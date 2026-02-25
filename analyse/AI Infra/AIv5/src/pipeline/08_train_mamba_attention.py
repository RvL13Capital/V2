"""
Training Pipeline for MambaAttention + Asymmetric Loss
=======================================================

CRITICAL: This pipeline ensures ZERO forward-looking bias throughout training.

Workflow:
1. Load temporal-safe sequences (built by temporal_safe_sequence_builder.py)
2. Split by DATE (train: past, validation: future)
3. Train MambaAttention with Asymmetric Loss
4. Validate ONLY on future data (never seen during training)
5. Track K4 recall, EV correlation, temporal integrity

Temporal Safety Mechanisms:
- Train/val split by snapshot_date (no random shuffle)
- Validation metrics computed ONLY on validation set
- No gradient updates on validation data
- Assert: max(train_dates) < min(val_dates)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime
from tqdm import tqdm
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.models.mamba_attention_classifier import MambaAttentionClassifier, create_mamba_attention_model
from training.losses.asymmetric_loss import AsymmetricLoss
from src.data.temporal_safe_sequence_builder import TemporalSafeSequenceBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatternSequenceDataset(Dataset):
    """
    PyTorch Dataset for pattern sequences.

    Each sample is:
    - sequence: (seq_len, n_features) temporal sequence
    - mask: (seq_len,) attention mask (1=valid, 0=padding)
    - label: int (K0-K5 outcome class)
    - metadata: dict (ticker, snapshot_date, etc.)
    """

    def __init__(self, sequences_df: pd.DataFrame):
        """
        Initialize dataset.

        Args:
            sequences_df: DataFrame from TemporalSafeSequenceBuilder
        """
        self.sequences_df = sequences_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.sequences_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, dict]:
        row = self.sequences_df.iloc[idx]

        # Get sequence and mask
        sequence = torch.tensor(row['sequence'], dtype=torch.float32)
        mask = torch.tensor(row['attention_mask'], dtype=torch.float32)
        label = int(row['outcome_class'])

        # Metadata
        metadata = {
            'ticker': row['ticker'],
            'snapshot_date': row['snapshot_date'],
            'days_in_pattern': row['days_in_pattern']
        }

        return sequence, mask, label, metadata


class TemporalSafeTrainer:
    """
    Trainer for MambaAttention with temporal safety guarantees.
    """

    def __init__(
        self,
        model: MambaAttentionClassifier,
        criterion: AsymmetricLoss,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.

        Args:
            model: MambaAttentionClassifier instance
            criterion: AsymmetricLoss instance
            device: torch.device (cuda or cpu)
            learning_rate: Learning rate for AdamW
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler (cosine annealing)
        self.scheduler = None  # Will be set after knowing max_epochs

        # Metrics tracking
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')
        self.best_k4_recall = 0.0
        self.patience_counter = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dict with training metrics
        """
        self.model.train()

        total_loss = 0
        all_preds = []
        all_labels = []
        k4_correct = 0
        k4_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")

        for sequences, masks, labels, metadata in pbar:
            # Move to device
            sequences = sequences.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.model(sequences, masks)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # K4 metrics
            k4_mask = (labels == 4)
            if k4_mask.sum() > 0:
                k4_correct += (preds[k4_mask] == 4).sum().item()
                k4_total += k4_mask.sum().item()

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        k4_recall = k4_correct / k4_total if k4_total > 0 else 0.0

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'k4_recall': k4_recall,
            'k4_count': k4_total
        }

        return metrics

    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict:
        """
        Validate on held-out data (TEMPORAL-SAFE).

        CRITICAL: This data is from FUTURE dates (never seen during training).
        No gradient updates allowed.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dict with validation metrics
        """
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        k4_correct = 0
        k4_total = 0
        k4_precision_denom = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [VAL]")

        with torch.no_grad():  # NO GRADIENTS ON VALIDATION
            for sequences, masks, labels, metadata in pbar:
                # Move to device
                sequences = sequences.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                # Forward pass ONLY (no backward)
                logits = self.model(sequences, masks)
                loss = self.criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # K4 metrics
                k4_mask = (labels == 4)
                k4_pred_mask = (preds == 4)

                if k4_mask.sum() > 0:
                    k4_correct += (preds[k4_mask] == 4).sum().item()
                    k4_total += k4_mask.sum().item()

                if k4_pred_mask.sum() > 0:
                    k4_precision_denom += k4_pred_mask.sum().item()

                # Update progress bar
                pbar.set_postfix({'val_loss': loss.item()})

        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

        # K4 metrics
        k4_recall = k4_correct / k4_total if k4_total > 0 else 0.0
        k4_precision = k4_correct / k4_precision_denom if k4_precision_denom > 0 else 0.0

        # Calculate EV correlation
        ev_values = self._calculate_ev(np.array(all_probs))
        actual_values = self._get_strategic_values(np.array(all_labels))
        ev_correlation = np.corrcoef(ev_values, actual_values)[0, 1] if len(ev_values) > 1 else 0.0

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'k4_recall': k4_recall,
            'k4_precision': k4_precision,
            'k4_count': k4_total,
            'ev_correlation': ev_correlation
        }

        return metrics

    def _calculate_ev(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate Expected Value from class probabilities.

        Args:
            probabilities: (n_samples, 6) class probabilities

        Returns:
            (n_samples,) expected values
        """
        # Strategic values for each class
        strategic_values = np.array([
            -2,    # K0_STAGNANT
            -0.2,  # K1_MINIMAL
            1,     # K2_QUALITY
            3,     # K3_STRONG
            10,    # K4_EXCEPTIONAL
            -10    # K5_FAILED
        ])

        # EV = Σ(p_i × value_i)
        ev = probabilities @ strategic_values

        return ev

    def _get_strategic_values(self, labels: np.ndarray) -> np.ndarray:
        """
        Get strategic values for actual labels.

        Args:
            labels: (n_samples,) class labels

        Returns:
            (n_samples,) strategic values
        """
        value_map = {
            0: -2,    # K0
            1: -0.2,  # K1
            2: 1,     # K2
            3: 3,     # K3
            4: 10,    # K4
            5: -10    # K5
        }

        return np.array([value_map[label] for label in labels])

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        early_stopping_patience: int = 15,
        save_dir: Path = Path("output/models")
    ) -> Dict:
        """
        Train model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save best model

        Returns:
            Dict with training history
        """
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs
        )

        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Starting training: max_epochs={max_epochs}, patience={early_stopping_patience}")

        for epoch in range(1, max_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history.append(train_metrics)

            # Validate
            val_metrics = self.validate(val_loader, epoch)
            self.val_history.append(val_metrics)

            # Learning rate step
            self.scheduler.step()

            # Log metrics
            logger.info(
                f"Epoch {epoch}/{max_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"K4 Recall: {val_metrics['k4_recall']:.4f}, "
                f"EV Corr: {val_metrics['ev_correlation']:.4f}"
            )

            # Save best model (based on K4 recall)
            if val_metrics['k4_recall'] > self.best_k4_recall:
                self.best_k4_recall = val_metrics['k4_recall']
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0

                # Save model
                model_path = save_dir / f"mamba_attention_asl_best_{timestamp}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics
                }, model_path)

                logger.info(f"✓ Saved best model: {model_path} (K4 recall: {val_metrics['k4_recall']:.4f})")

            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered (patience={early_stopping_patience})")
                break

        # Save training history
        history_path = save_dir / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump({
                'train_history': self.train_history,
                'val_history': self.val_history
            }, f, indent=2, default=str)

        logger.info(f"✓ Training complete! Best K4 recall: {self.best_k4_recall:.4f}")

        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_k4_recall': self.best_k4_recall,
            'best_val_loss': self.best_val_loss
        }


def temporal_train_val_split(
    sequences_df: pd.DataFrame,
    split_date: Optional[pd.Timestamp] = None,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split sequences by DATE (TEMPORAL-SAFE).

    CRITICAL: This ensures validation data is from FUTURE dates.
    No random shuffling - strict temporal ordering.

    Args:
        sequences_df: DataFrame with sequences and snapshot_date
        split_date: Explicit split date (optional, will calculate from ratio if None)
        train_ratio: Train/val ratio (default: 0.8)

    Returns:
        (train_df, val_df) tuple
    """
    # Ensure snapshot_date is datetime
    sequences_df['snapshot_date'] = pd.to_datetime(sequences_df['snapshot_date'])

    # Calculate split date if not provided
    if split_date is None:
        sorted_dates = sequences_df['snapshot_date'].sort_values()
        split_idx = int(len(sorted_dates) * train_ratio)
        split_date = sorted_dates.iloc[split_idx]

    logger.info(f"Temporal split date: {split_date}")

    # Split by date
    train_df = sequences_df[sequences_df['snapshot_date'] < split_date].copy()
    val_df = sequences_df[sequences_df['snapshot_date'] >= split_date].copy()

    # Validate temporal integrity
    max_train_date = train_df['snapshot_date'].max()
    min_val_date = val_df['snapshot_date'].min()

    if max_train_date >= min_val_date:
        raise ValueError(
            f"TEMPORAL INTEGRITY VIOLATION: "
            f"Max train date ({max_train_date}) >= Min val date ({min_val_date})\n"
            f"This indicates temporal leakage!"
        )

    logger.info(f"✓ Temporal split validated:")
    logger.info(f"  Train: {len(train_df)} samples, dates: {train_df['snapshot_date'].min()} to {max_train_date}")
    logger.info(f"  Val:   {len(val_df)} samples, dates: {min_val_date} to {val_df['snapshot_date'].max()}")

    # Log class distribution
    logger.info(f"Train outcome distribution:\n{train_df['outcome_class'].value_counts().sort_index()}")
    logger.info(f"Val outcome distribution:\n{val_df['outcome_class'].value_counts().sort_index()}")

    return train_df, val_df


def main():
    """
    Main training pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train MambaAttention with Asymmetric Loss")
    parser.add_argument('--sequences', type=str, required=True, help='Input sequences parquet')
    parser.add_argument('--model-size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--split-ratio', type=float, default=0.8)
    parser.add_argument('--output-dir', type=str, default='output/models')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load sequences
    logger.info(f"Loading sequences from {args.sequences}")
    sequences_df = pd.read_parquet(args.sequences)

    # Temporal train/val split
    train_df, val_df = temporal_train_val_split(sequences_df, train_ratio=args.split_ratio)

    # Create datasets
    train_dataset = PatternSequenceDataset(train_df)
    val_dataset = PatternSequenceDataset(val_df)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Can shuffle within train set (dates already separated)
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Never shuffle validation
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model
    logger.info(f"Creating MambaAttention model (size: {args.model_size})")
    model = create_mamba_attention_model(
        input_dim=69,
        n_classes=6,
        model_size=args.model_size
    )

    # Create loss
    criterion = AsymmetricLoss(
        gamma_pos=0,
        gamma_neg=4
    )

    # Create trainer
    trainer = TemporalSafeTrainer(
        model=model,
        criterion=criterion,
        device=device,
        learning_rate=args.learning_rate
    )

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_dir=Path(args.output_dir)
    )

    logger.info("✓ Training pipeline complete!")


if __name__ == "__main__":
    main()
