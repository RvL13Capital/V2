"""
Ablation Test: Does Attention Actually Help?
=============================================

This script tests whether the attention mechanism in HybridFeatureNetwork
provides meaningful improvement over LSTM + CNN alone.

Hypothesis: The attention layer (applied to CNN output, then mean-pooled)
may not be contributing much since:
1. It's applied to CNN features, not raw input
2. Mean pooling destroys temporal specificity
3. LSTM may be doing the heavy lifting

Test Design:
- Train model WITH attention (baseline)
- Train model WITHOUT attention (ablation)
- Compare EV-based metrics at various thresholds
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import argparse
from datetime import datetime

from models.temporal_hybrid_v18 import HybridFeatureNetwork
from models.asymmetric_loss import AsymmetricLoss
from config.constants import STRATEGIC_VALUES


class HybridFeatureNetworkNoAttention(nn.Module):
    """
    Ablation variant: LSTM + CNN without attention.

    This version skips the attention mechanism entirely,
    directly using mean-pooled CNN features.
    """

    def __init__(
        self,
        input_features: int = 10,  # 10 after composite disabled (was 14)
        sequence_length: int = 20,
        context_features: int = 5,
        lstm_hidden: int = 32,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
        cnn_channels: list = None,
        cnn_kernel_sizes: list = None,
        fusion_hidden: int = 64,
        fusion_dropout: float = 0.3,
        num_classes: Optional[int] = None
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64]
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [3, 5]
        if num_classes is None:
            num_classes = 3

        self.input_features = input_features
        self.sequence_length = sequence_length
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.num_classes = num_classes
        self.context_features = context_features
        self.use_context = context_features > 0

        # LSTM Path (same as original)
        self.engineered_branch = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        self.lstm_output_dim = lstm_hidden

        # CNN Path (same as original)
        self.conv_3 = nn.Conv1d(
            in_channels=input_features,
            out_channels=cnn_channels[0],
            kernel_size=cnn_kernel_sizes[0],
            padding=cnn_kernel_sizes[0] // 2
        )
        self.conv_5 = nn.Conv1d(
            in_channels=input_features,
            out_channels=cnn_channels[1],
            kernel_size=cnn_kernel_sizes[1],
            padding=cnn_kernel_sizes[1] // 2
        )

        self.cnn_output_dim = sum(cnn_channels)  # 96

        # NO ATTENTION - this is the ablation

        # Coil encoder (adjusted for no attention)
        branch_a_dim = self.lstm_output_dim + self.cnn_output_dim  # 32 + 96 = 128
        self.coil_encoder = nn.Sequential(
            nn.Linear(branch_a_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Context branch (same as original)
        if self.use_context:
            self.context_branch = nn.Sequential(
                nn.Linear(context_features, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            context_output_dim = 32
        else:
            context_output_dim = 0

        # Fusion (same as original)
        combined_dim = 64 + context_output_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)

        # LSTM path
        lstm_out, _ = self.engineered_branch(x)
        lstm_features = lstm_out[:, -1, :]  # (batch, 32)

        # CNN path
        x_conv = x.transpose(1, 2)  # (batch, 14, 20)
        conv3_out = F.relu(self.conv_3(x_conv))  # (batch, 32, 20)
        conv5_out = F.relu(self.conv_5(x_conv))  # (batch, 64, 20)
        conv_concat = torch.cat([conv3_out, conv5_out], dim=1)  # (batch, 96, 20)

        # NO ATTENTION - just mean pool directly
        cnn_features = conv_concat.mean(dim=2)  # (batch, 96)

        # Combine LSTM + CNN
        branch_a_combined = torch.cat([lstm_features, cnn_features], dim=1)  # (batch, 128)
        coil_embedding = self.coil_encoder(branch_a_combined)  # (batch, 64)

        # Context branch
        if self.use_context:
            if context is not None:
                context_embedding = self.context_branch(context)
            else:
                context_embedding = torch.zeros(batch_size, 32, device=coil_embedding.device)
            combined = torch.cat([coil_embedding, context_embedding], dim=1)
        else:
            combined = coil_embedding

        logits = self.fusion(combined)
        return logits


def normalize_sequences(sequences: np.ndarray) -> np.ndarray:
    """
    Normalize sequences for training.

    - Relativize OHLC + boundaries to day 0 close
    - Z-score normalize each feature across all sequences
    - Handle NaN by replacing with feature mean
    """
    sequences = sequences.copy()

    # Price features: 0-3 (OHLC), 12-13 (boundaries)
    # Relativize to day 0 close
    price_indices = [0, 1, 2, 3, 12, 13]
    for i, seq in enumerate(sequences):
        day0_close = seq[0, 3]  # close at day 0
        if day0_close > 0 and not np.isnan(day0_close):
            for idx in price_indices:
                sequences[i, :, idx] = (seq[:, idx] / day0_close) - 1.0

    # Replace NaN with 0 before computing stats
    nan_mask = np.isnan(sequences)
    sequences[nan_mask] = 0.0

    # Z-score normalize each feature across all sequences
    # Compute mean and std over (n_samples, n_timesteps) for each feature
    mean = sequences.mean(axis=(0, 1), keepdims=True)
    std = sequences.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0  # Prevent division by zero

    sequences = (sequences - mean) / std

    return sequences


def load_and_split_data(data_dir: Path) -> Tuple[
    torch.Tensor, torch.Tensor,  # train
    torch.Tensor, torch.Tensor,  # val
    torch.Tensor, torch.Tensor   # test
]:
    """
    Load sequences and labels, then split temporally.

    Uses metadata.parquet for start_date to ensure no look-ahead bias:
    - Train: start_date < 2023
    - Val: 2023 <= start_date < 2024
    - Test: start_date >= 2024
    """
    # Find the most recent files
    seq_files = list(data_dir.glob('sequences*.npy'))
    label_files = list(data_dir.glob('labels*.npy'))
    meta_files = list(data_dir.glob('metadata*.parquet'))

    if not seq_files:
        raise FileNotFoundError(f"No sequences.npy in {data_dir}")
    if not label_files:
        raise FileNotFoundError(f"No labels.npy in {data_dir}")
    if not meta_files:
        raise FileNotFoundError(f"No metadata.parquet in {data_dir}")

    # Use newest files
    seq_file = max(seq_files, key=lambda p: p.stat().st_mtime)
    label_file = max(label_files, key=lambda p: p.stat().st_mtime)
    meta_file = max(meta_files, key=lambda p: p.stat().st_mtime)

    print(f"  Loading: {seq_file.name}, {label_file.name}, {meta_file.name}")

    sequences = np.load(seq_file)
    labels = np.load(label_file)
    metadata = pd.read_parquet(meta_file)

    # Normalize sequences
    print("  Normalizing sequences...")
    sequences = normalize_sequences(sequences)

    # Parse dates
    if 'pattern_start_date' in metadata.columns:
        dates = pd.to_datetime(metadata['pattern_start_date'])
    elif 'start_date' in metadata.columns:
        dates = pd.to_datetime(metadata['start_date'])
    elif 'pattern_start' in metadata.columns:
        dates = pd.to_datetime(metadata['pattern_start'])
    else:
        print("  WARNING: No date column found, using random split")
        n = len(labels)
        perm = np.random.permutation(n)
        train_idx = perm[:int(0.7*n)]
        val_idx = perm[int(0.7*n):int(0.85*n)]
        test_idx = perm[int(0.85*n):]

        return (
            torch.tensor(sequences[train_idx], dtype=torch.float32),
            torch.tensor(labels[train_idx], dtype=torch.long),
            torch.tensor(sequences[val_idx], dtype=torch.float32),
            torch.tensor(labels[val_idx], dtype=torch.long),
            torch.tensor(sequences[test_idx], dtype=torch.float32),
            torch.tensor(labels[test_idx], dtype=torch.long)
        )

    # Temporal split
    train_mask = dates.dt.year < 2023
    val_mask = (dates.dt.year == 2023)
    test_mask = dates.dt.year >= 2024

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    print(f"  Temporal split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    return (
        torch.tensor(sequences[train_idx], dtype=torch.float32),
        torch.tensor(labels[train_idx], dtype=torch.long),
        torch.tensor(sequences[val_idx], dtype=torch.float32),
        torch.tensor(labels[val_idx], dtype=torch.long),
        torch.tensor(sequences[test_idx], dtype=torch.float32),
        torch.tensor(labels[test_idx], dtype=torch.long)
    )


def calculate_ev(probs: torch.Tensor) -> torch.Tensor:
    """Calculate expected value from class probabilities."""
    # probs shape: (N, 3)
    ev = (probs[:, 0] * STRATEGIC_VALUES[0] +   # Danger: -10
          probs[:, 1] * STRATEGIC_VALUES[1] +   # Noise: -1
          probs[:, 2] * STRATEGIC_VALUES[2])    # Target: +10
    return ev


def evaluate_model(model, sequences, labels, device):
    """Evaluate model and return metrics at various EV thresholds."""
    model.eval()

    with torch.no_grad():
        sequences = sequences.to(device)
        logits = model(sequences)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1).cpu().numpy()

    probs_np = probs.cpu().numpy()
    labels_np = labels.numpy()

    # Calculate EV for each sample
    ev_values = calculate_ev(probs).cpu().numpy()

    # Global accuracy
    accuracy = (preds == labels_np).mean() * 100

    # Metrics at various EV thresholds
    thresholds = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    results = {'accuracy': accuracy, 'thresholds': {}}

    for thresh in thresholds:
        mask = ev_values > thresh
        n_signals = mask.sum()

        if n_signals > 0:
            signal_labels = labels_np[mask]
            target_rate = (signal_labels == 2).sum() / n_signals * 100
            danger_rate = (signal_labels == 0).sum() / n_signals * 100
            actual_values = np.where(signal_labels == 0, -10,
                           np.where(signal_labels == 1, -1, 10))
            avg_value = actual_values.mean()
        else:
            target_rate = 0.0
            danger_rate = 0.0
            avg_value = 0.0

        results['thresholds'][thresh] = {
            'n_signals': int(n_signals),
            'pct_signals': n_signals / len(labels_np) * 100,
            'target_rate': target_rate,
            'danger_rate': danger_rate,
            'avg_value': avg_value
        }

    return results


def train_model(model, train_sequences, train_labels, val_sequences, val_labels,
                device, epochs=50, batch_size=32, lr=0.0005, patience=10):
    """Train model with early stopping."""
    model = model.to(device)

    # Class weights (inverse frequency)
    class_counts = np.bincount(train_labels.numpy(), minlength=3)
    total = class_counts.sum()
    class_weights = torch.tensor(total / (3 * class_counts + 1e-6), dtype=torch.float32).to(device)

    # Use standard CrossEntropyLoss for clean ablation comparison
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    n_train = len(train_sequences)

    for epoch in range(epochs):
        model.train()

        # Shuffle
        perm = torch.randperm(n_train)
        train_sequences = train_sequences[perm]
        train_labels = train_labels[perm]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_x = train_sequences[i:i+batch_size].to(device)
            batch_y = train_labels[i:i+batch_size].to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_sequences.to(device))
            val_loss = criterion(val_logits, val_labels.to(device)).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def print_comparison(results_with_attn, results_no_attn, split_name):
    """Print side-by-side comparison of results."""
    print(f"\n{'='*70}")
    print(f"ABLATION RESULTS - {split_name} Set")
    print(f"{'='*70}")

    print(f"\nGlobal Accuracy:")
    print(f"  With Attention:    {results_with_attn['accuracy']:.1f}%")
    print(f"  Without Attention: {results_no_attn['accuracy']:.1f}%")
    print(f"  Difference:        {results_with_attn['accuracy'] - results_no_attn['accuracy']:+.1f}%")

    print(f"\n{'EV Thresh':<12} {'Metric':<12} {'With Attn':<12} {'No Attn':<12} {'Delta':<10}")
    print("-" * 58)

    for thresh in [0.0, 2.0, 3.0, 4.0]:
        w = results_with_attn['thresholds'][thresh]
        n = results_no_attn['thresholds'][thresh]

        print(f"> {thresh:.1f}")
        print(f"{'':12} {'Signals':<12} {w['n_signals']:<12} {n['n_signals']:<12}")
        print(f"{'':12} {'Target %':<12} {w['target_rate']:.1f}%{'':<7} {n['target_rate']:.1f}%{'':<7} {w['target_rate']-n['target_rate']:+.1f}%")
        print(f"{'':12} {'Danger %':<12} {w['danger_rate']:.1f}%{'':<7} {n['danger_rate']:.1f}%{'':<7} {w['danger_rate']-n['danger_rate']:+.1f}%")
        print(f"{'':12} {'Avg Value':<12} {w['avg_value']:+.2f}{'':<7} {n['avg_value']:+.2f}{'':<7} {w['avg_value']-n['avg_value']:+.2f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Ablation test: attention vs no attention')
    parser.add_argument('--data-dir', type=str, default='output/sequences/eu',
                        help='Directory with sequence data')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)

    # Load data
    print("\nLoading data...")
    train_seq, train_labels, val_seq, val_labels, test_seq, test_labels = load_and_split_data(data_dir)

    print(f"  Train: {len(train_seq)} sequences")
    print(f"  Val:   {len(val_seq)} sequences")
    print(f"  Test:  {len(test_seq)} sequences")

    # Get input dimensions
    _, seq_len, n_features = train_seq.shape
    print(f"  Shape: ({seq_len} timesteps, {n_features} features)")

    # =========================================
    # Model 1: With Attention (Baseline)
    # =========================================
    print("\n" + "="*50)
    print("Training Model WITH Attention...")
    print("="*50)

    model_with_attn = HybridFeatureNetwork(
        input_features=n_features,
        sequence_length=seq_len,
        context_features=0  # No context for fair comparison
    )

    # Count parameters
    n_params_with = sum(p.numel() for p in model_with_attn.parameters())
    print(f"Parameters: {n_params_with:,}")

    model_with_attn = train_model(
        model_with_attn, train_seq, train_labels, val_seq, val_labels,
        device, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience
    )

    # =========================================
    # Model 2: Without Attention (Ablation)
    # =========================================
    print("\n" + "="*50)
    print("Training Model WITHOUT Attention...")
    print("="*50)

    # Reset seeds for fair comparison
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_no_attn = HybridFeatureNetworkNoAttention(
        input_features=n_features,
        sequence_length=seq_len,
        context_features=0
    )

    n_params_without = sum(p.numel() for p in model_no_attn.parameters())
    print(f"Parameters: {n_params_without:,}")
    print(f"Parameter reduction: {n_params_with - n_params_without:,} ({(n_params_with - n_params_without)/n_params_with*100:.1f}%)")

    model_no_attn = train_model(
        model_no_attn, train_seq, train_labels, val_seq, val_labels,
        device, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, patience=args.patience
    )

    # =========================================
    # Evaluate Both Models
    # =========================================
    print("\nEvaluating models...")

    # Validation set
    val_results_with = evaluate_model(model_with_attn, val_seq, val_labels, device)
    val_results_no = evaluate_model(model_no_attn, val_seq, val_labels, device)
    print_comparison(val_results_with, val_results_no, "Validation")

    # Test set
    test_results_with = evaluate_model(model_with_attn, test_seq, test_labels, device)
    test_results_no = evaluate_model(model_no_attn, test_seq, test_labels, device)
    print_comparison(test_results_with, test_results_no, "Test")

    # =========================================
    # Summary
    # =========================================
    print("\n" + "="*70)
    print("ABLATION SUMMARY")
    print("="*70)

    # Key metric: Target rate at EV > 2.0 (recommended threshold)
    thresh = 2.0
    test_w = test_results_with['thresholds'][thresh]
    test_n = test_results_no['thresholds'][thresh]

    print(f"\nKey Metric: Target Rate @ EV > 2.0 (Test Set)")
    print(f"  With Attention:    {test_w['target_rate']:.1f}% (N={test_w['n_signals']})")
    print(f"  Without Attention: {test_n['target_rate']:.1f}% (N={test_n['n_signals']})")

    delta = test_w['target_rate'] - test_n['target_rate']
    if abs(delta) < 2.0:
        conclusion = "ATTENTION PROVIDES MINIMAL BENEFIT"
    elif delta > 5.0:
        conclusion = "ATTENTION HELPS SIGNIFICANTLY"
    elif delta < -5.0:
        conclusion = "ATTENTION HURTS PERFORMANCE"
    else:
        conclusion = f"ATTENTION {'HELPS' if delta > 0 else 'HURTS'} MARGINALLY"

    print(f"\n  Conclusion: {conclusion} ({delta:+.1f}% difference)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = data_dir.parent / f'ablation_results_{timestamp}.txt'

    with open(output_file, 'w') as f:
        f.write(f"Ablation Test Results - {timestamp}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Parameters with attention: {n_params_with:,}\n")
        f.write(f"Parameters without attention: {n_params_without:,}\n\n")
        f.write(f"Test Set Results @ EV > 2.0:\n")
        f.write(f"  With Attention: {test_w['target_rate']:.1f}% target, {test_w['danger_rate']:.1f}% danger\n")
        f.write(f"  Without Attention: {test_n['target_rate']:.1f}% target, {test_n['danger_rate']:.1f}% danger\n")
        f.write(f"\nConclusion: {conclusion}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
