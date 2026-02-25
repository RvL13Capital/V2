"""
Production Prediction Script for MambaAttention
================================================

TEMPORAL-SAFE real-time prediction for consolidation patterns.

Usage:
    python predict_mamba_attention.py \\
        --model output/models/mamba_attention_asl_best_20251109_120000.pth \\
        --sequences output/sequences_temporal_safe_69f.parquet \\
        --output output/predictions_mamba_attention.parquet

This script:
1. Loads trained MambaAttention model
2. Loads temporal-safe sequences
3. Generates predictions with EV calculations
4. Saves results for validation

TEMPORAL SAFETY:
- Uses only sequences built from historical data
- No future data accessed during prediction
- All features calculated with temporal constraints
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import logging
from typing import Dict, Optional
from datetime import datetime
from tqdm import tqdm

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))

from training.models.mamba_attention_classifier import MambaAttentionClassifier, create_mamba_attention_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MambaAttentionPredictor:
    """
    Production predictor for MambaAttention model.

    Features:
    - Batch prediction with progress bars
    - Expected value (EV) calculation
    - Signal generation (STRONG/GOOD/MODERATE/AVOID)
    - Temporal safety validation
    """

    # Strategic values for each outcome class
    STRATEGIC_VALUES = {
        0: -2,    # K0_STAGNANT
        1: -0.2,  # K1_MINIMAL
        2: 1,     # K2_QUALITY
        3: 3,     # K3_STRONG
        4: 10,    # K4_EXCEPTIONAL
        5: -10    # K5_FAILED
    }

    # Signal thresholds
    SIGNAL_THRESHOLDS = {
        'STRONG': 5.0,
        'GOOD': 3.0,
        'MODERATE': 1.0
    }

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        model_size: str = 'base'
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint (.pth file)
            device: torch.device (defaults to CUDA if available)
            model_size: Model size preset ('small', 'base', 'large')
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_size = model_size

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        logger.info(f"Predictor initialized with device: {self.device}")

    def _load_model(self, model_path: str) -> MambaAttentionClassifier:
        """
        Load trained model from checkpoint.

        Args:
            model_path: Path to .pth file

        Returns:
            Loaded MambaAttentionClassifier
        """
        logger.info(f"Loading model from {model_path}")

        # Create model architecture
        model = create_mamba_attention_model(
            input_dim=69,
            n_classes=6,
            model_size=self.model_size
        )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"  Val metrics: {checkpoint.get('val_metrics', {})}")

        return model

    def predict_batch(
        self,
        sequences: np.ndarray,
        masks: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Predict on batch of sequences.

        Args:
            sequences: (n_samples, seq_len, n_features) numpy array
            masks: (n_samples, seq_len) attention masks
            batch_size: Batch size for prediction

        Returns:
            Dict with:
                - predictions: (n_samples,) predicted class indices
                - probabilities: (n_samples, 6) class probabilities
                - expected_values: (n_samples,) EV scores
                - signals: (n_samples,) signal classifications
        """
        n_samples = len(sequences)
        all_predictions = []
        all_probabilities = []

        # Process in batches
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), desc="Predicting"):
                batch_end = min(i + batch_size, n_samples)

                # Get batch
                batch_seq = torch.tensor(sequences[i:batch_end], dtype=torch.float32).to(self.device)
                batch_mask = torch.tensor(masks[i:batch_end], dtype=torch.float32).to(self.device)

                # Forward pass
                logits = self.model(batch_seq, batch_mask)
                probs = torch.softmax(logits, dim=1)

                # Store results
                all_probabilities.append(probs.cpu().numpy())
                all_predictions.append(torch.argmax(logits, dim=1).cpu().numpy())

        # Concatenate results
        probabilities = np.vstack(all_probabilities)
        predictions = np.concatenate(all_predictions)

        # Calculate expected values
        expected_values = self._calculate_expected_values(probabilities)

        # Generate signals
        signals = self._generate_signals(expected_values, probabilities)

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'expected_values': expected_values,
            'signals': signals
        }

    def _calculate_expected_values(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate Expected Value (EV) from class probabilities.

        EV = Σ(p_i × strategic_value_i)

        Args:
            probabilities: (n_samples, 6) class probabilities

        Returns:
            (n_samples,) expected values
        """
        strategic_values = np.array([
            self.STRATEGIC_VALUES[i] for i in range(6)
        ])

        # EV = probabilities @ values
        ev = probabilities @ strategic_values

        return ev

    def _generate_signals(
        self,
        expected_values: np.ndarray,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        Generate trading signals based on EV and failure probability.

        Signal logic:
        - STRONG: EV >= 5.0 AND P(K5) < 0.30
        - GOOD: EV >= 3.0 AND P(K5) < 0.30
        - MODERATE: EV >= 1.0 AND P(K5) < 0.30
        - AVOID: EV < 0 OR P(K5) >= 0.30

        Args:
            expected_values: (n_samples,) EV scores
            probabilities: (n_samples, 6) class probabilities

        Returns:
            (n_samples,) signal classifications (string array)
        """
        signals = np.empty(len(expected_values), dtype=object)

        # Get K5 (failure) probabilities
        k5_probs = probabilities[:, 5]

        for i in range(len(expected_values)):
            ev = expected_values[i]
            k5_prob = k5_probs[i]

            # AVOID if high failure probability or negative EV
            if k5_prob >= 0.30 or ev < 0:
                signals[i] = 'AVOID'
            elif ev >= self.SIGNAL_THRESHOLDS['STRONG']:
                signals[i] = 'STRONG'
            elif ev >= self.SIGNAL_THRESHOLDS['GOOD']:
                signals[i] = 'GOOD'
            elif ev >= self.SIGNAL_THRESHOLDS['MODERATE']:
                signals[i] = 'MODERATE'
            else:
                signals[i] = 'AVOID'

        return signals

    def predict_sequences_df(
        self,
        sequences_df: pd.DataFrame,
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Predict on sequences DataFrame.

        Args:
            sequences_df: DataFrame from TemporalSafeSequenceBuilder
            batch_size: Batch size for prediction

        Returns:
            DataFrame with original data + predictions
        """
        logger.info(f"Predicting on {len(sequences_df)} sequences")

        # Extract sequences and masks
        sequences = np.stack(sequences_df['sequence'].values)
        masks = np.stack(sequences_df['attention_mask'].values)

        # Predict
        results = self.predict_batch(sequences, masks, batch_size)

        # Add predictions to DataFrame
        predictions_df = sequences_df.copy()
        predictions_df['predicted_class'] = results['predictions']
        predictions_df['expected_value'] = results['expected_values']
        predictions_df['signal'] = results['signals']

        # Add class probabilities
        for i in range(6):
            predictions_df[f'prob_k{i}'] = results['probabilities'][:, i]

        logger.info("✓ Predictions complete")

        # Log signal distribution
        logger.info(f"Signal distribution:\n{predictions_df['signal'].value_counts()}")

        return predictions_df


def main():
    """
    Main prediction pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Predict with MambaAttention model")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--sequences', type=str, required=True, help='Path to sequences parquet')
    parser.add_argument('--output', type=str, required=True, help='Path to output predictions parquet')
    parser.add_argument('--model-size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--batch-size', type=int, default=32)

    args = parser.parse_args()

    # Initialize predictor
    predictor = MambaAttentionPredictor(
        model_path=args.model,
        model_size=args.model_size
    )

    # Load sequences
    logger.info(f"Loading sequences from {args.sequences}")
    sequences_df = pd.read_parquet(args.sequences)

    # Predict
    predictions_df = predictor.predict_sequences_df(
        sequences_df,
        batch_size=args.batch_size
    )

    # Save predictions
    logger.info(f"Saving predictions to {args.output}")
    predictions_df.to_parquet(args.output, index=False)

    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total sequences: {len(predictions_df)}")
    print(f"\nPredicted class distribution:")
    print(predictions_df['predicted_class'].value_counts().sort_index())
    print(f"\nSignal distribution:")
    print(predictions_df['signal'].value_counts())
    print(f"\nExpected Value statistics:")
    print(f"  Mean: {predictions_df['expected_value'].mean():.3f}")
    print(f"  Median: {predictions_df['expected_value'].median():.3f}")
    print(f"  Min: {predictions_df['expected_value'].min():.3f}")
    print(f"  Max: {predictions_df['expected_value'].max():.3f}")

    # High-value patterns
    strong_signals = predictions_df[predictions_df['signal'] == 'STRONG']
    print(f"\nSTRONG signals: {len(strong_signals)} ({len(strong_signals)/len(predictions_df)*100:.1f}%)")
    if len(strong_signals) > 0:
        print(f"  Avg EV: {strong_signals['expected_value'].mean():.3f}")
        print(f"  Avg P(K4): {strong_signals['prob_k4'].mean():.3f}")
        print(f"  Avg P(K5): {strong_signals['prob_k5'].mean():.3f}")

    print("="*60)

    logger.info("✓ Prediction pipeline complete!")


if __name__ == "__main__":
    main()
