"""
Probability Calibration for Focal Loss Models
==============================================

Implements Isotonic Regression calibration to map focal-loss-distorted
probabilities back to real-world frequencies.

The Problem:
    Focal loss multiplies the loss by (1-p_t)^gamma, which reweights gradients
    during training. This causes the model to produce overconfident predictions
    on easy samples and underconfident predictions on hard samples. The resulting
    softmax outputs do NOT represent true class frequencies.

The Solution:
    Isotonic Regression learns a monotonic mapping from predicted probabilities
    to actual observed frequencies on a hold-out validation set. This calibration
    is applied post-hoc during inference, BEFORE calculating Expected Value.

Usage:
    # During training (after model training completes)
    from utils.probability_calibration import ProbabilityCalibrator

    calibrator = ProbabilityCalibrator()
    calibrator.fit(val_probs, val_labels)  # Fit on validation set
    calibrator.save('output/models/calibrator.pkl')

    # During inference
    calibrator = ProbabilityCalibrator.load('output/models/calibrator.pkl')
    calibrated_probs = calibrator.calibrate(raw_probs)
    ev = calculate_expected_value(calibrated_probs)

Author: TRANS System v17
Date: January 2026
"""

import numpy as np
import pickle
import json
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

try:
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available - probability calibration disabled")

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Isotonic Regression-based probability calibrator for multi-class classification.

    Trains one isotonic regressor per class, mapping predicted P(class=c) to
    calibrated P(class=c) based on observed frequencies in validation data.

    Attributes:
        calibrators: Dict mapping class index to fitted IsotonicRegression
        n_classes: Number of classes
        is_fitted: Whether calibrator has been fitted
        calibration_stats: Statistics about calibration quality
    """

    def __init__(self, n_classes: int = 3):
        """
        Initialize calibrator.

        Args:
            n_classes: Number of classes (default 3 for Danger/Noise/Target)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for probability calibration. pip install scikit-learn")

        self.calibrators: Dict[int, IsotonicRegression] = {}
        self.n_classes = n_classes
        self.is_fitted = False
        self.calibration_stats: Dict = {}

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit isotonic regression calibrators on validation set.

        Args:
            y_prob: Predicted probabilities, shape (n_samples, n_classes)
            y_true: True labels, shape (n_samples,) with values in [0, n_classes-1]

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If shapes are inconsistent or insufficient samples
        """
        # Validate inputs
        if y_prob.ndim != 2:
            raise ValueError(f"y_prob must be 2D, got shape {y_prob.shape}")

        n_samples, n_classes = y_prob.shape
        if n_classes != self.n_classes:
            logger.warning(f"Updating n_classes from {self.n_classes} to {n_classes}")
            self.n_classes = n_classes

        if len(y_true) != n_samples:
            raise ValueError(f"y_prob has {n_samples} samples but y_true has {len(y_true)}")

        if n_samples < 100:
            logger.warning(f"Only {n_samples} samples for calibration - may be unreliable")

        # Fit one isotonic regressor per class
        for c in range(self.n_classes):
            # Create binary target: 1 if true class is c, 0 otherwise
            y_binary = (y_true == c).astype(np.float64)

            # Get predicted probability for this class
            p_class = y_prob[:, c].astype(np.float64)

            # Fit isotonic regression
            # out_of_bounds='clip' ensures predictions outside training range are clipped
            ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
            ir.fit(p_class, y_binary)

            self.calibrators[c] = ir

            # Compute calibration statistics
            calibrated = ir.predict(p_class)
            self.calibration_stats[c] = {
                'n_samples': int(n_samples),
                'n_positive': int(y_binary.sum()),
                'mean_raw_prob': float(p_class.mean()),
                'mean_calibrated_prob': float(calibrated.mean()),
                'actual_frequency': float(y_binary.mean()),
                'calibration_error': float(abs(calibrated.mean() - y_binary.mean()))
            }

        self.is_fitted = True

        # Log summary
        total_error_before = sum(
            abs(self.calibration_stats[c]['mean_raw_prob'] - self.calibration_stats[c]['actual_frequency'])
            for c in range(self.n_classes)
        )
        total_error_after = sum(
            self.calibration_stats[c]['calibration_error']
            for c in range(self.n_classes)
        )

        logger.info(f"Fitted probability calibrator on {n_samples} samples")
        logger.info(f"Total calibration error: {total_error_before:.4f} -> {total_error_after:.4f}")

        for c in range(self.n_classes):
            stats = self.calibration_stats[c]
            logger.info(f"  Class {c}: raw={stats['mean_raw_prob']:.3f}, "
                       f"calibrated={stats['mean_calibrated_prob']:.3f}, "
                       f"actual={stats['actual_frequency']:.3f}")

        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw probabilities.

        Args:
            y_prob: Raw predicted probabilities, shape (n_samples, n_classes) or (n_classes,)

        Returns:
            Calibrated probabilities, same shape as input, normalized to sum to 1

        Raises:
            RuntimeError: If calibrator has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before calibrating. Call fit() first.")

        # Handle single sample
        single_sample = y_prob.ndim == 1
        if single_sample:
            y_prob = y_prob.reshape(1, -1)

        n_samples, n_classes = y_prob.shape

        if n_classes != self.n_classes:
            raise ValueError(f"Expected {self.n_classes} classes, got {n_classes}")

        # Calibrate each class independently
        calibrated = np.zeros_like(y_prob, dtype=np.float64)

        for c in range(self.n_classes):
            if c in self.calibrators:
                calibrated[:, c] = self.calibrators[c].predict(y_prob[:, c].astype(np.float64))
            else:
                # Fallback: use raw probability
                calibrated[:, c] = y_prob[:, c]

        # Handle edge cases: all zeros or negative values
        calibrated = np.maximum(calibrated, 1e-8)

        # Re-normalize to sum to 1 (important for EV calculation)
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / row_sums

        if single_sample:
            calibrated = calibrated.squeeze(0)

        return calibrated

    def get_calibration_report(self) -> str:
        """Generate human-readable calibration report."""
        if not self.is_fitted:
            return "Calibrator not fitted"

        lines = ["Probability Calibration Report", "=" * 40]

        for c in range(self.n_classes):
            stats = self.calibration_stats[c]
            lines.append(f"\nClass {c}:")
            lines.append(f"  Samples: {stats['n_samples']} ({stats['n_positive']} positive)")
            lines.append(f"  Raw mean prob: {stats['mean_raw_prob']:.4f}")
            lines.append(f"  Calibrated mean prob: {stats['mean_calibrated_prob']:.4f}")
            lines.append(f"  Actual frequency: {stats['actual_frequency']:.4f}")
            lines.append(f"  Calibration error: {stats['calibration_error']:.4f}")

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """
        Save fitted calibrator to disk.

        Args:
            path: File path (should end in .pkl)
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted calibrator")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'calibrators': self.calibrators,
            'n_classes': self.n_classes,
            'calibration_stats': self.calibration_stats
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        # Also save stats as JSON for easy inspection
        json_path = path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'n_classes': self.n_classes,
                'calibration_stats': self.calibration_stats
            }, f, indent=2)

        logger.info(f"Saved calibrator to {path}")

    @classmethod
    def load(cls, path: str) -> 'ProbabilityCalibrator':
        """
        Load fitted calibrator from disk.

        Args:
            path: File path to .pkl file

        Returns:
            Loaded ProbabilityCalibrator instance
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)

        calibrator = cls(n_classes=state['n_classes'])
        calibrator.calibrators = state['calibrators']
        calibrator.calibration_stats = state['calibration_stats']
        calibrator.is_fitted = True

        logger.info(f"Loaded calibrator from {path}")
        return calibrator

    def get_state(self) -> dict:
        """
        Get calibrator state for embedding in checkpoint.

        Returns:
            Dictionary containing all state needed to reconstruct calibrator

        Raises:
            RuntimeError: If calibrator not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot get state from unfitted calibrator")

        return {
            'calibrators': self.calibrators,
            'n_classes': self.n_classes,
            'calibration_stats': self.calibration_stats
        }

    def load_state(self, state: dict) -> 'ProbabilityCalibrator':
        """
        Load calibrator state from embedded checkpoint data.

        Args:
            state: State dictionary from get_state()

        Returns:
            self (for method chaining)
        """
        self.calibrators = state['calibrators']
        self.n_classes = state['n_classes']
        self.calibration_stats = state.get('calibration_stats', {})
        self.is_fitted = True

        logger.info(f"Loaded calibrator state ({self.n_classes} classes)")
        return self


def compute_expected_calibration_error(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, Dict]:
    """
    Compute Expected Calibration Error (ECE) for probability predictions.

    ECE measures the average gap between predicted confidence and actual accuracy
    across probability bins. Lower is better.

    Args:
        y_prob: Predicted probabilities, shape (n_samples, n_classes)
        y_true: True labels, shape (n_samples,)
        n_bins: Number of bins for calibration curve

    Returns:
        Tuple of (ECE value, detailed bin statistics)
    """
    n_samples = len(y_true)

    # Get predicted class and confidence
    predicted_class = y_prob.argmax(axis=1)
    confidence = y_prob.max(axis=1)
    correct = (predicted_class == y_true).astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_stats = []

    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
        n_in_bin = in_bin.sum()

        if n_in_bin > 0:
            bin_accuracy = correct[in_bin].mean()
            bin_confidence = confidence[in_bin].mean()
            bin_ece = abs(bin_accuracy - bin_confidence) * (n_in_bin / n_samples)
            ece += bin_ece

            bin_stats.append({
                'bin': i,
                'lower': float(bin_lower),
                'upper': float(bin_upper),
                'n_samples': int(n_in_bin),
                'accuracy': float(bin_accuracy),
                'confidence': float(bin_confidence),
                'gap': float(abs(bin_accuracy - bin_confidence))
            })

    return float(ece), {'bins': bin_stats, 'total_ece': float(ece)}
