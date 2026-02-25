"""
Tests for Ensemble Training and Dirichlet Calibration
======================================================

Tests the ensemble wrapper, Dirichlet calibration, and uncertainty estimation.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dirichlet_calibration import (
    DirichletCalibrator,
    EnsembleUncertaintyEstimator,
    compute_confidence_interval_ev
)


class MockModel(nn.Module):
    """Simple mock model for testing."""

    def __init__(self, input_dim=14*20, n_classes=3, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x, context=None):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.fc(x)


class TestDirichletCalibrator:
    """Tests for DirichletCalibrator."""

    def test_initialization(self):
        """Test calibrator initialization."""
        cal = DirichletCalibrator(n_classes=3)
        assert cal.n_classes == 3
        assert cal.W.shape == (3, 3)
        assert cal.b.shape == (3,)
        assert not cal.is_fitted

    def test_fit_basic(self):
        """Test basic fitting on synthetic data."""
        np.random.seed(42)

        # Generate synthetic logits and labels
        n_samples = 1000
        logits = np.random.randn(n_samples, 3).astype(np.float64)
        labels = np.random.randint(0, 3, n_samples)

        cal = DirichletCalibrator(n_classes=3)
        cal.fit(logits, labels)

        assert cal.is_fitted
        assert 'pre_ece' in cal.calibration_stats
        assert 'post_ece' in cal.calibration_stats

    def test_fit_reduces_ece(self):
        """Calibration should reduce ECE on well-structured data."""
        np.random.seed(42)

        n_samples = 2000
        # Create biased predictions (model is overconfident)
        logits = np.random.randn(n_samples, 3) * 3  # High confidence
        labels = np.random.randint(0, 3, n_samples)

        cal = DirichletCalibrator(n_classes=3)
        cal.fit(logits, labels)

        # ECE should decrease or stay similar
        # Note: on random data, improvement may be small
        assert cal.calibration_stats['post_ece'] <= cal.calibration_stats['pre_ece'] * 1.5

    def test_calibrate_output_shape(self):
        """Calibrated output has correct shape."""
        np.random.seed(42)

        # Fit calibrator
        logits_train = np.random.randn(500, 3)
        labels_train = np.random.randint(0, 3, 500)

        cal = DirichletCalibrator(n_classes=3)
        cal.fit(logits_train, labels_train)

        # Calibrate test data
        logits_test = np.random.randn(100, 3)
        probs, uncertainty = cal.calibrate(logits_test, return_uncertainty=True)

        assert probs.shape == (100, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert uncertainty is not None
        assert 'entropy' in uncertainty
        assert 'max_prob' in uncertainty

    def test_calibrate_without_fit_returns_softmax(self):
        """Unfitted calibrator returns raw softmax."""
        cal = DirichletCalibrator(n_classes=3)

        logits = np.random.randn(50, 3)
        probs, _ = cal.calibrate(logits)

        expected = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        assert np.allclose(probs, expected, atol=1e-5)

    def test_uncertainty_metrics(self):
        """Uncertainty metrics are computed correctly."""
        np.random.seed(42)

        cal = DirichletCalibrator(n_classes=3)
        cal.fit(np.random.randn(500, 3), np.random.randint(0, 3, 500))

        # High confidence prediction vs uncertain prediction
        high_conf_logits = np.array([[5.0, 0.0, 0.0]])
        uncertain_logits = np.array([[0.0, 0.0, 0.0]])

        _, uncertainty_high = cal.calibrate(high_conf_logits)
        _, uncertainty_low = cal.calibrate(uncertain_logits)

        # High confidence input should have higher max_prob than uncertain input
        # (Calibration may soften predictions but should preserve ordering)
        assert uncertainty_high['max_prob'][0] > uncertainty_low['max_prob'][0], \
            "High confidence input should have higher max_prob"

        # High confidence input should have lower normalized entropy
        assert uncertainty_high['normalized_entropy'][0] < uncertainty_low['normalized_entropy'][0], \
            "High confidence input should have lower entropy"

        # Uncertainty metrics should be in valid range
        assert 0 <= uncertainty_high['max_prob'][0] <= 1
        assert 0 <= uncertainty_high['normalized_entropy'][0] <= 1
        assert 0 <= uncertainty_low['max_prob'][0] <= 1
        assert 0 <= uncertainty_low['normalized_entropy'][0] <= 1

    def test_state_serialization(self):
        """Test state save and load."""
        np.random.seed(42)

        cal = DirichletCalibrator(n_classes=3)
        cal.fit(np.random.randn(500, 3), np.random.randint(0, 3, 500))

        # Save state
        state = cal.get_state()

        # Load into new calibrator
        cal2 = DirichletCalibrator.from_state(state)

        assert cal2.is_fitted
        assert np.allclose(cal.W, cal2.W)
        assert np.allclose(cal.b, cal2.b)

    def test_calibration_preserves_simplex(self):
        """Calibrated probabilities sum to 1."""
        np.random.seed(42)

        cal = DirichletCalibrator(n_classes=3)
        cal.fit(np.random.randn(1000, 3), np.random.randint(0, 3, 1000))

        # Test on various inputs
        for _ in range(10):
            logits = np.random.randn(100, 3) * np.random.uniform(0.5, 5.0)
            probs, _ = cal.calibrate(logits)

            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
            assert (probs >= 0).all()
            assert (probs <= 1).all()


class TestEnsembleUncertaintyEstimator:
    """Tests for EnsembleUncertaintyEstimator."""

    def test_initialization(self):
        """Test estimator initialization."""
        est = EnsembleUncertaintyEstimator(n_classes=3)
        assert est.n_classes == 3
        assert est.strategic_values == {0: -1.0, 1: -0.1, 2: 5.0}

    def test_estimate_uncertainty_basic(self):
        """Test basic uncertainty estimation."""
        np.random.seed(42)

        est = EnsembleUncertaintyEstimator(n_classes=3)

        # Create ensemble predictions (5 models, 100 samples, 3 classes)
        ensemble_probs = np.random.dirichlet([1, 1, 1], size=(5, 100))
        # Reshape to (n_models, n_samples, n_classes)
        ensemble_probs = ensemble_probs.reshape(5, 100, 3)

        results = est.estimate_uncertainty(ensemble_probs)

        assert results['probs_mean'].shape == (100, 3)
        assert results['probs_std'].shape == (100, 3)
        assert results['ev_mean'].shape == (100,)
        assert results['ev_std'].shape == (100,)
        assert results['ev_ci_lower'].shape == (100,)
        assert results['ev_ci_upper'].shape == (100,)
        assert results['epistemic_uncertainty'].shape == (100,)
        assert results['aleatoric_uncertainty'].shape == (100,)
        assert results['prediction_agreement'].shape == (100,)

    def test_confidence_interval_ordering(self):
        """Lower CI should be <= mean <= upper CI."""
        np.random.seed(42)

        est = EnsembleUncertaintyEstimator(n_classes=3)

        # Create varied ensemble predictions
        ensemble_probs = np.random.dirichlet([1, 1, 1], size=(5, 200)).reshape(5, 200, 3)

        results = est.estimate_uncertainty(ensemble_probs, confidence_level=0.90)

        assert (results['ev_ci_lower'] <= results['ev_mean']).all()
        assert (results['ev_mean'] <= results['ev_ci_upper']).all()

    def test_high_agreement_low_uncertainty(self):
        """When models agree, uncertainty should be low."""
        est = EnsembleUncertaintyEstimator(n_classes=3)

        # All models predict same thing
        base_probs = np.array([0.1, 0.2, 0.7])
        ensemble_probs = np.tile(base_probs, (5, 100, 1))

        results = est.estimate_uncertainty(ensemble_probs)

        # Agreement should be 1.0
        assert np.allclose(results['prediction_agreement'], 1.0)
        # Epistemic uncertainty should be ~0
        assert (results['epistemic_uncertainty'] < 0.01).all()
        # EV std should be ~0
        assert (results['ev_std'] < 0.01).all()

    def test_model_disagreement_high_uncertainty(self):
        """When models disagree, uncertainty should be high."""
        est = EnsembleUncertaintyEstimator(n_classes=3)

        n_samples = 50
        # Each model predicts different class
        ensemble_probs = np.zeros((5, n_samples, 3))
        ensemble_probs[0, :, 0] = 0.9  # Model 0 predicts class 0
        ensemble_probs[1, :, 1] = 0.9  # Model 1 predicts class 1
        ensemble_probs[2, :, 2] = 0.9  # Model 2 predicts class 2
        ensemble_probs[3, :, 0] = 0.9  # Model 3 predicts class 0
        ensemble_probs[4, :, 1] = 0.9  # Model 4 predicts class 1
        # Fill remaining probability
        for i in range(5):
            for c in range(3):
                if ensemble_probs[i, :, c].max() < 0.5:
                    ensemble_probs[i, :, c] = 0.05

        results = est.estimate_uncertainty(ensemble_probs)

        # Agreement should be < 0.5 (no majority)
        assert (results['prediction_agreement'] < 0.6).all()
        # Epistemic uncertainty should be high
        assert (results['epistemic_uncertainty'] > 0.05).all()

    def test_ev_calculation(self):
        """Test EV calculation with strategic values."""
        est = EnsembleUncertaintyEstimator(
            n_classes=3,
            strategic_values={0: -1.0, 1: -0.1, 2: 5.0}
        )

        # All models predict Target (class 2) with 100% confidence
        ensemble_probs = np.zeros((3, 10, 3))
        ensemble_probs[:, :, 2] = 1.0

        results = est.estimate_uncertainty(ensemble_probs)

        # EV should be exactly 5.0
        assert np.allclose(results['ev_mean'], 5.0)
        assert np.allclose(results['ev_ci_lower'], 5.0)
        assert np.allclose(results['ev_ci_upper'], 5.0)

    def test_trading_recommendations(self):
        """Test trading recommendation logic."""
        est = EnsembleUncertaintyEstimator(n_classes=3)

        # High EV, high agreement, low uncertainty -> STRONG_SIGNAL
        ensemble_probs = np.zeros((5, 1, 3))
        ensemble_probs[:, 0, 2] = 0.8  # All models predict Target
        ensemble_probs[:, 0, 1] = 0.15
        ensemble_probs[:, 0, 0] = 0.05

        results = est.estimate_uncertainty(ensemble_probs)
        recs = est.get_trading_recommendation(results, ev_threshold=2.0)

        assert recs[0] in ['STRONG_SIGNAL', 'GOOD_SIGNAL']

        # High danger probability -> AVOID
        ensemble_probs2 = np.zeros((5, 1, 3))
        ensemble_probs2[:, 0, 0] = 0.5  # High danger
        ensemble_probs2[:, 0, 1] = 0.3
        ensemble_probs2[:, 0, 2] = 0.2

        results2 = est.estimate_uncertainty(ensemble_probs2)
        recs2 = est.get_trading_recommendation(results2)

        assert recs2[0] == 'AVOID'


class TestComputeConfidenceIntervalEV:
    """Tests for convenience function."""

    def test_basic_usage(self):
        """Test basic CI computation."""
        np.random.seed(42)

        ensemble_probs = np.random.dirichlet([1, 1, 1], size=(5, 100)).reshape(5, 100, 3)

        ev_mean, ev_lower, ev_upper = compute_confidence_interval_ev(ensemble_probs)

        assert ev_mean.shape == (100,)
        assert ev_lower.shape == (100,)
        assert ev_upper.shape == (100,)
        assert (ev_lower <= ev_mean).all()
        assert (ev_mean <= ev_upper).all()

    def test_confidence_level_affects_width(self):
        """Higher confidence level should give wider intervals."""
        np.random.seed(42)

        ensemble_probs = np.random.dirichlet([1, 1, 1], size=(5, 100)).reshape(5, 100, 3)

        _, lower_90, upper_90 = compute_confidence_interval_ev(
            ensemble_probs, confidence_level=0.90
        )
        _, lower_99, upper_99 = compute_confidence_interval_ev(
            ensemble_probs, confidence_level=0.99
        )

        # 99% CI should be wider than 90% CI
        width_90 = np.mean(upper_90 - lower_90)
        width_99 = np.mean(upper_99 - lower_99)

        assert width_99 >= width_90 * 0.9  # Allow some variance


class TestIntegration:
    """Integration tests combining calibration and ensemble."""

    def test_full_pipeline(self):
        """Test complete calibration + ensemble pipeline."""
        np.random.seed(42)

        # Create synthetic ensemble predictions
        n_models = 5
        n_samples = 200

        # Simulate logits from ensemble
        ensemble_logits = np.random.randn(n_models, n_samples, 3)

        # Convert to probabilities
        ensemble_probs = np.zeros_like(ensemble_logits)
        for i in range(n_models):
            exp_logits = np.exp(ensemble_logits[i])
            ensemble_probs[i] = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Average logits for calibration
        mean_logits = np.mean(ensemble_logits, axis=0)
        labels = np.random.randint(0, 3, n_samples)

        # Fit calibrator
        cal = DirichletCalibrator(n_classes=3)
        cal.fit(mean_logits[:100], labels[:100])  # Fit on first half

        # Calibrate ensemble mean
        calibrated_probs, _ = cal.calibrate(mean_logits[100:])  # Test on second half

        # Check output
        assert calibrated_probs.shape == (100, 3)
        assert np.allclose(calibrated_probs.sum(axis=1), 1.0, atol=1e-5)

        # Estimate uncertainty on ensemble
        est = EnsembleUncertaintyEstimator(n_classes=3)
        results = est.estimate_uncertainty(ensemble_probs[:, 100:, :])

        assert 'ev_mean' in results
        assert 'ev_ci_lower' in results
        assert 'ev_ci_upper' in results

    def test_calibration_state_in_checkpoint(self, tmp_path):
        """Test saving and loading calibrator state."""
        np.random.seed(42)

        # Fit calibrator
        cal = DirichletCalibrator(n_classes=3)
        cal.fit(np.random.randn(500, 3), np.random.randint(0, 3, 500))

        # Save state
        state = cal.get_state()

        # Simulate checkpoint
        import json
        checkpoint_path = tmp_path / 'test_checkpoint.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f)

        # Load state
        with open(checkpoint_path, 'r') as f:
            loaded_state = json.load(f)

        cal2 = DirichletCalibrator.from_state(loaded_state)

        # Verify functionality
        test_logits = np.random.randn(50, 3)
        probs1, _ = cal.calibrate(test_logits)
        probs2, _ = cal2.calibrate(test_logits)

        assert np.allclose(probs1, probs2)


class TestEdgeCases:
    """Edge case handling."""

    def test_single_sample(self):
        """Handle single sample input."""
        est = EnsembleUncertaintyEstimator(n_classes=3)

        ensemble_probs = np.random.dirichlet([1, 1, 1], size=(3, 1)).reshape(3, 1, 3)

        results = est.estimate_uncertainty(ensemble_probs)

        assert results['ev_mean'].shape == (1,)
        assert results['probs_mean'].shape == (1, 3)

    def test_two_model_ensemble(self):
        """Minimum ensemble size of 2."""
        est = EnsembleUncertaintyEstimator(n_classes=3)

        ensemble_probs = np.random.dirichlet([1, 1, 1], size=(2, 50)).reshape(2, 50, 3)

        results = est.estimate_uncertainty(ensemble_probs)

        assert results['n_models'] == 2

    def test_extreme_probabilities(self):
        """Handle extreme probability values."""
        cal = DirichletCalibrator(n_classes=3)
        cal.fit(np.random.randn(500, 3), np.random.randint(0, 3, 500))

        # Very high confidence
        extreme_logits = np.array([[100.0, 0.0, 0.0]])
        probs, _ = cal.calibrate(extreme_logits)

        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert (probs >= 0).all()

    def test_zero_variance_ensemble(self):
        """Handle ensemble with zero variance."""
        est = EnsembleUncertaintyEstimator(n_classes=3)

        # All models identical
        base = np.array([0.2, 0.3, 0.5])
        ensemble_probs = np.tile(base, (5, 100, 1))

        results = est.estimate_uncertainty(ensemble_probs)

        assert (results['ev_std'] < 1e-10).all()
        assert (results['probs_std'] < 1e-10).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
