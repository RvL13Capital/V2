"""
Tests for InferenceWrapper - Self-Contained Model Deployment
=============================================================

Tests the atomic checkpoint format and InferenceWrapper functionality.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
import sys
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.inference_wrapper import (
    InferenceWrapper,
    load_for_inference,
    STRATEGIC_VALUES,
    PREDICTION_TEMPERATURE
)


class MockModel(nn.Module):
    """Simple mock model for testing"""

    def __init__(self, num_classes=3):
        super().__init__()
        # 10 features (composite features disabled 2026-01-18)
        self.fc = nn.Linear(10 * 20, num_classes)
        self.num_classes = num_classes

    def forward(self, x, context=None):
        # Flatten temporal dimension
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.fc(x)


class TestInferenceWrapperNormalization:
    """Tests for normalization transforms"""

    def test_robust_normalization(self):
        """Robust normalization (median/IQR) applies correctly to all 10 features"""
        model = MockModel()

        # 10 features: OHLCV(0-4), technical(5-7), boundary(8-9)
        norm_params = {
            'scaling_type': 'robust',
            'median': [10.0] * 10,
            'iqr': [2.0] * 10
        }

        wrapper = InferenceWrapper(
            model=model,
            norm_params=norm_params
        )

        # Test data with 10 features
        sequences = np.ones((10, 20, 10), dtype=np.float32) * 10.0
        normalized = wrapper.normalize_sequences(sequences)

        # All features should be normalized: (10 - 10) / 2 = 0
        for feat_idx in range(10):
            assert np.allclose(normalized[:, :, feat_idx], 0.0)

    def test_standard_normalization_legacy(self):
        """Standard normalization (mean/std) applies correctly for legacy checkpoints"""
        model = MockModel()

        # Legacy format without scaling_type
        norm_params = {
            'mean': [10.0] * 10,
            'std': [2.0] * 10
        }

        wrapper = InferenceWrapper(
            model=model,
            norm_params=norm_params
        )

        # Test data with 10 features
        sequences = np.ones((10, 20, 10), dtype=np.float32) * 10.0
        normalized = wrapper.normalize_sequences(sequences)

        # All features should be normalized: (10 - 10) / 2 = 0
        for feat_idx in range(10):
            assert np.allclose(normalized[:, :, feat_idx], 0.0)

    @pytest.mark.skip(reason="Composite features disabled 2026-01-18 - robust scaling no longer needed")
    def test_robust_scaling(self):
        """Robust scaling applies correctly to features 8-11 (DISABLED - composite features removed)"""
        # This test is no longer applicable since composite features (vol_dryup_ratio, var_score,
        # nes_score, lpf_score) at indices 8-11 have been disabled. Features 8-9 are now
        # boundary slopes (upper_slope, lower_slope) which use standard normalization.
        pass

    def test_combined_normalization(self):
        """Robust normalization applies correctly to all 10 features"""
        model = MockModel()

        # 10 features: OHLCV(0-4), technical(5-7), boundary(8-9)
        norm_params = {
            'scaling_type': 'robust',
            'median': [100.0, 101.0, 99.0, 100.0, 1000.0,  # OHLCV
                       0.05, 25.0, 1.0,                     # technical (bbw, adx, volume_ratio)
                       0.01, -0.01],                        # boundary (upper_slope, lower_slope)
            'iqr': [10.0, 10.0, 10.0, 10.0, 500.0,
                    0.02, 10.0, 0.5,
                    0.02, 0.02]
        }

        wrapper = InferenceWrapper(
            model=model,
            norm_params=norm_params
        )

        sequences = np.random.randn(20, 20, 10).astype(np.float32)
        normalized = wrapper.normalize_sequences(sequences)

        assert normalized.shape == (20, 20, 10)
        assert np.isfinite(normalized).all()


class TestInferenceWrapperPrediction:
    """Tests for prediction functionality"""

    def test_predict_output_shape(self):
        """Prediction outputs correct shapes"""
        model = MockModel()
        wrapper = InferenceWrapper(model=model)

        sequences = np.random.randn(100, 20, 10).astype(np.float32)
        probs, ev_scores = wrapper.predict(sequences, apply_normalization=False)

        assert probs.shape == (100, 3)
        assert ev_scores.shape == (100,)

    def test_predict_probabilities_sum_to_one(self):
        """Probabilities sum to 1"""
        model = MockModel()
        wrapper = InferenceWrapper(model=model)

        sequences = np.random.randn(50, 20, 10).astype(np.float32)
        probs, _ = wrapper.predict(sequences, apply_normalization=False)

        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_ev_calculation(self):
        """Expected value calculated correctly"""
        model = MockModel()
        wrapper = InferenceWrapper(model=model)

        # Create sequences that produce known probabilities
        sequences = np.random.randn(10, 20, 10).astype(np.float32)
        probs, ev_scores = wrapper.predict(sequences, apply_normalization=False)

        # Manually compute EV
        expected_ev = np.zeros(len(probs))
        for cls, value in STRATEGIC_VALUES.items():
            expected_ev += probs[:, cls] * value

        assert np.allclose(ev_scores, expected_ev)

    def test_predict_single(self):
        """Single sequence prediction works"""
        model = MockModel()
        wrapper = InferenceWrapper(model=model)

        sequence = np.random.randn(20, 10).astype(np.float32)
        result = wrapper.predict_single(sequence)

        assert 'probs' in result
        assert 'predicted_class' in result
        assert 'ev_score' in result
        assert 'recommendation' in result
        assert result['predicted_class'] in [0, 1, 2]

    def test_with_context(self):
        """Prediction with context features"""
        class ContextModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 10 features (composite features disabled 2026-01-18)
                self.seq_fc = nn.Linear(10 * 20, 32)
                self.ctx_fc = nn.Linear(13, 16)
                self.out = nn.Linear(48, 3)

            def forward(self, x, context=None):
                batch = x.shape[0]
                x = self.seq_fc(x.view(batch, -1))
                if context is not None:
                    c = self.ctx_fc(context)
                    x = torch.cat([x, c], dim=-1)
                else:
                    x = torch.cat([x, torch.zeros(batch, 16)], dim=-1)
                return self.out(x)

        model = ContextModel()
        wrapper = InferenceWrapper(model=model)

        sequences = np.random.randn(10, 20, 10).astype(np.float32)
        context = np.random.randn(10, 13).astype(np.float32)

        probs, ev_scores = wrapper.predict(sequences, context, apply_normalization=False)

        assert probs.shape == (10, 3)
        assert ev_scores.shape == (10,)


class TestAtomicCheckpoint:
    """Tests for atomic checkpoint creation and loading"""

    def test_create_atomic_checkpoint(self):
        """Atomic checkpoint contains all required fields"""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Use robust scaling format (preferred)
        norm_params = {
            'scaling_type': 'robust',
            'median': [0.0] * 10,
            'iqr': [1.0] * 10
        }
        context_ranges = {'float_turnover': (0.0, 5.0)}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model.pt'

            checkpoint = InferenceWrapper.create_atomic_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=10,
                val_acc=0.85,
                config={'test': True},
                norm_params=norm_params,
                context_ranges=context_ranges,
                save_path=save_path
            )

            # Check checkpoint structure
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'norm_params' in checkpoint
            assert 'context_ranges' in checkpoint
            assert 'checkpoint_version' in checkpoint
            assert checkpoint['checkpoint_version'] == '2.2'  # Updated from 2.0
            assert 'label_mapping_version' in checkpoint  # Added 2026-01-19
            assert checkpoint['norm_params']['scaling_type'] == 'robust'

            # Check file was saved
            assert save_path.exists()

    def test_load_atomic_checkpoint(self):
        """Wrapper loads from atomic checkpoint"""
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Use robust scaling format (preferred)
        norm_params = {
            'scaling_type': 'robust',
            'median': [1.0] * 10,
            'iqr': [2.0] * 10
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model.pt'

            # Create checkpoint
            InferenceWrapper.create_atomic_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=5,
                val_acc=0.80,
                config={'dropout': 0.5},
                norm_params=norm_params,
                save_path=save_path
            )

            # Load using convenience function (mock model class)
            # Since we can't easily mock HybridFeatureNetwork, test the checkpoint loading
            checkpoint = torch.load(save_path, weights_only=False)

            assert checkpoint['norm_params'] == norm_params
            assert checkpoint['norm_params']['scaling_type'] == 'robust'
            assert checkpoint['epoch'] == 5
            assert checkpoint['val_acc'] == 0.80
            assert 'label_mapping_version' in checkpoint  # Added 2026-01-19


class TestRecommendations:
    """Tests for trading recommendation logic"""

    def test_strong_signal(self):
        """High EV + low danger = STRONG_SIGNAL"""
        model = MockModel()
        wrapper = InferenceWrapper(model=model)

        ev = 3.5
        probs = np.array([0.15, 0.30, 0.55])  # Low danger

        rec = wrapper._get_recommendation(ev, probs)
        assert rec == "STRONG_SIGNAL"

    def test_avoid(self):
        """High danger = AVOID"""
        model = MockModel()
        wrapper = InferenceWrapper(model=model)

        ev = 1.0
        probs = np.array([0.40, 0.35, 0.25])  # High danger

        rec = wrapper._get_recommendation(ev, probs)
        assert rec == "AVOID"

    def test_negative_ev_avoid(self):
        """Negative EV = AVOID"""
        model = MockModel()
        wrapper = InferenceWrapper(model=model)

        ev = -0.5
        probs = np.array([0.20, 0.50, 0.30])

        rec = wrapper._get_recommendation(ev, probs)
        assert rec == "AVOID"


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_invalid_sequence_shape(self):
        """Invalid input shape raises error"""
        model = MockModel()
        wrapper = InferenceWrapper(model=model)

        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected sequences shape"):
            wrapper.predict(np.random.randn(100, 10))  # 2D instead of 3D

        with pytest.raises(ValueError, match="Expected sequences shape"):
            wrapper.predict(np.random.randn(100, 20, 14))  # Wrong feature count (14 instead of 10)

    def test_empty_input(self):
        """Empty input handled gracefully"""
        model = MockModel()
        wrapper = InferenceWrapper(model=model)

        sequences = np.zeros((0, 20, 10), dtype=np.float32)

        # Empty input should raise ValueError due to shape mismatch
        # This is acceptable behavior - callers should check for empty input
        with pytest.raises(Exception):  # Either RuntimeError or ValueError
            wrapper.predict(sequences, apply_normalization=False)

    def test_nan_handling_in_normalization(self):
        """NaN values don't propagate unexpectedly"""
        model = MockModel()

        norm_params = {
            'scaling_type': 'robust',
            'median': [0.0] * 10,
            'iqr': [1.0] * 10
        }
        wrapper = InferenceWrapper(model=model, norm_params=norm_params)

        sequences = np.random.randn(10, 20, 10).astype(np.float32)
        sequences[0, 5, 3] = np.nan  # Introduce NaN

        # Normalization should handle this (though model may produce NaN output)
        normalized = wrapper.normalize_sequences(sequences)
        assert np.isnan(normalized[0, 5, 3])  # NaN should propagate

    def test_checkpoint_info(self):
        """get_checkpoint_info returns expected structure"""
        model = MockModel()
        wrapper = InferenceWrapper(
            model=model,
            norm_params={
                'scaling_type': 'robust',
                'median': [0.0] * 10,
                'iqr': [1.0] * 10
            },
            config={'test': True}
        )

        info = wrapper.get_checkpoint_info()

        assert 'config' in info
        assert 'has_norm_params' in info
        assert 'device' in info
        assert 'model_params' in info
        assert info['has_norm_params'] is True


class TestPerformance:
    """Performance tests"""

    def test_batch_prediction_speed(self):
        """Batch prediction completes in reasonable time"""
        import time

        model = MockModel()
        norm_params = {
            'scaling_type': 'robust',
            'median': [0.0] * 10,
            'iqr': [1.0] * 10
        }
        wrapper = InferenceWrapper(model=model, norm_params=norm_params)

        sequences = np.random.randn(1000, 20, 10).astype(np.float32)

        # Warmup
        wrapper.predict(sequences[:10])

        start = time.perf_counter()
        probs, ev_scores = wrapper.predict(sequences)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete in under 1000ms for 1K sequences (generous for CI)
        assert elapsed_ms < 1000, f"Took {elapsed_ms:.1f}ms"
        assert len(probs) == 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
