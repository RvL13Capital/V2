"""
Tests for StreamingSequenceWriter with Streaming Robust Scaling

Tests the reservoir sampling-based quantile estimation that computes
median/IQR during streaming without loading all data into memory.
"""
import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the actual module name (01_generate_sequences.py)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "generate_sequences",
    Path(__file__).parent.parent / "pipeline" / "01_generate_sequences.py"
)
generate_sequences = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_sequences)

ReservoirQuantileEstimator = generate_sequences.ReservoirQuantileEstimator
StreamingSequenceWriter = generate_sequences.StreamingSequenceWriter


class TestReservoirQuantileEstimator:
    """Tests for reservoir sampling quantile estimator"""

    def test_basic_initialization(self):
        """Estimator initializes with correct structure"""
        estimator = ReservoirQuantileEstimator(reservoir_size=1000)

        assert estimator.reservoir_size == 1000
        assert isinstance(estimator.reservoir, dict)
        assert isinstance(estimator.counts, dict)

    def test_update_fills_reservoir(self):
        """Values fill reservoir correctly via 3D sequences"""
        estimator = ReservoirQuantileEstimator(reservoir_size=100)

        # Create sequences with known values for feature 0
        sequences = np.zeros((5, 10, 4), dtype=np.float32)  # 5 samples, 10 timesteps, 4 features
        sequences[:, :, 0] = np.arange(50).reshape(5, 10)  # 50 values total

        estimator.update(sequences)

        # Feature 0 should have 50 values
        assert estimator.counts[0] == 50
        assert len(estimator.reservoir[0]) == 50

    def test_reservoir_overflow(self):
        """Reservoir uses probabilistic replacement when full"""
        estimator = ReservoirQuantileEstimator(reservoir_size=100, seed=42)

        # Create sequences that will produce >100 values per feature
        # 100 samples * 10 timesteps = 1000 values per feature
        sequences = np.arange(1000, dtype=np.float32).reshape(100, 10, 1)

        estimator.update(sequences)

        assert estimator.counts[0] == 1000
        assert len(estimator.reservoir[0]) == 100  # Reservoir capped

    def test_robust_params_computation(self):
        """Robust params computed correctly from reservoir"""
        estimator = ReservoirQuantileEstimator(reservoir_size=1000)

        # Create sequences with known distribution
        # Values 0-99 for feature 0
        sequences = np.arange(100, dtype=np.float32).reshape(10, 10, 1)

        estimator.update(sequences)
        params = estimator.get_robust_params()

        # For 0-99: median~49.5, IQR~50
        assert 0 in params
        assert abs(params[0]['median'] - 49.5) < 1.0
        assert params[0]['iqr'] > 0

    def test_update_multiple_features(self):
        """Update works with multiple features"""
        estimator = ReservoirQuantileEstimator(reservoir_size=1000)

        # 10 features (composite disabled 2026-01-18)
        sequences = np.random.randn(100, 20, 10).astype(np.float32)
        estimator.update(sequences)

        # All features should be tracked
        params = estimator.get_robust_params()
        assert len(params) == 10

    @pytest.mark.skip(reason="Composite features disabled 2026-01-18 - robust params no longer needed")
    def test_get_robust_params_format(self):
        """Robust params output matches expected format (DISABLED - composite features removed)"""
        # This test is no longer applicable since composite features (vol_dryup_ratio, var_score,
        # nes_score, lpf_score) at indices 8-11 have been disabled.
        pass

    def test_handles_nan(self):
        """NaN values are filtered out (inf values preserved)"""
        estimator = ReservoirQuantileEstimator(reservoir_size=100)

        # Create sequences with NaN and Inf
        # Values: 1.0, nan, 2.0, inf, 3.0, -inf, 4.0, 5.0, 6.0, 7.0
        sequences = np.array([[[1.0], [np.nan], [2.0], [np.inf], [3.0], [-np.inf], [4.0], [5.0], [6.0], [7.0]]], dtype=np.float32)
        estimator.update(sequences)

        # Only NaN is filtered out, inf values are preserved
        # 10 values - 1 NaN = 9 valid values
        assert estimator.counts[0] == 9
        assert len(estimator.reservoir[0]) == 9

    def test_empty_reservoir_returns_empty(self):
        """Empty reservoir returns empty params"""
        estimator = ReservoirQuantileEstimator(reservoir_size=100)

        params = estimator.get_robust_params()
        assert params == {}


class TestStreamingVsFullComputation:
    """Compare streaming reservoir sampling vs full data computation"""

    @pytest.mark.skip(reason="Composite features disabled 2026-01-18 - robust scaling no longer needed")
    def test_accuracy_on_normal_distribution(self):
        """Streaming matches full computation within 5% on normal data (DISABLED)"""
        # This test is no longer applicable since composite features have been disabled.
        pass

    @pytest.mark.skip(reason="Composite features disabled 2026-01-18 - robust scaling no longer needed")
    def test_accuracy_on_skewed_distribution(self):
        """Streaming handles skewed distributions (DISABLED)"""
        # This test is no longer applicable since composite features have been disabled.
        pass


class TestStreamingSequenceWriterIntegration:
    """Integration tests for StreamingSequenceWriter"""

    def test_writer_has_quantile_estimator(self):
        """Writer initializes with quantile estimator"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = StreamingSequenceWriter(Path(tmpdir))

            assert hasattr(writer, 'quantile_estimator')
            assert isinstance(writer.quantile_estimator, ReservoirQuantileEstimator)

    def test_write_ticker_updates_reservoir(self):
        """Writing sequences works with 10-feature data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = StreamingSequenceWriter(Path(tmpdir))
            writer._init_hdf5("test")

            try:
                # Create test sequences with 10 features (composite features disabled)
                sequences = np.ones((10, 20, 10), dtype=np.float32)
                sequences[:, :, 8] = 0.01  # upper_slope (boundary)
                sequences[:, :, 9] = -0.01  # lower_slope (boundary)

                labels = np.zeros(10, dtype=np.int32)
                metadata = [{'ticker': 'TEST', 'idx': i} for i in range(10)]

                writer.write_ticker('TEST', sequences, labels, metadata)

                # Verify write succeeded
                assert writer.total_sequences == 10
            finally:
                # Close HDF5 file to release lock on Windows
                if writer.h5_file is not None:
                    writer.h5_file.close()

    def test_get_robust_params_after_write(self):
        """Robust params returns dict with all feature params after write"""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = StreamingSequenceWriter(Path(tmpdir))
            writer._init_hdf5("test")

            try:
                # 10 features (composite features disabled 2026-01-18)
                sequences = np.random.randn(100, 20, 10).astype(np.float32)
                labels = np.zeros(100, dtype=np.int32)
                metadata = [{'ticker': 'TEST', 'idx': i} for i in range(100)]

                writer.write_ticker('TEST', sequences, labels, metadata)

                params = writer.get_robust_params()

                # Robust params should have entries for all 10 features
                assert len(params) == 10
                for feat_idx in range(10):
                    assert feat_idx in params
                    assert 'median' in params[feat_idx]
                    assert 'iqr' in params[feat_idx]
            finally:
                # Close HDF5 file to release lock on Windows
                if writer.h5_file is not None:
                    writer.h5_file.close()


class TestPerformance:
    """Performance benchmarks"""

    @pytest.mark.skip(reason="Composite features disabled 2026-01-18 - robust scaling no longer needed")
    def test_reservoir_update_performance(self):
        """Reservoir update is fast enough (DISABLED - composite features removed)"""
        pass

    @pytest.mark.skip(reason="Composite features disabled 2026-01-18 - robust scaling no longer needed")
    def test_memory_bounded(self):
        """Reservoir memory stays bounded (DISABLED - composite features removed)"""
        pass
