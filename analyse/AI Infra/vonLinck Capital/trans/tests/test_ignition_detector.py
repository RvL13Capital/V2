"""
Tests for Vectorized Ignition Detection
"""
import pytest
import numpy as np
from features.ignition_detector import (
    detect_sequence_ignition,
    detect_ignition_at_timestep,
    get_ignition_statistics,
    IgnitionResult,
    IGNITION_THRESHOLD,
    CLOSE_IDX,
    UPPER_BOUND_IDX,
    LOWER_BOUND_IDX,
    VOLUME_RATIO_IDX,
    ADX_IDX,
    OPEN_IDX,
)


class TestDetectSequenceIgnition:
    """Tests for main ignition detection function"""

    def test_basic_shape(self):
        """Output shapes match input batch size"""
        sequences = np.random.randn(100, 20, 10).astype(np.float32)
        scores, flags = detect_sequence_ignition(sequences)

        assert scores.shape == (100,)
        assert flags.shape == (100,)
        assert scores.dtype == np.float32
        assert flags.dtype == bool

    def test_score_bounds(self):
        """Scores are bounded [0, 1]"""
        sequences = np.random.randn(1000, 20, 10).astype(np.float32)
        scores, _ = detect_sequence_ignition(sequences)

        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_threshold_consistency(self):
        """Flags match threshold applied to scores"""
        sequences = np.random.randn(500, 20, 10).astype(np.float32)
        threshold = 0.5

        scores, flags = detect_sequence_ignition(sequences, threshold=threshold)

        expected_flags = scores >= threshold
        np.testing.assert_array_equal(flags, expected_flags)

    def test_return_components(self):
        """Component breakdown returned when requested"""
        sequences = np.random.randn(100, 20, 10).astype(np.float32)
        result = detect_sequence_ignition(sequences, return_components=True)

        assert isinstance(result, IgnitionResult)
        assert 'price_position' in result.components
        assert 'volume_spike' in result.components
        assert 'bbw_expansion' in result.components
        assert 'green_bar' in result.components
        assert 'adx_momentum' in result.components

        # Component shapes
        for name, values in result.components.items():
            assert values.shape == (100,), f"{name} has wrong shape"

    def test_high_position_high_score(self):
        """Price at upper boundary should increase score"""
        sequences = np.zeros((2, 20, 10), dtype=np.float32)

        # Sequence 0: Price at bottom of channel
        sequences[0, :, CLOSE_IDX] = 0.0
        sequences[0, :, UPPER_BOUND_IDX] = 1.0
        sequences[0, :, LOWER_BOUND_IDX] = 0.0

        # Sequence 1: Price at top of channel
        sequences[1, :, CLOSE_IDX] = 1.0
        sequences[1, :, UPPER_BOUND_IDX] = 1.0
        sequences[1, :, LOWER_BOUND_IDX] = 0.0

        result = detect_sequence_ignition(sequences, return_components=True)

        # High position should have higher price_position score
        assert result.components['price_position'][1] > result.components['price_position'][0]

    def test_volume_spike_effect(self):
        """High volume ratio should increase score"""
        sequences = np.zeros((2, 20, 10), dtype=np.float32)

        # Sequence 0: Normal volume (ratio = 1.0)
        sequences[0, :, VOLUME_RATIO_IDX] = 1.0

        # Sequence 1: High volume (ratio = 2.5)
        sequences[1, :, VOLUME_RATIO_IDX] = 2.5

        result = detect_sequence_ignition(sequences, return_components=True)

        assert result.components['volume_spike'][1] > result.components['volume_spike'][0]

    def test_green_bar_detection(self):
        """Green bars (close > open) should be detected"""
        sequences = np.zeros((2, 20, 10), dtype=np.float32)

        # Sequence 0: Red bar (close < open)
        sequences[0, -1, OPEN_IDX] = 1.0
        sequences[0, -1, CLOSE_IDX] = 0.5

        # Sequence 1: Green bar (close > open)
        sequences[1, -1, OPEN_IDX] = 0.5
        sequences[1, -1, CLOSE_IDX] = 1.0

        result = detect_sequence_ignition(sequences, return_components=True)

        assert result.components['green_bar'][0] == 0.0
        assert result.components['green_bar'][1] == 1.0

    def test_invalid_shape_raises(self):
        """Wrong input shape raises ValueError"""
        with pytest.raises(ValueError, match="3D tensor"):
            detect_sequence_ignition(np.random.randn(100, 10))  # 2D

        with pytest.raises(ValueError, match="10 features"):
            detect_sequence_ignition(np.random.randn(100, 20, 14))  # Wrong features (14 instead of 10)

    def test_empty_input(self):
        """Empty input handled gracefully"""
        sequences = np.zeros((0, 20, 10), dtype=np.float32)
        scores, flags = detect_sequence_ignition(sequences)

        assert len(scores) == 0
        assert len(flags) == 0


class TestDetectIgnitionAtTimestep:
    """Tests for timestep-specific detection"""

    def test_basic_functionality(self):
        """Can detect at specific timesteps"""
        sequences = np.random.randn(100, 20, 10).astype(np.float32)

        scores, flags = detect_ignition_at_timestep(sequences, timestep=15)
        assert scores.shape == (100,)
        assert flags.shape == (100,)

    def test_early_timestep_returns_zeros(self):
        """Very early timesteps return zero scores"""
        sequences = np.random.randn(100, 20, 10).astype(np.float32)

        scores, flags = detect_ignition_at_timestep(sequences, timestep=2)
        np.testing.assert_array_equal(scores, 0)
        np.testing.assert_array_equal(flags, False)

    def test_negative_indexing(self):
        """Negative timestep indices work"""
        sequences = np.random.randn(100, 20, 10).astype(np.float32)

        scores_neg, flags_neg = detect_ignition_at_timestep(sequences, timestep=-1)
        scores_pos, flags_pos = detect_ignition_at_timestep(sequences, timestep=19)

        # Results should be similar (though not identical due to implementation)
        assert abs(scores_neg.mean() - scores_pos.mean()) < 0.2


class TestGetIgnitionStatistics:
    """Tests for statistics calculation"""

    def test_basic_stats(self):
        """Basic statistics are computed"""
        sequences = np.random.randn(100, 20, 10).astype(np.float32)
        stats = get_ignition_statistics(sequences)

        assert 'n_sequences' in stats
        assert 'n_ignition' in stats
        assert 'ignition_rate' in stats
        assert 'mean_score' in stats
        assert stats['n_sequences'] == 100

    def test_with_labels(self):
        """Class breakdown computed with labels"""
        sequences = np.random.randn(100, 20, 10).astype(np.float32)
        labels = np.random.randint(0, 3, size=100)

        stats = get_ignition_statistics(sequences, labels)

        assert 'by_class' in stats
        assert 'target_lift' in stats


class TestPerformance:
    """Performance benchmarks"""

    def test_10k_under_100ms(self):
        """10K sequences processed in under 100ms"""
        import time

        sequences = np.random.randn(10000, 20, 10).astype(np.float32)

        # Warmup
        detect_sequence_ignition(sequences)

        # Timed run
        start = time.perf_counter()
        detect_sequence_ignition(sequences)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 100, f"Took {elapsed:.1f}ms, expected < 100ms"

    def test_large_batch(self):
        """Can handle large batches"""
        sequences = np.random.randn(50000, 20, 10).astype(np.float32)
        scores, flags = detect_sequence_ignition(sequences)

        assert len(scores) == 50000
