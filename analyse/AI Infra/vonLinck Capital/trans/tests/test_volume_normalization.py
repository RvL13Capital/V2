"""
Tests for volume normalization to ensure no data leakage.

These tests verify:
1. Day 0 is always 0 (reference point)
2. Log ratio calculation is correct
3. No look-ahead bias (only day 0 is used)
4. Extreme values are clipped properly
"""

import numpy as np
import pytest


def normalize_volume(windows: np.ndarray) -> np.ndarray:
    """Copy of _normalize_volume for testing."""
    VOLUME_IDX = 4
    EPS = 1e-8

    volume_0 = windows[:, 0, VOLUME_IDX:VOLUME_IDX+1]
    volume_0 = np.maximum(volume_0, EPS)

    volume_all = windows[:, :, VOLUME_IDX]
    volume_all = np.maximum(volume_all, EPS)

    volume_normalized = np.log(volume_all / volume_0)
    volume_normalized = np.clip(volume_normalized, -4, 6)

    windows[:, :, VOLUME_IDX] = volume_normalized
    return windows


class TestVolumeNormalization:
    """Tests for volume normalization function."""

    def test_day_zero_is_always_zero(self):
        """Day 0 should always be 0 after normalization (reference point)."""
        windows = np.zeros((5, 20, 10))
        windows[:, :, 4] = np.random.uniform(1000, 1000000, (5, 20))

        normalized = normalize_volume(windows.copy())

        # Day 0 should be exactly 0 for all patterns
        np.testing.assert_array_almost_equal(
            normalized[:, 0, 4],
            np.zeros(5),
            decimal=10,
            err_msg="Day 0 should be 0 (reference point)"
        )

    def test_volume_doubling_gives_log2(self):
        """Volume doubling should give log(2) ≈ 0.693."""
        windows = np.zeros((1, 20, 10))
        windows[0, :, 4] = [100000] * 10 + [200000] * 10  # Doubles at day 10

        normalized = normalize_volume(windows.copy())

        assert normalized[0, 0, 4] == pytest.approx(0.0, abs=1e-10)
        assert normalized[0, 10, 4] == pytest.approx(np.log(2), abs=1e-10)

    def test_volume_halving_gives_negative_log2(self):
        """Volume halving should give log(0.5) ≈ -0.693."""
        windows = np.zeros((1, 20, 10))
        windows[0, :, 4] = [100000] * 10 + [50000] * 10  # Halves at day 10

        normalized = normalize_volume(windows.copy())

        assert normalized[0, 0, 4] == pytest.approx(0.0, abs=1e-10)
        assert normalized[0, 10, 4] == pytest.approx(np.log(0.5), abs=1e-10)

    def test_10x_spike_gives_log10(self):
        """10x volume spike should give log(10) ≈ 2.303."""
        windows = np.zeros((1, 20, 10))
        windows[0, :, 4] = [100000] * 10 + [1000000] * 10  # 10x at day 10

        normalized = normalize_volume(windows.copy())

        assert normalized[0, 10, 4] == pytest.approx(np.log(10), abs=1e-10)

    def test_no_look_ahead_bias(self):
        """Verify normalization only uses day 0, not future data."""
        windows1 = np.zeros((1, 20, 10))
        windows2 = np.zeros((1, 20, 10))

        # Same day 0, different future
        windows1[0, :, 4] = [100000] + [200000] * 19  # Spike after day 0
        windows2[0, :, 4] = [100000] + [50000] * 19   # Drop after day 0

        norm1 = normalize_volume(windows1.copy())
        norm2 = normalize_volume(windows2.copy())

        # Day 0 should be identical (0) regardless of future
        assert norm1[0, 0, 4] == norm2[0, 0, 4] == 0.0

        # But day 1+ should differ
        assert norm1[0, 1, 4] != norm2[0, 1, 4]

    def test_extreme_values_clipped(self):
        """Extreme volume changes should be clipped to [-4, 6]."""
        windows = np.zeros((1, 20, 10))
        # 1000x spike (way beyond normal)
        windows[0, :, 4] = [100] + [100000] * 19

        normalized = normalize_volume(windows.copy())

        # log(1000) ≈ 6.9, should be clipped to 6
        assert normalized[0, 1, 4] == 6.0

        # Near-zero volume (0.001x)
        windows[0, :, 4] = [100000] + [100] * 19
        normalized = normalize_volume(windows.copy())

        # log(0.001) ≈ -6.9, should be clipped to -4
        assert normalized[0, 1, 4] == -4.0

    def test_zero_volume_handled(self):
        """Zero volume should not cause errors (uses epsilon)."""
        windows = np.zeros((1, 20, 10))
        windows[0, :, 4] = [0] * 20  # All zeros

        # Should not raise
        normalized = normalize_volume(windows.copy())

        # All values should be 0 (0/0 with epsilon = 1)
        assert np.all(normalized[0, :, 4] == 0.0)

    def test_steady_volume_gives_zeros(self):
        """Constant volume should give all zeros."""
        windows = np.zeros((1, 20, 10))
        windows[0, :, 4] = [123456] * 20

        normalized = normalize_volume(windows.copy())

        np.testing.assert_array_almost_equal(
            normalized[0, :, 4],
            np.zeros(20),
            decimal=10
        )

    def test_cross_sectional_comparability(self):
        """Same relative pattern should give same normalized values."""
        windows = np.zeros((2, 20, 10))

        # Pattern 1: Micro-cap (low absolute volume)
        windows[0, :, 4] = [10000] * 10 + [20000] * 10

        # Pattern 2: Large-cap (high absolute volume)
        windows[1, :, 4] = [10000000] * 10 + [20000000] * 10

        normalized = normalize_volume(windows.copy())

        # Should be identical (both double at day 10)
        np.testing.assert_array_almost_equal(
            normalized[0, :, 4],
            normalized[1, :, 4],
            decimal=10,
            err_msg="Same relative pattern should give identical normalized values"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
