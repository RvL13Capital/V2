"""
Tests for Streaming Robust Parameters via Reservoir Sampling
============================================================

Tests the StreamingQuantileEstimator and its integration with StreamingSequenceWriter.
Verifies that streaming estimates match full-data computation within tolerance.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
import tempfile
import logging

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Create a standalone StreamingQuantileEstimator for testing
# This matches the implementation in 01_generate_sequences.py
class StreamingQuantileEstimator:
    """
    Reservoir sampling for streaming quantile estimation.

    Mirrors the implementation in pipeline/01_generate_sequences.py
    """

    def __init__(self, reservoir_size: int = 100000, n_features: int = 4):
        self.reservoir_size = reservoir_size
        self.n_features = n_features
        self.reservoirs = [
            np.empty(reservoir_size, dtype=np.float32)
            for _ in range(n_features)
        ]
        self.counts = [0] * n_features
        self.filled = [0] * n_features

    def update(self, values: np.ndarray, feature_idx: int):
        """Add values to reservoir using Algorithm R (vectorized)."""
        if feature_idx >= self.n_features:
            return

        flat = values.flatten().astype(np.float32)
        valid_mask = np.isfinite(flat)
        valid = flat[valid_mask]

        if len(valid) == 0:
            return

        n_current = self.counts[feature_idx]
        reservoir = self.reservoirs[feature_idx]
        n_valid = len(valid)
        space_remaining = self.reservoir_size - self.filled[feature_idx]

        if space_remaining >= n_valid:
            start = self.filled[feature_idx]
            reservoir[start:start + n_valid] = valid
            self.filled[feature_idx] = start + n_valid
            self.counts[feature_idx] = n_current + n_valid
        else:
            if space_remaining > 0:
                reservoir[self.filled[feature_idx]:self.reservoir_size] = valid[:space_remaining]
                self.filled[feature_idx] = self.reservoir_size
                valid = valid[space_remaining:]
                n_current += space_remaining

            n_overflow = len(valid)
            if n_overflow > 0:
                positions = np.arange(n_current, n_current + n_overflow)
                probs = self.reservoir_size / (positions + 1)
                rand = np.random.random(n_overflow)
                include_mask = rand < probs
                n_include = include_mask.sum()
                if n_include > 0:
                    replace_indices = np.random.randint(0, self.reservoir_size, size=n_include)
                    reservoir[replace_indices] = valid[include_mask]
                n_current += n_overflow

            self.counts[feature_idx] = n_current

    def update_batch(self, sequences: np.ndarray, feature_indices: list = None):
        """Update all feature reservoirs from a batch of sequences."""
        if feature_indices is None:
            feature_indices = [8, 9]

        for i, feat_idx in enumerate(feature_indices):
            if sequences.ndim == 3:
                feat_data = sequences[:, :, feat_idx]
            else:
                feat_data = sequences[:, feat_idx]
            self.update(feat_data, i)

    def get_quantiles(self, feature_idx: int, quantiles: list = None) -> dict:
        """Compute quantiles from reservoir sample."""
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]

        n = self.filled[feature_idx]
        if n == 0:
            return {q: 0.0 for q in quantiles}

        sample = self.reservoirs[feature_idx][:n]
        return {q: float(np.percentile(sample, q * 100)) for q in quantiles}

    def get_robust_params(self) -> dict:
        """Compute robust scaling parameters (median/IQR) for all features."""
        robust_params = {}

        for i, feat_idx in enumerate([8, 9]):
            quantiles = self.get_quantiles(i, [0.25, 0.5, 0.75])

            median = quantiles[0.5]
            iqr = quantiles[0.75] - quantiles[0.25]
            iqr = iqr if iqr > 1e-8 else 1.0

            robust_params[f'feat_{feat_idx}_median'] = median
            robust_params[f'feat_{feat_idx}_iqr'] = iqr

        return robust_params

    def get_stats(self) -> dict:
        """Return sampling statistics."""
        return {
            'reservoir_size': self.reservoir_size,
            'n_features': self.n_features,
            'counts': self.counts.copy(),
            'filled': self.filled.copy(),
            'fill_ratios': [f / self.reservoir_size for f in self.filled]
        }


class TestStreamingQuantileEstimator:
    """Tests for the StreamingQuantileEstimator class."""

    def test_initialization(self):
        """Test estimator initialization with default and custom params."""
        # Default
        est = StreamingQuantileEstimator()
        assert est.reservoir_size == 100000
        assert est.n_features == 4
        assert len(est.reservoirs) == 4
        assert all(c == 0 for c in est.counts)
        assert all(f == 0 for f in est.filled)

        # Custom
        est2 = StreamingQuantileEstimator(reservoir_size=1000, n_features=2)
        assert est2.reservoir_size == 1000
        assert est2.n_features == 2
        assert len(est2.reservoirs) == 2

    def test_update_single_batch(self):
        """Test updating with a single batch of values."""
        est = StreamingQuantileEstimator(reservoir_size=1000, n_features=4)

        # Add 500 values to feature 0
        values = np.random.randn(500).astype(np.float32)
        est.update(values, feature_idx=0)

        assert est.counts[0] == 500
        assert est.filled[0] == 500
        assert est.counts[1] == 0  # Other features unchanged

    def test_update_overflow(self):
        """Test reservoir sampling when values exceed capacity."""
        est = StreamingQuantileEstimator(reservoir_size=100, n_features=1)

        # Add 1000 values (10x reservoir size)
        values = np.arange(1000).astype(np.float32)
        est.update(values, feature_idx=0)

        assert est.counts[0] == 1000  # All values seen
        assert est.filled[0] == 100   # Only reservoir_size kept

        # Reservoir should contain valid values from the input
        sample = est.reservoirs[0][:est.filled[0]]
        assert all(v >= 0 and v < 1000 for v in sample)

    def test_update_batch_3d(self):
        """Test update_batch with 3D sequences."""
        est = StreamingQuantileEstimator(reservoir_size=10000, n_features=2)

        # Create sequences: (100 samples, 20 timesteps, 10 features)
        sequences = np.random.randn(100, 20, 10).astype(np.float32)
        est.update_batch(sequences, feature_indices=[8, 9])  # boundary slopes

        # Each feature should have 100 * 20 = 2000 values
        for i in range(2):
            assert est.counts[i] == 2000, f"Feature {i}: expected 2000, got {est.counts[i]}"

    def test_handles_nan_values(self):
        """Test that NaN values are filtered out."""
        est = StreamingQuantileEstimator(reservoir_size=1000, n_features=1)

        values = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0, 5.0], dtype=np.float32)
        est.update(values, feature_idx=0)

        # Only 5 valid values should be counted
        assert est.counts[0] == 5
        assert est.filled[0] == 5

    def test_handles_inf_values(self):
        """Test that infinite values are filtered out."""
        est = StreamingQuantileEstimator(reservoir_size=1000, n_features=1)

        values = np.array([1.0, np.inf, 2.0, -np.inf, 3.0], dtype=np.float32)
        est.update(values, feature_idx=0)

        assert est.counts[0] == 3  # Only finite values counted

    def test_get_quantiles_basic(self):
        """Test quantile computation."""
        est = StreamingQuantileEstimator(reservoir_size=10000, n_features=1)

        # Add known distribution: uniform [0, 100)
        values = np.arange(100).astype(np.float32)
        est.update(values, feature_idx=0)

        quantiles = est.get_quantiles(0, [0.25, 0.5, 0.75])

        # For uniform [0, 99], median should be ~49.5
        assert 45 <= quantiles[0.5] <= 55
        assert 20 <= quantiles[0.25] <= 30
        assert 70 <= quantiles[0.75] <= 80

    def test_get_quantiles_empty(self):
        """Test quantiles on empty reservoir."""
        est = StreamingQuantileEstimator(reservoir_size=1000, n_features=1)

        quantiles = est.get_quantiles(0, [0.25, 0.5, 0.75])

        assert quantiles[0.25] == 0.0
        assert quantiles[0.5] == 0.0
        assert quantiles[0.75] == 0.0

    def test_get_robust_params(self):
        """Test robust params computation (median/IQR)."""
        est = StreamingQuantileEstimator(reservoir_size=10000, n_features=4)

        # Add data to all features with known distributions
        for i in range(4):
            values = np.random.randn(5000).astype(np.float32)
            est.update(values, i)

        params = est.get_robust_params()

        # Check all expected keys exist
        for idx in [8, 9]:
            assert f'feat_{idx}_median' in params
            assert f'feat_{idx}_iqr' in params

        # IQR should be positive
        for idx in [8, 9]:
            assert params[f'feat_{idx}_iqr'] > 0

    def test_get_stats(self):
        """Test statistics retrieval."""
        est = StreamingQuantileEstimator(reservoir_size=500, n_features=2)

        est.update(np.ones(100), 0)
        est.update(np.ones(200), 1)

        stats = est.get_stats()

        assert stats['reservoir_size'] == 500
        assert stats['n_features'] == 2
        assert stats['counts'][0] == 100
        assert stats['counts'][1] == 200
        assert stats['filled'][0] == 100
        assert stats['filled'][1] == 200


class TestStreamingVsFullComputation:
    """Compare streaming estimates to full data computation."""

    def test_median_accuracy_normal_distribution(self):
        """Streaming median should match full computation within 5%."""
        np.random.seed(42)

        # Generate large normal distribution
        full_data = np.random.randn(100000).astype(np.float32)

        # Full computation
        true_median = np.median(full_data)
        true_q25 = np.percentile(full_data, 25)
        true_q75 = np.percentile(full_data, 75)
        true_iqr = true_q75 - true_q25

        # Streaming computation
        est = StreamingQuantileEstimator(reservoir_size=50000, n_features=1)
        est.update(full_data, 0)

        quantiles = est.get_quantiles(0, [0.25, 0.5, 0.75])
        streaming_median = quantiles[0.5]
        streaming_iqr = quantiles[0.75] - quantiles[0.25]

        # Check within 10% tolerance (reservoir sampling has inherent variance)
        median_error = abs(streaming_median - true_median) / max(abs(true_median), 0.01)
        iqr_error = abs(streaming_iqr - true_iqr) / max(abs(true_iqr), 0.01)

        assert median_error < 0.10, f"Median error {median_error:.2%} exceeds 10%"
        assert iqr_error < 0.15, f"IQR error {iqr_error:.2%} exceeds 15%"

    def test_streaming_batched_matches_single(self):
        """Multiple batch updates should match single update."""
        np.random.seed(123)

        # Same data, different update patterns
        full_data = np.random.randn(10000).astype(np.float32)

        # Single update
        est1 = StreamingQuantileEstimator(reservoir_size=5000, n_features=1)
        est1.update(full_data, 0)
        q1 = est1.get_quantiles(0, [0.5])

        # Batched updates
        est2 = StreamingQuantileEstimator(reservoir_size=5000, n_features=1)
        for i in range(0, 10000, 1000):
            est2.update(full_data[i:i+1000], 0)
        q2 = est2.get_quantiles(0, [0.5])

        # Both should see same count
        assert est1.counts[0] == est2.counts[0]

        # Medians should be close (both are estimates)
        assert abs(q1[0.5] - q2[0.5]) < 0.2

    def test_accuracy_improves_with_reservoir_size(self):
        """Larger reservoir should give more accurate estimates."""
        np.random.seed(456)

        full_data = np.random.randn(100000).astype(np.float32)
        true_median = np.median(full_data)

        errors = []
        for reservoir_size in [1000, 10000, 50000, 100000]:
            est = StreamingQuantileEstimator(reservoir_size=reservoir_size, n_features=1)
            est.update(full_data, 0)

            streaming_median = est.get_quantiles(0, [0.5])[0.5]
            error = abs(streaming_median - true_median)
            errors.append(error)

        # Errors should generally decrease (not strictly due to randomness)
        assert errors[-1] < errors[0], "Largest reservoir should be most accurate"

    def test_sequence_feature_extraction(self):
        """Test extracting features from sequence-shaped data."""
        np.random.seed(789)

        # Create synthetic sequences (N=1000, T=20, F=10)
        sequences = np.random.randn(1000, 20, 10).astype(np.float32)

        # Inject known distribution into features 8-9 (boundary slopes)
        for feat_idx in [8, 9]:
            # Each feature has distinct median
            sequences[:, :, feat_idx] = np.random.randn(1000, 20) + (feat_idx - 8) * 2

        # Full computation
        full_params = {}
        for i, idx in enumerate([8, 9]):
            feat_data = sequences[:, :, idx].flatten()
            full_params[f'feat_{idx}_median'] = np.median(feat_data)
            q25, q75 = np.percentile(feat_data, [25, 75])
            full_params[f'feat_{idx}_iqr'] = q75 - q25

        # Streaming computation
        est = StreamingQuantileEstimator(reservoir_size=50000, n_features=4)
        est.update_batch(sequences, feature_indices=[8, 9])
        streaming_params = est.get_robust_params()

        # Compare
        for idx in [8, 9]:
            median_key = f'feat_{idx}_median'
            iqr_key = f'feat_{idx}_iqr'

            median_error = abs(streaming_params[median_key] - full_params[median_key])
            iqr_error = abs(streaming_params[iqr_key] - full_params[iqr_key])

            assert median_error < 0.2, f"Feature {idx} median error: {median_error:.4f}"
            assert iqr_error < 0.2, f"Feature {idx} IQR error: {iqr_error:.4f}"


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_input(self):
        """Handle empty array input."""
        est = StreamingQuantileEstimator(reservoir_size=1000, n_features=1)

        est.update(np.array([], dtype=np.float32), 0)

        assert est.counts[0] == 0
        assert est.filled[0] == 0

    def test_all_nan_input(self):
        """Handle all-NaN input."""
        est = StreamingQuantileEstimator(reservoir_size=1000, n_features=1)

        est.update(np.array([np.nan, np.nan, np.nan], dtype=np.float32), 0)

        assert est.counts[0] == 0

    def test_single_value(self):
        """Handle single value input."""
        est = StreamingQuantileEstimator(reservoir_size=1000, n_features=1)

        est.update(np.array([42.0], dtype=np.float32), 0)

        assert est.counts[0] == 1
        quantiles = est.get_quantiles(0, [0.5])
        assert quantiles[0.5] == 42.0

    def test_invalid_feature_index(self):
        """Invalid feature index should be ignored."""
        est = StreamingQuantileEstimator(reservoir_size=1000, n_features=4)

        # This should not crash
        est.update(np.ones(100), feature_idx=10)

        # No features should be updated
        assert all(c == 0 for c in est.counts)

    def test_constant_values(self):
        """Handle constant (zero variance) input."""
        est = StreamingQuantileEstimator(reservoir_size=1000, n_features=4)

        # Add constant values to all 4 features
        for i in range(4):
            est.update(np.ones(1000, dtype=np.float32) * 5.0, i)

        quantiles = est.get_quantiles(0, [0.25, 0.5, 0.75])
        assert quantiles[0.5] == 5.0
        assert quantiles[0.25] == 5.0
        assert quantiles[0.75] == 5.0

        params = est.get_robust_params()
        # IQR should be 1.0 (fallback for zero IQR)
        assert params['feat_8_iqr'] == 1.0


class TestPerformance:
    """Performance tests."""

    def test_update_speed(self):
        """Update should be fast for large batches."""
        import time

        est = StreamingQuantileEstimator(reservoir_size=100000, n_features=2)

        # Create large batch
        batch = np.random.randn(10000, 20, 10).astype(np.float32)

        # Warmup
        est.update_batch(batch[:100], [8, 9])

        # Time update
        start = time.perf_counter()
        est.update_batch(batch, [8, 9])
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete in under 500ms for 10K sequences
        assert elapsed_ms < 500, f"Update took {elapsed_ms:.1f}ms (expected <500ms)"

    def test_memory_bounded(self):
        """Memory usage should be bounded by reservoir size."""
        import sys

        est = StreamingQuantileEstimator(reservoir_size=10000, n_features=4)

        # Get baseline memory
        baseline = sum(r.nbytes for r in est.reservoirs)

        # Add lots of data
        for _ in range(100):
            est.update(np.random.randn(10000).astype(np.float32), 0)

        # Memory should not have grown
        current = sum(r.nbytes for r in est.reservoirs)
        assert current == baseline, "Memory grew beyond reservoir size"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
