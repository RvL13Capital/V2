"""
Tests for Temporal Train/Test Split Functionality
==================================================

Tests the temporal splitting utilities to ensure:
1. No temporal leakage (look-ahead bias)
2. Correct split ratios
3. Proper date ordering
4. Validation of edge cases
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.temporal_split import (
    temporal_train_test_split,
    split_sequences_temporal,
    validate_temporal_split,
    get_temporal_fold_statistics,
    EMBARGO_DAYS
)


class TestTemporalSplit(unittest.TestCase):
    """Test cases for temporal split functionality."""

    def setUp(self):
        """Set up test data."""
        # Create sample data with 100 days
        self.n_samples = 100
        start_date = datetime(2020, 1, 1)
        self.dates = pd.Series([
            start_date + timedelta(days=i) for i in range(self.n_samples)
        ])

        # Create sample sequences and labels
        self.sequences = np.random.randn(self.n_samples, 20, 10)
        self.labels = np.random.randint(0, 6, self.n_samples)

        # Create metadata
        self.metadata = pd.DataFrame({
            'ticker': ['TEST'] * self.n_samples,
            'sequence_end_date': self.dates,
            'pattern_start_date': self.dates - timedelta(days=30),
            'pattern_end_date': self.dates
        })

    def test_temporal_train_test_split_basic(self):
        """Test basic temporal train/test split (no embargo for backward compat)."""
        train_idx, test_idx = temporal_train_test_split(
            self.dates,
            split_ratio=0.8,
            min_train_samples=50,
            min_test_samples=10,
            embargo_days=0  # Disable embargo for unit test
        )

        # Check split sizes
        self.assertEqual(len(train_idx) + len(test_idx), self.n_samples)
        self.assertAlmostEqual(len(train_idx) / self.n_samples, 0.8, places=1)

        # Check no overlap
        self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)

    def test_no_temporal_leakage(self):
        """Test that there is no temporal leakage (no embargo for backward compat)."""
        train_idx, test_idx, train_end, test_start = temporal_train_test_split(
            self.dates,
            split_ratio=0.8,
            return_dates=True,
            min_train_samples=50,
            min_test_samples=10,
            embargo_days=0  # Disable embargo for unit test
        )

        # Get actual dates
        train_dates = self.dates.iloc[train_idx]
        test_dates = self.dates.iloc[test_idx]

        # Check that all training dates are before all test dates
        max_train_date = train_dates.max()
        min_test_date = test_dates.min()

        self.assertLessEqual(max_train_date, min_test_date,
                           f"Training data ({max_train_date}) overlaps with test data ({min_test_date})")

    def test_date_cutoff_strategy(self):
        """Test date cutoff strategy (no embargo for backward compat)."""
        cutoff_date = datetime(2020, 3, 1)  # ~60 days into the data

        train_idx, test_idx = temporal_train_test_split(
            self.dates,
            strategy='date_cutoff',
            cutoff_date=cutoff_date,
            min_train_samples=30,
            min_test_samples=10,
            embargo_days=0  # Disable embargo for unit test
        )

        train_dates = self.dates.iloc[train_idx]
        test_dates = self.dates.iloc[test_idx]

        # All training dates should be before cutoff
        self.assertTrue(all(train_dates < cutoff_date))
        # All test dates should be on or after cutoff
        self.assertTrue(all(test_dates >= cutoff_date))

    def test_split_sequences_temporal(self):
        """Test splitting sequences with temporal awareness (no embargo for backward compat)."""
        result = split_sequences_temporal(
            self.sequences,
            self.labels,
            self.metadata,
            date_column='sequence_end_date',
            split_ratio=0.8,
            embargo_days=0  # Disable embargo for unit test
        )

        # Check that all expected keys are present
        expected_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'train_idx', 'test_idx']
        for key in expected_keys:
            self.assertIn(key, result)

        # Check shapes
        self.assertEqual(result['X_train'].shape[1:], (20, 10))
        self.assertEqual(result['X_test'].shape[1:], (20, 10))

        # Check temporal ordering
        train_dates = self.metadata.iloc[result['train_idx']]['sequence_end_date']
        test_dates = self.metadata.iloc[result['test_idx']]['sequence_end_date']

        self.assertLessEqual(train_dates.max(), test_dates.min())

    def test_validation_detects_leakage(self):
        """Test that validation correctly detects temporal leakage."""
        # Create indices with intentional leakage
        train_idx = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        test_idx = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])

        # This should raise an error due to temporal leakage
        with self.assertRaises(ValueError) as context:
            validate_temporal_split(self.dates, train_idx, test_idx)

        self.assertIn("Temporal leakage detected", str(context.exception))

    def test_minimum_samples_requirement(self):
        """Test minimum samples requirement."""
        small_dates = self.dates[:10]  # Only 10 samples

        # Should raise error due to insufficient samples
        with self.assertRaises(ValueError) as context:
            temporal_train_test_split(
                small_dates,
                split_ratio=0.8,
                min_train_samples=100,
                min_test_samples=20
            )

        self.assertIn("Insufficient samples", str(context.exception))

    def test_shuffled_dates_handled_correctly(self):
        """Test that shuffled dates are handled correctly (no embargo for backward compat)."""
        # Shuffle the dates
        shuffled_indices = np.random.permutation(self.n_samples)
        shuffled_dates = self.dates.iloc[shuffled_indices]

        train_idx, test_idx = temporal_train_test_split(
            shuffled_dates,
            split_ratio=0.8,
            min_train_samples=50,
            min_test_samples=10,
            embargo_days=0  # Disable embargo for unit test
        )

        # The function should still sort and split correctly
        train_dates = shuffled_dates.iloc[train_idx]
        test_dates = shuffled_dates.iloc[test_idx]

        self.assertLessEqual(train_dates.max(), test_dates.min())

    def test_temporal_fold_statistics(self):
        """Test temporal fold statistics calculation (no embargo for backward compat)."""
        train_idx, test_idx = temporal_train_test_split(
            self.dates,
            split_ratio=0.8,
            min_train_samples=50,
            min_test_samples=10,
            embargo_days=0  # Disable embargo for unit test
        )

        stats = get_temporal_fold_statistics(self.dates, train_idx, test_idx)

        # Check expected keys
        expected_keys = ['train_size', 'test_size', 'train_start', 'train_end',
                        'test_start', 'test_end', 'train_days', 'test_days',
                        'gap_days', 'train_ratio']

        for key in expected_keys:
            self.assertIn(key, stats)

        # Verify statistics
        self.assertEqual(stats['train_size'], len(train_idx))
        self.assertEqual(stats['test_size'], len(test_idx))
        self.assertAlmostEqual(stats['train_ratio'], 0.8, places=1)

    def test_missing_metadata_column(self):
        """Test handling of missing metadata column."""
        # Remove the expected column
        bad_metadata = self.metadata.drop('sequence_end_date', axis=1)

        with self.assertRaises(ValueError) as context:
            split_sequences_temporal(
                self.sequences,
                self.labels,
                bad_metadata,
                date_column='sequence_end_date'
            )

        self.assertIn("Date column 'sequence_end_date' not found", str(context.exception))

    def test_class_distribution_warning(self):
        """Test warning when classes are missing from test set (no embargo for backward compat)."""
        # Create labels where class 5 only appears early
        labels = np.zeros(self.n_samples, dtype=int)
        labels[:10] = 5  # Class 5 only in first 10 samples
        labels[10:] = np.random.randint(0, 5, self.n_samples - 10)

        # This should work but potentially warn about missing class
        result = split_sequences_temporal(
            self.sequences,
            labels,
            self.metadata,
            split_ratio=0.8,
            embargo_days=0  # Disable embargo for unit test
        )

        # Class 5 should only be in training set
        self.assertIn(5, result['y_train'])
        self.assertNotIn(5, result['y_test'])

    def test_temporal_gap_calculation(self):
        """Test temporal gap calculation between train and test (no embargo for backward compat)."""
        # Create dates with a gap
        dates_with_gap = pd.Series([
            datetime(2020, 1, 1) + timedelta(days=i)
            for i in list(range(50)) + list(range(60, 110))  # 10-day gap
        ])

        train_idx, test_idx, train_end, test_start = temporal_train_test_split(
            dates_with_gap,
            split_ratio=0.5,
            return_dates=True,
            min_train_samples=30,
            min_test_samples=10,
            embargo_days=0  # Disable embargo for unit test
        )

        gap_days = (test_start - train_end).days
        self.assertGreater(gap_days, 0, "Should detect temporal gap")


    def test_embargo_enforced(self):
        """Test that embargo is enforced between train and test sets."""
        # Create longer date range to test embargo
        n_samples = 400
        start_date = datetime(2020, 1, 1)
        long_dates = pd.Series([
            start_date + timedelta(days=i) for i in range(n_samples)
        ])

        embargo_days = 50  # Use smaller embargo for test

        train_idx, test_idx, train_end, test_start = temporal_train_test_split(
            long_dates,
            split_ratio=0.6,
            return_dates=True,
            min_train_samples=100,
            min_test_samples=50,
            embargo_days=embargo_days
        )

        # Check that gap between train end and test start >= embargo_days
        gap_days = (test_start - train_end).days
        self.assertGreaterEqual(gap_days, embargo_days,
                               f"Gap ({gap_days}) should be >= embargo ({embargo_days})")

        # Verify that some samples were dropped in the embargo zone
        total_assigned = len(train_idx) + len(test_idx)
        self.assertLess(total_assigned, n_samples,
                       "Some samples should be dropped in embargo zone")


class TestTemporalIntegration(unittest.TestCase):
    """Integration tests for temporal splitting in the pipeline."""

    def test_pipeline_compatibility(self):
        """Test that temporal split integrates with the training pipeline."""
        # Create realistic pipeline data
        n_patterns = 50
        sequences_per_pattern = 20
        n_samples = n_patterns * sequences_per_pattern

        # Generate sequences
        sequences = np.random.randn(n_samples, 20, 10).astype(np.float32)
        labels = np.random.randint(0, 6, n_samples)

        # Generate metadata with realistic pattern dates
        metadata_rows = []
        base_date = datetime(2020, 1, 1)

        for i in range(n_patterns):
            pattern_start = base_date + timedelta(days=i*5)
            pattern_end = pattern_start + timedelta(days=30)

            for j in range(sequences_per_pattern):
                seq_end = pattern_start + timedelta(days=10+j)
                metadata_rows.append({
                    'ticker': f'TICK{i}',
                    'pattern_start_date': pattern_start,
                    'pattern_end_date': pattern_end,
                    'sequence_end_date': seq_end,
                    'label': labels[i*sequences_per_pattern + j]
                })

        metadata = pd.DataFrame(metadata_rows)

        # Test temporal split (no embargo for backward compat)
        result = split_sequences_temporal(
            sequences,
            labels,
            metadata,
            split_ratio=0.8,
            embargo_days=0  # Disable embargo for unit test
        )

        # Verify result structure for pipeline compatibility
        self.assertIsInstance(result['X_train'], np.ndarray)
        self.assertIsInstance(result['X_test'], np.ndarray)
        self.assertEqual(result['X_train'].dtype, np.float32)
        self.assertEqual(result['X_test'].shape[1:], (20, 10))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalSplit))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)