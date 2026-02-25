"""
Tests for Data Integrity Module

Verifies that the pipeline catches:
1. Row duplication (74x bug)
2. Temporal leakage (look-ahead bias)
3. Feature leakage (outcome columns as features)
4. Insufficient statistical power
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_integrity import (
    DataIntegrityChecker,
    DataIntegrityError,
    enforce_temporal_split,
    deduplicate_metadata,
    assert_no_duplicates,
    assert_temporal_split
)


class TestDuplicationCheck:
    """Tests for row duplication detection."""

    def test_no_duplicates_passes(self):
        """Clean data should pass duplication check."""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOG'],
            'pattern_end_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'upper_boundary': [100.0, 200.0, 150.0],
            'lower_boundary': [95.0, 190.0, 145.0],
            'label': [0, 1, 2]
        })

        checker = DataIntegrityChecker(df, strict=True)
        passed, details = checker.check_duplication()

        assert passed
        assert details['duplication_ratio'] == 1.0
        assert details['unique_patterns'] == 3

    def test_duplicates_detected(self):
        """Duplicated data should fail check."""
        # Create data with 5x duplication (simulating 74x bug)
        base_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'pattern_end_date': ['2024-01-01', '2024-01-02'],
            'upper_boundary': [100.0, 200.0],
            'lower_boundary': [95.0, 190.0],
            'label': [0, 1]
        })
        df = pd.concat([base_df] * 5, ignore_index=True)

        checker = DataIntegrityChecker(df, strict=False)
        passed, details = checker.check_duplication()

        assert not passed
        assert details['duplication_ratio'] == 5.0
        assert details['unique_patterns'] == 2

    def test_strict_mode_raises(self):
        """Strict mode should raise error on duplicates."""
        base_df = pd.DataFrame({
            'ticker': ['AAPL'],
            'pattern_end_date': ['2024-01-01'],
            'upper_boundary': [100.0],
            'lower_boundary': [95.0],
            'label': [0]
        })
        df = pd.concat([base_df] * 10, ignore_index=True)

        checker = DataIntegrityChecker(df, strict=True)

        with pytest.raises(DataIntegrityError, match="duplication"):
            checker.run_all_checks()

    def test_deduplicate_function(self):
        """Deduplication should produce unique patterns."""
        base_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'pattern_end_date': ['2024-01-01', '2024-01-02'],
            'upper_boundary': [100.0, 200.0],
            'lower_boundary': [95.0, 190.0],
            'label': [0, 1],
            'extra_col': [1, 2]
        })
        df = pd.concat([base_df] * 5, ignore_index=True)

        deduped = deduplicate_metadata(df)

        assert len(deduped) == 2
        assert 'extra_col' in deduped.columns


class TestTemporalIntegrity:
    """Tests for temporal leakage detection."""

    def test_clean_temporal_split_passes(self):
        """Properly ordered temporal split should pass."""
        df = pd.DataFrame({
            'ticker': ['A'] * 9,
            'pattern_end_date': pd.date_range('2022-01-01', periods=9, freq='M'),
            'upper_boundary': range(9),
            'lower_boundary': range(9),
            'label': [0, 1, 2] * 3
        })

        train_mask = df['pattern_end_date'] < '2022-07-01'
        val_mask = (df['pattern_end_date'] >= '2022-07-01') & (df['pattern_end_date'] < '2022-10-01')
        test_mask = df['pattern_end_date'] >= '2022-10-01'

        checker = DataIntegrityChecker(df, strict=True)
        passed, details = checker.check_temporal_integrity(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        assert passed
        assert len(details.get('issues', [])) == 0

    def test_temporal_leakage_detected(self):
        """Should detect when val/test have dates before train max."""
        # Create data where train includes FUTURE dates and val has PAST dates
        df = pd.DataFrame({
            'ticker': ['A', 'A', 'B', 'B', 'C', 'C'],
            'pattern_end_date': [
                '2022-01-01', '2024-01-01',  # A patterns span time
                '2022-06-01', '2024-06-01',  # B patterns span time
                '2023-01-01', '2023-06-01'   # C in middle
            ],
            'upper_boundary': range(6),
            'lower_boundary': range(6),
            'label': [0, 1, 2, 0, 1, 2]
        })

        # Create leaky split: train includes 2024, val includes 2022
        # Train: 2024-01-01, 2024-06-01 (indices 1, 3)
        # Val: 2022-01-01, 2022-06-01 (indices 0, 2) - BEFORE train max!
        train_mask = pd.Series([False, True, False, True, False, False])  # 2024 dates
        val_mask = pd.Series([True, False, True, False, False, False])    # 2022 dates (leakage!)
        test_mask = pd.Series([False, False, False, False, True, True])   # 2023 dates

        checker = DataIntegrityChecker(df, strict=False)
        passed, details = checker.check_temporal_integrity(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        # Should fail because val dates (2022) are before train max (2024)
        assert not passed
        assert 'val_before_train_max' in details or len(details.get('issues', [])) > 0

    def test_enforce_temporal_split(self):
        """enforce_temporal_split should create clean splits."""
        df = pd.DataFrame({
            'ticker': ['A'] * 12,
            'pattern_end_date': pd.date_range('2022-01-01', periods=12, freq='M'),
            'upper_boundary': range(12),
            'lower_boundary': range(12),
            'label': [0, 1, 2] * 4
        })

        train_df, val_df, test_df = enforce_temporal_split(
            df,
            train_end='2022-06-30',
            val_end='2022-09-30'
        )

        # Check sizes
        assert len(train_df) == 6  # Jan-Jun
        assert len(val_df) == 3    # Jul-Sep
        assert len(test_df) == 3   # Oct-Dec

        # Check no overlap
        train_dates = set(train_df['pattern_end_date'])
        val_dates = set(val_df['pattern_end_date'])
        test_dates = set(test_df['pattern_end_date'])

        assert len(train_dates & val_dates) == 0
        assert len(train_dates & test_dates) == 0
        assert len(val_dates & test_dates) == 0


class TestFeatureLeakage:
    """Tests for feature leakage detection."""

    def test_no_leakage_passes(self):
        """Clean feature list should pass."""
        df = pd.DataFrame({
            'ticker': ['A'],
            'pattern_end_date': ['2024-01-01'],
            'upper_boundary': [100],
            'lower_boundary': [95],
            'label': [0],
            'volume': [1000],
            'bbw': [0.05]
        })

        checker = DataIntegrityChecker(df)
        passed, details = checker.check_feature_leakage(
            feature_columns=['volume', 'bbw', 'upper_boundary']
        )

        assert passed
        assert len(details['leakage_columns']) == 0

    def test_leakage_detected(self):
        """Should detect outcome columns in feature list."""
        df = pd.DataFrame({
            'ticker': ['A'],
            'pattern_end_date': ['2024-01-01'],
            'upper_boundary': [100],
            'lower_boundary': [95],
            'label': [0],
            'max_r_achieved': [2.5],  # LEAKAGE!
            'breakout_class': [1]      # LEAKAGE!
        })

        checker = DataIntegrityChecker(df, strict=False)
        passed, details = checker.check_feature_leakage(
            feature_columns=['upper_boundary', 'max_r_achieved', 'breakout_class']
        )

        assert not passed
        assert 'max_r_achieved' in details['leakage_columns']
        assert 'breakout_class' in details['leakage_columns']


class TestStatisticalPower:
    """Tests for statistical power checks."""

    def test_adequate_power_passes(self):
        """Sufficient target events should pass."""
        # Need 500+ targets per new MIN_TARGET_EVENTS threshold
        df = pd.DataFrame({
            'ticker': ['A'] * 1500,
            'pattern_end_date': pd.date_range('2024-01-01', periods=1500),
            'upper_boundary': range(1500),
            'lower_boundary': range(1500),
            'label': [2] * 600 + [0] * 600 + [1] * 300  # 600 target events
        })

        checker = DataIntegrityChecker(df)
        passed, details = checker.check_statistical_power()

        assert passed
        assert details['target_events'] == 600

    def test_low_power_fails(self):
        """Insufficient target events should fail."""
        df = pd.DataFrame({
            'ticker': ['A'] * 100,
            'pattern_end_date': pd.date_range('2024-01-01', periods=100),
            'upper_boundary': range(100),
            'lower_boundary': range(100),
            'label': [2] * 10 + [0] * 50 + [1] * 40  # Only 10 target events
        })

        checker = DataIntegrityChecker(df, strict=False)
        passed, details = checker.check_statistical_power()

        assert not passed
        assert details['target_events'] == 10


class TestIntegration:
    """Integration tests for full validation flow."""

    def test_run_all_checks_clean_data(self):
        """All checks should pass for clean data."""
        np.random.seed(42)
        n = 1500  # Increased for new MIN_TARGET_EVENTS (500)

        df = pd.DataFrame({
            'ticker': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
            'pattern_end_date': pd.date_range('2022-01-01', periods=n, freq='D'),
            'upper_boundary': np.random.uniform(100, 200, n),
            'lower_boundary': np.random.uniform(90, 100, n),
            'label': np.random.choice([0, 1, 2], n, p=[0.3, 0.3, 0.4])
        })

        # Ensure enough targets (need 500+)
        df.loc[df.index[:600], 'label'] = 2

        checker = DataIntegrityChecker(df, strict=True)
        results = checker.run_all_checks()

        assert results['summary']['all_passed']
        assert len(results['summary']['issues']) == 0

    def test_run_all_checks_with_splits(self):
        """Full check with temporal splits should work."""
        np.random.seed(42)
        n = 2000  # Increased for statistical power with new thresholds

        df = pd.DataFrame({
            'ticker': np.random.choice(['A', 'B', 'C'], n),
            'pattern_end_date': pd.date_range('2022-01-01', periods=n, freq='D'),
            'upper_boundary': np.random.uniform(100, 200, n),
            'lower_boundary': np.random.uniform(90, 100, n),
            'label': np.random.choice([0, 1, 2], n, p=[0.3, 0.3, 0.4])
        })

        # Ensure enough targets for statistical power (need 500+)
        df.loc[df.index[:700], 'label'] = 2

        # Create proper temporal masks
        df['_date'] = pd.to_datetime(df['pattern_end_date'])
        train_mask = df['_date'] < '2023-07-01'
        val_mask = (df['_date'] >= '2023-07-01') & (df['_date'] < '2024-07-01')
        test_mask = df['_date'] >= '2024-07-01'

        checker = DataIntegrityChecker(df, strict=True)
        results = checker.run_all_checks(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            feature_columns=['upper_boundary', 'lower_boundary'],
            min_targets_per_split=30  # Lower threshold for test
        )

        assert results['summary']['all_passed']


class TestAssertFunctions:
    """Tests for convenience assertion functions."""

    def test_assert_no_duplicates_passes(self):
        """Should not raise for clean data."""
        df = pd.DataFrame({
            'ticker': ['A', 'B'],
            'pattern_end_date': ['2024-01-01', '2024-01-02'],
            'upper_boundary': [100, 200],
            'lower_boundary': [95, 190]
        })

        # Should not raise
        assert_no_duplicates(df, context="test")

    def test_assert_no_duplicates_raises(self):
        """Should raise for duplicated data."""
        df = pd.DataFrame({
            'ticker': ['A', 'A', 'A'],
            'pattern_end_date': ['2024-01-01', '2024-01-01', '2024-01-01'],
            'upper_boundary': [100, 100, 100],
            'lower_boundary': [95, 95, 95]
        })

        with pytest.raises(DataIntegrityError):
            assert_no_duplicates(df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
