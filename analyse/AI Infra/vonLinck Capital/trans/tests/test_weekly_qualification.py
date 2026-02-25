"""
Tests for Weekly Qualification Mode (Jan 2026)

Tests the weekly candle aggregation and 10-week qualification period
for longer-term consolidation pattern detection.

Run with: pytest tests/test_weekly_qualification.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestWeeklyAggregator:
    """Tests for utils/weekly_aggregator.py"""

    def test_resample_to_weekly_basic(self):
        """Test basic daily to weekly resampling"""
        from utils.weekly_aggregator import resample_to_weekly

        # Create 20 days of daily data (4 weeks)
        dates = pd.date_range(start='2024-01-01', periods=20, freq='B')
        daily_df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 20),
            'high': np.random.uniform(110, 120, 20),
            'low': np.random.uniform(90, 100, 20),
            'close': np.random.uniform(100, 110, 20),
            'volume': np.random.randint(1000, 10000, 20)
        }, index=dates)

        weekly_df, week_to_daily_map = resample_to_weekly(daily_df)

        # Should have fewer rows than daily
        assert len(weekly_df) < len(daily_df)
        # Each week should map to a daily date
        assert len(week_to_daily_map) == len(weekly_df)
        # Weekly volume should be sum of daily
        assert weekly_df['volume'].iloc[0] > daily_df['volume'].iloc[0]

    def test_resample_ohlcv_aggregation(self):
        """Test OHLCV aggregation rules: O=first, H=max, L=min, C=last, V=sum"""
        from utils.weekly_aggregator import resample_to_weekly

        # Create exactly 5 trading days (1 week)
        dates = pd.date_range(start='2024-01-08', end='2024-01-12', freq='B')  # Mon-Fri
        daily_df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [110, 115, 120, 118, 112],  # Max is 120
            'low': [95, 98, 97, 96, 99],        # Min is 95
            'close': [105, 106, 107, 108, 109],
            'volume': [1000, 2000, 3000, 4000, 5000]  # Sum is 15000
        }, index=dates)

        weekly_df, _ = resample_to_weekly(daily_df)

        assert len(weekly_df) == 1
        assert weekly_df['open'].iloc[0] == 100   # First
        assert weekly_df['high'].iloc[0] == 120   # Max
        assert weekly_df['low'].iloc[0] == 95     # Min
        assert weekly_df['close'].iloc[0] == 109  # Last
        assert weekly_df['volume'].iloc[0] == 15000  # Sum

    def test_week_to_daily_mapping(self):
        """Test that week-end dates map correctly to last trading day"""
        from utils.weekly_aggregator import resample_to_weekly

        # Create data with known week ending on Friday
        dates = pd.date_range(start='2024-01-08', end='2024-01-12', freq='B')
        daily_df = pd.DataFrame({
            'open': [100] * 5,
            'high': [110] * 5,
            'low': [90] * 5,
            'close': [105] * 5,
            'volume': [1000] * 5
        }, index=dates)

        _, week_to_daily_map = resample_to_weekly(daily_df)

        # The week ending Friday 2024-01-12 should map to 2024-01-12
        for week_end, daily_date in week_to_daily_map.items():
            assert daily_date in daily_df.index
            # Daily date should be within 6 days of week end
            assert (week_end - daily_date).days <= 6

    def test_calculate_weekly_indicators(self):
        """Test weekly indicator calculation"""
        from utils.weekly_aggregator import resample_to_weekly, calculate_weekly_indicators

        # Create 300 days of data (~60 weeks, enough for SMA_50)
        dates = pd.date_range(start='2023-01-01', periods=300, freq='B')
        daily_df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 300),
            'high': np.random.uniform(110, 120, 300),
            'low': np.random.uniform(90, 100, 300),
            'close': np.random.uniform(100, 110, 300),
            'volume': np.random.randint(1000, 10000, 300)
        }, index=dates)

        weekly_df, _ = resample_to_weekly(daily_df)
        weekly_df = calculate_weekly_indicators(weekly_df)

        # Should have indicator columns
        assert 'bbw_20' in weekly_df.columns
        assert 'sma_50' in weekly_df.columns
        assert 'volume_ratio_20' in weekly_df.columns
        assert 'range_ratio_20' in weekly_df.columns

        # Indicators should have values after warmup period
        assert weekly_df['bbw_20'].dropna().shape[0] > 0
        assert weekly_df['sma_50'].dropna().shape[0] > 0

    def test_validate_weekly_data_requirements(self):
        """Test validation of minimum weekly data requirements"""
        from utils.weekly_aggregator import validate_weekly_data_requirements

        # Create insufficient data (only 50 weeks, need 80)
        dates = pd.date_range(start='2022-01-01', periods=50, freq='W-FRI')
        small_weekly_df = pd.DataFrame({
            'close': np.random.uniform(100, 110, 50)
        }, index=dates)

        is_valid, msg = validate_weekly_data_requirements(small_weekly_df)
        assert not is_valid
        assert "Insufficient" in msg

        # Create sufficient data (100 weeks, need 80)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='W-FRI')
        large_weekly_df = pd.DataFrame({
            'close': np.random.uniform(100, 110, 100)
        }, index=dates)

        is_valid, msg = validate_weekly_data_requirements(large_weekly_df)
        assert is_valid

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        from utils.weekly_aggregator import resample_to_weekly, calculate_weekly_indicators

        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        weekly_df, week_map = resample_to_weekly(empty_df)
        assert len(weekly_df) == 0
        assert len(week_map) == 0

        result = calculate_weekly_indicators(weekly_df)
        assert len(result) == 0


class TestWeeklyConstants:
    """Tests for weekly qualification constants in config/constants.py"""

    def test_weekly_constants_exist(self):
        """Test that weekly constants are defined"""
        from config.constants import (
            WEEKLY_QUALIFICATION_PERIODS,
            WEEKLY_LOOKBACK_PERIODS,
            WEEKLY_MIN_HISTORY_WEEKS,
            WEEKLY_RESAMPLE_RULE
        )

        assert WEEKLY_QUALIFICATION_PERIODS == 10
        assert WEEKLY_LOOKBACK_PERIODS == 100
        assert WEEKLY_MIN_HISTORY_WEEKS == 80  # 50 weeks (SMA) + 30 weeks (window)
        assert WEEKLY_RESAMPLE_RULE == 'W-FRI'


class TestSleeperScannerV17Weekly:
    """Tests for weekly mode in sleeper_scanner_v17"""

    def test_candle_frequency_parameter(self):
        """Test that candle_frequency parameter is accepted"""
        from core.sleeper_scanner_v17 import find_sleepers_v17

        # Create minimal weekly data
        dates = pd.date_range(start='2020-01-01', periods=220, freq='W-FRI')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 220),
            'high': np.random.uniform(110, 120, 220),
            'low': np.random.uniform(90, 100, 220),
            'close': np.random.uniform(100, 110, 220),
            'volume': np.random.randint(100000, 1000000, 220),
            'Ticker': ['TEST'] * 220
        }, index=dates)

        # Should not raise an error
        result = find_sleepers_v17(df, candle_frequency='weekly')
        # Result may be None (no pattern detected) but function should run

    def test_minimum_data_requirement_weekly(self):
        """Test minimum data requirements for daily vs weekly mode"""
        from core.sleeper_scanner_v17 import find_sleepers_v17

        # Create data with 25 rows (less than weekly min of 30, less than daily min of 100)
        dates = pd.date_range(start='2020-01-01', periods=25, freq='W-FRI')
        small_df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 25),
            'high': np.random.uniform(110, 120, 25),
            'low': np.random.uniform(90, 100, 25),
            'close': np.random.uniform(100, 110, 25),
            'volume': np.random.randint(100000, 1000000, 25),
            'Ticker': ['TEST'] * 25
        }, index=dates)

        # Daily mode should return None (min 100 rows required)
        result_daily = find_sleepers_v17(small_df, candle_frequency='daily')
        assert result_daily is None  # Insufficient data for daily

        # Weekly mode should return None (min 30 rows required)
        result_weekly = find_sleepers_v17(small_df, candle_frequency='weekly')
        assert result_weekly is None  # Insufficient data for weekly

        # Create data with 50 rows (enough for weekly min of 30, not for daily min of 100)
        dates = pd.date_range(start='2020-01-01', periods=50, freq='W-FRI')
        medium_df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.randint(100000, 1000000, 50),
            'Ticker': ['TEST'] * 50
        }, index=dates)

        # Daily mode should return None (need 100)
        result_daily2 = find_sleepers_v17(medium_df, candle_frequency='daily')
        assert result_daily2 is None  # Still insufficient for daily

        # Weekly mode may process (min 30) - won't fail due to data length
        # Result may be None due to other filters but not data length


class TestConsolidationTrackerWeekly:
    """Tests for weekly mode in consolidation_tracker_refactored"""

    def test_weekly_mode_initialization(self):
        """Test tracker initializes with weekly mode parameters"""
        from core.aiv7_components.consolidation_tracker_refactored import RefactoredConsolidationTracker

        tracker = RefactoredConsolidationTracker(
            ticker='TEST',
            use_weekly_qualification=True,
            qualification_periods=10
        )

        assert tracker.use_weekly_qualification is True
        assert tracker.qualification_periods == 10
        assert tracker.candle_frequency == 'weekly'

    def test_get_state_includes_frequency(self):
        """Test that get_state() includes qualification frequency"""
        from core.aiv7_components.consolidation_tracker_refactored import RefactoredConsolidationTracker

        tracker = RefactoredConsolidationTracker(
            ticker='TEST',
            use_weekly_qualification=True
        )

        state = tracker.get_state()

        assert 'qualification_frequency' in state
        assert state['qualification_frequency'] == 'weekly'
        assert 'qualification_periods' in state


class TestPatternScannerWeekly:
    """Tests for weekly mode in pattern_scanner.py"""

    def test_weekly_mode_parameter(self):
        """Test scanner accepts use_weekly_qualification parameter"""
        from core.pattern_scanner import ConsolidationPatternScanner

        # Should not raise
        scanner = ConsolidationPatternScanner(
            use_weekly_qualification=True,
            disable_gcs=True,
            enable_market_cap=False
        )

        assert scanner.use_weekly_qualification is True


class TestPatternDateMapping:
    """Tests for pattern date mapping in weekly mode"""

    def test_enrich_pattern_with_daily_date(self):
        """Test pattern enrichment with daily date mapping"""
        from utils.weekly_aggregator import enrich_pattern_with_daily_date

        # Create a mock pattern from weekly detection
        pattern = {
            'ticker': 'TEST',
            'end_date': pd.Timestamp('2024-01-12'),  # Friday
            'upper_boundary': 110.0,
            'lower_boundary': 100.0
        }

        # Create mock week_to_daily mapping
        week_to_daily_map = {
            pd.Timestamp('2024-01-12'): pd.Timestamp('2024-01-12'),
            pd.Timestamp('2024-01-19'): pd.Timestamp('2024-01-18'),  # Thursday (Fri was holiday)
        }

        enriched = enrich_pattern_with_daily_date(pattern, week_to_daily_map)

        assert 'end_date_daily' in enriched
        assert 'qualification_frequency' in enriched
        assert enriched['qualification_frequency'] == 'weekly'
        assert enriched['end_date_daily'] == pd.Timestamp('2024-01-12')

    def test_get_daily_date_for_pattern(self):
        """Test getting daily date from week-end date"""
        from utils.weekly_aggregator import get_daily_date_for_pattern

        week_to_daily_map = {
            pd.Timestamp('2024-01-12'): pd.Timestamp('2024-01-12'),
        }

        # Exact match
        result = get_daily_date_for_pattern(pd.Timestamp('2024-01-12'), week_to_daily_map)
        assert result == pd.Timestamp('2024-01-12')

        # No match
        result = get_daily_date_for_pattern(pd.Timestamp('2025-01-12'), week_to_daily_map)
        assert result is None


class TestPatternReductionEstimation:
    """Tests for pattern count reduction estimation"""

    def test_estimate_pattern_reduction(self):
        """Test estimation of pattern count reduction in weekly mode"""
        from utils.weekly_aggregator import estimate_pattern_reduction

        # With 1000 daily patterns
        low, high = estimate_pattern_reduction(1000)

        # Expect 15-35% of patterns remain
        assert low == 150  # 15%
        assert high == 350  # 35%
        assert low < high


class TestIntegration:
    """Integration tests for weekly qualification flow"""

    def test_full_weekly_aggregation_flow(self):
        """Test complete flow: daily data -> weekly -> indicators"""
        from utils.weekly_aggregator import (
            resample_to_weekly,
            calculate_weekly_indicators,
            validate_weekly_data_requirements
        )

        # Create ~5 years of daily data
        dates = pd.date_range(start='2019-01-01', periods=1260, freq='B')
        daily_df = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(1260) * 0.5),
            'high': 105 + np.cumsum(np.random.randn(1260) * 0.5),
            'low': 95 + np.cumsum(np.random.randn(1260) * 0.5),
            'close': 100 + np.cumsum(np.random.randn(1260) * 0.5),
            'volume': np.random.randint(100000, 1000000, 1260)
        }, index=dates)

        # Fix OHLC relationships
        daily_df['high'] = daily_df[['open', 'high', 'low', 'close']].max(axis=1)
        daily_df['low'] = daily_df[['open', 'high', 'low', 'close']].min(axis=1)

        # Resample to weekly
        weekly_df, week_to_daily_map = resample_to_weekly(daily_df)

        # Validate requirements
        is_valid, msg = validate_weekly_data_requirements(weekly_df)
        assert is_valid, msg

        # Calculate indicators
        weekly_with_indicators = calculate_weekly_indicators(weekly_df)

        # All expected columns present
        expected_cols = ['open', 'high', 'low', 'close', 'volume',
                        'bbw_20', 'sma_50', 'sma_200', 'volume_ratio_20', 'range_ratio_20']
        for col in expected_cols:
            assert col in weekly_with_indicators.columns, f"Missing column: {col}"

        # SMA_200 should have values for later rows
        assert weekly_with_indicators['sma_200'].dropna().shape[0] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
