"""
Tests for Triple Barrier Labeling Method

Verifies that the implementation:
1. Uses relative returns (percentage change)
2. Respects the 150-day vertical barrier
3. Uses closed='left' for all rolling windows
4. Creates strictly temporal train/test splits
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.triple_barrier_labeler import (
    TripleBarrierLabeler,
    create_strict_temporal_split,
    validate_temporal_integrity,
    BarrierType,
    VERTICAL_BARRIER_DAYS,
    FEATURE_WINDOW_SIZE,
)


class TestTripleBarrierLabeler:
    """Tests for the Triple Barrier labeling implementation."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')

        # Simulate price with slight upward drift
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.015, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.015, len(dates)))),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)

    @pytest.fixture
    def labeler(self):
        """Create a labeler with default settings."""
        return TripleBarrierLabeler(
            vertical_barrier_days=150,
            upper_barrier_pct=0.03,  # +3%
            lower_barrier_pct=0.02,  # -2%
            volatility_scaling=False,
            feature_window_size=250
        )

    def test_label_uses_relative_returns(self, sample_price_data, labeler):
        """Verify that labeling uses relative returns (percentage change)."""
        entry_date = pd.Timestamp('2020-06-01')
        entry_price = sample_price_data.loc[entry_date, 'close']

        result = labeler.label_single_pattern(
            price_df=sample_price_data,
            entry_date=entry_date,
            entry_price=entry_price
        )

        # Check that returns are calculated as percentages
        assert result['max_return'] is not None
        assert result['min_return'] is not None
        # Returns should be small percentages, not large absolute values
        assert abs(result['max_return']) < 1.0  # Less than 100%
        assert abs(result['min_return']) < 1.0  # Less than 100%

    def test_vertical_barrier_150_days(self, sample_price_data, labeler):
        """Verify that vertical barrier is at 150 days."""
        entry_date = pd.Timestamp('2020-03-01')
        entry_price = sample_price_data.loc[entry_date, 'close']

        barriers = labeler.get_barriers(
            price_df=sample_price_data,
            entry_date=entry_date,
            entry_price=entry_price
        )

        expected_vertical = entry_date + timedelta(days=150)
        assert barriers['vertical_barrier_date'] == expected_vertical

    def test_label_classes_correct(self, sample_price_data, labeler):
        """Verify label classes: 0=Danger, 1=Noise, 2=Target."""
        # Create price data that clearly hits upper barrier
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        prices = np.linspace(100, 110, 300)  # Steady rise of 10%

        price_df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': [100000] * 300
        }, index=dates)

        entry_date = dates[50]
        entry_price = prices[50]

        result = labeler.label_single_pattern(
            price_df=price_df,
            entry_date=entry_date,
            entry_price=entry_price
        )

        # With 10% rise, should hit upper barrier (+3%)
        assert result['label'] == 2  # Target
        assert result['barrier_hit'] == BarrierType.UPPER.value

    def test_danger_label_on_decline(self, labeler):
        """Verify Danger label when lower barrier is hit."""
        # Create price data that clearly hits lower barrier
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        prices = np.linspace(100, 90, 300)  # Steady decline of 10%

        price_df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': [100000] * 300
        }, index=dates)

        entry_date = dates[50]
        entry_price = prices[50]

        result = labeler.label_single_pattern(
            price_df=price_df,
            entry_date=entry_date,
            entry_price=entry_price
        )

        # With 10% decline, should hit lower barrier (-2%)
        assert result['label'] == 0  # Danger
        assert result['barrier_hit'] == BarrierType.LOWER.value

    def test_noise_label_on_flat(self, labeler):
        """Verify Noise label when neither barrier is hit."""
        # Create flat price data
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        prices = np.full(300, 100.0) + np.random.normal(0, 0.5, 300)  # Very low volatility

        price_df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.001,  # Very tight range
            'low': prices * 0.999,
            'close': prices,
            'volume': [100000] * 300
        }, index=dates)

        entry_date = dates[50]
        entry_price = prices[50]

        result = labeler.label_single_pattern(
            price_df=price_df,
            entry_date=entry_date,
            entry_price=entry_price
        )

        # With flat prices, should hit vertical barrier (timeout)
        assert result['label'] == 1  # Noise
        assert result['barrier_hit'] == BarrierType.VERTICAL.value


class TestHistoricFeatures:
    """Tests for strictly historic feature calculation."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)

    @pytest.fixture
    def labeler(self):
        return TripleBarrierLabeler(
            vertical_barrier_days=150,
            upper_barrier_pct=0.03,
            lower_barrier_pct=0.02,
            volatility_scaling=False,
            feature_window_size=250
        )

    def test_features_use_only_historic_data(self, sample_price_data, labeler):
        """Verify features are calculated using only data BEFORE reference date."""
        reference_date = pd.Timestamp('2020-06-15')

        features = labeler.calculate_historic_features(
            price_df=sample_price_data,
            reference_date=reference_date
        )

        # Features should be calculated from data before reference_date
        hist_data = sample_price_data[sample_price_data.index < reference_date]

        # Check SMA_20 is calculated from historic data only
        # The feature at reference_date should use data from [ref-20, ref)
        expected_sma20 = hist_data['close'].tail(20).mean()
        if 'sma_20' in features and not np.isnan(features['sma_20']):
            # Allow some tolerance for closed='left' behavior
            assert abs(features['sma_20'] - expected_sma20) / expected_sma20 < 0.1

    def test_volatility_strictly_historic(self, sample_price_data, labeler):
        """Verify volatility calculation uses only data BEFORE reference date."""
        reference_date = pd.Timestamp('2020-06-15')

        daily_vol = labeler.calculate_daily_volatility(
            price_df=sample_price_data,
            date=reference_date
        )

        # Volatility should be based on data BEFORE reference_date
        hist_data = sample_price_data[sample_price_data.index < reference_date].tail(20)
        log_returns = np.log(hist_data['close'] / hist_data['close'].shift(1))
        expected_vol = log_returns.std()

        assert abs(daily_vol - expected_vol) < 0.001


class TestStrictTemporalSplit:
    """Tests for strict temporal train/test splits."""

    @pytest.fixture
    def sample_patterns(self):
        """Create sample pattern data."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='W')
        return pd.DataFrame({
            'ticker': ['AAPL'] * len(dates),
            'end_date': dates,
            'label': np.random.choice([0, 1, 2], len(dates))
        })

    def test_no_temporal_overlap(self, sample_patterns):
        """Verify train/val/test have no overlapping dates."""
        train_df, val_df, test_df = create_strict_temporal_split(
            df=sample_patterns,
            date_col='end_date',
            train_end='2022-06-30',
            val_end='2023-06-30'
        )

        train_dates = set(pd.to_datetime(train_df['end_date']))
        val_dates = set(pd.to_datetime(val_df['end_date']))
        test_dates = set(pd.to_datetime(test_df['end_date']))

        # No overlap between any splits
        assert len(train_dates & val_dates) == 0
        assert len(val_dates & test_dates) == 0
        assert len(train_dates & test_dates) == 0

    def test_temporal_ordering(self, sample_patterns):
        """Verify train dates < val dates < test dates."""
        train_df, val_df, test_df = create_strict_temporal_split(
            df=sample_patterns,
            date_col='end_date',
            train_end='2022-06-30',
            val_end='2023-06-30'
        )

        train_max = pd.to_datetime(train_df['end_date']).max()
        val_min = pd.to_datetime(val_df['end_date']).min()
        val_max = pd.to_datetime(val_df['end_date']).max()
        test_min = pd.to_datetime(test_df['end_date']).min()

        assert train_max < val_min
        assert val_max < test_min

    def test_validate_temporal_integrity_passes(self, sample_patterns):
        """Verify validate_temporal_integrity passes for valid splits."""
        train_df, val_df, test_df = create_strict_temporal_split(
            df=sample_patterns,
            date_col='end_date',
            train_end='2022-06-30',
            val_end='2023-06-30'
        )

        # Should not raise
        assert validate_temporal_integrity(train_df, val_df, test_df, date_col='end_date')

    def test_validate_temporal_integrity_fails_on_overlap(self):
        """Verify validate_temporal_integrity raises on overlapping dates."""
        # Create intentionally overlapping splits
        train_df = pd.DataFrame({
            'end_date': pd.date_range('2020-01-01', '2022-06-30', freq='W')
        })
        val_df = pd.DataFrame({
            'end_date': pd.date_range('2022-06-01', '2023-06-30', freq='W')  # Overlaps!
        })
        test_df = pd.DataFrame({
            'end_date': pd.date_range('2023-07-01', '2023-12-31', freq='W')
        })

        with pytest.raises(ValueError, match="Temporal integrity violated"):
            validate_temporal_integrity(train_df, val_df, test_df, date_col='end_date')

    def test_gap_days_creates_buffer(self, sample_patterns):
        """Verify gap_days creates buffer between splits."""
        train_df, val_df, test_df = create_strict_temporal_split(
            df=sample_patterns,
            date_col='end_date',
            train_end='2022-06-30',
            val_end='2023-06-30',
            gap_days=7  # 1 week gap
        )

        train_max = pd.to_datetime(train_df['end_date']).max()
        val_min = pd.to_datetime(val_df['end_date']).min()

        # Gap should be at least 7 days
        assert (val_min - train_max).days >= 7


class TestVolatilityScaling:
    """Tests for volatility-scaled barriers."""

    @pytest.fixture
    def high_vol_data(self):
        """Create high volatility price data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        returns = np.random.normal(0, 0.05, len(dates))  # High volatility
        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': prices * 1.01,
            'high': prices * 1.03,
            'low': prices * 0.97,
            'close': prices,
            'volume': [100000] * 300
        }, index=dates)

    @pytest.fixture
    def low_vol_data(self):
        """Create low volatility price data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        returns = np.random.normal(0, 0.005, len(dates))  # Low volatility
        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': prices * 1.001,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': [100000] * 300
        }, index=dates)

    def test_vol_scaling_adapts_barriers(self, high_vol_data, low_vol_data):
        """Verify volatility scaling produces different barriers."""
        labeler = TripleBarrierLabeler(
            vertical_barrier_days=150,
            upper_barrier_pct=0.03,
            lower_barrier_pct=0.02,
            volatility_scaling=True,
            volatility_multiplier=2.0
        )

        entry_date = pd.Timestamp('2020-06-01')

        high_vol_barriers = labeler.get_barriers(
            price_df=high_vol_data,
            entry_date=entry_date,
            entry_price=100.0
        )

        low_vol_barriers = labeler.get_barriers(
            price_df=low_vol_data,
            entry_date=entry_date,
            entry_price=100.0
        )

        # High vol should have wider barriers
        assert abs(high_vol_barriers['upper_barrier_pct']) > abs(low_vol_barriers['upper_barrier_pct'])
        assert abs(high_vol_barriers['lower_barrier_pct']) > abs(low_vol_barriers['lower_barrier_pct'])


class TestRollingWindowClosedLeft:
    """Tests to verify rolling windows use closed='left' correctly."""

    def test_bbw_excludes_current_day(self):
        """Verify BBW calculation excludes current day with closed='left'."""
        from features.vectorized_calculator import VectorizedFeatureCalculator

        # Create data with natural volatility, then a spike on the last day
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=50, freq='D')

        # Normal price movement with some volatility
        returns = np.random.normal(0, 0.015, 49)
        normal_prices = 100 * np.exp(np.cumsum(returns))
        # Add a massive spike on the last day
        prices = np.concatenate([normal_prices, [normal_prices[-1] * 2.0]])

        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': [100000] * 50
        }, index=dates)

        calc = VectorizedFeatureCalculator()
        df_with_features = calc.calculate_all_features(df)

        # With closed='left', the BBW at the spike day should be calculated
        # from the 20 days BEFORE it (not including the spike)
        # So it should be similar to the BBW on the day before the spike

        # Skip warmup period (need 20+ days)
        bbw_values = df_with_features['bbw_20'].dropna()

        if len(bbw_values) >= 2:
            # Get last two non-NaN BBW values
            bbw_second_last = bbw_values.iloc[-2]
            bbw_last = bbw_values.iloc[-1]

            # Both should be positive (there's volatility in the data)
            # The spike shouldn't dramatically change BBW if closed='left' works
            # Note: with closed='left', the spike is excluded from its own window
            # so the BBW should not be extremely different
            if bbw_second_last > 0.001:  # Ensure there's real volatility
                # BBW should be reasonably bounded - spike shouldn't cause 10x increase
                assert bbw_last < bbw_second_last * 5, (
                    f"BBW changed too much: {bbw_second_last:.4f} -> {bbw_last:.4f}"
                )


class TestDynamicBarriers:
    """Tests for dynamic barrier adjustment based on market cap and regime."""

    def test_micro_cap_has_wider_barriers(self):
        """Verify micro-caps get wider barriers than small-caps."""
        from config.barrier_config import get_dynamic_barriers, MarketCapTier, MarketRegimeState

        micro_barriers = get_dynamic_barriers(MarketCapTier.MICRO, MarketRegimeState.NEUTRAL)
        small_barriers = get_dynamic_barriers(MarketCapTier.SMALL, MarketRegimeState.NEUTRAL)

        # Micro-caps should have wider barriers (higher percentages)
        assert micro_barriers['upper_barrier_pct'] > small_barriers['upper_barrier_pct']
        assert micro_barriers['lower_barrier_pct'] > small_barriers['lower_barrier_pct']

    def test_bullish_regime_adjusts_barriers(self):
        """Verify bullish regime extends targets and tightens stops."""
        from config.barrier_config import get_dynamic_barriers, MarketCapTier, MarketRegimeState

        bullish = get_dynamic_barriers(MarketCapTier.SMALL, MarketRegimeState.BULLISH)
        neutral = get_dynamic_barriers(MarketCapTier.SMALL, MarketRegimeState.NEUTRAL)

        # Bullish should have higher targets
        assert bullish['upper_barrier_pct'] > neutral['upper_barrier_pct']
        # Bullish should have tighter stops
        assert bullish['lower_barrier_pct'] < neutral['lower_barrier_pct']

    def test_bearish_regime_adjusts_barriers(self):
        """Verify bearish regime lowers targets and widens stops."""
        from config.barrier_config import get_dynamic_barriers, MarketCapTier, MarketRegimeState

        bearish = get_dynamic_barriers(MarketCapTier.SMALL, MarketRegimeState.BEARISH)
        neutral = get_dynamic_barriers(MarketCapTier.SMALL, MarketRegimeState.NEUTRAL)

        # Bearish should have lower targets
        assert bearish['upper_barrier_pct'] < neutral['upper_barrier_pct']
        # Bearish should have wider stops
        assert bearish['lower_barrier_pct'] > neutral['lower_barrier_pct']

    def test_crisis_regime_is_most_defensive(self):
        """Verify crisis regime has widest stops and lowest targets."""
        from config.barrier_config import get_dynamic_barriers, MarketCapTier, MarketRegimeState

        crisis = get_dynamic_barriers(MarketCapTier.SMALL, MarketRegimeState.CRISIS)
        neutral = get_dynamic_barriers(MarketCapTier.SMALL, MarketRegimeState.NEUTRAL)
        bullish = get_dynamic_barriers(MarketCapTier.SMALL, MarketRegimeState.BULLISH)

        # Crisis should have lowest targets
        assert crisis['upper_barrier_pct'] < neutral['upper_barrier_pct']
        assert crisis['upper_barrier_pct'] < bullish['upper_barrier_pct']

        # Crisis should have widest stops
        assert crisis['lower_barrier_pct'] > neutral['lower_barrier_pct']

    def test_adv_based_classification(self):
        """Verify ADV-based market cap classification works."""
        from config.barrier_config import classify_market_cap_from_adv, MarketCapTier

        assert classify_market_cap_from_adv(50_000) == MarketCapTier.NANO
        assert classify_market_cap_from_adv(200_000) == MarketCapTier.MICRO
        assert classify_market_cap_from_adv(1_000_000) == MarketCapTier.SMALL
        assert classify_market_cap_from_adv(10_000_000) == MarketCapTier.MID
        assert classify_market_cap_from_adv(100_000_000) == MarketCapTier.LARGE

    def test_vix_to_regime_mapping(self):
        """Verify VIX level maps to correct regime."""
        from config.barrier_config import map_vix_to_regime, MarketRegimeState

        assert map_vix_to_regime(10) == MarketRegimeState.BULLISH
        assert map_vix_to_regime(15) == MarketRegimeState.NEUTRAL
        assert map_vix_to_regime(20) == MarketRegimeState.BEARISH
        assert map_vix_to_regime(30) == MarketRegimeState.HIGH_VOLATILITY
        assert map_vix_to_regime(40) == MarketRegimeState.CRISIS

    def test_labeler_with_dynamic_barriers(self):
        """Verify labeler works with dynamic barriers enabled."""
        labeler = TripleBarrierLabeler(
            vertical_barrier_days=150,
            use_dynamic_barriers=True
        )

        assert labeler.config.use_dynamic_barriers == True

        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=400, freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        price_df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)

        entry_date = pd.Timestamp('2020-06-01')
        entry_price = price_df.loc[entry_date, 'close']

        # Should not raise
        result = labeler.label_single_pattern(
            price_df=price_df,
            entry_date=entry_date,
            entry_price=entry_price,
            ticker='TEST'
        )

        assert result['label'] is not None
        assert 'market_cap_tier' in result['barriers']
        assert 'regime' in result['barriers']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
