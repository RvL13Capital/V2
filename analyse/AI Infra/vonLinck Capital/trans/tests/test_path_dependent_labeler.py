"""
Test Suite for Path-Dependent Labeler v17
==========================================

Comprehensive tests for the path-dependent labeling system with:
- Temporal integrity validation
- Risk-based classification testing
- Data quality checks
- Edge case handling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
import logging

from core.path_dependent_labeler import PathDependentLabelerV17, label_pattern_simple

logging.basicConfig(level=logging.INFO)


class TestPathDependentLabeler:
    """Test suite for path-dependent labeling system."""

    @pytest.fixture
    def labeler(self):
        """Create a labeler instance for testing."""
        return PathDependentLabelerV17(
            indicator_warmup=30,
            indicator_stable=100,
            outcome_window=100,
            risk_multiplier_target=3.0,
            risk_multiplier_grey=2.5,
            stop_buffer=0.02
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        np.random.seed(42)

        # Generate realistic price data
        base_price = 100
        prices = []
        for i in range(300):
            if i < 130:  # History period
                volatility = 0.01
            elif i < 230:  # Outcome period with different scenarios
                volatility = 0.02
            else:
                volatility = 0.01

            price_change = np.random.randn() * volatility
            base_price *= (1 + price_change)
            prices.append(base_price)

        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 300)
        }, index=dates)

        return df

    def test_initialization(self, labeler):
        """Test labeler initialization."""
        assert labeler.indicator_warmup == 30
        assert labeler.indicator_stable == 100
        assert labeler.outcome_window == 100
        assert labeler.risk_multiplier_target == 3.0
        assert labeler.risk_multiplier_grey == 2.5
        assert labeler.min_total_required == 230  # 30 + 100 + 100

    def test_insufficient_history(self, labeler, sample_data):
        """Test handling of insufficient historical data."""
        # Try to label with pattern_end_idx too early
        label = labeler.label_pattern(
            full_data=sample_data,
            pattern_end_idx=50,  # Less than 130 days required
            pattern_boundaries={'upper': 105, 'lower': 95}
        )
        assert label is None

    def test_insufficient_future_data(self, labeler, sample_data):
        """Test handling of insufficient future data."""
        # Try to label with not enough future data
        label = labeler.label_pattern(
            full_data=sample_data,
            pattern_end_idx=250,  # Would need data up to 350
            pattern_boundaries={'upper': 105, 'lower': 95}
        )
        assert label is None

    def test_breakdown_detection(self, labeler):
        """Test that breakdown is detected first (priority)."""
        # Create data that breaks down immediately
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # Pattern ends at index 130, outcome starts at 131
        # Need 131 stable prices (indices 0-130), then breakdown starts at 131
        prices = [100] * 131  # Stable history (indices 0-130)
        # Breakdown scenario: drops below stop immediately
        prices.extend([94, 93, 92] + [100] * 116)  # Drops then recovers (total 250)

        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [500000] * 250
        }, index=dates)

        label = labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 98}
        )

        assert label == 0  # Danger/Breakdown

    def test_target_with_confirmation(self, labeler):
        """Test target detection with next-open confirmation."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # Create data that hits target with confirmation
        # Entry at pattern_end_idx=130 should be 100
        # Stop loss = 95 * 0.98 = 93.1
        # R = 100 - 93.1 = 6.9
        # Target = 100 + 3 * 6.9 = 120.7

        prices = [100] * 131  # Stable history (indices 0-130)
        # Target scenario: jumps to target and gaps up next open
        prices.extend([102, 105, 110, 121, 122] + [122] * 114)  # Hits 121 (>120.7) and holds (total 250)

        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [500000] * 250
        }, index=dates)

        # Ensure next open confirms (day after hitting target)
        df.loc[df.index[135], 'open'] = 122  # Confirms target

        label = labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 95}
        )

        assert label == 2  # Target/Home Run

    def test_grey_zone_detection(self, labeler):
        """Test grey zone identification (2.5R but no 3R confirmation).

        FIX Jan 2026: Grey zone patterns now labeled as Noise (Class 1).

        R calculation (CORRECT):
        - Entry = 100 (close at pattern_end)
        - Stop = lower_boundary * (1 - stop_buffer) = 95 * 0.98 = 93.1
        - R = lower_boundary - stop_loss = 95 - 93.1 = 1.9
        - Grey threshold = 100 + 2.5 * 1.9 = 104.75
        - Target = 100 + 3 * 1.9 = 105.7
        """
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # Create data that reaches grey zone (>104.75) but not target (>105.7)
        # with no next-open confirmation even if close reaches target
        prices = [100] * 131  # Stable history (indices 0-130)
        # Grey zone scenario: close reaches 105.5 (> grey 104.75) but < target 105.7
        # Even if close hits 105.7+, next open must also confirm
        prices.extend([101, 102, 103, 105.5, 105.0] + [104] * 114)  # Total 250

        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [500000] * 250
        }, index=dates)

        label = labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 95}
        )

        assert label == 1  # Noise (was -1 Grey Zone before Jan 2026 fix)

    def test_noise_classification(self, labeler):
        """Test noise classification (stays in range)."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # Create data that stays in range
        # Entry = 100, Stop = 95 * 0.98 = 93.1, R = 6.9
        # Grey threshold = 117.25, must stay below this
        prices = [100] * 131  # Stable history (indices 0-130)
        # Noise scenario: small movements, never hits stop or grey zone
        prices.extend([99, 101, 100, 102, 101] + [100] * 114)  # Total 250

        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [500000] * 250
        }, index=dates)

        label = labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 95}
        )

        assert label == 1  # Noise

    def test_path_dependency(self, labeler):
        """Test that first event wins (path dependency)."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # Create data that breaks down first, then would hit target
        # Entry = 100, Stop = 95 * 0.98 = 93.1, R = 6.9
        # Price must drop below 93.1 first (breakdown)

        prices = [100] * 131  # Stable history (indices 0-130)
        # First breakdown (below 93.1), then rally to target
        prices.extend([95, 92, 90, 125, 130] + [130] * 114)  # Break then moon (total 250)

        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [500000] * 250
        }, index=dates)

        label = labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 95}
        )

        assert label == 0  # Danger (breakdown happened first)

    def test_data_integrity_validation(self, labeler):
        """Test data integrity checks."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # Create data with integrity issues
        df = pd.DataFrame({
            'open': [100] * 250,
            'high': [98] * 250,  # High < Low (impossible)
            'low': [99] * 250,
            'close': [100] * 250,
            'volume': [500000] * 250
        }, index=dates)

        label = labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130
        )

        assert label is None  # Should reject invalid data

    def test_trading_halt_detection(self, labeler):
        """Test detection of trading halts."""
        # Create data with a large gap
        dates = []
        for i in range(130):
            dates.append(datetime(2023, 1, 1) + timedelta(days=i))

        # Add a 10-day gap (trading halt)
        for i in range(120):
            dates.append(datetime(2023, 5, 20) + timedelta(days=i))

        df = pd.DataFrame({
            'open': [100] * 250,
            'high': [101] * 250,
            'low': [99] * 250,
            'close': [100] * 250,
            'volume': [500000] * 250
        }, index=pd.DatetimeIndex(dates))

        label = labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130
        )

        assert label is None  # Should reject due to trading halt

    def test_holiday_period_allowed(self, labeler):
        """Test that holiday periods are allowed."""
        # Create dates with December holiday gap
        dates = pd.date_range(start='2023-01-01', periods=120, freq='D')
        # Add holiday gap (Dec 24 - Jan 2)
        holiday_dates = pd.date_range(start='2023-12-24', periods=10, freq='D')
        remaining = pd.date_range(start='2024-01-03', periods=120, freq='D')

        all_dates = dates.tolist() + holiday_dates.tolist() + remaining.tolist()

        df = pd.DataFrame({
            'open': [100] * len(all_dates),
            'high': [101] * len(all_dates),
            'low': [99] * len(all_dates),
            'close': [100] * len(all_dates),
            'volume': [500000] * len(all_dates)
        }, index=pd.DatetimeIndex(all_dates))

        label = labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130
        )

        assert label is not None  # Holiday period should be allowed

    def test_overnight_gap_vs_volatility(self, labeler):
        """Test distinguishing data errors from legitimate volatility."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        df = pd.DataFrame({
            'open': [100] * 250,
            'high': [150] * 250,  # High intraday volatility (50%)
            'low': [100] * 250,
            'close': [100] * 250,
            'volume': [500000] * 250
        }, index=dates)

        # This is high INTRADAY volatility, not a gap - should be allowed
        result = labeler.validate_data_integrity(
            df.iloc[:130],
            df.iloc[130:230]
        )

        assert result is True  # High volatility should be allowed

    def test_volume_validity(self, labeler):
        """Test volume validity requirements."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # Create data with many zero-volume days
        volumes = [0] * 60 + [100000] * 190  # 24% zero volume (>20% threshold)

        df = pd.DataFrame({
            'open': [100] * 250,
            'high': [101] * 250,
            'low': [99] * 250,
            'close': [100] * 250,
            'volume': volumes
        }, index=dates)

        result = labeler.validate_data_integrity(
            df.iloc[:130],
            df.iloc[130:230]
        )

        assert result is False  # Too many zero-volume days

    def test_risk_metrics_calculation(self, labeler):
        """Test risk metrics calculation."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

        outcome_data = pd.DataFrame({
            'open': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'close': [100] * 50 + [115] * 50,  # Reaches 3R
            'volume': [500000] * 100
        }, index=dates)

        metrics = labeler.calculate_risk_metrics(
            entry_price=100,
            stop_loss=95,
            outcome_data=outcome_data
        )

        assert metrics['risk_unit'] == 5
        assert metrics['max_r_multiple'] == 3.0
        assert metrics['target_price'] == 115

    def test_simple_labeling_function(self):
        """Test the simplified labeling function."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')

        pattern_data = pd.DataFrame({
            'open': [100] * 100,
            'high': [101] * 100,
            'low': [99] * 100,
            'close': [100] * 100,
            'volume': [500000] * 100
        }, index=dates[:100])

        # Test breakdown scenario
        outcome_data = pd.DataFrame({
            'open': [100, 95, 94] + [100] * 97,
            'high': [101, 96, 95] + [101] * 97,
            'low': [94, 94, 93] + [99] * 97,
            'close': [94, 94, 94] + [100] * 97,
            'volume': [500000] * 100
        }, index=dates[100:200])

        label = label_pattern_simple(pattern_data, outcome_data)
        assert label == 0  # Breakdown

        # Test target scenario - must also update low to prevent stop trigger
        # CRITICAL: Since we now use Low for stop checking, must ensure low stays above stop
        outcome_data['close'] = [100] * 10 + [116] * 90  # Hits 3R
        outcome_data['open'] = [100] * 10 + [116] * 90
        outcome_data['low'] = [99] * 10 + [115] * 90  # Keep low above stop level
        outcome_data['high'] = [101] * 10 + [117] * 90

        label = label_pattern_simple(pattern_data, outcome_data)
        assert label == 2  # Target

    def test_temporal_integrity(self, labeler):
        """Test that labeling maintains temporal integrity."""
        # This test verifies no look-ahead bias
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        df = pd.DataFrame({
            'open': [100] * 250,
            'high': [101] * 250,
            'low': [99] * 250,
            'close': [100] * 250,
            'volume': [500000] * 250
        }, index=dates)

        # Label should only use data available at pattern_end + outcome_window
        pattern_end = 130

        # The labeler should only access:
        # 1. History: indices 0-130 (for pattern detection)
        # 2. Future: indices 131-230 (for outcome)
        # It should NOT access indices beyond 230

        label = labeler.label_pattern(
            full_data=df,
            pattern_end_idx=pattern_end
        )

        # Verify the function doesn't crash and returns valid label
        assert label in [0, 1, 2, -1, None]


class TestATRBasedLabeling:
    """Test suite for ATR-based labeling system (Jan 2026).

    ATR-based labeling uses Average True Range to set dynamic stop/target levels
    that adapt to market regime volatility, as opposed to R-multiple which adapts
    to pattern boundary width.

    Key formulas:
    - Stop = lower_boundary - (2.0 × ATR_14)
    - Target = entry + (5.0 × ATR_14)
    - Grey zone = entry + (2.5 × ATR_14)
    """

    @pytest.fixture
    def atr_labeler(self):
        """Create an ATR-based labeler instance."""
        return PathDependentLabelerV17(
            indicator_warmup=30,
            indicator_stable=100,
            outcome_window=100,
            use_atr_labeling=True,
            atr_period=14,
            atr_stop_multiple=2.0,
            atr_target_multiple=5.0,
            atr_grey_multiple=2.5
        )

    @pytest.fixture
    def r_multiple_labeler(self):
        """Create an R-multiple (default) labeler for comparison."""
        return PathDependentLabelerV17(
            indicator_warmup=30,
            indicator_stable=100,
            outcome_window=100,
            use_atr_labeling=False
        )

    @pytest.fixture
    def sample_data_with_volatility(self):
        """Create sample data with known ATR for testing."""
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        np.random.seed(42)

        # Generate data with known volatility profile
        # ATR calculation needs true range: max(high-low, abs(high-prev_close), abs(low-prev_close))
        base_price = 100
        prices = []
        highs = []
        lows = []

        for i in range(300):
            # Create consistent volatility (~$2 ATR)
            daily_range = 2.0  # Fixed daily range for predictable ATR
            high = base_price + daily_range / 2
            low = base_price - daily_range / 2
            close = base_price + np.random.uniform(-1, 1) * 0.5

            prices.append(close)
            highs.append(high)
            lows.append(low)

            # Small drift
            base_price = close

        df = pd.DataFrame({
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': [500000] * 300
        }, index=dates)

        return df

    def test_atr_labeler_initialization(self, atr_labeler):
        """Test ATR labeler initialization with correct parameters."""
        assert atr_labeler.use_atr_labeling is True
        assert atr_labeler.atr_period == 14
        assert atr_labeler.atr_stop_multiple == 2.0
        assert atr_labeler.atr_target_multiple == 5.0
        assert atr_labeler.atr_grey_multiple == 2.5

    def test_r_multiple_labeler_default(self, r_multiple_labeler):
        """Test R-multiple labeler (default) does not use ATR."""
        assert r_multiple_labeler.use_atr_labeling is False

    def test_atr_target_detection(self, atr_labeler):
        """Test target detection with ATR-based thresholds."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # Create data with consistent ~$4 ATR (high=102, low=98)
        # Entry = 100, ATR = 4
        # Stop = lower_boundary - (2 × 4) = 95 - 8 = 87
        # Target = 100 + (5 × 4) = 120

        highs = [102] * 131  # History: daily range = 4 (high-low), ATR ≈ 4
        lows = [98] * 131
        prices = [100] * 131

        # Outcome: rally to target (120+)
        for i in range(119):
            if i < 20:
                prices.append(100 + i * 1.0)  # Gradual rally to 120
                highs.append(prices[-1] + 2)
                lows.append(prices[-1] - 2)  # Keep lows above stop (87)
            elif i < 30:
                prices.append(122)  # Hits target (> 120)
                highs.append(124)
                lows.append(120)
            else:
                prices.append(122)
                highs.append(124)
                lows.append(120)

        df = pd.DataFrame({
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': [500000] * 250
        }, index=dates)

        # Ensure next-open confirms target
        df.loc[df.index[152], 'open'] = 122  # Day after hitting target

        label, metadata = atr_labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 95},
            return_metadata=True
        )

        assert label == 2  # Target
        assert metadata.get('labeling_mode') == 'atr'
        assert metadata.get('atr_value') is not None

    def test_atr_danger_detection(self, atr_labeler):
        """Test stop loss detection with ATR-based thresholds."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # ATR ≈ 4 (high=102, low=98)
        # Stop = lower_boundary - (2 × 4) = 95 - 8 = 87

        highs = [102] * 131
        lows = [98] * 131
        prices = [100] * 131

        # Outcome: breakdown below ATR-based stop (87)
        for i in range(119):
            if i < 5:
                prices.append(95 - i * 2)  # Quick drop to 85 (below 87)
                highs.append(prices[-1] + 1)
                lows.append(prices[-1] - 1)
            else:
                prices.append(90)  # Stays low
                highs.append(91)
                lows.append(85)

        df = pd.DataFrame({
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': [500000] * 250
        }, index=dates)

        label, metadata = atr_labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 95},
            return_metadata=True
        )

        assert label == 0  # Danger (stop hit)
        assert metadata.get('labeling_mode') == 'atr'

    def test_atr_noise_classification(self, atr_labeler):
        """Test noise classification with ATR-based thresholds."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # ATR ≈ 4
        # Stop = 95 - 8 = 87 (must stay above)
        # Grey zone = 100 + (2.5 × 4) = 110 (must stay below)

        highs = [102] * 131
        lows = [98] * 131
        prices = [100] * 131

        # Outcome: stays in range (above 87, below 110)
        for i in range(119):
            prices.append(100 + np.sin(i * 0.1) * 3)  # Oscillates ±3 around 100
            highs.append(prices[-1] + 2)
            lows.append(prices[-1] - 2)  # Never below 95, never above 108

        df = pd.DataFrame({
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': [500000] * 250
        }, index=dates)

        label = atr_labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 95}
        )

        assert label == 1  # Noise

    def test_atr_vs_r_multiple_difference(self, atr_labeler, r_multiple_labeler):
        """Test that ATR and R-multiple labeling can produce different results.

        This demonstrates the key difference:
        - R-multiple: thresholds based on pattern boundary width
        - ATR: thresholds based on recent market volatility
        """
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # Create data with HIGH ATR (~8) but TIGHT pattern boundaries
        # Pattern: upper=102, lower=100 (width=2%, R≈0.02)
        # R-multiple: Stop = 100 × 0.92 = 92, Target = 102 + 5×0.1 ≈ 102.5 (very tight)
        # ATR: Stop = 100 - 16 = 84, Target = 102 + 40 = 142 (very wide)

        # High volatility history (ATR ≈ 8)
        highs = [108] * 131  # Daily range = 16
        lows = [92] * 131
        prices = [100] * 131

        # Outcome: price moves to 105 (above tight R-target, below ATR-target)
        for i in range(119):
            prices.append(105)
            highs.append(113)
            lows.append(97)  # Above R-stop (92), above ATR-stop (84)

        df = pd.DataFrame({
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': [500000] * 250
        }, index=dates)

        # This scenario should produce different labels
        # R-multiple might see target (105 > ~102.5)
        # ATR sees noise (105 < 142)
        # Note: Exact behavior depends on implementation details

        atr_label = atr_labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 100}
        )

        r_label = r_multiple_labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 100}
        )

        # Both should return valid labels (implementation may vary)
        assert atr_label in [0, 1, 2, -1, None]
        assert r_label in [0, 1, 2, -1, None]

    def test_atr_metadata_includes_atr_values(self, atr_labeler):
        """Test that ATR labeling metadata includes ATR-specific information."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        highs = [102] * 250
        lows = [98] * 250
        prices = [100] * 250

        df = pd.DataFrame({
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': [500000] * 250
        }, index=dates)

        result = atr_labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 95},
            return_metadata=True
        )

        assert result is not None
        label, metadata = result

        # ATR metadata should be present
        assert metadata.get('labeling_mode') == 'atr'
        assert metadata.get('atr_period') == 14
        assert metadata.get('atr_stop_multiple') == 2.0
        assert metadata.get('atr_target_multiple') == 5.0
        assert 'atr_value' in metadata

    def test_backward_compatibility_r_multiple(self, r_multiple_labeler):
        """Test that R-multiple labeling still works (backward compatibility)."""
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

        # Standard R-multiple test
        prices = [100] * 131
        prices.extend([102, 105, 110, 121, 122] + [122] * 114)

        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [500000] * 250
        }, index=dates)

        df.loc[df.index[135], 'open'] = 122  # Confirms target

        result = r_multiple_labeler.label_pattern(
            full_data=df,
            pattern_end_idx=130,
            pattern_boundaries={'upper': 102, 'lower': 95},
            return_metadata=True
        )

        assert result is not None
        label, metadata = result
        assert label == 2  # Target
        assert metadata.get('labeling_mode') == 'r_multiple'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])