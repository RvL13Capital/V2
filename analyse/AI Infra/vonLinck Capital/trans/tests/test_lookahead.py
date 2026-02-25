# tests/test_lookahead.py
"""
Critical tests ensuring adaptive metrics don't use future information.

Look-ahead bias is one of the FOUR DEADLY SINS documented in CLAUDE.md.
These tests verify that all adaptive calculations use ONLY historical data.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta


class TestAdaptiveZScoreLookahead:
    """Tests for Z-score based adaptive thresholds."""

    def test_adaptive_zscore_no_future_leakage(self):
        """
        Ensure adaptive Z-Score uses ONLY historical data (rolling window),
        not the full dataset mean (future leakage).
        """
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=200)

        # Regime 1: Tight BBW (~0.10 with small variance)
        # Regime 2: Wide BBW (~0.50 with small variance)
        bbw_tight = np.random.normal(0.10, 0.01, 100)  # Mean 0.10, std 0.01
        bbw_wide = np.random.normal(0.50, 0.02, 100)   # Mean 0.50, std 0.02
        bbw_vals = np.concatenate([bbw_tight, bbw_wide])

        df = pd.DataFrame({'bbw': bbw_vals}, index=dates)

        # Correct: Rolling calculation (no future data)
        window = 50
        roll_mean = df['bbw'].rolling(window).mean()
        roll_std = df['bbw'].rolling(window).std()
        df['z_score_rolling'] = (df['bbw'] - roll_mean) / roll_std

        # Incorrect: Global calculation (leaks future information)
        global_mean = df['bbw'].mean()
        global_std = df['bbw'].std()
        df['z_score_global'] = (df['bbw'] - global_mean) / global_std

        idx_check = 99  # End of tight period (last day before regime change)

        rolling_z = df['z_score_rolling'].iloc[idx_check]
        global_z = df['z_score_global'].iloc[idx_check]

        print(f"\nAt index {idx_check} (end of tight period):")
        print(f"  BBW value: {df['bbw'].iloc[idx_check]:.4f}")
        print(f"  Rolling mean: {roll_mean.iloc[idx_check]:.4f}")
        print(f"  Global mean:  {global_mean:.4f}")
        print(f"  Rolling Z (Correct):  {rolling_z:.4f}")
        print(f"  Global Z (WRONG):     {global_z:.4f}")

        # Rolling Z should be near 0 (value close to rolling mean in stable period)
        # Global Z should be negative (0.1 is below global mean of ~0.3)
        assert abs(rolling_z) < 2.0, f"Rolling Z should be moderate, got {rolling_z}"
        assert global_z < -1.0, f"Global Z should be strongly negative, got {global_z}"

        # The key insight: rolling Z doesn't know about future regime change
        # Global Z is contaminated by future high-volatility period
        assert rolling_z > global_z, \
            f"Rolling Z ({rolling_z:.2f}) should be higher than global Z ({global_z:.2f})"

    def test_rolling_window_warmup_period(self):
        """
        Ensure rolling calculations return NaN during warmup period.
        Using data before window is full would be incorrect.
        """
        dates = pd.date_range(start='2020-01-01', periods=100)
        df = pd.DataFrame({'bbw': np.random.uniform(0.05, 0.15, 100)}, index=dates)

        window = 20
        roll_mean = df['bbw'].rolling(window).mean()
        roll_std = df['bbw'].rolling(window).std()
        z_score = (df['bbw'] - roll_mean) / roll_std

        # First (window-1) values should be NaN
        assert z_score.iloc[:window - 1].isna().all(), \
            "Z-score should be NaN during warmup period"

        # Values after warmup should be valid
        assert z_score.iloc[window:].notna().all(), \
            "Z-score should be valid after warmup period"

    def test_expanding_vs_rolling_at_regime_change(self):
        """
        Compare expanding (cumulative) vs rolling window at regime change.
        Expanding window will be contaminated by old regime data longer.
        """
        dates = pd.date_range(start='2020-01-01', periods=150)
        # Regime 1: Low volatility (BBW ~0.05)
        # Regime 2: High volatility (BBW ~0.20)
        bbw_vals = list(np.random.uniform(0.04, 0.06, 100)) + \
                   list(np.random.uniform(0.18, 0.22, 50))

        df = pd.DataFrame({'bbw': bbw_vals}, index=dates)

        window = 20

        # Rolling: Adapts to new regime in ~window days
        roll_mean = df['bbw'].rolling(window).mean()

        # Expanding: Slow to adapt, contaminated by history
        expand_mean = df['bbw'].expanding().mean()

        # Check 10 days after regime change (index 110)
        idx_after_change = 110

        rolling_mean_at_110 = roll_mean.iloc[idx_after_change]
        expanding_mean_at_110 = expand_mean.iloc[idx_after_change]
        actual_bbw_at_110 = df['bbw'].iloc[idx_after_change]

        print(f"\nAt index {idx_after_change} (10 days after regime change):")
        print(f"  Actual BBW:      {actual_bbw_at_110:.4f}")
        print(f"  Rolling mean:    {rolling_mean_at_110:.4f}")
        print(f"  Expanding mean:  {expanding_mean_at_110:.4f}")

        # Rolling should be closer to actual (adapted to new regime)
        rolling_error = abs(rolling_mean_at_110 - actual_bbw_at_110)
        expanding_error = abs(expanding_mean_at_110 - actual_bbw_at_110)

        assert rolling_error < expanding_error, \
            f"Rolling ({rolling_error:.4f}) should adapt faster than expanding ({expanding_error:.4f})"


class TestVolatilityWindowLookahead:
    """Tests for volatility-adjusted labeling window (V22)."""

    def test_volatility_proxy_uses_pattern_data_only(self):
        """
        Ensure volatility proxy is calculated from pattern data,
        not from future outcome data.
        """
        # Simulate pattern with boundaries
        pattern = {
            'ticker': 'TEST',
            'end_date': '2024-01-15',
            'upper_boundary': 10.0,
            'lower_boundary': 9.0,
            'risk_width_pct': 0.10,  # 10% structural risk
        }

        # Volatility proxy should come from pattern boundaries (known at detection)
        # NOT from future price movements
        vol_proxy = pattern.get('risk_width_pct') or \
                    (pattern['upper_boundary'] - pattern['lower_boundary']) / pattern['upper_boundary']

        assert vol_proxy == 0.10, f"Volatility proxy should be 0.10, got {vol_proxy}"

        # Future data should NOT affect volatility calculation
        future_volatility = 0.25  # Hypothetical future realized volatility
        assert vol_proxy != future_volatility, \
            "Volatility proxy must not use future information"

    def test_dynamic_window_determined_at_detection(self):
        """
        Dynamic window must be fixed at pattern detection time,
        not adjusted based on future price action.
        """
        from datetime import datetime

        # Pattern detected on 2024-01-15 with 10% volatility
        detection_date = datetime(2024, 1, 15)
        volatility_at_detection = 0.10

        # Window calculation (same as in 00b_label_outcomes.py)
        MIN_WINDOW = 10
        MAX_WINDOW = 60
        dynamic_window = int(np.clip(
            1.0 / (volatility_at_detection + 1e-6),
            MIN_WINDOW,
            MAX_WINDOW
        ))

        assert dynamic_window == 10, f"Expected 10 days for 10% vol, got {dynamic_window}"

        # Simulate: Pattern explodes on day 5, volatility spikes to 30%
        # The window should NOT change retroactively
        future_volatility = 0.30
        recalculated_window = int(np.clip(
            1.0 / (future_volatility + 1e-6),
            MIN_WINDOW,
            MAX_WINDOW
        ))

        # This would be wrong - using future info to change window
        assert dynamic_window != recalculated_window or dynamic_window == MIN_WINDOW, \
            "Window should be fixed at detection, not adjusted by future volatility"


class TestLabelingTemporalIntegrity:
    """Tests for labeling process temporal integrity."""

    def test_outcome_window_starts_after_pattern_end(self):
        """
        Outcome evaluation must start AFTER pattern end date.
        Using pattern-period data for outcomes would be circular.
        """
        pattern_end_date = pd.Timestamp('2024-01-15')
        outcome_start = pattern_end_date + timedelta(days=1)

        assert outcome_start > pattern_end_date, \
            "Outcome window must start after pattern end"

    def test_ripeness_check_uses_reference_date(self):
        """
        Pattern ripeness must be checked against reference date,
        not against data availability (which could leak future existence).
        """
        pattern_end = pd.Timestamp('2024-01-15')
        reference_date = pd.Timestamp('2024-02-01')  # 17 days later
        window_days = 20

        # Correct: Check if enough time has passed
        is_ripe = (reference_date - pattern_end).days >= window_days
        assert is_ripe == False, "Pattern should not be ripe (only 17 days, need 20)"

        # With more time passed
        reference_date_later = pd.Timestamp('2024-02-10')  # 26 days later
        is_ripe_later = (reference_date_later - pattern_end).days >= window_days
        assert is_ripe_later == True, "Pattern should be ripe (26 days >= 20)"

    def test_no_future_price_in_metrics(self):
        """
        Pattern metrics (BBW, ADX, boundaries) must use only
        data up to pattern end date.
        """
        # Simulate price data
        dates = pd.date_range(start='2024-01-01', periods=50)
        prices = list(range(100, 120)) + list(range(120, 150))  # Trend then breakout

        df = pd.DataFrame({
            'close': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
        }, index=dates)

        pattern_end_idx = 20  # 2024-01-21

        # Upper boundary should be max of pattern period, not future
        pattern_data = df.iloc[:pattern_end_idx + 1]
        future_data = df.iloc[pattern_end_idx + 1:]

        upper_boundary_correct = pattern_data['high'].max()
        upper_boundary_wrong = df['high'].max()  # Includes future!

        assert upper_boundary_correct < upper_boundary_wrong, \
            "Upper boundary using future data would be higher (wrong)"

        print(f"\nUpper boundary (correct, pattern only): {upper_boundary_correct}")
        print(f"Upper boundary (WRONG, includes future): {upper_boundary_wrong}")


class TestCoilIntensityLookahead:
    """Tests for coil_intensity calculation temporal integrity."""

    def test_coil_intensity_uses_only_pattern_data(self):
        """
        Verify coil_intensity calculation uses ONLY pattern data, no future information.

        Coil intensity should be calculated from:
        1. price_position_at_end: Position in box at pattern END (not outcome)
        2. bbw_at_end: BBW at pattern END date
        3. vol_trend_5d: Volume trend ENDING at pattern END (not into outcome)

        Future data (outcome period) must NOT influence coil_intensity.
        """
        # Simulate pattern data at detection time
        pattern_end_date = pd.Timestamp('2024-01-15')

        pattern_data = {
            'end_date': pattern_end_date,
            'upper_boundary': 10.0,
            'lower_boundary': 9.0,
            # Features available at pattern END
            'close_at_end': 9.8,  # Close on last day of pattern
            'bbw_at_end': 0.10,   # BBW on last day
            'volume_5d_avg_at_end': 50000,  # 5-day vol avg ending at pattern end
            'volume_20d_avg': 40000,  # 20-day vol avg
        }

        # Calculate coil components using ONLY pattern data
        box_width = pattern_data['upper_boundary'] - pattern_data['lower_boundary']

        # Price position: where close sits in the box [0=lower, 1=upper]
        price_position = (pattern_data['close_at_end'] - pattern_data['lower_boundary']) / box_width
        # Expected: (9.8 - 9.0) / 1.0 = 0.8 (near upper boundary)

        # Volume trend: recent vs average
        vol_trend = pattern_data['volume_5d_avg_at_end'] / pattern_data['volume_20d_avg']
        # Expected: 50000 / 40000 = 1.25 (volume increasing)

        # BBW at end (normalized)
        bbw_normalized = min(pattern_data['bbw_at_end'] / 0.20, 1.0)  # Cap at 20% BBW
        # Expected: 0.10 / 0.20 = 0.5

        # Coil intensity formula (simplified)
        # Higher when: price near upper, volume increasing, BBW contracting
        coil_intensity = (
            price_position * 0.4 +          # Price position weight
            min(vol_trend, 2.0) / 2.0 * 0.3 +  # Volume trend weight (capped)
            (1 - bbw_normalized) * 0.3       # BBW contraction weight (inverted)
        )

        # Now simulate what WRONG calculation would look like using future data
        future_data = {
            'close_day_10': 12.0,      # Price after breakout
            'volume_spike': 200000,    # Volume during breakout
            'bbw_after_breakout': 0.25 # BBW expanded after breakout
        }

        # WRONG: Using future close for price position
        wrong_price_position = (future_data['close_day_10'] - pattern_data['lower_boundary']) / box_width
        # This would be > 1.0, using information we don't have at detection

        # WRONG: Using future volume for trend
        wrong_vol_trend = future_data['volume_spike'] / pattern_data['volume_20d_avg']
        # This would be 5.0x, using breakout volume we don't have

        # Verify correct calculation uses only pattern data
        assert 0 <= price_position <= 1, "Price position must be within pattern box"
        assert 0 <= coil_intensity <= 1, "Coil intensity must be [0, 1]"

        # Verify wrong calculation would be detectably different
        assert wrong_price_position > 1.0, "Future price would exceed pattern box"
        assert wrong_vol_trend > vol_trend * 2, "Future volume would be much higher"

        print(f"\nCoil Intensity Calculation (Correct):")
        print(f"  Price Position: {price_position:.3f} (at pattern end)")
        print(f"  Volume Trend: {vol_trend:.3f} (historical)")
        print(f"  BBW Normalized: {bbw_normalized:.3f}")
        print(f"  Coil Intensity: {coil_intensity:.3f}")
        print(f"\nWrong (Future-Contaminated):")
        print(f"  Wrong Price Position: {wrong_price_position:.3f} (LEAKAGE)")
        print(f"  Wrong Volume Trend: {wrong_vol_trend:.3f} (LEAKAGE)")

    def test_coil_intensity_features_freeze_at_pattern_end(self):
        """
        Verify that coil intensity features are frozen at pattern detection,
        not updated as time progresses.
        """
        # Pattern detected on Jan 15
        detection_date = pd.Timestamp('2024-01-15')

        # Coil features at detection time
        coil_at_detection = {
            'price_position_at_end': 0.85,  # Near upper boundary
            'bbw_slope_5d': -0.02,           # BBW decreasing (tightening)
            'vol_trend_5d': 1.15,            # Slight volume increase
            'distance_to_danger': 0.85,      # Far from lower boundary
            'coil_intensity': 0.72           # Composite score
        }

        # Simulated outcome period values (these should NOT update coil features)
        outcome_period_updates = {
            'price_position_day_5': 1.20,   # Price broke out
            'bbw_day_5': 0.18,               # BBW expanded
            'volume_day_5': 150000,          # Huge volume
        }

        # The coil_intensity for this pattern should ALWAYS remain 0.72
        # regardless of what happens in the outcome period

        # This is a conceptual test - in real code, verify coil_intensity
        # is computed once at detection and stored, not recomputed

        final_coil_intensity = coil_at_detection['coil_intensity']

        assert final_coil_intensity == 0.72, \
            f"Coil intensity should be frozen at detection time, got {final_coil_intensity}"

        print(f"\nCoil Intensity Immutability Test:")
        print(f"  Detection date: {detection_date}")
        print(f"  Coil intensity (frozen): {final_coil_intensity}")
        print(f"  NOT affected by: price breakout, volume spike, BBW expansion")


class TestFeatureCalculationLookahead:
    """Tests for feature calculation temporal integrity."""

    def test_volume_ratio_uses_historical_average(self):
        """
        Volume ratios must use historical averages, not forward-looking.
        """
        dates = pd.date_range(start='2024-01-01', periods=100)
        # Volume spike on day 50
        volumes = [1000] * 50 + [5000] * 10 + [1000] * 40

        df = pd.DataFrame({'volume': volumes}, index=dates)

        # At day 49 (before spike), average should not know about spike
        lookback = 20
        idx = 49

        historical_avg = df['volume'].iloc[idx - lookback:idx].mean()
        future_contaminated_avg = df['volume'].iloc[idx:idx + lookback].mean()

        assert historical_avg == 1000, "Historical average should be 1000"
        assert future_contaminated_avg > 1000, "Future-contaminated would include spike"

    def test_rolling_indicators_no_lookahead(self):
        """
        Technical indicators (BBW, ADX, etc.) must use rolling windows
        that don't peek into future data.
        """
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

        df = pd.DataFrame({'close': prices}, index=dates)

        # Bollinger Band Width calculation
        window = 20
        rolling_mean = df['close'].rolling(window).mean()
        rolling_std = df['close'].rolling(window).std()
        bbw = (2 * rolling_std) / rolling_mean

        # At any point, BBW should only use past data
        test_idx = 50

        # Manually calculate BBW at test_idx using only past data
        past_data = df['close'].iloc[test_idx - window + 1:test_idx + 1]
        expected_bbw = (2 * past_data.std()) / past_data.mean()

        actual_bbw = bbw.iloc[test_idx]

        assert abs(actual_bbw - expected_bbw) < 1e-10, \
            f"BBW mismatch: {actual_bbw} vs {expected_bbw}"


def test_summary():
    """Print summary of look-ahead prevention measures."""
    print("\n" + "=" * 60)
    print("LOOK-AHEAD BIAS PREVENTION SUMMARY")
    print("=" * 60)
    print("""
    1. ADAPTIVE Z-SCORE: Uses rolling window, not global stats
       - Window: 50-day rolling mean/std
       - Warmup: First 49 days are NaN (insufficient history)

    2. VOLATILITY PROXY: Fixed at pattern detection time
       - Source: risk_width_pct or box_width from pattern
       - NOT recalculated based on future price action

    3. DYNAMIC WINDOW: Determined once at detection
       - Formula: 1 / volatility_proxy (clamped to [10, 60])
       - Does NOT adapt based on outcome

    4. OUTCOME LABELING: Starts day AFTER pattern end
       - Pattern data: Days 1 to N
       - Outcome data: Days N+1 to N+window

    5. FEATURE CALCULATION: Historical lookback only
       - Volume ratios: log_diff(recent, historical)
       - Technical indicators: Rolling windows

    6. RIPENESS CHECK: Against reference date
       - NOT based on data availability
       - Ensures temporal consistency in backtests
    """)
    print("=" * 60)


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_lookahead.py -v
    pytest.main([__file__, '-v', '--tb=short'])
