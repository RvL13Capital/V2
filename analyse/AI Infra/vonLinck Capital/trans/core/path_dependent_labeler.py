"""
Path-Dependent Labeling v17
============================

Implements risk-based, path-dependent labeling for consolidation patterns.
Uses "First Event Wins" logic with Risk Multiples (R) instead of fixed percentages.

Key Features:
- Risk-based classification (adapts to pattern volatility)
- Path-dependent evaluation (exits on first significant event)
- Next-open confirmation for targets (filters wicks)
- Grey zone exclusion (ambiguous patterns removed)
- Proper indicator warm-up handling
- Data integrity validation

Classes:
    0: Danger (Breakdown) - Capital destruction
    1: Noise (Base case) - Opportunity cost
    2: Target (Home Run) - High conviction winner
   -1: Grey Zone - Excluded from training

Author: TRANS System v17
Date: November 2024
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta

# Import market cap profiles for dynamic parameters
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.market_cap_profiles import get_profile, AssetProfile, MID_CAP_PROFILE

logger = logging.getLogger(__name__)


class PathDependentLabelerV17:
    """
    Path-dependent labeling system with proper temporal integrity.

    This labeler ensures no look-ahead bias by:
    1. Using indicator warm-up periods before pattern detection
    2. Only labeling after complete outcome windows
    3. Validating data integrity without filtering legitimate volatility
    4. Using risk-based thresholds that adapt to each pattern
    """

    def __init__(
        self,
        indicator_warmup: int = 30,
        indicator_stable: int = 100,
        outcome_window: int = 100,
        risk_multiplier_target: Optional[float] = None,
        risk_multiplier_grey: Optional[float] = None,
        stop_buffer: Optional[float] = None,
        halt_threshold_days: int = 7,
        max_overnight_gap: float = 0.30,  # 30% overnight gap threshold
        use_dynamic_profiles: bool = True,  # Enable market cap profiles
        default_market_cap_category: Optional[str] = None,  # Default category if not specified per pattern
        # ATR-based labeling parameters (Jan 2026)
        use_atr_labeling: bool = False,  # Use ATR instead of R-multiples
        atr_period: int = 14,            # ATR calculation period
        atr_stop_multiple: float = 2.0,  # Stop = lower - (N × ATR)
        atr_target_multiple: float = 5.0,  # Target = entry + (N × ATR)
        atr_grey_multiple: float = 2.5   # Grey = entry + (N × ATR)
    ):
        """
        Initialize the path-dependent labeler with dynamic market cap profiles.

        Args:
            indicator_warmup: Days needed for indicators to stabilize
            indicator_stable: Days of stable indicator data to use
            outcome_window: Days to track after pattern completion
            risk_multiplier_target: R-multiple for target (None = use profile default)
            risk_multiplier_grey: R-multiple for grey zone (None = use profile default)
            stop_buffer: Buffer below lower boundary for stop loss (None = use profile default)
            halt_threshold_days: Max days between consecutive trading days
            max_overnight_gap: Maximum overnight gap before flagging
            use_dynamic_profiles: If True, use market cap profiles; if False, use static defaults
            default_market_cap_category: Default market cap category (e.g., 'small_cap')
            use_atr_labeling: If True, use ATR-based stops/targets instead of R-multiples
            atr_period: Period for ATR calculation (default: 14)
            atr_stop_multiple: Stop = lower_boundary - (N × ATR) (default: 2.0)
            atr_target_multiple: Target = entry + (N × ATR) (default: 5.0)
            atr_grey_multiple: Grey zone = entry + (N × ATR) (default: 2.5)
        """
        self.indicator_warmup = indicator_warmup
        self.indicator_stable = indicator_stable
        self.outcome_window = outcome_window
        self.halt_threshold_days = halt_threshold_days
        self.max_overnight_gap = max_overnight_gap
        self.use_dynamic_profiles = use_dynamic_profiles

        # ATR-based labeling parameters
        self.use_atr_labeling = use_atr_labeling
        self.atr_period = atr_period
        self.atr_stop_multiple = atr_stop_multiple
        self.atr_target_multiple = atr_target_multiple
        self.atr_grey_multiple = atr_grey_multiple

        # Get default profile
        self.default_profile = get_profile(default_market_cap_category)

        # Set parameters (use provided values or profile defaults)
        if use_dynamic_profiles:
            self.risk_multiplier_target = risk_multiplier_target or self.default_profile.target_r
            self.risk_multiplier_grey = risk_multiplier_grey or self.default_profile.grey_zone_r
            self.stop_buffer = stop_buffer or self.default_profile.stop_buffer
        else:
            # Static defaults (backward compatibility)
            self.risk_multiplier_target = risk_multiplier_target or 3.0
            self.risk_multiplier_grey = risk_multiplier_grey or 2.5
            self.stop_buffer = stop_buffer or 0.02

        # Total data requirements
        self.min_history_required = indicator_warmup + indicator_stable
        self.min_total_required = self.min_history_required + outcome_window

        logger.info(f"Initialized PathDependentLabelerV17")
        logger.info(f"  - Labeling Mode: {'ATR-based' if use_atr_labeling else ('Dynamic Profiles' if use_dynamic_profiles else 'Static R-Multiple')}")
        if use_atr_labeling:
            logger.info(f"  - ATR Period: {atr_period} days")
            logger.info(f"  - Stop: lower - ({atr_stop_multiple} × ATR)")
            logger.info(f"  - Target: entry + ({atr_target_multiple} × ATR)")
            logger.info(f"  - Grey Zone: entry + ({atr_grey_multiple} × ATR)")
        else:
            if use_dynamic_profiles:
                logger.info(f"  - Default Profile: {self.default_profile.category.value}")
            logger.info(f"  - Target: {self.risk_multiplier_target}R")
            logger.info(f"  - Grey Zone: {self.risk_multiplier_grey}R")
            logger.info(f"  - Stop Buffer: {self.stop_buffer*100:.1f}%")
        logger.info(f"  - Warmup: {indicator_warmup} days")
        logger.info(f"  - Stable: {indicator_stable} days")
        logger.info(f"  - Outcome: {outcome_window} days")

    def label_pattern(
        self,
        full_data: pd.DataFrame,
        pattern_end_idx: int,
        pattern_boundaries: Optional[Dict[str, float]] = None,
        return_metadata: bool = False,
        market_cap_category: Optional[str] = None  # Dynamic profile resolution
    ) -> Optional[int]:
        """
        Label a pattern using path-dependent logic with risk multiples.

        Args:
            full_data: Complete DataFrame with warmup + pattern + outcome
            pattern_end_idx: Index where pattern completes
            pattern_boundaries: Optional dict with 'upper' and 'lower' boundaries
            return_metadata: If True, return (label, metadata_dict) instead of just label
            market_cap_category: Market cap category for dynamic profile (e.g., 'small_cap')
                                If None, uses default profile from __init__

        Returns:
            If return_metadata=False: Label: 0 (Danger), 1 (Noise), 2 (Target), -1 (Grey Zone), None (Invalid)
            If return_metadata=True: Tuple of (label, metadata_dict) where metadata contains:
                - outcome_day: Day (1-100) when outcome occurred (None for Noise)
                - entry_price, stop_loss, target_price, risk_unit
                - market_cap_category: Category used for this pattern

        Temporal Guarantee:
            This function should only be called after pattern_end_idx + outcome_window
            to ensure no look-ahead bias.
        """
        # Resolve profile for this specific pattern
        if self.use_dynamic_profiles:
            profile = get_profile(market_cap_category) if market_cap_category else self.default_profile
            risk_multiplier_target = profile.target_r
            risk_multiplier_grey = profile.grey_zone_r
            stop_buffer = profile.stop_buffer
        else:
            # Use static parameters
            risk_multiplier_target = self.risk_multiplier_target
            risk_multiplier_grey = self.risk_multiplier_grey
            stop_buffer = self.stop_buffer
            profile = None
        # Step 1: Validate sufficient data
        if pattern_end_idx < self.min_history_required:
            logger.warning(f"Insufficient history: {pattern_end_idx} < {self.min_history_required}")
            return None

        if pattern_end_idx + self.outcome_window >= len(full_data):
            logger.warning(f"Insufficient future data: need up to {pattern_end_idx + self.outcome_window}, have {len(full_data)}")
            return None

        # Step 2: Extract data slices
        warmup_start = pattern_end_idx - self.min_history_required
        pattern_data = full_data.iloc[warmup_start:pattern_end_idx+1].copy()
        outcome_data = full_data.iloc[pattern_end_idx+1:pattern_end_idx+self.outcome_window+1].copy()

        # Step 3: Validate data integrity
        if not self.validate_data_integrity(pattern_data, outcome_data):
            logger.debug("Data integrity validation failed")
            return None

        # Step 4: Calculate risk parameters from STABLE period only
        stable_data = pattern_data.iloc[self.indicator_warmup:].copy()

        # CRITICAL FIX: Use adj_close for split-adjusted pricing
        # Prevents reverse splits from creating false 900%+ gains
        price_col = 'adj_close' if 'adj_close' in full_data.columns else 'close'
        if price_col == 'close':
            ticker_str = full_data.get('ticker', ['UNKNOWN'])[0] if 'ticker' in full_data.columns else 'UNKNOWN'
            logger.warning(f"{ticker_str}: Using 'close' for entry price - not split-adjusted! Reverse splits will create false gains.")

        # Entry price is at the pattern END (last day of pattern)
        entry_price = full_data.iloc[pattern_end_idx][price_col]

        # Get lower boundary for stop calculation
        if pattern_boundaries and 'lower' in pattern_boundaries:
            lower_boundary = pattern_boundaries['lower']
        else:
            # Default: 20-day low
            recent_window = stable_data.iloc[-20:]
            lower_boundary = recent_window['low'].min()

        # ATR-based or R-multiple based calculation
        atr_value = None
        if self.use_atr_labeling:
            # ATR-based labeling (Jan 2026)
            # Calculate ATR at pattern end using lookback before pattern end
            from core.aiv7_components.indicators import calculate_atr
            lookback_start = max(0, pattern_end_idx - self.atr_period * 2)
            lookback_data = full_data.iloc[lookback_start:pattern_end_idx + 1].copy()
            atr_series = calculate_atr(lookback_data, period=self.atr_period)
            atr_value = atr_series.iloc[-1]

            if pd.isna(atr_value) or atr_value <= 0:
                logger.warning(f"Invalid ATR value: {atr_value}")
                return None

            # ATR-based calculations
            stop_loss = lower_boundary - (self.atr_stop_multiple * atr_value)
            target_price = entry_price + (self.atr_target_multiple * atr_value)
            grey_threshold = entry_price + (self.atr_grey_multiple * atr_value)
            R = atr_value  # Risk unit = ATR for consistent reporting

            logger.debug(f"ATR labeling: ATR={atr_value:.4f}, stop={stop_loss:.2f}, target={target_price:.2f}")
        else:
            # R-multiple based labeling (original)
            stop_loss = lower_boundary * (1 - stop_buffer)
            R = lower_boundary - stop_loss

            if R <= 0:
                logger.warning(f"Invalid risk unit: R={R:.4f} (lower={lower_boundary:.2f}, stop={stop_loss:.2f})")
                return None

            # R-multiple based calculations
            target_price = entry_price + (risk_multiplier_target * R)
            grey_threshold = entry_price + (risk_multiplier_grey * R)

        # Initialize metadata for tracking
        outcome_day = None
        label = None
        ambiguous_day_count = 0  # Track pessimistic execution triggers

        # ==========================================================================
        # DAILY OUTCOME EVALUATION WITH PESSIMISTIC EXECUTION RULE
        # ==========================================================================
        # WHY PESSIMISTIC EXECUTION IS NECESSARY FOR EOD DATA:
        #
        # End-of-Day (EOD) data provides: Open, High, Low, Close for each day.
        # It does NOT tell us the INTRADAY SEQUENCE of price movements.
        #
        # Problem: On a volatile day, we might see:
        #   - Day_Low = $9.50 (below stop at $10.00)
        #   - Day_High = $15.00 (above target at $14.00)
        #
        # Did the stock hit the stop first, then rally to target?
        # Or did it hit target first, then crash through the stop?
        # With EOD data, we simply cannot know.
        #
        # SOLUTION: PESSIMISTIC EXECUTION (Conservative Assumption)
        #   If BOTH stop and target are triggered on the same day:
        #   -> Assume STOP HIT FIRST (worst-case execution for the trader)
        #   -> Label as DANGER (Class 0)
        #
        # This is the ONLY safe assumption because:
        #   1. Backtests using EOD data cannot simulate intraday order execution
        #   2. Assuming target-first would overstate performance
        #   3. Real trading would face slippage/gaps that favor stops triggering first
        #   4. Models trained with optimistic assumptions fail in live trading
        #
        # The warning log helps track how often this ambiguity affects labeling.
        # ==========================================================================

        # Get ticker for logging (if available in the data)
        ticker_for_log = 'UNKNOWN'
        if 'ticker' in full_data.columns:
            ticker_for_log = full_data['ticker'].iloc[0] if len(full_data) > 0 else 'UNKNOWN'

        for t in range(len(outcome_data)):
            # Extract price data for this day
            day_data = outcome_data.iloc[t]
            low_t = day_data['low']
            high_t = day_data['high']
            close_t = day_data[price_col]

            # Get date for logging (if DatetimeIndex)
            if isinstance(outcome_data.index, pd.DatetimeIndex):
                day_date = outcome_data.index[t].strftime('%Y-%m-%d')
            else:
                day_date = f"Day {t+1}"

            # =================================================================
            # PESSIMISTIC EXECUTION RULE (Jan 2026)
            # =================================================================
            # Check for AMBIGUOUS intraday sequence FIRST:
            # If BOTH stop AND target could have been hit on the same day,
            # we CANNOT determine the true sequence from EOD data.
            # Default to STOP HIT (Danger) as the pessimistic assumption.
            # =================================================================
            stop_triggered = low_t <= stop_loss
            target_triggered = high_t >= target_price

            if stop_triggered and target_triggered:
                # AMBIGUOUS CASE: Both stop and target hit on same day
                # With EOD data, we cannot know which happened first.
                # PESSIMISTIC EXECUTION: Assume stop hit first (worst case for trader)
                ambiguous_day_count += 1
                logger.warning(
                    f"PESSIMISTIC EXECUTION: Ambiguous intraday sequence on {day_date} for {ticker_for_log}. "
                    f"Both stop (low={low_t:.2f} <= {stop_loss:.2f}) and target (high={high_t:.2f} >= {target_price:.2f}) "
                    f"triggered. Defaulting to STOP HIT (Danger) - cannot determine true sequence from EOD data."
                )
                outcome_day = t + 1  # Day 1 is first day after pattern end
                label = 0  # Danger (pessimistic assumption)
                break

            # Priority 1: Check for breakdown using LOW (worst-case timing)
            # This handles the unambiguous case where ONLY stop is triggered
            if stop_triggered:
                logger.debug(f"Breakdown detected on day {t+1}: low {low_t:.2f} <= {stop_loss:.2f}")
                outcome_day = t + 1  # Day 1 is first day after pattern end
                label = 0  # Danger
                break

            # Priority 2: Check for target with next-open confirmation
            # Target requires CLOSE >= target AND next day OPEN >= target
            # This filters out intraday wicks that don't hold
            if t < len(outcome_data) - 1:
                open_next = outcome_data.iloc[t + 1]['open']
                if close_t >= target_price and open_next >= target_price:
                    logger.debug(f"Target hit with confirmation on day {t+1}: {close_t:.2f} >= {target_price:.2f}")
                    outcome_day = t + 1
                    label = 2  # Target/Home Run
                    break

        # Post-loop: Check for noise (grey zone now converted to Noise)
        # FIX Jan 2026: Grey zone patterns now labeled as Noise (Class 1)
        # Rationale: If it didn't hit target with confirmation, it's effectively noise
        # This adds ~10-15% more training data and teaches model to reject ambiguous patterns
        if label is None:
            max_close = outcome_data[price_col].max()

            if max_close >= grey_threshold:
                # CHANGED: Grey zone -> Noise (opportunity cost)
                logger.debug(f"Grey zone -> Noise: max {max_close:.2f} >= {grey_threshold:.2f} but no confirmation")
                label = 1  # Noise (was -1 Grey Zone)
            else:
                # Default: Noise (stayed in range)
                logger.debug(f"Noise: price stayed between {stop_loss:.2f} and {grey_threshold:.2f}")
                label = 1  # Noise

        # Return with or without metadata
        if return_metadata:
            metadata = {
                'outcome_day': outcome_day,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'risk_unit': R,
                'grey_threshold': grey_threshold,
                'max_close': outcome_data[price_col].max(),
                'min_close': outcome_data[price_col].min(),
                'market_cap_category': market_cap_category,
                'profile_used': profile.category.value if profile else 'static',
                'stop_buffer_pct': stop_buffer * 100 if not self.use_atr_labeling else None,
                'target_r_multiple': risk_multiplier_target if not self.use_atr_labeling else None,
                'grey_r_multiple': risk_multiplier_grey if not self.use_atr_labeling else None,
                # Pessimistic Execution tracking (Jan 2026)
                # Tracks how many patterns were labeled Danger due to ambiguous intraday sequence
                'pessimistic_execution_applied': ambiguous_day_count > 0,
                # ATR-based labeling metadata
                'labeling_mode': 'atr' if self.use_atr_labeling else 'r_multiple',
                'atr_value': atr_value,
                'atr_period': self.atr_period if self.use_atr_labeling else None,
                'atr_stop_multiple': self.atr_stop_multiple if self.use_atr_labeling else None,
                'atr_target_multiple': self.atr_target_multiple if self.use_atr_labeling else None,
            }
            return label, metadata

        return label

    def validate_data_integrity(
        self,
        history: pd.DataFrame,
        future: pd.DataFrame
    ) -> bool:
        """
        Validate data integrity WITHOUT filtering legitimate volatility.

        Args:
            history: Historical data before pattern
            future: Outcome data after pattern

        Returns:
            True if data is valid, False otherwise
        """
        all_data = pd.concat([history, future])

        # Check 1: Trading halts (max gap between consecutive days)
        if not self._check_trading_halts(all_data):
            return False

        # Check 2: Data errors vs legitimate gaps
        if not self._check_data_consistency(all_data):
            return False

        # Check 3: OHLC relationship validity
        if not self._check_ohlc_validity(all_data):
            return False

        # Check 4: Sufficient non-zero volume days
        if not self._check_volume_validity(all_data):
            return False

        return True

    def _check_trading_halts(self, df: pd.DataFrame) -> bool:
        """Check for extended trading halts."""
        # Get date series - handle both DatetimeIndex and 'date' column
        if isinstance(df.index, pd.DatetimeIndex):
            date_series = df.index.to_series()
        elif 'date' in df.columns:
            date_series = pd.to_datetime(df['date'])
        else:
            # No date info available, skip halt check
            logger.debug("No date information available for halt check")
            return True

        # Remove timezone info if present (causes comparison issues)
        if hasattr(date_series.dtype, 'tz') and date_series.dt.tz is not None:
            date_series = date_series.dt.tz_localize(None)

        # Sort by date to ensure proper diff calculation
        date_series = date_series.sort_values()

        # Calculate gaps
        gaps = date_series.diff()
        max_gap = gaps.max()

        if pd.isna(max_gap):
            return True

        # Ensure max_gap is a timedelta (not float/int)
        if not isinstance(max_gap, (pd.Timedelta, timedelta)):
            # If gaps are numeric (days as floats), convert to timedelta
            try:
                max_gap = pd.Timedelta(days=float(max_gap))
                gaps = gaps.apply(lambda x: pd.Timedelta(days=float(x)) if pd.notna(x) else pd.NaT)
            except (TypeError, ValueError):
                logger.debug(f"Could not convert gap to timedelta: {type(max_gap)}")
                return True

        threshold = timedelta(days=self.halt_threshold_days)
        if max_gap > threshold:
            # Check if it's a known holiday period
            problem_indices = gaps[gaps > threshold].index

            for idx in problem_indices:
                # Get the actual date for this index
                if isinstance(df.index, pd.DatetimeIndex):
                    date = df.index[idx] if isinstance(idx, int) else idx
                elif 'date' in df.columns:
                    date = pd.to_datetime(df.loc[idx, 'date']) if idx in df.index else None
                else:
                    date = None

                if date is None:
                    continue

                # Allow December 24 - January 2 holiday period
                if hasattr(date, 'month') and hasattr(date, 'day'):
                    if (date.month == 12 and date.day >= 24) or (date.month == 1 and date.day <= 2):
                        continue

                gap_days = max_gap.days if hasattr(max_gap, 'days') else max_gap
                logger.debug(f"Trading halt detected: {gap_days} days gap at {date}")
                return False

        return True

    def _check_data_consistency(self, df: pd.DataFrame) -> bool:
        """
        Check for data errors WITHOUT filtering legitimate volatility.
        Distinguishes between gaps (open vs prev close) and volatility (high vs low).
        """
        if len(df) < 2:
            return True

        # Calculate overnight gaps (open vs previous close)
        opens = df['open'].values[1:]
        prev_closes = df['close'].values[:-1]

        # Avoid division by zero
        prev_closes = np.where(prev_closes == 0, 1e-10, prev_closes)
        gaps = np.abs(opens / prev_closes - 1)

        # Check for large overnight gaps
        large_gap_mask = gaps > self.max_overnight_gap

        if large_gap_mask.any():
            # Additional validation: Check if it's a consistent move (possible split)
            large_gap_indices = np.where(large_gap_mask)[0] + 1  # +1 because we started from index 1

            for idx in large_gap_indices:
                if idx >= len(df):
                    continue

                gap_ratio = df.iloc[idx]['open'] / df.iloc[idx-1]['close']

                # Check if high and low moved by similar ratio (indicates split/dividend)
                high_ratio = df.iloc[idx]['high'] / df.iloc[idx-1]['high']
                low_ratio = df.iloc[idx]['low'] / df.iloc[idx-1]['low']

                # If ratios are consistent, it's likely a split (acceptable)
                ratio_consistency = abs(high_ratio - gap_ratio) < 0.05 and abs(low_ratio - gap_ratio) < 0.05

                if not ratio_consistency:
                    logger.debug(f"Inconsistent gap detected at index {idx}: gap={gap_ratio:.2f}")
                    return False

        # Explicitly ALLOW high intraday volatility (we're looking for volatile stocks!)
        # Do NOT filter based on (High - Low) / Low

        return True

    def _check_ohlc_validity(self, df: pd.DataFrame) -> bool:
        """Check for impossible OHLC relationships."""
        # High must be >= Low
        if (df['high'] < df['low']).any():
            logger.debug("Invalid OHLC: High < Low")
            return False

        # Close must be between Low and High
        if ((df['close'] > df['high']) | (df['close'] < df['low'])).any():
            logger.debug("Invalid OHLC: Close outside High-Low range")
            return False

        # Open must be between Low and High
        if ((df['open'] > df['high']) | (df['open'] < df['low'])).any():
            logger.debug("Invalid OHLC: Open outside High-Low range")
            return False

        # Check for zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        if (df[price_cols] <= 0).any().any():
            logger.debug("Invalid prices: Zero or negative values")
            return False

        return True

    def _check_volume_validity(self, df: pd.DataFrame) -> bool:
        """Check for sufficient valid volume data."""
        if 'volume' not in df.columns:
            return True  # Volume not required

        # Calculate percentage of valid volume days
        valid_volume_days = (df['volume'] > 0).sum()
        total_days = len(df)
        valid_percentage = valid_volume_days / total_days if total_days > 0 else 0

        # Require at least 80% valid volume days
        if valid_percentage < 0.80:
            logger.debug(f"Insufficient valid volume: {valid_percentage:.1%} < 80%")
            return False

        return True

    def calculate_risk_metrics(
        self,
        entry_price: float,
        stop_loss: float,
        outcome_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate risk-based metrics for a pattern.

        Args:
            entry_price: Entry price at pattern completion
            stop_loss: Stop loss price
            outcome_data: DataFrame with outcome window data

        Returns:
            Dictionary with risk metrics
        """
        R = entry_price - stop_loss

        if R <= 0:
            return {
                'risk_unit': 0,
                'max_r_multiple': 0,
                'min_r_multiple': 0,
                'risk_reward_achieved': 0
            }

        # CRITICAL FIX: Use adj_close for split-adjusted pricing
        price_col = 'adj_close' if 'adj_close' in outcome_data.columns else 'close'

        # Calculate R-multiples achieved
        max_price = outcome_data[price_col].max()
        min_price = outcome_data[price_col].min()

        max_r_multiple = (max_price - entry_price) / R
        min_r_multiple = (min_price - entry_price) / R

        # Risk-reward achieved
        if min_r_multiple < -1:  # Stop hit
            risk_reward_achieved = -1
        else:
            risk_reward_achieved = max_r_multiple

        return {
            'risk_unit': R,
            'max_r_multiple': max_r_multiple,
            'min_r_multiple': min_r_multiple,
            'risk_reward_achieved': risk_reward_achieved,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target_price': entry_price + (self.risk_multiplier_target * R)
        }


def label_pattern_simple(
    pattern_data: pd.DataFrame,
    outcome_data: pd.DataFrame,
    risk_multiplier_target: float = 3.0,
    risk_multiplier_grey: float = 2.5,
    stop_buffer: float = 0.02
) -> int:
    """
    Simplified labeling function for quick testing.

    Implements PESSIMISTIC EXECUTION rule for EOD data integrity:
    If both stop and target are triggered on the same day, assumes stop hit first.

    Args:
        pattern_data: Data up to pattern completion
        outcome_data: 100-day outcome window
        risk_multiplier_target: R-multiple for target
        risk_multiplier_grey: R-multiple for grey zone
        stop_buffer: Buffer below boundary for stop

    Returns:
        Label: 0 (Danger), 1 (Noise), 2 (Target)
    """
    # CRITICAL FIX: Use adj_close for split-adjusted pricing
    price_col = 'adj_close' if 'adj_close' in pattern_data.columns else 'close'

    entry_price = pattern_data.iloc[-1][price_col]
    recent_low = pattern_data.iloc[-20:]['low'].min()
    stop_loss = recent_low * (1 - stop_buffer)

    R = entry_price - stop_loss
    if R <= 0:
        return 1  # Default to noise if invalid

    target = entry_price + (risk_multiplier_target * R)
    grey_threshold = entry_price + (risk_multiplier_grey * R)

    # Path-dependent evaluation with PESSIMISTIC EXECUTION
    for t in range(len(outcome_data) - 1):
        low_t = outcome_data.iloc[t]['low']
        high_t = outcome_data.iloc[t]['high']
        close_t = outcome_data.iloc[t][price_col]

        # =================================================================
        # PESSIMISTIC EXECUTION RULE (Jan 2026)
        # If BOTH stop AND target could have been hit on the same day,
        # we CANNOT determine the true sequence from EOD data.
        # Default to STOP HIT (Danger) as the pessimistic assumption.
        # =================================================================
        stop_triggered = low_t <= stop_loss
        target_triggered = high_t >= target

        if stop_triggered and target_triggered:
            # Ambiguous case: both triggered same day -> assume stop hit first
            return 0  # Danger (pessimistic)

        # Check breakdown using LOW (unambiguous case)
        if stop_triggered:
            return 0

        # Check target with next-open confirmation (uses Close)
        if close_t >= target:
            open_next = outcome_data.iloc[t + 1]['open']
            if open_next >= target:
                return 2

    # Check grey zone -> now converted to Noise (FIX Jan 2026)
    # If max close hit grey threshold but no confirmed target, it's still Noise
    if outcome_data[price_col].max() >= grey_threshold:
        return 1  # Noise (was -1 Grey Zone)

    return 1  # Noise


def apply_labels_to_patterns(
    patterns: list,
    df: pd.DataFrame,
    labeler: PathDependentLabelerV17 = None
) -> list:
    """
    Apply V17 path-dependent labels to detected patterns.

    This function is the single source of truth for labeling after detection.
    It ensures adj_close is used for split-adjusted outcome calculations.

    Args:
        patterns: List of pattern dictionaries from scanner (must have start_date, end_date, upper_boundary, lower_boundary)
        df: DataFrame with OHLCV + adj_close data (DatetimeIndex required)
        labeler: Optional PathDependentLabelerV17 instance (creates default if None)

    Returns:
        Labeled patterns with 'outcome_class' field set (0=Danger, 1=Noise, 2=Target)
        Patterns that cannot be labeled (insufficient data, grey zone) are filtered out.

    Raises:
        ValueError: If df missing adj_close (critical for correct labeling)

    Example:
        from core.path_dependent_labeler import apply_labels_to_patterns
        patterns = scanner.scan_ticker(ticker, df=df).patterns
        labeled = apply_labels_to_patterns(patterns, df)
    """
    # CRITICAL: Verify adj_close exists
    if 'adj_close' not in df.columns:
        raise ValueError(
            "DataFrame missing 'adj_close' column. "
            "Labeling requires split-adjusted prices to prevent false 900%+ gains from reverse splits. "
            "Add adj_close to data or use load_ticker_data with adj_close support."
        )

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'date' column")

    # Create labeler if not provided
    if labeler is None:
        labeler = PathDependentLabelerV17()

    labeled_patterns = []
    skipped_grey = 0
    skipped_invalid = 0

    for pattern in patterns:
        try:
            # Get pattern end index
            end_date = pd.to_datetime(pattern['end_date'])
            if end_date not in df.index:
                # Find nearest date
                idx_pos = df.index.get_indexer([end_date], method='nearest')[0]
                if idx_pos < 0 or idx_pos >= len(df):
                    skipped_invalid += 1
                    continue
                pattern_end_idx = idx_pos
            else:
                pattern_end_idx = df.index.get_loc(end_date)

            # Get boundaries
            pattern_boundaries = {
                'upper': pattern.get('upper_boundary'),
                'lower': pattern.get('lower_boundary')
            }

            # Apply label
            label = labeler.label_pattern(
                full_data=df,
                pattern_end_idx=pattern_end_idx,
                pattern_boundaries=pattern_boundaries,
                market_cap_category=pattern.get('market_cap_category')
            )

            # Filter invalid and grey zone
            if label is None:
                skipped_invalid += 1
                continue
            if label == -1:
                skipped_grey += 1
                continue

            # Update pattern with label
            pattern = pattern.copy()
            pattern['outcome_class'] = label
            pattern['labeling_version'] = 'v17'
            labeled_patterns.append(pattern)

        except Exception as e:
            logger.warning(f"Failed to label pattern {pattern.get('pattern_id', 'unknown')}: {e}")
            skipped_invalid += 1
            continue

    if skipped_grey > 0 or skipped_invalid > 0:
        logger.info(f"Labeling complete: {len(labeled_patterns)} labeled, {skipped_grey} grey zone, {skipped_invalid} invalid")

    return labeled_patterns