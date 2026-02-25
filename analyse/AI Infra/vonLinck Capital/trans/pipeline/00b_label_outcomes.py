"""
Step 0b: Label Pattern Outcomes (Structural Risk Labeling)
===========================================================

Implements Structural Risk Labeling using R-multiples with realistic EOD execution.

STRUCTURAL RISK FRAMEWORK:
    1. Technical_Floor = Pattern_Lower_Boundary (chart support/invalidation)
    2. Trigger_Price = Pattern_Upper_Boundary + $0.01 (breakout confirmation)
    3. R_Dollar = Trigger_Price - Technical_Floor (structural risk per share)
    4. Entry_Price = MAX(Trigger_Price, Open_{t+1}) (gap-adjusted entry)

GAP LIMIT RULE:
    If Entry_Price > (Trigger_Price + 0.5 * R_Dollar), mark as untradeable=True.
    Rationale: Massive gaps destroy R:R for retail execution, BUT they are strong
    momentum signals. The model should learn from them even though we can't trade them.
    FIX (Jan 2026): Previously these were excluded as NO_FILL - now included with flag.

OUTCOME CLASSES (40-day window):
    - Class 2 (TARGET): Entry + 3R reached AND max_volume > 2x vol_20d_avg
    - Class 0 (DANGER): Close below Technical_Floor (structural failure)
    - Class 1 (NOISE): Neither target nor floor hit within window

OUTPUT:
    - R_Multiplier achieved for every pattern (realized R:R)
    - labeled_patterns.parquet with outcome_class and R-metrics

Usage:
    python 00b_label_outcomes.py --input output/candidate_patterns.parquet
    python 00b_label_outcomes.py --input output/candidate_patterns.parquet --reference-date 2024-01-01
    python 00b_label_outcomes.py --input output/candidate_patterns.parquet --dry-run
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.aiv7_components import DataLoader
from utils.logging_config import setup_pipeline_logging
from config import (
    DEFAULT_OUTPUT_DIR,
    INDICATOR_WARMUP_DAYS,
    INDICATOR_STABLE_DAYS
)
from config.constants import (
    WEEKLY_STOP_BOX_FRACTION,
    WEEKLY_TARGET_R_MULTIPLE,
    WEEKLY_OUTCOME_WINDOW_DAYS
)

# Setup centralized logging
logger = setup_pipeline_logging('00b_label_outcomes')

# =============================================================================
# STRUCTURAL RISK LABELING CONSTANTS
# =============================================================================

# Outcome window (days after trigger)
OUTCOME_WINDOW_DAYS = 40  # Legacy fixed window (used when USE_DYNAMIC_WINDOW=False)

# Dynamic labeling window (V22 - Volatility-Adjusted)
# High volatility stocks resolve faster → shorter window
# Low volatility stocks need more time → longer window
# Formula: dynamic_days = 1 / volatility_proxy (clamped to [MIN, MAX])
MIN_OUTCOME_WINDOW = 10   # Minimum days for high-vol stocks
MAX_OUTCOME_WINDOW = 60   # Maximum days for low-vol stocks
USE_DYNAMIC_WINDOW = True # Toggle for A/B testing

# R-Multiple targets
TARGET_R_MULTIPLE = 3.0  # +3R for Target classification
GAP_LIMIT_R = 0.5        # Max gap before NO_FILL (0.5R above trigger)

# Volume requirement for Target (Jan 2026 fix: sustained + absolute minimum)
# To qualify as TARGET, must have 3+ CONSECUTIVE days where EACH day meets BOTH:
#   1. Volume > 2x 20d avg (relative surge)
#   2. Dollar volume > $50k (absolute minimum - prevents dormant stock false targets)
VOLUME_MULTIPLIER_TARGET = 2.0  # Must see 2x 20d avg volume
VOLUME_SUSTAINED_DAYS = 3       # Must sustain for 3+ consecutive days
MIN_DOLLAR_VOLUME_PER_DAY = 50_000  # Each day must have at least $50k traded

# Zombie Protocol: Force Danger after this many days without sufficient data
ZOMBIE_TIMEOUT = 150  # days

# Trigger offset from upper boundary
TRIGGER_OFFSET = 0.01  # $0.01 above upper boundary


def parse_args():
    parser = argparse.ArgumentParser(
        description='Label pattern outcomes using Structural Risk Labeling (R-multiples)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Structural Risk Framework:
    Technical_Floor = Lower_Boundary (chart support)
    Trigger_Price = Upper_Boundary + $0.01
    R_Dollar = Trigger_Price - Technical_Floor
    Entry_Price = MAX(Trigger_Price, Open_{t+1})

Gap Limit:
    If Entry_Price > Trigger_Price + 0.5R, pattern is marked untradeable=True (still labeled)

Outcome Classes (40-day window):
    Class 2 (TARGET): Entry + 3R hit AND volume > 2x 20d avg
    Class 0 (DANGER): Close below Technical_Floor
    Class 1 (NOISE): Neither target nor floor hit

Examples:
    python 00b_label_outcomes.py --input output/candidate_patterns.parquet
    python 00b_label_outcomes.py --input output/candidate_patterns.parquet --reference-date 2024-01-01
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to candidate_patterns.parquet'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for labeled patterns (default: labeled_patterns.parquet)'
    )
    parser.add_argument(
        '--reference-date',
        type=str,
        default=None,
        help='Reference date for ripeness check (default: today). Format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be labeled without writing output'
    )
    parser.add_argument(
        '--force-relabel',
        action='store_true',
        help='Re-label patterns that already have outcome_class (not recommended)'
    )
    parser.add_argument(
        '--enable-early-labeling',
        action='store_true',
        help='Enable early labeling for patterns that hit Target or Danger before 40 days. '
             'Maintains temporal integrity: only labels when today >= end_date + outcome_day.'
    )

    return parser.parse_args()


def get_volatility_proxy(pattern_row: pd.Series) -> float:
    """
    Get volatility proxy for dynamic window calculation.

    Uses risk_width_pct (structural volatility) as primary metric.
    Higher value = more volatile = shorter window needed.

    Args:
        pattern_row: Pattern metadata row (Series or dict-like)

    Returns:
        Volatility proxy (0.01 to 0.50 typical range)
    """
    # Primary: risk_width_pct (structural stop distance as % of price)
    if 'risk_width_pct' in pattern_row and pd.notna(pattern_row.get('risk_width_pct')):
        return float(pattern_row['risk_width_pct'])

    # Fallback: box_width (pattern boundary width)
    if 'box_width' in pattern_row and pd.notna(pattern_row.get('box_width')):
        return float(pattern_row['box_width'])

    # Second fallback: calculate from boundaries if available
    upper = pattern_row.get('upper_boundary')
    lower = pattern_row.get('lower_boundary')
    if upper is not None and lower is not None and upper > 0:
        calculated = (upper - lower) / upper
        if calculated > 0:
            return float(calculated)

    # Default: assume moderate volatility (5%)
    return 0.05


def check_pattern_ripeness(
    end_date: datetime,
    reference_date: datetime,
    volatility_proxy: float = 0.05,
    data_available_through_date: Optional[datetime] = None
) -> tuple:
    """
    Check if pattern has ripened based on volatility-adjusted window.

    Higher Volatility = Faster Resolution = Shorter Window.
    Formula: dynamic_days = 1 / volatility_proxy (clamped to [MIN, MAX])

    Examples:
        - 5% vol → 1/0.05 = 20 days
        - 10% vol → 1/0.10 = 10 days
        - 2% vol → 1/0.02 = 50 days

    IMPORTANT: This function now also validates that sufficient data exists
    to compute the outcome. A pattern is only ripe if:
    1. Enough calendar time has passed (reference_date >= end_date + dynamic_days)
    2. Data extends through the outcome window (if data_available_through_date provided)

    Args:
        end_date: Pattern end date
        reference_date: Current date for comparison
        volatility_proxy: Volatility measure (higher = more volatile)
        data_available_through_date: Optional date through which data is available.
                                     If provided, validates data covers outcome window.
                                     This prevents labeling patterns where we have
                                     calendar time but missing data (delisted, halted, etc.)

    Returns:
        Tuple of (is_ripe: bool, dynamic_window_days: int)

    Note:
        When data_available_through_date is None, only the time-based check is performed.
        For production use with survivorship-corrected data, always provide this parameter.
    """
    if USE_DYNAMIC_WINDOW:
        # Dynamic: Scale inversely with volatility
        dynamic_days = int(np.clip(
            1.0 / (volatility_proxy + 1e-6),
            MIN_OUTCOME_WINDOW,
            MAX_OUTCOME_WINDOW
        ))
    else:
        # Legacy: Fixed window
        dynamic_days = OUTCOME_WINDOW_DAYS

    ripe_date = end_date + timedelta(days=dynamic_days)
    is_ripe = ripe_date <= reference_date

    # NEW: Verify data extends through outcome window
    # This catches cases where calendar time has passed but data is missing
    # (e.g., stock was delisted, trading halted, or data gaps exist)
    if is_ripe and data_available_through_date is not None:
        data_available_through = pd.to_datetime(data_available_through_date)
        required_end = pd.to_datetime(end_date) + timedelta(days=dynamic_days)
        is_ripe = is_ripe and (data_available_through >= required_end)

    return is_ripe, dynamic_days


def check_early_closure_eligibility(
    end_date: datetime,
    reference_date: datetime,
    outcome_day: int,
    dynamic_window: int = None
) -> tuple:
    """
    Check if an early-closed pattern can be safely labeled.

    Args:
        end_date: Pattern end date
        reference_date: Reference date (today)
        outcome_day: Day (1-40) when outcome was determined
        dynamic_window: Volatility-adjusted window days (uses OUTCOME_WINDOW_DAYS if None)

    Returns:
        (is_eligible, labeling_method) tuple
    """
    days_elapsed = (reference_date - end_date).days
    window_days = dynamic_window if dynamic_window is not None else OUTCOME_WINDOW_DAYS

    # Case 1: Full window elapsed
    if days_elapsed >= window_days:
        return True, 'full_window'

    # Case 2: Early closure - outcome_day must have elapsed
    if outcome_day is not None and outcome_day > 0:
        if days_elapsed >= outcome_day:
            return True, 'early_closed'

    return False, None


def calculate_r_metrics(
    upper_boundary: float,
    lower_boundary: float,
    is_weekly: bool = False
) -> dict:
    """
    Calculate structural risk metrics.

    Args:
        upper_boundary: Pattern upper boundary
        lower_boundary: Pattern lower boundary (Technical Floor for daily)
        is_weekly: If True, use weekly-specific stop calculation (50% of box)

    Returns:
        Dictionary with R-metrics:
        - technical_floor: Support level (invalidation)
        - trigger_price: Breakout trigger (upper + $0.01)
        - r_dollar: Risk per share (trigger - floor)
        - target_price: 3R target (entry + 3R, calculated at labeling)
        - gap_limit: Maximum acceptable gap (trigger + 0.5R)
    """
    if upper_boundary is None or lower_boundary is None:
        return None

    if lower_boundary <= 0 or upper_boundary <= lower_boundary:
        return None

    trigger_price = upper_boundary + TRIGGER_OFFSET

    # Weekly mode: Stop at 50% of box width (not full box)
    # This makes 1R ~10% instead of ~20%, and +2R target ~20% (achievable)
    if is_weekly:
        box_width = upper_boundary - lower_boundary
        technical_floor = trigger_price - (box_width * WEEKLY_STOP_BOX_FRACTION)
    else:
        # Daily mode: Stop at lower boundary (full box)
        technical_floor = lower_boundary

    r_dollar = trigger_price - technical_floor

    if r_dollar <= 0:
        return None

    return {
        'technical_floor': technical_floor,
        'trigger_price': trigger_price,
        'r_dollar': r_dollar,
        'gap_limit': trigger_price + (GAP_LIMIT_R * r_dollar)
    }


def label_pattern_structural(
    pattern: dict,
    ticker_df: pd.DataFrame,
    reference_date: datetime = None,
    enable_early_labeling: bool = False
) -> dict:
    """
    Label a pattern using Structural Risk Labeling.

    Implements EOD execution simulation:
    1. Wait for trigger (close > upper_boundary)
    2. Entry at MAX(Trigger_Price, Open_{t+1})
    3. Check gap limit (Entry > Trigger + 0.5R = NO_FILL)
    4. Track to Target (3R) or Danger (floor breach) or Noise (timeout)

    Args:
        pattern: Pattern dictionary with boundaries
        ticker_df: Pre-loaded DataFrame with OHLCV data
        reference_date: Reference date for early labeling
        enable_early_labeling: Allow early labeling for closed trades

    Returns:
        Updated pattern dict with:
        - outcome_class: 0 (Danger), 1 (Noise), 2 (Target), or None (NO_FILL/Invalid)
        - r_multiplier_achieved: Actual R-multiple realized
        - entry_price, target_price, technical_floor, r_dollar
        - outcome_day, labeling_method
    """
    ticker = pattern['ticker']
    end_date = pd.to_datetime(pattern['end_date'])

    # Detect weekly mode from qualification_frequency
    is_weekly = pattern.get('qualification_frequency') == 'weekly'

    # Calculate R-metrics from boundaries (weekly uses 50% box for stop)
    r_metrics = calculate_r_metrics(
        upper_boundary=pattern.get('upper_boundary'),
        lower_boundary=pattern.get('lower_boundary'),
        is_weekly=is_weekly
    )

    if r_metrics is None:
        logger.debug(f"{ticker}: Invalid boundaries for R-calculation")
        return pattern

    technical_floor = r_metrics['technical_floor']
    trigger_price = r_metrics['trigger_price']
    r_dollar = r_metrics['r_dollar']
    gap_limit = r_metrics['gap_limit']

    # V22: Get outcome window
    # Weekly mode: Fixed 20-day window (patterns need weeks to resolve)
    # Daily mode: Volatility-adjusted dynamic window
    if is_weekly:
        dynamic_window = WEEKLY_OUTCOME_WINDOW_DAYS
        vol_proxy = pattern.get('volatility_proxy', 0.20)  # Use stored or default
    else:
        vol_proxy = get_volatility_proxy(pattern)
        _, dynamic_window = check_pattern_ripeness(end_date, reference_date or datetime.now(), vol_proxy)

    # Validate data (use dynamic window instead of fixed)
    if ticker_df is None or len(ticker_df) < INDICATOR_WARMUP_DAYS + dynamic_window:
        # Check Zombie Protocol
        if reference_date is not None:
            days_since_pattern = (reference_date - end_date).days
            if days_since_pattern > ZOMBIE_TIMEOUT:
                logger.warning(f"{ticker}: ZOMBIE - data timeout ({days_since_pattern}d)")
                pattern = pattern.copy()
                pattern['outcome_class'] = 0  # Danger (150 days dead money = opportunity cost)
                pattern['labeling_method'] = 'zombie_danger'
                pattern['r_multiplier_achieved'] = None
                pattern['labeled_at'] = datetime.now().isoformat()
                return pattern

        logger.debug(f"{ticker}: Insufficient data")
        return pattern

    # Find pattern end index
    try:
        if end_date not in ticker_df.index:
            idx_pos = ticker_df.index.get_indexer([end_date], method='nearest')[0]
            if idx_pos < 0 or idx_pos >= len(ticker_df):
                logger.debug(f"{ticker}: Cannot find pattern end date")
                return pattern
            pattern_end_idx = idx_pos
        else:
            pattern_end_idx = ticker_df.index.get_loc(end_date)
    except Exception as e:
        logger.debug(f"{ticker}: Index lookup failed: {e}")
        return pattern

    # Use adj_close if available (split-adjusted)
    price_col = 'adj_close' if 'adj_close' in ticker_df.columns else 'close'

    # Get outcome window data (day after pattern end) - use dynamic window
    outcome_start_idx = pattern_end_idx + 1
    outcome_end_idx = min(outcome_start_idx + dynamic_window, len(ticker_df))

    if outcome_end_idx <= outcome_start_idx:
        logger.debug(f"{ticker}: No outcome data available")
        return pattern

    outcome_data = ticker_df.iloc[outcome_start_idx:outcome_end_idx].copy()

    if len(outcome_data) == 0:
        return pattern

    # ==========================================================================
    # EXECUTION SIMULATION: Entry at Open_{t+1} with Gap Check
    # ==========================================================================

    # Day 1 (t+1) open is our potential entry
    open_t1 = outcome_data.iloc[0]['open']

    # Entry price is MAX(Trigger_Price, Open_{t+1})
    # This simulates a limit buy at Trigger_Price, filled at open if gap up
    entry_price = max(trigger_price, open_t1)

    # GAP LIMIT CHECK: If gap is too large, R:R is ruined for retail execution
    # BUT: Big gaps ARE strong momentum signals - the model should learn from them!
    # FIX (Jan 2026): Instead of excluding (NO_FILL), mark as untradeable but continue labeling.
    # The model learns "this pattern had strong momentum" even if we couldn't catch it.
    gap_too_large = entry_price > gap_limit
    gap_size_r = (entry_price - trigger_price) / r_dollar if r_dollar > 0 else 0

    if gap_too_large:
        logger.debug(f"{ticker}: GAP_UP - strong momentum signal (entry={entry_price:.2f} > limit={gap_limit:.2f})")
        # Continue to outcome evaluation - don't return early
        # The pattern will be labeled normally, but marked as untradeable

    # Recalculate R based on actual entry
    # Note: r_dollar is still based on structural risk (trigger - floor)
    # But target is based on entry price
    # Weekly mode uses +2R target (more achievable with tighter stop)
    target_r_multiple = WEEKLY_TARGET_R_MULTIPLE if is_weekly else TARGET_R_MULTIPLE
    target_price = entry_price + (target_r_multiple * r_dollar)

    # Calculate 20-day volume average BEFORE entry (no look-ahead)
    # Use data before outcome window
    vol_lookback_start = max(0, pattern_end_idx - 20)
    vol_lookback_data = ticker_df.iloc[vol_lookback_start:pattern_end_idx + 1]
    vol_20d_avg = vol_lookback_data['volume'].mean() if len(vol_lookback_data) > 0 else 0

    # ==========================================================================
    # PATH-DEPENDENT OUTCOME EVALUATION (Vectorized with Pessimistic Tie-Breaker)
    # ==========================================================================

    # Pre-compute R-multiples and volume metrics (vectorized)
    outcome_data['r_high'] = (outcome_data['high'] - entry_price) / r_dollar if r_dollar > 0 else 0
    outcome_data['r_low'] = (outcome_data['low'] - entry_price) / r_dollar if r_dollar > 0 else 0

    max_r_achieved = float(outcome_data['r_high'].max())
    min_r_achieved = float(outcome_data['r_low'].min())

    # Volume confirmation (Jan 2026 fix): Require SUSTAINED elevated volume
    # Each day must meet BOTH: 2x relative surge AND $50k absolute minimum
    # Prevents single-day manipulation spikes and dormant stock false targets
    dollar_volume = outcome_data['volume'] * outcome_data['close']
    volume_surge_mask = (
        (outcome_data['volume'] > VOLUME_MULTIPLIER_TARGET * vol_20d_avg) &
        (dollar_volume >= MIN_DOLLAR_VOLUME_PER_DAY)
    )

    # Check for 3+ consecutive days meeting both criteria
    # Use rolling sum to find consecutive runs
    consecutive_count = volume_surge_mask.rolling(window=VOLUME_SUSTAINED_DAYS, min_periods=VOLUME_SUSTAINED_DAYS).sum()
    volume_confirmed = (consecutive_count >= VOLUME_SUSTAINED_DAYS).any()

    # -------------------------------------------------------------------------
    # ANALYTICAL NOTE: Alternative Volume Confirmation Scenarios
    # -------------------------------------------------------------------------
    # Current implementation (STRICT): All 3 consecutive days must meet BOTH:
    #   - Volume > 2x 20d avg (relative surge)
    #   - Dollar volume >= $50k (absolute liquidity floor)
    #
    # Alternative scenario (RELAXED_SURGE): Could require:
    #   - All 3 days must have dollar volume >= $50k (liquidity always mandatory)
    #   - Only 2 of 3 days need volume > 2x 20d avg (surge can have 1 quiet day)
    #
    # Rationale for alternative: Some legitimate breakouts have institutional
    # accumulation patterns where Day 2 shows reduced relative volume (profit-taking)
    # but maintains high absolute dollar volume. The strict requirement may
    # filter out valid breakouts that consolidate briefly mid-move.
    #
    # To implement RELAXED_SURGE (for A/B testing), uncomment below:
    # ---------------------------------------------------------------------
    # liquidity_mask = dollar_volume >= MIN_DOLLAR_VOLUME_PER_DAY
    # surge_mask = outcome_data['volume'] > VOLUME_MULTIPLIER_TARGET * vol_20d_avg
    #
    # # All 3 days must have $50k+ liquidity
    # liquidity_consecutive = liquidity_mask.rolling(window=3, min_periods=3).sum()
    # has_liquidity_run = (liquidity_consecutive >= 3).any()
    #
    # # At least 2 of those 3 days must have 2x surge
    # # Find windows where liquidity_consecutive == 3, then check surge count
    # if has_liquidity_run:
    #     # For each valid 3-day liquidity window, count surge days
    #     combined = (liquidity_mask & surge_mask).rolling(window=3, min_periods=3).sum()
    #     volume_confirmed_relaxed = (combined >= 2).any()
    # else:
    #     volume_confirmed_relaxed = False
    # -------------------------------------------------------------------------

    # Create event masks
    # Jan 2026 fix: Use LOW price for stop check (stock can breach floor intraday but close above)
    # Previously used close price which missed 5-10% of actual stop hits
    stop_mask = outcome_data['low'] < technical_floor      # Low breaches floor = structural failure
    target_mask = outcome_data['high'] >= target_price     # High hit 3R target

    # Find first occurrence of each event (None if never occurred)
    first_stop_idx = stop_mask.idxmax() if stop_mask.any() else None
    first_target_idx = target_mask.idxmax() if target_mask.any() else None

    # Convert to day numbers (1-indexed) for outcome_day
    def idx_to_day(idx):
        if idx is None:
            return None
        return outcome_data.index.get_loc(idx) + 1

    first_stop_day = idx_to_day(first_stop_idx)
    first_target_day = idx_to_day(first_target_idx)

    # ==========================================================================
    # TEMPORAL LABELING WITH PESSIMISTIC TIE-BREAKER
    # If both stop and target triggered same day, assume stop hit first
    # (can't determine intraday sequence from EOD data)
    # ==========================================================================

    outcome_day = None
    label = None

    if first_stop_day and first_target_day and first_stop_day == first_target_day:
        # Same day conflict: pessimistic execution assumes stop hit first
        logger.debug(f"{ticker}: Pessimistic execution - both triggered day {first_stop_day}")
        outcome_day = first_stop_day
        label = 0  # Danger (pessimistic)

    elif first_target_day and (first_stop_day is None or first_target_day < first_stop_day):
        # Target hit before stop (or no stop)
        logger.debug(f"{ticker}: TARGET - {target_r_multiple}R hit on day {first_target_day} (volume_confirmed={volume_confirmed})")
        outcome_day = first_target_day
        label = 2  # Target

    elif first_stop_day:
        # Stop hit before target (or no target)
        close_at_stop = outcome_data.loc[first_stop_idx, price_col]
        logger.debug(f"{ticker}: DANGER - close {close_at_stop:.2f} < floor {technical_floor:.2f} on day {first_stop_day}")
        outcome_day = first_stop_day
        label = 0  # Danger

    else:
        # Neither stop nor target hit within window
        label = 1  # Noise
        logger.debug(f"{ticker}: NOISE - no definitive outcome in {dynamic_window} days (vol={vol_proxy:.2%})")

    # Calculate final R-multiple achieved
    # For Danger: use min_r_achieved (how much we lost)
    # For Target: use target_r_multiple (we exited at target: +2R weekly, +3R daily)
    # For Noise: use max_r_achieved (best we could have done)
    if label == 0:  # Danger
        r_multiplier_achieved = min_r_achieved
    elif label == 2:  # Target
        r_multiplier_achieved = target_r_multiple
    else:  # Noise
        r_multiplier_achieved = max_r_achieved

    # Early labeling eligibility check (use dynamic window)
    labeling_method = 'full_window'
    if enable_early_labeling and reference_date is not None:
        is_eligible, method = check_early_closure_eligibility(
            end_date=end_date,
            reference_date=reference_date,
            outcome_day=outcome_day,
            dynamic_window=dynamic_window
        )
        if not is_eligible:
            return pattern  # Not eligible yet
        labeling_method = method

    # Update pattern with results
    pattern = pattern.copy()
    pattern['outcome_class'] = label
    pattern['labeling_method'] = labeling_method
    # V22 for daily (volatility-adjusted), V23 for weekly (fixed parameters)
    pattern['labeling_version'] = 'v23_weekly' if is_weekly else 'v22_dynamic_window'
    pattern['labeled_at'] = datetime.now().isoformat()

    # Window and volatility metrics
    pattern['outcome_window_days'] = dynamic_window
    pattern['volatility_proxy'] = vol_proxy
    pattern['target_r_multiple'] = target_r_multiple  # Track which R-multiple was used

    # R-metrics
    pattern['technical_floor'] = technical_floor
    pattern['trigger_price'] = trigger_price
    pattern['entry_price'] = entry_price
    pattern['target_price'] = target_price
    pattern['r_dollar'] = r_dollar
    pattern['r_multiplier_achieved'] = r_multiplier_achieved
    pattern['max_r_achieved'] = max_r_achieved
    pattern['min_r_achieved'] = min_r_achieved
    pattern['outcome_day'] = outcome_day
    pattern['volume_confirmed'] = volume_confirmed
    pattern['vol_20d_avg'] = vol_20d_avg

    # Gap metrics
    pattern['gap_size_r'] = gap_size_r

    # UNTRADEABLE FLAG (Jan 2026): Patterns with gap > 0.5R can't be executed by retail
    # BUT they are strong momentum signals - the model should learn from them!
    # The prediction pipeline should filter these out at execution time, not here.
    pattern['untradeable'] = gap_too_large
    if gap_too_large:
        pattern['untradeable_reason'] = 'gap_exceeds_limit'

    # Pattern width for downstream filtering
    pattern['pattern_width'] = (pattern.get('upper_boundary', 0) - pattern.get('lower_boundary', 0)) / pattern.get('upper_boundary', 1) if pattern.get('upper_boundary', 0) > 0 else 0

    return pattern


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Step 0b: Structural Risk Labeling (R-Multiples)")
    logger.info("=" * 60)

    # Parse reference date
    if args.reference_date:
        reference_date = pd.to_datetime(args.reference_date)
        logger.info(f"Reference date: {reference_date.date()} (from argument)")
    else:
        reference_date = pd.Timestamp.now()
        logger.info(f"Reference date: {reference_date.date()} (today)")

    # Load candidate patterns
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    logger.info(f"Loading candidates from: {input_path}")
    df_candidates = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df_candidates):,} candidate patterns")

    # Convert end_date to datetime
    df_candidates['end_date'] = pd.to_datetime(df_candidates['end_date'])

    # Filter for patterns that need labeling
    if args.force_relabel:
        patterns_to_label = df_candidates
        logger.warning("Force relabeling ALL patterns (--force-relabel)")
    else:
        mask_unlabeled = df_candidates['outcome_class'].isna()
        patterns_to_label = df_candidates[mask_unlabeled]
        logger.info(f"Found {len(patterns_to_label):,} unlabeled patterns")

    if len(patterns_to_label) == 0:
        logger.info("No patterns to label!")
        return 0

    # Check ripeness with volatility-adjusted windows
    def check_ripe_with_vol(row):
        vol_proxy = get_volatility_proxy(row)
        is_ripe, window = check_pattern_ripeness(row['end_date'], reference_date, vol_proxy)
        return is_ripe

    ripe_mask = patterns_to_label.apply(check_ripe_with_vol, axis=1)
    ripe_patterns = patterns_to_label[ripe_mask]
    unripe_patterns = patterns_to_label[~ripe_mask]

    window_mode = "DYNAMIC (volatility-adjusted)" if USE_DYNAMIC_WINDOW else f"FIXED ({OUTCOME_WINDOW_DAYS}-day)"
    logger.info(f"Ripe patterns ({window_mode}): {len(ripe_patterns):,}")
    logger.info(f"Unripe patterns (waiting): {len(unripe_patterns):,}")

    # Early labeling mode
    if args.enable_early_labeling:
        logger.info("\nEarly Labeling: ENABLED")
        logger.info("  -> Checking unripe patterns for early closures")
        ripe_patterns = patterns_to_label
        unripe_patterns = pd.DataFrame()

    if len(ripe_patterns) == 0:
        logger.info("No patterns to label!")
        if len(unripe_patterns) > 0:
            # Find earliest ripening pattern using volatility-adjusted window
            def get_ripe_date(row):
                vol_proxy = get_volatility_proxy(row)
                _, window = check_pattern_ripeness(row['end_date'], reference_date, vol_proxy)
                return row['end_date'] + timedelta(days=window)

            ripe_dates = unripe_patterns.apply(get_ripe_date, axis=1)
            earliest_ripe = ripe_dates.min().date()
            logger.info(f"Earliest pattern will ripen on: {earliest_ripe}")
        return 0

    if args.dry_run:
        logger.info("\n[DRY RUN] Would label:")
        logger.info(f"  - Total ripe patterns: {len(ripe_patterns):,}")
        logger.info(f"  - Tickers: {ripe_patterns['ticker'].nunique()} unique")
        return 0

    # Initialize data loader
    data_loader = DataLoader()

    # Log labeling parameters
    logger.info("\n" + "=" * 40)
    logger.info("STRUCTURAL RISK LABELING PARAMETERS (V22)")
    logger.info("=" * 40)
    logger.info(f"Technical Floor: Lower Boundary (support)")
    logger.info(f"Trigger Price: Upper Boundary + ${TRIGGER_OFFSET}")
    logger.info(f"R_Dollar: Trigger - Floor (structural risk)")
    logger.info(f"Entry: MAX(Trigger, Open_{{t+1}})")
    logger.info(f"Gap Limit: Trigger + {GAP_LIMIT_R}R (marked untradeable if exceeded)")
    logger.info(f"Target: Entry + {TARGET_R_MULTIPLE}R")
    logger.info(f"Volume Req: {VOLUME_MULTIPLIER_TARGET}x 20d avg for TARGET")
    logger.info("-" * 40)
    logger.info("DYNAMIC WINDOW (V22 - Volatility-Adjusted):")
    logger.info(f"  Enabled: {USE_DYNAMIC_WINDOW}")
    if USE_DYNAMIC_WINDOW:
        logger.info(f"  Formula: days = 1 / volatility_proxy")
        logger.info(f"  Min Window: {MIN_OUTCOME_WINDOW} days (high vol)")
        logger.info(f"  Max Window: {MAX_OUTCOME_WINDOW} days (low vol)")
        logger.info(f"  Example: 5% vol → 20d, 10% vol → 10d, 2% vol → 50d")
    else:
        logger.info(f"  Fixed Window: {OUTCOME_WINDOW_DAYS} days")
    logger.info("=" * 40 + "\n")

    # Label patterns (grouped by ticker for efficiency)
    logger.info("Labeling patterns...")
    labeled_patterns = []
    failed_patterns = []
    untradeable_count = 0  # Gap patterns that are labeled but untradeable

    outcome_counts = {0: 0, 1: 0, 2: 0}
    labeling_method_counts = {'full_window': 0, 'early_closed': 0, 'zombie_protocol': 0}
    r_multiples = []
    window_days_list = []  # Track dynamic window distribution
    vol_proxy_list = []    # Track volatility proxy distribution

    unique_tickers = ripe_patterns['ticker'].unique()
    logger.info(f"Processing {len(ripe_patterns):,} patterns from {len(unique_tickers):,} tickers")

    for ticker in tqdm(unique_tickers, desc="Labeling by ticker"):
        ticker_patterns = ripe_patterns[ripe_patterns['ticker'] == ticker]

        # Calculate data range for all patterns of this ticker
        end_dates = pd.to_datetime(ticker_patterns['end_date'])
        earliest_end = end_dates.min()
        latest_end = end_dates.max()

        data_start = earliest_end - timedelta(days=INDICATOR_WARMUP_DAYS + INDICATOR_STABLE_DAYS + 50)
        data_end = latest_end + timedelta(days=OUTCOME_WINDOW_DAYS + 10)

        # Load ticker data once
        try:
            ticker_df = data_loader.load_ticker(
                ticker,
                start_date=data_start,
                end_date=data_end,
                validate=False
            )

            if ticker_df is not None and not isinstance(ticker_df.index, pd.DatetimeIndex):
                if 'date' in ticker_df.columns:
                    ticker_df['date'] = pd.to_datetime(ticker_df['date'])
                    ticker_df = ticker_df.set_index('date')

        except Exception as e:
            logger.warning(f"{ticker}: Data load failed: {e}")
            ticker_df = None

        # Label all patterns for this ticker
        for idx, row in ticker_patterns.iterrows():
            pattern = row.to_dict()
            labeled = label_pattern_structural(
                pattern=pattern,
                ticker_df=ticker_df,
                reference_date=reference_date,
                enable_early_labeling=args.enable_early_labeling
            )

            outcome_class = labeled.get('outcome_class')
            method = labeled.get('labeling_method', 'unknown')

            if outcome_class is None:
                # Invalid/failed pattern
                failed_patterns.append(labeled)
            else:
                labeled_patterns.append(labeled)
                outcome_counts[outcome_class] += 1
                if method in labeling_method_counts:
                    labeling_method_counts[method] += 1

                # Track untradeable (gap) patterns
                if labeled.get('untradeable', False):
                    untradeable_count += 1

                # Track R-multiples
                r_mult = labeled.get('r_multiplier_achieved')
                if r_mult is not None:
                    r_multiples.append(r_mult)

                # Track dynamic window metrics
                window_days = labeled.get('outcome_window_days')
                if window_days is not None:
                    window_days_list.append(window_days)
                vol_proxy = labeled.get('volatility_proxy')
                if vol_proxy is not None:
                    vol_proxy_list.append(vol_proxy)

    # Summary
    logger.info(f"\nLabeling complete:")
    logger.info(f"  - Successfully labeled: {len(labeled_patterns):,}")
    logger.info(f"  - Untradeable (gap > {GAP_LIMIT_R}R): {untradeable_count:,} (included in training)")
    logger.info(f"  - Failed/Invalid: {len(failed_patterns):,}")

    if labeled_patterns:
        logger.info("\nOutcome Class Distribution:")
        class_names = {0: 'Danger', 1: 'Noise', 2: 'Target'}
        for k, count in sorted(outcome_counts.items()):
            pct = 100 * count / len(labeled_patterns) if labeled_patterns else 0
            logger.info(f"  Class {k} ({class_names[k]}): {count:,} ({pct:.1f}%)")

        logger.info(f"\nLabeling Method Breakdown:")
        for method, count in labeling_method_counts.items():
            if count > 0:
                logger.info(f"  - {method}: {count:,}")

        if r_multiples:
            logger.info(f"\nR-Multiple Statistics:")
            logger.info(f"  - Mean R: {np.mean(r_multiples):.2f}")
            logger.info(f"  - Median R: {np.median(r_multiples):.2f}")
            logger.info(f"  - Min R: {np.min(r_multiples):.2f}")
            logger.info(f"  - Max R: {np.max(r_multiples):.2f}")

        if USE_DYNAMIC_WINDOW and window_days_list:
            logger.info(f"\nDynamic Window Statistics (V22):")
            logger.info(f"  - Mean Window: {np.mean(window_days_list):.1f} days")
            logger.info(f"  - Median Window: {np.median(window_days_list):.1f} days")
            logger.info(f"  - Min Window: {np.min(window_days_list)} days")
            logger.info(f"  - Max Window: {np.max(window_days_list)} days")
            if vol_proxy_list:
                logger.info(f"  - Mean Volatility: {np.mean(vol_proxy_list):.2%}")
                logger.info(f"  - Median Volatility: {np.median(vol_proxy_list):.2%}")

    # Prepare output
    df_labeled = pd.DataFrame(labeled_patterns) if labeled_patterns else pd.DataFrame()

    # Combine with already-labeled patterns
    if not args.force_relabel:
        already_labeled = df_candidates[~df_candidates['outcome_class'].isna()]
        if len(already_labeled) > 0 and len(df_labeled) > 0:
            df_labeled = pd.concat([already_labeled, df_labeled], ignore_index=True)
        elif len(already_labeled) > 0:
            df_labeled = already_labeled

    # Save outputs
    output_dir = Path(args.output).parent if args.output else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save labeled patterns
    if args.output:
        labeled_path = Path(args.output)
    else:
        labeled_path = output_dir / 'labeled_patterns.parquet'

    if len(df_labeled) > 0:
        df_labeled.to_parquet(labeled_path, index=False)
        logger.info(f"\nSaved {len(df_labeled):,} labeled patterns to {labeled_path}")

        # =====================================================================
        # GAP PULLBACK WATCHLIST (Jan 2026)
        # =====================================================================
        # Export gap patterns (gap > 0.5R) that eventually hit Target.
        # These represent strong momentum signals that retail couldn't catch,
        # but may pull back to offer a second chance entry.
        #
        # Use case: Monitor for pullbacks to prior consolidation range.
        # =====================================================================
        gap_target_mask = (
            (df_labeled.get('untradeable', pd.Series([False] * len(df_labeled))).fillna(False)) &
            (df_labeled.get('outcome_class', pd.Series([-1] * len(df_labeled))) == 2)  # Target
        )
        gap_targets = df_labeled[gap_target_mask]

        if len(gap_targets) > 0:
            watchlist_cols = [
                'ticker', 'end_date', 'trigger_price', 'entry_price',
                'gap_size_r', 'target_price', 'r_multiplier_achieved',
                'upper_boundary', 'lower_boundary', 'technical_floor'
            ]
            # Only include columns that exist
            watchlist_cols = [c for c in watchlist_cols if c in gap_targets.columns]
            watchlist_path = output_dir / 'gap_pullback_watchlist.csv'
            gap_targets[watchlist_cols].to_csv(watchlist_path, index=False)
            logger.info(f"Exported {len(gap_targets):,} gap pullback candidates to {watchlist_path}")
            logger.info(f"  -> These patterns gapped >0.5R but hit Target - watch for pullback entries")

    # Note: Gap patterns (formerly NO_FILL) are now included in labeled_patterns with untradeable=True

    # Save unlabeled patterns
    unlabeled_path = output_dir / 'unlabeled_patterns.parquet'
    df_unlabeled = pd.concat([
        unripe_patterns,
        pd.DataFrame(failed_patterns) if failed_patterns else pd.DataFrame()
    ], ignore_index=True)

    if len(df_unlabeled) > 0:
        df_unlabeled.to_parquet(unlabeled_path, index=False)
        logger.info(f"Saved {len(df_unlabeled):,} unlabeled patterns to {unlabeled_path}")

    # Save labeling summary
    summary_path = output_dir / f"labeling_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_path, 'w') as f:
        f.write("Structural Risk Labeling Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Reference Date: {reference_date.date()}\n")
        f.write(f"Input: {input_path}\n\n")

        f.write("STRUCTURAL RISK FRAMEWORK\n")
        f.write("-" * 30 + "\n")
        f.write(f"Technical Floor: Lower Boundary\n")
        f.write(f"Trigger Price: Upper Boundary + ${TRIGGER_OFFSET}\n")
        f.write(f"R_Dollar: Trigger - Floor\n")
        f.write(f"Entry: MAX(Trigger, Open_{{t+1}})\n")
        f.write(f"Gap Limit: {GAP_LIMIT_R}R above trigger\n")
        f.write(f"Target: {TARGET_R_MULTIPLE}R above entry\n")
        f.write(f"Volume Req: {VOLUME_MULTIPLIER_TARGET}x 20d avg\n\n")

        f.write("DYNAMIC WINDOW (V22 - Volatility-Adjusted)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Enabled: {USE_DYNAMIC_WINDOW}\n")
        if USE_DYNAMIC_WINDOW:
            f.write(f"Formula: days = 1 / volatility_proxy\n")
            f.write(f"Min Window: {MIN_OUTCOME_WINDOW} days\n")
            f.write(f"Max Window: {MAX_OUTCOME_WINDOW} days\n")
            if window_days_list:
                f.write(f"Actual Mean: {np.mean(window_days_list):.1f} days\n")
                f.write(f"Actual Median: {np.median(window_days_list):.1f} days\n")
        else:
            f.write(f"Fixed Window: {OUTCOME_WINDOW_DAYS} days\n")
        f.write("\n")

        f.write("RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Candidates loaded: {len(df_candidates):,}\n")
        f.write(f"Successfully labeled: {len(labeled_patterns):,}\n")
        f.write(f"Untradeable (gap > {GAP_LIMIT_R}R): {untradeable_count:,} (included in training)\n")
        f.write(f"Failed/Invalid: {len(failed_patterns):,}\n\n")

        f.write("Outcome Distribution:\n")
        for k, count in sorted(outcome_counts.items()):
            class_names = {0: 'Danger', 1: 'Noise', 2: 'Target'}
            f.write(f"  Class {k} ({class_names[k]}): {count:,}\n")

        if r_multiples:
            f.write(f"\nR-Multiple Statistics:\n")
            f.write(f"  Mean: {np.mean(r_multiples):.2f}R\n")
            f.write(f"  Median: {np.median(r_multiples):.2f}R\n")
            f.write(f"  Min: {np.min(r_multiples):.2f}R\n")
            f.write(f"  Max: {np.max(r_multiples):.2f}R\n")

        f.write(f"\nOutput: {labeled_path}\n")

    logger.info(f"Saved summary to {summary_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Structural Risk Labeling Complete")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
