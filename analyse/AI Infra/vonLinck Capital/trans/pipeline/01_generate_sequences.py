"""
Step 1: Generate Temporal Sequences from Detected Patterns
===========================================================

Reads patterns detected by AIv7 pattern detector and generates
temporal sequences for training.

Input:
    - Detected patterns (from AIv7 step 1)
    - OHLCV data with indicators

Output:
    - Temporal sequences: (n_windows, 20, 10) array
    - Labels: (n_windows,) array with K0-K5 classes
    - Metadata: Pattern information for tracking

Runtime: ~10-15 minutes for full dataset
Memory: ~500MB peak (streaming mode)

Usage:
    python 01_generate_sequences.py --input ../output/detected_patterns.parquet
    python 01_generate_sequences.py --limit 100  # Test with 100 patterns
    python 01_generate_sequences.py --parallel-streaming --streaming-workers 8
"""

import sys
import os
from pathlib import Path

# Load environment variables from .env file FIRST - override=True forces .env to take precedence
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path, override=True)

import argparse
import logging
import platform
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import tempfile
import shutil
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
import json
import hashlib

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pattern_detector import TemporalPatternDetector
from core.aiv7_components import DataLoader
from utils.logging_config import setup_pipeline_logging
from utils.market_phase_helper import get_market_phase
from config import (
    TEMPORAL_WINDOW_SIZE,
    MIN_PATTERN_DURATION,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEQUENCE_DIR,
    DEFAULT_LOG_DIR,
    METADATA_ENRICHMENT_FIELDS,
)
from config.context_features import (
    CONTEXT_FEATURES,
    CONTEXT_FEATURE_DEFAULTS,
    normalize_context_feature,
    log_diff,  # Jan 2026: Safe volume ratio calculation for dormant stocks
)

from utils.regime_features import create_regime_calculator  # Jan 2026: Regime-aware features
from utils.data_integrity import (
    DataIntegrityChecker,
    DataIntegrityError,
    assert_no_duplicates
)

# Setup centralized logging
logger = setup_pipeline_logging('01_generate_sequences')


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class InsufficientDataError(Exception):
    """Raised when ticker data is insufficient for feature extraction."""
    pass


class DataSchemaError(Exception):
    """Raised when required columns are missing from ticker data."""
    pass


# =============================================================================
# METADATA HYGIENE: FUTURE_ PREFIX CONVENTION
# =============================================================================
# Features that use information AFTER the pattern end_date MUST be prefixed
# with "FUTURE_" to clearly mark them as post-pattern (lookahead) data.
#
# Examples of FUTURE_ features:
#   - FUTURE_breakout_detected: Did breakout occur within 100 days?
#   - FUTURE_days_to_armed: How many days until stop/target hit?
#   - FUTURE_ignition_type: What type of ignition triggered the breakout?
#   - FUTURE_max_gain: Maximum gain achieved post-pattern
#   - FUTURE_stop_hit_day: Day number when stop loss was hit
#
# These features ARE ALLOWED in metadata for:
#   - Post-hoc analysis (understanding what happened)
#   - Calculating EV (Expected Value) on test sets
#   - Computing Lift metrics and Profit Factor
#   - Creating the "Answer Key" for backtesting
#   - Labeling (outcome_class is derived from future info, but is the TARGET)
#
# They MUST NOT be used as:
#   - Model input features (sequences tensor)
#   - Context features (context tensor)
#   - Any feature fed to the neural network
#
# The StreamingSequenceWriter will LOG A WARNING (not error) if FUTURE_
# columns are detected, as a reminder that they are for analysis only.
# =============================================================================

FUTURE_PREFIX = 'FUTURE_'


def mark_as_future_feature(column_name: str) -> str:
    """
    Mark a column as containing future (post-pattern) information.

    Use this when creating features that use information after pattern end_date.
    These features will be blocked from entering the training pipeline.

    Args:
        column_name: Original column name (e.g., 'breakout_detected')

    Returns:
        Prefixed column name (e.g., 'FUTURE_breakout_detected')
    """
    if column_name.startswith(FUTURE_PREFIX):
        return column_name
    return f"{FUTURE_PREFIX}{column_name}"


def is_future_feature(column_name: str) -> bool:
    """Check if a column is marked as a future (lookahead) feature."""
    return column_name.startswith(FUTURE_PREFIX)


# =============================================================================
# EVENT-ANCHOR NMS HELPER FUNCTION
# =============================================================================

def select_event_anchor(
    ohlcv_df: pd.DataFrame,
    upper_boundary: float,
    lower_boundary: float,
    cutoff_idx: int = None
) -> tuple:
    """
    SOTA Selection Logic: Anchors the sequence on the most significant
    Microstructure Event (Spring or Ignition) rather than just tightness.

    TEMPORAL INTEGRITY (Jan 2026):
    - Only considers events UP TO cutoff_idx (typically pattern.end_date)
    - Prevents lookahead bias by ignoring future microstructure events
    - Patterns are selected based on information available at detection time

    Args:
        ohlcv_df: DataFrame with columns ['high', 'low', 'close', 'volume', 'volume_ratio_20']
                  Index should be datetime
        upper_boundary: Upper boundary of the pattern (constant for cluster)
        lower_boundary: Lower boundary of the pattern (constant for cluster)
        cutoff_idx: Maximum index to search (exclusive). Events after this are ignored.
                    If None, searches entire ohlcv_df (legacy behavior, NOT recommended).

    Returns:
        (anchor_idx, anchor_type) - Index into ohlcv_df and type string
    """
    high = ohlcv_df['high'].values
    low = ohlcv_df['low'].values
    close = ohlcv_df['close'].values

    # Use volume_ratio_20 if available, otherwise calculate from raw volume
    if 'volume_ratio_20' in ohlcv_df.columns:
        vol_ratio = ohlcv_df['volume_ratio_20'].values
    else:
        # Calculate rolling 20-day average ratio
        volume = ohlcv_df['volume'].values
        vol_ma = pd.Series(volume).rolling(20, min_periods=5).mean().values
        vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)

    n_days = len(ohlcv_df)

    # TEMPORAL INTEGRITY: Limit search to cutoff_idx (prevents lookahead bias)
    if cutoff_idx is None:
        search_end = n_days  # Legacy behavior (full search)
    else:
        search_end = min(cutoff_idx + 1, n_days)  # +1 because range is exclusive

    # --- PRIORITY 1: THE SPRING (Undercut & Rally) ---
    spring_indices = []
    for t in range(5, search_end):
        recent_min = np.min(low[t-5:t])
        if (low[t] < recent_min) and (close[t] > recent_min):
            spring_indices.append(t)

    if spring_indices:
        return spring_indices[-1], "ANCHOR_SPRING"

    # --- PRIORITY 2: THE IGNITION WICK (The Trap) ---
    ignition_indices = []
    for t in range(5, search_end):
        is_push = high[t] >= (upper_boundary * 0.98)
        is_high_vol = vol_ratio[t] > 1.8

        if is_push and is_high_vol:
            ignition_indices.append(t)

    if ignition_indices:
        return ignition_indices[-1], "ANCHOR_IGNITION"

    # --- PRIORITY 3: FALLBACK (Structure) ---
    fallback_idx = min(search_end // 2, n_days - 1)
    return max(0, fallback_idx), "ANCHOR_COMPRESSION"


# =============================================================================
# PER-TICKER FILTER FUNCTIONS (True Streaming)
# =============================================================================
# These functions operate on a SINGLE ticker's patterns and data.
# They are designed for use within process_ticker_pipeline() where
# each worker processes one ticker independently.
# =============================================================================


def apply_nms_filter(
    ticker_patterns: pd.DataFrame,
    overlap_days: int = 10,
    selection_col: str = 'box_width',
    nms_mode: str = 'highlander'
) -> pd.DataFrame:
    """
    NMS filter for a single ticker's patterns (no global state).

    Args:
        ticker_patterns: DataFrame with patterns for ONE ticker
        overlap_days: Days overlap threshold for clustering
        selection_col: Column for selection (lower = better)
        nms_mode: 'highlander' (1 per cluster) or 'trinity' (3 per cluster)

    Returns:
        Filtered DataFrame with de-duplicated patterns
    """
    if len(ticker_patterns) <= 1:
        if nms_mode == 'trinity' and len(ticker_patterns) == 1:
            result = ticker_patterns.copy()
            result['nms_cluster_id'] = 0
            return result
        return ticker_patterns.copy()

    df = ticker_patterns.copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    df = df.sort_values('start_date')

    # Calculate gap and assign clusters
    df['gap'] = df['start_date'].diff().dt.days.fillna(overlap_days + 1)
    df['cluster'] = (df['gap'] > overlap_days).cumsum()

    kept_indices = []
    cluster_assignments = []

    for cluster_id, cluster_group in df.groupby('cluster'):
        if nms_mode == 'trinity' and len(cluster_group) >= 3:
            # TRINITY: Take Entry, Coil, Trigger
            sorted_cluster = cluster_group.sort_values('start_date')

            entry_idx = sorted_cluster.index[0]
            if selection_col in sorted_cluster.columns:
                coil_idx = sorted_cluster[selection_col].idxmin()
            else:
                coil_idx = sorted_cluster.index[-1]
            trigger_idx = sorted_cluster.index[len(sorted_cluster) // 2]

            selected = {entry_idx, coil_idx, trigger_idx}
            for idx in selected:
                kept_indices.append(idx)
                cluster_assignments.append((idx, cluster_id))
        else:
            # HIGHLANDER: Take best (tightest)
            if selection_col in cluster_group.columns:
                best_idx = cluster_group[selection_col].idxmin()
            else:
                best_idx = cluster_group.index[0]
            kept_indices.append(best_idx)
            cluster_assignments.append((best_idx, cluster_id))

    result = df.loc[kept_indices].drop(columns=['gap', 'cluster'], errors='ignore')

    # Add cluster_id for temporal split integrity
    # CRITICAL FIX (Jan 2026): Set nms_cluster_id for ALL modes, not just trinity
    # Without this, each pattern gets a unique hash-based ID causing:
    # - Ratio of 1.0 (no clustering)
    # - Data leakage between train/val/test splits
    # - Massive overfitting to duplicate patterns
    cluster_map = {idx: cid for idx, cid in cluster_assignments}
    result['nms_cluster_id'] = result.index.map(cluster_map)

    return result


def apply_physics_filter(
    ticker_patterns: pd.DataFrame,
    ticker_df: pd.DataFrame = None,
    allowed_market_caps: list = None,
    min_width_pct: float = 0.02,
    max_width_pct: float = 0.40,  # STRUCTURAL EFFICIENCY: Reject loose structures
    min_dollar_volume: float = 50000,
    max_consecutive_zero_days: int = 3,
    min_trend_extension: float = 0.6,
    max_trend_extension: float = 1.3,
    min_float_turnover: float = 0.05,  # 5% of float traded in 20 days
    mode: str = 'training'  # 'training' or 'inference'
) -> pd.DataFrame:
    """
    Physics filter for a single ticker's patterns (no global state).

    ==========================================================================
    LOBSTER TRAP PROTECTION (Retail Account Safety)
    ==========================================================================
    A "Lobster Trap" pattern occurs when:
    - Entry is easy: Low price, seemingly liquid, fills without issue
    - Exit is IMPOSSIBLE: When you need to sell, there are no buyers

    This happens in illiquid micro-caps where:
    - Market makers widen spreads during volatility
    - Order book depth evaporates on bad news
    - Your "paper profit" becomes a realized loss because you can't exit

    SOLUTION: Filter by DOLLAR VOLUME, not share volume.
    - 10,000 shares of a $0.50 stock = $5,000/day liquidity (TRAP!)
    - 500 shares of a $100 stock = $50,000/day liquidity (tradeable)

    The min_dollar_volume threshold (default: $25,000) ensures:
    - You can enter AND exit a $5,000 position within 1-2 days
    - Slippage stays reasonable (< 1% for small retail orders)
    - You're not the only buyer/seller in the market
    ==========================================================================

    Includes "Zombie Health Check" for free data sources:
    - Data Gaps: Drop patterns with >3 consecutive zero-volume days (data error)
      NOTE: DISABLED in training mode to preserve dormant lottery tickets!
    - Ghost Trades: Drop patterns where volume=0 but high!=low (impossible)
      NOTE: DISABLED in training mode to preserve dormant lottery tickets!
    - Minimum Dollar Volume: Enforce avg_dollar_vol_20d > min_dollar_volume (LOBSTER TRAP protection)

    CRITICAL (Jan 2026 - "Lottery Ticket" Patch):
    In TRAINING mode, ghost_trade and data_gap filters are DISABLED. The model
    needs to see dormant micro-caps (often weeks of zero volume before waking up)
    to learn the difference between "Good Dormant" (Coil → 500% explosion) and
    "Bad Dormant" (Delisting → Danger). Without this, the model is trained in a
    "Clean Room" of active stocks and never learns to distinguish dormancy types.

    In INFERENCE mode, these filters are ENABLED to avoid taking positions in
    patterns with data quality issues that could indicate stale/delisted tickers.

    Includes "Sideways Regime" filter:
    - Calculates trend_extension = close / SMA_200
    - Rejects trend_extension > 1.3 (Momentum stocks, not sleepers)
    - Rejects trend_extension < 0.6 (Crashing/falling knife)
    - Keeps "Sweet Spot" (0.6 to 1.3) where accumulation happens

    Includes "Structural Efficiency" filter (Jan 2026):
    - Calculates risk_width_pct = (upper - lower) / upper
    - Rejects risk_width_pct > 0.40 (40% structural stop = untradeable)
    - Even with risk-based sizing, a 40% stop implies broken/untradeable chart

    Args:
        ticker_patterns: DataFrame with patterns for ONE ticker
        ticker_df: Optional DataFrame with OHLCV data for Zombie Health Check
        allowed_market_caps: List of allowed market cap categories
        min_width_pct: Minimum pattern width percentage
        max_width_pct: Maximum pattern width percentage (default: 0.40)
                       STRUCTURAL EFFICIENCY filter - rejects "loose" patterns
                       where the structural stop is too far away (>40%)
        min_dollar_volume: Minimum average daily dollar volume (default: 25000)
                          This is the LOBSTER TRAP filter - protects against
                          easy-entry-impossible-exit scenarios in illiquid stocks
        max_consecutive_zero_days: Max consecutive zero-volume days allowed (default: 3)
        min_trend_extension: Minimum close/SMA_200 ratio (default: 0.6, rejects crashing)
        max_trend_extension: Maximum close/SMA_200 ratio (default: 1.3, rejects momentum)
        min_float_turnover: Minimum float turnover in 20 days (default: 0.05 = 5%)
                           Filters patterns with insufficient accumulation activity.
                           Low float turnover indicates no institutional interest.
        mode: 'training' or 'inference'. In training mode, ghost_trade and data_gap
              filters are DISABLED to preserve dormant "lottery ticket" patterns.
              In inference mode, all filters are applied.

    Returns:
        Filtered DataFrame with only tradeable patterns
    """
    if len(ticker_patterns) == 0:
        return ticker_patterns

    df = ticker_patterns.copy()

    # Market cap filter
    if allowed_market_caps is None:
        allowed_market_caps = ['Nano', 'Micro', 'Small']

    if 'market_cap_category' in df.columns:
        allowed_set = set()
        for cap in allowed_market_caps:
            allowed_set.add(cap)
            allowed_set.add(cap.lower())
            allowed_set.add(f"{cap.lower()}_cap")
        allowed_set.add('unknown')

        df = df[df['market_cap_category'].isin(allowed_set)]

    # Width filter (check both possible column names)
    width_col = None
    if 'width_pct' in df.columns:
        width_col = 'width_pct'
    elif 'pattern_width' in df.columns:
        width_col = 'pattern_width'

    if width_col:
        before = len(df)
        df = df[df[width_col] >= min_width_pct]
        if before > len(df):
            logger.debug(f"  Width filter ({width_col} >= {min_width_pct}): {before} -> {len(df)}")

    # ==========================================================================
    # STRUCTURAL EFFICIENCY FILTER (Jan 2026)
    # ==========================================================================
    # Rejects "loose" patterns where the structural stop is too far away.
    # A 40%+ risk_width means the stop loss is 40% below the trigger price.
    # Even with risk-based sizing, this implies a broken/untradeable chart.
    #
    # Calculation: risk_width_pct = (upper_boundary - lower_boundary) / upper_boundary
    # This is the percentage of price that is "at risk" when entering.
    # ==========================================================================
    if 'upper_boundary' in df.columns and 'lower_boundary' in df.columns:
        # Calculate risk_width_pct for each pattern
        upper = df['upper_boundary'].values
        lower = df['lower_boundary'].values
        # Avoid division by zero
        risk_width_pct = np.where(upper > 0, (upper - lower) / upper, 0.0)
        df = df.copy()
        df['risk_width_pct'] = risk_width_pct

        before = len(df)
        loose_mask = df['risk_width_pct'] <= max_width_pct
        loose_count = (~loose_mask).sum()
        df = df[loose_mask]
        if loose_count > 0:
            logger.info(
                f"  [STRUCTURAL EFFICIENCY] Filtered {loose_count} loose patterns "
                f"(risk_width > {max_width_pct*100:.0f}%) - untradeable R:R structure"
            )

    # Dollar volume filter (check both possible column names)
    volume_col = None
    if 'avg_dollar_volume' in df.columns:
        volume_col = 'avg_dollar_volume'
    elif 'liquidity' in df.columns:
        volume_col = 'liquidity'

    if volume_col:
        before = len(df)
        df = df[df[volume_col] >= min_dollar_volume]
        if before > len(df):
            logger.debug(f"  Volume filter ({volume_col} >= ${min_dollar_volume:,.0f}): {before} -> {len(df)}")

    # ==========================================================================
    # FLOAT TURNOVER FILTER (Jan 2026)
    # Ensures minimum accumulation activity (% of float traded in 20 days)
    # Low float turnover = no institutional interest, likely to stay dormant
    # ==========================================================================
    if 'float_turnover' in df.columns and min_float_turnover > 0:
        before = len(df)
        # Preserve NaN values (unknown float_turnover) - don't filter when data unavailable
        df = df[(df['float_turnover'] >= min_float_turnover) | (df['float_turnover'].isna())]
        filtered_count = before - len(df)
        if filtered_count > 0:
            logger.info(
                f"  [FLOAT TURNOVER] Filtered {filtered_count} low-activity patterns "
                f"(float_turnover < {min_float_turnover*100:.0f}%) - insufficient accumulation"
            )

    # ==========================================================================
    # ZOMBIE HEALTH CHECK (requires ticker_df)
    # Filters for data quality issues common in free data sources
    # ==========================================================================
    if ticker_df is not None and len(df) > 0:
        zombie_mask = pd.Series(True, index=df.index)  # Start with all patterns valid

        # Counters for detailed logging (helps understand WHY patterns are filtered)
        ghost_trade_count = 0
        data_gap_count = 0
        lobster_trap_count = 0  # CRITICAL: Tracks illiquid patterns that would trap retail traders
        sideways_momentum_count = 0
        sideways_crashing_count = 0

        for idx, pattern_row in df.iterrows():
            try:
                # Get pattern date range
                start_date = pd.to_datetime(pattern_row['start_date'])
                end_date = pd.to_datetime(pattern_row['end_date'])

                # Filter ticker_df to pattern period
                if isinstance(ticker_df.index, pd.DatetimeIndex):
                    pattern_data = ticker_df.loc[start_date:end_date]
                else:
                    mask = (ticker_df['date'] >= start_date) & (ticker_df['date'] <= end_date)
                    pattern_data = ticker_df[mask]

                if len(pattern_data) < 5:
                    # Too few data points - mark as zombie
                    zombie_mask[idx] = False
                    continue

                volume = pattern_data['volume'].values
                high = pattern_data['high'].values
                low = pattern_data['low'].values

                # --- Check 1: Ghost Trades ---
                # Volume == 0 but price moved (high != low) = impossible data error
                # DISABLED in training mode - preserve dormant lottery tickets!
                if mode == 'inference':
                    ghost_trades = (volume == 0) & (high != low)
                    if ghost_trades.any():
                        zombie_mask[idx] = False
                        ghost_trade_count += 1
                        logger.debug(f"  Zombie: Ghost trades detected (vol=0, price moved) - pattern {idx}")
                        continue

                # --- Check 2: Data Gaps ---
                # More than N consecutive days of zero volume = data error, not supply exhaustion
                # DISABLED in training mode - true dormant micro-caps have weeks of zero volume!
                if mode == 'inference' and max_consecutive_zero_days > 0:
                    consecutive_zeros = 0
                    max_consecutive = 0
                    for v in volume:
                        if v == 0:
                            consecutive_zeros += 1
                            max_consecutive = max(max_consecutive, consecutive_zeros)
                        else:
                            consecutive_zeros = 0

                    if max_consecutive > max_consecutive_zero_days:
                        zombie_mask[idx] = False
                        data_gap_count += 1
                        logger.debug(f"  Zombie: Data gap ({max_consecutive} consecutive zero-vol days) - pattern {idx}")
                        continue

                # --- Check 3: LOBSTER TRAP Protection (Dollar Volume) ---
                # =============================================================
                # WHY DOLLAR VOLUME, NOT SHARE VOLUME:
                # A $0.50 stock with 50,000 shares/day = $25,000 dollar volume
                # A $50 stock with 500 shares/day = $25,000 dollar volume
                # Both have SAME liquidity for a retail trader!
                #
                # Share volume alone is meaningless without price context.
                # A "high volume" penny stock can still be a Lobster Trap.
                #
                # This filter ensures you can EXIT your position, not just enter.
                # =============================================================
                close_prices = pattern_data['close'].values
                avg_dollar_vol_20d = np.mean(volume[-20:] * close_prices[-20:]) if len(volume) >= 20 else np.mean(volume * close_prices)

                if avg_dollar_vol_20d < min_dollar_volume:
                    zombie_mask[idx] = False
                    lobster_trap_count += 1
                    logger.debug(
                        f"  LOBSTER TRAP: Insufficient dollar volume "
                        f"(avg_dollar_vol_20d=${avg_dollar_vol_20d:,.0f} < ${min_dollar_volume:,.0f}) - pattern {idx}"
                    )
                    continue

                # --- Check 4: Sideways Regime ---
                # Reject momentum (>1.3x SMA_200) and crashing (<0.6x SMA_200) stocks
                # Keep "Sweet Spot" where accumulation happens
                if '_sma_200' in ticker_df.columns:
                    # Get SMA_200 at pattern end
                    if isinstance(ticker_df.index, pd.DatetimeIndex):
                        sma_200 = ticker_df.loc[:end_date, '_sma_200'].iloc[-1] if end_date in ticker_df.index or len(ticker_df.loc[:end_date]) > 0 else None
                        close_at_end = pattern_data['close'].iloc[-1] if len(pattern_data) > 0 else None
                    else:
                        end_mask = ticker_df['date'] <= end_date
                        sma_200 = ticker_df.loc[end_mask, '_sma_200'].iloc[-1] if end_mask.any() else None
                        close_at_end = pattern_data['close'].iloc[-1] if len(pattern_data) > 0 else None

                    if sma_200 is not None and close_at_end is not None and sma_200 > 0:
                        trend_extension = close_at_end / sma_200

                        if trend_extension > max_trend_extension:
                            zombie_mask[idx] = False
                            sideways_momentum_count += 1
                            logger.debug(f"  Sideways: Momentum stock (trend_ext={trend_extension:.2f} > {max_trend_extension}) - pattern {idx}")
                            continue

                        if trend_extension < min_trend_extension:
                            zombie_mask[idx] = False
                            sideways_crashing_count += 1
                            logger.debug(f"  Sideways: Crashing stock (trend_ext={trend_extension:.2f} < {min_trend_extension}) - pattern {idx}")
                            continue

            except Exception as e:
                # On any error, mark as zombie to be safe
                zombie_mask[idx] = False
                logger.debug(f"  Zombie: Error checking pattern {idx}: {e}")

        # Apply zombie filter
        before = len(df)
        df = df[zombie_mask]
        total_removed = before - len(df)

        if total_removed > 0:
            logger.debug(f"  Zombie Health Check: {before} -> {len(df)} ({total_removed} patterns removed)")

            # Detailed breakdown of WHY patterns were filtered
            # This helps understand data quality and liquidity issues
            if ghost_trade_count > 0:
                logger.debug(f"    - Ghost Trades (data error): {ghost_trade_count}")
            if data_gap_count > 0:
                logger.debug(f"    - Data Gaps (missing data): {data_gap_count}")
            if lobster_trap_count > 0:
                # CRITICAL: Log this prominently as it protects retail accounts
                logger.info(
                    f"  [LOBSTER TRAP PROTECTION] Filtered {lobster_trap_count} illiquid patterns "
                    f"(avg_dollar_vol < ${min_dollar_volume:,.0f}/day) - protecting against easy-entry-impossible-exit scenarios"
                )
            if sideways_momentum_count > 0:
                logger.debug(f"    - Momentum stocks (not sleepers): {sideways_momentum_count}")
            if sideways_crashing_count > 0:
                logger.debug(f"    - Crashing stocks (falling knife): {sideways_crashing_count}")

    return df


def apply_heartbeat_filter(
    ticker_patterns: pd.DataFrame,
    ticker_df: pd.DataFrame,
    max_volume_cv: float = 0.8
) -> pd.DataFrame:
    """
    Heartbeat filter for a single ticker (no global state).

    BEHAVIOR (Jan 2026 - Alpha Preservation Fix):
    Marks patterns with high volume CV (coefficient of variation) as is_noise=True.
    This is stored as METADATA ONLY - the true label is PRESERVED.

    WHY NO LABEL OVERRIDE:
    - High volume CV often indicates institutional accumulation (block trades)
    - "Erratic" volume can signal pre-catalyst positioning ("Power Plays")
    - Overriding labels to Noise destroyed ~15% of highest-beta alpha
    - The model should learn from TRUE price outcomes, not variance assumptions

    The is_noise flag is useful for:
    - Post-training analysis (comparing outcomes of high-CV vs low-CV patterns)
    - Feature engineering (can be added as a context feature if needed)
    - NOT for corrupting the target variable

    Args:
        ticker_patterns: DataFrame with patterns for ONE ticker
        ticker_df: Pre-loaded OHLCV data for the ticker
        max_volume_cv: Maximum volume CV threshold

    Returns:
        DataFrame with is_noise column added (True for CV > threshold)
        NOTE: Labels are NOT overridden - is_noise is metadata only
    """
    if len(ticker_patterns) == 0 or ticker_df is None:
        df = ticker_patterns.copy()
        df['is_noise'] = False
        return df

    df = ticker_patterns.copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])

    volume_cvs = []

    for idx, row in df.iterrows():
        start_date = row['start_date']
        end_date = row['end_date']

        # Handle timezone
        if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
            start_date = start_date.tz_localize(None)
        if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
            end_date = end_date.tz_localize(None)

        try:
            mask = (ticker_df.index >= start_date) & (ticker_df.index <= end_date)
            period_df = ticker_df[mask]

            if len(period_df) >= 5:
                volume = period_df['volume'].values
                vol_mean = np.mean(volume)
                vol_std = np.std(volume)
                cv = vol_std / vol_mean if vol_mean > 0 else np.nan
            else:
                cv = np.nan
        except:
            cv = np.nan

        volume_cvs.append(cv)

    df['volume_cv'] = volume_cvs
    df['is_noise'] = (df['volume_cv'] > max_volume_cv) & df['volume_cv'].notna()

    return df


def calculate_coil_features(
    ticker_patterns: pd.DataFrame,
    ticker_df: pd.DataFrame,
    danger_zone_pct: float = 0.25
) -> pd.DataFrame:
    """
    Calculate coil features for a single ticker (no global state).

    Uses local ticker_df instead of global cache.

    Args:
        ticker_patterns: DataFrame with patterns for ONE ticker
        ticker_df: Pre-loaded OHLCV data for the ticker
        danger_zone_pct: Lower percentile for danger zone

    Returns:
        DataFrame with coil features added
    """
    if len(ticker_patterns) == 0 or ticker_df is None:
        return ticker_patterns

    df = ticker_patterns.copy()
    df['end_date'] = pd.to_datetime(df['end_date'])

    # Initialize columns
    df['price_position_at_end'] = np.nan
    df['volume_shock'] = np.nan
    df['bbw_at_end'] = np.nan
    df['bbw_slope_5d'] = np.nan
    df['vol_trend_5d'] = np.nan
    df['coil_intensity'] = np.nan

    # Use pre-computed BBW if available (from precompute_rolling_features)
    if '_bbw_20' in ticker_df.columns:
        ticker_df['bbw_20'] = ticker_df['_bbw_20']
    elif 'bbw_20' not in ticker_df.columns:
        # Fallback calculation
        sma20 = ticker_df['close'].rolling(window=20).mean()
        std20 = ticker_df['close'].rolling(window=20).std()
        ticker_df = ticker_df.copy()
        ticker_df['bbw_20'] = (2 * std20) / sma20

    for idx, row in df.iterrows():
        end_date = row['end_date']
        upper = row.get('upper_boundary')
        lower = row.get('lower_boundary')

        if pd.isna(upper) or pd.isna(lower) or upper <= lower:
            continue

        if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
            end_date = end_date.tz_localize(None)

        try:
            mask = ticker_df.index <= end_date
            pattern_data = ticker_df[mask]

            if len(pattern_data) < 20:
                continue

            box_height = upper - lower
            close_at_end = pattern_data['close'].iloc[-1]

            # Price position
            price_position = np.clip((close_at_end - lower) / box_height, 0, 1)
            df.at[idx, 'price_position_at_end'] = price_position

            # Volume shock (max vol in last 3 days / 20d avg) - breakout precursor
            # FIX (Jan 2026): Use log_diff to handle dormant stocks
            vol_20d_avg_pre = pattern_data['volume'].iloc[-20:].mean() if len(pattern_data) >= 20 else pattern_data['volume'].mean()
            if len(pattern_data) >= 3:
                vol_last_3d_max = pattern_data['volume'].iloc[-3:].max()
                df.at[idx, 'volume_shock'] = log_diff(vol_last_3d_max, vol_20d_avg_pre)
            else:
                df.at[idx, 'volume_shock'] = 0.0  # Neutral default for log_diff

            # BBW at end
            bbw_at_end = pattern_data['bbw_20'].iloc[-1]
            df.at[idx, 'bbw_at_end'] = bbw_at_end

            # BBW slope
            if len(pattern_data) >= 5:
                bbw_5d_ago = pattern_data['bbw_20'].iloc[-5]
                if pd.notna(bbw_5d_ago) and pd.notna(bbw_at_end):
                    df.at[idx, 'bbw_slope_5d'] = (bbw_at_end - bbw_5d_ago) / 5

            # Volume trend
            # FIX (Jan 2026): Use log_diff to handle dormant stocks
            vol_20d_avg = pattern_data['volume'].iloc[-20:].mean()
            vol_5d_avg = pattern_data['volume'].iloc[-5:].mean()
            vol_trend = log_diff(vol_5d_avg, vol_20d_avg)
            df.at[idx, 'vol_trend_5d'] = vol_trend

            # Coil intensity
            # FIX (Jan 2026): Update vol_score for log_diff output
            pos_score = 1 - price_position
            bbw_score = 1 - min(bbw_at_end / 0.2, 1) if pd.notna(bbw_at_end) else 0.5
            # Map log_diff [-2, 2] to [1, 0]: negative (contracting) = high score
            vol_score = np.clip(0.5 - vol_trend / 4, 0.0, 1.0)
            df.at[idx, 'coil_intensity'] = (pos_score + bbw_score + vol_score) / 3

        except Exception:
            continue

    return df


def apply_event_anchor_nms(
    ticker_patterns: pd.DataFrame,
    ticker_df: pd.DataFrame,
    overlap_days: int = 10
) -> pd.DataFrame:
    """
    Event-anchor NMS for a single ticker (no global state).

    Uses local ticker_df instead of global cache.

    Args:
        ticker_patterns: DataFrame with patterns for ONE ticker
        ticker_df: Pre-loaded OHLCV data for the ticker
        overlap_days: Days overlap threshold for clustering

    Returns:
        Filtered DataFrame with event-anchored patterns
    """
    if len(ticker_patterns) <= 1:
        return ticker_patterns.copy()

    if ticker_df is None:
        # Fallback to standard NMS
        return apply_nms_filter(
            ticker_patterns, overlap_days, 'box_width', 'highlander'
        )

    df = ticker_patterns.copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df = df.sort_values('start_date')

    # Calculate volume_ratio_20 if needed
    if 'volume_ratio_20' not in ticker_df.columns and 'volume' in ticker_df.columns:
        ticker_df = ticker_df.copy()
        vol_ma = ticker_df['volume'].rolling(20, min_periods=5).mean()
        ticker_df['volume_ratio_20'] = np.where(vol_ma > 0, ticker_df['volume'] / vol_ma, 1.0)

    # Calculate gap and assign clusters
    df['gap'] = df['start_date'].diff().dt.days.fillna(overlap_days + 1)
    df['cluster'] = (df['gap'] > overlap_days).cumsum()

    kept_indices = []
    anchor_types = {}

    for cluster_id, cluster_group in df.groupby('cluster'):
        if len(cluster_group) == 1:
            kept_indices.append(cluster_group.index[0])
            anchor_types[cluster_group.index[0]] = "ANCHOR_SINGLE"
            continue

        # Get cluster date range
        cluster_start = cluster_group['start_date'].min()
        cluster_end = cluster_group['end_date'].max()

        if hasattr(cluster_start, 'tzinfo') and cluster_start.tzinfo is not None:
            cluster_start = cluster_start.tz_localize(None)
        if hasattr(cluster_end, 'tzinfo') and cluster_end.tzinfo is not None:
            cluster_end = cluster_end.tz_localize(None)

        try:
            cluster_ohlcv = ticker_df.loc[cluster_start:cluster_end]

            if len(cluster_ohlcv) < 10:
                # Fallback to tightest
                if 'box_width' in cluster_group.columns:
                    best_idx = cluster_group['box_width'].idxmin()
                else:
                    best_idx = cluster_group.index[0]
                kept_indices.append(best_idx)
                anchor_types[best_idx] = "ANCHOR_FALLBACK"
                continue

            # Get boundaries from tightest pattern
            if 'box_width' in cluster_group.columns:
                tightest_idx = cluster_group['box_width'].idxmin()
            else:
                tightest_idx = cluster_group.index[0]

            upper_boundary = cluster_group.loc[tightest_idx, 'upper_boundary']
            lower_boundary = cluster_group.loc[tightest_idx, 'lower_boundary']

            # TEMPORAL INTEGRITY: Find best anchor FOR EACH PATTERN
            anchor_priority = {
                'ANCHOR_SPRING': 3,
                'ANCHOR_IGNITION': 2,
                'ANCHOR_COMPRESSION': 1
            }

            pattern_scores = []
            for pat_idx, pat_row in cluster_group.iterrows():
                pat_end_date = pat_row['end_date']

                if hasattr(pat_end_date, 'tzinfo') and pat_end_date.tzinfo is not None:
                    pat_end_date = pat_end_date.tz_localize(None)

                # Find cutoff index for this pattern's end_date
                try:
                    cutoff_positions = cluster_ohlcv.index.get_indexer([pat_end_date], method='ffill')
                    cutoff_idx = cutoff_positions[0] if cutoff_positions[0] >= 0 else 0
                except:
                    cutoff_idx = 0

                # Find anchor for THIS pattern (limited to its visible range)
                if cutoff_idx >= 5:
                    anchor_idx, anchor_type = select_event_anchor(
                        cluster_ohlcv, upper_boundary, lower_boundary, cutoff_idx=cutoff_idx
                    )
                    priority = anchor_priority.get(anchor_type, 0)
                    score = priority * 1000 + anchor_idx
                else:
                    anchor_type = 'ANCHOR_COMPRESSION'
                    score = 0

                pattern_scores.append({
                    'idx': pat_idx,
                    'anchor_type': anchor_type,
                    'score': score
                })

            # Select pattern with highest score
            if pattern_scores:
                best = max(pattern_scores, key=lambda x: x['score'])
                best_idx = best['idx']
                anchor_types[best_idx] = best['anchor_type']
            else:
                if 'box_width' in cluster_group.columns:
                    best_idx = cluster_group['box_width'].idxmin()
                else:
                    best_idx = cluster_group.index[0]
                anchor_types[best_idx] = "ANCHOR_FALLBACK"

            kept_indices.append(best_idx)

        except Exception:
            # Fallback
            if 'box_width' in cluster_group.columns:
                best_idx = cluster_group['box_width'].idxmin()
            else:
                best_idx = cluster_group.index[0]
            kept_indices.append(best_idx)
            anchor_types[best_idx] = "ANCHOR_FALLBACK"

    result = df.loc[kept_indices].drop(columns=['gap', 'cluster'], errors='ignore')
    result['anchor_type'] = result.index.map(anchor_types)

    return result


# =============================================================================
# FEATURE COMPUTATION HELPERS
# =============================================================================

def precompute_rolling_features(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute rolling features ONCE for the entire ticker dataframe.

    This vectorizes feature calculation: instead of computing rolling windows
    for each pattern, we compute them once and use index lookups.

    Pre-computed columns added:
        - _vol_60d_sum: 60-day rolling volume sum
        - _vol_20d_avg: 20-day rolling volume average
        - _vol_60d_avg: 60-day rolling volume average
        - _vol_5d_avg: 5-day rolling volume average
        - _vol_6m_median: 6-month (126-day) rolling median volume (for Window-Anchor fix)
        - _sma_200: 200-day simple moving average
        - _high_52w: 52-week rolling high
        - _dollar_vol_20d_avg: 20-day average dollar volume
        - _dollar_vol_252d_avg: 252-day average dollar volume (for PIT-safe pseudo_turnover)
        - _return_20d: 20-day return
        - _bbw_20: Bollinger Band Width (if not present)
        - volume_ratio_20: Current volume / 20-day avg (for NMS event anchor)

    Args:
        ticker_df: DataFrame with OHLCV data

    Returns:
        DataFrame with pre-computed columns added (modifies in place)
    """
    # Volume rolling features
    ticker_df['_vol_60d_sum'] = ticker_df['volume'].rolling(60, min_periods=1).sum()
    ticker_df['_vol_20d_avg'] = ticker_df['volume'].rolling(20, min_periods=1).mean()
    ticker_df['_vol_60d_avg'] = ticker_df['volume'].rolling(60, min_periods=1).mean()
    ticker_df['_vol_5d_avg'] = ticker_df['volume'].rolling(5, min_periods=1).mean()

    # Deep Dormancy features (100d and 252d averages for dormancy detection)
    # Used for: vol_dryup_ratio = vol_20d / vol_100d, dormancy_shock = log10(vol_20d / vol_252d)
    ticker_df['_vol_100d_avg'] = ticker_df['volume'].rolling(100, min_periods=1).mean()
    ticker_df['_vol_252d_avg'] = ticker_df['volume'].rolling(252, min_periods=1).mean()

    # 6-month rolling median volume (Window-Anchor Fix)
    # CRITICAL: Used for volume normalization instead of arbitrary day-0 volume.
    # The LSTM sees "Wake Up" magnitude relative to dormant history, not pattern start.
    # Replace zeros with 1 to avoid division by zero in normalization.
    vol_6m_median = ticker_df['volume'].rolling(126, min_periods=1).median()
    ticker_df['_vol_6m_median'] = vol_6m_median.replace(0, 1)

    # Price rolling features
    ticker_df['_sma_200'] = ticker_df['close'].rolling(200, min_periods=1).mean()
    ticker_df['_high_52w'] = ticker_df['high'].rolling(252, min_periods=1).max()

    # Dollar volume (price * volume)
    dollar_vol = ticker_df['close'] * ticker_df['volume']
    ticker_df['_dollar_vol_20d_avg'] = dollar_vol.rolling(20, min_periods=1).mean()
    ticker_df['_dollar_vol_252d_avg'] = dollar_vol.rolling(252, min_periods=20).mean()  # For PIT-safe pseudo_turnover

    # 20-day return for relative strength
    ticker_df['_return_20d'] = ticker_df['close'].pct_change(20)

    # BBW if not present
    if 'bbw_20' not in ticker_df.columns:
        sma_20 = ticker_df['close'].rolling(20, min_periods=1).mean()
        std_20 = ticker_df['close'].rolling(20, min_periods=1).std()
        ticker_df['_bbw_20'] = (2 * std_20) / sma_20.replace(0, np.nan)
        ticker_df['_bbw_20'] = ticker_df['_bbw_20'].fillna(0.1)
    else:
        ticker_df['_bbw_20'] = ticker_df['bbw_20']

    # [FIX] Pre-compute standard volume ratio for NMS usage
    # This is used by select_event_anchor() for ignition detection
    if '_vol_20d_avg' in ticker_df.columns:
        vol_avg_safe = ticker_df['_vol_20d_avg'].replace(0, np.nan)
        ticker_df['volume_ratio_20'] = ticker_df['volume'] / vol_avg_safe
        ticker_df['volume_ratio_20'] = ticker_df['volume_ratio_20'].fillna(1.0)

    return ticker_df


# =============================================================================
# COHORT-BASED RELATIVE STRENGTH (Replacing SPY to avoid look-ahead bias)
# =============================================================================
# PROBLEM: EU stocks close at 17:30 CET, SPY closes at 22:00 CET.
#          Using SPY data for EU stocks creates a 4.5-hour look-ahead bias.
#
# SOLUTION: Use Point-In-Time (PIT) external index as baseline to avoid survivorship bias.
#           The cohort median from pattern-forming stocks is inflated because delisted
#           companies are excluded (survivors only).
#
# SURVIVORSHIP BIAS (Jan 2026 Fix):
#   - patterns_df['ticker'].unique() contains only stocks that survived long enough
#     to form consolidation patterns
#   - Delisted/bankrupt companies are excluded from the "universe"
#   - Median_survivor > Median_true (artificially inflated)
#   - RelativeStrength = R_stock - Median_survivor is systematically LOWER
#   - Model learns a conservative bias (artificially high hurdle rate)
#
# FIX: Use external PIT index (RSP/IWM for US, broad EU index for EU) as baseline.
#      Fallback to cohort median only if index data unavailable.
#
# Formula: relative_strength_cohort = (Close_t / Close_t-20) - Index_Return_20D
#          where Index_Return_20D = 20-day return of external index (e.g., RSP)
# =============================================================================

# Global cache for baseline returns (external index or cohort median)
_baseline_returns_cache: Optional[pd.Series] = None
_baseline_source: str = "unknown"  # Track which source was used

# Default external index tickers by region
DEFAULT_BASELINE_INDEX = {
    'US': 'RSP',    # Invesco S&P 500 Equal Weight (avoids mega-cap bias)
    'EU': 'EXSA.DE',  # iShares STOXX Europe 600 (broad EU exposure)
    'FALLBACK': 'IWM'  # iShares Russell 2000 (small-cap benchmark)
}


def load_pit_index_returns(
    index_ticker: str,
    loader: DataLoader = None
) -> Optional[pd.Series]:
    """
    Load 20-day returns from an external Point-In-Time index.

    This provides a survivorship-bias-free baseline for relative strength.

    Args:
        index_ticker: Ticker symbol of the external index (e.g., 'RSP', 'IWM', 'EXSA.DE')
        loader: DataLoader instance (created if None)

    Returns:
        pd.Series indexed by date with 20-day returns, or None if unavailable
    """
    if loader is None:
        loader = DataLoader()

    try:
        index_df = loader.load_ticker(index_ticker, validate=False)
        if index_df is None or len(index_df) < 25:
            logger.warning(f"[PIT INDEX] Could not load {index_ticker} (insufficient data)")
            return None

        # Ensure date index
        if 'date' in index_df.columns:
            index_df['date'] = pd.to_datetime(index_df['date'])
            index_df = index_df.set_index('date')

        if index_df.index.tz is not None:
            index_df.index = index_df.index.tz_localize(None)

        # Compute 20-day return: (Close_t / Close_t-20) - 1
        index_returns = index_df['close'].pct_change(20).dropna()

        logger.info(f"[PIT INDEX] Loaded {index_ticker}: {len(index_returns)} dates, "
                   f"range {index_returns.index.min().date()} to {index_returns.index.max().date()}")

        return index_returns

    except Exception as e:
        logger.warning(f"[PIT INDEX] Failed to load {index_ticker}: {e}")
        return None


def compute_universe_median_returns(
    patterns_df: pd.DataFrame,
    loader: DataLoader = None,
    baseline_index: str = None,
    region: str = 'EU'
) -> pd.Series:
    """
    Compute baseline 20-day returns for relative strength calculation.

    PRIORITY ORDER (to avoid survivorship bias):
    1. External PIT index (RSP/IWM for US, EXSA for EU) - PREFERRED
    2. Cohort median from patterns_df tickers - FALLBACK (has survivorship bias)

    SURVIVORSHIP BIAS WARNING:
    Using cohort median from patterns_df['ticker'].unique() introduces conservative
    bias because delisted companies are excluded. The "universe" contains only
    survivors, making Median_survivor > Median_true.

    Args:
        patterns_df: DataFrame with 'ticker' column (used for fallback cohort)
        loader: DataLoader instance (created if None)
        baseline_index: Override index ticker (e.g., 'RSP', 'IWM', 'EXSA.DE')
        region: 'US' or 'EU' to select default index (ignored if baseline_index set)

    Returns:
        pd.Series indexed by date with 20-day baseline returns
    """
    global _baseline_returns_cache, _baseline_source

    if _baseline_returns_cache is not None:
        return _baseline_returns_cache

    if loader is None:
        loader = DataLoader()

    # Determine which index to use
    if baseline_index:
        index_ticker = baseline_index
    else:
        index_ticker = DEFAULT_BASELINE_INDEX.get(region, DEFAULT_BASELINE_INDEX['FALLBACK'])

    logger.info("=" * 80)
    logger.info(f"COMPUTING BASELINE RETURNS FOR RELATIVE STRENGTH")
    logger.info(f"Attempting to load external PIT index: {index_ticker}")
    logger.info("=" * 80)

    # ATTEMPT 1: Load external PIT index (survivorship-bias-free)
    index_returns = load_pit_index_returns(index_ticker, loader)

    if index_returns is not None and len(index_returns) > 100:
        _baseline_returns_cache = index_returns.sort_index().ffill()
        _baseline_source = f"PIT_INDEX:{index_ticker}"
        logger.info(f"[OK] Using external PIT index: {index_ticker}")
        logger.info(f"     This avoids survivorship bias from cohort median")
        logger.info(f"     Date range: {_baseline_returns_cache.index.min()} to {_baseline_returns_cache.index.max()}")
        logger.info(f"     Return stats: mean={_baseline_returns_cache.mean():.4f}, std={_baseline_returns_cache.std():.4f}")
        return _baseline_returns_cache

    # ATTEMPT 2: Try fallback index
    if index_ticker != DEFAULT_BASELINE_INDEX['FALLBACK']:
        logger.warning(f"[WARN] Primary index {index_ticker} unavailable, trying fallback: {DEFAULT_BASELINE_INDEX['FALLBACK']}")
        fallback_returns = load_pit_index_returns(DEFAULT_BASELINE_INDEX['FALLBACK'], loader)

        if fallback_returns is not None and len(fallback_returns) > 100:
            _baseline_returns_cache = fallback_returns.sort_index().ffill()
            _baseline_source = f"PIT_INDEX:{DEFAULT_BASELINE_INDEX['FALLBACK']}"
            logger.info(f"[OK] Using fallback PIT index: {DEFAULT_BASELINE_INDEX['FALLBACK']}")
            return _baseline_returns_cache

    # FALLBACK: Cohort median (has survivorship bias - warn user)
    logger.warning("=" * 80)
    logger.warning("[SURVIVORSHIP BIAS WARNING] External index unavailable!")
    logger.warning("Falling back to COHORT MEDIAN (patterns_df tickers only)")
    logger.warning("")
    logger.warning("BIAS IMPACT: Cohort contains only survivor stocks (pattern-forming).")
    logger.warning("  - Delisted/bankrupt companies are EXCLUDED")
    logger.warning("  - Median_cohort > Median_true (artificially inflated)")
    logger.warning("  - RelativeStrength will be systematically LOWER than reality")
    logger.warning("  - Model learns conservative bias (15-25% signal distortion)")
    logger.warning("")
    logger.warning("RECOMMENDED: Add external index data (RSP, IWM, or EXSA.DE)")
    logger.warning("=" * 80)

    tickers = patterns_df['ticker'].unique()
    logger.info(f"[COHORT FALLBACK] Computing median from {len(tickers)} pattern-forming tickers...")

    # Collect 20-day returns from all tickers
    all_returns_by_date: Dict[pd.Timestamp, List[float]] = {}

    for ticker in tqdm(tickers, desc="Computing cohort returns (FALLBACK)", leave=False):
        try:
            ticker_df = loader.load_ticker(ticker, validate=False)
            if ticker_df is None or len(ticker_df) < 25:
                continue

            if 'date' in ticker_df.columns:
                ticker_df['date'] = pd.to_datetime(ticker_df['date'])
                ticker_df = ticker_df.set_index('date')

            if ticker_df.index.tz is not None:
                ticker_df.index = ticker_df.index.tz_localize(None)

            ticker_df['return_20d'] = ticker_df['close'].pct_change(20)

            for date, row in ticker_df.iterrows():
                if pd.notna(row['return_20d']):
                    if date not in all_returns_by_date:
                        all_returns_by_date[date] = []
                    all_returns_by_date[date].append(row['return_20d'])

        except Exception as e:
            logger.debug(f"[COHORT] Skipping {ticker}: {e}")
            continue

    # Compute median for each date
    median_returns = {}
    for date, returns_list in all_returns_by_date.items():
        if len(returns_list) >= 3:
            median_returns[date] = np.median(returns_list)

    _baseline_returns_cache = pd.Series(median_returns).sort_index().ffill()
    _baseline_source = f"COHORT_MEDIAN:{len(tickers)}_tickers"

    logger.info(f"[COHORT FALLBACK] Computed from {len(tickers)} tickers for {len(_baseline_returns_cache)} dates")
    logger.info(f"[COHORT FALLBACK] Date range: {_baseline_returns_cache.index.min()} to {_baseline_returns_cache.index.max()}")
    logger.info(f"[COHORT FALLBACK] Return stats: mean={_baseline_returns_cache.mean():.4f}, "
                f"std={_baseline_returns_cache.std():.4f}")
    logger.warning(f"[COHORT FALLBACK] REMINDER: This baseline has survivorship bias!")

    return _baseline_returns_cache


def get_baseline_return(date: pd.Timestamp) -> float:
    """
    Get the baseline 20-day return for a specific date.

    Returns the external PIT index return if available, otherwise cohort median.
    Use this for computing relative_strength_cohort.

    Args:
        date: The date to look up

    Returns:
        Baseline 20-day return for the date, or 0.0 if unavailable
    """
    global _baseline_returns_cache

    if _baseline_returns_cache is None:
        return 0.0

    # Normalize date
    date = pd.to_datetime(date)
    if hasattr(date, 'tz') and date.tz is not None:
        date = date.tz_localize(None)

    # Try exact match first
    if date in _baseline_returns_cache.index:
        return _baseline_returns_cache[date]

    # Fall back to nearest previous date (ffill behavior)
    mask = _baseline_returns_cache.index <= date
    if mask.any():
        return _baseline_returns_cache[mask].iloc[-1]

    return 0.0


# Keep old function name for backward compatibility
def get_universe_median_return(date: pd.Timestamp) -> float:
    """DEPRECATED: Use get_baseline_return() instead."""
    return get_baseline_return(date)


def extract_context_features(
    pattern_row: pd.Series,
    ticker_df: pd.DataFrame,
    pattern_end_idx: int
) -> Optional[np.ndarray]:
    """
    Extract 24 context features for Branch B (Gated Residual Network).

    Features (Original 7 - market context):
        0. retention_rate: (Close - Low) / (High - Low) - candle strength indicator
        1. trend_position: Current_Price / 200_SMA
        2. base_duration: log(1 + pattern_duration_days) / log(201)  [LOG NORMALIZED]
        3. relative_volume: Avg_Vol_20D / Avg_Vol_60D
        4. distance_to_high: (52W_High - Price) / 52W_High
        5. log_float: log10(shares_outstanding) - liquidity/size indicator
        6. log_dollar_volume: log10(avg_daily_dollar_volume) - tradability indicator

    Features (Deep Dormancy - sleeper detection):
        7. dormancy_shock: log10(vol_20d_avg / vol_252d_avg) - Is current activity highest in a year?
        8. vol_dryup_ratio: vol_20d_avg / (vol_100d_avg + eps) - Degree of volume exhaustion

    Features (Coil state at detection, BIAS-FREE):
        9. price_position_at_end: Close position in box (0=lower, 1=upper)
        10. volume_shock: max(volume[-3:]) / vol_20d_avg - breakout precursor signal
        11. bbw_slope_5d: BBW change over last 5 days (positive = expanding)
        12. vol_trend_5d: Recent volume vs 20d average (>1 = increasing)
        13. coil_intensity: Combined coil quality score (higher = better)

    Features (Cohort-based relative strength - Jan 2026):
        14. relative_strength_cohort: (Close_t / Close_t-20) - Universe_Median_Return_20D
            - Replaces SPY-based relative strength to avoid look-ahead bias for EU stocks
            - EU stocks close at 17:30 CET, SPY at 22:00 CET (4.5-hour gap)
            - Uses universe median computed from all tickers in the dataset

    Features (Structural Efficiency - Jan 2026):
        15. risk_width_pct: (Upper - Lower) / Upper - structural stop distance
        16. vol_contraction_intensity: Avg_Vol_5d / Avg_Vol_60d - volume drying up
        17. obv_divergence: OBV slope vs Price slope - smart money accumulation

    Features (Liquidity Fragility - Jan 2026):
        18. vol_decay_slope: Linear regression slope of log-volume over 20 days

    Features (Regime-Aware - Jan 2026):
        19. vix_regime_level: VIX normalized (0=calm, 1=crisis)
        20. vix_trend_20d: VIX 20d change (-0.5 to +0.5)
        21. market_breadth_200: % stocks above 200 SMA (0-1)
        22. risk_on_indicator: Risk-on/off composite (0-1)
        23. days_since_regime_change: Log-normalized days since regime shift

    Args:
        pattern_row: Row from patterns DataFrame with pattern metadata
        ticker_df: DataFrame with OHLCV data for the ticker
        pattern_end_idx: Index in ticker_df where pattern ends

    Returns:
        np.ndarray of shape (24,) with normalized context features, or None if data insufficient

    Raises:
        InsufficientDataError: If ticker_df has insufficient data for feature extraction
        DataSchemaError: If required columns are missing from ticker_df
    """
    # ==========================================================================
    # TOP-LEVEL VALIDATION
    # ==========================================================================
    MIN_REQUIRED_ROWS = 60  # Need at least 60 days for 60d rolling features

    if len(ticker_df) < MIN_REQUIRED_ROWS:
        raise InsufficientDataError(
            f"ticker_df has {len(ticker_df)} rows, need at least {MIN_REQUIRED_ROWS}"
        )

    if pattern_end_idx < 0 or pattern_end_idx >= len(ticker_df):
        raise InsufficientDataError(
            f"pattern_end_idx={pattern_end_idx} out of bounds for ticker_df with {len(ticker_df)} rows"
        )

    if pattern_end_idx < MIN_REQUIRED_ROWS:
        raise InsufficientDataError(
            f"pattern_end_idx={pattern_end_idx} too early, need at least {MIN_REQUIRED_ROWS} days of history"
        )

    # Validate required columns in ticker_df
    required_columns = ['close', 'high', 'volume']
    missing_columns = [col for col in required_columns if col not in ticker_df.columns]
    if missing_columns:
        raise DataSchemaError(f"Missing required columns in ticker_df: {missing_columns}")

    # Validate required fields in pattern_row
    required_pattern_fields = ['start_date', 'end_date', 'upper_boundary', 'lower_boundary']
    missing_fields = [f for f in required_pattern_fields if f not in pattern_row.index]
    if missing_fields:
        raise DataSchemaError(f"Missing required fields in pattern_row: {missing_fields}")

    # ==========================================================================
    # FEATURE EXTRACTION (no broad try-except)
    # ==========================================================================
    context = np.zeros(24, dtype=np.float32)  # 24 features (Jan 2026 - added 5 regime features)
    end_idx = pattern_end_idx

    # Access required data - let KeyError propagate if columns missing
    current_price = ticker_df['close'].iloc[end_idx]
    volume = ticker_df['volume']
    high = ticker_df['high']
    close = ticker_df['close']

    # Check for pre-computed columns (from precompute_rolling_features)
    has_precomputed = '_vol_60d_sum' in ticker_df.columns

    # --- Feature 1: Retention Rate (Jan 2026 - replaces float_turnover) ---
    # Formula: (Close - Low) / (High - Low)
    # Interpretation: Where did price close within the day's range?
    # - Near 0 = sellers won (close near low) = weak
    # - Near 1 = buyers won (close near high) = strong accumulation
    # This is NOT redundant with relative_volume (which measures quantity, not strength)
    EPS = 1e-8  # Define EPS early for all calculations

    high = ticker_df['high']
    low = ticker_df['low']

    close_at_end = close.iloc[end_idx]
    high_at_end = high.iloc[end_idx]
    low_at_end = low.iloc[end_idx]

    range_at_end = high_at_end - low_at_end
    if range_at_end > EPS:
        retention_rate = (close_at_end - low_at_end) / range_at_end
    else:
        retention_rate = CONTEXT_FEATURE_DEFAULTS['retention_rate']  # 0.5 (neutral)
    retention_rate = np.clip(retention_rate, 0.0, 1.0)  # Already bounded
    context[0] = retention_rate  # No additional normalization needed (already [0,1])

    # --- Feature 2: Trend Position ---
    if has_precomputed:
        sma_200 = ticker_df['_sma_200'].iloc[end_idx]
    elif end_idx >= 200:
        sma_200 = close.iloc[end_idx-200:end_idx].mean()
    else:
        sma_200 = close.iloc[:end_idx].mean()

    trend_position = current_price / sma_200 if sma_200 > 0 else 1.0
    context[1] = normalize_context_feature('trend_position', trend_position)

    # --- Feature 3: Base Duration (LOG NORMALIZED) ---
    start_date = pd.to_datetime(pattern_row['start_date'])
    end_date = pd.to_datetime(pattern_row['end_date'])
    duration_days = (end_date - start_date).days
    base_duration = np.log1p(duration_days) / np.log1p(200)
    base_duration = np.clip(base_duration, 0.0, 1.0)
    context[2] = base_duration

    # --- Feature 4: Relative Volume ---
    # FIX (Jan 2026): Use log_diff to handle dormant stocks where vol_60d ≈ 0
    # Old: vol_20d / vol_60d → Inf when vol_60d = 0
    # New: log_diff(vol_20d, vol_60d) → bounded, handles zeros gracefully
    if has_precomputed:
        vol_20d_avg = ticker_df['_vol_20d_avg'].iloc[end_idx]
        vol_60d_avg = ticker_df['_vol_60d_avg'].iloc[end_idx]
    else:
        vol_20d_avg = volume.iloc[max(0, end_idx-20):end_idx].mean()
        vol_60d_avg = volume.iloc[max(0, end_idx-60):end_idx].mean()

    relative_volume = log_diff(vol_20d_avg, vol_60d_avg)
    context[3] = normalize_context_feature('relative_volume', relative_volume)

    # --- Feature 5: Distance to High ---
    if has_precomputed:
        high_52w = ticker_df['_high_52w'].iloc[end_idx]
    elif end_idx >= 252:
        high_52w = high.iloc[end_idx-252:end_idx].max()
    else:
        high_52w = high.iloc[:end_idx].max()

    if high_52w > 0:
        distance_to_high = max(0.0, (high_52w - current_price) / high_52w)
    else:
        distance_to_high = CONTEXT_FEATURE_DEFAULTS['distance_to_high']
    context[4] = normalize_context_feature('distance_to_high', distance_to_high)

    # --- Feature 6: Log Float Proxy (PIT-SAFE) ---
    # CRITICAL: Uses only historical price/volume. No external metadata (shares_outstanding, market_cap).
    # Proxy: log10(dollar_vol_20d_avg) - captures liquidity/size without external data
    # Rationale: Higher float stocks tend to have higher dollar volume (correlation ~0.7)
    if has_precomputed:
        dollar_vol_20d_proxy = ticker_df['_dollar_vol_20d_avg'].iloc[end_idx]
    else:
        vol_20d_proxy = volume.iloc[max(0, end_idx-20):end_idx]
        close_20d_proxy = close.iloc[max(0, end_idx-20):end_idx]
        dollar_vol_20d_proxy = (vol_20d_proxy * close_20d_proxy).mean() if len(vol_20d_proxy) > 0 else EPS

    if dollar_vol_20d_proxy is not None and dollar_vol_20d_proxy > EPS and not np.isnan(dollar_vol_20d_proxy):
        log_float_proxy = np.log10(max(dollar_vol_20d_proxy, 1e4))  # Floor at $10k
    else:
        log_float_proxy = CONTEXT_FEATURE_DEFAULTS['log_float']
    context[5] = normalize_context_feature('log_float', log_float_proxy)

    # --- Feature 7: Log Dollar Volume ---
    if has_precomputed:
        dollar_vol_20d = ticker_df['_dollar_vol_20d_avg'].iloc[end_idx]
    else:
        vol_20d = volume.iloc[max(0, end_idx-20):end_idx]
        close_20d = close.iloc[max(0, end_idx-20):end_idx]
        dollar_vol_20d = (vol_20d * close_20d).mean()

    if dollar_vol_20d is not None and dollar_vol_20d > 0 and not np.isnan(dollar_vol_20d):
        log_dollar_volume = np.log10(max(dollar_vol_20d, 1e4))
    else:
        log_dollar_volume = CONTEXT_FEATURE_DEFAULTS['log_dollar_volume']
    context[6] = normalize_context_feature('log_dollar_volume', log_dollar_volume)

    # --- Feature 8: Dormancy Shock (Deep Dormancy) ---
    # FIX (Jan 2026): Use log_diff to handle dormant stocks where vol_252d ≈ 0
    # Old: log10(vol_20d / vol_252d) → -Inf or NaN when vol_252d = 0
    # New: log_diff(vol_20d, vol_252d) → bounded, handles zeros gracefully
    # Positive = "waking up", Negative = "deeply dormant"
    if has_precomputed and '_vol_252d_avg' in ticker_df.columns:
        vol_252d_avg = ticker_df['_vol_252d_avg'].iloc[end_idx]
    else:
        vol_252d_avg = volume.iloc[max(0, end_idx-252):end_idx].mean() if end_idx >= 20 else volume.iloc[:end_idx].mean()

    dormancy_shock = log_diff(vol_20d_avg, vol_252d_avg)
    context[7] = normalize_context_feature('dormancy_shock', dormancy_shock)

    # --- Feature 9: Volume Dryup Ratio (Deep Dormancy) ---
    # FIX (Jan 2026): Use log_diff to handle dormant stocks where vol_100d ≈ 0
    # Old: vol_20d / (vol_100d + eps) → extreme values when vol_100d ≈ 0
    # New: log_diff(vol_20d, vol_100d) → bounded, handles zeros gracefully
    # Low values indicate supply exhaustion (coiling)
    if has_precomputed and '_vol_100d_avg' in ticker_df.columns:
        vol_100d_avg = ticker_df['_vol_100d_avg'].iloc[end_idx]
    else:
        vol_100d_avg = volume.iloc[max(0, end_idx-100):end_idx].mean() if end_idx >= 20 else volume.iloc[:end_idx].mean()

    vol_dryup_ratio = log_diff(vol_20d_avg, vol_100d_avg)
    context[8] = normalize_context_feature('vol_dryup_ratio', vol_dryup_ratio)

    # --- Feature 10: Price Position at End ---
    upper = pattern_row['upper_boundary']
    lower = pattern_row['lower_boundary']

    if pd.isna(upper) or pd.isna(lower) or upper <= lower:
        raise InsufficientDataError(
            f"Invalid box boundaries: upper={upper}, lower={lower}"
        )

    box_height = upper - lower
    price_position = np.clip((current_price - lower) / box_height, 0.0, 1.0)
    context[9] = normalize_context_feature('price_position_at_end', price_position)

    # --- Feature 11: Volume Shock (Breakout Precursor) ---
    # FIX (Jan 2026): Use log_diff to handle dormant stocks where vol_20d ≈ 0
    # Old: vol_last_3d_max / (vol_20d + eps) → extreme values when vol_20d ≈ 0
    # New: log_diff(vol_last_3d_max, vol_20d) → bounded, handles zeros gracefully
    # High values indicate sudden volume spike often preceding breakouts
    if end_idx >= 3:
        vol_last_3d_max = volume.iloc[end_idx-3:end_idx].max()
        volume_shock = log_diff(vol_last_3d_max, vol_20d_avg)
    else:
        volume_shock = CONTEXT_FEATURE_DEFAULTS['volume_shock']
    context[10] = normalize_context_feature('volume_shock', volume_shock)

    # --- Feature 12: BBW Slope 5D ---
    if has_precomputed and end_idx >= 5:
        bbw_now = ticker_df['_bbw_20'].iloc[end_idx]
        bbw_5d_ago = ticker_df['_bbw_20'].iloc[end_idx - 5]
        if pd.notna(bbw_now) and pd.notna(bbw_5d_ago):
            bbw_slope_5d = (bbw_now - bbw_5d_ago) / 5
        else:
            bbw_slope_5d = CONTEXT_FEATURE_DEFAULTS['bbw_slope_5d']
    else:
        bbw_slope_5d = CONTEXT_FEATURE_DEFAULTS['bbw_slope_5d']
    context[11] = normalize_context_feature('bbw_slope_5d', bbw_slope_5d)

    # --- Feature 13: Vol Trend 5D ---
    # FIX (Jan 2026): Use log_diff to handle dormant stocks where vol_20d ≈ 0
    # Old: vol_5d / vol_20d → Inf when vol_20d = 0
    # New: log_diff(vol_5d, vol_20d) → bounded, handles zeros gracefully
    if has_precomputed:
        vol_5d_avg = ticker_df['_vol_5d_avg'].iloc[end_idx]
        vol_20d_avg_trend = ticker_df['_vol_20d_avg'].iloc[end_idx]
        vol_trend_5d = log_diff(vol_5d_avg, vol_20d_avg_trend)
    else:
        vol_trend_5d = CONTEXT_FEATURE_DEFAULTS['vol_trend_5d']
    context[12] = normalize_context_feature('vol_trend_5d', vol_trend_5d)

    # --- Feature 14: Coil Intensity ---
    # FIX (Jan 2026): Update vol_score calculation for log_diff output
    # log_diff range is roughly [-2, 2], so we normalize accordingly
    # vol_trend_5d < 0 means volume contracting (good for coil), > 0 means expanding
    pos_score = 1 - price_position
    bbw_at_end = ticker_df['_bbw_20'].iloc[end_idx] if has_precomputed else 0.1
    bbw_score = 1 - min(bbw_at_end / 0.2, 1) if pd.notna(bbw_at_end) else 0.5
    # Convert log_diff to score: negative (contracting) = high score, positive (expanding) = low score
    # Map [-2, 2] to [1, 0] approximately
    vol_score = np.clip(0.5 - vol_trend_5d / 4, 0.0, 1.0) if not pd.isna(vol_trend_5d) else 0.5
    coil_intensity = (pos_score + bbw_score + vol_score) / 3
    context[13] = normalize_context_feature('coil_intensity', coil_intensity)

    # --- Feature 15: Relative Strength vs Cohort (NO LOOK-AHEAD BIAS) ---
    # Formula: (Ticker_Close / Ticker_Close_20d) - Universe_Median_Return_20D
    # CRITICAL: Replaces SPY-based relative strength to avoid time-zone leakage
    #           EU stocks close at 17:30 CET, SPY at 22:00 CET (4.5-hour gap)
    #
    # Interpretation:
    #   - Positive: Ticker outperforming the universe median
    #   - Negative: Ticker underperforming the universe median
    #   - Near zero: Ticker in line with market
    if end_idx >= 20:
        close_20d_ago = close.iloc[end_idx - 20]
        if close_20d_ago > 0 and current_price > 0:
            ticker_return_20d = (current_price / close_20d_ago) - 1.0

            # Get pattern end date for universe median lookup
            pattern_end_date = pd.to_datetime(pattern_row['end_date'])
            universe_median = get_universe_median_return(pattern_end_date)

            relative_strength_cohort = ticker_return_20d - universe_median
        else:
            relative_strength_cohort = CONTEXT_FEATURE_DEFAULTS['relative_strength_cohort']
    else:
        relative_strength_cohort = CONTEXT_FEATURE_DEFAULTS['relative_strength_cohort']
    context[14] = normalize_context_feature('relative_strength_cohort', relative_strength_cohort)

    # --- Feature 16: Risk Width Percentage (Structural Efficiency) ---
    # Formula: (upper_boundary - lower_boundary) / upper_boundary
    # CRITICAL: Measures structural stop distance. Patterns with >40% are filtered.
    # Lower values = tighter structure = better R:R potential
    if upper > 0:
        risk_width_pct = (upper - lower) / upper
    else:
        risk_width_pct = CONTEXT_FEATURE_DEFAULTS['risk_width_pct']
    context[15] = normalize_context_feature('risk_width_pct', risk_width_pct)

    # --- Feature 17: Volume Contraction Intensity (Structural Efficiency) ---
    # FIX (Jan 2026): Use log_diff to handle dormant stocks where vol_60d ≈ 0
    # Old: vol_5d / vol_60d → Inf when vol_60d = 0
    # New: log_diff(vol_5d, vol_60d) → bounded, handles zeros gracefully
    # INTERPRETATION: Negative = volume contracting (supply exhaustion) = coiling spring
    # Best setups have volume contracting significantly before breakout
    if has_precomputed and '_vol_5d_avg' in ticker_df.columns:
        vol_5d_avg = ticker_df['_vol_5d_avg'].iloc[end_idx]
    else:
        vol_5d_avg = volume.iloc[max(0, end_idx-5):end_idx].mean() if end_idx >= 5 else volume.iloc[:end_idx].mean()

    vol_contraction_intensity = log_diff(vol_5d_avg, vol_60d_avg)
    context[16] = normalize_context_feature('vol_contraction_intensity', vol_contraction_intensity)

    # --- Feature 18: OBV Divergence (Structural Efficiency) ---
    # FIX (Jan 2026): Use LOG-DIFF of normalized slopes to handle dormant stocks
    # Formula: log_diff(normalized_slope(OBV), normalized_slope(Price))
    # INTERPRETATION: OBV rising while price flat = smart money accumulation
    # Positive divergence is bullish (accumulation before breakout)
    # Normalizing by std_y makes slopes unitless (correlation-like), enabling fair comparison
    if end_idx >= 20:
        close_window = close.iloc[max(0, end_idx-20):end_idx+1].values
        volume_window = volume.iloc[max(0, end_idx-20):end_idx+1].values

        cumulative_volume = np.sum(volume_window)

        # STRICT HANDLING: If cumulative_volume == 0, force to neutral
        # This prevents NaN/Inf from poisoning BatchNorm layers
        if cumulative_volume == 0 or len(close_window) < 2:
            obv_divergence = 0.0  # Neutral - cannot measure divergence with no volume
        else:
            # Calculate OBV: cumulative sum of signed volume
            price_changes = np.diff(close_window)
            obv_changes = np.where(price_changes > 0, volume_window[1:],
                                   np.where(price_changes < 0, -volume_window[1:], 0))
            obv = np.cumsum(np.insert(obv_changes, 0, 0))  # Start at 0

            if len(obv) >= 2:
                x = np.arange(len(obv))

                # Helper for normalized slope (robust to flat lines/zero std)
                # Normalizing by std_y makes the slope unitless (correlation-like)
                def get_normalized_slope(y_series):
                    std_y = np.std(y_series)
                    if std_y <= EPS:
                        return 0.0  # Flat line = no trend
                    # Slope = cov(x,y) / var(x)
                    # Normalized slope = slope / std(y)
                    slope = np.cov(x, y_series)[0, 1] / (np.var(x) + EPS)
                    return slope / std_y

                obv_slope_norm = get_normalized_slope(obv)
                price_slope_norm = get_normalized_slope(close_window)

                # LOG-DIFF of normalized slopes
                # Positive = OBV trend > Price trend (accumulation)
                # Negative = Price running without volume support (distribution)
                # Convert to absolute values for log_diff, then restore sign
                obv_abs = abs(obv_slope_norm)
                price_abs = abs(price_slope_norm)

                # log_diff handles magnitude comparison
                log_diff_magnitude = log_diff(obv_abs, price_abs)

                # Sign: positive if OBV slope >= price slope (accumulation)
                if obv_slope_norm >= price_slope_norm:
                    obv_divergence = abs(log_diff_magnitude)
                else:
                    obv_divergence = -abs(log_diff_magnitude)

                # Clip to reasonable range
                obv_divergence = np.clip(obv_divergence, -5.0, 5.0)
            else:
                obv_divergence = 0.0
    else:
        obv_divergence = 0.0  # Insufficient data
    context[17] = normalize_context_feature('obv_divergence', obv_divergence)

    # =========================================================================
    # Index 18: vol_decay_slope - Liquidity Fragility (Jan 2026)
    # =========================================================================
    # Linear regression slope of log-volume over 20 days.
    # Negative = liquidity declining (fragile, drying up)
    # Positive = liquidity increasing (accumulation, healthy interest)
    # =========================================================================
    from config.context_features import vol_decay_slope as calc_vol_decay_slope

    if 'volume' in ticker_df.columns and len(ticker_df) >= 20:
        recent_volumes = ticker_df['volume'].iloc[max(0, pattern_end_idx - 19):pattern_end_idx + 1].values
        if len(recent_volumes) >= 10:  # Need at least 10 days for meaningful slope
            vol_decay = calc_vol_decay_slope(recent_volumes, window=min(20, len(recent_volumes)))
        else:
            vol_decay = 0.0
    else:
        vol_decay = 0.0
    context[18] = normalize_context_feature('vol_decay_slope', vol_decay)

    # =========================================================================
    # Indices 19-23: Regime-Aware Features (Jan 2026)
    # =========================================================================
    # These features capture the broader market environment at pattern end_date:
    #   19: vix_regime_level - VIX normalized (0=calm, 1=crisis)
    #   20: vix_trend_20d - VIX 20d change (-0.5 to +0.5)
    #   21: market_breadth_200 - % stocks above 200 SMA (0-1)
    #   22: risk_on_indicator - Risk-on/off composite (0-1)
    #   23: days_since_regime_change - Log-normalized days since regime shift
    #
    # WHY: Model failed across ALL regimes (0.98-1.04x lift). These features
    # help the model learn regime-conditional breakout patterns.
    # =========================================================================
    pattern_end_date = pd.to_datetime(pattern_row['end_date'])

    try:
        # Get or create cached regime calculator (singleton pattern)
        # Uses create_regime_calculator() which auto-loads data from default paths
        if not hasattr(extract_context_features, '_regime_calculator'):
            extract_context_features._regime_calculator = create_regime_calculator()

        regime_calc = extract_context_features._regime_calculator

        # Extract regime features for pattern end date
        regime_features = regime_calc.get_features(pattern_end_date)

        context[19] = normalize_context_feature('vix_regime_level', regime_features.get('vix_regime_level', 0.25))
        context[20] = normalize_context_feature('vix_trend_20d', regime_features.get('vix_trend_20d', 0.0))
        context[21] = normalize_context_feature('market_breadth_200', regime_features.get('market_breadth_200', 0.5))
        context[22] = normalize_context_feature('risk_on_indicator', regime_features.get('risk_on_indicator', 0.5))
        context[23] = normalize_context_feature('days_since_regime_change', regime_features.get('days_since_regime_change', 0.5))
    except Exception as e:
        # Fallback to defaults if regime data unavailable
        logger.debug(f"Regime features unavailable for {pattern_end_date}: {e}")
        context[19] = normalize_context_feature('vix_regime_level', CONTEXT_FEATURE_DEFAULTS['vix_regime_level'])
        context[20] = normalize_context_feature('vix_trend_20d', CONTEXT_FEATURE_DEFAULTS['vix_trend_20d'])
        context[21] = normalize_context_feature('market_breadth_200', CONTEXT_FEATURE_DEFAULTS['market_breadth_200'])
        context[22] = normalize_context_feature('risk_on_indicator', CONTEXT_FEATURE_DEFAULTS['risk_on_indicator'])
        context[23] = normalize_context_feature('days_since_regime_change', CONTEXT_FEATURE_DEFAULTS['days_since_regime_change'])

    return context


# =============================================================================
# STREAMING SEQUENCE WRITER
# =============================================================================

class ReservoirQuantileEstimator:
    """
    Memory-efficient quantile estimation using reservoir sampling.

    Maintains a fixed-size reservoir for median/IQR calculation
    without storing all data points.
    """

    def __init__(self, reservoir_size: int = 10000, seed: int = 42):
        """
        Args:
            reservoir_size: Maximum samples to keep in reservoir
            seed: Random seed for reproducibility
        """
        self.reservoir_size = reservoir_size
        self.reservoir = {}  # {feature_idx: [samples]}
        self.counts = {}     # {feature_idx: total_seen}
        self.rng = np.random.RandomState(seed)

    def update(self, sequences: np.ndarray):
        """
        Update reservoir with new sequences using Algorithm R.

        Args:
            sequences: Array of shape (n_samples, seq_len, n_features)
        """
        n_samples, seq_len, n_features = sequences.shape

        for feat_idx in range(n_features):
            # Extract all values for this feature
            feat_values = sequences[:, :, feat_idx].flatten()
            feat_values = feat_values[~np.isnan(feat_values)]  # Remove NaN

            if len(feat_values) == 0:
                continue

            if feat_idx not in self.reservoir:
                self.reservoir[feat_idx] = []
                self.counts[feat_idx] = 0

            # Reservoir sampling (Algorithm R)
            for val in feat_values:
                self.counts[feat_idx] += 1
                if len(self.reservoir[feat_idx]) < self.reservoir_size:
                    self.reservoir[feat_idx].append(val)
                else:
                    # Replace with decreasing probability
                    j = self.rng.randint(0, self.counts[feat_idx])
                    if j < self.reservoir_size:
                        self.reservoir[feat_idx][j] = val

    def get_robust_params(self) -> dict:
        """
        Compute median and IQR from reservoir.

        Returns:
            dict: {feature_idx: {'median': float, 'iqr': float}}
        """
        params = {}
        for feat_idx, samples in self.reservoir.items():
            if len(samples) > 0:
                arr = np.array(samples)
                median = np.median(arr)
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                iqr = q3 - q1
                if iqr < 1e-6:
                    iqr = 1.0  # Avoid division by zero

                params[feat_idx] = {
                    'median': float(median),
                    'iqr': float(iqr),
                    'q1': float(q1),
                    'q3': float(q3),
                    'n_samples': self.counts[feat_idx]
                }
        return params

    def get_stats(self) -> dict:
        """Get sampling statistics for debugging."""
        return {
            'n_features': len(self.reservoir),
            'reservoir_sizes': {k: len(v) for k, v in self.reservoir.items()},
            'total_seen': dict(self.counts)
        }


class StreamingSequenceWriter:
    """
    Memory-efficient streaming writer for sequence data.

    Writes sequences directly to disk in chunks, avoiding memory accumulation.
    Uses HDF5 for efficient storage when available.
    """

    def __init__(
        self,
        output_dir: Path,
        window_size: int = 20,
        n_features: int = 10,
        metadata_batch_size: int = 50000,
        skip_npy_export: bool = False
    ):
        """
        Args:
            output_dir: Directory for output files
            window_size: Sequence window size
            n_features: Number of features per timestep
            metadata_batch_size: Rows before flushing metadata to disk
            skip_npy_export: If True, only produce HDF5 (no NPY files)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size
        self.n_features = n_features
        self.metadata_batch_size = metadata_batch_size
        self.skip_npy_export = skip_npy_export

        # Track counts
        self.total_sequences = 0
        self.class_counts = {}

        # Metadata accumulation
        self.metadata_buffer = []
        self.metadata_files = []

        # Memory-efficient quantile estimation
        self.quantile_estimator = ReservoirQuantileEstimator(reservoir_size=10000)

        # HDF5 support
        try:
            import h5py
            self.use_hdf5 = True
            self.h5_file = None
        except ImportError:
            self.use_hdf5 = False
            logger.warning("h5py not installed - using NPY files only")

        # NPY chunk files (if not using HDF5)
        self.npy_chunk_files = []
        self.npy_labels_files = []
        self.npy_context_files = []

    def _init_hdf5(self, timestamp: str):
        """Initialize HDF5 file with resizable datasets."""
        if not self.use_hdf5:
            return

        import h5py
        h5_path = self.output_dir / f"sequences_{timestamp}.h5"

        self.h5_file = h5py.File(h5_path, 'w')

        # Create resizable datasets
        self.h5_file.create_dataset(
            'sequences',
            shape=(0, self.window_size, self.n_features),
            maxshape=(None, self.window_size, self.n_features),
            dtype='float32',
            chunks=(min(1000, 100), self.window_size, self.n_features),
            compression='gzip',
            compression_opts=4
        )

        self.h5_file.create_dataset(
            'labels',
            shape=(0,),
            maxshape=(None,),
            dtype='int32',
            chunks=(min(10000, 1000),),
            compression='gzip'
        )

        self.h5_file.create_dataset(
            'context',
            shape=(0, 24),  # 24 context features (Jan 2026: 19 base + 5 regime features)
            maxshape=(None, 24),
            dtype='float32',
            chunks=(min(10000, 1000), 24),
            compression='gzip',
            compression_opts=4
        )

        logger.info(f"Initialized HDF5: {h5_path}")

    def write_ticker(
        self,
        ticker: str,
        sequences: np.ndarray,
        labels: np.ndarray,
        metadata_rows: list,
        context: np.ndarray = None
    ):
        """
        Write one ticker's sequences to storage.

        Args:
            ticker: Ticker symbol
            sequences: Array of shape (n, window_size, n_features)
            labels: Array of shape (n,)
            metadata_rows: List of metadata dicts
            context: Optional context features of shape (n, 24)

        Note:
            If metadata contains FUTURE_ prefixed columns, a warning is logged
            (these are allowed for post-hoc analysis but must not be used as model input).
        """
        if sequences is None or len(sequences) == 0:
            return

        # =======================================================================
        # FUTURE_ COLUMN ENFORCEMENT (Jan 2026 - Active Prevention)
        # =======================================================================
        # Any column prefixed with FUTURE_ indicates it uses post-pattern information
        # (e.g., FUTURE_max_gain, FUTURE_stop_hit_day, FUTURE_breakout_detected)
        #
        # These columns are ALLOWED in metadata for post-hoc analysis:
        #   - Calculating EV (Expected Value) on test set
        #   - Computing Lift metrics
        #   - Creating the "Answer Key" for backtesting
        #
        # They MUST NOT be used as model input features (sequences/context tensors).
        # The FUTURE_ prefix serves as documentation and a safety marker.
        #
        # ACTIVE ENFORCEMENT:
        #   1. Validate context array shape matches expected (24 features)
        #   2. Log FUTURE_* columns found in metadata for audit trail
        #   3. Strip FUTURE_* columns from metadata_rows if filter_future=True
        # =======================================================================

        # ENFORCEMENT #1: Validate context array dimensions
        if context is not None:
            expected_context_dim = 24  # NUM_CONTEXT_FEATURES: 19 base + 5 regime features (Jan 2026)
            actual_context_dim = context.shape[1] if context.ndim == 2 else context.shape[0]
            if actual_context_dim != expected_context_dim:
                raise ValueError(
                    f"Context array has {actual_context_dim} features, expected {expected_context_dim}. "
                    f"This could indicate FUTURE_* data leaking into model input!"
                )

        # ENFORCEMENT #2: Detect and log FUTURE_* columns in metadata
        if metadata_rows and len(metadata_rows) > 0:
            sample_row = metadata_rows[0]
            future_columns = [k for k in sample_row.keys() if k.startswith('FUTURE_')]
            if future_columns and not hasattr(self, '_future_columns_warned'):
                logger.warning(
                    f"LOOKAHEAD AUDIT: Metadata contains FUTURE_ prefixed columns: "
                    f"{future_columns}. These are ONLY for post-hoc analysis - "
                    f"model input (sequences/context) is validated to exclude these."
                )
                self._future_columns_warned = True  # Only warn once per writer instance

        n_seq = len(sequences)

        # Update class counts
        unique, counts = np.unique(labels, return_counts=True)
        for cls, cnt in zip(unique, counts):
            self.class_counts[int(cls)] = self.class_counts.get(int(cls), 0) + cnt

        # Update quantile estimator (for robust scaling)
        self.quantile_estimator.update(sequences)

        # Write to HDF5
        if self.use_hdf5 and self.h5_file is not None:
            old_len = self.h5_file['sequences'].shape[0]
            new_len = old_len + n_seq

            self.h5_file['sequences'].resize(new_len, axis=0)
            self.h5_file['sequences'][old_len:new_len] = sequences

            self.h5_file['labels'].resize(new_len, axis=0)
            self.h5_file['labels'][old_len:new_len] = labels

            if context is not None:
                self.h5_file['context'].resize(new_len, axis=0)
                self.h5_file['context'][old_len:new_len] = context

        # Accumulate metadata
        self.metadata_buffer.extend(metadata_rows)

        # Flush metadata periodically
        if len(self.metadata_buffer) >= self.metadata_batch_size:
            self._flush_metadata()

        self.total_sequences += n_seq

    def _flush_metadata(self):
        """Flush metadata buffer to disk."""
        if len(self.metadata_buffer) == 0:
            return

        batch_idx = len(self.metadata_files)
        meta_path = self.output_dir / f"metadata_batch_{batch_idx}.parquet"

        df = pd.DataFrame(self.metadata_buffer)
        df.to_parquet(meta_path, index=False)

        self.metadata_files.append(meta_path)
        self.metadata_buffer = []

        logger.debug(f"Flushed metadata batch {batch_idx}: {len(df)} rows")

    def finalize(self, timestamp: str) -> dict:
        """
        Finalize and return output file paths.

        Args:
            timestamp: Timestamp string for filenames

        Returns:
            dict: Paths to output files
        """
        # Flush remaining metadata
        self._flush_metadata()

        # Consolidate metadata files
        metadata_path = self.output_dir / f"metadata_{timestamp}.parquet"
        if len(self.metadata_files) > 0:
            dfs = [pd.read_parquet(f) for f in self.metadata_files]
            combined = pd.concat(dfs, ignore_index=True)
            combined.to_parquet(metadata_path, index=False)

            # Cleanup batch files
            for f in self.metadata_files:
                f.unlink()
        else:
            # Create empty metadata file
            pd.DataFrame().to_parquet(metadata_path, index=False)

        output_paths = {'metadata': str(metadata_path)}

        # Finalize HDF5
        if self.use_hdf5 and self.h5_file is not None:
            h5_path = self.h5_file.filename
            self.h5_file.close()
            output_paths['h5'] = str(h5_path)
            output_paths['sequences'] = str(h5_path)
            output_paths['labels'] = str(h5_path)
            output_paths['context'] = str(h5_path)

            # Save robust scaling params in format expected by 02_train_temporal.py
            raw_params = self.quantile_estimator.get_robust_params()

            # Convert from {feat_idx: {'median': ..., 'iqr': ...}} to {'median': [...], 'iqr': [...]}
            n_features = self.n_features
            median_list = [0.0] * n_features
            iqr_list = [1.0] * n_features  # Default to 1.0 to avoid division by zero

            for feat_idx, params in raw_params.items():
                if isinstance(feat_idx, int) and 0 <= feat_idx < n_features:
                    median_list[feat_idx] = params.get('median', 0.0)
                    iqr_list[feat_idx] = params.get('iqr', 1.0)

            robust_params = {
                'scaling_type': 'robust',
                'median': median_list,
                'iqr': iqr_list,
                'n_features': n_features,
                'raw_stats': raw_params  # Keep detailed stats for debugging
            }

            params_path = self.output_dir / f"robust_scaling_params_{timestamp}.json"
            with open(params_path, 'w') as f:
                json.dump(robust_params, f, indent=2)
            output_paths['robust_params'] = str(params_path)
            logger.info(f"Saved robust scaling params: {params_path}")

        return output_paths

    def get_stats(self) -> dict:
        """Get generation statistics."""
        return {
            'total_sequences': self.total_sequences,
            'class_counts': self.class_counts,
            'memory_mode': 'hdf5' if self.use_hdf5 else 'npy'
        }

    def get_robust_params(self) -> dict:
        """Get estimated robust scaling parameters."""
        return self.quantile_estimator.get_robust_params()


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def process_ticker_pipeline(
    ticker: str,
    ticker_patterns: pd.DataFrame,
    window_size: int,
    min_pattern_days: int,
    apply_physics: bool = False,
    allowed_market_caps: list = None,
    min_width_pct: float = 0.02,
    min_dollar_volume: float = 50000,
    apply_nms: bool = False,
    nms_mode: str = 'highlander',
    nms_overlap_days: int = 10,
    use_event_anchor: bool = False,
    apply_heartbeat: bool = False,
    max_volume_cv: float = 0.8,
    calc_coil_features: bool = False,
    danger_zone_pct: float = 0.25,
    physics_mode: str = 'training'  # 'training' or 'inference'
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[list]]:
    """
    Complete pipeline for a single ticker (TRUE STREAMING).

    Consolidates: data loading -> feature calculation -> filtering -> sequence generation.
    Each worker is fully self-contained with NO global state.

    This function is designed for ProcessPoolExecutor. It creates fresh
    instances per worker to avoid pickle/GCS client issues.

    Args:
        ticker: Ticker symbol
        ticker_patterns: DataFrame with patterns for this ticker
        window_size: Sequence window size
        min_pattern_days: Minimum pattern duration
        apply_physics: Apply physics filter
        allowed_market_caps: List of allowed market cap categories
        min_width_pct: Minimum pattern width percentage
        min_dollar_volume: Minimum average daily dollar volume
        apply_nms: Apply NMS filter
        nms_mode: 'highlander' or 'trinity'
        nms_overlap_days: Days overlap threshold for NMS
        use_event_anchor: Use event-anchor NMS
        apply_heartbeat: Apply heartbeat (volume CV) filter
        max_volume_cv: Maximum volume CV threshold
        calc_coil_features: Calculate coil features
        danger_zone_pct: Danger zone threshold
        physics_mode: 'training' or 'inference'. In training mode, dormancy filters
                      (ghost_trade, data_gap) are disabled to preserve lottery tickets.

    Returns:
        (sequences, labels, context, metadata_rows) or (None, None, None, None)
    """
    # Import inside worker to avoid module-level pickle issues
    from config import TemporalFeatureConfig, SequenceGenerationConfig

    if len(ticker_patterns) == 0:
        return None, None, None, None

    # 1. Create fresh instances (no global state)
    feature_config = TemporalFeatureConfig()
    sequence_config = SequenceGenerationConfig(
        window_size=window_size,
        min_pattern_duration=min_pattern_days
    )

    detector = TemporalPatternDetector(
        feature_config=feature_config,
        sequence_config=sequence_config,
        skip_composite_normalization=True
    )
    data_loader = DataLoader()

    # 2. Load ticker data LOCALLY (fresh instance, no global cache)
    try:
        ticker_df = data_loader.load_ticker(ticker, validate=False)
    except Exception as e:
        return None, None, None, None

    if ticker_df is None or len(ticker_df) == 0:
        return None, None, None, None

    # Normalize index
    if not isinstance(ticker_df.index, pd.DatetimeIndex):
        if 'date' in ticker_df.columns:
            ticker_df['date'] = pd.to_datetime(ticker_df['date'])
            ticker_df = ticker_df.set_index('date')

    if ticker_df.index.tz is not None:
        ticker_df.index = ticker_df.index.tz_localize(None)

    # 3. Pre-compute rolling features ONCE for this ticker
    ticker_df = precompute_rolling_features(ticker_df)

    # 4. Apply filters LOCALLY to this ticker's patterns
    patterns = ticker_patterns.copy()

    # Physics filter first (removes untradeable before expensive NMS)
    # Now includes Zombie Health Check for data quality issues
    # NOTE: In training mode, dormancy filters are disabled to preserve lottery tickets
    if apply_physics:
        patterns = apply_physics_filter(
            patterns,
            ticker_df=ticker_df,  # Pass for Zombie Health Check
            allowed_market_caps=allowed_market_caps,
            min_width_pct=min_width_pct,
            min_dollar_volume=min_dollar_volume,
            mode=physics_mode  # 'training' = preserve dormant patterns, 'inference' = filter them
        )

    if len(patterns) == 0:
        del ticker_df
        gc.collect()
        return None, None, None, None

    # NMS filter
    if apply_nms:
        if use_event_anchor:
            patterns = apply_event_anchor_nms(
                patterns, ticker_df, nms_overlap_days
            )
        else:
            patterns = apply_nms_filter(
                patterns, nms_overlap_days, 'box_width', nms_mode
            )

    if len(patterns) == 0:
        del ticker_df
        gc.collect()
        return None, None, None, None

    # Heartbeat filter (marks patterns with is_noise=True)
    if apply_heartbeat:
        patterns = apply_heartbeat_filter(
            patterns, ticker_df, max_volume_cv
        )
    else:
        patterns['is_noise'] = False

    # Coil features (adds columns for context)
    if calc_coil_features:
        patterns = calculate_coil_features(
            patterns, ticker_df, danger_zone_pct
        )

    # 5. Generate sequences
    all_sequences = []
    all_labels = []
    all_context = []
    all_metadata_rows = []
    skipped_patterns = 0
    hard_negatives = 0  # Patterns with is_noise=True (label overridden to 1)

    for _, pattern_row in patterns.iterrows():
        pattern_id = pattern_row.get('pattern_id', f"{ticker}_{pattern_row.get('start_date', 'unknown')}")

        # Generate sequences for this pattern
        sequences = detector.generate_sequences_for_pattern(
            ticker=ticker,
            pattern_start=pattern_row['start_date'],
            pattern_end=pattern_row['end_date'],
            df=ticker_df
        )

        if sequences is None or len(sequences) == 0:
            continue

        # Calculate pattern_end_idx for context feature extraction
        pattern_end_idx = pattern_row.get('end_idx')
        if pattern_end_idx is None:
            pattern_end_date = pd.to_datetime(pattern_row['end_date'])
            if isinstance(ticker_df.index, pd.DatetimeIndex):
                idx_positions = ticker_df.index.get_indexer([pattern_end_date], method='ffill')
                if idx_positions[0] < 0:
                    logger.warning(f"{ticker}: Pattern {pattern_id} end_date {pattern_end_date} not found in data, skipping")
                    skipped_patterns += 1
                    continue
                pattern_end_idx = idx_positions[0]
            else:
                logger.warning(f"{ticker}: Pattern {pattern_id} has non-datetime index, skipping")
                skipped_patterns += 1
                continue

        # Extract context features - catch specific errors
        try:
            context_features = extract_context_features(pattern_row, ticker_df, pattern_end_idx)
        except InsufficientDataError as e:
            logger.warning(f"{ticker}: Pattern {pattern_id} skipped - insufficient data: {e}")
            skipped_patterns += 1
            continue
        except DataSchemaError as e:
            logger.warning(f"{ticker}: Pattern {pattern_id} skipped - schema error: {e}")
            skipped_patterns += 1
            continue
        except KeyError as e:
            logger.warning(f"{ticker}: Pattern {pattern_id} skipped - missing column: {e}")
            skipped_patterns += 1
            continue

        if context_features is None:
            logger.warning(f"{ticker}: Pattern {pattern_id} skipped - context extraction returned None")
            skipped_patterns += 1
            continue

        # All validations passed - add to results
        all_sequences.append(sequences)

        # =====================================================================
        # HEARTBEAT FILTER: Hard Negative Training
        # =====================================================================
        # =====================================================================
        # ALPHA PRESERVATION FIX (Jan 2026)
        # =====================================================================
        # PREVIOUS BUG: Patterns with erratic volume (is_noise=True) had their
        # labels OVERRIDDEN to Class 1 (Noise), regardless of actual price outcome.
        #
        # WHY THIS WAS WRONG (Alpha Destruction):
        #   - "Erratic" volume often signals institutional accumulation (block trades)
        #   - High volume CV can indicate pre-catalyst positioning ("Power Plays")
        #   - By labeling these as Noise, we trained the model to AVOID the most
        #     explosive setups - destroying ~15% of highest-beta alpha
        #
        # FIX: Keep is_noise as metadata for analysis, but DO NOT corrupt the label.
        # The model should learn from TRUE price outcomes, not our variance assumptions.
        # If erratic volume truly predicts failure, the model will learn that from
        # the actual Danger/Noise outcomes - not from forced label overrides.
        # =====================================================================
        is_noise = pattern_row.get('is_noise', False)
        original_label = pattern_row['outcome_class']

        # CRITICAL: Use the TRUE label - DO NOT override based on volume variance
        label = original_label

        if is_noise:
            # Track for analysis only - DO NOT corrupt the target variable
            hard_negatives += 1
            logger.debug(
                f"{ticker}: Pattern {pattern_id} has erratic volume (is_noise=True) - "
                f"KEEPING TRUE LABEL={original_label} (no longer overriding to Noise)"
            )

        labels = np.full(len(sequences), label)
        all_labels.append(labels)

        context_batch = np.tile(context_features, (len(sequences), 1))
        all_context.append(context_batch)

        # Create metadata
        phase_value = get_market_phase(pattern_row['start_date']).value
        nms_cluster_id = pattern_row.get('nms_cluster_id', -1)

        # Generate unique cluster ID if not available
        if nms_cluster_id == -1:
            cluster_str = f"{ticker}_{pattern_row['start_date']}_{pattern_row['end_date']}"
            nms_cluster_id = int(hashlib.md5(cluster_str.encode()).hexdigest()[:8], 16)

        base_metadata = {
            'ticker': ticker,
            'label': label,
            'original_label': int(original_label) if pd.notna(original_label) else None,
            'is_noise': bool(is_noise),
            'pattern_start_date': pattern_row['start_date'],
            'pattern_end_date': pattern_row['end_date'],
            'pattern_id': pattern_id,
            'market_phase': phase_value,
            'nms_cluster_id': nms_cluster_id
        }

        # Add enrichment fields
        for field in METADATA_ENRICHMENT_FIELDS:
            if field in pattern_row.index:
                value = pattern_row.get(field)
                if pd.notna(value):
                    if isinstance(value, (np.integer, np.floating)):
                        value = float(value)
                    base_metadata[field] = value

        for seq_idx in range(len(sequences)):
            metadata_row = base_metadata.copy()
            metadata_row['sequence_idx'] = seq_idx
            all_metadata_rows.append(metadata_row)

    if skipped_patterns > 0:
        logger.debug(f"{ticker}: Skipped {skipped_patterns} patterns due to data issues")

    if hard_negatives > 0:
        logger.debug(
            f"{ticker}: {hard_negatives} patterns with erratic volume (is_noise=True) - "
            f"keeping TRUE labels (metadata tracks is_noise for analysis)"
        )

    # 6. Explicit cleanup
    del ticker_df
    gc.collect()

    if len(all_sequences) == 0:
        return None, None, None, None

    # Concatenate results
    sequences = np.concatenate(all_sequences, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    context = np.concatenate(all_context, axis=0)

    # ==========================================================================
    # NaN VALIDATION (Critical for training stability)
    # ==========================================================================
    if np.isnan(sequences).any():
        nan_count = np.isnan(sequences).sum()
        nan_pct = nan_count / sequences.size * 100
        raise ValueError(
            f"NaN produced in sequence generation! "
            f"Found {nan_count} NaN values ({nan_pct:.2f}% of data). "
            f"Check volume normalization and feature extraction."
        )

    if np.isnan(context).any():
        nan_count = np.isnan(context).sum()
        nan_pct = nan_count / context.size * 100
        raise ValueError(
            f"NaN produced in context features! "
            f"Found {nan_count} NaN values ({nan_pct:.2f}% of data). "
            f"Check extract_context_features() for division by zero."
        )

    return sequences, labels, context, all_metadata_rows


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Generate temporal sequences from patterns (True Streaming)')
    parser.add_argument('--input', type=str,
                       default=f'{DEFAULT_OUTPUT_DIR}/detected_patterns.parquet',
                       help='Path to detected patterns file')
    parser.add_argument('--output-dir', type=str,
                       default=DEFAULT_SEQUENCE_DIR,
                       help='Output directory for sequences')
    parser.add_argument('--window-size', type=int, default=TEMPORAL_WINDOW_SIZE,
                       help='Sequence window size')
    parser.add_argument('--min-pattern-days', type=int, default=MIN_PATTERN_DURATION,
                       help='Minimum pattern duration')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of patterns (for testing)')

    # Streaming options
    parser.add_argument('--parallel-streaming', action='store_true',
                        help='Enable parallel processing using ProcessPoolExecutor')
    parser.add_argument('--streaming-workers', type=int, default=4,
                        help='Number of worker processes for parallel streaming (default: 4)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output (summary only)')
    parser.add_argument('--metadata-batch-size', type=int, default=50000,
                        help='Metadata rows before flushing to disk')
    parser.add_argument('--skip-npy-export', action='store_true',
                        help='Skip .npy export (keep HDF5 only)')
    parser.add_argument('--claude-safe', action='store_true',
                        help='Minimal output mode for running within Claude Code terminal')

    # Data Quality Filters
    parser.add_argument('--apply-nms', action='store_true',
                        help='Apply NMS to de-duplicate overlapping patterns')
    parser.add_argument('--nms-mode', type=str, default='highlander', choices=['highlander', 'trinity'],
                        help='NMS mode: highlander (1 per cluster) or trinity (3 per cluster)')
    parser.add_argument('--nms-overlap-days', type=int, default=10,
                        help='Days overlap threshold for NMS clustering (default: 10)')
    parser.add_argument('--use-event-anchor', action='store_true',
                        help='Use event-anchor NMS instead of standard NMS')
    parser.add_argument('--dedup-overlap-days', type=int, default=5,
                        help='Days threshold for overlapping window deduplication (default: 5). '
                             'Deduplication is now mandatory; this only adjusts the overlap window.')
    parser.add_argument('--apply-physics-filter', action='store_true',
                        help='Apply physics filter to remove untradeable patterns')
    parser.add_argument('--mode', type=str, default='training', choices=['training', 'inference'],
                        help='Pipeline mode: training (preserve dormant lottery tickets) or inference (strict filters)')
    parser.add_argument('--allowed-market-caps', type=str, default='Nano,Micro,Small',
                        help='Comma-separated list of allowed market cap categories')
    parser.add_argument('--min-width-pct', type=float, default=0.02,
                        help='Minimum pattern width percentage (default: 0.02 = 2%%)')
    parser.add_argument('--min-dollar-volume', type=float, default=50000,
                        help='Minimum average daily dollar volume (default: 50000). '
                             'LIQUIDITY FLOOR: Alpha validation filter - if alpha disappears with this '
                             'threshold, the alpha was fake (untradeable illiquid stocks). '
                             'Patterns with lower liquidity are filtered to protect small retail accounts.')

    # Heartbeat filter
    parser.add_argument('--apply-heartbeat-filter', action='store_true',
                        help='Mark erratic-volume patterns as noise')
    parser.add_argument('--max-volume-cv', type=float, default=0.8,
                        help='Maximum volume CV threshold (default: 0.8)')

    # Coil features
    parser.add_argument('--calculate-coil-features', action='store_true',
                        help='Calculate bias-free coil features for training')
    parser.add_argument('--danger-zone-pct', type=float, default=0.25,
                        help='Danger zone threshold (default: 0.25 = lower 25%% of box)')

    # Relative strength baseline (survivorship bias fix)
    parser.add_argument('--baseline-index', type=str, default=None,
                        help='External PIT index for relative strength baseline (e.g., RSP, IWM, EXSA.DE). '
                             'If not specified, uses default for region. '
                             'IMPORTANT: Using external index avoids survivorship bias from cohort median.')
    parser.add_argument('--region', type=str, default='EU', choices=['US', 'EU'],
                        help='Region for default baseline index selection (default: EU). '
                             'US -> RSP (S&P 500 Equal Weight), EU -> EXSA.DE (STOXX Europe 600)')

    # NOTE: Dollar bar experiment (Jan 2026) resulted in 99% data loss.
    # EU micro/small-caps average <1 dollar bar per day, making 20-bar windows impossible.
    # Keeping daily time bars as the only supported option.
    # See output/reports/VonLinck_TRANS_Technical_Report_20260128.pdf, Section XII for details.

    return parser.parse_args()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    args = parse_args()

    # Claude-safe mode: minimal output
    if args.claude_safe:
        logger.setLevel(logging.ERROR)
        import warnings
        warnings.filterwarnings('ignore')
        os.environ['TQDM_DISABLE'] = '1'
        print("[CLAUDE-SAFE] Minimal output mode - only errors and final summary")
    elif args.quiet:
        logger.setLevel(logging.WARNING)
        import warnings
        warnings.filterwarnings('ignore')

    logger.info("=" * 60)
    logger.info("Step 1: Generate Temporal Sequences (True Streaming)")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detected patterns
    logger.info(f"Loading patterns from {args.input}")
    patterns_df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(patterns_df)} patterns")

    # ================================================================
    # MANDATORY DATA INTEGRITY CHECK (Jan 2026)
    # Prevents 74x duplication bug from propagating through pipeline
    # ================================================================
    if not args.claude_safe:
        logger.info("=" * 60)
        logger.info("DATA INTEGRITY CHECK")
        logger.info("=" * 60)

        date_col = 'pattern_end_date' if 'pattern_end_date' in patterns_df.columns else 'end_date'
        label_col = 'outcome_class' if 'outcome_class' in patterns_df.columns else 'label'
        checker = DataIntegrityChecker(patterns_df, date_col=date_col, label_col=label_col, strict=False)

        dup_passed, dup_details = checker.check_duplication()
        if dup_passed:
            logger.info(f"[PASS] Input patterns: {dup_details['unique_patterns']:,} unique")
        else:
            logger.warning(f"[WARN] Input has {dup_details['duplication_ratio']:.1f}x duplication")
            logger.warning(f"       Rows: {dup_details['total_rows']:,}, Unique: {dup_details['unique_patterns']:,}")
            logger.warning("       Consider deduplicating input patterns")

        # Statistical power check
        if 'label' in patterns_df.columns or 'outcome_class' in patterns_df.columns:
            power_passed, power_details = checker.check_statistical_power()
            if power_passed:
                logger.info(f"[PASS] Target events: {power_details['target_events']:,}")
            else:
                logger.warning(f"[WARN] Low power: {power_details['target_events']:,} target events")

        logger.info("=" * 60)

    # ================================================================
    # MANDATORY DEDUPLICATION (Jan 2026)
    # Cross-ticker and overlap deduplication are now ALWAYS applied.
    # This removes ~73.6% redundant patterns (ETF duplicates, overlapping windows).
    # If alpha disappears after dedup, the alpha was fake.
    # ================================================================
    from utils.data_integrity import deduplicate_cross_ticker, deduplicate_overlapping_windows

    original_count = len(patterns_df)
    logger.info("=" * 60)
    logger.info("MANDATORY DEDUPLICATION (Cross-Ticker + Overlap)")
    logger.info("=" * 60)

    # ALWAYS apply cross-ticker deduplication (removes ETF/cross-listing duplicates)
    patterns_df = deduplicate_cross_ticker(
        patterns_df,
        verbose=not args.claude_safe
    )

    # ALWAYS apply overlap deduplication (removes same consolidation at different dates)
    # Default: 5-day overlap window
    overlap_days = args.dedup_overlap_days if args.dedup_overlap_days > 0 else 5
    patterns_df = deduplicate_overlapping_windows(
        patterns_df,
        overlap_days=overlap_days,
        verbose=not args.claude_safe
    )

    removed = original_count - len(patterns_df)
    logger.info(f"Deduplication complete: {original_count:,} -> {len(patterns_df):,} ({removed:,} removed, {100*removed/original_count:.1f}%)")
    logger.info("=" * 60)

    # Get unique tickers
    tickers = patterns_df['ticker'].unique()
    logger.info(f"Processing {len(tickers)} tickers with TRUE STREAMING")

    # ==========================================================================
    # RELATIVE STRENGTH BASELINE: Pre-compute returns (PIT index or cohort median)
    # ==========================================================================
    # CRITICAL: Must be computed BEFORE processing tickers to avoid look-ahead bias.
    #           This replaces SPY-based relative strength which has time-zone leakage
    #           (EU closes at 17:30 CET, SPY at 22:00 CET = 4.5-hour gap).
    #
    # SURVIVORSHIP BIAS FIX (Jan 2026):
    #   - Cohort median from patterns_df has survivorship bias (only survivors)
    #   - External PIT index (RSP, IWM, EXSA.DE) avoids this bias
    #   - Model learns correct relative strength without conservative distortion
    logger.info("=" * 60)
    logger.info("Step 0: Computing Baseline Returns for Relative Strength")
    logger.info("=" * 60)
    compute_universe_median_returns(
        patterns_df,
        baseline_index=args.baseline_index,
        region=args.region
    )
    logger.info("=" * 60)

    # Limit for testing
    if args.limit:
        patterns_df = patterns_df.head(args.limit)
        tickers = patterns_df['ticker'].unique()
        logger.info(f"Limited to {len(patterns_df)} patterns ({len(tickers)} tickers)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize streaming writer
    skip_npy = args.skip_npy_export or args.claude_safe
    writer = StreamingSequenceWriter(
        output_dir=output_dir,
        window_size=args.window_size,
        n_features=10,
        metadata_batch_size=args.metadata_batch_size,
        skip_npy_export=skip_npy
    )

    if writer.use_hdf5:
        writer._init_hdf5(timestamp)

    # Build filter arguments dict
    filter_args = {
        'window_size': args.window_size,
        'min_pattern_days': args.min_pattern_days,
        'apply_physics': args.apply_physics_filter,
        'allowed_market_caps': [cap.strip() for cap in args.allowed_market_caps.split(',')] if args.apply_physics_filter else None,
        'min_width_pct': args.min_width_pct,
        'min_dollar_volume': args.min_dollar_volume,
        'apply_nms': args.apply_nms,
        'nms_mode': args.nms_mode,
        'nms_overlap_days': args.nms_overlap_days,
        'use_event_anchor': args.use_event_anchor,
        'apply_heartbeat': args.apply_heartbeat_filter,
        'max_volume_cv': args.max_volume_cv,
        'calc_coil_features': args.calculate_coil_features,
        'danger_zone_pct': args.danger_zone_pct,
        'physics_mode': args.mode,  # 'training' preserves dormant lottery tickets
    }

    logger.info(f"Filter settings: physics={args.apply_physics_filter}, nms={args.apply_nms} ({args.nms_mode}), "
               f"heartbeat={args.apply_heartbeat_filter}, coil={args.calculate_coil_features}, mode={args.mode}")

    # Progress bar
    pbar = tqdm(
        total=len(tickers),
        desc="TRUE STREAMING",
        disable=args.quiet or args.claude_safe
    )

    # Use ProcessPoolExecutor for parallel processing
    n_workers = args.streaming_workers if args.parallel_streaming else 1

    if n_workers > 1:
        logger.info(f"Using {n_workers} parallel workers")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit each ticker as a separate task
            futures = {}
            for ticker in tickers:
                ticker_patterns = patterns_df[patterns_df['ticker'] == ticker]
                future = executor.submit(
                    process_ticker_pipeline,
                    ticker=ticker,
                    ticker_patterns=ticker_patterns,
                    **filter_args
                )
                futures[future] = ticker

            # Collect results as they complete
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    sequences, labels, context, metadata = future.result()

                    if sequences is not None and len(sequences) > 0:
                        writer.write_ticker(ticker, sequences, labels, metadata, context)

                except Exception as e:
                    logger.warning(f"{ticker}: Worker failed: {e}")

                pbar.update(1)
                gc.collect()

    else:
        # Sequential mode
        logger.info("Using sequential processing (single worker)")

        for ticker in tickers:
            ticker_patterns = patterns_df[patterns_df['ticker'] == ticker]

            sequences, labels, context, metadata = process_ticker_pipeline(
                ticker=ticker,
                ticker_patterns=ticker_patterns,
                **filter_args
            )

            if sequences is not None and len(sequences) > 0:
                writer.write_ticker(ticker, sequences, labels, metadata, context)

            pbar.update(1)
            gc.collect()

    pbar.close()

    # Finalize
    if writer.total_sequences == 0:
        logger.error("No sequences generated!")
        return

    output_paths = writer.finalize(timestamp)
    stats = writer.get_stats()

    logger.info(f"\nGenerated {stats['total_sequences']:,} total sequences")
    logger.info(f"Memory mode: {stats['memory_mode']}")

    # Class distribution
    logger.info("\nClass distribution:")
    for cls in sorted(stats['class_counts'].keys()):
        count = stats['class_counts'][cls]
        pct = 100 * count / stats['total_sequences']
        logger.info(f"  K{cls}: {count:,} ({pct:.2f}%)")

    # Output paths
    logger.info(f"\nSaved sequences to {output_paths['sequences']}")
    logger.info(f"Saved labels to {output_paths['labels']}")
    logger.info(f"Saved context to {output_paths.get('context', 'N/A')}")
    logger.info(f"Saved metadata to {output_paths['metadata']}")
    if 'robust_params' in output_paths:
        logger.info(f"Saved robust scaling params to {output_paths['robust_params']}")
    if 'h5' in output_paths:
        logger.info(f"HDF5 archive: {output_paths['h5']}")

    # ================================================================
    # FINAL OUTPUT INTEGRITY CHECK (Jan 2026)
    # Verify generated metadata is not duplicated
    # ================================================================
    if 'metadata' in output_paths and not args.claude_safe:
        try:
            output_meta = pd.read_parquet(output_paths['metadata'])
            out_checker = DataIntegrityChecker(output_meta, strict=False)
            out_dup_passed, out_dup_details = out_checker.check_duplication()

            if out_dup_passed:
                logger.info(f"\n[PASS] Output integrity: {out_dup_details['unique_patterns']:,} unique patterns")
            else:
                logger.error(f"\n[FAIL] Output has {out_dup_details['duplication_ratio']:.1f}x duplication!")
                logger.error(f"       This will cause inflated statistics in downstream analysis.")
                logger.error(f"       Check sequence generation logic for bugs.")
        except Exception as e:
            logger.warning(f"Could not verify output integrity: {e}")

    # Summary
    if args.quiet:
        total = stats['total_sequences']
        print(f"[OK] {total:,} sequences | {len(tickers)} tickers | TRUE STREAMING")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("TRUE STREAMING Complete")
        logger.info("=" * 60)
        logger.info(f"Tickers processed: {len(tickers)}")
        logger.info(f"Mode: true_streaming (workers={n_workers})")

        # CRITICAL: Leakage prevention reminder
        logger.info("\n" + "-" * 60)
        logger.info("REMINDER: Use GroupKFold on 'nms_cluster_id' in training")
        logger.info("to prevent leakage between related patterns.")
        logger.info("-" * 60)


if __name__ == "__main__":
    main()
