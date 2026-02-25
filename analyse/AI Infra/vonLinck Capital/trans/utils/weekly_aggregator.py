"""
Weekly Candle Aggregator for TRAnS Pattern Detection
====================================================

Aggregates daily OHLCV data into weekly candles for longer-term
consolidation pattern qualification (10-week instead of 10-day).

This module provides:
- resample_to_weekly(): Converts daily OHLCV to weekly candles
- calculate_weekly_indicators(): Adds technical indicators on weekly timeframe
- Weekly-to-daily date mapping for outcome labeling

The weekly qualification approach looks for consolidation over ~2.5 months
(10 weeks) rather than ~2 weeks (10 days), which filters for stronger
accumulation patterns.

Usage:
    from utils.weekly_aggregator import resample_to_weekly, calculate_weekly_indicators

    # Aggregate daily to weekly
    weekly_df, week_to_daily_map = resample_to_weekly(daily_df)

    # Add weekly indicators
    weekly_df = calculate_weekly_indicators(weekly_df)

    # Use in pattern detection
    pattern = find_sleepers_v17(weekly_df, candle_frequency='weekly')
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ================================================================================
# AGGREGATION RULES
# ================================================================================

# Standard OHLCV aggregation for weekly candles
WEEKLY_OHLCV = {
    'open': 'first',   # Monday's open (or first trading day of week)
    'high': 'max',     # Week's high
    'low': 'min',      # Week's low
    'close': 'last',   # Friday's close (or last trading day of week)
    'volume': 'sum'    # Total weekly volume
}

# Week-ending rule: 'W-FRI' means weeks end on Friday (standard trading week)
DEFAULT_RESAMPLE_RULE = 'W-FRI'


# ================================================================================
# WEEKLY AGGREGATION
# ================================================================================

def resample_to_weekly(
    daily_df: pd.DataFrame,
    preserve_last_partial: bool = False,
    resample_rule: str = DEFAULT_RESAMPLE_RULE
) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, pd.Timestamp]]:
    """
    Aggregate daily OHLCV data to weekly candles.

    Uses 'W-FRI' offset (weeks ending Friday) for trading week alignment.
    This ensures weekly candles align with the standard trading week.

    Args:
        daily_df: DataFrame with daily OHLCV data. Must have DatetimeIndex
                  and columns: open, high, low, close, volume (lowercase).
        preserve_last_partial: If True, keeps the last partial week.
                               If False (default), drops incomplete weeks.
        resample_rule: Pandas resample rule (default: 'W-FRI' for trading weeks)

    Returns:
        Tuple of:
        - weekly_df: Weekly OHLCV DataFrame indexed by week-end date
        - week_to_daily_map: Dict mapping week_end_date -> last_daily_trading_date
                            This is used for outcome labeling on daily prices.

    Example:
        >>> daily_df = pd.read_parquet('AAPL.parquet')
        >>> weekly_df, mapping = resample_to_weekly(daily_df)
        >>> print(f'Daily rows: {len(daily_df)}, Weekly rows: {len(weekly_df)}')
        >>> # Pattern end_date_daily = mapping[pattern_end_date] for outcome labeling
    """
    if daily_df.empty:
        logger.warning("Empty DataFrame provided to resample_to_weekly")
        return pd.DataFrame(), {}

    # Ensure DatetimeIndex
    df = daily_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'date' column")

    # Sort by date (required for proper resampling)
    df = df.sort_index()

    # Ensure lowercase column names
    df.columns = df.columns.str.lower()

    # Validate required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Resample to weekly
    weekly_df = df[required_cols].resample(resample_rule).agg(WEEKLY_OHLCV)

    # Drop rows with NaN (incomplete weeks at start/end)
    if not preserve_last_partial:
        weekly_df = weekly_df.dropna()

    # Build date mapping: week_end_date -> last_actual_trading_date in that week
    # This is critical for outcome labeling which uses daily prices
    week_to_daily_map = {}

    for week_end in weekly_df.index:
        # Find all trading days in this week
        week_start = week_end - pd.Timedelta(days=6)
        mask = (df.index >= week_start) & (df.index <= week_end)

        if mask.any():
            # Last actual trading day in this week
            last_trading_day = df.index[mask][-1]
            week_to_daily_map[week_end] = last_trading_day

    logger.info(
        f"Resampled {len(daily_df)} daily rows to {len(weekly_df)} weekly rows "
        f"({len(week_to_daily_map)} weeks with mapping)"
    )

    return weekly_df, week_to_daily_map


# ================================================================================
# WEEKLY INDICATORS
# ================================================================================

def calculate_weekly_indicators(
    weekly_df: pd.DataFrame,
    bbw_period: int = 20,
    sma_periods: Tuple[int, int] = (20, 50),  # Changed: 20/50 weeks (~5mo/1yr) instead of 50/200
    volume_ratio_period: int = 20,
    range_ratio_period: int = 20
) -> pd.DataFrame:
    """
    Calculate technical indicators on weekly candles.

    These indicators are the weekly equivalents of the daily indicators
    used in pattern detection, but calculated on weekly timeframe.

    IMPORTANT: Uses shorter SMA periods than daily mode because:
    - Daily mode: SMA_50/200 (50/200 days = ~2.5mo/10mo)
    - Weekly mode: SMA_20/50 (20/50 weeks = ~5mo/1yr)
    This keeps warmup requirements reasonable while preserving trend info.

    Args:
        weekly_df: Weekly OHLCV DataFrame from resample_to_weekly()
        bbw_period: Period for Bollinger Band Width (default: 20 weeks)
        sma_periods: Periods for SMA calculations (default: 20, 50 weeks)
        volume_ratio_period: Period for volume ratio (default: 20 weeks)
        range_ratio_period: Period for range ratio (default: 20 weeks)

    Returns:
        DataFrame with added indicator columns:
        - bbw_20: Bollinger Band Width (20-week)
        - sma_50: 50-week Simple Moving Average (medium-term)
        - sma_200: Uses sma_50 for compatibility (long-term proxy)
        - volume_ratio_20: Current week volume vs 20-week average
        - range_ratio_20: Current week range vs 20-week average

    Note:
        Weekly SMA calculations use shorter periods to reduce warmup
        requirements while maintaining meaningful trend signals.
    """
    if weekly_df.empty:
        return weekly_df

    df = weekly_df.copy()

    # Ensure lowercase columns
    df.columns = df.columns.str.lower()

    # -------------------------------------------------------------------------
    # BBW (Bollinger Band Width) - 20-week
    # Measures volatility contraction (key consolidation indicator)
    # -------------------------------------------------------------------------
    sma = df['close'].rolling(bbw_period).mean()
    std = df['close'].rolling(bbw_period).std()
    # BBW = (2 * std * 2) / sma * 100 = 4 * std / sma * 100
    # This represents the percentage width of Bollinger Bands
    df['bbw_20'] = ((std * 4) / sma) * 100

    # -------------------------------------------------------------------------
    # SMAs for Market Structure Check
    # Weekly mode uses shorter periods: sma_20/sma_50 instead of sma_50/sma_200
    # Used for: close > sma_short > sma_long (healthy uptrend)
    # -------------------------------------------------------------------------
    sma_short, sma_long = sma_periods
    # Use standard column names for compatibility with pattern_scanner
    df['sma_50'] = df['close'].rolling(sma_short).mean()   # 20-week in weekly mode
    df['sma_200'] = df['close'].rolling(sma_long).mean()   # 50-week in weekly mode

    # -------------------------------------------------------------------------
    # Volume Ratio (current week vs 20-week average)
    # Low volume during consolidation = accumulation in progress
    # -------------------------------------------------------------------------
    volume_avg = df['volume'].rolling(volume_ratio_period).mean()
    df['volume_ratio_20'] = df['volume'] / volume_avg

    # -------------------------------------------------------------------------
    # Range Ratio (current week range vs 20-week average)
    # Compressed range = volatility contraction
    # -------------------------------------------------------------------------
    weekly_range = df['high'] - df['low']
    range_avg = weekly_range.rolling(range_ratio_period).mean()
    df['range_ratio_20'] = weekly_range / range_avg

    # -------------------------------------------------------------------------
    # ADX (Average Directional Index) - Optional
    # Note: ADX on weekly is less common, but we include for compatibility
    # -------------------------------------------------------------------------
    # Calculate ADX if needed (simplified weekly version)
    if 'adx' not in df.columns:
        df['adx'] = _calculate_weekly_adx(df, period=14)

    # -------------------------------------------------------------------------
    # Dollar Volume (for liquidity checks)
    # -------------------------------------------------------------------------
    if 'DollarVol' not in df.columns:
        df['DollarVol'] = df['close'] * df['volume']

    logger.debug(
        f"Calculated weekly indicators: BBW, SMA_50, SMA_200, "
        f"volume_ratio, range_ratio (period={bbw_period})"
    )

    return df


def _calculate_weekly_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX) on weekly data.

    ADX measures trend strength. Values below 25 indicate low trending
    (potential consolidation), while values above 25 indicate trending.

    Args:
        df: Weekly OHLCV DataFrame
        period: ADX calculation period (default: 14 weeks)

    Returns:
        Series of ADX values
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed averages
    atr = pd.Series(tr).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()

    return adx


# ================================================================================
# PATTERN DATE MAPPING
# ================================================================================

def get_daily_date_for_pattern(
    pattern_end_date: pd.Timestamp,
    week_to_daily_map: Dict[pd.Timestamp, pd.Timestamp]
) -> Optional[pd.Timestamp]:
    """
    Get the actual daily trading date for a pattern detected on weekly candles.

    When a pattern is detected using weekly qualification, we need to map
    the pattern's week-end date back to the actual last trading day of
    that week for outcome labeling purposes.

    Args:
        pattern_end_date: Pattern end date from weekly detection
        week_to_daily_map: Mapping from resample_to_weekly()

    Returns:
        The last daily trading date in that week, or None if not found

    Example:
        >>> pattern_end = pd.Timestamp('2024-01-19')  # Friday week-end
        >>> daily_date = get_daily_date_for_pattern(pattern_end, mapping)
        >>> print(daily_date)  # '2024-01-19' if Friday traded, else Thursday etc.
    """
    if pattern_end_date in week_to_daily_map:
        return week_to_daily_map[pattern_end_date]

    # Try to find nearest date if exact match not found
    if week_to_daily_map:
        dates = sorted(week_to_daily_map.keys())
        for date in dates:
            if abs((date - pattern_end_date).days) <= 2:
                logger.debug(
                    f"Using nearest date mapping: {pattern_end_date} -> {date} -> "
                    f"{week_to_daily_map[date]}"
                )
                return week_to_daily_map[date]

    logger.warning(f"No daily date mapping found for pattern end date: {pattern_end_date}")
    return None


def enrich_pattern_with_daily_date(
    pattern: Dict,
    week_to_daily_map: Dict[pd.Timestamp, pd.Timestamp]
) -> Dict:
    """
    Add end_date_daily field to a pattern detected with weekly qualification.

    The end_date_daily is used by 00b_label_outcomes.py for outcome labeling
    on daily price data.

    Args:
        pattern: Pattern dictionary from weekly detection
        week_to_daily_map: Mapping from resample_to_weekly()

    Returns:
        Pattern dictionary with added 'end_date_daily' and 'qualification_frequency'
    """
    pattern = pattern.copy()

    # Get the pattern's weekly end date
    pattern_end = pattern.get('end_date')
    if pattern_end is not None:
        if not isinstance(pattern_end, pd.Timestamp):
            pattern_end = pd.to_datetime(pattern_end)

        # Map to daily date
        daily_date = get_daily_date_for_pattern(pattern_end, week_to_daily_map)
        pattern['end_date_daily'] = daily_date

    # Mark as weekly qualification
    pattern['qualification_frequency'] = 'weekly'

    return pattern


# ================================================================================
# VALIDATION UTILITIES
# ================================================================================

def validate_weekly_data_requirements(
    weekly_df: pd.DataFrame,
    min_weeks: int = 80  # ~1.5 years: 50 weeks for SMA_50 + 30 weeks min_window
) -> Tuple[bool, str]:
    """
    Validate that weekly data meets minimum requirements for pattern detection.

    Weekly mode data requirements (updated Jan 2026):
    - SMA_50 (long-term) needs 50 weeks (~1 year)
    - Plus min_window for pattern detection (30 weeks)
    - Plus buffer for forward tracking (20 weeks)

    Args:
        weekly_df: Weekly OHLCV DataFrame
        min_weeks: Minimum weeks required (default: 80 = 50 + 30)

    Returns:
        Tuple of (is_valid, message)
    """
    if weekly_df.empty:
        return False, "Empty weekly DataFrame"

    num_weeks = len(weekly_df)
    if num_weeks < min_weeks:
        return False, (
            f"Insufficient weekly data: {num_weeks} weeks. "
            f"Weekly mode requires at least {min_weeks} weeks (~{min_weeks // 52} years) "
            f"for indicator calculation and pattern detection."
        )

    # Check for gaps larger than 2 weeks (data quality)
    if isinstance(weekly_df.index, pd.DatetimeIndex):
        gaps = weekly_df.index.to_series().diff()
        max_gap = gaps.max()
        if pd.notna(max_gap) and max_gap > pd.Timedelta(days=21):
            return False, f"Data gap too large: {max_gap.days} days between weeks"

    return True, f"Weekly data validated: {num_weeks} weeks"


def estimate_pattern_reduction(
    daily_pattern_count: int,
    avg_daily_pattern_duration: float = 20
) -> Tuple[int, int]:
    """
    Estimate pattern count reduction when switching from daily to weekly.

    Weekly qualification (10 weeks = ~50 trading days) is ~5x stricter
    than daily qualification (10 days), so we expect fewer but higher
    quality patterns.

    Args:
        daily_pattern_count: Number of patterns detected with daily mode
        avg_daily_pattern_duration: Average pattern duration in days

    Returns:
        Tuple of (estimated_weekly_count_low, estimated_weekly_count_high)

    Note:
        This is a rough estimate. Actual reduction depends on:
        - Market conditions during the period
        - Stock volatility characteristics
        - Data quality
    """
    # Empirical reduction factors from backtesting
    # Weekly mode typically finds 15-35% of daily patterns
    reduction_low = 0.15
    reduction_high = 0.35

    low_estimate = int(daily_pattern_count * reduction_low)
    high_estimate = int(daily_pattern_count * reduction_high)

    return low_estimate, high_estimate
