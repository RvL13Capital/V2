"""
Microstructure-Aware Sleeper Scanner V17

Designed to differentiate between "Messy Volatility" (Bad) and
"Thin Liquidity Volatility" (Acceptable).

Identifies consolidation patterns with optional accumulation detection.

Now supports Dynamic Asset Profiles based on market cap category.

V17.1 (Jan 2026): Added check_thresholds() with:
  - BBW Z-score based tightness (6-month history)
  - Market structure check (close > sma_50 > sma_200) - REPLACES ADX
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple

from config import ENABLE_ACCUMULATION_DETECTION

# Constants for relative threshold checking
BBW_HISTORY_DAYS = 126  # 6 months of trading days for BBW Z-score calculation

# Market Cap-Based Width Limits (Asset Physics) - WICK TIGHTNESS
# UPDATED: 1.5x multiplier based on grid search empirical validation (2025-12-16)
# Small-cap natural consolidation width is 25-30%, not 10-15%
# Tighter limits led to 75.9% danger rate, looser limits reduced to 70.4%
# These limits now apply to HIGH/LOW ranges (wick tightness) - allows intraday volatility
WIDTH_LIMITS = {
    'nano_cap':  0.60,   # 60% (was 40%) - Extreme volatility (lottery tickets)
    'micro_cap': 0.45,   # 45% (was 30%) - High volatility but showing structure
    'small_cap': 0.30,   # 30% (was 20%) - Natural consolidation width 25-30%
    'mid_cap':   0.225,  # 22.5% (was 15%) - Standard behavior, real consolidation
    'large_cap': 0.15,   # 15% (was 10%) - Efficient markets, real coiling
    'mega_cap':  0.075,  # 7.5% (was 5%) - Exceptional tightness (FAANG)
    'default':   0.225   # Fallback to mid-cap baseline (22.5%, was 15%)
}

# Weekly mode multiplier for WIDTH_LIMITS
# Weekly candles have ~2x wider wicks due to accumulating 5 days of price action
WEEKLY_WIDTH_MULTIPLIER = 1.5

# Body Tightness: Strict threshold for close price consolidation (ACCUMULATION SIGNATURE)
# This captures "strong hands holding tight" while allowing intraday volatility
# Pre-breakout micro-caps show tight closes (accumulation) + wide intraday ranges (shake out weak hands)
BODY_TIGHTNESS_MAX = 0.15  # 15% maximum close-to-close range (strict) for DAILY

# Weekly mode has naturally wider candle ranges (5 days of movement per candle)
# A 10-week consolidation on weekly candles shows accumulation over ~2.5 months
# We relax the threshold to 40% for weekly mode (~2.7x daily)
# This captures longer-term consolidations with wider natural ranges
BODY_TIGHTNESS_MAX_WEEKLY = 0.40  # 40% for weekly candles


def check_thresholds(
    df: pd.DataFrame,
    tightness_zscore: Optional[float] = None,
    candle_frequency: str = 'daily'
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if current conditions meet relative thresholds.

    Uses:
    - BBW Z-score (6-month history) instead of static percentile
    - Market structure (close > sma_50 > sma_200) instead of ADX

    This function replaces ADX-based trend filtering with a market structure
    check that verifies the stock is in a healthy uptrend configuration.

    Args:
        df: DataFrame with OHLCV data. Must have 'close' column.
            Optionally 'bbw_20', 'sma_50', 'sma_200' for enhanced filtering.
        tightness_zscore: Max Z-Score threshold for BBW (e.g., -1.0 means
                         current BBW must be at least 1 std dev below the
                         6-month mean). If None, BBW check is skipped.
        candle_frequency: 'daily' or 'weekly' - affects minimum data requirements

    Returns:
        Tuple of:
        - passed: bool - True if all enabled checks pass
        - metrics: dict with computed values:
            - bbw_zscore: float or None - Z-score of current BBW vs 6-month history
            - market_structure_ok: bool or None - True if close > sma_50 > sma_200
    """
    metrics = {'bbw_zscore': None, 'market_structure_ok': None}

    # Need minimum data for reliable calculations
    # Weekly mode uses smaller windows (30 weeks vs 60 days)
    min_data = 20 if candle_frequency == 'weekly' else 60
    if len(df) < min_data:
        return False, metrics

    # Get current values (last row of window)
    current_close = df['close'].iloc[-1]

    # ==========================================================================
    # CHECK 1: BBW Z-score (6-month history)
    # Measures if volatility contraction is unusual FOR THIS STOCK
    # ==========================================================================
    if tightness_zscore is not None and 'bbw_20' in df.columns:
        current_bbw = df['bbw_20'].iloc[-1]

        if pd.notna(current_bbw):
            # Use up to 6 months of history, excluding current value
            lookback = min(BBW_HISTORY_DAYS, len(df) - 1)

            if lookback >= 20:  # Need minimum history for meaningful stats
                hist_bbw = df['bbw_20'].iloc[-lookback-1:-1]
                hist_bbw = hist_bbw.dropna()

                if len(hist_bbw) >= 20:
                    mean_bbw = hist_bbw.mean()
                    std_bbw = hist_bbw.std()

                    if std_bbw > 1e-6:  # Avoid division by zero
                        bbw_zscore = (current_bbw - mean_bbw) / std_bbw
                        metrics['bbw_zscore'] = round(bbw_zscore, 4)

                        # Z-score > threshold means NOT tight enough
                        # e.g., tightness_zscore=-1.0 requires BBW to be 1 std below mean
                        if bbw_zscore > tightness_zscore:
                            return False, metrics

    # ==========================================================================
    # CHECK 2: Market Structure (REPLACES ADX)
    # Requires healthy uptrend: close > sma_50 > sma_200
    # This ensures we're looking at consolidation in an UPTREND, not a downtrend
    # ==========================================================================
    sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else None
    sma_200 = df['sma_200'].iloc[-1] if 'sma_200' in df.columns else None

    if sma_50 is not None and sma_200 is not None:
        if pd.notna(sma_50) and pd.notna(sma_200):
            market_structure_ok = (current_close > sma_50 > sma_200)
            metrics['market_structure_ok'] = market_structure_ok

            if not market_structure_ok:
                return False, metrics

    return True, metrics


def find_sleepers_v17(
    df: pd.DataFrame,
    min_liquidity_dollar: float = 50000,
    precomputed: bool = False,
    market_cap_category: Optional[str] = None,  # Dynamic asset profile
    # Adaptive thresholds (Jan 2026)
    tightness_zscore: Optional[float] = None,
    min_float_turnover: Optional[float] = None,
    shares_outstanding: Optional[float] = None,  # Required for float turnover
    # Weekly qualification mode (Jan 2026)
    candle_frequency: str = 'daily'  # 'daily' or 'weekly'
) -> Optional[Dict[str, Any]]:
    """
    Microstructure-aware consolidation pattern scanner with DUAL TIGHTNESS METRICS.

    Scans for consolidation patterns using two distinct tightness checks:
    1. WICK TIGHTNESS (Loose): 40-50% allowance using high/low ranges
       - Captures intraday volatility (weak hands being shaken out)
       - Market cap-based WIDTH_LIMITS (nano: 60%, micro: 45%, small: 30%)
    2. BODY TIGHTNESS (Strict): <15% range using close prices only
       - Captures accumulation (strong hands holding tight)
       - Fixed threshold: BODY_TIGHTNESS_MAX = 15%

    This dual approach identifies pre-breakout micro-caps showing the accumulation
    signature: tight closes (accumulation) + wide intraday ranges (shake out weak hands).

    OPTIMIZED: Supports pre-computed features to avoid redundant calculations
    when called in loops (2-3x speedup when precomputed=True).

    Args:
        df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
            Must also have 'Ticker' column if you want ticker in output
        min_liquidity_dollar: Minimum average dollar volume threshold
        precomputed: If True, assumes DollarVol, Vol_50MA, Pct_Change already computed
        market_cap_category: Market cap category (e.g., 'small_cap', 'nano_cap')
                            If None, falls back to price-based heuristic
        tightness_zscore: Max Z-Score for body tightness (e.g., -1.0 = 1 std dev tighter than avg).
                         When set, adds adaptive body tightness check based on stock's own history.
        min_float_turnover: Minimum 20d float turnover (e.g., 0.10 = 10% of float traded).
                           Requires accumulation activity to detect pattern.
        shares_outstanding: Number of shares outstanding (required for float turnover calculation)
        candle_frequency: Candle frequency for qualification ('daily' or 'weekly').
                         When 'weekly', patterns are detected on weekly candles
                         for longer-term consolidation (10 weeks vs 10 days).
                         Weekly mode requires ~4 years of data for SMA_200.

    Returns:
        Dictionary with pattern details if found, None otherwise

    Output Dictionary:
        - ticker: Stock ticker symbol
        - status: 'CONSOLIDATING' (always, accumulation doesn't affect status)
        - current_price: Current closing price
        - upper_lid: Upper boundary (95th percentile of CLOSE prices - body)
        - trigger: Breakout trigger price (upper_lid * 1.05)
        - box_width: Body width as percentage (close price range - strict)
        - wick_width: Wick width as percentage (high/low range - loose)
        - allowed_width: Maximum allowed wick width for this asset class
        - body_tightness_max: Maximum allowed body width (15% fixed)
        - is_thin_stock: Boolean indicating thin liquidity characteristics
        - accumulation_count: Number of accumulation days (0 if disabled)
        - liquidity: Average dollar volume
    """

    # 0. DATA PREP & SAFETY
    # Ensure we have enough data
    # Weekly mode: 30 periods (indicators pre-computed with shorter SMAs 20/50)
    # Daily mode: 100 periods (for SMA_50/200 computation)
    # Note: Weekly uses sliding window approach with pre-computed indicators
    min_periods = 30 if candle_frequency == 'weekly' else 100
    if len(df) < min_periods:
        return None

    df = df.copy()  # Work on a copy to avoid SettingWithCopy warnings

    # OPTIMIZATION: Only compute if not already present (2-3x speedup in loops)
    if not precomputed or 'DollarVol' not in df.columns:
        df['DollarVol'] = df['close'] * df['volume']
    if not precomputed or 'Vol_50MA' not in df.columns:
        df['Vol_50MA'] = df['volume'].rolling(50).mean()
    if not precomputed or 'Pct_Change' not in df.columns:
        df['Pct_Change'] = df['close'].pct_change()

    # 1. LIQUIDITY GATE (The Bouncer)
    avg_dollar_vol = df['DollarVol'].rolling(20).mean().iloc[-1]
    current_price = df['close'].iloc[-1]

    # REFINEMENT: "Ghost" Check
    # If more than 20% of the last 20 days had ZERO volume, it's a ghost town.
    # This kills stocks that trade by appointment only.
    recent_volume = df['volume'].iloc[-20:]
    zero_volume_days = (recent_volume == 0).sum()

    if (pd.isna(avg_dollar_vol) or
        avg_dollar_vol < min_liquidity_dollar or
        zero_volume_days > 4):  # Max 4 days of 0 vol in a month
        return None

    # 1.5 RELATIVE THRESHOLD CHECK (Jan 2026 - Replaces ADX)
    # Uses BBW Z-score and market structure (close > sma_50 > sma_200)
    # This check is applied BEFORE tightness checks but AFTER liquidity gate
    threshold_metrics = {'bbw_zscore': None, 'market_structure_ok': None}

    # Compute SMAs if not present (needed for market structure check)
    if 'sma_50' not in df.columns:
        df['sma_50'] = df['close'].rolling(50).mean()
    if 'sma_200' not in df.columns:
        df['sma_200'] = df['close'].rolling(200).mean()

    # Run relative threshold check
    passed, threshold_metrics = check_thresholds(df, tightness_zscore, candle_frequency)
    if not passed:
        return None

    # 2. DEFINE WINDOWS
    lookback = 60
    recent = df.iloc[-lookback:]

    # 3. DUAL TIGHTNESS METRICS (The Accumulation Signature)
    #
    # For WEEKLY mode: Use only last 3 weeks for boundary calculation
    # This focuses on recent price action rather than full 10-week qualification period
    # Rationale: Breakout traders care about recent support/resistance, not historical extremes
    if candle_frequency == 'weekly':
        from config.constants import WEEKLY_BOUNDARY_WEEKS
        boundary_window = df.iloc[-WEEKLY_BOUNDARY_WEEKS:]  # Last 3 weeks
    else:
        boundary_window = recent  # Full lookback for daily

    # A. WICK TIGHTNESS (Loose) - Captures intraday volatility
    #    Uses high/low to allow for shaking out weak hands
    #    For weekly: boundaries from last 3 weeks, tightness check from full window
    wick_upper = boundary_window['high'].max()  # Use max/min for weekly boundaries
    wick_lower = boundary_window['low'].min()
    wick_height_pct = (wick_upper - wick_lower) / wick_lower if wick_lower > 0 else 0

    # B. BODY TIGHTNESS (Strict) - Captures accumulation
    #    Uses close prices to detect tight holding patterns
    #    For weekly: use actual high/low of boundary window (not percentile)
    if candle_frequency == 'weekly':
        # Weekly mode: Use actual H/L of last 3 weeks as boundaries
        body_upper = boundary_window['high'].max()
        body_lower = boundary_window['low'].min()
    else:
        # Daily mode: Use percentiles of close prices (original behavior)
        body_upper = np.percentile(recent['close'], 95)
        body_lower = np.percentile(recent['close'], 10)
    body_height_pct = (body_upper - body_lower) / body_lower if body_lower > 0 else 0

    # 4. INTELLIGENT TIGHTNESS (Dynamic Asset Physics)

    # A. Wick Tightness Check (Loose: 40-50% depending on asset class)
    if market_cap_category is not None:
        # Use market cap profile for max wick width
        wick_allowed = WIDTH_LIMITS.get(market_cap_category, WIDTH_LIMITS['default'])
    else:
        # FALLBACK: Price-based heuristic (when market cap unavailable)
        # Set to 40-50% range for micro/small caps
        if current_price < 2.00:
            wick_allowed = 0.50  # 50% for nano-caps
        elif current_price < 10.00:
            wick_allowed = 0.45  # 45% for micro/small caps
        else:
            wick_allowed = 0.40  # 40% for larger

    # Weekly mode: Apply multiplier to wick width limits
    # Weekly candles have naturally wider wicks due to 5 days of price action
    if candle_frequency == 'weekly':
        wick_allowed = wick_allowed * WEEKLY_WIDTH_MULTIPLIER

    # REFINEMENT: Efficiency Ratio (The "Thinness" Buffer)
    # We calculate the average candle range (High-Low) relative to price.
    # If a stock has huge candle wicks but closes tight, it's "Thin", not "Bad".
    # We give "Thin" stocks a 5% bonus to the allowed WICK width.
    avg_candle_size = (df['high'] - df['low']).rolling(20).mean().iloc[-1] / current_price
    # Weekly mode: Use 10% threshold (vs 4% for daily) due to wider candle ranges
    thin_threshold = 0.10 if candle_frequency == 'weekly' else 0.04
    is_thin = avg_candle_size > thin_threshold

    final_wick_allowed = wick_allowed + (0.05 if is_thin else 0.0)

    # CRITICAL FILTER 1: Wick tightness (allows intraday volatility)
    # Rejects if HIGH/LOW range is too wide (extreme volatility)
    if wick_height_pct > final_wick_allowed:
        return None

    # CRITICAL FILTER 2: Body tightness (strict accumulation check)
    # Rejects if CLOSE price range is too wide (no accumulation)
    # Weekly mode uses relaxed threshold (30%) due to natural weekly candle width
    body_tightness_threshold = BODY_TIGHTNESS_MAX_WEEKLY if candle_frequency == 'weekly' else BODY_TIGHTNESS_MAX
    if body_height_pct > body_tightness_threshold:
        return None

    # ==========================================================================
    # ADAPTIVE FILTERS (Jan 2026)
    # ==========================================================================

    # ADAPTIVE FILTER A: Z-Score Tightness
    # Measures if current body tightness is unusually tight FOR THIS STOCK
    # Z-score < -1.0 means body width is 1 std dev tighter than historical average
    body_width_zscore = None
    if tightness_zscore is not None:
        # Calculate historical body width distribution (last 252 trading days)
        lookback_body = min(252, len(df) - 60)
        if lookback_body > 20:
            historical_body_widths = []
            for i in range(lookback_body):
                # Handle i=0 case separately to avoid :-None slice
                if i == 0:
                    hist_window = df.iloc[-60:]
                else:
                    hist_window = df.iloc[-(60 + i):-i]
                if len(hist_window) >= 60:
                    hist_body_upper = np.percentile(hist_window['close'], 95)
                    hist_body_lower = np.percentile(hist_window['close'], 10)
                    hist_body_width = (hist_body_upper - hist_body_lower) / (hist_body_lower + 1e-9)
                    historical_body_widths.append(hist_body_width)

            if len(historical_body_widths) >= 20:
                hist_mean = np.mean(historical_body_widths)
                hist_std = np.std(historical_body_widths)
                if hist_std > 1e-6:
                    body_width_zscore = (body_height_pct - hist_mean) / hist_std

                    # Reject if not tight enough relative to stock's own history
                    if body_width_zscore > tightness_zscore:
                        return None

    # ADAPTIVE FILTER B: Float Turnover
    # Measures accumulated trading activity as % of float
    # float_turnover = sum(volume_20d) / shares_outstanding
    float_turnover = None
    if min_float_turnover is not None and shares_outstanding is not None and shares_outstanding > 0:
        vol_20d_sum = df['volume'].iloc[-20:].sum()
        float_turnover = vol_20d_sum / shares_outstanding

        # Reject if insufficient accumulation activity
        if float_turnover < min_float_turnover:
            return None

    # ==========================================================================
    # END ADAPTIVE FILTERS
    # ==========================================================================

    # 5. ACCUMULATION (The Hidden Hand) - OPTIONAL
    accumulation_count = 0

    if ENABLE_ACCUMULATION_DETECTION:
        # Calculate accumulation days: huge volume spikes with tiny price moves
        acc_window = df.iloc[-90:]
        is_huge_vol = acc_window['volume'] > (acc_window['Vol_50MA'] * 3)
        is_tiny_move = abs(acc_window['Pct_Change']) < 0.025
        is_active = acc_window['volume'] > 0

        accumulation_days = acc_window[is_huge_vol & is_tiny_move & is_active]
        accumulation_count = len(accumulation_days)

    # 6. TRIGGER & STATUS
    trigger_price = body_upper * 1.05  # Use body boundary (accumulation zone)

    # Status is always CONSOLIDATING (accumulation doesn't affect status)
    status = 'CONSOLIDATING'

    return {
        'ticker': df['Ticker'].iloc[0] if 'Ticker' in df.columns else 'Unknown',
        'status': status,
        'current_price': current_price,
        'upper_lid': body_upper,  # Body boundary (close-based, strict)
        'trigger': trigger_price,
        'box_width': round(body_height_pct, 4),  # Body width (close range - strict)
        'wick_width': round(wick_height_pct, 4),  # NEW: Wick width (high/low range - loose)
        'allowed_width': round(final_wick_allowed, 4),  # Max allowed wick width
        'body_tightness_max': body_tightness_threshold,  # Max allowed body width (15% daily / 30% weekly)
        'is_thin_stock': is_thin,  # Helps you know execution will be tricky
        'accumulation_count': accumulation_count,
        'liquidity': int(avg_dollar_vol),
        'market_cap_category': market_cap_category,  # Asset physics profile used
        'base_allowed_width': round(wick_allowed, 4),  # Wick width before thin buffer
        # Adaptive metrics (Jan 2026)
        'body_width_zscore': round(body_width_zscore, 4) if body_width_zscore is not None else None,
        'float_turnover': round(float_turnover, 4) if float_turnover is not None else None,
        # Relative threshold metrics (Jan 2026 - check_thresholds)
        'bbw_zscore': threshold_metrics.get('bbw_zscore'),
        'market_structure_ok': threshold_metrics.get('market_structure_ok'),
        # Weekly qualification mode (Jan 2026)
        'candle_frequency': candle_frequency
    }


def calculate_outcome_class(max_gain_pct: float, breakdown: bool = False) -> int:
    """
    Calculate outcome class (K0-K5) from gain percentage.

    Maps gain percentages to TRANS outcome classification system:
    - K0: <5% gain (stagnant)
    - K1: 5-15% gain (minimal)
    - K2: 15-35% gain (quality)
    - K3: 35-75% gain (strong)
    - K4: >75% gain (exceptional)
    - K5: Breakdown/failed pattern

    Args:
        max_gain_pct: Maximum gain percentage achieved
        breakdown: Whether pattern broke down (failed)

    Returns:
        Outcome class (0-5)
    """
    if breakdown:
        return 5  # K5 - Failed

    if max_gain_pct < 5.0:
        return 0  # K0 - Stagnant
    elif max_gain_pct < 15.0:
        return 1  # K1 - Minimal
    elif max_gain_pct < 35.0:
        return 2  # K2 - Quality
    elif max_gain_pct < 75.0:
        return 3  # K3 - Strong
    else:
        return 4  # K4 - Exceptional
