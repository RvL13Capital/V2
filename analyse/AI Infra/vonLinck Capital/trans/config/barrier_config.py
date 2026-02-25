"""
Dynamic Barrier Configuration for Triple Barrier Labeling
==========================================================

Adjusts upper and lower barriers based on:
1. Market Cap Tier (at time of pattern) - micro-caps need wider stops/targets
2. Market Regime (bullish/bearish) - affects risk tolerance and target amplitude

Philosophy:
- MICRO-CAPS: More volatile → wider stops to avoid noise, higher targets (40%+ potential)
- SMALL-CAPS: Medium volatility → balanced stops/targets
- MID/LARGE-CAPS: Lower volatility → tighter stops, smaller but more consistent targets

- BULLISH REGIME: Risk-on → can afford tighter stops, aim for bigger targets
- BEARISH REGIME: Risk-off → wider stops (more volatility), take profits faster
- CRISIS: Capital preservation → very wide stops, quick exits

Usage:
    from config.barrier_config import get_dynamic_barriers, MarketCapTier, MarketRegimeState

    barriers = get_dynamic_barriers(
        market_cap_tier=MarketCapTier.MICRO,
        regime=MarketRegimeState.BULLISH
    )

    # barriers = {'upper_barrier_pct': 0.08, 'lower_barrier_pct': 0.04, ...}
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


# =============================================================================
# MARKET CAP TIERS
# =============================================================================

class MarketCapTier(Enum):
    """Market capitalization tiers based on dollar volume or market cap."""
    NANO = "nano"       # < $50M market cap or < $100K ADV
    MICRO = "micro"     # $50M - $300M market cap or $100K - $500K ADV
    SMALL = "small"     # $300M - $2B market cap or $500K - $5M ADV
    MID = "mid"         # $2B - $10B market cap or $5M - $50M ADV
    LARGE = "large"     # > $10B market cap or > $50M ADV
    UNKNOWN = "unknown" # Fallback when data unavailable


# ADV-based thresholds (Average Daily Dollar Volume)
# Used when market cap API unavailable (common for EU stocks)
ADV_TIER_THRESHOLDS = {
    'nano': 100_000,      # < $100K ADV
    'micro': 500_000,     # $100K - $500K ADV
    'small': 5_000_000,   # $500K - $5M ADV
    'mid': 50_000_000,    # $5M - $50M ADV
    # Above $50M = large
}

# Market cap thresholds (when available)
MARKET_CAP_TIER_THRESHOLDS = {
    'nano': 50_000_000,       # < $50M
    'micro': 300_000_000,     # $50M - $300M
    'small': 2_000_000_000,   # $300M - $2B
    'mid': 10_000_000_000,    # $2B - $10B
    # Above $10B = large
}


# =============================================================================
# MARKET REGIME STATES
# =============================================================================

class MarketRegimeState(Enum):
    """Simplified market regime for barrier adjustment."""
    BULLISH = "bullish"         # Risk-on, trending up
    NEUTRAL = "neutral"         # Sideways, normal volatility
    BEARISH = "bearish"         # Risk-off, trending down
    HIGH_VOLATILITY = "high_vol"  # Elevated VIX (18-25)
    CRISIS = "crisis"           # VIX > 25, defensive mode


# =============================================================================
# BARRIER CONFIGURATION MATRICES
# =============================================================================

@dataclass
class BarrierThresholds:
    """Barrier thresholds for a specific market cap + regime combination."""
    upper_barrier_pct: float    # Profit-taking threshold (e.g., 0.05 = +5%)
    lower_barrier_pct: float    # Stop-loss threshold (e.g., 0.03 = -3%)
    rationale: str              # Explanation for these values

    # Optional adjustments
    min_holding_days: int = 5   # Minimum days before considering exit
    volatility_scalar: float = 1.0  # Multiplier for vol-scaled barriers


# BASE BARRIERS BY MARKET CAP TIER
# These are the starting points, modified by regime
BASE_BARRIERS_BY_CAP = {
    MarketCapTier.NANO: BarrierThresholds(
        upper_barrier_pct=0.12,   # +12% target (nano-caps can double)
        lower_barrier_pct=0.06,   # -6% stop (very volatile, need room)
        rationale="Nano-caps: Extreme volatility, need wide stops. High reward potential.",
        min_holding_days=3,
        volatility_scalar=1.3
    ),
    MarketCapTier.MICRO: BarrierThresholds(
        upper_barrier_pct=0.08,   # +8% target (micro-caps have 50%+ potential)
        lower_barrier_pct=0.04,   # -4% stop (volatile but not extreme)
        rationale="Micro-caps: High volatility, meaningful stops. Targets 40%+ moves.",
        min_holding_days=5,
        volatility_scalar=1.2
    ),
    MarketCapTier.SMALL: BarrierThresholds(
        upper_barrier_pct=0.05,   # +5% target (small-caps: 20-30% potential)
        lower_barrier_pct=0.025,  # -2.5% stop (medium volatility)
        rationale="Small-caps: Moderate volatility. Balanced risk/reward.",
        min_holding_days=7,
        volatility_scalar=1.0
    ),
    MarketCapTier.MID: BarrierThresholds(
        upper_barrier_pct=0.04,   # +4% target (mid-caps: steady gains)
        lower_barrier_pct=0.02,   # -2% stop (lower volatility)
        rationale="Mid-caps: Lower volatility, tighter execution possible.",
        min_holding_days=10,
        volatility_scalar=0.9
    ),
    MarketCapTier.LARGE: BarrierThresholds(
        upper_barrier_pct=0.03,   # +3% target (large-caps: slow and steady)
        lower_barrier_pct=0.015,  # -1.5% stop (minimal noise)
        rationale="Large-caps: Lowest volatility, consistent but smaller moves.",
        min_holding_days=14,
        volatility_scalar=0.8
    ),
    MarketCapTier.UNKNOWN: BarrierThresholds(
        upper_barrier_pct=0.05,   # Conservative default
        lower_barrier_pct=0.03,
        rationale="Unknown market cap: Using conservative small-cap assumptions.",
        min_holding_days=7,
        volatility_scalar=1.0
    ),
}


# REGIME ADJUSTMENT MULTIPLIERS
# Applied to base barriers based on market regime
REGIME_ADJUSTMENTS = {
    MarketRegimeState.BULLISH: {
        'upper_multiplier': 1.2,   # Aim higher in bull markets (+20% target)
        'lower_multiplier': 0.8,   # Tighter stops OK (-20% tighter)
        'rationale': "Bullish: Risk-on, extend targets, tighten stops"
    },
    MarketRegimeState.NEUTRAL: {
        'upper_multiplier': 1.0,   # Standard targets
        'lower_multiplier': 1.0,   # Standard stops
        'rationale': "Neutral: Use base parameters"
    },
    MarketRegimeState.BEARISH: {
        'upper_multiplier': 0.7,   # Lower targets (take profits faster)
        'lower_multiplier': 1.3,   # Wider stops (more volatility)
        'rationale': "Bearish: Take profits quickly, wider stops for chop"
    },
    MarketRegimeState.HIGH_VOLATILITY: {
        'upper_multiplier': 0.8,   # Slightly reduced targets
        'lower_multiplier': 1.5,   # Much wider stops
        'rationale': "High Vol: Wider stops essential, targets still achievable"
    },
    MarketRegimeState.CRISIS: {
        'upper_multiplier': 0.5,   # Very reduced targets (capital preservation)
        'lower_multiplier': 2.0,   # Very wide stops (extreme volatility)
        'rationale': "Crisis: Capital preservation mode, quick exits"
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def classify_market_cap_from_adv(adv: float) -> MarketCapTier:
    """
    Classify market cap tier from Average Daily Dollar Volume.

    This is the fallback when market cap data is unavailable (common for EU stocks).
    ADV correlates with market cap for actively traded stocks.

    Args:
        adv: Average daily dollar volume in USD

    Returns:
        MarketCapTier enum value
    """
    if adv < ADV_TIER_THRESHOLDS['nano']:
        return MarketCapTier.NANO
    elif adv < ADV_TIER_THRESHOLDS['micro']:
        return MarketCapTier.MICRO
    elif adv < ADV_TIER_THRESHOLDS['small']:
        return MarketCapTier.SMALL
    elif adv < ADV_TIER_THRESHOLDS['mid']:
        return MarketCapTier.MID
    else:
        return MarketCapTier.LARGE


def classify_market_cap(market_cap: Optional[float] = None, adv: Optional[float] = None) -> MarketCapTier:
    """
    Classify market cap tier from market cap or ADV.

    Prefers market cap if available, falls back to ADV.

    Args:
        market_cap: Market capitalization in USD (if available)
        adv: Average daily dollar volume in USD (fallback)

    Returns:
        MarketCapTier enum value
    """
    if market_cap is not None and market_cap > 0:
        if market_cap < MARKET_CAP_TIER_THRESHOLDS['nano']:
            return MarketCapTier.NANO
        elif market_cap < MARKET_CAP_TIER_THRESHOLDS['micro']:
            return MarketCapTier.MICRO
        elif market_cap < MARKET_CAP_TIER_THRESHOLDS['small']:
            return MarketCapTier.SMALL
        elif market_cap < MARKET_CAP_TIER_THRESHOLDS['mid']:
            return MarketCapTier.MID
        else:
            return MarketCapTier.LARGE

    if adv is not None and adv > 0:
        return classify_market_cap_from_adv(adv)

    return MarketCapTier.UNKNOWN


def map_vix_to_regime(vix: float) -> MarketRegimeState:
    """
    Map VIX level to market regime state.

    Args:
        vix: Current VIX level

    Returns:
        MarketRegimeState enum value
    """
    if vix >= 35:
        return MarketRegimeState.CRISIS
    elif vix >= 25:
        return MarketRegimeState.HIGH_VOLATILITY
    elif vix >= 18:
        return MarketRegimeState.BEARISH  # Elevated VIX often means bearish
    elif vix >= 12:
        return MarketRegimeState.NEUTRAL
    else:
        return MarketRegimeState.BULLISH  # Low VIX typically bullish


def map_trend_to_regime(
    price: float,
    sma_50: float,
    sma_200: float,
    vix: Optional[float] = None
) -> MarketRegimeState:
    """
    Map trend indicators to market regime state.

    Args:
        price: Current price
        sma_50: 50-day simple moving average
        sma_200: 200-day simple moving average
        vix: Optional VIX level for refinement

    Returns:
        MarketRegimeState enum value
    """
    # Check VIX first for crisis/high vol
    if vix is not None:
        if vix >= 35:
            return MarketRegimeState.CRISIS
        elif vix >= 25:
            return MarketRegimeState.HIGH_VOLATILITY

    # Trend-based classification
    if price > sma_50 > sma_200:
        # Strong uptrend
        return MarketRegimeState.BULLISH
    elif price < sma_50 < sma_200:
        # Strong downtrend
        return MarketRegimeState.BEARISH
    elif price > sma_200:
        # Above long-term average but choppy
        return MarketRegimeState.NEUTRAL
    else:
        # Below long-term average
        return MarketRegimeState.BEARISH


# =============================================================================
# MAIN BARRIER CALCULATION
# =============================================================================

def get_dynamic_barriers(
    market_cap_tier: MarketCapTier,
    regime: MarketRegimeState,
    base_volatility: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate dynamic barriers based on market cap and regime.

    Args:
        market_cap_tier: Market capitalization tier
        regime: Current market regime state
        base_volatility: Optional realized volatility for additional scaling

    Returns:
        Dictionary with barrier configuration:
            - upper_barrier_pct: Profit-taking threshold
            - lower_barrier_pct: Stop-loss threshold
            - min_holding_days: Minimum holding period
            - rationale: Explanation
            - market_cap_tier: Input tier
            - regime: Input regime
    """
    # Get base barriers for market cap tier
    base = BASE_BARRIERS_BY_CAP.get(market_cap_tier, BASE_BARRIERS_BY_CAP[MarketCapTier.UNKNOWN])

    # Get regime adjustments
    adjustment = REGIME_ADJUSTMENTS.get(regime, REGIME_ADJUSTMENTS[MarketRegimeState.NEUTRAL])

    # Calculate adjusted barriers
    upper_barrier = base.upper_barrier_pct * adjustment['upper_multiplier']
    lower_barrier = base.lower_barrier_pct * adjustment['lower_multiplier']

    # Apply volatility scaling if provided
    if base_volatility is not None and base_volatility > 0:
        # Scale relative to expected volatility (roughly 2% daily for micro-caps)
        expected_vol = 0.02
        vol_ratio = base_volatility / expected_vol

        # Moderate scaling: don't let volatility dominate
        vol_adjustment = 0.5 + 0.5 * min(vol_ratio, 2.0)  # Clamp to [0.5, 1.5]

        upper_barrier *= vol_adjustment * base.volatility_scalar
        lower_barrier *= vol_adjustment * base.volatility_scalar

    # Ensure minimum barrier widths
    upper_barrier = max(upper_barrier, 0.01)  # At least 1%
    lower_barrier = max(lower_barrier, 0.01)  # At least 1%

    return {
        'upper_barrier_pct': round(upper_barrier, 4),
        'lower_barrier_pct': round(lower_barrier, 4),
        'min_holding_days': base.min_holding_days,
        'volatility_scalar': base.volatility_scalar,
        'rationale': f"{base.rationale} | {adjustment['rationale']}",
        'market_cap_tier': market_cap_tier.value,
        'regime': regime.value,
    }


def get_barriers_for_pattern(
    ticker: str,
    pattern_date: pd.Timestamp,
    price_df: pd.DataFrame,
    market_cap: Optional[float] = None,
    vix: Optional[float] = None,
    price_col: str = 'close',
    volume_col: str = 'volume'
) -> Dict[str, Any]:
    """
    Calculate barriers for a specific pattern using historical data.

    This function computes barriers using ONLY data available at pattern_date.
    No look-ahead bias.

    Args:
        ticker: Stock ticker symbol
        pattern_date: Date of the pattern (barriers calculated at this point)
        price_df: Historical price data (DatetimeIndex)
        market_cap: Optional market cap if known
        vix: Optional VIX level at pattern_date
        price_col: Column name for close prices
        volume_col: Column name for volume

    Returns:
        Dictionary with barrier configuration
    """
    # Get data BEFORE pattern date only (strict temporal integrity)
    hist_data = price_df[price_df.index < pattern_date].copy()

    if len(hist_data) < 50:
        # Insufficient data, use defaults
        return get_dynamic_barriers(
            market_cap_tier=MarketCapTier.UNKNOWN,
            regime=MarketRegimeState.NEUTRAL
        )

    # Calculate ADV for market cap classification
    recent_data = hist_data.tail(40)  # 40-day ADV
    if volume_col in recent_data.columns:
        avg_volume = recent_data[volume_col].mean()
        avg_price = recent_data[price_col].mean()
        adv = avg_volume * avg_price
    else:
        adv = None

    # Classify market cap tier
    market_cap_tier = classify_market_cap(market_cap=market_cap, adv=adv)

    # Determine regime
    if vix is not None:
        regime = map_vix_to_regime(vix)
    else:
        # Use price/SMA trend if VIX not available
        current_price = hist_data[price_col].iloc[-1]

        sma_50 = hist_data[price_col].rolling(50, closed='left').mean().iloc[-1]
        sma_200 = hist_data[price_col].rolling(200, closed='left').mean().iloc[-1] if len(hist_data) >= 200 else sma_50

        regime = map_trend_to_regime(current_price, sma_50, sma_200, vix)

    # Calculate realized volatility for additional scaling
    log_returns = np.log(hist_data[price_col] / hist_data[price_col].shift(1))
    daily_vol = log_returns.rolling(20, closed='left').std().iloc[-1]

    # Get dynamic barriers
    barriers = get_dynamic_barriers(
        market_cap_tier=market_cap_tier,
        regime=regime,
        base_volatility=daily_vol
    )

    # Add calculated metadata
    barriers['adv_calculated'] = adv
    barriers['daily_volatility_at_pattern'] = daily_vol
    barriers['ticker'] = ticker
    barriers['pattern_date'] = pattern_date

    return barriers


# =============================================================================
# BARRIER SUMMARY TABLE
# =============================================================================

def print_barrier_matrix():
    """Print a summary table of all barrier combinations."""
    print("=" * 90)
    print("DYNAMIC BARRIER CONFIGURATION MATRIX")
    print("=" * 90)
    print(f"{'Market Cap Tier':<15} {'Regime':<15} {'Upper %':<10} {'Lower %':<10} {'Min Days':<10}")
    print("-" * 90)

    for cap_tier in MarketCapTier:
        if cap_tier == MarketCapTier.UNKNOWN:
            continue
        for regime in MarketRegimeState:
            barriers = get_dynamic_barriers(cap_tier, regime)
            print(
                f"{cap_tier.value:<15} "
                f"{regime.value:<15} "
                f"+{barriers['upper_barrier_pct']*100:.1f}%{'':>4} "
                f"-{barriers['lower_barrier_pct']*100:.1f}%{'':>4} "
                f"{barriers['min_holding_days']}"
            )
        print()

    print("=" * 90)


if __name__ == "__main__":
    print_barrier_matrix()

    print("\nExample: Micro-cap in Bullish regime")
    barriers = get_dynamic_barriers(MarketCapTier.MICRO, MarketRegimeState.BULLISH)
    for k, v in barriers.items():
        print(f"  {k}: {v}")

    print("\nExample: Small-cap in Crisis regime")
    barriers = get_dynamic_barriers(MarketCapTier.SMALL, MarketRegimeState.CRISIS)
    for k, v in barriers.items():
        print(f"  {k}: {v}")
