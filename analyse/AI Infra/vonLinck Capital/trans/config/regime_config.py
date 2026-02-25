"""
Market Regime Configuration for Dynamic Strategy Adjustment
===========================================================

Defines regime-specific parameter adjustments for:
- R-multiple targets
- Position sizing
- Stop loss placement
- Signal thresholds

Market regimes are detected using VIX levels, trend indicators,
and other macro signals.

Usage:
    from config.regime_config import get_regime_params, detect_regime

    # Get current regime parameters
    regime = detect_regime(vix=22, spy_trend='bullish')
    params = get_regime_params(regime)

    # Apply to position sizing
    adjusted_r_target = params['r_target']
    position_scale = params['position_scale']

NOTE: This is a configuration file. The actual regime detection logic
should be implemented in utils/regime_detector.py (not yet created).
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications."""
    HIGH_VOLATILITY = "high_vol"      # VIX > 25
    ELEVATED_VOLATILITY = "elevated"  # VIX 18-25
    NORMAL = "normal"                 # VIX 12-18
    LOW_VOLATILITY = "low_vol"        # VIX < 12
    CRISIS = "crisis"                 # VIX > 35


@dataclass
class RegimeParameters:
    """Parameter adjustments for a specific regime."""
    regime: MarketRegime
    description: str

    # Target adjustment
    r_target: float           # R-multiple target (base is 3.0)
    r_target_rationale: str

    # Position sizing
    position_scale: float     # Multiplier for position size (1.0 = full)
    max_positions: int        # Maximum concurrent positions

    # Stop adjustment
    stop_buffer_multiplier: float  # Multiplier for stop buffer (1.0 = standard)

    # Signal threshold adjustment
    min_ev_threshold: float   # Minimum EV for signal (higher = more selective)
    max_danger_prob: float    # Maximum P(Danger) allowed

    # Execution rules
    allow_overnight_holds: bool
    require_volume_confirmation: bool


# ============================================================================
# Regime Parameter Definitions
# ============================================================================

REGIME_PARAMETERS: Dict[MarketRegime, RegimeParameters] = {

    MarketRegime.HIGH_VOLATILITY: RegimeParameters(
        regime=MarketRegime.HIGH_VOLATILITY,
        description="High volatility regime (VIX > 25): Reduce exposure, tighten targets",

        # Lower target - capture gains faster in volatile conditions
        r_target=2.0,
        r_target_rationale=(
            "In high-vol, breakouts are more likely to reverse. "
            "2R target captures gains before mean reversion."
        ),

        # Reduce position size significantly
        position_scale=0.5,  # Half size
        max_positions=3,     # Fewer concurrent positions

        # Wider stop buffer to avoid volatility stops
        stop_buffer_multiplier=1.5,

        # Higher selectivity
        min_ev_threshold=5.0,   # Only strong signals
        max_danger_prob=0.20,   # Lower danger tolerance

        # Conservative execution
        allow_overnight_holds=True,  # Still allowed but smaller size
        require_volume_confirmation=True
    ),

    MarketRegime.ELEVATED_VOLATILITY: RegimeParameters(
        regime=MarketRegime.ELEVATED_VOLATILITY,
        description="Elevated volatility (VIX 18-25): Moderate adjustments",

        r_target=2.5,
        r_target_rationale="Slightly reduced target, still allows for meaningful gains",

        position_scale=0.75,
        max_positions=4,

        stop_buffer_multiplier=1.25,

        min_ev_threshold=4.0,
        max_danger_prob=0.22,

        allow_overnight_holds=True,
        require_volume_confirmation=True
    ),

    MarketRegime.NORMAL: RegimeParameters(
        regime=MarketRegime.NORMAL,
        description="Normal volatility (VIX 12-18): Standard parameters",

        r_target=3.0,
        r_target_rationale="Standard 3R target for balanced risk/reward",

        position_scale=1.0,
        max_positions=5,

        stop_buffer_multiplier=1.0,

        min_ev_threshold=3.0,
        max_danger_prob=0.25,

        allow_overnight_holds=True,
        require_volume_confirmation=True
    ),

    MarketRegime.LOW_VOLATILITY: RegimeParameters(
        regime=MarketRegime.LOW_VOLATILITY,
        description="Low volatility (VIX < 12): Extended targets, larger positions",

        # Higher target - breakouts in low vol tend to be persistent
        r_target=4.0,
        r_target_rationale=(
            "Low vol breakouts are more likely to trend. "
            "Extended target captures larger moves."
        ),

        position_scale=1.0,
        max_positions=6,  # Can hold more concurrent positions

        # Tighter stop buffer - lower noise
        stop_buffer_multiplier=0.85,

        # More permissive
        min_ev_threshold=2.5,
        max_danger_prob=0.28,

        allow_overnight_holds=True,
        require_volume_confirmation=True
    ),

    MarketRegime.CRISIS: RegimeParameters(
        regime=MarketRegime.CRISIS,
        description="Crisis regime (VIX > 35): Minimal exposure or cash",

        # Minimum target - get out quickly
        r_target=1.5,
        r_target_rationale="Crisis conditions - any gain is good, exit fast",

        position_scale=0.25,  # Quarter size maximum
        max_positions=2,      # Very limited exposure

        stop_buffer_multiplier=2.0,  # Wide stops for extreme volatility

        min_ev_threshold=6.0,   # Only exceptional setups
        max_danger_prob=0.15,   # Very low danger tolerance

        allow_overnight_holds=False,  # Day trades only
        require_volume_confirmation=True
    ),
}


# ============================================================================
# VIX Thresholds for Regime Detection
# ============================================================================

VIX_THRESHOLDS = {
    'crisis': 35.0,
    'high_vol': 25.0,
    'elevated': 18.0,
    'normal': 12.0,
    # Below 12 = low_vol
}


def detect_regime_from_vix(vix: float) -> MarketRegime:
    """
    Detect market regime from VIX level.

    Args:
        vix: Current VIX level

    Returns:
        MarketRegime enum value
    """
    if vix >= VIX_THRESHOLDS['crisis']:
        return MarketRegime.CRISIS
    elif vix >= VIX_THRESHOLDS['high_vol']:
        return MarketRegime.HIGH_VOLATILITY
    elif vix >= VIX_THRESHOLDS['elevated']:
        return MarketRegime.ELEVATED_VOLATILITY
    elif vix >= VIX_THRESHOLDS['normal']:
        return MarketRegime.NORMAL
    else:
        return MarketRegime.LOW_VOLATILITY


def get_regime_params(regime: MarketRegime) -> RegimeParameters:
    """
    Get parameters for a specific regime.

    Args:
        regime: MarketRegime enum value

    Returns:
        RegimeParameters for the regime
    """
    return REGIME_PARAMETERS[regime]


def get_regime_params_from_vix(vix: float) -> RegimeParameters:
    """
    Get regime parameters directly from VIX level.

    Args:
        vix: Current VIX level

    Returns:
        RegimeParameters for detected regime
    """
    regime = detect_regime_from_vix(vix)
    return get_regime_params(regime)


# ============================================================================
# Hybrid Regime Detection (VIX + Trend)
# ============================================================================

def detect_regime(
    vix: float,
    spy_trend: str = 'neutral',
    spy_above_sma200: Optional[bool] = None
) -> MarketRegime:
    """
    Detect regime using multiple signals.

    Args:
        vix: Current VIX level
        spy_trend: SPY trend ('bullish', 'bearish', 'neutral')
        spy_above_sma200: Whether SPY is above 200-day SMA

    Returns:
        MarketRegime with context-aware adjustments

    Note:
        This is a simplified version. Production should include:
        - VIX term structure (contango/backwardation)
        - Credit spreads
        - Put/call ratios
        - Sector breadth
    """
    base_regime = detect_regime_from_vix(vix)

    # Bearish trend in elevated VIX -> treat as high vol
    if spy_trend == 'bearish' and base_regime == MarketRegime.ELEVATED_VOLATILITY:
        return MarketRegime.HIGH_VOLATILITY

    # Bullish trend in elevated VIX -> may be fine to treat as normal
    if spy_trend == 'bullish' and base_regime == MarketRegime.ELEVATED_VOLATILITY:
        return MarketRegime.NORMAL

    # Below SMA200 in any regime -> bump up one level
    if spy_above_sma200 is False and base_regime in [
        MarketRegime.NORMAL,
        MarketRegime.LOW_VOLATILITY
    ]:
        return MarketRegime.ELEVATED_VOLATILITY

    return base_regime


# ============================================================================
# Utility Functions
# ============================================================================

def adjust_target_for_pattern(
    base_params: RegimeParameters,
    pattern_volatility: float
) -> float:
    """
    Adjust R-target based on pattern-specific volatility.

    Args:
        base_params: Regime parameters
        pattern_volatility: Pattern's volatility proxy (e.g., BBW)

    Returns:
        Adjusted R-target
    """
    # High volatility patterns -> lower target
    # Low volatility patterns -> can keep higher target

    if pattern_volatility > 0.15:  # Very volatile pattern
        return min(base_params.r_target, 2.0)
    elif pattern_volatility < 0.05:  # Very tight pattern
        return base_params.r_target + 0.5  # Can extend target
    else:
        return base_params.r_target


def get_position_size_adjustment(
    base_params: RegimeParameters,
    account_heat: float
) -> float:
    """
    Adjust position size based on account heat.

    Args:
        base_params: Regime parameters
        account_heat: Current account risk exposure (0-1)

    Returns:
        Adjusted position scale (0-1)
    """
    base_scale = base_params.position_scale

    # Reduce further if account is already hot
    if account_heat > 0.8:
        return base_scale * 0.25  # Quarter size
    elif account_heat > 0.5:
        return base_scale * 0.5   # Half size
    elif account_heat > 0.3:
        return base_scale * 0.75  # Three-quarter size

    return base_scale


def summarize_regime(regime: MarketRegime) -> Dict[str, Any]:
    """
    Get summary of regime parameters for display.

    Args:
        regime: MarketRegime enum value

    Returns:
        Dictionary with key parameters
    """
    params = get_regime_params(regime)

    return {
        'regime': regime.value,
        'description': params.description,
        'r_target': params.r_target,
        'position_scale': params.position_scale,
        'max_positions': params.max_positions,
        'min_ev': params.min_ev_threshold,
        'max_danger': params.max_danger_prob
    }


# ============================================================================
# Print regime summary when run directly
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MARKET REGIME CONFIGURATION")
    print("=" * 70)

    for regime in MarketRegime:
        params = get_regime_params(regime)
        print(f"\n{regime.value.upper()}: {params.description}")
        print(f"  R-Target: {params.r_target} ({params.r_target_rationale[:50]}...)")
        print(f"  Position Scale: {params.position_scale}")
        print(f"  Max Positions: {params.max_positions}")
        print(f"  Min EV: {params.min_ev_threshold}")
        print(f"  Max P(Danger): {params.max_danger_prob}")

    print("\n" + "=" * 70)
    print("VIX THRESHOLDS")
    print("=" * 70)
    for level, threshold in VIX_THRESHOLDS.items():
        print(f"  {level}: VIX >= {threshold}")
