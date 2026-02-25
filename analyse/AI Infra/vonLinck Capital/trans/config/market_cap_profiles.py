"""
Market Cap Asset Profiles
==========================

Dynamic parameter profiles based on market cap category.
Market cap dictates the "physics" of the stock (volatility, noise floor, liquidity).

Strategy:
- Nano/Micro: Extreme noise requires loose stops, higher targets (4-5R)
- Small: "Sweet Spot" for volatility contraction setups (3.5R)
- Mid: Standard technical behavior (3.0R)
- Large/Mega: High efficiency, lower targets (2-2.5R) as doubling is rare

Author: TRANS System
Date: December 2025
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class MarketCapCategory(Enum):
    """Market cap categories."""
    NANO_CAP = "nano_cap"      # < $50M
    MICRO_CAP = "micro_cap"    # $50M - $300M
    SMALL_CAP = "small_cap"    # $300M - $2B
    MID_CAP = "mid_cap"        # $2B - $10B
    LARGE_CAP = "large_cap"    # $10B - $200B
    MEGA_CAP = "mega_cap"      # > $200B


@dataclass
class AssetProfile:
    """
    Dynamic parameters for a market cap category.

    IMPORTANT: These are adjustable for backtesting! Default values are conservative
    and should be optimized through systematic backtesting.

    Attributes:
        category: Market cap category
        stop_buffer_min: Minimum stop buffer below lower boundary (decimal)
        stop_buffer_max: Maximum stop buffer below lower boundary (decimal)
        target_r_min: Minimum R-multiple for target
        target_r_max: Maximum R-multiple for target
        max_box_width: Maximum allowed box width for qualifying (decimal)
        grey_zone_r: R-multiple threshold for grey zone (between noise and target)
        rationale: Strategic reasoning for these parameters
    """
    category: MarketCapCategory
    stop_buffer_min: float
    stop_buffer_max: float
    target_r_min: float
    target_r_max: float
    max_box_width: float
    grey_zone_r: float
    rationale: str

    @property
    def stop_buffer(self) -> float:
        """Default stop buffer (midpoint of range)."""
        return (self.stop_buffer_min + self.stop_buffer_max) / 2.0

    @property
    def target_r(self) -> float:
        """Default target R-multiple (midpoint of range)."""
        return (self.target_r_min + self.target_r_max) / 2.0

    def copy_with_overrides(
        self,
        stop_buffer_min: Optional[float] = None,
        stop_buffer_max: Optional[float] = None,
        target_r_min: Optional[float] = None,
        target_r_max: Optional[float] = None,
        max_box_width: Optional[float] = None,
        grey_zone_r: Optional[float] = None
    ) -> 'AssetProfile':
        """Create a copy with selective parameter overrides for backtesting."""
        return AssetProfile(
            category=self.category,
            stop_buffer_min=stop_buffer_min if stop_buffer_min is not None else self.stop_buffer_min,
            stop_buffer_max=stop_buffer_max if stop_buffer_max is not None else self.stop_buffer_max,
            target_r_min=target_r_min if target_r_min is not None else self.target_r_min,
            target_r_max=target_r_max if target_r_max is not None else self.target_r_max,
            max_box_width=max_box_width if max_box_width is not None else self.max_box_width,
            grey_zone_r=grey_zone_r if grey_zone_r is not None else self.grey_zone_r,
            rationale=self.rationale
        )


# Market Cap Profile Definitions
# ===============================
#
# UPDATED 2025-12-16: 1.5x multiplier based on grid search empirical validation
# Grid search (15 combinations, 10 years data) showed:
# - Tighter limits (0.6x) → 75.9% danger rate
# - Optimal limits (1.5x) → 70.4% danger rate (5.5pp improvement)
# Small-cap natural consolidation width is 25-30%, not 10-15%
#
# Adjust these via copy_with_overrides() for backtesting optimization.

NANO_CAP_PROFILE = AssetProfile(
    category=MarketCapCategory.NANO_CAP,
    stop_buffer_min=0.04,     # 4%
    stop_buffer_max=0.06,     # 6%
    target_r_min=4.0,         # 4R
    target_r_max=5.0,         # 5R
    max_box_width=0.60,       # 60% (was 40%) - extreme volatility, lottery tickets
    grey_zone_r=3.5,          # 3.5R
    rationale=(
        "Nano-caps: Extreme noise, lottery ticket dynamics. "
        "60% width (empirically validated 2025-12-16) allows for substantial volatility."
    )
)

MICRO_CAP_PROFILE = AssetProfile(
    category=MarketCapCategory.MICRO_CAP,
    stop_buffer_min=0.035,    # 3.5%
    stop_buffer_max=0.045,    # 4.5%
    target_r_min=3.5,         # 3.5R
    target_r_max=4.5,         # 4.5R
    max_box_width=0.45,       # 45% (was 30%) - high volatility but showing structure
    grey_zone_r=3.0,          # 3.0R
    rationale=(
        "Micro-caps: High volatility but starting to show structure. "
        "45% width (empirically validated 2025-12-16) accepts legitimate consolidation patterns."
    )
)

SMALL_CAP_PROFILE = AssetProfile(
    category=MarketCapCategory.SMALL_CAP,
    stop_buffer_min=0.025,    # 2.5%
    stop_buffer_max=0.035,    # 3.5%
    target_r_min=3.0,         # 3R
    target_r_max=4.0,         # 4R
    max_box_width=0.30,       # 30% (was 20%) - natural consolidation width 25-30%
    grey_zone_r=2.5,          # 2.5R
    rationale=(
        'Small-caps: The "Sweet Spot" for consolidation breakouts. '
        '30% width (empirically validated 2025-12-16) matches natural consolidation behavior. '
        'Grid search: 20% width → 75.9% danger rate, 30% width → 70.4% danger rate.'
    )
)

MID_CAP_PROFILE = AssetProfile(
    category=MarketCapCategory.MID_CAP,
    stop_buffer_min=0.02,     # 2%
    stop_buffer_max=0.025,    # 2.5%
    target_r_min=2.5,         # 2.5R
    target_r_max=3.5,         # 3.5R
    max_box_width=0.225,      # 22.5% (was 15%) - standard behavior with realistic tolerance
    grey_zone_r=2.0,          # 2.0R
    rationale=(
        "Mid-caps: Standard technical behavior. "
        "22.5% width (empirically validated 2025-12-16) = real consolidation, not market indecision."
    )
)

LARGE_CAP_PROFILE = AssetProfile(
    category=MarketCapCategory.LARGE_CAP,
    stop_buffer_min=0.015,    # 1.5%
    stop_buffer_max=0.02,     # 2%
    target_r_min=2.0,         # 2R
    target_r_max=3.0,         # 3R
    max_box_width=0.15,       # 15% (was 10%) - efficient markets with realistic tolerance
    grey_zone_r=1.5,          # 1.5R
    rationale=(
        "Large-caps: Efficient markets with high liquidity. "
        "15% width (empirically validated 2025-12-16) filters for REAL coiling, not drift."
    )
)

MEGA_CAP_PROFILE = AssetProfile(
    category=MarketCapCategory.MEGA_CAP,
    stop_buffer_min=0.01,     # 1%
    stop_buffer_max=0.015,    # 1.5%
    target_r_min=1.5,         # 1.5R
    target_r_max=2.5,         # 2.5R
    max_box_width=0.075,      # 7.5% (was 5%) - very tight but realistic for FAANG
    grey_zone_r=1.2,          # 1.2R
    rationale=(
        "Mega-caps (FAANG): Extremely efficient. "
        "7.5% width (empirically validated 2025-12-16) = exceptional tightness. Doubling is rare."
    )
)

# Profile Registry
# ================

MARKET_CAP_PROFILES = {
    MarketCapCategory.NANO_CAP: NANO_CAP_PROFILE,
    MarketCapCategory.MICRO_CAP: MICRO_CAP_PROFILE,
    MarketCapCategory.SMALL_CAP: SMALL_CAP_PROFILE,
    MarketCapCategory.MID_CAP: MID_CAP_PROFILE,
    MarketCapCategory.LARGE_CAP: LARGE_CAP_PROFILE,
    MarketCapCategory.MEGA_CAP: MEGA_CAP_PROFILE,
}

# Default profile for unknown/missing market cap
DEFAULT_PROFILE = MID_CAP_PROFILE


def get_profile(market_cap_category: Optional[str]) -> AssetProfile:
    """
    Get asset profile for a market cap category.

    Args:
        market_cap_category: Market cap category string (e.g., 'small_cap')
                            or None to use default (mid-cap)

    Returns:
        AssetProfile with appropriate parameters
    """
    # Default to mid-cap if not specified
    if market_cap_category is None:
        return DEFAULT_PROFILE

    # Convert string to enum
    try:
        category = MarketCapCategory(market_cap_category)
    except ValueError:
        # Unknown category - default to mid-cap
        return DEFAULT_PROFILE

    return MARKET_CAP_PROFILES[category]


def get_profile_from_market_cap(market_cap: Optional[float]) -> AssetProfile:
    """
    Get asset profile directly from market cap value.

    Args:
        market_cap: Market cap in dollars (e.g., 1_000_000_000 for $1B)

    Returns:
        AssetProfile with appropriate parameters
    """
    if market_cap is None:
        return MID_CAP_PROFILE

    # Determine category
    if market_cap >= 200_000_000_000:  # >= $200B
        category = MarketCapCategory.MEGA_CAP
    elif market_cap >= 10_000_000_000:  # >= $10B
        category = MarketCapCategory.LARGE_CAP
    elif market_cap >= 2_000_000_000:   # >= $2B
        category = MarketCapCategory.MID_CAP
    elif market_cap >= 300_000_000:     # >= $300M
        category = MarketCapCategory.SMALL_CAP
    elif market_cap >= 50_000_000:      # >= $50M
        category = MarketCapCategory.MICRO_CAP
    else:                                # < $50M
        category = MarketCapCategory.NANO_CAP

    return MARKET_CAP_PROFILES[category]


def get_profile_summary() -> str:
    """Get a formatted summary of all profiles."""
    lines = [
        "Market Cap Asset Profiles",
        "=" * 80,
        "",
        f"{'Category':<12} {'Stop Buffer':<15} {'Target R':<12} {'Max Width':<12} {'Rationale':<30}",
        "-" * 80,
    ]

    for category in [
        MarketCapCategory.NANO_CAP,
        MarketCapCategory.SMALL_CAP,
        MarketCapCategory.MID_CAP,
        MarketCapCategory.LARGE_CAP,
    ]:
        profile = MARKET_CAP_PROFILES[category]

        # Format ranges
        if profile.stop_buffer_min == profile.stop_buffer_max:
            stop_str = f"{profile.stop_buffer_min*100:.1f}%"
        else:
            stop_str = f"{profile.stop_buffer_min*100:.1f}%-{profile.stop_buffer_max*100:.1f}%"

        if profile.target_r_min == profile.target_r_max:
            target_str = f"{profile.target_r_min:.1f}R"
        else:
            target_str = f"{profile.target_r_min:.1f}R-{profile.target_r_max:.1f}R"

        width_str = f"{profile.max_box_width*100:.0f}%"

        # Shorten rationale for display
        rationale_short = profile.rationale.split('.')[0][:35]

        lines.append(
            f"{category.value:<12} {stop_str:<15} {target_str:<12} "
            f"{width_str:<12} {rationale_short:<30}"
        )

    return "\n".join(lines)


def create_backtest_profiles(
    max_width_multiplier: float = 1.0,
    stop_buffer_multiplier: float = 1.0,
    target_r_multiplier: float = 1.0
) -> dict:
    """
    Create a modified set of profiles for backtesting parameter sensitivity.

    Args:
        max_width_multiplier: Multiply all max_box_width values (e.g., 0.8 = 20% tighter)
        stop_buffer_multiplier: Multiply all stop_buffer values
        target_r_multiplier: Multiply all target R-multiples

    Returns:
        Dictionary of modified profiles keyed by category

    Example:
        # Test with 20% tighter width thresholds
        profiles = create_backtest_profiles(max_width_multiplier=0.8)

        # Test with 50% wider stops (more forgiving)
        profiles = create_backtest_profiles(stop_buffer_multiplier=1.5)

        # Test with higher targets
        profiles = create_backtest_profiles(target_r_multiplier=1.2)
    """
    modified = {}

    for category, profile in MARKET_CAP_PROFILES.items():
        modified[category] = profile.copy_with_overrides(
            stop_buffer_min=profile.stop_buffer_min * stop_buffer_multiplier,
            stop_buffer_max=profile.stop_buffer_max * stop_buffer_multiplier,
            target_r_min=profile.target_r_min * target_r_multiplier,
            target_r_max=profile.target_r_max * target_r_multiplier,
            max_box_width=profile.max_box_width * max_width_multiplier,
            grey_zone_r=profile.grey_zone_r * target_r_multiplier
        )

    return modified


def get_backtest_config_summary(profiles: dict) -> str:
    """Get summary of a custom backtest profile configuration."""
    lines = [
        "Backtest Profile Configuration",
        "=" * 80,
        "",
        f"{'Category':<12} {'Stop Buffer':<15} {'Target R':<12} {'Max Width':<12}",
        "-" * 80,
    ]

    for category in [
        MarketCapCategory.NANO_CAP,
        MarketCapCategory.MICRO_CAP,
        MarketCapCategory.SMALL_CAP,
        MarketCapCategory.MID_CAP,
        MarketCapCategory.LARGE_CAP,
        MarketCapCategory.MEGA_CAP,
    ]:
        profile = profiles[category]

        # Format ranges
        if profile.stop_buffer_min == profile.stop_buffer_max:
            stop_str = f"{profile.stop_buffer_min*100:.1f}%"
        else:
            stop_str = f"{profile.stop_buffer_min*100:.1f}%-{profile.stop_buffer_max*100:.1f}%"

        if profile.target_r_min == profile.target_r_max:
            target_str = f"{profile.target_r_min:.1f}R"
        else:
            target_str = f"{profile.target_r_min:.1f}R-{profile.target_r_max:.1f}R"

        width_str = f"{profile.max_box_width*100:.0f}%"

        lines.append(
            f"{category.value:<12} {stop_str:<15} {target_str:<12} {width_str:<12}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # Test the profile system
    print(get_profile_summary())
    print("\n")

    # Test profile retrieval
    test_caps = [
        ("Nano ($30M)", 30_000_000),
        ("Micro ($100M)", 100_000_000),
        ("Small ($1B)", 1_000_000_000),
        ("Mid ($5B)", 5_000_000_000),
        ("Large ($50B)", 50_000_000_000),
        ("Mega ($500B)", 500_000_000_000),
    ]

    print("\nProfile Resolution Examples:")
    print("-" * 80)
    for name, market_cap in test_caps:
        profile = get_profile_from_market_cap(market_cap)
        print(f"{name:<20} -> Stop: {profile.stop_buffer*100:.1f}%, "
              f"Target: {profile.target_r:.1f}R, "
              f"Max Width: {profile.max_box_width*100:.0f}%")

    # Test backtest configuration
    print("\n\nBacktest Configuration Example (20% tighter widths):")
    print("=" * 80)
    tight_profiles = create_backtest_profiles(max_width_multiplier=0.8)
    print(get_backtest_config_summary(tight_profiles))
