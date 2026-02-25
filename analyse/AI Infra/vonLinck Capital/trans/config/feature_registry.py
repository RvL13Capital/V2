"""
Dynamic Feature Registry for TRAnS Pipeline
============================================

Centralizes all feature definitions to eliminate hardcoded indices scattered
throughout the codebase. Both feature extraction and training code import
from this single source of truth.

Design Principles:
    1. Single source of truth - all feature metadata in one place
    2. Type safety - features are looked up by name, not magic numbers
    3. Transformation tracking - knows which features need log-diff, bounds, etc.
    4. Forward compatibility - easy to add/remove features without hunting indices

Usage:
    from config.feature_registry import FeatureRegistry

    # Get indices for volume ratio features
    vol_indices = FeatureRegistry.get_indices_by_type('log_diff')

    # Check if a feature needs transformation
    if FeatureRegistry.needs_log_diff('relative_volume'):
        value = log_diff(vol_20d, vol_60d)

    # Get feature by index
    feature_name = FeatureRegistry.get_name(3)  # 'relative_volume'

Jan 2026 - Created to replace hardcoded VOLUME_RATIO_INDICES
"""

from typing import Final, List, Dict, Optional, Set, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class FeatureType(Enum):
    """Classification of feature transformation requirements."""
    BOUNDED = 'bounded'         # Already bounded [0, 1], no transform needed
    RATIO = 'ratio'             # Simple ratio, may need clipping
    LOG_DIFF = 'log_diff'       # Volume ratio - use log1p(num) - log1p(denom)
    LOG_SCALE = 'log_scale'     # Already log-scaled (e.g., log_float)
    SLOPE = 'slope'             # Derivative/slope feature
    COMPOSITE = 'composite'     # Derived from other features


@dataclass
class FeatureSpec:
    """Complete specification for a context feature."""
    name: str
    index: int
    feature_type: FeatureType
    description: str
    formula: str
    default: float
    range_min: float
    range_max: float
    dormant_safe: bool  # True if stable for zero-volume stocks

    def __post_init__(self):
        """Validate feature specification."""
        if self.range_max <= self.range_min:
            raise ValueError(f"Invalid range for {self.name}: [{self.range_min}, {self.range_max}]")


class FeatureRegistry:
    """
    Centralized registry of all context features.

    Provides type-safe access to feature metadata, eliminating magic numbers
    throughout the codebase.

    Class Attributes:
        FEATURES: List of all FeatureSpec objects in schema order
        _by_name: Dict mapping feature name to FeatureSpec
        _by_index: Dict mapping index to FeatureSpec
    """

    # =========================================================================
    # FEATURE DEFINITIONS (24 Context Features - Jan 2026)
    # =========================================================================
    # Order matters! Index = position in this list.
    # =========================================================================

    FEATURES: Final[List[FeatureSpec]] = [
        # Index 0: Candle strength indicator (replaces float_turnover Jan 2026)
        FeatureSpec(
            name='retention_rate',
            index=0,
            feature_type=FeatureType.BOUNDED,
            description='Candle strength: (Close-Low)/(High-Low)',
            formula='(close - low) / (high - low)',
            default=0.5,
            range_min=0.0,
            range_max=1.0,
            dormant_safe=True
        ),
        # Index 1: Trend position relative to 200 SMA
        FeatureSpec(
            name='trend_position',
            index=1,
            feature_type=FeatureType.RATIO,
            description='Price vs 200-day SMA',
            formula='close / sma_200',
            default=1.0,
            range_min=0.7,
            range_max=1.5,
            dormant_safe=True
        ),
        # Index 2: Consolidation duration (log-normalized)
        FeatureSpec(
            name='base_duration',
            index=2,
            feature_type=FeatureType.BOUNDED,
            description='Days in consolidation (log-normalized)',
            formula='log(1 + days) / log(201)',
            default=0.3,
            range_min=0.0,
            range_max=1.0,
            dormant_safe=True
        ),
        # Index 3: Recent vs historical volume (LOG-DIFF)
        FeatureSpec(
            name='relative_volume',
            index=3,
            feature_type=FeatureType.LOG_DIFF,
            description='Volume 20d vs 60d average',
            formula='log_diff(vol_20d_avg, vol_60d_avg)',
            default=0.0,
            range_min=-2.0,
            range_max=2.0,
            dormant_safe=False  # UNSTABLE when both ≈ 0
        ),
        # Index 4: Distance below 52-week high
        FeatureSpec(
            name='distance_to_high',
            index=4,
            feature_type=FeatureType.BOUNDED,
            description='Percentage below 52-week high',
            formula='(high_52w - close) / high_52w',
            default=0.15,
            range_min=0.0,
            range_max=0.5,
            dormant_safe=True
        ),
        # Index 5: Float size (log10 scale)
        FeatureSpec(
            name='log_float',
            index=5,
            feature_type=FeatureType.LOG_SCALE,
            description='Log10 of shares outstanding',
            formula='log10(shares_outstanding)',
            default=7.0,
            range_min=5.0,
            range_max=10.0,
            dormant_safe=True
        ),
        # Index 6: Tradability indicator (log10 scale)
        FeatureSpec(
            name='log_dollar_volume',
            index=6,
            feature_type=FeatureType.LOG_SCALE,
            description='Log10 of avg daily dollar volume',
            formula='log10(price * volume_20d_avg)',
            default=5.5,
            range_min=4.0,
            range_max=9.0,
            dormant_safe=True
        ),
        # Index 7: Dormancy detection (LOG-DIFF)
        FeatureSpec(
            name='dormancy_shock',
            index=7,
            feature_type=FeatureType.LOG_DIFF,
            description='Volume 20d vs 252d (yearly) average',
            formula='log_diff(vol_20d_avg, vol_252d_avg)',
            default=0.0,
            range_min=-5.0,
            range_max=5.0,
            dormant_safe=False  # UNSTABLE - division by near-zero yearly avg
        ),
        # Index 8: Volume exhaustion (LOG-DIFF)
        FeatureSpec(
            name='vol_dryup_ratio',
            index=8,
            feature_type=FeatureType.LOG_DIFF,
            description='Volume 20d vs 100d average',
            formula='log_diff(vol_20d_avg, vol_100d_avg)',
            default=0.0,
            range_min=-3.0,
            range_max=3.0,
            dormant_safe=False  # UNSTABLE when both ≈ 0
        ),
        # Index 9: Position in consolidation box
        FeatureSpec(
            name='price_position_at_end',
            index=9,
            feature_type=FeatureType.BOUNDED,
            description='Close position in box (0=lower, 1=upper)',
            formula='(close - lower_boundary) / (upper_boundary - lower_boundary)',
            default=0.5,
            range_min=0.0,
            range_max=1.0,
            dormant_safe=True
        ),
        # Index 10: Recent volume spike (LOG-DIFF)
        FeatureSpec(
            name='volume_shock',
            index=10,
            feature_type=FeatureType.LOG_DIFF,
            description='Max volume in last 3 days vs 20d average',
            formula='log_diff(max(volume[-3:]), vol_20d_avg)',
            default=0.0,
            range_min=-2.0,
            range_max=3.0,
            dormant_safe=False  # UNSTABLE when avg ≈ 0
        ),
        # Index 11: Bollinger Band width trend
        FeatureSpec(
            name='bbw_slope_5d',
            index=11,
            feature_type=FeatureType.SLOPE,
            description='BBW change over last 5 days',
            formula='(bbw_at_end - bbw_5d_ago) / 5',
            default=0.0,
            range_min=-0.02,
            range_max=0.02,
            dormant_safe=True
        ),
        # Index 12: Recent volume trend (LOG-DIFF)
        FeatureSpec(
            name='vol_trend_5d',
            index=12,
            feature_type=FeatureType.LOG_DIFF,
            description='Volume 5d vs 20d average',
            formula='log_diff(vol_5d_avg, vol_20d_avg)',
            default=0.0,
            range_min=-2.0,
            range_max=2.0,
            dormant_safe=False  # UNSTABLE when both ≈ 0
        ),
        # Index 13: Composite coil quality score
        FeatureSpec(
            name='coil_intensity',
            index=13,
            feature_type=FeatureType.COMPOSITE,
            description='Combined coil quality (position + BBW + volume)',
            formula='(pos_score + bbw_score + vol_score) / 3',
            default=0.5,
            range_min=0.0,
            range_max=1.0,
            dormant_safe=False  # Depends on volume features
        ),
        # Index 14: Relative strength vs universe
        FeatureSpec(
            name='relative_strength_cohort',
            index=14,
            feature_type=FeatureType.RATIO,
            description='Ticker 20d return vs universe median',
            formula='ticker_return_20d - universe_median_return_20d',
            default=0.0,
            range_min=-0.30,
            range_max=0.30,
            dormant_safe=True
        ),
        # Index 15: Structural stop distance
        FeatureSpec(
            name='risk_width_pct',
            index=15,
            feature_type=FeatureType.RATIO,
            description='Pattern risk as percentage of price',
            formula='(upper_boundary - lower_boundary) / upper_boundary',
            default=0.10,
            range_min=0.02,
            range_max=0.40,
            dormant_safe=True
        ),
        # Index 16: Volume contraction intensity (LOG-DIFF)
        FeatureSpec(
            name='vol_contraction_intensity',
            index=16,
            feature_type=FeatureType.LOG_DIFF,
            description='Volume 5d vs 60d average',
            formula='log_diff(vol_5d_avg, vol_60d_avg)',
            default=-0.3,
            range_min=-3.0,
            range_max=2.0,
            dormant_safe=False  # UNSTABLE when both ≈ 0
        ),
        # Index 17: OBV vs Price divergence (LOG-DIFF Jan 2026)
        FeatureSpec(
            name='obv_divergence',
            index=17,
            feature_type=FeatureType.LOG_DIFF,
            description='OBV change vs Price change (log-scaled)',
            formula='sign(obv_delta)*log1p(|obv_delta|) - sign(price_delta)*log1p(|price_delta|)',
            default=0.0,
            range_min=-5.0,
            range_max=5.0,
            dormant_safe=True  # FIXED: Returns 0.0 when cumulative_volume == 0
        ),
        # Index 18: Liquidity Fragility - Volume Decay Slope (Jan 2026)
        FeatureSpec(
            name='vol_decay_slope',
            index=18,
            feature_type=FeatureType.SLOPE,
            description='Volume trend over 20 days (negative = declining liquidity)',
            formula='linregress(log1p(volume[-20:]), x=[0..19]).slope',
            default=0.0,
            range_min=-0.5,
            range_max=0.5,
            dormant_safe=True  # Returns 0 when volume is flat/zero
        ),
        # =====================================================================
        # REGIME-AWARE FEATURES (Jan 2026 - Cross-Regime Generalization)
        # =====================================================================
        # These features capture market-wide conditions that affect breakout
        # success rates differently in bull vs bear vs volatile markets.
        # =====================================================================
        # Index 19: VIX Regime Level - Current volatility environment
        FeatureSpec(
            name='vix_regime_level',
            index=19,
            feature_type=FeatureType.BOUNDED,
            description='VIX normalized to regime level (0=calm, 1=crisis)',
            formula='clip((vix - 10) / 40, 0, 1)',
            default=0.25,
            range_min=0.0,
            range_max=1.0,
            dormant_safe=True  # External data, not stock-specific
        ),
        # Index 20: VIX Trend 20D - Volatility direction
        FeatureSpec(
            name='vix_trend_20d',
            index=20,
            feature_type=FeatureType.SLOPE,
            description='VIX 20-day change (negative = vol falling, positive = vol rising)',
            formula='clip((vix_current / vix_20d_ago - 1), -0.5, 0.5)',
            default=0.0,
            range_min=-0.5,
            range_max=0.5,
            dormant_safe=True
        ),
        # Index 21: Market Breadth 200 - % stocks above 200 SMA
        FeatureSpec(
            name='market_breadth_200',
            index=21,
            feature_type=FeatureType.BOUNDED,
            description='Percentage of universe above 200 SMA (0=narrow, 1=broad)',
            formula='pct_above_sma200 / 100',
            default=0.5,
            range_min=0.0,
            range_max=1.0,
            dormant_safe=True
        ),
        # Index 22: Risk-On Indicator - Risk appetite composite
        FeatureSpec(
            name='risk_on_indicator',
            index=22,
            feature_type=FeatureType.BOUNDED,
            description='Risk-on/off composite (0=risk-off, 0.5=neutral, 1=risk-on)',
            formula='(spy-tlt) + (jnk-tlt) - (gld-spy), normalized',
            default=0.5,
            range_min=0.0,
            range_max=1.0,
            dormant_safe=True
        ),
        # Index 23: Days Since Regime Change - Regime maturity
        FeatureSpec(
            name='days_since_regime_change',
            index=23,
            feature_type=FeatureType.BOUNDED,
            description='Log-normalized days since last bull/bear crossover',
            formula='log1p(days) / log1p(500)',
            default=0.5,
            range_min=0.0,
            range_max=1.0,
            dormant_safe=True
        ),
    ]

    # Build lookup dictionaries at class definition time
    _by_name: Dict[str, FeatureSpec] = {f.name: f for f in FEATURES}
    _by_index: Dict[int, FeatureSpec] = {f.index: f for f in FEATURES}

    # =========================================================================
    # CLASS METHODS - Feature Lookup
    # =========================================================================

    @classmethod
    def get(cls, name: str) -> FeatureSpec:
        """Get feature specification by name."""
        if name not in cls._by_name:
            raise KeyError(f"Unknown feature: '{name}'. Valid: {list(cls._by_name.keys())}")
        return cls._by_name[name]

    @classmethod
    def get_by_index(cls, index: int) -> FeatureSpec:
        """Get feature specification by index."""
        if index not in cls._by_index:
            raise KeyError(f"Invalid index: {index}. Valid: 0-{len(cls.FEATURES)-1}")
        return cls._by_index[index]

    @classmethod
    def get_name(cls, index: int) -> str:
        """Get feature name by index."""
        return cls.get_by_index(index).name

    @classmethod
    def get_index(cls, name: str) -> int:
        """Get feature index by name."""
        return cls.get(name).index

    @classmethod
    def get_default(cls, name: str) -> float:
        """Get default value for a feature."""
        return cls.get(name).default

    @classmethod
    def get_range(cls, name: str) -> tuple:
        """Get (min, max) range for a feature."""
        f = cls.get(name)
        return (f.range_min, f.range_max)

    # =========================================================================
    # CLASS METHODS - Feature Type Queries
    # =========================================================================

    @classmethod
    def get_by_type(cls, feature_type: FeatureType) -> List[FeatureSpec]:
        """Get all features of a specific type."""
        return [f for f in cls.FEATURES if f.feature_type == feature_type]

    @classmethod
    def get_indices_by_type(cls, feature_type: FeatureType) -> List[int]:
        """Get indices of all features of a specific type."""
        return [f.index for f in cls.FEATURES if f.feature_type == feature_type]

    @classmethod
    def get_names_by_type(cls, feature_type: FeatureType) -> List[str]:
        """Get names of all features of a specific type."""
        return [f.name for f in cls.FEATURES if f.feature_type == feature_type]

    @classmethod
    def needs_log_diff(cls, name: str) -> bool:
        """Check if a feature requires log-diff transformation."""
        return cls.get(name).feature_type == FeatureType.LOG_DIFF

    @classmethod
    def is_dormant_safe(cls, name: str) -> bool:
        """Check if a feature is stable for zero-volume stocks."""
        return cls.get(name).dormant_safe

    @classmethod
    def get_dormant_unsafe_features(cls) -> List[FeatureSpec]:
        """Get features that are unstable for dormant stocks."""
        return [f for f in cls.FEATURES if not f.dormant_safe]

    # =========================================================================
    # CLASS METHODS - Convenience Accessors
    # =========================================================================

    @classmethod
    def log_diff_indices(cls) -> List[int]:
        """Get indices of all LOG_DIFF features (volume ratios)."""
        return cls.get_indices_by_type(FeatureType.LOG_DIFF)

    @classmethod
    def bounded_indices(cls) -> List[int]:
        """Get indices of all BOUNDED features (no transformation needed)."""
        return cls.get_indices_by_type(FeatureType.BOUNDED)

    @classmethod
    def all_names(cls) -> List[str]:
        """Get all feature names in schema order."""
        return [f.name for f in cls.FEATURES]

    @classmethod
    def num_features(cls) -> int:
        """Get total number of features."""
        return len(cls.FEATURES)

    # =========================================================================
    # CLASS METHODS - Validation
    # =========================================================================

    @classmethod
    def validate_array(cls, context: np.ndarray) -> bool:
        """
        Validate context array shape matches registry.

        Args:
            context: Context array of shape (batch_size, n_features) or (n_features,)

        Returns:
            True if valid

        Raises:
            ValueError: If shape doesn't match
        """
        expected = cls.num_features()
        if context.ndim == 1:
            actual = context.shape[0]
        elif context.ndim == 2:
            actual = context.shape[1]
        else:
            raise ValueError(f"Context must be 1D or 2D, got {context.ndim}D")

        if actual != expected:
            raise ValueError(
                f"Context has {actual} features, expected {expected}. "
                f"Features: {cls.all_names()}"
            )
        return True

    @classmethod
    def describe(cls) -> str:
        """Return human-readable description of all features."""
        lines = [
            "TRAnS Context Feature Registry",
            "=" * 60,
            f"Total Features: {cls.num_features()}",
            "",
        ]

        # Group by type
        for ftype in FeatureType:
            features = cls.get_by_type(ftype)
            if features:
                lines.append(f"\n{ftype.value.upper()} ({len(features)} features):")
                lines.append("-" * 40)
                for f in features:
                    safe = "✓" if f.dormant_safe else "⚠"
                    lines.append(f"  {f.index:2d}. {f.name:<28} {safe}")
                    lines.append(f"      {f.description}")

        lines.append("\n" + "=" * 60)
        lines.append("Legend: ✓ = Dormant-safe | ⚠ = Needs volume floor")

        return "\n".join(lines)


# =============================================================================
# BACKWARDS COMPATIBILITY EXPORTS
# =============================================================================
# These match the existing imports from context_features.py

# List of volume ratio feature indices (LOG_DIFF type)
VOLUME_RATIO_INDICES: Final[List[int]] = FeatureRegistry.log_diff_indices()

# List of volume ratio feature names
VOLUME_RATIO_FEATURES: Final[List[str]] = FeatureRegistry.get_names_by_type(FeatureType.LOG_DIFF)


# =============================================================================
# DOLLAR BAR FEATURES (Jan 2026)
# =============================================================================
# These features are specific to dollar bar sampling (Lopez de Prado methodology).
# They replace time-based volume features when bar_type='dollar' is used.
# =============================================================================

@dataclass
class DollarBarFeatureSpec:
    """Feature specification for dollar bar specific features."""
    name: str
    index: int  # Offset from DOLLAR_BAR_FEATURE_START_INDEX
    description: str
    formula: str
    default: float
    range_min: float
    range_max: float


# Dollar bar feature definitions
# These are added to sequences when bar_type='dollar'
DOLLAR_BAR_FEATURES: Final[List[DollarBarFeatureSpec]] = [
    DollarBarFeatureSpec(
        name='bars_per_day',
        index=0,
        description='Activity intensity: count(bars) / count(calendar_days)',
        formula='1.0 / num_days_in_bar',
        default=1.0,
        range_min=0.1,
        range_max=10.0
    ),
    DollarBarFeatureSpec(
        name='time_between_bars',
        index=1,
        description='Dormancy detection: log-scaled bar duration',
        formula='log1p(bar_duration_hours)',
        default=3.0,  # log1p(24) ≈ 3.2 for 1-day bar
        range_min=0.0,
        range_max=8.0  # log1p(3000) ≈ 8 for multi-week bar
    ),
    DollarBarFeatureSpec(
        name='volume_vs_threshold',
        index=2,
        description='Volume deviation from threshold (should be ~1.0)',
        formula='bar_dollar_volume / threshold',
        default=1.0,
        range_min=0.5,
        range_max=2.0
    ),
]

# Starting index for dollar bar features in context vector
# These are added after the standard 24 context features
DOLLAR_BAR_FEATURE_START_INDEX: Final[int] = 24


class DollarBarFeatureRegistry:
    """
    Registry for dollar bar specific features.

    These features replace or supplement time-based features when
    using dollar bar sampling instead of daily bars.

    Usage:
        from config.feature_registry import DollarBarFeatureRegistry

        # Get all dollar bar feature names
        names = DollarBarFeatureRegistry.all_names()

        # Get index for a specific feature
        idx = DollarBarFeatureRegistry.get_index('bars_per_day')
    """

    _features = {f.name: f for f in DOLLAR_BAR_FEATURES}
    _by_index = {f.index: f for f in DOLLAR_BAR_FEATURES}

    @classmethod
    def get(cls, name: str) -> DollarBarFeatureSpec:
        """Get feature specification by name."""
        if name not in cls._features:
            raise KeyError(f"Unknown dollar bar feature: '{name}'")
        return cls._features[name]

    @classmethod
    def get_by_index(cls, index: int) -> DollarBarFeatureSpec:
        """Get feature specification by index."""
        if index not in cls._by_index:
            raise KeyError(f"Invalid dollar bar feature index: {index}")
        return cls._by_index[index]

    @classmethod
    def get_index(cls, name: str) -> int:
        """Get absolute index (including offset) for a feature."""
        return DOLLAR_BAR_FEATURE_START_INDEX + cls.get(name).index

    @classmethod
    def all_names(cls) -> List[str]:
        """Get all dollar bar feature names."""
        return [f.name for f in DOLLAR_BAR_FEATURES]

    @classmethod
    def num_features(cls) -> int:
        """Get number of dollar bar features."""
        return len(DOLLAR_BAR_FEATURES)

    @classmethod
    def get_defaults(cls) -> Dict[str, float]:
        """Get default values for all dollar bar features."""
        return {f.name: f.default for f in DOLLAR_BAR_FEATURES}

    @classmethod
    def get_ranges(cls) -> Dict[str, Tuple[float, float]]:
        """Get (min, max) ranges for all dollar bar features."""
        return {f.name: (f.range_min, f.range_max) for f in DOLLAR_BAR_FEATURES}

    @classmethod
    def describe(cls) -> str:
        """Return human-readable description of dollar bar features."""
        lines = [
            "Dollar Bar Features",
            "=" * 50,
            f"Start index: {DOLLAR_BAR_FEATURE_START_INDEX}",
            f"Number of features: {cls.num_features()}",
            "",
        ]

        for f in DOLLAR_BAR_FEATURES:
            abs_idx = DOLLAR_BAR_FEATURE_START_INDEX + f.index
            lines.append(f"{abs_idx}. {f.name}")
            lines.append(f"   {f.description}")
            lines.append(f"   Range: [{f.range_min}, {f.range_max}], Default: {f.default}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print(FeatureRegistry.describe())
    print()
    print(f"LOG_DIFF indices: {VOLUME_RATIO_INDICES}")
    print(f"LOG_DIFF features: {VOLUME_RATIO_FEATURES}")
    print()
    print("Dormant-unsafe features:")
    for f in FeatureRegistry.get_dormant_unsafe_features():
        print(f"  {f.index}: {f.name}")
