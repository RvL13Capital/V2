"""
Modernized Pattern Model for AIv4

Uses dataclasses for clean, type-safe pattern representation.
Replaces AIv3's class-based pattern with modern Python idioms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class PatternPhase(str, Enum):
    """Pattern lifecycle phases."""
    NONE = "NONE"
    QUALIFYING = "QUALIFYING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class PatternBoundaries:
    """Fixed pattern boundaries (set once during qualification)."""
    upper: float
    lower: float
    power: float  # upper * 1.005
    range_pct: float  # (upper - lower) / lower


@dataclass
class RecentMetrics:
    """Recent window metrics (last 20 days) - updated daily."""
    # Averages
    avg_bbw: Optional[float] = None
    avg_adx: Optional[float] = None
    avg_volume_ratio: Optional[float] = None
    avg_range_ratio: Optional[float] = None

    # Trends (slopes)
    bbw_slope: Optional[float] = None
    volume_slope: Optional[float] = None
    adx_slope: Optional[float] = None

    # Stability (standard deviations)
    bbw_std: Optional[float] = None
    volume_std: Optional[float] = None
    price_volatility: Optional[float] = None


@dataclass
class BaselineMetrics:
    """Baseline window metrics (days -50 to -20)."""
    bbw_avg: Optional[float] = None
    adx_avg: Optional[float] = None
    volume_avg: Optional[float] = None
    range_avg: Optional[float] = None
    volatility: Optional[float] = None
    bbw_std: Optional[float] = None
    volume_std: Optional[float] = None


@dataclass
class CompressionMetrics:
    """Compression ratios (recent / baseline)."""
    bbw_compression: Optional[float] = None  # < 1.0 = tightening
    volume_compression: Optional[float] = None  # < 1.0 = quieter
    range_compression: Optional[float] = None  # < 1.0 = tighter range
    volatility_stability: Optional[float] = None  # < 1.0 = more stable
    overall_compression: Optional[float] = None  # Combined score


@dataclass
class PricePosition:
    """Price position within pattern boundaries."""
    position_in_range: Optional[float] = None  # 0.0 (lower) to 1.0 (upper)
    distance_from_upper_pct: Optional[float] = None  # % from breakout
    distance_from_lower_pct: Optional[float] = None  # % from breakdown


@dataclass
class FeatureSnapshot:
    """A single snapshot of pattern features at a specific date."""
    snapshot_date: datetime
    days_since_activation: int
    recent: RecentMetrics
    compression: CompressionMetrics
    price_position: PricePosition

    def to_dict(self) -> dict:
        """Convert to flat dictionary for training."""
        return {
            'snapshot_date': self.snapshot_date,
            'days_since_activation': self.days_since_activation,

            # Recent metrics
            'avg_bbw_20d': self.recent.avg_bbw,
            'avg_adx_20d': self.recent.avg_adx,
            'avg_volume_ratio_20d': self.recent.avg_volume_ratio,
            'avg_range_ratio_20d': self.recent.avg_range_ratio,
            'bbw_slope_20d': self.recent.bbw_slope,
            'volume_slope_20d': self.recent.volume_slope,
            'adx_slope_20d': self.recent.adx_slope,
            'bbw_std_20d': self.recent.bbw_std,
            'volume_std_20d': self.recent.volume_std,
            'price_volatility_20d': self.recent.price_volatility,

            # Compression metrics
            'bbw_compression_ratio': self.compression.bbw_compression,
            'volume_compression_ratio': self.compression.volume_compression,
            'range_compression_ratio': self.compression.range_compression,
            'volatility_stability_ratio': self.compression.volatility_stability,
            'overall_compression': self.compression.overall_compression,

            # Price position
            'price_position_in_range': self.price_position.position_in_range,
            'price_distance_from_upper_pct': self.price_position.distance_from_upper_pct,
            'price_distance_from_lower_pct': self.price_position.distance_from_lower_pct,
        }


@dataclass
class ConsolidationPattern:
    """
    Represents a detected consolidation pattern.

    Modernized with dataclasses for:
    - Type safety
    - Automatic __init__, __repr__, __eq__
    - Cleaner code
    - Better IDE support
    """
    # Basic pattern info
    ticker: str
    start_date: datetime
    start_idx: int
    start_price: float

    # Pattern lifecycle
    phase: PatternPhase = PatternPhase.QUALIFYING
    end_date: Optional[datetime] = None
    end_idx: Optional[int] = None
    end_price: Optional[float] = None

    # Temporal tracking (LEAK-FREE)
    days_since_activation: int = 0  # Always known (current day)
    days_qualifying: int = 0

    # Activation tracking
    activation_date: Optional[datetime] = None
    activation_idx: Optional[int] = None

    # Qualification period tracking
    qualification_highs: list[float] = field(default_factory=list)
    qualification_lows: list[float] = field(default_factory=list)

    # Pattern boundaries (set once during qualification)
    boundaries: Optional[PatternBoundaries] = None

    # Feature snapshots (multiple per pattern)
    snapshots: list[FeatureSnapshot] = field(default_factory=list)

    # Current metrics (updated daily)
    recent_metrics: RecentMetrics = field(default_factory=RecentMetrics)
    baseline_metrics: BaselineMetrics = field(default_factory=BaselineMetrics)
    compression_metrics: CompressionMetrics = field(default_factory=CompressionMetrics)
    price_position: PricePosition = field(default_factory=PricePosition)

    # Outcome (for historical patterns only)
    breakout_date: Optional[datetime] = None
    breakout_direction: Optional[str] = None  # 'UP' or 'DOWN'
    max_gain: Optional[float] = None

    # Analysis-only fields (LEAKED - not for training!)
    total_days_in_pattern: int = 0  # Only known after completion

    def set_boundaries(self) -> None:
        """Set fixed pattern boundaries from qualification highs/lows."""
        if not self.qualification_highs or not self.qualification_lows:
            raise ValueError("Cannot set boundaries without qualification data")

        upper = max(self.qualification_highs)
        lower = min(self.qualification_lows)
        power = upper * 1.005
        range_pct = (upper - lower) / lower if lower > 0 else 0

        self.boundaries = PatternBoundaries(
            upper=upper,
            lower=lower,
            power=power,
            range_pct=range_pct
        )

    def take_snapshot(self, date: datetime) -> FeatureSnapshot:
        """Create a feature snapshot at the current state."""
        snapshot = FeatureSnapshot(
            snapshot_date=date,
            days_since_activation=self.days_since_activation,
            recent=self.recent_metrics,
            compression=self.compression_metrics,
            price_position=self.price_position,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def to_training_samples(self) -> list[dict]:
        """
        Convert pattern to multiple training samples (one per snapshot).

        NOTE: Outcomes are NOT included here - they will be calculated
        per-snapshot during labeling, each with its own 100-day window
        starting from snapshot_date (not pattern end_date).
        """
        samples = []

        for snapshot in self.snapshots:
            sample = {
                # Pattern identification
                'ticker': self.ticker,
                'pattern_id': f"{self.ticker}_{self.start_date.strftime('%Y-%m-%d')}",

                # Pattern dates
                'start_date': self.start_date,
                'end_date': self.end_date,  # Pattern end (for reference)
                'snapshot_date': snapshot.snapshot_date,  # THIS date used for outcome evaluation
                'snapshot_idx': snapshot.days_since_activation,

                # Pattern boundaries (static)
                'upper_boundary': self.boundaries.upper if self.boundaries else None,
                'lower_boundary': self.boundaries.lower if self.boundaries else None,
                'power_boundary': self.boundaries.power if self.boundaries else None,
                'boundary_range_pct': self.boundaries.range_pct if self.boundaries else None,

                # Temporal features (leak-free)
                'days_since_activation': snapshot.days_since_activation,

                # All snapshot features
                **snapshot.to_dict(),
            }
            samples.append(sample)

        return samples

    def to_dict(self) -> dict:
        """Convert full pattern to dictionary."""
        return {
            'ticker': self.ticker,
            'start_date': self.start_date,
            'start_idx': self.start_idx,
            'start_price': self.start_price,
            'phase': self.phase.value,
            'end_date': self.end_date,
            'end_idx': self.end_idx,
            'end_price': self.end_price,
            'days_since_activation': self.days_since_activation,
            'days_qualifying': self.days_qualifying,
            'activation_date': self.activation_date,
            'activation_idx': self.activation_idx,
            'boundaries': {
                'upper': self.boundaries.upper,
                'lower': self.boundaries.lower,
                'power': self.boundaries.power,
                'range_pct': self.boundaries.range_pct,
            } if self.boundaries else None,
            'num_snapshots': len(self.snapshots),
            'breakout_date': self.breakout_date,
            'breakout_direction': self.breakout_direction,
            'max_gain': self.max_gain,
            'total_days_in_pattern': self.total_days_in_pattern,  # Analysis only!
        }
