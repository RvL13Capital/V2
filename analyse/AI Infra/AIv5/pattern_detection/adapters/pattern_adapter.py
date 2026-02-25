"""
Pattern Adapter - Bridges old class-based and new dataclass-based Pattern models.

Provides conversion methods to transform between AIv3's class-based ConsolidationPattern
and AIv4's modern dataclass-based ConsolidationPattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pattern_detection.models import (
    ConsolidationPattern as NewPattern,
    PatternPhase,
    PatternBoundaries,
    RecentMetrics,
    BaselineMetrics,
    CompressionMetrics,
    PricePosition,
)

if TYPE_CHECKING:
    from pattern_detection.state_machine.consolidation_tracker import (
        ConsolidationPattern as OldPattern,
    )


class PatternAdapter:
    """
    Adapter to convert between old and new Pattern representations.

    Provides bidirectional conversion:
    - from_old: Converts AIv3 class-based pattern → AIv4 dataclass pattern
    - to_old: Converts AIv4 dataclass pattern → AIv3 class-based pattern (if needed)
    """

    @staticmethod
    def from_old(old_pattern: 'OldPattern') -> NewPattern:
        """
        Convert old class-based pattern to new dataclass pattern.

        Args:
            old_pattern: AIv3 ConsolidationPattern instance

        Returns:
            NewPattern: AIv4 dataclass-based pattern
        """
        # Create boundaries if they exist
        boundaries = None
        if old_pattern.upper_boundary is not None:
            boundaries = PatternBoundaries(
                upper=old_pattern.upper_boundary,
                lower=old_pattern.lower_boundary,
                power=old_pattern.power_boundary,
                range_pct=old_pattern.boundary_range_pct,
            )

        # Create recent metrics
        recent_metrics = RecentMetrics(
            avg_bbw=old_pattern.avg_bbw_20d,
            avg_adx=old_pattern.avg_adx_20d,
            avg_volume_ratio=old_pattern.avg_volume_ratio_20d,
            avg_range_ratio=old_pattern.avg_range_ratio_20d,
            bbw_slope=old_pattern.bbw_slope_20d,
            volume_slope=old_pattern.volume_slope_20d,
            adx_slope=old_pattern.adx_slope_20d,
            bbw_std=old_pattern.bbw_std_20d,
            volume_std=old_pattern.volume_std_20d,
            price_volatility=None,  # Not in old pattern
        )

        # Create baseline metrics
        baseline_metrics = BaselineMetrics(
            bbw_avg=old_pattern.baseline_bbw_avg,
            adx_avg=old_pattern.baseline_adx_avg,
            volume_avg=old_pattern.baseline_volume_avg,
            range_avg=old_pattern.baseline_range_avg,
            volatility=old_pattern.baseline_volatility,
            bbw_std=old_pattern.baseline_bbw_std,
            volume_std=old_pattern.baseline_volume_std,
        )

        # Create compression metrics
        compression_metrics = CompressionMetrics(
            bbw_compression=old_pattern.bbw_compression_ratio,
            volume_compression=old_pattern.volume_compression_ratio,
            range_compression=old_pattern.range_compression_ratio,
            volatility_stability=old_pattern.volatility_stability_ratio,
            overall_compression=old_pattern.overall_compression,
        )

        # Create price position
        price_position = PricePosition(
            position_in_range=old_pattern.price_position_in_range,
            distance_from_upper_pct=old_pattern.price_distance_from_upper_pct,
            distance_from_lower_pct=getattr(old_pattern, 'price_distance_from_lower_pct', None),
        )

        # Create new pattern
        new_pattern = NewPattern(
            ticker=old_pattern.ticker,
            start_date=old_pattern.start_date,
            start_idx=old_pattern.start_idx,
            start_price=old_pattern.start_price,
            phase=PatternPhase(old_pattern.phase.value),
        )

        # Set optional fields
        new_pattern.end_date = old_pattern.end_date
        new_pattern.end_idx = old_pattern.end_idx
        new_pattern.end_price = old_pattern.end_price

        new_pattern.boundaries = boundaries
        new_pattern.days_since_activation = old_pattern.days_since_activation
        new_pattern.total_days_in_pattern = old_pattern.total_days_in_pattern
        new_pattern.days_qualifying = old_pattern.days_qualifying

        new_pattern.activation_date = old_pattern.activation_date
        new_pattern.activation_idx = old_pattern.activation_idx

        new_pattern.qualification_highs = old_pattern.qualification_highs.copy() if old_pattern.qualification_highs else []
        new_pattern.qualification_lows = old_pattern.qualification_lows.copy() if old_pattern.qualification_lows else []

        new_pattern.recent_metrics = recent_metrics
        new_pattern.baseline_metrics = baseline_metrics
        new_pattern.compression_metrics = compression_metrics
        new_pattern.price_position = price_position

        # Outcome
        new_pattern.breakout_date = old_pattern.breakout_date
        new_pattern.breakout_direction = old_pattern.breakout_direction
        new_pattern.max_gain = old_pattern.max_gain

        # Note: feature_snapshots are not migrated as they use a different structure
        # The new pattern will build snapshots using FeatureSnapshot dataclass

        return new_pattern

    @staticmethod
    def to_dict(pattern: NewPattern) -> dict:
        """
        Convert new pattern to dictionary (for backwards compatibility).

        Args:
            pattern: AIv4 dataclass pattern

        Returns:
            dict: Dictionary representation matching old format
        """
        return pattern.to_dict()

    @staticmethod
    def update_metrics_from_dict(pattern: NewPattern, metrics: dict) -> None:
        """
        Update pattern metrics from a dictionary (helper for incremental updates).

        Args:
            pattern: Pattern to update
            metrics: Dictionary of metric values
        """
        # Update recent metrics if provided
        if any(k in metrics for k in ['avg_bbw_20d', 'avg_adx_20d', 'bbw_slope_20d']):
            if pattern.recent_metrics is None:
                pattern.recent_metrics = RecentMetrics()

            for key in ['avg_bbw', 'avg_adx', 'avg_volume_ratio', 'avg_range_ratio',
                       'bbw_slope', 'volume_slope', 'adx_slope', 'bbw_std', 'volume_std']:
                old_key = f"{key}_20d" if not key.endswith('_20d') else key
                if old_key in metrics:
                    setattr(pattern.recent_metrics, key, metrics[old_key])

        # Update baseline metrics if provided
        if any(k in metrics for k in ['baseline_bbw_avg', 'baseline_adx_avg']):
            if pattern.baseline_metrics is None:
                pattern.baseline_metrics = BaselineMetrics()

            for key in ['bbw_avg', 'adx_avg', 'volume_avg', 'range_avg',
                       'volatility', 'bbw_std', 'volume_std']:
                dict_key = f"baseline_{key}"
                if dict_key in metrics:
                    setattr(pattern.baseline_metrics, key, metrics[dict_key])

        # Update compression metrics if provided
        if any(k in metrics for k in ['bbw_compression_ratio', 'overall_compression']):
            if pattern.compression_metrics is None:
                pattern.compression_metrics = CompressionMetrics()

            for key in ['bbw_compression', 'volume_compression', 'range_compression',
                       'volatility_stability', 'overall_compression']:
                dict_key = f"{key}_ratio" if key != 'overall_compression' else key
                if dict_key in metrics:
                    setattr(pattern.compression_metrics, key, metrics[dict_key])

        # Update price position if provided
        if any(k in metrics for k in ['price_position_in_range', 'price_distance_from_upper_pct']):
            if pattern.price_position is None:
                pattern.price_position = PricePosition()

            for key in ['position_in_range', 'distance_from_upper_pct', 'distance_from_lower_pct']:
                dict_key = f"price_{key}"
                if dict_key in metrics:
                    setattr(pattern.price_position, key, metrics[dict_key])
