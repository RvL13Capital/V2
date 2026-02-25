"""
Tests for modernized Pattern dataclass model.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from pattern_detection.models import (
    ConsolidationPattern,
    PatternPhase,
    PatternBoundaries,
    RecentMetrics,
    FeatureSnapshot,
)


class TestConsolidationPattern:
    """Test ConsolidationPattern dataclass."""

    def test_initialization(self):
        """Test pattern initialization with required fields."""
        pattern = ConsolidationPattern(
            ticker="AAPL",
            start_date=datetime(2024, 1, 1),
            start_idx=0,
            start_price=150.0
        )

        assert pattern.ticker == "AAPL"
        assert pattern.start_date == datetime(2024, 1, 1)
        assert pattern.start_idx == 0
        assert pattern.start_price == 150.0
        assert pattern.phase == PatternPhase.QUALIFYING
        assert pattern.days_since_activation == 0
        assert len(pattern.snapshots) == 0

    def test_set_boundaries(self, sample_pattern):
        """Test setting fixed boundaries from qualification data."""
        # Add qualification data
        sample_pattern.qualification_highs = [102, 103, 104, 103, 102]
        sample_pattern.qualification_lows = [98, 97, 96, 97, 98]

        # Set boundaries
        sample_pattern.set_boundaries()

        assert sample_pattern.boundaries is not None
        assert sample_pattern.boundaries.upper == 104.0  # max(highs)
        assert sample_pattern.boundaries.lower == 96.0   # min(lows)
        assert sample_pattern.boundaries.power == pytest.approx(104.52, rel=1e-2)  # 104 * 1.005
        assert sample_pattern.boundaries.range_pct == pytest.approx(0.0833, rel=1e-2)  # (104-96)/96

    def test_set_boundaries_without_data(self, sample_pattern):
        """Test setting boundaries fails without qualification data."""
        with pytest.raises(ValueError, match="Cannot set boundaries"):
            sample_pattern.set_boundaries()

    def test_take_snapshot(self, active_pattern):
        """Test taking a feature snapshot."""
        from pattern_detection.models import RecentMetrics, CompressionMetrics, PricePosition

        # Set up metrics
        active_pattern.recent_metrics = RecentMetrics(
            avg_bbw=0.012,
            avg_adx=28.0,
            bbw_slope=-0.001
        )
        active_pattern.compression_metrics = CompressionMetrics(
            bbw_compression=0.85,
            overall_compression=0.90
        )
        active_pattern.price_position = PricePosition(
            position_in_range=0.5,
            distance_from_upper_pct=0.02
        )

        # Take snapshot
        snapshot = active_pattern.take_snapshot(datetime(2024, 1, 15))

        assert snapshot.snapshot_date == datetime(2024, 1, 15)
        assert snapshot.days_since_activation == active_pattern.days_since_activation
        assert snapshot.recent.avg_bbw == 0.012
        assert len(active_pattern.snapshots) == 1

    def test_to_training_samples_empty(self, sample_pattern):
        """Test to_training_samples with no snapshots."""
        samples = sample_pattern.to_training_samples()
        assert samples == []

    def test_to_training_samples_with_snapshots(self, active_pattern):
        """Test to_training_samples creates samples from snapshots."""
        from pattern_detection.models import RecentMetrics, CompressionMetrics, PricePosition

        # Add metrics
        active_pattern.recent_metrics = RecentMetrics(avg_bbw=0.012, avg_adx=28.0)
        active_pattern.compression_metrics = CompressionMetrics(bbw_compression=0.85)
        active_pattern.price_position = PricePosition(position_in_range=0.5)

        # Take multiple snapshots
        active_pattern.take_snapshot(datetime(2024, 1, 15))
        active_pattern.days_since_activation = 20
        active_pattern.take_snapshot(datetime(2024, 1, 30))

        # Convert to training samples
        samples = active_pattern.to_training_samples()

        assert len(samples) == 2
        assert samples[0]['ticker'] == 'TEST'
        assert samples[0]['snapshot_date'] == datetime(2024, 1, 15)
        assert samples[1]['snapshot_date'] == datetime(2024, 1, 30)
        assert 'upper_boundary' in samples[0]
        assert 'avg_bbw_20d' in samples[0]

    def test_to_dict(self, active_pattern):
        """Test converting pattern to dictionary."""
        pattern_dict = active_pattern.to_dict()

        assert pattern_dict['ticker'] == 'TEST'
        assert pattern_dict['phase'] == 'ACTIVE'
        assert pattern_dict['boundaries'] is not None
        assert pattern_dict['boundaries']['upper'] == 104.0
        assert pattern_dict['num_snapshots'] == 0


class TestPatternPhase:
    """Test PatternPhase enum."""

    def test_enum_values(self):
        """Test enum has correct values."""
        assert PatternPhase.NONE.value == "NONE"
        assert PatternPhase.QUALIFYING.value == "QUALIFYING"
        assert PatternPhase.ACTIVE.value == "ACTIVE"
        assert PatternPhase.COMPLETED.value == "COMPLETED"
        assert PatternPhase.FAILED.value == "FAILED"

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert PatternPhase.ACTIVE == PatternPhase.ACTIVE
        assert PatternPhase.ACTIVE != PatternPhase.QUALIFYING


class TestPatternBoundaries:
    """Test PatternBoundaries dataclass."""

    def test_boundaries_creation(self):
        """Test creating boundaries."""
        boundaries = PatternBoundaries(
            upper=105.0,
            lower=95.0,
            power=105.525,  # 105 * 1.005
            range_pct=0.105  # (105-95)/95
        )

        assert boundaries.upper == 105.0
        assert boundaries.lower == 95.0
        assert boundaries.power == 105.525
        assert boundaries.range_pct == pytest.approx(0.105, rel=1e-3)


class TestRecentMetrics:
    """Test RecentMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating recent metrics."""
        metrics = RecentMetrics(
            avg_bbw=0.012,
            avg_adx=28.5,
            bbw_slope=-0.001,
            volume_std=0.15
        )

        assert metrics.avg_bbw == 0.012
        assert metrics.avg_adx == 28.5
        assert metrics.bbw_slope == -0.001
        assert metrics.volume_std == 0.15

    def test_metrics_defaults(self):
        """Test metrics default to None."""
        metrics = RecentMetrics()

        assert metrics.avg_bbw is None
        assert metrics.avg_adx is None
        assert metrics.bbw_slope is None


class TestFeatureSnapshot:
    """Test FeatureSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating feature snapshot."""
        recent = RecentMetrics(avg_bbw=0.012, avg_adx=28.0)
        from pattern_detection.models import CompressionMetrics, PricePosition

        compression = CompressionMetrics(bbw_compression=0.85)
        position = PricePosition(position_in_range=0.5)

        snapshot = FeatureSnapshot(
            snapshot_date=datetime(2024, 1, 15),
            days_since_activation=5,
            recent=recent,
            compression=compression,
            price_position=position
        )

        assert snapshot.snapshot_date == datetime(2024, 1, 15)
        assert snapshot.days_since_activation == 5
        assert snapshot.recent.avg_bbw == 0.012

    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        recent = RecentMetrics(avg_bbw=0.012, avg_adx=28.0, bbw_slope=-0.001)
        from pattern_detection.models import CompressionMetrics, PricePosition

        compression = CompressionMetrics(
            bbw_compression=0.85,
            overall_compression=0.90
        )
        position = PricePosition(
            position_in_range=0.5,
            distance_from_upper_pct=0.02
        )

        snapshot = FeatureSnapshot(
            snapshot_date=datetime(2024, 1, 15),
            days_since_activation=5,
            recent=recent,
            compression=compression,
            price_position=position
        )

        snapshot_dict = snapshot.to_dict()

        assert 'snapshot_date' in snapshot_dict
        assert 'days_since_activation' in snapshot_dict
        assert snapshot_dict['avg_bbw_20d'] == 0.012
        assert snapshot_dict['avg_adx_20d'] == 28.0
        assert snapshot_dict['bbw_compression_ratio'] == 0.85
        assert snapshot_dict['price_position_in_range'] == 0.5
