"""
Tests for Modern Pattern Tracker.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pattern_detection.state_machine import ModernPatternTracker
from pattern_detection.models import PatternPhase
from shared.config import ConsolidationCriteria


class TestModernPatternTracker:
    """Test ModernPatternTracker state machine."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ModernPatternTracker('TEST')

        assert tracker.ticker == 'TEST'
        assert tracker.phase == PatternPhase.NONE
        assert tracker.current_pattern is None
        assert len(tracker.completed_patterns) == 0

    def test_start_qualification(self):
        """Test starting qualification phase."""
        tracker = ModernPatternTracker('TEST')

        # Create indicators that meet consolidation criteria
        indicators = {
            'bbw_percentile': 0.25,  # Below 0.30
            'adx': 28.0,  # Below 32
            'volume_ratio_20': 0.30,  # Below 0.35
            'range_ratio': 0.60,  # Below 0.65
        }

        row = pd.Series({
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000000,
        })

        # Process day
        result = tracker.process_day(
            date=datetime(2024, 1, 1),
            idx=0,
            row=row,
            indicators=indicators
        )

        # Should start qualifying
        assert tracker.phase == PatternPhase.QUALIFYING
        assert tracker.current_pattern is not None
        assert tracker.current_pattern.ticker == 'TEST'

    def test_failed_qualification(self):
        """Test failed qualification (criteria not met continuously)."""
        tracker = ModernPatternTracker('TEST')

        # Start qualification
        good_indicators = {
            'bbw_percentile': 0.25,
            'adx': 28.0,
            'volume_ratio_20': 0.30,
            'range_ratio': 0.60,
        }

        row = pd.Series({
            'open': 100.0, 'high': 101.0, 'low': 99.0,
            'close': 100.5, 'volume': 1000000,
        })

        tracker.process_day(datetime(2024, 1, 1), 0, row, good_indicators)
        assert tracker.phase == PatternPhase.QUALIFYING

        # Fail criteria on day 2
        bad_indicators = {
            'bbw_percentile': 0.50,  # Too high
            'adx': 45.0,  # Too high
            'volume_ratio_20': 0.80,  # Too high
            'range_ratio': 0.90,  # Too high
        }

        tracker.process_day(datetime(2024, 1, 2), 1, row, bad_indicators)

        # Should reset to NONE
        assert tracker.phase == PatternPhase.NONE
        assert tracker.current_pattern is None

    def test_successful_activation(self):
        """Test successful pattern activation after 10 days."""
        tracker = ModernPatternTracker('TEST')

        good_indicators = {
            'bbw_percentile': 0.25,
            'adx': 28.0,
            'volume_ratio_20': 0.30,
            'range_ratio': 0.60,
        }

        # Simulate 10 days of qualification (need 11 calls: day 0-10)
        for day in range(11):
            row = pd.Series({
                'open': 100.0 + day * 0.1,
                'high': 101.0 + day * 0.1,
                'low': 99.0 + day * 0.1,
                'close': 100.5 + day * 0.1,
                'volume': 1000000,
            })

            result = tracker.process_day(
                date=datetime(2024, 1, 1) + timedelta(days=day),
                idx=day,
                row=row,
                indicators=good_indicators
            )

        # Should be ACTIVE after 10 days
        assert tracker.phase == PatternPhase.ACTIVE
        assert tracker.current_pattern is not None
        assert tracker.current_pattern.phase == PatternPhase.ACTIVE
        assert tracker.current_pattern.boundaries is not None
        assert tracker.current_pattern.boundaries.upper > 0
        assert tracker.current_pattern.boundaries.lower > 0

    def test_breakout_detection(self):
        """Test power breakout detection."""
        tracker = ModernPatternTracker('TEST')

        # Activate pattern first
        self._activate_pattern(tracker)

        # Get power boundary
        power = tracker.current_pattern.boundaries.power

        # Create price that breaks power boundary
        row = pd.Series({
            'open': power - 1,
            'high': power + 1,  # Breaks power
            'low': 99.0,
            'close': power + 0.5,
            'volume': 2000000,
        })

        result = tracker.process_day(
            date=datetime(2024, 1, 15),
            idx=14,
            row=row,
            indicators={}
        )

        # Should complete pattern with UP breakout
        assert tracker.phase == PatternPhase.NONE  # Reset
        assert len(tracker.completed_patterns) == 1
        assert tracker.completed_patterns[0].phase == PatternPhase.COMPLETED
        assert tracker.completed_patterns[0].breakout_direction == 'UP'

    def test_breakdown_detection(self):
        """Test lower boundary breakdown detection."""
        tracker = ModernPatternTracker('TEST')

        # Activate pattern first
        self._activate_pattern(tracker)

        # Get lower boundary
        lower = tracker.current_pattern.boundaries.lower

        # Create price that breaks lower boundary
        row = pd.Series({
            'open': lower + 1,
            'high': 101.0,
            'low': lower - 1,  # Breaks lower
            'close': lower - 0.5,
            'volume': 2000000,
        })

        result = tracker.process_day(
            date=datetime(2024, 1, 15),
            idx=14,
            row=row,
            indicators={}
        )

        # Should complete pattern with DOWN breakdown
        assert tracker.phase == PatternPhase.NONE  # Reset
        assert len(tracker.completed_patterns) == 1
        assert tracker.completed_patterns[0].phase == PatternPhase.FAILED
        assert tracker.completed_patterns[0].breakout_direction == 'DOWN'

    def test_timeout_detection(self):
        """Test max duration timeout."""
        criteria = ConsolidationCriteria(max_pattern_days=5)
        tracker = ModernPatternTracker('TEST', criteria=criteria)

        # Activate pattern
        self._activate_pattern(tracker)

        # Simulate staying in range for max_pattern_days
        for day in range(5):
            row = pd.Series({
                'open': 100.0,
                'high': 100.5,
                'low': 99.5,
                'close': 100.0,
                'volume': 1000000,
            })

            result = tracker.process_day(
                date=datetime(2024, 1, 15) + timedelta(days=day),
                idx=14 + day,
                row=row,
                indicators={}
            )

        # Should timeout after max_pattern_days
        assert tracker.phase == PatternPhase.NONE
        assert len(tracker.completed_patterns) == 1
        assert tracker.completed_patterns[0].breakout_direction is None  # Timeout

    def test_reset(self):
        """Test tracker reset."""
        tracker = ModernPatternTracker('TEST')

        # Activate pattern
        self._activate_pattern(tracker)
        assert tracker.phase == PatternPhase.ACTIVE

        # Reset
        tracker.reset()

        assert tracker.phase == PatternPhase.NONE
        assert tracker.current_pattern is None

    def test_criteria_checking(self):
        """Test consolidation criteria evaluation."""
        tracker = ModernPatternTracker('TEST')

        # 3 of 4 criteria met - should qualify
        indicators_3_of_4 = {
            'bbw_percentile': 0.25,  # ✓
            'adx': 28.0,  # ✓
            'volume_ratio_20': 0.30,  # ✓
            'range_ratio': 0.70,  # ✗ (too high)
        }

        assert tracker._meets_consolidation_criteria(indicators_3_of_4)

        # Only 2 of 4 criteria met - should not qualify
        indicators_2_of_4 = {
            'bbw_percentile': 0.25,  # ✓
            'adx': 28.0,  # ✓
            'volume_ratio_20': 0.80,  # ✗
            'range_ratio': 0.90,  # ✗
        }

        assert not tracker._meets_consolidation_criteria(indicators_2_of_4)

    # Helper methods

    def _activate_pattern(self, tracker: ModernPatternTracker):
        """Helper to activate a pattern."""
        good_indicators = {
            'bbw_percentile': 0.25,
            'adx': 28.0,
            'volume_ratio_20': 0.30,
            'range_ratio': 0.60,
            'bbw_20': 0.012,
            'volume_ratio_20': 0.30,
            'daily_range': 1.0,
        }

        # Simulate 10 days to activate
        # Day 0: Start qualifying (days_qualifying = 0)
        # Days 1-9: Continue qualifying (days_qualifying = 1-9)
        # Day 10: Reach days_qualifying = 10, activate
        for day in range(11):  # Need 11 calls (0-10)
            row = pd.Series({
                'open': 100.0,
                'high': 101.0,
                'low': 99.0,
                'close': 100.5,
                'volume': 1000000,
            })

            tracker.process_day(
                date=datetime(2024, 1, 1) + timedelta(days=day),
                idx=day,
                row=row,
                indicators=good_indicators
            )

        assert tracker.phase == PatternPhase.ACTIVE
