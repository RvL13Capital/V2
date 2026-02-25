"""
Refactored Consolidation Tracker - Clean State Machine with Delegated Components
================================================================================

Lightweight orchestrator that delegates to specialized components.
Maintains strict temporal ordering and prevents data leaks.

This is the refactored version that replaces the 924-line monolith
with clean separation of concerns.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

# Import refactored components
from .consolidation import (
    PatternState,
    PatternPhase,
    TemporalFeatureExtractor,
    SequentialStateManager,
    TemporalBoundaryManager,
    TemporalStatisticsCalculator,
    TemporalTrainingDataGenerator
)

logger = logging.getLogger(__name__)


class RefactoredConsolidationTracker:
    """
    Orchestrates consolidation pattern tracking using specialized components.

    This refactored version delegates responsibilities:
    - State management → SequentialStateManager
    - Feature extraction → TemporalFeatureExtractor
    - Boundary management → TemporalBoundaryManager
    - Statistics → TemporalStatisticsCalculator
    - Training data → TemporalTrainingDataGenerator
    """

    def __init__(
        self,
        ticker: str,
        bbw_threshold_percentile: int = 30,
        adx_threshold: float = 32,
        volume_ratio_threshold: float = 0.35,
        range_ratio_threshold: float = 0.65,
        # Weekly qualification mode (Jan 2026)
        qualification_periods: int = 10,  # 10 days (daily) or 10 weeks (weekly)
        use_weekly_qualification: bool = False
    ):
        """
        Initialize refactored tracker with components.

        Args:
            ticker: Stock ticker symbol
            bbw_threshold_percentile: BBW percentile threshold
            adx_threshold: ADX threshold for low trending
            volume_ratio_threshold: Volume ratio threshold
            range_ratio_threshold: Range ratio threshold
            qualification_periods: Number of periods for qualification phase.
                                  Default 10 (10 days for daily, 10 weeks for weekly)
            use_weekly_qualification: If True, periods are weeks instead of days.
                                     This enables ~2.5 month qualification windows.
        """
        self.ticker = ticker

        # Thresholds for consolidation criteria
        self.bbw_threshold_percentile = bbw_threshold_percentile
        self.adx_threshold = adx_threshold
        self.volume_ratio_threshold = volume_ratio_threshold
        self.range_ratio_threshold = range_ratio_threshold

        # Weekly qualification mode (Jan 2026)
        self.qualification_periods = qualification_periods
        self.use_weekly_qualification = use_weekly_qualification
        self.candle_frequency = 'weekly' if use_weekly_qualification else 'daily'

        # Initialize components
        self.state_manager = SequentialStateManager()
        self.feature_extractor = TemporalFeatureExtractor()
        self.boundary_manager = TemporalBoundaryManager()
        self.stats_calculator = TemporalStatisticsCalculator()
        self.training_generator = TemporalTrainingDataGenerator()

        # Data storage
        self.full_data = None
        self.completed_patterns: List[PatternState] = []

        # Qualification tracking
        self.qualification_highs: List[float] = []
        self.qualification_lows: List[float] = []

        # Violation tracking (for consecutive close violations)
        self.consecutive_violations = 0
        self.last_violation_type: Optional[str] = None

        mode = "weekly" if use_weekly_qualification else "daily"
        logger.info(
            f"Initialized RefactoredConsolidationTracker for {ticker} "
            f"(mode={mode}, qualification_periods={qualification_periods})"
        )

    @property
    def state(self) -> PatternState:
        """
        Get current pattern state (backward compatibility property).

        Returns:
            Current PatternState
        """
        return self.state_manager.get_state()

    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set full DataFrame for lookback operations.

        Args:
            data: Full price DataFrame with indicators
        """
        self.full_data = data
        logger.debug(f"Set data with {len(data)} rows for {self.ticker}")

    def update(
        self,
        date: datetime,
        idx: int,
        price_data: pd.DataFrame,
        indicators: pd.DataFrame
    ) -> None:
        """
        Daily update maintaining exact same sequence as original.

        PRESERVED SEQUENCE:
        1. Check current phase
        2. Evaluate transition criteria (lookback only)
        3. Transition if criteria met
        4. Extract features at current point
        5. Save snapshot if in ACTIVE phase
        6. Return current state

        Args:
            date: Current date
            idx: Current index in data
            price_data: Price OHLCV data
            indicators: Technical indicators
        """
        # Get current state
        current_state = self.state_manager.get_state()
        current_phase = current_state.phase if current_state else PatternPhase.NONE

        # EXACT SAME ORDER as original implementation:

        if current_phase == PatternPhase.NONE:
            # Check if we should start qualification
            if self._check_start_qualification(date, idx, indicators):
                # Initialize new pattern state
                start_price = price_data.iloc[idx]['close']
                new_state = self.state_manager.initialize_state(
                    ticker=self.ticker,
                    start_date=date,
                    start_idx=idx,
                    start_price=start_price
                )
                # Transition to QUALIFYING
                self.state_manager.transition(
                    PatternPhase.QUALIFYING,
                    date
                )
                # Reset qualification tracking
                self.qualification_highs = [price_data.iloc[idx]['high']]
                self.qualification_lows = [price_data.iloc[idx]['low']]

        elif current_phase == PatternPhase.QUALIFYING:
            # Track qualification progress
            self._check_qualification_progress(date, idx, price_data, indicators)

        elif current_phase == PatternPhase.ACTIVE:
            # Check for pattern completion
            self._check_active_pattern(date, idx, price_data)

    def _check_start_qualification(
        self,
        date: datetime,
        idx: int,
        indicators: pd.DataFrame
    ) -> bool:
        """
        Check if consolidation criteria are met to start qualification.

        Args:
            date: Current date
            idx: Current index
            indicators: Technical indicators

        Returns:
            True if qualification should start
        """
        # Need at least 100 days of history for percentile calculation
        if idx < 100:
            return False

        return self._meets_consolidation_criteria(idx, indicators)

    def _check_qualification_progress(
        self,
        date: datetime,
        idx: int,
        price_data: pd.DataFrame,
        indicators: pd.DataFrame
    ) -> None:
        """
        Track qualification phase progress (days 1-10).

        Args:
            date: Current date
            idx: Current index
            price_data: Price data
            indicators: Technical indicators
        """
        current_state = self.state_manager.get_state()

        # Update day counter
        self.state_manager.update_day_counters()

        # Track highs and lows during qualification
        self.qualification_highs.append(price_data.iloc[idx]['high'])
        self.qualification_lows.append(price_data.iloc[idx]['low'])

        # Check if still meeting criteria
        if not self._meets_consolidation_criteria(idx, indicators):
            # Failed qualification - reset
            logger.info(f"{self.ticker}: Failed qualification at day {current_state.days_qualifying}")
            self.state_manager.reset()
            self.qualification_highs = []
            self.qualification_lows = []
            return

        # Check if ready to activate (day 10)
        if self.state_manager.can_activate():
            self._activate_pattern(date, idx)

    def _activate_pattern(self, date: datetime, idx: int) -> None:
        """
        Activate pattern after successful 10-day qualification.

        Sets boundaries and transitions to ACTIVE phase.

        Args:
            date: Activation date
            idx: Current index
        """
        logger.info(f"{self.ticker}: Activating pattern at {date}")

        # Set boundaries from qualification data
        current_state = self.state_manager.get_state()
        updated_state = self.boundary_manager.set_boundaries_on_state(
            current_state,
            self.qualification_highs,
            self.qualification_lows,
            date
        )

        # Transition to ACTIVE
        self.state_manager.current_state = updated_state
        self.state_manager.transition(
            PatternPhase.ACTIVE,
            date
        )

        # Take initial feature snapshot
        self._save_feature_snapshot(date, idx)

    def _check_active_pattern(
        self,
        date: datetime,
        idx: int,
        price_data: pd.DataFrame
    ) -> None:
        """
        Monitor active pattern for completion with CONSECUTIVE CLOSE requirement.

        CRITICAL: Requires 2 consecutive closes outside boundaries to complete pattern.
        This filters out 1-day fakeouts common in micro-caps (stop hunts).

        Args:
            date: Current date
            idx: Current index
            price_data: Price data
        """
        current_state = self.state_manager.get_state()

        # Update day counter
        self.state_manager.update_day_counters()

        # Check boundary violations (now checks CLOSE, not high/low)
        current_row = price_data.iloc[idx]
        violation, violation_type = self.boundary_manager.check_boundary_violations(
            current_state,
            current_row['close'],
            current_row['high'],
            current_row['low']
        )

        if violation:
            # Track consecutive violations of the SAME TYPE
            if violation_type == self.last_violation_type:
                self.consecutive_violations += 1
            else:
                # New violation type, reset counter
                self.consecutive_violations = 1
                self.last_violation_type = violation_type

            # REQUIREMENT: 2 Consecutive Closes to Complete Pattern
            # Filters out 1-day fakeouts common in micro-caps
            if self.consecutive_violations >= 2:
                # Pattern completed after 2 consecutive closes
                self._complete_pattern(date, idx, violation_type, current_row['close'])
                return
        else:
            # Reset on re-entry (filters out bull/bear traps)
            self.consecutive_violations = 0
            self.last_violation_type = None

            # Randomly save feature snapshots (for training diversity)
            if np.random.random() < 0.3:  # 30% chance each day
                self._save_feature_snapshot(date, idx)

    def _complete_pattern(
        self,
        date: datetime,
        idx: int,
        breakout_direction: str,
        end_price: float
    ) -> None:
        """
        Complete pattern and prepare for next detection.

        Args:
            date: Completion date
            idx: Current index
            breakout_direction: 'BREAKOUT_UP' or 'BREAKDOWN'
            end_price: Price at completion
        """
        logger.info(
            f"{self.ticker}: Pattern completed with {breakout_direction} at {date}"
        )

        # Update state with completion info
        current_state = self.state_manager.get_state()
        completed_state = current_state._replace(
            end_date=date,
            end_idx=idx,
            end_price=end_price,
            breakout_direction='UP' if breakout_direction == 'BREAKOUT_UP' else 'DOWN'
        )

        # Transition to completed
        phase = PatternPhase.COMPLETED if breakout_direction == 'BREAKOUT_UP' else PatternPhase.FAILED
        self.state_manager.current_state = completed_state
        self.state_manager.transition(phase, date)

        # Store completed pattern
        self.completed_patterns.append(self.state_manager.get_state())

        # Reset for next pattern detection
        self.state_manager.reset()
        self.qualification_highs = []
        self.qualification_lows = []
        self.consecutive_violations = 0
        self.last_violation_type = None

    def _save_feature_snapshot(self, date: datetime, idx: int) -> None:
        """
        Save feature snapshot during ACTIVE phase for training.

        Args:
            date: Snapshot date
            idx: Current index
        """
        if self.full_data is None:
            return

        current_state = self.state_manager.get_state()

        # Extract features at current point
        features = self.feature_extractor.extract_features_at_point(
            self.full_data,
            idx,
            current_state.start_idx
        )

        # Add boundary position features
        if current_state.upper_boundary is not None:
            current_price = self.full_data.iloc[idx]['close']
            position_features = self.boundary_manager.calculate_price_position_features(
                current_state,
                current_price
            )
            features.update(position_features)

        # Add metadata
        features['snapshot_date'] = date.isoformat()
        features['days_since_activation'] = current_state.days_since_activation

        # Save snapshot
        updated_state = current_state.with_snapshot(features)
        self.state_manager.current_state = updated_state

    def _meets_consolidation_criteria(
        self,
        idx: int,
        indicators: pd.DataFrame
    ) -> bool:
        """
        Check if consolidation criteria are met.

        BBW must be below threshold + at least 2 of 3 other criteria.

        Updated Jan 2026: Replaced ADX check with market structure check
        (close > sma_50 > sma_200) to ensure consolidation in uptrend.

        Args:
            idx: Current index
            indicators: Technical indicators

        Returns:
            True if criteria are met
        """
        if idx < 100:  # Need history for percentile
            return False

        # Get current indicator values
        current_ind = indicators.iloc[idx]

        # BBW must be below threshold percentile
        bbw_history = indicators['bbw_20'].iloc[max(0, idx-100):idx]
        bbw_percentile = self.stats_calculator.calculate_percentile(
            bbw_history,
            current_ind['bbw_20'],
            end_idx=idx
        )

        if bbw_percentile >= self.bbw_threshold_percentile:
            return False

        # Check other criteria (need at least 2 of 3)
        criteria_met = 0

        # REPLACED: ADX < threshold (low trending)
        # NEW: Market structure check (close > sma_50 > sma_200)
        # Ensures consolidation is in an UPTREND, not a downtrend
        current_close = current_ind.get('close', 0)
        sma_50 = current_ind.get('sma_50')
        sma_200 = current_ind.get('sma_200')

        if sma_50 is not None and sma_200 is not None:
            if pd.notna(sma_50) and pd.notna(sma_200):
                # Market structure check: close > sma_50 > sma_200 (healthy uptrend)
                if current_close > sma_50 > sma_200:
                    criteria_met += 1
        else:
            # Fallback to legacy ADX if SMAs not available
            if current_ind.get('adx', 100) < self.adx_threshold:
                criteria_met += 1

        # Volume ratio < threshold (low volume)
        if current_ind.get('volume_ratio_20', 1.0) < self.volume_ratio_threshold:
            criteria_met += 1

        # Range ratio < threshold (compressed range)
        if current_ind.get('range_ratio_20', 1.0) < self.range_ratio_threshold:
            criteria_met += 1

        return criteria_met >= 2

    def get_current_pattern(self) -> Optional[PatternState]:
        """Get current active pattern if any."""
        state = self.state_manager.get_state()
        if state and state.phase in [PatternPhase.QUALIFYING, PatternPhase.ACTIVE]:
            return state
        return None

    def get_completed_patterns(self) -> List[PatternState]:
        """Get all completed patterns."""
        return self.completed_patterns

    def generate_training_data(self) -> List[Dict[str, Any]]:
        """
        Generate training data from completed patterns.

        Returns:
            List of training samples
        """
        return self.training_generator.create_batch_samples(
            self.completed_patterns,
            only_completed=True
        )

    def get_state(self) -> Dict[str, Any]:
        """
        Get current tracker state for serialization.

        Returns:
            State dictionary
        """
        current_state = self.state_manager.get_state()

        return {
            'ticker': self.ticker,
            'current_phase': current_state.phase.value if current_state else 'NONE',
            'current_pattern': current_state.to_dict() if current_state else None,
            'completed_patterns': len(self.completed_patterns),
            'total_training_samples': sum(
                len(p.feature_snapshots) for p in self.completed_patterns
            ),
            # Weekly qualification mode (Jan 2026)
            'qualification_frequency': self.candle_frequency,
            'qualification_periods': self.qualification_periods
        }