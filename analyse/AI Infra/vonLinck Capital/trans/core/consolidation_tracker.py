"""
Consolidation Pattern Tracker - STANDALONE IMPLEMENTATION
==========================================================

This is a standalone reference implementation of the consolidation pattern detector.

ARCHITECTURAL NOTE:
- This implementation is kept separate from core/aiv7_components
- Has its own API design (process_day, check_qualification_criteria, etc.)
- Includes MATURE state (not in refactored core version)
- 100% test coverage (23/23 tests passing)
- Designed for standalone use in Consol/ analysis scripts

For the main TRANS system, see: core/aiv7_components/consolidation_tracker_refactored.py

This version implements state machine for detecting consolidation patterns with qualification phases.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class PatternState(Enum):
    """State machine phases for consolidation pattern detection"""
    NONE = "none"
    QUALIFYING = "qualifying"
    ACTIVE = "active"
    MATURE = "mature"  # After 30 days in channel, boundaries adjusted
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ConsolidationPattern:
    """Data class for storing consolidation pattern information"""
    ticker: str
    start_date: datetime
    qualification_start: datetime
    qualification_end: Optional[datetime] = None
    active_start: Optional[datetime] = None
    end_date: Optional[datetime] = None
    state: PatternState = PatternState.NONE

    # Boundaries
    upper_boundary: float = 0.0
    lower_boundary: float = 0.0
    power_boundary: float = 0.0  # Upper * 1.005 (0.5% buffer)

    # Adjusted boundaries (after 30 days in channel)
    adjusted_upper: float = 0.0
    adjusted_lower: float = 0.0
    adjusted_power: float = 0.0

    # Metrics during qualification
    avg_bbw: float = 0.0
    avg_adx: float = 0.0
    avg_volume_ratio: float = 0.0
    avg_range_ratio: float = 0.0

    # Pattern characteristics
    days_in_qualification: int = 0
    days_active: int = 0
    days_in_channel: int = 0  # Days staying within boundaries
    days_mature: int = 0  # Days after boundary adjustment
    boundary_violations: int = 0
    maturity_date: Optional[datetime] = None  # When pattern became mature

    # Features for ML
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrackPointPattern:
    """
    Lightweight pattern wrapper compatible with TemporalPatternDetector API.

    This class provides the interface expected by pattern_detector.py:
    - is_terminal() method
    - phase attribute (PatternPhase from aiv7_components)
    - start_idx, end_idx for data slicing
    - upper_boundary, lower_boundary for feature calculation
    - start_date, end_date for temporal tracking
    """
    ticker: str
    start_date: datetime
    end_date: datetime
    start_idx: int
    end_idx: int
    start_price: float
    upper_boundary: Optional[float] = None
    lower_boundary: Optional[float] = None
    phase: Optional['PatternPhase'] = None  # aiv7_components.PatternPhase

    def is_terminal(self) -> bool:
        """Check if pattern is in terminal state (COMPLETED or FAILED)."""
        if self.phase is None:
            return False
        # Import here to avoid circular imports
        from .aiv7_components import PatternPhase as AIv7PatternPhase
        return self.phase in [
            AIv7PatternPhase.COMPLETED,
            AIv7PatternPhase.FAILED,
            AIv7PatternPhase.RECOVERING,
            AIv7PatternPhase.DEAD
        ]


class ConsolidationTracker:
    """
    Tracks consolidation patterns through state machine phases:
    NONE → QUALIFYING (10 days) → ACTIVE → COMPLETED/FAILED
    """

    def __init__(self,
                 ticker: str,
                 qualification_days: int = 10,
                 channel_days_required: int = 30,
                 bbw_percentile: float = 30,
                 adx_threshold: float = 32,
                 volume_threshold: float = 0.35,
                 range_threshold: float = 0.65):
        """
        Initialize consolidation tracker with thresholds

        Args:
            ticker: Stock symbol
            qualification_days: Days required to qualify (default 10)
            bbw_percentile: BBW percentile threshold (default 30th)
            adx_threshold: Maximum ADX for low trending (default 32)
            volume_threshold: Max volume ratio vs 20-day avg (default 0.35)
            range_threshold: Max daily range ratio vs 20-day avg (default 0.65)
        """
        self.ticker = ticker
        self.qualification_days = qualification_days
        self.channel_days_required = channel_days_required

        # Thresholds
        self.bbw_percentile = bbw_percentile
        self.adx_threshold = adx_threshold
        self.volume_threshold = volume_threshold
        self.range_threshold = range_threshold

        # Current pattern tracking
        self.current_pattern: Optional[ConsolidationPattern] = None
        self.completed_patterns: List[ConsolidationPattern] = []

        # Qualification tracking
        self.qualification_counter = 0
        self.qualification_metrics = []

        # Violation tracking (for consecutive close violations)
        self.consecutive_violations = 0

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators needed for consolidation detection
        Uses centralized indicator calculation from utils.indicators

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with additional indicator columns
        """
        # Use centralized calculation to avoid duplication
        try:
            from utils.indicators import calculate_all_indicators
        except ImportError:
            # Try absolute import from Consol package
            import sys
            import os
            utils_path = os.path.join(os.path.dirname(__file__), 'utils')
            if utils_path not in sys.path:
                sys.path.insert(0, utils_path)
            from indicators import calculate_all_indicators

        return calculate_all_indicators(data)

    def check_qualification_criteria(self, row: pd.Series) -> bool:
        """
        Check if current day meets qualification criteria

        Args:
            row: Series with indicator data for current day

        Returns:
            True if all criteria met, False otherwise
        """
        return (
            row['bbw_percentile'] < self.bbw_percentile and
            row['adx'] < self.adx_threshold and
            row['volume_ratio'] < self.volume_threshold and
            row['range_ratio'] < self.range_threshold
        )

    def establish_boundaries(self, data: pd.DataFrame, end_idx: int) -> Tuple[float, float, float]:
        """
        Establish boundaries after qualification phase using PERCENTILE-BASED ROBUSTNESS.

        Uses 90th/10th percentile of High/Low to create a "Volatility Envelope"
        that ignores the top 10% of manipulation spikes (scam wicks).

        This allows patterns to survive longer (60-100 days) by not being
        killed by single stop-hunt events.

        Args:
            data: DataFrame with price data
            end_idx: Index of qualification end

        Returns:
            Tuple of (upper_boundary, lower_boundary, power_boundary)
            Returns (None, None, None) if pattern is invalid
        """
        # Use qualification period data to establish boundaries
        start_idx = max(0, end_idx - self.qualification_days)
        period_data = data.iloc[start_idx:end_idx + 1]

        # OPTION B: 90th/10th Percentile of High/Low (Robust Boundaries)
        # This crops out the top 10% of "scam wicks" that destroy pattern duration
        upper_boundary = period_data['high'].quantile(0.90)
        lower_boundary = period_data['low'].quantile(0.10)

        # VALIDATION: Reject if price is invalid
        if lower_boundary < 1e-10:  # Division by zero guard
            return None, None, None

        # SAFETY: Prevent collapse on flat stocks (zero variance)
        if upper_boundary <= lower_boundary:
            center = period_data['close'].mean()
            upper_boundary = center * 1.02
            lower_boundary = center * 0.98

        # VALIDATION: Reject if range is too small (< 0.3% range)
        # This filters out ultra-stable mutual fund NAVs while allowing tight consolidations
        range_pct = ((upper_boundary - lower_boundary) / lower_boundary) * 100
        if range_pct < 0.3:  # Minimum 0.3% range required
            return None, None, None

        # VALIDATION: Check if this is a dead/frozen mutual fund
        # If ALL days in qualification period have zero volume, reject
        zero_volume_days = (period_data['volume'] == 0).sum()
        if zero_volume_days == len(period_data):  # ALL days zero volume = mutual fund NAV
            return None, None, None

        power_boundary = upper_boundary * 1.005  # 0.5% buffer above upper

        return upper_boundary, lower_boundary, power_boundary

    def process_day(self, data: pd.DataFrame, idx: int) -> Optional[str]:
        """
        Process single day through state machine

        Args:
            data: DataFrame with all indicator data
            idx: Current day index

        Returns:
            Signal type if pattern completes/fails, None otherwise
        """
        if idx >= len(data):
            return None

        row = data.iloc[idx]
        current_date = row.name if isinstance(row.name, datetime) else datetime.now()

        # State machine logic
        if self.current_pattern is None:
            # Check if we should start qualifying
            if self.check_qualification_criteria(row):
                self.qualification_counter = 1
                self.qualification_metrics = [{
                    'bbw': row['bbw'],
                    'adx': row['adx'],
                    'volume_ratio': row['volume_ratio'],
                    'range_ratio': row['range_ratio']
                }]

                self.current_pattern = ConsolidationPattern(
                    ticker=self.ticker,
                    start_date=current_date,
                    qualification_start=current_date,
                    state=PatternState.QUALIFYING
                )

        elif self.current_pattern.state == PatternState.QUALIFYING:
            # Continue qualification or fail
            if self.check_qualification_criteria(row):
                self.qualification_counter += 1
                self.qualification_metrics.append({
                    'bbw': row['bbw'],
                    'adx': row['adx'],
                    'volume_ratio': row['volume_ratio'],
                    'range_ratio': row['range_ratio']
                })

                # Check if qualification complete
                if self.qualification_counter >= self.qualification_days:
                    # Calculate average metrics
                    metrics_df = pd.DataFrame(self.qualification_metrics)
                    self.current_pattern.avg_bbw = metrics_df['bbw'].mean()
                    self.current_pattern.avg_adx = metrics_df['adx'].mean()
                    self.current_pattern.avg_volume_ratio = metrics_df['volume_ratio'].mean()
                    self.current_pattern.avg_range_ratio = metrics_df['range_ratio'].mean()
                    self.current_pattern.days_in_qualification = self.qualification_counter

                    # Establish boundaries
                    upper, lower, power = self.establish_boundaries(data, idx)

                    # Check if boundaries are valid (not None)
                    if upper is None or lower is None or power is None:
                        # Pattern failed validation (insufficient range/volume)
                        self.current_pattern = None
                        self.qualification_counter = 0
                        self.qualification_metrics = []
                        return None

                    self.current_pattern.upper_boundary = upper
                    self.current_pattern.lower_boundary = lower
                    self.current_pattern.power_boundary = power

                    # Transition to ACTIVE
                    self.current_pattern.state = PatternState.ACTIVE
                    self.current_pattern.qualification_end = current_date
                    self.current_pattern.active_start = current_date

            else:
                # Failed qualification
                self.current_pattern = None
                self.qualification_counter = 0
                self.qualification_metrics = []

        elif self.current_pattern.state == PatternState.ACTIVE:
            # ACTIVE state: Must stay in channel for 30 days before becoming MATURE
            self.current_pattern.days_active += 1

            # THE "IRON CORE" CHECK: Wicks are free, only CLOSES count as structural breaks
            # Check if CLOSE is within boundaries (allow intraday volatility)
            # Boundaries are "Volatility Envelope" (90th % High) as limit for Closes
            is_violation = False

            if row['close'] > self.current_pattern.upper_boundary:
                is_violation = True
            elif row['close'] < self.current_pattern.lower_boundary:
                is_violation = True

            if is_violation:
                self.consecutive_violations += 1
            else:
                # Reset on re-entry (filters out bull/bear traps)
                self.consecutive_violations = 0
                self.current_pattern.days_in_channel += 1

            # REQUIREMENT: 2 Consecutive Closes to Kill Pattern
            # Filters out 1-day fakeouts common in micro-caps
            if self.consecutive_violations >= 2:
                # Determine if breakout or breakdown
                if row['close'] < self.current_pattern.lower_boundary:
                    # Breakdown: 2 consecutive closes below lower boundary
                    self.current_pattern.state = PatternState.FAILED
                    self.current_pattern.end_date = current_date
                    self.completed_patterns.append(self.current_pattern)

                    signal = "BREAKDOWN"
                    self.current_pattern = None
                    self.consecutive_violations = 0
                    return signal

                elif row['close'] > self.current_pattern.upper_boundary:
                    # Breakout: 2 consecutive closes above upper boundary
                    self.current_pattern.state = PatternState.COMPLETED
                    self.current_pattern.end_date = current_date
                    self.completed_patterns.append(self.current_pattern)

                    features = self.extract_pattern_features(data, idx)
                    self.current_pattern.features = features

                    signal = "BREAKOUT"
                    self.current_pattern = None
                    self.consecutive_violations = 0
                    return signal

            # Check if ready to transition to MATURE state (after 30 days in channel)
            if self.current_pattern.days_in_channel >= self.channel_days_required:
                # Transition to MATURE state and adjust boundaries
                self.current_pattern.state = PatternState.MATURE
                self.current_pattern.maturity_date = current_date

                # Calculate adjusted boundaries based on last 20 days (using percentiles)
                start_idx = max(0, idx - 20)
                recent_data = data.iloc[start_idx:idx + 1]

                # NEW: Use percentiles for robustness
                new_upper = recent_data['high'].quantile(0.90)
                new_lower = recent_data['low'].quantile(0.10)

                # ONLY EXPAND, NEVER CONTRACT (Natural Drift tolerance)
                if new_upper > self.current_pattern.upper_boundary:
                    self.current_pattern.adjusted_upper = new_upper
                else:
                    self.current_pattern.adjusted_upper = self.current_pattern.upper_boundary

                if new_lower < self.current_pattern.lower_boundary:
                    self.current_pattern.adjusted_lower = new_lower
                else:
                    self.current_pattern.adjusted_lower = self.current_pattern.lower_boundary

                self.current_pattern.adjusted_power = self.current_pattern.adjusted_upper * 1.005

                return "MATURE_PATTERN"

        elif self.current_pattern.state == PatternState.MATURE:
            # MATURE state: Volume requirement removed, adjusted boundaries
            self.current_pattern.days_mature += 1

            # Use adjusted boundaries for mature patterns
            upper_bound = self.current_pattern.adjusted_upper
            lower_bound = self.current_pattern.adjusted_lower

            # THE "IRON CORE" CHECK: Same logic as ACTIVE state
            is_violation = False

            if row['close'] > upper_bound:
                is_violation = True
            elif row['close'] < lower_bound:
                is_violation = True

            if is_violation:
                self.consecutive_violations += 1
            else:
                # Reset on re-entry (filters out bull/bear traps)
                self.consecutive_violations = 0

            # REQUIREMENT: 2 Consecutive Closes to Complete/Fail Pattern
            if self.consecutive_violations >= 2:
                # Determine if breakout or breakdown
                if row['close'] < lower_bound:
                    # Breakdown: 2 consecutive closes below lower boundary
                    self.current_pattern.state = PatternState.FAILED
                    self.current_pattern.end_date = current_date
                    self.completed_patterns.append(self.current_pattern)

                    signal = "MATURE_BREAKDOWN"
                    self.current_pattern = None
                    self.consecutive_violations = 0
                    return signal

                elif row['close'] > upper_bound:
                    # Breakout: 2 consecutive closes above upper boundary
                    self.current_pattern.state = PatternState.COMPLETED
                    self.current_pattern.end_date = current_date
                    self.completed_patterns.append(self.current_pattern)

                    features = self.extract_pattern_features(data, idx)
                    self.current_pattern.features = features

                    signal = "MATURE_BREAKOUT"
                    self.current_pattern = None
                    self.consecutive_violations = 0
                    return signal

        return None

    def extract_pattern_features(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        """
        Extract features from completed pattern for ML

        Args:
            data: DataFrame with all data
            idx: Index of pattern completion

        Returns:
            Dictionary of features
        """
        if self.current_pattern is None:
            return {}

        # Calculate range compression with division-by-zero guard
        if self.current_pattern.lower_boundary > 1e-10:
            range_compression = ((self.current_pattern.upper_boundary -
                                 self.current_pattern.lower_boundary) /
                                self.current_pattern.lower_boundary * 100)
        else:
            range_compression = 0.0

        features = {
            'days_in_qualification': self.current_pattern.days_in_qualification,
            'days_active': self.current_pattern.days_active,
            'avg_bbw': self.current_pattern.avg_bbw,
            'avg_adx': self.current_pattern.avg_adx,
            'avg_volume_ratio': self.current_pattern.avg_volume_ratio,
            'avg_range_ratio': self.current_pattern.avg_range_ratio,
            'boundary_violations': self.current_pattern.boundary_violations,
            'range_compression': range_compression
        }

        # Add current market conditions
        if idx < len(data):
            row = data.iloc[idx]

            # Calculate price position with division-by-zero guard
            boundary_range = self.current_pattern.upper_boundary - self.current_pattern.lower_boundary
            if boundary_range > 1e-10:
                price_position = ((row['close'] - self.current_pattern.lower_boundary) /
                                 boundary_range)
            else:
                price_position = 0.5  # Default to middle if no range

            features.update({
                'breakout_volume_ratio': row['volume_ratio'],
                'breakout_bbw': row['bbw'],
                'breakout_adx': row['adx'],
                'price_position': price_position
            })

        return features

    def scan(self, data: pd.DataFrame) -> List[Dict]:
        """
        Scan entire dataset for consolidation patterns

        Args:
            data: DataFrame with OHLCV data

        Returns:
            List of pattern signals
        """
        # Calculate indicators
        df = self.calculate_indicators(data)

        signals = []
        for i in range(len(df)):
            signal = self.process_day(df, i)
            if signal:
                signals.append({
                    'date': df.index[i] if hasattr(df, 'index') else i,
                    'signal': signal,
                    'pattern': self.current_pattern.__dict__ if self.current_pattern else None
                })

        return signals

    def track_point(
        self,
        date: datetime,
        close: float,
        high: float,
        low: float,
        volume: float,
        bbw: float = 0,
        adx: float = 0,
        current_idx: int = 0
    ) -> Tuple[Optional['TrackPointPattern'], Optional['PatternState']]:
        """
        Track a single price point and update pattern state.

        This method provides a point-by-point interface for pattern detection,
        compatible with the TemporalPatternDetector API.

        Args:
            date: Current date/datetime
            close: Closing price
            high: High price
            low: Low price
            volume: Volume
            bbw: Bollinger Band Width (optional)
            adx: ADX indicator (optional)
            current_idx: Index in the data series

        Returns:
            Tuple of (pattern, phase) where:
            - pattern: TrackPointPattern object if pattern exists, None otherwise
            - phase: PatternState enum value

        Note:
            The returned pattern object is a lightweight wrapper compatible
            with the TemporalPatternDetector's expectations.
        """
        # Build internal tracking data
        if not hasattr(self, '_track_data'):
            self._track_data = []
            self._track_idx = 0

        # Append new point
        self._track_data.append({
            'date': date,
            'close': close,
            'high': high,
            'low': low,
            'volume': volume,
            'bbw': bbw,
            'adx': adx,
            'bbw_percentile': self._calculate_bbw_percentile(bbw) if bbw > 0 else 50,
            'volume_ratio': self._calculate_volume_ratio(volume),
            'range_ratio': self._calculate_range_ratio(high, low)
        })

        # Convert to DataFrame row for processing
        df = pd.DataFrame(self._track_data)
        df.index = pd.to_datetime([d['date'] for d in self._track_data])

        # Process the current day
        idx = len(self._track_data) - 1
        signal = self.process_day(df, idx)

        # Return pattern and phase
        if self.current_pattern is not None:
            # Create a wrapper pattern compatible with TemporalPatternDetector
            wrapper = TrackPointPattern(
                ticker=self.ticker,
                start_date=self.current_pattern.start_date,
                end_date=self.current_pattern.end_date or date,
                start_idx=max(0, idx - self.current_pattern.days_active - self.current_pattern.days_in_qualification),
                end_idx=current_idx,
                start_price=self.current_pattern.lower_boundary or close,
                upper_boundary=self.current_pattern.upper_boundary,
                lower_boundary=self.current_pattern.lower_boundary,
                phase=self._map_state_to_phase(self.current_pattern.state)
            )
            return wrapper, self.current_pattern.state

        # Check completed patterns from this call
        if signal and len(self.completed_patterns) > 0:
            last_completed = self.completed_patterns[-1]
            wrapper = TrackPointPattern(
                ticker=self.ticker,
                start_date=last_completed.start_date,
                end_date=last_completed.end_date or date,
                start_idx=max(0, idx - last_completed.days_active - last_completed.days_in_qualification),
                end_idx=current_idx,
                start_price=last_completed.lower_boundary or close,
                upper_boundary=last_completed.upper_boundary,
                lower_boundary=last_completed.lower_boundary,
                phase=self._map_state_to_phase(last_completed.state)
            )
            return wrapper, last_completed.state

        return None, PatternState.NONE

    def _map_state_to_phase(self, state: PatternState) -> 'PatternPhase':
        """Map internal PatternState to aiv7_components PatternPhase."""
        # Import here to avoid circular imports
        from .aiv7_components import PatternPhase as AIv7PatternPhase

        mapping = {
            PatternState.NONE: AIv7PatternPhase.NONE,
            PatternState.QUALIFYING: AIv7PatternPhase.QUALIFYING,
            PatternState.ACTIVE: AIv7PatternPhase.ACTIVE,
            PatternState.MATURE: AIv7PatternPhase.ACTIVE,  # Mature is still active
            PatternState.COMPLETED: AIv7PatternPhase.COMPLETED,
            PatternState.FAILED: AIv7PatternPhase.FAILED,
        }
        return mapping.get(state, AIv7PatternPhase.NONE)

    def _calculate_bbw_percentile(self, bbw: float) -> float:
        """Calculate BBW percentile from historical data."""
        if not hasattr(self, '_track_data') or len(self._track_data) < 20:
            return 50.0  # Default to median if insufficient history

        bbw_history = [d['bbw'] for d in self._track_data[-100:] if d['bbw'] > 0]
        if not bbw_history:
            return 50.0

        count_below = sum(1 for b in bbw_history if b < bbw)
        return (count_below / len(bbw_history)) * 100

    def _calculate_volume_ratio(self, volume: float) -> float:
        """Calculate volume ratio vs 20-day average."""
        if not hasattr(self, '_track_data') or len(self._track_data) < 20:
            return 1.0

        vol_history = [d['volume'] for d in self._track_data[-20:]]
        avg_vol = np.mean(vol_history) if vol_history else volume
        return volume / avg_vol if avg_vol > 0 else 1.0

    def _calculate_range_ratio(self, high: float, low: float) -> float:
        """Calculate daily range ratio vs 20-day average."""
        if not hasattr(self, '_track_data') or len(self._track_data) < 20:
            return 1.0

        daily_range = high - low
        range_history = [d['high'] - d['low'] for d in self._track_data[-20:]]
        avg_range = np.mean(range_history) if range_history else daily_range
        return daily_range / avg_range if avg_range > 0 else 1.0

    def get_current_state(self) -> Dict:
        """Get current pattern state"""
        if self.current_pattern:
            state_info = {
                'state': self.current_pattern.state.value,
                'days_active': self.current_pattern.days_active,
                'days_in_channel': self.current_pattern.days_in_channel,
                'days_mature': self.current_pattern.days_mature,
                'upper_boundary': self.current_pattern.upper_boundary,
                'lower_boundary': self.current_pattern.lower_boundary,
                'power_boundary': self.current_pattern.power_boundary,
                'boundary_violations': self.current_pattern.boundary_violations,
                'features': self.current_pattern.features
            }

            # Add adjusted boundaries if pattern is mature
            if self.current_pattern.state == PatternState.MATURE:
                state_info.update({
                    'adjusted_upper': self.current_pattern.adjusted_upper,
                    'adjusted_lower': self.current_pattern.adjusted_lower,
                    'adjusted_power': self.current_pattern.adjusted_power,
                    'maturity_date': self.current_pattern.maturity_date
                })

            return state_info
        return {'state': PatternState.NONE.value}