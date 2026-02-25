"""
Pattern State Container - Immutable State with Temporal Metadata
================================================================

Stores pattern state with timestamps to ensure temporal integrity.
All state changes create new instances (immutable pattern).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum


class PatternPhase(Enum):
    """Pattern lifecycle phases with strict ordering."""
    NONE = "NONE"  # Not in consolidation
    QUALIFYING = "QUALIFYING"  # First 10 days - testing consolidation criteria
    ACTIVE = "ACTIVE"  # After 10 days - active consolidation pattern
    COMPLETED = "COMPLETED"  # Broke out successfully (2 consecutive closes above)
    FAILED = "FAILED"  # Broke down (2 consecutive closes below) - signals allowed
    RECOVERING = "RECOVERING"  # FAILED + High > upper_boundary - no signals
    DEAD = "DEAD"  # FAILED + Low breached stop level - no signals


# State priority ranking (lower index = better, higher index = worse)
# Used when multiple state conditions are met on same candle
STATE_PRIORITY = {
    PatternPhase.COMPLETED: 0,   # Best outcome
    PatternPhase.RECOVERING: 1,  # Recovery attempt
    PatternPhase.FAILED: 2,      # Breakdown
    PatternPhase.DEAD: 3,        # Worst outcome
}


@dataclass(frozen=True)
class PatternState:
    """
    Immutable pattern state container with temporal metadata.

    All fields are frozen after creation to prevent accidental
    temporal violations. State transitions create new instances.
    """

    # Core identifiers
    ticker: str
    start_date: datetime
    start_idx: int
    start_price: float

    # Current phase
    phase: PatternPhase = PatternPhase.NONE

    # Temporal tracking (LEAK-FREE)
    days_qualifying: int = 0
    days_since_activation: int = 0  # Only incremented after ACTIVE

    # Phase transition timestamps
    qualification_started_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Boundaries (set once at activation, never changed)
    upper_boundary: Optional[float] = None
    lower_boundary: Optional[float] = None
    power_boundary: Optional[float] = None  # upper * 1.005
    boundary_set_date: Optional[datetime] = None
    boundary_range_pct: Optional[float] = None

    # Qualification period data (days 1-10)
    qualification_highs: List[float] = field(default_factory=list)
    qualification_lows: List[float] = field(default_factory=list)

    # Completion data
    end_date: Optional[datetime] = None
    end_idx: Optional[int] = None
    end_price: Optional[float] = None
    breakout_direction: Optional[str] = None  # 'UP' or 'DOWN'

    # New state tracking for 2-consecutive-close logic
    consecutive_close_above: int = 0  # Days with close > upper_boundary
    consecutive_close_below: int = 0  # Days with close < lower_boundary
    last_active_lowest_close: Optional[float] = None  # Stop level for DEAD detection
    signals_allowed: bool = False  # True for ACTIVE and FAILED

    # Features captured at specific points in time
    # Key: datetime when features were calculated
    # Value: feature dictionary at that point
    temporal_features: Dict[datetime, Dict[str, Any]] = field(default_factory=dict)

    # Feature snapshots for training (captured during ACTIVE phase)
    feature_snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def with_phase_transition(
        self,
        new_phase: PatternPhase,
        transition_date: datetime,
        **kwargs
    ) -> 'PatternState':
        """
        Create new state with phase transition.

        Ensures temporal ordering is maintained:
        NONE → QUALIFYING → ACTIVE → COMPLETED/FAILED

        Args:
            new_phase: Target phase
            transition_date: When transition occurs
            **kwargs: Additional fields to update

        Returns:
            New PatternState instance
        """
        # Validate transition
        valid_transitions = {
            PatternPhase.NONE: [PatternPhase.QUALIFYING],
            PatternPhase.QUALIFYING: [PatternPhase.ACTIVE, PatternPhase.NONE],
            PatternPhase.ACTIVE: [PatternPhase.COMPLETED, PatternPhase.FAILED],
            PatternPhase.FAILED: [PatternPhase.RECOVERING, PatternPhase.DEAD, PatternPhase.NONE],
            PatternPhase.RECOVERING: [PatternPhase.DEAD, PatternPhase.NONE],  # Can become DEAD or start new
            PatternPhase.COMPLETED: [PatternPhase.NONE],  # Can start new consolidation
            PatternPhase.DEAD: [PatternPhase.NONE]  # Can start new consolidation
        }

        if new_phase not in valid_transitions[self.phase]:
            raise ValueError(
                f"Invalid transition: {self.phase} → {new_phase}. "
                f"Valid: {valid_transitions[self.phase]}"
            )

        # Set transition timestamps and signals_allowed
        updates = kwargs.copy()
        updates['phase'] = new_phase

        if new_phase == PatternPhase.QUALIFYING:
            updates['qualification_started_at'] = transition_date
            updates['signals_allowed'] = False
        elif new_phase == PatternPhase.ACTIVE:
            updates['activated_at'] = transition_date
            updates['signals_allowed'] = True  # ACTIVE allows signals (if close < upper)
        elif new_phase == PatternPhase.COMPLETED:
            updates['completed_at'] = transition_date
            updates['end_date'] = transition_date
            updates['signals_allowed'] = False
        elif new_phase == PatternPhase.FAILED:
            updates['completed_at'] = transition_date
            updates['end_date'] = transition_date
            updates['signals_allowed'] = True  # FAILED allows signals
        elif new_phase == PatternPhase.RECOVERING:
            updates['signals_allowed'] = False  # RECOVERING does not allow signals
        elif new_phase == PatternPhase.DEAD:
            updates['signals_allowed'] = False

        # Create new instance with updates
        return self._replace(**updates)

    def _replace(self, **kwargs) -> 'PatternState':
        """
        Create new instance with updated fields.

        Internal method to maintain immutability while allowing updates.
        """
        # Get current values as dict
        current_vals = {
            'ticker': self.ticker,
            'start_date': self.start_date,
            'start_idx': self.start_idx,
            'start_price': self.start_price,
            'phase': self.phase,
            'days_qualifying': self.days_qualifying,
            'days_since_activation': self.days_since_activation,
            'qualification_started_at': self.qualification_started_at,
            'activated_at': self.activated_at,
            'completed_at': self.completed_at,
            'upper_boundary': self.upper_boundary,
            'lower_boundary': self.lower_boundary,
            'power_boundary': self.power_boundary,
            'boundary_set_date': self.boundary_set_date,
            'boundary_range_pct': self.boundary_range_pct,
            'qualification_highs': self.qualification_highs,
            'qualification_lows': self.qualification_lows,
            'end_date': self.end_date,
            'end_idx': self.end_idx,
            'end_price': self.end_price,
            'breakout_direction': self.breakout_direction,
            'consecutive_close_above': self.consecutive_close_above,
            'consecutive_close_below': self.consecutive_close_below,
            'last_active_lowest_close': self.last_active_lowest_close,
            'signals_allowed': self.signals_allowed,
            'temporal_features': self.temporal_features,
            'feature_snapshots': self.feature_snapshots
        }

        # Update with new values
        current_vals.update(kwargs)

        # Create new instance
        return PatternState(**current_vals)

    def with_boundaries(
        self,
        upper: float,
        lower: float,
        set_date: datetime
    ) -> 'PatternState':
        """
        Create new state with boundaries set.

        Boundaries are immutable once set (temporal integrity).

        Args:
            upper: Upper boundary from qualification highs
            lower: Lower boundary from qualification lows
            set_date: When boundaries were established

        Returns:
            New PatternState with boundaries
        """
        if self.boundary_set_date is not None:
            raise ValueError(
                f"Boundaries already set at {self.boundary_set_date}. "
                "Boundaries are immutable for temporal integrity."
            )

        return self._replace(
            upper_boundary=upper,
            lower_boundary=lower,
            power_boundary=upper * 1.005,
            boundary_set_date=set_date,
            boundary_range_pct=(upper - lower) / lower if lower > 0 else 0
        )

    def with_features_at_point(
        self,
        features: Dict[str, Any],
        calculation_date: datetime
    ) -> 'PatternState':
        """
        Add features calculated at specific point in time.

        Args:
            features: Feature dictionary
            calculation_date: When features were calculated

        Returns:
            New PatternState with features added
        """
        new_temporal_features = self.temporal_features.copy()
        new_temporal_features[calculation_date] = features

        return self._replace(temporal_features=new_temporal_features)

    def with_snapshot(
        self,
        snapshot: Dict[str, Any]
    ) -> 'PatternState':
        """
        Add feature snapshot for training.

        Snapshots are only taken during ACTIVE phase.

        Args:
            snapshot: Feature snapshot dictionary

        Returns:
            New PatternState with snapshot added
        """
        if self.phase != PatternPhase.ACTIVE:
            raise ValueError(
                f"Snapshots only taken during ACTIVE phase, not {self.phase}"
            )

        new_snapshots = self.feature_snapshots.copy()
        new_snapshots.append(snapshot)

        return self._replace(feature_snapshots=new_snapshots)

    def increment_days(self) -> 'PatternState':
        """
        Increment day counters based on current phase.

        Returns:
            New PatternState with updated counters
        """
        updates = {}

        if self.phase == PatternPhase.QUALIFYING:
            updates['days_qualifying'] = self.days_qualifying + 1
        elif self.phase == PatternPhase.ACTIVE:
            updates['days_since_activation'] = self.days_since_activation + 1

        return self._replace(**updates) if updates else self

    def is_terminal(self) -> bool:
        """Check if pattern is in terminal state (COMPLETED, RECOVERING, or DEAD)."""
        return self.phase in [
            PatternPhase.COMPLETED,
            PatternPhase.RECOVERING,
            PatternPhase.DEAD
        ]

    def is_signals_allowed(self, close_price: Optional[float] = None) -> bool:
        """
        Check if signals are allowed at current close price.

        Signal rules:
        - ACTIVE: signals allowed only if close < upper_boundary
        - FAILED: signals always allowed (recovery play)
        - RECOVERING: no signals (recovery in progress)
        - Other phases: no signals

        Args:
            close_price: Current close price. Required for ACTIVE phase.

        Returns:
            True if signals allowed, False otherwise
        """
        # FAILED always allows signals (recovery play)
        if self.phase == PatternPhase.FAILED:
            return True

        # ACTIVE requires close < upper_boundary
        if self.phase == PatternPhase.ACTIVE:
            if close_price is None:
                return False
            if self.upper_boundary is None:
                return False
            return close_price < self.upper_boundary

        # RECOVERING and all other phases: no signals
        return False

    def update_consecutive_counters(
        self,
        close_price: float,
        low_price: float
    ) -> 'PatternState':
        """
        Update consecutive close counters based on current day's price.

        Args:
            close_price: Today's closing price
            low_price: Today's low price

        Returns:
            New PatternState with updated counters
        """
        if self.phase != PatternPhase.ACTIVE:
            return self

        updates = {}

        # Update last_active_lowest_close
        if self.last_active_lowest_close is None or close_price < self.last_active_lowest_close:
            updates['last_active_lowest_close'] = close_price

        if close_price > self.upper_boundary:
            # Close above upper bound
            updates['consecutive_close_above'] = self.consecutive_close_above + 1
            updates['consecutive_close_below'] = 0
        elif close_price < self.lower_boundary:
            # Close below lower bound
            updates['consecutive_close_below'] = self.consecutive_close_below + 1
            updates['consecutive_close_above'] = 0
        else:
            # Inside corridor - reset counters
            updates['consecutive_close_above'] = 0
            updates['consecutive_close_below'] = 0

        return self._replace(**updates) if updates else self

    def should_complete(self) -> bool:
        """Check if pattern should transition to COMPLETED (2 closes above)."""
        return self.phase == PatternPhase.ACTIVE and self.consecutive_close_above >= 2

    def should_fail(self) -> bool:
        """Check if pattern should transition to FAILED (2 closes below)."""
        return self.phase == PatternPhase.ACTIVE and self.consecutive_close_below >= 2

    def should_recover(self, current_high: float) -> bool:
        """Check if FAILED pattern should become RECOVERING (high > upper_boundary)."""
        return (
            self.phase == PatternPhase.FAILED and
            self.upper_boundary is not None and
            current_high > self.upper_boundary
        )

    def should_die(self, current_low: float) -> bool:
        """Check if FAILED/RECOVERING pattern should become DEAD (low breaches stop)."""
        return (
            self.phase in [PatternPhase.FAILED, PatternPhase.RECOVERING] and
            self.last_active_lowest_close is not None and
            current_low < self.last_active_lowest_close
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Maintains all temporal metadata for analysis.
        """
        return {
            # Identifiers
            'ticker': self.ticker,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'start_idx': self.start_idx,
            'start_price': self.start_price,

            # Phase info
            'phase': self.phase.value,
            'days_qualifying': self.days_qualifying,
            'days_since_activation': self.days_since_activation,

            # Timestamps
            'qualification_started_at': (
                self.qualification_started_at.isoformat()
                if self.qualification_started_at else None
            ),
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,

            # Boundaries
            'upper_boundary': self.upper_boundary,
            'lower_boundary': self.lower_boundary,
            'power_boundary': self.power_boundary,
            'boundary_set_date': (
                self.boundary_set_date.isoformat()
                if self.boundary_set_date else None
            ),
            'boundary_range_pct': self.boundary_range_pct,

            # Completion
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'end_idx': self.end_idx,
            'end_price': self.end_price,
            'breakout_direction': self.breakout_direction,

            # New state tracking
            'consecutive_close_above': self.consecutive_close_above,
            'consecutive_close_below': self.consecutive_close_below,
            'last_active_lowest_close': self.last_active_lowest_close,
            'signals_allowed': self.signals_allowed,

            # Features
            'num_snapshots': len(self.feature_snapshots),
            'num_temporal_features': len(self.temporal_features)
        }