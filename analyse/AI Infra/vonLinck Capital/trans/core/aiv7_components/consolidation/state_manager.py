"""
Sequential State Manager - State Machine with Temporal Ordering
===============================================================

Manages pattern phase transitions with strict temporal ordering.
Ensures valid state transitions and maintains transition history.
"""

from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import logging
from .pattern_state import PatternPhase, PatternState, STATE_PRIORITY

logger = logging.getLogger(__name__)


class SequentialStateManager:
    """
    Manages state transitions with strict sequential ordering.

    Enforces the valid transition sequence:
    NONE → QUALIFYING → ACTIVE → COMPLETED/FAILED → NONE
    """

    # Valid state transitions
    # 2 consecutive closes required for COMPLETED/FAILED from ACTIVE
    # FAILED can become RECOVERING (high > upper) or DEAD (low < stop)
    # RECOVERING can become DEAD (low < stop)
    # All terminal states can start new QUALIFYING through NONE
    VALID_TRANSITIONS = {
        PatternPhase.NONE: [PatternPhase.QUALIFYING],
        PatternPhase.QUALIFYING: [PatternPhase.ACTIVE, PatternPhase.NONE],
        PatternPhase.ACTIVE: [PatternPhase.COMPLETED, PatternPhase.FAILED],
        PatternPhase.FAILED: [PatternPhase.RECOVERING, PatternPhase.DEAD, PatternPhase.NONE],
        PatternPhase.RECOVERING: [PatternPhase.DEAD, PatternPhase.NONE],
        PatternPhase.COMPLETED: [PatternPhase.NONE],
        PatternPhase.DEAD: [PatternPhase.NONE]
    }

    def __init__(self):
        """Initialize state manager."""
        self.transition_history: List[Tuple[PatternPhase, PatternPhase, datetime]] = []
        self.current_state: Optional[PatternState] = None

    def initialize_state(
        self,
        ticker: str,
        start_date: datetime,
        start_idx: int,
        start_price: float
    ) -> PatternState:
        """
        Initialize a new pattern state.

        Args:
            ticker: Stock ticker
            start_date: Pattern start date
            start_idx: Start index in data
            start_price: Starting price

        Returns:
            New PatternState instance
        """
        self.current_state = PatternState(
            ticker=ticker,
            start_date=start_date,
            start_idx=start_idx,
            start_price=start_price,
            phase=PatternPhase.NONE
        )
        return self.current_state

    def transition(
        self,
        to_phase: PatternPhase,
        transition_date: datetime,
        **kwargs
    ) -> PatternState:
        """
        Perform state transition with validation.

        Args:
            to_phase: Target phase
            transition_date: When transition occurs
            **kwargs: Additional state updates

        Returns:
            Updated PatternState

        Raises:
            ValueError: If transition is invalid
        """
        if self.current_state is None:
            raise ValueError("No current state to transition from")

        from_phase = self.current_state.phase

        # Validate transition
        if not self.is_valid_transition(from_phase, to_phase):
            raise ValueError(
                f"Invalid transition: {from_phase.value} → {to_phase.value}. "
                f"Valid transitions: {[p.value for p in self.VALID_TRANSITIONS[from_phase]]}"
            )

        # Log transition
        logger.info(
            f"{self.current_state.ticker}: {from_phase.value} → {to_phase.value} "
            f"at {transition_date}"
        )

        # Record in history
        self.transition_history.append((from_phase, to_phase, transition_date))

        # Create new state with transition
        self.current_state = self.current_state.with_phase_transition(
            to_phase,
            transition_date,
            **kwargs
        )

        return self.current_state

    def is_valid_transition(
        self,
        from_phase: PatternPhase,
        to_phase: PatternPhase
    ) -> bool:
        """
        Check if transition is valid.

        Args:
            from_phase: Current phase
            to_phase: Target phase

        Returns:
            True if valid, False otherwise
        """
        return to_phase in self.VALID_TRANSITIONS.get(from_phase, [])

    def get_phase(self) -> Optional[PatternPhase]:
        """Get current phase."""
        return self.current_state.phase if self.current_state else None

    def get_state(self) -> Optional[PatternState]:
        """Get current state."""
        return self.current_state

    def reset(self) -> None:
        """Reset state manager for new pattern detection."""
        if self.current_state and not self.current_state.is_terminal():
            logger.warning(
                f"Resetting non-terminal state: {self.current_state.phase.value}"
            )

        self.current_state = None
        # Keep history for analysis

    def can_start_qualification(self) -> bool:
        """Check if qualification can start."""
        return (
            self.current_state is None or
            self.current_state.phase == PatternPhase.NONE
        )

    def can_activate(self) -> bool:
        """Check if pattern can be activated."""
        return (
            self.current_state is not None and
            self.current_state.phase == PatternPhase.QUALIFYING and
            self.current_state.days_qualifying >= 10
        )

    def can_complete(self) -> bool:
        """Check if pattern can complete."""
        return (
            self.current_state is not None and
            self.current_state.phase == PatternPhase.ACTIVE
        )

    def is_in_pattern(self) -> bool:
        """Check if currently in an active pattern."""
        return (
            self.current_state is not None and
            self.current_state.phase in [
                PatternPhase.QUALIFYING,
                PatternPhase.ACTIVE
            ]
        )

    def get_transition_history(self) -> List[Dict[str, Any]]:
        """
        Get transition history for analysis.

        Returns:
            List of transition records
        """
        return [
            {
                'from': from_phase.value,
                'to': to_phase.value,
                'timestamp': timestamp.isoformat()
            }
            for from_phase, to_phase, timestamp in self.transition_history
        ]

    def validate_sequence(self) -> bool:
        """
        Validate that transition history follows valid sequence.

        Returns:
            True if all transitions are valid
        """
        for from_phase, to_phase, _ in self.transition_history:
            if not self.is_valid_transition(from_phase, to_phase):
                logger.error(
                    f"Invalid transition in history: "
                    f"{from_phase.value} → {to_phase.value}"
                )
                return False
        return True

    def get_phase_duration(self, phase: PatternPhase) -> Optional[int]:
        """
        Get duration spent in specific phase.

        Args:
            phase: Phase to check

        Returns:
            Duration in days, or None if not in that phase
        """
        if self.current_state is None:
            return None

        if phase == PatternPhase.QUALIFYING:
            return self.current_state.days_qualifying
        elif phase == PatternPhase.ACTIVE:
            return self.current_state.days_since_activation
        else:
            return None

    def update_day_counters(self) -> PatternState:
        """
        Increment day counters based on current phase.

        Returns:
            Updated PatternState
        """
        if self.current_state is None:
            raise ValueError("No current state to update")

        self.current_state = self.current_state.increment_days()
        return self.current_state

    def process_price_action(
        self,
        close_price: float,
        low_price: float,
        current_date: datetime,
        high_price: Optional[float] = None
    ) -> PatternState:
        """
        Process daily price action and handle state transitions.

        Implements:
        - ACTIVE -> COMPLETED: 2 consecutive closes above upper boundary
        - ACTIVE -> FAILED: 2 consecutive closes below lower boundary
        - FAILED -> RECOVERING: high > upper_boundary
        - FAILED -> DEAD: low < last_active_lowest_close
        - RECOVERING -> DEAD: low < last_active_lowest_close

        Priority (worse state wins on same candle):
        COMPLETED(0) < RECOVERING(1) < FAILED(2) < DEAD(3)

        Args:
            close_price: Today's closing price
            low_price: Today's low price
            current_date: Current date for transition timestamps
            high_price: Today's high price (needed for RECOVERING detection)

        Returns:
            Updated PatternState (may have transitioned)
        """
        if self.current_state is None:
            raise ValueError("No current state to process")

        # Update consecutive counters if in ACTIVE phase
        if self.current_state.phase == PatternPhase.ACTIVE:
            self.current_state = self.current_state.update_consecutive_counters(
                close_price, low_price
            )

            # Check for transitions - priority: worse state wins
            should_complete = self.current_state.should_complete()
            should_fail = self.current_state.should_fail()

            if should_complete and should_fail:
                # Both conditions met - FAILED is worse (priority 2 > 0)
                self.current_state = self.transition(
                    PatternPhase.FAILED,
                    current_date,
                    end_price=close_price,
                    breakout_direction='DOWN',
                    signals_allowed=True
                )
            elif should_fail:
                self.current_state = self.transition(
                    PatternPhase.FAILED,
                    current_date,
                    end_price=close_price,
                    breakout_direction='DOWN',
                    signals_allowed=True
                )
            elif should_complete:
                self.current_state = self.transition(
                    PatternPhase.COMPLETED,
                    current_date,
                    end_price=close_price,
                    breakout_direction='UP',
                    signals_allowed=False
                )

        # Check for RECOVERING or DEAD transition if in FAILED phase
        elif self.current_state.phase == PatternPhase.FAILED:
            should_recover = (
                high_price is not None and
                self.current_state.should_recover(high_price)
            )
            should_die = self.current_state.should_die(low_price)

            if should_recover and should_die:
                # Both conditions met - DEAD is worse (priority 3 > 1)
                self.current_state = self.transition(
                    PatternPhase.DEAD,
                    current_date,
                    signals_allowed=False
                )
            elif should_die:
                self.current_state = self.transition(
                    PatternPhase.DEAD,
                    current_date,
                    signals_allowed=False
                )
            elif should_recover:
                self.current_state = self.transition(
                    PatternPhase.RECOVERING,
                    current_date,
                    signals_allowed=False
                )

        # Check for DEAD transition if in RECOVERING phase
        elif self.current_state.phase == PatternPhase.RECOVERING:
            if self.current_state.should_die(low_price):
                self.current_state = self.transition(
                    PatternPhase.DEAD,
                    current_date,
                    signals_allowed=False
                )

        return self.current_state

    def can_fail(self) -> bool:
        """Check if pattern can transition to FAILED (2 consecutive closes below)."""
        return (
            self.current_state is not None and
            self.current_state.phase == PatternPhase.ACTIVE and
            self.current_state.consecutive_close_below >= 2
        )

    def can_recover(self, current_high: float) -> bool:
        """Check if FAILED pattern should become RECOVERING."""
        return (
            self.current_state is not None and
            self.current_state.should_recover(current_high)
        )

    def can_die(self, current_low: float) -> bool:
        """Check if FAILED/RECOVERING pattern should become DEAD."""
        return (
            self.current_state is not None and
            self.current_state.should_die(current_low)
        )

    def is_signals_allowed(self, close_price: Optional[float] = None) -> bool:
        """
        Check if current state allows signals.

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
        if self.current_state is None:
            return False

        phase = self.current_state.phase

        # FAILED always allows signals (recovery play)
        if phase == PatternPhase.FAILED:
            return True

        # ACTIVE requires close < upper_boundary
        if phase == PatternPhase.ACTIVE:
            if close_price is None:
                return False
            if self.current_state.upper_boundary is None:
                return False
            return close_price < self.current_state.upper_boundary

        # RECOVERING and all other phases: no signals
        return False