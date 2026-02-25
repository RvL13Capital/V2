"""
Temporal Boundary Manager - Boundary Calculation and Tracking
=============================================================

Manages consolidation pattern boundaries with temporal integrity.
Boundaries are set once at activation and remain immutable.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import logging
from .pattern_state import PatternState

logger = logging.getLogger(__name__)


class TemporalBoundaryManager:
    """
    Manages pattern boundaries with temporal constraints.

    Boundaries are calculated from qualification period data
    and frozen at activation time to maintain temporal integrity.
    """

    def __init__(self, power_buffer: float = 0.005):
        """
        Initialize boundary manager.

        Args:
            power_buffer: Buffer above upper boundary for power boundary (default 0.5%)
        """
        self.power_buffer = power_buffer

    def calculate_boundaries_from_qualification(
        self,
        qualification_highs: List[float],
        qualification_lows: List[float],
        set_date: datetime
    ) -> Dict[str, Any]:
        """
        Calculate boundaries from qualification period data using PERCENTILE-BASED ROBUSTNESS.

        Uses 90th/10th percentile of High/Low to create a "Volatility Envelope"
        that ignores the top 10% of manipulation spikes (scam wicks).

        This allows patterns to survive longer (60-100 days) by not being
        killed by single stop-hunt events.

        Boundaries are set ONCE at pattern activation (day 10) and
        remain fixed throughout the pattern's active phase.

        Args:
            qualification_highs: High prices during days 1-10
            qualification_lows: Low prices during days 1-10
            set_date: When boundaries are being set

        Returns:
            Dictionary with boundary values
        """
        if not qualification_highs or not qualification_lows:
            raise ValueError("Qualification data required for boundary calculation")

        if len(qualification_highs) < 10 or len(qualification_lows) < 10:
            logger.warning(
                f"Insufficient qualification data: "
                f"{len(qualification_highs)} highs, {len(qualification_lows)} lows"
            )

        # OPTION B: 90th/10th Percentile of High/Low (Robust Boundaries)
        # This crops out the top 10% of "scam wicks" that destroy pattern duration
        upper_boundary = float(np.percentile(qualification_highs, 90))
        lower_boundary = float(np.percentile(qualification_lows, 10))

        # SAFETY: Prevent collapse on flat stocks (zero variance)
        if upper_boundary <= lower_boundary:
            # Use mean of highs/lows and add small buffer
            center_high = float(np.mean(qualification_highs))
            center_low = float(np.mean(qualification_lows))
            center = (center_high + center_low) / 2
            upper_boundary = center * 1.02
            lower_boundary = center * 0.98

        # Power boundary is upper + buffer
        power_boundary = upper_boundary * (1 + self.power_buffer)

        # Calculate range percentage
        boundary_range_pct = (
            (upper_boundary - lower_boundary) / lower_boundary
            if lower_boundary > 0 else 0
        )

        boundaries = {
            'upper_boundary': upper_boundary,
            'lower_boundary': lower_boundary,
            'power_boundary': power_boundary,
            'boundary_range_pct': boundary_range_pct,
            'boundary_set_date': set_date
        }

        logger.info(
            f"Boundaries set at {set_date} (90th/10th percentile): "
            f"Upper={upper_boundary:.2f}, Lower={lower_boundary:.2f}, "
            f"Power={power_boundary:.2f}, Range={boundary_range_pct:.2%}"
        )

        return boundaries

    def set_boundaries_on_state(
        self,
        state: PatternState,
        qualification_highs: List[float],
        qualification_lows: List[float],
        set_date: datetime
    ) -> PatternState:
        """
        Set boundaries on pattern state.

        Args:
            state: Current pattern state
            qualification_highs: Highs from qualification period
            qualification_lows: Lows from qualification period
            set_date: When boundaries are set

        Returns:
            New PatternState with boundaries set
        """
        # Ensure boundaries haven't been set already (temporal integrity)
        if state.boundary_set_date is not None:
            raise ValueError(
                f"Boundaries already set at {state.boundary_set_date}. "
                "Boundaries are immutable for temporal integrity."
            )

        # Calculate boundaries
        boundaries = self.calculate_boundaries_from_qualification(
            qualification_highs,
            qualification_lows,
            set_date
        )

        # Return new state with boundaries
        return state.with_boundaries(
            upper=boundaries['upper_boundary'],
            lower=boundaries['lower_boundary'],
            set_date=set_date
        )

    def check_boundary_violations(
        self,
        state: PatternState,
        current_price: float,
        current_high: float,
        current_low: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if CLOSE price violates pattern boundaries (THE "IRON CORE" CHECK).

        Wicks are free - only CLOSES count as structural breaks.
        Boundaries are "Volatility Envelope" (90th % High) as limit for Closes.

        This allows intraday volatility (stop hunts) without killing the pattern,
        enabling patterns to survive 60-100+ days.

        Args:
            state: Current pattern state with boundaries
            current_price: Current closing price (THIS IS WHAT WE CHECK)
            current_high: Current high price (IGNORED for violation check)
            current_low: Current low price (IGNORED for violation check)

        Returns:
            Tuple of (violation_occurred, violation_type)
            violation_type can be 'BREAKOUT_UP', 'BREAKDOWN', or None
        """
        if state.upper_boundary is None or state.lower_boundary is None:
            logger.warning("Boundaries not set, cannot check violations")
            return False, None

        # Check for breakout: CLOSE above upper boundary
        # NOTE: We check against upper_boundary (not power_boundary) for close
        if current_price > state.upper_boundary:
            logger.info(
                f"BREAKOUT detected: Close {current_price:.2f} > "
                f"Upper boundary {state.upper_boundary:.2f}"
            )
            return True, 'BREAKOUT_UP'

        # Check for breakdown: CLOSE below lower boundary
        if current_price < state.lower_boundary:
            logger.info(
                f"BREAKDOWN detected: Close {current_price:.2f} < "
                f"Lower boundary {state.lower_boundary:.2f}"
            )
            return True, 'BREAKDOWN'

        # Intraday wicks outside boundaries are ALLOWED and ignored
        return False, None

    def calculate_price_position_features(
        self,
        state: PatternState,
        current_price: float
    ) -> Dict[str, float]:
        """
        Calculate price position relative to boundaries.

        Args:
            state: Pattern state with boundaries
            current_price: Current price

        Returns:
            Dictionary of price position features
        """
        if state.upper_boundary is None or state.lower_boundary is None:
            return {
                'price_position_in_range': 0.5,
                'price_distance_from_upper_pct': 0.0,
                'price_distance_from_lower_pct': 0.0
            }

        # Position within range (0 = at lower, 1 = at upper)
        range_width = state.upper_boundary - state.lower_boundary
        if range_width > 0:
            price_position = (current_price - state.lower_boundary) / range_width
            price_position = max(0.0, min(1.0, price_position))  # Clamp to [0, 1]
        else:
            price_position = 0.5

        # Distance from boundaries as percentage
        distance_from_upper_pct = (
            (state.upper_boundary - current_price) / current_price
            if current_price > 0 else 0.0
        )

        distance_from_lower_pct = (
            (current_price - state.lower_boundary) / current_price
            if current_price > 0 else 0.0
        )

        return {
            'price_position_in_range': price_position,
            'price_distance_from_upper_pct': distance_from_upper_pct,
            'price_distance_from_lower_pct': distance_from_lower_pct
        }

    def validate_boundary_integrity(
        self,
        state: PatternState
    ) -> bool:
        """
        Validate boundary integrity and relationships.

        Args:
            state: Pattern state to validate

        Returns:
            True if boundaries are valid
        """
        if state.upper_boundary is None or state.lower_boundary is None:
            # Boundaries not set yet - this is valid
            return True

        # Check boundary relationships
        if state.upper_boundary <= state.lower_boundary:
            logger.error(
                f"Invalid boundaries: upper {state.upper_boundary} <= "
                f"lower {state.lower_boundary}"
            )
            return False

        if state.power_boundary <= state.upper_boundary:
            logger.error(
                f"Invalid power boundary: {state.power_boundary} <= "
                f"upper {state.upper_boundary}"
            )
            return False

        # Check boundary range is reasonable (not too wide)
        if state.boundary_range_pct > 0.5:  # 50% range
            logger.warning(
                f"Unusually wide boundary range: {state.boundary_range_pct:.2%}"
            )

        return True

    def get_boundary_metrics(
        self,
        state: PatternState,
        price_data: pd.DataFrame,
        current_idx: int
    ) -> Dict[str, Any]:
        """
        Get comprehensive boundary metrics.

        Args:
            state: Pattern state with boundaries
            price_data: Price DataFrame
            current_idx: Current index position

        Returns:
            Dictionary of boundary metrics
        """
        if state.upper_boundary is None:
            return {}

        metrics = {
            'upper_boundary': state.upper_boundary,
            'lower_boundary': state.lower_boundary,
            'power_boundary': state.power_boundary,
            'boundary_range_pct': state.boundary_range_pct,
            'boundary_set_date': state.boundary_set_date.isoformat()
                if state.boundary_set_date else None
        }

        # Add current price position if available
        if current_idx < len(price_data):
            current_price = price_data.iloc[current_idx]['close']
            position_features = self.calculate_price_position_features(
                state,
                current_price
            )
            metrics.update(position_features)

        # Days since boundaries set
        if state.boundary_set_date and state.activated_at:
            metrics['days_since_boundary_set'] = state.days_since_activation

        return metrics

    def are_boundaries_set(self, state: PatternState) -> bool:
        """Check if boundaries have been set."""
        return state.upper_boundary is not None