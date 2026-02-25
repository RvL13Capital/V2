"""
MicroCap Bond Filter - Strict Boundary Logic for Consolidation Patterns
========================================================================

Implements Day-50 Bond Logic with milestone adjustments.

Phase 1 (Days 0-49): Survival Mode - No close breaks allowed (with 0.5% tolerance)
Phase 2 (Day 50+):   Boundary Adjustment - Tightening at milestones
Phase 3 (Post-50):   Exit conditions with breakdown confirmation

Key Features:
- 0.5% wick tolerance to prevent false failures from intraday noise
- Multiple milestone adjustments (50, 150, 300, 450, 600 days)
- Breakdown confirmation (2 consecutive closes OR 5+ lows in 10 days)
- Boundaries only tighten, never expand
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("BondFilter")


@dataclass
class BondState:
    """Structured return type for bond evaluation."""
    status: str                    # "ACTIVE", "BROKEN", "BREAKOUT", "BREAKDOWN"
    current_high: float            # The valid upper boundary for today
    current_low: float             # The valid lower boundary for today
    milestone_reached: bool        # Has any milestone been reached?
    current_milestone: int = 0     # Which milestone (0, 50, 150, 300...)
    break_day: Optional[int] = None
    tightening_pct: float = 0.0
    consecutive_closes_below: int = 0
    lows_below_10d: int = 0
    original_high: float = 0.0     # Original upper boundary (never changes)
    original_low: float = 0.0      # Original lower boundary (never changes)


class MicroCapBondFilter:
    """
    Implements the strict Day-50 Bond Logic.

    Phase 1 (0-49): Survival Mode (No Close breaks allowed with tolerance).
    Phase 2 (50+):  Boundary Adjustment (Lock-in at milestones).
    Phase 3 (51+):  Exit conditions with breakdown confirmation.

    Optimizations:
    - 0.5% wick tolerance (from BreakoutAI1.0)
    - Vectorized break checking
    - Multiple milestone support
    """

    def __init__(self):
        # Configuration
        self.MILESTONES = [50, 150, 300, 450, 600, 750, 900]  # Days for boundary adjustments
        self.FLOOR_LOOKBACK = 40   # Days to look back for new floor
        self.CEILING_LOOKBACK = 50  # Days to look back for new ceiling
        self.WICK_TOLERANCE = 0.005  # 0.5% tolerance for wicks

        # Breakdown confirmation settings
        self.CONSECUTIVE_CLOSES_REQUIRED = 2
        self.LOWS_BELOW_IN_10D_REQUIRED = 5

    def evaluate_bond(
        self,
        df: pd.DataFrame,
        initial_high: float,
        initial_low: float
    ) -> BondState:
        """
        Evaluates the bond health and calculates boundaries.

        Args:
            df: DataFrame starting from Day 0 of the bond (ACTIVE phase).
            initial_high: The ceiling established at Day 0 (activation).
            initial_low: The floor established at Day 0 (activation).

        Returns:
            BondState with current status and boundaries.
        """
        days_active = len(df)

        if days_active == 0:
            return BondState(
                status="WAITING_FOR_DATA",
                current_high=initial_high,
                current_low=initial_low,
                milestone_reached=False,
                original_high=initial_high,
                original_low=initial_low
            )

        # Calculate tolerance-adjusted boundaries for Phase 1
        upper_tolerance = initial_high * (1 + self.WICK_TOLERANCE)
        lower_tolerance = initial_low * (1 - self.WICK_TOLERANCE)

        # --- PHASE 1: SURVIVAL CHECK (Days 0 to first milestone - 1) ---
        first_milestone = self.MILESTONES[0]  # Day 50
        check_window = min(days_active, first_milestone)
        survival_slice = df.iloc[:check_window]

        # Get close column (handle both cases)
        close_col = 'close' if 'close' in df.columns else 'Close'

        # Vectorized break check with tolerance
        broken_high = survival_slice[close_col] > upper_tolerance
        broken_low = survival_slice[close_col] < lower_tolerance

        if broken_high.any() or broken_low.any():
            # Find the first day it broke
            if broken_high.any():
                first_break_idx = broken_high.idxmax()
            else:
                first_break_idx = broken_low.idxmax()

            # Get integer location
            try:
                loc = df.index.get_loc(first_break_idx)
            except:
                loc = list(df.index).index(first_break_idx)

            return BondState(
                status="BROKEN",
                current_high=initial_high,
                current_low=initial_low,
                milestone_reached=False,
                break_day=int(loc),
                original_high=initial_high,
                original_low=initial_low
            )

        # If we are here, the bond survived the check window.

        # --- PHASE 2: MILESTONE ADJUSTMENTS ---
        if days_active < first_milestone:
            # Not yet reached first milestone. Boundaries remain initial.
            return BondState(
                status="ACTIVE",
                current_high=initial_high,
                current_low=initial_low,
                milestone_reached=False,
                current_milestone=0,
                original_high=initial_high,
                original_low=initial_low
            )

        # Determine which milestone we're at
        current_milestone = 0
        for milestone in self.MILESTONES:
            if days_active >= milestone:
                current_milestone = milestone
            else:
                break

        # Calculate adjusted boundaries based on current milestone
        new_high, new_low, tightening = self._calculate_adjusted_boundaries(
            df, initial_high, initial_low, current_milestone
        )

        # --- PHASE 3: POST-MILESTONE EXIT CONDITIONS ---
        if days_active > first_milestone:
            # Check for breakout (close > new upper)
            latest_close = df[close_col].iloc[-1]

            if latest_close > new_high * (1 + self.WICK_TOLERANCE):
                return BondState(
                    status="BREAKOUT",
                    current_high=new_high,
                    current_low=new_low,
                    milestone_reached=True,
                    current_milestone=current_milestone,
                    tightening_pct=tightening,
                    original_high=initial_high,
                    original_low=initial_low
                )

            # Check for breakdown with confirmation
            breakdown_confirmed, consec_closes, lows_count = self._check_breakdown_confirmation(
                df, new_low
            )

            if breakdown_confirmed:
                return BondState(
                    status="BREAKDOWN",
                    current_high=new_high,
                    current_low=new_low,
                    milestone_reached=True,
                    current_milestone=current_milestone,
                    tightening_pct=tightening,
                    consecutive_closes_below=consec_closes,
                    lows_below_10d=lows_count,
                    original_high=initial_high,
                    original_low=initial_low
                )

        # Pattern is still active
        return BondState(
            status="ACTIVE",
            current_high=round(new_high, 4),
            current_low=round(new_low, 4),
            milestone_reached=True,
            current_milestone=current_milestone,
            tightening_pct=round(tightening, 1),
            original_high=initial_high,
            original_low=initial_low
        )

    def _calculate_adjusted_boundaries(
        self,
        df: pd.DataFrame,
        initial_high: float,
        initial_low: float,
        milestone: int
    ) -> tuple:
        """
        Calculate adjusted boundaries at a milestone.

        Uses fixed 40-day low and 50-day high lookbacks.
        Boundaries only tighten, never expand.

        Returns:
            (new_high, new_low, tightening_pct)
        """
        # Get column names
        low_col = 'low' if 'low' in df.columns else 'Low'
        high_col = 'high' if 'high' in df.columns else 'High'

        # Get data up to the milestone
        milestone_slice = df.iloc[:milestone]

        if len(milestone_slice) < self.FLOOR_LOOKBACK:
            # Not enough data, use initial boundaries
            return initial_high, initial_low, 0.0

        # 1. Calculate Potential New Floor (Low of past 40 days)
        floor_slice = milestone_slice[low_col].tail(self.FLOOR_LOOKBACK)
        lowest_low_40 = floor_slice.min()

        # Logic: Adjust ONLY if higher or equal (Tightening from below)
        new_low = max(initial_low, lowest_low_40)

        # 2. Calculate Potential New Ceiling (High of past 50 days)
        ceiling_slice = milestone_slice[high_col].tail(self.CEILING_LOOKBACK)
        highest_high_50 = ceiling_slice.max()

        # Logic: Adjust ONLY if lower or equal (Tightening from above)
        new_high = min(initial_high, highest_high_50)

        # Calculate tightening percentage
        initial_range = initial_high - initial_low
        new_range = new_high - new_low

        if initial_range > 0:
            tightening = (1 - (new_range / initial_range)) * 100
        else:
            tightening = 0.0

        return new_high, new_low, tightening

    def _check_breakdown_confirmation(
        self,
        df: pd.DataFrame,
        lower_boundary: float
    ) -> tuple:
        """
        Check if breakdown is confirmed with coil logic.

        Confirmation requires EITHER:
        - 2 consecutive closes below lower boundary, OR
        - 5+ days in last 10 where low was below lower boundary

        Returns:
            (is_confirmed, consecutive_closes, lows_below_count)
        """
        # Get column names
        close_col = 'close' if 'close' in df.columns else 'Close'
        low_col = 'low' if 'low' in df.columns else 'Low'

        # Get last 10 days
        recent = df.tail(10)

        if len(recent) < 2:
            return False, 0, 0

        # Check for consecutive closes below
        closes = recent[close_col]
        consecutive_closes = 0

        # Count consecutive closes from the end
        for i in range(len(closes) - 1, -1, -1):
            if closes.iloc[i] < lower_boundary:
                consecutive_closes += 1
            else:
                break

        # Check for lows below boundary in last 10 days
        lows = recent[low_col]
        lows_below_count = (lows < lower_boundary).sum()

        # Determine if confirmed
        is_confirmed = (
            consecutive_closes >= self.CONSECUTIVE_CLOSES_REQUIRED or
            lows_below_count >= self.LOWS_BELOW_IN_10D_REQUIRED
        )

        return is_confirmed, consecutive_closes, lows_below_count

    def get_next_milestone(self, days_active: int) -> Optional[int]:
        """Get the next milestone day from current position."""
        for milestone in self.MILESTONES:
            if days_active < milestone:
                return milestone
        return None

    def get_days_to_milestone(self, days_active: int) -> int:
        """Get days remaining until next milestone."""
        next_milestone = self.get_next_milestone(days_active)
        if next_milestone:
            return next_milestone - days_active
        return 0


# Convenience function for direct usage
def evaluate_bond_health(
    df: pd.DataFrame,
    initial_high: float,
    initial_low: float
) -> BondState:
    """
    Convenience function to evaluate bond health.

    Args:
        df: DataFrame with price data from pattern start
        initial_high: Upper boundary at activation
        initial_low: Lower boundary at activation

    Returns:
        BondState with current status
    """
    filter = MicroCapBondFilter()
    return filter.evaluate_bond(df, initial_high, initial_low)


if __name__ == "__main__":
    # Test the bond filter
    import numpy as np

    # Create sample data (60 days)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=60, freq='D')

    # Simulate consolidation with slight random walk
    base_price = 10.0
    prices = base_price + np.cumsum(np.random.randn(60) * 0.05)

    df = pd.DataFrame({
        'date': dates,
        'open': prices - 0.1,
        'high': prices + 0.2,
        'low': prices - 0.2,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 60)
    }).set_index('date')

    # Initial boundaries
    initial_high = df['high'].iloc[:10].max()
    initial_low = df['low'].iloc[:10].min()

    print(f"Initial boundaries: {initial_low:.2f} - {initial_high:.2f}")

    # Test bond filter
    bond_filter = MicroCapBondFilter()
    result = bond_filter.evaluate_bond(df, initial_high, initial_low)

    print(f"\nBond Status: {result.status}")
    print(f"Current boundaries: {result.current_low:.2f} - {result.current_high:.2f}")
    print(f"Milestone reached: {result.milestone_reached}")
    print(f"Current milestone: {result.current_milestone}")
    print(f"Tightening: {result.tightening_pct:.1f}%")

    if result.status == "BROKEN":
        print(f"Break day: {result.break_day}")
    elif result.status == "BREAKDOWN":
        print(f"Consecutive closes below: {result.consecutive_closes_below}")
        print(f"Lows below in 10d: {result.lows_below_10d}")
