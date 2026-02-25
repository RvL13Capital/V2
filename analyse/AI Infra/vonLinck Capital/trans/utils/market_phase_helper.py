"""
Market Phase Helper
===================

Simple helper to look up S&P 500 market phase (Bull/Bear) for any date.
Uses pre-computed phase data from data/market_phases/USA500IDXUSD_phases.csv.

Usage:
    from utils.market_phase_helper import get_market_phase, MarketPhase

    phase = get_market_phase('2020-03-15')
    print(f"Phase: {phase.name}")  # BEAR
    print(f"Value: {phase.value}")  # 0

    # For batch lookups (more efficient)
    phases = get_market_phases(['2020-03-15', '2024-01-01'])
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, List
from datetime import datetime
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)

# Market phase enum
class MarketPhase(IntEnum):
    """Market phase enumeration."""
    BEAR = 0
    BULL = 1
    UNKNOWN = -1

    def __str__(self):
        return self.name


# Global cache for phase data
_phase_cache: Optional[Dict[str, int]] = None
_phase_df: Optional[pd.DataFrame] = None


def _load_phase_data() -> Dict[str, int]:
    """Load phase data from CSV into memory (singleton pattern)."""
    global _phase_cache, _phase_df

    if _phase_cache is not None:
        return _phase_cache

    # Find the phases CSV
    base_path = Path(__file__).parent.parent
    csv_path = base_path / 'data' / 'market_phases' / 'USA500IDXUSD_phases.csv'

    if not csv_path.exists():
        logger.warning(f"Market phase data not found: {csv_path}")
        _phase_cache = {}
        return _phase_cache

    # Load CSV
    logger.info(f"Loading market phase data from {csv_path}")
    df = pd.read_csv(csv_path)
    _phase_df = df

    # Build lookup dictionary: date_str -> phase
    _phase_cache = {}
    for _, row in df.iterrows():
        date_str = str(row['date'])
        phase = row['phase']
        if pd.notna(phase):
            _phase_cache[date_str] = int(phase)

    logger.info(f"Loaded {len(_phase_cache)} days of market phase data")
    return _phase_cache


def get_market_phase(date_input: Union[str, datetime, pd.Timestamp]) -> MarketPhase:
    """
    Get market phase for a specific date.

    Args:
        date_input: Date as string ('YYYY-MM-DD'), datetime, or Timestamp

    Returns:
        MarketPhase enum (BULL=1, BEAR=0, UNKNOWN=-1)
    """
    cache = _load_phase_data()

    # Normalize date to string
    if isinstance(date_input, (datetime, pd.Timestamp)):
        date_str = date_input.strftime('%Y-%m-%d')
    else:
        date_str = str(pd.to_datetime(date_input).strftime('%Y-%m-%d'))

    phase = cache.get(date_str)

    if phase is None:
        # Try to find nearest previous date (handles weekends/holidays)
        return _get_nearest_phase(date_str, cache)

    return MarketPhase(phase)


def _get_nearest_phase(date_str: str, cache: Dict[str, int]) -> MarketPhase:
    """Find nearest previous date with phase data."""
    target_date = pd.to_datetime(date_str)

    # Look back up to 10 days for nearest trading day
    for i in range(1, 11):
        prev_date = (target_date - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
        if prev_date in cache:
            return MarketPhase(cache[prev_date])

    return MarketPhase.UNKNOWN


def get_market_phases(dates: List[Union[str, datetime]]) -> Dict[str, MarketPhase]:
    """
    Get market phases for multiple dates (batch lookup).

    Args:
        dates: List of dates

    Returns:
        Dictionary mapping date strings to MarketPhase
    """
    cache = _load_phase_data()
    results = {}

    for date in dates:
        if isinstance(date, (datetime, pd.Timestamp)):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(pd.to_datetime(date).strftime('%Y-%m-%d'))

        results[date_str] = get_market_phase(date)

    return results


def get_phase_stats() -> Dict[str, any]:
    """Get summary statistics about market phases."""
    cache = _load_phase_data()

    if not cache:
        return {'error': 'No phase data available'}

    bull_days = sum(1 for p in cache.values() if p == 1)
    bear_days = sum(1 for p in cache.values() if p == 0)
    total_days = len(cache)

    # Get date range
    dates = sorted(cache.keys())

    return {
        'total_days': total_days,
        'bull_days': bull_days,
        'bear_days': bear_days,
        'bull_pct': round(bull_days / total_days * 100, 1) if total_days > 0 else 0,
        'bear_pct': round(bear_days / total_days * 100, 1) if total_days > 0 else 0,
        'start_date': dates[0] if dates else None,
        'end_date': dates[-1] if dates else None
    }


def get_current_phase() -> Dict[str, any]:
    """Get current market phase information."""
    global _phase_df

    # Ensure data is loaded
    _load_phase_data()

    if _phase_df is None or len(_phase_df) == 0:
        return {'error': 'No phase data available'}

    # Get latest row with valid phase
    valid_rows = _phase_df[_phase_df['phase'].notna()]
    if len(valid_rows) == 0:
        return {'error': 'No valid phase data'}

    latest = valid_rows.iloc[-1]

    return {
        'date': latest['date'],
        'phase': 'BULL' if latest['phase'] == 1 else 'BEAR',
        'phase_value': int(latest['phase']),
        'days_in_phase': int(latest['days_in_phase']),
        'crossover_date': latest['crossover_date'],
        'fast_ma': round(latest['fast_ma'], 2),
        'slow_ma': round(latest['slow_ma'], 2)
    }


# For convenience, export these at module level
def is_bull_market(date: Union[str, datetime]) -> bool:
    """Check if date is in a bull market."""
    return get_market_phase(date) == MarketPhase.BULL


def is_bear_market(date: Union[str, datetime]) -> bool:
    """Check if date is in a bear market."""
    return get_market_phase(date) == MarketPhase.BEAR


if __name__ == '__main__':
    # Quick test
    print("Market Phase Helper Test")
    print("=" * 40)

    # Test current phase
    current = get_current_phase()
    print(f"\nCurrent Market Phase: {current}")

    # Test specific dates
    test_dates = ['2020-03-15', '2020-11-01', '2024-01-01', '2024-06-15']
    print(f"\nTest dates:")
    for d in test_dates:
        phase = get_market_phase(d)
        print(f"  {d}: {phase.name} ({phase.value})")

    # Stats
    stats = get_phase_stats()
    print(f"\nStatistics: {stats}")
