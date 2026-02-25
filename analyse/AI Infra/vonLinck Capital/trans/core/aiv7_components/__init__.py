"""
AIv7 Core Components - V17 TRANS Version
=========================================

This package contains essential components for TRANS v17:
- ConsolidationTracker: Pattern detection state machine (REFACTORED modular version)
- BondFilter: Boundary management with milestone adjustments
- DataLoader: Data loading with GCS support
- Technical Indicators: BBW, ADX, RSI, etc.

These are standalone versions that don't require the full AIv7 codebase.

V17 REFACTORING:
- Now uses ONLY the refactored modular version (350 lines)
- Legacy version removed as part of v17-only migration
"""

# Import refactored version as ConsolidationTracker (standard name)
from .consolidation_tracker_refactored import RefactoredConsolidationTracker as ConsolidationTracker

# Import supporting types from refactored version
from .consolidation import PatternPhase, PatternState

# Use PatternState as ConsolidationPattern for backward compatibility
ConsolidationPattern = PatternState

from .bond_filter import (
    MicroCapBondFilter,
    BondState,
    evaluate_bond_health
)

from .data_loader import DataLoader

from .indicators import (
    calculate_bbw,
    calculate_adx,
    calculate_rsi,
    calculate_atr,
    calculate_all_indicators
)

__all__ = [
    # Pattern detection (V17 - refactored only)
    'ConsolidationTracker',
    'PatternPhase',
    'PatternState',
    'ConsolidationPattern',  # Temporary - for backward compatibility

    # Bond filter
    'MicroCapBondFilter',
    'BondState',
    'evaluate_bond_health',

    # Data loading
    'DataLoader',

    # Technical indicators
    'calculate_bbw',
    'calculate_adx',
    'calculate_rsi',
    'calculate_atr',
    'calculate_all_indicators'
]
