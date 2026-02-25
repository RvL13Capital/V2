"""Core pattern detection module."""

from .consolidation_tracker import (
    ConsolidationTracker,
    ConsolidationPattern,
    PatternPhase
)
from .stateful_detector import StatefulDetector

__all__ = [
    'ConsolidationTracker',
    'ConsolidationPattern',
    'PatternPhase',
    'StatefulDetector'
]
