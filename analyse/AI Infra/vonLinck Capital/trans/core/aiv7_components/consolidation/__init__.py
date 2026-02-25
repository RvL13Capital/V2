"""
Consolidation Pattern Detection Components
==========================================

Modular components for temporal consolidation pattern tracking.
Each component maintains strict temporal integrity - no future data leaks.

Components:
- PatternState: Immutable pattern state container
- TemporalFeatureExtractor: Lookback-only feature extraction
- SequentialStateManager: Phase transition state machine
- TemporalBoundaryManager: Boundary calculation and tracking
- TemporalStatisticsCalculator: Statistical calculations with temporal safety
- TemporalTrainingDataGenerator: ML sample generation after completion
"""

from .pattern_state import PatternState, PatternPhase
from .feature_extractor import TemporalFeatureExtractor
from .state_manager import SequentialStateManager
from .boundary_manager import TemporalBoundaryManager
from .statistics_calculator import TemporalStatisticsCalculator
from .training_generator import TemporalTrainingDataGenerator

__all__ = [
    'PatternState',
    'PatternPhase',
    'TemporalFeatureExtractor',
    'SequentialStateManager',
    'TemporalBoundaryManager',
    'TemporalStatisticsCalculator',
    'TemporalTrainingDataGenerator'
]