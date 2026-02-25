"""
Features Module for TRANS Temporal Architecture

Provides vectorized feature calculation and extraction utilities.
"""

from .vectorized_calculator import VectorizedFeatureCalculator
from .ignition_detector import (
    detect_sequence_ignition,
    detect_ignition_at_timestep,
    get_ignition_statistics,
    IgnitionResult,
    IGNITION_THRESHOLD,
)
from .ignition_prioritizer import (
    score_patterns_for_ignition,
    prioritize_and_oversample,
    stratified_oversample,
    get_ignition_statistics as get_prioritization_statistics,
)

__all__ = [
    'VectorizedFeatureCalculator',
    # Ignition detection (sequence-level)
    'detect_sequence_ignition',
    'detect_ignition_at_timestep',
    'get_ignition_statistics',
    'IgnitionResult',
    'IGNITION_THRESHOLD',
    # Ignition prioritization (pattern-level)
    'score_patterns_for_ignition',
    'prioritize_and_oversample',
    'stratified_oversample',
    'get_prioritization_statistics',
]