"""
AIv3 System Core Module
Centralized components for pattern detection and analysis
"""

from .config import SystemConfig, get_config
from .data_loader import UnifiedDataLoader, get_data_loader
from .pattern_detector import UnifiedPatternDetector, Pattern

__all__ = [
    'SystemConfig',
    'get_config',
    'UnifiedDataLoader',
    'get_data_loader',
    'UnifiedPatternDetector',
    'Pattern'
]

__version__ = '3.0.0'