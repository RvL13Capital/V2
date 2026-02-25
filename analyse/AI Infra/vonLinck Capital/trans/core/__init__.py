"""
Core pattern detection components for TRANS architecture.
"""

from .pattern_detector import TemporalPatternDetector
from .pattern_scanner import (
    ConsolidationPatternScanner,
    ScanResult,
    UniverseScanResult,
    scan_for_patterns
)
from .exceptions import (
    SequenceGenerationError,
    TemporalConsistencyError,
    ValidationError,
    ConfigError,
    DataIntegrityError
)

__all__ = [
    'TemporalPatternDetector',
    'ConsolidationPatternScanner',
    'ScanResult',
    'UniverseScanResult',
    'scan_for_patterns',
    'SequenceGenerationError',
    'TemporalConsistencyError',
    'ValidationError',
    'ConfigError',
    'DataIntegrityError'
]
