"""
Backtesting Module for TRANS Temporal Architecture
===================================================

Comprehensive backtesting system with full historical pattern detection.
Ensures all consolidation patterns are found across entire date range.
"""

from .temporal_backtester import TemporalBacktester, BacktestConfig, BacktestResults
from .performance_metrics import PerformanceMetrics, PatternStatistics

__all__ = [
    'TemporalBacktester',
    'BacktestConfig',
    'BacktestResults',
    'PerformanceMetrics',
    'PatternStatistics'
]
