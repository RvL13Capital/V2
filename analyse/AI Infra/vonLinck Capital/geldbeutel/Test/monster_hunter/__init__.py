"""
Micro-Cap Monster Hunter System

A mean-reversion system for identifying coiled micro-caps with high potential energy.
Philosophy: Buy dead, sell alive.
"""

from .manipulation_screen import ManipulationScreen, is_toxic
from .feature_engine import (
    CoilDetector,
    prepare_features,
    calculate_adx,
    calculate_rvol,
    validate_coil_candidate
)
from .backtest_engine import BacktestEngine
from .model_brain import MonsterModel, PurgedTimeSeriesCV, compute_profit_factor

__version__ = "0.2.0"
__all__ = [
    # Gatekeeper
    "ManipulationScreen",
    "is_toxic",
    # Feature Engine
    "CoilDetector",
    "prepare_features",
    "calculate_adx",
    "calculate_rvol",
    "validate_coil_candidate",
    # Backtest
    "BacktestEngine",
    # ML
    "MonsterModel",
    "PurgedTimeSeriesCV",
    "compute_profit_factor",
]
