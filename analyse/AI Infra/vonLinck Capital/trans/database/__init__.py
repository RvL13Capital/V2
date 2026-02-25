"""
Database Package for TRANS Production System
============================================

Provides database models, connection management, and utilities.
"""

from .models import (
    Base,
    Pattern,
    Prediction,
    ModelVersion,
    SystemLog,
    MetricSnapshot,
    TaskQueue,
    LabelingVersion,
    PatternStatus
)

from .connection import (
    DatabaseConfig,
    DatabaseManager,
    db_manager,
    get_db_session,
    init_database,
    close_database
)

__all__ = [
    # Models
    'Base',
    'Pattern',
    'Prediction',
    'ModelVersion',
    'SystemLog',
    'MetricSnapshot',
    'TaskQueue',
    'LabelingVersion',
    'PatternStatus',

    # Connection management
    'DatabaseConfig',
    'DatabaseManager',
    'db_manager',
    'get_db_session',
    'init_database',
    'close_database'
]