"""
Database Models for TRANS Production System
===========================================

SQLAlchemy models for pattern storage, predictions, and system tracking.
Supports both v17 (3-class) and legacy (K0-K5) labeling systems.
"""

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class LabelingVersion(enum.Enum):
    """Labeling system version"""
    LEGACY = "legacy"  # K0-K5 (6 classes)
    V17 = "v17"  # Path-dependent (3 classes)


class PatternStatus(enum.Enum):
    """Pattern lifecycle status"""
    DETECTED = "detected"
    LABELED = "labeled"
    PREDICTED = "predicted"
    VALIDATED = "validated"
    INVALID = "invalid"


class Pattern(Base):
    """
    Consolidation pattern record.

    Stores detected patterns with their features and outcomes.
    Supports both v17 and legacy labeling systems.
    """
    __tablename__ = 'patterns'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_id = Column(String(100), unique=True, nullable=False, index=True)

    # Ticker and timing
    ticker = Column(String(20), nullable=False, index=True)
    start_date = Column(DateTime, nullable=False, index=True)
    end_date = Column(DateTime, nullable=False, index=True)
    detection_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Pattern boundaries
    upper_boundary = Column(Float, nullable=False)
    lower_boundary = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float)
    risk_unit = Column(Float)  # R = entry - stop (v17 specific)

    # Pattern characteristics
    box_width = Column(Float)
    is_thin_stock = Column(Boolean, default=False)
    accumulation_count = Column(Integer, default=0)
    liquidity = Column(Float)  # Average dollar volume

    # Labeling
    labeling_version = Column(Enum(LabelingVersion), nullable=False)
    outcome_class = Column(Integer)  # 0-5 for legacy, 0-2 for v17
    strategic_value = Column(Float)

    # v17 specific fields
    is_grey_zone = Column(Boolean, default=False)  # v17 grey zone patterns

    # Legacy specific fields
    max_gain_pct = Column(Float)  # Maximum gain achieved (legacy)
    breakout_direction = Column(String(10))  # UP/DOWN (legacy)

    # Status tracking
    status = Column(Enum(PatternStatus), default=PatternStatus.DETECTED)

    # Metadata
    data_quality_score = Column(Float)  # 0-1 quality metric
    features = Column(JSON)  # Additional features as JSON
    notes = Column(Text)

    # Relationships
    predictions = relationship("Prediction", back_populates="pattern")

    # Indexes for common queries
    __table_args__ = (
        Index('idx_ticker_date', 'ticker', 'end_date'),
        Index('idx_status_version', 'status', 'labeling_version'),
        Index('idx_outcome', 'outcome_class', 'strategic_value'),
    )


class Prediction(Base):
    """
    Model predictions for patterns.

    Stores ML model outputs and expected values.
    """
    __tablename__ = 'predictions'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to pattern
    pattern_id = Column(String(100), ForeignKey('patterns.pattern_id'), nullable=False)

    # Model information
    model_version_id = Column(Integer, ForeignKey('model_versions.id'), nullable=False)
    prediction_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Predictions
    predicted_class = Column(Integer, nullable=False)  # 0-5 or 0-2
    class_probabilities = Column(JSON, nullable=False)  # {0: 0.2, 1: 0.5, 2: 0.3}
    expected_value = Column(Float, nullable=False)  # EV calculation
    confidence = Column(Float)  # Model confidence 0-1

    # Signal strength
    signal_strength = Column(String(20))  # STRONG/GOOD/MODERATE/WEAK/AVOID

    # Validation (filled in later)
    actual_outcome = Column(Integer)
    prediction_correct = Column(Boolean)
    value_captured = Column(Float)  # Actual strategic value realized

    # Relationships
    pattern = relationship("Pattern", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")

    # Indexes
    __table_args__ = (
        Index('idx_pattern_model', 'pattern_id', 'model_version_id'),
        Index('idx_prediction_date', 'prediction_date'),
        Index('idx_signal_strength', 'signal_strength'),
    )


class ModelVersion(Base):
    """
    Track model versions and performance.

    Manages different model versions for A/B testing and rollback.
    """
    __tablename__ = 'model_versions'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Version info
    version = Column(String(50), unique=True, nullable=False)
    labeling_version = Column(Enum(LabelingVersion), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Model details
    architecture = Column(String(100))  # 'hybrid_lstm_cnn_attention'
    num_classes = Column(Integer, nullable=False)  # 3 or 6
    parameters = Column(JSON)  # Model hyperparameters

    # Training info
    training_start_date = Column(DateTime)
    training_end_date = Column(DateTime)
    training_samples = Column(Integer)
    training_metrics = Column(JSON)  # Loss, accuracy, etc.

    # Validation metrics
    validation_metrics = Column(JSON)  # Precision, recall, F1, etc.
    expected_value_correlation = Column(Float)  # EV correlation

    # Status
    is_active = Column(Boolean, default=False)
    is_production = Column(Boolean, default=False)

    # File paths
    model_path = Column(String(500))  # Path to saved model file
    checkpoint_path = Column(String(500))  # Path to training checkpoint

    # Notes
    description = Column(Text)
    changelog = Column(Text)

    # Relationships
    predictions = relationship("Prediction", back_populates="model_version")

    # Indexes
    __table_args__ = (
        Index('idx_version_active', 'version', 'is_active'),
        Index('idx_created_at', 'created_at'),
        CheckConstraint('num_classes IN (3, 6)', name='check_num_classes'),
    )


class SystemLog(Base):
    """
    System-wide logging for audit and debugging.

    Tracks all significant events in the system.
    """
    __tablename__ = 'system_logs'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Log details
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    level = Column(String(20), nullable=False, index=True)  # DEBUG/INFO/WARNING/ERROR
    component = Column(String(100), nullable=False, index=True)  # scanner/labeler/predictor

    # Message
    message = Column(Text, nullable=False)

    # Context
    ticker = Column(String(20), index=True)
    pattern_id = Column(String(100), index=True)
    model_version_id = Column(Integer, ForeignKey('model_versions.id'))

    # Error tracking
    error_type = Column(String(200))
    error_trace = Column(Text)

    # Performance metrics
    execution_time_ms = Column(Integer)
    memory_usage_mb = Column(Float)

    # Request tracking
    request_id = Column(String(100), index=True)  # For tracing
    user_id = Column(String(100))

    # Indexes for common queries
    __table_args__ = (
        Index('idx_timestamp_level', 'timestamp', 'level'),
        Index('idx_component_ticker', 'component', 'ticker'),
    )


class MetricSnapshot(Base):
    """
    Time-series metrics for monitoring.

    Stores periodic snapshots of system performance.
    """
    __tablename__ = 'metric_snapshots'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    period = Column(String(20), nullable=False)  # HOURLY/DAILY/WEEKLY

    # Pattern metrics
    patterns_detected = Column(Integer, default=0)
    patterns_labeled = Column(Integer, default=0)
    patterns_predicted = Column(Integer, default=0)

    # Classification distribution (JSON for flexibility)
    outcome_distribution = Column(JSON)  # {0: 10, 1: 50, 2: 5}

    # Performance metrics
    avg_detection_time_ms = Column(Float)
    avg_labeling_time_ms = Column(Float)
    avg_prediction_time_ms = Column(Float)

    # Quality metrics
    avg_data_quality_score = Column(Float)
    patterns_grey_zone = Column(Integer, default=0)  # v17 specific
    patterns_invalid = Column(Integer, default=0)

    # Model performance
    model_version_id = Column(Integer, ForeignKey('model_versions.id'))
    prediction_accuracy = Column(Float)
    expected_value_mean = Column(Float)
    signal_distribution = Column(JSON)  # {STRONG: 5, GOOD: 10, ...}

    # System health
    cpu_usage_percent = Column(Float)
    memory_usage_mb = Column(Float)
    disk_usage_gb = Column(Float)
    api_response_time_ms = Column(Float)
    error_rate = Column(Float)  # Errors per 1000 operations

    # Indexes
    __table_args__ = (
        Index('idx_timestamp_period', 'timestamp', 'period'),
        UniqueConstraint('timestamp', 'period', name='unique_snapshot'),
    )


class TaskQueue(Base):
    """
    Async task queue for background processing.

    Tracks long-running tasks like scanning and training.
    """
    __tablename__ = 'task_queue'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(100), unique=True, nullable=False, index=True)

    # Task details
    task_type = Column(String(50), nullable=False, index=True)  # scan/label/predict/train
    priority = Column(Integer, default=5)  # 1=highest, 10=lowest

    # Payload
    parameters = Column(JSON, nullable=False)

    # Status
    status = Column(String(20), default='pending', index=True)  # pending/running/completed/failed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Results
    result = Column(JSON)
    error_message = Column(Text)

    # Execution tracking
    worker_id = Column(String(100))
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Indexes
    __table_args__ = (
        Index('idx_status_priority', 'status', 'priority'),
        Index('idx_task_type_status', 'task_type', 'status'),
    )


class MarketPhase(Base):
    """
    Market phase classification (bull/bear) based on index MA crossovers.

    Stores daily market regime for backtesting and analysis.
    Supports multiple index symbols (USA500, NASDAQ, etc.).
    """
    __tablename__ = 'market_phases'

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Index identification
    index_symbol = Column(String(50), nullable=False, index=True)  # USA500IDXUSD, etc.
    date = Column(DateTime, nullable=False, index=True)

    # Classification
    phase = Column(Integer, nullable=False)  # 1 = Bull, 0 = Bear

    # Moving averages
    fast_ma = Column(Float, nullable=False)  # 50-day MA
    slow_ma = Column(Float, nullable=False)  # 200-day MA
    fast_period = Column(Integer, nullable=False, default=50)
    slow_period = Column(Integer, nullable=False, default=200)

    # Transition tracking
    crossover_date = Column(DateTime)  # Last transition date
    days_in_phase = Column(Integer, default=0)

    # Configuration
    hysteresis_pct = Column(Float, default=0.05)  # Hysteresis used

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Unique constraint: One phase per index/date
    __table_args__ = (
        Index('idx_index_date', 'index_symbol', 'date'),
        UniqueConstraint('index_symbol', 'date', name='unique_index_date'),
        CheckConstraint('phase IN (0, 1)', name='check_phase_binary'),
    )


# Alias for API compatibility
TaskStatus = TaskQueue


# View for active patterns (convenience)
class ActivePatternView:
    """
    SQL View definition for active patterns needing predictions.

    To be created as:
    CREATE VIEW active_patterns AS
    SELECT p.*, pred.expected_value, pred.signal_strength
    FROM patterns p
    LEFT JOIN predictions pred ON p.pattern_id = pred.pattern_id
    WHERE p.status = 'labeled'
    AND p.outcome_class IS NOT NULL
    AND (pred.id IS NULL OR pred.model_version_id != (
        SELECT id FROM model_versions WHERE is_production = true LIMIT 1
    ))
    """
    pass