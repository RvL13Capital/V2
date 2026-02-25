"""
Configuration Module for TRANS Temporal Architecture

Provides centralized configuration management for the entire system.
"""

from .constants import (
    # Outcome classes
    OutcomeClass,
    STRATEGIC_VALUES,
    OutcomeClass as OUTCOME_VALUES,  # Alias for backward compatibility

    # Signal thresholds
    SignalStrength,
    SIGNAL_THRESHOLDS,

    # Temporal model constants
    TEMPORAL_WINDOW_SIZE,
    QUALIFICATION_PHASE,
    VALIDATION_PHASE,
    MIN_PATTERN_DURATION,
    FEATURE_DIM,
    NUM_CLASSES,
    PatternPhase,

    # Feature flags
    ENABLE_ACCUMULATION_DETECTION,
    MIN_LIQUIDITY_DOLLAR,
    USE_GRN_CONTEXT,
    USE_PROBABILITY_CALIBRATION,
    DRIFT_CRITICAL_FEATURES,

    # Share dilution filter (US stocks only)
    ENABLE_SHARE_DILUTION_FILTER,
    MAX_DILUTION_PCT,
    DILUTION_LOOKBACK_MONTHS,
    SEC_EDGAR_RATE_LIMIT_RPM,
    SEC_EDGAR_CACHE_TTL_DAYS,
    SEC_EDGAR_USER_AGENT,
    EU_SUFFIXES,

    # Risk-based labeling constants
    RISK_MULTIPLIER_TARGET,
    RISK_MULTIPLIER_GREY,
    STOP_BUFFER_PERCENT,
    INDICATOR_WARMUP_DAYS,
    INDICATOR_STABLE_DAYS,
    OUTCOME_WINDOW_DAYS,
    HALT_THRESHOLD_DAYS,
    MAX_OVERNIGHT_GAP,
    MIN_VOLUME_VALIDITY_PCT,

    # Data validation
    MIN_DATA_LENGTH,
    MIN_TRAINING_SAMPLES,
    MAX_TEMPORAL_GAP_DAYS,
    MIN_VOLUME,
    MAX_PRICE_RATIO,

    # Model training
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    GAMMA_NEG,
    GAMMA_POS,
    GAMMA_PER_CLASS,
    CLASS_WEIGHTS,
    PREDICTION_TEMPERATURE,

    # Directories
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_SEQUENCE_DIR,
    DEFAULT_PREDICTION_DIR,

    # Metadata enrichment
    METADATA_ENRICHMENT_FIELDS,

    # Helper functions
    get_strategic_value,
    get_signal_strength,
    calculate_expected_value
)

from .temporal_features import (
    TemporalFeatureConfig,
    SequenceGenerationConfig,
    ModelArchitectureConfig
)

__all__ = [
    # Classes and Enums
    'OutcomeClass',
    'SignalStrength',
    'PatternPhase',
    'TemporalFeatureConfig',
    'SequenceGenerationConfig',
    'ModelArchitectureConfig',

    # Constants
    'STRATEGIC_VALUES',
    'SIGNAL_THRESHOLDS',
    'TEMPORAL_WINDOW_SIZE',
    'QUALIFICATION_PHASE',
    'VALIDATION_PHASE',
    'MIN_PATTERN_DURATION',
    'FEATURE_DIM',
    'NUM_CLASSES',
    'ENABLE_ACCUMULATION_DETECTION',
    'MIN_LIQUIDITY_DOLLAR',
    'USE_GRN_CONTEXT',
    'USE_PROBABILITY_CALIBRATION',
    'DRIFT_CRITICAL_FEATURES',

    # Share dilution filter
    'ENABLE_SHARE_DILUTION_FILTER',
    'MAX_DILUTION_PCT',
    'DILUTION_LOOKBACK_MONTHS',
    'SEC_EDGAR_RATE_LIMIT_RPM',
    'SEC_EDGAR_CACHE_TTL_DAYS',
    'SEC_EDGAR_USER_AGENT',
    'EU_SUFFIXES',

    # Risk-based labeling constants
    'RISK_MULTIPLIER_TARGET',
    'RISK_MULTIPLIER_GREY',
    'STOP_BUFFER_PERCENT',
    'INDICATOR_WARMUP_DAYS',
    'INDICATOR_STABLE_DAYS',
    'OUTCOME_WINDOW_DAYS',
    'HALT_THRESHOLD_DAYS',
    'MAX_OVERNIGHT_GAP',
    'MIN_VOLUME_VALIDITY_PCT',

    # Data validation
    'MIN_DATA_LENGTH',
    'MIN_TRAINING_SAMPLES',
    'MAX_TEMPORAL_GAP_DAYS',
    'MIN_VOLUME',
    'MAX_PRICE_RATIO',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_EPOCHS',
    'GAMMA_NEG',
    'GAMMA_POS',
    'GAMMA_PER_CLASS',
    'CLASS_WEIGHTS',
    'PREDICTION_TEMPERATURE',
    'DEFAULT_DATA_DIR',
    'DEFAULT_OUTPUT_DIR',
    'DEFAULT_LOG_DIR',
    'DEFAULT_MODEL_DIR',
    'DEFAULT_SEQUENCE_DIR',
    'DEFAULT_PREDICTION_DIR',

    # Metadata enrichment
    'METADATA_ENRICHMENT_FIELDS',

    # Functions
    'get_strategic_value',
    'get_signal_strength',
    'calculate_expected_value'
]