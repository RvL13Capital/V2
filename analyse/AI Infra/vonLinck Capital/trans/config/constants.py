"""
Centralized Constants for TRANS Temporal Architecture

Single source of truth for all constants used across the system.
Eliminates duplication and ensures consistency.
"""

from enum import Enum
from typing import Dict, Final


# ================================================================================
# OUTCOME CLASSES AND STRATEGIC VALUES (V17 Path-Dependent)
# ================================================================================
# CRITICAL: These mappings are HARDCODED and must NEVER be changed without
# retraining all models. The integer IDs are baked into checkpoints.
# ================================================================================

class OutcomeClass(Enum):
    """Path-dependent outcome classifications with risk-based thresholds

    3-class system using Risk Multiples (R) instead of fixed percentages.
    Values calibrated for realistic R-based trading:
    - Danger: -2R (stop loss hit at -2R level)
    - Noise: -0.1R (opportunity cost / fees)
    - Target: +5R (conservative average win)
    """
    DANGER = (0, -2.0, "Stop Loss - Costs 2R")
    NOISE = (1, -0.1, "Base case - Opportunity cost / fees")
    TARGET = (2, 5.0, "Winner - Average win (conservative)")
    GREY_ZONE = (-1, None, "Ambiguous - Excluded from training")

    def __init__(self, class_id: int, strategic_value: float, description: str):
        self.class_id = class_id
        self.strategic_value = strategic_value
        self.description = description


# =============================================================================
# HARDCODED LABEL MAPPING (CRITICAL - DO NOT CHANGE)
# =============================================================================
# This mapping is the SINGLE SOURCE OF TRUTH for class IDs.
# Changing these values will BREAK all trained models.
#
# Why hardcoded: sklearn.LabelEncoder sorts labels, which could change
# mapping between training runs or library versions. We explicitly define
# the mapping to ensure consistency across all code paths.
# =============================================================================

# Name -> ID mapping (used for encoding)
LABEL_NAME_TO_ID: Final[Dict[str, int]] = {
    'Danger': 0,
    'Noise': 1,
    'Target': 2,
}

# ID -> Name mapping (used for decoding)
LABEL_ID_TO_NAME: Final[Dict[int, str]] = {
    0: 'Danger',
    1: 'Noise',
    2: 'Target',
}

# Checkpoint validation hash (changes if mapping changes)
# SHA256 of "Danger:0,Noise:1,Target:2" = ensures mapping consistency
LABEL_MAPPING_VERSION: Final[str] = "v1_DNT_012"


def validate_label_mapping(checkpoint_mapping_version: str = None) -> bool:
    """
    Validate that label mapping matches expected version.

    Call this at inference time to catch mapping mismatches before predictions.

    Args:
        checkpoint_mapping_version: Version string from checkpoint (optional)

    Returns:
        True if mapping is valid

    Raises:
        ValueError: If mapping version doesn't match
    """
    if checkpoint_mapping_version is not None:
        if checkpoint_mapping_version != LABEL_MAPPING_VERSION:
            raise ValueError(
                f"Label mapping mismatch! "
                f"Checkpoint has '{checkpoint_mapping_version}', "
                f"but code expects '{LABEL_MAPPING_VERSION}'. "
                f"This model was trained with a different label mapping and cannot be used."
            )

    # Validate internal consistency
    for name, id_ in LABEL_NAME_TO_ID.items():
        if LABEL_ID_TO_NAME.get(id_) != name:
            raise ValueError(
                f"Internal label mapping inconsistency: "
                f"LABEL_NAME_TO_ID['{name}']={id_} but "
                f"LABEL_ID_TO_NAME[{id_}]='{LABEL_ID_TO_NAME.get(id_)}'"
            )

    # Validate strategic values match
    for id_ in LABEL_ID_TO_NAME.keys():
        if id_ not in STRATEGIC_VALUES:
            raise ValueError(
                f"Missing strategic value for class {id_} ({LABEL_ID_TO_NAME[id_]})"
            )

    return True


def encode_label(label_name: str) -> int:
    """
    Encode a label name to its integer ID.

    Args:
        label_name: One of 'Danger', 'Noise', 'Target'

    Returns:
        Integer class ID (0, 1, or 2)

    Raises:
        ValueError: If label_name is not recognized
    """
    if label_name not in LABEL_NAME_TO_ID:
        raise ValueError(
            f"Unknown label '{label_name}'. "
            f"Valid labels: {list(LABEL_NAME_TO_ID.keys())}"
        )
    return LABEL_NAME_TO_ID[label_name]


def decode_label(label_id: int) -> str:
    """
    Decode an integer label ID to its name.

    Args:
        label_id: Integer class ID (0, 1, or 2)

    Returns:
        Label name ('Danger', 'Noise', or 'Target')

    Raises:
        ValueError: If label_id is not recognized
    """
    if label_id not in LABEL_ID_TO_NAME:
        raise ValueError(
            f"Unknown label ID {label_id}. "
            f"Valid IDs: {list(LABEL_ID_TO_NAME.keys())}"
        )
    return LABEL_ID_TO_NAME[label_id]


# Strategic values for EV calculation (default / bull market)
STRATEGIC_VALUES: Final[Dict[int, float]] = {
    0: -2.0,   # Danger: Costs 2R (Stop Loss at -2R level)
    1: -0.1,   # Noise: Opportunity Cost / Fees
    2: 5.0,    # Target: Average Win (conservative estimate)
    # Note: -1 (Grey zone) is excluded from training
}

# ================================================================================
# REGIME-AWARE STRATEGIC VALUES (Jan 2026 - Bear Mode Configuration)
# ================================================================================
# In bear markets, success probabilities are lower and losses more frequent.
# These adjusted values reflect more conservative expectations:
#   - Target value reduced (breakouts more likely to fail)
#   - Noise penalty increased (opportunity cost higher in drawdowns)
#   - Danger value unchanged (stop loss is stop loss)
#
# Market Phase Mapping:
#   0 = Bear (Close < SMA_200 < SMA_50)
#   1 = Bull (Close > SMA_50 > SMA_200)
#   2 = Sideways/Transition
# ================================================================================

STRATEGIC_VALUES_BULL: Final[Dict[int, float]] = {
    0: -2.0,   # Danger: Same (stop is stop)
    1: -0.1,   # Noise: Low opportunity cost in bull
    2: 5.0,    # Target: Strong breakouts succeed
}

STRATEGIC_VALUES_BEAR: Final[Dict[int, float]] = {
    0: -2.0,   # Danger: Same (stop is stop)
    1: -0.2,   # Noise: Higher opportunity cost (could be losing elsewhere)
    2: 3.0,    # Target: Reduced (breakouts fail more often)
}

STRATEGIC_VALUES_SIDEWAYS: Final[Dict[int, float]] = {
    0: -2.0,   # Danger: Same
    1: -0.15,  # Noise: Moderate opportunity cost
    2: 4.0,    # Target: Between bull and bear
}


def get_strategic_values(market_phase: int = 1) -> Dict[int, float]:
    """
    Return strategic values based on market regime.

    Args:
        market_phase: Market regime (0=Bear, 1=Bull, 2=Sideways)

    Returns:
        Dictionary mapping outcome class to strategic value

    Example:
        >>> values = get_strategic_values(0)  # Bear market
        >>> print(values)
        {0: -2.0, 1: -0.2, 2: 3.0}
    """
    if market_phase == 0:  # Bear
        return STRATEGIC_VALUES_BEAR.copy()
    elif market_phase == 2:  # Sideways
        return STRATEGIC_VALUES_SIDEWAYS.copy()
    else:  # Bull (default)
        return STRATEGIC_VALUES_BULL.copy()


# ================================================================================
# SIGNAL THRESHOLDS
# ================================================================================

class SignalStrength(Enum):
    """Signal strength thresholds based on Expected Value"""
    STRONG = 5.0      # High confidence signal
    GOOD = 3.0        # Good confidence signal
    MODERATE = 1.0    # Moderate confidence signal
    WEAK = 0.0        # Weak/no signal
    AVOID = -1.0      # Negative EV, avoid


# Signal thresholds dictionary
SIGNAL_THRESHOLDS: Final[Dict[str, float]] = {
    'STRONG': 5.0,
    'GOOD': 3.0,
    'MODERATE': 1.0,
    'WEAK': 0.0,
    'AVOID': -1.0
}

# Trading filters
MIN_PATTERN_WIDTH: Final[float] = 0.025    # 2.5% minimum width (patterns below this are untradeable due to spread)
PREDICTION_TEMPERATURE: Final[float] = 1.5  # Temperature scaling for calibrated probabilities


# ================================================================================
# TEMPORAL MODEL CONSTANTS
# ================================================================================

# Sequence generation parameters
TEMPORAL_WINDOW_SIZE: Final[int] = 20          # Timesteps per sequence
QUALIFICATION_PHASE: Final[int] = 10           # Timesteps 1-10
VALIDATION_PHASE: Final[int] = 10              # Timesteps 11-20
MIN_PATTERN_DURATION: Final[int] = 20          # Minimum for valid pattern
FEATURE_DIM: Final[int] = 10                   # Features per timestep (composite features disabled)
NUM_CLASSES: Final[int] = 3                    # V17: Danger, Noise, Target

# Class names for reporting and evaluation (V17 3-class system)
CLASS_NAMES: Final[Dict[int, str]] = {
    0: 'Danger',
    1: 'Noise',
    2: 'Target'
}

# Class names as list (for sklearn target_names parameter)
CLASS_NAME_LIST: Final[list] = ['Danger', 'Noise', 'Target']

# Pattern phases
class PatternPhase(Enum):
    """Pattern lifecycle phases"""
    NONE = "none"
    QUALIFYING = "qualifying"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


# ================================================================================
# PROCESSING MODES (Training vs Inference)
# ================================================================================

class ProcessingMode(Enum):
    """
    Processing mode for pattern detection and labeling.

    TRAINING: Strict filtering - only patterns with definitive outcomes
              Excludes: unripe patterns, grey zones (-1)
              Use for: Model training, backtesting

    INFERENCE: Permissive - includes patterns without outcomes
               Assigns placeholder (-2) for unripe/insufficient data
               Use for: Live predictions, API endpoints
    """
    TRAINING = "training"
    INFERENCE = "inference"


# Label placeholders for inference mode
LABEL_PLACEHOLDER_UNRIPE: Final[int] = -2    # Pattern not yet ripe (insufficient outcome data)
LABEL_GREY_ZONE: Final[int] = -1              # DEPRECATED (Jan 2026): Grey zone now converted to Class 1 (Noise)
                                              # Was: Ambiguous pattern (v17: 2.5R but not 3R)
                                              # Now: These patterns are labeled as Noise to add training data

# Valid labels for training (exclude -1, -2)
VALID_TRAINING_LABELS: Final[list] = [0, 1, 2]  # Danger, Noise, Target


# ================================================================================
# FEATURE FLAGS
# ================================================================================

# Pattern detection features
ENABLE_ACCUMULATION_DETECTION: Final[bool] = True  # Enable/disable accumulation detection in scanner

# Scanner parameters
MIN_LIQUIDITY_DOLLAR: Final[float] = 50000  # Minimum daily dollar volume

# Model architecture flags
USE_GRN_CONTEXT: Final[bool] = True   # ENABLED: Context Branch with GRN gating
                                      # When True: Branch B (GRN Context Gating) modulates temporal signal
                                      # Features: 8 context features including relative_strength_spy

# Probability calibration (Isotonic Regression post-hoc calibration)
USE_PROBABILITY_CALIBRATION: Final[bool] = True  # ENABLED: Calibrate focal loss probabilities
                                                  # Focal loss distorts P(class) - Isotonic Regression
                                                  # maps predicted probs to true class frequencies

# Drift monitoring critical features
DRIFT_CRITICAL_FEATURES: Final[list] = [
    'vol_dryup_ratio',  # PRIMARY - micro-cap cycle indicator (Sleeper regime)
    'bbw',              # Volatility contraction (Sleeper regime)
]


# ================================================================================
# SHARE DILUTION FILTER (US STOCKS ONLY)
# ================================================================================

# Share dilution thresholds
ENABLE_SHARE_DILUTION_FILTER: Final[bool] = True      # Enable/disable US dilution filter
MAX_DILUTION_PCT: Final[float] = 20.0                 # Reject US stocks with >20% dilution in trailing 12 months
DILUTION_LOOKBACK_MONTHS: Final[int] = 12             # Trailing period for dilution calculation

# SEC EDGAR API settings (free, no API key required)
SEC_EDGAR_RATE_LIMIT_RPM: Final[int] = 300            # 5 req/sec (conservative for SEC)
SEC_EDGAR_CACHE_TTL_DAYS: Final[int] = 7              # Cache dilution data for 7 days
SEC_EDGAR_USER_AGENT: Final[str] = "TRANS-System/1.0 (pattern-detection@vonlinck.com)"  # Required by SEC

# EU market suffixes (dilution filter does NOT apply to these)
EU_SUFFIXES: Final[tuple] = (
    '.DE', '.L', '.PA', '.MI', '.MC', '.AS', '.LS',
    '.BR', '.SW', '.ST', '.OL', '.CO', '.HE', '.IR', '.VI'
)


# ================================================================================
# RISK-BASED LABELING CONSTANTS
# ================================================================================

# Risk-based thresholds
RISK_MULTIPLIER_TARGET: Final[float] = 5.0   # 5R for target confirmation (UPDATED: filters "noise winners", forces AI to learn stocks that double)
RISK_MULTIPLIER_GREY: Final[float] = 2.5     # 2.5R for grey zone threshold
STOP_BUFFER_PERCENT: Final[float] = 0.08     # 8% buffer below boundary for stop (UPDATED: micro-caps stop-hunt aggressively, wider buffer prevents false "Danger" labels)

# Data requirements for proper indicator warm-up
INDICATOR_WARMUP_DAYS: Final[int] = 30       # Days for indicators to stabilize
INDICATOR_STABLE_DAYS: Final[int] = 100      # Days of stable indicator data
OUTCOME_WINDOW_DAYS: Final[int] = 100        # Days to track after pattern

# Data integrity thresholds
HALT_THRESHOLD_DAYS: Final[int] = 10         # Max gap between trading days (weekends + holidays OK)
MAX_OVERNIGHT_GAP: Final[float] = 0.30       # 30% max overnight gap (vs data error)
MIN_VOLUME_VALIDITY_PCT: Final[float] = 0.80 # 80% of days must have valid volume


# ================================================================================
# ATR-BASED LABELING CONSTANTS (Jan 2026)
# ================================================================================
# Alternative to R-multiple labeling that adapts to market regime volatility.
# ATR (Average True Range) captures recent market conditions better than
# pattern-specific boundaries.

USE_ATR_LABELING: Final[bool] = False        # Feature flag (default: off, use R-multiples)
ATR_PERIOD: Final[int] = 14                  # Standard ATR period (14 days)
ATR_STOP_MULTIPLE: Final[float] = 2.0        # Stop = lower_boundary - (2.0 × ATR_14)
ATR_TARGET_MULTIPLE: Final[float] = 5.0      # Target = entry + (5.0 × ATR_14)
ATR_GREY_MULTIPLE: Final[float] = 2.5        # Grey zone = entry + (2.5 × ATR_14)

# Key differences from R-multiple:
# - R-multiple: Stop/target adapt to pattern boundary width (pattern-specific)
# - ATR: Stop/target adapt to recent volatility (market regime-specific)
# - ATR better handles regime changes (high vol = wider stops, low vol = tighter)


# ================================================================================
# TRIPLE BARRIER LABELING CONSTANTS (Jan 2026 - Marcos Lopez de Prado Method)
# ================================================================================
# Implementation of the Triple Barrier Method from "Advances in Financial ML"
# Uses relative returns (percentage change) instead of absolute prices.
#
# Three Barriers:
#   1. Upper (Profit-Taking): +threshold % return achieved
#   2. Lower (Stop-Loss): -threshold % return hit
#   3. Vertical (Time): Maximum holding period expires
#
# Labels determined by which barrier is touched FIRST:
#   - 2 (Target): Upper barrier hit first
#   - 0 (Danger): Lower barrier hit first
#   - 1 (Noise): Vertical barrier (timeout)
#
# IMPORTANT: All rolling windows use closed='left' for strict temporal integrity
# ================================================================================

USE_TRIPLE_BARRIER_LABELING: Final[bool] = False  # Feature flag (enable to use Triple Barrier)

# Window configuration
TRIPLE_BARRIER_FEATURE_WINDOW: Final[int] = 250   # Days of history for feature calculation
TRIPLE_BARRIER_VERTICAL_DAYS: Final[int] = 150    # Maximum holding period (vertical barrier)
TRIPLE_BARRIER_MIN_DATA_DAYS: Final[int] = 300    # Minimum data required (feature + outcome)

# Barrier thresholds (as decimals - e.g., 0.03 = 3%)
TRIPLE_BARRIER_UPPER_PCT: Final[float] = 0.03     # Profit-taking threshold (+3%)
TRIPLE_BARRIER_LOWER_PCT: Final[float] = 0.02     # Stop-loss threshold (-2%)

# Volatility scaling configuration
TRIPLE_BARRIER_VOL_SCALING: Final[bool] = True    # Scale barriers by realized volatility
TRIPLE_BARRIER_VOL_MULTIPLIER: Final[float] = 2.0 # Multiplier for vol-scaled barriers
TRIPLE_BARRIER_VOL_LOOKBACK: Final[int] = 20      # Days for volatility estimation

# Temporal integrity requirements
# CRITICAL: All features must use rolling(window, closed='left')
# This ensures no information from the current day leaks into calculations
TRIPLE_BARRIER_STRICT_TEMPORAL: Final[bool] = True


# ================================================================================
# DATA VALIDATION CONSTANTS
# ================================================================================

# Minimum data requirements
MIN_DATA_LENGTH: Final[int] = 50               # Minimum rows for valid data
MIN_TRAINING_SAMPLES: Final[int] = 100         # Minimum samples for training

# Temporal integrity
MAX_TEMPORAL_GAP_DAYS: Final[int] = 10         # Max gap (weekends + holidays OK)
MAX_DUPLICATE_TIMESTAMPS: Final[int] = 0       # No duplicates allowed

# OHLCV validation thresholds
MIN_VOLUME: Final[float] = 0.0                 # Volume must be non-negative
MAX_PRICE_RATIO: Final[float] = 10.0           # Max high/low ratio (sanity check)


# ================================================================================
# PERFORMANCE OPTIMIZATION CONSTANTS
# ================================================================================

# Caching parameters
CACHE_SIZE: Final[int] = 128                   # LRU cache size
CACHE_TTL_SECONDS: Final[int] = 3600          # Cache time-to-live

# Batch processing
DEFAULT_BATCH_SIZE: Final[int] = 64            # Default batch size for training
MAX_BATCH_SIZE: Final[int] = 512               # Maximum batch size

# Performance thresholds
MAX_PROCESSING_TIME_SECONDS: Final[int] = 300  # Max time for single ticker


# ================================================================================
# MODEL TRAINING CONSTANTS
# ================================================================================

# Training hyperparameters
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_LEARNING_RATE: Final[float] = 0.001
DEFAULT_EPOCHS: Final[int] = 100
DEFAULT_EARLY_STOP_PATIENCE: Final[int] = 10

# Asymmetric loss parameters - PRECISION FOCUS (Hard Negative Mining)
# High gamma = focus on hard examples (those the model gets wrong)
# Low gamma = downweight easy examples
GAMMA_NEG: Final[float] = 4.0                  # High gamma for negatives (focus on hard negatives)
GAMMA_POS: Final[float] = 0.5                  # Low gamma for positives (don't hunt for easy targets)

# Per-class gamma (overrides gamma_neg/gamma_pos when specified)
# K0 (Danger): High gamma = model must be VERY sure before predicting Target
# K2 (Target): Low gamma = accept missing some Targets for precision
GAMMA_PER_CLASS: Final[Dict[int, float]] = {
    0: 4.0,    # Danger - focus on hard negatives (prevent False Positives)
    1: 2.0,    # Noise - moderate focus
    2: 0.5,    # Target - low focus (don't hunt for easy targets)
}

# Class weights for imbalance (3 classes)
# PRECISION FOCUS (Jan 2026): Inverted weights for Hard Negative Mining
# Philosophy: Penalize False Positives > Missing Targets
# - High K0 weight (5.0): Heavy penalty for predicting Target when actual is Danger
# - Low K2 weight (1.0): Accept missing some Targets to avoid False Positives
CLASS_WEIGHTS: Final[Dict[int, float]] = {
    0: 5.0,    # Danger - HEAVY penalty for False Positives
    1: 1.0,    # Noise - base case
    2: 1.0,    # Target - low weight, prioritize precision over recall
}

# ================================================================================
# TEMPORAL SPLIT CONFIGURATION
# ================================================================================

# Temporal split settings to prevent look-ahead bias
USE_TEMPORAL_SPLIT: Final[bool] = True                    # Always use temporal split for production
TEMPORAL_SPLIT_RATIO: Final[float] = 0.8                  # 80% train, 20% test (by time)
TEMPORAL_SPLIT_STRATEGY: Final[str] = "percentage"        # Options: "percentage", "date_cutoff", "walk_forward"
TEMPORAL_VAL_SIZE: Final[float] = 0.1                     # 10% validation from training period
MIN_TRAIN_SAMPLES: Final[int] = 100                       # Minimum samples required for training
MIN_TEST_SAMPLES: Final[int] = 20                         # Minimum samples required for testing

# Temporal integrity validation
ENFORCE_TEMPORAL_INTEGRITY: Final[bool] = True            # Validate no look-ahead bias
MAX_TEMPORAL_GAP_DAYS: Final[int] = 10                   # Maximum allowed gap in days (accounts for weekends/holidays)
TEMPORAL_DATE_COLUMN: Final[str] = "sequence_end_date"    # Column to use for temporal splitting


# ================================================================================
# FILE PATHS AND DIRECTORIES
# ================================================================================

# Default directories
DEFAULT_DATA_DIR: Final[str] = "data/raw"
DEFAULT_OUTPUT_DIR: Final[str] = "output"
DEFAULT_LOG_DIR: Final[str] = "logs"
DEFAULT_MODEL_DIR: Final[str] = "output/models"
DEFAULT_SEQUENCE_DIR: Final[str] = "output/sequences"
DEFAULT_PREDICTION_DIR: Final[str] = "output/predictions"


# ================================================================================
# LOGGING CONFIGURATION
# ================================================================================

# Log levels
LOG_LEVEL_DEFAULT: Final[str] = "INFO"
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


# ================================================================================
# METADATA ENRICHMENT CONFIGURATION
# ================================================================================

# Fields to copy from pattern_row into metadata.parquet for downstream analysis
# These are computed by calculate_coil_features() and prioritize_and_oversample()
METADATA_ENRICHMENT_FIELDS: Final[list] = [
    # Coil features (from calculate_coil_features)
    'price_position_at_end',   # Position in box [0=lower, 1=upper]
    'distance_to_danger',      # Distance to danger zone [0-1]
    'bbw_slope_5d',            # BBW slope over 5 days
    'vol_trend_5d',            # Volume trend ratio
    'coil_intensity',          # Combined coil quality [0-1]
    # Ignition features (from prioritize_and_oversample)
    'ignition_score',          # Ignition potential [0-1]
    'recency_score',           # Pattern recency [0.5-1]
    'priority_score',          # Composite priority [0-1]
    'oversample_copy',         # Oversample copy number (0 = original)
    # Pattern metrics
    'box_width',               # Pattern box width (%)
    'pattern_days',            # Duration in days
    'upper_boundary',          # Upper boundary price
    'lower_boundary',          # Lower boundary price
]


# ================================================================================
# WEEKLY QUALIFICATION MODE (Jan 2026)
# ================================================================================
# Weekly candle qualification for longer-term consolidation patterns.
# Uses weekly (instead of daily) candles for the 10-period qualification window.
#
# Benefits:
# - Filters for stronger accumulation (10 weeks = ~2.5 months vs 10 days)
# - Reduces pattern count by 65-85% (higher quality signals)
# - Wider boundaries (from weekly H/L) improve risk/reward
#
# Trade-offs:
# - Requires ~4 years of data for SMA_200 calculation
# - Fewer patterns detected (may miss shorter-term opportunities)
# ================================================================================

# Qualification period in weeks (equivalent to 10 days in daily mode)
WEEKLY_QUALIFICATION_PERIODS: Final[int] = 10          # 10 weeks = ~2.5 months

# Lookback for percentile calculations (100 weeks = ~2 years)
WEEKLY_LOOKBACK_PERIODS: Final[int] = 100              # 100 weeks for BBW percentile

# Minimum history required for weekly mode (~1.5 years for SMA_50 + windows)
WEEKLY_MIN_HISTORY_WEEKS: Final[int] = 80              # 50 weeks (SMA) + 30 weeks (window)

# Pandas resample rule (weeks ending Friday for trading week alignment)
WEEKLY_RESAMPLE_RULE: Final[str] = 'W-FRI'             # Week ends Friday

# ================================================================================
# WEEKLY LABELING PARAMETERS (Jan 2026)
# ================================================================================
# Adjusted parameters for weekly patterns based on empirical analysis:
# - Weekly boundaries are naturally wider (~17% vs ~5% for daily)
# - Using full box as stop (1R = 17%) makes +3R target (51%) unreachable
# - Solution: Tighter stop, lower target, appropriate window
# ================================================================================

# Boundary calculation: Use last N weeks of qualifying period (not full 10 weeks)
# Rationale: Recent price action is more relevant for breakout trading
WEEKLY_BOUNDARY_WEEKS: Final[int] = 3                  # Last 3 weeks for H/L boundaries

# Stop distance as fraction of box width
# Full box (1.0) = 17% stop → +3R requires 51% gain (unreachable)
# 75% box (0.75) = 12.8% stop → +2R requires 25.6% gain (achievable)
# Half box (0.5) = 8.5% stop → +2R requires 17% gain (too tight, 92% stopped out)
WEEKLY_STOP_BOX_FRACTION: Final[float] = 0.75          # 75% of box width

# Target R-multiple for weekly patterns
# Daily uses +3R, but weekly with tighter stop can use +2R
WEEKLY_TARGET_R_MULTIPLE: Final[float] = 2.0           # +2R target (vs +3R for daily)

# Fixed outcome window for weekly patterns (days)
# Weekly patterns need more time to resolve than daily
# 100-day analysis shows +2R hit rate of 40.5% (> 33% breakeven)
WEEKLY_OUTCOME_WINDOW_DAYS: Final[int] = 100           # 100 trading days = 20 weeks


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def get_strategic_value(outcome_class: int) -> float:
    """Get strategic value for an outcome class

    Args:
        outcome_class: Class ID (0=Danger, 1=Noise, 2=Target, -1=Grey Zone)

    Returns:
        Strategic value for the class

    Raises:
        ValueError: If outcome_class is invalid
    """
    if outcome_class == -1:
        raise ValueError("Grey zone patterns (-1) should be excluded from training")
    if outcome_class not in STRATEGIC_VALUES:
        raise ValueError(f"Invalid outcome class: {outcome_class}")
    return STRATEGIC_VALUES[outcome_class]


def get_signal_strength(expected_value: float) -> str:
    """Determine signal strength from expected value

    Args:
        expected_value: Calculated EV

    Returns:
        Signal strength label
    """
    if expected_value >= SIGNAL_THRESHOLDS['STRONG']:
        return 'STRONG'
    elif expected_value >= SIGNAL_THRESHOLDS['GOOD']:
        return 'GOOD'
    elif expected_value >= SIGNAL_THRESHOLDS['MODERATE']:
        return 'MODERATE'
    elif expected_value >= SIGNAL_THRESHOLDS['WEAK']:
        return 'WEAK'
    else:
        return 'AVOID'


def calculate_expected_value(
    class_probabilities: Dict[int, float]
) -> float:
    """Calculate expected value from class probabilities

    Args:
        class_probabilities: Dictionary of {class_id: probability}
                           Should only contain classes 0, 1, 2 (not -1)

    Returns:
        Expected value

    Example:
        >>> probs = {0: 0.1, 1: 0.6, 2: 0.3}  # Danger, Noise, Target
        >>> ev = calculate_expected_value(probs)
        >>> print(f"EV: {ev:.2f}")
    """
    ev = 0.0
    for class_id, prob in class_probabilities.items():
        if class_id == -1:
            raise ValueError("Grey zone (-1) should not be in probability distribution")
        strategic_value = get_strategic_value(class_id)
        ev += prob * strategic_value
    return ev