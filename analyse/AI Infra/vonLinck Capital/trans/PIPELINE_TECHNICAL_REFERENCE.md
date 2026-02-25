# TRAnS Pipeline Technical Reference

> **Version:** Verified from codebase January 2026
> **Status:** Production EU | EU Model: 21.1% Top-15 (1.85x lift)

---

## Pipeline Overview

```
OHLCV Data
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 0: DETECT PATTERNS                                      │
│ Script: pipeline/00_detect_patterns.py                       │
│ Output: candidate_patterns.parquet (NO labels)               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼  ⏳ 40-day ripening period
    │
┌─────────────────────────────────────────────────────────────┐
│ Step 0b: LABEL OUTCOMES                                      │
│ Script: pipeline/00b_label_outcomes.py                       │
│ Output: labeled_patterns.parquet (outcome_class 0/1/2)       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: GENERATE SEQUENCES                                   │
│ Script: pipeline/01_generate_sequences.py                    │
│ Output: sequences.h5 + metadata.parquet                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: TRAIN MODEL                                          │
│ Script: pipeline/02_train_temporal.py                        │
│ Output: best_model.pt + norm_params.json                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: PREDICT                                              │
│ Script: pipeline/03_predict_temporal.py                      │
│ Output: nightly_orders.csv                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 0: Detect Patterns

**Script:** `pipeline/00_detect_patterns.py`

### Purpose
Scan tickers for consolidation patterns using `ConsolidationPatternScanner` state machine.

### Detection Thresholds
```python
# Source: pipeline/00_detect_patterns.py lines 138-161
# Legacy (percentile-based)
--bbw-threshold 0.30      # BBW percentile threshold
--adx-threshold 32.0      # ADX threshold
--volume-threshold 0.35   # Volume ratio threshold
--range-threshold 0.65    # Range ratio threshold

# Adaptive (Jan 2026) - RECOMMENDED
--tightness-zscore -1.0   # Z-score based: 1 std dev tighter than stock's own avg
--min-float-turnover 0.10 # Require 10% of float traded in 20d (accumulation filter)
```

### Adaptive vs Legacy Thresholds

| Method | Parameter | Description |
|--------|-----------|-------------|
| Legacy | `--bbw-threshold 0.30` | Static: BBW < 30th percentile of last 100 days |
| Adaptive | `--tightness-zscore -1.0` | Dynamic: Body width 1 std dev below stock's own mean |

**Why Adaptive is Better:**
- Large-caps with naturally tight BBW may never hit 30th percentile
- Micro-caps with high BBW hit 30th too easily
- Z-score measures relative tightness *for this specific stock*

```python
# Z-Score calculation (sleeper_scanner_v17.py)
body_width_zscore = (current_body_width - historical_mean) / historical_std
# Reject if: body_width_zscore > tightness_zscore (e.g., > -1.0)
```

### Ticker Loading
```python
# Source: pipeline/00_detect_patterns.py lines 192-255
# Supports:
# - File path: data/eu_tickers.txt
# - Comma-separated: AAPL,MSFT,GOOGL
# - GCS bucket: "ALL"
```

### Output Schema
- `candidate_patterns.parquet`
  - `ticker`, `pattern_id`, `start_date`, `end_date`
  - `upper_boundary`, `lower_boundary`
  - Technical features at detection time
  - **NO `outcome_class`** (temporal integrity)

### Command
```bash
python pipeline/00_detect_patterns.py \
    --tickers data/eu_tickers.txt \
    --start-date 2020-01-01 \
    --workers 8
```

---

## Step 0b: Label Outcomes

**Script:** `pipeline/00b_label_outcomes.py`

### Purpose
Apply Structural R-Labeling after 40-day ripening period.

### Labeling Constants
```python
# Source: pipeline/00b_label_outcomes.py lines 60-74
OUTCOME_WINDOW_DAYS = 40
TARGET_R_MULTIPLE = 3.0      # +3R for Target classification
GAP_LIMIT_R = 0.5            # Max gap before untradeable flag
VOLUME_MULTIPLIER_TARGET = 2.0
ZOMBIE_TIMEOUT = 150         # Days before pattern → Noise
TRIGGER_OFFSET = 0.01        # $0.01 above upper boundary
```

### R-Metrics Calculation
```python
# Source: pipeline/00b_label_outcomes.py lines 185-222
def calculate_r_metrics(upper_boundary, lower_boundary):
    technical_floor = lower_boundary
    trigger_price = upper_boundary + TRIGGER_OFFSET  # +$0.01
    r_dollar = trigger_price - technical_floor
    gap_limit = trigger_price + (GAP_LIMIT_R * r_dollar)

    return {
        'technical_floor': technical_floor,
        'trigger_price': trigger_price,
        'r_dollar': r_dollar,
        'gap_limit': gap_limit
    }
```

### Entry Simulation
```python
# Source: pipeline/00b_label_outcomes.py lines 318-339
# Day 1 (t+1) open is potential entry
open_t1 = outcome_data.iloc[0]['open']

# Entry price = MAX(Trigger_Price, Open_{t+1})
entry_price = max(trigger_price, open_t1)

# Gap check - if too large, mark untradeable but continue labeling
gap_too_large = entry_price > gap_limit

# Target price based on actual entry
target_price = entry_price + (TARGET_R_MULTIPLE * r_dollar)
```

### Outcome Classes
```python
# Source: config/constants.py lines 19-31
class OutcomeClass(Enum):
    DANGER = (0, -2.0, "Stop Loss - Costs 2R")
    NOISE = (1, -0.1, "Base case - Opportunity cost / fees")
    TARGET = (2, 5.0, "Winner - Average win (conservative)")
    GREY_ZONE = (-1, None, "Ambiguous - Excluded from training")
```

### Label Mapping
```python
# Source: config/constants.py lines 51-62
# CRITICAL - DO NOT CHANGE (breaks trained models)
LABEL_NAME_TO_ID = {
    'Danger': 0,
    'Noise': 1,
    'Target': 2,
}
LABEL_MAPPING_VERSION = "v1_DNT_012"
```

### Command
```bash
python pipeline/00b_label_outcomes.py \
    --input output/candidate_patterns.parquet \
    --enable-early-labeling
```

---

## Step 1: Generate Sequences

**Script:** `pipeline/01_generate_sequences.py`

### Purpose
Convert labeled patterns to tensor format for neural network training.

### Temporal Features (10 per timestep)
```python
# Source: config/temporal_features.py lines 32-72
# Shape: (N, 20, 10)

# Market data features [0-4]
market_features = ['open', 'high', 'low', 'close', 'volume']

# Technical indicators [5-7]
technical_features = ['bbw_20', 'adx', 'volume_ratio_20']

# Boundary slopes [8-9]
# Note: Changed from static box to dynamic slopes (2026-01-18)
boundary_features = ['upper_boundary', 'lower_boundary']
```

### Context Features (18 static)
```python
# Source: config/context_features.py lines 22-47
# Shape: (N, 18)

CONTEXT_FEATURES = [
    'retention_rate',              # 0: (Close-Low)/(High-Low) candle strength
    'trend_position',              # 1: Current_Price / 200_SMA
    'base_duration',               # 2: Days_Since_20Pct_High / 200 (log-normalized)
    'relative_volume',             # 3: log_diff(vol_20d, vol_60d)
    'distance_to_high',            # 4: (52W_High - Price) / 52W_High
    'log_float',                   # 5: log10(shares_outstanding)
    'log_dollar_volume',           # 6: log10(avg_daily_dollar_volume)
    'dormancy_shock',              # 7: log_diff(vol_20d, vol_252d)
    'vol_dryup_ratio',             # 8: log_diff(vol_20d, vol_100d)
    'price_position_at_end',       # 9: Close position in box [0,1]
    'volume_shock',                # 10: log_diff(max_vol_3d, avg_vol_20d)
    'bbw_slope_5d',                # 11: BBW change over last 5 days
    'vol_trend_5d',                # 12: log_diff(avg_vol_5d, avg_vol_20d)
    'coil_intensity',              # 13: Combined coil quality score
    'relative_strength_cohort',    # 14: Ticker 20d return vs universe median
    'risk_width_pct',              # 15: (Upper - Lower) / Upper
    'vol_contraction_intensity',   # 16: log_diff(avg_vol_5d, avg_vol_60d)
    'obv_divergence',              # 17: OBV slope vs Price slope
]

NUM_CONTEXT_FEATURES = 18
```

### Log-Diff Transformation
```python
# Source: config/context_features.py lines 73-97
# Fixes NaN/Inf on dormant stocks where denominator ≈ 0

def log_diff(numerator: float, denominator: float) -> float:
    """
    Safe volume ratio calculation using log-difference.
    Replaces raw ratio (num/denom) which explodes on dormant stocks.

    Returns:
        log1p(numerator) - log1p(denominator)
        Positive = volume increasing
        Negative = volume decreasing
    """
    return np.log1p(max(numerator, 0)) - np.log1p(max(denominator, 0))

# Features using log-diff (lines 117-130)
VOLUME_RATIO_FEATURES = [
    'relative_volume',           # Index 3
    'dormancy_shock',            # Index 7
    'vol_dryup_ratio',           # Index 8
    'volume_shock',              # Index 10
    'vol_trend_5d',              # Index 12
    'vol_contraction_intensity', # Index 16
    'obv_divergence',            # Index 17
]
```

### Output
- `sequences.h5` - HDF5 with datasets:
  - `sequences`: Shape `(N, 20, 10)`
  - `labels`: Shape `(N,)`
  - `context`: Shape `(N, 18)`
- `metadata.parquet` - Pattern tracking info

### Command
```bash
python pipeline/01_generate_sequences.py \
    --input output/labeled_patterns.parquet \
    --apply-nms \
    --apply-physics-filter \
    --mode training
```

---

## Step 2: Train Model

**Script:** `pipeline/02_train_temporal.py`

### Purpose
Train LSTM+CNN+Attention neural network.

### Model Architecture (V18)
```python
# Source: models/temporal_hybrid_v18.py lines 1-21
"""
Multi-Modal Hybrid Temporal Model V18: Context-Query Attention (CQA)

Architecture:
  Input -> [LSTM] -------------------------> [Narrative State]
  Input -> [CNN] -> [Self-Attention] ------> [Geometric Structure]
                 -> [Cross-Attention] -----> [Context Relevance] <--- Context (Query)
  Context -> [GRN] ------------------------> [Base Probability]

Output: Concatenation of all 4 signals -> Classifier
"""
```

### RoPE (Rotary Positional Embedding)
```python
# Source: models/temporal_hybrid_v18.py lines 30-100
class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE encodes position by rotating query/key vectors, allowing the model to:
    1. Learn relative positions naturally through dot-product attention
    2. Generalize to different sequence lengths
    3. Detect valid breakouts at ANY timestep without human bias

    Key advantage over manual bias ramps:
    - No hardcoded assumptions about "when breakouts should occur"
    - Model learns position-dependent patterns from data
    """
```

### Model Constants
```python
# Source: config/constants.py lines 193-206
TEMPORAL_WINDOW_SIZE = 20          # Timesteps per sequence
QUALIFICATION_PHASE = 10           # Timesteps 1-10
VALIDATION_PHASE = 10              # Timesteps 11-20
MIN_PATTERN_DURATION = 20          # Minimum for valid pattern
FEATURE_DIM = 10                   # Features per timestep
NUM_CLASSES = 3                    # Danger, Noise, Target

CLASS_NAMES = {
    0: 'Danger',
    1: 'Noise',
    2: 'Target'
}
```

### Strategic Values
```python
# Source: config/constants.py lines 155-160
STRATEGIC_VALUES = {
    0: -2.0,   # Danger: Costs 2R (Stop Loss)
    1: -0.1,   # Noise: Opportunity Cost / Fees
    2: 5.0,    # Target: Average Win (conservative)
}
```

### Signal Thresholds
```python
# Source: config/constants.py lines 176-182
SIGNAL_THRESHOLDS = {
    'STRONG': 5.0,
    'GOOD': 3.0,
    'MODERATE': 1.0,
    'WEAK': 0.0,
    'AVOID': -1.0
}
```

### LazyHDF5Dataset
```python
# Source: pipeline/02_train_temporal.py lines 152-273
class LazyHDF5Dataset(Dataset):
    """
    Memory-efficient dataset that reads from HDF5 on demand.
    Applies robust scaling (median/IQR) on-the-fly.

    CRITICAL (Jan 2026 Fix): File handle is lazy-opened in __getitem__, NOT __init__.
    Opening h5py.File in __init__ makes the dataset unpickleable, causing DataLoader
    workers to fail silently (GPU starvation, 0% utilization during load).
    """

    def __getitem__(self, idx):
        # Apply robust scaling on-the-fly
        if self.train_median is not None and self.train_iqr is not None:
            seq = (seq - self.train_median) / self.train_iqr
```

### Window Jitter Augmentation
```python
# Source: pipeline/02_train_temporal.py lines 65-102
def apply_window_jitter(seq: np.ndarray, jitter_range: tuple = (-3, 2)) -> np.ndarray:
    """
    Apply random window jittering to a sequence.

    Forces the model to recognize patterns regardless of whether the breakout
    occurs at index 18, 19, or 20 - utilizing RoPE layers correctly.
    """
```

### Feature Flags
```python
# Source: config/constants.py lines 262-268
USE_GRN_CONTEXT = True              # Context Branch with GRN gating
USE_PROBABILITY_CALIBRATION = True  # Isotonic Regression post-hoc calibration
```

### Command
```bash
python pipeline/02_train_temporal.py \
    --sequences output/sequences/eu/sequences.h5 \
    --metadata output/sequences/eu/metadata.parquet \
    --epochs 100 \
    --use-coil-focal
```

---

## Step 3: Predict

**Script:** `pipeline/03_predict_temporal.py`

### Purpose
Generate position-sized trade orders with risk management.

### Risk Configuration
```python
# Source: pipeline/03_predict_temporal.py lines 70-73
DEFAULT_RISK_UNIT_DOLLARS = 250.0      # Max loss per trade
DEFAULT_MAX_CAPITAL_PER_TRADE = 5000.0 # Hard cap on position size
DEFAULT_ADV_LIQUIDITY_PCT = 0.04       # Never exceed 4% of ADV
DEFAULT_TRIGGER_OFFSET = 0.01          # $0.01 above upper boundary
```

### Intraday Volume Profile
```python
# Source: pipeline/03_predict_temporal.py lines 94-117
# Cumulative % of daily volume by time (minutes since 9:30 AM)
INTRADAY_VOLUME_PROFILE = {
    0: 0.00,     # 9:30 AM - market open
    15: 0.08,    # 9:45 AM - opening rush
    30: 0.15,    # 10:00 AM - still elevated
    45: 0.20,    # 10:15 AM
    60: 0.25,    # 10:30 AM - first hour done (~25%)
    90: 0.32,    # 11:00 AM
    120: 0.38,   # 11:30 AM
    150: 0.43,   # 12:00 PM - lunch lull starts
    180: 0.48,   # 12:30 PM
    210: 0.52,   # 1:00 PM - midday trough
    240: 0.57,   # 1:30 PM
    270: 0.62,   # 2:00 PM - picking up
    300: 0.68,   # 2:30 PM
    330: 0.76,   # 3:00 PM - closing rush
    360: 0.85,   # 3:30 PM
    390: 1.00,   # 4:00 PM - market close
}

DYNAMIC_PACING_MULTIPLIER = 1.5  # Vol_Current must exceed 1.5x expected
V19_STATIC_PACING_PCT = 0.30     # Fallback: 30% at 10:00 AM
```

### Expected Volume Calculation
```python
# Source: pipeline/03_predict_temporal.py lines 120-150
def get_expected_volume_pct(minutes_since_open: int) -> float:
    """
    Get expected cumulative volume percentage at a given time.
    Uses linear interpolation between known points in the intraday profile.
    """
```

### Command
```bash
python pipeline/03_predict_temporal.py \
    --model output/models/best_model.pt \
    --sequences output/sequences/live/sequences.h5 \
    --metadata output/sequences/live/metadata.parquet \
    --risk-unit 250 \
    --max-capital 5000 \
    --min-signal GOOD
```

---

## Unified Pipeline

**Script:** `main_pipeline.py`

### Region Configuration
```python
# Source: main_pipeline.py lines 122-167
configs = {
    'EU': {
        'baseline_index': 'EXSA.DE',  # iShares EURO STOXX
        'default_tickers': 'data/eu_tickers.txt',
        'output_prefix': 'eu',
        'model_name': 'eu_model',
        'seq_dir': Path('output/sequences/eu'),
        'patterns_file': Path('output/eu_candidate_patterns.parquet'),
        'labeled_patterns_file': Path('output/eu_labeled_patterns.parquet'),
    },
    'US': {
        'baseline_index': 'RSP',  # Equal-weight S&P 500
        'default_tickers': 'data/us_tickers.txt',
        'output_prefix': 'us',
        'model_name': 'us_model',
        'seq_dir': Path('output/sequences/us'),
        'patterns_file': Path('output/us_candidate_patterns.parquet'),
        'labeled_patterns_file': Path('output/us_labeled_patterns.parquet'),
    },
    'GLOBAL': {
        'baseline_index': None,  # Cohort-based only
        'default_tickers': 'data/all_tickers.txt',
        'output_prefix': 'global',
        'model_name': 'production_model',
        'seq_dir': Path('output/sequences'),
        'patterns_file': Path('output/candidate_patterns.parquet'),
        'labeled_patterns_file': Path('output/labeled_patterns.parquet'),
    }
}
```

### Pipeline Constants
```python
# Source: main_pipeline.py lines 68-77
OUTCOME_WINDOW_DAYS = 40   # From 00b_label_outcomes.py
ZOMBIE_TIMEOUT_DAYS = 150  # Force Noise if no data after this many days

STEP_DETECT = 'detect'
STEP_LABEL = 'label'
STEP_GENERATE = 'generate'
STEP_TRAIN = 'train'

PIPELINE_ORDER = [STEP_DETECT, STEP_LABEL, STEP_GENERATE, STEP_TRAIN]
```

### Full Pipeline Command
```bash
python main_pipeline.py \
    --region EU \
    --full-pipeline \
    --apply-nms \
    --apply-physics-filter \
    --mode training \
    --use-coil-focal \
    --epochs 100
```

### Individual Step Commands
```bash
# Detect patterns
python main_pipeline.py --region EU --step detect --tickers data/eu_tickers.txt

# Label outcomes
python main_pipeline.py --region EU --step label --enable-early-labeling

# Generate sequences
python main_pipeline.py --region EU --step generate --apply-nms --apply-physics-filter --mode training

# Train model
python main_pipeline.py --region EU --step train --epochs 100 --use-coil-focal
```

---

## Key File References

| File | Purpose | Key Lines |
|------|---------|-----------|
| `pipeline/00_detect_patterns.py` | Pattern detection | 138-161 (thresholds) |
| `pipeline/00b_label_outcomes.py` | Outcome labeling | 60-74 (constants), 185-222 (R-metrics) |
| `pipeline/01_generate_sequences.py` | Sequence generation | 119-143 (FUTURE_ prefix) |
| `pipeline/02_train_temporal.py` | Model training | 152-273 (LazyHDF5Dataset) |
| `pipeline/03_predict_temporal.py` | Prediction | 70-73 (risk), 94-117 (pacing) |
| `config/constants.py` | System constants | 19-31 (classes), 155-182 (values/thresholds) |
| `config/context_features.py` | Context features | 22-47 (18 features), 73-97 (log_diff) |
| `config/temporal_features.py` | Temporal features | 32-72 (10 features) |
| `models/temporal_hybrid_v18.py` | Model architecture | 1-21 (overview), 30-100 (RoPE) |
| `main_pipeline.py` | Orchestrator | 68-77 (constants), 122-167 (regions) |

---

## Validation Checklist

```bash
# 1. Check NMS cluster ratio (should be > 1.0, ideally 3-5)
python -c "import pandas as pd; df=pd.read_parquet('output/metadata.parquet'); \
  print(f'Ratio: {len(df)/df.nms_cluster_id.nunique():.1f}')"

# 2. Check for cross-ticker duplicates
python -c "import pandas as pd; df=pd.read_parquet('output/metadata.parquet'); \
  df['sig']=df['pattern_end_date'].astype(str)+df['upper_boundary'].round(2).astype(str); \
  print(f'Duplicates: {df.duplicated(\"sig\").sum()}/{len(df)}')"

# 3. Verify data alignment
python -c "import h5py; import pandas as pd; import numpy as np; \
  meta=pd.read_parquet('output/metadata.parquet'); \
  with h5py.File('output/sequences.h5','r') as f: labels=f['labels'][:]; \
  assert np.array_equal(meta['label'].values, labels), 'MISALIGNED!'; \
  print('Alignment OK')"

# 4. Validate sequence integrity
python scripts/validate_sequence_integrity.py --seq-dir output/sequences/eu
```

---

*Generated from codebase verification - January 2026*
