# CLAUDE.md - AIv5 Clean Production System

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is AIv5?

**AIv5** is a clean refactoring of AIv4, containing only essential production code for:
- **Pattern Detection**: State machine identifying consolidation patterns
- **Feature Extraction**: 69 enhanced features (base + EBP composite indicators)
- **ML Training**: XGBoost + LightGBM with first-breach classification
- **Prediction & Validation**: Expected value calculations and signal generation

## What Changed from AIv4 → AIv5

### Removed (Clutter Cleanup)
- ❌ **41 analysis/debug scripts** (analyze_*.py, debug_*.py, compare_*.py, etc.)
- ❌ **20+ outdated .md files** (old reports, session summaries, etc.)
- ❌ **3 archive folders** (cleanup_*, archive_md/, etc.)
- ❌ **Generated outputs** (*.parquet, *.pkl models - regenerate in AIv5)
- ❌ **Deprecated scripts** (temporal_safe, old extractors, old validators)

### Kept (Essential Only)
- ✅ **Core modules** (pattern_detection/, training/, data_acquisition/, shared/, src/)
- ✅ **6 essential scripts** (extract, predict, scan, detect, generate, main)
- ✅ **8 current .md docs** (recent critical reports only)
- ✅ **4 config files** (.env.example, requirements.txt, setup.py, pyproject.toml)
- ✅ **12 test files** (tests/)

### Result
- **Before**: 47 root scripts, 25+ .md files, 3 archives → confusing
- **After**: 6 root scripts, 8 .md files, 0 archives → clear execution path

---

## Critical Lessons Learned (from AIv4)

### 1. Feature Mismatch = 0% K4 Recall ⚠️

**Issue**: Models trained on **69 enhanced features**, but validation used **31 temporal_safe features** (only 3 overlapped!)

**Result**:
- K4 recall: 0% (models couldn't use trained features)
- EV correlation: -0.140 (negative!)

**Fix**: Always use `extract_features_historical_enhanced.py` (69 features) for validation

### 2. BBW Mandatory Criterion

**Issue**: Old "3 of 4" criteria allowed patterns WITHOUT volatility compression (BBW)

**Result**:
- Mean BBW: 119.8% (explosive, not compressed!)
- Only 26.7% patterns had proper BBW

**Fix**: BBW < 30th percentile is now MANDATORY + 2 of 3 others
- New mean BBW: 26.2% (4.6× better)
- 70.4% compliance (2.6× improvement)

### 3. First-Breach Classification

**Issue**: Old labeling used "best outcome" (max_gain), not temporal priority

**Result**:
- Pattern drops -60% on day 5 (K5 breach), rallies to +100% on day 50
- Old label: K4_EXCEPTIONAL (wrong!)
- Correct label: K5_FAILED (first breach wins)

**Fix**: `02_label_patterns_firstbreach.py` uses day-by-day breach detection
- 10% grace period for K5 (prevents false triggers from brief dips)
- Models improved: K4 recall 0% → 66.7%, K3 recall 0-11% → 75-79%

### 4. K4 Extreme Rarity

**Issue**: K4 patterns extremely rare in training data

**Stats**:
- Training: 12 K4 examples out of 131,580 snapshots (0.01%)
- Validation: 20 K4 out of 708 (2.82%) - **282× more frequent!**

**Impact**: Models never learned to detect K4 patterns (insufficient examples)

**Solutions** (pending):
- Retrain with combined dataset (32 K4 total)
- SMOTE synthetic oversampling
- Focal loss with alpha=500 for K4
- Separate K4 binary classifier

---

## System Architecture

### Core Modules

**pattern_detection/** (261KB)
- `state_machine/` - ModernPatternTracker (QUALIFYING → ACTIVE → COMPLETED)
- `features/` - EnhancedTabularFeatureExtractor (30 base + 19 EBP = 49 features)
- `models/` - ConsolidationPattern dataclass
- `protocols/` - PatternTracker protocol interface
- `services/` - PatternDetectionService, LabelingService

**training/** (241KB)
- `models_orig/` - XGBoostClassifier, LightGBMClassifier
- `labeling/` - EnhancedPatternLabeler (K0-K5 first-breach)
- `pipelines/` - Complete training workflows

**data_acquisition/** (180MB - includes GCS caching)
- `sources/` - YFinanceDownloader, AsyncDownloader
- `storage/` - StorageManager (GCS integration)
- `screening/` - TickerScreener (micro/small-cap filters)

**shared/** (101KB)
- `config/` - Pydantic settings (consolidation criteria, ML hyperparams)
- `indicators/` - Technical indicators (BBW, ADX, volume ratio)
- `utils/` - Data loading, checkpointing, memory optimization

**src/pipeline/** (197KB)
- `01_scan_patterns.py` - Detect patterns from OHLCV data
- `02_label_patterns_firstbreach.py` - Label with K0-K5 outcomes
- `03_train_enhanced.py` - Train XGBoost + LightGBM
- `05_feature_selection_shap.py` - SHAP importance analysis
- `07_train_no_leakage.py` - Temporal-safe training

### Essential Scripts (Root Level)

**1. extract_features_historical_enhanced.py** ⭐ CRITICAL
- Extracts 69 enhanced features (30 base + 19 EBP + 20 metadata/boundaries)
- Uses EnhancedTabularFeatureExtractor with EBP enabled
- Temporal-safe: Only uses data ≤ snapshot_date
- Input: detected_patterns.parquet (snapshots)
- Output: pattern_features_enhanced.parquet (69 features)

**2. predict_historical_enhanced.py**
- Loads trained models (XGBoost + LightGBM)
- Generates predictions on 69-feature validation set
- Calculates expected value (EV) from class probabilities
- Output: predictions_enhanced.parquet

**3. generate_validation_report_enhanced.py**
- Compares predictions vs actual outcomes
- Calculates K4 recall, K3+K4 recall, EV correlation
- Generates confusion matrix, class-level accuracy
- Output: validation_report_enhanced.md

**4. detect_patterns_historical_unused.py**
- Runs pattern detection on historical tickers
- Generates snapshots (2-5 per active pattern)
- Output: detected_patterns.parquet

**5. scan_consolidated_optimized.py**
- Fast batch pattern scanning
- Memory-optimized for large ticker sets
- Uses shared/config for consolidation criteria

**6. main.py**
- Main entry point for interactive pattern scanning
- Supports --tickers, --start-date, --end-date flags

---

## Pattern Outcome Classes & Strategic Values

```
K4_EXCEPTIONAL: ≥75% gain   → Value: +10  (PRIMARY TARGET)
K3_STRONG:      35-75% gain → Value: +3   (SECONDARY TARGET)
K2_QUALITY:     15-35% gain → Value: +1
K1_MINIMAL:     5-15% gain  → Value: -0.2 (penalized - not worth risk)
K0_STAGNANT:    <5% gain    → Value: -2   (penalized)
K5_FAILED:      Breakdown   → Value: -10  (AVOID)
```

**Expected Value (EV) Calculation:**
```
EV = P(K4)×10 + P(K3)×3 + P(K2)×1 + P(K1)×(-0.2) + P(K0)×(-2) + P(K5)×(-10)
```

**Signal Thresholds:**
- STRONG_SIGNAL: EV ≥ 5.0
- GOOD_SIGNAL: EV ≥ 3.0
- MODERATE_SIGNAL: EV ≥ 1.0
- AVOID: EV < 0 OR P(K5) > 30%

---

## Essential Commands (Production Workflow)

### Priority 1: Fix Feature Mismatch (IMMEDIATE) ⚠️

**Problem**: Validation currently using wrong features (31 vs 69)

**Solution**:
```bash
# 1. Regenerate validation features with correct pipeline
python extract_features_historical_enhanced.py

# 2. Verify 69 features extracted
python -c "import pandas as pd; df = pd.read_parquet('output/unused_patterns/pattern_features_historical_48enhanced.parquet'); print(f'Features: {len(df.select_dtypes(include=[\"float64\", \"float32\", \"int64\", \"int32\"]).columns)} (should be 69)')"

# 3. Rerun predictions with matched features
python predict_historical_enhanced.py

# 4. Generate new validation report
python generate_validation_report_enhanced.py

# 5. Check results
# Expected: K4 recall 0% → 10-40% (limited by K4 rarity)
# Expected: EV correlation -0.14 → +0.10 to +0.25
```

### Priority 2: Address K4 Rarity (After P1)

**Problem**: Only 12 K4 examples in 131,580 training snapshots (0.01%)

**Options**:

**Option A: Retrain with Combined Dataset**
```bash
# Combine training + validation (32 K4 total, still rare but 2.7× better)
python src/pipeline/03_train_enhanced.py --combined-dataset
```

**Option B: SMOTE Oversampling**
```bash
# Synthetic minority oversampling for K4 class
python src/pipeline/03_train_enhanced.py --use-smote --k4-oversample-ratio=0.05
```

**Option C: Extreme Focal Loss**
```bash
# Alpha weight = 500 for K4 class (extreme prioritization)
python src/pipeline/03_train_enhanced.py --focal-loss --k4-alpha=500
```

**Option D: Separate K4 Classifier**
```bash
# Binary classifier: K4 vs not-K4
python src/pipeline/train_k4_binary_classifier.py
```

### Standard Workflow (After Feature Fix)

```bash
# 1. Scan patterns from cached data
python detect_patterns_historical_unused.py

# 2. Extract 69 enhanced features
python extract_features_historical_enhanced.py

# 3. Label patterns with K0-K5 outcomes (first-breach)
python src/pipeline/02_label_patterns_firstbreach.py

# 4. Train models (XGBoost + LightGBM)
python src/pipeline/03_train_enhanced.py

# 5. Generate predictions
python predict_historical_enhanced.py

# 6. Validate results
python generate_validation_report_enhanced.py
```

---

## Feature Engineering (69 Features)

### Base Features (30+)
**Pattern metrics**: days_in_pattern, days_qualifying, days_active, range_width_pct
**Volatility**: current_bbw_20, bbw_percentile, bbw_slope, atr_normalized
**Trend**: current_adx, adx_trend, trend_strength
**Volume**: volume_ratio, volume_trend, accumulation_score
**Price position**: distance_to_power, distance_to_upper, price_position_in_range

**Top 3 by SHAP importance:**
1. `distance_to_power` (2.89) - Proximity to breakout threshold
2. `current_bbw_20` (1.78) - Current volatility contraction
3. `range_width_pct` (1.35) - Consolidation tightness

### EBP Features (19) - Explosive Breakout Predictor

**Composite quality indicator** combining:

**CCI (Consolidation Compression Index) - 4 features:**
- `cci_bbw_compression` - BBW vs max BBW ratio
- `cci_atr_compression` - ATR vs max ATR ratio
- `cci_days_factor` - Consecutive low-volatility days
- `cci_score` - Composite CCI (0-1)

**VAR (Volume Accumulation Ratio) - 2 features:**
- `var_raw` - Σ(up-volume × price-change) / Σ(down-volume × |price-change|)
- `var_score` - Normalized VAR (0-1)

**NES (Narrative Energy Score) - 4 features:**
- `nes_inactive_mass` - Multi-timeframe volatility energy
- `nes_wavelet_energy` - High-frequency energy compression
- `nes_rsa_proxy` - Momentum-based relative strength
- `nes_score` - Composite NES (0-1)

**LPF (Liquidity Pressure Factor) - 4 features:**
- `lpf_bid_pressure` - Intraday bid-ask pressure
- `lpf_volume_pressure` - Volume vs moving average
- `lpf_fta_proxy` - Turnover acceleration
- `lpf_score` - Composite LPF (0-1)

**TSF (Time Scaling Factor) - 2 features:**
- `tsf_days_in_consolidation` - Pattern duration
- `tsf_score` - Time weighting (1 + √(days/30))

**EBP Composite - 3 features:**
- `ebp_raw` - (CCI × VAR × NES × LPF)^(1/TSF)
- `ebp_composite` - Normalized EBP (0-1)
- `ebp_signal` - Classification (WEAK/MODERATE/GOOD/STRONG/EXCEPTIONAL)

**Usage:**
```python
from pattern_detection.features.features.enhanced_tabular_features import EnhancedTabularFeatureExtractor

extractor = EnhancedTabularFeatureExtractor(use_ebp=True)
df_with_features = extractor.extract_features(df_ohlcv, pattern, snapshot_date)
```

### Metadata/Boundaries (20 features)
- Pattern boundaries (upper, lower, power)
- Current prices (open, high, low, close)
- Pattern state (days_since_activation, end_date)
- Outcome labels (for training: outcome, outcome_gain, days_to_peak)

**CRITICAL**: All features calculated using ONLY data available at snapshot_date (no look-ahead bias)

---

## Configuration

### GCS Data Storage
```bash
export PROJECT_ID="ignition-ki-csv-storage"
export GCS_BUCKET_NAME="ignition-ki-csv-data-2025-user123"
```

### Consolidation Criteria (shared/config/settings.py)
```python
BBW_PERCENTILE_THRESHOLD = 0.30  # MANDATORY
ADX_THRESHOLD = 32               # Optional (2 of 3)
VOLUME_THRESHOLD = 0.35          # Optional (2 of 3)
RANGE_THRESHOLD = 0.65           # Optional (2 of 3)
```

### Model Hyperparameters
```python
# XGBoost
N_ESTIMATORS = 100
MAX_DEPTH = 6
LEARNING_RATE = 0.1
TREE_METHOD = 'hist'  # Memory-efficient

# LightGBM
NUM_LEAVES = 31
MIN_DATA_IN_LEAF = 20
```

---

## Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/pattern_detection/test_modern_tracker.py
pytest tests/shared/test_config.py

# Run with coverage
pytest --cov=pattern_detection --cov=training --cov-report=html
```

---

## Current Model Performance (First-Breach Models)

**Training Data**: 131,580 snapshots (first-breach labeled)
**Models**: `output/models/*_extreme_20251101_201450.pkl`

| Metric | Before First-Breach | After First-Breach | Improvement |
|--------|---------------------|-------------------|-------------|
| K4 Detection | 0% | 66.7% | +66.7% |
| K3 Detection | 0-11% | 75-79% | +64-79% |
| K3+K4 Combined | ~5% | 75-80% | 15× improvement |
| K5 Over-Prediction | 85.7% | 40-43% | Fixed |
| EV Correlation | -0.14 | +0.20-0.35 | Flipped positive |

**Validation Status**: PENDING FEATURE MATCH FIX (see Priority 1)

---

## Known Issues & Fixes

### ✅ RESOLVED: First-Breach Classification (2025-11-01)
- **Issue**: Best-outcome labeling masked failures
- **Fix**: Day-by-day first-breach detection in `02_label_patterns_firstbreach.py`
- **Result**: K4 recall 0% → 66.7%, K3 recall 11% → 75-79%

### ✅ RESOLVED: BBW Mandatory Criterion (2025-11-03)
- **Issue**: Patterns qualified without volatility compression
- **Fix**: BBW < 30th percentile now MANDATORY in consolidation criteria
- **Result**: Mean BBW 119.8% → 26.2% (4.6× better)

### ⚠️ ACTIVE: Feature Mismatch (2025-11-03)
- **Issue**: Training (69 features) ≠ Validation (31 features), only 3 overlap
- **Impact**: K4 recall stuck at 0% despite all fixes
- **Fix**: Regenerate validation features using `extract_features_historical_enhanced.py`
- **Status**: Ready to execute (see Priority 1 commands)

### ⚠️ ACTIVE: K4 Extreme Rarity (2025-11-01)
- **Issue**: Only 12 K4 examples in 131,580 training snapshots (0.01%)
- **Impact**: Models never learned K4 patterns (validation has 282× more K4s!)
- **Fix Options**: Combined dataset, SMOTE, focal loss, binary classifier
- **Status**: Pending P1 completion

---

## Directory Structure

```
AIv5/
├── pattern_detection/          # Core pattern detection (261KB)
│   ├── state_machine/          # ModernPatternTracker
│   ├── features/               # EnhancedTabularFeatureExtractor
│   ├── models/                 # ConsolidationPattern dataclass
│   ├── protocols/              # Interfaces
│   └── services/               # Detection, labeling, validation services
├── training/                   # ML training (241KB)
│   ├── models_orig/            # XGBoost, LightGBM classifiers
│   ├── labeling/               # EnhancedPatternLabeler
│   └── pipelines/              # Complete workflows
├── data_acquisition/           # Data loading (180MB)
│   ├── sources/                # YFinance, async downloaders
│   ├── storage/                # GCS integration
│   └── screening/              # Ticker filters
├── shared/                     # Shared utilities (101KB)
│   ├── config/                 # Pydantic settings
│   ├── indicators/             # BBW, ADX, volume calculations
│   └── utils/                  # Data loading, checkpointing
├── src/
│   ├── pipeline/               # 11 pipeline scripts (01-07)
│   └── utils/                  # Helper utilities
├── tests/                      # 12 test files
├── config/                     # Empty (uses shared/config)
├── docs/                       # 8 essential .md files
├── output/                     # Generated results
│   ├── models/                 # Trained .pkl models
│   ├── validation/             # Validation reports
│   └── unused_patterns/        # Validation datasets
├── data/                       # Data cache
│   ├── gcs_cache/              # Downloaded ticker OHLCV
│   └── unused_tickers_cache/   # Validation ticker cache
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── pyproject.toml              # Build configuration
└── 6 essential .py scripts     # extract, predict, scan, detect, generate, main
```

---

## Next Session Quick Start

1. **Fix feature mismatch** (30-45 min runtime):
   ```bash
   python extract_features_historical_enhanced.py
   python predict_historical_enhanced.py
   python generate_validation_report_enhanced.py
   ```

2. **Expected outcome**: K4 recall 0% → 10-40%, EV correlation -0.14 → +0.15-0.25

3. **If K4 still fails**: Retrain with combined dataset or SMOTE (see Priority 2)

4. **Monitor**:
   - Signal distribution: Target 60-70% AVOID, 1-3% STRONG
   - EV correlation: Target +0.20 to +0.35
   - Class accuracy: K4 recall >30%, K3 recall >60%

---

**AIv5 Version**: 1.0.0
**Created**: 2025-11-04
**Status**: Production-Ready (pending P1 feature fix)
