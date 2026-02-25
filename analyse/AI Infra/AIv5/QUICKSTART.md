# AIv5 Quick Start Guide

**Goal**: Fix feature mismatch and restore K4 detection capability

---

## 🚨 Priority 1: Fix Feature Mismatch (CRITICAL)

### The Problem

Models trained on **69 enhanced features**, but validation using **31 temporal_safe features**.
- Only 3 features overlap
- K4 recall: 0% (models can't use trained features)
- EV correlation: -0.140 (negative!)

### The Solution (30-45 min)

```bash
# 1. Extract correct features (69 enhanced)
python extract_features_historical_enhanced.py
# Expected: pattern_features_historical_48enhanced.parquet
# Runtime: ~30-45 minutes for 708 snapshots

# 2. Verify feature count
python -c "import pandas as pd; df = pd.read_parquet('output/unused_patterns/pattern_features_historical_48enhanced.parquet'); print(f'Features: {len(df.select_dtypes(include=[\"float64\", \"float32\", \"int64\", \"int32\"]).columns)} (should be 69)')"

# 3. Rerun predictions with matched features
python predict_historical_enhanced.py
# Expected: predictions_historical_enhanced.parquet
# Runtime: ~5 minutes

# 4. Generate validation report
python generate_validation_report_enhanced.py
# Expected: VALIDATION_REPORT_ENHANCED.md
# Runtime: ~2 minutes
```

### Expected Results

| Metric | Before (Feature Mismatch) | After (Feature Match) | Improvement |
|--------|---------------------------|----------------------|-------------|
| K4 Recall | 0% | 10-40% | +10-40% |
| K3+K4 Recall | 14.9% | 40-60% | 3-4× better |
| EV Correlation | -0.140 | +0.10 to +0.25 | Flipped positive |

**Note**: K4 recall may still be limited by extreme rarity in training data (0.01% prevalence)

---

## 📊 Priority 2: Address K4 Extreme Rarity (If K4 Still Fails)

### The Problem

K4 patterns extremely rare in training:
- Training: 12 K4 examples out of 131,580 snapshots (0.01%)
- Validation: 20 K4 out of 708 (2.82%) - **282× more frequent!**
- Models never learned to recognize K4 patterns

### Solution Options

#### Option A: Retrain with Combined Dataset
```bash
# Combine training + validation (32 K4 total, still rare but 2.7× better)
python src/pipeline/03_train_enhanced.py --combined-dataset

# Expected: 132,288 snapshots, 32 K4 (0.024%)
# Runtime: ~30-60 minutes
```

#### Option B: SMOTE Oversampling
```bash
# Synthetic minority oversampling for K4 class
python src/pipeline/03_train_enhanced.py --use-smote --k4-oversample-ratio=0.05

# Expected: K4 prevalence 0.01% → 5% (synthetic)
# Runtime: ~45-90 minutes
```

#### Option C: Extreme Focal Loss
```bash
# Alpha weight = 500 for K4 class (extreme prioritization)
python src/pipeline/03_train_enhanced.py --focal-loss --k4-alpha=500

# Expected: Model focuses heavily on K4 examples
# Runtime: ~30-60 minutes
```

#### Option D: Separate K4 Binary Classifier
```bash
# Binary classifier: K4 vs not-K4
python src/pipeline/train_k4_binary_classifier.py

# Expected: Specialized K4 detector
# Runtime: ~20-40 minutes
# NOTE: This script may need to be created
```

**Recommendation**: Try Option A first (combined dataset), then Option C (focal loss) if needed.

---

## 🔧 Installation & Setup

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure GCS Access
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# PROJECT_ID=ignition-ki-csv-storage
# GCS_BUCKET_NAME=ignition-ki-csv-data-2025-user123
```

### 3. Verify Installation
```bash
# Test imports
python -c "from pattern_detection.features.features.enhanced_tabular_features import EnhancedTabularFeatureExtractor; print('✅ Pattern detection OK')"

python -c "from shared.config.settings import get_settings; settings = get_settings(); print('✅ Config OK')"

python -c "import xgboost as xgb; import lightgbm as lgb; print('✅ ML libraries OK')"
```

---

## 📁 Essential Files (Only 6 Scripts!)

### Production Workflow

1. **detect_patterns_historical_unused.py**
   - Scans OHLCV data for consolidation patterns
   - Generates snapshots (2-5 per active pattern)
   - Output: `output/unused_patterns/detected_patterns.parquet`

2. **extract_features_historical_enhanced.py** ⭐ CRITICAL
   - Extracts 69 enhanced features (30 base + 19 EBP + 20 metadata)
   - Temporal-safe: Uses only data ≤ snapshot_date
   - Output: `output/unused_patterns/pattern_features_historical_48enhanced.parquet`

3. **src/pipeline/02_label_patterns_firstbreach.py**
   - Labels patterns with K0-K5 outcomes
   - First-breach logic: Temporal priority over best outcome
   - Output: `output/patterns_labeled_enhanced_firstbreach.parquet`

4. **src/pipeline/03_train_enhanced.py**
   - Trains XGBoost + LightGBM ensemble
   - Focal loss with extreme class weighting (K4 alpha=30-500)
   - Output: `output/models/*_extreme_*.pkl`

5. **predict_historical_enhanced.py**
   - Loads trained models, generates predictions
   - Calculates expected value (EV) from class probabilities
   - Output: `output/unused_patterns/predictions_historical_enhanced.parquet`

6. **generate_validation_report_enhanced.py**
   - Compares predictions vs actual outcomes
   - Confusion matrix, K4 recall, EV correlation
   - Output: `output/unused_patterns/VALIDATION_REPORT_ENHANCED.md`

---

## 📈 Understanding the Metrics

### Primary Success Metrics

**K4 Recall** (Primary Target)
- % of actual K4 patterns correctly detected
- Target: >30% (current: 0% due to feature mismatch)
- Exceptional: >50%

**K3+K4 Combined Recall**
- % of profitable patterns (≥35% gain) detected
- Target: >60%
- Exceptional: >75%

**EV Correlation**
- Correlation between predicted EV and actual gain
- Target: +0.15 to +0.25
- Exceptional: >+0.30

### Signal Distribution (Target)

- AVOID: 60-70% (EV < 0 or P(K5) > 30%)
- WEAK: 5-10%
- MODERATE: 10-15%
- GOOD: 5-10%
- STRONG: 1-3% (EV ≥ 5.0)

### Pattern Outcome Classes

```
K4_EXCEPTIONAL: ≥75% gain   → Value: +10  (PRIMARY TARGET)
K3_STRONG:      35-75% gain → Value: +3
K2_QUALITY:     15-35% gain → Value: +1
K1_MINIMAL:     5-15% gain  → Value: -0.2 (penalized)
K0_STAGNANT:    <5% gain    → Value: -2   (penalized)
K5_FAILED:      Breakdown   → Value: -10  (AVOID)
```

**Expected Value Formula:**
```
EV = P(K4)×10 + P(K3)×3 + P(K2)×1 + P(K1)×(-0.2) + P(K0)×(-2) + P(K5)×(-10)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific modules
pytest tests/pattern_detection/test_modern_tracker.py
pytest tests/shared/test_config.py

# Run with coverage
pytest --cov=pattern_detection --cov=training --cov-report=html
```

---

## 📚 Documentation

### Essential Reading (in order)

1. **QUICKSTART.md** (this file) - Immediate actions
2. **CLAUDE.md** - Complete system architecture
3. **docs/TRAINING_VALIDATION_COMPARISON_20251103.md** - Root cause analysis
4. **docs/MODEL_IMPROVEMENT_REPORT.md** - First-breach success story
5. **docs/VALIDATION_REPORT_BBW_FIX_20251103.md** - BBW fix results

### Historical Context

- **docs/ACTION_PLAN_FEATURE_FIXES.md** - Feature calculation bugs identified
- **docs/BBW_FIX_VERIFICATION.md** - BBW mandatory criterion validation
- **docs/CONSOLIDATION_CRITERIA_FIX_SUMMARY.md** - Criteria evolution
- **docs/PROJECT_COMPLETE.md** - First-breach implementation summary

---

## ⚡ Common Issues

### Issue 1: ModuleNotFoundError
```bash
# Solution: Install in development mode
pip install -e .
```

### Issue 2: GCS Access Denied
```bash
# Solution: Check .env credentials
cat .env  # Verify PROJECT_ID and GCS_BUCKET_NAME

# Test GCS access
python -c "from data_acquisition.storage.storage_manager import StorageManager; mgr = StorageManager(); print('✅ GCS OK')"
```

### Issue 3: Out of Memory (8GB RAM)
```bash
# Solution: Enable memory optimization
export MEMORY_MODE=aggressive

# Or: Reduce batch size in training
python src/pipeline/03_train_enhanced.py --batch-size 5000
```

### Issue 4: K4 Recall Still 0% After Feature Fix
```bash
# Likely cause: K4 extreme rarity in training (0.01%)
# Solution: Proceed to Priority 2 (retrain with combined dataset or SMOTE)
```

---

## 🎯 Success Checklist

After completing Priority 1, verify:

- [ ] 69 features extracted (not 31!)
- [ ] Predictions generated with matched features
- [ ] Validation report shows K4 recall >0% (ideally >10%)
- [ ] EV correlation is positive (+0.10 or higher)
- [ ] Signal distribution: 60-70% AVOID, 1-3% STRONG

If K4 recall still low (<10%):

- [ ] Proceed to Priority 2 (retrain with combined dataset)
- [ ] Or: Use SMOTE oversampling
- [ ] Or: Train K4-specific binary classifier

---

## 🚀 Production Deployment (After Validation)

```bash
# 1. Copy trained models
cp output/models/*_extreme_*.pkl /production/models/

# 2. Schedule daily pattern scanning
# (Add to crontab or task scheduler)
0 18 * * * /path/to/venv/bin/python detect_patterns_historical_unused.py

# 3. Extract features for new patterns
0 19 * * * /path/to/venv/bin/python extract_features_historical_enhanced.py

# 4. Generate predictions
0 20 * * * /path/to/venv/bin/python predict_historical_enhanced.py

# 5. Alert on STRONG signals (EV ≥ 5.0)
# (Integrate with notification system)
```

---

## 📞 Support

**Issues**: Check `docs/TRAINING_VALIDATION_COMPARISON_20251103.md` for troubleshooting

**Architecture Questions**: See `CLAUDE.md` for detailed system design

**Next Steps**: After Priority 1, review validation report and decide on Priority 2 approach

---

**AIv5 Quick Start Version**: 1.0.0
**Last Updated**: 2025-11-04
**Status**: Ready for Priority 1 execution
