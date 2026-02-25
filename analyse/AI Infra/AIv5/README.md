# AIv5 - Clean Production Pattern Detection System

**Version**: 5.0.0
**Created**: 2025-11-04
**Status**: Production-Ready (pending feature fix)

---

## What is AIv5?

AIv5 is a **clean refactoring** of AIv4, removing clutter and keeping only essential production code for:

- ✅ **Pattern Detection** - State machine identifying consolidation patterns
- ✅ **Feature Extraction** - 69 enhanced features (30 base + 19 EBP + 20 metadata)
- ✅ **ML Training** - XGBoost + LightGBM with first-breach classification
- ✅ **Prediction & Validation** - Expected value calculations and signal generation

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Configure GCS Access
```bash
cp .env.example .env
# Edit .env with your PROJECT_ID and GCS_BUCKET_NAME
```

### 3. Verify Installation
```bash
python test_imports.py
# Expected: 16/18 imports successful (88.9%)
```

### 4. Run Priority 1 Fix (Feature Mismatch)
```bash
# Extract correct 69 features
python extract_features_historical_enhanced.py

# Generate predictions
python predict_historical_enhanced.py

# Validate results
python generate_validation_report_enhanced.py
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.**

---

## What Changed from AIv4 → AIv5

### Removed (Clutter Cleanup)
- ❌ **41 analysis/debug scripts** (analyze_*.py, debug_*.py, compare_*.py, etc.)
- ❌ **20+ outdated .md files** (old reports, session summaries, archives)
- ❌ **3 archive folders** (cleanup_*, archive_md/, etc.)
- ❌ **Generated outputs** (*.parquet, *.pkl models - regenerate in AIv5)

### Kept (Essential Only)
- ✅ **Core modules** (pattern_detection/, training/, data_acquisition/, shared/, src/)
- ✅ **6 essential scripts** (extract, predict, scan, detect, generate, main)
- ✅ **8 current .md docs** (recent critical reports only)
- ✅ **4 config files** (.env.example, requirements.txt, setup.py, pyproject.toml)
- ✅ **12 test files** (tests/)

### Result
- **Before**: 47 root scripts, 25+ .md files → confusing
- **After**: 6 root scripts, 8 .md files → clear execution path

---

## File Structure

```
AIv5/
├── pattern_detection/          # Core pattern detection (261KB)
├── training/                   # ML training (241KB)
├── data_acquisition/           # Data loading (180MB)
├── shared/                     # Shared utilities (101KB)
├── src/
│   ├── pipeline/              # 11 pipeline scripts (01-07)
│   └── utils/                 # Helper utilities
├── tests/                     # 12 test files
├── docs/                      # 8 essential .md files
├── output/                    # Empty (for results)
├── data/                      # Empty (for cache)
├── CLAUDE.md                  # Complete system architecture
├── QUICKSTART.md              # Immediate action guide
├── README.md                  # This file
├── test_imports.py            # Import verification script
├── 6 essential .py scripts    # extract, predict, scan, detect, generate, main
└── 4 config files             # .env.example, requirements.txt, setup.py, pyproject.toml
```

---

## Essential Scripts (6 Total)

1. **extract_features_historical_enhanced.py** ⭐ CRITICAL
   - Extracts 69 enhanced features for validation
   - **Must use this** (not temporal_safe version)

2. **predict_historical_enhanced.py**
   - Generates predictions with matched features

3. **generate_validation_report_enhanced.py**
   - Validates model performance

4. **detect_patterns_historical_unused.py**
   - Scans for consolidation patterns

5. **scan_consolidated_optimized.py**
   - Fast batch pattern scanning

6. **main.py**
   - Interactive pattern scanning entry point

---

## Critical Issues & Fixes

### ⚠️ ACTIVE: Feature Mismatch (Priority 1)
**Issue**: Training (69 features) ≠ Validation (31 features)
**Impact**: K4 recall stuck at 0%
**Fix**: Run `extract_features_historical_enhanced.py`
**Expected**: K4 recall 0% → 10-40%

### ⚠️ ACTIVE: K4 Extreme Rarity (Priority 2)
**Issue**: Only 12 K4 examples in 131,580 training snapshots (0.01%)
**Impact**: Models never learned K4 patterns
**Fix**: Retrain with combined dataset or SMOTE

### ✅ RESOLVED: First-Breach Classification
**Result**: K4 recall 0% → 66.7%, K3 recall 11% → 75-79%

### ✅ RESOLVED: BBW Mandatory Criterion
**Result**: Mean BBW 119.8% → 26.2% (4.6× better)

---

## Documentation

### Start Here
1. **[QUICKSTART.md](QUICKSTART.md)** - Immediate actions & Priority 1 fix
2. **[CLAUDE.md](CLAUDE.md)** - Complete system architecture & commands

### Historical Context (in docs/)
- **TRAINING_VALIDATION_COMPARISON_20251103.md** - Root cause analysis
- **MODEL_IMPROVEMENT_REPORT.md** - First-breach success story
- **VALIDATION_REPORT_BBW_FIX_20251103.md** - BBW fix results
- **ACTION_PLAN_FEATURE_FIXES.md** - Feature calculation bugs
- **BBW_FIX_VERIFICATION.md** - BBW mandatory criterion
- **CONSOLIDATION_CRITERIA_FIX_SUMMARY.md** - Criteria evolution
- **PROJECT_COMPLETE.md** - First-breach implementation
- **README.md** - AIv4 original readme

---

## Testing

```bash
# Run all tests
pytest

# Run specific modules
pytest tests/pattern_detection/
pytest tests/shared/

# Verify imports
python test_imports.py
# Expected: 16/18 imports successful (88.9%)
```

**Known Import Issues**:
- `pattern_detection.state_machine` - Internal import path issue (fixable)
- `src.pipeline` - Syntax error in __init__.py (fixable)
- These do NOT affect production scripts

---

## Current Model Performance

**Training Data**: 131,580 snapshots (first-breach labeled)
**Models**: `output/models/*_extreme_20251101_201450.pkl` (from AIv4)

| Metric | Before First-Breach | After First-Breach |
|--------|---------------------|-------------------|
| K4 Detection | 0% | 66.7% |
| K3 Detection | 0-11% | 75-79% |
| K3+K4 Combined | ~5% | 75-80% |
| EV Correlation | -0.14 | +0.20-0.35 |

**Validation Status**: PENDING FEATURE MATCH FIX (Priority 1)

---

## Next Steps

1. **Fix feature mismatch** (30-45 min):
   ```bash
   python extract_features_historical_enhanced.py
   python predict_historical_enhanced.py
   python generate_validation_report_enhanced.py
   ```

2. **Review results**: Check K4 recall, EV correlation

3. **If K4 still fails**: Retrain with combined dataset (Priority 2)

4. **Deploy**: Move to production after validation succeeds

---

## Support

**Quick Reference**: See [QUICKSTART.md](QUICKSTART.md)
**Architecture**: See [CLAUDE.md](CLAUDE.md)
**Troubleshooting**: See docs/TRAINING_VALIDATION_COMPARISON_20251103.md

---

**AIv5 - Clean, Focused, Production-Ready**
