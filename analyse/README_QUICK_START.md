# Quick Start Guide - Phase 1 Feature Extraction

## ⚡ Fastest Way to Get Started

### 1. Run Test (1 minute)
```batch
cd analyse
test_pipeline.bat
```
**Validates:** Feature extraction works on 10 patterns

---

### 2. Run Complete Analysis (30-90 minutes) ⭐ RECOMMENDED
```batch
cd analyse
run_complete_analysis.bat
```

**This single command does everything:**
- ✅ Processes ALL 3,548+ tickers from GCS (parallel)
- ✅ Detects 10,000-15,000+ consolidation patterns
- ✅ Extracts 47 features per pattern (35 volume + 12 others)
- ✅ Prepares ML-ready training data (70/15/15 split)
- ✅ Generates comprehensive analysis report
- ✅ 5-10x faster than sequential (parallel processing)

**What you get:**
```
output/
├── patterns_parallel_*.parquet              # All detected patterns
├── complete_analysis_report_*.txt           # ⭐ READ THIS FIRST

AI Infra/data/
├── features/parallel_pattern_features_*.parquet   # Feature-enriched data
└── raw/
    ├── production_training_data.parquet     # ML-ready (use for training)
    └── production_training_metadata.json    # Dataset statistics
```

---

## What the Report Contains

The `complete_analysis_report_*.txt` gives you:

✅ **Dataset size** (patterns, tickers, date range)
✅ **Feature counts** (volume, pattern, trend)
✅ **Data quality checks** (NaN, infinities, duplicates)
✅ **Class distribution** (K0-K5 outcomes)
✅ **Binary target stats** (K3_K4 positive class rate)
✅ **Top tickers** by pattern count
✅ **Next steps** recommendations for ML training

---

## Next Steps After Complete Analysis

### 1. Review the Report
```batch
# Open the report file (in output/ folder)
notepad output\complete_analysis_report_*.txt
```

Look for:
- ✅ `[OK]` markers = All good
- ⚠️ `[WARNING]` markers = Review before training
- 📊 Check positive class rate (should be ~10%)

### 2. Train the Model
```batch
cd "AI Infra\hybrid_model"
python integrated_self_training.py train --features volume --target K3_K4_binary
```

### 3. Validate Model Performance
```batch
python integrated_self_training.py validate --test-size 0.3
```

### 4. Run Backtest
```batch
python automated_backtesting.py --start-date 2022-01-01 --end-date 2024-01-01
```

---

## Alternative Options

### Parallel Only (No Report)
```batch
run_full_pipeline_parallel.bat
```
30-90 min, no report generation

### Sequential (Slower)
```batch
run_full_pipeline.bat
```
4-8 hours, processes one ticker at a time

### Test on Subset
```batch
python extract_all_features_parallel.py --limit 100 --workers 10
```
5-10 min, processes 100 tickers only

---

## Requirements

- ✅ GCS credentials (`gcs-key.json` in analyse/ folder)
- ✅ Virtual environment activated
- ✅ Internet connection (for GCS access)
- ✅ ~500MB disk space for output
- ✅ ~1-2GB RAM (for parallel processing)

---

## Troubleshooting

### "GCS credentials not found"
**Fix:** Ensure `gcs-key.json` exists in the `analyse/` directory

### "No patterns detected"
**Fix:** Check GCS bucket has data:
```python
from core import get_data_loader
loader = get_data_loader()
print(f"Available tickers: {len(loader.list_available_tickers())}")
```

### "Out of memory"
**Fix:** Reduce workers:
```batch
python extract_all_features_parallel.py --workers 5
```

### "Too slow"
**Fix:** Increase workers (if you have cores available):
```batch
python extract_all_features_parallel.py --workers 20
```

---

## Performance Comparison

| Method | Runtime | Patterns | Workers | Report |
|--------|---------|----------|---------|--------|
| **Complete Analysis (Parallel)** ⭐ | 30-90 min | 10K-15K | Auto (10-20) | ✅ Yes |
| Parallel (No Report) | 30-90 min | 10K-15K | Auto (10-20) | ❌ No |
| Sequential | 4-8 hours | 10K-15K | 1 | ❌ No |
| Test | 1 min | 10 | 1 | ❌ No |

---

## File Locations

All scripts are in: `C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\`

**Test:**
- `test_pipeline.bat`

**Production:**
- `run_complete_analysis.bat` ⭐ RECOMMENDED
- `run_full_pipeline_parallel.bat`
- `run_full_pipeline.bat`

**Manual:**
- `extract_all_features_parallel.py` (Python script)
- `extract_all_features.py` (Sequential version)

---

## Expected Results

After successful run:

**Patterns:** 10,000 - 15,000+
**Tickers:** 3,548+
**Features:** 47 (35 volume + 5 pattern + 5 trend + 2 metadata)
**Positive Class (K3/K4):** ~1,000 examples (~10%)
**Training Data:** 70% train, 15% val, 15% test
**Runtime:** 30-90 minutes (parallel)

---

**Last Updated:** October 7, 2025
**Status:** ✅ Phase 1 Complete - Ready for Phase 2 (Model Training)
