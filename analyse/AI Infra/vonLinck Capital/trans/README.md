# TRAnS - Temporal Retail Analysis System

## Production-Ready V22 | Micro/Small-Cap Consolidation Breakouts

This directory contains a **fully self-contained** temporal sequence-based pattern detection system optimized for retail traders ($10k-$100k accounts).

**Status**: ✅ Production V22 | Dynamic Labeling Window | EOD Execution | 383 Tests Passing

**Core Paradigm**: Static Snapshots → Temporal Sequences

**Recent Cleanup**: Removed 67 obsolete files (experimental scripts, duplicate utilities, obsolete docs) to improve maintainability and AI context efficiency while preserving 100% production functionality.

## Installation (Standalone)

```bash
# 1. Install dependencies
cd trans
pip install -r requirements.txt

# 2. Create directories
mkdir -p data/raw output/{sequences,models,predictions} logs
```

**See Quick Start section below for setup, or [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) for complete documentation.**

---

## V22 Features (Jan 2026)

### Verified Implementations

| Feature | Status | Location |
|---------|--------|----------|
| **Adaptive Detection** | ✅ Verified | `--tightness-zscore` flag, `close > sma_50 > sma_200` |
| **EOD Execution** | ✅ Verified | BUY STOP LIMIT orders, 10 AM pacing disabled |
| **Float Rotation Filter** | ✅ Verified | `--min-float-turnover` filter applied |
| **Dynamic Outcome Window** | ✅ Verified | V22: 10-60 days based on volatility |

### Dynamic Labeling Window (V22)

```python
# High volatility = faster resolution = shorter window
# Low volatility = slower resolution = longer window
volatility_proxy = risk_width_pct or box_width
dynamic_window = clip(1 / volatility_proxy, MIN=10, MAX=60)

# Examples:
#   10% vol → 10 days (fast movers)
#   5% vol  → 20 days (typical)
#   2% vol  → 50 days (tight coils)
```

### EOD Execution (No Intraday Monitoring)

```python
# Order Type: BUY STOP LIMIT
# - Stop Price: Upper_Boundary + $0.01
# - Limit Price: Stop + 0.5R (caps slippage)
# - If Open > Limit: DO NOT CHASE (cancel order)
```

### Adaptive Detection

```bash
# Use Z-score instead of static percentile
python pipeline/00_detect_patterns.py \
    --tightness-zscore -1.0 \
    --min-float-turnover 0.10
```

---

## Quick Start

### Pattern Detection & Training Pipeline (V22)

```bash
# 0. Detect patterns with adaptive thresholds
python pipeline/00_detect_patterns.py \
    --tickers AAPL,MSFT,GOOGL \
    --start-date 2020-01-01 \
    --tightness-zscore -1.0 \
    --min-float-turnover 0.10

# 0b. Label outcomes with dynamic window (V22)
python pipeline/00b_label_outcomes.py \
    --input output/candidate_patterns.parquet

# 1. Generate sequences with NMS and physics filter
python pipeline/01_generate_sequences.py \
    --input output/labeled_patterns.parquet \
    --apply-nms \
    --apply-physics-filter \
    --mode training

# 2. Train with Coil-Aware Focal Loss
python pipeline/02_train_temporal.py \
    --use-coil-focal \
    --epochs 100

# 3. Generate risk-based orders (EOD execution)
python pipeline/03_predict_temporal.py \
    --model output/best_model.pt \
    --risk-unit 250 \
    --max-capital 5000

# Run all tests (383 tests)
python -m pytest tests/ -v
```

### Full Historical Backtesting (NEW!)

**CRITICAL FEATURE**: Scans ENTIRE historical dataset for ALL patterns (not just recent ones)

```bash
# Run 4-year comprehensive backtest
python run_backtest.py --start-date 2020-01-01 --end-date 2024-01-01 --tickers AAPL,MSFT,GOOGL

# Run from ticker file with full history scan
python run_backtest.py --start-date 2019-01-01 --tickers-file universe.txt --output output/my_backtest

# Quick example (see examples/run_4year_backtest_example.py)
python examples/run_4year_backtest_example.py
```

**What Makes This Special:**
- ✅ Scans **ENTIRE** date range, not just recent patterns
- ✅ Finds **ALL** consolidation patterns across full history
- ✅ Labels completed patterns with actual outcomes (K0-K5)
- ✅ Generates comprehensive performance metrics
- ✅ Walk-forward validation (no look-ahead bias)
- ✅ Saves all historical patterns for analysis
```

## Architecture

### Input Format
- **Shape**: (batch_size, 20 timesteps, 14 features)
- **Timesteps 1-10**: Qualification phase (pattern emergence)
- **Timestep 10**: Activation point (boundaries established)
- **Timesteps 11-20**: Validation period (stability verification)

### 14 Features Per Timestep
```python
features = [
    'upper_boundary',      # Pattern structural boundary
    'lower_boundary',      # Pattern structural boundary
    'bbw',                 # Bollinger Band Width
    'adx',                 # Average Directional Index
    'volume_ratio',        # Volume vs baseline
    'range_ratio',         # Daily range vs baseline
    'cci',                 # Commodity Channel Index
    'rsi',                 # Relative Strength Index
    'price_position',      # (price - lower) / (upper - lower)
    'days_in_pattern',     # Pattern age
    'boundary_range_pct',  # (upper - lower) / lower
    'close',               # Closing price
    'volume',              # Raw volume
    'atr'                  # Average True Range
]
```

## Key Components from AIv7

### ✅ Refactored & Improved
1. **Pattern Detection** (`core/aiv7_components/consolidation_tracker_refactored.py`)
   - **REFACTORED**: Clean modular version (350 lines vs 1,092 lines monolith)
   - ConsolidationTracker state machine
   - 10-day qualification logic
   - Boundary detection
   - Uses modular components: PatternState, FeatureExtractor, StateManager, BoundaryManager

2. **Asymmetric Loss** (`models/asymmetric_loss.py`)
   - Handles K4 class imbalance
   - γ_easy=4.0 for K0-K2, γ_hard=1.0 for K3-K5

3. **Labeling System** (`core/pattern_labeler.py`)
   - K0-K5 outcome classes
   - Strategic value assignments
   - Expected Value calculation

### 🔄 Adapted Components
1. **VAE Concepts** → Temporal VAE (optional pre-training)
2. **Data Loader** → Sequence Generator
3. **Feature Engineering** → 14 raw features (vs 92 engineered)

## Directory Structure

```
trans/
├── README.md                    # This file
├── run_backtest.py              # CLI for full historical backtesting (NEW)
├── core/                        # Core pattern detection (from AIv7)
│   ├── pattern_detector.py     # ConsolidationTracker wrapper
│   ├── pattern_scanner.py      # High-performance pattern scanner
│   └── aiv7_components/        # Extracted AIv7 components
│       ├── consolidation_tracker_refactored.py  # Main implementation (350 lines)
│       ├── consolidation/      # Modular components
│       ├── data_loader.py
│       └── indicators.py
├── models/                      # Neural architectures
│   ├── temporal_hybrid.py      # LSTM + CNN + Attention
│   └── asymmetric_loss.py      # ASL from AIv7 (PyTorch)
├── backtesting/                 # Full historical backtesting (NEW)
│   ├── temporal_backtester.py  # Comprehensive backtest engine
│   └── performance_metrics.py  # Performance tracking & metrics
├── pipeline/                    # Training pipeline scripts
│   ├── 00_detect_patterns.py   # Pattern scanner (adaptive Z-score)
│   ├── 00b_label_outcomes.py   # V22 dynamic window labeling
│   ├── 01_generate_sequences.py # NMS + physics filter
│   ├── 02_train_temporal.py    # Coil-Aware Focal Loss
│   ├── 03_predict_temporal.py  # EOD BUY STOP LIMIT orders
│   └── 04_evaluate.py
├── examples/                    # Example scripts (NEW)
│   └── run_4year_backtest_example.py
├── tests/                       # Test suite (383 tests)
│   ├── test_lookahead.py       # Look-ahead bias prevention (11 tests)
│   ├── test_temporal_integrity.py
│   ├── test_temporal_split.py
│   └── ...
├── utils/                       # Utilities
│   ├── data_utils.py           # Data loading helpers
│   ├── metrics.py              # Evaluation metrics
│   └── visualization.py        # Training plots, UMAP
├── docs/                        # Documentation
│   └── BACKTEST_PARAMETER_OPTIMIZATION.md  # Parameter tuning guide
└── output/                      # Results (gitignored)
    ├── sequences/
    ├── models/
    ├── predictions/
    └── backtest/                # Backtest results (NEW)
```

## Performance Targets

| Metric | AIv7 Baseline | Temporal Target |
|--------|---------------|-----------------|
| K4 Recall | 30-50% | 40-60% |
| K3 Recall | 80-90% | 85-95% |
| EV Correlation | >0.30 | >0.40 |
| Training Time | 60-80 min | 40-60 min |
| Feature Engineering | 30-45 min | 5-10 min |

## Key Advantages

1. **Temporal Context**: Full sequence captures pattern evolution
2. **Feature Efficiency**: 14 features (vs 92) with model learning dynamics
3. **Natural Augmentation**: Sliding windows create multiple training samples
4. **Discovery Capability**: CNN + Attention find hidden patterns
5. **ASL Integration**: Handles K4 rarity without SMOTE
6. **Modular Architecture**: Clean 350-line orchestrator with specialized components

## Dependencies

```bash
pip install torch>=2.0.0
pip install pandas numpy
pip install scikit-learn
pip install matplotlib seaborn
pip install tqdm
```

## Credits

- Base pattern detection: AIv7 ConsolidationTracker
- Asymmetric Loss: AIv7 SOTA upgrade
- Temporal architecture: New paradigm shift
