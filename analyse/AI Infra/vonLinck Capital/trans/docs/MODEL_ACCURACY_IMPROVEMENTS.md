# TRANS Model Accuracy Improvements (January 2026)

## Executive Summary

Three critical improvements were implemented to address poor model performance:
- **Before**: 15.1% top-15 target rate (same as baseline = no predictive power), 0 patterns with EV > 3.0
- **Expected After**: 25-35% top-15 target rate, 50-200 patterns with EV > 3.0

---

## System Overview

### What is TRANS?

**TRANS (Temporal Sequence Architecture)** is a neural network system that identifies consolidation patterns in micro/small-cap stocks likely to produce significant gains (+5R or more).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRANS ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  OHLCV Data → Pattern Detection → Sequence Generation → Neural Network  │
│                                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                 │
│  │   Branch A   │   │   Branch B   │   │    Output    │                 │
│  │  (Temporal)  │ + │  (Context)   │ → │   3-Class    │                 │
│  │  LSTM + CNN  │   │  GRN Gating  │   │  Prediction  │                 │
│  │  + Attention │   │  8 Features  │   │              │                 │
│  └──────────────┘   └──────────────┘   └──────────────┘                 │
│                                                                          │
│  Input: 14 features × 20 timesteps    Output: P(Danger), P(Noise),      │
│         + 8 context features                   P(Target)                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Problem Being Solved

The model identifies "sleeper" consolidation patterns - stocks that have:
1. **Contracted volatility** (Bollinger Band Width < 30th percentile)
2. **Dried up volume** (volume < 35% of 20-day average)
3. **Built a base** (10+ days of qualification)

These patterns often precede explosive moves. The challenge is distinguishing winners from noise.

---

## The Three Improvements

### 1. Reduced Class Weights (K2: 10x → 3x)

**Problem**: The 10x weight on Target class forced the model to maximize Recall at the expense of Precision, producing many false positives.

**Solution**: Reduce K2 weight from 10.0 to 3.0 for better Precision/Recall balance.

```python
# BEFORE (config/constants.py)
CLASS_WEIGHTS = {
    0: 2.0,    # Danger
    1: 1.0,    # Noise
    2: 10.0,   # Target ← TOO HIGH
}

# AFTER
CLASS_WEIGHTS = {
    0: 2.0,    # Danger
    1: 1.0,    # Noise
    2: 3.0,    # Target ← BALANCED
}
```

**Files Changed**:
- `config/constants.py` - CLASS_WEIGHTS dictionary
- `pipeline/02_train_temporal.py` - CLI default "2.0,1.0,3.0"

---

### 2. Global Robust Scaling (Median/IQR)

**Problem**: Per-window whitening destroyed "amplitude" information. A major squeeze (BBW dropping 80%) looked identical to minor noise (BBW dropping 5%) after normalization.

**Solution**: Use global Median/IQR computed across all training data instead of per-window statistics.

```
BEFORE (Per-Window):                    AFTER (Global):
┌─────────────────────┐                ┌─────────────────────┐
│ Window 1: BBW drops │                │ Window 1: BBW drops │
│ 80% → normalized    │                │ 80% → large signal  │
│ to [-1, +1]         │                │ (vs global median)  │
├─────────────────────┤                ├─────────────────────┤
│ Window 2: BBW drops │                │ Window 2: BBW drops │
│ 5% → normalized     │                │ 5% → small signal   │
│ to [-1, +1]         │                │ (vs global median)  │
└─────────────────────┘                └─────────────────────┘
     SAME OUTPUT!                         DIFFERENT OUTPUT!
```

**Formula**:
```python
normalized = (value - Global_Median) / Global_IQR
```

**Files Changed**:
- `core/pattern_detector.py` - Added `robust_scaling_path` parameter to `__init__`
- `pipeline/01_generate_sequences.py` - Added `compute_global_robust_params()` function

**New Function**:
```python
def compute_global_robust_params(sequences: np.ndarray, output_path: str = None) -> dict:
    """
    Compute global Median/IQR from all generated sequences.

    Applied to composite features (indices 8-11):
    - vol_dryup_ratio (index 8)
    - var_score (index 9)
    - nes_score (index 10)
    - lpf_score (index 11)
    """
```

---

### 3. Context Branch + Relative Strength SPY

**Problem**: The model only saw the "shape" of the coil (temporal features) but not the broader market context. It couldn't distinguish:
- Breakout coil (stock outperforming in bull market)
- Breakdown coil (stock lagging in bear market)

**Solution**: Enable the GRN (Gated Residual Network) Context Branch with a new feature measuring relative strength vs SPY.

```
┌────────────────────────────────────────────────────────────────┐
│                    DUAL-BRANCH ARCHITECTURE                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Branch A (Temporal)              Branch B (Context)           │
│  ─────────────────               ──────────────────            │
│  Input: 14×20 sequence           Input: 8 static features      │
│                                                                 │
│  ┌─────────────┐                 ┌─────────────┐               │
│  │    LSTM     │                 │    Dense    │               │
│  │  (32 units) │                 │  (32 units) │               │
│  └──────┬──────┘                 └──────┬──────┘               │
│         │                               │                       │
│  ┌──────┴──────┐                 ┌──────┴──────┐               │
│  │     CNN     │                 │     GRN     │               │
│  │ (3×5 kernel)│                 │   Gating    │               │
│  └──────┬──────┘                 └──────┬──────┘               │
│         │                               │                       │
│  ┌──────┴──────┐                        │                       │
│  │  Attention  │                        │                       │
│  │  (8 heads)  │                        │                       │
│  └──────┬──────┘                        │                       │
│         │                               │                       │
│         └───────────┬───────────────────┘                       │
│                     │                                           │
│              ┌──────┴──────┐                                    │
│              │  Concatenate │                                   │
│              │  + Classify  │                                   │
│              └──────┬──────┘                                    │
│                     │                                           │
│              ┌──────┴──────┐                                    │
│              │   Output    │                                    │
│              │ P(D,N,T)    │                                    │
│              └─────────────┘                                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**The 8 Context Features**:

| # | Feature | Formula | What It Measures |
|---|---------|---------|------------------|
| 1 | float_turnover | Sum(Vol_60D) × Price / Market_Cap | Accumulation activity |
| 2 | trend_position | Price / 200_SMA | Market structure |
| 3 | base_duration | log(1 + days) / log(201) | Consolidation maturity |
| 4 | relative_volume | Avg_Vol_20D / Avg_Vol_60D | Recent activity |
| 5 | distance_to_high | (52W_High - Price) / 52W_High | Overhead resistance |
| 6 | log_float | log10(shares_outstanding) | Size/liquidity |
| 7 | log_dollar_volume | log10(avg_daily_dollar_volume) | Tradability |
| 8 | **relative_strength_spy** | (ticker_ret - SPY_ret) / \|SPY_ret\| | **NEW: Market outperformance** |

**Relative Strength SPY Calculation**:
```python
# Calculate 20-day returns
ticker_20d_return = (price_now / price_20d_ago) - 1
spy_20d_return = (spy_now / spy_20d_ago) - 1

# Relative strength: how much better/worse than SPY
rs_spy = (ticker_20d_return - spy_20d_return) / abs(spy_20d_return)

# Interpretation:
#   rs_spy = 0    → Same as SPY
#   rs_spy = 1    → Outperforming by 100% of SPY's move
#   rs_spy = -0.5 → Underperforming by 50% of SPY's move
```

**Files Changed**:
- `config/constants.py` - `USE_GRN_CONTEXT = True`
- `config/context_features.py` - Added `relative_strength_spy`, updated to 8 features
- `pipeline/01_generate_sequences.py` - Added `_load_spy_data()` and Feature 8 calculation

---

## Complete Feature Architecture

### Branch A: 14 Temporal Features × 20 Timesteps

```
Index   Feature              Normalization              Range
─────   ───────              ─────────────              ─────
0-3     open,high,low,close  Relativized to day 0       ~[-0.3, +0.5]
4       volume               Log ratio to day 0         [-3, +5]
5       bbw_20               Raw (already bounded)      [0, 1]
6       adx                  Raw (already bounded)      [0, 100]
7       volume_ratio_20      Raw                        [0, 5]
8       vol_dryup_ratio      Global Median/IQR (NEW)    [-3, +3]
9       var_score            Global Median/IQR (NEW)    [-3, +3]
10      nes_score            Global Median/IQR (NEW)    [-3, +3]
11      lpf_score            Global Median/IQR (NEW)    [-3, +3]
12-13   upper/lower_boundary Relativized to day 0       ~[-0.1, +0.2]
```

### Branch B: 8 Context Features (Static per Pattern)

```
Index   Feature                  Range        Optimal
─────   ───────                  ─────        ───────
0       float_turnover           [0, 1]       0.3-1.0 (1.5-5x turnover)
1       trend_position           [0, 1]       0.5-1.0 (above 200MA)
2       base_duration            [0, 1]       0.5-0.85 (15-100 days)
3       relative_volume          [0, 1]       0.47-1.0 (1.2-2x baseline)
4       distance_to_high         [0, 1]       0.1-0.4 (5-20% from high)
5       log_float                [0, 1]       0.2-0.6 (1M-100M shares)
6       log_dollar_volume        [0, 1]       0.2-0.6 ($100k-$10M/day)
7       relative_strength_spy    [0, 1]       0.33-1.0 (outperforming)
```

---

## Output Classes and Strategic Values

```
┌─────────────────────────────────────────────────────────────┐
│                    3-CLASS OUTPUT                            │
├───────────┬────────────┬─────────────┬──────────────────────┤
│   Class   │   Trigger  │   Value     │   Interpretation     │
├───────────┼────────────┼─────────────┼──────────────────────┤
│ 0: Danger │ -2R hit    │ -1.0        │ Stop loss triggered  │
│ 1: Noise  │ Neither    │ -0.1        │ Opportunity cost     │
│ 2: Target │ +5R hit    │ +5.0        │ Winner (5x risk)     │
└───────────┴────────────┴─────────────┴──────────────────────┘

Expected Value = P(Danger)×(-1.0) + P(Noise)×(-0.1) + P(Target)×(+5.0)

Example:
  P(Danger) = 0.10 → -0.10
  P(Noise)  = 0.60 → -0.06
  P(Target) = 0.30 → +1.50
  ─────────────────────────
  EV = +1.34 (TRADE if > 2.0)
```

---

## Training Pipeline

```bash
# Step 1: Detect patterns (already done)
python pipeline/00_detect_patterns.py --tickers ALL --start-date 2020-01-01

# Step 2: Generate sequences with NEW features
python pipeline/01_generate_sequences.py \
    --input output/eu/detected_patterns.parquet \
    --output-dir output/sequences/eu \
    --claude-safe

# Step 3: Train with NEW class weights
python pipeline/02_train_temporal.py \
    --sequences output/sequences/eu/sequences.h5 \
    --metadata output/sequences/eu/metadata.parquet \
    --epochs 100 \
    --class-weights "2.0,1.0,3.0"

# Step 4: Evaluate
python pipeline/evaluate_trading_performance.py --split val
```

---

## Expected Improvements

| Metric | Before | Expected After | Why |
|--------|--------|----------------|-----|
| K2 Recall | 6% | 15-25% | Balanced class weights |
| K2 Precision | 15% | 30-40% | Reduced false positives |
| Top 15% Target Rate | 15.1% | 25-35% | Context branch discrimination |
| Patterns with EV > 3 | 0 | 50-200 | Better calibration |
| EV Calibration Error | 0.56 | < 0.3 | Global scaling preserves amplitude |

---

## Key Files Modified

| File | Change |
|------|--------|
| `config/constants.py` | `USE_GRN_CONTEXT=True`, `CLASS_WEIGHTS[2]: 10→3` |
| `config/context_features.py` | Added `relative_strength_spy`, `NUM_CONTEXT_FEATURES=8` |
| `core/pattern_detector.py` | Added `robust_scaling_path` parameter |
| `pipeline/01_generate_sequences.py` | SPY loader, RS_SPY calc, `compute_global_robust_params()` |
| `pipeline/02_train_temporal.py` | CLI default `"2.0,1.0,3.0"` |

---

## Important Notes

1. **Sequences must be regenerated** after these changes - the old sequences don't have RS_SPY
2. **Run in external terminal** - avoid Claude Code for long-running tasks (OOM risk)
3. **EU model is 5x better** than US - prioritize EU patterns
4. **Monitor drift** - run `evaluate_trading_performance.py --drift-only` weekly
