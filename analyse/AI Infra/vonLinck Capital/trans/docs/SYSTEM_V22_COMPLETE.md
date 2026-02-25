# TRANS V22 "Clean Slate" - Complete System Architecture

## Executive Summary

**TRANS** (Temporal Sequence Architecture) is a production-ready machine learning system for detecting consolidation patterns in European micro/small-cap stocks that are likely to produce significant upward moves (+5R or greater).

The system has evolved through 22 versions, culminating in the **V22 "Clean Slate"** release (January 2026) which fundamentally restructured data quality, eliminating a critical data leakage bug that inflated apparent pattern counts by 20x.

### Key Metrics (V22)

| Metric | Value | Significance |
|--------|-------|--------------|
| **Val Precision @ Top 15%** | 26.7% | Win rate on highest-confidence signals |
| **Test Precision @ Top 15%** | 31.4% | Out-of-sample validation |
| **Unique Patterns** | 2,416 | Down from 81,512 (97% were duplicates) |
| **Pattern Overlap** | 0% | Previously 71% had 1-day gaps |
| **Target Base Rate** | 25.3% (val) | Clean data has higher quality signals |

---

## Table of Contents

1. [System Philosophy](#1-system-philosophy)
2. [Data Pipeline Architecture](#2-data-pipeline-architecture)
3. [Pattern Detection Engine](#3-pattern-detection-engine)
4. [Feature Engineering](#4-feature-engineering)
5. [Data Quality Filters (Operation Clean Slate)](#5-data-quality-filters-operation-clean-slate)
6. [Neural Network Architecture](#6-neural-network-architecture)
7. [Training Pipeline](#7-training-pipeline)
8. [Inference & Evaluation](#8-inference--evaluation)
9. [Historical Bug Fixes](#9-historical-bug-fixes)
10. [Production Deployment](#10-production-deployment)

---

## 1. System Philosophy

### 1.1 Core Hypothesis

Consolidation patterns in micro/small-cap stocks exhibit predictable temporal signatures before significant moves. By encoding 20 days of price action into a learnable representation, we can identify which consolidations will break out profitably versus fail.

### 1.2 Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Temporal Integrity** | No future data leakage at any stage |
| **Quality Over Quantity** | 2,416 unique events > 81,512 overlapping windows |
| **Precision Focus** | Optimize for win rate on traded signals, not global accuracy |
| **Micro-Cap Specialization** | System tuned for $50M-$2B market cap stocks |
| **Path Dependence** | Outcomes depend on the journey, not just endpoints |

### 1.3 What We Predict

**3-Class Classification System (V17 Labeling)**

| Class | Name | Definition | Strategic Value |
|-------|------|------------|-----------------|
| 0 | **Danger** | Price hits -2R (stop loss) first | -1.0 |
| 1 | **Noise** | Neither target nor stop hit in 100 days | -0.1 |
| 2 | **Target** | Price hits +5R (profit target) first | +5.0 |

**Expected Value Calculation:**
```
EV = P(Danger) × (-1.0) + P(Noise) × (-0.1) + P(Target) × (+5.0)
```

---

## 2. Data Pipeline Architecture

### 2.1 Two-Registry System

The system uses a **two-registry architecture** to guarantee temporal integrity:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  OHLCV + adj_close                                                       │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────────────────────────────┐                               │
│  │  00_detect_patterns.py               │                               │
│  │  ─────────────────────               │                               │
│  │  • State machine detection           │                               │
│  │  • NO labeling (outcome_class=NULL)  │                               │
│  │  • Outputs: candidate_patterns.parquet│                               │
│  └──────────────────────────────────────┘                               │
│        │                                                                 │
│        │ (Wait 100+ days for outcome window)                            │
│        ▼                                                                 │
│  ┌──────────────────────────────────────┐                               │
│  │  00b_label_outcomes.py               │                               │
│  │  ─────────────────────               │                               │
│  │  • Path-dependent labeling (V17)     │                               │
│  │  • Only labels patterns with         │                               │
│  │    end_date + 100 days <= today      │                               │
│  │  • Outputs: labeled_patterns.parquet │                               │
│  └──────────────────────────────────────┘                               │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────────────────────────────┐                               │
│  │  01_generate_sequences.py            │  ◄── Operation Clean Slate   │
│  │  ─────────────────────               │                               │
│  │  • Physics Filter (market cap, etc.) │                               │
│  │  • NMS Filter (de-duplication)       │                               │
│  │  • 14 features × 20 timesteps        │                               │
│  │  • Outputs: sequences.h5, metadata   │                               │
│  └──────────────────────────────────────┘                               │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────────────────────────────┐                               │
│  │  02_train_temporal.py                │                               │
│  │  ─────────────────────               │                               │
│  │  • Temporal split (date-based)       │                               │
│  │  • Z-score normalization (0-7, 12-13)│                               │
│  │  • Robust scaling (8-11)             │                               │
│  │  • Asymmetric loss training          │                               │
│  │  • Outputs: best_model.pt            │                               │
│  └──────────────────────────────────────┘                               │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────────────────────────────┐                               │
│  │  03_predict_temporal.py              │                               │
│  │  ─────────────────────               │                               │
│  │  • Load model + scaling params       │                               │
│  │  • Generate class probabilities      │                               │
│  │  • Calculate Expected Value          │                               │
│  └──────────────────────────────────────┘                               │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────────────────────────────┐                               │
│  │  evaluate_trading_performance.py     │                               │
│  │  ─────────────────────               │                               │
│  │  • Precision @ Top K%                │                               │
│  │  • EV threshold analysis             │                               │
│  │  • Drift monitoring                  │                               │
│  └──────────────────────────────────────┘                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Why Two Registries?

**Problem (Pre-V22):** Labeling patterns at detection time creates look-ahead bias. The model learns from labels that used future price data.

**Solution:** Split detection from labeling:
- **Candidate Registry:** Patterns detected in real-time, no labels
- **Outcome Registry:** Labels applied only after 100-day outcome window closes

---

## 3. Pattern Detection Engine

### 3.1 State Machine Architecture

Patterns progress through a deterministic state machine:

```
┌─────────┐    10 days     ┌─────────┐    Variable    ┌───────────┐
│  NONE   │ ──────────────►│QUALIFYING│ ─────────────►│  ACTIVE   │
└─────────┘   (conditions  └─────────┘   (conditions  └───────────┘
              met daily)                  still met)        │
                                                            │
                                              ┌─────────────┴─────────────┐
                                              ▼                           ▼
                                        ┌───────────┐              ┌──────────┐
                                        │ COMPLETED │              │  FAILED  │
                                        │ (Breakout)│              │(Breakdown)│
                                        └───────────┘              └──────────┘
```

### 3.2 Qualification Criteria

A pattern enters QUALIFYING state when ALL conditions are met for 10 consecutive days:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **BBW (Bollinger Band Width)** | < 30th percentile | Volatility contraction |
| **ADX (Trend Strength)** | < 32 | Low trending, coiling energy |
| **Volume** | < 35% of 20-day average | Decreasing participation |
| **Daily Range** | < 65% of 20-day average | Price compression |

### 3.3 Microstructure Filters (V17)

Additional filters from `sleeper_scanner_v17.py`:

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| **Liquidity Gate** | $50k daily volume | Ensure tradeable |
| **Ghost Detection** | <20% zero-volume days | Avoid illiquid stocks |
| **Thin Stock Flag** | Spread analysis | Execution cost awareness |

---

## 4. Feature Engineering

### 4.1 The 14 Temporal Features

Each pattern generates a tensor of shape `(20 timesteps, 14 features)`:

```python
# Feature Index Map (CRITICAL - memorize these)

# Market Data (4 features) - Relativized to Day 0 Close
# Formula: (Price_t / Price_0) - 1
[0] open      # Intraday open relative to pattern start
[1] high      # Intraday high relative to pattern start
[2] low       # Intraday low relative to pattern start
[3] close     # Close relative to pattern start

# Volume (1 feature) - Log Ratio to Day 0
# Formula: log(Volume_t / Volume_0)
[4] volume    # Range: typically [-3, +5]
              # Interpretation: 0 = same, +0.69 = doubled, -0.69 = halved

# Technical Indicators (3 features) - Already Bounded
[5] bbw_20          # Bollinger Band Width (20-period)
[6] adx             # Average Directional Index [0-100]
[7] volume_ratio_20 # Volume / 20-day SMA

# Composite Scores (4 features) - Global Robust Scaling Applied
# Formula: (value - median) / IQR (applied during training)
[8] vol_dryup_ratio  # Volume dryup indicator (< 0.3 = imminent move)
[9] var_score        # Volatility-adjusted return score
[10] nes_score       # Normalized energy score
[11] lpf_score       # Low-pass filtered momentum

# Boundaries (2 features) - Relativized to Day 0 Close
[12] upper_boundary  # Pattern ceiling relative to start
[13] lower_boundary  # Pattern floor relative to start
```

### 4.2 Normalization Strategy (V22)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEATURE NORMALIZATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  RAW SEQUENCES (14 features × 20 timesteps)                             │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Z-SCORE NORMALIZATION (Features 0-7, 12-13)                     │   │
│  │  ────────────────────────────────────────────                    │   │
│  │  • Computed on TRAINING set only                                 │   │
│  │  • Formula: (X - mean) / std                                     │   │
│  │  • Applied to val/test using TRAIN params                        │   │
│  │  • Saved to: norm_params_*.json                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│        │                                                                 │
│        ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  ROBUST SCALING (Features 8-11 only)                             │   │
│  │  ───────────────────────────────────                             │   │
│  │  • Computed on TRAINING set only                                 │   │
│  │  • Formula: (X - median) / IQR                                   │   │
│  │  • More robust to outliers than z-score                          │   │
│  │  • Saved to: robust_scaling_params_*.json                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│        │                                                                 │
│        ▼                                                                 │
│  NORMALIZED SEQUENCES (ready for model)                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Why Robust Scaling for Composite Features?

**Problem (Pre-V22):** Composite features (8-11) were calculated with per-window whitening, destroying absolute level information. A `vol_dryup_ratio` of 0.2 (very dry) looked the same as 1.5 (normal) after per-window normalization.

**Solution (V22):**
1. Save RAW composite values during sequence generation
2. Apply GLOBAL robust scaling during training
3. Preserves absolute levels: truly "dried up" volume now has negative scaled values

---

## 5. Data Quality Filters (Operation Clean Slate)

### 5.1 The Problem Discovered

Analysis revealed a critical data quality issue:

| Finding | Impact |
|---------|--------|
| 71% of patterns started 1 day apart | 95% data overlap between consecutive patterns |
| 81,512 "patterns" were really ~12k events | 7x artificial inflation |
| Train/val/test shared near-identical sequences | Severe data leakage |
| Large/Mega caps had 0% target rate | Unpredictable patterns polluting data |

### 5.2 NMS Filter (Protocol "Highlander")

**Principle:** *There can be only one* sequence per consolidation event.

**Algorithm:**
```python
def apply_nms_filter(patterns_df, overlap_days=10, selection_col='box_width'):
    """
    Non-Maximum Suppression for overlapping patterns.

    1. Group patterns by ticker
    2. Sort by start_date
    3. Calculate gap to previous pattern
    4. Assign cluster IDs (new cluster when gap > overlap_days)
    5. Keep only the pattern with LOWEST box_width per cluster
       (tightest consolidation = best representative)
    """
```

**Results:**
- Input: 81,512 patterns
- Output: 4,094 patterns
- Reduction: 95%

### 5.3 Physics Filter

**Principle:** Remove patterns that are physically impossible to trade profitably.

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| **Market Cap** | Nano, Micro, Small only | Large/Mega have 0% target rate |
| **Pattern Width** | ≥ 2% | Slippage eats edge on narrow patterns |
| **Dollar Volume** | ≥ $50k/day | Need liquidity to execute |

**Results:**
- Further reduces 4,094 → 2,416 patterns
- All remaining patterns are theoretically tradeable

### 5.4 Combined Filter Command

```bash
python pipeline/01_generate_sequences.py \
    --input output/detected_patterns.parquet \
    --output-dir output/sequences/eu_clean \
    --apply-nms \
    --nms-overlap-days 10 \
    --apply-physics-filter \
    --allowed-market-caps Nano,Micro,Small \
    --min-width-pct 0.02 \
    --min-dollar-volume 50000 \
    --skip-npy-export
```

---

## 6. Neural Network Architecture

### 6.1 HybridFeatureNetwork (V18)

The model combines three processing pathways:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HYBRID FEATURE NETWORK (V18)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: (batch, 20 timesteps, 14 features)                              │
│                    │                                                     │
│         ┌─────────┴─────────┐                                           │
│         ▼                   ▼                                           │
│  ┌─────────────┐    ┌─────────────┐                                     │
│  │  BRANCH A   │    │  BRANCH B   │                                     │
│  │  Temporal   │    │  GRN Context│                                     │
│  └─────────────┘    └─────────────┘                                     │
│         │                   │                                           │
│         ▼                   │                                           │
│  ┌─────────────────────┐    │                                           │
│  │      LSTM           │    │                                           │
│  │  ─────────────────  │    │                                           │
│  │  • 2 layers         │    │                                           │
│  │  • 32 hidden units  │    │                                           │
│  │  • Bidirectional    │    │                                           │
│  │  • Dropout: 0.2     │    │                                           │
│  └─────────────────────┘    │                                           │
│         │                   │                                           │
│         ▼                   │                                           │
│  ┌─────────────────────┐    │                                           │
│  │   Multi-Head        │    │                                           │
│  │   Attention         │    │                                           │
│  │  ─────────────────  │    │                                           │
│  │  • 8 heads          │    │                                           │
│  │  • Learn temporal   │    │                                           │
│  │    importance       │    │                                           │
│  └─────────────────────┘    │                                           │
│         │                   │                                           │
│         ▼                   │                                           │
│  ┌─────────────────────┐    │                                           │
│  │   CNN (Spatial)     │    │                                           │
│  │  ─────────────────  │    │                                           │
│  │  • Kernel sizes:    │    │                                           │
│  │    [3, 5]           │    │                                           │
│  │  • Local patterns   │    │                                           │
│  └─────────────────────┘    │                                           │
│         │                   │                                           │
│         ▼                   ▼                                           │
│  ┌─────────────────────────────────────┐                                │
│  │         FUSION LAYER                │                                │
│  │  ─────────────────────────────────  │                                │
│  │  • Concatenate temporal + context   │                                │
│  │  • GRN gating mechanism             │                                │
│  │  • 64-dim coil embedding            │                                │
│  └─────────────────────────────────────┘                                │
│                    │                                                     │
│                    ▼                                                     │
│  ┌─────────────────────────────────────┐                                │
│  │         OUTPUT HEAD                 │                                │
│  │  ─────────────────────────────────  │                                │
│  │  • Linear → 3 classes               │                                │
│  │  • Softmax for probabilities        │                                │
│  └─────────────────────────────────────┘                                │
│                    │                                                     │
│                    ▼                                                     │
│  Output: [P(Danger), P(Noise), P(Target)]                               │
│                                                                          │
│  Parameters: ~142,467                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Context Features (Branch B)

When `USE_GRN_CONTEXT = True`, 8 additional context features modulate the temporal signal:

| Feature | Description |
|---------|-------------|
| `float_turnover` | Trading activity relative to float |
| `trend_position` | Position within longer-term trend |
| `base_duration` | How long pattern has been forming |
| `relative_volume` | Current vs historical volume |
| `distance_to_high` | Distance from 52-week high |
| `log_float` | Log of shares available |
| `log_dollar_volume` | Log of daily dollar volume |
| `relative_strength_spy` | Performance vs S&P 500 |

### 6.3 Attention Mechanism

The attention layer learns which timesteps matter most:

- **Setup Phase (Days 0-2):** Pattern formation signals
- **Coil Phase (Days 15-19):** Pre-breakout compression
- **Attention Std:** Higher = model found discriminative patterns

---

## 7. Training Pipeline

### 7.1 Temporal Split (Enforced)

**No random splitting allowed** - it causes data leakage.

```python
# Strict temporal split based on pattern end date
train_cutoff = "2024-01-01"  # Everything before
val_cutoff = "2024-07-01"    # Between train and val cutoff

# Split assignment
train: pattern_end_date < train_cutoff
val:   train_cutoff <= pattern_end_date < val_cutoff
test:  pattern_end_date >= val_cutoff
```

### 7.2 Loss Function: Asymmetric Focal Loss

**Philosophy:** Penalize false positives (predicting Target when it's Danger) more heavily than false negatives.

```python
# Per-class configuration
CLASS_WEIGHTS = {0: 5.0, 1: 1.0, 2: 1.0}  # Heavy penalty on Danger
GAMMA_PER_CLASS = {0: 4.0, 1: 2.0, 2: 0.5}  # Focus on hard Danger examples
LABEL_SMOOTHING = 0.01  # Prevent overconfidence
```

### 7.3 Training Command

```bash
python pipeline/02_train_temporal.py \
    --sequences output/sequences/eu_clean/sequences_*.h5 \
    --metadata output/sequences/eu_clean/metadata_*.parquet \
    --epochs 100 \
    --batch-size 64 \
    --use-asl \
    --train-cutoff 2024-01-01 \
    --patience 15
```

### 7.4 Training Output Files

| File | Contents |
|------|----------|
| `best_model_*.pt` | Model weights + config |
| `norm_params_*.json` | Z-score parameters (mean, std) |
| `robust_scaling_params_*.json` | Robust scaling parameters (median, IQR) |
| `training_history_*.json` | Loss/accuracy per epoch |

---

## 8. Inference & Evaluation

### 8.1 Prediction Pipeline

```python
# Inference flow
1. Load model + norm_params + robust_scaling_params
2. Apply z-score normalization (features 0-7, 12-13)
3. Apply robust scaling (features 8-11)
4. Forward pass → class probabilities
5. Calculate EV = Σ(P(class) × Value(class))
6. Generate signal based on EV threshold
```

### 8.2 Primary Evaluation Metric

**Precision @ Top 15%** - The only metric that matters for live trading.

```
1. Sort all patterns by Predicted EV (descending)
2. Take top 15% of patterns
3. Count how many are actually Target (class 2)
4. Precision = Target hits / Total in top 15%
```

### 8.3 V22 Results

| Split | Patterns | Top 15% Precision | Base Rate | Lift |
|-------|----------|-------------------|-----------|------|
| **Val** | 304 | 26.7% | 25.3% | 1.05x |
| **Test** | 917 | 31.4% | ~21% | 1.5x |

### 8.4 Evaluation Command

```bash
python pipeline/evaluate_trading_performance.py \
    --model output/models/best_model_*.pt \
    --metadata output/sequences/eu_clean/metadata_*.parquet \
    --sequences-dir output/sequences/eu_clean \
    --split test \
    --train-cutoff 2024-01-01 \
    --val-cutoff 2024-07-01
```

---

## 9. Historical Bug Fixes

### 9.1 Complete Bug Fix Timeline

| Date | Bug | Severity | Fix |
|------|-----|----------|-----|
| **2026-01-10** | Pattern overlap (71% 1-day gaps) | CRITICAL | NMS + Physics filters |
| **2026-01-10** | Composite features unscaled | HIGH | Global robust scaling |
| **2026-01-09** | Eval split mismatch | MEDIUM | Date-based cutoffs |
| **2026-01-08** | Look-ahead in detection | CRITICAL | Two-Registry System |
| **2026-01-07** | torch.compile incompatible | LOW | Refactored forward() |
| **2026-01-06** | ADX warmup artifact | HIGH | 30-day warmup prefix |
| **2025-12-31** | Raw volume dominating | HIGH | Log-ratio normalization |
| **2025-12-17** | Raw close for labeling | HIGH | Use adj_close |
| **2025-12-15** | adj_close stripped | MEDIUM | Preserve in data loader |

### 9.2 Lessons Learned

1. **Always check for data leakage** - overlapping windows are subtle
2. **Validate feature distributions** - raw vs normalized can differ wildly
3. **Temporal splits are mandatory** - random splits hide problems
4. **Test on truly out-of-sample data** - val accuracy can be misleading

---

## 10. Production Deployment

### 10.1 File Structure

```
trans/
├── pipeline/
│   ├── 00_detect_patterns.py      # Candidate registry
│   ├── 00b_label_outcomes.py      # Outcome registry
│   ├── 01_generate_sequences.py   # Feature extraction + filters
│   ├── 02_train_temporal.py       # Model training
│   ├── 03_predict_temporal.py     # Inference
│   └── evaluate_trading_performance.py
├── models/
│   └── temporal_hybrid_v18.py     # Neural network
├── core/
│   ├── pattern_detector.py        # Sequence generation
│   ├── pattern_scanner.py         # Detection orchestrator
│   ├── sleeper_scanner_v17.py     # Microstructure filters
│   └── path_dependent_labeler.py  # V17 labeling
├── config/
│   ├── constants.py               # Strategic values, thresholds
│   └── context_features.py        # GRN feature definitions
├── output/
│   ├── models/                    # Trained models + params
│   ├── sequences/                 # Generated sequences
│   │   └── eu_clean/              # V22 clean data
│   └── evaluation/                # Evaluation reports
└── docs/
    └── SYSTEM_V22_COMPLETE.md     # This document
```

### 10.2 Standard Workflow

```bash
# 1. Detect patterns (real-time, no labels)
python pipeline/00_detect_patterns.py --tickers EU_TICKERS --start-date 2020-01-01

# 2. Label outcomes (run periodically, only labels old patterns)
python pipeline/00b_label_outcomes.py --input output/candidate_patterns.parquet

# 3. Generate clean sequences
python pipeline/01_generate_sequences.py \
    --input output/labeled_patterns.parquet \
    --apply-nms --apply-physics-filter \
    --output-dir output/sequences/eu_clean

# 4. Train model
python pipeline/02_train_temporal.py \
    --sequences output/sequences/eu_clean/sequences_*.h5 \
    --metadata output/sequences/eu_clean/metadata_*.parquet \
    --epochs 100 --train-cutoff 2024-01-01

# 5. Evaluate
python pipeline/evaluate_trading_performance.py \
    --split test --train-cutoff 2024-01-01 --val-cutoff 2024-07-01
```

### 10.3 Key Configuration

```python
# config/constants.py

# Strategic Values (V17)
STRATEGIC_VALUES = {0: -1.0, 1: -0.1, 2: 5.0}

# Class Weights (Precision Focus)
CLASS_WEIGHTS = {0: 5.0, 1: 1.0, 2: 1.0}
GAMMA_PER_CLASS = {0: 4.0, 1: 2.0, 2: 0.5}

# Feature Flags
USE_GRN_CONTEXT = True  # Enable context branch
INDICATOR_WARMUP_DAYS = 30  # ADX/BBW warmup

# Drift Monitoring
DRIFT_CRITICAL_FEATURES = ['vol_dryup_ratio', 'bbw']
```

---

## Appendix A: Quick Reference Commands

### Generate Clean Sequences
```bash
python pipeline/01_generate_sequences.py \
    --input output/detected_patterns.parquet \
    --output-dir output/sequences/eu_clean \
    --apply-nms --apply-physics-filter
```

### Train Model
```bash
python pipeline/02_train_temporal.py \
    --sequences output/sequences/eu_clean/sequences_*.h5 \
    --metadata output/sequences/eu_clean/metadata_*.parquet \
    --epochs 100 --train-cutoff 2024-01-01 --use-asl
```

### Evaluate
```bash
python pipeline/evaluate_trading_performance.py \
    --split val --train-cutoff 2024-01-01 --val-cutoff 2024-07-01
```

### Drift Check
```bash
python pipeline/evaluate_trading_performance.py --drift-only
```

---

## Appendix B: Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| V22 | 2026-01-10 | Operation Clean Slate (NMS, Physics, Robust Scaling) |
| V21 | 2026-01-10 | Raw composite features (skip per-window whitening) |
| V20 | 2026-01-09 | Balanced class weights |
| V19 | 2026-01-08 | Two-Registry System |
| V18 | 2026-01-06 | Context-Query Attention, ADX warmup fix |
| V17 | 2025-12-17 | Path-dependent labeling, adj_close |

---

*Document generated: 2026-01-10*
*System Version: V22 "Clean Slate"*
*Author: Claude Code*
