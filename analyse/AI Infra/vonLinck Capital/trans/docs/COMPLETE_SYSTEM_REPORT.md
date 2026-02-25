# Temporal Consolidation Pattern Detection System
## Complete System Report - EU & US Markets

**Version:** V18 Ensemble
**Date:** January 2026
**Status:** EU Production-Ready | US Requires Further Development

---

## Executive Summary

This document presents the complete Temporal Consolidation Pattern Detection System, designed to identify micro/small-cap stocks poised for significant upward moves from consolidation phases. The system uses deep learning (LSTM + CNN + Attention) to analyze 20-day temporal sequences and predict breakout outcomes.

### Key Results

| Market | EV Correlation | Monotonicity | Top 15% Target | Recommendation |
|--------|----------------|--------------|----------------|----------------|
| **EU** | **+0.088** | **YES** | 31.4% | Production Ready |
| US | -0.033 | NO | 32.2% | Do Not Use EV |

**Bottom Line:** The EU model successfully learned to rank patterns by expected value. The US model failed to learn meaningful EV rankings despite identical architecture and training.

---

## Table of Contents

1. [The Problem: Why Consolidation Patterns?](#1-the-problem-why-consolidation-patterns)
2. [The Hypothesis: Coiled Spring Theory](#2-the-hypothesis-coiled-spring-theory)
3. [System Architecture](#3-system-architecture)
4. [Data Pipeline & Temporal Integrity](#4-data-pipeline--temporal-integrity)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Architecture](#6-model-architecture)
7. [Training Methodology](#7-training-methodology)
8. [EU Market Results](#8-eu-market-results)
9. [US Market Results](#9-us-market-results)
10. [Comparative Analysis](#10-comparative-analysis)
11. [Conclusions & Recommendations](#11-conclusions--recommendations)

---

## 1. The Problem: Why Consolidation Patterns?

### Market Observation

Stock prices don't move linearly. They alternate between:
- **Trending phases:** Directional moves with momentum
- **Consolidation phases:** Sideways movement, building energy

### The Opportunity

Consolidation phases represent periods of equilibrium where supply and demand are balanced. When this equilibrium breaks, significant moves often follow. The challenge is identifying which consolidations will break upward versus fail.

### Why This Matters

```
Traditional Approach:
- Buy breakouts after they happen
- Chase momentum, often entering late
- High slippage, crowded trades

Our Approach:
- Identify patterns BEFORE breakout
- Position during quiet consolidation
- Better entry prices, less competition
```

### The Statistical Challenge

Most consolidations fail or go nowhere. Historical data shows:
- ~50% hit stop-loss (Danger)
- ~20% drift sideways (Noise)
- ~30% achieve target (Target)

The goal: Build a model that identifies the 30% with higher probability.

---

## 2. The Hypothesis: Coiled Spring Theory

### Core Idea

A consolidation pattern is like a coiled spring. The tighter and longer the coil, the more potential energy stored. When released, this energy converts to kinetic energy (price movement).

### Observable Indicators of "Coiling"

| Indicator | What It Measures | Ideal State |
|-----------|------------------|-------------|
| **BBW (Bollinger Band Width)** | Volatility contraction | Decreasing |
| **ADX** | Trend strength | Low (<32) |
| **Volume** | Trading activity | Drying up |
| **Daily Range** | Price volatility | Compressing |
| **Duration** | Time in consolidation | 10-50 days |

### The "Perfect Coil"

```
Ideal Pattern Characteristics:
├── Volatility: Contracting (BBW in lowest 30%)
├── Trend: Absent (ADX < 32)
├── Volume: Drying up (<35% of average)
├── Duration: 10+ qualifying days
├── Price: Coiled near support (lower boundary)
└── Context: Near 52-week high, above 200MA
```

### Why Machine Learning?

Human pattern recognition is:
- Subjective and inconsistent
- Unable to process thousands of patterns
- Prone to recency bias

ML can:
- Learn subtle feature interactions
- Process entire market daily
- Maintain consistent criteria
- Quantify uncertainty

---

## 3. System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATTERN DETECTION PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Raw OHLCV  │───▶│   Pattern    │───▶│   Candidate  │       │
│  │     Data     │    │  Detection   │    │   Patterns   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│                                          ┌──────────────┐       │
│                                          │  Wait 100    │       │
│                                          │    Days      │       │
│                                          └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Labeled    │◀───│   Outcome    │◀───│   Evaluate   │       │
│  │   Patterns   │    │   Labeling   │    │   Results    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                                                        │
│         ▼                                                        │
├─────────────────────────────────────────────────────────────────┤
│                    SEQUENCE GENERATION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Extract    │───▶│  Calculate   │───▶│   Generate   │       │
│  │  14 Temporal │    │ 13 Context   │    │  Sequences   │       │
│  │   Features   │    │   Features   │    │  (N,20,14)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    MODEL TRAINING                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Temporal   │    │   Ensemble   │    │   Dirichlet  │       │
│  │    Split     │───▶│   Training   │───▶│ Calibration  │       │
│  │ (No Leakage) │    │  (5 Models)  │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    INFERENCE                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Live      │───▶│   Ensemble   │───▶│   Expected   │       │
│  │   Pattern    │    │  Prediction  │    │    Value     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│                                          ┌──────────────┐       │
│                                          │   Trading    │       │
│                                          │   Signal     │       │
│                                          └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| Pattern Detection | Identify consolidation boxes | `core/sleeper_scanner_v17.py` |
| Outcome Labeling | Classify results (100d later) | `core/path_dependent_labeler.py` |
| Sequence Generation | Create temporal features | `pipeline/01_generate_sequences.py` |
| Model Training | Train LSTM+CNN ensemble | `pipeline/02_train_temporal.py` |
| Inference | Production predictions | `models/ensemble_wrapper.py` |

---

## 4. Data Pipeline & Temporal Integrity

### The Two-Registry System

**Critical Design Decision:** Separate pattern detection from outcome labeling to prevent look-ahead bias.

```
Registry 1: Candidate Patterns (Real-time)
├── Detected at pattern end date
├── outcome_class = NULL (unknown)
├── Only uses data available at detection time
└── Can be used for live trading alerts

Registry 2: Labeled Patterns (Historical)
├── Labeled 100+ days after detection
├── outcome_class = 0, 1, or 2
├── Uses future data for labeling only
└── Used for model training
```

### Why 100 Days?

Consolidation breakouts need time to play out:
- Early exit misses big moves
- Too long includes unrelated price action
- 100 days balances both concerns

### R-Multiple Labeling

Instead of arbitrary percentage gains, we use R-multiples based on pattern structure:

```
R = lower_boundary - stop_loss

Entry:  Upper boundary (breakout trigger)
Stop:   2R below lower boundary
Target: 5R above entry

Example:
├── Upper: $14.02
├── Lower: $12.26
├── R = $0.24
├── Stop:   $11.78 (-2R from lower)
└── Target: $15.22 (+5R from entry)
```

### Outcome Classes

| Class | Name | Condition | Strategic Value |
|-------|------|-----------|-----------------|
| 0 | Danger | Hit -2R stop | -1.0 |
| 1 | Noise | Neither hit in 100d | -0.1 |
| 2 | Target | Hit +5R target | +5.0 |

### Temporal Split (No Leakage)

```
Timeline:
2020 ────────── 2024-01-01 ────── 2024-07-01 ────── 2025
     TRAIN           VAL              TEST

Rules:
├── Train only on patterns ending before 2024-01-01
├── Validate on 2024-01 to 2024-07
├── Test on 2024-07 onwards
└── NO random shuffling (preserves temporal order)
```

---

## 5. Feature Engineering

### 14 Temporal Features (per timestep)

The model sees 20 days of these features, capturing how the pattern evolves:

| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 0-3 | OHLC | Open, High, Low, Close | Relativized to Day 0 |
| 4 | Volume | Trading volume | log(vol_t / vol_0) |
| 5 | BBW_20 | Bollinger Band Width | Raw (0-1 scale) |
| 6 | ADX | Average Directional Index | Raw (0-100) |
| 7 | Volume_Ratio | Vol / 20-day average | Ratio |
| 8 | Vol_Dryup | Volume contraction score | Robust scaled |
| 9 | VAR_Score | Volatility-adjusted return | Robust scaled |
| 10 | NES_Score | Normalized efficiency | Robust scaled |
| 11 | LPF_Score | Low-pass filtered price | Robust scaled |
| 12 | Upper_Bound | Consolidation ceiling | Relativized |
| 13 | Lower_Bound | Consolidation floor | Relativized |

### 13 Context Features (static per pattern)

These capture the broader context that temporal features alone miss:

**Original 8 (Market Context):**

| Feature | What It Captures | Optimal Range |
|---------|------------------|---------------|
| float_turnover | Accumulation activity | 1.5 - 5.0 |
| trend_position | Price vs 200MA | 1.1 - 1.5 |
| base_duration | Consolidation maturity | 15-100 days |
| relative_volume | Recent vs historical vol | 1.2 - 2.0 |
| distance_to_high | Overhead resistance | 5-20% |
| log_float | Share float (liquidity) | 1M - 100M |
| log_dollar_volume | Tradability | $100k - $10M/day |
| relative_strength_spy | Relative performance | Positive |

**New 5 (Coil Features - Jan 2026):**

| Feature | What It Captures | Why It Matters |
|---------|------------------|----------------|
| price_position_at_end | Position in box (0-1) | Low = coiled spring |
| distance_to_danger | Distance from stop zone | Risk assessment |
| bbw_slope_5d | BBW trend direction | Contracting = good |
| vol_trend_5d | Recent volume trend | Drying up = good |
| coil_intensity | Combined coil score | Higher = better odds |

### Feature Philosophy

```
Temporal Features (20 x 14):
"How is the pattern evolving day-by-day?"
├── Captures shape of consolidation
├── Volume dynamics over time
├── Volatility contraction progression
└── Boundary tests and reactions

Context Features (13):
"What's the broader setup?"
├── Market structure (trend position)
├── Liquidity (float, dollar volume)
├── Accumulation signals (turnover)
└── Coil quality at detection
```

---

## 6. Model Architecture

### HybridFeatureNetwork V18

```
                    ┌─────────────────────────────────────┐
                    │         INPUT LAYER                  │
                    │  Sequences: (N, 20, 14)              │
                    │  Context:   (N, 13)                  │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │   TEMPORAL BRANCH   │         │   CONTEXT BRANCH    │
        │                     │         │                     │
        │  ┌───────────────┐  │         │  ┌───────────────┐  │
        │  │ Bidirectional │  │         │  │     GRN       │  │
        │  │     LSTM      │  │         │  │   (Gated      │  │
        │  │  (32 hidden)  │  │         │  │   Residual    │  │
        │  └───────────────┘  │         │  │   Network)    │  │
        │         │           │         │  └───────────────┘  │
        │         ▼           │         │                     │
        │  ┌───────────────┐  │         └─────────────────────┘
        │  │   1D CNN      │  │                    │
        │  │  (64 filters) │  │                    │
        │  └───────────────┘  │                    │
        │         │           │                    │
        │         ▼           │                    │
        │  ┌───────────────┐  │                    │
        │  │  Multi-Head   │  │                    │
        │  │  Attention    │  │                    │
        │  └───────────────┘  │                    │
        └─────────────────────┘                    │
                    │                              │
                    └───────────────┬──────────────┘
                                    ▼
                    ┌─────────────────────────────────────┐
                    │         FUSION LAYER                 │
                    │  Concatenate + Dense + Dropout       │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │         OUTPUT LAYER                 │
                    │  3 classes: Danger, Noise, Target    │
                    │  Softmax probabilities               │
                    └─────────────────────────────────────┘
```

### Why This Architecture?

| Component | Purpose |
|-----------|---------|
| **Bi-LSTM** | Capture sequential dependencies in both directions |
| **1D CNN** | Detect local patterns (volume spikes, price patterns) |
| **Attention** | Learn which timesteps matter most |
| **GRN** | Process context features with gating (handle missing data) |
| **Fusion** | Combine temporal shape with contextual setup |

### Ensemble Configuration

```
5 Models with:
├── Different random seeds (reproducibility)
├── Varied dropout (0.3 - 0.45)
├── Varied hidden dimensions
└── Dirichlet calibration (better probabilities)
```

### Expected Value Calculation

```python
EV = P(Danger) × (-1.0) + P(Noise) × (-0.1) + P(Target) × (+5.0)

Example:
P(Danger) = 0.30 → -0.30
P(Noise)  = 0.40 → -0.04
P(Target) = 0.30 → +1.50
─────────────────────────
EV = +1.16 (Positive expectation)
```

---

## 7. Training Methodology

### Loss Function: Coil-Aware Focal Loss

Standard cross-entropy fails because:
- Class imbalance (50% Danger, 20% Noise, 30% Target)
- Model predicts majority class
- Misses rare but valuable Target patterns

**Coil-Aware Focal Loss:**

```python
# Class weights (penalize Danger mistakes heavily)
weights = {Danger: 5.0, Noise: 1.0, Target: 1.0}

# Gamma per class (focus on hard examples)
gamma = {Danger: 4.0, Noise: 2.0, Target: 0.5}

# Coil strength weighting
# High-coil patterns get boosted gradients
coil_weight = 1.0 + coil_intensity × 3.0
```

### Dirichlet Calibration

Neural networks often produce overconfident probabilities. Dirichlet calibration:
- Learns a transformation matrix on validation set
- Maps raw logits to calibrated probabilities
- Reduces Expected Calibration Error (ECE)

```
EU Model:
├── Pre-calibration ECE: 0.0608
├── Post-calibration ECE: 0.0319
└── Improvement: 47.5%

US Model:
├── Pre-calibration ECE: 0.0957
├── Post-calibration ECE: 0.0264
└── Improvement: 72.4%
```

### Early Stopping

```
Patience: 20 epochs
Monitor: Validation accuracy
Restore: Best weights
```

### Data Augmentation

**NMS (Non-Maximum Suppression):**
- Multiple overlapping patterns for same consolidation
- Keep only the "best" pattern per cluster
- Prevents data leakage between train/test

**Physics Filter:**
- Remove large/mega caps (not enough volatility)
- Remove illiquid stocks (<$50k/day volume)
- Remove too-narrow patterns (<2% width)

---

## 8. EU Market Results

### Dataset Overview

| Metric | Value |
|--------|-------|
| Total Samples | 4,847 |
| Train | 2,384 (49.2%) |
| Validation | 651 (13.4%) |
| Test | 1,812 (37.4%) |
| Unique Tickers | 373 |
| Date Range | 2020-07 to 2025-08 |
| Duplicates Removed | 16.3% |

### Class Distribution (Test Set)

| Class | Count | Percentage |
|-------|-------|------------|
| Danger | 932 | 51.4% |
| Noise | 504 | 27.8% |
| Target | 376 | 20.8% |

### Performance Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Overall Accuracy | 27.6% | Expected (predicts Target heavily) |
| EV-Outcome Pearson | **+0.088** | **POSITIVE - Working** |
| EV-Outcome Spearman | +0.042 | Positive rank correlation |
| Monotonicity | **YES** | Higher EV = Higher Target Rate |

### Monotonicity Check

| Quartile | Avg EV | Target Rate | Danger Rate |
|----------|--------|-------------|-------------|
| Q1 (lowest) | 0.77 | 17.4% | 51.0% |
| Q2 | 0.98 | 18.8% | 55.4% |
| Q3 | 1.18 | 20.1% | 51.0% |
| Q4 (highest) | 1.55 | **26.7%** | 48.3% |

**Key Finding:** Q4 (highest EV) has 53% higher Target Rate than Q1 (26.7% vs 17.4%)

### Top 15% Analysis

| Metric | Value |
|--------|-------|
| Count | 271 patterns |
| EV Threshold | >= 1.43 |
| Target Rate | **31.4%** |
| Danger Rate | 55.7% |
| Profit per Trade | **+0.998** |
| Total Profit | +270.5 |

### EV Bucket Performance

| EV Range | Count | Target% | Danger% | Profit/Trade |
|----------|-------|---------|---------|--------------|
| [0.0, 0.5) | 9 | 0.0% | 22.2% | -0.300 |
| [0.5, 1.0) | 723 | 18.3% | 53.0% | +0.354 |
| [1.0, 1.5) | 884 | 20.5% | 48.5% | +0.507 |
| [1.5, 2.0) | 168 | **31.5%** | 59.5% | +0.973 |
| [2.0, 2.5) | 25 | **36.0%** | 64.0% | +1.160 |
| [2.5, 5.0) | 3 | 33.3% | 66.7% | +1.000 |

### Uncertainty Quantification

| Metric | Value |
|--------|-------|
| Mean CI Width (90%) | 0.607 |
| Epistemic Uncertainty | 0.0018 |
| Aleatoric Uncertainty | 1.054 |
| Model Agreement | 87.0% |

### EU Conclusion

**The EU model successfully learned to rank patterns by expected value.**

- Higher EV predictions correlate with better outcomes
- Monotonicity is preserved across quartiles
- Top 15% achieves 31.4% Target Rate vs 20.8% baseline (1.5x lift)
- Profitable at +0.998 per trade

---

## 9. US Market Results

### Dataset Overview

| Metric | Value |
|--------|-------|
| Total Samples | 3,719 |
| Train | 1,364 |
| Validation | 462 |
| Test | 1,893 |
| Unique Tickers | ~2,500 |
| Date Range | 2020-07 to 2025-08 |
| Duplicates Removed | 27.9% |

### Class Distribution (Test Set)

| Class | Count | Percentage |
|-------|-------|------------|
| Danger | 1,014 | 53.6% |
| Noise | 343 | 18.1% |
| Target | 536 | 28.3% |

### Performance Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Overall Accuracy | 28.3% | Similar to EU |
| EV-Outcome Pearson | **-0.033** | **NEGATIVE - Broken** |
| EV-Outcome Spearman | -0.157 | Negative rank correlation |
| Monotonicity | **NO** | EV ranking unreliable |

### Monotonicity Check

| Quartile | Avg EV | Target Rate | Danger Rate |
|----------|--------|-------------|-------------|
| Q1 (lowest) | 0.89 | 28.3% | 35.9% |
| Q2 | 1.28 | 31.5% | 51.2% |
| Q3 | 1.47 | **24.5%** | 61.9% |
| Q4 (highest) | 1.75 | 28.9% | 65.2% |

**Key Problem:** Q3 has LOWEST Target Rate (24.5%), breaking monotonicity.

### Top 15% Analysis

| Metric | Value |
|--------|-------|
| Count | 283 patterns |
| EV Threshold | >= 1.92 |
| Target Rate | 32.2% |
| Danger Rate | **63.6%** |
| Profit per Trade | +0.967 |
| Total Profit | +273.8 |

### EV Bucket Performance

| EV Range | Count | Target% | Danger% | Profit/Trade |
|----------|-------|---------|---------|--------------|
| [0.0, 0.5) | 2 | 50.0% | 50.0% | +2.000 |
| [0.5, 1.0) | 262 | 25.2% | 33.6% | +0.882 |
| [1.0, 1.5) | 920 | **30.5%** | 50.0% | +1.008 |
| [1.5, 2.0) | 499 | 23.6% | 66.1% | +0.511 |
| [2.0, 2.5) | 210 | 33.3% | 64.3% | +1.021 |

**Note:** [1.0, 1.5) has higher Target Rate than [1.5, 2.0), confirming broken ranking.

### US Conclusion

**The US model failed to learn meaningful EV rankings.**

- Negative correlation means higher EV = worse outcomes
- Monotonicity is broken
- Adding coil features did not fix the issue
- Model overfits to predicting Target class (99.9%)

---

## 10. Comparative Analysis

### Side-by-Side Comparison

| Metric | EU | US | Winner |
|--------|----|----|--------|
| Test Samples | 1,812 | 1,893 | Similar |
| Baseline Target Rate | 20.8% | 28.3% | US higher |
| EV Correlation | **+0.088** | -0.033 | **EU** |
| Monotonicity | **YES** | NO | **EU** |
| Top 15% Target | 31.4% | 32.2% | Similar |
| Top 15% Danger | **55.7%** | 63.6% | **EU** |
| Profit/Trade | **+0.998** | +0.967 | **EU** |
| Model Reliability | **High** | Low | **EU** |

### Why Does EU Work But US Doesn't?

**Hypothesis 1: Different Market Dynamics**
```
EU Market:
├── Smaller, less efficient markets
├── Less algorithmic trading
├── Consolidations may follow more "textbook" patterns
└── Retail-driven breakouts

US Market:
├── Highly efficient, competitive
├── Heavy algorithmic trading
├── Patterns quickly arbitraged away
└── Institutional-driven moves
```

**Hypothesis 2: Data Quality**
```
EU Data:
├── Cleaner exchange listings
├── Less duplicate tickers (16.3%)
├── More distinct patterns

US Data:
├── Many related tickers (ETF variants, ADRs)
├── Higher duplicate rate (27.9%)
├── Noisier patterns
```

**Hypothesis 3: Feature Relevance**
```
EU:
├── Coil features were developed on EU data
├── Features optimized for EU market behavior
└── Better feature-outcome alignment

US:
├── Coil features may not capture US dynamics
├── Different market structure requires different features
└── Need US-specific feature engineering
```

### Visual Comparison: Monotonicity

```
EU Model (WORKING):
Target Rate by EV Quartile

Q1 |████████████████░░░░░░░░░░░░░░| 17.4%
Q2 |█████████████████░░░░░░░░░░░░░| 18.8%
Q3 |██████████████████░░░░░░░░░░░░| 20.1%
Q4 |██████████████████████████░░░░| 26.7%  ← Highest
     ↑ Monotonically increasing = GOOD


US Model (BROKEN):
Target Rate by EV Quartile

Q1 |███████████████████████████░░░| 28.3%
Q2 |██████████████████████████████| 31.5%  ← Highest
Q3 |█████████████████████████░░░░░| 24.5%  ← Lowest!
Q4 |████████████████████████████░░| 28.9%
     ↑ Not monotonic = BROKEN
```

---

## 11. Conclusions & Recommendations

### Summary of Findings

| Finding | Implication |
|---------|-------------|
| EU model has positive EV correlation | EV ranking is reliable for EU |
| US model has negative EV correlation | EV ranking is unreliable for US |
| Coil features didn't fix US | Problem is more fundamental |
| Both models profitable in Top 15% | Raw predictions still useful |
| US has higher baseline Target rate | Market is inherently different |

### Recommendations by Market

#### EU Market: Production Ready

```
RECOMMENDED USAGE:
├── Primary Filter: EV >= 1.43 (Top 15%)
├── Expected: 31.4% Target Rate, +0.998/trade
├── Position Sizing: Use CI lower bound
│   ├── CI > 1.0: Full position
│   ├── CI > 0.5: 3/4 position
│   └── CI < 0.0: Skip
└── Monitoring: Retrain if PSI > 0.25
```

#### US Market: Requires Different Approach

```
DO NOT USE:
├── EV-based signal ranking
├── Top-N by EV selection

ALTERNATIVE APPROACHES:
├── Use P(Target) > 0.35 directly
├── Apply EU-learned heuristics
├── Consider fundamental filters
└── Baseline (28.3%) may be sufficient

FUTURE WORK:
├── US-specific feature engineering
├── Different model architecture
├── Investigate market microstructure
└── Consider sector-specific models
```

### Production Deployment

| Market | Status | Action |
|--------|--------|--------|
| EU | Ready | Deploy with EV >= 1.5 filter |
| US | Not Ready | Use P(Target) or baseline |

### Model Files

| File | Purpose |
|------|---------|
| `output/models/ensemble/ensemble_combined.pt` | Latest ensemble checkpoint |
| `output/sequences/eu_dedup/` | EU training data |
| `output/sequences/us_coil_dedup/` | US training data |
| `docs/MODEL_ANALYSIS_REPORT.md` | EU detailed analysis |
| `docs/MODEL_ANALYSIS_REPORT_US.md` | US detailed analysis |

### Code Example

```python
from models.ensemble_wrapper import EnsembleWrapper

# Load ensemble
ensemble = EnsembleWrapper.from_ensemble_checkpoint(
    'output/models/ensemble/ensemble_combined.pt',
    device='cuda'
)

# Get predictions with uncertainty
results = ensemble.predict_with_uncertainty(sequences, context)

# EU Market: Use EV ranking
eu_signals = results['ev_mean'] >= 1.5

# US Market: Use P(Target) directly (EV unreliable)
us_signals = results['probs_mean'][:, 2] >= 0.35
```

---

## Appendix A: Key Metrics Definitions

| Metric | Definition |
|--------|------------|
| **EV (Expected Value)** | P(D)×(-1) + P(N)×(-0.1) + P(T)×(+5) |
| **Monotonicity** | Higher EV quartiles have higher Target rates |
| **ECE** | Expected Calibration Error (lower = better) |
| **Epistemic Uncertainty** | Model disagreement (reducible) |
| **Aleatoric Uncertainty** | Data noise (irreducible) |
| **R-Multiple** | Risk unit based on pattern structure |

## Appendix B: File Structure

```
trans/
├── config/
│   ├── constants.py          # Strategic values, thresholds
│   └── context_features.py   # 13 context feature definitions
├── core/
│   ├── sleeper_scanner_v17.py    # Pattern detection
│   └── path_dependent_labeler.py # Outcome labeling
├── models/
│   ├── temporal_hybrid_v18.py    # Model architecture
│   ├── inference_wrapper.py      # Single model inference
│   └── ensemble_wrapper.py       # Ensemble inference
├── pipeline/
│   ├── 00_detect_patterns.py     # Pattern detection
│   ├── 00b_label_outcomes.py     # Outcome labeling
│   ├── 01_generate_sequences.py  # Feature extraction
│   └── 02_train_temporal.py      # Model training
├── losses/
│   └── coil_focal_loss.py        # Custom loss function
├── utils/
│   ├── dirichlet_calibration.py  # Probability calibration
│   └── temporal_split.py         # Data splitting
└── docs/
    ├── COMPLETE_SYSTEM_REPORT.md # This document
    ├── MODEL_ANALYSIS_REPORT.md  # EU detailed analysis
    └── MODEL_ANALYSIS_REPORT_US.md # US detailed analysis
```

---

*End of Complete System Report*

**Document Version:** 1.0
**Last Updated:** January 14, 2026
**Author:** Temporal Pattern Detection System
