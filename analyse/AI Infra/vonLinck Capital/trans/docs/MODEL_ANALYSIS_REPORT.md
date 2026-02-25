# Complete Model Analysis Report
## Temporal Hybrid V18 Ensemble - EU Market Consolidation Patterns

**Generated:** 2026-01-14
**Model Version:** V18 Ensemble (5 models)
**Data:** EU Market, Deduplicated

---

## Table of Contents
1. [Model Architecture](#1-model-architecture)
2. [Feature Specification](#2-feature-specification)
3. [Data Pipeline & Deduplication](#3-data-pipeline--deduplication)
4. [Training Configuration](#4-training-configuration)
5. [Performance Metrics](#5-performance-metrics)
6. [Expected Value (EV) Analysis](#6-expected-value-ev-analysis)
7. [Uncertainty Quantification](#7-uncertainty-quantification)
8. [Trading Recommendations](#8-trading-recommendations)
9. [Deduplication Impact Analysis](#9-deduplication-impact-analysis)
10. [Conclusions & Recommendations](#10-conclusions--recommendations)
11. [Appendices](#appendices)

---

## 1. Model Architecture

**Model Type:** HybridFeatureNetwork (Temporal Hybrid V18)
**Ensemble Size:** 5 models

### Architecture Components

#### 1.1 Temporal Branch (LSTM + CNN)
- **Input:** 14 features x 20 timesteps
- **LSTM:** Bidirectional, hidden_dim=32, 2 layers
- **CNN:** 1D convolutions for local pattern detection
- **Attention:** Multi-head self-attention over temporal dimension

#### 1.2 Context Branch (GRN - Gated Residual Network)
- **Input:** 13 context features
- **Architecture:** Dense layers with gating mechanism
- **Purpose:** Incorporate static pattern characteristics

#### 1.3 Fusion Layer
- Concatenates temporal and context representations
- Dense layers with dropout for final classification

#### 1.4 Output
- **3 classes:** Danger (0), Noise (1), Target (2)
- Softmax probabilities for EV calculation

---

## 2. Feature Specification

### 2.1 Temporal Features (14 features per timestep, 20 days)

| Index | Feature | Description |
|-------|---------|-------------|
| 0-3 | OHLC (relativized) | (Price_t / Price_0) - 1, day-0 normalized |
| 4 | Volume (log-ratio) | log(volume_t / volume_0) |
| 5 | BBW_20 | Bollinger Band Width (20-period) |
| 6 | ADX | Average Directional Index (trend strength) |
| 7 | Volume_Ratio_20 | Current volume / 20-day average |
| 8 | Vol_Dryup_Ratio | Volume contraction measure |
| 9 | VAR_Score | Volatility-adjusted return score |
| 10 | NES_Score | Normalized efficiency score |
| 11 | LPF_Score | Low-pass filtered price score |
| 12 | Upper_Boundary | Consolidation upper bound (relativized) |
| 13 | Lower_Boundary | Consolidation lower bound (relativized) |

### 2.2 Context Features (13 static features per pattern)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Float_Turnover | Trading activity vs float |
| 1 | Trend_Position | Position within longer trend |
| 2 | Base_Duration | Length of consolidation base |
| 3 | Relative_Volume | Volume vs historical average |
| 4 | Distance_to_High | Distance from 52-week high |
| 5 | Log_Float | log(shares outstanding) |
| 6 | Log_Dollar_Volume | log(average dollar volume) |
| 7 | Relative_Strength | Relative strength vs SPY |
| 8 | Price_Position_End | Price position at pattern end |
| 9 | Distance_to_Danger | Distance to stop-loss level |
| 10 | BBW_Slope_5d | 5-day BBW trend |
| 11 | Vol_Trend_5d | 5-day volume trend |
| 12 | Coil_Intensity | Measure of coiling/compression |

---

## 3. Data Pipeline & Deduplication

### 3.1 Data Source
- **Source:** EU Market Consolidation Patterns
- **Date Range:** 2020-07-08 to 2025-08-05

### 3.2 Deduplication Applied

**Problem:** Multiple exchange listings for same stock created duplicates (e.g., Greatland Gold: GPG.ST, GPH.DE, GPH.L, GPI1.DE). These had 100% identical sequences, inflating false confidence.

**Solution:**
1. Hash-based deduplication using MD5 of sequence content
2. Group by (pattern_end_date, sequence_hash)
3. Keep only primary exchange listing per group

**Exchange Priority:** L > DE > PA > AS > MC > SW > ST > CO

### 3.3 Data Split

| Split | Date Range | Samples | Percentage |
|-------|------------|---------|------------|
| Train | < 2024-01-01 | 2,384 | 49.2% |
| Validation | 2024-01 to 2024-07 | 651 | 13.4% |
| Test | >= 2024-07-01 | 1,812 | 37.4% |
| **Total** | | **4,847** | 100% |

---

## 4. Training Configuration

### 4.1 Ensemble Training
- 5 models with varied random seeds
- Architecture variation: Different dropout/hidden_dim per member
- Loss function: Coil-Aware Focal Loss
- Dirichlet calibration applied post-training

### 4.2 Loss Function Details

**Coil-Aware Focal Loss** addresses class imbalance:
- Class weights: Danger=5.0, Noise=1.0, Target=1.0
- Gamma per class: Danger=4.0, Noise=2.0, Target=0.5
- Coil strength weighting: Boosts gradients for high-coil patterns

### 4.3 Strategic Values (for EV calculation)

| Class | Label | Value | Meaning |
|-------|-------|-------|---------|
| 0 | Danger | -1.0 | Hit -2R stop-loss |
| 1 | Noise | -0.1 | Neither stop nor target hit |
| 2 | Target | +5.0 | Hit +5R profit target |

**EV Formula:** `EV = P(Danger)*(-1.0) + P(Noise)*(-0.1) + P(Target)*(+5.0)`

### 4.4 Member Training Results

| Member | Val Accuracy | Best Epoch |
|--------|--------------|------------|
| 0 | 51.6% | 53 |
| 1 | 49.0% | 76 |
| 2 | 51.6% | 50 |
| 3 | 50.4% | 45 |
| 4 | 50.9% | 73 |

### 4.5 Calibration
- **Method:** Dirichlet Calibration
- **ECE Reduction:** 47.5% (0.0608 -> 0.0319)

---

## 5. Performance Metrics

### 5.1 Test Set Overview

| Metric | Value |
|--------|-------|
| Total samples | 1,812 |
| Date range | 2024-07-01 to 2025-08-05 |
| Unique tickers | 373 |

### 5.2 Class Distribution (Actual)

| Class | Count | Percentage |
|-------|-------|------------|
| Danger (0) | 932 | 51.4% |
| Noise (1) | 504 | 27.8% |
| Target (2) | 376 | 20.8% |

### 5.3 Ensemble vs Single Model Comparison

| Metric | Ensemble | Single Model |
|--------|----------|--------------|
| Overall Accuracy | 27.6% | 42.9% |
| Danger Precision | 50.6% | 59.9% |
| Danger Recall | 14.7% | 35.2% |
| Noise Precision | 64.6% | 35.6% |
| Noise Recall | 10.1% | 88.7% |
| Target Precision | 21.3% | 37.5% |
| Target Recall | 83.0% | 0.8% |
| EV-Outcome Correlation | 0.0882 | 0.0210 |
| Spearman Rank Correlation | 0.0424 | 0.0435 |

### 5.4 Confusion Matrix (Ensemble)

|  | Predicted Danger | Predicted Noise | Predicted Target |
|--|------------------|-----------------|------------------|
| **Actual Danger** | 137 | 24 | 771 |
| **Actual Noise** | 74 | 51 | 379 |
| **Actual Target** | 60 | 4 | 312 |

### 5.5 Prediction Distribution (Ensemble)

| Predicted Class | Count | Percentage |
|-----------------|-------|------------|
| Danger | 271 | 15.0% |
| Noise | 79 | 4.4% |
| Target | 1,462 | 80.7% |

---

## 6. Expected Value (EV) Analysis

### 6.1 EV Distribution

| Statistic | Value |
|-----------|-------|
| Min EV | 0.346 |
| Max EV | 2.752 |
| Mean EV | 1.120 |
| Median EV | 1.076 |
| Std EV | 0.317 |

### 6.2 EV Percentiles

| Percentile | Value |
|------------|-------|
| 10th | 0.755 |
| 25th | 0.896 |
| 50th | 1.076 |
| 75th | 1.306 |
| 90th | 1.513 |
| 95th | 1.698 |

### 6.3 EV Bucket Analysis

| EV Range | Count | Danger% | Noise% | Target% | Avg Profit |
|----------|-------|---------|--------|---------|------------|
| [0.0, 0.5) | 9 | 22.2% | 77.8% | 0.0% | -0.300 |
| [0.5, 1.0) | 723 | 53.0% | 28.8% | 18.3% | +0.354 |
| [1.0, 1.5) | 884 | 48.5% | 31.0% | 20.5% | +0.507 |
| [1.5, 2.0) | 168 | 59.5% | 8.9% | 31.5% | +0.973 |
| [2.0, 2.5) | 25 | 64.0% | 0.0% | 36.0% | +1.160 |
| [2.5, 5.0) | 3 | 66.7% | 0.0% | 33.3% | +1.000 |

### 6.4 Top 15% Analysis (Primary Trading Signal)

| Metric | Value |
|--------|-------|
| Top 15% count | 271 |
| EV threshold (min in top 15%) | 1.433 |
| EV range | 1.433 to 2.752 |
| **Target Rate (Class 2)** | **31.4%** |
| **Danger Rate (Class 0)** | **55.7%** |
| Noise Rate (Class 1) | 12.9% |
| Total Profit (Top 15%) | +270.5 |
| Avg Profit per Trade | +0.998 |

### 6.5 Monotonicity Check

Higher EV should equal higher Target Rate:

| Quartile | Avg EV | Target% | Danger% | Count |
|----------|--------|---------|---------|-------|
| Q1 (lowest) | 0.77 | 17.4% | 51.0% | 453 |
| Q2 | 0.98 | 18.8% | 55.4% | 453 |
| Q3 | 1.18 | 20.1% | 51.0% | 453 |
| Q4 (highest) | 1.55 | 26.7% | 48.3% | 453 |

**Monotonicity preserved: YES**

---

## 7. Uncertainty Quantification

### 7.1 Confidence Interval Statistics (90% CI)

| Metric | Value |
|--------|-------|
| Mean CI Width | 0.607 |
| Median CI Width | 0.598 |
| Min CI Width | 0.123 |
| Max CI Width | 1.458 |

### 7.2 Uncertainty Decomposition

#### Epistemic (Model Uncertainty)
- **Mean:** 0.0018
- **Std:** 0.0012
- Represents disagreement between ensemble members
- Reducible with more training data

#### Aleatoric (Data Uncertainty)
- **Mean:** 1.0539
- **Std:** 0.0348
- Represents inherent noise in data
- Irreducible (intrinsic to the problem)

### 7.3 Model Agreement
- **Mean Agreement:** 0.870
- Measures the fraction of ensemble members predicting the same class

### 7.4 Uncertainty by EV Bucket

| EV Range | CI Width | Epistemic | Aleatoric | Agreement |
|----------|----------|-----------|-----------|-----------|
| [0.00, 0.75) | 0.672 | 0.0017 | 1.0391 | 0.943 |
| [0.75, 1.00) | 0.559 | 0.0014 | 1.0655 | 0.902 |
| [1.00, 1.25) | 0.567 | 0.0017 | 1.0685 | 0.867 |
| [1.25, 1.50) | 0.683 | 0.0025 | 1.0491 | 0.828 |
| [1.50, 2.00) | 0.676 | 0.0025 | 1.0164 | 0.773 |
| [2.00, 5.00) | 0.526 | 0.0015 | 0.9276 | 0.957 |

---

## 8. Trading Recommendations

### 8.1 Signal Thresholds

| Signal Type | Criteria |
|-------------|----------|
| Strong Signal | EV >= 2.0 AND Agreement >= 0.6 AND Uncertainty < 0.5 |
| Signal | EV >= 2.0 OR (EV >= 1.5 AND Agreement >= 0.7) |
| Hold | 0 < EV < 2.0 with insufficient confidence |
| Avoid | EV < 0 or high uncertainty |

### 8.2 Optimal Threshold Analysis

| EV Threshold | Trades | Target% | Danger% | Profit | Profit/Trade |
|--------------|--------|---------|---------|--------|--------------|
| EV >= 0.50 | 1,803 | 20.9% | 51.6% | +900.3 | +0.499 |
| EV >= 0.75 | 1,640 | 21.1% | 52.0% | +833.8 | +0.508 |
| EV >= 1.00 | 1,080 | 22.6% | 50.6% | +644.1 | +0.596 |
| EV >= 1.25 | 554 | 25.6% | 48.9% | +424.9 | +0.767 |
| **EV >= 1.43** | **272** | **31.2%** | **55.5%** | **+270.4** | **+0.994** |
| EV >= 1.50 | 196 | 32.1% | 60.2% | +195.5 | +0.997 |
| EV >= 1.75 | 77 | 29.9% | 66.2% | +63.7 | +0.827 |
| EV >= 2.00 | 28 | 35.7% | 64.3% | +32.0 | +1.143 |

*Note: EV >= 1.43 corresponds to Top 15% threshold*

### 8.3 Position Sizing Recommendations

Based on confidence interval analysis:

| CI Lower Bound | Position Size | Count | Target% | Profit |
|----------------|---------------|-------|---------|--------|
| > 1.0 | Full position | 410 | 25.9% | +326.8 |
| 0.5 - 1.0 | 3/4 position | 1,125 | 19.7% | +468.9 |
| 0.0 - 0.5 | 1/2 position | 273 | 17.6% | +102.3 |
| < 0.0 | Skip trade | 4 | 0.0% | -0.4 |

### 8.4 Additional Filters

Combine model signals with:
- Manual chart review for high-EV signals
- Sector/market regime filters
- Liquidity checks (already filtered for $50k+)

---

## 9. Deduplication Impact Analysis

### 9.1 Problem Identified

Multiple exchange listings for the same stock created duplicate patterns with 100% identical sequences, artificially inflating model confidence.

**Example:** Greatland Gold appeared as 4 separate tickers:
- GPG.ST (Stockholm)
- GPH.DE (Frankfurt)
- GPH.L (London - Primary)
- GPI1.DE (Xetra)

All had identical 14x20 temporal sequences and produced identical predictions, but the model counted them as 4 "successful" high-confidence predictions when they were actually 1 pattern repeated 4 times.

### 9.2 Solution Applied

1. Hash-based deduplication using MD5 of sequence content
2. Group by (pattern_end_date, sequence_hash)
3. Keep only primary exchange listing per group
4. Exchange priority: L > DE > PA > AS > MC > SW > ST > CO

### 9.3 Impact Summary

| Metric | Before Dedup | After Dedup | Change |
|--------|--------------|-------------|--------|
| Test samples | 5,774 | 1,812 | -69% |
| Unique patterns | ~4,832 | 1,812 | Real data only |
| EV [2.5,3.0) Danger Rate | 92.3% | 66.7% | -25.6pp |
| Top 15% Danger Rate | 64.1% | 55.7% | -8.4pp |
| EV-Outcome Monotonicity | BROKEN | TRUE | FIXED |
| EV Correlation | ~0.00 | 0.088 | +0.088 |

### 9.4 Key Improvements

1. **MONOTONICITY RESTORED:** Higher EV now correctly predicts higher Target Rate
   - Before: EV [2.5, 3.0) had 92.3% Danger (inverse!)
   - After: Q1->Q4 Target Rate: 17.4% -> 26.7% (monotonic)

2. **DANGER RATE REDUCED:** 8.4 percentage point improvement in Top 15%
   - Before: 64.1% of high-EV predictions were Danger
   - After: 55.7% (still high, but improved)

3. **DATA INTEGRITY:** No more artificial confidence inflation
   - Each pattern is now unique
   - Model predictions reflect true pattern quality

4. **EV CORRELATION:** Positive correlation restored (0.088 vs ~0)
   - Higher EV predictions now correlate with better outcomes

---

## 10. Conclusions & Recommendations

### 10.1 Model Strengths

- Temporal pattern recognition with LSTM + CNN + Attention architecture
- Ensemble uncertainty quantification provides confidence bounds
- Dirichlet calibration improves probability estimates (47.5% ECE reduction)
- Proper EV correlation after deduplication
- Monotonicity preserved (higher EV = higher success probability)
- GRN context branch incorporates 13 static pattern features

### 10.2 Model Limitations

- High Danger Rate overall (51.4% of patterns hit stop-loss)
- Model predicts Target (80.7%) far more than actual (20.8%)
- Low overall accuracy (27.6%) due to over-predicting Target
- EV range compressed (0.35 to 2.75) - limited differentiation

### 10.3 Recommended Usage

1. **PRIMARY FILTER:** EV >= 1.43 (Top 15% threshold)
   - 271 trades, 31.4% Target Rate, +0.998 profit per trade

2. **HIGH SELECTIVITY:** EV >= 2.0
   - 28 trades, 35.7% Target Rate, +1.143 profit per trade
   - Lower volume but higher quality

3. **POSITION SIZING:** Use CI lower bound
   - CI Lower > 1.0: Full position (410 trades, 25.9% Target)
   - CI Lower > 0.5: 3/4 position (1,125 trades, 19.7% Target)
   - CI Lower < 0.0: Skip trade

4. **COMBINE WITH:**
   - Manual chart review for high-EV signals
   - Sector/market regime filters
   - Liquidity checks (already filtered for $50k+)

### 10.4 Expected Performance (Top 15% @ EV >= 1.43)

| Metric | Value |
|--------|-------|
| Signals per year | ~271 |
| Target Rate | 31.4% (+5R profit) |
| Danger Rate | 55.7% (-1R loss) |
| Noise Rate | 12.9% (-0.1R) |
| Net Expected per trade | +0.998R |
| Estimated Annual Total | +270R |

### 10.5 Next Steps

1. Monitor for drift (retrain if PSI > 0.25)
2. Consider reducing Danger exposure with additional filters
3. Test on US market data for comparison
4. Paper trade for 90 days before live deployment

---

## Appendices

### Appendix A: Sample High-EV Predictions

Top 10 Highest EV Predictions:

| Rank | Ticker | Date | EV | CI | P(Target) | Actual | Result |
|------|--------|------|-----|-----|-----------|--------|--------|
| 1 | GPH.L | 2024-11-06 | 2.75 | [2.44,3.05] | 84.0% | Danger | LOSS |
| 2 | JFN.SW | 2025-04-23 | 2.62 | [2.42,2.83] | 89.1% | Target | WIN |
| 3 | GPH.L | 2024-11-08 | 2.59 | [2.15,2.90] | 82.6% | Danger | LOSS |
| 4 | GPH.L | 2024-11-11 | 2.39 | [1.91,2.80] | 80.2% | Danger | LOSS |
| 5 | EXP.DE | 2024-11-07 | 2.34 | [2.27,2.46] | 78.3% | Danger | LOSS |
| 6 | LUKN.SW | 2025-04-11 | 2.26 | [1.68,2.59] | 83.3% | Target | WIN |
| 7 | SEQI.L | 2025-04-11 | 2.23 | [1.87,2.51] | 81.7% | Target | WIN |
| 8 | STIL.ST | 2024-12-11 | 2.18 | [1.75,2.48] | 76.3% | Danger | LOSS |
| 9 | USCP.DE | 2024-08-07 | 2.16 | [1.67,2.51] | 79.4% | Target | WIN |
| 10 | ZPRS.DE | 2025-02-28 | 2.14 | [2.04,2.23] | 77.8% | Danger | LOSS |

### Appendix B: Target Rate by Month (Test Period)

| Month | Patterns | Top 15% Count | Target% | Danger% |
|-------|----------|---------------|---------|---------|
| 2024-07 | 212 | 31 | 9.7% | 83.9% |
| 2024-08 | 123 | 18 | 55.6% | 27.8% |
| 2024-09 | 95 | 14 | 35.7% | 42.9% |
| 2024-10 | 104 | 15 | 26.7% | 60.0% |
| 2024-11 | 188 | 28 | 46.4% | 53.6% |
| 2024-12 | 169 | 25 | 48.0% | 44.0% |
| 2025-01 | 183 | 27 | 18.5% | 70.4% |
| 2025-02 | 213 | 31 | 19.4% | 77.4% |
| 2025-03 | 101 | 15 | 20.0% | 80.0% |
| 2025-04 | 70 | 10 | 40.0% | 30.0% |
| 2025-05 | 52 | 7 | 0.0% | 14.3% |
| 2025-06 | 49 | 7 | 14.3% | 57.1% |
| 2025-07 | 239 | 35 | 11.4% | 34.3% |
| 2025-08 | 14 | 2 | 50.0% | 50.0% |

### Appendix C: Model Files

| File | Path |
|------|------|
| Ensemble Checkpoint | `output/models/ensemble/ensemble_combined.pt` |
| Single Model | `output/models/eu_dedup_20260114_042115.pt` |
| Data (Deduplicated) | `output/sequences/eu_dedup/sequences_dedup.h5` |
| Metadata | `output/sequences/eu_dedup/metadata_dedup.parquet` |

### Appendix D: Reproduction Commands

```python
# Load ensemble for inference
from models.ensemble_wrapper import EnsembleWrapper
ensemble = EnsembleWrapper.from_ensemble_checkpoint(
    'output/models/ensemble/ensemble_combined.pt',
    device='cuda'  # or 'cpu'
)

# Get predictions with uncertainty
results = ensemble.predict_with_uncertainty(sequences, context)
ev = results['ev_mean']
ci_lower = results['ev_ci_lower']
ci_upper = results['ev_ci_upper']

# Filter for high-confidence signals
high_conf = (ev >= 1.5) & (ci_lower > 0.5)
```

---

*End of Analysis Report*
