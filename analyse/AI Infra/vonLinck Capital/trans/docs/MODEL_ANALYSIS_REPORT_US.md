# Complete Model Analysis Report - US Market
## Temporal Hybrid V18 Ensemble - US Market Consolidation Patterns

**Generated:** 2026-01-14
**Model Version:** V18 Ensemble (5 models)
**Data:** US Market, Deduplicated

---

## Table of Contents
1. [Model Architecture](#1-model-architecture)
2. [Data Overview](#2-data-overview)
3. [Class Distribution](#3-class-distribution)
4. [Performance Metrics](#4-performance-metrics)
5. [Expected Value (EV) Analysis](#5-expected-value-ev-analysis)
6. [Top 15% Analysis](#6-top-15-analysis)
7. [Uncertainty Quantification](#7-uncertainty-quantification)
   - 7.5 [Probability Calibration](#75-probability-calibration-reliability-diagrams)
   - 7.6 [Signal Quality Analysis (Top-N)](#76-signal-quality-analysis-top-n-performance)
   - 7.7 [Cumulative Lift Curve](#77-cumulative-lift-curve-model-efficiency)
8. [Trading Thresholds](#8-trading-thresholds)
9. [US vs EU Comparison](#9-us-vs-eu-comparison)
10. [Sample Predictions](#10-sample-predictions)
11. [Conclusions & Recommendations](#11-conclusions--recommendations)

---

## Executive Summary

**CRITICAL FINDING: The US model has NEGATIVE EV-to-outcome correlation (-0.054), meaning higher EV predictions are associated with WORSE outcomes. This model should NOT be used for EV-based trading decisions.**

| Key Metric | US Model | EU Model | Assessment |
|------------|----------|----------|------------|
| EV Correlation | -0.054 | +0.088 | US is inverted |
| Monotonicity | NO | YES | US is broken |
| Top 15% Target Rate | 30.4% | 31.4% | Similar |
| Baseline Target Rate | 28.4% | 20.8% | US higher |

---

## 1. Model Architecture

**Model Type:** HybridFeatureNetwork (Temporal Hybrid V18)
**Ensemble Size:** 5 models with Dirichlet calibration

### Architecture Components

1. **Temporal Branch (LSTM + CNN)**
   - Input: 14 features x 20 timesteps
   - LSTM: Bidirectional, hidden_dim=32, 2 layers
   - CNN: 1D convolutions for local pattern detection
   - Attention: Multi-head self-attention

2. **Context Branch (GRN)**
   - Input: 13 context features
   - **Note:** US data only has 8 original features
   - Features 8-12 padded with defaults (coil features unavailable)

3. **Output:** 3 classes (Danger, Noise, Target)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Ensemble Size | 5 models |
| Architecture Variation | Yes (varied dropout) |
| Loss Function | Coil-Aware Focal Loss |
| Dirichlet Calibration | Yes (72.4% ECE reduction) |
| Pre-calibration ECE | 0.0957 |
| Post-calibration ECE | 0.0264 |

---

## 2. Data Overview

### Source
- **Market:** US Stocks
- **Original Samples:** 21,739
- **After Deduplication:** 15,030 (-30.9%)

### Deduplication Applied
**Problem:** Related tickers (ETF variants, ADRs) had identical sequences
- Example: SGOL/SGPPF, GLD/GLDG, CEF/CEFA

**Solution:** Hash-based deduplication keeping shorter ticker names (typically primary listings)

### Date Range
- **Full Dataset:** 2020-07-09 to 2025-08-01
- **Unique Tickers:** 3,013

### Temporal Split

| Split | Date Range | Samples | Percentage |
|-------|------------|---------|------------|
| Train | < 2024-01-01 | 7,397 | 49.2% |
| Validation | 2024-01 to 2024-07 | 1,550 | 10.3% |
| Test | >= 2024-07-01 | 6,083 | 40.5% |

---

## 3. Class Distribution

### Test Set Distribution (Actual)

| Class | Count | Percentage |
|-------|-------|------------|
| Danger (0) | 3,306 | 54.3% |
| Noise (1) | 1,048 | 17.2% |
| Target (2) | 1,729 | 28.4% |

### Prediction Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Predicted Danger | 1,091 | 17.9% |
| Predicted Noise | 4,091 | 67.3% |
| Predicted Target | 901 | 14.8% |

**Note:** Model predominantly predicts Noise (67.3%), unlike EU model which predicted Target (80.7%)

---

## 4. Performance Metrics

### Overall Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 34.1% |
| EV-Outcome Pearson | -0.0544 |
| EV-Outcome Spearman | -0.0955 |

### Per-Class Metrics

| Class | Precision | Recall |
|-------|-----------|--------|
| Danger | 72.4% | 23.9% |
| Noise | 24.3% | 94.8% |
| Target | 32.3% | 16.8% |

### Confusion Matrix

|  | Pred Danger | Pred Noise | Pred Target |
|--|-------------|------------|-------------|
| **Actual Danger** | 790 | 1,952 | 564 |
| **Actual Noise** | 8 | 994 | 46 |
| **Actual Target** | 293 | 1,145 | 291 |

---

## 5. Expected Value (EV) Analysis

### EV Distribution

| Statistic | Value |
|-----------|-------|
| Min | 0.590 |
| Max | 2.389 |
| Mean | 1.695 |
| Median | 1.778 |
| Std | 0.269 |

### EV Percentiles

| Percentile | Value |
|------------|-------|
| 10th | 1.233 |
| 25th | 1.613 |
| 50th | 1.778 |
| 75th | 1.870 |
| 90th | 1.937 |
| 95th | 1.982 |

### EV Bucket Analysis

| EV Range | Count | Danger% | Noise% | Target% | Avg Profit |
|----------|-------|---------|--------|---------|------------|
| [0.5, 1.0) | 126 | 31.7% | 37.3% | 31.0% | +1.193 |
| [1.0, 1.5) | 996 | 41.2% | 25.6% | 33.2% | +1.224 |
| [1.5, 2.0) | 4,720 | 58.2% | 14.6% | 27.2% | +0.765 |
| [2.0, 2.5) | 241 | 46.1% | 23.2% | 30.7% | +1.051 |

**CRITICAL:** Lower EV buckets have HIGHER Target rates and LOWER Danger rates. This is inverted from expected behavior.

---

## 6. Top 15% Analysis

### Performance (n=912)

| Metric | Value |
|--------|-------|
| EV Threshold (min) | 1.909 |
| EV Range | 1.909 to 2.389 |
| Target Rate | 30.4% |
| Danger Rate | 56.7% |
| Noise Rate | 12.9% |
| Total Profit | +856.2 |
| Avg Profit/Trade | +0.939 |

### Monotonicity Check

| Quartile | Avg EV | Target% | Danger% | Count |
|----------|--------|---------|---------|-------|
| Q1 (lowest) | 1.30 | **30.8%** | 41.9% | 1,521 |
| Q2 | 1.71 | 26.5% | 54.0% | 1,520 |
| Q3 | 1.82 | 26.6% | 62.1% | 1,521 |
| Q4 (highest) | 1.94 | 29.8% | 59.4% | 1,521 |

**Monotonicity Preserved: NO**

Q1 (lowest EV) has the HIGHEST Target rate at 30.8%. This confirms the model's EV predictions are not useful for ranking.

---

## 7. Uncertainty Quantification

### Confidence Interval Statistics (90% CI)

| Metric | Value |
|--------|-------|
| Mean CI Width | 0.367 |
| Median CI Width | 0.304 |

### Uncertainty Decomposition

| Type | Mean | Std |
|------|------|-----|
| Epistemic (Model) | 0.0019 | 0.0018 |
| Aleatoric (Data) | 0.9835 | 0.0963 |

### Model Agreement

| Metric | Value |
|--------|-------|
| Mean Agreement | 0.843 |

---

## 7.5 Probability Calibration (Reliability Diagrams)

Despite Dirichlet calibration during training, the US model shows calibration issues.

### Brier Score & ECE

| Class | Brier Score | ECE | Assessment |
|-------|-------------|-----|------------|
| **Target** | 0.2157 | 0.0887 | Poor |
| Noise | 0.1539 | 0.1249 | Moderate |
| Danger | 0.2802 | 0.1976 | Poor |

**Interpretation:**
- **Target Class:** Poor calibration (Brier > 0.2). Model overestimates P(Target).
- **Noise Class:** Best calibrated of the three.
- **Danger Class:** Severe underconfidence - actual Danger rate much higher than predicted.

### Calibration Plots

![Target Probability Calibration](../output/visualizations/us_calibration_target_20260114_203932.png)

*Target class shows poor calibration - predicted probabilities consistently overestimate actual frequency.*

![All Classes Calibration](../output/visualizations/us_calibration_all_classes_20260114_203932.png)

*Comparison across classes. Note: Good calibration does NOT fix the negative EV correlation.*

### Key Insight

Even if probabilities were perfectly calibrated, the **negative EV correlation** means the EV calculation itself is flawed for US data. Better calibration would not solve the fundamental ranking problem.

---

## 7.6 Signal Quality Analysis (Top-N Performance)

**WARNING:** EV-based ranking is NOT effective for US market.

### Top-N Performance Summary

| Top N % | Samples | Target Rate | Lift | Danger Rate | EV Threshold |
|---------|---------|-------------|------|-------------|--------------|
| Top 5% | 95 | 32.6% | 1.15x | 64.2% | 2.141 |
| Top 10% | 190 | 33.2% | **1.17x** | 64.7% | 2.020 |
| Top 15% | 284 | 32.4% | 1.14x | 63.4% | 1.921 |
| Top 20% | 379 | 29.8% | 1.05x | 64.4% | 1.807 |
| Top 25% | 474 | 28.9% | 1.02x | 65.2% | 1.699 |
| Top 30% | 568 | 27.8% | **0.98x** | 65.3% | 1.612 |
| **Baseline** | 1893 | 28.3% | 1.00x | 53.6% | - |

**Critical Observations:**
1. **All Lift values < 1.2x** - Model provides minimal ranking value
2. **Top 30% has NEGATIVE lift (0.98x)** - Worse than random selection
3. **Danger rate increases** with Top-N selection (inverse of desired behavior)

### Lift Curve Analysis

![Top-N Lift Curve](../output/visualizations/us_top_n_lift_curve_20260114_203932.png)

*Four-panel analysis showing broken ranking behavior. Note: Lift never reaches 1.5x at any depth.*

### Signal Quality Table

![Top-N Summary Table](../output/visualizations/us_top_n_summary_table_20260114_203932.png)

*All rows red (Lift < 1.2x) - EV-based selection is ineffective.*

### US vs EU Lift Comparison

| Selection | US Lift | EU Lift | Difference |
|-----------|---------|---------|------------|
| Top 10% | 1.17x | 1.62x | **-0.45x** |
| Top 15% | 1.14x | 1.51x | **-0.37x** |
| Top 20% | 1.05x | 1.35x | **-0.30x** |

**Conclusion:** EU model provides 3-4x more ranking value than US model.

### 7.7 Cumulative Lift Curve (Model Efficiency)

The cumulative lift curve shows the overall ranking efficiency across the entire population.

#### AULC Comparison

| Metric | US Model | EU Model | Assessment |
|--------|----------|----------|------------|
| **AULC** | **1.011** | 1.204 | US nearly random |
| Lift @ 10% | 1.17x | 1.62x | US -27% |
| Lift @ 15% | 1.14x | 1.51x | US -25% |
| Lift @ 20% | 1.05x | 1.35x | US -22% |

**AULC = 1.011 means:** US model is only 1.1% better than random selection on average - essentially no predictive value.

#### Cumulative Lift Curve Visualization

![Cumulative Lift Curve](../output/visualizations/us_cumulative_lift_curve_20260114_205556.png)

*Left: US lift curve never reaches 1.5x (green line) and dips BELOW random in the 30-50% range. Right: Target rate barely exceeds baseline.*

#### Critical Observations

1. **Never Reaches Good Lift:** The curve never touches 1.5x (the green "good lift" threshold)
2. **Dips Below Random:** Between 30-50% of population, lift falls BELOW 1.0 (red shading)
3. **Flat Curve:** Unlike EU's smooth decay, US curve is nearly flat around 1.0
4. **AULC ≈ 1.0:** Confirms the model provides essentially no ranking value

#### Why This Happens

The negative EV correlation (-0.033) means:
- High EV predictions do NOT correspond to better outcomes
- The ranking is essentially random noise
- EV-based selection provides no edge over random selection

#### Practical Implication

**DO NOT use EV-based ranking for US market.** The AULC of 1.011 proves that sorting by EV is no better than random selection.

---

## 8. Trading Thresholds

| EV Threshold | Trades | Target% | Danger% | Profit | Profit/Trade |
|--------------|--------|---------|---------|--------|--------------|
| EV >= 0.50 | 6,083 | 28.4% | 54.3% | +5,234.2 | +0.860 |
| EV >= 1.00 | 5,957 | 28.4% | 54.8% | +5,083.9 | +0.853 |
| EV >= 1.50 | 4,961 | 27.4% | 57.6% | +3,864.4 | +0.779 |
| EV >= 1.75 | 3,436 | 28.0% | 60.8% | +2,678.3 | +0.779 |
| EV >= 1.90 | 1,044 | 31.0% | 57.0% | +1,012.5 | +0.970 |
| EV >= 2.00 | 241 | 30.7% | 46.1% | +253.4 | +1.051 |

**Note:** Higher EV thresholds do NOT consistently improve Target rate, confirming EV is not a valid ranking metric for US.

---

## 9. US vs EU Comparison

| Metric | US Market | EU Market |
|--------|-----------|-----------|
| **Data** | | |
| Total Test Samples | 6,083 | 1,812 |
| Unique Tickers | 3,013 | 373 |
| Duplicates Removed | 30.9% | 16.3% |
| **Distribution** | | |
| Danger Rate | 54.3% | 51.4% |
| Noise Rate | 17.2% | 27.8% |
| Target Rate | 28.4% | 20.8% |
| **Performance** | | |
| Overall Accuracy | 34.1% | 27.6% |
| EV-Outcome Correlation | **-0.054** | **+0.088** |
| Monotonicity | **NO** | **YES** |
| **Top 15%** | | |
| Target Rate | 30.4% | 31.4% |
| Danger Rate | 56.7% | 55.7% |
| Profit per Trade | +0.939 | +0.998 |

### Key Differences

1. **US has NEGATIVE EV correlation** - Model predictions are inverted
2. **US has higher baseline Target rate** (28.4% vs 20.8%)
3. **US has larger sample size** (6,083 vs 1,812)
4. **US lacks coil context features** (5 of 13 features are defaults)

---

## 10. Sample Predictions

### Top 10 Highest EV Predictions

| Rank | Ticker | Date | EV | P(Target) | Actual | Result |
|------|--------|------|-----|-----------|--------|--------|
| 1 | UZF | 2025-07-29 | 2.39 | 55.6% | Danger | LOSS |
| 2 | DFIC | 2025-05-06 | 2.36 | 27.2% | Noise | LOSS |
| 3 | IEUR | 2025-05-06 | 2.36 | 30.7% | Noise | LOSS |
| 4 | AWAY | 2025-07-21 | 2.32 | 30.1% | Danger | LOSS |
| 5 | DIHP | 2025-05-02 | 2.29 | 42.3% | Noise | LOSS |
| 6 | SHLD | 2025-02-10 | 2.27 | 31.6% | Target | WIN |
| 7 | MAYZ | 2025-05-27 | 2.26 | 15.5% | Noise | LOSS |
| 8 | FNDF | 2025-05-08 | 2.26 | 23.5% | Target | WIN |
| 9 | DEEF | 2025-05-06 | 2.26 | 19.6% | Noise | LOSS |
| 10 | FEUS | 2024-09-05 | 2.25 | 38.8% | Target | WIN |

**Win Rate in Top 10:** 3/10 (30%) - below baseline Target rate of 28.4%

---

## 11. Conclusions & Recommendations

### Model Assessment

| Aspect | Assessment |
|--------|------------|
| EV Correlation | **FAILED** (negative correlation) |
| Monotonicity | **FAILED** (Q1 has highest Target rate) |
| Calibration | Pass (72.4% ECE reduction) |
| Profitability | Pass (+0.939 per trade in Top 15%) |

### Root Causes of Poor EV Correlation

1. **Missing Coil Features:** 5 of 13 context features are defaults
   - price_position_at_end
   - distance_to_danger
   - bbw_slope_5d
   - vol_trend_5d
   - coil_intensity

2. **Different Market Dynamics:** US has naturally higher Target rate (28.4% vs EU 20.8%)

3. **Training Data Mismatch:** Model may have overfit to patterns not generalizing to test period

### Recommendations

#### DO NOT USE
- EV-based signal ranking for US market
- Top-N by EV selection strategy

#### ALTERNATIVE APPROACHES

1. **Use P(Target) Directly**
   - Filter by P(Target) > 0.35
   - Ignore EV calculation for US

2. **Regenerate US Sequences**
   - Include proper coil features
   - Retrain with full 13 context features

3. **Use EU Model Insights**
   - Apply EU patterns/thresholds to US stocks
   - EU model has proven EV correlation

4. **Baseline Strategy**
   - US baseline Target rate (28.4%) is already high
   - Random selection may perform similarly to model

### Model Files

| File | Path |
|------|------|
| Ensemble Checkpoint | `output/models/ensemble/ensemble_combined.pt` |
| Individual Members | `output/models/us_dedup_ensemble_ensemble_0X_*.pt` |
| Data (Deduplicated) | `output/sequences/us_dedup/sequences_dedup.h5` |
| Metadata | `output/sequences/us_dedup/metadata_dedup.parquet` |

### Usage Code

```python
# Load ensemble (WARNING: EV not reliable for US)
from models.ensemble_wrapper import EnsembleWrapper

ensemble = EnsembleWrapper.from_ensemble_checkpoint(
    'output/models/ensemble/ensemble_combined.pt',
    device='cuda'
)

results = ensemble.predict_with_uncertainty(sequences, context)

# For US: Use P(Target) directly, NOT EV
p_target = results['probs_mean'][:, 2]
high_target = p_target > 0.35  # Alternative to EV-based selection
```

---

## Summary Table

| Metric | US Value | Recommendation |
|--------|----------|----------------|
| EV Correlation | -0.054 | Do not use EV |
| Monotonicity | No | Do not use EV ranking |
| Top 15% Target | 30.4% | Still profitable |
| Baseline Target | 28.4% | Consider baseline strategy |
| Missing Features | 5/13 | Regenerate with coil data |

---

*End of US Market Analysis Report*
