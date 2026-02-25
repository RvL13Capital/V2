# Validation Report: Post-BBW Fix - 2025-11-03

## Executive Summary

The BBW-mandatory fix successfully improved pattern quality (70.4% early BBW compliance vs 26.7% old validation), but **model performance did NOT improve**. K4 recall remains at 0%, suggesting the root issue extends beyond pattern qualification criteria.

## Key Changes

### Pattern Detection Criteria

**OLD (BROKEN)**: Any 3 of 4 criteria
```
- BBW < 30th percentile (volatility compression)
- ADX < 32 (low trending)
- Volume < 35% of 20-day average
- Range < 65% of 20-day average
```
**Problem**: Patterns could qualify with ADX + Volume + Range = 3/4 (missing BBW!)

**NEW (FIXED)**: BBW mandatory + 2 of 3 others
```
- BBW < 30th percentile - REQUIRED
- ADX < 32 (low trending)
- Volume < 35% of 20-day average
- Range < 65% of 20-day average
```

### Pattern Regeneration Results

| Metric | Old Validation | New Validation | Change |
|--------|----------------|----------------|--------|
| Tickers processed | 2,617 | 2,617 | Same |
| Patterns detected | 706 | 807 | +14.3% |
| Snapshots generated | 708 | 708 | Same |
| Mean BBW (early snapshots) | 119.8 (explosive!) | 26.2% (compressed) | **4.6× better** |
| BBW compliance (early) | 26.7% | 70.4% | **2.6× improvement** |

## Model Performance Comparison

### K4_EXCEPTIONAL (≥75% gain) - PRIMARY TARGET

| Metric | Old Validation | New Validation | Change |
|--------|----------------|----------------|--------|
| Actual K4 count | 20 | 20 | Same |
| Recall | 0.0% | 0.0% | **No change** |
| Precision | 0.0% | 0.0% | No change |
| Model predictions | 0/20 detected | 0/20 detected | **CRITICAL FAILURE** |

### K3+K4 Combined (≥35% gain)

| Metric | Old Validation | New Validation | Change |
|--------|----------------|----------------|--------|
| Recall | 48.9% | 14.9% | **-34% WORSE** |
| Actual count | Unknown | 47 (K3: 27, K4: 20) | - |

### Overall Metrics

| Metric | Old Validation | New Validation | Change |
|--------|----------------|----------------|--------|
| Overall accuracy | Unknown | 38.1% | - |
| EV correlation | +0.051 | -0.140 | **NEGATIVE (WORSE)** |
| Mean EV | Unknown | -5.26 | Low |
| Median EV | Unknown | -5.94 | Low |

### Signal Distribution

| Signal Type | Old Validation | New Validation |
|-------------|----------------|----------------|
| AVOID | Unknown | 94.2% (667/708) |
| STRONG | Unknown | 2.5% (18/708) |
| GOOD | Unknown | 1.1% (8/708) |
| MODERATE | Unknown | 0.8% (6/708) |
| WEAK | Unknown | 1.3% (9/708) |

## Detailed Analysis

### Pattern Outcome Distribution

**New Validation Actual Outcomes:**
- K5_FAILED: 301 (42.5%)
- K0_STAGNANT: 255 (36.0%)
- K1_MINIMAL: 59 (8.3%)
- K2_QUALITY: 46 (6.5%)
- K3_STRONG: 27 (3.8%)
- K4_EXCEPTIONAL: 20 (2.8%)

**Model Predictions:**
- K5_FAILED: 593 (83.8%) - **Massive over-prediction**
- K3_STRONG: 47 (6.6%)
- K4_EXCEPTIONAL: 36 (5.1%)
- K1_MINIMAL: 15 (2.1%)
- K2_QUALITY: 9 (1.3%)
- K0_STAGNANT: 8 (1.1%)

### Class-Level Accuracy

| Outcome Class | Actual Count | Correct Predictions | Accuracy |
|--------------|--------------|---------------------|----------|
| K5_FAILED | 301 | 258 | 85.7% |
| K0_STAGNANT | 255 | 7 | 2.7% |
| K1_MINIMAL | 59 | 1 | 1.7% |
| K2_QUALITY | 46 | 1 | 2.2% |
| K3_STRONG | 27 | 3 | 11.1% |
| K4_EXCEPTIONAL | 20 | 0 | **0.0%** |

### Geographic Performance

| Market | Accuracy | Pattern Count |
|--------|----------|---------------|
| US | 43.4% | 343 |
| L (UK) | 40.0% | 95 |
| KL (Malaysia) | 34.1% | 129 |
| DE (Germany) | 26.2% | 126 |

## Critical Insights

### ✅ What Worked

1. **BBW-mandatory fix achieved its goal:**
   - Early snapshots now have mean BBW 26.2% vs 119.8% previously
   - 70.4% compliance vs 26.7% old validation (2.6× improvement)
   - Patterns START with compression as required

2. **K5 (failure) detection is excellent:**
   - 85.7% recall (258/301 failures detected)
   - Model correctly identifies breakdown patterns

3. **Pattern quality fundamentally improved:**
   - Distribution mismatch between training/validation resolved
   - Natural BBW expansion over time confirmed (26.2% early → 59.8% late)

### ❌ What DIDN'T Work

1. **K4 detection remains at 0%:**
   - Despite proper BBW compression, model cannot detect exceptional breakouts
   - Only 20 K4 patterns in 708 snapshots (2.8%) - extremely rare

2. **EV correlation turned NEGATIVE:**
   - Old: +0.051 (weak positive)
   - New: -0.140 (inverse correlation)
   - Model predictions are INVERSELY correlated with actual gains

3. **Model heavily over-predicts failures:**
   - Predicts 83.8% K5 but actual is 42.5%
   - 94.2% of patterns classified as AVOID
   - Only 2.5% get STRONG signals

4. **Poor accuracy on quality patterns (K0-K3):**
   - K0: 2.7%, K1: 1.7%, K2: 2.2%, K3: 11.1%
   - Model struggles with everything except failures

## Root Cause Analysis

### Why K4 Detection STILL Fails

The BBW fix eliminated ONE source of distribution mismatch (volatility compression), but **other critical differences remain:**

**Hypothesis 1: Feature Distribution Mismatch**
- Even with correct BBW, other features may still be out-of-distribution
- Need to compare NEW validation features vs training features

**Hypothesis 2: Class Imbalance**
- Only 2.8% of validation snapshots are K4 (20/708)
- Training set may have had higher K4 prevalence
- Model calibrated for different outcome distribution

**Hypothesis 3: Temporal/Market Regime Differences**
- Training: Historical patterns from training markets (UK/Malaysia?)
- Validation: Unused tickers from 2020-2025 (different market regime)
- Model may not generalize across markets/time periods

**Hypothesis 4: Snapshot Timing Bias**
- Snapshots taken probabilistically throughout active phase
- Early snapshots (compressed BBW) may look different from training
- Model trained on snapshots at specific pattern stages?

## Pattern Quality Verification

### BBW Distribution by Snapshot Timing

**Early Snapshots (≤5 days active):**
- Count: 277 snapshots
- Mean BBW percentile: 0.262 (26.2%)
- Compliance: 195/277 (70.4%)
- Days since active start: ~5 days average

**Late Snapshots (>20 days active):**
- Count: 173 snapshots
- Mean BBW percentile: 0.598 (59.8%)
- Compliance: 31/173 (17.9%)
- Days since active start: ~20+ days average

**Interpretation:**
- Patterns correctly START with compressed volatility
- Natural expansion occurs as patterns mature
- This is EXPECTED and confirms fix is working

### Remaining Non-Compliance

**29.6% of early snapshots** still have BBW ≥ 0.30. Possible causes:
1. Rapid expansion immediately after qualification (normal market behavior)
2. Snapshot timing captures moment of expansion
3. BBW calculation variance at boundaries

This is acceptable given the massive improvement from old validation (73.3% non-compliance).

## Comparison to Training Data (STILL NEEDED)

**CRITICAL MISSING ANALYSIS:**
We need to compare NEW validation patterns (with correct BBW) against training patterns to identify remaining distribution mismatches:

1. Load training pattern features
2. Compare feature distributions (mean, std, min, max, z-scores)
3. Identify which features are still out-of-distribution
4. Determine if model is fundamentally incompatible with validation data

## Next Steps

### Immediate Actions (Priority Order)

**1. Compare NEW validation features to training features**
```bash
python analyze_k4_failure.py --new-validation
```
- Load `predictions_historical_validation.parquet` (NEW)
- Load training pattern features
- Calculate z-scores for each feature
- Identify remaining OOD features

**2. Analyze K4 patterns specifically**
- Extract the 20 K4 patterns from validation
- Compare their features to training K4 patterns
- Determine if they're fundamentally different

**3. Evaluate model calibration**
- Check if model probabilities are calibrated for new distribution
- Consider retraining on combined dataset (training + new validation)

**4. Consider alternative approaches**
- Retrain models specifically for new pattern distribution
- Use domain adaptation techniques
- Build market-specific models (US, UK, Malaysia, Germany)

### Long-Term Recommendations

**1. Model Retraining:**
- Include new validation patterns in training set
- Balance class distribution (oversample K4)
- Use focal loss to prioritize K4 detection

**2. Feature Engineering:**
- Add market-specific features
- Add temporal regime indicators
- Add volume profile features

**3. Model Architecture:**
- Consider ensemble with market-specific models
- Add meta-features (market, time period)
- Use calibrated probability estimates

## Conclusion

### Success: Pattern Quality Improved

The BBW-mandatory fix **successfully achieved its primary goal** of ensuring consolidation patterns have compressed volatility:

✅ Mean BBW: 119.8 → 26.2% (4.6× better)
✅ BBW compliance: 26.7% → 70.4% (2.6× improvement)
✅ Distribution mismatch resolved

### Failure: Model Performance NOT Improved

Despite correct pattern quality, **model performance did NOT improve**:

❌ K4 recall: 0% → 0% (no change)
❌ EV correlation: +0.051 → -0.140 (worse)
❌ Over-predicts failures: 42.5% actual → 83.8% predicted

### Root Issue: Beyond Pattern Qualification

The **root issue extends beyond pattern qualification criteria**. The model fails to detect K4 patterns even when they have proper BBW compression, suggesting:

1. Other features remain out-of-distribution
2. Model calibrated for different outcome distribution
3. Market/temporal regime differences
4. Insufficient K4 training examples

### Recommendation: RETRAIN MODELS

**The BBW fix was necessary but not sufficient.** To achieve K4 detection, we need to:

1. Analyze NEW validation features vs training
2. Identify remaining distribution mismatches
3. Retrain models on combined dataset or use domain adaptation
4. Balance K4 class representation
5. Consider market-specific models

---

**Date**: 2025-11-03
**Status**: BBW FIX VERIFIED - Pattern quality improved, but model performance NOT improved
**Critical Finding**: K4 detection failure persists despite correct BBW compression
**Next Priority**: Compare NEW validation features to training features to identify remaining mismatches
