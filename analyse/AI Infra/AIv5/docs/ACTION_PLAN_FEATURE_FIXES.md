# Action Plan: Fix Feature Calculation Bugs

**Date**: 2025-11-02
**Priority**: CRITICAL
**Status**: Ready to Execute

---

## Executive Summary

K4 failure analysis revealed **18 out of 48 features** (37.5%) have calculation bugs in the simplified feature extraction script. These bugs cause the model to completely fail at detecting K4 patterns (0% recall).

**Root Cause**: Simplified script (`extract_features_historical_simple.py`) calculates features incorrectly, causing astronomical z-scores (up to 3.8 billion) and 100% out-of-distribution values.

---

## Critical Bugs Identified

### 1. `volatility_compression_ratio`
**Status**: BROKEN (z-score: +3.8 BILLION)
**OOD**: 93.8%

**Problem**: Producing astronomical values indicating divide-by-zero or unit error

**Current calculation** (lines 275-277 in `extract_features_historical_simple.py`):
```python
if baseline_volatility > 0:
    volatility_compression_ratio = current_volatility / baseline_volatility
else:
    volatility_compression_ratio = 0
```

**Likely issue**: `current_volatility` or `baseline_volatility` calculation is wrong

**Fix strategy**:
- Check calculation of `current_volatility` (should be recent rolling std)
- Check calculation of `baseline_volatility` (should be earlier period std)
- Verify units match (both should be percentages or both absolute)
- Add bounds checking (ratio should be 0-10, not billions)

---

### 2. `baseline_volume_avg`
**Status**: BROKEN (z-score: +64 MILLION)
**OOD**: 95.0%

**Problem**: Unit mismatch or incorrect averaging period

**Current calculation** (lines 223-229):
```python
# Baseline period (days 10-30 of pattern)
baseline_start_idx = max(0, len(df_pattern) - days_in_pattern)
baseline_end_idx = max(0, len(df_pattern) - max(10, days_in_pattern - 20))

if baseline_start_idx < baseline_end_idx:
    baseline_volume_avg = df_pattern.iloc[baseline_start_idx:baseline_end_idx]['volume'].mean()
else:
    baseline_volume_avg = df_pattern['volume'].mean()
```

**Likely issue**: Indexing logic is backwards or average includes outliers

**Fix strategy**:
- Verify indexing logic (forward or backward from snapshot_date?)
- Check for outlier handling (one huge volume day can skew average)
- Compare against training data typical values (should be in thousands/millions, not billions)

---

### 3. Price Distance Features (100% OOD)

**Features affected**:
- `distance_from_power_pct` (z-score: +1,275, 100% OOD)
- `price_distance_from_upper_pct` (z-score: +1,267, 100% OOD)
- `price_distance_from_lower_pct` (z-score: +1,555, 50% OOD)

**Problem**: Incorrect percentage calculation or wrong boundary values

**Current calculation** (lines 243-248):
```python
if power > 0:
    distance_from_power_pct = ((power - current_price) / current_price) * 100
else:
    distance_from_power_pct = 0

# Similar for upper and lower
```

**Likely issue**: Formula is inverted or boundaries are wrong

**Expected formula**:
```python
distance_from_power_pct = ((current_price - power) / power) * 100
# Positive if above power, negative if below
```

**Fix strategy**:
- Verify formula matches full pipeline
- Check boundary values (upper, lower, power) are correct
- Ensure percentages are reasonable (-100 to +100 range)

---

### 4. `price_volatility_20d` (100% OOD)
**Status**: BROKEN (z-score: +196)

**Current calculation** (lines 321-323):
```python
recent_20 = df_pattern.tail(min(20, len(df_pattern)))
price_volatility_20d = recent_20['close'].pct_change().std()
```

**Likely issue**: Result is decimal (e.g., 0.02 = 2%) not percentage (2.0)

**Fix**: Multiply by 100 to convert to percentage
```python
price_volatility_20d = recent_20['close'].pct_change().std() * 100
```

---

### 5. `consolidation_quality_score` (100% OOD)
**Status**: BROKEN

**Current calculation** (lines 290-299):
```python
# Multiple component calculation
quality_score = (
    (1 - bbw_percentile) * 0.3 +
    (1 - normalized_adx) * 0.3 +
    (1 - volume_ratio) * 0.2 +
    (1 - range_ratio) * 0.2
)
consolidation_quality_score = quality_score
```

**Likely issues**:
- Component values out of expected range (0-1)
- Missing normalization
- Formula doesn't match full pipeline

**Fix strategy**:
- Verify each component is properly normalized (0-1)
- Check full pipeline formula
- Add bounds checking (score should be 0-1)

---

### 6. `consolidation_channel_width_pct`
**Status**: INCORRECT (z-score: +19.66, 80% OOD)

**Current calculation** (lines 209-211):
```python
if lower > 0:
    consolidation_channel_width_pct = ((upper - lower) / lower) * 100
```

**Likely issue**: Should be divided by midpoint or close, not lower

**Expected formula**:
```python
mid = (upper + lower) / 2
consolidation_channel_width_pct = ((upper - lower) / mid) * 100
```

---

## Bugs by Severity

### CRITICAL (Fix Immediately)
1. `volatility_compression_ratio` (z-score: 3.8 billion)
2. `baseline_volume_avg` (z-score: 64 million)
3. `distance_from_power_pct` (100% OOD)
4. `price_distance_from_upper_pct` (100% OOD)
5. `price_volatility_20d` (100% OOD)

### HIGH (Fix Soon)
6. `consolidation_quality_score` (100% OOD)
7. `consolidation_channel_width_pct` (80% OOD)
8. `baseline_bbw_avg` (90% OOD)
9. `nes_rsa_proxy` (100% OOD)

### MEDIUM (Review)
10. `current_price`, `current_high`, `current_low`, `start_price` (85% OOD)
11. Various EBP components (70-80% OOD)

---

## Recommended Fix Process

### Step 1: Create Fixed Version of Script
1. Create `extract_features_historical_FIXED.py`
2. Fix the 5 CRITICAL bugs first
3. Add validation checks (reasonable ranges)
4. Add debug logging

### Step 2: Validate Fixes
1. Run on single K4 pattern
2. Compare features to training data
3. Check z-scores are <2
4. Verify no values >100% OOD

### Step 3: Regenerate All Features
1. Run fixed script on all 706 patterns
2. Save to `pattern_features_historical_48enhanced_v2.parquet`
3. Verify feature distributions match training

### Step 4: Regenerate Predictions
1. Run `predict_historical_enhanced.py` with new features
2. Save to `predictions_historical_enhanced_v2.parquet`
3. Check K4 recall improves from 0%

### Step 5: Regenerate Reports
1. Run `generate_validation_report_enhanced.py` with new predictions
2. Compare results to initial validation
3. Document improvements

---

## Expected Improvements After Fixes

### K4 Detection
- **Current**: 0% recall (completely broken)
- **Expected**: 40-60% recall (closer to test set 66.7%)

### K3+K4 Combined
- **Current**: 48.9% recall
- **Expected**: 65-75% recall (closer to test set 75-80%)

### EV Correlation
- **Current**: +0.051 (weak positive)
- **Expected**: +0.15 to +0.25 (moderate positive)

### Feature Distributions
- **Current**: 18 features >50% OOD
- **Expected**: 0-2 features >50% OOD

---

## Alternative Approach: Use Full Pipeline

Instead of fixing the simplified script, could use the full pipeline:

**Option A: Fix Simplified Script** (Recommended)
- Pros: Standalone, no dependencies, already mostly written
- Cons: Need to fix ~18 features
- Time: 2-4 hours

**Option B: Use Full Pipeline** (Alternative)
- Pros: Features guaranteed correct
- Cons: Complex imports, dependencies, slower
- Time: 1-2 hours setup + longer runtime

**Recommendation**: Fix simplified script (Option A) because:
1. Already mostly correct (30 of 48 features work)
2. Faster runtime for 706 patterns
3. Bugs are identified and fixable
4. No complex dependencies

---

## Implementation Timeline

**Today (Next 2-3 hours)**:
- [ ] Fix 5 CRITICAL bugs in simplified script
- [ ] Test on single K4 pattern
- [ ] Verify fixes with z-scores

**Today (Next 1-2 hours)**:
- [ ] Regenerate features for all 706 patterns
- [ ] Regenerate predictions
- [ ] Quick validation check

**Tomorrow**:
- [ ] Regenerate full validation report
- [ ] Compare before/after results
- [ ] Document improvements
- [ ] Update deployment guide if results are good

---

## Success Criteria

**Minimum Acceptable**:
- K4 recall > 0% (anything better than current)
- K3+K4 recall > 55%
- Features: <5 with >50% OOD
- Z-scores: All <10

**Good Performance**:
- K4 recall > 30%
- K3+K4 recall > 65%
- EV correlation > +0.15
- Features: All <50% OOD

**Excellent Performance**:
- K4 recall > 50%
- K3+K4 recall > 70%
- EV correlation > +0.20
- Features: All match training distribution

---

## Rollback Plan

If fixes don't work:
1. Keep original `extract_features_historical_simple.py` (backup)
2. Document what was tried
3. Fall back to Option B (full pipeline)
4. Or: Accept current limitations, deploy with caveat

---

## Next Steps

1. **Immediate**: Fix the 5 CRITICAL bugs
2. **Validate**: Test on sample pattern
3. **Deploy**: Regenerate features for all 706 patterns
4. **Evaluate**: Check if K4 detection improves

---

*Created: 2025-11-02*
*Priority: CRITICAL*
*Owner: Feature Engineering Team*
