# Consolidation Criteria Fix - 2025-11-03

## Root Cause Found

**Issue**: K4 detection showed 0% recall because 73% of validation patterns were NOT real consolidations.

**Root Cause**: The "3 of 4" consolidation criteria allowed patterns to qualify WITHOUT Bollinger Band Width (BBW) compression - the KEY indicator of consolidation.

## The Problem

### Old Criteria (BROKEN)
```python
# Required ANY 3 of these 4:
- BBW < 30th percentile (volatility compression)
- ADX < 32 (low trending)
- Volume < 35% of 20-day average
- Range < 65% of 20-day average

# Problem: Pattern could qualify with ADX + Volume + Range = 3/4
# This allowed patterns WITHOUT compressed volatility!
```

### Evidence
**Validation K4 Patterns**:
- Only 8/30 (26.7%) met BBW criterion
- Mean BBW: 119.8 (explosive volatility!)
- Mean channel width: 153% (wide, loose)
- These were NOT consolidations

**Training K4 Patterns** (by luck):
- Most met ALL 4 criteria including BBW
- Mean BBW: 12.3 (properly compressed)
- Mean channel width: 4.1% (tight)
- These WERE consolidations

## The Fix

### New Criteria (FIXED)
```python
# BBW is MANDATORY + at least 2 of the other 3:
bbw_met = bbw_percentile < threshold  # REQUIRED
other_criteria = [adx_ok, volume_ok, range_ok]
return bbw_met and (sum(other_criteria) >= 2)
```

This ensures EVERY pattern has compressed volatility before qualifying.

## Files Modified

1. **`pattern_detection/state_machine_orig/consolidation_tracker.py`** (Line 869-901)
   - Used by training/test pattern detection
   - Changed `_meets_consolidation_criteria()` method

2. **`detect_patterns_historical_unused.py`** (Line 136-157)
   - Used by validation pattern detection
   - Changed consolidation criteria check

## Expected Impact

### Pattern Filtering
- Will filter out ~70% of current validation patterns (the fake ones)
- From 706 → ~200-250 true consolidations
- Only patterns with compressed volatility will qualify

### Model Performance (After Regeneration)
- K4 Recall: 0% → 40-60% (target)
- K3+K4 Recall: 48.9% → 65-75% (target)
- EV Correlation: +0.051 → +0.15 to +0.25 (target)
- Feature distributions will match training

## Next Steps

1. ✅ **COMPLETED**: Fix consolidation criteria in both scripts
2. **PENDING**: Regenerate validation patterns with `detect_patterns_historical_unused.py`
3. **PENDING**: Extract features for filtered patterns
4. **PENDING**: Generate predictions with corrected validation set
5. **PENDING**: Compare results (before/after fix)

## Why This Matters

The model was CORRECTLY rejecting validation K4 patterns because they were NOT consolidations! 

- Training learned: "Consolidation = tight, compressed, BBW ~12"
- Validation had: "Fake consolidations = loose, volatile, BBW ~120"
- Model correctly said: "These don't look like my training data"

The fix ensures training, test, and validation ALL use the same strict definition of consolidation.

---

*Date: 2025-11-03*
*Status: FIX IMPLEMENTED*
*Next: REGENERATE VALIDATION DATA*
