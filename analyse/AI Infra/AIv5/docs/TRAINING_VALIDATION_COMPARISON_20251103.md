# Training vs Validation Comparison Report - 2025-11-03

## Executive Summary

**CRITICAL FINDING**: Training and validation datasets use **completely different feature extraction pipelines**, with only **3 of 69 features in common**. This explains why K4 detection remains at 0% despite fixing the BBW criterion.

## Dataset Overview

### Training Data
- **File**: `output/patterns_labeled_enhanced_firstbreach_20251101.parquet`
- **Records**: 131,580 snapshots
- **Features**: 69 numeric features (enhanced feature set with EBP components)
- **K4 prevalence**: 0.01% (12 snapshots) - EXTREMELY RARE
- **Outcome distribution**:
  - K0_STAGNANT: 51.4% (67,691)
  - K5_FAILED: 23.7% (31,200)
  - K1_MINIMAL: 20.8% (27,366)
  - K2_QUALITY: 3.8% (4,935)
  - K3_STRONG: 0.3% (376)
  - K4_EXCEPTIONAL: 0.0% (12)

### Validation Data (with BBW-mandatory fix)
- **File**: `output/unused_patterns/pattern_features_full_29.parquet`
- **Records**: 708 snapshots
- **Features**: 31 numeric features (temporal_safe feature set)
- **K4 prevalence**: 2.82% (20 snapshots) - **282x more frequent than training!**
- **Outcome distribution**:
  - K5_FAILED: 42.5% (301)
  - K0_STAGNANT: 36.0% (255)
  - K1_MINIMAL: 8.3% (59)
  - K2_QUALITY: 6.5% (46)
  - K3_STRONG: 3.8% (27)
  - K4_EXCEPTIONAL: 2.8% (20)

## Critical Issue: Feature Set Mismatch

### Common Features (Only 3!)
1. `current_adx`
2. `current_range_ratio`
3. `days_since_activation`

### Training-Only Features (66 features)
**Enhanced/EBP features**:
- `cci_score`, `cci_bbw_compression`, `cci_atr_compression`, `cci_days_factor`
- `var_score`, `var_raw`
- `nes_score`, `nes_inactive_mass`, `nes_wavelet_energy`, `nes_rsa_proxy`
- `lpf_score`, `lpf_bid_pressure`, `lpf_volume_pressure`, `lpf_fta_proxy`
- `tsf_score`, `tsf_days_in_consolidation`
- `ebp_raw`, `ebp_composite`

**Advanced derived features**:
- `baseline_volatility`, `baseline_bbw_avg`, `baseline_volume_avg`
- `bbw_compression_ratio`, `volatility_compression_ratio`
- `bbw_slope_20d`, `adx_slope_20d`, `bbw_std_20d`
- `channel_narrowing_pct`, `breakout_direction`

### Validation-Only Features (28 features)
**Temporal_safe features**:
- `current_bbw`, `current_volume_ratio`
- `bbw_trend_5d`, `bbw_trend_10d`, `volume_trend_10d`
- `distance_to_upper`, `distance_to_lower`, `distance_to_power`
- `consolidation_width`, `position_in_range`
- `bbw_percentile`, `bbw_acceleration`
- `volume_spike`, `volume_vs_50d`
- `daily_volatility`
- `days_since_bbw_low`, `days_since_volume_spike`
- `close_vs_vwap`, `high_low_spread`
- `ma_5`, `ma_10`, `ma_20`
- `price_vs_ma5`, `price_vs_ma20`
- `price_momentum_20d`
- `days_in_qualification`

## Out-of-Distribution Features (2 OOD from 3 common)

| Feature | Training Mean | Validation Mean | Z-Score | Difference | Out of Range |
|---------|---------------|-----------------|---------|------------|--------------|
| `current_range_ratio` | 0.012 | 0.827 | 2.46 | **+6,700%** | Yes |
| `current_adx` | 20.806 | 29.401 | 0.84 | +41.3% | Yes |

## K4-Specific Analysis

### K4 Prevalence
- **Training**: 0.01% (12/131,580) - Extremely rare
- **Validation**: 2.82% (20/708) - **282x more frequent**
- **Impact**: Model never learned to detect K4 patterns (insufficient training examples)

### K4 Feature Comparison (3 common features)

| Feature | Training K4 Mean | Validation K4 Mean | K4 Z-Score |
|---------|------------------|---------------------|------------|
| `current_adx` | 16.305 | 36.416 | **+2.15** |
| `days_since_activation` | 81.250 | 17.600 | **-1.61** |
| `current_range_ratio` | 2.325 | 0.434 | -0.52 |

**Interpretation**:
- Validation K4 patterns have much higher ADX (stronger trends)
- Validation K4 patterns are captured much earlier in their lifecycle
- Completely different pattern characteristics

## Root Cause Analysis

### Why K4 Detection Fails (0% Recall)

**Primary Cause: Feature Set Incompatibility**
1. Models trained on 69 enhanced features
2. Validation provides only 31 temporal_safe features
3. Only 3 features overlap
4. Models cannot function with 95% of expected features missing

**Secondary Causes**:
1. **K4 Extreme Rarity in Training**: Only 12 K4 examples out of 131K snapshots (0.01%)
   - Model never learned to recognize K4 patterns
   - K4 class essentially invisible during training

2. **Outcome Distribution Mismatch**:
   - Training: Heavily biased toward K0 (51.4%) and K5 (23.7%)
   - Validation: More balanced with higher K4 prevalence (2.82%)

3. **Feature Value Distributions**:
   - Even the 3 common features have very different value ranges
   - `current_range_ratio`: 6,700% difference!
   - `current_adx`: 41.3% difference

4. **Pattern Lifecycle Timing**:
   - Training K4: Captured at ~81 days active
   - Validation K4: Captured at ~18 days active
   - Different stages of pattern evolution

## Impact on Model Performance

### Predictions with Incompatible Features
The `predict_historical_validation.py` script somehow ran predictions despite feature mismatch. Possible scenarios:
1. Models filled missing features with zeros/defaults (degraded performance)
2. Models used only the 3 common features (terrible predictive power)
3. Feature matching logic silently failed

**Result**: K4 recall = 0%, EV correlation = -0.140 (negative!)

### Why BBW Fix Didn't Help
- BBW fix improved pattern quality (26.2% mean BBW vs 119.8% old)
- But validation feature extraction used different pipeline than training
- Models never saw the corrected patterns in their expected feature space
- Like training a model on color images, then testing on black & white sketches

## Recommendations

### Immediate Actions (Priority 1)

**1. Regenerate Validation Features with Training Pipeline**
```bash
# Use the same feature extraction as training
python extract_features_enhanced.py --input output/unused_patterns/detected_patterns_historical.parquet
```
- Must use enhanced feature extraction (69+ features)
- Include all EBP components (CCI, VAR, NES, LPF, TSF, composite)
- Match exact feature names and calculation logic

**2. Rerun Predictions with Matched Features**
```bash
# After regenerating features
python predict_historical_validation.py --features output/unused_patterns/pattern_features_enhanced.parquet
```

**3. Expected Results After Feature Matching**
- K4 recall: 0% → likely still low (5-15%) due to extreme rarity in training
- But model will at least USE the correct features
- Can properly evaluate if K4 rarity is the remaining issue

### Long-Term Solutions (Priority 2)

**4. Retrain Models on Combined Dataset**
- Combine training + new validation (131,580 + 708 = 132,288 snapshots)
- K4 examples: 12 + 20 = 32 (still very rare, but 2.7x more)
- Use focal loss with alpha=50 for K4 class (extreme oversampling)

**5. Address K4 Extreme Rarity**
- Expand training dataset to include more historical K4 patterns
- Consider synthetic minority oversampling (SMOTE) for K4
- Train separate K4-specific binary classifier (K4 vs not-K4)

**6. Standardize Feature Extraction Pipeline**
- Document canonical feature extraction process
- Create unit tests to verify feature consistency
- Version control feature extraction code with data

## Validation Report Updates

### Previous Conclusion (INCOMPLETE)
> "The BBW fix was necessary but not sufficient. Root issue extends beyond pattern qualification criteria."

### Updated Conclusion (COMPLETE)
> "The BBW fix successfully improved pattern quality. However, **feature extraction pipeline mismatch** prevents models from evaluating validation patterns. Training (69 enhanced features) and validation (31 temporal_safe features) share only 3 features. This incompatibility, combined with K4 extreme rarity in training (0.01%), causes 0% K4 recall. **Must regenerate validation features using training pipeline before assessing model performance.**"

## Files Generated

1. `output/validation_analysis/feature_distribution_comparison.csv` - Full feature comparison
2. `output/validation_analysis/ood_features.csv` - Out-of-distribution features
3. `output/validation_analysis/k4_feature_comparison.csv` - K4-specific analysis

## Next Steps

1. ✅ **COMPLETED**: Identified feature set mismatch as root cause
2. **PENDING**: Regenerate validation features with enhanced pipeline
3. **PENDING**: Rerun predictions with matched features
4. **PENDING**: Evaluate K4 detection with correct features
5. **PENDING**: If K4 still fails, expand training set or use specialized K4 model

---

**Date**: 2025-11-03
**Status**: ROOT CAUSE IDENTIFIED - Feature pipeline mismatch
**Critical Finding**: Only 3 of 69 features match between training and validation
**Next Priority**: Regenerate validation features using enhanced extraction pipeline
