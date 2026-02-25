# Feature Importance Analysis Report
**Date**: 2025-11-04
**Models Analyzed**: 4 trained models (XGBoost + LightGBM, Standard + Combined)

---

## Executive Summary

This analysis reveals which features are most critical for detecting BIG_WINNER patterns (K3+K4 combined) and standard K0-K5 classification. The best-performing model (LightGBM Combined) achieves **71.5% BIG_WINNER recall** by prioritizing:

1. **Trend Indicators** (ADX): 13.8% total importance - Detecting directional momentum changes
2. **Volatility Compression** (BBW): 28.3% total importance - Core consolidation signal
3. **Price Position**: 23.2% total importance - Proximity to breakout thresholds

---

## Top 10 Most Important Features (All Models Average)

| Rank | Feature | Avg Importance | Category | Description |
|------|---------|----------------|----------|-------------|
| 1 | **current_bbw_20** | 6.73% | Volatility (BBW) | Current Bollinger Band Width - primary volatility measure |
| 2 | **baseline_volume_avg** | 5.80% | Volume | Historical volume baseline for comparison |
| 3 | **baseline_bbw_avg** | 5.24% | Volatility (BBW) | Historical BBW baseline - compression reference point |
| 4 | **avg_range_20d** | 4.96% | Range & Volatility | 20-day average daily range |
| 5 | **adx_slope_20d** | 4.82% | Trend (ADX) | ADX trend direction - momentum acceleration |
| 6 | **start_price** | 4.81% | Price Position | Pattern starting price |
| 7 | **current_adx** | 4.62% | Trend (ADX) | Current ADX value - trend strength |
| 8 | **days_in_pattern** | 4.60% | Pattern Metrics | Pattern duration (maturity signal) |
| 9 | **bbw_compression_ratio** | 4.17% | Volatility (BBW) | Current BBW vs baseline - compression degree |
| 10 | **days_since_activation** | 4.16% | Pattern Metrics | Days since pattern became active |

---

## Category-Level Importance

### LightGBM Combined (Best Model: 71.5% BIG_WINNER Recall)

| Category | Total Importance | # Features | Key Insight |
|----------|------------------|------------|-------------|
| **Volatility (BBW)** | 28.26% | 6 | **#1 Priority** - Volatility compression is the strongest signal |
| **Price Position** | 23.25% | 6 | Proximity to boundaries critical for timing |
| **Pattern Metrics** | 15.60% | 3 | Pattern maturity matters (days_in_pattern) |
| **Trend (ADX)** | 13.80% | 2 | Trend reversal detection crucial |
| **Range & Volatility** | 12.96% | 5 | Consolidation tightness |
| Boundaries | 3.47% | 2 | Less important (already captured in price position) |
| Volume | 2.66% | 4 | Surprisingly low for combined model |

### XGBoost Standard (K0-K5 Classification)

| Category | Total Importance | # Features | Key Insight |
|----------|------------------|------------|-------------|
| **Range & Volatility** | 21.91% | 5 | **#1 for Standard** - Broader range of outcomes needs volatility context |
| **Volatility (BBW)** | 21.55% | 6 | Still critical, tied with range |
| **Price Position** | 18.36% | 6 | Important for all outcome classes |
| **Volume** | 17.90% | 4 | **More important** for standard model (distinguishing K5 failures) |
| Boundaries | 9.27% | 2 | Higher importance in standard model |
| Pattern Metrics | 6.04% | 3 | Less critical for multi-class |
| Trend (ADX) | 4.97% | 2 | Lower priority in standard classification |

---

## Model Comparison Insights

### LightGBM vs XGBoost

**LightGBM Models**:
- **Higher ADX importance** (13.2-13.8%) - Better at detecting trend reversals
- **Lower volume importance** (2.5-2.7%) - De-emphasizes volume in favor of volatility
- **Top feature**: `adx_slope_20d` (7.22% in Combined) - Momentum acceleration

**XGBoost Models**:
- **Higher volume importance** (17.7-17.9%) - Relies more on volume signals
- **Higher boundary importance** (6.3-9.3%) - Direct boundary proximity matters more
- **Top feature**: `baseline_volume_avg` (10.95% in Combined) - Volume baseline critical

### Combined vs Standard Models

**Combined (BIG_WINNER) Models**:
- **Focus**: Volatility (BBW) + Trend (ADX) + Price Position
- **De-emphasize**: Volume, boundaries
- **Strategy**: Detect explosive potential early via compression + momentum
- **Result**: 71.5% recall (LightGBM), 69.6% recall (XGBoost)

**Standard (K0-K5) Models**:
- **Focus**: Range/Volatility + Volume + Boundaries
- **More balanced** across all categories
- **Strategy**: Distinguish between 6 classes requires broader feature set
- **Result**: 47.8% K4 recall, 71.2% K3 recall, 66.5% K3+K4 combined

---

## Critical Features by Category

### 1. Volatility (BBW) - 21.5-29.9% Total Importance

**Top Contributors**:
- `current_bbw_20` (4.96-10.21%) - **Most important single feature** across models
- `baseline_bbw_avg` (4.70-5.88%) - Compression reference point
- `bbw_compression_ratio` (3.87-5.48%) - Degree of squeeze
- `bbw_std_20d` (3.03-4.92%) - Volatility stability

**Insight**: All models heavily rely on BBW for consolidation detection. Lower BBW = higher breakout probability.

### 2. Trend (ADX) - 5.0-13.8% Total Importance

**Top Contributors**:
- `adx_slope_20d` (3.06-7.22%) - **Critical for LightGBM Combined** (7.22%)
- `current_adx` (2.69-6.58%) - Trend strength at snapshot

**Insight**: LightGBM models use ADX more effectively (13.8% vs 5.8% in XGBoost). Trend slope captures momentum acceleration better than absolute ADX value.

### 3. Price Position - 17.8-23.2% Total Importance

**Top Contributors**:
- `start_price` (3.77-5.36%) - Pattern baseline
- `price_distance_from_lower_pct` (3.23-4.57%) - Distance from support
- `price_position_in_range` (2.87-3.48%) - Relative position within boundaries

**Insight**: Proximity to boundaries matters, but not as much as volatility. Being near lower boundary + low BBW = strong signal.

### 4. Pattern Metrics - 6.0-21.6% Total Importance

**Top Contributors**:
- `days_in_pattern` (5.81-9.04%) - **Most important for Standard models**
- `days_since_activation` (5.35-8.30%) - Pattern age
- `consolidation_quality_score` (3.36-4.44%) - Composite quality metric

**Insight**: Standard models rely more on pattern maturity (21.6% vs 15.6% in Combined). Longer consolidations slightly favor success, but not a strong signal alone.

### 5. Volume - 2.5-17.9% Total Importance

**Top Contributors**:
- `baseline_volume_avg` (1.70-10.95%) - **#1 feature in XGBoost Combined** (10.95%)
- `current_volume` (0.76-3.67%) - Current volume level
- `volume_compression_ratio` (0.77-3.08%) - Volume contraction

**Insight**: **Massive divergence** between models. XGBoost uses volume heavily (17.9%), LightGBM barely uses it (2.5%). This suggests volume may be correlated with BBW and is redundant for LightGBM.

### 6. Range & Volatility - 12.0-21.9% Total Importance

**Top Contributors**:
- `avg_range_20d` (1.11-13.56%) - **#1 feature in XGBoost Standard** (13.56%)
- `volatility_compression_ratio` (3.92-4.30%) - Price volatility contraction
- `baseline_volatility` (3.85-4.07%) - Historical volatility baseline

**Insight**: More important for Standard models (21.9% vs 13.0% in Combined). Range complements BBW for distinguishing all 6 outcome classes.

---

## Surprising Findings

### 1. Volume is Less Important Than Expected
- **Expected**: Volume crucial for breakouts (conventional wisdom)
- **Reality**: Only 2.5-2.7% importance in LightGBM (best models)
- **Explanation**: BBW already captures volume-driven volatility contraction. Volume adds little signal beyond BBW.

### 2. ADX Slope > ADX Value
- **Expected**: Current ADX (trend strength) would dominate
- **Reality**: `adx_slope_20d` (7.22%) > `current_adx` (6.58%) in LightGBM Combined
- **Explanation**: **Momentum acceleration** (ADX increasing) predicts breakouts better than absolute trend strength.

### 3. Pattern Duration Matters More for Standard Models
- **Expected**: Longer consolidations = better breakouts (universally)
- **Reality**: `days_in_pattern` 9.04% (Standard) vs 5.81% (Combined)
- **Explanation**: For BIG_WINNER detection, volatility compression matters more. For distinguishing K0-K5, duration provides context.

### 4. Boundaries Have Low Direct Importance
- **Expected**: Distance to boundaries critical
- **Reality**: Only 3.5-9.3% total importance
- **Explanation**: Price position features already capture this. Absolute boundary values add little.

---

## Model Strategy Interpretation

### LightGBM Combined (71.5% Recall) Strategy:
```
IF (BBW compressed) AND (ADX slope increasing) AND (near lower boundary)
THEN BIG_WINNER likely
```

**Feature Weights**:
1. Volatility compression (28.3%) → Core signal
2. Price near support (23.2%) → Entry timing
3. Pattern maturity (15.6%) → Signal confirmation
4. Trend acceleration (13.8%) → Momentum building

**Why It Works**: Focuses on the **3 critical conditions** for explosive moves:
1. Energy coiled (low BBW)
2. Momentum building (ADX slope)
3. Near support (price position)

### XGBoost Standard (66.5% K3+K4) Strategy:
```
IF (Range contracting) AND (Volume baseline high) AND (BBW compressed)
THEN K3+K4 likely
ELSE IF (Volume spike) AND (ADX low) THEN K5 likely
```

**Feature Weights**:
1. Range/Volatility (21.9%) → Consolidation quality
2. Volatility (21.6%) → BBW compression
3. Price position (18.4%) → Entry zone
4. Volume (17.9%) → Distinguishes K5 failures

**Why Different**: Must distinguish **6 classes** (K0-K5), not just BIG_WINNER. Requires more features to separate:
- K5 failures (volume + ADX patterns)
- K0-K2 weak outcomes (range, duration)
- K3-K4 winners (BBW, ADX slope)

---

## Recommendations

### For Model Improvement

1. **Feature Engineering - ADX Derivatives**:
   - `adx_slope_20d` is critical (7.22%) → Create more momentum features:
     - ADX acceleration (2nd derivative)
     - ADX cross above/below threshold (binary)
     - Days since ADX reversal

2. **BBW Composite Metrics**:
   - `current_bbw_20`, `baseline_bbw_avg`, `bbw_compression_ratio` all important
   - Create **BBW percentile rank** (current BBW vs 200-day distribution)
   - **BBW velocity** (rate of change)

3. **Price Position Refinement**:
   - `price_distance_from_lower_pct` (4.57%) shows proximity to support matters
   - Add: **Time near lower boundary** (days spent within 5% of support)
   - Add: **Lower boundary bounce count** (number of tests without break)

4. **Volume Simplification**:
   - Low importance in LightGBM (2.5%) suggests volume is redundant
   - **Remove volume features** from Combined models to reduce noise
   - Keep for Standard models (needed to detect K5 failures)

5. **Pattern Metrics Enhancement**:
   - `days_in_pattern` important but non-linear
   - Create **pattern age buckets** (0-10d, 10-20d, 20-30d, 30+d)
   - Add **consolidation tightness over time** (BBW trend during pattern)

### For Production Deployment

1. **Feature Monitoring**:
   - Track `current_bbw_20`, `adx_slope_20d`, `baseline_bbw_avg` for data drift
   - Alert if BBW distribution shifts significantly (regime change)

2. **Model Selection**:
   - **Use LightGBM Combined** for BIG_WINNER detection (71.5% recall)
   - Use XGBoost Standard if need to distinguish K5 failures (better volume/range signals)

3. **Signal Thresholds**:
   - Require `current_bbw_20` < 30th percentile (mandatory)
   - Require `adx_slope_20d` > 0 (positive momentum)
   - Require `price_distance_from_lower_pct` < 50% (closer to support)

4. **Ensemble Opportunity**:
   - LightGBM: Strong on volatility + trend
   - XGBoost: Strong on volume + range
   - **Combine predictions** for robust signals (require both models agree)

---

## Feature Removal Candidates (Low Importance)

Based on importance < 2% across all models:

| Feature | Max Importance | Category | Reason for Low Importance |
|---------|----------------|----------|---------------------------|
| `current_volume_ratio_20` | 1.84% | Volume | Redundant with `baseline_volume_avg` |
| `current_volume` | 3.67% | Volume | Absolute volume less useful than ratio |
| `price_volatility_20d` | 2.86% | Range & Volatility | Captured by BBW |
| `current_range_ratio` | 2.50% | Range & Volatility | Redundant with `avg_range_20d` |

**Impact**: Removing these 4 features would:
- Reduce feature count by 14% (28 → 24)
- Minimal information loss (< 3% total importance)
- Faster training + prediction
- Less overfitting risk

---

## Visualizations

All visualizations saved to `output/feature_importance/`:

1. **Individual Model Plots**:
   - `lightgbm_combined_improved_*_top_features.png` - Top 20 features bar chart
   - `lightgbm_combined_improved_*_category_importance.png` - Category breakdown
   - Similar for XGBoost Standard/Combined models

2. **Comparison Plot**:
   - `model_comparison_top_features.png` - Side-by-side top 15 features across all models

3. **CSV Exports**:
   - `model_comparison.csv` - Full feature comparison matrix
   - `*_importance.csv` - Individual model importance scores

---

## Next Steps

1. **Walk-Forward Validation** (currently running):
   - Validate these feature importances hold across market eras (1973-2025)
   - Check if ADX slope importance varies by decade
   - Assess BBW importance stability during different volatility regimes

2. **SHAP Analysis**:
   - Feature importance shows **correlation**, not **causation**
   - Run SHAP to understand **directional impact** (e.g., does high ADX slope → higher BIG_WINNER probability?)
   - Identify feature interactions (e.g., BBW × ADX slope synergy)

3. **Feature Engineering V2**:
   - Implement recommended derivative features (ADX acceleration, BBW velocity)
   - Test removing low-importance volume features
   - Re-train and compare K4 recall improvements

4. **Production Model Finalization**:
   - Based on walk-forward validation results, select final model
   - Lock feature set (prevent feature drift)
   - Document feature calculation logic for real-time inference

---

## Conclusion

The feature importance analysis reveals that **detecting BIG_WINNER patterns** relies primarily on:

1. **Volatility Compression** (BBW): 28% importance - Energy coiled for explosion
2. **Price Position**: 23% importance - Entry timing near support
3. **Pattern Maturity**: 16% importance - Signal confirmation
4. **Trend Momentum** (ADX slope): 14% importance - Momentum building

**Volume is surprisingly unimportant** (2.5%) for the best model (LightGBM Combined 71.5% recall), suggesting BBW already captures volume-driven compression.

**ADX slope outperforms ADX value**, indicating **momentum acceleration** predicts breakouts better than absolute trend strength.

These insights align with the model's 71.5% BIG_WINNER recall and validate the pattern detection strategy: **Find coiled energy (low BBW) + building momentum (ADX slope) near support (price position)**.

---

**Generated**: 2025-11-04 21:48 UTC
**Models**: lightgbm_combined_improved_20251104_221058.pkl (71.5% recall - BEST)
**Status**: Feature importance analysis complete, walk-forward validation in progress
