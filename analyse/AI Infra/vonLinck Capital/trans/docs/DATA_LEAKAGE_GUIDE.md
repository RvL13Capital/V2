# Four Deadly Sins of Data Leakage

These bugs caused Lift to collapse from 1.71x to 0.99x. **NEVER repeat them.**

---

## Principle 1: Windows Wildcard Resolution

```python
# BROKEN: Windows does NOT expand wildcards like Unix
h5py.File("output/*.h5", 'r')  # FAILS on Windows!

# CORRECT: Always resolve wildcards explicitly
import glob
files = sorted(glob.glob("output/*.h5"))
h5py.File(files[-1], 'r')  # Use most recent
```

**Sin:** Passed `*.h5` directly to h5py on Windows -> silently loaded nothing.

---

## Principle 2: NMS Cluster IDs for ALL Modes

```python
# BROKEN: Only set cluster IDs for 'trinity' mode
if nms_mode == 'trinity':
    result['nms_cluster_id'] = cluster_map[idx]

# CORRECT: Set cluster IDs for ALL modes
# Otherwise each pattern gets unique ID -> ratio = 1.0 -> data leakage!
cluster_map = {idx: cid for idx, cid in cluster_assignments}
result['nms_cluster_id'] = result.index.map(cluster_map)  # ALWAYS
```

**Sin:** NMS cluster ratio was 1.0 (each pattern = own cluster) -> train/val/test splits leaked.

---

## Principle 3: Cross-Ticker Deduplication

```python
# MISSED: ETFs tracking same underlying have IDENTICAL patterns
# GLD and GLDG both have pattern on 2024-01-15 with same boundaries
# Model sees "different" patterns that are actually duplicates

# CORRECT: Deduplicate by pattern signature across ALL tickers
signature = f"{end_date}_{upper_boundary:.2f}_{lower_boundary:.2f}"
dedup_df = df.drop_duplicates(subset=['signature'])
```

**Sin:** 62% of patterns were cross-ticker duplicates -> massive train/test leakage.

---

## Principle 4: Metadata-HDF5 Index Alignment

```python
# BROKEN: sequence_idx column was all zeros after deduplication
test_data = sequences[meta.loc[test_mask, 'sequence_idx']]  # All zeros!

# CORRECT: When meta and HDF5 are in same order, use row indices
test_idx = np.where(test_mask)[0]  # 0, 1, 2, ... N-1
test_data = sequences[test_idx]

# ALWAYS verify alignment:
assert np.array_equal(meta['label'].values, hdf5_labels), "Misaligned!"
```

**Sin:** All test predictions used index 0 -> 100% predicted same class.

---

## Validation Checklist Before Training

```bash
# 1. Check NMS cluster ratio (should be > 1.0, ideally 3-5)
python -c "import pandas as pd; df=pd.read_parquet('metadata.parquet'); \
  print(f'Ratio: {len(df)/df.nms_cluster_id.nunique():.1f}')"

# 2. Check for cross-ticker duplicates
python -c "import pandas as pd; df=pd.read_parquet('metadata.parquet'); \
  df['sig']=df['pattern_end_date'].astype(str)+df['upper_boundary'].round(2).astype(str); \
  print(f'Duplicates: {df.duplicated(\"sig\").sum()}/{len(df)}')"

# 3. Verify data alignment
python -c "import h5py; import pandas as pd; import numpy as np; \
  meta=pd.read_parquet('metadata.parquet'); \
  with h5py.File('sequences.h5','r') as f: labels=f['labels'][:]; \
  assert np.array_equal(meta['label'].values, labels), 'MISALIGNED!'; \
  print('Alignment OK')"
```

---

## Data Integrity Checks (Automated)

The pipeline now **automatically blocks** training if data integrity fails:

```python
from utils.data_integrity import DataIntegrityChecker, DataIntegrityError

checker = DataIntegrityChecker(df, date_col='pattern_end_date', strict=True)
results = checker.run_all_checks(
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
    feature_columns=feature_cols
)
# Raises DataIntegrityError if:
#   - Duplication ratio > 1.1x
#   - Val dates <= train max (temporal leakage)
#   - Test dates <= val max (temporal leakage)
#   - Leakage columns in features
#   - Target events < 100 (insufficient power)
```

### Integrity Thresholds

| Check | Threshold | What It Catches |
|-------|-----------|-----------------|
| Duplication | ratio <= 1.1x | 74x LSTM timestep expansion bug |
| Temporal Integrity | val_min > train_max | Look-ahead bias in splits |
| Feature Leakage | 0 leakage cols | Outcome columns as features |
| Statistical Power | targets >= 100 | Insufficient sample size |

### Forbidden Feature Columns

```python
LEAKAGE_COLUMNS = {
    'breakout_class', 'label_3R', 'label_4R', 'max_r_achieved',
    'outcome_class', 'target_hit', 'stop_hit', 'final_r',
    'days_to_target', 'days_to_stop', 'breakout_date'
}
```

---

## Principle 5: Labeling Bias Prevention (Jan 2026)

### 5a. Stop Loss Must Use Low Price

```python
# BROKEN: Used close price - misses intraday stop breaches
stop_mask = outcome_data['close'] < technical_floor  # 5-10% missed!

# CORRECT: Stock can breach floor intraday but close above
stop_mask = outcome_data['low'] < technical_floor
```

**Sin:** Patterns that hit stop intraday but closed above were labeled TARGET/NOISE instead of DANGER.

### 5b. Volume Confirmation Requires Sustained Activity

```python
# BROKEN: Single-day spike qualified as TARGET
volume_confirmed = (volume > 2 * vol_20d_avg).any()  # Manipulation-prone!

# CORRECT: Require 3 consecutive days where EACH day meets BOTH criteria
dollar_volume = volume * close
volume_surge_mask = (
    (volume > 2.0 * vol_20d_avg) &      # Relative surge (2x average)
    (dollar_volume >= 50_000)            # Absolute floor ($50k/day)
)
consecutive_count = volume_surge_mask.rolling(window=3, min_periods=3).sum()
volume_confirmed = (consecutive_count >= 3).any()
```

**Sin:** Single-day manipulation spikes and dormant stocks with tiny trades qualified as TARGET.

### 5c. Untradeable Patterns Must Be Filtered at Prediction

```python
# BROKEN: Gap patterns (>0.5R) included in model training but NOT filtered at prediction
# Model learns "gaps = strong momentum = success" but retail can't execute these!

# CORRECT: Filter untradeable patterns in 03_predict_temporal.py
if 'untradeable' in merged.columns:
    tradeable_mask = ~merged['untradeable'].fillna(False)
    merged = merged[tradeable_mask]  # Only generate orders for executable patterns
```

**Sin:** Optimistic bias - model accuracy inflated by patterns retail traders cannot actually trade.

---

## Principle 6: Metadata-HDF5 Label Alignment Verification

```python
# BROKEN: Assumed metadata and HDF5 are aligned without verification
# After deduplication, order can silently change

# CORRECT: Explicit alignment check in training pipeline
label_col = 'outcome_class' if 'outcome_class' in metadata.columns else 'label'
metadata_labels = metadata[label_col].values.astype(int)
h5_labels = labels.astype(int)
mismatches = np.sum(metadata_labels != h5_labels)

if mismatches > 0:
    raise DataIntegrityError(
        f"Metadata labels do not match HDF5 labels: {mismatches} mismatches. "
        f"Regenerate sequences to fix alignment."
    )
```

**Sin:** Silent prediction errors where all sequences mapped to wrong labels.
