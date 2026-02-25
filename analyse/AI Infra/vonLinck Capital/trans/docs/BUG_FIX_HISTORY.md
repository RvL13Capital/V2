# Bug Fix History

## Critical Fixes (Jan 2026)

| Issue | Impact | Fix |
|-------|--------|-----|
| **High P(Danger) in GOOD signals** | 58.9% of GOOD signals were high-danger | `--max-danger-prob 0.25` guardrail |
| **Trinity mode temporal leakage** | Patterns clustered across years (99.7% train) | `--disable-trinity` for training |
| **Stop loss used close price** | 5-10% of DANGER labels missed (intraday breach, close above) | Use `low` price for stop check |
| **Untradeable patterns not filtered** | Gap >0.5R patterns generated unexecutable orders | Filter `untradeable=True` in 03_predict |
| **Single-day volume spike** | Manipulation spikes qualified as TARGET | Require 3 consecutive days meeting criteria |
| **No absolute volume floor** | Dormant stocks with tiny trades qualified | Each day must have $50k+ dollar volume |
| **Metadata-HDF5 misalignment** | Silent prediction errors (all map to idx 0) | Explicit label alignment check in training |
| **74x Row Duplication** | V24 had 76,479 rows but only 1,034 unique patterns | Mandatory integrity checks in pipeline |
| **VIX lift overstated** | 1.60x claimed, 1.05x actual | Corrected analysis |
| **Fixed 40-day window** | Low-vol patterns timeout, high-vol waste time | V22: Dynamic window (10-60 days) |
| **Windows wildcard bug** | Files not loading on Windows | glob.glob() resolution |
| **NMS cluster IDs missing** | Ratio=1.0, massive leakage | Set for ALL NMS modes |
| **Cross-ticker duplicates** | 62% ETF duplicates leaked | Dedupe by pattern signature |
| **Metadata index misaligned** | All predictions = class 0 | Use np.where() row indices |
| **Trinity split imbalance** | Val=44/Test=45 (0.1%) | `--disable-trinity` for pattern-level splits |
| **ADV 10% too loose** | Untradeable in micro-caps | Tightened to 4% |
| **Zombie -> Danger** | Insufficient data != failure | Reclassified to Noise |
| **NO_FILL deleted data** | Model blind to strongest signals | Label with `untradeable=True` |
| **Volume ratios NaN/Inf** | Features explode on dormant | Log-diff transformation |
| **Hardcoded indices** | Maintenance nightmare | Dynamic Feature Registry |

---

## Full Chronological History

| Date | Issue | Fix |
|------|-------|-----|
| 2026-01-27 | UK exclusion experiment | REJECTED - lift dropped 4.61x to 1.12x, keep UK in training |
| 2026-01-27 | High P(Danger) in GOOD signals (58.9%) | Added `--max-danger-prob` guardrail in 03_predict_temporal.py |
| 2026-01-27 | Trinity mode temporal leakage | Model 4.61x lift with `--disable-trinity` (was 1.15x) |
| 2026-01-27 | EU market cap API 99% fail | `--skip-market-cap-api` + 40-day ADV fallback |
| 2026-01-26 | Stop loss used close price | Use `low` price in 00b_label_outcomes.py:487 |
| 2026-01-26 | Untradeable patterns not filtered at prediction | Filter `untradeable=True` in 03_predict_temporal.py |
| 2026-01-26 | Single-day volume spike qualified TARGET | Require 3 consecutive days with both 2x surge AND $50k |
| 2026-01-26 | No absolute volume floor | MIN_DOLLAR_VOLUME_PER_DAY = $50k per day |
| 2026-01-26 | Metadata-HDF5 label alignment not verified | Explicit check in 02_train_temporal.py (Check 4) |
| 2026-01-25 | Box width effect validated | 1.54x lift for tight patterns (use risk_width_pct < 0.05) |
| 2026-01-25 | V24 models archived | Moved to archive_v24_no_signal/ (0.98x lift = no signal) |
| 2026-01-25 | 74x row duplication in V24 weekly | Mandatory DataIntegrityChecker in pipeline |
| 2026-01-25 | VIX filter lift overstated (1.60x) | Corrected to 1.05x (not significant) |
| 2026-01-25 | No deduplication enforcement | auto-blocks if ratio > 1.1x |
| 2026-01-25 | No temporal integrity checks | auto-blocks if val dates <= train max |
| 2026-01-25 | Feature leakage undetected | auto-detects outcome columns in features |
| 2026-01-24 | Fixed 40-day window suboptimal | V22: Dynamic window = 1/volatility (10-60 days) |
| 2026-01-24 | Day-by-day labeling loop slow | V22: Vectorized masks + pessimistic tie-breaker |
| 2026-01-24 | No look-ahead bias tests | tests/test_lookahead.py (11 tests) |
| 2026-01-22 | Trinity + temporal split = 99.7% train | `--disable-trinity` uses pattern-level splits |
| 2026-01-22 | Windows wildcard not expanded | glob.glob() before h5py.File() |
| 2026-01-22 | NMS cluster_id only for 'trinity' | Set cluster IDs for ALL NMS modes |
| 2026-01-22 | ETFs have identical patterns | Cross-ticker deduplication by signature |
| 2026-01-22 | sequence_idx column all zeros | Use np.where(mask)[0] for row indices |
| 2026-01-21 | Static pacing ignores intraday profile | V20 dynamic pacing (1.5x time-weighted) |
| 2026-01-21 | V18 too complex for ablation | V20 ablation architecture (4 modes) |
| 2026-01-21 | ADV 10% too loose | Tightened to 4% |
| 2026-01-21 | Zombie -> Danger wrong | Reclassified to Noise (insufficient data != failure) |
| 2026-01-21 | NO_FILL gaps deleted | Label normally + untradeable flag |
| 2026-01-21 | Volume ratios NaN/Inf | log_diff(num, denom) |
| 2026-01-21 | Hardcoded [3,7,8,10,16] | config/feature_registry.py |
| 2026-01-21 | Missing vol_trend_5d (idx 12) | Added to VOLUME_RATIO_INDICES |
| 2026-01-21 | float_turnover redundant | retention_rate (candle strength) |
| 2026-01-21 | Dormant filtered in training | --mode training preserves |
| 2026-01-21 | Volume ratios saturate | Log1p indices [3,7,8,10,12,16] |
| 2026-01-21 | Context not scaled | Robust scaling from C_train |
| 2026-01-21 | EOD 2x volume impossible | Intraday Pacing @ 10AM |
| 2026-01-21 | Volume in labeling | V19: execution rule only |
| 2026-01-20 | No exit strategy | Justice Exit Plan |
| 2026-01-20 | No position sizing | Risk-Based + ADV clamp |
| 2026-01-20 | 100d labeling unrealistic | 40d + structural R |
| 2026-01-19 | Zombie patterns stuck | ZOMBIE_TIMEOUT = 150 |
| 2026-01-18 | 71% pattern overlap | NMS + Physics filters |

---

## V24 Postmortem

V24 ML models and VIX filter strategy were **discarded** after rigorous temporal validation:

| Approach | Claimed Lift | Actual Lift (Temporal) | Status |
|----------|--------------|------------------------|--------|
| V24 ML Model | 6.67x | **0.98x** | NO SIGNAL |
| VIX >= 20 Filter | 1.60x | **~1.05x** | NOT SIGNIFICANT |

**Root Causes:**
- ML lift was due to look-ahead bias from random train/test splitting
- VIX filter lift was inflated by 74x row duplication bug
- Neither approach generalizes across market regimes

**Archived to:** `output/models/archive_v24_no_signal/`

---

## Trinity Split Fix

**Problem:** Trinity mode clusters patterns by NMS, but clusters span multiple years.

| Split | Trinity Mode (BROKEN) | Pattern-Level (FIXED) |
|-------|----------------------|----------------------|
| Train | 29,394 (99.7%) | 17,419 (59.1%) |
| Val | 44 (0.1%) | 3,090 (10.5%) |
| Test | 45 (0.2%) | 8,974 (30.4%) |

**Solution:** **ALWAYS use `--disable-trinity` for training.**

**Model Performance Comparison:**
| Model | Trinity Mode | Lift |
|-------|--------------|------|
| Jan 25 | Enabled (BROKEN) | 1.15x |
| Jan 27 | Disabled (FIXED) | **4.61x** |
