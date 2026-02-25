# CLAUDE.md

## System Identity

**TRAnS** - **T**emporal **R**etail **An**alysis **S**ystem for micro/small-cap consolidation breakouts.

| Parameter | Value |
|-----------|-------|
| Account Size | $10,000 - $100,000 |
| Risk Per Trade | $250 (fixed R-unit) |
| Max Position | $5,000 or 4% ADV |
| Execution | Manual/semi-automated, EOD |

**Status**: Production EU | **EU Lift**: 2.7x | **Labeling**: V24 (Triple Barrier) | **Model**: Jan 30, 2026

**Note**: This directory is NOT a git repository. No version control commits possible here.

---

## Quick Reference

```bash
# FULL PIPELINE WITH TRIPLE BARRIER LABELING (RECOMMENDED)
python main_pipeline.py --region EU --step detect --tickers data/eu_tickers.txt
python pipeline/00c_label_triple_barrier.py --input output/candidate_patterns.parquet --dynamic-barriers
python main_pipeline.py --region EU --step generate --apply-nms --apply-physics-filter --mode training
python main_pipeline.py --region EU --step train --epochs 100 --use-coil-focal --disable-trinity

# LEGACY PIPELINE (V22 Structural Risk Labeling)
python main_pipeline.py --region EU --full-pipeline --apply-nms --apply-physics-filter \
    --mode training --use-coil-focal --epochs 100 --skip-market-cap-api --disable-trinity

# NIGHTLY ORDERS
python pipeline/03_predict_temporal.py --model output/models/production_model.pt \
    --sequences output/sequences_latest.h5 --metadata output/metadata_latest.parquet \
    --risk-unit 250 --max-capital 5000 --min-signal GOOD --max-danger-prob 0.25

# TESTS
python -m pytest tests/ -v
python -m pytest tests/test_triple_barrier.py -v   # Triple Barrier + Dynamic Barriers
python -m pytest tests/test_lookahead.py -v        # Look-ahead bias
python -m pytest tests/test_data_integrity.py -v   # 74x bug prevention
```

---

## Triple Barrier Labeling (V24) - Primary Method

The Triple Barrier Method (Lopez de Prado, 2018) labels patterns using **relative returns** with three barriers:

| Barrier | Description | Outcome |
|---------|-------------|---------|
| **Upper** | Profit-taking threshold hit | Label 2 (Target) |
| **Lower** | Stop-loss threshold hit | Label 0 (Danger) |
| **Vertical** | Time limit (150 days) expires | Label 1 (Noise) |

### Key Parameters

```python
# Triple Barrier Configuration
FEATURE_WINDOW_SIZE = 250      # Days of history for feature calculation
VERTICAL_BARRIER_DAYS = 150    # Maximum holding period (labeling window)

# CRITICAL: All rolling windows use closed='left'
# This ensures features use ONLY data BEFORE the current observation
# Example: rolling(20, closed='left') at day T uses days [T-20, T)
```

### Dynamic Barrier Adjustment

Barriers are adjusted based on **market cap tier** and **market regime** at pattern time:

#### By Market Cap Tier

| Tier | Upper (Target) | Lower (Stop) | Min Days | Rationale |
|------|----------------|--------------|----------|-----------|
| **Nano** | +12% | -6% | 3 | Extreme volatility, high reward potential |
| **Micro** | +8% | -4% | 5 | High volatility, 40%+ moves possible |
| **Small** | +5% | -2.5% | 7 | Balanced risk/reward |
| **Mid** | +4% | -2% | 10 | Lower volatility, tighter execution |
| **Large** | +3% | -1.5% | 14 | Lowest volatility, consistent |

#### By Market Regime

| Regime | Upper Mult | Lower Mult | Effect |
|--------|------------|------------|--------|
| **Bullish** | 1.2x | 0.8x | Extend targets, tighten stops (risk-on) |
| **Neutral** | 1.0x | 1.0x | Base parameters |
| **Bearish** | 0.7x | 1.3x | Reduce targets, widen stops (risk-off) |
| **High Vol** | 0.8x | 1.5x | Wider stops essential |
| **Crisis** | 0.5x | 2.0x | Capital preservation mode |

#### ADV-Based Market Cap Classification

When market cap API is unavailable (common for EU stocks), ADV is used:

| ADV Range | Classification |
|-----------|----------------|
| < $100K | Nano |
| $100K - $500K | Micro |
| $500K - $5M | Small |
| $5M - $50M | Mid |
| > $50M | Large |

#### VIX-Based Regime Detection

| VIX Level | Regime |
|-----------|--------|
| < 12 | Bullish |
| 12 - 18 | Neutral |
| 18 - 25 | Bearish |
| 25 - 35 | High Volatility |
| > 35 | Crisis |

### Usage Examples

```bash
# Dynamic barriers (RECOMMENDED) - auto-adjusts per market cap + regime
python pipeline/00c_label_triple_barrier.py \
    --input output/candidate_patterns.parquet \
    --dynamic-barriers

# With explicit market cap and VIX columns
python pipeline/00c_label_triple_barrier.py \
    --input output/candidate_patterns.parquet \
    --dynamic-barriers \
    --market-cap-col market_cap \
    --vix-col vix_at_pattern

# Fixed barriers (simple, no adjustment)
python pipeline/00c_label_triple_barrier.py \
    --input output/candidate_patterns.parquet \
    --upper-barrier 0.05 --lower-barrier 0.03

# With volatility scaling (barriers = volatility × multiplier)
python pipeline/00c_label_triple_barrier.py \
    --input output/candidate_patterns.parquet \
    --volatility-scaling --volatility-multiplier 2.0

# Create strict temporal splits
python pipeline/00c_label_triple_barrier.py \
    --input output/candidate_patterns.parquet \
    --dynamic-barriers \
    --create-splits --train-end 2022-06-30 --val-end 2023-06-30 --gap-days 7
```

### Example Calculations

**Micro-cap in Bullish Regime:**
```
Base barriers: +8% target, -4% stop
Bullish adjustment: 1.2x upper, 0.8x lower
Final: +9.6% target, -3.2% stop
```

**Small-cap in Crisis Regime:**
```
Base barriers: +5% target, -2.5% stop
Crisis adjustment: 0.5x upper, 2.0x lower
Final: +2.5% target, -5.0% stop (capital preservation)
```

---

## Pipeline Overview

```
00_detect_patterns.py        → candidate_patterns.parquet
00c_label_triple_barrier.py  → labeled_patterns_triple_barrier.parquet (V24: Triple Barrier)
01_generate_sequences.py     → sequences.h5 + metadata.parquet
02_train_temporal.py         → best_model.pt
03_predict_temporal.py       → nightly_orders.csv
```

### Labeling Options

| Script | Method | Window | Based On |
|--------|--------|--------|----------|
| `00c_label_triple_barrier.py` | **V24 Triple Barrier** | 150 days | Relative returns + Dynamic barriers |
| `00b_label_outcomes.py` | V22 Structural Risk | 10-60 days | R-multiples (legacy) |

---

## Temporal Integrity Requirements

### Strict `closed='left'` Rolling Windows

All feature calculations use `rolling(window, closed='left')` to ensure **strictly historic** data:

```python
# CORRECT: closed='left' excludes current observation
sma_20 = df['close'].rolling(20, closed='left').mean()
# At index i, uses data from [i-20, i) - excludes day i

# WRONG: Default includes current observation (look-ahead bias!)
sma_20 = df['close'].rolling(20).mean()
# At index i, uses data from [i-19, i] - includes day i
```

### Strict Temporal Train/Test Splits

No overlapping dates between splits:

```python
from utils.triple_barrier_labeler import create_strict_temporal_split

train_df, val_df, test_df = create_strict_temporal_split(
    df=patterns_df,
    date_col='end_date',
    train_end='2022-06-30',
    val_end='2023-06-30',
    gap_days=7  # Optional buffer between splits
)
```

---

## Critical Files

| File | Purpose |
|------|---------|
| `pipeline/00c_label_triple_barrier.py` | **V24 Triple Barrier labeling pipeline** |
| `utils/triple_barrier_labeler.py` | Triple Barrier implementation |
| `config/barrier_config.py` | Dynamic barrier configuration |
| `features/vectorized_calculator.py` | Feature calc with `closed='left'` |
| `main_pipeline.py` | Unified orchestrator |
| `models/temporal_hybrid_v18.py` | Model architecture (V18 CNN + GRN) |
| `models/temporal_hybrid_unified.py` | Unified model with ablation modes |
| `output/models/production_model.pt` | Production weights |
| `config/feature_registry.py` | Feature indices (no hardcoding!) |
| `utils/data_integrity.py` | 74x bug prevention |

---

## Essential CLI Flags

| Flag | Description |
|------|-------------|
| `--dynamic-barriers` | **RECOMMENDED** - Adjust barriers per market cap + regime |
| `--apply-nms` | **REQUIRED** - Remove 71% pattern overlap |
| `--apply-physics-filter` | **REQUIRED** - Remove untradeable patterns ($50k liquidity floor) |
| `--mode training` | Preserves dormant stocks |
| `--use-coil-focal` | **REQUIRED** - Coil-Aware Focal Loss |
| `--disable-trinity` | **REQUIRED FOR TRAINING** - Pattern-level temporal splits |
| `--max-danger-prob 0.25` | **RECOMMENDED** - Reject patterns with P(Danger) > 25% |

---

## Key Constants

```python
# Triple Barrier (V24) - PRIMARY
FEATURE_WINDOW_SIZE = 250      # Days of history for features
VERTICAL_BARRIER_DAYS = 150    # Maximum holding period
# Dynamic barriers calculated per pattern based on market cap + regime

# Structural Risk (V22) - LEGACY
MIN_OUTCOME_WINDOW = 10        # High-vol stocks
MAX_OUTCOME_WINDOW = 60        # Low-vol stocks
TARGET_R_MULTIPLE = 3.0
GAP_LIMIT_R = 0.5

# Liquidity Floor
MIN_DOLLAR_VOLUME = 50000      # $50k minimum avg dollar volume

# Execution
RISK_UNIT_DOLLARS = 250.0
MAX_CAPITAL_PER_TRADE = 5000.0
ADV_LIQUIDITY_PCT = 0.04
MAX_DANGER_PROB = 0.25         # Guardrail: reject if P(Danger) > 25%

# Labels: 0=Danger, 1=Noise, 2=Target
```

---

## Data Integrity (Mandatory)

Pipeline auto-blocks if:
- Duplication ratio > 1.1x
- Val dates <= train max (temporal leakage)
- Outcome columns in features
- Rolling windows not using `closed='left'`

```python
from utils.data_integrity import DataIntegrityChecker
checker = DataIntegrityChecker(df, date_col='pattern_end_date', strict=True)
```

---

## EU Exchange Performance

| Exchange | Patterns | Target Rate | Recommendation |
|----------|----------|-------------|----------------|
| Denmark (.CO) | 313 (6%) | **7.7%** | FOCUS |
| Italy (.MI) | 458 (9%) | **5.5%** | FOCUS |
| Netherlands (.AS) | 165 (3%) | 4.8% | Good |
| Sweden (.ST) | 522 (10%) | 4.2% | Good |
| Germany (.DE) | 1,286 (25%) | 1.9% | Caution (dominates data) |
| France (.PA) | 934 (18%) | 2.4% | Caution |
| UK (.L) | 27 (0.5%) | 0% | Low quality |

---

## Key Findings

| Finding | Value | Action |
|---------|-------|--------|
| Triple Barrier V24 | 250d features, 150d window | **PRIMARY labeling method** |
| Dynamic barriers | Market cap + regime adjusted | Micro-caps get wider stops/targets |
| `closed='left'` | Strict temporal integrity | **MANDATORY** for all rolling windows |
| Jan 30 model (EU) | 8% Top-15 (2.7x lift) | Current production model |
| Deduplication | 73.5% removed | **MANDATORY** (always applied) |
| Liquidity floor | $50k min | Balances alpha validation vs data sparsity |
| Danger filter 25% | Removes 58% bad | `--max-danger-prob 0.25` |

---

## Known Limitations

| Limitation | Mitigation |
|------------|------------|
| Dormant volume ratios | Log-diff transformation |
| Broker slippage | 0.5R gap limit + untradeable filter |
| Can't short | Long-only focus |
| Intraday stop breach | Use `low` price for stop check (not close) |
| EU market cap APIs fail 99% | ADV-based classification fallback |
| High P(Danger) in GOOD signals | `--max-danger-prob 0.25` guardrail |
| No git version control | This directory is not a git repo |

---

## Notes

- **Triple Barrier V24** - Primary labeling with 250d features, 150d window, dynamic barriers
- **Dynamic barriers** - Micro-caps: +8%/-4%, Small-caps: +5%/-2.5%, regime-adjusted
- **`closed='left'`** - **MANDATORY** for all rolling window calculations
- **Deduplication** - **MANDATORY** (73.6% redundant patterns removed)
- **Liquidity floor** - $50k minimum dollar volume
- **Feature indices** - always import from `config.feature_registry`
- **Model imports** - use `models.temporal_hybrid_unified`
- **Retrain** every 90 days or PSI > 0.25
- **Danger filter** - use `--max-danger-prob 0.25` for production

---

## Extended Documentation

See `docs/` for detailed information:
- `docs/SYSTEM_OVERVIEW.md` - Complete system description
- `docs/ARCHITECTURE.md` - Model architecture and feature schema
- `docs/BUG_FIX_HISTORY.md` - Historical fixes and lessons learned
- `docs/DATA_LEAKAGE_GUIDE.md` - Four deadly sins of leakage
- `docs/VERIFICATION_LOG.md` - Implementation verification details
