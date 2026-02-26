# CLAUDE.md — geldbeutel V2.0

## System Identity

**geldbeutel V2.0** — Thermodynamic Swing Liquidity Sweep Engine for MNQ/NQ futures.

| Parameter | Value |
|-----------|-------|
| Instrument | MNQ (Micro E-mini Nasdaq 100) |
| Timeframe | 1-minute bars, 24/5 session |
| Data Source | CME NQ unadjusted futures (continuous front-month, raw) |
| Structural Scan | Every `STRUCTURAL_RESOLUTION` bars (default: 5 = effective 5-min) |
| Tick Size | 0.25 points |
| Point Value | $2.00 per point per contract |
| Risk Per Trade | 1% of equity |
| Max Contracts | 20 (hard margin cap) |
| Optimization | Mixed-Integer GA (7 genes) inside Rolling Walk-Forward |

**Status**: Kernel built. Data loader ready. Awaiting CME 1-min data source.

**Predecessor**: V1.0 (sealed, archived) — Calmar 1.12 OOS on 9-year CFD data.

never present fabricated data

---

## Architectural Changes from V1.0

Blueprint source: `Desktop/notes/2terfix.txt` — 5-step V2 architecture.

### Step 1 — True OOS Isolation (Blueprint bug fixed)
`oos_start_idx` and `initial_capital` are passed directly into Numba. Phase 1 & 2
(execution) are gated: `if t >= oos_start_idx`. **Phase 3 (structural mapping) runs
continuously through the IS overhang**, ensuring the state machine has full structural
memory when OOS begins. The blueprint's `continue` placement would have skipped
Phase 3 — this is corrected: the execution lock uses a `phase3_skip` flag rather
than an early `continue`.

### Step 2 — Structural Regime Gate (No Indicators)
Tracks the outcome of the **last two macro sweeps** per side:
- `1` = reclaim success, `-1` = depth death, `0` = no prior outcome
- Two consecutive depth deaths on the long side → `long_locked = True`
- The long side is unlocked when a **short sweep succeeds** (reclaims)
- Symmetrical for shorts
- No ATR, no MA, no phenomenological filter — pure price-physics memory

### Step 3 — Thermodynamic Volume Friction
Requires `cumulative sweep volume > vol_mult × rolling_median_vol[t]` before a
pending signal is generated. Real stop-runs produce forced-liquidation volume
spikes. `vol_mult` is the 7th gene, evolved by the GA. `rolling_median_vol` is
pre-computed in `data_loader_v2.py` (200-bar rolling median).

### Step 4 — Roll Date Handling + Swing Gene Bounds
- `is_roll_date` boolean array passed into Numba
- On roll: close open position at open price ± slippage, zero all structural memory
- Gene bounds rescaled from V1.0 intraday to V2.0 swing dimension:

| Gene | V1.0 (5-min CFD) | V2.0 (1-min CME swing) | Physical Meaning |
|------|-----------------|----------------------|-----------------|
| `w` | [72, 576] bars | **[2000, 8000]** bars | 1.4–5.6 days macro lookback |
| `m` | [6, 144] bars | **[400, 2760]** bars | 6.7h–2 day level maturity |
| `d_min` | [0.02, 0.50]% | **[0.10, 1.50]%** | 20–300 pt structural sweep (NQ@20k) |
| `d_max` | [0.10, 2.00]% | **[1.00, 15.00]%** | depth-death ceiling (avoids false lockouts) |
| `v` | [1, 10] checks | **[2, 200]** checks | VACUUM duration tolerance |
| `beta` | [1.0, 8.0] R | **[3.0, 15.0]** R | absorbs overnight gap risk |
| `vol_mult` | N/A (new) | **[0.2, 1.5]** | volume friction multiplier (thin modern NQ book) |

### Step 5 — Pure Calmar Fitness with Hard Gates
```
fitness = net / dd   (if feasible, else -999999)
```
- Hard gate: `min_trades = 15` — statistical significance on 12-month IS
- Hard gate: `min_avg_hold = 500 bars` (~8h) — eliminates scalp-degenerate genomes
- Hard gate: `IS Calmar ≥ 0.5` — rejects low-edge genomes with positive net but weak risk-adjusted return
- Pure Calmar `net/dd` within the feasible region — honest edge per unit risk
- Mark-to-market uses true intrabar MAE: `equity[t] = capital + (q_low[t] - entry_px) × size × PT_VAL` for longs
- V1.0 validated: simple Calmar achieved Calmar 1.12 OOS on 9-year CFD data

---

## The 7-Dimensional Genome

| Gene | Symbol | Type | Range | Physical Meaning |
|------|--------|------|-------|-----------------|
| Macro Lookback | `w` | int | [2000, 8000] | Structural scope (1.4–5.6 days) |
| Anchor Maturity | `m` | int | [400, 2760] | Level must age (stops accumulate) |
| Min Sweep Depth | `d_min` | float | [0.10, 1.50]% | Minimum institutional capitulation (20 pts at NQ@20k) |
| Max Excursion | `d_max` | float | [1.00, 15.00]% | Maximum before hypothesis dies |
| Reclaim Tolerance | `v` | int | [2, 200] | Structural checks in VACUUM |
| Reward Asymmetry | `beta` | float | [3.0, 15.0] | R-multiple for take-profit |
| Volume Gate | `vol_mult` | float | [0.2, 1.5] | Sweep vol vs median (thin post-2022 NQ book) |

**Constraints**: `w > m`, `d_min < d_max`, `v >= 2`, `beta > 1.0`, `vol_mult > 0`. Violations return -999999.

---

## Ensemble OOS Deployment (V5+)

Instead of deploying a single apex genome per WFO window, the GA returns an ensemble
of `ENSEMBLE_SIZE=3` diverse genomes from a hall-of-fame accumulated across all generations.

| Parameter | Value |
|-----------|-------|
| `ENSEMBLE_SIZE` | 3 (genomes deployed per OOS window) |
| `DIVERSITY_MIN_GENES` | 3 (must differ on >= 3 of 7 genes) |
| `DIVERSITY_THRESHOLD_PCT` | 0.15 (each differing gene >= 15% of range) |

**Hall-of-Fame**: Every feasible genome across all 50 GA generations is accumulated,
deduplicated using gene-distance (not float epsilon — prevents near-clone flooding),
and the top 50 are retained. The ensemble is greedy-selected:
apex first, then the highest-fitness genome passing the diversity filter against all
already-selected members. Fallback pads with best-fitness non-clones if diversity
filter yields fewer than N.

**Ensemble Equity**: Each genome receives full `compounding_capital`. The ensemble
equity curve is the element-wise mean of N individual OOS equity curves. This is
mathematically equivalent to 1/N capital allocation while preserving position sizing.

**`run_ga()` return contract**: `(ensemble_list, best_ever_fitness, gen_log)` where
`ensemble_list = [(genome_dict, fitness), ...]` of length `ENSEMBLE_SIZE`.

---

## Architecture

```
geldbeutel/V2.0/
├── wfo_matrix_v2.py          # Sealed kernel + GA + Rolling WFO (main entry point)
├── data_loader_v2.py         # CME data ingestion + roll detection + volume prep
├── NQ_CME_1min.parquet       # Prepared dataset (generated by data_loader_v2.py)
├── wfo_equity_curve_v2.parquet   # OOS compounding curve (generated by WFO)
├── wfo_window_results_v2.csv     # Per-window results (generated by WFO)
└── CLAUDE.md                 # This file
```

---

## Performance Characteristics

With `STRUCTURAL_RESOLUTION = 5` (default):
- Phase 3 fires every 5th 1-min bar → effective 5-min structural detection
- 1-min execution fidelity in Phase 1/2 is preserved
- Estimated WFO runtime: **25–40 hours** on modern CPU (24/12 windows)
- Per-genome evaluation: ~0.5–1.5 seconds (vs. ~0.007s for V1.0)
- Set `STRUCTURAL_RESOLUTION = 15` for ~15-min structural scanning (~3× faster)
- Set `STRUCTURAL_RESOLUTION = 1` for full 1-min structural fidelity (~5× slower)

**Root cause of cost**: `w` up to 20,000 bars means the pristine-scan inner loop
(`for i in range(min_idx+1, t)`) scans up to 20,000 elements per structural check.

---

## Data Pipeline

### CME NQ 1-Minute Data Requirements
- **Instrument**: NQ front-month futures (unadjusted, continuous)
- **Resolution**: 1-minute OHLCV bars
- **Horizon**: Minimum 10 years recommended (WFO needs 24mo IS + 12mo OOS per window)
- **Timestamps**: Any timezone (loader converts to US/Eastern)
- **Volume**: Must be real CME exchange volume (contracts/bar), not tick count
- **Format**: CSV or Parquet with standard OHLCV columns

### Why Unadjusted (Not Back-Adjusted)
Back-adjustment modifies historical prices to remove roll gaps. On NQ with ~50-point
typical roll gaps, back-adjustment can push 2008–2010 prices to negative territory.
This destroys percentage-based `d_min` math. Unadjusted data preserves real price
levels; roll dates are handled surgically inside the kernel via `is_roll_date`.

### Data Preparation
```bash
# Prepare CME data (CSV input)
python data_loader_v2.py --input path/to/nq_1min.csv

# Prepare with explicit timezone
python data_loader_v2.py --input path/to/nq_1min.csv --timezone America/New_York

# Validate an existing prepared dataset
python data_loader_v2.py --test

# Custom output path
python data_loader_v2.py --input data.csv --output NQ_CME_1min.parquet
```

### Manual Roll Date Override
If the automatic roll detection misses or double-detects rolls, create
`roll_dates_override.csv` in V2.0/ with a single column `date` (YYYY-MM-DD):
```csv
date
2024-03-14
2024-06-13
2024-09-12
2024-12-12
```
This completely replaces gap-detection (Method B) for any dates listed.

---

## Walk-Forward Optimization Protocol

| Parameter | Value |
|-----------|-------|
| Training window | 24 months |
| Trading window | 12 months (true OOS) |
| Overhang | `w` bars pre-OOS for structural memory warm-up |
| GA population | 80 genomes |
| GA generations | 50 |
| Selection | Tournament (k=3) |
| Crossover | Simulated Binary (SBX, eta=2) |
| Mutation | Polynomial (15% rate) + integer snapping |
| Elitism | Top 10% carry forward |
| Min trades for fitness | 15 |
| OOS stitching | Compounding (scaled to running capital) |

---

## Development Commands

```bash
# Install dependencies
pip install numpy pandas numba pyarrow

# Step 1: Prepare CME data
python data_loader_v2.py --input your_nq_1min_data.csv

# Step 2: Validate prepared data
python data_loader_v2.py --test

# Step 3: Numba warmup + single genome test
python -c "
from wfo_matrix_v2 import *
import numpy as np
n = 5000
d = np.ones(n)
m = np.full(n, 600, dtype=np.int64)
v = np.ones(n) * 1000.0
mv = np.ones(n) * 800.0
r = np.zeros(n, dtype=np.bool_)
eq, tr, bars = swing_liquidity_sweep_engine(d,d,d,d,m,v,mv,r,100000.0,0,3000,500,0.3,2.0,20,5.0,1.5,5)
print(f'Bars: {len(eq)}, Trades: {tr}, Bars in trade: {bars}')
"

# Step 4: Full WFO (12–24h runtime)
python wfo_matrix_v2.py
```

---

## Output Files

| File | Contents |
|------|----------|
| `NQ_CME_1min.parquet` | Prepared 1-min OHLCV + is_roll_date + rolling_median_vol |
| `wfo_equity_curve_v2.parquet` | Global OOS compounding equity curve |
| `wfo_window_results_v2.csv` | Per-window: genome, fitness, OOS net, DD, Calmar, trades, avg_hold |

---

## Key Performance Metrics

1. **OOS Calmar**: net / max_drawdown on unseen data (target > 1.0)
2. **Profitable Windows**: % of WFO windows with positive OOS (target > 55%)
3. **Avg Hold Time**: bars in trade per trade (swing should be 500–5000 bars = 8–83h)
4. **Regime Gate Activation Rate**: % of windows where long or short lock triggers
5. **Volume Gate Hit Rate**: % of valid sweeps that pass the vol_mult filter

---

## Known Constraints

| Constraint | Mitigation |
|------------|------------|
| CME data source not included | data_loader_v2.py accepts any CSV/Parquet; see Data Requirements |
| 12–24h WFO runtime | STRUCTURAL_RESOLUTION tunable; start with 15 for rapid prototyping |
| Unadjusted data has roll gaps | is_roll_date zeros structural memory; positions closed on roll |
| Low swing trade frequency (~20–40/yr) | Fitness min_trades=15; IS Calmar≥0.5 gate; Calmar×√trades respects low-frequency edge |
| Regime gate can over-lock in trending markets | Gate resets on opposite-side success; self-correcting over time |

---

## Relationship to Other Systems

| System | Location | Interaction |
|--------|----------|-------------|
| **V1.0** | `geldbeutel/V1.0/` | Archived predecessor. Calmar 1.12 OOS. Different data/gene space. |
| **TRAnS** | `vonLinck Capital/trans/` | Independent — EU equities, daily TF |
| **AIv3** | `AI Infra/` root | Independent — US micro-cap equities |
| **TTDLS Theory** | `Desktop/notes/Theorie .txt` | V2.0 is the swing implementation of this theory |
