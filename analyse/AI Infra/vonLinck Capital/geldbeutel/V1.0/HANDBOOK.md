# geldbeutel V1.0 — Operator Handbook

## What This System Does

This system hunts for **liquidity sweep reclaims** on MNQ/NQ futures using 5-minute bars. It detects when institutional capital sweeps resting stop-loss clusters below structural support (or above resistance), absorbs the selling, and reclaims the level — trapping breakout traders offside. It enters on the reclaim, stops at the sweep wick extreme, and targets a multiple of the risk.

The system does **not** predict price direction. It identifies mechanical order-flow events where trapped capital provides forced-covering fuel.

---

## How To Run It

### Step 1: Download Data

```bash
cd vonLinck Capital/geldbeutel/V1.0/
python dukascopy_downloader.py
```

Downloads 5-minute tick data from Dukascopy's free API (Nasdaq 100 CFD), aggregates to 5-min OHLCV bars, saves to `NQ_CFD_5min_2017_2026.parquet`. Takes 60–90 minutes. Checkpoints every 30 days — safe to interrupt and resume.

### Step 2: Run the Walk-Forward Optimization

```bash
python wfo_matrix.py
```

Trains a Genetic Algorithm on 6-month rolling windows, then trades the next 3 months blind with the locked genome. Stitches all out-of-sample segments into a compounding equity curve.

**Output:**
- `wfo_equity_curve.parquet` — the global OOS equity curve
- `wfo_window_results.csv` — per-window breakdown (genome, P&L, drawdown, Calmar)

### Step 3: Read the Results

```python
import pandas as pd

# Equity curve
eq = pd.read_parquet("wfo_equity_curve.parquet")
print(eq.describe())

# Window-by-window
wr = pd.read_csv("wfo_window_results.csv")
print(wr[['window','oos_net','oos_dd','oos_calmar','oos_trades']].to_string())
```

---

## What To Look For In Results

### Good Signs
- **OOS Calmar > 1.0** across most windows
- **Profitable windows > 55%** of total
- **Genome stability**: `w`, `d_min`, `d_max` cluster within a regime (not random noise)
- **Trade count > 30** per IS window (statistical significance)
- **Max DD < 15%** of peak equity on the global curve

### Bad Signs
- **All genomes hit -999999 fitness**: Gene bounds may be too narrow for current regime, or data is insufficient
- **0 trades in OOS**: Genome is over-fit to training noise — too restrictive
- **Genome parameters wildly different each window**: Market structure is non-stationary beyond what the WFO can track
- **Calmar < 0 in majority of windows**: No edge exists in this parameter space for current data

---

## The 6 Genes — What They Mean In Plain English

| Gene | Plain English | Tuning Intuition |
|------|--------------|------------------|
| **w** (lookback) | How far back to scan for structural levels | Larger = bigger levels with more stops, but fewer signals |
| **m** (maturity) | How old a level must be before it counts | Too low = chasing noise levels. Too high = missing fresh setups |
| **d_min** (min depth) | How far price must pierce below the level | Too low = false triggers from noise. Too high = missing valid sweeps |
| **d_max** (max depth) | When to give up — the level is truly broken | Too low = premature invalidation. Too high = catching falling knives |
| **v** (velocity) | How many bars price can stay below before abort | v=1 = ultra-aggressive (must reclaim immediately). v=5 = patient |
| **beta** (reward) | How many R-multiples for the take-profit target | Low beta = high win rate, small wins. High beta = low win rate, big wins |

---

## The Execution Flow — What Happens On Each Bar

```
Bar t opens
│
├─ Any pending order from yesterday?
│  ├─ Both long AND short pending? → Abort both (chaotic node)
│  ├─ Long pending: Does open[t] >= level? → Execute with slippage
│  └─ Short pending: Does open[t] <= level? → Execute with slippage
│
├─ Currently in a trade?
│  ├─ Check gap through stop → exit at open with slippage
│  ├─ Check gap through target → exit at open (no slippage on limits)
│  ├─ Check wick hit stop → exit at stop with slippage
│  ├─ Check wick pierced through target → exit at target
│  └─ No fill → mark-to-market, skip signal generation
│
├─ Flat — scan for new setups
│  ├─ LONG: Find lowest pristine pivot in [t-w, t-m]
│  │  └─ Did price breach it today AND was above it yesterday?
│  │     └─ Enter VACUUM state, track sweep extreme
│  │
│  ├─ In VACUUM? Check gates:
│  │  1. Swept too deep? → abort
│  │  2. Stayed below too long? → abort
│  │  3. Reclaimed the level? → signal if RTH and depth valid
│  │  4. None of above → increment timer
│  │
│  └─ SHORT: Mirror of above with highs
│
└─ Record equity[t]
```

---

## Risk Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Risk per trade | 1% of equity | Standard Kelly fraction for uncertain edge |
| Stop placement | 1 tick below sweep extreme | If this level fails, the thesis is dead |
| Slippage budget | 2 ticks per fill | Pessimistic for MNQ; absorbs CFD tracking error |
| Max contracts | 20 | Hard margin cap prevents over-leverage |
| Min risk distance | 2.0 points | Prevents sub-spread micro-scalps |

Position size is calculated as:
```
qty = int(equity * 0.01 / (risk_pts * $2.00))
```

Where `risk_pts = (entry + slippage) - (E_min - 1 tick) + slippage`.

---

## Data Considerations

### Why Dukascopy CFD Data Works

The system was designed to survive on free, imperfect data:

1. **Pre-quantization** snaps all prices to the MNQ 0.25 tick grid, destroying CFD pricing noise
2. **2-tick slippage** per fill absorbs the CFD-to-CME spread differential
3. **d_min depth gate** ensures only sweeps deep enough to dwarf the 1–2 point tracking error are tradeable

If an edge survives these penalties, it will perform better on real CME data.

### Data Gaps
- Dukascopy has no data on weekends or during market holidays — this is correct behavior
- Some hours may have zero ticks (low liquidity ETH sessions) — these produce no 5-min bars
- The downloader checkpoints every 30 days and can resume from interruption

---

## Walk-Forward Optimization Explained

```
Time ─────────────────────────────────────────────────────►

Window 0:  [====TRAIN 6mo====][==TRADE 3mo==]
Window 1:        [====TRAIN 6mo====][==TRADE 3mo==]
Window 2:              [====TRAIN 6mo====][==TRADE 3mo==]
Window 3:                    [====TRAIN 6mo====][==TRADE 3mo==]
                                                    ...

Only TRADE segments are stitched into the equity curve.
Each TRAIN segment produces one apex genome via GA.
The apex genome is LOCKED during its TRADE segment.
```

The **overhang** trick: At the start of each TRADE segment, the kernel needs `w` bars of lookback to map structural levels. These bars come from the end of the training period. The kernel processes them but execution is locked until the true OOS start.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: numba` | Numba not installed | `pip install numba` |
| Download stalls at X% | Dukascopy rate limit or network issue | Script auto-retries. Kill and rerun — checkpoints resume |
| `NQ_CFD_5min_2017_2026.parquet not found` | Data not downloaded yet | Run `python dukascopy_downloader.py` first |
| All genomes return -999999 | Gene bounds too narrow or not enough bars | Check data has 500+ bars per IS window |
| 0 trades in OOS window | Genome too restrictive | Widen gene bounds or increase `pop_size` |
| Numba first-call slow (30s+) | JIT compilation | Normal. Only happens once per session |
| MemoryError | Too many bars in memory | Should not happen — 600k float64 rows = ~30 MB |

---

## Modifying The System

### Changing Gene Bounds

Edit `GENE_BOUNDS` in `wfo_matrix.py`:

```python
GENE_BOUNDS = {
    'w':     (72,   576,   'int'),     # Lookback range in bars
    'm':     (6,    144,   'int'),     # Anchor maturity range
    'd_min': (0.02, 0.50,  'float'),   # Min sweep depth %
    'd_max': (0.10, 2.00,  'float'),   # Max excursion %
    'v':     (1,    10,    'int'),     # Reclaim velocity
    'beta':  (1.0,  8.0,   'float'),   # Reward asymmetry
}
```

### Changing WFO Windows

In the `__main__` block of `wfo_matrix.py`:

```python
results = run_wfo(df, train_months=6, trade_months=3, pop_size=80, generations=40)
```

### Changing Friction Constants

In the `absolute_liquidity_sweep_engine` function (sealed kernel — modify with extreme caution):

```python
TICK = 0.25        # MNQ minimum tick
PT_VAL = 2.00      # MNQ point value ($2/point/contract)
COMMISSION = 1.00   # Round-trip commission per contract
SLIPPAGE = 0.50     # 2-tick slippage penalty per fill
MIN_RISK_PTS = 2.00 # Minimum risk distance
```

### DO NOT MODIFY

The tau gate order and fitness function have been mathematically audited. Do not change:

```python
# Tau: gate BEFORE increment
if delta_pct > d_max: s_l = 0       # 1. Depth death
elif tau_l > v: s_l = 0             # 2. Time death
elif q_close[t] > L_star: ...       # 3. Reclaim check
else: tau_l += 1                    # 4. Failed reclaim → count

# Fitness: subtraction below zero
if net < 0: return net - dd         # Monotonic negative gradient
return net / dd                     # Calmar above zero
```

Altering either creates exploitable leaks that the GA will weaponize.

---

## Weekly Operational Cadence

1. **Friday close**: Download latest week of data (downloader appends via checkpoint)
2. **Weekend**: Rerun `wfo_matrix.py` — new apex genome for the upcoming quarter
3. **Monday open**: Deploy locked genome parameters to execution (manual or automated)
4. **Intra-week**: No parameter changes. The genome is frozen until next WFO cycle.

The system is a sonar ping: it measures where institutions are currently sweeping and how fast they're reclaiming. It adapts the boundary conditions to match the current heat of the market.
