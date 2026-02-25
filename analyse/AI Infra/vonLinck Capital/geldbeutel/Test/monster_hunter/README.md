# Micro-Cap Monster Hunter System

**Status:** Research / Backtest Prototype
**Objective:** Identify 2.0x+ baggers in micro-cap equities (8-12 week horizon).

## Disclaimer

**WARNING:** This code models high-risk strategies in illiquid markets.

1. **Slippage:** The simulation imposes a deterministic 15% slippage tax on stocks < $3.00. Real-world execution may be worse.
2. **Survivorship Bias:** Results are only valid if used with a **Point-in-Time (PIT)** database that includes delisted companies. Using standard data feeds will result in massively inflated returns.
3. **Losses:** This strategy aims for low win-rates but high payouts. Expect extended drawdowns.

## Architecture

* **Trigger:** Volume Pivot (High of Max Vol Day in last 40 days).
* **Confirmation:** RVOL > 3.0, Accumulation Ratio, LDI.
* **Execution:** Aggressive cut on failed breakouts (Close < Trigger).
* **Risk:** Dynamic wide stops (`min(entry*0.75, pivot*0.98)`).

## Installation

```bash
# From the project root
pip install -e .

# Or install dependencies directly
pip install pandas numpy xgboost scikit-learn pytest
```

## Project Structure

```
monster_hunter/
├── __init__.py           # Package exports
├── feature_engine.py     # Data pipeline & Feature definitions
├── backtest_engine.py    # Simulation logic (Execution, Slippage, Risk)
├── model_brain.py        # XGBoost pipeline & Purged CV
├── tests/
│   ├── __init__.py
│   └── test_core.py      # Unit & Integration tests
└── README.md             # This file
```

## Quick Start

```python
from monster_hunter import VolumePivotTrigger, prepare_features, BacktestEngine

# 1. Load your Point-in-Time OHLCV data
# df = load_your_data()  # Must have: open, high, low, close, volume

# 2. Prepare features
df = prepare_features(df)

# 3. Initialize engines
trigger_calc = VolumePivotTrigger(window=40)
engine = BacktestEngine(
    slippage_low_price=0.15,  # 15% for stocks < $3
    slippage_high_price=0.08, # 8% for stocks >= $3
    target_multiple=2.0,      # 2x target
    time_stop_days=60         # 60-day max hold
)

# 4. Run backtest
results = []
for i in range(100, len(df)):
    trigger, pivot_low = trigger_calc.compute_trigger(df, i)

    if trigger:
        res = engine.simulate_trade(df, i, trigger, pivot_low)
        if res['outcome'] != 'no_trigger':
            results.append(res)
```

## Key Components

### VolumePivotTrigger

Identifies the breakout level based on volume analysis:
- Looks back 40 trading days
- Finds the day with maximum volume
- Returns that day's High (trigger) and Low (stop reference)

### BacktestEngine

Simulates trade execution with realistic constraints:
- **Slippage Model:** 15% for sub-$3 stocks, 8% for higher-priced stocks
- **Entry:** On breakout day if High > Trigger AND RVOL > 3.0
- **Failed Breakout:** Immediate exit if Close < Trigger
- **Stop Loss:** min(Entry × 0.75, Pivot_Low × 0.98)
- **Target:** Entry × 2.0 (100% gain)
- **Time Stop:** Exit at close after 60 days if neither target nor stop hit

### MonsterModel

XGBoost classifier with proper time-series validation:
- Purged cross-validation to prevent look-ahead bias
- Feature importance analysis
- Probability predictions for ranking candidates

## Running Tests

```bash
# From project root
pytest monster_hunter/tests/test_core.py -v

# With coverage
pytest monster_hunter/tests/test_core.py --cov=monster_hunter
```

## Trade Outcomes

| Outcome | Label | Description |
|---------|-------|-------------|
| `monster_target` | 1 | Hit 2x target |
| `stop_hit` | 0 | Stop loss triggered |
| `failed_breakout` | 0 | Closed below trigger on entry day |
| `time_stop` | 0 | Held 60 days without target/stop |
| `no_trigger` | 0 | Breakout conditions not met |

## Performance Metrics

The system is optimized for **Profit Factor**:

```python
from monster_hunter import compute_profit_factor

pnl_series = pd.Series([r['pnl'] for r in results])
pf = compute_profit_factor(pnl_series)
print(f"Profit Factor: {pf:.2f}")
```

Target metrics:
- Profit Factor > 1.5
- Win Rate: 15-25% (low, but high payout on winners)
- Average Winner: ~100% (by design)
- Average Loser: -25% to -35%

## Integration with AIv3

This system can complement the existing consolidation pattern detector:

1. **Pre-filter:** Use AIv3 to identify consolidation patterns
2. **Volume Trigger:** Apply Monster Hunter to find volume pivot entries
3. **Combined Signal:** Higher confidence when both systems agree

## Notes & Best Practices

- **Data Quality:** Requires Point-in-Time data including delisted stocks
- **Universe:** Best suited for micro/small-cap stocks with high volatility
- **Position Sizing:** Never risk more than 1-2% of portfolio per trade
- **Diversification:** Run multiple setups simultaneously to smooth returns
- **Market Conditions:** Performance varies with market regime - monitor for changes
