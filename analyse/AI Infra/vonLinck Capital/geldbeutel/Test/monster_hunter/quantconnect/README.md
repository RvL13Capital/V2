# Monster Hunter: QuantConnect Acid Test

## Purpose

This algorithm validates the Monster Hunter strategy's profitability using QuantConnect's survivorship-bias-free data. Run this **before** investing time in ML optimization.

## Quick Start

### Step 1: Login to QuantConnect

1. Go to [quantconnect.com](https://www.quantconnect.com)
2. Create account or login
3. Navigate to **Algorithm Lab**

### Step 2: Create New Algorithm

1. Click **Create New Algorithm**
2. Select **Python**
3. Name: `MonsterHunterAcidTest`

### Step 3: Upload Code

1. Delete the template code
2. Copy entire contents of `MonsterHunterAlgorithm.py`
3. Paste into the editor
4. Click **Build** (should compile without errors)

### Step 4: Configure Backtest

The algorithm is pre-configured with:
- **Start**: January 1, 2014
- **End**: December 31, 2024
- **Cash**: $100,000
- **Resolution**: Daily

No additional configuration needed.

### Step 5: Run Backtest

1. Click **Backtest** button
2. Wait 10-20 minutes (10 years of data)
3. Monitor progress in the console

### Step 6: Analyze Results

Look for these lines in the console output:

```
==================================================
MONSTER HUNTER ACID TEST RESULTS
==================================================
Total Trades:       XXX
Win Rate:           XX.X%
Profit Factor:      X.XX    <-- KEY METRIC
Avg PnL per Trade:  X.X%
Avg Winner:         XX.X%
Avg Loser:          -XX.X%
--------------------------------------------------
Outcome Breakdown:
  failed_breakout: XX (X.X%)
  stop_hit: XX (X.X%)
  target_hit: XX (X.X%)
  time_stop: XX (X.X%)
==================================================
```

## Decision Matrix

| Profit Factor | Verdict | Action |
|---------------|---------|--------|
| **> 1.3** | VIABLE | Proceed with ML optimization |
| **0.9 - 1.2** | MARGINAL | Test tighter filters first |
| **< 0.9** | FAILS | Redesign entry logic |

## Strategy Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PIVOT_LOOKBACK` | 40 days | Window for volume pivot |
| `RVOL_THRESHOLD` | 3.0 | Min relative volume for entry |
| `MIN_PRICE` | $0.60 | Floor price filter |
| `MAX_PRICE` | $10.00 | Ceiling (micro-cap proxy) |
| `MIN_DOLLAR_VOL` | $50,000 | Liquidity filter |
| `TARGET_MULTIPLE` | 2.0 | 100% profit target |
| `TIME_STOP_DAYS` | 60 | Max holding period |
| `SLIPPAGE_LOW` | 15% | Slippage for stocks < $3 |
| `SLIPPAGE_HIGH` | 8% | Slippage for stocks >= $3 |
| `POSITION_SIZE` | 2% | Per-trade allocation |
| `MAX_POSITIONS` | 20 | Concurrent position limit |

## Slippage Model

The strategy applies a **brutal slippage tax** to simulate real micro-cap execution:

```
Price < $3.00  --> 15% slippage
Price >= $3.00 --> 8% slippage
```

This is intentionally punishing. If the strategy survives this, it has real edge.

## Trade Lifecycle

1. **Universe Filter**: Coarse selection for $0.60-$10 stocks with $50k+ dollar volume
2. **Setup Detection**: Find volume pivot (high of max volume day in 40 days)
3. **Entry Signal**: Price breaks above trigger with RVOL > 3.0
4. **Slippage Applied**: Buy price = Trigger * (1 + slippage_rate)
5. **Stop Calculation**: min(Entry * 0.75, Pivot_Low * 0.98)
6. **Failed Breakout Check**: If EOD close < trigger, exit immediately
7. **Position Management**: Monitor for stop, target (2x), or time stop (60 days)

## Troubleshooting

### "No trades executed"

- Universe may be too restrictive
- Try lowering `MIN_DOLLAR_VOL` to 25000
- Try raising `MAX_PRICE` to 15.00

### Slow backtest

- 10 years with 500 symbols takes 15-20 minutes on free tier
- Reduce universe size by lowering the `[:500]` limit in `CoarseSelectionFunction`

### Compilation errors

- Ensure you copied the **entire** file including imports
- Check for any copy/paste formatting issues

## Understanding Results

### Target Hit Rate

Expect 15-25% of trades to hit the 2x target. This is normal for high-payoff strategies.

### Failed Breakouts

Should be 10-20% of trades. These are stopped quickly, limiting damage.

### Time Stops

Trades that didn't hit stop or target in 60 days. Usually small gains or losses.

## Next Steps

After running the backtest:

1. **PF > 1.3**: Return to local codebase and train XGBoost model to optimize signal selection
2. **PF 0.9-1.2**: Experiment with filters (higher RVOL, tighter universe)
3. **PF < 0.9**: Analyze trade log to understand why the edge is gone

## Files

- `MonsterHunterAlgorithm.py` - Complete QC algorithm (copy this to QC)
- `README.md` - This file
