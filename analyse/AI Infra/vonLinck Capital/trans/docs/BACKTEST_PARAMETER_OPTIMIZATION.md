# Backtest Parameter Optimization Guide

## Overview

The Dynamic Asset Profiles system is **fully configurable for backtesting**. This allows systematic optimization of:
- **Max Box Width** - Pattern tightness thresholds
- **Stop Buffer** - Stop loss distance below boundaries
- **Target R-Multiples** - Expected gain targets
- **Grey Zone R** - Ambiguous outcome thresholds

## New Tighter Defaults (2025-12-16)

Based on empirical observation: **Real consolidations are TIGHT. Loose patterns are chop.**

| Market Cap | Max Width | Stop Buffer | Target R | Rationale |
|------------|-----------|-------------|----------|-----------|
| **Nano** | 40% | 5.0% | 4.5R | Lottery tickets - extreme volatility acceptable |
| **Micro** | 30% | 4.0% | 4.0R | High volatility but showing structure |
| **Small** | 20% | 3.0% | 3.5R | **"Sweet Spot"** - requires TIGHT patterns |
| **Mid** | 15% | 2.2% | 3.0R | Standard behavior - real consolidation |
| **Large** | 10% | 1.8% | 2.5R | Efficient markets - real coiling |
| **Mega** | 5% | 1.2% | 2.0R | Exceptional tightness (FAANG) |

**Previous defaults were 40/35/30/25/20/15%** - these new values are **40-75% tighter** for mid-to-mega caps.

---

## Why These Parameters Matter

### Max Box Width
**Impact**: Pattern detection rate (filter)
- **Tighter** (lower %) = Fewer patterns, higher quality
- **Looser** (higher %) = More patterns, more noise
- **Test range**: 0.5x to 1.5x of defaults

### Stop Buffer
**Impact**: Risk management and breakdown detection
- **Tighter** (lower %) = Earlier stop-outs, more false breakdowns
- **Looser** (higher %) = More forgiving, fewer false breakdowns
- **Test range**: 0.5x to 2.0x of defaults

### Target R-Multiple
**Impact**: Pattern labeling and success criteria
- **Higher** R = Harder targets, fewer "Target" labels
- **Lower** R = Easier targets, more "Target" labels
- **Test range**: 0.7x to 1.5x of defaults

---

## How to Use in Backtests

### Example 1: Test Tighter Width Thresholds

```python
from config.market_cap_profiles import create_backtest_profiles, get_backtest_config_summary

# Test 20% tighter widths (0.8x multiplier)
tight_profiles = create_backtest_profiles(max_width_multiplier=0.8)

# Results:
# - Nano: 40% -> 32%
# - Small: 20% -> 16%
# - Mid: 15% -> 12%
# - Large: 10% -> 8%

# Use in labeler
from core.path_dependent_labeler import PathDependentLabelerV17

labeler = PathDependentLabelerV17(
    use_dynamic_profiles=True,
    default_market_cap_category='small_cap'
)

# Override per pattern with custom profile
custom_profile = tight_profiles[MarketCapCategory.SMALL_CAP]
label, metadata = labeler.label_pattern(
    full_data=df,
    pattern_end_idx=130,
    market_cap_category='small_cap'  # Uses tight_profiles via override
)
```

### Example 2: Grid Search Parameter Sweep

```python
import numpy as np
from itertools import product

# Define parameter grid
width_multipliers = [0.6, 0.8, 1.0, 1.2, 1.5]
stop_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
target_multipliers = [0.8, 1.0, 1.2]

results = []

for width_mult, stop_mult, target_mult in product(
    width_multipliers,
    stop_multipliers,
    target_multipliers
):
    # Create custom profiles
    profiles = create_backtest_profiles(
        max_width_multiplier=width_mult,
        stop_buffer_multiplier=stop_mult,
        target_r_multiplier=target_mult
    )

    # Run backtest with these profiles
    backtest_results = run_backtest(
        tickers=['AAPL', 'MSFT', ...],
        profiles=profiles,
        start_date='2020-01-01',
        end_date='2024-01-01'
    )

    results.append({
        'width_mult': width_mult,
        'stop_mult': stop_mult,
        'target_mult': target_mult,
        'win_rate': backtest_results['win_rate'],
        'avg_r': backtest_results['avg_r'],
        'sharpe': backtest_results['sharpe']
    })

# Find optimal parameters
best = max(results, key=lambda x: x['sharpe'])
print(f"Optimal: Width={best['width_mult']}x, "
      f"Stop={best['stop_mult']}x, Target={best['target_mult']}x")
```

### Example 3: Per-Category Optimization

```python
# Test different parameters for each market cap category
optimal_params = {}

for category in ['nano_cap', 'micro_cap', 'small_cap', 'mid_cap', 'large_cap', 'mega_cap']:
    best_sharpe = -999
    best_width = None

    for width_mult in np.arange(0.5, 1.5, 0.1):
        profiles = create_backtest_profiles(max_width_multiplier=width_mult)

        # Backtest only this category
        results = run_category_backtest(
            category=category,
            profiles=profiles,
            tickers=get_tickers_for_category(category)
        )

        if results['sharpe'] > best_sharpe:
            best_sharpe = results['sharpe']
            best_width = width_mult

    optimal_params[category] = {
        'width_multiplier': best_width,
        'sharpe': best_sharpe
    }

# Results might show:
# - Nano: 1.0x (keep 40%)
# - Small: 0.7x (14% instead of 20%)
# - Large: 0.5x (5% instead of 10%)
```

---

## Expected Backtest Findings

### Hypothesis 1: Tighter = Better (Up to a Point)
**Test**: Sweep width_multiplier from 0.5x to 1.5x
**Expected**: Optimal around 0.7-0.9x (tighter than defaults)
**Why**: Loose patterns dilute signal with choppy markets

### Hypothesis 2: Small Caps Tolerate More Width
**Test**: Compare optimal width_mult across categories
**Expected**: Nano/Micro ~1.0x, Small ~0.8x, Large ~0.6x
**Why**: Larger caps have less legitimate volatility

### Hypothesis 3: Stop Buffer Sweet Spot
**Test**: Sweep stop_multiplier from 0.5x to 2.0x
**Expected**: Optimal around 1.0-1.2x (current or slightly wider)
**Why**: Too tight = false breakdowns, too loose = missed risk

### Hypothesis 4: Higher Targets for Small Caps
**Test**: Compare optimal target_mult across categories
**Expected**: Small caps ~1.2x, Large caps ~0.8x
**Why**: Small caps have explosive potential, large caps grind

---

## Backtesting Workflow

```
1. Define Parameter Grid
   └─> width_mult, stop_mult, target_mult ranges

2. For each parameter combination:
   ├─> Create custom profiles
   ├─> Scan historical data for patterns
   ├─> Label patterns with custom profiles
   ├─> Calculate performance metrics
   └─> Store results

3. Analyze Results:
   ├─> Identify optimal parameters per category
   ├─> Check for overfitting (train vs test)
   ├─> Validate on out-of-sample period
   └─> Update DEFAULT_PROFILE values

4. Production Deployment:
   └─> Use optimized parameters as new defaults
```

---

## Key Metrics to Track

| Metric | Description | Target |
|--------|-------------|--------|
| **Pattern Count** | Total patterns detected | Enough for statistical significance (>100) |
| **Win Rate** | % of Target outcomes | 20-40% (depends on R-multiple) |
| **Avg R** | Average risk-multiple achieved | >1.5R overall |
| **Sharpe Ratio** | Risk-adjusted returns | >1.0 |
| **Max Drawdown** | Worst losing streak | <30% |
| **Recovery Factor** | Profit / Max DD | >2.0 |
| **Label Distribution** | Danger/Noise/Target balance | ~15/60/25 |

---

## Production Update Process

1. **Run comprehensive backtest** with parameter grid
2. **Identify optimal parameters** for each category
3. **Validate on holdout period** (e.g., most recent year)
4. **Update profile definitions** in `config/market_cap_profiles.py`
5. **Re-run existing tests** to ensure compatibility
6. **Deploy to production** with monitoring

---

## Example: Systematic Optimization Script

```python
"""
Systematic parameter optimization for dynamic asset profiles.
"""

import pandas as pd
from config.market_cap_profiles import create_backtest_profiles
from core.path_dependent_labeler import PathDependentLabelerV17
from backtesting.temporal_backtester import TemporalBacktester

def optimize_profiles(
    tickers: list,
    start_date: str,
    end_date: str,
    width_range=(0.5, 1.5, 0.1),
    stop_range=(0.5, 2.0, 0.25),
    target_range=(0.7, 1.5, 0.1)
):
    """
    Run grid search optimization on asset profile parameters.

    Returns:
        DataFrame with results for each parameter combination
    """
    results = []

    width_multipliers = np.arange(*width_range)
    stop_multipliers = np.arange(*stop_range)
    target_multipliers = np.arange(*target_range)

    total_combinations = len(width_multipliers) * len(stop_multipliers) * len(target_multipliers)
    print(f"Testing {total_combinations} parameter combinations...")

    for i, (w, s, t) in enumerate(product(width_multipliers, stop_multipliers, target_multipliers)):
        if i % 10 == 0:
            print(f"Progress: {i}/{total_combinations}")

        # Create custom profiles
        profiles = create_backtest_profiles(
            max_width_multiplier=w,
            stop_buffer_multiplier=s,
            target_r_multiplier=t
        )

        # Run backtest
        bt = TemporalBacktester(
            labeler=PathDependentLabelerV17(use_dynamic_profiles=True),
            custom_profiles=profiles
        )

        metrics = bt.run(tickers, start_date, end_date)

        results.append({
            'width_mult': w,
            'stop_mult': s,
            'target_mult': t,
            **metrics
        })

    return pd.DataFrame(results)

# Run optimization
results_df = optimize_profiles(
    tickers=['AAPL', 'MSFT', 'AMD', 'NVDA', ...],
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Find best
best = results_df.nlargest(10, 'sharpe')
print(best)

# Save results
results_df.to_csv('output/parameter_optimization.csv', index=False)
```

---

## Research Questions to Answer

1. **What is the optimal max_box_width for each category?**
   - Current: 40/30/20/15/10/5%
   - Hypothesis: Tighter is better (30/20/15/10/7/3%?)

2. **Does stop_buffer need to vary by category?**
   - Current: Varies 1-5%
   - Hypothesis: Could be simplified to 2-3% across all categories

3. **Are target R-multiples too aggressive?**
   - Current: 4.5R for nano, 2.0R for mega
   - Hypothesis: Lower targets (3R / 1.5R) might improve win rate

4. **Does grey_zone_r matter?**
   - Current: Used to exclude ambiguous patterns
   - Hypothesis: Could be removed or simplified

5. **Should parameters adapt to market regime?**
   - Bull market: Tighter stops, higher targets?
   - Bear market: Wider stops, lower targets?

**Run these backtests to get empirical answers!**
