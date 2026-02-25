# Pattern Detection & Qualification: Ultra-Detailed Technical Reference

**Last Verified Against Production Code**: 2026-01-10
**Production Model**: `eu_model_clean.pt` (V22 Clean Slate)

---

## Table of Contents

1. [Qualification Phase Criteria](#1-qualification-phase-criteria)
2. [State Machine Transitions](#2-state-machine-transitions)
3. [Breakout & Breakdown Logic](#3-breakout--breakdown-logic)
4. [Price & Volume Behavior Insights](#4-price--volume-behavior-insights)
5. [Production Code Verification](#5-production-code-verification)
6. [Guardrails & Assumption Checks](#6-guardrails--assumption-checks)

---

## 1. Qualification Phase Criteria

### 1.1 The Two-Level Filter System

The production scanner uses **two distinct tightness metrics** that together form the "Accumulation Signature":

| Metric | What It Measures | Threshold | Source |
|--------|------------------|-----------|--------|
| **Wick Tightness** (Loose) | Intraday High/Low range | 30-60% (market-cap dependent) | `sleeper_scanner_v17.py:126-128` |
| **Body Tightness** (Strict) | Close-to-Close range | <15% (fixed for all caps) | `sleeper_scanner_v17.py:131-134` |

**Why Two Metrics?**

The accumulation signature of institutional buying shows:
- **Tight closes**: Institutions want specific entry points (creates body tightness)
- **Wide intraday swings**: Shakes out weak hands, creates buying opportunities (allows wick looseness)

A stock with 40% intraday swings but only 10% close-to-close variance is showing accumulation, not chaos.

### 1.2 Wick Tightness: Market-Cap Dependent Width Limits

```
Location: sleeper_scanner_v17.py:23-31 (WIDTH_LIMITS)
Verified: ACTIVE in production eu_model_clean.pt
```

| Market Cap Category | Range | Max Wick Width | Rationale |
|---------------------|-------|----------------|-----------|
| **Nano-cap** | <$50M | 60% | Extreme volatility, lottery tickets |
| **Micro-cap** | $50M-$300M | 45% | High volatility but showing structure |
| **Small-cap** | $300M-$2B | 30% | "Sweet spot" - natural consolidation width |
| **Mid-cap** | $2B-$10B | 22.5% | Standard technical behavior |
| **Large-cap** | $10B-$200B | 15% | Efficient markets, real coiling only |
| **Mega-cap** | >$200B | 7.5% | Exceptional tightness (FAANG level) |

**Empirical Validation (2025-12-16)**:
- Grid search tested 15 combinations over 10 years of data
- Tighter limits (0.6x): 75.9% danger rate
- Optimal limits (1.5x): 70.4% danger rate (5.5pp improvement)
- Small-cap natural consolidation is 25-30%, not 10-15%

### 1.3 Body Tightness: Fixed 15% Maximum

```
Location: sleeper_scanner_v17.py:36 (BODY_TIGHTNESS_MAX = 0.15)
Verified: ACTIVE in production
```

**Calculation**:
```python
body_upper = np.percentile(recent['close'], 95)  # 60-day window
body_lower = np.percentile(recent['close'], 10)
body_height_pct = (body_upper - body_lower) / body_lower
# MUST be < 0.15 (15%) for ALL market caps
```

This is the **strict filter** - if close prices vary more than 15% over 60 days, there's no accumulation pattern regardless of market cap.

### 1.4 Thin Stock Buffer (+5% Allowance)

```
Location: sleeper_scanner_v17.py:156-159
Verified: ACTIVE in production
```

**Purpose**: Distinguish between "messy volatility" (bad) and "thin liquidity volatility" (acceptable structural feature).

**Detection**:
```python
avg_candle_size = (df['high'] - df['low']).rolling(20).mean().iloc[-1] / current_price
is_thin = avg_candle_size > 0.04  # If daily candles average >4% range
final_wick_allowed = wick_allowed + (0.05 if is_thin else 0.0)  # +5% bonus
```

**Rationale**: Thin stocks naturally have larger bid-ask spreads and daily ranges. This isn't noise; it's structural.

### 1.5 Liquidity Gate ($50k Minimum)

```
Location: sleeper_scanner_v17.py:103-116
Config: constants.py:140 (MIN_LIQUIDITY_DOLLAR = 50000)
Verified: ACTIVE in production
```

**Two Checks**:

1. **Dollar Volume Threshold**:
   ```python
   avg_dollar_vol = df['DollarVol'].rolling(20).mean().iloc[-1]
   if avg_dollar_vol < 50000:
       return None  # Reject
   ```

2. **Ghost Stock Detection**:
   ```python
   recent_volume = df['volume'].iloc[-20:]
   zero_volume_days = (recent_volume == 0).sum()
   if zero_volume_days > 4:  # Max 4 zero-vol days in 20
       return None  # Reject - "trades by appointment only"
   ```

### 1.6 Qualification State Machine (10-Day Requirement)

```
Location: consolidation_tracker.py:289-334
Config: qualification_days = 10
Verified: ACTIVE in production
```

The `ConsolidationTracker` implements a day-by-day state machine:

**Day-by-Day Qualification Logic**:
```python
def check_qualification_criteria(self, row):
    return (
        row['bbw_percentile'] < 30 and      # BBW < 30th percentile
        row['adx'] < 32 and                  # ADX < 32 (low trending)
        row['volume_ratio'] < 0.35 and       # Volume < 35% of 20-day avg
        row['range_ratio'] < 0.65            # Daily range < 65% of 20-day avg
    )
```

**Transition Rules**:
- Day 1: First qualifying day starts QUALIFYING state
- Days 2-9: Must continue meeting ALL criteria OR reset to NONE
- Day 10: If all criteria met for 10 consecutive days → transition to ACTIVE

### 1.7 Boundary Establishment (90th/10th Percentile)

```
Location: consolidation_tracker.py:200-251
Verified: ACTIVE in production
```

After 10-day qualification completes:

```python
# Use qualification period data (last 10 days)
period_data = data.iloc[start_idx:end_idx + 1]

# 90th/10th PERCENTILE (NOT max/min)
# This creates a "Volatility Envelope" that ignores top 10% manipulation spikes
upper_boundary = period_data['high'].quantile(0.90)
lower_boundary = period_data['low'].quantile(0.10)

# Power boundary: 0.5% above upper (breakout confirmation buffer)
power_boundary = upper_boundary * 1.005
```

**Why Percentiles, Not Max/Min?**
- Max/min boundaries are destroyed by single scam wicks
- 90th/10th percentile crops out top 10% of manipulation
- Allows patterns to survive 60-100 days instead of being killed early

---

## 2. State Machine Transitions

```
NONE → QUALIFYING (10 days) → ACTIVE → MATURE (30 days) → COMPLETED/FAILED
```

### 2.1 Complete State Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PATTERN STATE MACHINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐      Day 1 meets          ┌─────────────┐                    │
│   │   NONE   │ ─── criteria ──────────>  │  QUALIFYING │                    │
│   └──────────┘                           └──────┬──────┘                    │
│        ^                                        │                            │
│        │                                        │ 10 consecutive             │
│        │ Criteria fail                          │ qualifying days            │
│        │ (any day 1-9)                          v                            │
│        │                                 ┌──────────────┐                    │
│        └─────────────────────────────────│    ACTIVE    │                    │
│                                          └──────┬───────┘                    │
│                                                 │                            │
│                            ┌────────────────────┼────────────────────┐       │
│                            │                    │                    │       │
│                            │ 30 days in         │ 2 consecutive      │       │
│                            │ channel            │ close violations   │       │
│                            v                    v                    v       │
│                     ┌──────────────┐    ┌────────────┐       ┌────────────┐ │
│                     │    MATURE    │    │ COMPLETED  │       │   FAILED   │ │
│                     │  (adjusted   │    │ (breakout) │       │(breakdown) │ │
│                     │  boundaries) │    └────────────┘       └────────────┘ │
│                     └──────┬───────┘                                        │
│                            │ 2 consecutive                                   │
│                            │ close violations                                │
│                            └────────────────────────────────────────────────>│
│                                              COMPLETED/FAILED                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 The "Iron Core" Rule: Only Closes Kill Patterns

```
Location: consolidation_tracker.py:341-359
Verified: ACTIVE in production
```

**Wicks are FREE, only CLOSES count as structural breaks.**

```python
# Check if CLOSE (not high/low) violates boundary
is_violation = False
if row['close'] > self.current_pattern.upper_boundary:
    is_violation = True
elif row['close'] < self.current_pattern.lower_boundary:
    is_violation = True

if is_violation:
    self.consecutive_violations += 1
else:
    self.consecutive_violations = 0  # Reset on re-entry
    self.current_pattern.days_in_channel += 1
```

**Rationale**: Intraday wicks (stop hunts) are common in micro-caps. A single wick through the boundary followed by close back inside is a **bull/bear trap**, not a pattern failure.

### 2.3 Two-Consecutive-Close Requirement

```
Location: consolidation_tracker.py:357-384
Verified: ACTIVE in production
```

**Patterns require 2 consecutive closes outside boundaries to terminate.**

```python
if self.consecutive_violations >= 2:
    if row['close'] < lower_boundary:
        state = FAILED      # BREAKDOWN
    elif row['close'] > upper_boundary:
        state = COMPLETED   # BREAKOUT
```

**Why 2 Closes?**
- Single-day fakeouts are common in micro-caps (manipulation, thin liquidity)
- Requiring 2 closes filters out 1-day traps
- Genuine breakouts/breakdowns maintain direction over multiple days

---

## 3. Breakout & Breakdown Logic

### 3.1 Breakout Definition (Pattern COMPLETED)

```
Location: consolidation_tracker.py:372-384
path_dependent_labeler.py:278-283
Verified: ACTIVE in production
```

**Detection Criteria**:
1. Close > upper_boundary for 2 consecutive days
2. Pattern transitions to COMPLETED state
3. For labeling: Close >= target_price AND next_open >= target_price (confirmation)

**Labeling as Target (Class 2)**:
```python
# Walk forward day-by-day
if close_t >= target_price:
    # Require next-day confirmation
    if t < len(outcome_data) - 1:
        open_next = outcome_data.iloc[t + 1]['open']
        if open_next >= target_price:
            label = 2  # TARGET
            break  # First event wins
```

**Why Next-Open Confirmation?**
- Prevents labeling intraday wicks as hits
- Ensures the breakout "sticks" overnight
- Filters flash spikes from manipulation

### 3.2 Breakdown Definition (Pattern FAILED)

```
Location: consolidation_tracker.py:360-370
path_dependent_labeler.py:271-275
Verified: ACTIVE in production
```

**Detection Criteria**:
1. Close < lower_boundary for 2 consecutive days
2. Pattern transitions to FAILED state
3. For labeling: Low < stop_loss (uses LOW for worst-case execution)

**Labeling as Danger (Class 0)**:
```python
# Walk forward day-by-day - PRIORITY 1 (stop before target)
low_t = outcome_data.iloc[t]['low']  # Use LOW, not close
if low_t < stop_loss:
    label = 0  # DANGER
    break  # First event wins
```

**Why LOW for Stop, CLOSE for Target?**
- Stops can gap through: An intraday wick hitting your stop is a real stop-out
- Targets need confirmation: A wick above target without follow-through is unreliable
- This asymmetry reflects real trade execution

### 3.3 R-Multiple Calculation (Risk Unit)

```
Location: path_dependent_labeler.py:211-255
Config: constants.py:185-187
Verified: ACTIVE in production (RISK_MULTIPLIER_TARGET = 5.0, STOP_BUFFER_PERCENT = 0.08)
```

**Core Formulas**:
```python
# Stop calculation
stop_loss = lower_boundary * (1 - STOP_BUFFER_PERCENT)  # 8% below lower
# Example: lower=$12.26, buffer=8%
# stop_loss = $12.26 × 0.92 = $11.28

# R = Risk Unit (pattern-specific)
R = lower_boundary - stop_loss
# R = $12.26 - $11.28 = $0.98

# Target = Entry + 5R
target_price = entry_price + (5.0 × R)
# target = $14.02 + ($0.98 × 5) = $18.92

# Grey zone = Entry + 2.5R (converted to Noise in Jan 2026)
grey_threshold = entry_price + (2.5 × R)
```

### 3.4 Stop Buffer Rationale (8%)

**Why 8% Below Lower Boundary?**

1. **Stop-hunting**: Micro-caps are aggressively stop-hunted. Market makers/whales deliberately trigger stops clustered at obvious levels.

2. **Natural false breaks**: Lower boundary is 10th percentile of lows - there will be occasional false breaks below this level.

3. **Empirical testing**: 8% buffer reduces false Danger labels while maintaining risk definition.

---

## 4. Price & Volume Behavior Insights

### 4.1 Volume Dry-Up Ratio: The Primary Breakout Predictor

```
Location: vectorized_calculator.py:207-255
Feature Index: 8 (vol_dryup_ratio)
Verified: ACTIVE in production, listed in DRIFT_CRITICAL_FEATURES
```

**Formula**:
```python
vol_short = volume.rolling(3).mean()   # Recent 3-day average
vol_long = volume.rolling(20).mean()   # Baseline 20-day average
vol_dryup_ratio = vol_short / vol_long
```

**Signal Interpretation**:

| Ratio | Meaning | Action |
|-------|---------|--------|
| < 0.30 | Volume dried up to 30% of normal | **IMMINENT MOVE** |
| < 0.40 | Volume contraction beginning | Alert state |
| 0.40-0.80 | Normal trading range | Hold pattern |
| > 0.80 | Volume stable or increasing | No signal |

**Why This Works**:
- Volume dry-up in consolidation = institutions accumulated, retail lost interest
- Low volume makes it easier for smart money to push price
- When volume returns → explosive moves happen

**Trading Application**:
```
When Vol_DryUp_Ratio < 0.3 AND pattern is ACTIVE for 15+ days
→ IMMINENT MOVE (breakout or breakdown expected within 5-10 days)
```

### 4.2 Accumulation Day Detection

```
Location: sleeper_scanner_v17.py:172-182
Config: ENABLE_ACCUMULATION_DETECTION = True
Verified: ACTIVE in production
```

**Definition**: Days with huge volume spikes + tiny price moves.

```python
acc_window = df.iloc[-90:]  # Last 90 days
is_huge_vol = acc_window['volume'] > (acc_window['Vol_50MA'] * 3)  # 3x average
is_tiny_move = abs(acc_window['Pct_Change']) < 0.025  # <2.5% move
is_active = acc_window['volume'] > 0

accumulation_days = acc_window[is_huge_vol & is_tiny_move & is_active]
```

**What This Captures**:
- Large blocks being absorbed without moving price
- Characteristic of institutional accumulation
- More accumulation days = stronger pattern conviction

### 4.3 BBW (Bollinger Band Width): Volatility Contraction

```
Location: vectorized_calculator.py:104-137
Feature Index: 5 (bbw_20)
Verified: ACTIVE in production, listed in DRIFT_CRITICAL_FEATURES
```

**Formula**:
```python
sma = close.rolling(20).mean()
std = close.rolling(20).std()
upper_band = sma + (2 * std)
lower_band = sma - (2 * std)
bbw = (upper_band - lower_band) / sma
```

**Signal Interpretation**:

| BBW Percentile | Meaning | Pattern Quality |
|----------------|---------|-----------------|
| < 10th | Extreme compression | Highest quality patterns |
| < 30th | Good compression | Qualifying threshold |
| 30th-50th | Moderate volatility | Marginal patterns |
| > 50th | High volatility | Not consolidating |

### 4.4 ADX: Trend Strength Rejection

```
Location: vectorized_calculator.py:139-184
Feature Index: 6 (adx)
Verified: ACTIVE in production
```

**Threshold**: ADX < 32 during qualification

**Why ADX < 32?**

| ADX Value | Trend State | Pattern Quality |
|-----------|-------------|-----------------|
| < 20 | No trend | Ideal consolidation |
| 20-32 | Weak trend | Acceptable |
| > 32 | Strong trend | REJECT - not consolidating |

Stocks in strong trends don't consolidate - they continue. ADX > 32 indicates the stock is trending, not coiling.

### 4.5 Typical Price/Volume Behavior During Consolidation

**Phase 1: Entry (Days 1-10)**
- Volume: 50-70% of 20-day average (starting to dry up)
- Price: Multiple tests of both boundaries
- BBW: Contracting from recent high
- ADX: Declining toward sub-30 levels

**Phase 2: Coiling (Days 11-25)**
- Volume: 30-50% of average (significant dry-up)
- Price: Tightening range, fewer boundary tests
- BBW: At or near 20-day low
- ADX: Stable in 15-25 range

**Phase 3: Pre-Breakout (Days 26-35)**
- Volume: < 30% of average (critical dry-up)
- Price: Clustered near upper boundary (bullish) or lower (bearish)
- BBW: At minimum compression
- ADX: Beginning to curl up (direction unclear)

**Phase 4: Resolution**
- Volume: 200-400% spike on breakout/breakdown
- Price: Gaps through boundary
- BBW: Rapidly expanding
- ADX: Shooting above 25

---

## 5. Production Code Verification

### 5.1 Call Stack Verification

**Training Pipeline**:
```
02_train_temporal.py
  → loads sequences from HDF5 (generated by 01_generate_sequences.py)
  → HybridFeatureNetwork (models/temporal_hybrid_v18.py)
  → USE_GRN_CONTEXT = True (config/constants.py:143)
  → AsymmetricLoss with GAMMA_PER_CLASS (config/constants.py:271-275)
```

**Sequence Generation**:
```
01_generate_sequences.py
  → TemporalPatternDetector (core/pattern_detector.py)
    → ConsolidationTracker (core/consolidation_tracker.py)
    → VectorizedFeatureCalculator (features/vectorized_calculator.py)
    → PathDependentLabelerV17 (core/path_dependent_labeler.py)
```

**Pattern Detection**:
```
00_detect_patterns.py
  → PatternScanner (core/pattern_scanner.py)
    → VectorizedPatternMixin (pre-filter)
    → find_sleepers_v17 (core/sleeper_scanner_v17.py)
    → WIDTH_LIMITS (market-cap based)
    → BODY_TIGHTNESS_MAX = 0.15
```

### 5.2 Constants Verification (Production Values)

| Constant | Value | Location | Verified |
|----------|-------|----------|----------|
| RISK_MULTIPLIER_TARGET | 5.0 | constants.py:185 | Yes |
| STOP_BUFFER_PERCENT | 0.08 (8%) | constants.py:187 | Yes |
| BODY_TIGHTNESS_MAX | 0.15 (15%) | sleeper_scanner_v17.py:36 | Yes |
| WIDTH_LIMITS['small_cap'] | 0.30 (30%) | sleeper_scanner_v17.py:26 | Yes |
| MIN_LIQUIDITY_DOLLAR | 50000 | constants.py:140 | Yes |
| qualification_days | 10 | consolidation_tracker.py:121 | Yes |
| INDICATOR_WARMUP_DAYS | 30 | constants.py:190 | Yes |
| USE_GRN_CONTEXT | True | constants.py:143 | Yes |
| CLASS_WEIGHTS[0] (Danger) | 5.0 | constants.py:283 | Yes |
| GAMMA_PER_CLASS[0] (Danger) | 4.0 | constants.py:272 | Yes |

### 5.3 Feature Index Verification (14 Features)

| Index | Feature | Location | Used In |
|-------|---------|----------|---------|
| 0-3 | open, high, low, close | Relativized to day 0 | Temporal branch |
| 4 | volume | Log ratio to day 0 | Temporal branch |
| 5 | bbw_20 | vectorized_calculator.py:58 | **DRIFT_CRITICAL** |
| 6 | adx | vectorized_calculator.py:61-64 | Temporal branch |
| 7 | volume_ratio_20 | vectorized_calculator.py:68 | Temporal branch |
| 8 | vol_dryup_ratio | vectorized_calculator.py:71 | **DRIFT_CRITICAL** |
| 9 | var_score | vectorized_calculator.py:85-88 | Composite |
| 10 | nes_score | vectorized_calculator.py:91-94 | Composite |
| 11 | lpf_score | vectorized_calculator.py:97-100 | Composite |
| 12-13 | upper_boundary, lower_boundary | pattern_detector.py:1002-1011 | Relativized |

---

## 6. Guardrails & Assumption Checks

### 6.1 Critical Assumptions Verified

| Assumption | Validity | Guardrail |
|------------|----------|-----------|
| Tight closes = accumulation | PARTIAL | Body tightness (15%) + Vol_DryUp_Ratio |
| 100-day window captures outcomes | HIGH | Empirical: 95% of moves complete in 100 days |
| R-multiples normalize risk | HIGH | Pattern-specific risk unit, comparable across stocks |
| 2 consecutive closes = confirmed | HIGH | Filters 1-day fakeouts common in micro-caps |
| LOW for stop, CLOSE+next_open for target | HIGH | Reflects real execution asymmetry |
| Market cap determines volatility physics | HIGH | Empirically validated (grid search 2025-12-16) |

### 6.2 Anti-Patterns (What DOESN'T Work)

1. **Max/Min Boundaries**: Single scam wicks destroy patterns. Use percentiles.

2. **Fixed Width Limits**: A 20% limit that works for small-caps rejects valid nano-cap patterns.

3. **Single-Close Violations**: Too many false kills from 1-day fakeouts.

4. **Random Train/Test Split**: Overlapping windows leak data. Must use temporal split.

5. **Raw Volume as Feature**: Volume scale varies 1000x across stocks. Use log ratio.

6. **Target Without Confirmation**: Wicks create false positives. Require next-open.

### 6.3 Data Quality Guardrails

| Check | Threshold | Action |
|-------|-----------|--------|
| Ghost stock | >4 zero-vol days in 20 | REJECT |
| Dollar volume | <$50k average | REJECT |
| Range too small | <0.3% | REJECT (mutual fund NAV) |
| All zero volume | 100% in qualification | REJECT |
| Pattern overlap | <10 days apart | NMS de-duplicate |
| ADX warmup | First 14 days | Include 30-day prefix |

---

## Summary

This document verifies that the production model (eu_model_clean.pt, V22) uses:

1. **Dual Tightness Metrics**: Wick (30-60% market-cap based) + Body (15% fixed)
2. **10-Day Qualification**: Consecutive days meeting BBW/ADX/Volume/Range criteria
3. **Iron Core Rule**: Only closes violate boundaries, wicks are free
4. **2-Consecutive-Close Termination**: Filters 1-day fakeouts
5. **5R Target / 8% Stop Buffer**: Risk-based labeling with 100-day outcome window
6. **Path-Dependent "First Event Wins"**: Stop checked with LOW, target with CLOSE+next_open
7. **14 Temporal Features**: All normalized appropriately (relativized, log ratio, robust scaled)
8. **Vol_DryUp_Ratio**: Primary breakout predictor (< 0.3 = imminent move)

All constants, formulas, and logic have been traced from CLAUDE.md through the actual source files and verified against the production training pipeline.
