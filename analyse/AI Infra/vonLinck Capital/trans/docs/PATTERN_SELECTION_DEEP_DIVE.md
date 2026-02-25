# TRANS Pattern Selection & Labeling: Complete Technical Deep-Dive

## Part 1: The Philosophy Behind Pattern Selection

### Core Market Thesis

The TRANS system hunts for **"sleeper" stocks** - micro/small-cap equities in a specific microstructure state that precedes explosive moves. The fundamental insight is:

**"Tight closes + wide intraday ranges = Strong hands accumulating while shaking out weak hands"**

This is the **Accumulation Signature**. Institutional buyers create tight closing prices (they want specific entry points) while allowing intraday volatility (which flushes out retail/weak holders). The pattern manifests as:

- **Body Tightness** (Close prices): Very narrow range (<15%) - accumulation
- **Wick Tightness** (High/Low): Looser range (30-60% depending on market cap) - shakeouts

### Why Consolidation Patterns Work

The system targets consolidations because they represent:

1. **Energy Compression**: Volatility contraction (measured by BBW) before expansion
2. **Information Asymmetry**: Smart money accumulates before public catalysts
3. **Defined Risk**: Clear boundaries provide natural stop-loss levels

### The Two-Registry Separation (Temporal Integrity)

The system enforces **strict temporal separation** between:

| Registry | Script | Output | Purpose |
|----------|--------|--------|---------|
| **Candidate Registry** | `00_detect_patterns.py` | `candidate_patterns.parquet` | Pattern detection (NO future data) |
| **Outcome Registry** | `00b_label_outcomes.py` | `labeled_patterns.parquet` | Label assignment (100+ days later) |

**Why this matters**: If you label patterns at detection time, the model can "learn" artifacts that leak future information. The two-registry system guarantees that during feature engineering, the model CANNOT see any outcome data.

---

## Part 2: Pattern Selection Process (Detection)

### Step 1: Liquidity Gate ($50k minimum)

```
Location: sleeper_scanner_v17.py:103-116
```

**Purpose**: Filter out untradeable "ghost" stocks.

**Checks**:
1. Average 20-day dollar volume >= $50,000
2. No more than 4 days with zero volume in last 20 days

**Rationale**: Patterns in illiquid stocks can't be traded without significant slippage. "Ghost" stocks trade by appointment only.

### Step 2: Vectorized Pre-Filter (95% Loop Reduction)

```
Location: pattern_scanner.py:125-181 (VectorizedPatternMixin)
```

**Before**: Iterate every day, run expensive V17 checks.
**After**: Use numpy sliding_window_view to calculate percentiles in O(1) operations.

```python
# Creates view of all 60-day windows at once
v_high = sliding_window_view(highs, window_size)  # Shape: (N-60, 60)

# Calculate 95th percentile for EVERY window simultaneously
wick_upper = np.percentile(v_high, 95, axis=1)
```

**Filter Applied**: Only windows with `wick < 60%` AND `body < 15%` proceed to full V17 checks.

### Step 3: Dual Tightness Metrics (The Accumulation Signature)

```
Location: sleeper_scanner_v17.py:122-168
```

**Metric A - Wick Tightness (Loose)**:
```python
wick_upper = np.percentile(recent['high'], 95)
wick_lower = np.percentile(recent['low'], 10)
wick_height_pct = (wick_upper - wick_lower) / wick_lower
```

Captures intraday volatility. Market-cap dependent limits:

| Category | Max Wick Width |
|----------|---------------|
| Nano-cap | 60% |
| Micro-cap | 45% |
| Small-cap | 30% |
| Mid-cap | 22.5% |
| Large-cap | 15% |
| Mega-cap | 7.5% |

**Metric B - Body Tightness (Strict)**:
```python
body_upper = np.percentile(recent['close'], 95)
body_lower = np.percentile(recent['close'], 10)
body_height_pct = (body_upper - body_lower) / body_lower
# MUST be < 15% for ALL market caps
```

**Why Both?**: A stock can have 40% intraday swings (wick) but close within 10% range (body). This is the accumulation signature - wide intraday ranges (shaking out weak hands) with tight closes (strong hands holding).

### Step 4: Thin Stock Buffer

```
Location: sleeper_scanner_v17.py:152-159
```

```python
avg_candle_size = (df['high'] - df['low']).rolling(20).mean().iloc[-1] / current_price
is_thin = avg_candle_size > 0.04  # >4% average daily range

final_wick_allowed = wick_allowed + (0.05 if is_thin else 0.0)  # +5% bonus
```

**Rationale**: Thin stocks naturally have wider spreads and larger daily ranges. This isn't "messy volatility" (bad), it's "thin liquidity volatility" (acceptable structural feature).

### Step 5: Market Cap-Based Asset Physics

```
Location: pattern_scanner.py:766-797
```

Each pattern gets its **own** market cap category based on the **qualification period** (first 10 days):

```python
# Check ALL days in qualification period
for day_idx in range(start_idx, qual_end_idx):
    day_mc = historical_mc_indexed.loc[day_date, 'market_cap']
    categories_during_qualification.add(day_category)

# Use WIDEST category (most permissive)
pattern_market_cap_category = self._select_widest_category(categories)
```

**Why widest?**: If a stock transitions from micro-cap to small-cap during qualification, we use micro-cap limits (45%) to avoid false rejections.

### Step 6: Pattern Boundary Extraction

```
Location: sleeper_scanner_v17.py:183-205
```

Output includes:
- `upper_lid`: 95th percentile of CLOSE prices (body upper boundary)
- `lower_boundary`: Derived from box_width
- `trigger`: upper_lid x 1.05 (5% above boundary = breakout confirmation)
- `box_width`: Body height percentage (strict metric)
- `wick_width`: Wick height percentage (loose metric)

---

## Part 3: The Labeling Process (Ultra-Detailed Step-by-Step)

### Overview: "First Event Wins" Path-Dependent Labeling

```
Location: path_dependent_labeler.py
```

The labeler simulates **realistic trading** by walking forward day-by-day through the 100-day outcome window, checking for stop or target hits in the ORDER they would occur in reality.

### Step 1: Temporal Safety Check

```python
# pattern_end_idx + outcome_window must be within available data
if pattern_end_idx + self.outcome_window >= len(full_data):
    return None  # Can't label - insufficient future data
```

**Guarantee**: We only label patterns where the full 100-day outcome window has elapsed.

### Step 2: Calculate Risk Parameters

```
Location: path_dependent_labeler.py:211-255
```

**Entry Price** (using adj_close for split adjustment):
```python
price_col = 'adj_close' if 'adj_close' in full_data.columns else 'close'
entry_price = full_data.iloc[pattern_end_idx][price_col]
```

**Stop Loss Calculation** (R-Multiple mode):
```python
stop_loss = lower_boundary * (1 - stop_buffer)  # 8% below boundary
R = lower_boundary - stop_loss  # Risk unit

# Example: lower=$12.26, buffer=8%
# stop_loss = $12.26 x 0.92 = $11.28
# R = $12.26 - $11.28 = $0.98
```

**Target & Grey Zone**:
```python
target_price = entry_price + (5.0 x R)      # +5R for Target (Class 2)
grey_threshold = entry_price + (2.5 x R)    # +2.5R for Grey Zone
```

### Step 3: Path-Dependent Walk-Forward Loop

```
Location: path_dependent_labeler.py:263-284
```

```python
for t in range(len(outcome_data)):
    low_t = outcome_data.iloc[t]['low']       # Intraday low
    close_t = outcome_data.iloc[t][price_col]  # Adjusted close

    # PRIORITY 1: Check stop using LOW (worst-case execution)
    if low_t < stop_loss:
        outcome_day = t + 1
        label = 0  # Danger
        break  # FIRST EVENT WINS

    # PRIORITY 2: Check target with next-day confirmation
    if t < len(outcome_data) - 1:
        open_next = outcome_data.iloc[t + 1]['open']
        if close_t >= target_price and open_next >= target_price:
            outcome_day = t + 1
            label = 2  # Target
            break  # FIRST EVENT WINS
```

**Critical Design Decisions**:

1. **Stop checked using LOW**: Intraday wicks can gap through stops. Using close would miss gapped breakdowns.

2. **Target requires next-open confirmation**: Prevents labeling wicks as "hits". Price must close above target AND open above target next day.

3. **Stop has priority over target**: If both triggered same day, stop wins. This is conservative and reflects real execution.

### Step 4: Classify Remaining Patterns

```
Location: path_dependent_labeler.py:287-300
```

If the loop completes without stop/target hit:

```python
if label is None:
    max_close = outcome_data[price_col].max()

    if max_close >= grey_threshold:
        # Grey zone -> Noise (Jan 2026 change)
        label = 1  # Was -1, now 1
    else:
        label = 1  # Noise (stayed in range)
```

**Jan 2026 Change**: Grey zone (patterns that hit 2.5R but not 5R) are now labeled as Noise (Class 1) instead of being excluded (-1). Rationale: If it didn't hit target with confirmation, it's effectively noise. This adds ~10-15% more training data.

### Step 5: Return Label + Metadata

```python
metadata = {
    'outcome_day': outcome_day,     # Day 1-100 when outcome occurred
    'entry_price': entry_price,
    'stop_loss': stop_loss,
    'target_price': target_price,
    'risk_unit': R,
    'labeling_mode': 'r_multiple',  # or 'atr'
    'market_cap_category': market_cap_category
}
return label, metadata
```

### Final Class Distribution

| Class | Name | Condition | Strategic Value | Typical Rate |
|-------|------|-----------|-----------------|--------------|
| 0 | Danger | Low < stop_loss (first) | -1.0 | ~40-50% |
| 1 | Noise | Neither stop nor target hit | -0.1 | ~35-45% |
| 2 | Target | Close>=target + Next_open>=target | +5.0 | ~15-25% |

---

## Part 4: Critical Assumption Check (Ultra-Think Deliberation)

### Assumption 1: "Tight closes indicate accumulation"

**Validity**: PARTIAL

**Evidence For**:
- Institutional execution algorithms prefer specific price levels
- VWAP/TWAP strategies create tight closing patterns
- Academic research on price clustering supports this

**Risks**:
- Could be low liquidity artifact, not accumulation
- Could be market maker manipulation (painting the tape)
- Could be algorithmic arbitrage, not directional accumulation

**Guardrail**: The $50k liquidity gate + ghost detection helps filter manipulation, but 20% of detected patterns may still be false positives from structural artifacts.

### Assumption 2: "100-day outcome window captures pattern resolution"

**Validity**: HIGH for this market segment

**Evidence For**:
- Micro/small-cap catalysts typically play out within 1-3 months
- SEC filing windows align with 90-day cadence
- Earnings cycles are quarterly

**Risks**:
- Some patterns may be "early" (catalyst in month 4-6)
- Market regime changes can invalidate patterns mid-resolution

**Guardrail**: The +5R target is aggressive enough that genuinely explosive moves are captured early in the window. The 100-day period is conservative.

### Assumption 3: "R-multiples are the correct risk metric"

**Validity**: DEBATABLE

**Evidence For**:
- Pattern-specific (adapts to boundary width)
- Creates comparable risk units across different stocks

**Risks**:
- Ignores market regime (high vol periods need wider stops)
- Ignores position sizing / portfolio context
- 8% stop buffer may be too tight for nano-caps, too wide for large-caps

**Guardrail**: ATR-based labeling (Jan 2026 addition) provides alternative that adapts to market regime. A/B testing recommended.

### Assumption 4: "Next-open confirmation prevents wick gaming"

**Validity**: HIGH

**Evidence For**:
- Intraday wicks are common in micro-caps (manipulation, thin liquidity)
- Requiring next-open prevents labeling flash crashes/spikes as events

**Risks**:
- Gap-up scenarios may see next-open below target despite legitimate breakout
- May slightly undercount true targets

**Guardrail**: This is conservative by design. Missing some targets is acceptable; false targets are costly.

### Assumption 5: "Historical market cap is appropriate for pattern categorization"

**Validity**: HIGH (important fix)

**Evidence For**:
- A $50M stock in 2020 is micro-cap then, regardless of today's size
- WIDTH_LIMITS must match the stock's volatility profile AT THAT TIME
- Many micro-caps become small-caps after breakout -> using current cap would mislabel historical patterns

**Risks**:
- Historical market cap data may have gaps
- Splits/mergers can create discontinuities

**Guardrail**: `market_cap_source` field tracks whether historical, nearest-date, or fallback was used. Patterns with 'unavailable' source use price-based heuristic.

### Assumption 6: "Look-ahead bias is fully eliminated"

**Validity**: CRITICAL CHECK PASSED (with Two-Registry System)

**Guarantee Chain**:
1. `00_detect_patterns.py` outputs `outcome_class = NULL`
2. `00b_label_outcomes.py` ONLY labels patterns where `end_date + 100 days <= today`
3. `01_generate_sequences.py` uses features from pattern period ONLY
4. `02_train_temporal.py` uses ENFORCED temporal split (no random)

**Remaining Risks**:
- Developer error using `--with-labeling` flag in production
- Bug in sequence generator using future indices

**Guardrail**: Tests in `tests/test_temporal_integrity.py` validate no look-ahead.

---

## Part 5: System Guardrails Summary

### Detection Guardrails

| Guardrail | Location | Purpose |
|-----------|----------|---------|
| Liquidity Gate ($50k) | sleeper_scanner_v17.py:103 | Filter untradeable stocks |
| Ghost Detection (4 zero-vol days) | sleeper_scanner_v17.py:111 | Filter dead stocks |
| Body Tightness (15% max) | sleeper_scanner_v17.py:166 | Require accumulation signature |
| Wick Tightness (market-cap based) | sleeper_scanner_v17.py:163 | Allow appropriate volatility |
| Share Dilution Filter (US, >20%) | pattern_scanner.py:322-343 | Reject diluting stocks |

### Labeling Guardrails

| Guardrail | Location | Purpose |
|-----------|----------|---------|
| adj_close requirement | path_dependent_labeler.py:204-209 | Prevent split artifacts |
| Stop checked using LOW | path_dependent_labeler.py:271 | Worst-case execution |
| Target requires next-open confirm | path_dependent_labeler.py:278-283 | Filter wicks |
| 80% valid volume days | path_dependent_labeler.py:498-512 | Data integrity |
| Trading halt detection | path_dependent_labeler.py:364-426 | Data integrity |

### Temporal Integrity Guardrails

| Guardrail | Location | Purpose |
|-----------|----------|---------|
| Two-Registry separation | 00_detect + 00b_label | Prevent look-ahead |
| 100-day outcome window check | path_dependent_labeler.py:187 | Only label ripe patterns |
| Enforced temporal split | 02_train_temporal.py | No random shuffle |
| Indicator warmup (30 days) | constants.py:190 | Valid ADX/BBW values |

### Training Guardrails

| Guardrail | Location | Purpose |
|-----------|----------|---------|
| NMS De-duplication | 01_generate_sequences.py | Remove overlapping patterns |
| Physics Filter | 01_generate_sequences.py | Remove untradeable patterns |
| Robust Scaling (features 8-11) | 02_train_temporal.py | Normalize composite features |
| Class Weights (K0: 5.0) | constants.py:282 | Penalize false positives |
| Gamma Per Class (K0: 4.0) | constants.py:271 | Focus on hard negatives |

---

## Conclusion

This system represents a sophisticated attempt to capture institutional accumulation patterns while maintaining strict temporal integrity. The key innovations are:

1. **Dual-tightness metrics** (body vs wick) - captures the accumulation signature
2. **Two-registry separation** - guarantees no look-ahead bias
3. **Path-dependent "first event wins" labeling** - simulates realistic trade execution
4. **Market-cap-aware asset physics** - adapts to volatility profiles by stock size
5. **Conservative target confirmation** - prioritizes precision over recall

The system is designed to be conservative: it would rather miss targets than generate false positives, because in live trading, false positives cost real money.

---

## Related Documentation

For even more detailed technical specifications, see:

- **[PATTERN_DETECTION_QUALIFICATION_GUIDE.md](./PATTERN_DETECTION_QUALIFICATION_GUIDE.md)**: Ultra-detailed reference covering:
  - Complete qualification phase criteria with exact thresholds
  - State machine transitions with code references
  - Breakout/breakdown logic with production verification
  - Price/volume behavior insights (Vol_DryUp_Ratio, BBW, ADX)
  - All constants verified against production code (eu_model_clean.pt)
