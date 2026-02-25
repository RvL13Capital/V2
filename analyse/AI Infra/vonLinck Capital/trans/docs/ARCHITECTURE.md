# Architecture

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANS PIPELINE V22 (RETAIL EDITION)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  00_detect_patterns.py ──► candidate_patterns.parquet (outcome_class=NULL) │
│           │                                                                 │
│           ▼ (Wait 10-60 days for ripening - volatility-adjusted)           │
│                                                                             │
│  00b_label_outcomes.py ──► labeled_patterns.parquet                        │
│           │                 • V22: Dynamic Window (1/volatility clamped)   │
│           │                 • Vectorized outcome eval + pessimistic ties   │
│           │                 • Target = +3R hit | Danger = Close < Floor    │
│           ▼                                                                 │
│                                                                             │
│  01_generate_sequences.py ──► sequences.h5 + metadata.parquet              │
│           │                   • --mode training (preserves dormant)        │
│           │                   • --mode inference (strict filters)          │
│           ▼                                                                 │
│                                                                             │
│  02_train_temporal.py ──► best_model.pt + norm_params.json                 │
│           │               • Log1p + Robust Scaling                          │
│           │               • CoilAwareFocalLoss                              │
│           ▼                                                                 │
│                                                                             │
│  03_predict_temporal.py ──► nightly_orders.csv                             │
│                             • Risk-Based Position Sizing (retail-safe)     │
│                             • Intraday Pacing (Vol @ 10AM > 30% avg)       │
│                             • Justice Exit Plan (T1/T2/T3)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## V18 Model Architecture (Production)

```
┌─────────────────────────────────────────────────────────────────┐
│              TEMPORAL HYBRID NETWORK (RETAIL)                    │
│         Pattern Recognition, NOT Speed Competition               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BRANCH A: Temporal (10×20)          BRANCH B: Context (18)    │
│  ┌─────────────────────┐             ┌─────────────────────┐   │
│  │   LSTM + RoPE       │             │   GRN (Gated        │   │
│  │   (h0/c0 from GRN)  │◄────────────│   Residual Network) │   │
│  └─────────┬───────────┘             └─────────────────────┘   │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────────┐                                       │
│  │   1D CNN            │                                       │
│  │   (Pattern Shapes)  │                                       │
│  └─────────┬───────────┘                                       │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────────┐                                       │
│  │   Multi-Head        │                                       │
│  │   Attention         │                                       │
│  └─────────┬───────────┘                                       │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────────┐                                       │
│  │   3-Class Output    │──► [Danger, Noise, Target]            │
│  └─────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design Choices:**
- **Context-Conditioned LSTM**: h0/c0 initialized from GRN (knows float/regime before seeing prices)
- **RoPE**: Rotary Positional Embeddings (breakouts valid at any timestep position)
- **Early Stop on Top-15% Precision**: Not global accuracy (retail cares about best signals)

---

## Feature Schema

### 10 Temporal Features (per timestep)

| Index | Feature | Description | Dormant OK? |
|-------|---------|-------------|-------------|
| 0-3 | OHLC | `(Price_t / Price_0) - 1` | Yes |
| 4 | Volume | `log(volume_t / vol_6m_median)` | Caution if median=0 |
| 5 | BBW_20 | Bollinger Band Width | Yes |
| 6 | ADX | Average Directional Index | Yes |
| 7 | Volume_Ratio_20 | Current vol / 20d avg | **UNSTABLE** |
| 8 | Upper_Slope | Upper boundary convergence | Yes |
| 9 | Lower_Slope | Lower boundary convergence | Yes |

### 24 Context Features (static)

| Index | Feature | Transform | Description |
|-------|---------|-----------|-------------|
| 0 | `retention_rate` | bounded | (Close-Low)/(High-Low) candle strength |
| 1 | `trend_position` | ratio | Price vs SMA_200 |
| 2 | `base_duration` | bounded | Days in consolidation (log-normalized) |
| 3 | `relative_volume` | **LOG-DIFF** | log_diff(vol_20d, vol_60d) |
| 4 | `distance_to_high` | bounded | % below 52-week high |
| 5 | `log_float` | log-scale | log(shares outstanding) |
| 6 | `log_dollar_volume` | log-scale | log(price * volume) |
| 7 | `dormancy_shock` | **LOG-DIFF** | log_diff(vol_20d, vol_252d) |
| 8 | `vol_dryup_ratio` | **LOG-DIFF** | log_diff(vol_20d, vol_100d) |
| 9 | `price_position_at_end` | bounded | Position in box [0,1] |
| 10 | `volume_shock` | **LOG-DIFF** | log_diff(max_3d, avg_20d) |
| 11 | `bbw_slope_5d` | slope | BBW trend |
| 12 | `vol_trend_5d` | **LOG-DIFF** | log_diff(vol_5d, vol_20d) |
| 13 | `coil_intensity` | composite | Composite coil score |
| 14 | `relative_strength_cohort` | ratio | vs universe median |
| 15 | `risk_width_pct` | ratio | (Upper-Lower)/Upper |
| 16 | `vol_contraction_intensity` | **LOG-DIFF** | log_diff(vol_5d, vol_60d) |
| 17 | `obv_divergence` | **LOG-DIFF** | OBV vs Price change |
| 18 | `vol_decay_slope` | slope | Volume trend over 20 days |
| 19 | `vix_regime_level` | bounded | VIX normalized (0=calm, 1=crisis) |
| 20 | `vix_trend_20d` | slope | VIX 20d change |
| 21 | `market_breadth_200` | bounded | % stocks above 200 SMA |
| 22 | `risk_on_indicator` | bounded | Risk-on/off composite |
| 23 | `days_since_regime_change` | bounded | Days since bull/bear cross |

**LOG-DIFF** = `log1p(num) - log1p(denom)` (handles zeros gracefully)

---

## Structural Risk Framework

```python
# Structural Risk (Retail-Realistic)
Technical_Floor = Lower_Boundary        # Support level = stop loss
Trigger_Price   = Upper_Boundary + $0.01
R_Dollar        = Trigger - Floor       # Risk per share

# Entry Execution (assumes slippage)
Entry = MAX(Trigger, Open_{t+1})        # Can't get exact trigger price

# GAP HANDLING
if gap > 0.5R:
    untradeable = True   # Strong signal, but retail can't catch it
    # Continue labeling - don't delete!

# Labels (dynamic window: 10-60 days based on volatility)
Class 0 (Danger): Close < Floor         # Value: -2.0
Class 1 (Noise):  Neither               # Value: -0.1
Class 2 (Target): +3R price hit         # Value: +5.0
```

---

## V22 Vectorized Labeling

```python
stop_mask = outcome_data[price_col] < technical_floor
target_mask = outcome_data['high'] >= target_price

first_stop = stop_mask.idxmax() if stop_mask.any() else None
first_target = target_mask.idxmax() if target_mask.any() else None

# PESSIMISTIC TIE-BREAKER: Same day = stop wins
if first_stop == first_target:
    label = 0  # Danger
elif first_target and (not first_stop or first_target < first_stop):
    label = 2  # Target
elif first_stop:
    label = 0  # Danger
else:
    label = 1  # Noise (timeout)
```

---

## Justice Exit Plan

| Tier | Target | Sell | Rationale |
|------|--------|------|-----------|
| T1 (Bank) | +3R | 50% | Lock in profit |
| T2 (Runner) | +5R | 25% | Capture extension |
| T3 (Moon) | +10R | 25% + trail | Let winners run |

---

## Data Filters

### Training vs Inference Mode

| Filter | Training | Inference | Why? |
|--------|----------|-----------|------|
| Ghost Trades | OFF | ON | Model must learn from messy data |
| Data Gaps (>3d) | OFF | ON | Preserve dormant lottery tickets |
| Lobster Trap ($25k) | ON | ON | Always protect liquidity |
| Sideways Regime | ON | ON | Avoid momentum/crashing |
| Structural (>40%) | ON | ON | Reject untradeable R:R |

---

## Weekly Qualification Mode

Use `--use-weekly-qualification` for longer-term patterns (10 weeks vs 10 days).

| Aspect | Daily Mode | Weekly Mode |
|--------|------------|-------------|
| Qualification Period | 10 days | 10 weeks |
| Pattern Duration | Short | Long sideways/dormant |
| Target Stocks | Quick setups | Slow accumulators |

---

## Bear-Aware Split

| Split | Date Range | Content |
|-------|------------|---------|
| Train | 2020-2022 | Includes 2022 bear market |
| Val | 2023 | Bear-to-bull transition |
| Test | 2024+ | Bull market |

Use `--bear-aware-split` for production training.
