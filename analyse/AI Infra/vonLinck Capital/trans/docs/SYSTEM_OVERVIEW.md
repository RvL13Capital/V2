# TRAnS System Overview

**T**emporal **R**etail **An**alysis **S**ystem for micro/small-cap consolidation breakout detection.

---

## Executive Summary

TRAnS is a production-grade machine learning system that identifies consolidation patterns in micro/small-cap stocks with high probability of significant upward breakouts. The system uses a four-step temporal pipeline with strict separation of concerns to prevent look-ahead bias.

| Parameter | Value |
|-----------|-------|
| Account Size Target | $10,000 - $100,000 |
| Risk Per Trade | $250 (fixed R-unit) |
| Max Position | $5,000 or 4% ADV |
| Execution | Manual/semi-automated, EOD |
| Current Performance | EU: 2.7x lift (Jan 30, 2026) |

---

## Pipeline Architecture

```
Raw OHLCV Data
      |
      v
+---------------------------+
| STEP 0: DETECT PATTERNS   |  00_detect_patterns.py
| ConsolidationScanner      |  -> candidate_patterns.parquet
+---------------------------+
      |
      v  [40+ day ripening delay]
+---------------------------+
| STEP 0b: LABEL OUTCOMES   |  00b_label_outcomes.py
| R-Multiple Classification |  -> labeled_patterns.parquet
+---------------------------+
      |
      v
+---------------------------+
| STEP 1: GENERATE SEQUENCES|  01_generate_sequences.py
| 20x10 Temporal Tensors    |  -> sequences.h5 + metadata.parquet
| NMS + Physics Filters     |
+---------------------------+
      |
      v
+---------------------------+
| STEP 2: TRAIN MODEL       |  02_train_temporal.py
| HybridFeatureNetwork      |  -> best_model.pt
| Temporal Split + CoilLoss |
+---------------------------+
      |
      v
+---------------------------+
| STEP 3: PREDICT & EXECUTE |  03_predict_temporal.py
| EV Calculation + Sizing   |  -> nightly_orders.csv
+---------------------------+
```

---

## 1. Pattern Detection (Step 0)

### What It Does

The pattern scanner identifies consolidation patterns using a two-phase state machine:

**Phase 1: Qualification (Days 1-10)**
- BBW (Bollinger Band Width) < 30th percentile - volatility contraction
- ADX < 32 - low trending strength
- Volume < 35% of 20-day average - suppressed activity
- Daily Range < 65% of 20-day average - tight price action

**Phase 2: Active (Day 10+)**
Pattern monitored until breakout (close > upper boundary) or failure (close < lower boundary).

### Key Concept: The "Coiling Spring"

Consolidation patterns represent potential energy building in a stock:
- Tight ranges indicate indecision and accumulation
- Low volume suggests distribution/accumulation phase
- When the "spring" releases, moves can be explosive

### Output

`candidate_patterns.parquet` with pattern boundaries, duration, and metadata. **No outcome labels** at this stage to prevent look-ahead bias.

---

## 2. Outcome Labeling (Step 0b)

### Structural Risk Framework

Instead of percentage gains, TRAnS uses **R-multiples** (risk multiples):

```
Technical_Floor = Pattern_Lower_Boundary (chart support)
Trigger_Price = Pattern_Upper_Boundary + $0.01
R_Dollar = Trigger_Price - Technical_Floor (risk per share)
```

### Label Classes

| Class | Name | Condition | Strategic Value |
|-------|------|-----------|-----------------|
| 0 | DANGER | Close below Technical_Floor | -2.0 |
| 1 | NOISE | Neither target nor stop within window | -0.1 |
| 2 | TARGET | Hit +3R with volume confirmation | +5.0 |

### Volume Confirmation (Critical)

TARGET requires sustained buying pressure:
- Volume > 2x 20-day average
- Dollar volume > $50,000/day
- **3 consecutive days** meeting both conditions

This prevents false positives from dormant stocks gapping on zero volume.

### Dynamic Labeling Window (V22)

Outcome window adapts to volatility:
- High volatility stocks: 10-day window (resolve faster)
- Low volatility stocks: 60-day window (need more time)

---

## 3. Sequence Generation (Step 1)

### Feature Engineering

Each pattern becomes a **20x10 tensor** (20 timesteps, 10 features per timestep):

| Index | Feature | Purpose |
|-------|---------|---------|
| 0-3 | OHLC | Price action |
| 4 | Volume | Trading activity |
| 5 | BBW | Volatility contraction |
| 6 | ADX | Trend strength |
| 7 | Volume Ratio | Relative volume |
| 8-9 | Boundary Slopes | Pattern geometry |

### Context Features (24 Static)

Complementary features capturing "potential energy":

- **Market Context**: trend position, distance to 52-week high, float size
- **Dormancy Features**: volume shock, dryup ratio
- **Coil Features**: price position, BBW slope, coil intensity
- **Regime Features**: VIX level, market breadth, risk-on indicator

### Log-Diff Transformation

Volume ratios use `log_diff(a, b) = log1p(a) - log1p(b)` to handle dormant stocks with near-zero volumes gracefully.

### Filtering Pipeline

1. **NMS Filter**: Removes 71-73% of overlapping patterns
2. **Physics Filter**: Removes untradeable patterns (liquidity < $50k, gap > 0.5R)
3. **Deduplication**: Cross-ticker + 5-day overlap removal (mandatory)

---

## 4. Model Architecture (Step 2)

### HybridFeatureNetwork

A multi-modal architecture combining temporal patterns with static context:

```
Temporal Input (20x10)          Context Input (24)
        |                              |
        v                              v
+---------------+               +-------------+
| BRANCH A      |               | BRANCH B    |
| LSTM + CNN    |               | GRN         |
| (Dynamics)    |               | (Context)   |
+---------------+               +-------------+
        |                              |
        +--------- CQA (Cross-Modal) --+
                      |
                      v
              +---------------+
              | Fusion Layers |
              +---------------+
                      |
                      v
              +---------------+
              | 3-Class Output|
              | (D, N, T)     |
              +---------------+
```

### Key Components

1. **Context-Conditioned LSTM**: Hidden state initialized from context embedding
2. **Split-Feature Attention**: Separates price vs volume attention
3. **Rotary Positional Embedding (RoPE)**: Position-agnostic pattern recognition
4. **Gated Residual Network (GRN)**: Processes static context features
5. **Context-Query Attention (CQA)**: Cross-modal interaction

### Training

- **Loss**: Coil-Aware Focal Loss (amplifies learning on high-quality coils)
- **Split**: Temporal cluster-based (not random) to prevent leakage
- **Augmentation**: Window jittering, time warping, feature masking

---

## 5. Prediction & Execution (Step 3)

### Expected Value Calculation

```
EV = P(Danger) x (-2.0) + P(Noise) x (-0.1) + P(Target) x (+5.0)
```

| Signal | EV Threshold | Action |
|--------|--------------|--------|
| STRONG | >= 5.0 | High confidence trade |
| GOOD | >= 3.0 | Good confidence trade |
| MODERATE | >= 1.0 | Consider with caution |
| WEAK | >= 0.0 | Weak signal |
| AVOID | < 0.0 | Do not trade |

### Danger Filter

If P(Danger) > 25%, signal is downgraded to AVOID regardless of EV.

### Position Sizing

```
shares = min(
    risk_unit / risk_per_share,      # Risk-based
    max_capital / trigger_price,      # Capital limit
    0.04 * avg_dollar_volume / price  # Liquidity limit
)
```

### Execution Strategy

**Buy Stop Limit Order**:
- Stop Price: Upper boundary + $0.01 (breakout trigger)
- Limit Price: Stop + 0.5R (slippage cap)
- Stop Loss: Lower boundary (technical floor)

**Exit Plan (Justice Tiers)**:
- +3R: Sell 50%
- +5R: Sell 25%
- +10R: Hold 25% with trailing stop

---

## 6. Data Integrity Safeguards

### The "74x Bug" Prevention

Pipeline auto-blocks if:
- Duplication ratio > 1.1x
- Validation dates <= training max (temporal leakage)
- Outcome columns in features
- Metadata-HDF5 label mismatch

### Look-Ahead Bias Prevention

1. **Temporal Ripening**: 40+ day delay between detection and labeling
2. **Walk-Forward Validation**: Test data is always future relative to training
3. **Event Anchor Restriction**: Only searches up to pattern.end_date
4. **FUTURE_* Columns**: Post-pattern metadata blocked from model input

---

## 7. Quick Reference

### Required CLI Flags

```bash
python main_pipeline.py --region EU --full-pipeline \
    --apply-nms \                 # Remove overlapping patterns
    --apply-physics-filter \      # Remove untradeable patterns
    --mode training \             # Preserve dormant stocks
    --use-coil-focal \            # Coil-aware loss function
    --disable-trinity \           # Pattern-level temporal splits
    --skip-market-cap-api         # Use PIT market cap estimation
```

### Key Files

| File | Purpose |
|------|---------|
| `main_pipeline.py` | Unified orchestrator |
| `models/temporal_hybrid_unified.py` | Model architecture |
| `config/feature_registry.py` | Feature definitions |
| `utils/data_integrity.py` | Validation checks |

### Performance Metrics

| Metric | Target |
|--------|--------|
| Top-15% Precision | > 2x baseline lift |
| EV Correlation | > 0.3 |
| Danger Detection | High accuracy on K0 |

---

## 8. System Philosophy

### Why Consolidation Patterns?

Consolidation represents a battle between buyers and sellers reaching equilibrium. When this equilibrium breaks, the resulting move often exceeds what random price action would suggest.

### Why R-Multiples?

Fixed percentage targets ignore risk. A 10% gain on a 2% risk pattern is excellent (5R). The same 10% on a 15% risk pattern is poor (0.67R). R-multiples normalize for structural risk.

### Why Temporal Integrity?

Financial ML is uniquely susceptible to look-ahead bias. A model that "knows" a pattern will break out will appear to predict perfectly but fail in production. TRAnS enforces strict temporal separation at every step.

### Why Coil-Aware Loss?

Not all patterns are equal. Patterns with strong coil characteristics (tight ranges, volume dryup, building momentum) deserve more attention during training. The loss function amplifies learning on these high-quality setups.

---

## 9. Limitations

| Limitation | Mitigation |
|------------|------------|
| Dormant stock volume ratios | Log-diff transformation |
| Broker slippage | 0.5R gap limit + untradeable filter |
| Long-only | Focus on bullish breakouts |
| Intraday stop breach | Use low price for stop check |
| EU market cap APIs fail 99% | PIT estimation with ADV fallback |

---

## 10. Maintenance Schedule

- **Daily**: Run nightly orders, execute signals
- **Weekly**: Monitor signal quality, check drift
- **Quarterly**: Retrain model (90-day cycle)
- **Triggered**: Retrain if PSI > 0.25 (distribution shift)
