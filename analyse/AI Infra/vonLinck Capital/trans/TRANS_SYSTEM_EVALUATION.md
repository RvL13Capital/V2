# TRANS System: Comprehensive In-Depth Evaluation

**Generated:** 2026-01-06
**System Version:** v17 (Production)
**Model Architecture:** Hybrid LSTM + CNN + Attention

---

## Executive Summary

**TRANS (Temporal Sequence Architecture)** is a sophisticated production-ready ML system for detecting consolidation patterns in micro/small-cap stocks with potential for 40%+ breakout gains. The system transforms raw OHLCV data into 14-feature x 20-timestep temporal sequences and uses a hybrid neural network (LSTM + CNN + Attention) to predict 3-class outcomes (Danger/Noise/Target).

---

## 1. System Architecture Overview

### 1.1 Core Innovation

The key innovation is replacing **static 92-feature snapshots** with **temporal sequences (14x20)** that capture pattern evolution over time. This enables the model to learn:
- How volatility contracts before breakouts (BBW trajectory)
- Volume dryup patterns preceding explosive moves
- Price position dynamics within consolidation boundaries

### 1.2 Five-Stage Pipeline

```
00_detect_patterns -> 01_generate_sequences -> 02_train_temporal -> 03_predict_temporal -> 04_evaluate
```

| Stage | Input | Output | Key Component |
|-------|-------|--------|---------------|
| 00 | OHLCV+adj_close | detected_patterns.parquet | State Machine Scanner |
| 01 | Patterns + OHLCV | sequences.npy + labels.npy | PathDependentLabelerV17 |
| 02 | Sequences + Labels | best_model.pt | HybridFeatureNetwork |
| 03 | Model + Sequences | predictions.parquet | EV Calculation |
| 04 | Predictions | metrics.json | Precision@Top15% |

---

## 2. Pattern Detection System (State Machine)

### 2.1 State Machine Architecture

**Location:** `core/consolidation_tracker.py`, `core/sleeper_scanner_v17.py`

```
NONE -> QUALIFYING (10d) -> ACTIVE -> [COMPLETED | FAILED | RECOVERING | DEAD]
```

#### State Definitions:

**NONE:** Initial state, no pattern detected

**QUALIFYING (Days 1-10):** ALL conditions must hold simultaneously:
- `BBW < 30th percentile` - Volatility contraction
- `ADX < 32` - Low trend strength (coiling)
- `Volume < 35% of 20-day avg` - Supply drying up
- `Daily Range < 65% of 20-day avg` - Price tightening

**ACTIVE (Day 10+):** Pattern boundaries established
- Upper boundary = max(highs during qualification)
- Lower boundary = min(lows during qualification)
- Monitor for breakout/breakdown

**COMPLETED:** 2 consecutive closes above upper boundary (breakout)

**FAILED:** 2 consecutive closes below lower boundary (breakdown)

### 2.2 Microstructure-Aware Detection (V17)

**Location:** `core/sleeper_scanner_v17.py:127-412`

Three critical quality gates:

1. **Liquidity Gate ($50k):**
   ```python
   daily_dollar_volume = close * volume
   if daily_dollar_volume < 50000:
       reject_pattern()
   ```

2. **Ghost Detection (20% threshold):**
   ```python
   zero_volume_pct = (volume == 0).sum() / len(volume)
   if zero_volume_pct > 0.20:
       reject_pattern()  # "Ghost stock" - too illiquid
   ```

3. **Intelligent Tightness (Dual Metrics):**
   - **Wick Tightness:** `(high - low) / close` - Target: 40-50%
   - **Body Tightness:** `abs(close - open) / close` - Target: <15%
   - Purpose: Distinguish thin liquidity (signal) from messy volatility (noise)

### 2.3 Vol_DryUp_Ratio (Primary Indicator)

**Formula:**
```python
vol_dryup_ratio = mean(volume[-3:]) / mean(volume[-20:])
```

**Interpretation:**
- `< 0.3` = Imminent move likely (volume severely dried up)
- `0.3-0.5` = Building pressure
- `> 0.5` = Normal activity

---

## 3. Labeling System (V17 Path-Dependent)

### 3.1 R-Multiple Framework

**Location:** `core/path_dependent_labeler.py:58-245`

**Critical Definition:**
```python
R = lower_boundary - stop_loss  # Pattern-specific risk unit
entry = upper_boundary  # Entry at breakout
stop_loss = lower_boundary - (STOP_BUFFER_PERCENT/100 * lower_boundary)
```

**Example:**
```
Upper = $14.02, Lower = $12.26, Stop Buffer = 5%
Stop Loss = $12.26 x 0.95 = $11.65
R = $12.26 - $11.65 = $0.61

For +5R Target: Target Price = $14.02 + (5 x $0.61) = $17.07 (+21.7%)
For -2R Danger: Danger Price = $14.02 - (2 x $0.61) = $12.80
```

### 3.2 Three-Class System

| Class | Name | Strategic Value | Trigger | Meaning |
|-------|------|-----------------|---------|---------|
| 0 | DANGER | -1.0 | -2R hit | Catastrophic failure |
| 1 | NOISE | -0.1 | Neither | No edge realized |
| 2 | TARGET | +5.0 | +5R hit | Explosive breakout |

### 3.3 "First Event Wins" Rule

**Location:** `core/path_dependent_labeler.py:147-178`

```python
for day in range(100):  # 100-day outcome window
    current_price = df.loc[day, 'adj_close']  # Uses adj_close!

    if current_price <= danger_price:
        return 0  # DANGER - immediate exit

    if current_price >= target_price:
        return 2  # TARGET - immediate exit

return 1  # NOISE - neither triggered
```

**Critical:** Uses `adj_close` (not `close`) to handle stock splits correctly. Fixed 2025-12-17.

### 3.4 Regional R-Multiple Configurations

| Region | Config | Optimal EV | Notes |
|--------|--------|------------|-------|
| **EU** | +4R/-1.5R | +5.42 | 5x more profitable |
| **US** | +6R/-2R | +1.12 | Higher volatility |

---

## 4. Feature Engineering (14 Temporal Features)

### 4.1 Feature Index Map

**Location:** `features/vectorized_calculator.py:45-180`

```python
TEMPORAL_FEATURES = {
    # Market data (0-4) - RELATIVIZED to day 0 close
    0: 'open',      # (open_t / close_0) - 1
    1: 'high',      # (high_t / close_0) - 1
    2: 'low',       # (low_t / close_0) - 1
    3: 'close',     # (close_t / close_0) - 1
    4: 'volume',    # log(volume_t / volume_0) - LOG RATIO

    # Technical indicators (5-7)
    5: 'bbw_20',           # Bollinger Band Width (0-1 scale)
    6: 'adx',              # Average Directional Index (0-100)
    7: 'volume_ratio_20',  # volume / 20-day avg

    # Composite scores (8-11) - WINDOW-NORMALIZED
    8: 'vol_dryup_ratio',  # Supply vacuum signal
    9: 'var_score',        # Volume Accumulation Profile
    10: 'nes_score',       # Energy Concentration Index
    11: 'lpf_score',       # Liquidity Flow Pressure

    # Boundaries (12-13) - RELATIVIZED to day 0 close
    12: 'upper_boundary',  # (upper - close_0) / close_0
    13: 'lower_boundary',  # (lower - close_0) / close_0
}
```

### 4.2 Normalization Strategies

**Location:** `core/pattern_detector.py:312-380`

| Feature Group | Normalization | Rationale |
|---------------|---------------|-----------|
| Price (0-3) | Relative to day 0 | No look-ahead |
| Volume (4) | log(v_t / v_0) | Range ~[-3, +5] |
| Technical (5-7) | Raw (bounded) | ADX=0-100, BBW bounded |
| Composite (8-11) | Window-normalized | (x - mean) / std |
| Boundaries (12-13) | Relative to day 0 | Consistent scale |

### 4.3 ADX Warmup Fix (2026-01-06)

**Problem:** ADX requires 14-day warmup. Sequences starting at pattern day 0 had invalid ADX values (~96 instead of ~25-35).

**Solution:** Include `INDICATOR_WARMUP_DAYS` (30 days) of prefix data before pattern start.

**Location:** `core/pattern_detector.py:_generate_sliding_windows()` with `warmup_offset` parameter

---

## 5. Neural Network Architecture

### 5.1 HybridFeatureNetwork

**Location:** `models/temporal_hybrid.py:35-250`

```
Input: (batch, 20 timesteps, 14 features)
              |
      +-------+--------+
      |                |
   Branch A         Branch B
   (Temporal)       (Spatial)
      |                |
   LSTM(32,2)      CNN[3,5,7]
   Bidirectional   MaxPool
      |                |
   Attention(8)        |
      |                |
   Pool -> 64          96
      +-------+--------+
              |
         Concat(160)
              |
         Linear(64)
              |
         Dropout(0.3)
              |
         Linear(3)
              |
         Output: [P(Danger), P(Noise), P(Target)]
```

### 5.2 Branch A: LSTM + Attention (Temporal Coil Shape)

```python
self.lstm = nn.LSTM(
    input_size=14,
    hidden_size=32,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.2
)
# Output: (batch, 20, 64)

self.attention = nn.MultiheadAttention(
    embed_dim=64,
    num_heads=8,
    dropout=0.1,
    batch_first=True
)
# Output: (batch, 20, 64) -> mean pool -> (batch, 64)
```

**Purpose:** Capture temporal dependencies and how patterns evolve over 20 days.

### 5.3 Branch B: CNN (Spatial Features) - CURRENTLY DISABLED

**Location:** `config/constants.py:15`

```python
USE_GRN_CONTEXT: Final[bool] = False  # Pure Temporal/Spatial mode
```

When `USE_GRN_CONTEXT = False`:
- Branch B (GRN Context Gating) is disabled
- Only Branch A (LSTM + Attention) is active
- Simpler, more interpretable architecture

### 5.4 AsymmetricLoss (Focal Loss Variant)

**Location:** `models/asymmetric_loss.py:6-111`

```python
class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 2.0,    # Focus on hard negatives
        gamma_pos: float = 1.0,    # Focus on hard positives
        label_smoothing: float = 0.01  # Prevent mode collapse
    ):
        ...

    def forward(self, logits, targets):
        # Log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)

        # Focal weight: (1 - p_t)^gamma
        p_t = torch.exp(log_probs.gather(1, targets.unsqueeze(1)))
        focal_weight = (1.0 - p_t).pow(gamma_t)

        # Cross-entropy with label smoothing
        ce_loss = F.cross_entropy(logits, targets,
                                   label_smoothing=self.label_smoothing)

        return (focal_weight * ce_loss).mean()
```

**Purpose:** Handle class imbalance by down-weighting easy examples, up-weighting hard examples.

---

## 6. Training Pipeline

### 6.1 Temporal Split (Critical for Temporal Integrity)

**Location:** `utils/temporal_split.py:33-149`

```python
def temporal_train_test_split(dates, split_ratio=0.8, strategy='percentage'):
    """
    ENFORCED: No random shuffle.
    Train on past data, test on future data.
    """
    sorted_indices = dates.argsort().values
    train_size = int(n_samples * split_ratio)

    train_indices = sorted_indices[:train_size]  # Earlier dates
    test_indices = sorted_indices[train_size:]   # Later dates

    # Validate no temporal leakage
    validate_temporal_split(dates, train_indices, test_indices)
```

**Critical:** Random split is BLOCKED. This prevents window overlap leakage.

### 6.2 Training Configuration

**Location:** `pipeline/02_train_temporal.py:180-320`

```python
# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)

# Early stopping
early_stopping_patience = 15
```

### 6.3 Training Loop

```python
for epoch in range(max_epochs):
    model.train()
    for batch in train_loader:
        sequences, labels = batch
        optimizer.zero_grad()
        logits = model(sequences)
        loss = asymmetric_loss(logits, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = evaluate(model, val_loader)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            break
```

---

## 7. Expected Value (EV) System

### 7.1 Strategic Values

**Location:** `config/constants.py:25-35`

```python
STRATEGIC_VALUES = {
    0: -1.0,  # DANGER - catastrophic
    1: -0.1,  # NOISE - slight negative (transaction costs)
    2: +5.0   # TARGET - highly profitable
}
```

### 7.2 EV Calculation

**Location:** `pipeline/03_predict_temporal.py:71-87`

```python
def calculate_expected_value(class_probs: np.ndarray) -> np.ndarray:
    """
    EV = P(Danger) x (-1.0) + P(Noise) x (-0.1) + P(Target) x (+5.0)

    Example:
    P(Danger)=0.10, P(Noise)=0.30, P(Target)=0.60
    EV = 0.10 x (-1.0) + 0.30 x (-0.1) + 0.60 x (+5.0)
       = -0.10 - 0.03 + 3.00
       = 2.87
    """
    values = np.array([STRATEGIC_VALUES[i] for i in range(3)])
    ev = np.sum(class_probs * values, axis=1)
    return ev
```

### 7.3 Signal Thresholds

```python
SIGNAL_THRESHOLDS = {
    'STRONG': 5.0,    # Ultra-selective (<1% of patterns)
    'GOOD': 3.0,      # RECOMMENDED live trading threshold
    'MODERATE': 1.0,  # Borderline
    # EV < 0 = AVOID
}
```

---

## 8. Evaluation System

### 8.1 Primary Metric: Precision @ Top 15%

**Location:** `pipeline/evaluate_trading_performance.py:318-377`

```python
def compute_precision_at_top_k(df, k_percent=15.0):
    """
    THE ONLY METRIC THAT MATTERS FOR LIVE TRADING

    1. Sort all predictions by predicted EV (descending)
    2. Take top k%
    3. Calculate: what % actually hit target?
    """
    df_sorted = df.sort_values('predicted_ev', ascending=False)
    k_count = int(len(df_sorted) * k_percent / 100)
    top_k = df_sorted.head(k_count)

    target_hits = (top_k['actual_label'] == 2).sum()
    precision = target_hits / k_count

    return precision  # Target: 40-60%
```

**Why This Matters:**
- Global accuracy includes patterns you'll never trade (Noise/Danger)
- Precision @ Top 15% = your actual live trading win rate
- If 50% of top 15% hit target -> 50% win rate in live trading

### 8.2 Drift Monitoring (PSI)

**Location:** `pipeline/evaluate_trading_performance.py:466-673`

**Formula:**
```python
PSI = sum((current_% - reference_%) x ln(current_% / reference_%))
```

**Thresholds:**
- PSI < 0.1: No drift
- PSI 0.1-0.25: Moderate drift (monitor)
- **PSI > 0.25: RETRAIN IMMEDIATELY**

**Monitored Features:**
```python
DRIFT_CRITICAL_FEATURES = [
    'vol_dryup_ratio',  # PRIMARY - micro-cap cycle indicator
    'bbw',              # Volatility contraction
]
```

**Use Case:** Detect "AI Season" -> "Bio Season" sector rotations in micro-caps.

---

## 9. Temporal Integrity Guarantees

### 9.1 No Look-Ahead Bias Mechanisms

1. **State Machine:** Can only see past and current day data
2. **Temporal Split:** Train on past, test on future (no random)
3. **Feature Normalization:** Relative to day 0 (no future data)
4. **Indicator Warmup:** 30-day prefix data before pattern start
5. **100-Day Outcome Window:** Labels only after outcome period completes

### 9.2 Temporal Validation

**Location:** `utils/temporal_split.py:235-272`

```python
def validate_temporal_split(dates, train_indices, test_indices):
    """Validates no temporal leakage exists."""
    train_dates = pd.to_datetime(dates.iloc[train_indices])
    test_dates = pd.to_datetime(dates.iloc[test_indices])

    max_train_date = train_dates.max()
    min_test_date = test_dates.min()

    if max_train_date > min_test_date:
        raise ValueError("Temporal leakage detected!")
```

---

## 10. Critical Files Reference

### 10.1 Tier 1: Must-Read Files (Core Understanding)

| File | Lines | Purpose |
|------|-------|---------|
| `models/temporal_hybrid.py` | 250 | Neural network architecture |
| `core/path_dependent_labeler.py` | 500 | V17 3-class labeling |
| `core/sleeper_scanner_v17.py` | 412 | Microstructure-aware detection |
| `config/constants.py` | 200 | Strategic values, thresholds |
| `pipeline/02_train_temporal.py` | 450 | Training loop |

### 10.2 Tier 2: Important Files (Deep Understanding)

| File | Lines | Purpose |
|------|-------|---------|
| `core/consolidation_tracker.py` | 450 | State machine implementation |
| `core/pattern_detector.py` | 600 | Sequence generation |
| `features/vectorized_calculator.py` | 400 | 14-feature extraction |
| `models/asymmetric_loss.py` | 111 | Focal loss for imbalance |
| `pipeline/evaluate_trading_performance.py` | 1057 | Trading metrics + drift |

### 10.3 Tier 3: Supporting Files (Complete Understanding)

| File | Purpose |
|------|---------|
| `utils/temporal_split.py` | Temporal train/test split |
| `config/temporal_features.py` | Feature configuration |
| `ml/drift_monitor.py` | PSI calculation |
| `pipeline/03_predict_temporal.py` | EV prediction |
| `utils/data_loader.py` | GCS/local data loading |

### 10.4 Documentation Files

| File | Purpose |
|------|---------|
| `SYSTEM_OVERVIEW.md` | Complete functional overview |
| `CLAUDE.md` | AI guidance + quick reference |
| `README.md` | Quick start |
| `docs/BACKTEST_PARAMETER_OPTIMIZATION.md` | Parameter tuning |

---

## 11. Key Nuances & Design Decisions

### 11.1 Why 20 Timesteps?

- **10 days qualification** + **10 days validation**
- Qualification: Pattern formation (volatility contraction)
- Validation: Pattern stability (boundaries hold)

### 11.2 Why 14 Features (Not 92)?

Old system had 92 static features -> overfitting, poor generalization.
14 temporal features x 20 timesteps = 280 datapoints per pattern.
Temporal structure captures **how** patterns evolve, not just **what** they look like.

### 11.3 Why 3 Classes (Not 6)?

Original K0-K5 system was too granular:
- K0 (Stagnant) and K1 (Minimal) -> collapsed into NOISE
- K3 (Strong) and K4 (Exceptional) -> collapsed into TARGET
- K5 (Failed) -> DANGER

3 classes align with trading decisions: TRADE / PASS / AVOID.

### 11.4 Why +5R Target?

- +5R at entry = ~50% gain (explosive move)
- Smaller gains (K1/K2) not worth transaction costs + opportunity cost
- Only hunt "home runs" - matches micro-cap volatility profile

### 11.5 Why EU > US (5x More Profitable)?

- EU micro-caps have less algo competition
- Lower liquidity = sharper breakouts
- Fewer pattern "failures" from institutional selling

### 11.6 Why AsymmetricLoss (Not CrossEntropy)?

- Class imbalance: 40-50% Noise, 30% Danger, 20% Target
- CrossEntropy would predict everything as Noise
- Focal loss down-weights easy examples
- gamma=2.0 focuses on hard examples (borderline patterns)

### 11.7 Why Log-Ratio for Volume?

Raw volume (e.g., 20,000 shares) would dominate other features.
```python
log(volume_t / volume_0)
# 0 = same as day 0
# +0.69 = doubled
# -0.69 = halved
# +2.3 = 10x spike
```
Range ~[-3, +5] instead of [0, 100M].

---

## 12. Production Considerations

### 12.1 Model Retraining Protocol

1. Trigger: PSI > 0.25 on vol_dryup_ratio
2. Regenerate sequences (last 90 days only)
3. Train fresh model (don't use old weights)
4. Validate: Precision @ Top 15% >= 40%
5. Deploy only if validation improves

### 12.2 Live Trading Rules

```python
if predicted_ev >= 3.0 and prob_danger < 0.20:
    SIGNAL = "BUY"
    entry = pattern.upper_boundary
    stop = pattern.lower_boundary * 0.95
    target = entry + (5 * (pattern.lower_boundary - stop))
else:
    SIGNAL = "PASS"
```

### 12.3 Model Parameters (~77k)

```
LSTM:      14x32 + 32x32x4 = 4,544 (x2 layers x2 bidirectional)
Attention: 64x64x3 = 12,288
Fusion:    160x64 = 10,240
Output:    64x3 = 192
Total:     ~77,000 parameters
```

---

## 13. System Strengths

1. **Temporal Structure:** Captures pattern evolution, not just snapshots
2. **Path-Dependent Labeling:** Labels based on actual risk/reward outcomes
3. **Microstructure Awareness:** Distinguishes thin liquidity from noise
4. **Drift Detection:** Proactive retraining triggers
5. **Focus on Actionable Metrics:** Precision @ Top 15%, not global accuracy
6. **Temporal Integrity:** Multiple safeguards against look-ahead bias

---

## 14. Potential Improvements

1. **Walk-Forward Validation:** Not yet implemented (`temporal_split.py:296`)
2. **Dynamic R-Multipliers:** Market-cap-aware thresholds (currently static)
3. **GRN Context Branch:** Disabled; could be re-enabled for richer context
4. **Multi-Horizon Prediction:** Currently fixed 100-day outcome window
5. **Ensemble Methods:** Single model; could benefit from model averaging

---

## 15. File Count Summary

```
Total Python files:     ~85
Core files:             12
Pipeline files:         8
Model files:            6
Feature files:          4
Utility files:          15
Test files:             25
Config files:           5
API/Infrastructure:     10
```

---

## 16. Quick Reference Commands

```bash
# Core Pipeline
python pipeline/00_detect_patterns.py --tickers AAPL,MSFT --start-date 2020-01-01
python pipeline/01_generate_sequences.py --input output/detected_patterns.parquet
python pipeline/02_train_temporal.py --sequences output/sequences/sequences.npy
python pipeline/03_predict_temporal.py --model output/models/best_model.pt
python pipeline/04_evaluate.py --predictions output/predictions/predictions.parquet

# Live Trading Evaluation
python pipeline/evaluate_trading_performance.py --split val
python pipeline/evaluate_trading_performance.py --drift-only  # Weekly drift check

# Tests
python -m pytest tests/ -v
```

---

This evaluation covers every major component and nuance of the TRANS system. The critical files listed in Section 10 provide a complete roadmap for an expert-level understanding of the model architecture, data flow, and design decisions.
