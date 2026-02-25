# TRANS System - Complete Functional Overview

**Last Updated:** 2025-12-21
**System Version:** v17 (Production)
**Test Coverage:** 183/184 tests passing (99.5%)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Pipeline (5 Stages)](#core-pipeline-5-stages)
4. [Pattern Detection System](#pattern-detection-system)
5. [Labeling System (v17)](#labeling-system-v17)
6. [Feature Engineering](#feature-engineering)
7. [Neural Network Architecture](#neural-network-architecture)
8. [Prediction & Evaluation](#prediction--evaluation)
9. [Production Infrastructure](#production-infrastructure)
10. [Data Flow Diagram](#data-flow-diagram)
11. [Key Modules Reference](#key-modules-reference)

---

## Executive Summary

**TRANS (Temporal Sequence Architecture)** is a production-ready ML system that predicts which consolidation patterns will lead to explosive price moves (40%+ gains) in micro/small-cap stocks.

### What Makes It Unique

- **Temporal Sequences**: Analyzes 20-day sequences (not static snapshots) to capture pattern evolution
- **Path-Dependent Labeling**: Labels based on actual outcome path, not just final price
- **Risk-Based Classes**: 3 classes (Danger/Noise/Target) aligned with risk/reward ratios
- **Hybrid Neural Network**: LSTM + CNN + Attention learns both temporal and spatial patterns
- **Microstructure-Aware**: Differentiates thin liquidity (signal) from messy volatility (noise)

### Performance Metrics

| Metric | Target | Production Status |
|--------|--------|-------------------|
| Precision @ Top 15% | 40-60% | ✅ Primary live trading metric |
| Pattern Detection | 380ms/ticker | ✅ 3.35x faster than baseline |
| Test Pass Rate | 99%+ | ✅ 183/184 passing |
| Context Efficiency | 45% smaller | ✅ After 2025-12-21 cleanup |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRANS SYSTEM v17                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   RAW DATA   │→→│   PATTERN    │→→│   SEQUENCE   │         │
│  │  OHLCV+adj   │  │  DETECTION   │  │  GENERATION  │         │
│  │   (GCS/Local)│  │  (Sleeper v17)│  │  (20×14)     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                           ↓                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  EVALUATION  │←←│  PREDICTION  │←←│   TRAINING   │         │
│  │ (Precision@15%)│  │  (EV calc)   │  │ (Hybrid NN)  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3-Layer Architecture

**Layer 1: Data & Pattern Detection**
- Data loading (GCS/local with caching)
- Pattern scanning (state machine + microstructure)
- Quality gates (liquidity, ghost detection)

**Layer 2: ML Pipeline**
- Temporal sequence generation
- Risk-based labeling (v17)
- Feature extraction (14 temporal features)
- Model training (hybrid LSTM+CNN+Attention)

**Layer 3: Production Services**
- REST API (FastAPI)
- Prediction service
- Drift monitoring (PSI-based)
- Evaluation metrics

---

## Core Pipeline (5 Stages)

### Stage 0: Pattern Detection
**File:** `pipeline/00_detect_patterns.py`

**Function:** Scan historical price data to identify consolidation patterns

**Input:**
- OHLCV data + adj_close (CSV/Parquet from GCS or local)
- Tickers list or file
- Date range

**Process:**
1. Load price data for each ticker
2. Calculate technical indicators (BBW, ADX, volume ratio, range ratio)
3. Run state machine to detect consolidation phases
4. Apply quality gates:
   - Liquidity gate: $50k minimum daily dollar volume
   - Ghost detection: Reject if >20% zero-volume days
   - Intelligent tightness: Separate thin liquidity from messy volatility
5. Save detected patterns with metadata

**Output:** `output/detected_patterns.parquet`

**Columns:**
```python
{
    'ticker', 'pattern_id', 'start_date', 'end_date',
    'start_idx', 'end_idx', 'days_in_pattern',
    'upper_boundary', 'lower_boundary', 'entry_price',
    'market_cap', 'market_cap_category',
    'liquidity_ok', 'ghost_free', 'state'
}
```

**Command:**
```bash
python pipeline/00_detect_patterns.py \
  --tickers AAPL,MSFT,GOOGL \
  --start-date 2020-01-01 \
  --workers 4
```

---

### Stage 1: Sequence Generation
**File:** `pipeline/01_generate_sequences.py`

**Function:** Convert detected patterns into temporal sequences for ML training

**Input:**
- `output/detected_patterns.parquet`
- Historical OHLCV data

**Process:**
1. For each pattern:
   - Verify 30-day warmup data available
   - Verify 100-day outcome data available
   - Label using PathDependentLabelerV17 (3-class system)
2. Generate sequences using sliding window:
   - Window size: 20 timesteps
   - Stride: 1 day
   - Example: 50-day pattern → 31 sequences (natural augmentation)
3. Extract 14 features per timestep
4. Split into train/val/test (70/15/15)
5. Save as NumPy arrays

**Output:**
- `output/sequences/train/*.npy` (sequences and labels)
- `output/sequences/val/*.npy`
- `output/sequences/test/*.npy`
- `output/sequences/metadata.parquet` (pattern info + labels)

**Sequence Shape:** `(num_windows, 20 timesteps, 14 features)`

**Command:**
```bash
python pipeline/01_generate_sequences.py \
  --input output/detected_patterns.parquet \
  --warmup-days 30 \
  --outcome-days 100
```

---

### Stage 2: Model Training
**File:** `pipeline/02_train_temporal.py`

**Function:** Train hybrid neural network to predict pattern outcomes

**Input:**
- `output/sequences/train/*.npy`
- `output/sequences/val/*.npy`

**Process:**
1. Load sequences and labels
2. Create DataLoaders (batch_size=64)
3. Initialize TemporalHybridModel:
   - LSTM branch (32 hidden, 2 layers, bidirectional)
   - CNN branch (kernels 3,5,7 with max pooling)
   - Attention mechanism (8 heads)
   - Fusion layer → 3 output classes
4. Training loop:
   - Optimizer: Adam (lr=0.001)
   - Loss: AsymmetricLoss (gamma=2.0 for class imbalance)
   - LR scheduler: ReduceLROnPlateau
   - Early stopping: patience=15 epochs
5. Save best model based on validation loss

**Output:**
- `output/models/best_model.pt` (model weights)
- `output/models/training_history.csv` (loss/accuracy per epoch)

**Command:**
```bash
python pipeline/02_train_temporal.py \
  --architecture hybrid \
  --use-asl \
  --epochs 100 \
  --batch-size 64
```

---

### Stage 3: Prediction
**File:** `pipeline/03_predict_temporal.py`

**Function:** Generate predictions for validation/test set

**Input:**
- `output/models/best_model.pt`
- `output/sequences/test/*.npy` (or val)

**Process:**
1. Load trained model
2. Set model to eval mode (disable dropout)
3. For each sequence:
   - Forward pass through model
   - Get logits → apply softmax → class probabilities
   - Calculate Expected Value (EV):
     ```
     EV = P(Danger)×(-10) + P(Noise)×(-1) + P(Target)×(+10)
     ```
4. Save predictions with metadata

**Output:** `output/predictions/predictions.parquet`

**Columns:**
```python
{
    'pattern_id', 'ticker', 'start_date',
    'prob_danger', 'prob_noise', 'prob_target',
    'predicted_class', 'predicted_ev',
    'true_label', 'market_cap_category'
}
```

**Command:**
```bash
python pipeline/03_predict_temporal.py \
  --model output/models/best_model.pt \
  --split test
```

---

### Stage 4: Evaluation
**File:** `pipeline/04_evaluate.py`

**Function:** Calculate comprehensive performance metrics

**Input:** `output/predictions/predictions.parquet`

**Process:**
1. **Class Metrics:**
   - Precision, Recall, F1 per class
   - Confusion matrix
   - Class distribution analysis

2. **Trading Metrics (CRITICAL):**
   - **Precision @ Top 15%**: Of highest-EV predictions, what % hit target?
   - EV > 3.0 target rate (live trading threshold)
   - EV > 3.0 signal rate (selectivity)
   - Danger rate in top predictions (risk control)

3. **Calibration:**
   - Predicted EV vs actual EV correlation
   - Calibration error by EV bucket
   - Reliability curves

4. **Drift Detection:**
   - PSI (Population Stability Index) for key features
   - Alert if PSI > 0.25 (retrain immediately)

**Output:**
- `output/evaluation/metrics.json`
- `output/evaluation/confusion_matrix.png`
- `output/evaluation/calibration_curve.png`
- Console report with pass/fail thresholds

**Command:**
```bash
python pipeline/04_evaluate.py \
  --predictions output/predictions/predictions.parquet
```

---

## Pattern Detection System

### State Machine Architecture

**File:** `core/aiv7_components/consolidation/state_manager.py`

**States:**
```
NONE → QUALIFYING → ACTIVE → [COMPLETED | FAILED | RECOVERING | DEAD]
```

**State Definitions:**

1. **NONE**: No pattern detected
   - Initial state, reset after pattern ends

2. **QUALIFYING** (Days 1-10):
   - **Entry Conditions** (ALL must hold):
     - BBW < 30th percentile (volatility contraction)
     - ADX < 32 (low trending strength)
     - Volume < 35% of 20-day average (drying up)
     - Daily Range < 65% of 20-day average (tightening)
   - **Duration**: 10 consecutive days
   - **Exit**: Any condition fails → reset to NONE

3. **ACTIVE** (Day 10+):
   - **Entry**: Successfully completed 10-day qualification
   - **Boundaries**: Upper/lower established from qualification phase
   - **Monitoring**:
     - Track closes relative to boundaries
     - Count boundary violations
     - Monitor volume spikes
   - **Exit Triggers**:
     - Breakout (2 consecutive closes above upper) → COMPLETED
     - Breakdown (2 consecutive closes below lower) → FAILED
     - Max duration exceeded → DEAD
     - Ghost detection (too many zero-volume days) → DEAD

4. **Terminal States**:
   - **COMPLETED**: Successful breakout above upper boundary
   - **FAILED**: Breakdown below lower boundary
   - **RECOVERING**: Brief violation, attempting to recover
   - **DEAD**: Pattern invalidated (too long, ghost, etc.)

### Microstructure-Aware Detection

**File:** `core/sleeper_scanner_v17.py`

**Key Innovations:**

1. **Liquidity Gate**
   ```python
   min_liquidity_dollar = 50000
   daily_dollar_volume = close * volume
   if daily_dollar_volume < min_liquidity_dollar:
       reject_pattern()
   ```

2. **Ghost Detection**
   ```python
   zero_volume_pct = (volume == 0).sum() / len(volume)
   if zero_volume_pct > 0.20:  # More than 20% zero-volume days
       reject_pattern()
   ```

3. **Intelligent Tightness** (Dual Metrics)
   - **Wick Tightness**: (high - low) / close
     - Target: 40-50% for accumulation
   - **Body Tightness**: abs(close - open) / close
     - Target: <15% for controlled accumulation
   - **Purpose**: Differentiate thin liquidity (signal) from messy volatility (noise)

4. **Vol_DryUp_Ratio**
   ```python
   vol_dryup_ratio = mean(volume[-3:]) / mean(volume[-20:])
   if vol_dryup_ratio < 0.3:
       # Imminent move likely (volume severely dried up)
       flag_as_high_quality()
   ```

### Boundary Calculation

**File:** `core/aiv7_components/consolidation/boundary_manager.py`

**Method:** Conservative approach using qualification phase data

```python
# During 10-day qualification:
highs = df['high'].iloc[qualification_start:qualification_end]
lows = df['low'].iloc[qualification_start:qualification_end]

# Boundaries
upper_boundary = highs.max()  # Resistance level
lower_boundary = lows.min()   # Support level

# Boundary range (volatility measure)
boundary_range_pct = (upper_boundary - lower_boundary) / lower_boundary * 100
```

**Validation:**
- Boundaries must be at least 5% apart (min_boundary_range_pct)
- Boundaries updated if pattern extends qualification
- Used for breakout/breakdown detection in ACTIVE phase

---

## Labeling System (v17)

**File:** `core/path_dependent_labeler.py`

### Path-Dependent Risk-Based Labeling

**Core Concept:** Label patterns based on risk/reward from entry, not just final price.

**Configuration (as of 2025-12-17):**
```python
RISK_MULTIPLIER_TARGET = 5.0   # 5R = +50% from entry (only explosive moves)
RISK_MULTIPLIER_GREY = 2.5     # Grey zone: 2.5R to 5R
STOP_BUFFER_PERCENT = 5.0      # 5% below lower boundary for stop loss
```

**Risk Calculation:**
```python
entry_price = pattern.entry_price
stop_loss = pattern.lower_boundary * (1 - STOP_BUFFER_PERCENT/100)
initial_risk = entry_price - stop_loss  # R

target_price = entry_price + (RISK_MULTIPLIER_TARGET * initial_risk)
grey_zone_price = entry_price + (RISK_MULTIPLIER_GREY * initial_risk)
danger_price = stop_loss
```

### 3-Class System

**Class 0: DANGER** (Value: -10)
- **Trigger**: Price hits stop loss (lower_boundary - 5%)
- **When**: Any time during 100-day outcome window
- **Meaning**: Risk realized, pattern failed catastrophically
- **Label immediately, ignore future recovery**

**Class 1: NOISE** (Value: -1)
- **Trigger**: Neither target nor danger hit
- **Gain Range**: Between -2R and +5R
- **Meaning**: Pattern didn't deliver meaningful edge
- **Most common class** (~40-50% of patterns)

**Class 2: TARGET** (Value: +10)
- **Trigger**: Price hits 5R target (+50% from entry)
- **When**: Any time during 100-day outcome window
- **Meaning**: Explosive move, exactly what we're hunting
- **Label immediately** ("First Event Wins")

### First Event Wins Rule

```python
for day in outcome_window:
    current_price = df.loc[day, 'adj_close']

    # Check danger FIRST (risk control)
    if current_price <= danger_price:
        return 0  # DANGER - exit immediately

    # Check target
    if current_price >= target_price:
        return 2  # TARGET - exit immediately

# If loop completes without trigger
return 1  # NOISE
```

**Critical:** Uses `adj_close` to handle stock splits correctly (fixed 2025-12-17).

### Dynamic Profiles (Market Cap Aware)

The system adjusts thresholds based on market cap category:

```python
profiles = {
    'micro_cap': {
        'risk_multiplier_target': 7.0,   # Higher targets for micros
        'stop_buffer_percent': 8.0       # Wider stops (more volatile)
    },
    'small_cap': {
        'risk_multiplier_target': 5.0,
        'stop_buffer_percent': 5.0
    },
    'mid_cap': {
        'risk_multiplier_target': 3.0,
        'stop_buffer_percent': 3.0
    }
}
```

**Mode:** Currently using "Static Parameters" (mid_cap profile for all), but can enable dynamic mode.

---

## Feature Engineering

### 14 Temporal Features

**File:** `features/vectorized_calculator.py`

Each of the 20 timesteps has 14 features:

**1-2. Pattern Structure**
- `upper_boundary`: Resistance level (static during pattern)
- `lower_boundary`: Support level (static during pattern)

**3-6. Consolidation Indicators**
- `bbw`: Bollinger Band Width (volatility measure)
  ```python
  bbw = (upper_bb - lower_bb) / middle_bb
  ```
- `adx`: Average Directional Index (trend strength)
  - ADX < 25 = weak trend (good for consolidation)
- `volume_ratio`: Current volume / 20-day average
  - < 0.35 = drying up (qualification criterion)
- `range_ratio`: Daily range / 20-day average range
  - < 0.65 = tightening (qualification criterion)

**7-9. Technical Indicators**
- `cci`: Commodity Channel Index (overbought/oversold)
  ```python
  cci = (typical_price - sma) / (0.015 * mean_deviation)
  ```
- `rsi`: Relative Strength Index (momentum)
  - < 30 = oversold, > 70 = overbought
- `atr`: Average True Range (volatility in dollars)

**10-12. Pattern Context**
- `price_position`: Where price sits in pattern
  ```python
  price_position = (close - lower_boundary) / (upper_boundary - lower_boundary)
  ```
  - 0.0 = at lower boundary
  - 0.5 = middle of pattern
  - 1.0 = at upper boundary
- `days_in_pattern`: Pattern age (0 to N)
- `boundary_range_pct`: Pattern width relative to price
  ```python
  boundary_range_pct = (upper_boundary - lower_boundary) / lower_boundary * 100
  ```

**13-14. Raw Data**
- `close`: Closing price
- `volume`: Trading volume

### Feature Computation Pipeline

**Key Methods:**

1. **`calculate_bbw(df, window=20)`**
   - Calculates Bollinger Band Width
   - Uses 2 standard deviations
   - Vectorized with rolling windows

2. **`calculate_adx(df, window=14)`**
   - Calculates Average Directional Index
   - +DI, -DI, then ADX smoothing
   - Measures trend strength (not direction)

3. **`calculate_cci(df, window=20)`**
   - Commodity Channel Index
   - Typical price: (high + low + close) / 3
   - Mean absolute deviation calculation

4. **`calculate_rsi(df, window=14)`**
   - Relative Strength Index
   - Gain/loss ratio smoothed
   - Returns 0-100 scale

5. **`calculate_all_features(df, pattern_info)`**
   - **Master function** called by sequence generator
   - Adds all 14 features to DataFrame
   - Handles missing data gracefully
   - Returns numpy arrays (no pandas Series - critical fix 2025-11-24)

### Normalization

**Not applied during sequence generation.** The neural network learns to handle raw feature scales through:
- Batch normalization in CNN layers
- LSTM's inherent ability to handle varying scales
- Attention mechanism learns relative importance

---

## Neural Network Architecture

### Hybrid Model: LSTM + CNN + Attention

**File:** `models/temporal_hybrid.py`

**Class:** `TemporalHybridModel`

**Input Shape:** `(batch_size, 20 timesteps, 14 features)`

**Architecture Diagram:**

```
Input: (batch, 20, 14)
        ↓
    ┌───┴────┐
    │        │
 Branch A  Branch B
  (Coil)   (Context)
    │        │
  LSTM     CNN
    │        │
Attention    │
    │        │
    └───┬────┘
        ↓
     Fusion
        ↓
    Dropout
        ↓
  FC → Logits (3)
        ↓
    Softmax
        ↓
Probabilities (3)
```

### Component Breakdown

**1. Branch A: LSTM + Attention (Temporal Coil Shape)**

```python
# Bi-directional LSTM
self.lstm = nn.LSTM(
    input_size=14,
    hidden_size=32,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.3
)
# Output: (batch, 20, 64)  # 32*2 from bidirectional

# Multi-head attention
self.attention = nn.MultiheadAttention(
    embed_dim=64,
    num_heads=8,
    dropout=0.3,
    batch_first=True
)
# Output: (batch, 20, 64)

# Global pooling
attention_output_pooled = attention_output.mean(dim=1)  # (batch, 64)
```

**Purpose:** Capture temporal dependencies and pattern evolution over 20 days.

**2. Branch B: CNN (Spatial Features)**

```python
# 1D Convolutions with different kernel sizes
self.conv1 = nn.Conv1d(14, 32, kernel_size=3, padding=1)  # Short patterns
self.conv2 = nn.Conv1d(14, 32, kernel_size=5, padding=2)  # Medium patterns
self.conv3 = nn.Conv1d(14, 32, kernel_size=7, padding=3)  # Long patterns

# Max pooling
pooled = F.max_pool1d(conv_out, kernel_size=20)  # (batch, 96, 1)

# Flatten
cnn_features = pooled.view(batch_size, -1)  # (batch, 96)
```

**Purpose:** Extract spatial patterns and local features across the sequence.

**3. Fusion Layer**

```python
# Concatenate branches
combined = torch.cat([lstm_features, cnn_features], dim=1)  # (batch, 64+96=160)

# Fusion
self.fusion = nn.Linear(160, 64)
fused = F.relu(self.fusion(combined))  # (batch, 64)
```

**4. Output Layer**

```python
self.dropout = nn.Dropout(0.5)
self.fc = nn.Linear(64, 3)  # 3 classes

# Forward pass
x = self.dropout(fused)
logits = self.fc(x)  # (batch, 3)
return logits
```

### Training Details

**Loss Function:** Asymmetric Loss (ASL)

**File:** `models/asymmetric_loss.py`

```python
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma=2.0):
        self.gamma = gamma

    def forward(self, logits, targets):
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Get probability of true class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Asymmetric weighting
        focal_weight = (1 - pt) ** self.gamma

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Weighted loss
        loss = focal_weight * ce_loss
        return loss.mean()
```

**Purpose:** Down-weight easy examples (high confidence), up-weight hard examples. Handles class imbalance without SMOTE.

**Optimizer:** Adam
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)
```

**LR Scheduler:** ReduceLROnPlateau
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)
```

**Early Stopping:** Patience = 15 epochs (no improvement on validation loss)

---

## Prediction & Evaluation

### Expected Value Calculation

**File:** `config/constants.py`

**Strategic Values:**
```python
STRATEGIC_VALUES = {
    0: -10,  # DANGER - avoid at all costs
    1: -1,   # NOISE - slight negative (transaction costs, opportunity cost)
    2: +10   # TARGET - highly profitable
}
```

**EV Formula:**
```python
def calculate_expected_value(probs_dict):
    """
    probs_dict = {0: p_danger, 1: p_noise, 2: p_target}
    """
    ev = sum(probs_dict[k] * STRATEGIC_VALUES[k] for k in range(3))
    return ev

# Example:
probs = {0: 0.10, 1: 0.40, 2: 0.50}
ev = 0.10*(-10) + 0.40*(-1) + 0.50*(+10)
   = -1.0 - 0.4 + 5.0
   = 3.6  # GOOD SIGNAL
```

### Signal Thresholds

```python
EV_STRONG = 5.0      # Conservative, high conviction
EV_GOOD = 3.0        # ⭐ Live trading threshold (recommended)
EV_MODERATE = 1.0    # Borderline
# EV < 0.0 = AVOID
```

**Live Trading Rule:**
```python
if predicted_ev >= EV_GOOD and prob_danger < 0.20:
    signal = "BUY"
    position_size = calculate_kelly_criterion(ev, prob_target)
else:
    signal = "PASS"
```

### Precision @ Top 15%

**The Primary Metric for Live Trading**

**Concept:** Global accuracy is irrelevant. What matters is the accuracy of patterns you actually trade.

**Calculation:**
```python
# 1. Sort all predictions by EV (descending)
df_sorted = predictions.sort_values('predicted_ev', ascending=False)

# 2. Take top 15%
top_15_pct = int(len(df_sorted) * 0.15)
top_predictions = df_sorted.head(top_15_pct)

# 3. Calculate precision
targets_hit = (top_predictions['true_label'] == 2).sum()
precision_at_15 = targets_hit / top_15_pct

# Target: 40-60%
```

**Why This Matters:**
- You only trade the highest-EV signals (top 15%)
- If 50% of those hit target → 50% win rate in live trading
- If 30% of those hit target → model needs retraining
- Global accuracy on all patterns is meaningless for trading

### Drift Monitoring

**File:** `pipeline/evaluate_trading_performance.py`

**Problem:** Micro-cap sector rotates quickly (30-90 days). "AI Season" → "Bio Season" → model silently degrades.

**Solution:** Population Stability Index (PSI)

**Formula:**
```python
def calculate_psi(reference_dist, current_dist):
    """
    PSI = Σ (current_% - reference_%) * ln(current_% / reference_%)
    """
    psi = 0
    for i in range(len(bins)):
        if current_pct[i] > 0 and reference_pct[i] > 0:
            psi += (current_pct[i] - reference_pct[i]) * \
                   np.log(current_pct[i] / reference_pct[i])
    return psi
```

**Interpretation:**
- PSI < 0.1: No significant drift
- PSI 0.1-0.25: Moderate drift, monitor closely
- **PSI > 0.25: RETRAIN IMMEDIATELY** 🚨

**Monitored Features:**
1. Vol_DryUp_Ratio (PRIMARY)
2. volume_ratio
3. bbw
4. rsi
5. cci

**Automated Alert:**
```bash
python pipeline/evaluate_trading_performance.py --drift-only

# If PSI > 0.25:
# ⚠️ CRITICAL DRIFT DETECTED
# Feature: vol_dryup_ratio, PSI: 0.38
#
# RETRAIN IMMEDIATELY:
# 1. python pipeline/00_detect_patterns.py --start-date 2024-09-01
# 2. python pipeline/01_generate_sequences.py --force-relabel
# 3. python pipeline/02_train_temporal.py --epochs 100
```

---

## Production Infrastructure

### REST API

**File:** `api/main.py` (FastAPI)

**Endpoints:**

1. **Health Check**
   ```
   GET /health
   Response: {"status": "healthy", "model_loaded": true}
   ```

2. **Scan Patterns**
   ```
   POST /api/scan
   Body: {
     "ticker": "AAPL",
     "start_date": "2023-01-01",
     "end_date": "2024-01-01"
   }
   Response: {
     "patterns_found": 3,
     "patterns": [...],
     "processing_time_ms": 380
   }
   ```

3. **Predict Pattern**
   ```
   POST /api/predict
   Body: {
     "pattern_id": "AAPL_20230515_001",
     "sequence": [[...], ...],  // 20x14 array
   }
   Response: {
     "predicted_class": 2,
     "probabilities": {0: 0.10, 1: 0.30, 2: 0.60},
     "expected_value": 5.6,
     "signal": "STRONG"
   }
   ```

4. **Batch Scan**
   ```
   POST /api/batch-scan
   Body: {
     "tickers": ["AAPL", "MSFT", "GOOGL"],
     "start_date": "2023-01-01"
   }
   Response: {
     "total_patterns": 15,
     "by_ticker": {...},
     "processing_time_ms": 1200
   }
   ```

**Deployment:**
```bash
# Development
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker-compose up -d
```

**API Documentation:** http://localhost:8000/docs

### Database Layer

**File:** `database/models.py`

**7 Production Models:**

1. **Pattern** - Detected consolidation patterns
2. **Sequence** - Generated temporal sequences
3. **Prediction** - Model predictions with EV
4. **ModelVersion** - Trained model metadata
5. **DriftReport** - PSI monitoring results
6. **TrainingRun** - Training history and metrics
7. **Alert** - System alerts (drift, failures, etc.)

**ORM:** SQLAlchemy
**Supported DBs:** PostgreSQL, MySQL, SQLite

**Migration:**
```bash
python -m database.migrate
python -m database.migrate --verify
```

### Monitoring

**Logging:** Structured JSON logs
```python
logger.info("Pattern detected", extra={
    "ticker": "AAPL",
    "pattern_id": "AAPL_20230515_001",
    "days_in_pattern": 45,
    "market_cap": 2500000000
})
```

**Metrics:** Prometheus-compatible
- Pattern detection rate
- Prediction latency
- Model accuracy by market cap
- Drift scores

**Dashboards:** Grafana (http://localhost:3000)
- Real-time pattern scanning
- Model performance over time
- Drift monitoring alerts
- System health metrics

---

## Data Flow Diagram

### Complete System Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                 │
├─────────────────────────────────────────────────────────────────────┤
│  GCS Bucket (gs://ignition-ki-csv-data-2025-user123/tickers/)       │
│  └─ 2,567 tickers × 5-10 years = 5,000 files (CSV/Parquet)          │
│                                                                       │
│  Local Cache (data/raw/)                                             │
│  └─ Fallback for offline development                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      00_DETECT_PATTERNS.PY                           │
├─────────────────────────────────────────────────────────────────────┤
│  ConsolidationPatternScanner                                         │
│  ├─ Load OHLCV + adj_close                                           │
│  ├─ Calculate indicators (BBW, ADX, volume_ratio, range_ratio)      │
│  ├─ State machine (NONE→QUALIFYING→ACTIVE→COMPLETED/FAILED)         │
│  ├─ Quality gates:                                                   │
│  │  ├─ Liquidity: $50k min daily dollar volume                      │
│  │  ├─ Ghost: <20% zero-volume days                                 │
│  │  └─ Tightness: Wick 40-50%, Body <15%                            │
│  └─ Output: detected_patterns.parquet                                │
│     Columns: ticker, pattern_id, start_date, end_date, boundaries,  │
│              market_cap, state, days_in_pattern                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    01_GENERATE_SEQUENCES.PY                          │
├─────────────────────────────────────────────────────────────────────┤
│  PathDependentLabelerV17                                             │
│  ├─ For each pattern:                                                │
│  │  ├─ Verify 30-day warmup + 100-day outcome data                  │
│  │  ├─ Calculate risk: R = entry_price - stop_loss                  │
│  │  ├─ Set targets: Danger=-2R, Target=+5R (50%)                    │
│  │  ├─ Track outcome path (adj_close):                              │
│  │  │  └─ First Event Wins: Danger=0, Target=2, else Noise=1        │
│  │  └─ Label pattern                                                 │
│  │                                                                    │
│  VectorizedFeatureCalculator                                         │
│  ├─ Sliding window (20 timesteps, stride=1)                          │
│  ├─ Extract 14 features per timestep:                                │
│  │  ├─ Boundaries (2), Indicators (6), Technicals (3),              │
│  │  ├─ Context (3), Raw (2)                                          │
│  └─ Natural augmentation: 50-day pattern → 31 sequences              │
│                                                                       │
│  Output:                                                              │
│  ├─ sequences/train/sequences_*.npy (70%)                            │
│  ├─ sequences/val/sequences_*.npy (15%)                              │
│  ├─ sequences/test/sequences_*.npy (15%)                             │
│  └─ sequences/metadata.parquet                                       │
│     Shape: (num_windows, 20, 14)                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      02_TRAIN_TEMPORAL.PY                            │
├─────────────────────────────────────────────────────────────────────┤
│  TemporalHybridModel                                                 │
│  ├─ Architecture:                                                    │
│  │  ├─ Branch A: LSTM (32 hidden, 2 layers, bidirectional)          │
│  │  │           → Attention (8 heads)                                │
│  │  │           → Pooling → (batch, 64)                              │
│  │  │                                                                 │
│  │  ├─ Branch B: CNN (kernels 3,5,7)                                 │
│  │  │           → Max pooling → (batch, 96)                          │
│  │  │                                                                 │
│  │  ├─ Fusion: Concat → Linear(160, 64) → ReLU                      │
│  │  └─ Output: Dropout(0.5) → Linear(64, 3)                         │
│  │                                                                    │
│  Training Loop:                                                       │
│  ├─ Optimizer: Adam (lr=0.001, weight_decay=1e-5)                   │
│  ├─ Loss: AsymmetricLoss (gamma=2.0)                                │
│  ├─ Scheduler: ReduceLROnPlateau (patience=5)                       │
│  ├─ Early stopping: patience=15 epochs                              │
│  └─ Save best model on validation loss                              │
│                                                                       │
│  Output:                                                              │
│  ├─ models/best_model.pt                                             │
│  └─ models/training_history.csv                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    03_PREDICT_TEMPORAL.PY                            │
├─────────────────────────────────────────────────────────────────────┤
│  Load best_model.pt                                                  │
│  ├─ Set eval mode (disable dropout)                                 │
│  ├─ For each sequence in test set:                                  │
│  │  ├─ Forward pass: sequence → logits                              │
│  │  ├─ Softmax: logits → [p_danger, p_noise, p_target]             │
│  │  └─ Calculate EV:                                                 │
│  │     EV = p_danger×(-10) + p_noise×(-1) + p_target×(+10)          │
│  │                                                                    │
│  └─ Save predictions with metadata                                   │
│                                                                       │
│  Output: predictions/predictions.parquet                             │
│  Columns: pattern_id, ticker, prob_*, predicted_ev, true_label      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       04_EVALUATE.PY                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Classification Metrics:                                             │
│  ├─ Per-class precision, recall, F1                                 │
│  ├─ Confusion matrix                                                 │
│  └─ Class distribution                                               │
│                                                                       │
│  Trading Metrics (CRITICAL):                                         │
│  ├─ Precision @ Top 15%:                                             │
│  │  └─ Sort by EV → Take top 15% → % that hit target                │
│  │     Target: 40-60%                                                │
│  │                                                                    │
│  ├─ EV > 3.0 Analysis:                                               │
│  │  ├─ Target rate (should be 40-60%)                               │
│  │  ├─ Signal rate (should be 10-20%)                               │
│  │  └─ Danger rate (should be <20%)                                 │
│  │                                                                    │
│  └─ EV Calibration:                                                  │
│     ├─ Predicted EV vs Actual EV correlation                        │
│     ├─ Calibration error by EV bucket                               │
│     └─ Target: correlation >0.30, error <2.0                        │
│                                                                       │
│  Drift Monitoring:                                                   │
│  ├─ PSI calculation for 5 key features                              │
│  └─ Alert if PSI > 0.25 (retrain needed)                            │
│                                                                       │
│  Output:                                                              │
│  ├─ evaluation/metrics.json                                          │
│  ├─ evaluation/confusion_matrix.png                                  │
│  └─ evaluation/calibration_curve.png                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION DEPLOYMENT                            │
├─────────────────────────────────────────────────────────────────────┤
│  API Server (FastAPI):                                               │
│  ├─ POST /api/scan → Detect patterns for ticker                     │
│  ├─ POST /api/predict → Get EV for pattern                          │
│  └─ GET /health → System status                                      │
│                                                                       │
│  Database (PostgreSQL/MySQL/SQLite):                                 │
│  ├─ Store patterns, predictions, alerts                             │
│  └─ Track drift, model versions                                      │
│                                                                       │
│  Monitoring (Grafana + Prometheus):                                  │
│  ├─ Pattern detection rate                                           │
│  ├─ Model accuracy over time                                         │
│  └─ Drift alerts                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Modules Reference

### Core Modules

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| **ConsolidationPatternScanner** | `core/pattern_scanner.py` | 450 | Orchestrates pattern detection |
| **SleeperScannerV17** | `core/sleeper_scanner_v17.py` | 400 | Microstructure-aware detection |
| **ConsolidationTracker** | `core/aiv7_components/consolidation/tracker.py` | 500 | State machine implementation |
| **PathDependentLabelerV17** | `core/path_dependent_labeler.py` | 500 | Risk-based 3-class labeling |
| **VectorizedFeatureCalculator** | `features/vectorized_calculator.py` | 600 | Extracts 14 temporal features |

### Model Modules

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| **TemporalHybridModel** | `models/temporal_hybrid.py` | 200 | LSTM+CNN+Attention architecture |
| **AsymmetricLoss** | `models/asymmetric_loss.py` | 100 | Handles class imbalance |
| **ModelManager** | `models/model_manager.py` | 300 | Versioning, checkpointing |

### Pipeline Scripts

| Script | Lines | Purpose |
|--------|-------|---------|
| `00_detect_patterns.py` | 200 | Scan tickers for patterns |
| `01_generate_sequences.py` | 250 | Generate 20×14 sequences |
| `02_train_temporal.py` | 300 | Train hybrid neural network |
| `03_predict_temporal.py` | 150 | Generate predictions + EV |
| `04_evaluate.py` | 200 | Calculate trading metrics |
| `evaluate_trading_performance.py` | 400 | Drift monitoring (PSI) |

### Utility Modules

| Module | File | Purpose |
|--------|------|---------|
| **DataLoader** | `utils/data_loader.py` | Load OHLCV from GCS/local |
| **MarketCapFetcher** | `utils/market_cap_fetcher.py` | Fetch market cap (4 API sources) |
| **Indicators** | `utils/indicators.py` | Calculate BBW, ADX, RSI, etc. |
| **DriftMonitor** | `ml/drift_monitor.py` | PSI calculation |

### API & Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| **API Server** | `api/main.py` | FastAPI REST endpoints |
| **Database Models** | `database/models.py` | SQLAlchemy ORM (7 models) |
| **Analytics Dashboard** | `analysis/dashboard/app.py` | Plotly Dash visualization |
| **CLI Tool** | `analysis/trans_analyze.py` | Command-line analysis |

---

## Production Checklist

### Pre-Deployment

- [ ] Verify GCS credentials: `PROJECT_ID`, `GCS_BUCKET_NAME`, `GOOGLE_APPLICATION_CREDENTIALS`
- [ ] Run full test suite: `python -m pytest tests/ -v` (expect 183/184 passing)
- [ ] Verify pipeline: Run 00→01→02→03→04 on sample data
- [ ] Check model performance: Precision @ Top 15% ≥ 40%
- [ ] Validate EV correlation: >0.30
- [ ] Test API endpoints: `curl http://localhost:8000/health`

### Post-Deployment

- [ ] Monitor drift weekly: `python pipeline/evaluate_trading_performance.py --drift-only`
- [ ] Track Precision @ Top 15% daily
- [ ] Alert if PSI > 0.25 (retrain immediately)
- [ ] Retrain quarterly (90 days) minimum
- [ ] Backup model versions before updates

### Live Trading

- [ ] Only signal if `predicted_ev >= 3.0` AND `prob_danger < 0.20`
- [ ] Position size: Kelly criterion or fixed %
- [ ] Stop loss: `lower_boundary × (1 - 0.05)` (5% buffer)
- [ ] Target: `entry_price + (5.0 × initial_risk)` (5R = +50%)
- [ ] Max holding: 100 days or until signal exit
- [ ] Diversify: 10-20 concurrent positions minimum

---

## Appendix: Critical Configuration Values

### Pattern Detection Thresholds

```python
# Qualification criteria (ALL must hold for 10 days)
BBW_PERCENTILE_THRESHOLD = 30        # Volatility contraction
ADX_THRESHOLD = 32                   # Low trending
VOLUME_RATIO_THRESHOLD = 0.35        # Volume drying up
RANGE_RATIO_THRESHOLD = 0.65         # Range tightening

# Quality gates
MIN_LIQUIDITY_DOLLAR = 50000         # $50k daily volume
GHOST_THRESHOLD = 0.20               # Max 20% zero-volume days

# Microstructure
WICK_TIGHTNESS_TARGET = 0.45         # 40-50% for accumulation
BODY_TIGHTNESS_TARGET = 0.15         # <15% for control
VOL_DRYUP_THRESHOLD = 0.30           # <0.3 = imminent move
```

### Labeling Configuration

```python
# Risk multipliers (updated 2025-12-17)
RISK_MULTIPLIER_TARGET = 5.0         # +50% gain (5R)
RISK_MULTIPLIER_GREY = 2.5           # Grey zone threshold
STOP_BUFFER_PERCENT = 5.0            # 5% below lower boundary

# Time windows
WARMUP_DAYS = 30                     # Indicator calculation period
STABLE_DAYS = 100                    # Pattern stability verification
OUTCOME_DAYS = 100                   # Outcome evaluation window
```

### Trading Thresholds

```python
# Expected Value
EV_STRONG = 5.0                      # Conservative, high conviction
EV_GOOD = 3.0                        # ⭐ Live trading threshold
EV_MODERATE = 1.0                    # Borderline

# Strategic Values
STRATEGIC_VALUES = {
    0: -10,  # DANGER
    1: -1,   # NOISE
    2: +10   # TARGET
}

# Risk control
MAX_DANGER_PROBABILITY = 0.20        # <20% danger in top signals
MIN_TARGET_PRECISION = 0.40          # ≥40% Precision @ Top 15%
```

### Model Hyperparameters

```python
# Architecture
LSTM_HIDDEN = 32
LSTM_LAYERS = 2
CNN_CHANNELS = [32, 64, 128]
ATTENTION_HEADS = 8
DROPOUT = 0.3

# Training
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GAMMA_ASL = 2.0                      # Asymmetric loss focal parameter
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Sequence
SEQUENCE_LENGTH = 20                 # Timesteps
NUM_FEATURES = 14                    # Features per timestep
```

### Drift Monitoring

```python
# PSI thresholds
PSI_NO_DRIFT = 0.1                   # No action needed
PSI_MODERATE = 0.25                  # Monitor closely
PSI_CRITICAL = 0.25                  # 🚨 RETRAIN IMMEDIATELY

# Monitored features
DRIFT_FEATURES = [
    'vol_dryup_ratio',               # PRIMARY
    'volume_ratio',
    'bbw',
    'rsi',
    'cci'
]

# Check frequency
DRIFT_CHECK_FREQUENCY = 7            # Weekly
```

---

**END OF SYSTEM OVERVIEW**

*For specific implementation details, see individual module documentation and CLAUDE.md.*
