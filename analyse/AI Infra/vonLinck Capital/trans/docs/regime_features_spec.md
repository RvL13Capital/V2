# Regime-Aware Features Specification

## Problem Statement

The TRAnS system currently fails to generalize from 2020-2023 training data to 2024+ test data, with both V18 and V20 architectures achieving approximately 0.99x lift (random chance) on the test set. While the system has a `market_phase` column (bull=1, bear=0) based on 50/200 MA crossovers on the S&P 500, this binary indicator is not providing sufficient regime information for cross-regime generalization.

### Root Cause Analysis

1. **Binary Phase Limitation**: The current `market_phase` is a simple binary (bull/bear) based on MA crossovers with 5% hysteresis. This misses nuanced regime characteristics like:
   - Volatility regimes (calm vs turbulent)
   - Risk appetite (risk-on vs risk-off)
   - Market breadth (narrow vs broad rallies)
   - Credit conditions

2. **Regime Shift in 2024**: The test period (2024+) may represent a fundamentally different market regime:
   - Post-COVID normalization
   - Rising interest rate environment
   - AI/tech concentration effects
   - Different small-cap dynamics

3. **Missing Cross-Asset Information**: The system only uses individual stock features and lacks broader market context that could indicate regime shifts.

---

## Proposed Regime-Aware Features

### Feature 1: VIX Regime Level

**Description**: Captures the current volatility regime using the VIX index, normalized to regime categories.

**Calculation**:
```python
def vix_regime_level(vix_value: float) -> float:
    """
    Normalize VIX to regime level (0-1 scale).

    Interpretation:
        0.0-0.25: Low volatility regime (VIX < 15)
        0.25-0.50: Normal volatility regime (VIX 15-20)
        0.50-0.75: Elevated volatility regime (VIX 20-30)
        0.75-1.00: High volatility / Crisis regime (VIX > 30)

    Formula: clip((vix - 10) / 40, 0, 1)
    """
    return np.clip((vix_value - 10) / 40, 0.0, 1.0)
```

**Why It Helps**:
- Consolidation breakouts behave differently in low vs high volatility environments
- High VIX often indicates fear/uncertainty where breakouts may fail
- Low VIX may indicate complacency where breakouts have more room to run
- Micro-caps especially sensitive to volatility regime changes

**Data Requirements**:
- Daily VIX closing prices (^VIX or VIX index)
- Available from: Yahoo Finance (free), CBOE (official), TwelveData API
- Historical data needed: 2019-present minimum

**Look-Ahead Bias Risk**: LOW
- Use VIX value as of pattern end_date (T-0 or T-1)
- VIX is publicly available at market close

---

### Feature 2: VIX Trend (20-Day Change)

**Description**: Captures whether volatility is rising or falling, indicating regime transition.

**Calculation**:
```python
def vix_trend_20d(vix_current: float, vix_20d_ago: float) -> float:
    """
    VIX change over 20 days, normalized.

    Formula: clip((vix_current / vix_20d_ago - 1), -0.5, 0.5)

    Interpretation:
        < -0.2: Volatility collapsing (bullish for breakouts)
        -0.2 to 0.2: Stable volatility
        > 0.2: Volatility expanding (bearish for breakouts)
    """
    if vix_20d_ago <= 0:
        return 0.0
    change = (vix_current / vix_20d_ago) - 1.0
    return np.clip(change, -0.5, 0.5)
```

**Why It Helps**:
- Rising VIX during consolidation may signal impending breakdown
- Falling VIX indicates calming conditions favorable for orderly breakouts
- Trend more informative than level alone

**Data Requirements**: Same as Feature 1 (VIX data)

**Look-Ahead Bias Risk**: LOW
- Uses historical VIX values only

---

### Feature 3: Market Breadth (% Stocks Above 200 SMA)

**Description**: Measures market breadth by calculating the percentage of stocks trading above their 200-day moving average.

**Calculation**:
```python
def market_breadth_200(universe_close: pd.DataFrame) -> pd.Series:
    """
    Calculate daily market breadth.

    Args:
        universe_close: DataFrame of close prices (date x tickers)

    Returns:
        Series of daily breadth percentages (0-100)

    Formula: (count of stocks where close > sma_200) / total_stocks * 100

    Interpretation:
        < 30%: Extremely narrow market / bearish
        30-50%: Weak breadth
        50-70%: Healthy breadth
        > 70%: Strong breadth / potential overextension
    """
    sma_200 = universe_close.rolling(200).mean()
    above_sma = (universe_close > sma_200).astype(int)
    breadth = above_sma.sum(axis=1) / above_sma.count(axis=1) * 100
    return breadth
```

**Feature for Model**:
```python
def breadth_regime_normalized(breadth_pct: float) -> float:
    """Normalize breadth to 0-1 scale."""
    return np.clip(breadth_pct / 100, 0.0, 1.0)
```

**Why It Helps**:
- Narrow markets (few leaders) have different breakout dynamics than broad markets
- Micro-cap breakouts more likely to succeed when breadth is improving
- Breadth divergences (price up, breadth down) signal regime risk

**Data Requirements**:
- Daily close prices for universe of stocks (e.g., S&P 500, Russell 2000)
- Can be computed from existing data if universe is defined
- Pre-computed breadth indices available from some data providers

**Implementation Options**:
1. **Option A (Recommended)**: Pre-compute breadth from GCS ticker universe
2. **Option B**: Use external breadth indices (McClellan Breadth, MMFI)
3. **Option C**: Compute from S&P 500 constituents

**Look-Ahead Bias Risk**: MEDIUM
- Must ensure breadth is calculated using ONLY stocks that were constituents at that time
- Survivorship bias risk if using current universe composition
- Recommend using point-in-time constituent data or a stable ETF proxy (RSP equal-weight)

---

### Feature 4: Risk-On/Risk-Off Indicator

**Description**: Composite indicator measuring risk appetite using relative performance of risk assets.

**Calculation**:
```python
def risk_on_indicator(
    spy_return_20d: float,  # S&P 500 20-day return
    tlt_return_20d: float,  # Long-term treasury 20-day return
    jnk_return_20d: float,  # High-yield bond 20-day return
    gld_return_20d: float   # Gold 20-day return
) -> float:
    """
    Risk-On / Risk-Off composite indicator.

    Logic:
        Risk-On (positive score):
            - Stocks outperforming bonds (SPY > TLT)
            - High-yield outperforming (credit spreads tightening)
            - Gold underperforming (no flight to safety)

        Risk-Off (negative score):
            - Bonds outperforming stocks
            - High-yield underperforming (credit spreads widening)
            - Gold outperforming (flight to safety)

    Formula:
        score = (spy - tlt) + (jnk - tlt) - (gld - spy)
        normalized = clip(score, -0.2, 0.2) / 0.4 + 0.5  # Map to 0-1

    Interpretation:
        < 0.3: Strong risk-off (avoid breakout entries)
        0.3-0.7: Neutral
        > 0.7: Strong risk-on (favorable for breakouts)
    """
    equity_vs_bonds = spy_return_20d - tlt_return_20d
    credit_strength = jnk_return_20d - tlt_return_20d
    gold_safe_haven = gld_return_20d - spy_return_20d

    score = equity_vs_bonds + credit_strength - gold_safe_haven
    normalized = np.clip(score, -0.2, 0.2) / 0.4 + 0.5
    return normalized
```

**Why It Helps**:
- Risk appetite directly affects willingness to bid up micro-caps
- Credit spreads are leading indicator of market stress
- Gold strength signals defensive positioning
- Composite captures multiple risk dimensions

**Data Requirements**:
- Daily ETF prices: SPY, TLT, JNK/HYG, GLD
- Available from: Yahoo Finance, TwelveData, most data providers
- Low cost / free data

**Look-Ahead Bias Risk**: LOW
- All inputs are publicly available at market close
- Use T-1 values if computing morning of entry day

---

### Feature 5: Days Since Regime Change

**Description**: Time elapsed since last bull/bear transition, capturing regime maturity.

**Calculation**:
```python
def days_since_regime_change(current_date: pd.Timestamp, last_crossover_date: pd.Timestamp) -> float:
    """
    Log-normalized days since last MA crossover.

    Formula: log1p(days) / log1p(500)  # Normalized to ~[0, 1]

    Interpretation:
        < 0.4: Early in regime (< 30 days) - uncertain
        0.4-0.6: Establishing (30-100 days)
        0.6-0.8: Mature regime (100-250 days)
        > 0.8: Extended regime (> 250 days) - potential for reversal
    """
    days = (current_date - last_crossover_date).days
    return np.log1p(days) / np.log1p(500)
```

**Why It Helps**:
- Breakouts in early regime phases face more uncertainty
- Mature regimes provide more stable backdrop
- Extended regimes may be at higher risk of reversal
- Already partially available from existing `market_phase` infrastructure

**Data Requirements**:
- Uses existing `USA500IDXUSD_phases.csv` data
- Column `crossover_date` already computed
- Minimal additional implementation

**Look-Ahead Bias Risk**: LOW
- Based on historical crossover dates only

---

## Feature Integration Plan

### Phase 1: Data Acquisition

```
1. VIX Data Pipeline
   - Source: Yahoo Finance (^VIX) or TwelveData
   - Storage: data/market_regime/vix_daily.parquet
   - Update: Nightly via download_market_data.py

2. ETF Data Pipeline (SPY, TLT, JNK, GLD)
   - Source: Yahoo Finance or existing GCS bucket
   - Storage: data/market_regime/etf_daily.parquet
   - Update: Nightly

3. Breadth Data (if computing internally)
   - Source: Compute from existing GCS ticker data
   - Storage: data/market_regime/breadth_daily.parquet
   - Update: Weekly (computationally expensive)
```

### Phase 2: Feature Registry Update

Update `config/context_features.py`:

```python
# Add new regime features (indices 18-22)
CONTEXT_FEATURES: Final[List[str]] = [
    # ... existing 18 features ...

    # Regime features (Jan 2026)
    'vix_regime_level',           # Index 18: VIX normalized (0=calm, 1=crisis)
    'vix_trend_20d',              # Index 19: VIX 20d change (-0.5 to +0.5)
    'market_breadth_200',         # Index 20: % stocks > 200 SMA (0-1)
    'risk_on_indicator',          # Index 21: Risk-on/off composite (0-1)
    'days_since_regime_change',   # Index 22: Log-normalized days (0-1)
]

NUM_CONTEXT_FEATURES: Final[int] = 23  # Updated from 18
```

### Phase 3: Pipeline Integration

1. **01_generate_sequences.py**:
   - Load regime data from `data/market_regime/` directory
   - Add regime feature extraction to `extract_context_features()`
   - Ensure point-in-time correctness (use pattern_end_date for lookup)

2. **02_train_temporal.py**:
   - Update context dimension from 18 to 23
   - Add regime features to normalization ranges
   - Consider regime-stratified validation

3. **03_predict_temporal.py**:
   - Load current regime state for live predictions
   - Add regime features to inference pipeline

### Phase 4: Model Architecture Update

**Option A (Minimal Change)**:
- Simply add 5 features to existing GRN input
- Let model learn to use regime information

**Option B (Regime-Aware Architecture)**:
```python
class RegimeGatedFusion(nn.Module):
    """
    Gate fusion weights based on regime features.

    Idea: Different branches may be more/less relevant
    depending on market regime. In high-VIX, weight
    context features higher. In low-VIX, weight
    temporal patterns higher.
    """
    def __init__(self, regime_dim=5, num_branches=4):
        self.gate_network = nn.Sequential(
            nn.Linear(regime_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_branches),
            nn.Softmax(dim=-1)
        )

    def forward(self, regime_features, branch_outputs):
        gates = self.gate_network(regime_features)
        # gates shape: (batch, 4)
        # Weighted combination of branches
        return sum(g * b for g, b in zip(gates.unbind(-1), branch_outputs))
```

---

## Validation Strategy

### 1. Regime Distribution Analysis

Before training, verify regime feature distribution across splits:

```python
# Expected output:
# Split      | VIX_mean | Breadth_mean | Risk_On_mean | Days_Regime
# -----------|----------|--------------|--------------|------------
# Train      |   18.5   |    58.2%     |    0.52      |   125
# Val        |   15.2   |    62.1%     |    0.61      |   180
# Test       |   13.8   |    71.5%     |    0.68      |   210

# WARNING: If test regime is drastically different from train,
# model may still fail to generalize even with regime features.
```

### 2. Regime-Stratified Performance

Evaluate model separately by regime:

```python
# Evaluate lift by VIX regime
for regime in ['low', 'normal', 'elevated', 'high']:
    subset = test_df[test_df['vix_regime'] == regime]
    lift = compute_lift(subset)
    print(f"{regime}: Lift = {lift:.2f}x (n={len(subset)})")
```

### 3. Ablation Study

Compare performance with and without regime features:

| Configuration | Val Lift | Test Lift | Notes |
|--------------|----------|-----------|-------|
| Baseline (18 features) | 1.85x | 0.99x | Current production |
| + VIX only | TBD | TBD | Single regime indicator |
| + VIX + Breadth | TBD | TBD | Add market context |
| + All 5 Regime Features | TBD | TBD | Full specification |
| + Regime-Gated Fusion | TBD | TBD | Architecture change |

---

## Implementation Checklist

- [ ] **Data Pipeline**
  - [ ] Create `scripts/download_regime_data.py` for VIX and ETF data
  - [ ] Create `data/market_regime/` directory structure
  - [ ] Add VIX to nightly data update script
  - [ ] Add ETF prices (SPY, TLT, JNK, GLD) to data pipeline

- [ ] **Feature Engineering**
  - [ ] Create `utils/regime_features.py` with calculation functions
  - [ ] Add unit tests for each regime feature calculation
  - [ ] Verify point-in-time correctness (no look-ahead)
  - [ ] Add to feature registry in `config/context_features.py`

- [ ] **Pipeline Updates**
  - [ ] Update `01_generate_sequences.py` to extract regime features
  - [ ] Update context normalization ranges
  - [ ] Update model input dimensions (18 -> 23)
  - [ ] Add regime features to metadata for analysis

- [ ] **Validation**
  - [ ] Run regime distribution analysis across splits
  - [ ] Execute ablation study
  - [ ] Evaluate regime-stratified performance
  - [ ] Document findings

- [ ] **Documentation**
  - [ ] Update CLAUDE.md with new feature indices
  - [ ] Add regime features to feature schema table
  - [ ] Document data sources and update frequency

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Insufficient regime variance in training data** | HIGH | May need to extend training data back to 2018 to include more regimes |
| **Look-ahead bias in breadth calculation** | MEDIUM | Use point-in-time constituent lists or stable ETF proxies |
| **VIX data gaps/holidays** | LOW | Forward-fill missing values; VIX typically available |
| **Regime features still insufficient** | MEDIUM | Consider additional features: Fed policy, sector rotation, yield curve |
| **Model complexity increase** | LOW | Only 5 additional features; minimal computational impact |

---

## Alternative / Future Features to Consider

If the proposed 5 features prove insufficient, consider these additional regime indicators:

1. **Yield Curve Slope** (10Y - 2Y Treasury): Leading recession indicator
2. **Sector Rotation Score**: Cyclicals vs Defensives relative performance
3. **Put/Call Ratio**: Sentiment indicator from options market
4. **McClellan Oscillator**: Technical breadth momentum
5. **Fed Funds Rate Regime**: Tightening vs easing cycle
6. **Dollar Strength Index**: Currency regime affects small-cap performance
7. **Junk Bond Spread** (HY OAS): Credit market stress indicator

---

## Summary

This specification proposes adding 5 regime-aware features to address the cross-regime generalization failure:

| # | Feature | Data Source | Look-Ahead Risk |
|---|---------|-------------|-----------------|
| 18 | VIX Regime Level | Yahoo/CBOE | LOW |
| 19 | VIX Trend 20D | Yahoo/CBOE | LOW |
| 20 | Market Breadth 200 | Computed/External | MEDIUM |
| 21 | Risk-On Indicator | ETF prices | LOW |
| 22 | Days Since Regime Change | Existing phases.csv | LOW |

The features are designed to be:
- **Temporally safe**: No look-ahead bias when properly implemented
- **Data-efficient**: Available from free/low-cost sources
- **Interpretable**: Clear relationship to market regimes
- **Complementary**: Each captures a different aspect of regime

Implementation should proceed in phases, starting with data acquisition, then feature integration, and finally model architecture updates if needed.
