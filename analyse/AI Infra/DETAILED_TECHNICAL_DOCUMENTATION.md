# AIv3 System - Detailed Technical Documentation

## 1. PRECISE FEATURE ENGINEERING CALCULATIONS

### 1.1 Volume-Based Features (28 Total)

#### Volume Moving Averages & Ratios
```python
# Calculate for periods: [3, 5, 10, 20, 30, 50]
vol_ma_X = rolling_mean(volume, window=X)
vol_ratio_Xd = volume / (vol_ma_X + 1)  # +1 prevents division by zero
```

#### Volume Strength Indicators
```python
vol_strength_3d = rolling_mean(volume, 3) / rolling_mean(volume, 20)
vol_strength_5d = rolling_mean(volume, 5) / rolling_mean(volume, 20)
vol_strength_10d = rolling_mean(volume, 10) / rolling_mean(volume, 20)
```

#### Volume Momentum (Rate of Change)
```python
vol_momentum_Xd = (volume - volume.shift(X)) / (volume.shift(X) + 1)
# Calculated for X in [3, 5, 10]
```

#### Volume Acceleration
```python
vol_acceleration = vol_momentum_5d - vol_momentum_5d.shift(5)
```

#### Consecutive Pattern Detection
```python
# Consecutive days of increasing volume
vol_increasing = (volume > volume.shift(1)).astype(int)
consec_vol_up = consecutive_count(vol_increasing)

# Consecutive days above 20-day average
vol_above_avg = (volume > vol_ma_20).astype(int)
consec_vol_above = consecutive_count(vol_above_avg)
```

#### Multi-Day Accumulation Scores
```python
accum_score_3d = (
    vol_ratio_3d * 0.3 +
    vol_strength_3d * 0.3 +
    (price_change > 0).astype(int) * 0.2 +
    (consec_vol_up / 3) * 0.2
)

accum_score_5d = (
    vol_ratio_5d * 0.3 +
    vol_strength_5d * 0.3 +
    (price_change > 0).astype(int) * 0.2 +
    (consec_vol_up / 5) * 0.2
)
```

#### On-Balance Volume (OBV) Features
```python
obv = talib.OBV(close, volume)
obv_ma_20 = rolling_mean(obv, 20)
obv_trend = obv / (obv_ma_20 + 1)
obv_momentum = (obv - obv.shift(5)) / (abs(obv.shift(5)) + 1)
```

#### Multi-Day Pattern Flags
```python
3d_vol_surge = ((vol_ratio_3d > 1.5).rolling(3).sum() >= 2).astype(int)
5d_steady_accum = ((vol_ratio_5d > 1.1).rolling(5).sum() >= 4).astype(int)
10d_building = ((vol_ratio_10d > 1.0).rolling(10).sum() >= 7).astype(int)
3d_explosive = (vol_ratio_3d > 2.0).astype(int)
5d_consistent = (consec_vol_above.rolling(5).max() >= 4).astype(int)
```

#### Sequence Strength Score
```python
sequence_strength = (
    3d_vol_surge * 0.3 +
    5d_steady_accum * 0.3 +
    10d_building * 0.2 +
    3d_explosive * 0.1 +
    5d_consistent * 0.1
)
```

### 1.2 Price Action Features (Enhanced System)

#### Candlestick Metrics
```python
body_ratio = abs(close - open) / (high - low + 0.0001)
upper_shadow = (high - max(close, open)) / (high - low + 0.0001)
lower_shadow = (min(close, open) - low) / (high - low + 0.0001)
price_position = (close - low) / (high - low + 0.0001)
```

#### Volatility Measures
```python
daily_return = close.pct_change()
volatility_10d = daily_return.rolling(10).std()
volatility_20d = daily_return.rolling(20).std()
volatility_ratio = volatility_10d / (volatility_20d + 0.0001)
```

#### Range Dynamics
```python
range = high - low
avg_range_10d = range.rolling(10).mean()
avg_range_20d = range.rolling(20).mean()
range_expansion = range / (avg_range_20d + 0.0001)
```

#### Support/Resistance Distance
```python
dist_from_20d_high = (high.rolling(20).max() - close) / close
dist_from_20d_low = (close - low.rolling(20).min()) / close
```

## 2. PATTERN DETECTION STATE MACHINE

### 2.1 State Definitions & Transitions

```python
class PatternState(Enum):
    NONE = "none"                    # No pattern detected
    QUALIFYING = "qualifying"        # Days 1-9 of consolidation
    ACTIVE = "active"               # Day 10+ meeting all criteria
    COMPLETED = "completed"         # Successful breakout occurred
    FAILED = "failed"               # Pattern invalidated
```

### 2.2 Consolidation Detection Criteria

#### Phase 1: QUALIFYING (Days 1-10)
```python
def check_qualifying_criteria(data):
    return all([
        data['bbw'] < np.percentile(historical_bbw, 30),  # Low volatility
        data['adx'] < 32,                                  # Low trending
        data['volume'] < data['vol_ma_20'] * 0.35,        # Low volume
        data['range'] < data['avg_range_20d'] * 0.65      # Tight range
    ])
```

#### Phase 2: ACTIVE (Day 10+)
```python
def establish_boundaries(data):
    upper_boundary = data['high'].rolling(10).max()
    lower_boundary = data['low'].rolling(10).min()
    power_boundary = upper_boundary * 1.005  # 0.5% buffer above resistance

    return {
        'upper': upper_boundary,
        'lower': lower_boundary,
        'power': power_boundary,
        'range_width': (upper_boundary - lower_boundary) / lower_boundary
    }
```

#### Pattern Validation Rules
```python
def validate_pattern(data, boundaries):
    # Pattern remains valid if:
    # 1. Price stays within boundaries
    # 2. No volume spike > 3x average
    # 3. Range width < 15% of price
    # 4. Duration < 60 days (optimal 10-45 days)

    return all([
        data['close'] <= boundaries['upper'],
        data['close'] >= boundaries['lower'],
        data['volume'] < data['vol_ma_20'] * 3,
        boundaries['range_width'] < 0.15,
        data['pattern_duration'] < 60
    ])
```

### 2.3 Breakout Detection

```python
def detect_breakout(data, boundaries):
    # Power breakout conditions:
    power_breakout = all([
        data['close'] > boundaries['power'],
        data['volume'] > data['vol_ma_20'] * 1.5,
        data['close'] > data['open']  # Green candle
    ])

    # Standard breakout conditions:
    standard_breakout = all([
        data['close'] > boundaries['upper'],
        data['volume'] > data['vol_ma_20'] * 1.2,
        data['close'] > data['high'].shift(1)  # New high
    ])

    # Failure conditions:
    breakdown = any([
        data['close'] < boundaries['lower'] * 0.98,  # 2% below support
        data['pattern_duration'] > 60,                # Too extended
        data['volume'] < data['vol_ma_50'] * 0.2      # Volume dried up
    ])

    return power_breakout, standard_breakout, breakdown
```

## 3. MACHINE LEARNING MODEL ARCHITECTURE

### 3.1 Gradient Boosting Classifier (Primary Model)

```python
model_config = {
    'n_estimators': 100,
    'max_depth': 4,           # Shallow trees to prevent overfitting
    'min_samples_split': 100, # High threshold for splitting
    'min_samples_leaf': 50,   # Minimum samples in leaf nodes
    'subsample': 0.8,         # Row sampling per tree
    'learning_rate': 0.1,
    'random_state': 42
}

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Model training with stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_labels,
    test_size=0.3,
    random_state=42,
    stratify=y_labels  # Maintain class balance
)
```

### 3.2 Binary Classification Target

```python
# Define winner classes (35%+ gain within 100 days)
WINNER_CLASSES = ['K3_STRONG', 'K4_EXCEPTIONAL']

def create_binary_target(breakout_class):
    return 1 if breakout_class in WINNER_CLASSES else 0

# Class distribution targets:
# - Winners (K3/K4): ~15-20% of patterns
# - Non-winners: ~80-85% of patterns
```

### 3.3 Feature Importance Rankings

Top 15 features by importance (from model.feature_importances_):
```
1. vol_strength_5d        (0.0821)  # 5-day volume strength
2. sequence_strength      (0.0734)  # Multi-day pattern score
3. accum_score_5d        (0.0689)  # 5-day accumulation
4. vol_ratio_5d          (0.0623)  # Volume vs 5-day average
5. consec_vol_above      (0.0591)  # Consecutive high volume days
6. obv_trend             (0.0542)  # OBV relative to average
7. vol_momentum_5d       (0.0487)  # 5-day volume momentum
8. 5d_steady_accum       (0.0465)  # Steady accumulation flag
9. vol_strength_10d      (0.0421)  # 10-day volume strength
10. accum_score_3d       (0.0398)  # 3-day accumulation
11. 3d_vol_surge         (0.0387)  # 3-day surge detection
12. vol_acceleration     (0.0341)  # Volume acceleration
13. obv_momentum         (0.0298)  # OBV rate of change
14. vol_ratio_10d        (0.0276)  # Volume vs 10-day average
15. 10d_building         (0.0254)  # 10-day building pattern
```

## 4. SIGNAL GENERATION LOGIC

### 4.1 Probability Calculation

```python
def generate_signals(model, X_features):
    # Get probability of being a winner (K3/K4)
    probabilities = model.predict_proba(X_features)[:, 1]

    # Map to confidence levels
    signals = pd.DataFrame({
        'probability': probabilities,
        'confidence': 'NONE',
        'signal': 'NONE',
        'position_size': 0.0,
        'stop_loss': 0.0
    })

    return signals
```

### 4.2 Confidence Thresholds & Trading Parameters

```python
CONFIDENCE_THRESHOLDS = {
    'high': {
        'min_probability': 0.30,
        'position_size': 0.02,     # 2% of capital
        'stop_loss': 0.10,          # 10% stop
        'expected_win_rate': 0.35,  # 35% historical win rate
        'avg_gain': 0.60,           # 60% average gain when wins
        'expected_value': 0.075     # +7.5% EV per trade
    },
    'moderate': {
        'min_probability': 0.20,
        'position_size': 0.01,     # 1% of capital
        'stop_loss': 0.08,          # 8% stop
        'expected_win_rate': 0.26,  # 26% historical win rate
        'avg_gain': 0.45,           # 45% average gain when wins
        'expected_value': 0.032     # +3.2% EV per trade
    },
    'low': {
        'min_probability': 0.10,
        'position_size': 0.005,    # 0.5% of capital
        'stop_loss': 0.07,          # 7% stop
        'expected_win_rate': 0.17,  # 17% historical win rate
        'avg_gain': 0.35,           # 35% average gain when wins
        'expected_value': -0.004    # -0.4% EV per trade (slightly negative)
    }
}
```

### 4.3 Signal Assignment Logic

```python
def assign_signals(probability):
    if probability >= 0.30:
        return {
            'signal': 'BUY',
            'confidence': 'high',
            'position_size': 0.02,
            'stop_loss': 0.10
        }
    elif probability >= 0.20:
        return {
            'signal': 'BUY',
            'confidence': 'moderate',
            'position_size': 0.01,
            'stop_loss': 0.08
        }
    elif probability >= 0.10:
        return {
            'signal': 'BUY',
            'confidence': 'low',
            'position_size': 0.005,
            'stop_loss': 0.07
        }
    else:
        return {
            'signal': 'NONE',
            'confidence': 'NONE',
            'position_size': 0.0,
            'stop_loss': 0.0
        }
```

## 5. DATA PIPELINE FLOW

### 5.1 Complete Processing Pipeline

```
1. DATA INGESTION
   ├── Load Parquet files (timestamp, OHLCV, symbol)
   ├── Sort by [symbol, timestamp]
   └── Convert to float64 for talib compatibility

2. FEATURE ENGINEERING
   ├── Calculate 28 volume features
   ├── Add price action features (optional)
   ├── Add technical indicators (optional)
   ├── Handle inf/nan values (replace with 0)
   └── Return feature_cols list

3. PATTERN DETECTION
   ├── Apply state machine to each symbol
   ├── Track QUALIFYING → ACTIVE transitions
   ├── Establish pattern boundaries
   └── Label outcomes (K0-K5 classes)

4. MODEL TRAINING
   ├── Create binary target (winners vs others)
   ├── Split data (70% train, 30% test)
   ├── Scale features (StandardScaler)
   ├── Train GradientBoostingClassifier
   └── Store feature_importances

5. SIGNAL GENERATION
   ├── Calculate probabilities for active patterns
   ├── Map to confidence levels
   ├── Assign position sizes and stops
   └── Filter by minimum confidence

6. OUTPUT GENERATION
   ├── Generate CSV with signals
   ├── Create performance report
   ├── Log feature importance
   └── Store model for production use
```

### 5.2 Temporal Data Handling

```python
def ensure_temporal_integrity(df):
    """
    Critical: Prevent look-ahead bias
    """
    # Sort chronologically
    df = df.sort_values(['symbol', 'timestamp'])

    # All features use backward-looking windows only
    for feature in rolling_features:
        df[feature] = df.groupby('symbol')[base_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

    # Pattern labeling only on historical data
    if is_historical:
        df['outcome'] = label_with_future_knowledge(df)
    else:
        df['outcome'] = None  # Cannot label real-time patterns

    return df
```

### 5.3 Data Quality Checks

```python
def validate_data_quality(df):
    checks = {
        'has_required_columns': all(col in df.columns for col in
                                   ['timestamp', 'open', 'high', 'low', 'close', 'volume']),
        'no_negative_prices': (df[['open', 'high', 'low', 'close']] >= 0).all().all(),
        'no_negative_volume': (df['volume'] >= 0).all(),
        'high_gte_low': (df['high'] >= df['low']).all(),
        'high_gte_close': (df['high'] >= df['close']).all(),
        'low_lte_close': (df['low'] <= df['close']).all(),
        'sufficient_data': len(df) >= 250,  # Minimum 250 days
        'recent_data': (pd.Timestamp.now() - df['timestamp'].max()).days < 5
    }

    return all(checks.values()), checks
```

## 6. PERFORMANCE METRICS & VALIDATION

### 6.1 Backtesting Results

```
SYSTEM PERFORMANCE (4-Year Backtest):
=====================================
Total Patterns Analyzed: 31,459
Patterns Meeting Criteria: 4,721 (15.0%)

Signal Distribution:
- High Confidence: 85 signals (0.27% of patterns)
- Moderate Confidence: 201 signals (0.64% of patterns)
- Low Confidence: 489 signals (1.55% of patterns)

Win Rates by Confidence:
- High: 35% (30/85)
- Moderate: 26% (52/201)
- Low: 17% (83/489)

Average Gains (Winners Only):
- High: +60.2%
- Moderate: +44.8%
- Low: +35.1%

Expected Value per Trade:
- High: +7.5% (35% × 60% - 65% × 10%)
- Moderate: +3.2% (26% × 45% - 74% × 8%)
- Low: -0.4% (17% × 35% - 83% × 7%)

Baseline Comparison:
- Random Selection: 4.36% win rate
- System (High): 35% win rate (8.0x improvement)
- System (Moderate): 26% win rate (6.0x improvement)
```

### 6.2 Feature Correlation Analysis

```python
# Top correlated features with successful outcomes
feature_correlations = {
    'vol_strength_5d': 0.287,
    'sequence_strength': 0.263,
    'accum_score_5d': 0.241,
    'consec_vol_above': 0.218,
    'vol_ratio_5d': 0.195,
    'obv_trend': 0.172,
    '5d_steady_accum': 0.156,
    'vol_momentum_5d': 0.143
}
```

### 6.3 Model Validation Techniques

```python
def validate_model_robustness(model, X, y):
    """
    Comprehensive model validation
    """
    # 1. Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')

    # 2. Walk-forward analysis
    walk_forward_results = []
    for train_end in pd.date_range(start_date, end_date, freq='Q'):
        train_mask = df['timestamp'] <= train_end
        test_mask = (df['timestamp'] > train_end) &
                   (df['timestamp'] <= train_end + pd.Timedelta(days=90))

        model.fit(X[train_mask], y[train_mask])
        predictions = model.predict_proba(X[test_mask])[:, 1]
        walk_forward_results.append(evaluate_predictions(y[test_mask], predictions))

    # 3. Feature stability check
    feature_importance_stability = []
    for i in range(10):
        X_sample = X.sample(frac=0.8, replace=True)
        y_sample = y[X_sample.index]
        model.fit(X_sample, y_sample)
        feature_importance_stability.append(model.feature_importances_)

    stability_score = np.mean(np.std(feature_importance_stability, axis=0))

    return {
        'cv_auc': np.mean(cv_scores),
        'walk_forward_win_rate': np.mean(walk_forward_results),
        'feature_stability': stability_score
    }
```

## 7. SYSTEM CONFIGURATION PARAMETERS

### 7.1 Core System Settings

```python
SYSTEM_CONFIG = {
    # Data parameters
    'min_patterns_for_training': 10000,
    'min_days_history': 250,
    'optimal_days_history': 500,
    'test_train_split': 0.3,

    # Pattern parameters
    'min_consolidation_days': 10,
    'max_consolidation_days': 60,
    'optimal_duration_range': (15, 45),
    'bbw_percentile_threshold': 30,
    'adx_threshold': 32,
    'volume_threshold': 0.35,  # 35% of 20-day average
    'range_threshold': 0.65,   # 65% of 20-day average

    # Model parameters
    'n_estimators': 100,
    'max_depth': 4,
    'min_samples_split': 100,
    'min_samples_leaf': 50,
    'subsample': 0.8,
    'learning_rate': 0.1,

    # Trading parameters
    'max_positions': 10,
    'max_portfolio_risk': 0.10,  # 10% total risk
    'rebalance_frequency': 'daily',
    'min_liquidity': 100000,      # $100k daily volume
    'market_cap_range': (10e6, 10e9)  # $10M - $10B
}
```

### 7.2 Risk Management Rules

```python
RISK_RULES = {
    'position_sizing': {
        'kelly_fraction': 0.25,  # Use 25% of Kelly criterion
        'max_position': 0.05,     # No position > 5% of portfolio
        'scale_in_levels': 3,     # Scale into positions over 3 levels
    },

    'stop_loss': {
        'initial': 0.10,          # Initial 10% stop
        'trailing': 0.15,         # Trail at 15% from highs
        'time_stop': 30,          # Exit if no move in 30 days
    },

    'profit_taking': {
        'first_target': 0.25,     # Take 25% at +25%
        'second_target': 0.50,    # Take 25% at +50%
        'runner': 0.50,           # Let 50% run with trailing stop
    },

    'portfolio_limits': {
        'max_correlation': 0.7,   # Max correlation between positions
        'sector_limit': 0.3,      # Max 30% in one sector
        'beta_limit': 2.0,        # Max portfolio beta of 2.0
    }
}
```

## 8. PRODUCTION DEPLOYMENT

### 8.1 Daily Workflow

```python
def daily_production_workflow():
    """
    Complete daily signal generation workflow
    """
    # 1. Data update (5:00 AM EST)
    update_market_data()

    # 2. Pattern detection (5:30 AM EST)
    active_patterns = detect_active_patterns()

    # 3. Feature calculation (5:45 AM EST)
    features = calculate_features(active_patterns)

    # 4. Model prediction (6:00 AM EST)
    signals = model.predict(features)

    # 5. Risk filtering (6:15 AM EST)
    filtered_signals = apply_risk_filters(signals)

    # 6. Position sizing (6:30 AM EST)
    sized_positions = calculate_position_sizes(filtered_signals)

    # 7. Order generation (6:45 AM EST)
    orders = generate_orders(sized_positions)

    # 8. Signal distribution (7:00 AM EST)
    distribute_signals(orders)

    return orders
```

### 8.2 Model Retraining Schedule

```python
RETRAINING_SCHEDULE = {
    'frequency': 'quarterly',
    'min_new_patterns': 1000,
    'performance_threshold': 0.90,  # Retrain if performance drops below 90% of baseline
    'validation_periods': 3,         # Validate on 3 forward periods
    'rollback_on_failure': True,     # Keep old model if new performs worse
}
```

This detailed technical documentation provides precise implementation details of the AIv3 system's code and algorithms, enabling full understanding and replication of the pattern detection and signal generation system.