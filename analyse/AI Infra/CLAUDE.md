
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

never present fabricated data

## Project Overview

This repository contains multiple trading systems spanning different asset classes, timeframes, and methodologies. Each subsystem has its own `CLAUDE.md` with detailed specifications.

### System Map

| System | Location | Asset Class | Timeframe | Method |
|--------|----------|-------------|-----------|--------|
| **AIv3** | Root (`core/`, `ml/`) | Micro/small-cap equities | Daily | State machine + XGBoost |
| **TRAnS** | `vonLinck Capital/trans/` | EU equities | Daily | CNN + GRN temporal model |
| **geldbeutel V1.0** | `vonLinck Capital/geldbeutel/V1.0/` | MNQ/NQ futures | 5-minute | Liquidity sweep kernel + GA/WFO |
| **Monster Hunter** | `vonLinck Capital/geldbeutel/Test/monster_hunter/` | Micro-cap equities | Daily | Volume pivot + XGBoost (research) |

---

## geldbeutel V1.0 — Discrete Liquidity Sweep Engine

**See `vonLinck Capital/geldbeutel/V1.0/CLAUDE.md` for full specification.**

Implements the Thermodynamic Theory of Discrete Liquidity Sweeps (TTDLS). A Numba-JIT state machine detects pristine liquidity pool breaches on MNQ 5-minute data, executes asymmetric timeframe arbitrage (macro-level detection, micro-level risk), and adapts via Mixed-Integer GA inside a Rolling Walk-Forward Optimization matrix.

```bash
# Quick start (from geldbeutel/V1.0/)
python dukascopy_downloader.py   # Download 5-min NQ CFD data (2017-2026)
python wfo_matrix.py             # Run GA + Rolling WFO
```

Key files: `wfo_matrix.py` (sealed kernel + GA + WFO), `dukascopy_downloader.py` (data pipeline).

---

## AIv3 — Consolidation Pattern Detector

Advanced consolidation pattern detection system designed to identify micro/small-cap stocks poised for significant upward moves (40%+ gains). The system uses a sophisticated state machine, XGBoost machine learning, and strategic value assignment to predict breakout patterns from consolidation phases.

- **Strategic Focus**: Identifying consolidation patterns with 75%+ breakout potential

See `ATTENTION_README.md` for details on the experimental attention system.

## Architecture Components

### 1. Pattern Detection State Machine (`core/`)
- **ConsolidationTracker**: Tracks individual ticker through qualification → active → completed phases
- **StatefulDetector**: Manages multiple ticker trackers with temporal consistency
- **Qualification Phase (Days 1-10)**:
  - BBW < 30th percentile (volatility contraction)
  - ADX < 32 (low trending)
  - Volume < 35% of 20-day average
  - Daily Range < 65% of 20-day average
- **Active Phase (Day 10+)**: Pattern monitored until breakout or failure

### 2. Machine Learning Pipeline (`ml/`)
- **StatefulPatternLabeler**: Path-dependent labeling with day-by-day state tracking
- **XGBoostClassifier**: Multi-class prediction (K0-K5) with hyperparameter optimization
- **ExpectedValuePredictor**: Calculates EV from class probabilities
- **PatternValueSystem**: Strategic value assignments for each outcome class


## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run pattern scanning (main entry point)
python main.py scan --tickers ALL --start-date 2024-01-01

# Run 4-year backtest (recommended minimum)
python run_4year_backtest.py

# Run comprehensive test suite
python run_tests.py
python run_tests.py --type unit
python run_tests.py --type integration

# Pattern detection testing
python test_pattern_detection.py

# Feature extraction testing
python test_feature_extraction.py

# Hyperparameter optimization (when retraining)
python optimize/optuna_search.py --trials 50 --model xgboost
```

## Data Pipeline

### GCS Data Storage
```bash
export PROJECT_ID="ignition-ki-csv-storage"
export GCS_BUCKET_NAME="ignition-ki-csv-data-2025-user123"
```

### Input Format
- **Primary Storage**: Google Cloud Storage (GCS) buckets
- **Local Cache**: `data/raw/` directory
- **File Format**: CSV or Parquet files
- **Required Columns**: date, open, high, low, close, volume, ticker
- **Frequency**: Daily OHLCV data

### Loading Data
```python
from utils.data_loader import load_ticker_data

# Load from GCS
df = load_ticker_data('AAPL', start_date='2015-01-01')

# Load multiple tickers
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = {ticker: load_ticker_data(ticker) for ticker in tickers}
```

## Critical Implementation Details

### Temporal Integrity (CRITICAL)
- **NO look-ahead bias**: State machine ensures strict temporal ordering
- **Walk-forward validation**: Train only on past data, predict on future
- **Pattern labeling**: Only applied after 100-day outcome period completes
- **Real-time predictions**: Use only current and historical state
- **4-year minimum**: Backtest period for meaningful pattern statistics
- **90-day retraining**: Model updates with quarterly cadence

### Feature Engineering Focus
The system extracts 30+ technical features during pattern phases:
- **BBW (Bollinger Band Width)**: Volatility contraction measurement
- **ADX (Average Directional Index)**: Trend strength indicator
- **Volume characteristics**: Relative volume, volume spikes, accumulation
- **Price position**: Distance from boundaries, range compression
- **Temporal features**: BBW slope, ADX trend, pattern duration
- **Pattern metrics**: Days in qualification, days active, boundary violations




## Testing & Validation

```bash
# Run comprehensive test suite
python run_tests.py

# Run specific test categories
python run_tests.py --type unit
python run_tests.py --type integration

# Test pattern detection logic
python test_pattern_detection.py

# Test feature extraction
python test_feature_extraction.py

# Run 4-year backtest validation
python run_4year_backtest.py

# Validate temporal integrity (no look-ahead bias)
python backtesting/temporal_backtester.py --validate-temporal
```

## Model Retraining Protocol

1. **Collect new data**: Minimum 90 days of additional market data
2. **Update pattern labels**: Apply StatefulPatternLabeler to historical patterns
3. **Feature engineering**: Extract features from completed patterns
4. **Validate temporal integrity**: Ensure no look-ahead bias
5. **Hyperparameter optimization**: Optuna search with 50-100 trials
6. **Cross-validation**: Stratified time-series splits
7. **Performance comparison**: Must exceed baseline by >5%
8. **Deploy**: Only if validation metrics improve

## Production Deployment Checklist

- [ ] Verify GCS credentials configured (`PROJECT_ID`, `GCS_BUCKET_NAME`)
- [ ] Test data loading: `python utils/data_loader.py --test`
- [ ] Run pattern detection tests: `python test_pattern_detection.py`
- [ ] Run 4-year backtest: `python run_4year_backtest.py`
- [ ] Validate EV correlation > 0.3
- [ ] Confirm signal thresholds in config (EV ≥ 5.0 for strong signals)
- [ ] Set up monitoring for:
  - Pattern detection rate (should be 2-5% strong signals)
  - Model prediction drift
  - Class distribution changes
  - K5 (failure) prediction accuracy

## Expected Value (EV) Calculation

```python
EV = Σ(probability_i × value_i)

Where:
- probability_i = model's predicted probability for class K_i
- value_i = strategic value for class K_i

Example:
P(K4) = 0.15 → EV contribution = 0.15 × 10 = +1.5
P(K3) = 0.25 → EV contribution = 0.25 × 3  = +0.75
P(K2) = 0.30 → EV contribution = 0.30 × 1  = +0.3
P(K1) = 0.15 → EV contribution = 0.15 × -0.2 = -0.03
P(K0) = 0.10 → EV contribution = 0.10 × -2 = -0.2
P(K5) = 0.05 → EV contribution = 0.05 × -10 = -0.5
Total EV = +1.82 (MODERATE_SIGNAL)
```

## Key Performance Metrics to Monitor

1. **EV Correlation**: Correlation between predicted and actual values (target >0.3)
2. **Class Accuracy**: Precision/recall for each K0-K5 class
3. **Signal Quality Rate**: % of patterns generating high-confidence signals
4. **Value Capture**: Total strategic value realized from signals
5. **Failure Prediction**: Accuracy in identifying K5 (breakdown) patterns

## Notes & Best Practices

- System optimized for **micro/small-cap stocks** with high volatility potential
- Not a complete trading system - pattern detection and signal generation only
- Requires **substantial historical data** (4+ years) for meaningful results
- Performance varies with market conditions - monitor for regime changes
- Focus on patterns with **failure probability < 20%** for best risk/reward
- Diversify across multiple patterns to reduce single-pattern risk