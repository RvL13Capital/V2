# Walk-Forward Validation & Extended Performance Metrics

## 📋 Übersicht

Dieses Modul implementiert eine rigorose **Walk-Forward Validierung** mit **erweiterten Performance-Metriken** für das AIv3 Pattern Detection System.

### ✨ Hauptmerkmale

1. **Walk-Forward Validation**
   - Zeitbasierte Datensplits (kein Look-Ahead Bias)
   - Quarterly Retraining (alle 90 Tage)
   - 100-Tage Evaluation Windows
   - Minimum 4 Jahre Backtest-Periode

2. **Extended Performance Metrics**
   - **Profit Factor**: Gross Profit / Gross Loss
   - **Sharpe Ratio**: Risikoadjustierte Rendite
   - **Sortino Ratio**: Downside-fokussierte Sharpe-Variante
   - **Maximum Drawdown**: Größter Peak-to-Trough Verlust
   - **Calmar Ratio**: Return / Max Drawdown
   - **Value at Risk (VaR)**: Potenzielle Verluste bei 95% Konfidenz
   - **Conditional VaR (CVaR)**: Expected Shortfall
   - **Recovery Factor**: Total Return / Max Drawdown

3. **Comprehensive Reporting**
   - Publication-ready Berichte
   - Per-Window Performance-Tracking
   - Strategy Quality Assessment (0-10 Score)
   - Actionable Recommendations

---

## 🚀 Quick Start

### Installation

Alle benötigten Dependencies sind bereits in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Verwendung

#### Option 1: Comprehensive Backtester (Empfohlen)

```python
from comprehensive_backtester import ComprehensiveBacktester
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Load your pattern data
df = pd.read_parquet('data/patterns.parquet')

# Initialize backtester
backtester = ComprehensiveBacktester(
    initial_train_years=2.0,      # 2 Jahre initiales Training
    retrain_frequency_days=90,    # Retraining alle 90 Tage
    test_window_days=100,         # 100-Tage Test Windows
    min_train_samples=500,        # Min. 500 Patterns
    risk_free_rate=0.02           # 2% risikofreier Zinssatz
)

# Define features
feature_columns = ['vol_strength_5d', 'bbw', 'adx', ...]  # Your features

# Run backtest
results = backtester.run_comprehensive_backtest(
    df=df,
    model_class=GradientBoostingClassifier,
    feature_columns=feature_columns,
    target_column='outcome_class',
    date_column='start_date',
    model_params={'n_estimators': 100, 'max_depth': 5}
)

# Generate report
report = backtester.generate_publication_report()
print(report)

# Save results
backtester.save_results(output_dir='output/backtest')
```

#### Option 2: Standalone Walk-Forward Validator

```python
from walk_forward_validator import WalkForwardValidator

validator = WalkForwardValidator(
    initial_train_days=730,
    retrain_frequency_days=90,
    test_window_days=100
)

# Run validation
performances = validator.validate_model(
    df=df,
    model_class=GradientBoostingClassifier,
    feature_columns=feature_columns
)

# Generate report
report = validator.generate_comprehensive_report()
validator.save_results()
```

#### Option 3: Standalone Extended Metrics

```python
from extended_performance_metrics import PerformanceMetricsCalculator

calculator = PerformanceMetricsCalculator(risk_free_rate=0.02)

# Calculate all metrics
metrics = calculator.calculate_all_metrics(
    predictions=df,
    outcome_column='outcome_class',
    gain_column='max_gain'
)

# Generate report
report = calculator.generate_metrics_report(metrics)
print(report)
```

---

## 📊 Output Files

Nach dem Backtest werden folgende Dateien erstellt:

```
output/comprehensive_backtest/
├── backtest_report_YYYYMMDD_HHMMSS.txt      # Hauptbericht (Publication-ready)
├── backtest_results_YYYYMMDD_HHMMSS.json    # Vollständige Ergebnisse als JSON
├── performance_summary_YYYYMMDD_HHMMSS.csv  # Performance-Metriken als CSV
└── walk_forward_details_YYYYMMDD_HHMMSS.csv # Per-Window Details als CSV
```

---

## 📈 Key Performance Metrics Explained

### Trading Metrics

| Metrik | Beschreibung | Interpretation |
|--------|--------------|----------------|
| **Win Rate** | Gewinnende Trades / Gesamt Trades | >25% = Gut für AIv3 System |
| **Profit Factor** | Gross Profit / Gross Loss | >1.5 = Profitabel, >2.0 = Sehr gut |
| **Expectancy** | Erwarteter Gewinn pro Trade | >0 = Profitabel |

### Risk-Adjusted Metrics

| Metrik | Beschreibung | Interpretation |
|--------|--------------|----------------|
| **Sharpe Ratio** | (Return - Risk-Free Rate) / Std Dev | >1.0 = Gut, >1.5 = Sehr gut |
| **Sortino Ratio** | Wie Sharpe, aber nur Downside Risk | >1.5 = Gut |
| **Calmar Ratio** | Annual Return / Max Drawdown | >3.0 = Gut |

### Risk Metrics

| Metrik | Beschreibung | Interpretation |
|--------|--------------|----------------|
| **Max Drawdown** | Größter Peak-to-Trough Verlust | <-20% = Akzeptabel, <-15% = Gut |
| **VaR (95%)** | Max. Verlust bei 95% Konfidenz | Risiko-Management Schwelle |
| **CVaR (95%)** | Durchschnittlicher Tail-Loss | Expected Shortfall |

---

## 🎯 AIv3-Spezifische Metriken

### Pattern Outcome Classes

```
K4 (Exceptional): >75% Gewinn  → Strategischer Wert: +10
K3 (Strong):      35-75% Gewinn → Strategischer Wert: +3
K2 (Quality):     15-35% Gewinn → Strategischer Wert: +1
K1 (Minimal):     5-15% Gewinn  → Strategischer Wert: -0.2
K0 (Stagnant):    <5% Gewinn    → Strategischer Wert: -2
K5 (Failed):      Breakdown     → Strategischer Wert: -10
```

### Expected Value (EV) per Pattern

```
EV = Σ(Probability_i × Strategic_Value_i)

Interpretation:
- EV > +3.0: STRONG_SIGNAL
- EV > +1.0: GOOD_SIGNAL
- EV > 0:    MODERATE_SIGNAL
- EV < 0:    AVOID
```

---

## 🔧 Konfiguration

### Walk-Forward Parameter

```python
backtester = ComprehensiveBacktester(
    initial_train_years=2.0,      # 2-4 Jahre empfohlen
    retrain_frequency_days=90,    # 60-120 Tage (quarterly/monthly)
    test_window_days=100,         # 60-120 Tage
    min_train_samples=500,        # 300-1000 je nach Datenmenge
    risk_free_rate=0.02           # Aktuelle 10Y Treasury Rate
)
```

### Model Parameter Example

```python
model_params = {
    'n_estimators': 100,          # 50-200
    'max_depth': 5,               # 3-8 (prevent overfitting)
    'learning_rate': 0.1,         # 0.01-0.3
    'min_samples_split': 10,      # 5-20
    'min_samples_leaf': 5,        # 3-10
    'random_state': 42
}
```

---

## 📝 Beispiel Backtest-Run

```bash
# Mit synthetic data (Test)
python run_comprehensive_backtest.py

# Mit echten Daten
python run_comprehensive_backtest.py --use-real-data --data-path ./data/patterns/
```

### Erwartete Output

```
====================================================================================================
COMPREHENSIVE BACKTEST - AIv3 PATTERN DETECTION SYSTEM
====================================================================================================

EXECUTIVE SUMMARY
Testing Period: 2020-01-01 to 2024-12-31 (1824 days)
Total Patterns Analyzed: 2,000
Walk-Forward Windows: 12
Retraining Frequency: Every 90 days

Key Performance Indicators:
  • Overall Win Rate: 25.7%
  • Sharpe Ratio: 1.45
  • Profit Factor: 2.18
  • Maximum Drawdown: -18.3%
  • Total Return: +127.4%
  • Expected Value per Pattern: +0.47

PATTERN OUTCOME DISTRIBUTION
K4 Exceptional Rate (>75% gains): 7.5%
K3 Strong Rate (35-75% gains): 18.2%
Combined K3+K4 Success Rate: 25.7%
K5 Failure Rate (Breakdowns): 6.4%

STRATEGY QUALITY ASSESSMENT
Overall Quality Score: 8/10
Assessment: EXCELLENT - Strategy demonstrates strong, consistent performance

RECOMMENDATIONS
1. Strategy shows strong performance across all key metrics.
2. Consider paper trading before live deployment.
3. Monitor performance closely for regime changes.
```

---

## ⚠️ Wichtige Hinweise

### Temporal Integrity

Das System garantiert **kein Look-Ahead Bias**:

- ✅ Training nur auf vergangenen Daten
- ✅ Test immer auf zukünftigen Daten (relative zum Training)
- ✅ Features werden zeitlich korrekt berechnet
- ✅ Model wird alle 90 Tage neu trainiert

### Daten-Anforderungen

Minimum für verlässliche Ergebnisse:

- **4 Jahre** historische Daten (1460+ Tage)
- **500+ Patterns** pro Training-Window
- **Saubere Outcome-Labels** (K0-K5)
- **Zeitstempel** für jedes Pattern

### Performance-Erwartungen

Realistische Ziele für AIv3 System:

| Metrik | Minimum | Gut | Exzellent |
|--------|---------|-----|-----------|
| Win Rate | 15% | 25% | 35% |
| Sharpe Ratio | 0.8 | 1.2 | 1.8 |
| Profit Factor | 1.3 | 1.8 | 2.5 |
| Max Drawdown | -25% | -18% | -12% |
| K3+K4 Hit Rate | 18% | 25% | 35% |

---

## 🐛 Troubleshooting

### Fehler: "Insufficient training data"

**Lösung**: Reduziere `min_train_samples` oder erhöhe `initial_train_years`

```python
backtester = ComprehensiveBacktester(
    initial_train_years=1.5,      # Reduziert von 2.0
    min_train_samples=300         # Reduziert von 500
)
```

### Warnung: "RuntimeWarning: divide by zero"

**Ursache**: Windows mit 0 Trades (Model macht keine Predictions)

**Lösung**: Normal bei synthetischen Daten. Bei echten Daten:
- Überprüfe Feature Quality
- Adjustiere Model-Parameter
- Überprüfe Klassen-Balancierung

### Hohe Variabilität in Win Rates

**Empfehlung**:
- Verwende Ensemble-Methods (Random Forest, XGBoost)
- Füge mehr robuste Features hinzu
- Erhöhe Training-Sample Size
- Verwende Feature Selection

---

## 📚 Weiterführende Dokumentation

### Module

1. **`walk_forward_validator.py`**: Kernlogik der Walk-Forward Validation
2. **`extended_performance_metrics.py`**: Berechnung aller Financial Metrics
3. **`comprehensive_backtester.py`**: Integration beider Systeme
4. **`run_comprehensive_backtest.py`**: Beispiel-Skript und Entry Point

### Referenzen

- **Walk-Forward Analysis**: [Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies]
- **Sharpe Ratio**: [Sharpe, W.F. (1994). The Sharpe Ratio]
- **Profit Factor**: Standard Trading Metric
- **Maximum Drawdown**: [Magdon-Ismail et al. (2004)]

---

## 🔄 Integration mit bestehendem System

### Mit AI Infra Model Trainer

```python
from AI Infra.train_best_model import ModelTrainer
from comprehensive_backtester import ComprehensiveBacktester

# Train model
trainer = ModelTrainer()
data = trainer.load_training_data('data/raw')
X, y = trainer.prepare_features(data)

# Create DataFrame for backtesting
df_for_backtest = data.copy()
# ... prepare features as columns

# Run comprehensive backtest
backtester = ComprehensiveBacktester()
results = backtester.run_comprehensive_backtest(
    df=df_for_backtest,
    model_class=VolumePatternModel,
    feature_columns=list(X.columns)
)
```

### Mit Pattern Detection

```python
from core.pattern_detector import PatternDetector
from comprehensive_backtester import ComprehensiveBacktester

# Detect patterns
detector = PatternDetector()
patterns = detector.detect_patterns(price_data)

# Convert to DataFrame
patterns_df = pd.DataFrame(patterns)

# Run backtest
backtester = ComprehensiveBacktester()
results = backtester.run_comprehensive_backtest(...)
```

---

## 📧 Support

Bei Fragen oder Problemen:

1. Überprüfe diese README
2. Siehe Beispiel in `run_comprehensive_backtest.py`
3. Überprüfe die generierten Reports in `output/`
4. Konsultiere die Modul-Docstrings

---

## 📅 Version History

### v1.0.0 (2025-10-09)
- ✅ Initial Release
- ✅ Walk-Forward Validation System
- ✅ Extended Performance Metrics
- ✅ Comprehensive Backtester Integration
- ✅ Publication-Ready Reports
- ✅ Example Scripts & Documentation

---

**Status**: ✅ Production Ready

**Last Updated**: 2025-10-09

**Author**: AIv3 Team
