# AIv5 System Architecture - Visual Overview

## 🏗️ High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   AIv5 CONSOLIDATION PATTERN DETECTOR                    │
│                              Predicting 75%+ Breakouts from Micro-Cap Stocks            │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   DATA SOURCE    │────▶│ PATTERN DETECTOR │────▶│ FEATURE EXTRACT  │────▶│   ML PREDICT     │
│   GCS/YFinance   │     │   State Machine   │     │   69 Features    │     │  XGBoost/DL/Ens  │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                         │                         │
         ▼                        ▼                         ▼                         ▼
    OHLCV Data              Active Patterns          Feature Vectors            K0-K5 Classes
    (Daily Bars)            (Consolidation)           (Technical)              + EV Scores
```

## 📊 Detailed Component Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         DATA ACQUISITION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                   │
│  ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐        │
│  │  YFinance    │     │  Alpha Vantage  │     │ Twelve Data  │     │   GCS Storage   │        │
│  │  Downloader  │────▶│   Downloader    │────▶│  Downloader  │────▶│    Manager      │        │
│  └──────────────┘     └─────────────────┘     └──────────────┘     └─────────────────┘        │
│         │                                                                    │                   │
│         └────────────────────────────────────────────────────────────────────┘                   │
│                                              │                                                   │
│                                              ▼                                                   │
│                                    ┌──────────────────┐                                         │
│                                    │  Modern Foreman  │                                         │
│                                    │     Service      │                                         │
│                                    └──────────────────┘                                         │
│                                              │                                                   │
└──────────────────────────────────────────────┼───────────────────────────────────────────────────┘
                                               ▼
                                      Raw OHLCV Data (CSV/Parquet)
                                               │
┌──────────────────────────────────────────────┼───────────────────────────────────────────────────┐
│                                   PATTERN DETECTION LAYER                                         │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                              │                                                   │
│                                              ▼                                                   │
│                                 ┌────────────────────────┐                                      │
│                                 │  Stateful Detector     │                                      │
│                                 │  (Multi-ticker Mgr)    │                                      │
│                                 └────────────────────────┘                                      │
│                                         │         │                                              │
│                    ┌────────────────────┼─────────┼────────────────────┐                        │
│                    ▼                    ▼         ▼                    ▼                        │
│          ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐                        │
│          │ Tracker (AAPL)   │ │ Tracker (MSFT)   │ │ Tracker (NVDA)   │  ...                  │
│          └──────────────────┘ └──────────────────┘ └──────────────────┘                        │
│                    │                    │                    │                                   │
│                    └────────────────────┴────────────────────┘                                   │
│                                         │                                                        │
│                              State Machine per Ticker:                                          │
│                                         │                                                        │
│     ┌──────────┐      ┌──────────────┐      ┌──────────┐      ┌────────────┐                  │
│     │   NONE   │─────▶│  QUALIFYING  │─────▶│  ACTIVE  │─────▶│ COMPLETED  │                  │
│     │          │      │  (Days 1-10) │      │ (Day 10+)│      │   /FAILED  │                  │
│     └──────────┘      └──────────────┘      └──────────┘      └────────────┘                  │
│                                                    │                                             │
│                             Qualification Criteria:│                                             │
│                             • BBW < 30th percentile│                                             │
│                             • ADX < 32             │                                             │
│                             • Volume < 35% of avg  │                                             │
│                             • Range < 65% of avg   │                                             │
│                                                    ▼                                             │
│                                          Active Patterns (snapshots)                             │
│                                                    │                                             │
└────────────────────────────────────────────────────┼─────────────────────────────────────────────┘
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      FEATURE EXTRACTION LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                    │                                             │
│                                                    ▼                                             │
│                              ┌─────────────────────────────────────┐                            │
│                              │  Canonical Feature Extractor        │                            │
│                              │       (69 Features Total)           │                            │
│                              └─────────────────────────────────────┘                            │
│                                                    │                                             │
│                   ┌────────────────────────────────┼────────────────────────────────┐           │
│                   ▼                                ▼                                ▼           │
│        ┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐      │
│        │ Core Features(26)│           │  EBP Features(19)│           │Derived Feats(24) │      │
│        ├──────────────────┤           ├──────────────────┤           ├──────────────────┤      │
│        │• days_in_pattern │           │• cci_score       │           │• avg_range_20d   │      │
│        │• current_bbw_20  │           │• var_score       │           │• bbw_std_20d     │      │
│        │• volume_ratio    │           │• nes_score       │           │• volume_spike    │      │
│        │• distance_to_pwr │           │• lpf_score       │           │• range_expansion │      │
│        │• current_adx     │           │• ebp_composite   │           │• bbw_acceleration│      │
│        │• price_position  │           │• tsf_score       │           │• trend_clarity   │      │
│        │• ...             │           │• ...             │           │• ...             │      │
│        └──────────────────┘           └──────────────────┘           └──────────────────┘      │
│                                                    │                                             │
│                                                    ▼                                             │
│                                         Feature Vector (69 dims)                                │
│                                                    │                                             │
└────────────────────────────────────────────────────┼─────────────────────────────────────────────┘
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      MACHINE LEARNING LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                    │                                             │
│                              ┌─────────────────────┴─────────────────────┐                      │
│                              │         Model Factory                      │                      │
│                              └─────────────────────┬─────────────────────┘                      │
│                                                    │                                             │
│                   ┌────────────────────┬───────────┼──────────┬────────────────┐                │
│                   ▼                    ▼           ▼          ▼                ▼                │
│        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐     │
│        │XGBoost Classifier│ │XGBoost Regressor │ │  Deep Learning   │ │    Ensemble      │     │
│        │  (K0-K5 classes) │ │ (gain prediction)│ │ CNN-BiLSTM-Attn  │ │  (Weighted Avg)  │     │
│        └──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘     │
│                   │                                           │                 │                │
│                   └───────────────────┬───────────────────────┘                 │                │
│                                       ▼                                         ▼                │
│                            Class Probabilities:                      Expected Value (EV):        │
│                            P(K0) = 0.10                             EV = Σ(P(Ki) × Value(Ki))   │
│                            P(K1) = 0.15                                                          │
│                            P(K2) = 0.30                             Signal Strength:            │
│                            P(K3) = 0.25                             • STRONG: EV ≥ 5.0          │
│                            P(K4) = 0.15                             • GOOD:   EV ≥ 3.0          │
│                            P(K5) = 0.05                             • MODERATE: EV ≥ 1.0        │
│                                       │                             • AVOID: EV < 0             │
└───────────────────────────────────────┼──────────────────────────────────────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    OUTCOME CLASSIFICATION                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                   │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│    │K4 EXCEPTION │  │ K3 STRONG   │  │ K2 QUALITY  │  │ K1 MINIMAL  │  │K0 STAGNANT  │        │
│    │  ≥75% gain  │  │ 35-75% gain │  │ 15-35% gain │  │  5-15% gain │  │  <5% gain   │        │
│    │  Value: +10 │  │  Value: +3  │  │  Value: +1  │  │ Value: -0.2 │  │  Value: -2  │        │
│    └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                           ┌─────────────┐                                        │
│                                           │ K5 FAILED   │                                        │
│                                           │  Breakdown  │                                        │
│                                           │ Value: -10  │                                        │
│                                           └─────────────┘                                        │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PIPELINE                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  Historical Data          Pattern Detection        Feature Extraction              │
│       (4+ years)               (Batch)                 (Canonical)                 │
│           │                       │                         │                      │
│           ▼                       ▼                         ▼                      │
│    ┌──────────┐           ┌──────────────┐         ┌──────────────┐              │
│    │  OHLCV   │──────────▶│Scan Patterns │────────▶│Extract Feats │              │
│    │   Data   │           │  Historical  │         │ 69 Features  │              │
│    └──────────┘           └──────────────┘         └──────────────┘              │
│                                   │                         │                      │
│                                   ▼                         ▼                      │
│                           Pattern Snapshots         Feature Vectors                │
│                            (~340K samples)           (69 dimensions)               │
│                                   │                         │                      │
│                                   └─────────┬───────────────┘                      │
│                                             ▼                                      │
│                                    ┌──────────────┐                               │
│                                    │Label Outcomes│                               │
│                                    │  (K0-K5)     │                               │
│                                    └──────────────┘                               │
│                                             │                                      │
│                                             ▼                                      │
│                                     Training Data                                  │
│                                   (Features + Labels)                             │
│                                             │                                      │
│                          ┌──────────────────┼──────────────────┐                  │
│                          ▼                  ▼                  ▼                  │
│                  ┌──────────────┐   ┌──────────────┐  ┌──────────────┐          │
│                  │Train XGBoost │   │Train LightGBM│  │Train Deep NN │          │
│                  │  Classifier  │   │  Classifier  │  │  CNN-BiLSTM  │          │
│                  └──────────────┘   └──────────────┘  └──────────────┘          │
│                          │                  │                  │                  │
│                          └──────────────────┼──────────────────┘                  │
│                                             ▼                                      │
│                                      Trained Models                                │
│                                        (.pkl files)                                │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                              PREDICTION PIPELINE                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│     Live/Recent Data        Real-time Detection       Feature Extraction          │
│         (Today)                (Streaming)               (Canonical)              │
│           │                         │                         │                    │
│           ▼                         ▼                         ▼                    │
│    ┌──────────┐            ┌──────────────┐         ┌──────────────┐            │
│    │New OHLCV │───────────▶│State Machine │────────▶│Extract Feats │            │
│    │   Bar    │            │   Updates    │         │ 69 Features  │            │
│    └──────────┘            └──────────────┘         └──────────────┘            │
│                                     │                         │                    │
│                                     ▼                         ▼                    │
│                              Active Patterns           Feature Vector              │
│                             (ACTIVE phase)              (69 dims)                 │
│                                     │                         │                    │
│                                     └────────┬────────────────┘                    │
│                                              ▼                                     │
│                                     ┌──────────────┐                             │
│                                     │Load Models   │                             │
│                                     │& Predict     │                             │
│                                     └──────────────┘                             │
│                                              │                                     │
│                                              ▼                                     │
│                                    Predictions + EV                               │
│                                              │                                     │
│                           ┌──────────────────┼──────────────────┐                │
│                           ▼                  ▼                  ▼                │
│                    ┌──────────┐      ┌──────────┐      ┌──────────┐            │
│                    │  STRONG  │      │   GOOD   │      │  AVOID   │            │
│                    │  SIGNAL  │      │  SIGNAL  │      │  SIGNAL  │            │
│                    │  EV ≥ 5  │      │  3≤EV<5  │      │  EV < 0  │            │
│                    └──────────┘      └──────────┘      └──────────┘            │
│                                                                                    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 Key System Characteristics

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM PERFORMANCE METRICS                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Pattern Detection Rate:     █████░░░░░░░░░░░  30-40% qualify          │
│  Active Pattern Rate:        ███░░░░░░░░░░░░░  15-20% become active    │
│  Strong Signal Rate:         █░░░░░░░░░░░░░░░   2-5% generate signals  │
│                                                                          │
│  K4 Detection (≥75% gain):   ██░░░░░░░░░░░░░░  10-15% of patterns      │
│  K3 Detection (35-75%):      ████░░░░░░░░░░░░  20-25% of patterns      │
│  K5 Avoidance (failures):    ████████████░░░░  80-85% accuracy         │
│                                                                          │
│  Expected Value Range:       [-10 ←─────────────→ +10]                  │
│                                   │     │     │                         │
│                               Avoid  Moderate Strong                    │
│                                                                          │
│  Model Retraining:           Every 90 days (quarterly)                  │
│  Backtest Period:            4+ years minimum                           │
│  Feature Count:              69 canonical features                      │
│  Training Samples:           ~340K pattern snapshots                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📁 Simplified Directory Structure

```
AIv5/
│
├── pattern_detection/           [PATTERN DETECTION ENGINE]
│   ├── state_machine/          ← ConsolidationTracker, StatefulDetector
│   ├── features/               ← CanonicalFeatureExtractor (69 features)
│   └── adapters/               ← Pattern adapters
│
├── training/                    [MACHINE LEARNING]
│   └── models/
│       └── implementations/    ← XGBoost, LightGBM, Deep Learning
│           ├── xgboost_classifier.py
│           ├── pattern_value_system.py
│           └── deep_learning/
│               ├── cnn_bilstm_attention_model.py
│               └── ensemble_predictor.py
│
├── data_acquisition/            [DATA LAYER]
│   ├── sources/                ← YFinance, AlphaVantage, TwelveData
│   └── services/               ← ModernForemanService
│
├── src/pipeline/               [CORE PIPELINES]
│   ├── 01_scan_patterns.py    ← Detect patterns
│   ├── 02_label_patterns.py   ← Assign K0-K5 outcomes
│   └── 03_train_models.py     ← Train ML models
│
├── shared/                     [SHARED UTILITIES]
│   ├── config/                 ← Pydantic settings
│   └── indicators/             ← BBW, ADX, Volume indicators
│
├── tests/                      [TEST SUITE]
│   └── test_canonical_feature_extractor.py
│
├── output/                     [RESULTS]
│   ├── models/                 ← Trained .pkl files
│   └── validation/             ← Backtest results
│
└── data/                       [DATA STORAGE]
    └── gcs_cache/              ← Local OHLCV cache
```

## 🚀 System Strengths After Simplification

1. **Unified Feature Pipeline**: Single source of truth (69 features)
2. **Clean Architecture**: No duplicate directories or competing implementations
3. **State Machine Integrity**: Temporal consistency guaranteed
4. **Modular Design**: Easy to swap ML models or data sources
5. **Production Ready**: Simplified structure ready for deployment

The AIv5 system is now a streamlined consolidation pattern detector with clear data flow, unified features, and maintainable architecture.