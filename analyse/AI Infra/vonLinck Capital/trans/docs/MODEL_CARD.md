# TRAnS Model Card

**T**emporal **R**etail **An**alysis **S**ystem for Micro/Small-Cap Consolidation Breakouts

---

## Model Description

### Overview

TRAnS is a deep learning system designed to identify high-probability consolidation breakout patterns in micro/small-cap equities. The model predicts the likelihood of three outcome classes:

| Class | Name | Description | Strategic Value |
|-------|------|-------------|-----------------|
| 0 | Danger | Pattern breaks down, stop loss hit | -2.0 R |
| 1 | Noise | No clear outcome within window | -0.1 R |
| 2 | Target | Breakout achieves +3R target | +5.0 R |

### Architecture

**HybridFeatureNetwork (V18)**:
- **CNN Branch**: 2-layer 1D convolution for local pattern extraction
- **LSTM Branch**: 64-unit bidirectional LSTM for temporal dynamics
- **GRN Context Branch**: Gated Residual Network for static context features
- **Fusion MLP**: Multi-layer perceptron combining all branches

**Input Features**:
- Sequence: 20 timesteps × 10 features (BBW, ADX, volume ratios, momentum, etc.)
- Context: 24 static features (pattern metrics, market cap, sector, etc.)

**Output**: 3-class probability distribution

### Training Data

| Attribute | Value |
|-----------|-------|
| Region | EU micro/small-caps |
| Date Range | 2015-01-01 to 2024-12-31 |
| Patterns | ~6,240 (after deduplication) |
| Target Rate | ~4% |
| Training Split | Pre-2023 |
| Validation Split | 2023 |
| Test Split | 2024 |

### Performance Metrics

**Jan 30, 2026 Model (Production)**:

| Metric | Value |
|--------|-------|
| Top-15 Precision | 8% |
| Top-15 Lift | 2.7x baseline |
| Global Accuracy | ~45% |
| F1 (Target class) | ~0.20 |
| Danger Recall | ~70% |

---

## Intended Use

### Primary Use Case

Identifying consolidation patterns in EU micro/small-cap equities with high breakout probability for semi-automated EOD (end-of-day) trading decisions.

### Recommended Context

- **Account Size**: $10,000 - $100,000
- **Risk Per Trade**: $250 fixed R-unit
- **Max Position**: $5,000 or 4% ADV
- **Execution**: Manual/semi-automated, EOD
- **Holding Period**: 10-60 days (volatility-dependent)

### Out-of-Scope Uses

This model is **NOT** designed for:

1. **Intraday trading**: Model predictions are based on EOD data
2. **Large-cap stocks**: Trained on micro/small-caps with different dynamics
3. **US market**: Trained on EU equities (cross-region validation pending)
4. **Automated execution**: Requires human review of signals
5. **Short selling**: Long-only pattern detection
6. **Crypto/Forex**: Equity-specific patterns only

---

## Known Biases

### Survivorship Bias

**Status**: PARTIALLY ADDRESSED

- Training data includes some delisted stocks
- `utils/survivorship_checker.py` provides detection tools
- **Mitigation**: Always validate on out-of-sample periods

### Geographic Bias

**Status**: KNOWN LIMITATION

| Exchange | Patterns | Target Rate | Note |
|----------|----------|-------------|------|
| Germany (.DE) | 25% | 1.9% | Dominates data |
| France (.PA) | 18% | 2.4% | Large sample |
| Sweden (.ST) | 10% | 4.2% | Good signal |
| Denmark (.CO) | 6% | 7.7% | Best rate |
| UK (.L) | 0.5% | 0% | Poor quality |

**Mitigation**: Consider exchange-specific signal thresholds

### Market Regime Bias

**Status**: PARTIALLY ADDRESSED

Training period includes:
- Post-QE bull (2015-2016)
- COVID crash/recovery (2020)
- Bear market (2022)
- AI bull rally (2023-2024)

**Mitigation**:
- `config/regime_config.py` provides regime-aware parameters
- Model should be retrained after extended new regime exposure

### Selection Bias

**Status**: ADDRESSED

- Patterns require $50k minimum dollar volume (liquidity floor)
- Deduplication removes 73.6% overlapping patterns
- Cross-ticker dedup removes ETF mirror patterns

---

## Failure Modes

### Known Failure Scenarios

| Scenario | Description | Detection | Mitigation |
|----------|-------------|-----------|------------|
| Low liquidity | <$50k ADV causes execution problems | Physics filter | Auto-rejected |
| Gap breakouts | >0.5R gap destroys R:R | Gap check | Marked untradeable |
| Extended halts | Missing data during halt | Halt detection | Zombie protocol |
| Regime shift | New market conditions | Drift monitoring | Retrain trigger |
| Sector rotation | Single sector drives patterns | Sector concentration | Diversification |

### High-Risk Predictions

The model has elevated error rates when:

1. **P(Danger) > 25%**: 58.9% of these failed historically
2. **Box width < 2.5%**: Too tight for meaningful stop
3. **Volume < 20-day average**: Insufficient conviction
4. **ADX > 32**: Trending, not consolidating

### Error Handling

The pipeline includes guards:

```python
# Danger filter (production recommended)
--max-danger-prob 0.25

# Liquidity filter (mandatory)
--min-dollar-volume 50000

# Gap filter (auto-applied)
untradeable=True when gap > 0.5R
```

---

## Limitations

### Data Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Variable data depth | Some tickers lack 2015 history | Adaptive regime splits |
| Missing delisted data | Some survivorship bias remains | Track survival status |
| EOD data only | No intraday patterns | EOD execution only |
| EU market cap APIs fail | 99% API failure rate | ADV-based fallback |

### Model Limitations

| Limitation | Impact | Future Work |
|------------|--------|-------------|
| Long-only | Misses short opportunities | Not planned |
| 3-class output | No gradient of success | Keep simple |
| Static window | Not adaptive to volatility | V22 dynamic window helps |
| No market context | Ignores macro environment | Regime features exist |

### Execution Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Slippage not modeled | Realized R may differ | 0.5R gap limit |
| Intraday stops not caught | May miss exact stop hit | Use LOW price for checks |
| Extended hours excluded | Gaps more common | Gap penalty in labeling |

---

## Ethical Considerations

### Market Impact

- Intended for retail-scale positions ($250-$5,000)
- Micro-cap stocks have limited liquidity
- Large positions could impact prices
- **Recommendation**: Limit to 4% of 20-day ADV per position

### Information Asymmetry

- Model identifies public patterns
- No insider information used
- No front-running capability
- Signals available EOD only

### Risk Disclosure

**This model is not financial advice.**

- Past performance does not guarantee future results
- All trading involves risk of loss
- The model may underperform in future market conditions
- Users should understand the methodology before trading

---

## Model Versioning

### Current Version

| Attribute | Value |
|-----------|-------|
| Model Version | V18 |
| Labeling Version | V22 (Dynamic Window) |
| Training Date | Jan 30, 2026 |
| Checkpoint | `production_model.pt` |
| Region | EU |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| V18 | 2025-01 | CNN + LSTM + GRN architecture |
| V22 Label | 2026-01 | Dynamic outcome window |
| Dedup | 2026-01 | Mandatory 73.6% deduplication |
| Liquidity | 2026-01 | $50k floor (reverted from $100k) |

### Retraining Protocol

Retrain when:
1. **90+ days elapsed** since last training
2. **PSI > 0.25** (population stability index)
3. **New market regime** extended exposure
4. **Target rate shift** > 20% from training

---

## Contact and Support

- **Repository Issues**: [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- **Documentation**: See `docs/SYSTEM_OVERVIEW.md` for detailed system description

---

## Citation

If referencing this model in research:

```
TRAnS: Temporal Retail Analysis System for Consolidation Breakouts
Version 1.0, January 2026
vonLinck Capital
```

---

*Last Updated: January 31, 2026*
