# AIv4 - Consolidation Pattern Detection System

Advanced pattern detection system for identifying micro/small-cap stocks poised for significant upward moves.

## Quick Start

```bash
# 1. Scan patterns
python scan_patterns_with_enhanced_features.py --input data/gcs_cache --output output/

# 2. Label patterns
python label_patterns_parquet.py --input output/patterns_enhanced_*.parquet

# 3. Train models
python train_both_models.py --patterns output/patterns_labeled_*.parquet

# 4. Run validation
python run_validation_batch2.py --model-dir output/models/

# 5. Analyze results
python comprehensive_architecture_analysis.py
```

## Documentation

- **AIV4_SYSTEM_OVERVIEW.md** - Complete system description, workflow, and architecture
- **CLAUDE.md** - Project instructions and implementation details
- **DIRECTORY_STRUCTURE.md** - File organization

## System Architecture

- **Pattern Detection**: State machine (QUALIFYING → ACTIVE → COMPLETED)
- **ML Models**: XGBoost + LightGBM ensemble (50/50 weighted)
- **Outcome Classes**: K0-K5 (Stagnant to Exceptional)
- **Validation**: 194 patterns from 99 small-cap tickers

## Critical Issue (2025-10-27)

**Classification Logic Problem**: Current system uses "best outcome" logic, but requirements specify "first breach" logic. 94.3% of validation patterns would change classification. See CLAUDE.md for details.

## Project Status

- ✓ Pattern scanning system operational
- ✓ Model training pipeline complete
- ✓ Validation framework established
- ⚠️ Classification logic needs correction (first-breach implementation pending)
- ⚠️ Models trained on mis-labeled data (retraining required)

For detailed status and next steps, see **CLAUDE.md**.
