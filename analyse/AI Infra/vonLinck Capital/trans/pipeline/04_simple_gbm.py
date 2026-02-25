"""
Simplified Gradient Boosting Model for Consolidation Breakouts

PHILOSOPHY:
- 10 features max (interpretable)
- Target: 10%+ price move (Class 2 = Target)
- Metric: Precision-Recall AUC (not accuracy)
- Reality check: >15% hit rate in top-15% or signal doesn't exist
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import h5py
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    classification_report,
    roc_auc_score
)
import joblib

# ============================================================================
# FEATURE DEFINITIONS (from 24 context features, pick top 10)
# ============================================================================

# Full 24 context features (from CLAUDE.md)
FULL_FEATURE_NAMES = [
    'retention_rate',           # 0
    'trend_position',           # 1
    'base_duration',            # 2
    'relative_volume',          # 3 (log-diff)
    'distance_to_high',         # 4
    'log_float',                # 5
    'log_dollar_volume',        # 6
    'dormancy_shock',           # 7 (log-diff)
    'vol_dryup_ratio',          # 8 (log-diff)
    'price_position_at_end',    # 9
    'volume_shock',             # 10 (log-diff)
    'bbw_slope_5d',             # 11
    'vol_trend_5d',             # 12 (log-diff)
    'coil_intensity',           # 13
    'relative_strength_cohort', # 14
    'risk_width_pct',           # 15
    'vol_contraction_intensity',# 16 (log-diff)
    'obv_divergence',           # 17 (log-diff)
    'market_phase_bull',        # 18 (one-hot)
    'market_phase_bear',        # 19 (one-hot)
    'market_phase_sideways',    # 20 (one-hot)
    'market_phase_recovery',    # 21 (one-hot)
    'regime_vix_level',         # 22
    'regime_trend_strength',    # 23
]

# TOP 10 features to use (most interpretable and likely predictive)
TOP_10_INDICES = [
    13,  # coil_intensity - composite squeeze score
    15,  # risk_width_pct - box tightness
    11,  # bbw_slope_5d - volatility contracting?
    9,   # price_position_at_end - where in box
    3,   # relative_volume - accumulation
    10,  # volume_shock - spike detection
    2,   # base_duration - time in consolidation
    1,   # trend_position - above/below SMA200
    6,   # log_dollar_volume - liquidity
    4,   # distance_to_high - how beaten down
]

TOP_10_NAMES = [FULL_FEATURE_NAMES[i] for i in TOP_10_INDICES]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(sequences_path: str, metadata_path: str):
    """
    Load sequences (context features) and labels from HDF5 + metadata.
    """
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")

    # Load HDF5
    with h5py.File(sequences_path, 'r') as f:
        context = f['context'][:]
        labels = f['labels'][:]
        print(f"Context shape: {context.shape}")
        print(f"Labels shape: {labels.shape}")

    # Load metadata for temporal splitting
    meta = pd.read_parquet(metadata_path)
    print(f"Metadata shape: {meta.shape}")

    # Convert labels: Class 2 (Target) = 1, else = 0
    # This is our "10%+ move" proxy (Class 2 = hit +3R target)
    y = (labels == 2).astype(int)

    print(f"\nLabel distribution:")
    print(f"  Original: Class0={np.sum(labels==0):,}, Class1={np.sum(labels==1):,}, Class2={np.sum(labels==2):,}")
    print(f"  Binary:   Negative={np.sum(y==0):,} ({np.mean(y==0)*100:.1f}%), Positive={np.sum(y==1):,} ({np.mean(y==1)*100:.1f}%)")

    # Extract top 10 features
    X = context[:, TOP_10_INDICES]
    print(f"\nUsing {len(TOP_10_INDICES)} features: {TOP_10_NAMES}")

    return X, y, meta


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_gbm(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, test_size: float = 0.2):
    """
    Train Gradient Boosting with temporal split and class weighting.
    """
    print(f"\n{'='*60}")
    print("TRAINING GRADIENT BOOSTING MODEL")
    print(f"{'='*60}")

    # Temporal split based on pattern_end_date
    meta['date'] = pd.to_datetime(meta['pattern_end_date'])
    meta = meta.sort_values('date')

    # Get sorted indices
    sorted_idx = meta.index.values

    # Split: oldest 80% for train, newest 20% for test
    split_point = int(len(sorted_idx) * (1 - test_size))
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nTemporal split:")
    print(f"  Train: {len(X_train):,} samples ({y_train.mean()*100:.1f}% positive)")
    print(f"  Test:  {len(X_test):,} samples ({y_test.mean()*100:.1f}% positive)")
    print(f"  Train dates: {meta.iloc[train_idx[:1]]['date'].values[0]} to {meta.iloc[train_idx[-1:]]['date'].values[0]}")
    print(f"  Test dates:  {meta.iloc[test_idx[:1]]['date'].values[0]} to {meta.iloc[test_idx[-1:]]['date'].values[0]}")

    # Calculate class weight (handle imbalance)
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    if n_pos > 0:
        pos_weight = n_neg / n_pos  # Upweight minority class
    else:
        pos_weight = 1.0

    print(f"\nClass weighting: pos_weight={pos_weight:.1f}x")

    # Create sample weights
    sample_weights = np.where(y_train == 1, pos_weight, 1.0)

    # Train model - simple, interpretable
    model = GradientBoostingClassifier(
        n_estimators=100,          # Not too many trees
        max_depth=3,               # Shallow trees = interpretable
        learning_rate=0.1,
        min_samples_leaf=50,       # Prevent overfitting
        subsample=0.8,
        random_state=42,
        verbose=0
    )

    print("\nTraining GBM...")
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("Done.")

    return model, X_train, X_test, y_train, y_test, train_idx, test_idx


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate with PR-AUC and top-15% hit rate.

    THE KEY METRIC: If top-15% hit rate < 15%, the signal doesn't exist.
    """
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")

    # Get probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # ===================
    # PR-AUC (PRIMARY METRIC)
    # ===================
    pr_auc = average_precision_score(y_test, y_prob)
    baseline_pr = y_test.mean()  # Random classifier baseline

    print(f"\n📊 PRECISION-RECALL AUC")
    print(f"   PR-AUC: {pr_auc:.3f}")
    print(f"   Baseline (random): {baseline_pr:.3f}")
    print(f"   Improvement: {pr_auc/baseline_pr:.1f}x over random")

    # ===================
    # TOP-15% HIT RATE (REALITY CHECK)
    # ===================
    n_top = max(1, int(len(y_test) * 0.15))
    top_indices = np.argsort(y_prob)[-n_top:]
    top_hit_rate = y_test[top_indices].mean() * 100

    baseline_rate = y_test.mean() * 100
    lift = top_hit_rate / baseline_rate if baseline_rate > 0 else 0

    print(f"\n🎯 TOP-15% ANALYSIS (n={n_top})")
    print(f"   Hit rate in top-15%: {top_hit_rate:.1f}%")
    print(f"   Baseline rate: {baseline_rate:.1f}%")
    print(f"   Lift: {lift:.2f}x")

    # THE REALITY CHECK
    print(f"\n{'='*60}")
    if top_hit_rate >= 15:
        print(f"✅ SIGNAL EXISTS: {top_hit_rate:.1f}% >= 15% threshold")
        signal_exists = True
    else:
        print(f"❌ NO SIGNAL: {top_hit_rate:.1f}% < 15% threshold")
        print(f"   The model cannot reliably identify 10%+ movers")
        signal_exists = False
    print(f"{'='*60}")

    # ===================
    # FEATURE IMPORTANCE
    # ===================
    print(f"\n📈 FEATURE IMPORTANCE:")
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.iterrows():
        bar = '█' * int(row['importance'] * 40)
        print(f"   {row['feature']:28s} {row['importance']:.3f} {bar}")

    # ===================
    # PRECISION AT DIFFERENT RECALL LEVELS
    # ===================
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    print(f"\n📉 PRECISION @ RECALL:")
    for target_recall in [0.1, 0.2, 0.3, 0.5, 0.7]:
        idx = np.argmin(np.abs(recall - target_recall))
        if idx < len(precision):
            print(f"   Recall={target_recall:.0%}: Precision={precision[idx]:.1%}")

    # ===================
    # TOP PREDICTIONS ANALYSIS
    # ===================
    print(f"\n🔍 TOP-15% PREDICTIONS BREAKDOWN:")
    top_probs = y_prob[top_indices]
    top_actual = y_test[top_indices]
    print(f"   Probability range: [{top_probs.min():.3f}, {top_probs.max():.3f}]")
    print(f"   True positives: {top_actual.sum()} / {n_top}")
    print(f"   False positives: {n_top - top_actual.sum()} / {n_top}")

    # ROC-AUC for comparison
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"\n   (ROC-AUC for reference: {roc_auc:.3f})")

    return {
        'pr_auc': pr_auc,
        'top_15_hit_rate': top_hit_rate,
        'lift': lift,
        'baseline': baseline_rate,
        'signal_exists': signal_exists,
        'feature_importance': importance
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Simplified GBM for breakout prediction')
    parser.add_argument('--sequences', type=str, required=True,
                        help='Path to sequences HDF5 file')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to metadata parquet file')
    parser.add_argument('--output-dir', type=str, default='output/gbm',
                        help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')

    args = parser.parse_args()

    print("""
================================================================
     SIMPLIFIED GRADIENT BOOSTING - BREAKOUT DETECTION
================================================================
  Features:  10 (interpretable)
  Target:    Class 2 (10%+ move, hit +3R)
  Metric:    PR-AUC + Top-15% hit rate
  Reality:   >15% hit rate in top-15% or NO SIGNAL
================================================================
""")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, meta = load_data(args.sequences, args.metadata)

    # Train
    model, X_train, X_test, y_train, y_test, train_idx, test_idx = train_gbm(
        X, y, meta, args.test_size
    )

    # Evaluate
    results = evaluate_model(model, X_test, y_test, TOP_10_NAMES)

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = output_dir / f'gbm_model_{timestamp}.joblib'
    joblib.dump({
        'model': model,
        'feature_names': TOP_10_NAMES,
        'feature_indices': TOP_10_INDICES,
        'results': results
    }, model_path)
    print(f"\n💾 Model saved: {model_path}")

    # Save results summary
    results_path = output_dir / f'results_{timestamp}.txt'
    with open(results_path, 'w') as f:
        f.write("SIMPLIFIED GBM RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"PR-AUC: {results['pr_auc']:.3f}\n")
        f.write(f"Top-15% Hit Rate: {results['top_15_hit_rate']:.1f}%\n")
        f.write(f"Lift: {results['lift']:.2f}x\n")
        f.write(f"Baseline: {results['baseline']:.1f}%\n")
        f.write(f"Signal Exists: {results['signal_exists']}\n\n")
        f.write("Feature Importance:\n")
        f.write(results['feature_importance'].to_string())
    print(f"📄 Results saved: {results_path}")

    # Final verdict
    print(f"\n{'='*60}")
    if results['signal_exists']:
        print("✅ CONCLUSION: Signal exists. Proceed to backtesting.")
    else:
        print("❌ CONCLUSION: No reliable signal found.")
        print("   Options:")
        print("   1. Try different features")
        print("   2. Use EU data (showed 1.85x lift)")
        print("   3. Accept that these patterns don't predict 10%+ moves")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    main()
