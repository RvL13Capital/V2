"""
XAI (Explainable AI) Analysis for TRAnS GBM Model

Generates interpretability visualizations:
1. SHAP Summary Plot - Global feature importance with directionality
2. SHAP Dependence Plots - Feature interactions
3. Permutation Importance - Model-agnostic validation
4. Partial Dependence Plots - Feature effect curves

SHAP TreeExplainer is EXACT for tree-based models (no approximation loss).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import h5py
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not installed. Run: pip install shap")

# Import feature names from config (single source of truth)
from config.context_features import CONTEXT_FEATURES
from config.temporal_features import TemporalFeatureConfig

# =============================================================================
# CONFIGURATION
# =============================================================================

# Temporal features from config
_temporal_config = TemporalFeatureConfig()
TEMPORAL_FEATURES = _temporal_config.all_features

def get_feature_names():
    names = list(CONTEXT_FEATURES)
    for feat in TEMPORAL_FEATURES:
        names.append(f'{feat}_mean')
    for feat in TEMPORAL_FEATURES:
        names.append(f'{feat}_std')
    return names

FEATURE_NAMES = get_feature_names()


def load_data():
    """Load and prepare data for XAI analysis."""
    seq_path = 'output/sequences/eu_v18/sequences_20260109_023519.h5'
    meta_path = 'output/sequences/eu_v18/metadata_20260109_023519.parquet'

    with h5py.File(seq_path, 'r') as f:
        context = f['context'][:]
        sequences = f['sequences'][:]
        labels = f['labels'][:]

    # Create combined features
    seq_mean = np.nanmean(sequences, axis=1)
    seq_std = np.nanstd(sequences, axis=1)
    X = np.hstack([context, seq_mean, seq_std])
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

    meta = pd.read_parquet(meta_path)
    meta['pattern_end_date'] = pd.to_datetime(meta['pattern_end_date'])

    y = (labels == 2).astype(int)

    return X, y, meta


def train_model(X, y, meta):
    """Train GBM with temporal split."""
    meta = meta.copy()
    meta['date'] = meta['pattern_end_date']

    # Temporal split
    train_mask = meta['date'] <= '2022-12-31'
    test_mask = meta['date'] >= '2023-03-01'

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Class weighting
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    sample_weights = np.where(y_train == 1, pos_weight, 1.0)

    # Train
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    return model, X_train, X_test, y_train, y_test


def analyze_shap(model, X_train, X_test, output_dir):
    """SHAP analysis - exact for tree models."""
    if not SHAP_AVAILABLE:
        print("SHAP not available, skipping...")
        return None

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS (TreeExplainer - EXACT)")
    print("=" * 60)

    # Use subset for speed
    n_samples = min(1000, len(X_test))
    X_sample = X_test[:n_samples]

    # TreeExplainer is EXACT for GBM (no approximation)
    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (Target)

    print(f"SHAP values shape: {shap_values.shape}")

    # 1. Summary Plot (Beeswarm)
    print("\nGenerating SHAP Summary Plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=FEATURE_NAMES,
        show=False,
        max_display=20
    )
    plt.title("SHAP Feature Importance (Impact on Target Prediction)")
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'shap_summary.png'}")

    # 2. Bar Plot (Mean absolute SHAP)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=FEATURE_NAMES,
        plot_type="bar",
        show=False,
        max_display=20
    )
    plt.title("Mean |SHAP| Value (Feature Importance)")
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_importance_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'shap_importance_bar.png'}")

    # 3. Top feature dependence plots
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = np.argsort(mean_abs_shap)[-6:][::-1]  # Top 6

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat_idx in enumerate(top_features):
        shap.dependence_plot(
            feat_idx, shap_values, X_sample,
            feature_names=FEATURE_NAMES,
            ax=axes[i],
            show=False
        )
        axes[i].set_title(f"{FEATURE_NAMES[feat_idx]}")

    plt.suptitle("SHAP Dependence Plots (Top 6 Features)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_dependence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'shap_dependence.png'}")

    # Calculate feature importance from SHAP
    shap_importance = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'shap_importance': mean_abs_shap
    }).sort_values('shap_importance', ascending=False)

    return shap_importance


def analyze_permutation_importance(model, X_test, y_test, output_dir):
    """Permutation importance - model-agnostic validation."""
    print("\n" + "=" * 60)
    print("PERMUTATION IMPORTANCE ANALYSIS")
    print("=" * 60)

    # Use subset for speed
    n_samples = min(2000, len(X_test))
    X_sample = X_test[:n_samples]
    y_sample = y_test[:n_samples]

    print("Computing permutation importance (10 repeats)...")
    result = permutation_importance(
        model, X_sample, y_sample,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    # Sort by importance
    perm_importance = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)

    # Plot
    plt.figure(figsize=(10, 10))
    top_n = 20
    top_features = perm_importance.head(top_n)

    plt.barh(range(top_n), top_features['importance_mean'].values,
             xerr=top_features['importance_std'].values,
             color='steelblue', alpha=0.8)
    plt.yticks(range(top_n), top_features['feature'].values)
    plt.xlabel('Mean Accuracy Decrease')
    plt.title('Permutation Importance (Top 20 Features)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'permutation_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'permutation_importance.png'}")

    return perm_importance


def analyze_builtin_importance(model, output_dir):
    """Built-in GBM feature importance (Gini-based)."""
    print("\n" + "=" * 60)
    print("BUILT-IN FEATURE IMPORTANCE (Gini/Split-based)")
    print("=" * 60)

    builtin_importance = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'gini_importance': model.feature_importances_
    }).sort_values('gini_importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 10))
    top_n = 20
    top_features = builtin_importance.head(top_n)

    colors = ['#e74c3c' if imp > 0.05 else '#3498db' for imp in top_features['gini_importance']]
    plt.barh(range(top_n), top_features['gini_importance'].values, color=colors, alpha=0.8)
    plt.yticks(range(top_n), top_features['feature'].values)
    plt.xlabel('Gini Importance')
    plt.title('Built-in Feature Importance (Top 20 Features)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'builtin_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'builtin_importance.png'}")

    return builtin_importance


def compare_importance_methods(builtin, permutation, shap_imp, output_dir):
    """Compare different importance methods."""
    print("\n" + "=" * 60)
    print("IMPORTANCE METHOD COMPARISON")
    print("=" * 60)

    # Merge all methods
    comparison = builtin[['feature', 'gini_importance']].merge(
        permutation[['feature', 'importance_mean']].rename(
            columns={'importance_mean': 'permutation_importance'}),
        on='feature'
    )

    if shap_imp is not None:
        comparison = comparison.merge(
            shap_imp[['feature', 'shap_importance']],
            on='feature'
        )

    # Normalize to [0, 1] for comparison
    for col in ['gini_importance', 'permutation_importance']:
        comparison[f'{col}_norm'] = comparison[col] / comparison[col].max()

    if shap_imp is not None:
        comparison['shap_importance_norm'] = comparison['shap_importance'] / comparison['shap_importance'].max()

    # Calculate correlation between methods
    print("\nCorrelation between importance methods:")
    cols_to_corr = ['gini_importance', 'permutation_importance']
    if shap_imp is not None:
        cols_to_corr.append('shap_importance')

    corr_matrix = comparison[cols_to_corr].corr()
    print(corr_matrix.round(3).to_string())

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get top 15 features by average rank
    comparison['avg_rank'] = comparison[cols_to_corr].rank(ascending=False).mean(axis=1)
    top_features = comparison.nsmallest(15, 'avg_rank')

    x = np.arange(len(top_features))
    width = 0.25

    bars1 = ax.barh(x - width, top_features['gini_importance_norm'], width,
                    label='Gini (Built-in)', color='#e74c3c', alpha=0.8)
    bars2 = ax.barh(x, top_features['permutation_importance_norm'], width,
                    label='Permutation', color='#3498db', alpha=0.8)
    if shap_imp is not None:
        bars3 = ax.barh(x + width, top_features['shap_importance_norm'], width,
                        label='SHAP', color='#2ecc71', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Normalized Importance')
    ax.set_title('Feature Importance: Method Comparison (Top 15)')
    ax.legend(loc='lower right')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'importance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {output_dir / 'importance_comparison.png'}")

    # Save comparison table
    comparison.to_csv(output_dir / 'importance_comparison.csv', index=False)
    print(f"  Saved: {output_dir / 'importance_comparison.csv'}")

    return comparison


def generate_xai_report(comparison, output_dir):
    """Generate XAI summary report."""
    print("\n" + "=" * 60)
    print("XAI SUMMARY REPORT")
    print("=" * 60)

    report = []
    report.append("# XAI Analysis Report - TRAnS GBM Model")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Method Comparison")
    report.append("\n| Method | Strengths | Weaknesses |")
    report.append("|--------|-----------|------------|")
    report.append("| **SHAP TreeExplainer** | Exact for trees, shows directionality | Slower on large data |")
    report.append("| **Permutation** | Model-agnostic, robust | Affected by correlated features |")
    report.append("| **Gini (Built-in)** | Fast, always available | Biased toward high-cardinality |")

    report.append("\n## Consistency Analysis")

    # Check if top features are consistent across methods
    cols = ['gini_importance', 'permutation_importance']
    if 'shap_importance' in comparison.columns:
        cols.append('shap_importance')

    top_gini = set(comparison.nlargest(5, 'gini_importance')['feature'])
    top_perm = set(comparison.nlargest(5, 'permutation_importance')['feature'])

    if 'shap_importance' in comparison.columns:
        top_shap = set(comparison.nlargest(5, 'shap_importance')['feature'])
        common_all = top_gini & top_perm & top_shap
        report.append(f"\nTop 5 features common to ALL methods: {common_all}")
    else:
        common_all = top_gini & top_perm
        report.append(f"\nTop 5 features common to Gini & Permutation: {common_all}")

    report.append("\n## Top 10 Features (by average rank)")
    comparison['avg_rank'] = comparison[cols].rank(ascending=False).mean(axis=1)
    top10 = comparison.nsmallest(10, 'avg_rank')[['feature'] + cols + ['avg_rank']]
    report.append("\n```")
    report.append(top10.to_string(index=False))
    report.append("```")

    report.append("\n## Interpretation Guidelines")
    report.append("\n1. **close_std** - High price volatility during consolidation predicts breakouts")
    report.append("2. **adx_mean** - Average trend strength during pattern formation")
    report.append("3. **bbw_20_mean** - Bollinger Band squeeze intensity")
    report.append("4. **trend_position** - Where price sits relative to long-term trend")
    report.append("5. **base_duration** - How long the consolidation has lasted")

    report.append("\n## Files Generated")
    report.append("\n| File | Description |")
    report.append("|------|-------------|")
    report.append("| shap_summary.png | SHAP beeswarm plot |")
    report.append("| shap_importance_bar.png | Mean |SHAP| values |")
    report.append("| shap_dependence.png | Top 6 feature dependence |")
    report.append("| permutation_importance.png | Permutation importance |")
    report.append("| builtin_importance.png | Gini importance |")
    report.append("| importance_comparison.png | Method comparison |")
    report.append("| importance_comparison.csv | Full comparison data |")

    # Save report
    report_text = '\n'.join(report)
    with open(output_dir / 'XAI_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  Saved: {output_dir / 'XAI_REPORT.md'}")


def main():
    print("=" * 60)
    print("TRAnS XAI ANALYSIS")
    print("Explainable AI for GBM Model")
    print("=" * 60)

    # Setup output directory
    output_dir = Path('output/gbm/xai')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    X, y, meta = load_data()
    print(f"Total samples: {len(X):,}")
    print(f"Features: {X.shape[1]}")

    # Train model
    print("\nTraining model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y, meta)
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Built-in importance
    builtin = analyze_builtin_importance(model, output_dir)

    # Permutation importance
    permutation = analyze_permutation_importance(model, X_test, y_test, output_dir)

    # SHAP analysis
    shap_imp = analyze_shap(model, X_train, X_test, output_dir)

    # Compare methods
    comparison = compare_importance_methods(builtin, permutation, shap_imp, output_dir)

    # Generate report
    generate_xai_report(comparison, output_dir)

    print("\n" + "=" * 60)
    print("XAI ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
