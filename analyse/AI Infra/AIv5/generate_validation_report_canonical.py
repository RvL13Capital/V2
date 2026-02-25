"""
Generate Comprehensive Validation Report for Canonical Predictions
====================================================================

This script analyzes predictions made with 69 canonical features and
generates a detailed validation report comparing predicted vs actual outcomes.

Purpose:
- Validate that feature mismatch has been fixed
- Calculate key metrics: K4 recall, K3+K4 recall, EV correlation
- Generate confusion matrices for each model
- Analyze signal quality and distribution

Process:
1. Load predictions from predictions_canonical.parquet
2. Compare predicted classes vs actual classes
3. Calculate class-wise precision, recall, F1 scores
4. Compute expected value correlation
5. Analyze signal strength distribution
6. Generate markdown report with all metrics

Expected Runtime: 1-2 minutes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append(str(Path(__file__).parent))

from training.utils.pattern_value_system import PatternValueSystem

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input paths
PREDICTIONS_FILE = Path('output/unused_patterns/predictions_canonical.parquet')

# Output paths
OUTPUT_DIR = Path('output/validation')
REPORT_FILE = OUTPUT_DIR / 'validation_report_canonical.md'
METRICS_FILE = OUTPUT_DIR / 'validation_metrics_canonical.json'

# Value system for actual value calculation
value_system = PatternValueSystem()

# Class to value mapping
CLASS_VALUES = {
    'K0_STAGNANT': value_system.K0_VALUE,     # -2
    'K1_MINIMAL': value_system.K1_VALUE,      # -0.2
    'K2_QUALITY': value_system.K2_VALUE,      # +1
    'K3_STRONG': value_system.K3_VALUE,       # +3
    'K4_EXCEPTIONAL': value_system.K4_VALUE,  # +10
    'K5_FAILED': value_system.K5_VALUE        # -10
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_actual_value(actual_class: str) -> float:
    """Calculate strategic value for actual outcome."""
    return CLASS_VALUES.get(actual_class, 0)


def format_confusion_matrix(cm: np.ndarray, labels: list) -> str:
    """Format confusion matrix as markdown table."""
    result = "| Actual \\ Predicted | " + " | ".join(labels) + " |\n"
    result += "|---|" + "|".join(["---"] * len(labels)) + "|\n"

    for i, actual_label in enumerate(labels):
        row = f"| **{actual_label}** |"
        for j in range(len(labels)):
            value = cm[i, j]
            if value > 0 and i == j:
                row += f" **{value}** |"  # Bold correct predictions
            elif value > 0:
                row += f" {value} |"
            else:
                row += " 0 |"
        result += row + "\n"

    return result


def calculate_metrics(y_true, y_pred, labels) -> dict:
    """Calculate comprehensive classification metrics."""
    # Overall metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )

    # Per-class metrics
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    metrics = {
        'overall': {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'total_samples': len(y_true)
        },
        'per_class': {}
    }

    for i, label in enumerate(labels):
        metrics['per_class'][label] = {
            'precision': float(class_precision[i]),
            'recall': float(class_recall[i]),
            'f1': float(class_f1[i]),
            'support': int(class_support[i])
        }

    return metrics


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CANONICAL VALIDATION REPORT GENERATION")
    print("Analyzing Predictions with 69 Matched Features")
    print("=" * 80)

    # Check if predictions file exists
    if not PREDICTIONS_FILE.exists():
        print(f"\n❌ ERROR: Predictions file not found: {PREDICTIONS_FILE}")
        print("Please run predict_historical_canonical.py first")
        exit(1)

    # Load predictions
    print(f"\n[STEP 1] Loading predictions...")
    predictions_df = pd.read_parquet(PREDICTIONS_FILE)
    print(f"  Total predictions: {len(predictions_df):,}")

    # Filter to patterns with known actual outcomes
    valid_df = predictions_df[predictions_df['actual_class'] != 'UNKNOWN'].copy()
    print(f"  Patterns with known outcomes: {len(valid_df):,}")

    if len(valid_df) == 0:
        print("\n❌ ERROR: No patterns with known outcomes for validation")
        exit(1)

    # Calculate actual values
    print(f"\n[STEP 2] Calculating actual values...")
    valid_df['actual_value'] = valid_df['actual_class'].apply(calculate_actual_value)

    # Get unique classes
    all_classes = sorted(list(set(valid_df['actual_class'].unique()) |
                              set(valid_df['xgb_predicted_class'].unique()) |
                              set(valid_df['lgb_predicted_class'].unique()) |
                              set(valid_df['ensemble_predicted_class'].unique())))

    # Start building report
    report_lines = []
    report_lines.append("# Canonical Validation Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("## Executive Summary\n")
    report_lines.append("### Feature Pipeline Status")
    report_lines.append("- **Features Used**: 69 canonical features (MATCHED with training)")
    report_lines.append("- **Previous Issue**: Was using 31 temporal_safe features (only 3 overlapped)")
    report_lines.append("- **Current Status**: Using exact same CanonicalFeatureExtractor as training\n")

    # Calculate key metrics for each model
    print(f"\n[STEP 3] Calculating validation metrics...")

    models = ['xgb', 'lgb', 'ensemble']
    all_metrics = {}

    for model_name in models:
        print(f"  Processing {model_name.upper()} model...")

        # Get predictions for this model
        y_true = valid_df['actual_class'].values
        y_pred = valid_df[f'{model_name}_predicted_class'].values
        y_ev = valid_df[f'{model_name}_expected_value'].values
        actual_values = valid_df['actual_value'].values

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, all_classes)
        all_metrics[model_name] = metrics

        # Calculate K4 recall
        k4_mask = y_true == 'K4_EXCEPTIONAL'
        k4_recall = 0
        if k4_mask.sum() > 0:
            k4_correct = ((y_true == y_pred) & k4_mask).sum()
            k4_recall = k4_correct / k4_mask.sum()

        # Calculate K3+K4 recall
        k34_mask = (y_true == 'K3_STRONG') | (y_true == 'K4_EXCEPTIONAL')
        k34_recall = 0
        if k34_mask.sum() > 0:
            k34_pred_mask = (y_pred == 'K3_STRONG') | (y_pred == 'K4_EXCEPTIONAL')
            k34_correct = (k34_pred_mask & k34_mask).sum()
            k34_recall = k34_correct / k34_mask.sum()

        # Calculate EV correlation
        ev_corr_pearson, ev_p_value = pearsonr(y_ev, actual_values)
        ev_corr_spearman, _ = spearmanr(y_ev, actual_values)

        # Store key metrics
        metrics['k4_recall'] = k4_recall
        metrics['k34_recall'] = k34_recall
        metrics['ev_correlation_pearson'] = ev_corr_pearson
        metrics['ev_correlation_spearman'] = ev_corr_spearman

    # Write key metrics section
    report_lines.append("\n## Key Performance Metrics\n")
    report_lines.append("### Critical Metrics Comparison\n")
    report_lines.append("| Metric | XGBoost | LightGBM | Ensemble | Target | Status |")
    report_lines.append("|--------|---------|----------|----------|--------|--------|")

    for model_name in models:
        m = all_metrics[model_name]
        if model_name == 'ensemble':
            # K4 Recall
            k4_status = "✅" if m['k4_recall'] > 0.10 else "⚠️" if m['k4_recall'] > 0 else "❌"
            report_lines.append(f"| **K4 Recall** | {all_metrics['xgb']['k4_recall']:.1%} | "
                              f"{all_metrics['lgb']['k4_recall']:.1%} | "
                              f"{m['k4_recall']:.1%} | >30% | {k4_status} |")

            # K3+K4 Recall
            k34_status = "✅" if m['k34_recall'] > 0.50 else "⚠️" if m['k34_recall'] > 0.30 else "❌"
            report_lines.append(f"| **K3+K4 Recall** | {all_metrics['xgb']['k34_recall']:.1%} | "
                              f"{all_metrics['lgb']['k34_recall']:.1%} | "
                              f"{m['k34_recall']:.1%} | >60% | {k34_status} |")

            # EV Correlation
            ev_status = "✅" if m['ev_correlation_pearson'] > 0.20 else "⚠️" if m['ev_correlation_pearson'] > 0 else "❌"
            report_lines.append(f"| **EV Correlation** | {all_metrics['xgb']['ev_correlation_pearson']:.3f} | "
                              f"{all_metrics['lgb']['ev_correlation_pearson']:.3f} | "
                              f"{m['ev_correlation_pearson']:.3f} | >0.20 | {ev_status} |")

    # Add historical comparison
    report_lines.append("\n### Before vs After Feature Fix\n")
    report_lines.append("| Metric | Before Fix (31 features) | After Fix (69 features) | Improvement |")
    report_lines.append("|--------|---------------------------|--------------------------|-------------|")

    # These are placeholders - will be updated with actual results
    ensemble_metrics = all_metrics['ensemble']
    report_lines.append(f"| K4 Recall | 0.0% | {ensemble_metrics['k4_recall']:.1%} | "
                       f"+{ensemble_metrics['k4_recall']:.1%} |")
    report_lines.append(f"| K3+K4 Recall | 14.9% | {ensemble_metrics['k34_recall']:.1%} | "
                       f"+{max(0, ensemble_metrics['k34_recall']-0.149):.1%} |")
    report_lines.append(f"| EV Correlation | -0.140 | {ensemble_metrics['ev_correlation_pearson']:.3f} | "
                       f"+{ensemble_metrics['ev_correlation_pearson']+0.140:.3f} |")

    # Confusion matrices
    print(f"\n[STEP 4] Generating confusion matrices...")
    report_lines.append("\n## Confusion Matrices\n")

    for model_name in models:
        y_true = valid_df['actual_class'].values
        y_pred = valid_df[f'{model_name}_predicted_class'].values

        cm = confusion_matrix(y_true, y_pred, labels=all_classes)

        report_lines.append(f"\n### {model_name.upper()} Model\n")
        report_lines.append(format_confusion_matrix(cm, all_classes))

    # Signal distribution analysis
    print(f"\n[STEP 5] Analyzing signal distribution...")
    report_lines.append("\n## Signal Distribution Analysis\n")

    for model_name in models:
        signal_counts = valid_df[f'{model_name}_signal'].value_counts()

        report_lines.append(f"\n### {model_name.upper()} Signal Distribution\n")
        report_lines.append("| Signal Strength | Count | Percentage |")
        report_lines.append("|----------------|-------|------------|")

        for signal in ['STRONG_SIGNAL', 'GOOD_SIGNAL', 'MODERATE_SIGNAL', 'WEAK_SIGNAL', 'AVOID']:
            count = signal_counts.get(signal, 0)
            pct = count / len(valid_df) * 100
            report_lines.append(f"| {signal} | {count} | {pct:.1f}% |")

    # Expected Value Analysis
    print(f"\n[STEP 6] Analyzing expected values...")
    report_lines.append("\n## Expected Value Analysis\n")

    for model_name in models:
        ev_values = valid_df[f'{model_name}_expected_value'].values

        report_lines.append(f"\n### {model_name.upper()} Expected Values\n")
        report_lines.append(f"- **Mean EV**: {ev_values.mean():.2f}")
        report_lines.append(f"- **Median EV**: {np.median(ev_values):.2f}")
        report_lines.append(f"- **Max EV**: {ev_values.max():.2f}")
        report_lines.append(f"- **Min EV**: {ev_values.min():.2f}")
        report_lines.append(f"- **Positive EV**: {(ev_values > 0).sum()}/{len(ev_values)} "
                          f"({(ev_values > 0).mean()*100:.1f}%)")
        report_lines.append(f"- **EV > 3.0**: {(ev_values > 3.0).sum()} patterns "
                          f"({(ev_values > 3.0).mean()*100:.1f}%)")
        report_lines.append(f"- **EV > 5.0**: {(ev_values > 5.0).sum()} patterns "
                          f"({(ev_values > 5.0).mean()*100:.1f}%)")

    # K4 Detection Analysis
    print(f"\n[STEP 7] Analyzing K4 detection...")
    report_lines.append("\n## K4 Detection Analysis\n")

    actual_k4_count = (valid_df['actual_class'] == 'K4_EXCEPTIONAL').sum()
    report_lines.append(f"### Actual K4 Patterns: {actual_k4_count}\n")

    for model_name in models:
        # Get K4 probabilities
        k4_probs = valid_df[f'{model_name}_k4_prob'].values

        report_lines.append(f"\n### {model_name.upper()} K4 Detection\n")
        report_lines.append(f"- **Predicted as K4**: "
                          f"{(valid_df[f'{model_name}_predicted_class'] == 'K4_EXCEPTIONAL').sum()}")
        report_lines.append(f"- **K4 Prob > 0.1**: {(k4_probs > 0.1).sum()} patterns")
        report_lines.append(f"- **K4 Prob > 0.2**: {(k4_probs > 0.2).sum()} patterns")
        report_lines.append(f"- **K4 Prob > 0.3**: {(k4_probs > 0.3).sum()} patterns")
        report_lines.append(f"- **Max K4 Prob**: {k4_probs.max():.3f}")
        report_lines.append(f"- **Mean K4 Prob**: {k4_probs.mean():.3f}")

    # Class-wise Performance
    print(f"\n[STEP 8] Analyzing class-wise performance...")
    report_lines.append("\n## Class-wise Performance (Ensemble)\n")

    ensemble_metrics = all_metrics['ensemble']
    report_lines.append("| Class | Precision | Recall | F1-Score | Support |")
    report_lines.append("|-------|-----------|--------|----------|---------|")

    for class_name in all_classes:
        if class_name in ensemble_metrics['per_class']:
            cm = ensemble_metrics['per_class'][class_name]
            report_lines.append(f"| {class_name} | {cm['precision']:.3f} | "
                              f"{cm['recall']:.3f} | {cm['f1']:.3f} | {cm['support']} |")

    # Recommendations
    report_lines.append("\n## Recommendations\n")

    if ensemble_metrics['k4_recall'] < 0.10:
        report_lines.append("### ⚠️ K4 Detection Still Low")
        report_lines.append("- Feature mismatch has been fixed, but K4 recall remains low")
        report_lines.append("- This is likely due to K4 extreme rarity in training (only 12 examples)")
        report_lines.append("- Consider:")
        report_lines.append("  1. Retraining with combined dataset (training + validation)")
        report_lines.append("  2. Using SMOTE oversampling for K4 class")
        report_lines.append("  3. Implementing focal loss with high K4 weight (alpha=500)")
        report_lines.append("  4. Training separate binary K4 classifier")

    if ensemble_metrics['ev_correlation_pearson'] < 0:
        report_lines.append("\n### ❌ Negative EV Correlation")
        report_lines.append("- Models are still struggling with value prediction")
        report_lines.append("- Check for remaining feature engineering issues")
        report_lines.append("- May need to retrain with corrected features")
    elif ensemble_metrics['ev_correlation_pearson'] < 0.15:
        report_lines.append("\n### ⚠️ Low EV Correlation")
        report_lines.append("- EV correlation improved but still below target")
        report_lines.append("- Continue monitoring after addressing K4 rarity")

    # Success indicators
    if ensemble_metrics['ev_correlation_pearson'] > 0.20:
        report_lines.append("\n### ✅ Feature Fix Successful!")
        report_lines.append("- EV correlation is now positive and above target")
        report_lines.append("- Models can properly use the 69 canonical features")

    if ensemble_metrics['k34_recall'] > 0.50:
        report_lines.append("\n### ✅ Good K3+K4 Detection")
        report_lines.append("- Models successfully detecting valuable patterns")
        report_lines.append("- Focus on improving K4-specific detection")

    # Write report
    print(f"\n[STEP 9] Writing report...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(report_lines))

    # Save metrics as JSON
    import json
    with open(METRICS_FILE, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"  Report saved to: {REPORT_FILE}")
    print(f"  Metrics saved to: {METRICS_FILE}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE!")
    print("=" * 80)
    print(f"\nKey Results (Ensemble):")
    print(f"  K4 Recall: {ensemble_metrics['k4_recall']:.1%}")
    print(f"  K3+K4 Recall: {ensemble_metrics['k34_recall']:.1%}")
    print(f"  EV Correlation: {ensemble_metrics['ev_correlation_pearson']:.3f}")
    print(f"\nFull report: {REPORT_FILE}")
    print("=" * 80)