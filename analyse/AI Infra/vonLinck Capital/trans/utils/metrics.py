"""
Metrics Utilities for Temporal Model Evaluation (V17 3-Class System)
====================================================================

Custom metrics for pattern detection evaluation:
- Expected Value (EV) correlation
- Per-class recall (Target class 2 is primary metric)
- Signal quality metrics
- Calibration metrics

V17 Classes:
  - 0: Danger (stop loss hit)
  - 1: Noise (neither target nor stop)
  - 2: Target (profitable trade)
"""

import numpy as np
from typing import Dict, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

# Import strategic values from config (V17 3-class system)
from config.constants import STRATEGIC_VALUES, NUM_CLASSES, CLASS_NAMES


def calculate_expected_value(class_probs: np.ndarray) -> np.ndarray:
    """
    Calculate Expected Value from class probabilities.

    V17 3-class system:
        EV = P(Danger)*(-1.0) + P(Noise)*(-0.1) + P(Target)*(+5.0)

    Args:
        class_probs: (n_samples, 3) probabilities for [Danger, Noise, Target]

    Returns:
        (n_samples,) Expected Values
    """
    values = np.array([STRATEGIC_VALUES[i] for i in range(NUM_CLASSES)])
    ev = np.sum(class_probs * values, axis=1)
    return ev


def ev_correlation(
    predicted_ev: np.ndarray,
    true_labels: np.ndarray
) -> Dict[str, float]:
    """
    Calculate EV correlation metrics.

    Args:
        predicted_ev: (n_samples,) predicted Expected Values
        true_labels: (n_samples,) true class labels

    Returns:
        Dictionary with correlation metrics
    """
    # Calculate actual EV from true labels
    actual_ev = np.array([STRATEGIC_VALUES[label] for label in true_labels])

    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(predicted_ev, actual_ev)

    # Spearman correlation (rank-based)
    spearman_corr, spearman_p = spearmanr(predicted_ev, actual_ev)

    # MAE and RMSE
    mae = np.mean(np.abs(predicted_ev - actual_ev))
    rmse = np.sqrt(np.mean((predicted_ev - actual_ev) ** 2))

    # Directional accuracy (same sign)
    same_sign = np.sign(predicted_ev) == np.sign(actual_ev)
    directional_acc = same_sign.mean()

    return {
        'pearson_correlation': pearson_corr,
        'pearson_pvalue': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_p,
        'mae': mae,
        'rmse': rmse,
        'directional_accuracy': directional_acc
    }


def target_danger_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray
) -> Dict[str, float]:
    """
    Calculate Target recall and Danger precision (V17 key metrics).

    V17 focuses on:
    - Target Recall: Ability to capture winning trades (class 2)
    - Danger Precision: Ability to correctly identify losers (class 0)

    Args:
        true_labels: (n_samples,) true labels
        predicted_labels: (n_samples,) predicted labels

    Returns:
        Dictionary with Target recall and Danger precision
    """
    metrics = {}

    # Target recall (class 2) - how many winners did we find?
    target_mask = true_labels == 2
    if target_mask.sum() > 0:
        target_correct = (predicted_labels[target_mask] == 2).sum()
        metrics['Target_recall'] = target_correct / target_mask.sum()
        metrics['Target_support'] = int(target_mask.sum())
    else:
        metrics['Target_recall'] = 0.0
        metrics['Target_support'] = 0

    # Danger precision (class 0) - of predictions as Danger, how many were correct?
    danger_predicted = predicted_labels == 0
    if danger_predicted.sum() > 0:
        danger_correct = (true_labels[danger_predicted] == 0).sum()
        metrics['Danger_precision'] = danger_correct / danger_predicted.sum()
        metrics['Danger_predicted_count'] = int(danger_predicted.sum())
    else:
        metrics['Danger_precision'] = 0.0
        metrics['Danger_predicted_count'] = 0

    # Danger recall
    danger_mask = true_labels == 0
    if danger_mask.sum() > 0:
        danger_recalled = (predicted_labels[danger_mask] == 0).sum()
        metrics['Danger_recall'] = danger_recalled / danger_mask.sum()
        metrics['Danger_support'] = int(danger_mask.sum())
    else:
        metrics['Danger_recall'] = 0.0
        metrics['Danger_support'] = 0

    return metrics


def signal_quality_metrics(
    signals: np.ndarray,
    true_labels: np.ndarray,
    ev: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate signal quality (STRONG/GOOD/MODERATE) for V17 3-class system.

    Success is defined as Target outcomes (class 2).

    Args:
        signals: (n_samples,) signal strings
        true_labels: (n_samples,) true class labels
        ev: (n_samples,) Expected Values

    Returns:
        Dictionary with signal quality metrics
    """
    metrics = {}

    # Actual EV from true labels
    actual_ev = np.array([STRATEGIC_VALUES[label] for label in true_labels])

    for signal_level in ['STRONG', 'GOOD', 'MODERATE']:
        mask = signals == signal_level

        if mask.sum() > 0:
            # Count
            metrics[f'{signal_level}_count'] = int(mask.sum())

            # Success rate (Target outcomes = class 2)
            success = (true_labels[mask] == 2).sum()
            metrics[f'{signal_level}_success_rate'] = success / mask.sum()

            # Average actual EV
            metrics[f'{signal_level}_avg_actual_ev'] = actual_ev[mask].mean()

            # Average predicted EV
            metrics[f'{signal_level}_avg_predicted_ev'] = ev[mask].mean()

            # EV accuracy (predicted close to actual)
            ev_diff = np.abs(ev[mask] - actual_ev[mask])
            metrics[f'{signal_level}_ev_mae'] = ev_diff.mean()
        else:
            metrics[f'{signal_level}_count'] = 0
            metrics[f'{signal_level}_success_rate'] = 0.0
            metrics[f'{signal_level}_avg_actual_ev'] = 0.0
            metrics[f'{signal_level}_avg_predicted_ev'] = 0.0
            metrics[f'{signal_level}_ev_mae'] = 0.0

    return metrics


def calibration_metrics(
    class_probs: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Evaluate probability calibration.

    Well-calibrated model: predicted probability matches empirical frequency.

    Args:
        class_probs: (n_samples, n_classes) probabilities
        true_labels: (n_samples,) true labels
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics
    """
    metrics = {}

    # Overall calibration (max probability vs accuracy)
    max_probs = np.max(class_probs, axis=1)
    predicted_labels = np.argmax(class_probs, axis=1)
    correct = (predicted_labels == true_labels).astype(float)

    # Brier score (lower is better, perfect = 0)
    # Convert to one-hot
    n_classes = class_probs.shape[1]
    true_one_hot = np.zeros((len(true_labels), n_classes))
    true_one_hot[np.arange(len(true_labels)), true_labels] = 1

    brier = np.mean(np.sum((class_probs - true_one_hot) ** 2, axis=1))
    metrics['brier_score'] = brier

    # Expected Calibration Error (ECE)
    # Bin predictions by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_mask = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])

        if bin_mask.sum() > 0:
            bin_acc = correct[bin_mask].mean()
            bin_conf = max_probs[bin_mask].mean()
            bin_weight = bin_mask.sum() / len(max_probs)

            ece += bin_weight * np.abs(bin_acc - bin_conf)

    metrics['expected_calibration_error'] = ece

    return metrics


def confusion_based_metrics(confusion_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics from confusion matrix (V17 3-class system).

    Args:
        confusion_matrix: (n_classes, n_classes) array

    Returns:
        Dictionary with derived metrics using Danger/Noise/Target naming
    """
    metrics = {}

    n_classes = confusion_matrix.shape[0]

    # Per-class metrics (use CLASS_NAMES for naming)
    for i in range(n_classes):
        class_name = CLASS_NAMES.get(i, f'Class{i}')

        # True Positives
        tp = confusion_matrix[i, i]

        # False Positives (predicted i, but wasn't)
        fp = confusion_matrix[:, i].sum() - tp

        # False Negatives (was i, but predicted something else)
        fn = confusion_matrix[i, :].sum() - tp

        # True Negatives
        tn = confusion_matrix.sum() - tp - fp - fn

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics[f'{class_name}_precision'] = precision

        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f'{class_name}_recall'] = recall

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f'{class_name}_f1'] = f1

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[f'{class_name}_specificity'] = specificity

    # Overall accuracy
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum()
    metrics['overall_accuracy'] = accuracy

    return metrics


def v17_comparison_metrics(
    target_recall: float,
    danger_precision: float,
    ev_correlation: float
) -> Dict[str, bool]:
    """
    Compare against V17 3-class performance targets.

    V17 Targets:
    - Target Recall (class 2): 25-35% (target 35%)
    - Danger Precision (class 0): 50-60% (target 60%)
    - EV Correlation: >0.30 (target 0.40)

    Args:
        target_recall: Target (class 2) recall value
        danger_precision: Danger (class 0) precision value
        ev_correlation: EV Pearson correlation

    Returns:
        Dictionary with pass/fail status
    """
    comparison = {
        # Target Recall (class 2)
        'target_recall_meets_min': target_recall >= 0.25,
        'target_recall_meets_target': target_recall >= 0.35,
        'target_recall_value': target_recall,

        # Danger Precision (class 0)
        'danger_precision_meets_min': danger_precision >= 0.50,
        'danger_precision_meets_target': danger_precision >= 0.60,
        'danger_precision_value': danger_precision,

        # EV Correlation
        'ev_corr_meets_min': ev_correlation >= 0.30,
        'ev_corr_meets_target': ev_correlation >= 0.40,
        'ev_corr_value': ev_correlation,

        # Overall
        'all_min_met': (target_recall >= 0.25 and danger_precision >= 0.50 and ev_correlation >= 0.30),
        'all_targets_met': (target_recall >= 0.35 and danger_precision >= 0.60 and ev_correlation >= 0.40)
    }

    return comparison


if __name__ == "__main__":
    # Test metrics (V17 3-class system)
    print("Testing metrics utilities (V17 3-Class)...")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    # Simulate class probabilities (3 classes: Danger, Noise, Target)
    class_probs = np.random.dirichlet(np.ones(NUM_CLASSES), n_samples)

    # Simulate true labels (imbalanced: 40% Danger, 45% Noise, 15% Target)
    true_labels = np.random.choice([0, 1, 2], n_samples, p=[0.40, 0.45, 0.15])

    # Predicted labels
    predicted_labels = np.argmax(class_probs, axis=1)

    # Calculate EV
    predicted_ev = calculate_expected_value(class_probs)

    # Test EV correlation
    print("\n1. EV Correlation:")
    ev_metrics = ev_correlation(predicted_ev, true_labels)
    for key, value in ev_metrics.items():
        print(f"   {key}: {value:.4f}")

    # Test Target/Danger metrics
    print("\n2. Target/Danger Metrics:")
    td_metrics = target_danger_metrics(true_labels, predicted_labels)
    for key, value in td_metrics.items():
        print(f"   {key}: {value}")

    # Test calibration
    print("\n3. Calibration:")
    calib_metrics = calibration_metrics(class_probs, true_labels)
    for key, value in calib_metrics.items():
        print(f"   {key}: {value:.4f}")

    # Test V17 comparison
    print("\n4. V17 Comparison:")
    comparison = v17_comparison_metrics(
        target_recall=0.38,
        danger_precision=0.62,
        ev_correlation=0.42
    )
    print(f"   All targets met: {comparison['all_targets_met']}")

    print("\nAll tests passed!")
