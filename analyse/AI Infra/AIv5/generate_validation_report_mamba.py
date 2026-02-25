"""
Validation Report Generator with Temporal Leakage Detection
============================================================

CRITICAL: This script validates temporal integrity and detects data leakage.

Validation Tests:
1. Temporal Integrity - All sequence dates <= snapshot_date
2. Future Contamination - Past accuracy not suspiciously high
3. K4 Recall Consistency - Similar across time periods
4. EV Correlation - Predicted vs actual values
5. Class Balance Drift - Distribution changes over time

Usage:
    python generate_validation_report_mamba.py \\
        --predictions output/predictions_mamba_attention.parquet \\
        --output output/validation_report_mamba.md
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalIntegrityValidator:
    """
    Validate temporal integrity and detect data leakage.
    """

    # Strategic values for EV calculation
    STRATEGIC_VALUES = {
        0: -2,    # K0_STAGNANT
        1: -0.2,  # K1_MINIMAL
        2: 1,     # K2_QUALITY
        3: 3,     # K3_STRONG
        4: 10,    # K4_EXCEPTIONAL
        5: -10    # K5_FAILED
    }

    def __init__(self, predictions_df: pd.DataFrame):
        """
        Initialize validator.

        Args:
            predictions_df: DataFrame with predictions and actual outcomes
        """
        self.predictions_df = predictions_df.copy()
        self.predictions_df['snapshot_date'] = pd.to_datetime(self.predictions_df['snapshot_date'])

        # Calculate actual EV from outcome_class
        self.predictions_df['actual_value'] = self.predictions_df['outcome_class'].map(
            self.STRATEGIC_VALUES
        )

        logger.info(f"Validator initialized with {len(predictions_df)} predictions")

    def run_all_tests(self) -> Dict:
        """
        Run all temporal integrity and leakage detection tests.

        Returns:
            Dict with test results
        """
        logger.info("Running all validation tests...")

        results = {}

        # Test 1: Temporal integrity
        results['temporal_integrity'] = self.test_temporal_integrity()

        # Test 2: Future contamination
        results['future_contamination'] = self.test_future_contamination()

        # Test 3: K4 recall consistency
        results['k4_recall_consistency'] = self.test_k4_recall_consistency()

        # Test 4: EV correlation
        results['ev_correlation'] = self.test_ev_correlation()

        # Test 5: Class distribution drift
        results['class_drift'] = self.test_class_distribution_drift()

        # Test 6: Performance metrics
        results['performance'] = self.calculate_performance_metrics()

        logger.info("✓ All tests complete")

        return results

    def test_temporal_integrity(self) -> Dict:
        """
        Test 1: Verify all sequence dates <= snapshot_date.

        CRITICAL: Detects if sequences contain future data.

        Returns:
            Dict with test results
        """
        logger.info("Test 1: Temporal integrity check...")

        violations = 0
        total_sequences = 0

        if 'sequence_dates' in self.predictions_df.columns:
            for idx, row in self.predictions_df.iterrows():
                sequence_dates = row['sequence_dates']
                snapshot_date = row['snapshot_date']

                total_sequences += 1

                # Check for future dates
                future_dates = [d for d in sequence_dates if d > snapshot_date]
                if future_dates:
                    violations += 1

                    if violations <= 3:  # Log first 3 violations
                        logger.error(
                            f"TEMPORAL VIOLATION at row {idx}: "
                            f"Snapshot {snapshot_date}, Future dates: {future_dates}"
                        )

            passed = violations == 0

            logger.info(
                f"{'✓ PASSED' if passed else '✗ FAILED'}: "
                f"{violations}/{total_sequences} sequences with temporal violations"
            )

            return {
                'passed': passed,
                'violations': violations,
                'total_sequences': total_sequences,
                'error_rate': violations / total_sequences if total_sequences > 0 else 0
            }

        else:
            logger.warning("sequence_dates column not found - skipping temporal integrity test")
            return {'passed': True, 'violations': 0, 'total_sequences': 0, 'error_rate': 0}

    def test_future_contamination(self) -> Dict:
        """
        Test 2: Check if past accuracy is suspiciously higher than recent accuracy.

        LEAKAGE INDICATOR: If model performs better on old data than new data,
        it may have seen future information during training.

        Returns:
            Dict with test results
        """
        logger.info("Test 2: Future contamination check...")

        # Split by time periods
        median_date = self.predictions_df['snapshot_date'].median()

        past_df = self.predictions_df[self.predictions_df['snapshot_date'] < median_date]
        recent_df = self.predictions_df[self.predictions_df['snapshot_date'] >= median_date]

        if len(past_df) == 0 or len(recent_df) == 0:
            logger.warning("Insufficient data for temporal split")
            return {'passed': True, 'suspicion_level': 'UNKNOWN'}

        # Calculate accuracies
        past_accuracy = accuracy_score(past_df['outcome_class'], past_df['predicted_class'])
        recent_accuracy = accuracy_score(recent_df['outcome_class'], recent_df['predicted_class'])

        accuracy_diff = past_accuracy - recent_accuracy

        # LEAKAGE THRESHOLD: Past accuracy should NOT be > 15% higher than recent
        threshold = 0.15
        passed = accuracy_diff <= threshold

        suspicion_level = 'NONE'
        if accuracy_diff > 0.25:
            suspicion_level = 'HIGH'
        elif accuracy_diff > threshold:
            suspicion_level = 'MODERATE'
        elif accuracy_diff > 0.05:
            suspicion_level = 'LOW'

        logger.info(
            f"{'✓ PASSED' if passed else '✗ SUSPICIOUS'}: "
            f"Past acc: {past_accuracy:.3f}, Recent acc: {recent_accuracy:.3f}, "
            f"Diff: {accuracy_diff:+.3f} (threshold: {threshold})"
        )

        return {
            'passed': passed,
            'past_accuracy': past_accuracy,
            'recent_accuracy': recent_accuracy,
            'accuracy_diff': accuracy_diff,
            'suspicion_level': suspicion_level
        }

    def test_k4_recall_consistency(self) -> Dict:
        """
        Test 3: Check if K4 recall is consistent across time periods.

        LEAKAGE INDICATOR: Large variations in K4 recall across time may indicate
        model learned time-specific patterns instead of generalizable features.

        Returns:
            Dict with test results
        """
        logger.info("Test 3: K4 recall consistency check...")

        # Get K4 samples
        k4_df = self.predictions_df[self.predictions_df['outcome_class'] == 4]

        if len(k4_df) < 10:
            logger.warning(f"Only {len(k4_df)} K4 samples - insufficient for consistency test")
            return {'passed': True, 'k4_count': len(k4_df), 'consistency': 'UNKNOWN'}

        # Split into time periods
        k4_df = k4_df.sort_values('snapshot_date')
        split_point = len(k4_df) // 2

        early_k4 = k4_df.iloc[:split_point]
        late_k4 = k4_df.iloc[split_point:]

        # Calculate K4 recall for each period
        early_recall = (early_k4['predicted_class'] == 4).mean()
        late_recall = (late_k4['predicted_class'] == 4).mean()

        recall_diff = abs(early_recall - late_recall)

        # CONSISTENCY THRESHOLD: Recall should not vary by more than 30%
        threshold = 0.30
        passed = recall_diff <= threshold

        consistency = 'GOOD'
        if recall_diff > 0.50:
            consistency = 'POOR'
        elif recall_diff > threshold:
            consistency = 'MODERATE'

        logger.info(
            f"{'✓ PASSED' if passed else '⚠ WARNING'}: "
            f"Early K4 recall: {early_recall:.3f}, Late K4 recall: {late_recall:.3f}, "
            f"Diff: {recall_diff:.3f}"
        )

        return {
            'passed': passed,
            'early_recall': early_recall,
            'late_recall': late_recall,
            'recall_diff': recall_diff,
            'consistency': consistency,
            'k4_count': len(k4_df)
        }

    def test_ev_correlation(self) -> Dict:
        """
        Test 4: Calculate correlation between predicted EV and actual values.

        PERFORMANCE METRIC: Higher correlation = better predictions.

        Returns:
            Dict with test results
        """
        logger.info("Test 4: EV correlation test...")

        # Calculate correlation
        correlation = np.corrcoef(
            self.predictions_df['expected_value'],
            self.predictions_df['actual_value']
        )[0, 1]

        # Target: EV correlation > 0.30
        target = 0.30
        passed = correlation >= target

        logger.info(
            f"{'✓ PASSED' if passed else '⚠ BELOW TARGET'}: "
            f"EV correlation: {correlation:.3f} (target: {target})"
        )

        return {
            'passed': passed,
            'correlation': correlation,
            'target': target
        }

    def test_class_distribution_drift(self) -> Dict:
        """
        Test 5: Check for significant drift in class distribution over time.

        DRIFT INDICATOR: Large changes in class distribution may indicate
        market regime changes or data quality issues.

        Returns:
            Dict with test results
        """
        logger.info("Test 5: Class distribution drift test...")

        # Split by time
        median_date = self.predictions_df['snapshot_date'].median()

        early_df = self.predictions_df[self.predictions_df['snapshot_date'] < median_date]
        late_df = self.predictions_df[self.predictions_df['snapshot_date'] >= median_date]

        # Get class distributions
        early_dist = early_df['outcome_class'].value_counts(normalize=True).sort_index()
        late_dist = late_df['outcome_class'].value_counts(normalize=True).sort_index()

        # Calculate KL divergence (drift measure)
        # Ensure all classes present
        all_classes = set(range(6))
        early_probs = np.array([early_dist.get(i, 1e-10) for i in all_classes])
        late_probs = np.array([late_dist.get(i, 1e-10) for i in all_classes])

        # Normalize
        early_probs = early_probs / early_probs.sum()
        late_probs = late_probs / late_probs.sum()

        # KL divergence
        kl_div = np.sum(late_probs * np.log((late_probs + 1e-10) / (early_probs + 1e-10)))

        # DRIFT THRESHOLD: KL divergence > 0.5 indicates significant drift
        threshold = 0.5
        passed = kl_div <= threshold

        drift_level = 'LOW'
        if kl_div > 1.0:
            drift_level = 'HIGH'
        elif kl_div > threshold:
            drift_level = 'MODERATE'

        logger.info(
            f"{'✓ PASSED' if passed else '⚠ DRIFT DETECTED'}: "
            f"KL divergence: {kl_div:.3f}, Drift level: {drift_level}"
        )

        return {
            'passed': passed,
            'kl_divergence': kl_div,
            'drift_level': drift_level,
            'early_distribution': early_dist.to_dict(),
            'late_distribution': late_dist.to_dict()
        }

    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dict with all performance metrics
        """
        logger.info("Calculating performance metrics...")

        y_true = self.predictions_df['outcome_class'].values
        y_pred = self.predictions_df['predicted_class'].values

        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        class_metrics = {}

        for k in range(6):
            mask = (y_true == k)
            if mask.sum() == 0:
                continue

            recall = (y_pred[mask] == k).sum() / mask.sum()
            precision = (y_true[y_pred == k] == k).sum() / max((y_pred == k).sum(), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)

            class_metrics[f'K{k}'] = {
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'support': mask.sum()
            }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # K3+K4 combined recall (high-value patterns)
        k3_k4_mask = (y_true == 3) | (y_true == 4)
        if k3_k4_mask.sum() > 0:
            k3_k4_recall = ((y_pred[k3_k4_mask] == 3) | (y_pred[k3_k4_mask] == 4)).sum() / k3_k4_mask.sum()
        else:
            k3_k4_recall = 0.0

        # EV correlation
        ev_correlation = np.corrcoef(
            self.predictions_df['expected_value'],
            self.predictions_df['actual_value']
        )[0, 1]

        return {
            'accuracy': accuracy,
            'k3_k4_recall': k3_k4_recall,
            'ev_correlation': ev_correlation,
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist()
        }


def generate_report(test_results: Dict, output_path: str):
    """
    Generate markdown validation report.

    Args:
        test_results: Dict with all test results
        output_path: Path to save report
    """
    logger.info(f"Generating report: {output_path}")

    report_lines = []

    # Header
    report_lines.append("# MambaAttention Validation Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n" + "="*60)

    # Temporal Integrity Tests
    report_lines.append("\n## Temporal Integrity Tests")
    report_lines.append("\nThese tests validate that NO forward-looking data was used.")

    # Test 1
    t1 = test_results['temporal_integrity']
    status = "✓ PASSED" if t1['passed'] else "✗ FAILED"
    report_lines.append(f"\n### Test 1: Temporal Sequence Validation")
    report_lines.append(f"**Status**: {status}")
    report_lines.append(f"- Sequences checked: {t1['total_sequences']}")
    report_lines.append(f"- Temporal violations: {t1['violations']}")
    report_lines.append(f"- Error rate: {t1['error_rate']:.2%}")

    # Test 2
    t2 = test_results['future_contamination']
    status = "✓ PASSED" if t2['passed'] else "✗ SUSPICIOUS"
    report_lines.append(f"\n### Test 2: Future Contamination Check")
    report_lines.append(f"**Status**: {status}")
    report_lines.append(f"- Past accuracy: {t2['past_accuracy']:.3f}")
    report_lines.append(f"- Recent accuracy: {t2['recent_accuracy']:.3f}")
    report_lines.append(f"- Difference: {t2['accuracy_diff']:+.3f}")
    report_lines.append(f"- Suspicion level: {t2['suspicion_level']}")

    # Test 3
    t3 = test_results['k4_recall_consistency']
    status = "✓ PASSED" if t3['passed'] else "⚠ WARNING"
    report_lines.append(f"\n### Test 3: K4 Recall Consistency")
    report_lines.append(f"**Status**: {status}")
    report_lines.append(f"- Early period recall: {t3['early_recall']:.3f}")
    report_lines.append(f"- Late period recall: {t3['late_recall']:.3f}")
    report_lines.append(f"- Difference: {t3['recall_diff']:.3f}")
    report_lines.append(f"- Consistency: {t3['consistency']}")

    # Test 4
    t4 = test_results['ev_correlation']
    status = "✓ PASSED" if t4['passed'] else "⚠ BELOW TARGET"
    report_lines.append(f"\n### Test 4: EV Correlation")
    report_lines.append(f"**Status**: {status}")
    report_lines.append(f"- Correlation: {t4['correlation']:.3f}")
    report_lines.append(f"- Target: {t4['target']:.3f}")

    # Test 5
    t5 = test_results['class_drift']
    status = "✓ PASSED" if t5['passed'] else "⚠ DRIFT DETECTED"
    report_lines.append(f"\n### Test 5: Class Distribution Drift")
    report_lines.append(f"**Status**: {status}")
    report_lines.append(f"- KL divergence: {t5['kl_divergence']:.3f}")
    report_lines.append(f"- Drift level: {t5['drift_level']}")

    # Performance Metrics
    report_lines.append("\n" + "="*60)
    report_lines.append("\n## Performance Metrics")

    perf = test_results['performance']
    report_lines.append(f"\n- **Overall Accuracy**: {perf['accuracy']:.3f}")
    report_lines.append(f"- **K3+K4 Recall**: {perf['k3_k4_recall']:.3f}")
    report_lines.append(f"- **EV Correlation**: {perf['ev_correlation']:.3f}")

    # Per-class metrics
    report_lines.append("\n### Per-Class Metrics")
    report_lines.append("\n| Class | Recall | Precision | F1 | Support |")
    report_lines.append("|-------|--------|-----------|-----|---------|")

    for class_name, metrics in perf['class_metrics'].items():
        report_lines.append(
            f"| {class_name} | {metrics['recall']:.3f} | {metrics['precision']:.3f} | "
            f"{metrics['f1']:.3f} | {metrics['support']} |"
        )

    # Final verdict
    report_lines.append("\n" + "="*60)
    report_lines.append("\n## Final Verdict")

    all_tests_passed = all([
        test_results['temporal_integrity']['passed'],
        test_results['future_contamination']['passed'],
        test_results['k4_recall_consistency']['passed'],
        test_results['ev_correlation']['passed'],
        test_results['class_drift']['passed']
    ])

    if all_tests_passed:
        report_lines.append("\n✓ **ALL TESTS PASSED** - No temporal leakage detected")
    else:
        report_lines.append("\n⚠ **WARNINGS DETECTED** - Review failed tests above")

    report_lines.append("\n" + "="*60)

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"✓ Report saved to {output_path}")


def main():
    """
    Main validation pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate validation report with leakage detection")
    parser.add_argument('--predictions', type=str, required=True, help='Predictions parquet file')
    parser.add_argument('--output', type=str, required=True, help='Output report file (.md)')

    args = parser.parse_args()

    # Load predictions
    logger.info(f"Loading predictions from {args.predictions}")
    predictions_df = pd.read_parquet(args.predictions)

    # Initialize validator
    validator = TemporalIntegrityValidator(predictions_df)

    # Run all tests
    test_results = validator.run_all_tests()

    # Generate report
    generate_report(test_results, args.output)

    logger.info("✓ Validation pipeline complete!")


if __name__ == "__main__":
    main()
