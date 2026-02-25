#!/usr/bin/env python3
"""
Prediction Drift Monitor
========================

Weekly PSI check between baseline predictions and current predictions.
Integrates with existing ml/drift_monitor.py infrastructure.

Usage:
    # Check drift against baseline
    python scripts/monitor_drift.py --baseline output/baseline_predictions.parquet \
        --current output/latest_predictions.parquet

    # Generate new baseline from training data
    python scripts/monitor_drift.py --generate-baseline \
        --training-context output/sequences/eu/context_train.npy \
        --output output/drift_baseline.json

    # Quick check (uses default paths)
    python scripts/monitor_drift.py --quick-check

Exit codes:
    0: No significant drift
    1: Warning-level drift (monitor)
    2: Critical drift (retrain recommended)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import json
import logging

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.drift_monitor import DriftMonitor, PSI_THRESHOLDS, DriftReport
from utils.logging_config import setup_pipeline_logging

logger = setup_pipeline_logging('monitor_drift')


def check_weekly_drift(
    baseline_path: str,
    current_predictions_path: str,
    feature_cols: list = None,
    output_report_path: str = None
) -> tuple:
    """
    Check PSI between baseline and current predictions.

    Args:
        baseline_path: Path to baseline drift reference (JSON from fit_reference)
        current_predictions_path: Path to current predictions (Parquet)
        feature_cols: Columns to check for drift (default: prob columns + EV)
        output_report_path: Optional path to save report JSON

    Returns:
        (passed: bool, report: DriftReport)
    """
    logger.info("=" * 60)
    logger.info("PREDICTION DRIFT CHECK")
    logger.info("=" * 60)

    # Load baseline
    if not Path(baseline_path).exists():
        logger.error(f"Baseline not found: {baseline_path}")
        logger.info("Generate baseline with: --generate-baseline")
        return False, None

    logger.info(f"Loading baseline: {baseline_path}")
    monitor = DriftMonitor.load_reference(Path(baseline_path))

    # Load current predictions
    logger.info(f"Loading predictions: {current_predictions_path}")
    current_df = pd.read_parquet(current_predictions_path)
    logger.info(f"  Samples: {len(current_df):,}")

    # Default feature columns for prediction drift
    if feature_cols is None:
        feature_cols = []
        # Probability columns
        for col in ['danger_prob', 'noise_prob', 'target_prob']:
            if col in current_df.columns:
                feature_cols.append(col)
        # Expected Value
        if 'expected_value' in current_df.columns:
            feature_cols.append('expected_value')
        # Predicted class distribution (as numeric)
        if 'predicted_class' in current_df.columns:
            feature_cols.append('predicted_class')

    if not feature_cols:
        logger.error("No feature columns found in predictions!")
        return False, None

    logger.info(f"Checking columns: {feature_cols}")

    # Filter to available columns
    available_cols = [c for c in feature_cols if c in current_df.columns]
    current_subset = current_df[available_cols].copy()

    # Analyze drift
    report = monitor.analyze_drift(current_subset)

    # Log summary
    logger.info(report.summary())

    # Save report if requested
    if output_report_path:
        report_dict = report.to_dict()
        report_dict['baseline_path'] = str(baseline_path)
        report_dict['predictions_path'] = str(current_predictions_path)

        with open(output_report_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        logger.info(f"Report saved: {output_report_path}")

    # Determine pass/fail
    passed = not report.action_required

    if report.overall_alert == 'critical':
        logger.error("CRITICAL DRIFT DETECTED - Retraining recommended!")
        logger.error(f"  PSI threshold: {PSI_THRESHOLDS['high']}")
        return False, report
    elif report.overall_alert == 'warning':
        logger.warning("WARNING: Moderate drift detected - Monitor closely")
        return True, report
    else:
        logger.info("PASSED: No significant drift detected")
        return True, report


def generate_baseline(
    training_data_path: str,
    output_path: str,
    feature_cols: list = None
) -> None:
    """
    Generate drift baseline from training data.

    Args:
        training_data_path: Path to training context/predictions
        output_path: Path to save baseline JSON
        feature_cols: Columns to include in baseline
    """
    logger.info("=" * 60)
    logger.info("GENERATING DRIFT BASELINE")
    logger.info("=" * 60)

    # Load training data
    logger.info(f"Loading training data: {training_data_path}")

    if training_data_path.endswith('.npy'):
        data = np.load(training_data_path)
        if feature_cols is None:
            feature_cols = [f'feat_{i}' for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=feature_cols[:data.shape[1]])
    elif training_data_path.endswith('.parquet'):
        df = pd.read_parquet(training_data_path)
        if feature_cols is not None:
            df = df[[c for c in feature_cols if c in df.columns]]
    else:
        raise ValueError(f"Unsupported file format: {training_data_path}")

    logger.info(f"  Samples: {len(df):,}")
    logger.info(f"  Features: {list(df.columns)}")

    # Create and fit monitor
    monitor = DriftMonitor(
        n_bins=10,
        psi_threshold=PSI_THRESHOLDS['medium'],
        ks_threshold=0.2
    )
    monitor.fit_reference(df)

    # Save baseline
    monitor.save_reference(Path(output_path))
    logger.info(f"Baseline saved: {output_path}")


def quick_check(predictions_dir: str = 'output/predictions') -> int:
    """
    Quick drift check using most recent predictions.

    Returns exit code.
    """
    pred_dir = Path(predictions_dir)

    # Find most recent prediction file
    pred_files = sorted(pred_dir.glob('predictions_*.parquet'))
    if not pred_files:
        logger.error(f"No prediction files in {predictions_dir}")
        return 1

    current_path = pred_files[-1]
    logger.info(f"Using most recent predictions: {current_path}")

    # Find baseline
    baseline_candidates = [
        pred_dir / 'drift_baseline.json',
        pred_dir.parent / 'drift_baseline.json',
        Path('output/drift_baseline.json')
    ]

    baseline_path = None
    for bp in baseline_candidates:
        if bp.exists():
            baseline_path = bp
            break

    if baseline_path is None:
        logger.warning("No baseline found - generating from first prediction file")
        if len(pred_files) < 2:
            logger.error("Need at least 2 prediction files for drift check")
            return 1
        baseline_df = pd.read_parquet(pred_files[0])
        baseline_path = pred_dir / 'drift_baseline.json'

        monitor = DriftMonitor()
        # Use probability columns for drift detection
        prob_cols = ['danger_prob', 'noise_prob', 'target_prob', 'expected_value']
        available = [c for c in prob_cols if c in baseline_df.columns]
        if available:
            monitor.fit_reference(baseline_df[available])
            monitor.save_reference(baseline_path)
            logger.info(f"Generated baseline: {baseline_path}")
        else:
            logger.error("No probability columns in prediction file")
            return 1

    # Run check
    passed, report = check_weekly_drift(
        baseline_path=str(baseline_path),
        current_predictions_path=str(current_path),
        output_report_path=str(pred_dir / f'drift_report_{datetime.now():%Y%m%d_%H%M%S}.json')
    )

    if report is None:
        return 1
    elif report.overall_alert == 'critical':
        return 2
    elif report.overall_alert == 'warning':
        return 0  # Warning but still pass
    else:
        return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prediction Drift Monitor - Weekly PSI Check',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--baseline', type=str,
                        help='Path to baseline drift reference (JSON)')
    parser.add_argument('--current', type=str,
                        help='Path to current predictions (Parquet)')
    parser.add_argument('--output-report', type=str,
                        help='Path to save drift report JSON')

    parser.add_argument('--generate-baseline', action='store_true',
                        help='Generate new baseline from training data')
    parser.add_argument('--training-data', type=str,
                        help='Path to training data for baseline')
    parser.add_argument('--output', type=str, default='output/drift_baseline.json',
                        help='Output path for baseline')

    parser.add_argument('--quick-check', action='store_true',
                        help='Quick check using most recent predictions')
    parser.add_argument('--predictions-dir', type=str, default='output/predictions',
                        help='Directory for predictions (with --quick-check)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Mode 1: Generate baseline
    if args.generate_baseline:
        if not args.training_data:
            logger.error("--training-data required with --generate-baseline")
            return 1
        generate_baseline(args.training_data, args.output)
        return 0

    # Mode 2: Quick check
    if args.quick_check:
        return quick_check(args.predictions_dir)

    # Mode 3: Full check
    if args.baseline and args.current:
        passed, report = check_weekly_drift(
            baseline_path=args.baseline,
            current_predictions_path=args.current,
            output_report_path=args.output_report
        )

        if report is None:
            return 1
        elif report.overall_alert == 'critical':
            return 2
        return 0

    # No valid mode
    logger.error("Specify --baseline + --current, --generate-baseline, or --quick-check")
    return 1


if __name__ == "__main__":
    sys.exit(main())
