"""
Train K4 Anomaly Detection Model

Implements anomaly detection approach for K4_EXCEPTIONAL patterns.
Better suited for extreme rarity (0.03%) than classification.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.parquet_helper import read_data
from src.models.anomaly_detector_k4 import K4AnomalyDetector, evaluate_anomaly_detector

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = Path('output')
MODEL_DIR = Path('output/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data_for_anomaly_detection(df):
    """Prepare features and labels for anomaly detection training."""

    # Define feature columns (exclude target and metadata)
    exclude_cols = [
        'outcome_class', 'max_gain_from_upper_pct', 'max_loss_from_lower_pct',
        'days_to_max_gain', 'days_to_max_loss', 'ticker', 'activation_date',
        'snapshot_date', 'pattern_id', 'start_date', 'end_date', 'phase',
        'breakout_direction', 'went_below_boundary', 'days_to_below_boundary',
        'reached_anxious', 'first_breach_day', 'excluded', 'passed_20_day_filter',
        'days_survived', 'refinement_applied', 'original_upper_boundary',
        'original_lower_boundary', 'refined_upper_boundary', 'refined_lower_boundary',
        'original_channel_width_pct', 'refined_channel_width_pct', 'channel_narrowing_pct',
        'ebp_signal',  # Exclude text column
        'end_price'  # This column has 100% NaN values
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Create numeric labels from outcome classes
    class_mapping = {
        'K0_STAGNANT': 0,
        'K1_MINIMAL': 1,
        'K2_QUALITY': 2,
        'K3_STRONG': 3,
        'K4_EXCEPTIONAL': 4,
        'K5_FAILED': 5
    }

    df['outcome_numeric'] = df['outcome_class'].map(class_mapping)

    # Handle NaN values more intelligently
    # Fill NaN values with reasonable defaults (forward fill, then backward fill, then 0)
    for col in feature_cols:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill().fillna(0)

    X = df[feature_cols].values
    y = df['outcome_numeric'].values

    return X, y, feature_cols


def analyze_anomaly_patterns(detector, X_test, y_test, feature_names):
    """Detailed analysis of detected anomalies."""

    logger.info("\n" + "="*70)
    logger.info("DETAILED ANOMALY PATTERN ANALYSIS")
    logger.info("="*70)

    # Get anomaly scores and predictions
    anomaly_scores = detector.get_anomaly_scores(X_test)
    predictions = detector.predict(X_test)
    proba = detector.predict_proba(X_test)

    # Analyze score distribution
    logger.info("\nAnomaly Score Statistics:")
    logger.info(f"  Mean: {anomaly_scores.mean():.3f}")
    logger.info(f"  Std: {anomaly_scores.std():.3f}")
    logger.info(f"  Min: {anomaly_scores.min():.3f}")
    logger.info(f"  Max: {anomaly_scores.max():.3f}")
    logger.info(f"  Median: {np.median(anomaly_scores):.3f}")

    # Score percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    logger.info("\nAnomaly Score Percentiles:")
    for p in percentiles:
        score = np.percentile(anomaly_scores, p)
        logger.info(f"  {p:3d}th percentile: {score:.3f}")

    # Class-specific analysis
    logger.info("\nClass-Specific Anomaly Detection:")
    for cls in range(6):
        cls_mask = (y_test == cls)
        if cls_mask.sum() > 0:
            cls_detected = predictions[cls_mask].sum()
            cls_scores = anomaly_scores[cls_mask]
            cls_proba = proba[cls_mask, 1]

            logger.info(f"\nClass K{cls} ({cls_mask.sum()} samples):")
            logger.info(f"  Detected as anomaly: {cls_detected}/{cls_mask.sum()} ({cls_detected/cls_mask.sum():.1%})")
            logger.info(f"  Mean anomaly score: {cls_scores.mean():.3f}")
            logger.info(f"  Mean P(K4): {cls_proba.mean():.3f}")

            # Top 3 most anomalous in this class
            if len(cls_scores) >= 3:
                top_3_idx = np.argsort(cls_scores)[:3]
                logger.info(f"  Top 3 anomaly scores: {cls_scores[top_3_idx]}")

    # Feature analysis for top anomalies
    if detector.feature_importance_ is not None:
        logger.info("\nFeature Importance for Anomaly Detection:")
        top_10_features = np.argsort(detector.feature_importance_)[-10:][::-1]

        for i, feat_idx in enumerate(top_10_features, 1):
            if feat_idx < len(feature_names):
                feat_name = feature_names[feat_idx]
                importance = detector.feature_importance_[feat_idx]
                logger.info(f"  {i:2d}. {feat_name:30s}: {importance:.4f}")

    # Threshold analysis
    logger.info("\nThreshold Analysis for K4 Detection:")
    thresholds = np.percentile(anomaly_scores, [0.01, 0.03, 0.05, 0.1, 0.5, 1.0])

    for threshold in thresholds:
        mask = anomaly_scores <= threshold
        n_detected = mask.sum()

        if n_detected > 0:
            k4_in_detected = ((y_test == 4) & mask).sum()
            k3_in_detected = ((y_test == 3) & mask).sum()

            percentile = (anomaly_scores <= threshold).mean() * 100

            logger.info(f"\nThreshold {threshold:.3f} (bottom {percentile:.2f}%):")
            logger.info(f"  Total detected: {n_detected}")
            logger.info(f"  K4 detected: {k4_in_detected}")
            logger.info(f"  K3 detected: {k3_in_detected}")

            if (y_test == 4).sum() > 0:
                k4_recall = k4_in_detected / (y_test == 4).sum()
                logger.info(f"  K4 Recall: {k4_recall:.1%}")

    return anomaly_scores, predictions


def train_anomaly_detector():
    """Main training function for K4 anomaly detection."""

    logger.info("\n" + "="*80)
    logger.info("K4 ANOMALY DETECTION TRAINING PIPELINE")
    logger.info("="*80)

    # Find the latest enhanced labeled file
    labeled_files = list(INPUT_DIR.glob('patterns_labeled_enhanced_*.parquet'))
    if not labeled_files:
        logger.error("No enhanced labeled pattern files found!")
        return

    latest_file = max(labeled_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"\nLoading data from: {latest_file}")

    # Load data
    df = read_data(latest_file)
    logger.info(f"Loaded {len(df):,} labeled patterns")

    # Show outcome distribution
    logger.info("\nOutcome Distribution:")
    outcome_counts = df['outcome_class'].value_counts()
    for outcome, count in outcome_counts.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {outcome:15s}: {count:7,} ({pct:6.3f}%)")

    # Highlight K4 rarity
    k4_count = len(df[df['outcome_class'] == 'K4_EXCEPTIONAL'])
    k3_count = len(df[df['outcome_class'] == 'K3_STRONG'])
    logger.info(f"\nTarget Anomalies:")
    logger.info(f"  K4_EXCEPTIONAL: {k4_count:,} ({k4_count/len(df)*100:.3f}%)")
    logger.info(f"  K3_STRONG: {k3_count:,} ({k3_count/len(df)*100:.3f}%)")
    logger.info(f"  Combined K3+K4: {k3_count + k4_count:,} ({(k3_count + k4_count)/len(df)*100:.2f}%)")

    # Prepare data
    X, y, feature_names = prepare_data_for_anomaly_detection(df)
    logger.info(f"\nFeature matrix shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")

    # Show test set K4 distribution
    test_k4 = (y_test == 4).sum()
    test_k3 = (y_test == 3).sum()
    logger.info(f"Test set K4: {test_k4} samples")
    logger.info(f"Test set K3: {test_k3} samples")

    # Initialize and train detector
    contamination = (k3_count + k4_count) / len(df)  # Actual K3+K4 rate

    detector = K4AnomalyDetector(
        contamination=contamination * 2,  # Allow some buffer for detection
        n_estimators=200,
        max_samples='auto',
        random_state=42,
        use_refinement=True,
        anomaly_threshold=-0.5
    )

    logger.info(f"\nTraining with contamination={contamination*2:.3%}")
    detector.fit(X_train, y_train)

    # Evaluate on test set
    logger.info("\n" + "="*70)
    logger.info("TEST SET EVALUATION")
    logger.info("="*70)

    metrics = evaluate_anomaly_detector(detector, X_test, y_test)

    # Detailed pattern analysis
    anomaly_scores, predictions = analyze_anomaly_patterns(detector, X_test, y_test, feature_names)

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODEL_DIR / f'k4_anomaly_detector_{timestamp}.pkl'
    detector.save_model(str(model_path))

    # Save evaluation results
    results = {
        'timestamp': timestamp,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'contamination': contamination,
        'k4_recall': metrics['k4_recall'],
        'k3_recall': metrics['k3_recall'],
        'false_positive_rate': metrics['false_positive_rate'],
        'n_predicted_k4': metrics['n_predicted_k4'],
        'test_k4_count': test_k4,
        'test_k3_count': test_k3
    }

    results_df = pd.DataFrame([results])
    results_path = MODEL_DIR / f'anomaly_detector_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"K4 Recall: {metrics['k4_recall']:.1%}")
    logger.info(f"K3 Recall: {metrics['k3_recall']:.1%}")
    logger.info(f"False Positive Rate: {metrics['false_positive_rate']:.2%}")
    logger.info(f"Model saved to: {model_path}")
    logger.info("="*80)

    return detector, metrics


if __name__ == "__main__":
    detector, metrics = train_anomaly_detector()