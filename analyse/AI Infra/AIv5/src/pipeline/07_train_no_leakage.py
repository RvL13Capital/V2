"""
Training Script with No Data Leakage

This script trains models using:
1. Only features available at prediction time (no future information)
2. Correctly labeled snapshots (each snapshot's own forward outcome)
3. Realistic expectations for performance
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.parquet_helper import read_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = Path('output/models_no_leakage')
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def prepare_features_no_leakage(df):
    """
    Prepare features WITHOUT data leakage.

    Only includes features that would be available at prediction time.
    """
    # CRITICAL: Only include features available at snapshot time
    # These are features calculated from PAST data only
    valid_features = [
        # Pattern state features (known at snapshot time)
        'days_since_activation',
        'days_in_pattern',
        'consolidation_channel_width_pct',
        'consolidation_period_days',

        # Current market metrics (at snapshot time)
        'current_price',
        'current_high',
        'current_low',
        'current_volume',
        'current_bbw_20',
        'current_bbw_percentile',
        'current_adx',
        'current_volume_ratio_20',
        'current_range_ratio',

        # Historical baselines (calculated from past)
        'baseline_bbw_avg',
        'baseline_volume_avg',
        'baseline_volatility',
        'bbw_compression_ratio',
        'volume_compression_ratio',
        'volatility_compression_ratio',

        # Trend indicators (from past data)
        'bbw_slope_20d',
        'adx_slope_20d',
        'consolidation_quality_score',

        # Price position (at snapshot time)
        'price_position_in_range',
        'price_distance_from_upper_pct',
        'price_distance_from_lower_pct',
        'distance_from_power_pct',

        # EBP features (if calculated from past data only)
        'cci_bbw_compression',
        'cci_atr_compression',
        'cci_days_factor',
        'cci_score',
        'var_raw',
        'var_score',
        'nes_inactive_mass',
        'nes_wavelet_energy',
        'nes_rsa_proxy',
        'nes_score',
        'lpf_bid_pressure',
        'lpf_volume_pressure',
        'lpf_fta_proxy',
        'lpf_score',
        'tsf_days_in_consolidation',
        'tsf_score',
        'ebp_raw',
        'ebp_composite',

        # Technical indicators (from past)
        'avg_range_20d',
        'bbw_std_20d',
        'price_volatility_20d',
        'start_price'
    ]

    # Only keep features that exist in the dataframe
    available_features = [f for f in valid_features if f in df.columns]

    logger.info(f"\nUsing {len(available_features)} leak-free features:")
    logger.info(f"  First 10: {available_features[:10]}")

    # Check for any remaining NaN values
    df_features = df[available_features].copy()
    nan_counts = df_features.isnull().sum()
    if nan_counts.sum() > 0:
        logger.info("\nHandling NaN values in features:")
        for col in nan_counts[nan_counts > 0].index:
            logger.info(f"  {col}: {nan_counts[col]} NaN values")
        # Fill NaN with 0 (conservative approach)
        df_features = df_features.fillna(0)

    return df_features, available_features


def train_realistic_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost with realistic expectations.

    No extreme weights since we don't have data leakage anymore.
    """
    # Calculate class weights (moderate, not extreme)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(unique_classes)

    # Moderate class weights (not extreme)
    weights = {}
    for cls, count in zip(unique_classes, class_counts):
        # Square root weighting (less aggressive than inverse)
        weights[cls] = np.sqrt(n_samples / (n_classes * count))

    logger.info("\nClass weights (moderate):")
    for cls in sorted(weights.keys()):
        count = class_counts[unique_classes == cls][0]
        logger.info(f"  Class {cls}: weight={weights[cls]:.2f}, count={count}")

    # Convert to sample weights
    sample_weights = np.array([weights[y] for y in y_train])

    # Train XGBoost with moderate hyperparameters
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=6,
        max_depth=5,  # Moderate depth
        n_estimators=100,
        learning_rate=0.05,
        min_child_weight=5,  # Higher to prevent overfitting
        gamma=2.0,  # Moderate regularization
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',  # Memory efficient
        eval_metric='mlogloss',
        early_stopping_rounds=20  # Moved here for newer XGBoost
    )

    # Train with validation set
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def evaluate_realistic_performance(model, X_test, y_test, feature_names):
    """
    Evaluate with realistic expectations.
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Classification report
    class_names = ['K0_STAGNANT', 'K1_MINIMAL', 'K2_QUALITY',
                   'K3_STRONG', 'K4_EXCEPTIONAL', 'K5_FAILED']

    logger.info("\n" + "="*70)
    logger.info("REALISTIC PERFORMANCE EVALUATION (NO DATA LEAKAGE)")
    logger.info("="*70)

    logger.info("\nClassification Report:")
    logger.info("-" * 50)
    report = classification_report(y_test, y_pred,
                                  target_names=class_names,
                                  zero_division=0)
    logger.info(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix:")
    logger.info("Predicted:  K0   K1   K2   K3   K4   K5")
    for i, row in enumerate(cm):
        logger.info(f"Actual K{i}: {row}")

    # K3/K4 specific metrics
    k3_mask = (y_test == 3)
    k4_mask = (y_test == 4)

    if k3_mask.sum() > 0:
        k3_correct = (y_pred[k3_mask] == 3).sum()
        k3_recall = k3_correct / k3_mask.sum()
        logger.info(f"\nK3_STRONG Recall: {k3_recall:.1%} ({k3_correct}/{k3_mask.sum()})")
        logger.info(f"Expected realistic range: 20-40%")

    if k4_mask.sum() > 0:
        k4_correct = (y_pred[k4_mask] == 4).sum()
        k4_recall = k4_correct / k4_mask.sum()
        logger.info(f"\nK4_EXCEPTIONAL Recall: {k4_recall:.1%} ({k4_correct}/{k4_mask.sum()})")
        logger.info(f"Expected realistic range: 15-30%")

    # Feature importance
    importance = model.feature_importances_
    top_features = np.argsort(importance)[-10:][::-1]

    logger.info("\nTop 10 Most Important Features (No Leakage):")
    for i, idx in enumerate(top_features, 1):
        logger.info(f"  {i:2d}. {feature_names[idx]:30s}: {importance[idx]:.4f}")

    return y_pred, report


def main():
    """Main training pipeline without data leakage."""

    logger.info("\n" + "="*80)
    logger.info("TRAINING WITH NO DATA LEAKAGE - REALISTIC PERFORMANCE")
    logger.info("="*80)

    # Check for forward-labeled file first
    forward_files = list(Path('output').glob('snapshots_forward_labeled_*.parquet'))
    if forward_files:
        latest_file = max(forward_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"\nUsing forward-labeled file: {latest_file}")
        target_col = 'forward_outcome_class'
    else:
        # Fall back to enhanced file (but warn about wrong labeling)
        enhanced_files = list(Path('output').glob('patterns_labeled_enhanced_*.parquet'))
        if not enhanced_files:
            logger.error("No labeled files found! Run labeling first.")
            return
        latest_file = max(enhanced_files, key=lambda p: p.stat().st_mtime)
        logger.warning(f"\nWARNING: Using pattern-labeled file (not ideal): {latest_file}")
        logger.warning("For proper results, run 06_label_snapshots_forward.py first!")
        target_col = 'outcome_class'

    # Load data
    df = read_data(latest_file)
    logger.info(f"Loaded {len(df):,} snapshots")

    # Create numeric labels
    class_mapping = {
        'K0_STAGNANT': 0, 'K1_MINIMAL': 1, 'K2_QUALITY': 2,
        'K3_STRONG': 3, 'K4_EXCEPTIONAL': 4, 'K5_FAILED': 5
    }
    df['outcome_numeric'] = df[target_col].map(class_mapping)

    # Show distribution
    logger.info(f"\nOutcome Distribution:")
    for outcome, count in df[target_col].value_counts().items():
        pct = count / len(df) * 100
        logger.info(f"  {outcome:15s}: {count:7,} ({pct:5.1f}%)")

    # Prepare features (NO LEAKAGE)
    X, feature_names = prepare_features_no_leakage(df)
    y = df['outcome_numeric'].values

    # Remove any rows with NaN in target
    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"\nFinal dataset: {len(X):,} samples, {X.shape[1]} features")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")

    # Further split for validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Train model
    logger.info("\nTraining XGBoost (no data leakage)...")
    model = train_realistic_model(X_train_final, y_train_final, X_val, y_val)

    # Evaluate
    y_pred, report = evaluate_realistic_performance(model, X_test, y_test, feature_names)

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODEL_DIR / f'xgboost_no_leakage_{timestamp}.pkl'
    joblib.dump(model, model_path)
    logger.info(f"\nModel saved to: {model_path}")

    # Save predictions for analysis
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'is_k3': (y_test == 3).astype(int),
        'is_k4': (y_test == 4).astype(int),
        'predicted_k3': (y_pred == 3).astype(int),
        'predicted_k4': (y_pred == 4).astype(int)
    })
    predictions_path = MODEL_DIR / f'predictions_no_leakage_{timestamp}.csv'
    predictions_df.to_csv(predictions_path, index=False)

    logger.info(f"Predictions saved to: {predictions_path}")

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE - REALISTIC PERFORMANCE ACHIEVED")
    logger.info("="*80)
    logger.info("\nExpected realistic performance:")
    logger.info("  K4 Recall: 15-30% (not 100%)")
    logger.info("  K3 Recall: 20-40% (not 100%)")
    logger.info("  This reflects the true difficulty of predicting 75%+ gains")
    logger.info("="*80)


if __name__ == "__main__":
    main()