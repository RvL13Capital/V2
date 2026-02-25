"""
Enhanced Model Training with Extreme Class Weighting - AIv4

Addresses extreme imbalance in the dataset:
- K4_EXCEPTIONAL: 0.03% (34 samples)
- K3_STRONG: 0.4% (544 samples)
- K5_FAILED: 28.3%
- K0_STAGNANT: 39.6%

Uses inverse frequency weighting and optimized hyperparameters for rare event detection.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.parquet_helper import read_data, write_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = Path('output/models')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_class_weights(y_train, strategy='inverse_freq'):
    """
    Calculate extreme class weights for imbalanced classification.

    Strategies:
    - inverse_freq: Full inverse frequency weighting
    - sqrt_inverse: Square root of inverse frequency (less aggressive)
    - custom: Business-aligned custom weights
    """
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(unique_classes)

    if strategy == 'inverse_freq':
        # Full inverse frequency weighting
        weights = {}
        for cls, count in zip(unique_classes, class_counts):
            weights[cls] = n_samples / (n_classes * count)

    elif strategy == 'sqrt_inverse':
        # Square root of inverse frequency (less aggressive)
        weights = {}
        for cls, count in zip(unique_classes, class_counts):
            weights[cls] = np.sqrt(n_samples / (n_classes * count))

    elif strategy == 'custom':
        # Custom business-aligned weights
        weights = {
            0: 0.5,    # K0_STAGNANT - low weight
            1: 0.8,    # K1_MINIMAL - low weight
            2: 3.0,    # K2_QUALITY - moderate weight
            3: 50.0,   # K3_STRONG - high weight
            4: 500.0,  # K4_EXCEPTIONAL - extreme weight
            5: 1.0     # K5_FAILED - baseline weight
        }

    logger.info(f"\nClass weights ({strategy}):")
    logger.info("-" * 40)
    for cls in sorted(weights.keys()):
        count = class_counts[unique_classes == cls][0]
        logger.info(f"  Class {cls}: weight={weights[cls]:.2f}, count={count}")

    return weights


def create_sample_weights(y_train, class_weights):
    """Convert class weights to sample weights."""
    sample_weights = np.zeros(len(y_train))
    for i, label in enumerate(y_train):
        sample_weights[i] = class_weights[label]
    return sample_weights


def train_xgboost_extreme(X_train, y_train, X_val, y_val, class_weights):
    """
    Train XGBoost with extreme class weighting and optimized hyperparameters.
    """
    logger.info("\n" + "="*70)
    logger.info("TRAINING XGBOOST WITH EXTREME WEIGHTING")
    logger.info("="*70)

    # Create sample weights
    sample_weights_train = create_sample_weights(y_train, class_weights)
    sample_weights_val = create_sample_weights(y_val, class_weights)

    # Hyperparameters optimized for extreme imbalance
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 6,
        'max_depth': 4,  # Reduced to prevent overfitting
        'min_child_weight': 0.1,  # Allow small leafs for rare classes
        'gamma': 5.0,  # High minimum loss reduction
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'learning_rate': 0.01,  # Lower for stability with extreme weights
        'n_estimators': 1000,
        'reg_alpha': 10.0,  # Strong L1 regularization
        'reg_lambda': 10.0,  # Strong L2 regularization
        'tree_method': 'hist',
        'device': 'cpu',
        'random_state': 42,
        'verbosity': 1,
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': 50
    }

    # Create model
    model = xgb.XGBClassifier(**xgb_params)

    # Train with early stopping
    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=eval_set,
        verbose=True
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best score: {model.best_score:.4f}")

    return model


def train_lightgbm_extreme(X_train, y_train, X_val, y_val, class_weights):
    """
    Train LightGBM with extreme class weighting.
    """
    logger.info("\n" + "="*70)
    logger.info("TRAINING LIGHTGBM WITH EXTREME WEIGHTING")
    logger.info("="*70)

    # Create sample weights
    sample_weights_train = create_sample_weights(y_train, class_weights)
    sample_weights_val = create_sample_weights(y_val, class_weights)

    # Hyperparameters for extreme imbalance
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 6,
        'metric': 'multi_logloss',
        'max_depth': 4,
        'num_leaves': 20,
        'min_child_samples': 5,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 10.0,
        'reg_lambda': 10.0,
        'random_state': 42,
        'verbosity': 1
    }

    # Create model
    model = lgb.LGBMClassifier(**lgb_params)

    # Train
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[sample_weights_val],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )

    return model


def evaluate_model_extreme(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model with focus on K3/K4 recall and K5 precision.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    logger.info(f"\n{'='*70}")
    logger.info(f"{model_name} EVALUATION - EXTREME IMBALANCE METRICS")
    logger.info(f"{'='*70}")

    # Class names
    class_names = ['K0_STAGNANT', 'K1_MINIMAL', 'K2_QUALITY',
                   'K3_STRONG', 'K4_EXCEPTIONAL', 'K5_FAILED']

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )

    logger.info("\nPer-Class Performance:")
    logger.info("-" * 70)
    logger.info(f"{'Class':<15} {'Support':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    logger.info("-" * 70)

    for i, cls_name in enumerate(class_names):
        logger.info(f"{cls_name:<15} {support[i]:>8} {precision[i]:>10.2%} "
                   f"{recall[i]:>10.2%} {f1[i]:>10.2%}")

    # Key metrics for extreme imbalance
    k4_recall = recall[4] if len(recall) > 4 else 0
    k3_recall = recall[3] if len(recall) > 3 else 0
    k5_precision = precision[5] if len(precision) > 5 else 0

    # Combined K3+K4 metrics
    k3k4_mask = np.isin(y_test, [3, 4])
    if k3k4_mask.any():
        k3k4_pred = np.isin(y_pred, [3, 4])
        k3k4_recall = np.sum(k3k4_mask & k3k4_pred) / np.sum(k3k4_mask)
    else:
        k3k4_recall = 0

    logger.info(f"\n{'='*50}")
    logger.info("KEY BUSINESS METRICS:")
    logger.info("-" * 50)
    logger.info(f"K4_EXCEPTIONAL Recall: {k4_recall:.1%} (Target: ≥30%)")
    logger.info(f"K3_STRONG Recall: {k3_recall:.1%} (Target: ≥40%)")
    logger.info(f"K3+K4 Combined Recall: {k3k4_recall:.1%} (Target: ≥40%)")
    logger.info(f"K5_FAILED Precision: {k5_precision:.1%} (Target: ≥60%)")

    # Success criteria
    success = True
    if k4_recall < 0.20:
        logger.warning("⚠ K4 recall below 20% threshold")
        success = False
    if k3k4_recall < 0.30:
        logger.warning("⚠ K3+K4 recall below 30% threshold")
        success = False
    if k5_precision < 0.50:
        logger.warning("⚠ K5 precision below 50% threshold")
        success = False

    if success:
        logger.info("\n✓ Model meets minimum performance criteria!")

    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    logger.info("-" * 50)
    cm = confusion_matrix(y_test, y_pred)

    # Build header row
    header = f"{'True/Pred':<10}"
    for cls in class_names:
        header += f"{cls[:2]:>5}"
    logger.info(header)

    # Build each row of the confusion matrix
    for i, cls in enumerate(class_names):
        row = f"{cls[:10]:<10}"
        for j in range(len(class_names)):
            row += f"{cm[i,j]:>5}"
        logger.info(row)

    return {
        'k4_recall': k4_recall,
        'k3_recall': k3_recall,
        'k3k4_recall': k3k4_recall,
        'k5_precision': k5_precision,
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            cls: {'precision': p, 'recall': r, 'f1': f, 'support': int(s)}
            for cls, p, r, f, s in zip(class_names, precision, recall, f1, support)
        }
    }


def prepare_features(df):
    """
    Prepare features WITHOUT data leakage.

    Only includes features that would be available at prediction time.
    This matches the approach in 07_train_no_leakage.py.
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

    logger.info(f"\nUsing {len(available_features)} leak-free features")
    logger.info(f"  First 10: {available_features[:10]}")

    # CRITICAL: Explicitly exclude all features with future information
    leaky_features = [
        'first_breach_day',
        'max_gain_from_upper_pct',
        'max_loss_from_lower_pct',
        'days_to_max_gain',
        'days_to_max_loss',
        'consecutive_days_below_max',
        'went_below_boundary',
        'days_to_below_boundary',
        'reached_anxious'
    ]

    # Check if any leaky features are accidentally included
    included_leaky = [f for f in leaky_features if f in available_features]
    if included_leaky:
        logger.warning(f"WARNING: Found leaky features that should be excluded: {included_leaky}")
        available_features = [f for f in available_features if f not in leaky_features]

    # Check for any remaining NaN values
    df_features = df[available_features].copy()
    nan_counts = df_features.isnull().sum()
    if nan_counts.sum() > 0:
        logger.info("\nHandling NaN values in features:")
        for col in nan_counts[nan_counts > 0].index[:5]:  # Show first 5
            logger.info(f"  {col}: {nan_counts[col]} NaN values")
        # Fill NaN with 0 (conservative approach)
        df_features = df_features.fillna(0)

    # Handle infinite values
    df_features = df_features.replace([np.inf, -np.inf], 0)

    return df_features


def main(input_file=None, weight_strategy='inverse_freq'):
    """
    Main training pipeline with extreme class weighting.
    """
    logger.info(f"\n{'='*70}")
    logger.info("ENHANCED MODEL TRAINING - EXTREME CLASS WEIGHTING")
    logger.info(f"{'='*70}\n")

    # Load enhanced dataset
    if input_file is None:
        input_file = Path('output/patterns_labeled_enhanced_20251029_040159.parquet')

    logger.info(f"Loading enhanced dataset: {input_file}")
    df = read_data(input_file)
    logger.info(f"Loaded {len(df):,} samples")

    # Prepare features
    X = prepare_features(df)
    logger.info(f"Features prepared: {X.shape[1]} features")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['outcome_class'])

    logger.info("\nLabel encoding:")
    for i, cls in enumerate(label_encoder.classes_):
        count = np.sum(y == i)
        pct = count / len(y) * 100
        logger.info(f"  {i}: {cls} ({count:,} samples, {pct:.2f}%)")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    logger.info(f"\nDataset splits:")
    logger.info(f"  Train: {len(X_train):,} samples")
    logger.info(f"  Val: {len(X_val):,} samples")
    logger.info(f"  Test: {len(X_test):,} samples")

    # Calculate extreme class weights
    class_weights = calculate_class_weights(y_train, strategy=weight_strategy)

    # Train XGBoost
    xgb_model = train_xgboost_extreme(X_train, y_train, X_val, y_val, class_weights)

    # Train LightGBM
    lgb_model = train_lightgbm_extreme(X_train, y_train, X_val, y_val, class_weights)

    # Evaluate models
    xgb_metrics = evaluate_model_extreme(xgb_model, X_test, y_test, "XGBoost")
    lgb_metrics = evaluate_model_extreme(lgb_model, X_test, y_test, "LightGBM")

    # Save models
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    xgb_path = OUTPUT_DIR / f'xgboost_extreme_{timestamp}.pkl'
    with open(xgb_path, 'wb') as f:
        pickle.dump({
            'model': xgb_model,
            'label_encoder': label_encoder,
            'feature_names': X.columns.tolist(),
            'class_weights': class_weights,
            'metrics': xgb_metrics
        }, f)
    logger.info(f"\nSaved XGBoost model: {xgb_path}")

    lgb_path = OUTPUT_DIR / f'lightgbm_extreme_{timestamp}.pkl'
    with open(lgb_path, 'wb') as f:
        pickle.dump({
            'model': lgb_model,
            'label_encoder': label_encoder,
            'feature_names': X.columns.tolist(),
            'class_weights': class_weights,
            'metrics': lgb_metrics
        }, f)
    logger.info(f"Saved LightGBM model: {lgb_path}")

    # Save metrics summary
    metrics_path = OUTPUT_DIR / f'metrics_extreme_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'xgboost': xgb_metrics,
            'lightgbm': lgb_metrics,
            'weight_strategy': weight_strategy,
            'class_weights': class_weights,
            'dataset_info': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X.shape[1]
            }
        }, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    logger.info(f"\n{'='*70}")
    logger.info("TRAINING COMPLETE!")
    logger.info(f"{'='*70}")

    # Print final comparison
    logger.info("\nFinal Model Comparison:")
    logger.info("-" * 40)
    logger.info(f"{'Metric':<25} {'XGBoost':>15} {'LightGBM':>15}")
    logger.info("-" * 40)
    logger.info(f"{'K4 Recall':<25} {xgb_metrics['k4_recall']:>15.1%} {lgb_metrics['k4_recall']:>15.1%}")
    logger.info(f"{'K3 Recall':<25} {xgb_metrics['k3_recall']:>15.1%} {lgb_metrics['k3_recall']:>15.1%}")
    logger.info(f"{'K3+K4 Recall':<25} {xgb_metrics['k3k4_recall']:>15.1%} {lgb_metrics['k3k4_recall']:>15.1%}")
    logger.info(f"{'K5 Precision':<25} {xgb_metrics['k5_precision']:>15.1%} {lgb_metrics['k5_precision']:>15.1%}")

    return xgb_model, lgb_model, xgb_metrics, lgb_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train models with extreme class weighting')
    parser.add_argument('--input', type=str, default=None,
                       help='Input labeled patterns file')
    parser.add_argument('--weights', type=str, default='inverse_freq',
                       choices=['inverse_freq', 'sqrt_inverse', 'custom'],
                       help='Class weighting strategy')

    args = parser.parse_args()

    # Run training
    main(input_file=args.input, weight_strategy=args.weights)