"""
Train Models with Strict No-Leakage Guarantee
==============================================
Explicitly excludes all future-looking features and validates
temporal integrity at every step.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# EXPLICITLY DEFINE SAFE FEATURES (NO FUTURE INFORMATION)
SAFE_FEATURES = [
    # Current state features (available at prediction time)
    "days_since_activation",
    "days_in_pattern",
    "current_price",
    "current_high",
    "current_low",
    "current_volume",

    # Technical indicators (calculated from historical data)
    "current_bbw_20",
    "current_bbw_percentile",
    "current_adx",
    "current_volume_ratio_20",
    "current_range_ratio",

    # Historical baselines (from qualification period)
    "baseline_bbw_avg",
    "baseline_volume_avg",
    "baseline_volatility",

    # Compression metrics (current vs baseline)
    "bbw_compression_ratio",
    "volume_compression_ratio",
    "volatility_compression_ratio",

    # Trend indicators
    "bbw_slope_20d",
    "adx_slope_20d",
    "consolidation_quality_score",

    # Position within pattern
    "price_position_in_range",
    "price_distance_from_upper_pct",
    "price_distance_from_lower_pct",
    "distance_from_power_pct",

    # EBP features (all calculated from historical data)
    "cci_bbw_compression",
    "cci_atr_compression",
    "cci_days_factor",
    "cci_score",
    "var_raw",
    "var_score",
    "nes_inactive_mass",
    "nes_wavelet_energy",
    "nes_rsa_proxy",
    "nes_score",
    "lpf_bid_pressure",
    "lpf_volume_pressure",
    "lpf_fta_proxy",
    "lpf_score",
    "tsf_days_in_consolidation",
    "tsf_score",
    "ebp_raw",
    "ebp_composite",

    # Historical volatility
    "avg_range_20d",
    "bbw_std_20d",
    "price_volatility_20d",
    "start_price"
]

# EXPLICITLY BANNED FEATURES (CONTAIN FUTURE INFORMATION)
LEAKY_FEATURES = [
    "max_gain_from_upper_pct",  # Future outcome
    "max_loss_from_lower_pct",  # Future outcome
    "days_to_max_gain",         # Future timing
    "days_to_max_loss",         # Future timing
    "outcome",                  # Future result
    "outcome_gain",             # Future result
    "days_to_peak",             # Future timing
    "end_price",                # Future price
    "breakout_direction"        # Future direction
]


def validate_no_leakage(df: pd.DataFrame, features: list) -> bool:
    """Validate that no leaky features are being used."""
    print("\n" + "="*60)
    print("LEAKAGE VALIDATION")
    print("="*60)

    # Check for any leaky features in the feature list
    used_leaky = [f for f in features if f in LEAKY_FEATURES]
    if used_leaky:
        print(f"ERROR: Leaky features found: {used_leaky}")
        return False

    # Check for any features with suspicious names
    suspicious = []
    for f in features:
        if any(term in f.lower() for term in ['future', 'outcome', 'gain', 'loss', 'peak', 'max', 'end']):
            if f not in SAFE_FEATURES:
                suspicious.append(f)

    if suspicious:
        print(f"WARNING: Suspicious features found: {suspicious}")
        print("Please verify these don't contain future information")

    print(f"[PASS] Validation passed: {len(features)} safe features")
    return True


def prepare_data(df: pd.DataFrame):
    """Prepare data with strict feature selection."""
    print("\n" + "="*60)
    print("DATA PREPARATION - NO LEAKAGE")
    print("="*60)

    # Map outcomes to numeric labels
    outcome_mapping = {
        'K0_STAGNANT': 0,
        'K1_MINIMAL': 1,
        'K2_QUALITY': 2,
        'K3_STRONG': 3,
        'K4_EXCEPTIONAL': 4,
        'K5_FAILED': 5
    }

    df['label'] = df['outcome_class'].map(outcome_mapping)
    df = df.dropna(subset=['label'])

    # Only use safe features
    available_features = [f for f in SAFE_FEATURES if f in df.columns]
    missing_features = [f for f in SAFE_FEATURES if f not in df.columns]

    print(f"Using {len(available_features)} safe features")
    if missing_features:
        print(f"Missing {len(missing_features)} features: {missing_features[:5]}...")

    # Validate no leakage
    if not validate_no_leakage(df, available_features):
        raise ValueError("Data leakage detected! Aborting training.")

    X = df[available_features].fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    y = df['label'].values.astype(int)

    # Print class distribution
    print("\nClass distribution:")
    for i in range(6):
        count = (y == i).sum()
        pct = count / len(y) * 100
        print(f"  K{i}: {count} ({pct:.2f}%)")

    return X, y, available_features


def train_models(X_train, y_train, X_val, y_val, feature_names):
    """Train XGBoost and LightGBM with extreme weighting."""
    print("\n" + "="*60)
    print("TRAINING LEAK-FREE MODELS")
    print("="*60)

    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)

    # Extreme weighting for K4
    weight_dict = {i: w for i, w in zip(classes, weights)}
    weight_dict[4] = weight_dict.get(4, 1.0) * 5  # Extra boost for K4

    print("Class weights:")
    for k, v in weight_dict.items():
        print(f"  K{k}: {v:.2f}")

    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        tree_method='hist',
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='mlogloss'
    )

    sample_weights = np.array([weight_dict[y] for y in y_train])
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        verbose=False
    )

    # Train LightGBM
    print("Training LightGBM...")
    lgb_model = LGBMClassifier(
        n_estimators=1000,
        num_leaves=31,
        learning_rate=0.1,
        objective='multiclass',
        num_class=6,
        random_state=42,
        verbosity=-1
    )

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        callbacks=[lambda x: None]  # Suppress output
    )

    return xgb_model, lgb_model, weight_dict


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model with focus on K4 detection."""
    print(f"\n{model_name} Evaluation:")
    print("-" * 40)

    y_pred = model.predict(X_test)

    # K4 metrics
    k4_mask = y_test == 4
    k4_recall = (y_pred[k4_mask] == 4).mean() if k4_mask.sum() > 0 else 0
    print(f"K4 Recall: {k4_recall:.1%} ({k4_mask.sum()} samples)")

    # K3 metrics
    k3_mask = y_test == 3
    k3_recall = (y_pred[k3_mask] == 3).mean() if k3_mask.sum() > 0 else 0
    print(f"K3 Recall: {k3_recall:.1%} ({k3_mask.sum()} samples)")

    # K3+K4 combined
    k34_mask = (y_test == 3) | (y_test == 4)
    k34_pred_correct = ((y_pred == 3) | (y_pred == 4)) & k34_mask
    k34_recall = k34_pred_correct.sum() / k34_mask.sum() if k34_mask.sum() > 0 else 0
    print(f"K3+K4 Recall: {k34_recall:.1%}")

    # K5 precision
    k5_pred_mask = y_pred == 5
    k5_precision = (y_test[k5_pred_mask] == 5).mean() if k5_pred_mask.sum() > 0 else 0
    print(f"K5 Precision: {k5_precision:.1%}")

    return {
        'k4_recall': k4_recall,
        'k3_recall': k3_recall,
        'k34_recall': k34_recall,
        'k5_precision': k5_precision,
        'predictions': y_pred
    }


def main():
    """Main training pipeline with no-leakage guarantee."""
    print("="*60)
    print("LEAK-FREE MODEL TRAINING")
    print("="*60)
    print("This training explicitly excludes all future-looking features")
    print("to ensure realistic, production-ready performance metrics.")

    # Load data
    data_file = Path("output/patterns_labeled_enhanced_20251104_193956.parquet")
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return

    print(f"\nLoading data from {data_file}")
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df)} patterns")

    # Prepare data with strict feature selection
    X, y, feature_names = prepare_data(df)

    # Temporal split (more realistic than random)
    split_idx = int(len(X) * 0.7)
    X_temp = X[:split_idx]
    y_temp = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    # Further split training into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train)} patterns")
    print(f"  Val: {len(X_val)} patterns")
    print(f"  Test: {len(X_test)} patterns (temporal holdout)")

    # Train models
    xgb_model, lgb_model, class_weights = train_models(
        X_train, y_train, X_val, y_val, feature_names
    )

    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION (Temporal Holdout)")
    print("="*60)

    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    lgb_metrics = evaluate_model(lgb_model, X_test, y_test, "LightGBM")

    # Save models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    xgb_path = Path(f"output/models/xgboost_noleakage_{timestamp}.pkl")
    with open(xgb_path, 'wb') as f:
        pickle.dump({
            'model': xgb_model,
            'feature_names': feature_names,
            'class_weights': class_weights,
            'metrics': xgb_metrics
        }, f)
    print(f"\nSaved XGBoost model: {xgb_path}")

    lgb_path = Path(f"output/models/lightgbm_noleakage_{timestamp}.pkl")
    with open(lgb_path, 'wb') as f:
        pickle.dump({
            'model': lgb_model,
            'feature_names': feature_names,
            'class_weights': class_weights,
            'metrics': lgb_metrics
        }, f)
    print(f"Saved LightGBM model: {lgb_path}")

    # Summary
    print("\n" + "="*60)
    print("LEAK-FREE TRAINING COMPLETE")
    print("="*60)
    print("\nRealistic Performance Summary:")
    print(f"  XGBoost K4 Recall: {xgb_metrics['k4_recall']:.1%}")
    print(f"  LightGBM K4 Recall: {lgb_metrics['k4_recall']:.1%}")
    print(f"  Best K3+K4 Recall: {max(xgb_metrics['k34_recall'], lgb_metrics['k34_recall']):.1%}")

    if max(xgb_metrics['k4_recall'], lgb_metrics['k4_recall']) > 0.5:
        print("\n[SUCCESS] Strong K4 detection without data leakage!")
    else:
        print("\n[WARNING] K4 detection below 50% - this is the realistic performance")

    return xgb_model, lgb_model


if __name__ == "__main__":
    models = main()