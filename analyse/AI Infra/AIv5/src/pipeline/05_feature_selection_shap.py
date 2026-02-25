"""
Feature Selection using SHAP

Identifies the most important features for K3/K4 detection using SHAP values.
Reduces feature set from 52 to ~20 most important features to prevent overfitting.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.parquet_helper import read_data, write_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = Path('output')
OUTPUT_DIR = Path('output/feature_selection')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data_for_shap(df):
    """Prepare data for SHAP analysis."""

    # Define feature columns (same as anomaly detector)
    exclude_cols = [
        'outcome_class', 'max_gain_from_upper_pct', 'max_loss_from_lower_pct',
        'days_to_max_gain', 'days_to_max_loss', 'ticker', 'activation_date',
        'snapshot_date', 'pattern_id', 'start_date', 'end_date', 'phase',
        'breakout_direction', 'went_below_boundary', 'days_to_below_boundary',
        'reached_anxious', 'first_breach_day', 'excluded', 'passed_20_day_filter',
        'days_survived', 'refinement_applied', 'original_upper_boundary',
        'original_lower_boundary', 'refined_upper_boundary', 'refined_lower_boundary',
        'original_channel_width_pct', 'refined_channel_width_pct', 'channel_narrowing_pct',
        'ebp_signal', 'end_price'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Create binary labels for high-value patterns (K3+K4)
    df['is_high_value'] = df['outcome_class'].isin(['K3_STRONG', 'K4_EXCEPTIONAL']).astype(int)

    # Create multi-class labels
    class_mapping = {
        'K0_STAGNANT': 0,
        'K1_MINIMAL': 1,
        'K2_QUALITY': 2,
        'K3_STRONG': 3,
        'K4_EXCEPTIONAL': 4,
        'K5_FAILED': 5
    }
    df['outcome_numeric'] = df['outcome_class'].map(class_mapping)

    # Handle NaN values
    for col in feature_cols:
        if df[col].isnull().any():
            df[col] = df[col].ffill().bfill().fillna(0)

    X = df[feature_cols]
    y_binary = df['is_high_value'].values
    y_multi = df['outcome_numeric'].values

    return X, y_binary, y_multi, feature_cols


def train_model_for_shap(X_train, y_train, model_type='binary'):
    """Train XGBoost model for SHAP analysis."""

    if model_type == 'binary':
        # Binary classification for K3+K4 detection
        # Calculate class weights
        pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            min_child_weight=1,
            gamma=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=-1
        )
    else:
        # Multi-class classification
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=6,
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            min_child_weight=1,
            gamma=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_train, y_train)
    return model


def calculate_shap_values(model, X_train, X_test):
    """Calculate SHAP values for feature importance."""

    logger.info("\nCalculating SHAP values...")

    # Create SHAP explainer
    explainer = shap.Explainer(model, X_train)

    # Calculate SHAP values for test set
    shap_values = explainer(X_test)

    return explainer, shap_values


def analyze_feature_importance(shap_values, X_test, feature_names, y_test=None):
    """Analyze and visualize feature importance."""

    # Get SHAP values array
    if hasattr(shap_values, 'values'):
        if len(shap_values.values.shape) == 3:
            # Multi-class: average across classes
            shap_array = np.abs(shap_values.values).mean(axis=2)
        else:
            shap_array = shap_values.values
    else:
        shap_array = shap_values

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_array).mean(axis=0)

    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)

    logger.info("\n" + "="*70)
    logger.info("FEATURE IMPORTANCE (SHAP)")
    logger.info("="*70)

    # Print top 20 features
    logger.info("\nTop 20 Most Important Features:")
    for i, row in importance_df.head(20).iterrows():
        logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")

    # If we have K3/K4 labels, analyze their specific importance
    if y_test is not None:
        k3k4_mask = np.isin(y_test, [3, 4])
        if k3k4_mask.sum() > 0:
            k3k4_shap = np.abs(shap_array[k3k4_mask]).mean(axis=0)
            k3k4_importance = pd.DataFrame({
                'feature': feature_names,
                'k3k4_importance': k3k4_shap
            }).sort_values('k3k4_importance', ascending=False)

            logger.info("\nTop 10 Features for K3/K4 Detection:")
            for i, row in k3k4_importance.head(10).iterrows():
                logger.info(f"  {row['feature']:30s}: {row['k3k4_importance']:.4f}")

    return importance_df


def select_top_features(importance_df, n_features=20):
    """Select top N features based on SHAP importance."""

    top_features = importance_df.head(n_features)['feature'].tolist()

    logger.info(f"\n{'='*70}")
    logger.info(f"SELECTED TOP {n_features} FEATURES")
    logger.info(f"{'='*70}")

    for i, feature in enumerate(top_features, 1):
        importance = importance_df[importance_df['feature'] == feature]['importance'].values[0]
        logger.info(f"  {i:2d}. {feature:30s}: {importance:.4f}")

    # Group features by category
    ebp_features = [f for f in top_features if any(x in f for x in ['ebp', 'cci', 'var', 'nes', 'lpf', 'tsf'])]
    price_features = [f for f in top_features if 'price' in f or 'boundary' in f or 'distance' in f]
    volatility_features = [f for f in top_features if 'bbw' in f or 'volatility' in f or 'adx' in f or 'atr' in f]
    volume_features = [f for f in top_features if 'volume' in f]
    time_features = [f for f in top_features if 'days' in f or 'period' in f]

    logger.info("\nFeature Categories:")
    logger.info(f"  EBP Features: {len(ebp_features)} - {ebp_features[:5]}")
    logger.info(f"  Price Features: {len(price_features)} - {price_features[:5]}")
    logger.info(f"  Volatility Features: {len(volatility_features)} - {volatility_features[:5]}")
    logger.info(f"  Volume Features: {len(volume_features)} - {volume_features[:5]}")
    logger.info(f"  Time Features: {len(time_features)} - {time_features[:5]}")

    return top_features


def train_reduced_model(X_train, y_train, X_test, y_test, selected_features):
    """Train model with reduced feature set and evaluate."""

    logger.info("\n" + "="*70)
    logger.info("TRAINING MODEL WITH REDUCED FEATURES")
    logger.info("="*70)

    # Use only selected features
    X_train_reduced = X_train[selected_features]
    X_test_reduced = X_test[selected_features]

    # Train model
    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=3,  # Reduced depth for simpler model
        n_estimators=50,  # Fewer trees
        learning_rate=0.1,
        min_child_weight=1,
        gamma=2.0,  # Higher regularization
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_reduced, y_train)

    # Evaluate
    train_score = model.score(X_train_reduced, y_train)
    test_score = model.score(X_test_reduced, y_test)

    # Get predictions for K3+K4
    y_pred = model.predict(X_test_reduced)
    y_pred_proba = model.predict_proba(X_test_reduced)[:, 1]

    # Calculate metrics for K3+K4
    k3k4_mask = (y_test == 1)
    if k3k4_mask.sum() > 0:
        k3k4_detected = y_pred[k3k4_mask].sum()
        k3k4_recall = k3k4_detected / k3k4_mask.sum()

        # False positives
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fp_rate = fp / (y_test == 0).sum()

        logger.info(f"\nModel Performance:")
        logger.info(f"  Training accuracy: {train_score:.3f}")
        logger.info(f"  Test accuracy: {test_score:.3f}")
        logger.info(f"  K3+K4 Recall: {k3k4_recall:.1%} ({k3k4_detected}/{k3k4_mask.sum()})")
        logger.info(f"  False Positive Rate: {fp_rate:.2%}")

    return model, X_train_reduced, X_test_reduced


def save_selected_features(selected_features, output_dir):
    """Save selected features to file."""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save as text file
    features_file = output_dir / f'selected_features_{timestamp}.txt'
    with open(features_file, 'w') as f:
        f.write(f"# Selected Features using SHAP Analysis\n")
        f.write(f"# Generated: {datetime.now()}\n")
        f.write(f"# Total features: {len(selected_features)}\n\n")
        for i, feature in enumerate(selected_features, 1):
            f.write(f"{i:2d}. {feature}\n")

    # Save as Python list for easy import
    py_file = output_dir / f'selected_features_{timestamp}.py'
    with open(py_file, 'w') as f:
        f.write('"""Selected features from SHAP analysis."""\n\n')
        f.write('SELECTED_FEATURES = [\n')
        for feature in selected_features:
            f.write(f'    "{feature}",\n')
        f.write(']\n')

    logger.info(f"\nFeatures saved to:")
    logger.info(f"  {features_file}")
    logger.info(f"  {py_file}")

    return features_file


def main():
    """Main feature selection pipeline."""

    logger.info("\n" + "="*80)
    logger.info("SHAP-BASED FEATURE SELECTION PIPELINE")
    logger.info("="*80)

    # Load data
    labeled_files = list(INPUT_DIR.glob('patterns_labeled_enhanced_*.parquet'))
    if not labeled_files:
        logger.error("No enhanced labeled pattern files found!")
        return

    latest_file = max(labeled_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"\nLoading data from: {latest_file}")

    df = read_data(latest_file)
    logger.info(f"Loaded {len(df):,} labeled patterns")

    # Prepare data
    X, y_binary, y_multi, feature_names = prepare_data_for_shap(df)
    logger.info(f"\nFeature matrix shape: {X.shape}")
    logger.info(f"High-value patterns (K3+K4): {y_binary.sum():,} ({y_binary.mean()*100:.2f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Also split multi-class labels for detailed analysis
    _, _, _, y_test_multi = train_test_split(
        X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )

    logger.info(f"\nTraining set: {X_train.shape[0]:,} samples")
    logger.info(f"Test set: {X_test.shape[0]:,} samples")
    logger.info(f"K3+K4 in test: {y_test.sum()} samples")

    # Train model for SHAP analysis
    logger.info("\nTraining XGBoost model for SHAP analysis...")
    model = train_model_for_shap(X_train, y_train, model_type='binary')

    # Calculate SHAP values
    explainer, shap_values = calculate_shap_values(model, X_train, X_test)

    # Analyze feature importance
    importance_df = analyze_feature_importance(
        shap_values, X_test, feature_names, y_test_multi
    )

    # Select top features
    n_features = 20
    selected_features = select_top_features(importance_df, n_features)

    # Train and evaluate reduced model
    reduced_model, X_train_reduced, X_test_reduced = train_reduced_model(
        X_train, y_train, X_test, y_test, selected_features
    )

    # Save results
    save_selected_features(selected_features, OUTPUT_DIR)

    # Save importance DataFrame
    importance_file = OUTPUT_DIR / f'feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    importance_df.to_csv(importance_file, index=False)
    logger.info(f"  {importance_file}")

    # Create summary plots
    try:
        # SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values.values if hasattr(shap_values, 'values') else shap_values,
            X_test,
            feature_names=feature_names,
            show=False,
            max_display=20
        )
        plt.tight_layout()
        plot_file = OUTPUT_DIR / f'shap_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"  {plot_file}")
        plt.close()

        # Feature importance bar plot
        plt.figure(figsize=(10, 8))
        top_20 = importance_df.head(20)
        plt.barh(range(len(top_20)), top_20['importance'].values)
        plt.yticks(range(len(top_20)), top_20['feature'].values)
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Top 20 Features by SHAP Importance')
        plt.tight_layout()
        bar_plot_file = OUTPUT_DIR / f'feature_importance_bar_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(bar_plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"  {bar_plot_file}")
        plt.close()

    except Exception as e:
        logger.warning(f"Could not create plots: {e}")

    logger.info("\n" + "="*80)
    logger.info("FEATURE SELECTION COMPLETE")
    logger.info("="*80)

    return selected_features, importance_df


if __name__ == "__main__":
    selected_features, importance_df = main()