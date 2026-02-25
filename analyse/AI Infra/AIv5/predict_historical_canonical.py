"""
Generate Predictions Using 69 Canonical Features
=================================================

This script loads the trained models and generates predictions on the
validation dataset using the correct 69 canonical features.

Purpose:
- Fix the feature mismatch between training (69 features) and validation
- Load XGBoost and LightGBM models trained on 69 features
- Generate predictions using matched feature set
- Calculate expected value (EV) from class probabilities

Process:
1. Load trained models from output/models/
2. Load canonical features from pattern_features_canonical_69.parquet
3. Extract only the 69 model features in correct order
4. Generate class predictions and probabilities
5. Calculate expected value using strategic value assignments
6. Save predictions to predictions_canonical.parquet

Expected Runtime: 2-3 minutes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append(str(Path(__file__).parent))

from pattern_detection.features.canonical_feature_extractor import CanonicalFeatureExtractor

# Define PatternValueSystem inline since training.utils doesn't exist
class PatternValueSystem:
    """Strategic value assignments for pattern outcomes."""
    K0_VALUE = -2     # Stagnant
    K1_VALUE = -0.2   # Minimal
    K2_VALUE = 1      # Quality
    K3_VALUE = 3      # Strong
    K4_VALUE = 10     # Exceptional
    K5_VALUE = -10    # Failed

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model paths
XGBOOST_MODEL = Path('output/models/xgboost_extreme_20251101_201450.pkl')
LIGHTGBM_MODEL = Path('output/models/lightgbm_extreme_20251101_201450.pkl')

# Input/Output paths
FEATURES_FILE = Path('output/unused_patterns/pattern_features_canonical_69.parquet')
OUTPUT_FILE = Path('output/unused_patterns/predictions_canonical.parquet')

# Get canonical feature list
# IMPORTANT: Models were trained on 48 features, not the full 69
extractor = CanonicalFeatureExtractor()
all_features = extractor.get_feature_names()
MODEL_EXPECTED_FEATURES = all_features[:48]  # Use only first 48 features to match training

# Strategic values for EV calculation
value_system = PatternValueSystem()

# Class mappings
CLASS_MAP = {
    0: 'K0_STAGNANT',
    1: 'K1_MINIMAL',
    2: 'K2_QUALITY',
    3: 'K3_STRONG',
    4: 'K4_EXCEPTIONAL',
    5: 'K5_FAILED'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_model(model_path: Path):
    """Load a pickled model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        # Handle both direct model and dictionary format
        if isinstance(model_data, dict) and 'model' in model_data:
            return model_data['model']
        return model_data


def calculate_expected_value(probabilities: np.ndarray) -> float:
    """
    Calculate expected value from class probabilities.

    Args:
        probabilities: Array of 6 class probabilities [K0, K1, K2, K3, K4, K5]

    Returns:
        Expected value based on strategic value assignments
    """
    # Strategic values for each class
    values = [
        value_system.K0_VALUE,  # -2
        value_system.K1_VALUE,  # -0.2
        value_system.K2_VALUE,  # +1
        value_system.K3_VALUE,  # +3
        value_system.K4_VALUE,  # +10
        value_system.K5_VALUE   # -10
    ]

    # Calculate weighted sum
    ev = np.sum(probabilities * values)
    return ev


def get_signal_strength(ev: float, failure_prob: float) -> str:
    """
    Determine signal strength based on EV and failure probability.

    Args:
        ev: Expected value
        failure_prob: Probability of K5_FAILED outcome

    Returns:
        Signal strength category
    """
    # Check failure probability threshold
    if failure_prob > 0.30:
        return 'AVOID'

    # Categorize by EV
    if ev >= 5.0:
        return 'STRONG_SIGNAL'
    elif ev >= 3.0:
        return 'GOOD_SIGNAL'
    elif ev >= 1.0:
        return 'MODERATE_SIGNAL'
    elif ev >= 0:
        return 'WEAK_SIGNAL'
    else:
        return 'AVOID'


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CANONICAL PREDICTION GENERATION")
    print("Using 69 Matched Features")
    print("=" * 80)

    # Check if models exist
    if not XGBOOST_MODEL.exists():
        print(f"\n[ERROR] ERROR: XGBoost model not found: {XGBOOST_MODEL}")
        exit(1)
    if not LIGHTGBM_MODEL.exists():
        print(f"\n[ERROR] ERROR: LightGBM model not found: {LIGHTGBM_MODEL}")
        exit(1)

    # Check if features file exists
    if not FEATURES_FILE.exists():
        print(f"\n[ERROR] ERROR: Features file not found: {FEATURES_FILE}")
        print("Please run extract_features_historical_enhanced.py first")
        exit(1)

    # Load models
    print(f"\n[STEP 1] Loading trained models...")
    xgb_model = load_model(XGBOOST_MODEL)
    lgb_model = load_model(LIGHTGBM_MODEL)
    print(f"  [OK] XGBoost model loaded")
    print(f"  [OK] LightGBM model loaded")

    # Load features
    print(f"\n[STEP 2] Loading canonical features...")
    features_df = pd.read_parquet(FEATURES_FILE)
    print(f"  Total patterns: {len(features_df):,}")

    # Extract only the 69 model features
    print(f"\n[STEP 3] Extracting model features...")
    available_features = [col for col in MODEL_EXPECTED_FEATURES if col in features_df.columns]
    missing_features = set(MODEL_EXPECTED_FEATURES) - set(available_features)

    print(f"  Expected features: {len(MODEL_EXPECTED_FEATURES)}")
    print(f"  Available features: {len(available_features)}")

    if missing_features:
        print(f"  [WARNING] WARNING: Missing {len(missing_features)} features: {list(missing_features)[:5]}...")
        # Fill missing features with appropriate defaults
        for feature in missing_features:
            if 'ratio' in feature or 'pct' in feature:
                features_df[feature] = 1.0
            elif 'days' in feature or 'count' in feature:
                features_df[feature] = 0
            else:
                features_df[feature] = 0.0

    # Ensure features are in correct order
    X = features_df[MODEL_EXPECTED_FEATURES].values
    print(f"  Feature matrix shape: {X.shape}")

    # Generate predictions
    print(f"\n[STEP 4] Generating predictions...")

    # XGBoost predictions
    xgb_proba = xgb_model.predict_proba(X)
    xgb_pred = xgb_model.predict(X)

    # LightGBM predictions
    lgb_proba = lgb_model.predict_proba(X)
    lgb_pred = lgb_model.predict(X)

    # Ensemble predictions (average probabilities)
    ensemble_proba = (xgb_proba + lgb_proba) / 2
    ensemble_pred = np.argmax(ensemble_proba, axis=1)

    print(f"  [OK] XGBoost predictions generated")
    print(f"  [OK] LightGBM predictions generated")
    print(f"  [OK] Ensemble predictions computed")

    # Calculate expected values
    print(f"\n[STEP 5] Calculating expected values...")

    xgb_ev = np.array([calculate_expected_value(probs) for probs in xgb_proba])
    lgb_ev = np.array([calculate_expected_value(probs) for probs in lgb_proba])
    ensemble_ev = np.array([calculate_expected_value(probs) for probs in ensemble_proba])

    # Create results DataFrame
    results_df = pd.DataFrame({
        # Metadata
        'ticker': features_df['ticker'],
        'snapshot_date': features_df['snapshot_date'],
        'actual_class': features_df.get('actual_class', 'UNKNOWN'),

        # XGBoost predictions
        'xgb_predicted_class': [CLASS_MAP[p] for p in xgb_pred],
        'xgb_k0_prob': xgb_proba[:, 0],
        'xgb_k1_prob': xgb_proba[:, 1],
        'xgb_k2_prob': xgb_proba[:, 2],
        'xgb_k3_prob': xgb_proba[:, 3],
        'xgb_k4_prob': xgb_proba[:, 4],
        'xgb_k5_prob': xgb_proba[:, 5],
        'xgb_expected_value': xgb_ev,

        # LightGBM predictions
        'lgb_predicted_class': [CLASS_MAP[p] for p in lgb_pred],
        'lgb_k0_prob': lgb_proba[:, 0],
        'lgb_k1_prob': lgb_proba[:, 1],
        'lgb_k2_prob': lgb_proba[:, 2],
        'lgb_k3_prob': lgb_proba[:, 3],
        'lgb_k4_prob': lgb_proba[:, 4],
        'lgb_k5_prob': lgb_proba[:, 5],
        'lgb_expected_value': lgb_ev,

        # Ensemble predictions
        'ensemble_predicted_class': [CLASS_MAP[p] for p in ensemble_pred],
        'ensemble_k0_prob': ensemble_proba[:, 0],
        'ensemble_k1_prob': ensemble_proba[:, 1],
        'ensemble_k2_prob': ensemble_proba[:, 2],
        'ensemble_k3_prob': ensemble_proba[:, 3],
        'ensemble_k4_prob': ensemble_proba[:, 4],
        'ensemble_k5_prob': ensemble_proba[:, 5],
        'ensemble_expected_value': ensemble_ev,
    })

    # Add signal strength
    results_df['xgb_signal'] = results_df.apply(
        lambda x: get_signal_strength(x['xgb_expected_value'], x['xgb_k5_prob']), axis=1
    )
    results_df['lgb_signal'] = results_df.apply(
        lambda x: get_signal_strength(x['lgb_expected_value'], x['lgb_k5_prob']), axis=1
    )
    results_df['ensemble_signal'] = results_df.apply(
        lambda x: get_signal_strength(x['ensemble_expected_value'], x['ensemble_k5_prob']), axis=1
    )

    # Save predictions
    print(f"\n[STEP 6] Saving predictions...")
    results_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"  Saved to: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

    # Display summary statistics
    print(f"\n[STEP 7] Prediction Summary:")
    print("-" * 50)

    # Class distribution
    print("\nEnsemble predicted class distribution:")
    print(results_df['ensemble_predicted_class'].value_counts())

    # Signal distribution
    print("\nEnsemble signal distribution:")
    print(results_df['ensemble_signal'].value_counts())

    # EV statistics
    print(f"\nExpected Value statistics (Ensemble):")
    print(f"  Mean: {ensemble_ev.mean():.2f}")
    print(f"  Median: {np.median(ensemble_ev):.2f}")
    print(f"  Max: {ensemble_ev.max():.2f}")
    print(f"  Min: {ensemble_ev.min():.2f}")
    print(f"  Positive EV: {(ensemble_ev > 0).sum()}/{len(ensemble_ev)} ({(ensemble_ev > 0).mean()*100:.1f}%)")

    # K4 detection
    k4_detected = (ensemble_proba[:, 4] > 0.1).sum()
    print(f"\nK4 Detection (>10% probability): {k4_detected}/{len(ensemble_proba)} patterns")

    print("\n" + "=" * 80)
    print("PREDICTION GENERATION COMPLETE!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("Next step: Run validation report to compare with actual outcomes")
    print("=" * 80)