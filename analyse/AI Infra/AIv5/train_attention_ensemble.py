"""
Train Attention-Based Ensemble with V3 Features
===============================================
Integrates advanced feature engineering, attention mechanism,
Optuna optimization, and character rating system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from feature_engineering_v3 import AdvancedFeatureEngineerV3
from attention_xgboost_ensemble import AttentionXGBoostEnsemble
# from data_acquisition.storage.storage_manager import StorageManager


def extract_v3_features(df_patterns: pd.DataFrame) -> pd.DataFrame:
    """
    Extract V3 features for all patterns.
    """
    print("\n" + "="*60)
    print("EXTRACTING V3 FEATURES")
    print("="*60)

    feature_engineer = AdvancedFeatureEngineerV3()
    # storage_manager = StorageManager()

    all_features = []

    # Group by ticker for efficiency
    ticker_groups = df_patterns.groupby('ticker')
    total_patterns = len(df_patterns)
    processed = 0

    for ticker, group in ticker_groups:
        # Load ticker data
        try:
            # Try multiple data locations
            ticker_file = None
            for dir_path in ['data/gcs_tickers_massive/tickers', 'data/massive_gcs_cache/tickers', 'data_acquisition/storage']:
                file_path = Path(dir_path) / f"{ticker}.parquet"
                if file_path.exists():
                    ticker_file = file_path
                    break

            if ticker_file is None:
                continue

            ticker_data = pd.read_parquet(ticker_file)
            if ticker_data is None or ticker_data.empty:
                continue

            # Process each pattern
            for idx, pattern_row in group.iterrows():
                pattern = {
                    'upper_boundary': pattern_row['upper_boundary'],
                    'lower_boundary': pattern_row['lower_boundary'],
                    'power_boundary': pattern_row.get('power_boundary',
                                                     pattern_row['upper_boundary'] * 1.005),
                    'days_in_pattern': pattern_row.get('days_in_pattern', 0),
                    'days_qualifying': pattern_row.get('days_qualifying', 10),
                    'days_active': pattern_row.get('days_active', 0)
                }

                snapshot_date = pd.to_datetime(pattern_row['snapshot_date'])

                # Extract features
                features = feature_engineer.extract_all_features(
                    ticker_data, pattern, snapshot_date
                )

                # Add pattern metadata
                features['ticker'] = ticker
                features['snapshot_date'] = snapshot_date
                features['outcome_class'] = pattern_row.get('outcome_class', 'UNKNOWN')

                all_features.append(features)

                processed += 1
                if processed % 1000 == 0:
                    print(f"Processed {processed}/{total_patterns} patterns...")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # Combine all features
    df_features = pd.concat(all_features, ignore_index=True)
    print(f"\nExtracted features for {len(df_features)} patterns")
    print(f"Total features: {len(df_features.columns) - 3}")  # Exclude metadata

    return df_features


def prepare_training_data(df_features: pd.DataFrame):
    """
    Prepare data for training.
    """
    print("\n" + "="*60)
    print("PREPARING TRAINING DATA")
    print("="*60)

    # Encode labels
    label_encoder = LabelEncoder()
    outcome_mapping = {
        'K0_STAGNANT': 0,
        'K1_MINIMAL': 1,
        'K2_QUALITY': 2,
        'K3_STRONG': 3,
        'K4_EXCEPTIONAL': 4,
        'K5_FAILED': 5
    }

    df_features['label'] = df_features['outcome_class'].map(outcome_mapping)

    # Remove rows with unknown outcomes
    df_features = df_features.dropna(subset=['label'])
    df_features['label'] = df_features['label'].astype(int)

    # Separate features and labels
    feature_cols = [col for col in df_features.columns
                   if col not in ['ticker', 'snapshot_date', 'outcome_class', 'label']]

    X = df_features[feature_cols]
    y = df_features['label'].values

    # Handle missing values
    X = X.fillna(0)

    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X)}")

    # Class distribution
    print("\nClass distribution:")
    for class_id, class_name in enumerate(['K0', 'K1', 'K2', 'K3', 'K4', 'K5']):
        count = (y == class_id).sum()
        pct = count / len(y) * 100
        print(f"  {class_name}: {count} ({pct:.2f}%)")

    return X, y, feature_cols


def main():
    """
    Main training pipeline.
    """
    print("="*60)
    print("ATTENTION-BASED ENSEMBLE TRAINING WITH V3 FEATURES")
    print("="*60)

    # Load race-labeled dataset
    pattern_file = Path('output/patterns_labeled_enhanced_20251104_193956.parquet')

    if not pattern_file.exists():
        print(f"Error: {pattern_file} not found!")
        print("Please run the race-based labeling pipeline first.")
        return

    print(f"\nLoading patterns from: {pattern_file}")
    df_patterns = pd.read_parquet(pattern_file)
    print(f"Loaded {len(df_patterns)} patterns")

    # Option 1: Extract V3 features (takes time)
    extract_new = input("\nExtract new V3 features? (y/n): ").lower() == 'y'

    if extract_new:
        df_features = extract_v3_features(df_patterns)
        # Save extracted features
        df_features.to_parquet('output/features_v3_enhanced.parquet', index=False)
        print("Saved V3 features to output/features_v3_enhanced.parquet")
    else:
        # Load existing features if available
        feature_file = Path('output/features_v3_enhanced.parquet')
        if feature_file.exists():
            print(f"Loading existing features from {feature_file}")
            df_features = pd.read_parquet(feature_file)
        else:
            print("No existing features found. Extracting new features...")
            df_features = extract_v3_features(df_patterns)
            df_features.to_parquet('output/features_v3_enhanced.parquet', index=False)

    # Prepare training data
    X, y, feature_names = prepare_training_data(df_features)

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Initialize ensemble
    ensemble = AttentionXGBoostEnsemble(
        n_trials=50,  # Use 100+ for production
        use_gpu=False  # Set to True if GPU available
    )

    # Train with Optuna optimization
    optimize = input("\nRun Optuna optimization? (y/n): ").lower() == 'y'

    ensemble.train(
        X_train, y_train,
        X_val, y_val,
        optimize=optimize,
        feature_selection_method='importance'
    )

    # Evaluate on test set
    ensemble.evaluate(X_test, y_test)

    # Save models
    ensemble.save_models('output/models/ensemble')

    # Generate predictions with character ratings
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS WITH CHARACTER RATINGS")
    print("="*60)

    predictions, probabilities, character_ratings = ensemble.predict_with_character_rating(X_test)

    # Create results dataframe
    results = pd.DataFrame({
        'prediction': predictions,
        'character_rating': character_ratings,
        'k4_probability': probabilities[:, 4] if probabilities.shape[1] > 4 else 0,
        'k3_probability': probabilities[:, 3] if probabilities.shape[1] > 3 else 0,
        'actual': y_test
    })

    # Find high-potential patterns
    high_potential = results[
        (results['character_rating'] >= 8) |
        (results['k4_probability'] >= 0.3) |
        (results['k3_probability'] >= 0.5)
    ]

    print(f"\nHigh-Potential Patterns: {len(high_potential)}")
    print(f"  Average character rating: {high_potential['character_rating'].mean():.2f}")
    print(f"  Average K4 probability: {high_potential['k4_probability'].mean():.2%}")
    print(f"  Average K3 probability: {high_potential['k3_probability'].mean():.2%}")

    # Check actual outcomes
    k4_actual = (high_potential['actual'] == 4).sum()
    k3_actual = (high_potential['actual'] == 3).sum()
    print(f"\nActual outcomes in high-potential patterns:")
    print(f"  K4: {k4_actual} ({k4_actual/len(high_potential):.1%})")
    print(f"  K3: {k3_actual} ({k3_actual/len(high_potential):.1%})")

    # Save results
    results.to_csv('output/ensemble_predictions_v3.csv', index=False)
    print(f"\nResults saved to output/ensemble_predictions_v3.csv")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

    # Display feature importance
    if ensemble.attention_weights is not None:
        print("\nTop 20 Features by Attention Weight:")
        feature_importance = pd.DataFrame({
            'feature': ensemble.selected_features,
            'attention_weight': ensemble.attention_weights
        }).sort_values('attention_weight', ascending=False).head(20)

        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['attention_weight']:.4f}")


if __name__ == "__main__":
    main()