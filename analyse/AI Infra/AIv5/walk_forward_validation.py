"""
Walk-Forward Validation
=======================
Simulates production deployment with quarterly retraining.
Tests model performance on truly unseen future data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


class WalkForwardValidator:
    """
    Performs walk-forward validation with quarterly retraining.
    """

    def __init__(self, retrain_frequency_days=90):
        self.retrain_frequency = retrain_frequency_days
        self.results = []

    def prepare_features(self, df, feature_names):
        """Extract features and labels."""
        # Use original labels (not enhanced) for true validation
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

        X = df[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        y = df['label'].values.astype(int)

        return X, y, df

    def prepare_combined_labels(self, df, feature_names):
        """Prepare BIG_WINNER combined labels."""
        def map_to_combined(label):
            if label in ['K3_STRONG', 'K4_EXCEPTIONAL']:
                return 'BIG_WINNER'
            elif label == 'K2_QUALITY':
                return 'MODERATE'
            elif label in ['K0_STAGNANT', 'K1_MINIMAL']:
                return 'WEAK'
            else:
                return 'FAILED'

        df['combined_label'] = df['outcome_class'].apply(map_to_combined)

        combined_mapping = {
            'WEAK': 0,
            'MODERATE': 1,
            'BIG_WINNER': 2,
            'FAILED': 3
        }

        df['label'] = df['combined_label'].map(combined_mapping)
        df = df.dropna(subset=['label'])

        X = df[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        y = df['label'].values.astype(int)

        return X, y, df

    def train_model(self, X_train, y_train, num_classes, model_type='lightgbm'):
        """Train a model with proper class weighting."""
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = {i: w for i, w in zip(classes, weights)}

        # Apply extra weighting for rare classes
        if num_classes == 6:  # Standard
            if 4 in weight_dict:
                weight_dict[4] *= 3
            if 3 in weight_dict:
                weight_dict[3] *= 2
        else:  # Combined
            if 2 in weight_dict:
                weight_dict[2] *= 2.5

        sample_weights = np.array([weight_dict[y] for y in y_train])

        if model_type == 'lightgbm':
            model = LGBMClassifier(
                n_estimators=300,
                num_leaves=40,
                learning_rate=0.05,
                min_data_in_leaf=20,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                verbosity=-1
            )
        else:
            model = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist',
                random_state=42
            )

        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model

    def evaluate_window(self, model, X_test, y_test, window_name, num_classes):
        """Evaluate model on a validation window."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        results = {
            'window': window_name,
            'samples': len(y_test),
            'predictions': y_pred,
            'probabilities': y_proba,
            'actuals': y_test
        }

        if num_classes == 6:
            # K4 metrics
            k4_mask = y_test == 4
            if k4_mask.sum() > 0:
                results['k4_count'] = k4_mask.sum()
                results['k4_recall'] = (y_pred[k4_mask] == 4).mean()
                results['k4_precision'] = (y_test[y_pred == 4] == 4).mean() if (y_pred == 4).sum() > 0 else 0

            # K3 metrics
            k3_mask = y_test == 3
            if k3_mask.sum() > 0:
                results['k3_count'] = k3_mask.sum()
                results['k3_recall'] = (y_pred[k3_mask] == 3).mean()

            # K3+K4 combined
            k34_mask = (y_test == 3) | (y_test == 4)
            if k34_mask.sum() > 0:
                k34_pred = ((y_pred == 3) | (y_pred == 4)) & k34_mask
                results['k34_recall'] = k34_pred.sum() / k34_mask.sum()
                results['k34_count'] = k34_mask.sum()

        else:  # Combined
            winner_mask = y_test == 2
            if winner_mask.sum() > 0:
                results['winner_count'] = winner_mask.sum()
                results['winner_recall'] = (y_pred[winner_mask] == 2).mean()
                results['winner_precision'] = (y_test[y_pred == 2] == 2).mean() if (y_pred == 2).sum() > 0 else 0

        # Overall accuracy
        results['accuracy'] = (y_pred == y_test).mean()

        return results

    def run_walk_forward(self, df, feature_names, model_type='combined'):
        """
        Run walk-forward validation.
        """
        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION")
        print("="*60)
        print(f"Model type: {model_type}")
        print(f"Retrain frequency: {self.retrain_frequency} days")

        # Sort by date
        df = df.sort_values('snapshot_date')
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

        # Get date range
        min_date = df['snapshot_date'].min()
        max_date = df['snapshot_date'].max()

        print(f"Date range: {min_date.date()} to {max_date.date()}")
        print(f"Total patterns: {len(df)}")

        # Define training windows
        # Start with 2 years of training data
        initial_train_end = min_date + timedelta(days=730)

        current_train_end = initial_train_end
        window_num = 1

        all_results = []

        while current_train_end < max_date:
            # Define validation window (next 90 days)
            val_start = current_train_end
            val_end = val_start + timedelta(days=self.retrain_frequency)

            if val_end > max_date:
                val_end = max_date

            # Get training data (all data up to val_start)
            train_mask = df['snapshot_date'] < val_start
            val_mask = (df['snapshot_date'] >= val_start) & (df['snapshot_date'] < val_end)

            df_train = df[train_mask]
            df_val = df[val_mask]

            if len(df_val) < 50:  # Skip if too few validation samples
                print(f"\nWindow {window_num}: Skipping (only {len(df_val)} samples)")
                current_train_end = val_end
                window_num += 1
                continue

            print(f"\n{'='*60}")
            print(f"WINDOW {window_num}")
            print(f"{'='*60}")
            print(f"Training: {min_date.date()} to {val_start.date()} ({len(df_train)} samples)")
            print(f"Validation: {val_start.date()} to {val_end.date()} ({len(df_val)} samples)")

            # Prepare data
            if model_type == 'combined':
                X_train, y_train, _ = self.prepare_combined_labels(df_train, feature_names)
                X_val, y_val, _ = self.prepare_combined_labels(df_val, feature_names)
                num_classes = 4
            else:
                X_train, y_train, _ = self.prepare_features(df_train, feature_names)
                X_val, y_val, _ = self.prepare_features(df_val, feature_names)
                num_classes = 6

            # Train model
            print(f"Training model on {len(X_train)} samples...")
            model = self.train_model(X_train, y_train, num_classes)

            # Evaluate
            window_name = f"Window_{window_num}_{val_start.strftime('%Y%m%d')}"
            results = self.evaluate_window(model, X_val, y_val, window_name, num_classes)

            # Print results
            print(f"\nValidation Results:")
            print(f"  Samples: {results['samples']}")
            print(f"  Accuracy: {results['accuracy']:.1%}")

            if model_type == 'combined':
                if 'winner_recall' in results:
                    print(f"  BIG_WINNER Recall: {results['winner_recall']:.1%} ({results['winner_count']} samples)")
                    print(f"  BIG_WINNER Precision: {results['winner_precision']:.1%}")
            else:
                if 'k4_recall' in results:
                    print(f"  K4 Recall: {results['k4_recall']:.1%} ({results['k4_count']} samples)")
                    print(f"  K4 Precision: {results['k4_precision']:.1%}")
                if 'k3_recall' in results:
                    print(f"  K3 Recall: {results['k3_recall']:.1%} ({results['k3_count']} samples)")
                if 'k34_recall' in results:
                    print(f"  K3+K4 Recall: {results['k34_recall']:.1%} ({results['k34_count']} samples)")

            all_results.append(results)

            # Move to next window
            current_train_end = val_end
            window_num += 1

        return all_results

    def print_summary(self, results, model_type):
        """Print summary statistics across all windows."""
        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("="*60)

        print(f"\nTotal validation windows: {len(results)}")

        if model_type == 'combined':
            # BIG_WINNER metrics
            winner_recalls = [r['winner_recall'] for r in results if 'winner_recall' in r]
            winner_precisions = [r['winner_precision'] for r in results if 'winner_precision' in r]

            if winner_recalls:
                print(f"\nBIG_WINNER Performance:")
                print(f"  Average Recall: {np.mean(winner_recalls):.1%} ± {np.std(winner_recalls):.1%}")
                print(f"  Median Recall: {np.median(winner_recalls):.1%}")
                print(f"  Min/Max Recall: {np.min(winner_recalls):.1%} / {np.max(winner_recalls):.1%}")
                print(f"  Average Precision: {np.mean(winner_precisions):.1%}")

        else:
            # K4 metrics
            k4_recalls = [r['k4_recall'] for r in results if 'k4_recall' in r]
            k3_recalls = [r['k3_recall'] for r in results if 'k3_recall' in r]
            k34_recalls = [r['k34_recall'] for r in results if 'k34_recall' in r]

            if k4_recalls:
                print(f"\nK4 Performance:")
                print(f"  Average Recall: {np.mean(k4_recalls):.1%} ± {np.std(k4_recalls):.1%}")
                print(f"  Median Recall: {np.median(k4_recalls):.1%}")
                print(f"  Min/Max Recall: {np.min(k4_recalls):.1%} / {np.max(k4_recalls):.1%}")

            if k3_recalls:
                print(f"\nK3 Performance:")
                print(f"  Average Recall: {np.mean(k3_recalls):.1%} ± {np.std(k3_recalls):.1%}")

            if k34_recalls:
                print(f"\nK3+K4 Combined Performance:")
                print(f"  Average Recall: {np.mean(k34_recalls):.1%} ± {np.std(k34_recalls):.1%}")

        # Overall accuracy
        accuracies = [r['accuracy'] for r in results]
        print(f"\nOverall Accuracy:")
        print(f"  Average: {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}")

        print("\n" + "="*60)
        print("PRODUCTION READINESS ASSESSMENT")
        print("="*60)

        if model_type == 'combined' and winner_recalls:
            avg_recall = np.mean(winner_recalls)
            std_recall = np.std(winner_recalls)

            print(f"\nBIG_WINNER Detection:")
            print(f"  Expected recall: {avg_recall:.1%}")
            print(f"  Consistency (std): {std_recall:.1%}")

            if avg_recall > 0.60 and std_recall < 0.15:
                print("  Status: ✓ READY FOR PRODUCTION")
                print("  Consistent performance with good detection rate")
            elif avg_recall > 0.50:
                print("  Status: ~ ACCEPTABLE")
                print("  Moderate performance, monitor closely")
            else:
                print("  Status: X NEEDS IMPROVEMENT")
                print("  Performance below production threshold")

        elif k34_recalls:
            avg_recall = np.mean(k34_recalls)
            std_recall = np.std(k34_recalls)

            print(f"\nK3+K4 Detection:")
            print(f"  Expected recall: {avg_recall:.1%}")
            print(f"  Consistency (std): {std_recall:.1%}")

            if avg_recall > 0.50 and std_recall < 0.15:
                print("  Status: ✓ READY FOR PRODUCTION")
            elif avg_recall > 0.40:
                print("  Status: ~ ACCEPTABLE")
            else:
                print("  Status: X NEEDS IMPROVEMENT")


def main():
    """Main walk-forward validation pipeline."""
    print("="*60)
    print("WALK-FORWARD VALIDATION PIPELINE")
    print("="*60)
    print("Simulating production deployment with quarterly retraining")

    # Load dataset (use original labels for true validation)
    data_file = Path("output/patterns_labeled_enhanced_20251104_193956.parquet")
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return

    print(f"\nLoading data from {data_file}")
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df)} patterns")

    # Safe features (no leakage)
    safe_features = [
        "days_since_activation", "days_in_pattern", "current_price",
        "current_high", "current_low", "current_volume",
        "current_bbw_20", "current_bbw_percentile", "current_adx",
        "current_volume_ratio_20", "current_range_ratio",
        "baseline_bbw_avg", "baseline_volume_avg", "baseline_volatility",
        "bbw_compression_ratio", "volume_compression_ratio", "volatility_compression_ratio",
        "bbw_slope_20d", "adx_slope_20d", "consolidation_quality_score",
        "price_position_in_range", "price_distance_from_upper_pct",
        "price_distance_from_lower_pct", "distance_from_power_pct",
        "avg_range_20d", "bbw_std_20d", "price_volatility_20d", "start_price"
    ]

    available_features = [f for f in safe_features if f in df.columns]
    print(f"Using {len(available_features)} features")

    # Initialize validator
    validator = WalkForwardValidator(retrain_frequency_days=90)

    # Run validation for combined model (recommended)
    print("\n" + "="*70)
    print("VALIDATING COMBINED (BIG_WINNER) MODEL")
    print("="*70)

    results_combined = validator.run_walk_forward(df, available_features, model_type='combined')
    validator.print_summary(results_combined, 'combined')

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"output/walk_forward_results_{timestamp}.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump({
            'results': results_combined,
            'model_type': 'combined',
            'features': available_features
        }, f)
    print(f"\nResults saved to: {results_file}")

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()