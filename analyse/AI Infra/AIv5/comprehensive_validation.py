"""
Comprehensive Validation Framework
===================================
Tests model performance on various unseen datasets to detect
oversampling risks and validate realistic performance metrics.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveValidator:
    """
    Validates model performance across multiple dimensions:
    1. Temporal validation (different time periods)
    2. Ticker validation (unseen tickers)
    3. Cross-validation folds
    4. Data leakage detection
    5. Oversampling risk assessment
    """

    def __init__(self, model_path: str):
        """Load the trained model."""
        with open(model_path, 'rb') as f:
            loaded_obj = pickle.load(f)
            # Handle both direct model and dict format
            if isinstance(loaded_obj, dict):
                # If it's a dict, look for the model in common keys
                if 'model' in loaded_obj:
                    self.model = loaded_obj['model']
                    self.feature_names = loaded_obj.get('feature_names', None)
                elif 'classifier' in loaded_obj:
                    self.model = loaded_obj['classifier']
                    self.feature_names = loaded_obj.get('feature_names', None)
                else:
                    # Try to find the model object
                    for key, value in loaded_obj.items():
                        if hasattr(value, 'predict'):
                            self.model = value
                            self.feature_names = loaded_obj.get('feature_names', None)
                            break
                    else:
                        raise ValueError(f"Could not find model in dict: {loaded_obj.keys()}")
            else:
                self.model = loaded_obj
                self.feature_names = None
        print(f"Loaded model from {model_path}")
        if self.feature_names:
            print(f"Using {len(self.feature_names)} features from model")

    def validate_temporal_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Test model on different time periods to ensure no look-ahead bias.
        """
        print("\n" + "="*60)
        print("TEMPORAL VALIDATION - Testing Different Time Periods")
        print("="*60)

        results = {}

        # Split data by time periods
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

        periods = {
            '2020-2021': (df['snapshot_date'] >= '2020-01-01') & (df['snapshot_date'] < '2022-01-01'),
            '2022': (df['snapshot_date'] >= '2022-01-01') & (df['snapshot_date'] < '2023-01-01'),
            '2023': (df['snapshot_date'] >= '2023-01-01') & (df['snapshot_date'] < '2024-01-01'),
            '2024': (df['snapshot_date'] >= '2024-01-01')
        }

        for period_name, mask in periods.items():
            period_data = df[mask]
            if len(period_data) < 100:
                continue

            print(f"\n[{period_name}] Samples: {len(period_data)}")

            # Prepare features
            X = self._prepare_features(period_data)
            y_true = period_data['label'].values

            # Predict
            y_pred = self.model.predict(X)

            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred, period_data)
            results[period_name] = metrics

            # Print key metrics
            print(f"  K4 Recall: {metrics['k4_recall']:.1%}")
            print(f"  K3 Recall: {metrics['k3_recall']:.1%}")
            print(f"  K5 Precision: {metrics['k5_precision']:.1%}")

        return results

    def validate_unseen_tickers(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Test on tickers that were never seen during training.
        """
        print("\n" + "="*60)
        print("UNSEEN TICKER VALIDATION - Testing Generalization")
        print("="*60)

        # Find tickers only in test set
        train_tickers = set(train_df['ticker'].unique())
        test_tickers = set(test_df['ticker'].unique())
        unseen_tickers = test_tickers - train_tickers

        print(f"Training tickers: {len(train_tickers)}")
        print(f"Test tickers: {len(test_tickers)}")
        print(f"Unseen tickers: {len(unseen_tickers)}")

        # Test on unseen tickers
        unseen_data = test_df[test_df['ticker'].isin(unseen_tickers)]

        if len(unseen_data) < 50:
            print("Insufficient unseen ticker data")
            return {}

        print(f"\nTesting on {len(unseen_data)} patterns from unseen tickers")

        # Prepare and predict
        X = self._prepare_features(unseen_data)
        y_true = unseen_data['label'].values
        y_pred = self.model.predict(X)

        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, unseen_data)

        print(f"\nUnseen Ticker Performance:")
        print(f"  K4 Recall: {metrics['k4_recall']:.1%}")
        print(f"  K3 Recall: {metrics['k3_recall']:.1%}")
        print(f"  K3+K4 Combined: {metrics['k34_recall']:.1%}")
        print(f"  K5 Precision: {metrics['k5_precision']:.1%}")

        return metrics

    def detect_data_leakage(self, df: pd.DataFrame) -> Dict:
        """
        Check for potential data leakage by analyzing feature correlations.
        """
        print("\n" + "="*60)
        print("DATA LEAKAGE DETECTION")
        print("="*60)

        leakage_indicators = {}

        # Check for future-looking features
        suspicious_features = []
        feature_cols = self._get_feature_columns(df)

        for col in feature_cols:
            if any(term in col.lower() for term in ['future', 'outcome', 'gain', 'peak', 'max_price']):
                suspicious_features.append(col)

        if suspicious_features:
            print(f"WARNING: Potential future-looking features detected:")
            for feat in suspicious_features:
                print(f"  - {feat}")
            leakage_indicators['suspicious_features'] = suspicious_features
        else:
            print("No obvious future-looking features detected")

        # Check feature-label correlation
        print("\nFeature-Label Correlation Analysis:")
        X = self._prepare_features(df)
        y = df['label'].values

        high_corr_features = []
        for i, col in enumerate(feature_cols[:len(X.columns)]):
            corr = np.corrcoef(X.iloc[:, i], y)[0, 1]
            if abs(corr) > 0.7:  # High correlation threshold
                high_corr_features.append((col, corr))
                print(f"  HIGH CORRELATION: {col} = {corr:.3f}")

        leakage_indicators['high_correlation_features'] = high_corr_features

        return leakage_indicators

    def assess_oversampling_risk(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Check if model is overfitting to oversampled K4 patterns.
        """
        print("\n" + "="*60)
        print("OVERSAMPLING RISK ASSESSMENT")
        print("="*60)

        # Check class distribution
        train_dist = train_df['outcome_class'].value_counts(normalize=True)
        test_dist = test_df['outcome_class'].value_counts(normalize=True)

        print("\nClass Distribution Comparison:")
        print("Class          Train%    Test%    Difference")
        print("-" * 50)

        risk_metrics = {}
        for class_name in ['K4_EXCEPTIONAL', 'K3_STRONG', 'K5_FAILED']:
            train_pct = train_dist.get(class_name, 0) * 100
            test_pct = test_dist.get(class_name, 0) * 100
            diff = train_pct - test_pct
            print(f"{class_name:15s} {train_pct:6.2f}%  {test_pct:6.2f}%  {diff:+7.2f}%")

            if class_name == 'K4_EXCEPTIONAL' and diff > 0.5:
                risk_metrics['k4_oversampling'] = True
                print(f"  WARNING: K4 may be oversampled in training")

        # Test performance degradation
        print("\nPerformance on Original vs Augmented K4s:")

        # Separate original and potentially augmented K4s
        k4_patterns = test_df[test_df['outcome_class'] == 'K4_EXCEPTIONAL']
        if len(k4_patterns) > 0:
            # Assume patterns with very similar features might be augmented
            X_k4 = self._prepare_features(k4_patterns)
            y_k4_true = np.full(len(k4_patterns), 4)  # K4 label
            y_k4_pred = self.model.predict(X_k4)

            k4_recall = (y_k4_pred == 4).mean()
            print(f"  Overall K4 Recall: {k4_recall:.1%}")

            # Check for suspiciously high recall
            if k4_recall > 0.95:
                risk_metrics['suspiciously_high_recall'] = True
                print(f"  WARNING: K4 recall >95% suggests possible overfitting")

        return risk_metrics

    def run_cross_validation(self, df: pd.DataFrame, n_folds: int = 5) -> Dict:
        """
        Run k-fold cross-validation to assess model stability.
        """
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION - {n_folds} Folds")
        print("="*60)

        from sklearn.model_selection import StratifiedKFold

        X = self._prepare_features(df)
        y = df['label'].values

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_val = X.iloc[val_idx]
            y_val = y[val_idx]

            y_pred = self.model.predict(X_val)

            # Calculate K4 and K3 recall
            k4_mask = y_val == 4
            k3_mask = y_val == 3

            k4_recall = (y_pred[k4_mask] == 4).mean() if k4_mask.sum() > 0 else 0
            k3_recall = (y_pred[k3_mask] == 3).mean() if k3_mask.sum() > 0 else 0

            fold_results.append({
                'fold': fold,
                'k4_recall': k4_recall,
                'k3_recall': k3_recall,
                'k4_samples': k4_mask.sum(),
                'k3_samples': k3_mask.sum()
            })

            print(f"Fold {fold}: K4={k4_recall:.1%} ({k4_mask.sum()} samples), "
                  f"K3={k3_recall:.1%} ({k3_mask.sum()} samples)")

        # Calculate statistics
        k4_recalls = [r['k4_recall'] for r in fold_results if r['k4_samples'] > 0]
        k3_recalls = [r['k3_recall'] for r in fold_results if r['k3_samples'] > 0]

        print(f"\nCross-Validation Summary:")
        print(f"  K4 Mean: {np.mean(k4_recalls):.1%} +/- {np.std(k4_recalls):.1%}")
        print(f"  K3 Mean: {np.mean(k3_recalls):.1%} +/- {np.std(k3_recalls):.1%}")

        # Check for high variance (overfitting indicator)
        if np.std(k4_recalls) > 0.2:
            print("  WARNING: High K4 variance suggests instability")

        return fold_results

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix from dataframe."""
        # Use feature names from model if available
        if self.feature_names:
            # Only use columns that exist in both model features and dataframe
            available_features = [col for col in self.feature_names if col in df.columns]
            if len(available_features) < len(self.feature_names):
                print(f"Warning: Using {len(available_features)}/{len(self.feature_names)} features")
            X = df[available_features].fillna(0)
        else:
            feature_cols = self._get_feature_columns(df)
            X = df[feature_cols].fillna(0)

        X = X.replace([np.inf, -np.inf], 0)
        return X

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns (excluding metadata and labels)."""
        exclude_cols = ['ticker', 'snapshot_date', 'outcome_class', 'label',
                       'outcome', 'outcome_gain', 'days_to_peak', 'qualification_start',
                       'qualification_end', 'upper_boundary', 'lower_boundary',
                       'power_boundary']

        return [col for col in df.columns if col not in exclude_cols]

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          df: pd.DataFrame) -> Dict:
        """Calculate comprehensive metrics."""
        metrics = {}

        # K4 recall
        k4_mask = y_true == 4
        metrics['k4_recall'] = (y_pred[k4_mask] == 4).mean() if k4_mask.sum() > 0 else 0
        metrics['k4_count'] = k4_mask.sum()

        # K3 recall
        k3_mask = y_true == 3
        metrics['k3_recall'] = (y_pred[k3_mask] == 3).mean() if k3_mask.sum() > 0 else 0
        metrics['k3_count'] = k3_mask.sum()

        # K3+K4 combined
        k34_mask = (y_true == 3) | (y_true == 4)
        k34_pred_correct = ((y_pred == 3) | (y_pred == 4)) & k34_mask
        metrics['k34_recall'] = k34_pred_correct.sum() / k34_mask.sum() if k34_mask.sum() > 0 else 0

        # K5 precision
        k5_pred_mask = y_pred == 5
        metrics['k5_precision'] = (y_true[k5_pred_mask] == 5).mean() if k5_pred_mask.sum() > 0 else 0

        return metrics


def main():
    """Run comprehensive validation suite."""

    print("="*60)
    print("COMPREHENSIVE MODEL VALIDATION SUITE")
    print("="*60)

    # Load the trained model
    model_path = Path("output/models/lightgbm_extreme_20251104_194341.pkl")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    validator = ComprehensiveValidator(model_path)

    # Load datasets
    print("\nLoading validation datasets...")

    # Load the full labeled dataset
    data_path = Path("output/patterns_labeled_enhanced_20251104_193956.parquet")
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        return

    df_full = pd.read_parquet(data_path)
    print(f"Loaded {len(df_full)} total patterns")

    # Encode labels
    outcome_mapping = {
        'K0_STAGNANT': 0,
        'K1_MINIMAL': 1,
        'K2_QUALITY': 2,
        'K3_STRONG': 3,
        'K4_EXCEPTIONAL': 4,
        'K5_FAILED': 5
    }
    df_full['label'] = df_full['outcome_class'].map(outcome_mapping)

    # Split into train and test (80/20)
    split_date = df_full['snapshot_date'].quantile(0.8)
    train_df = df_full[df_full['snapshot_date'] <= split_date]
    test_df = df_full[df_full['snapshot_date'] > split_date]

    print(f"Train: {len(train_df)} patterns")
    print(f"Test: {len(test_df)} patterns")

    # Run validation suite
    results = {}

    # 1. Temporal Validation
    temporal_results = validator.validate_temporal_consistency(df_full)
    results['temporal'] = temporal_results

    # 2. Unseen Ticker Validation
    unseen_results = validator.validate_unseen_tickers(train_df, test_df)
    results['unseen_tickers'] = unseen_results

    # 3. Data Leakage Detection
    leakage_results = validator.detect_data_leakage(df_full)
    results['data_leakage'] = leakage_results

    # 4. Oversampling Risk Assessment
    oversampling_results = validator.assess_oversampling_risk(train_df, test_df)
    results['oversampling'] = oversampling_results

    # 5. Cross-Validation
    cv_results = validator.run_cross_validation(test_df, n_folds=5)
    results['cross_validation'] = cv_results

    # Generate summary report
    print("\n" + "="*60)
    print("VALIDATION SUMMARY REPORT")
    print("="*60)

    print("\n1. TEMPORAL CONSISTENCY:")
    if temporal_results:
        k4_recalls = [r['k4_recall'] for r in temporal_results.values() if r.get('k4_count', 0) > 0]
        if k4_recalls:
            print(f"   K4 Recall Range: {min(k4_recalls):.1%} - {max(k4_recalls):.1%}")
            print(f"   K4 Recall StdDev: {np.std(k4_recalls):.1%}")

    print("\n2. UNSEEN TICKER PERFORMANCE:")
    if unseen_results:
        print(f"   K4 Recall: {unseen_results.get('k4_recall', 0):.1%}")
        print(f"   K3+K4 Recall: {unseen_results.get('k34_recall', 0):.1%}")

    print("\n3. DATA LEAKAGE RISK:")
    if leakage_results.get('suspicious_features'):
        print(f"   WARNING: {len(leakage_results['suspicious_features'])} suspicious features")
    else:
        print("   No obvious leakage detected")

    print("\n4. OVERSAMPLING RISK:")
    if oversampling_results.get('k4_oversampling'):
        print("   WARNING: K4 may be oversampled")
    if oversampling_results.get('suspiciously_high_recall'):
        print("   WARNING: Suspiciously high K4 recall")
    if not oversampling_results:
        print("   No clear oversampling detected")

    print("\n5. CROSS-VALIDATION STABILITY:")
    if cv_results:
        k4_recalls = [r['k4_recall'] for r in cv_results if r['k4_samples'] > 0]
        if k4_recalls:
            cv_std = np.std(k4_recalls)
            if cv_std > 0.2:
                print(f"   WARNING: High variance ({cv_std:.1%})")
            else:
                print(f"   Stable across folds (StdDev: {cv_std:.1%})")

    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)

    # Calculate realistic K4 recall estimate
    realistic_recalls = []
    if temporal_results:
        realistic_recalls.extend([r['k4_recall'] for r in temporal_results.values()
                                 if r.get('k4_count', 0) > 5])
    if unseen_results and unseen_results.get('k4_count', 0) > 5:
        realistic_recalls.append(unseen_results['k4_recall'])

    if realistic_recalls:
        realistic_k4_recall = np.median(realistic_recalls)
        print(f"\nRealistic K4 Recall Estimate: {realistic_k4_recall:.1%}")
        print(f"(Based on temporal and unseen ticker validation)")

        if realistic_k4_recall < 0.5:
            print("\nWARNING: Real-world K4 recall may be lower than training results")
        elif realistic_k4_recall > 0.7:
            print("\nSUCCESS: Model shows strong generalization to unseen data")
    else:
        print("\nInsufficient K4 samples for realistic estimate")

    return results


if __name__ == "__main__":
    results = main()