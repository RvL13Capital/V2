"""
Validation Service for AIv3

Encapsulates model validation logic including:
- Walk-forward validation with temporal windows
- EV correlation calculation
- Cross-validation with temporal ordering
- Performance metrics aggregation

This service consolidates Step 6 of the training pipeline.
"""

from typing import Dict, Optional, List, Any
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from config import SystemConfig
from ml import XGBoostClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.temporal_utils import ensure_temporal_order, generate_walk_forward_windows, validate_no_future_leakage

logger = logging.getLogger(__name__)


class ValidationService:
    """
    Service for model validation with temporal integrity.

    Encapsulates:
    - Walk-forward validation (expanding window strategy)
    - Temporal window generation
    - No look-ahead bias validation
    - Performance metrics calculation
    - Window-by-window results tracking

    Usage:
        validation_service = ValidationService(config)

        # Perform walk-forward validation
        results = validation_service.walk_forward_validation(
            features_df,
            trained_model,
            n_windows=4
        )

        # Get detailed window results
        window_results = validation_service.get_window_results()
    """

    def __init__(self, config: SystemConfig):
        """
        Initialize validation service.

        Args:
            config: System configuration
        """
        self.config = config

        # Results storage
        self.validation_results = None
        self.window_results = None

        logger.info("ValidationService initialized")

    def walk_forward_validation(
        self,
        features_df: pd.DataFrame,
        model: Optional[XGBoostClassifier] = None,
        params: Optional[Dict] = None,
        initial_train_pct: float = 0.6,
        test_period_pct: float = 0.2,
        step_pct: float = 0.1
    ) -> Dict:
        """
        Perform walk-forward validation with expanding windows.

        Args:
            features_df: DataFrame with features and outcome_class
            model: Pre-trained model (if None, trains new model for each window)
            params: Hyperparameters for training (if model is None)
            initial_train_pct: Initial training size as % of data (default: 60%)
            test_period_pct: Test period size as % of data (default: 20%)
            step_pct: Step size for window advancement as % of data (default: 10%)

        Returns:
            Dictionary with validation metrics and results
        """
        logger.info("Starting walk-forward validation...")

        # Ensure temporal ordering
        if 'start_date' in features_df.columns:
            features_df = features_df.copy()
            features_df['start_date'] = pd.to_datetime(features_df['start_date'])
            features_df = ensure_temporal_order(features_df, date_column='start_date')
            date_col = 'start_date'
        else:
            logger.warning("No start_date column - assuming data is chronologically sorted")
            date_col = None

        # Prepare data
        X, y, feature_cols = self._prepare_data(features_df)

        logger.info(f"Prepared data: {len(X)} patterns, {len(feature_cols)} features")

        # Generate walk-forward windows
        windows = self._generate_windows(
            features_df, X, y, date_col,
            initial_train_pct, test_period_pct, step_pct
        )

        if not windows:
            logger.warning("Insufficient data for walk-forward validation. Using single split.")
            return self._fallback_single_split(X, y, params)

        logger.info(f"Generated {len(windows)} walk-forward windows")

        # Train and evaluate on each window
        window_results = []
        all_predictions = []
        all_actuals = []

        for window in windows:
            logger.info(f"Processing window {window['window_num']}/{len(windows)}")

            # Extract train/test data
            X_train = X.iloc[window['train_start_idx']:window['train_end_idx']]
            y_train = y.iloc[window['train_start_idx']:window['train_end_idx']]
            X_test = X.iloc[window['test_start_idx']:window['test_end_idx']]
            y_test = y.iloc[window['test_start_idx']:window['test_end_idx']]

            # Validate temporal integrity
            if date_col and 'train_end_date' in window:
                train_dates = features_df[date_col].iloc[window['train_start_idx']:window['train_end_idx']]
                test_dates = features_df[date_col].iloc[window['test_start_idx']:window['test_end_idx']]
                try:
                    validate_no_future_leakage(train_dates, test_dates, raise_error=True)
                except ValueError as e:
                    logger.error(f"Future leakage detected in window {window['window_num']}: {e}")
                    continue

            # Train model for this window (or use pre-trained model)
            if model is None:
                if params is None:
                    params = self._get_default_params()

                window_model = XGBoostClassifier(
                    **params,
                    memory_efficient_mode=self.config.memory.xgb_memory_efficient_mode
                )
                window_model.train(X_train, y_train)
            else:
                window_model = model

            # Predict on test set
            y_pred = window_model.predict(X_test)

            # Calculate metrics
            window_accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )

            # Store results
            window_results.append({
                'window_num': window['window_num'],
                'accuracy': float(window_accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'train_size': len(X_train),
                'test_size': len(X_test)
            })

            # Aggregate for overall metrics
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test)

            logger.debug(f"Window {window['window_num']}: accuracy={window_accuracy:.3f}")

        # Calculate overall metrics
        overall_accuracy = accuracy_score(all_actuals, all_predictions)
        avg_window_accuracy = np.mean([w['accuracy'] for w in window_results])
        accuracy_std = np.std([w['accuracy'] for w in window_results])

        self.window_results = window_results
        self.validation_results = {
            'num_windows': len(windows),
            'overall_accuracy': float(overall_accuracy),
            'avg_window_accuracy': float(avg_window_accuracy),
            'accuracy_std': float(accuracy_std),
            'window_results': window_results,
            'total_test_predictions': len(all_predictions),
            'validation_strategy': 'walk_forward_expanding_window'
        }

        logger.info(f"Validation complete: overall_accuracy={overall_accuracy:.3f}, "
                   f"avg_window_accuracy={avg_window_accuracy:.3f}")

        return self.validation_results

    def _prepare_data(self, features_df: pd.DataFrame):
        """Prepare features and target from dataframe."""
        exclude_cols = ['ticker', 'pattern_id', 'outcome_class', 'max_gain', 'max_loss',
                       'start_date', 'end_date', 'breakout_date']

        feature_cols = [col for col in features_df.columns
                       if col not in exclude_cols and
                       features_df[col].dtype in ['int64', 'float64', 'float32', 'int32', 'bool']]

        X = features_df[feature_cols].fillna(0)
        y = features_df['outcome_class']

        return X, y, feature_cols

    def _generate_windows(
        self,
        features_df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        date_col: Optional[str],
        initial_train_pct: float,
        test_period_pct: float,
        step_pct: float
    ) -> List[Dict]:
        """Generate walk-forward windows."""
        n_samples = len(X)
        initial_train_size = int(n_samples * initial_train_pct)
        test_period_size = max(int(n_samples * test_period_pct), 10)
        step_size = max(int(n_samples * step_pct), 5)

        logger.info(f"Window configuration: total={n_samples}, "
                   f"initial_train={initial_train_size}, "
                   f"test_period={test_period_size}, "
                   f"step={step_size}")

        if date_col:
            try:
                windows = generate_walk_forward_windows(
                    df=features_df,
                    initial_train_size=initial_train_size,
                    test_period_size=test_period_size,
                    step_size=step_size,
                    date_column=date_col
                )
            except Exception as e:
                logger.warning(f"Error generating windows with date column: {e}. Using manual generation.")
                windows = self._generate_windows_manual(
                    n_samples, initial_train_size, test_period_size, step_size
                )
        else:
            windows = self._generate_windows_manual(
                n_samples, initial_train_size, test_period_size, step_size
            )

        return windows

    def _generate_windows_manual(
        self,
        n_samples: int,
        initial_train_size: int,
        test_period_size: int,
        step_size: int
    ) -> List[Dict]:
        """Generate windows manually without date column."""
        windows = []
        train_end = initial_train_size
        window_num = 1

        while train_end + test_period_size <= n_samples:
            windows.append({
                'window_num': window_num,
                'train_start_idx': 0,
                'train_end_idx': train_end,
                'test_start_idx': train_end,
                'test_end_idx': min(train_end + test_period_size, n_samples),
                'train_size': train_end,
                'test_size': min(test_period_size, n_samples - train_end)
            })
            train_end += step_size
            window_num += 1

        return windows

    def _fallback_single_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Optional[Dict]
    ) -> Dict:
        """Fallback to single temporal split if insufficient data."""
        logger.info("Using single temporal split (80-20)")

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if params is None:
            params = self._get_default_params()

        model = XGBoostClassifier(
            **params,
            memory_efficient_mode=self.config.memory.xgb_memory_efficient_mode
        )
        model.train(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.validation_results = {
            'num_windows': 1,
            'overall_accuracy': float(accuracy),
            'avg_window_accuracy': float(accuracy),
            'accuracy_std': 0.0,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'validation_strategy': 'single_temporal_split',
            'notes': 'Fallback due to insufficient data'
        }

        logger.info(f"Single split validation: accuracy={accuracy:.3f}")
        return self.validation_results

    def _get_default_params(self) -> Dict:
        """Get default hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }

    # Accessor methods
    def get_validation_results(self) -> Optional[Dict]:
        """Get overall validation results."""
        return self.validation_results

    def get_window_results(self) -> Optional[List[Dict]]:
        """Get window-by-window results."""
        return self.window_results

    def get_overall_accuracy(self) -> Optional[float]:
        """Get overall validation accuracy."""
        if self.validation_results:
            return self.validation_results.get('overall_accuracy')
        return None

    def get_avg_window_accuracy(self) -> Optional[float]:
        """Get average window accuracy."""
        if self.validation_results:
            return self.validation_results.get('avg_window_accuracy')
        return None

    def print_summary(self):
        """Print validation summary."""
        if not self.validation_results:
            print("No validation results available")
            return

        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Strategy: {self.validation_results.get('validation_strategy', 'N/A')}")
        print(f"Windows: {self.validation_results['num_windows']}")
        print(f"Overall accuracy: {self.validation_results['overall_accuracy']:.3f}")
        print(f"Avg window accuracy: {self.validation_results.get('avg_window_accuracy', 0):.3f}")
        print(f"Accuracy std: {self.validation_results.get('accuracy_std', 0):.3f}")

        if self.window_results:
            print(f"\nWindow-by-window results:")
            for w in self.window_results:
                print(f"  Window {w['window_num']}: {w['accuracy']:.3f} "
                      f"(train={w['train_size']}, test={w['test_size']})")
