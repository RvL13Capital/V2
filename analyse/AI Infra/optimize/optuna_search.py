"""
Optuna-based hyperparameter optimization for dual ML models.
"""

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Union
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging
import json
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Bayesian hyperparameter optimization using Optuna."""

    def __init__(self,
                 model_type: str = 'lightgbm',
                 n_trials: int = 100,
                 n_jobs: int = 1,
                 timeout: Optional[int] = None,
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None):
        """
        Initialize Optuna optimizer.

        Args:
            model_type: Type of model ('lightgbm', 'sequential', 'ensemble')
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            timeout: Timeout in seconds
            study_name: Name for the study
            storage: Database URL for distributed optimization
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.study_name = study_name or f"{model_type}_optimization"
        self.storage = storage
        self.best_params = None
        self.best_score = None
        self.study = None

    def create_lightgbm_objective(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_val: np.ndarray,
                                 n_classes: int = 3) -> Callable:
        """
        Create objective function for LightGBM optimization.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_classes: Number of classes

        Returns:
            Objective function
        """
        def objective(trial: Trial) -> float:
            # Suggest hyperparameters
            params = {
                'objective': 'multiclass' if n_classes > 2 else 'binary',
                'num_class': n_classes if n_classes > 2 else 1,
                'metric': 'multi_logloss' if n_classes > 2 else 'binary_logloss',
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0001, 0.1, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
                'early_stopping_rounds': 50,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }

            # Handle DART specific parameters
            if params['boosting_type'] == 'dart':
                params['drop_rate'] = trial.suggest_float('drop_rate', 0.0, 0.3)
                params['max_drop'] = trial.suggest_int('max_drop', 10, 100)
                params['skip_drop'] = trial.suggest_float('skip_drop', 0.4, 0.7)

            # Train model
            from models.lightgbm_model import LightGBMBreakoutClassifier

            model = LightGBMBreakoutClassifier(n_classes=n_classes, params=params)

            # Use pruning callback for early stopping
            pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'multi_logloss')

            try:
                metrics = model.train(
                    X_train, y_train,
                    X_val, y_val
                )

                # Get validation predictions
                y_pred = model.predict(X_val)

                # Calculate validation score
                if n_classes > 2:
                    score = f1_score(y_val, y_pred, average='weighted')
                else:
                    score = f1_score(y_val, y_pred)

                # Report intermediate values for pruning
                trial.report(score, step=model.best_iteration)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return 0.0

        return objective

    def create_sequential_objective(self,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_val: np.ndarray,
                                  y_val: np.ndarray,
                                  sequence_length: int,
                                  n_features: int,
                                  n_classes: int = 3) -> Callable:
        """
        Create objective function for Sequential model optimization.

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            sequence_length: Length of sequences
            n_features: Number of features
            n_classes: Number of classes

        Returns:
            Objective function
        """
        def objective(trial: Trial) -> float:
            # Suggest CNN architecture
            n_cnn_layers = trial.suggest_int('n_cnn_layers', 1, 4)
            cnn_filters = []
            cnn_kernel_sizes = []

            for i in range(n_cnn_layers):
                cnn_filters.append(
                    trial.suggest_categorical(f'cnn_filters_{i}', [32, 64, 128, 256])
                )
                cnn_kernel_sizes.append(
                    trial.suggest_categorical(f'cnn_kernel_{i}', [3, 5, 7, 9])
                )

            # Suggest LSTM architecture
            n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 3)
            lstm_units = []

            for i in range(n_lstm_layers):
                lstm_units.append(
                    trial.suggest_categorical(f'lstm_units_{i}', [32, 64, 128, 256])
                )

            # Other hyperparameters
            attention_units = trial.suggest_categorical('attention_units', [64, 128, 256])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

            # Build and train model
            from models.sequential_model import CNNBiLSTMAttentionModel
            import tensorflow as tf

            model = CNNBiLSTMAttentionModel(
                sequence_length=sequence_length,
                n_features=n_features,
                n_classes=n_classes,
                cnn_filters=cnn_filters,
                cnn_kernel_sizes=cnn_kernel_sizes,
                lstm_units=lstm_units,
                attention_units=attention_units,
                dropout_rate=dropout_rate
            )

            # Build model
            model.build_model()

            # Custom optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile_model(optimizer=optimizer)

            # Pruning callback
            pruning_callback = optuna.integration.TFKerasPruningCallback(
                trial, 'val_accuracy'
            )

            # Train with early stopping
            model.callbacks = [pruning_callback]
            model.create_callbacks(patience=10)

            try:
                history = model.train(
                    X_train, y_train,
                    X_val, y_val,
                    epochs=50,
                    batch_size=batch_size
                )

                # Get best validation score
                score = history['best_val_accuracy']

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return 0.0

        return objective

    def create_ensemble_objective(self,
                                lightgbm_model,
                                sequential_model,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray) -> Callable:
        """
        Create objective function for ensemble optimization.

        Args:
            lightgbm_model: Trained LightGBM model
            sequential_model: Trained Sequential model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Objective function
        """
        def objective(trial: Trial) -> float:
            # Suggest ensemble weights
            lgb_weight = trial.suggest_float('lgb_weight', 0.0, 1.0)
            seq_weight = 1.0 - lgb_weight

            # Suggest meta-learner parameters
            use_meta_learner = trial.suggest_categorical('use_meta_learner', [True, False])

            if use_meta_learner:
                meta_params = {
                    'n_estimators': trial.suggest_int('meta_n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('meta_max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('meta_learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('meta_subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('meta_colsample', 0.5, 1.0)
                }

                # Get base model predictions
                lgb_train_pred = lightgbm_model.predict_proba(X_train['tabular'])
                seq_train_pred = sequential_model.predict_proba(X_train['sequential'])

                lgb_val_pred = lightgbm_model.predict_proba(X_val['tabular'])
                seq_val_pred = sequential_model.predict_proba(X_val['sequential'])

                # Stack predictions
                train_meta = np.hstack([lgb_train_pred, seq_train_pred])
                val_meta = np.hstack([lgb_val_pred, seq_val_pred])

                # Train meta-learner
                from xgboost import XGBClassifier
                meta_learner = XGBClassifier(**meta_params, random_state=42)
                meta_learner.fit(train_meta, y_train)

                # Get predictions
                y_pred = meta_learner.predict(val_meta)

            else:
                # Simple weighted average
                lgb_pred = lightgbm_model.predict_proba(X_val['tabular'])
                seq_pred = sequential_model.predict_proba(X_val['sequential'])

                ensemble_pred = lgb_weight * lgb_pred + seq_weight * seq_pred
                y_pred = np.argmax(ensemble_pred, axis=1)

            # Calculate score
            score = f1_score(y_val, y_pred, average='weighted')

            return score

        return objective

    def optimize(self,
                objective: Callable,
                direction: str = 'maximize',
                n_trials: Optional[int] = None,
                callbacks: Optional[List] = None) -> optuna.Study:
        """
        Run optimization study.

        Args:
            objective: Objective function to optimize
            direction: Optimization direction ('minimize' or 'maximize')
            n_trials: Number of trials (overrides init value)
            callbacks: Additional callbacks

        Returns:
            Optuna study object
        """
        # Create sampler and pruner
        sampler = TPESampler(seed=42, n_startup_trials=10)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        # Create or load study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            load_if_exists=True
        )

        # Add default callbacks
        default_callbacks = [
            optuna.visualization.matplotlib.plot_intermediate_values,
        ]

        if callbacks:
            default_callbacks.extend(callbacks)

        # Run optimization
        n_trials = n_trials or self.n_trials

        self.study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            catch=(Exception,)
        )

        # Get best parameters
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info(f"Optimization complete. Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.study

    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.

        Returns:
            DataFrame with trial results
        """
        if self.study is None:
            return pd.DataFrame()

        trials_df = self.study.trials_dataframe()
        return trials_df

    def plot_optimization_results(self, save_path: Optional[Path] = None):
        """
        Plot optimization results.

        Args:
            save_path: Path to save plots
        """
        if self.study is None:
            logger.warning("No study to plot")
            return

        import matplotlib.pyplot as plt

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot optimization history
        ax = axes[0, 0]
        trial_numbers = [t.number for t in self.study.trials]
        trial_values = [t.value for t in self.study.trials if t.value is not None]
        ax.plot(trial_numbers[:len(trial_values)], trial_values, 'bo-', alpha=0.6)
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Objective Value')
        ax.set_title('Optimization History')
        ax.grid(True, alpha=0.3)

        # Plot parameter importance
        ax = axes[0, 1]
        try:
            importance = optuna.importance.get_param_importances(self.study)
            params = list(importance.keys())
            values = list(importance.values())

            ax.barh(params, values)
            ax.set_xlabel('Importance')
            ax.set_title('Parameter Importance')
        except Exception as e:
            ax.text(0.5, 0.5, 'Not enough trials', ha='center', va='center')

        # Plot parameter relationships
        ax = axes[1, 0]
        if len(self.study.trials) > 10:
            try:
                from optuna.visualization.matplotlib import plot_parallel_coordinate
                plot_parallel_coordinate(self.study, target_name='Score')
            except Exception:
                ax.text(0.5, 0.5, 'Cannot create parallel plot', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'Not enough trials', ha='center', va='center')

        # Plot best vs trial
        ax = axes[1, 1]
        best_values = []
        current_best = float('-inf') if self.study.direction == optuna.study.StudyDirection.MAXIMIZE else float('inf')

        for trial in self.study.trials:
            if trial.value is not None:
                if self.study.direction == optuna.study.StudyDirection.MAXIMIZE:
                    current_best = max(current_best, trial.value)
                else:
                    current_best = min(current_best, trial.value)
            best_values.append(current_best)

        ax.plot(range(len(best_values)), best_values, 'g-', linewidth=2)
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Best Value So Far')
        ax.set_title('Best Value Progression')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / 'optimization_results.png', dpi=100, bbox_inches='tight')

        plt.show()

    def save_results(self, path: Union[str, Path]):
        """
        Save optimization results.

        Args:
            path: Save path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        with open(path / 'best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)

        # Save study
        if self.study:
            joblib.dump(self.study, path / 'study.pkl')

            # Save trials dataframe
            trials_df = self.study.trials_dataframe()
            trials_df.to_csv(path / 'trials.csv', index=False)

            # Save importance if available
            try:
                importance = optuna.importance.get_param_importances(self.study)
                importance_df = pd.DataFrame(
                    list(importance.items()),
                    columns=['parameter', 'importance']
                )
                importance_df.to_csv(path / 'parameter_importance.csv', index=False)
            except Exception:
                pass

        logger.info(f"Results saved to {path}")

    def load_study(self, path: Union[str, Path]):
        """
        Load saved study.

        Args:
            path: Path to saved study
        """
        path = Path(path)

        if (path / 'study.pkl').exists():
            self.study = joblib.load(path / 'study.pkl')
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            logger.info(f"Study loaded from {path}")
        else:
            logger.warning(f"No study found at {path}")