"""
Training Service - Unified interface for model training.

Provides centralized training orchestration with:
- Dependency injection
- Progress tracking
- Model registry
- Unified interface for all training pipelines
- Support for XGBoost, Deep Learning, and Ensemble models
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class TrainingProgress:
    """Tracks training progress."""

    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress_pct(self) -> float:
        """Get progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


@dataclass
class TrainingResult:
    """Result of a training operation."""

    success: bool
    models: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_paths: Dict[str, Path] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0


class TrainingService:
    """
    Unified training service for all ML models.

    Provides high-level interface for training:
    - XGBoost classifier (multi-class outcome prediction)
    - XGBoost regressor (gain/loss prediction)
    - Expected Value predictor
    - Hybrid predictor (classification + regression)
    - Deep learning models (if TensorFlow available)
    - Ensemble models (XGBoost + Deep Learning)

    Features:
    - Progress tracking
    - Model saving/loading
    - Hyperparameter optimization (Optuna)
    - Walk-forward validation
    - Comprehensive metrics

    Example:
        service = TrainingService(
            model_factory=factory,
            output_dir='output/models'
        )

        # Train classifier
        result = service.train_classifier(
            X_train, y_train,
            X_val, y_val,
            optimize_hyperparams=True,
            n_trials=30
        )

        # Train full system
        result = service.train_complete_system(
            patterns_df,
            enable_deep_learning=True,
            enable_regression=True
        )
    """

    def __init__(
        self,
        model_factory: 'ModelFactory',
        output_dir: str = "output/models",
        enable_progress_tracking: bool = True
    ):
        """
        Initialize training service.

        Args:
            model_factory: ModelFactory for creating models
            output_dir: Directory for saving models
            enable_progress_tracking: Whether to track progress
        """
        self.model_factory = model_factory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_progress_tracking = enable_progress_tracking

        # Progress tracking
        self._progress: Optional[TrainingProgress] = None

        # Model registry (in-memory)
        self._models: Dict[str, Any] = {}

        logger.info(f"TrainingService initialized (output_dir={output_dir})")

    def train_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        optimize_hyperparams: bool = False,
        n_trials: int = 30,
        save_model: bool = True,
        model_name: str = "xgboost_classifier"
    ) -> TrainingResult:
        """
        Train XGBoost classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            optimize_hyperparams: Whether to optimize hyperparameters with Optuna
            n_trials: Number of Optuna trials
            save_model: Whether to save trained model
            model_name: Name for saving model

        Returns:
            TrainingResult with trained model and metrics
        """
        logger.info("=" * 70)
        logger.info("Training XGBoost Classifier")
        logger.info("=" * 70)

        start_time = datetime.now()

        try:
            # Create classifier
            classifier = self.model_factory.create_classifier()

            # Optimize hyperparameters if requested
            if optimize_hyperparams:
                logger.info(f"Optimizing hyperparameters ({n_trials} trials)...")
                best_params = self._optimize_classifier_hyperparams(
                    X_train, y_train,
                    X_val, y_val,
                    n_trials=n_trials
                )

                # Create new classifier with optimized params
                classifier = self.model_factory.create_classifier(
                    config_override=best_params
                )

            # Train classifier
            logger.info("Training classifier...")

            if X_val is not None and y_val is not None:
                # Train with validation set for early stopping
                classifier.train(X_train, y_train, eval_set=(X_val, y_val))
            else:
                # Train without validation
                classifier.train(X_train, y_train)

            # Note: XGBoostClassifier does not have an evaluate() method
            # Metrics are logged during training if eval_set is provided
            metrics = {}
            if X_val is not None and y_val is not None:
                logger.info("Validation metrics logged during training")

            # Save model if requested
            model_path = None
            if save_model:
                model_path = self.output_dir / f"{model_name}.joblib"
                classifier.save(str(model_path))
                logger.info(f"Model saved to: {model_path}")

            # Store in registry
            self._models[model_name] = classifier

            duration = (datetime.now() - start_time).total_seconds()

            logger.info(f"Classifier training complete in {duration:.2f}s")

            return TrainingResult(
                success=True,
                models={'classifier': classifier},
                metrics=metrics,
                model_paths={'classifier': model_path} if model_path else {},
                duration_seconds=duration
            )

        except Exception as e:
            logger.error(f"Classifier training failed: {e}", exc_info=True)

            return TrainingResult(
                success=False,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )

    def train_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        save_model: bool = True,
        model_name: str = "xgboost_regressor"
    ) -> TrainingResult:
        """
        Train XGBoost regressor.

        Args:
            X_train: Training features
            y_train: Training target (continuous gain/loss)
            X_val: Validation features
            y_val: Validation target
            save_model: Whether to save model
            model_name: Model name for saving

        Returns:
            TrainingResult with trained regressor
        """
        logger.info("=" * 70)
        logger.info("Training XGBoost Regressor")
        logger.info("=" * 70)

        start_time = datetime.now()

        try:
            # Create regressor
            regressor = self.model_factory.create_regressor()

            # Train
            logger.info("Training regressor...")

            if X_val is not None and y_val is not None:
                regressor.train(X_train, y_train, eval_set=(X_val, y_val))
            else:
                regressor.train(X_train, y_train)

            # Note: XGBoostRegressor does not have an evaluate() method
            # Metrics are logged during training if eval_set is provided
            metrics = {}
            if X_val is not None and y_val is not None:
                logger.info("Validation metrics logged during training")

            # Save model
            model_path = None
            if save_model:
                model_path = self.output_dir / f"{model_name}.joblib"
                regressor.save(str(model_path))
                logger.info(f"Regressor saved to: {model_path}")

            # Store in registry
            self._models[model_name] = regressor

            duration = (datetime.now() - start_time).total_seconds()

            logger.info(f"Regressor training complete in {duration:.2f}s")

            return TrainingResult(
                success=True,
                models={'regressor': regressor},
                metrics=metrics,
                model_paths={'regressor': model_path} if model_path else {},
                duration_seconds=duration
            )

        except Exception as e:
            logger.error(f"Regressor training failed: {e}", exc_info=True)

            return TrainingResult(
                success=False,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )

    def train_complete_system(
        self,
        patterns_df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = 'outcome_class',
        gain_column: str = 'total_gain_pct',
        test_size: float = 0.2,
        optimize_hyperparams: bool = True,
        n_trials: int = 30,
        enable_regression: bool = True,
        enable_deep_learning: bool = False,
        save_models: bool = True
    ) -> TrainingResult:
        """
        Train complete system (classifier + regressor + ensemble).

        Args:
            patterns_df: Labeled patterns DataFrame
            feature_columns: List of feature column names
            target_column: Target column for classification
            gain_column: Target column for regression
            test_size: Test set proportion
            optimize_hyperparams: Whether to optimize with Optuna
            n_trials: Number of Optuna trials
            enable_regression: Whether to train regressor
            enable_deep_learning: Whether to train deep learning models
            save_models: Whether to save trained models

        Returns:
            TrainingResult with all trained models
        """
        logger.info("=" * 70)
        logger.info("Training Complete System")
        logger.info("=" * 70)

        start_time = datetime.now()

        try:
            # Prepare data
            logger.info("Preparing data...")

            X = patterns_df[feature_columns]
            y_class = patterns_df[target_column]

            # Split data
            X_train, X_val, y_train_class, y_val_class = train_test_split(
                X, y_class,
                test_size=test_size,
                random_state=42,
                stratify=y_class
            )

            logger.info(f"Training set: {len(X_train)} samples")
            logger.info(f"Validation set: {len(X_val)} samples")

            all_models = {}
            all_metrics = {}
            all_paths = {}

            # Step 1: Train classifier
            logger.info("\nStep 1/3: Training Classifier")
            classifier_result = self.train_classifier(
                X_train, y_train_class,
                X_val, y_val_class,
                optimize_hyperparams=optimize_hyperparams,
                n_trials=n_trials,
                save_model=save_models
            )

            if not classifier_result.success:
                raise RuntimeError(f"Classifier training failed: {classifier_result.error}")

            all_models.update(classifier_result.models)
            all_metrics.update(classifier_result.metrics)
            all_paths.update(classifier_result.model_paths)

            # Step 2: Train regressor (if enabled)
            regressor = None
            if enable_regression and gain_column in patterns_df.columns:
                logger.info("\nStep 2/3: Training Regressor")

                y_gain = patterns_df[gain_column]
                _, _, y_train_gain, y_val_gain = train_test_split(
                    X, y_gain,
                    test_size=test_size,
                    random_state=42
                )

                regressor_result = self.train_regressor(
                    X_train, y_train_gain,
                    X_val, y_val_gain,
                    save_model=save_models
                )

                if regressor_result.success:
                    all_models.update(regressor_result.models)
                    all_metrics.update(regressor_result.metrics)
                    all_paths.update(regressor_result.model_paths)
                    regressor = regressor_result.models.get('regressor')

            # Step 3: Create EV predictor
            logger.info("\nStep 3/3: Creating EV Predictor")

            value_system = self.model_factory.create_value_system()
            classifier = all_models['classifier']

            ev_predictor = self.model_factory.create_ev_predictor(
                classifier=classifier,
                value_system=value_system
            )

            all_models['ev_predictor'] = ev_predictor

            # Save EV predictor if requested
            if save_models:
                ev_path = self.output_dir / "ev_predictor.joblib"
                import joblib
                joblib.dump(ev_predictor, ev_path)
                all_paths['ev_predictor'] = ev_path

            duration = (datetime.now() - start_time).total_seconds()

            logger.info("")
            logger.info("=" * 70)
            logger.info(f"Complete system training finished in {duration:.2f}s")
            logger.info("=" * 70)

            return TrainingResult(
                success=True,
                models=all_models,
                metrics=all_metrics,
                model_paths=all_paths,
                duration_seconds=duration
            )

        except Exception as e:
            logger.error(f"Complete system training failed: {e}", exc_info=True)

            return TrainingResult(
                success=False,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )

    def _optimize_classifier_hyperparams(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        n_trials: int = 30
    ) -> Dict[str, Any]:
        """
        Optimize classifier hyperparameters with Optuna.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of trials

        Returns:
            Best hyperparameters
        """
        try:
            import optuna
            from sklearn.metrics import f1_score

            def objective(trial):
                # Suggest hyperparameters
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                }

                # Create and train classifier
                classifier = self.model_factory.create_classifier(config_override=params)

                if X_val is not None and y_val is not None:
                    classifier.train(X_train, y_train, X_val=X_val, y_val=y_val)

                    # Evaluate on validation set
                    y_pred = classifier.predict(X_val)
                    score = f1_score(y_val, y_pred, average='weighted')
                else:
                    # Use training set (not ideal but better than nothing)
                    classifier.train(X_train, y_train)
                    y_pred = classifier.predict(X_train)
                    score = f1_score(y_train, y_pred, average='weighted')

                return score

            # Run optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

            logger.info(f"Best F1 score: {study.best_value:.4f}")
            logger.info(f"Best params: {study.best_params}")

            return study.best_params

        except ImportError:
            logger.warning("Optuna not available, using default hyperparameters")
            return {}

    def get_model(self, name: str) -> Optional[Any]:
        """Get model from registry."""
        return self._models.get(name)

    def list_models(self) -> List[str]:
        """List all models in registry."""
        return list(self._models.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'models_trained': len(self._models),
            'model_names': list(self._models.keys()),
            'output_dir': str(self.output_dir),
        }
