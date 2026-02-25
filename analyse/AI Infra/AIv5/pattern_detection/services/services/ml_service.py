"""
ML Service for AIv3

Encapsulates machine learning training pipeline including:
- Hyperparameter optimization (Optuna)
- Model training (XGBoost, Deep Learning)
- Ensemble creation
- Model evaluation and metrics

This service consolidates Steps 4-5 of the training pipeline.
"""

from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

from config import SystemConfig
from ml import XGBoostClassifier, ExpectedValuePredictor, PatternValueSystem
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

logger = logging.getLogger(__name__)

# Try to import deep learning components
try:
    from ml.deep_learning import (
        CNNBiLSTMAttentionModel,
        DeepLearningPredictor,
        EnsemblePredictor,
        SequenceDataProcessor
    )
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    CNNBiLSTMAttentionModel = None
    DeepLearningPredictor = None
    EnsemblePredictor = None
    SequenceDataProcessor = None


class MLService:
    """
    Service for machine learning model training and optimization.

    Encapsulates:
    - Hyperparameter optimization with Optuna
    - XGBoost classifier training
    - Deep learning model training (optional)
    - Ensemble model creation
    - Model evaluation and metrics

    Usage:
        ml_service = MLService(config)

        # Optimize hyperparameters
        best_params = ml_service.optimize_hyperparameters(
            features_df,
            n_trials=50
        )

        # Train models
        trained_models = ml_service.train_models(
            features_df,
            params=best_params,
            enable_deep_learning=True
        )

        # Get trained classifier
        classifier = ml_service.get_classifier()
        ev_predictor = ml_service.get_ev_predictor()
    """

    def __init__(self, config: SystemConfig, memory_optimizer: Any = None):
        """
        Initialize ML service.

        Args:
            config: System configuration
            memory_optimizer: Optional memory optimizer for resource management
        """
        self.config = config
        self.memory_optimizer = memory_optimizer

        # Model storage
        self.classifier = None
        self.ev_predictor = None
        self.dl_model = None
        self.sequence_processor = None
        self.ensemble = None

        # Training results
        self.optimization_results = None
        self.training_results = None

        # Value system
        self.value_system = PatternValueSystem()

        logger.info("MLService initialized")

    def optimize_hyperparameters(
        self,
        features_df: pd.DataFrame,
        n_trials: int = 50,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Optimize hyperparameters using Optuna.

        Args:
            features_df: DataFrame with features and outcome_class column
            n_trials: Number of Optuna optimization trials
            output_dir: Optional directory to save optimization results

        Returns:
            Dictionary with best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")

        try:
            import optuna
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import make_scorer
        except ImportError:
            logger.warning("Optuna not installed. Using default parameters.")
            return self._get_default_params()

        # Prepare data
        X, y, feature_cols = self._prepare_data(features_df)

        logger.info(f"Prepared data: {len(X)} patterns, {len(feature_cols)} features")

        # Custom strategic value scorer
        def strategic_value_scorer(y_true, y_pred):
            total_value = 0
            for true_class, pred_class in zip(y_true, y_pred):
                true_value = self.value_system.get_class_value(true_class)
                pred_value = self.value_system.get_class_value(pred_class)

                if true_class == pred_class:
                    total_value += abs(true_value)
                else:
                    total_value -= abs(true_value - pred_value) * 0.5

            return total_value / len(y_true)

        # Optuna objective function
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
            }

            # Add memory-efficient parameters
            if self.config.memory.xgb_memory_efficient_mode:
                params['tree_method'] = 'hist'
                params['max_bin'] = self.config.memory.xgb_max_bin

            # 5-fold time series cross-validation
            cv = TimeSeriesSplit(n_splits=5)
            fold_scores = []

            for train_idx, val_idx in cv.split(X):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]

                # Calculate sample weights for class balancing
                fold_weights = compute_sample_weight(class_weight='balanced', y=y_train_fold)

                # Train classifier
                fold_classifier = XGBoostClassifier(**params)
                fold_classifier.train(X_train_fold, y_train_fold, sample_weight=fold_weights)

                # Evaluate
                y_pred = fold_classifier.predict(X_val_fold)
                fold_score = strategic_value_scorer(y_val_fold, y_pred)
                fold_scores.append(fold_score)

            return np.mean(fold_scores)

        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            study_name=f'aiv3_ml_service_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

        best_params = study.best_params
        best_value = study.best_value

        # Store results
        self.optimization_results = {
            'best_params': best_params,
            'best_value': float(best_value),
            'n_trials': n_trials,
            'study_name': study.study_name
        }

        # Save results if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            optuna_file = output_dir / f'optuna_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(optuna_file, 'w') as f:
                json.dump(self.optimization_results, f, indent=2)
            logger.info(f"Optimization results saved to {optuna_file}")

        logger.info(f"Optimization complete: best_value={best_value:.4f}")
        return best_params

    def train_models(
        self,
        features_df: pd.DataFrame,
        params: Optional[Dict] = None,
        enable_deep_learning: bool = False,
        ticker_data: Optional[Dict[str, pd.DataFrame]] = None,
        labeled_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Train machine learning models.

        Args:
            features_df: DataFrame with features and outcome_class
            params: Hyperparameters (if None, uses defaults)
            enable_deep_learning: Whether to train deep learning model
            ticker_data: Required if enable_deep_learning=True (for sequence generation)
            labeled_df: Required if enable_deep_learning=True (for pattern info)

        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Starting model training...")

        if params is None:
            params = self._get_default_params()
            logger.info("Using default hyperparameters")

        # Prepare data
        X, y, feature_cols = self._prepare_data(features_df)

        # Split train/test
        X_train, X_test, y_train, y_test = self._split_data(X, y)

        # Train XGBoost
        logger.info("Training XGBoost classifier...")
        xgb_metrics = self._train_xgboost(X_train, y_train, X_test, y_test, params)

        results = {
            'xgboost': xgb_metrics,
            'deep_learning': None,
            'ensemble': None
        }

        # Train deep learning (RECOMMENDED for optimal ensemble performance)
        # When both XGBoost + DL are trained, ensemble predictor combines insights
        if enable_deep_learning and DEEP_LEARNING_AVAILABLE:
            if ticker_data is None or labeled_df is None:
                logger.warning("Deep learning enabled but ticker_data/labeled_df not provided. Skipping DL training.")
            else:
                try:
                    logger.info("Training deep learning model...")
                    dl_metrics = self._train_deep_learning(
                        features_df, labeled_df, ticker_data,
                        X_train, X_test, y_train, y_test
                    )
                    results['deep_learning'] = dl_metrics

                    # Create ensemble
                    if self.dl_model is not None and self.sequence_processor is not None:
                        logger.info("Creating ensemble predictor...")
                        ensemble_metrics = self._create_ensemble()
                        results['ensemble'] = ensemble_metrics

                except Exception as e:
                    logger.error(f"Deep learning training failed: {e}", exc_info=True)
                    results['deep_learning'] = {'error': str(e)}

        self.training_results = results
        logger.info("Model training complete")
        return results

    def _prepare_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Prepare features and target from dataframe."""
        exclude_cols = ['ticker', 'pattern_id', 'outcome_class', 'max_gain', 'max_loss',
                       'start_date', 'end_date', 'breakout_date']

        feature_cols = [col for col in features_df.columns
                       if col not in exclude_cols and
                       features_df[col].dtype in ['int64', 'float64', 'float32', 'int32', 'bool']]

        X = features_df[feature_cols].fillna(0)
        y = features_df['outcome_class']

        return X, y, feature_cols

    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train/test sets with stratification if possible."""
        class_counts = y.value_counts()
        min_class_count = class_counts.min()

        if min_class_count >= 2 and len(y) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        else:
            logger.warning(f"Insufficient data ({len(y)} patterns, min class={min_class_count}). Using all for training.")
            X_train, y_train = X, y
            X_test, y_test = X.iloc[:0], y.iloc[:0]

        return X_train, X_test, y_train, y_test

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Dict
    ) -> Dict:
        """Train XGBoost classifier and return metrics."""
        # Calculate sample weights
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        # Initialize classifier
        self.classifier = XGBoostClassifier(
            **params,
            tree_method=self.config.memory.xgb_tree_method,
            max_bin=self.config.memory.xgb_max_bin,
            memory_efficient_mode=self.config.memory.xgb_memory_efficient_mode
        )

        # Train
        if len(X_test) > 0:
            metrics = self.classifier.train(
                X_train, y_train,
                eval_set=(X_test, y_test),
                sample_weight=sample_weights
            )
        else:
            metrics = self.classifier.train(
                X_train, y_train,
                eval_set=None,
                sample_weight=sample_weights
            )

        # Create EV predictor
        self.ev_predictor = ExpectedValuePredictor(self.classifier, self.value_system)

        # Evaluate
        if len(X_test) > 0:
            y_pred = self.classifier.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )

            metrics.update({
                'test_accuracy': float(test_accuracy),
                'test_precision': float(precision),
                'test_recall': float(recall),
                'test_f1': float(f1),
                'train_size': len(X_train),
                'test_size': len(X_test)
            })

        logger.info(f"XGBoost training complete. Accuracy: {metrics.get('accuracy', 'N/A')}")
        return metrics

    def _train_deep_learning(
        self,
        features_df: pd.DataFrame,
        labeled_df: pd.DataFrame,
        ticker_data: Dict[str, pd.DataFrame],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """Train deep learning model and return metrics."""
        # Initialize sequence processor
        self.sequence_processor = SequenceDataProcessor(
            sequence_length=self.config.deep_learning.sequence_length,
            feature_columns=['open', 'high', 'low', 'close', 'volume',
                           'bbw_20', 'adx', 'rsi_14', 'atr_14',
                           'volume_ratio_20', 'daily_range',
                           'bbw_percentile', 'daily_range_avg']
        )

        # Create sequences
        sequences_list = []
        labels_list = []

        for idx, row in labeled_df.iterrows():
            ticker = row['ticker']
            if ticker not in ticker_data:
                continue

            df = ticker_data[ticker]
            try:
                sequence = self.sequence_processor.create_sequences_for_pattern(
                    df, pattern_end_idx=row.get('end_idx', len(df) - 1)
                )
                sequences_list.append(sequence)
                labels_list.append(row['outcome_class'])
            except Exception as e:
                logger.debug(f"Failed to create sequence for {ticker}: {e}")

        if not sequences_list:
            logger.error("No sequences created for DL training")
            return {'error': 'No sequences created'}

        X_seq = np.array(sequences_list)
        y_seq = np.array(labels_list)

        # Split sequences (matching tabular split)
        if len(X_test) > 0:
            X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42
            )
        else:
            X_seq_train, y_seq_train = X_seq, y_seq
            X_seq_test, y_seq_test = X_seq[:0], y_seq[:0]

        # Initialize DL model
        self.dl_model = CNNBiLSTMAttentionModel(
            sequence_length=self.config.deep_learning.sequence_length,
            n_features=X_seq.shape[2],
            n_classes=6,  # K0-K5
            cnn_filters=self.config.deep_learning.cnn_filters,
            lstm_units=self.config.deep_learning.lstm_units,
            attention_heads=self.config.deep_learning.attention_heads,
            dropout_rate=self.config.deep_learning.dropout_rate
        )

        # Train
        history = self.dl_model.train(
            X_seq_train, y_seq_train,
            validation_split=0.1,
            epochs=self.config.deep_learning.epochs,
            batch_size=self.config.deep_learning.batch_size,
            early_stopping_patience=self.config.deep_learning.early_stopping_patience
        )

        metrics = {
            'train_size': len(X_seq_train),
            'test_size': len(X_seq_test),
            'sequence_shape': X_seq.shape,
            'epochs_trained': len(history.history.get('loss', []))
        }

        logger.info(f"Deep learning training complete. Epochs: {metrics['epochs_trained']}")
        return metrics

    def _create_ensemble(self) -> Dict:
        """Create ensemble predictor combining XGBoost and DL."""
        dl_predictor = DeepLearningPredictor(self.dl_model, self.value_system)

        self.ensemble = EnsemblePredictor(
            xgboost_predictor=self.ev_predictor,
            dl_predictor=dl_predictor,
            value_system=self.value_system,
            strategy=self.config.deep_learning.ensemble_strategy,
            xgb_weight=self.config.deep_learning.ensemble_xgb_weight,
            dl_weight=self.config.deep_learning.ensemble_dl_weight
        )

        # Replace ev_predictor with ensemble
        self.ev_predictor = self.ensemble

        logger.info("Ensemble predictor created")
        return {'ensemble_strategy': self.config.deep_learning.ensemble_strategy}

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
    def get_classifier(self) -> Optional[XGBoostClassifier]:
        """Get trained XGBoost classifier."""
        return self.classifier

    def get_ev_predictor(self) -> Optional[ExpectedValuePredictor]:
        """Get EV predictor (may be ensemble if DL enabled)."""
        return self.ev_predictor

    def get_dl_model(self) -> Optional[Any]:
        """Get trained deep learning model."""
        return self.dl_model

    def get_ensemble(self) -> Optional[Any]:
        """Get ensemble predictor."""
        return self.ensemble

    def get_optimization_results(self) -> Optional[Dict]:
        """Get hyperparameter optimization results."""
        return self.optimization_results

    def get_training_results(self) -> Optional[Dict]:
        """Get training results and metrics."""
        return self.training_results
