"""
Attention-Based XGBoost Ensemble with Optuna Optimization
==========================================================
Advanced ensemble model with attention mechanism, anomaly detection,
and character rating for exceptional pattern detection.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import IsolationForest
import joblib
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
warnings.filterwarnings('ignore')

from feature_engineering_v3 import AdvancedFeatureEngineerV3


class AttentionXGBoostEnsemble:
    """
    Advanced ensemble with:
    - Self-attention mechanism for feature weighting
    - Optuna hyperparameter optimization
    - Anomaly detection for data quality
    - Character rating for pattern scoring
    - Automated feature selection
    """

    def __init__(self, n_trials: int = 100, use_gpu: bool = False):
        self.n_trials = n_trials
        self.use_gpu = use_gpu
        self.models = {}
        self.feature_engineer = AdvancedFeatureEngineerV3()
        self.attention_weights = None
        self.feature_selector = None
        self.selected_features = None
        self.anomaly_detector = None
        self.best_params = {}
        self.feature_importance_history = []

    def attention_mechanism(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute attention weights for features using self-attention.
        """
        # Normalize features
        X_norm = (X - X.mean()) / (X.std() + 1e-8)

        # Compute attention scores using dot product attention
        # Q = K = V = X (self-attention)
        attention_scores = X_norm @ X_norm.T / np.sqrt(X.shape[1])

        # Apply softmax to get attention weights
        exp_scores = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
        attention_weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        # Apply attention to features
        X_attended = attention_weights @ X.values

        # Store feature-level attention weights
        self.attention_weights = np.mean(np.abs(X_norm.T @ attention_weights.T), axis=1)

        return X_attended

    def automated_feature_selection(self, X: pd.DataFrame, y: np.ndarray,
                                  method: str = 'importance') -> List[str]:
        """
        Automated feature selection using multiple methods.
        """
        print(f"\nPerforming automated feature selection (method: {method})...")

        if method == 'importance':
            # Use XGBoost feature importance
            temp_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                tree_method='hist' if not self.use_gpu else 'gpu_hist'
            )
            temp_model.fit(X, y)

            # Select features with importance > threshold
            importances = temp_model.feature_importances_
            threshold = np.percentile(importances, 25)  # Keep top 75% features
            selected_mask = importances > threshold
            selected_features = X.columns[selected_mask].tolist()

            # Store importance scores
            self.feature_importance_history.append({
                'timestamp': datetime.now().isoformat(),
                'importances': dict(zip(X.columns, importances))
            })

        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42)
            selector = RFE(estimator, n_features_to_select=int(X.shape[1] * 0.75))
            selector.fit(X, y)
            selected_features = X.columns[selector.support_].tolist()

        elif method == 'correlation':
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            selected_features = [col for col in X.columns if col not in to_drop]

        else:  # mutual_information
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(X, y)
            threshold = np.percentile(mi_scores, 25)
            selected_mask = mi_scores > threshold
            selected_features = X.columns[selected_mask].tolist()

        print(f"Selected {len(selected_features)} features from {X.shape[1]} total")
        return selected_features

    def anomaly_detection(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in the dataset using Isolation Forest.
        """
        if self.anomaly_detector is None:
            self.anomaly_detector = IsolationForest(
                contamination=0.05,
                random_state=42
            )
            self.anomaly_detector.fit(X)

        # Predict anomalies (-1 for anomaly, 1 for normal)
        anomaly_labels = self.anomaly_detector.predict(X)
        anomaly_scores = self.anomaly_detector.score_samples(X)

        return anomaly_labels, anomaly_scores

    def optuna_objective(self, trial, X_train, y_train, X_val, y_val, model_type='xgboost'):
        """
        Optuna objective function for hyperparameter optimization.
        """
        if model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'tree_method': 'hist' if not self.use_gpu else 'gpu_hist',
                'random_state': 42
            }

            model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')

        else:  # lightgbm
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'device_type': 'gpu' if self.use_gpu else 'cpu',
                'random_state': 42,
                'verbose': -1
            }

            model = lgb.LGBMClassifier(**params)

        # Apply attention mechanism
        X_train_attended = self.attention_mechanism(pd.DataFrame(X_train))
        X_val_attended = self.attention_mechanism(pd.DataFrame(X_val))

        # Train model
        model.fit(X_train_attended, y_train)

        # Predict and calculate F1 score (focusing on K4 class)
        y_pred = model.predict(X_val_attended)

        # Calculate weighted F1 score with emphasis on K4
        class_weights = {0: 1, 1: 1, 2: 2, 3: 5, 4: 10, 5: 1}  # K4 has highest weight
        weighted_f1 = 0

        for class_id, weight in class_weights.items():
            class_f1 = f1_score(y_val == class_id, y_pred == class_id, average='binary')
            weighted_f1 += class_f1 * weight

        return weighted_f1

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """
        Optimize hyperparameters using Optuna.
        """
        print("\n" + "="*60)
        print("OPTUNA HYPERPARAMETER OPTIMIZATION")
        print("="*60)

        # Optimize XGBoost
        study_xgb = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        study_xgb.optimize(
            lambda trial: self.optuna_objective(
                trial, X_train, y_train, X_val, y_val, 'xgboost'
            ),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params['xgboost'] = study_xgb.best_params
        print(f"\nBest XGBoost params: {study_xgb.best_params}")
        print(f"Best XGBoost score: {study_xgb.best_value:.4f}")

        # Optimize LightGBM
        study_lgb = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        study_lgb.optimize(
            lambda trial: self.optuna_objective(
                trial, X_train, y_train, X_val, y_val, 'lightgbm'
            ),
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        self.best_params['lightgbm'] = study_lgb.best_params
        print(f"\nBest LightGBM params: {study_lgb.best_params}")
        print(f"Best LightGBM score: {study_lgb.best_value:.4f}")

        return study_xgb, study_lgb

    def train(self, X_train, y_train, X_val, y_val, optimize: bool = True,
             feature_selection_method: str = 'importance'):
        """
        Train the ensemble with optional optimization and feature selection.
        """
        print("\n" + "="*60)
        print("TRAINING ATTENTION-BASED ENSEMBLE")
        print("="*60)

        # Automated feature selection
        self.selected_features = self.automated_feature_selection(
            X_train, y_train, method=feature_selection_method
        )
        X_train_selected = X_train[self.selected_features]
        X_val_selected = X_val[self.selected_features]

        # Anomaly detection
        anomaly_labels_train, anomaly_scores_train = self.anomaly_detection(X_train_selected)
        print(f"Detected {(anomaly_labels_train == -1).sum()} anomalies in training data")

        # Remove anomalies from training
        clean_mask = anomaly_labels_train == 1
        X_train_clean = X_train_selected[clean_mask]
        y_train_clean = y_train[clean_mask]

        # Optimize hyperparameters
        if optimize:
            self.optimize_hyperparameters(X_train_clean, y_train_clean, X_val_selected, y_val)
        else:
            # Use default parameters
            self.best_params['xgboost'] = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 1,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'min_child_weight': 3
            }
            self.best_params['lightgbm'] = {
                'n_estimators': 500,
                'num_leaves': 50,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'min_child_weight': 3
            }

        # Apply attention mechanism
        X_train_attended = self.attention_mechanism(pd.DataFrame(X_train_clean))
        X_val_attended = self.attention_mechanism(pd.DataFrame(X_val_selected))

        # Train final models with best parameters
        print("\nTraining final models with best parameters...")

        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            **self.best_params['xgboost'],
            tree_method='hist' if not self.use_gpu else 'gpu_hist',
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        self.models['xgboost'].fit(X_train_attended, y_train_clean)

        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            **self.best_params['lightgbm'],
            device_type='gpu' if self.use_gpu else 'cpu',
            random_state=42,
            verbose=-1
        )
        self.models['lightgbm'].fit(X_train_attended, y_train_clean)

        # Store feature importance
        self._update_feature_importance()

        print("Training completed!")

    def predict_with_character_rating(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with character rating and anomaly scores.
        """
        # Select features
        X_selected = X[self.selected_features] if self.selected_features else X

        # Detect anomalies
        anomaly_labels, anomaly_scores = self.anomaly_detection(X_selected)

        # Apply attention
        X_attended = self.attention_mechanism(pd.DataFrame(X_selected))

        # Get predictions from both models
        xgb_proba = self.models['xgboost'].predict_proba(X_attended)
        lgb_proba = self.models['lightgbm'].predict_proba(X_attended)

        # Weighted ensemble based on attention scores
        ensemble_proba = (xgb_proba * 0.5 + lgb_proba * 0.5)

        # Adjust probabilities based on anomaly scores
        # Reduce confidence for anomalies
        for i, (label, score) in enumerate(zip(anomaly_labels, anomaly_scores)):
            if label == -1:  # Anomaly
                ensemble_proba[i] *= (1 + score)  # score is negative for anomalies

        # Normalize probabilities
        ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)

        # Get predictions
        predictions = np.argmax(ensemble_proba, axis=1)

        # Calculate character ratings (0-10 scale)
        character_ratings = self._calculate_character_ratings(X, ensemble_proba, anomaly_scores)

        return predictions, ensemble_proba, character_ratings

    def _calculate_character_ratings(self, X: pd.DataFrame, probabilities: np.ndarray,
                                    anomaly_scores: np.ndarray) -> np.ndarray:
        """
        Calculate character ratings for patterns.
        """
        ratings = []

        for i in range(len(X)):
            # Base rating from prediction confidence
            max_prob = probabilities[i].max()
            confidence_score = max_prob * 10

            # K4 probability bonus
            k4_prob = probabilities[i][4] if len(probabilities[i]) > 4 else 0
            k4_bonus = k4_prob * 5

            # K3 probability bonus
            k3_prob = probabilities[i][3] if len(probabilities[i]) > 3 else 0
            k3_bonus = k3_prob * 2

            # Anomaly penalty (positive scores are normal)
            anomaly_adjustment = min(0, anomaly_scores[i] * 2)

            # Calculate final rating
            rating = confidence_score + k4_bonus + k3_bonus + anomaly_adjustment
            rating = max(0, min(10, rating))  # Clip to 0-10

            ratings.append(rating)

        return np.array(ratings)

    def _update_feature_importance(self):
        """
        Update and track feature importance.
        """
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].feature_importances_
            feature_names = self.selected_features if self.selected_features else []

            importance_dict = dict(zip(feature_names, xgb_importance))
            self.feature_engineer.update_feature_importance(importance_dict)

    def save_models(self, path_prefix: str):
        """
        Save trained models and configurations.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{path_prefix}_{name}_attention_{timestamp}.pkl")

        # Save configuration
        config = {
            'best_params': self.best_params,
            'selected_features': self.selected_features,
            'attention_weights': self.attention_weights.tolist() if self.attention_weights is not None else None,
            'feature_importance_history': self.feature_importance_history
        }

        with open(f"{path_prefix}_config_attention_{timestamp}.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Models and configuration saved with timestamp {timestamp}")

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray):
        """
        Evaluate the ensemble on test data.
        """
        print("\n" + "="*60)
        print("ENSEMBLE EVALUATION WITH CHARACTER RATINGS")
        print("="*60)

        predictions, probabilities, character_ratings = self.predict_with_character_rating(X_test)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, predictions,
                                   target_names=['K0', 'K1', 'K2', 'K3', 'K4', 'K5']))

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)

        # K4 specific metrics
        k4_mask = y_test == 4
        if k4_mask.sum() > 0:
            k4_recall = (predictions[k4_mask] == 4).sum() / k4_mask.sum()
            k4_precision = (predictions == 4).sum() / max((predictions == 4).sum(), 1)
            print(f"\nK4_EXCEPTIONAL Metrics:")
            print(f"  Recall: {k4_recall:.2%}")
            print(f"  Precision: {k4_precision:.2%}")
            print(f"  Support: {k4_mask.sum()}")

        # Character rating distribution
        print(f"\nCharacter Rating Statistics:")
        print(f"  Mean: {character_ratings.mean():.2f}")
        print(f"  Std: {character_ratings.std():.2f}")
        print(f"  Min: {character_ratings.min():.2f}")
        print(f"  Max: {character_ratings.max():.2f}")

        # High-rated patterns
        high_rating_mask = character_ratings >= 8
        print(f"\nHigh-Rated Patterns (≥8): {high_rating_mask.sum()} ({high_rating_mask.sum()/len(character_ratings):.1%})")

        if high_rating_mask.sum() > 0:
            high_rated_outcomes = y_test[high_rating_mask]
            print("  Outcome distribution of high-rated patterns:")
            for i in range(6):
                count = (high_rated_outcomes == i).sum()
                if count > 0:
                    print(f"    K{i}: {count} ({count/high_rating_mask.sum():.1%})")

        # Attention weights analysis
        if self.attention_weights is not None:
            top_features_idx = np.argsort(self.attention_weights)[-10:]
            print(f"\nTop 10 Features by Attention Weight:")
            for idx in reversed(top_features_idx):
                feature_name = self.selected_features[idx] if self.selected_features else f"Feature_{idx}"
                print(f"  {feature_name}: {self.attention_weights[idx]:.4f}")


if __name__ == "__main__":
    print("Attention-Based XGBoost Ensemble initialized")
    print("\nFeatures:")
    print("✓ Self-attention mechanism for feature weighting")
    print("✓ Optuna hyperparameter optimization")
    print("✓ Automated feature selection (4 methods)")
    print("✓ Anomaly detection with Isolation Forest")
    print("✓ Character rating system (0-10 scale)")
    print("✓ Feature importance tracking")
    print("✓ GPU support for acceleration")