"""
model_brain.py

XGBoost Classifier pipeline with Purged Time Series Cross Validation.
Optimized for Profit Factor validation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import precision_score, recall_score
from typing import List, Generator


class PurgedTimeSeriesCV:
    """
    Purged Time Series Splitter.
    Ensures no overlap between train and test sets with an embargo period.
    """
    def __init__(self, n_splits: int = 5, embargo_days: int = 10, test_size_days: int = 60):
        self.n_splits = n_splits
        self.embargo = embargo_days
        self.test_size = test_size_days

    def split(self, X: pd.DataFrame, y=None, groups=None) -> Generator:
        """
        Yields indices for train/test.
        Assumes X is sorted by Date index.
        """
        indices = np.arange(len(X))

        # Simple implementation: Moving window or expanding window?
        # Prompt asks for Purged Time Series CV. Usually implies Walk-Forward.
        # We will iterate backwards from end of dataset.

        test_points = self.test_size

        for i in range(self.n_splits):
            # Define Test Range
            test_end = len(X) - (i * test_points)
            test_start = test_end - test_points

            if test_start <= 0:
                break

            # Define Train Range (Purged/Embargoed)
            # Train ends before (Test Start - Embargo)
            train_end = test_start - self.embargo

            if train_end <= 0:
                break

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits


def compute_profit_factor(pnl_series: pd.Series) -> float:
    """
    Profit Factor = Gross Profit / Abs(Gross Loss).
    """
    winners = pnl_series[pnl_series > 0].sum()
    losers = pnl_series[pnl_series < 0].sum()

    if losers == 0:
        return np.inf if winners > 0 else 0.0

    return winners / abs(losers)


class MonsterModel:
    """
    XGBoost-based classifier for Monster Hunter strategy.
    Predicts probability of a trade hitting the 2x target.
    """

    def __init__(self, params: dict = None):
        self.params = params if params else {
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'random_state': 42
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.feature_names_ = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model on labeled data."""
        self.feature_names_ = list(X_train.columns)
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return binary predictions."""
        return self.model.predict(X_test)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return probability of positive class (monster target hit)."""
        return self.model.predict_proba(X_test)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance scores."""
        if self.feature_names_ is None:
            raise ValueError("Model must be trained before getting feature importance")

        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save(self, path: str):
        """Save model to disk."""
        self.model.save_model(path)

    def load(self, path: str):
        """Load model from disk."""
        self.model.load_model(path)
