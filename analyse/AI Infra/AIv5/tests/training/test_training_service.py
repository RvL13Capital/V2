"""
Tests for Training Service Layer.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

from training.models import ModelFactory, ModelConfig
from training.services import TrainingService, TrainingProgress, TrainingResult


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()

        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.1
        assert config.tree_method == 'hist'
        assert config.memory_efficient_mode is False

    def test_custom_values(self):
        """Test custom configuration."""
        config = ModelConfig(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            memory_efficient_mode=True
        )

        assert config.n_estimators == 200
        assert config.max_depth == 8
        assert config.learning_rate == 0.05
        assert config.memory_efficient_mode is True


class TestModelFactory:
    """Test ModelFactory."""

    def test_initialization(self):
        """Test factory initialization."""
        config = ModelConfig(n_estimators=50)
        factory = ModelFactory(config=config, enable_deep_learning=False)

        assert factory.config.n_estimators == 50
        assert factory.enable_deep_learning is False

    def test_create_classifier(self):
        """Test creating XGBoost classifier."""
        factory = ModelFactory()

        classifier = factory.create_classifier()

        assert classifier is not None
        assert hasattr(classifier, 'train')
        assert hasattr(classifier, 'predict')

    def test_create_classifier_with_override(self):
        """Test classifier creation with parameter override."""
        factory = ModelFactory()

        classifier = factory.create_classifier(
            config_override={'n_estimators': 50, 'max_depth': 4}
        )

        assert classifier is not None
        assert classifier.n_estimators == 50
        assert classifier.max_depth == 4

    def test_create_regressor(self):
        """Test creating XGBoost regressor."""
        factory = ModelFactory()

        regressor = factory.create_regressor()

        assert regressor is not None
        assert hasattr(regressor, 'train')
        assert hasattr(regressor, 'predict')

    def test_create_value_system(self):
        """Test creating pattern value system."""
        factory = ModelFactory()

        value_system = factory.create_value_system()

        assert value_system is not None
        assert hasattr(value_system, 'calculate_expected_value')

    def test_create_ev_predictor(self):
        """Test creating Expected Value predictor."""
        factory = ModelFactory()

        # Create classifier first
        classifier = factory.create_classifier()

        # Create EV predictor
        ev_predictor = factory.create_ev_predictor(classifier)

        assert ev_predictor is not None
        assert hasattr(ev_predictor, 'predict_ev')

    def test_create_ev_predictor_with_value_system(self):
        """Test EV predictor with custom value system."""
        factory = ModelFactory()

        classifier = factory.create_classifier()
        value_system = factory.create_value_system()

        ev_predictor = factory.create_ev_predictor(classifier, value_system)

        assert ev_predictor is not None
        assert ev_predictor.value_system == value_system


class TestTrainingProgress:
    """Test TrainingProgress tracking."""

    def test_initialization(self):
        """Test progress initialization."""
        progress = TrainingProgress()

        assert progress.current_step == ""
        assert progress.total_steps == 0
        assert progress.completed_steps == 0

    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        progress = TrainingProgress(total_steps=10, completed_steps=5)

        assert progress.progress_pct == 50.0

    def test_progress_percentage_zero_steps(self):
        """Test progress when total steps is zero."""
        progress = TrainingProgress(total_steps=0, completed_steps=0)

        assert progress.progress_pct == 0.0

    def test_elapsed_time(self):
        """Test elapsed time tracking."""
        progress = TrainingProgress()

        # Should be very small (just created)
        assert progress.elapsed_seconds >= 0
        assert progress.elapsed_seconds < 1.0


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_success_result(self):
        """Test successful training result."""
        result = TrainingResult(
            success=True,
            models={'classifier': Mock()},
            metrics={'accuracy': 0.95},
            duration_seconds=10.5
        )

        assert result.success is True
        assert 'classifier' in result.models
        assert result.metrics['accuracy'] == 0.95
        assert result.duration_seconds == 10.5

    def test_failure_result(self):
        """Test failed training result."""
        result = TrainingResult(
            success=False,
            error="Training failed due to insufficient data",
            duration_seconds=2.0
        )

        assert result.success is False
        assert result.error is not None
        assert "insufficient data" in result.error


class TestTrainingService:
    """Test TrainingService."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)

        # Generate sample features
        n_samples = 1000
        n_features = 10

        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )

        # Generate sample labels (6 classes: K0-K5)
        y = pd.Series(np.random.choice(['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], n_samples))

        # Generate sample gains
        y_gain = pd.Series(np.random.randn(n_samples) * 0.3)  # ~30% std

        return X, y, y_gain

    def test_initialization(self, temp_output_dir):
        """Test service initialization."""
        factory = ModelFactory()
        service = TrainingService(factory, output_dir=temp_output_dir)

        assert service.model_factory == factory
        assert service.output_dir == Path(temp_output_dir)
        assert service.output_dir.exists()

    def test_train_classifier_basic(self, temp_output_dir, sample_data):
        """Test basic classifier training."""
        X, y, _ = sample_data

        factory = ModelFactory()
        service = TrainingService(factory, output_dir=temp_output_dir)

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train classifier
        result = service.train_classifier(
            X_train, y_train,
            X_val, y_val,
            optimize_hyperparams=False,
            save_model=True
        )

        # Check result
        assert result.success is True
        assert 'classifier' in result.models
        assert result.duration_seconds > 0

        # Check model was saved
        assert 'classifier' in result.model_paths
        assert result.model_paths['classifier'].exists()

    def test_train_classifier_without_validation(self, temp_output_dir, sample_data):
        """Test classifier training without validation set."""
        X, y, _ = sample_data

        factory = ModelFactory()
        service = TrainingService(factory, output_dir=temp_output_dir)

        result = service.train_classifier(
            X, y,
            save_model=False
        )

        assert result.success is True
        assert 'classifier' in result.models

    def test_train_regressor(self, temp_output_dir, sample_data):
        """Test regressor training."""
        X, _, y_gain = sample_data

        factory = ModelFactory()
        service = TrainingService(factory, output_dir=temp_output_dir)

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_gain[:split_idx], y_gain[split_idx:]

        result = service.train_regressor(
            X_train, y_train,
            X_val, y_val,
            save_model=True
        )

        assert result.success is True
        assert 'regressor' in result.models

    def test_train_complete_system(self, temp_output_dir, sample_data):
        """Test training complete system."""
        X, y, y_gain = sample_data

        # Create full DataFrame
        patterns_df = X.copy()
        patterns_df['outcome_class'] = y
        patterns_df['total_gain_pct'] = y_gain

        factory = ModelFactory()
        service = TrainingService(factory, output_dir=temp_output_dir)

        result = service.train_complete_system(
            patterns_df,
            feature_columns=X.columns.tolist(),
            target_column='outcome_class',
            gain_column='total_gain_pct',
            optimize_hyperparams=False,
            enable_regression=True,
            enable_deep_learning=False,  # Skip DL for speed
            save_models=True
        )

        assert result.success is True
        assert 'classifier' in result.models
        assert 'regressor' in result.models
        assert 'ev_predictor' in result.models

    def test_model_registry(self, temp_output_dir, sample_data):
        """Test model registry functionality."""
        X, y, _ = sample_data

        factory = ModelFactory()
        service = TrainingService(factory, output_dir=temp_output_dir)

        # Train classifier
        result = service.train_classifier(X, y, save_model=False)

        # Check model is in registry
        model = service.get_model('xgboost_classifier')
        assert model is not None

        # List models
        models = service.list_models()
        assert 'xgboost_classifier' in models

    def test_get_statistics(self, temp_output_dir):
        """Test getting training statistics."""
        factory = ModelFactory()
        service = TrainingService(factory, output_dir=temp_output_dir)

        stats = service.get_statistics()

        assert 'models_trained' in stats
        assert 'model_names' in stats
        assert 'output_dir' in stats
        assert stats['output_dir'] == str(temp_output_dir)


# Integration tests

class TestTrainingIntegration:
    """Integration tests for training workflow."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_full_training_workflow(self, temp_output_dir):
        """Test complete training workflow end-to-end."""
        # Generate realistic data
        np.random.seed(42)
        n_samples = 500

        # Features
        X = pd.DataFrame({
            'avg_bbw_20d': np.random.rand(n_samples) * 0.05,
            'avg_adx_20d': np.random.rand(n_samples) * 40,
            'volume_ratio': np.random.rand(n_samples) * 2,
            'range_ratio': np.random.rand(n_samples) * 1.5,
            'pattern_duration': np.random.randint(10, 50, n_samples),
        })

        # Outcomes
        y = pd.Series(np.random.choice(['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], n_samples))
        y_gain = pd.Series(np.random.randn(n_samples) * 0.4)

        # Create patterns DataFrame
        patterns_df = X.copy()
        patterns_df['outcome_class'] = y
        patterns_df['total_gain_pct'] = y_gain

        # Create factory and service
        config = ModelConfig(n_estimators=10, max_depth=3)  # Small for speed
        factory = ModelFactory(config=config)
        service = TrainingService(factory, output_dir=temp_output_dir)

        # Train complete system
        result = service.train_complete_system(
            patterns_df,
            feature_columns=X.columns.tolist(),
            optimize_hyperparams=False,
            enable_regression=True,
            save_models=True
        )

        # Verify results
        assert result.success is True
        assert len(result.models) >= 3  # At least classifier, regressor, ev_predictor

        # Verify models were saved
        saved_files = list(Path(temp_output_dir).glob('*.joblib'))
        assert len(saved_files) >= 2  # At least classifier and regressor

        # Test prediction with EV predictor
        ev_predictor = result.models['ev_predictor']
        predictions = ev_predictor.predict_ev(X[:10])

        assert len(predictions) == 10
        assert 'expected_value' in predictions.columns
        assert 'signal_strength' in predictions.columns
