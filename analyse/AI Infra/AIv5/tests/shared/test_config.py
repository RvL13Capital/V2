"""
Tests for modernized Pydantic configuration.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.config import (
    Settings,
    get_settings,
    reload_settings,
    ConsolidationCriteria,
    MLConfig,
    DeepLearningConfig,
)


class TestConsolidationCriteria:
    """Test consolidation criteria configuration."""

    def test_default_values(self):
        """Test default criteria values."""
        criteria = ConsolidationCriteria()

        assert criteria.bbw_percentile_threshold == 0.30
        assert criteria.adx_threshold == 32.0
        assert criteria.volume_ratio_threshold == 0.35
        assert criteria.range_ratio_threshold == 0.65
        assert criteria.qualifying_days == 10
        assert criteria.max_pattern_days == 90

    def test_validation_percentile(self):
        """Test BBW percentile validation (0-1)."""
        # Valid
        criteria = ConsolidationCriteria(bbw_percentile_threshold=0.5)
        assert criteria.bbw_percentile_threshold == 0.5

        # Invalid - too low
        with pytest.raises(ValidationError):
            ConsolidationCriteria(bbw_percentile_threshold=-0.1)

        # Invalid - too high
        with pytest.raises(ValidationError):
            ConsolidationCriteria(bbw_percentile_threshold=1.5)

    def test_validation_adx(self):
        """Test ADX threshold validation."""
        # Valid
        criteria = ConsolidationCriteria(adx_threshold=25.0)
        assert criteria.adx_threshold == 25.0

        # Invalid - negative
        with pytest.raises(ValidationError):
            ConsolidationCriteria(adx_threshold=-10.0)

        # Invalid - too high
        with pytest.raises(ValidationError):
            ConsolidationCriteria(adx_threshold=150.0)


class TestMLConfig:
    """Test ML configuration."""

    def test_default_values(self):
        """Test default ML config values."""
        config = MLConfig()

        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.1
        assert config.random_state == 42

    def test_validation_estimators(self):
        """Test n_estimators validation (>= 1)."""
        # Valid
        config = MLConfig(n_estimators=50)
        assert config.n_estimators == 50

        # Invalid
        with pytest.raises(ValidationError):
            MLConfig(n_estimators=0)

        with pytest.raises(ValidationError):
            MLConfig(n_estimators=-10)

    def test_validation_max_depth(self):
        """Test max_depth validation (1-20)."""
        # Valid
        config = MLConfig(max_depth=10)
        assert config.max_depth == 10

        # Invalid - too low
        with pytest.raises(ValidationError):
            MLConfig(max_depth=0)

        # Invalid - too high
        with pytest.raises(ValidationError):
            MLConfig(max_depth=30)


class TestDeepLearningConfig:
    """Test deep learning configuration."""

    def test_default_values(self):
        """Test default DL config values."""
        config = DeepLearningConfig()

        assert config.enabled is True
        assert config.sequence_length == 30
        assert config.lstm_units == 128
        assert config.attention_heads == 4
        assert config.ensemble_strategy == 'weighted_average'

    def test_ensemble_weights_validation(self):
        """Test ensemble weights sum to 1.0."""
        # Valid - weights sum to 1.0
        config = DeepLearningConfig(
            ensemble_xgb_weight=0.4,
            ensemble_dl_weight=0.6
        )
        assert config.ensemble_xgb_weight + config.ensemble_dl_weight == 1.0

        # Invalid - weights don't sum to 1.0
        with pytest.raises(ValidationError):
            DeepLearningConfig(
                ensemble_xgb_weight=0.3,
                ensemble_dl_weight=0.5  # Sum = 0.8, should be 1.0
            )

    def test_ensemble_strategy_literal(self):
        """Test ensemble strategy accepts only valid literals."""
        # Valid
        config = DeepLearningConfig(ensemble_strategy='voting')
        assert config.ensemble_strategy == 'voting'

        config = DeepLearningConfig(ensemble_strategy='adaptive')
        assert config.ensemble_strategy == 'adaptive'

        # Invalid
        with pytest.raises(ValidationError):
            DeepLearningConfig(ensemble_strategy='invalid_strategy')


class TestSettings:
    """Test master Settings class."""

    def test_get_settings_singleton(self):
        """Test get_settings returns same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reload_settings(self):
        """Test reload_settings creates new instance."""
        settings1 = get_settings()
        settings2 = reload_settings(ml=MLConfig(n_estimators=200))

        assert settings1 is not settings2
        assert settings2.ml.n_estimators == 200

    def test_nested_config_access(self):
        """Test accessing nested configuration."""
        settings = get_settings()

        # Access nested configs
        assert hasattr(settings, 'consolidation')
        assert hasattr(settings, 'ml')
        assert hasattr(settings, 'deep_learning')

        # Access nested values
        assert settings.consolidation.bbw_percentile_threshold == 0.30
        assert settings.ml.n_estimators == 100

    def test_default_log_level(self):
        """Test default log level is INFO."""
        settings = get_settings()
        assert settings.log_level == 'INFO'

    def test_environment_variable_override(self, monkeypatch):
        """Test environment variables override defaults."""
        # Set environment variable
        monkeypatch.setenv('ML_N_ESTIMATORS', '200')

        # Reload settings
        settings = reload_settings()

        assert settings.ml.n_estimators == 200

        # Clean up
        monkeypatch.delenv('ML_N_ESTIMATORS', raising=False)
