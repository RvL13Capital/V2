"""
AIv4 Configuration - Modernized with Pydantic

Provides type-safe configuration with environment variable support and validation.
Replaces AIv3's dataclass-based config with Pydantic BaseSettings.
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pathlib import Path


class ConsolidationCriteria(BaseSettings):
    """
    Consolidation pattern detection criteria.

    Required criteria (3 of 4 must be met for consecutive days):
    - BBW < 30th percentile (volatility contraction)
    - ADX < 32 (low trending environment)
    - Volume < 35% of 20-day average (low volume)
    - Daily range < 65% of average (tight trading range)
    """
    bbw_percentile_threshold: float = Field(0.30, ge=0, le=1, description="BBW < 30th percentile")
    adx_threshold: float = Field(32.0, ge=0, le=100, description="ADX < 32 (low trending)")
    volume_ratio_threshold: float = Field(0.35, ge=0, le=1, description="Volume < 35% of 20-day average")
    range_ratio_threshold: float = Field(0.65, ge=0, le=1, description="Range < 65% of average")
    qualifying_days: int = Field(10, ge=1, description="Days to qualify pattern")
    max_pattern_days: int = Field(90, ge=1, description="Max pattern duration")

    model_config = {"env_prefix": "CONSOLIDATION_"}


class MemoryConfig(BaseSettings):
    """
    Memory optimization configuration for low-RAM systems.
    """
    enable_optimization: bool = Field(True, description="Enable memory optimization")
    aggressive_mode: bool = Field(False, description="Use aggressive optimization")
    memory_warning_threshold_pct: float = Field(80.0, ge=0, le=100)

    use_parquet_cache: bool = True
    parquet_compression: Literal['snappy', 'gzip'] = 'snappy'
    downcast_dtypes: bool = True
    batch_size_tickers: int = Field(10, ge=0)
    force_gc_between_batches: bool = True

    xgb_tree_method: Literal['hist', 'auto', 'exact'] = 'hist'
    xgb_max_bin: int = Field(256, ge=16, le=512)
    xgb_colsample_bytree: float = Field(0.8, ge=0, le=1)
    xgb_subsample: float = Field(0.8, ge=0, le=1)
    xgb_memory_efficient_mode: bool = False

    selective_feature_loading: bool = True
    drop_intermediate_features: bool = True
    max_features_in_memory: int = Field(100, ge=1)

    model_config = {"env_prefix": "MEMORY_"}


class MLConfig(BaseSettings):
    """Machine learning configuration for XGBoost models."""
    n_estimators: int = Field(100, ge=1)
    max_depth: int = Field(6, ge=1, le=20)
    learning_rate: float = Field(0.1, gt=0, le=1)
    random_state: int = 42
    min_samples_for_training: int = Field(100, ge=1)
    test_size: float = Field(0.2, gt=0, lt=1)

    model_config = {"env_prefix": "ML_"}


class DeepLearningConfig(BaseSettings):
    """
    Deep Learning configuration for CNN-BiLSTM-Attention model.
    """
    enabled: bool = Field(True, description="Enable deep learning models")
    sequence_length: int = Field(30, ge=1)

    cnn_filters: list[int] = Field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: list[int] = Field(default_factory=lambda: [3, 3, 3])
    lstm_units: int = Field(128, ge=1)
    attention_heads: int = Field(4, ge=1)
    dropout_rate: float = Field(0.3, ge=0, le=1)

    epochs: int = Field(100, ge=1)
    batch_size: int = Field(32, ge=1)
    learning_rate: float = Field(0.001, gt=0)
    early_stopping_patience: int = Field(15, ge=1)
    reduce_lr_patience: int = Field(7, ge=1)
    validation_split: float = Field(0.1, gt=0, lt=1)

    ensemble_strategy: Literal['weighted_average', 'voting', 'adaptive'] = 'weighted_average'
    ensemble_xgb_weight: float = Field(0.5, ge=0, le=1)
    ensemble_dl_weight: float = Field(0.5, ge=0, le=1)

    save_models: bool = True
    model_save_dir: str = "models/deep_learning"

    @field_validator('ensemble_dl_weight')
    @classmethod
    def weights_sum_to_one(cls, v, info):
        """Ensure ensemble weights sum to 1.0"""
        if 'ensemble_xgb_weight' in info.data:
            total = info.data['ensemble_xgb_weight'] + v
            if not (0.99 <= total <= 1.01):  # Allow small floating point error
                raise ValueError(f"Ensemble weights must sum to 1.0, got {total}")
        return v

    model_config = {"env_prefix": "DL_"}


class RegressionConfig(BaseSettings):
    """Regression model configuration for gain prediction."""
    enabled: bool = Field(True, description="Enable regression models")

    xgboost_n_estimators: int = Field(100, ge=1)
    xgboost_max_depth: int = Field(6, ge=1, le=20)
    xgboost_learning_rate: float = Field(0.1, gt=0, le=1)
    xgboost_objective: Literal['reg:squarederror', 'reg:pseudohubererror', 'reg:tweedie'] = 'reg:pseudohubererror'
    xgboost_huber_slope: float = Field(1.0, gt=0)

    dl_epochs: int = Field(100, ge=1)
    dl_batch_size: int = Field(32, ge=1)
    dl_learning_rate: float = Field(0.001, gt=0)
    dl_loss: Literal['huber', 'mse', 'mae'] = 'huber'

    gain_clip_max: float = Field(3.0, gt=0)

    save_models: bool = True
    model_save_dir: str = "models/regression"

    model_config = {"env_prefix": "REGRESSION_"}


class OutcomeConfig(BaseSettings):
    """Pattern outcome classification thresholds."""
    k4_threshold: float = Field(0.75, ge=0, description="75%+ gain = K4_EXCEPTIONAL")
    k3_threshold: float = Field(0.35, ge=0, description="35%+ gain = K3_STRONG")
    k2_threshold: float = Field(0.15, ge=0, description="15%+ gain = K2_QUALITY")
    k1_threshold: float = Field(0.05, ge=0, description="5%+ gain = K1_MINIMAL")
    lookforward_days: int = Field(100, ge=1, description="Outcome evaluation window")

    model_config = {"env_prefix": "OUTCOME_"}


class DataConfig(BaseSettings):
    """Data loading and storage configuration."""
    min_price: float = Field(0.05, gt=0, description="Minimum stock price filter")
    production_start_date: str = Field("2000-01-01", description="Default start date for training")
    min_years_data: float = Field(2.0, ge=0, description="Minimum years of historical data")

    # GCS configuration
    project_id: str | None = Field(None, description="GCS Project ID", validation_alias="PROJECT_ID")
    gcs_bucket_name: str | None = Field(None, description="GCS Bucket Name", validation_alias="GCS_BUCKET_NAME")
    use_gcs: bool = Field(True, description="Use Google Cloud Storage")

    # Local storage
    local_data_dir: Path = Field(Path("data/raw"), description="Local data directory")
    cache_dir: Path = Field(Path("data/cache"), description="Cache directory")

    model_config = {"env_prefix": "DATA_"}


class PathConfig(BaseSettings):
    """File and directory paths."""
    output_dir: Path = Field(Path("output"), description="Output directory")
    models_dir: Path = Field(Path("models"), description="Trained models directory")
    logs_dir: Path = Field(Path("logs"), description="Logs directory")

    model_config = {"env_prefix": "PATH_"}


class Settings(BaseSettings):
    """
    Master configuration for AIv4 system.

    Loads from environment variables with prefix support.
    Can be overridden via .env file or explicit parameters.

    Example:
        # Load from environment
        settings = Settings()

        # Override specific values
        settings = Settings(ml=MLConfig(n_estimators=200))

        # Load from .env file
        settings = Settings(_env_file='.env')
    """
    consolidation: ConsolidationCriteria = Field(default_factory=ConsolidationCriteria)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    deep_learning: DeepLearningConfig = Field(default_factory=DeepLearningConfig)
    regression: RegressionConfig = Field(default_factory=RegressionConfig)
    outcomes: OutcomeConfig = Field(default_factory=OutcomeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    # Global settings
    debug: bool = Field(False, description="Debug mode", validation_alias="DEBUG")
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = Field('INFO', description="Log level", validation_alias="LOG_LEVEL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings(**kwargs) -> Settings:
    """Reload settings with optional overrides."""
    global _settings
    _settings = Settings(**kwargs)
    return _settings
