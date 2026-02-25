"""
Production Configuration Management
====================================

Centralized production configuration with environment-based settings,
feature flags, and deployment configurations.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL",
        "sqlite:///trans_production.db"
    ))
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    use_pool: bool = True
    retry_on_disconnect: bool = True
    max_retries: int = 3


@dataclass
class ModelConfig:
    """Model configuration."""
    model_dir: str = "models/artifacts"
    default_version: Optional[str] = None
    auto_load_production: bool = True
    cache_models: bool = True
    max_cached_models: int = 3
    device: str = field(default_factory=lambda: "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu")
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class APIConfig:
    """API configuration."""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "4")))
    reload: bool = False
    cors_origins: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(","))
    api_key_header: str = "X-API-Key"
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    request_timeout: int = 300  # seconds
    max_request_size: int = 100 * 1024 * 1024  # 100 MB


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: str = "logs"
    use_json_format: bool = True
    enable_console: bool = True
    enable_file: bool = True
    enable_database: bool = False
    rotate_size_mb: int = 100
    rotate_count: int = 10
    include_stacktrace: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool = True
    collection_interval: int = 60  # seconds
    retention_hours: int = 24
    export_prometheus: bool = True
    export_json: bool = True
    alert_email_enabled: bool = False
    alert_email_recipients: List[str] = field(default_factory=list)
    alert_slack_enabled: bool = False
    alert_slack_webhook: Optional[str] = field(default_factory=lambda: os.getenv("SLACK_WEBHOOK"))


@dataclass
class ScannerConfig:
    """Pattern scanner configuration."""
    min_liquidity_dollar: float = field(default_factory=lambda: float(os.getenv("MIN_LIQUIDITY", "50000")))
    max_patterns_per_ticker: int = 10
    parallel_workers: int = 4
    batch_size: int = 100
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    enable_accumulation_detection: bool = field(default_factory=lambda: os.getenv("ENABLE_ACCUMULATION", "true").lower() == "true")

    # V17 specific settings
    use_v17_labeling: bool = True
    exclude_grey_zone: bool = True
    min_risk_multiple: float = 1.0
    max_risk_multiple: float = 10.0


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    gcs_enabled: bool = field(default_factory=lambda: os.getenv("GCS_ENABLED", "false").lower() == "true")
    gcs_bucket: Optional[str] = field(default_factory=lambda: os.getenv("GCS_BUCKET_NAME"))
    gcs_project: Optional[str] = field(default_factory=lambda: os.getenv("PROJECT_ID"))
    gcs_credentials: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    cache_local: bool = True
    max_cache_size_gb: float = 10.0


@dataclass
class FeatureFlags:
    """Feature flags for gradual rollout and A/B testing."""
    enable_v17_labeling: bool = True
    enable_hybrid_model: bool = True
    enable_async_predictions: bool = True
    enable_pattern_caching: bool = True
    enable_model_ab_testing: bool = False
    enable_advanced_metrics: bool = True
    enable_auto_retraining: bool = False
    enable_data_validation: bool = True
    enable_circuit_breakers: bool = True
    enable_graceful_degradation: bool = True


@dataclass
class ProductionConfig:
    """Complete production configuration."""
    environment: Environment = field(default_factory=lambda: Environment(
        os.getenv("ENVIRONMENT", "development").lower()
    ))

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)

    # Security
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "change-this-in-production"))
    api_keys: List[str] = field(default_factory=lambda: os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else [])

    # Performance
    max_concurrent_requests: int = 100
    request_timeout: int = 300

    # Deployment
    version: str = field(default_factory=lambda: os.getenv("APP_VERSION", "1.0.0"))
    deployment_id: str = field(default_factory=lambda: os.getenv("DEPLOYMENT_ID", "local"))

    def __post_init__(self):
        """Validate and adjust configuration based on environment."""
        # Adjust settings based on environment
        if self.environment == Environment.PRODUCTION:
            self.logging.level = "WARNING"
            self.database.echo = False
            self.api.reload = False
            self.monitoring.enabled = True
            self.feature_flags.enable_circuit_breakers = True

        elif self.environment == Environment.DEVELOPMENT:
            self.logging.level = "DEBUG"
            self.database.echo = True
            self.api.reload = True
            self.monitoring.collection_interval = 10

        elif self.environment == Environment.TESTING:
            self.database.url = "sqlite:///:memory:"
            self.logging.enable_file = False
            self.monitoring.enabled = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON."""
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, Path):
                return str(obj)
            return str(obj)

        return json.dumps(self.to_dict(), default=serialize, indent=indent)

    def save(self, path: Path):
        """Save configuration to file."""
        path = Path(path)
        if path.suffix == ".json":
            with open(path, 'w') as f:
                f.write(self.to_json())
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    @classmethod
    def load(cls, path: Path) -> "ProductionConfig":
        """Load configuration from file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Parse environment
        if "environment" in data:
            data["environment"] = Environment(data["environment"])

        return cls(**data)

    def validate(self) -> List[str]:
        """
        Validate configuration settings.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Database validation
        if self.environment == Environment.PRODUCTION:
            if "sqlite" in self.database.url:
                errors.append("SQLite not recommended for production")

        # API validation
        if self.api.port < 1024 and os.name != 'nt':
            errors.append("Port < 1024 requires root privileges on Unix")

        # Security validation
        if self.environment == Environment.PRODUCTION:
            if self.secret_key == "change-this-in-production":
                errors.append("Default secret key used in production")
            if not self.api_keys:
                errors.append("No API keys configured for production")

        # GCS validation
        if self.data.gcs_enabled:
            if not self.data.gcs_bucket:
                errors.append("GCS enabled but bucket not configured")
            if not self.data.gcs_credentials:
                errors.append("GCS enabled but credentials not configured")

        # Model validation
        if self.model.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append("CUDA requested but not available")
            except ImportError:
                errors.append("PyTorch not installed for CUDA validation")

        return errors


# Global configuration instance
_config: Optional[ProductionConfig] = None


def get_config() -> ProductionConfig:
    """Get or create global configuration."""
    global _config
    if not _config:
        # Try to load from file first
        config_path = Path(os.getenv("CONFIG_PATH", "config/production.yaml"))
        if config_path.exists():
            _config = ProductionConfig.load(config_path)
        else:
            _config = ProductionConfig()
    return _config


def reload_config(path: Optional[Path] = None):
    """
    Reload configuration from file or environment.

    Args:
        path: Optional path to configuration file
    """
    global _config
    if path:
        _config = ProductionConfig.load(path)
    else:
        _config = ProductionConfig()
    return _config


if __name__ == "__main__":
    # Test configuration
    config = get_config()

    print("Production Configuration")
    print("=" * 50)
    print(f"Environment: {config.environment.value}")
    print(f"Database: {config.database.url}")
    print(f"API: {config.api.host}:{config.api.port}")
    print(f"Logging: {config.logging.level}")
    print(f"Model Device: {config.model.device}")
    print()

    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid")

    # Save example configuration
    example_path = Path("config/example.json")
    example_path.parent.mkdir(exist_ok=True)
    config.save(example_path)
    print(f"\nExample configuration saved to: {example_path}")