"""
Centralized Configuration Module for AIv3 System
Handles all system configuration, credentials, and settings
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class GCSConfig:
    """Google Cloud Storage configuration"""
    project_id: str = "ignition-ki-csv-storage"
    bucket_name: str = "ignition-ki-csv-data-2025-user123"
    credentials_path: Optional[str] = None

    def __post_init__(self):
        if not self.credentials_path:
            self.credentials_path = self._find_credentials()

    def _find_credentials(self) -> str:
        """Find GCS credentials file in common locations"""
        possible_paths = [
            Path("./gcs-key.json"),
            Path("./gcs_credentials.json"),
            Path(r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0 (1).json"),
            Path(r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json"),
        ]

        # Check environment variable
        if env_path := os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            possible_paths.insert(0, Path(env_path))

        for path in possible_paths:
            if path.exists():
                logger.info(f"Found GCS credentials at: {path}")
                return str(path)

        raise FileNotFoundError(
            "GCS credentials file not found. Please set GOOGLE_APPLICATION_CREDENTIALS "
            "environment variable or place credentials in project directory."
        )


@dataclass
class PatternDetectionConfig:
    """Pattern detection parameters"""
    # Consolidation criteria
    min_duration_days: int = 5
    max_duration_days: int = 100

    # BBW (Bollinger Band Width) thresholds
    bbw_percentile_threshold: float = 30.0  # Below 30th percentile

    # Volume thresholds
    volume_ratio_threshold: float = 0.35  # Below 35% of 20-day average

    # Range thresholds
    range_ratio_threshold: float = 0.65  # Below 65% of 20-day average

    # ADX threshold
    adx_threshold: float = 32.0  # Below 32 indicates low trending

    # Breakout parameters
    breakout_multiplier: float = 1.005  # 0.5% above upper boundary
    volume_spike_threshold: float = 1.5  # 50% above average

    # Outcome evaluation
    evaluation_days: int = 30  # Days to track after breakout

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PatternDetectionConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


@dataclass
class OutcomeClassConfig:
    """Outcome classification and strategic values"""
    # Outcome thresholds (percentage gains)
    explosive_threshold: float = 75.0  # K4: >75% gain
    strong_threshold: float = 35.0     # K3: 35-75% gain
    quality_threshold: float = 15.0    # K2: 15-35% gain
    minimal_threshold: float = 5.0     # K1: 5-15% gain

    # Strategic values
    values = {
        'K4_EXPLOSIVE': 10.0,    # Exceptional moves
        'K3_STRONG': 3.0,        # Strong moves
        'K2_QUALITY': 1.0,       # Quality moves
        'K1_MINIMAL': -0.2,      # Minimal moves (not worth risk)
        'K0_STAGNANT': -2.0,     # Stagnant (<5% gain)
        'K5_FAILED': -10.0       # Failed patterns (breakdown)
    }

    # Signal thresholds (Expected Value)
    strong_signal_ev: float = 5.0
    good_signal_ev: float = 3.0
    moderate_signal_ev: float = 1.0
    max_failure_probability: float = 0.3

    def classify_outcome(self, gain_pct: float) -> str:
        """Classify outcome based on gain percentage"""
        if gain_pct >= self.explosive_threshold:
            return 'K4_EXPLOSIVE'
        elif gain_pct >= self.strong_threshold:
            return 'K3_STRONG'
        elif gain_pct >= self.quality_threshold:
            return 'K2_QUALITY'
        elif gain_pct >= self.minimal_threshold:
            return 'K1_MINIMAL'
        elif gain_pct >= 0:
            return 'K0_STAGNANT'
        else:
            return 'K5_FAILED'

    def get_value(self, outcome_class: str) -> float:
        """Get strategic value for outcome class"""
        return self.values.get(outcome_class, 0.0)


@dataclass
class AnalysisConfig:
    """Analysis and processing configuration"""
    # Data processing
    batch_size: int = 100
    max_tickers: Optional[int] = None
    parallel_workers: int = 4

    # Time windows
    lookback_days: int = 365
    min_data_points: int = 100

    # Output settings
    output_dir: Path = Path("./output")
    save_intermediate: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class SystemConfig:
    """Main system configuration singleton"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.gcs = GCSConfig()
            self.pattern = PatternDetectionConfig()
            self.outcome = OutcomeClassConfig()
            self.analysis = AnalysisConfig()
            self.initialized = True
            self._load_custom_config()

    def _load_custom_config(self):
        """Load custom configuration from file if exists"""
        config_file = Path("./config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)

                # Update configurations
                if 'gcs' in custom_config:
                    for key, value in custom_config['gcs'].items():
                        setattr(self.gcs, key, value)

                if 'pattern' in custom_config:
                    self.pattern = PatternDetectionConfig.from_dict(custom_config['pattern'])

                if 'analysis' in custom_config:
                    for key, value in custom_config['analysis'].items():
                        if hasattr(self.analysis, key):
                            setattr(self.analysis, key, value)

                logger.info("Loaded custom configuration from config.json")
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}")

    def save_config(self, filepath: str = "./config.json"):
        """Save current configuration to file"""
        config_dict = {
            'gcs': {
                'project_id': self.gcs.project_id,
                'bucket_name': self.gcs.bucket_name,
            },
            'pattern': {
                'min_duration_days': self.pattern.min_duration_days,
                'bbw_percentile_threshold': self.pattern.bbw_percentile_threshold,
                'volume_ratio_threshold': self.pattern.volume_ratio_threshold,
                'range_ratio_threshold': self.pattern.range_ratio_threshold,
            },
            'outcome': {
                'explosive_threshold': self.outcome.explosive_threshold,
                'strong_threshold': self.outcome.strong_threshold,
                'values': self.outcome.values,
            },
            'analysis': {
                'batch_size': self.analysis.batch_size,
                'output_dir': str(self.analysis.output_dir),
            }
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")


def get_config() -> SystemConfig:
    """Get system configuration instance"""
    return SystemConfig()