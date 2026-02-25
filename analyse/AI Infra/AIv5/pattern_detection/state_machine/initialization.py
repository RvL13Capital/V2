"""
Common Initialization Utilities

Provides standardized setup functions used across multiple scripts:
- Logging configuration
- System configuration loading
- Service initialization
- Common argument parsing
"""

import sys
import io
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from config import SystemConfig
from services import DataService, PatternDetectionService


def setup_windows_encoding():
    """
    Fix Windows encoding issues for console output.

    This prevents UnicodeEncodeError when printing non-ASCII characters
    like checkmarks (✓) and crosses (✗) in Windows terminals.
    """
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace'
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding='utf-8',
            errors='replace'
        )


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    include_timestamp: bool = True
) -> None:
    """
    Setup standardized logging configuration.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path (auto-generated if None)
        include_timestamp: Include timestamp in log filename
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    # Add file handler if log_file specified or auto-generate
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    elif include_timestamp:
        # Auto-generate timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'aiv3_{timestamp}.log'
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {level} level")
    if log_file:
        logger.info(f"Log file: {log_file}")


def initialize_system(
    custom_config: Optional[Dict[str, Any]] = None,
    print_config: bool = True
) -> SystemConfig:
    """
    Initialize system configuration.

    Args:
        custom_config: Optional custom configuration overrides
        print_config: Print configuration summary

    Returns:
        SystemConfig instance
    """
    config = SystemConfig(custom_config=custom_config)

    if print_config:
        config.print_config()

    return config


def initialize_services(config: Optional[SystemConfig] = None) -> Dict[str, Any]:
    """
    Initialize all core services.

    Args:
        config: SystemConfig instance (creates default if None)

    Returns:
        Dictionary with initialized services:
        - config: SystemConfig instance
        - data_service: DataService instance
        - pattern_service: PatternDetectionService instance
        - labeling_service: LabelingService instance
        - feature_service: FeatureService instance
        - ml_service: MLService instance
        - validation_service: ValidationService instance
    """
    if config is None:
        config = SystemConfig()

    logger = logging.getLogger(__name__)
    logger.info("Initializing services...")

    # Initialize services
    from services import (
        DataService, PatternDetectionService,
        LabelingService, FeatureService,
        MLService, ValidationService
    )

    data_service = DataService(config)
    pattern_service = PatternDetectionService(config)
    labeling_service = LabelingService(config)
    feature_service = FeatureService(config)
    ml_service = MLService(config)
    validation_service = ValidationService(config)

    logger.info("Services initialized successfully (6 services)")

    return {
        'config': config,
        'data_service': data_service,
        'pattern_service': pattern_service,
        'labeling_service': labeling_service,
        'feature_service': feature_service,
        'ml_service': ml_service,
        'validation_service': validation_service
    }


def setup_system(
    log_level: str = 'INFO',
    custom_config: Optional[Dict[str, Any]] = None,
    print_config: bool = True
) -> Dict[str, Any]:
    """
    Complete system setup: encoding, logging, config, services.

    This is the one-stop initialization function for most scripts.

    Args:
        log_level: Logging level
        custom_config: Optional configuration overrides
        print_config: Print configuration summary

    Returns:
        Dictionary with:
        - config: SystemConfig instance
        - data_service: DataService instance
        - pattern_service: PatternDetectionService instance

    Example:
        from core.initialization import setup_system

        # Complete initialization
        services = setup_system(log_level='INFO')
        config = services['config']
        data_service = services['data_service']
        pattern_service = services['pattern_service']

        # Load data
        ticker_data = data_service.load_tickers('ALL', min_years=2.0)

        # Scan patterns
        results = pattern_service.scan_patterns(ticker_data)
    """
    # Fix Windows encoding
    setup_windows_encoding()

    # Setup logging
    setup_logging(level=log_level)

    # Initialize configuration
    config = initialize_system(custom_config=custom_config, print_config=print_config)

    # Initialize services
    services = initialize_services(config)

    return services


def get_output_directory(base_dir: str = 'output', subdir: Optional[str] = None) -> Path:
    """
    Get or create output directory.

    Args:
        base_dir: Base output directory
        subdir: Optional subdirectory name

    Returns:
        Path to output directory
    """
    if subdir:
        output_dir = Path(base_dir) / subdir
    else:
        output_dir = Path(base_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def generate_output_filename(
    prefix: str,
    extension: str = 'csv',
    timestamp: bool = True,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Generate timestamped output filename.

    Args:
        prefix: Filename prefix (e.g., 'patterns', 'results')
        extension: File extension (without dot)
        timestamp: Include timestamp in filename
        output_dir: Output directory (current dir if None)

    Returns:
        Path to output file

    Example:
        output_file = generate_output_filename('patterns', 'csv')
        # Returns: patterns_20240115_143022.csv
    """
    if timestamp:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{ts}.{extension}"
    else:
        filename = f"{prefix}.{extension}"

    if output_dir:
        return output_dir / filename
    else:
        return Path(filename)


def validate_data_requirements(
    ticker_data: Dict,
    min_tickers: int = 1,
    min_bars_per_ticker: int = 100
) -> bool:
    """
    Validate that loaded data meets minimum requirements.

    Args:
        ticker_data: Dictionary of ticker -> DataFrame
        min_tickers: Minimum number of tickers required
        min_bars_per_ticker: Minimum bars per ticker

    Returns:
        True if requirements met, False otherwise
    """
    logger = logging.getLogger(__name__)

    if not ticker_data:
        logger.error("No ticker data loaded")
        return False

    if len(ticker_data) < min_tickers:
        logger.error(f"Insufficient tickers: {len(ticker_data)} < {min_tickers} required")
        return False

    # Check each ticker has enough data
    insufficient_tickers = []
    for ticker, df in ticker_data.items():
        if len(df) < min_bars_per_ticker:
            insufficient_tickers.append((ticker, len(df)))

    if insufficient_tickers:
        logger.warning(f"{len(insufficient_tickers)} tickers have insufficient data:")
        for ticker, bars in insufficient_tickers[:5]:  # Show first 5
            logger.warning(f"  {ticker}: {bars} bars < {min_bars_per_ticker} required")

    logger.info(f"Data validation passed: {len(ticker_data)} tickers loaded")
    return True
