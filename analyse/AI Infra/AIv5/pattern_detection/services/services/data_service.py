"""
Data Service - Centralized Data Loading and Management

Provides a clean, high-level interface for loading ticker data with:
- Automatic configuration from SystemConfig
- Ticker list resolution (ALL vs specific tickers)
- Minimum data requirements validation
- Memory-optimized batch processing
- GCS integration with local caching
"""

import logging
from typing import List, Dict, Optional, Union
from datetime import datetime
import pandas as pd

from config import SystemConfig
from utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


class DataService:
    """
    High-level service for data loading operations.

    Consolidates duplicate data loading logic from main.py,
    train_complete_system.py, and scan_existing_data.py.

    Usage:
        config = SystemConfig()
        data_service = DataService(config)

        # Load all tickers
        data = data_service.load_tickers('ALL', min_years=2.0)

        # Load specific tickers
        data = data_service.load_tickers(['AAPL', 'MSFT'])
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize data service.

        Args:
            config: SystemConfig instance (creates default if None)
        """
        self.config = config or SystemConfig()

        # Initialize data loader with config settings
        cache_format = 'parquet' if self.config.memory.use_parquet_cache else 'csv'

        self.data_loader = DataLoader(
            project_id=self.config.data.gcs_project_id,
            bucket_name=self.config.data.gcs_bucket_name,
            use_memory_optimization=self.config.memory.enable_optimization,
            aggressive_optimization=self.config.memory.aggressive_mode,
            cache_format=cache_format,
            min_price=self.config.data.min_price
        )

        logger.info(f"DataService initialized (GCS: {self.config.data.gcs_bucket_name}, "
                   f"min_price: ${self.config.data.min_price:.2f})")

    def load_tickers(
        self,
        tickers: Union[str, List[str]],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        min_years: Optional[float] = None,
        min_days: Optional[int] = None,
        use_cache: bool = True,
        limit: Optional[int] = None,
        use_complete_history: Optional[bool] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load ticker data with automatic ticker resolution and validation.

        Args:
            tickers: 'ALL' to load all available tickers, or list of specific tickers
            start_date: Start date (YYYY-MM-DD or datetime) - ignored if use_complete_history=True
            end_date: End date (YYYY-MM-DD or datetime) - ignored if use_complete_history=True
            min_years: Minimum years of data required (overrides config and min_days)
            min_days: Minimum days of data required (overrides config)
            use_cache: Use local cache (default: True)
            limit: Limit number of tickers when using 'ALL' (for testing)
            use_complete_history: Override config setting (default: use config value)

        Returns:
            Dictionary mapping ticker -> DataFrame with OHLCV data
        """
        # Resolve ticker list
        ticker_list = self._resolve_tickers(tickers, limit)

        if not ticker_list:
            logger.error("No tickers to load")
            return {}

        # Determine minimum data requirements
        min_data_days = self._calculate_min_data_days(min_years, min_days)

        # Determine if we should use complete history
        if use_complete_history is None:
            use_complete_history = self.config.data.use_complete_history

        # Log configuration
        logger.info(f"Loading data for {len(ticker_list)} tickers")
        logger.info(f"Minimum requirement: {min_data_days} days ({min_data_days/252:.1f} years)")
        logger.info(f"Use complete history: {use_complete_history}")

        # Determine batch size for memory optimization
        batch_size = None
        if self.config.memory.batch_size_tickers > 0:
            batch_size = self.config.memory.batch_size_tickers
            logger.info(f"Using batch processing: {batch_size} tickers per batch")

        # Load data
        # If use_complete_history is True, ignore start_date/end_date
        effective_start = None if use_complete_history else start_date
        effective_end = None if use_complete_history else end_date

        ticker_data = self.data_loader.load_multiple_tickers(
            tickers=ticker_list,
            start_date=effective_start,
            end_date=effective_end,
            use_cache=use_cache,
            min_data_days=min_data_days,
            batch_size=batch_size,
            force_gc_between_batches=self.config.memory.force_gc_between_batches
        )

        if not ticker_data:
            logger.error("No data loaded. Check GCS credentials and data availability.")
            logger.error(f"Or relax min_data_points requirement (currently {min_data_days} days)")

        return ticker_data

    def load_single_ticker(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Load data for a single ticker.

        Args:
            ticker: Ticker symbol
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime)
            use_cache: Use local cache
            force_refresh: Force download from GCS

        Returns:
            DataFrame with OHLCV data
        """
        return self.data_loader.load_ticker_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
            force_refresh=force_refresh
        )

    def get_available_tickers(self, from_gcs: bool = True) -> List[str]:
        """
        Get list of available tickers.

        Args:
            from_gcs: If True, list from GCS; if False, list from local cache

        Returns:
            List of ticker symbols
        """
        return self.data_loader.get_available_tickers(from_gcs=from_gcs)

    def _resolve_tickers(self, tickers: Union[str, List[str]], limit: Optional[int] = None) -> List[str]:
        """
        Resolve ticker specification to list of ticker symbols.

        Args:
            tickers: 'ALL' or list of specific tickers
            limit: Limit number of tickers (for 'ALL' only)

        Returns:
            List of ticker symbols
        """
        if isinstance(tickers, str) and tickers.upper() == 'ALL':
            # Get all available tickers from GCS
            ticker_list = self.data_loader.get_available_tickers(from_gcs=True)

            if not ticker_list:
                logger.error("No tickers found in GCS or cache")
                logger.info(f"Check GCS bucket: gs://{self.config.data.gcs_bucket_name}/tickers/")
                return []

            # Apply limit if specified
            if limit and limit > 0:
                ticker_list = ticker_list[:limit]
                logger.info(f"Limited to first {limit} tickers")

            logger.info(f"Resolved 'ALL' to {len(ticker_list)} tickers")
            return ticker_list

        elif isinstance(tickers, str):
            # Single ticker as string or comma-separated
            if ',' in tickers:
                ticker_list = [t.strip() for t in tickers.split(',')]
            else:
                ticker_list = [tickers.strip()]

            logger.info(f"Resolved to {len(ticker_list)} specific tickers")
            return ticker_list

        elif isinstance(tickers, list):
            # Already a list
            logger.info(f"Using provided list of {len(tickers)} tickers")
            return tickers

        else:
            logger.error(f"Invalid ticker specification: {type(tickers)}")
            return []

    def _calculate_min_data_days(
        self,
        min_years: Optional[float] = None,
        min_days: Optional[int] = None
    ) -> int:
        """
        Calculate minimum data days from various specifications.

        Priority: min_years > min_days > config.data.min_data_points

        Args:
            min_years: Minimum years of data required
            min_days: Minimum days of data required

        Returns:
            Minimum number of trading days required
        """
        if min_years is not None:
            days = int(min_years * 252)  # Convert years to trading days
            logger.info(f"Using min_years: {min_years} years = {days} days")
            return days

        if min_days is not None:
            logger.info(f"Using min_days: {min_days} days")
            return min_days

        # Default to config
        days = self.config.data.min_data_points
        logger.debug(f"Using config min_data_points: {days} days ({days/252:.1f} years)")
        return days

    def get_cache_info(self, ticker: str) -> Optional[Dict]:
        """
        Get cache information for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with cache info or None
        """
        return self.data_loader.get_cache_info(ticker)

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """
        Clear cached data.

        Args:
            ticker: Specific ticker to clear, or None to clear all
        """
        self.data_loader.clear_cache(ticker)

        if ticker:
            logger.info(f"Cleared cache for {ticker}")
        else:
            logger.info("Cleared all cached data")

    def get_cache_size_mb(self) -> float:
        """
        Get total cache size in MB.

        Returns:
            Cache size in MB
        """
        return self.data_loader.get_cache_size_mb()
