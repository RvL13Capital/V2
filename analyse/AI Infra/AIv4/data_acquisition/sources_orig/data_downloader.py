"""
Data Downloader - Alpha Vantage Historical Data Retrieval

Downloads maximum available historical OHLCV data using Alpha Vantage API
with intelligent 4-key rotation and retry logic.
"""

import logging
import requests
import pandas as pd
import time
from typing import Optional, Dict, Tuple
from datetime import datetime
from config.foreman_config import ForemanConfig
from data_acquisition.sources_orig.api_key_manager import APIKeyManager


logger = logging.getLogger(__name__)


class DataDownloader:
    """
    Downloads historical stock data from Alpha Vantage

    Features:
    - Full historical data (TIME_SERIES_DAILY_ADJUSTED)
    - Automatic API key rotation via APIKeyManager
    - Retry logic with exponential backoff
    - Data quality validation
    """

    def __init__(self, config: ForemanConfig, api_key_manager: APIKeyManager):
        """
        Initialize Data Downloader

        Args:
            config: Foreman configuration
            api_key_manager: API key manager for rotation
        """
        self.config = config
        self.api_key_manager = api_key_manager
        self.base_url = config.api.alpha_vantage_base_url

        # Statistics
        self.downloads_successful = 0
        self.downloads_failed = 0
        self.total_data_points = 0

        logger.info("DataDownloader initialized with Alpha Vantage API")

    def download_ticker_data(
        self,
        ticker: str,
        exchange_suffix: str = ''
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Download historical data for a single ticker

        Args:
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix (e.g., '.DE', '.L')

        Returns:
            Tuple of (DataFrame with OHLCV data, metadata dict)
        """
        full_ticker = f"{ticker}{exchange_suffix}"

        logger.info(f"Downloading data for {full_ticker}")

        # Try download with retry
        max_retries = self.config.retry.max_retries
        backoff = self.config.retry.initial_backoff_seconds

        for attempt in range(max_retries):
            try:
                # Get API key from manager
                api_key = self.api_key_manager.get_available_key(
                    wait_if_needed=True,
                    max_wait_seconds=300  # Wait up to 5 minutes
                )

                if not api_key:
                    logger.error(f"No API key available for {full_ticker} after waiting")
                    break

                # Download data
                df, metadata = self._download_from_alpha_vantage(full_ticker, api_key)

                if df is not None and len(df) > 0:
                    # Validate data quality
                    if self._validate_data(df, full_ticker):
                        logger.info(
                            f"Successfully downloaded {len(df)} data points for {full_ticker} "
                            f"(from {df.index.min()} to {df.index.max()})"
                        )
                        self.downloads_successful += 1
                        self.total_data_points += len(df)
                        return df, metadata
                    else:
                        logger.warning(f"Data validation failed for {full_ticker}")

                # If we got here, download failed
                logger.warning(f"Download attempt {attempt+1}/{max_retries} failed for {full_ticker}")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = min(
                        backoff * (self.config.retry.backoff_multiplier ** attempt),
                        self.config.retry.max_backoff_seconds
                    )
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error downloading {full_ticker} (attempt {attempt+1}): {e}")

                if attempt < max_retries - 1:
                    wait_time = min(
                        backoff * (self.config.retry.backoff_multiplier ** attempt),
                        self.config.retry.max_backoff_seconds
                    )
                    time.sleep(wait_time)

        # All retries failed
        logger.error(f"Failed to download data for {full_ticker} after {max_retries} attempts")
        self.downloads_failed += 1

        return None, {}

    def _download_from_alpha_vantage(
        self,
        ticker: str,
        api_key: str
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Perform actual Alpha Vantage API call

        Args:
            ticker: Full ticker symbol with exchange suffix
            api_key: Alpha Vantage API key

        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        try:
            # Alpha Vantage TIME_SERIES_DAILY_ADJUSTED
            # This provides maximum historical data (20+ years for many stocks)
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': ticker,
                'outputsize': 'full',  # Get maximum available history
                'apikey': api_key,
                'datatype': 'json'
            }

            logger.debug(f"Calling Alpha Vantage API for {ticker}")

            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.config.retry.request_timeout_seconds
            )

            response.raise_for_status()

            data = response.json()

            # Check for errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                return None, {}

            if 'Note' in data:
                # Rate limit message
                logger.warning(f"Alpha Vantage rate limit hit: {data['Note']}")
                return None, {}

            # Extract time series data
            if 'Time Series (Daily)' not in data:
                logger.error(f"No time series data found for {ticker}")
                logger.debug(f"Response keys: {data.keys()}")
                return None, {}

            time_series = data['Time Series (Daily)']
            metadata = data.get('Meta Data', {})

            # Convert to DataFrame
            df = self._parse_time_series(time_series)

            return df, metadata

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limit exceeded for {ticker}")
            else:
                logger.error(f"HTTP error for {ticker}: {e}")
            return None, {}

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {ticker}: {e}")
            return None, {}

        except Exception as e:
            logger.error(f"Unexpected error downloading {ticker}: {e}")
            return None, {}

    def _parse_time_series(self, time_series: Dict) -> pd.DataFrame:
        """
        Parse Alpha Vantage time series data into DataFrame

        Args:
            time_series: Time series dictionary from API

        Returns:
            DataFrame with OHLCV columns
        """
        # Convert to list of records
        records = []

        for date_str, values in time_series.items():
            record = {
                'date': pd.to_datetime(date_str),
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(float(values['6. volume']))  # Convert to int
            }
            records.append(record)

        # Create DataFrame
        df = pd.DataFrame(records)

        # Set date as index and sort
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        return df

    def _validate_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Validate downloaded data quality

        Args:
            df: Downloaded DataFrame
            ticker: Ticker symbol

        Returns:
            True if data passes validation
        """
        # Check minimum data points
        if len(df) < self.config.data_quality.min_data_points:
            logger.warning(
                f"{ticker}: Insufficient data points "
                f"({len(df)} < {self.config.data_quality.min_data_points})"
            )
            return False

        # Check required columns
        required = set(self.config.data_quality.required_columns)
        actual = set(df.columns)

        if not required.issubset(actual):
            missing = required - actual
            logger.error(f"{ticker}: Missing columns: {missing}")
            return False

        # Validate OHLC relationships
        if self.config.data_quality.validate_ohlc:
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()

            if invalid_ohlc > 0:
                logger.warning(f"{ticker}: {invalid_ohlc} rows with invalid OHLC relationships")
                # Don't fail validation, just warn

        # Validate positive prices
        if self.config.data_quality.validate_positive_prices:
            negative_prices = (
                (df['open'] <= 0) |
                (df['high'] <= 0) |
                (df['low'] <= 0) |
                (df['close'] <= 0)
            ).sum()

            if negative_prices > 0:
                logger.error(f"{ticker}: {negative_prices} rows with non-positive prices")
                return False

        # Validate positive volume
        if self.config.data_quality.validate_positive_volume:
            negative_volume = (df['volume'] < 0).sum()

            if negative_volume > 0:
                logger.warning(f"{ticker}: {negative_volume} rows with negative volume")
                # Don't fail validation for volume issues

        logger.debug(f"{ticker}: Data validation passed")
        return True

    def get_statistics(self) -> Dict:
        """Get download statistics"""
        return {
            'downloads_successful': self.downloads_successful,
            'downloads_failed': self.downloads_failed,
            'total_data_points': self.total_data_points,
            'success_rate': (
                self.downloads_successful / max(1, self.downloads_successful + self.downloads_failed)
            )
        }

    def print_statistics(self):
        """Print download statistics"""
        stats = self.get_statistics()

        logger.info("=" * 60)
        logger.info("Data Downloader Statistics")
        logger.info("=" * 60)
        logger.info(f"Successful downloads: {stats['downloads_successful']}")
        logger.info(f"Failed downloads: {stats['downloads_failed']}")
        logger.info(f"Success rate: {stats['success_rate']*100:.1f}%")
        logger.info(f"Total data points: {stats['total_data_points']}")

        if stats['downloads_successful'] > 0:
            avg_points = stats['total_data_points'] / stats['downloads_successful']
            logger.info(f"Average data points per ticker: {avg_points:.0f}")

        logger.info("=" * 60)
