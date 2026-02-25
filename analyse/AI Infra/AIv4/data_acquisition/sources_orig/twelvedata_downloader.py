"""
TwelveData Downloader - Alternative to Alpha Vantage

Downloads historical stock data from TwelveData API as an alternative
to Alpha Vantage when rate limits are hit or for better performance.

TwelveData Advantages:
- Higher rate limits on free tier (8 calls/min per key)
- Better international stock support
- More reliable data quality
"""

import logging
import requests
import pandas as pd
import time
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
from config.foreman_config import ForemanConfig
from data_acquisition.sources_orig.api_key_manager import APIKeyManager


logger = logging.getLogger(__name__)


class TwelveDataDownloader:
    """
    Downloads historical stock data from TwelveData API

    Features:
    - Full historical data (time_series endpoint)
    - Automatic API key rotation via APIKeyManager
    - Retry logic with exponential backoff
    - Data quality validation
    - Better international exchange support
    """

    def __init__(self, config: ForemanConfig, api_key_manager: APIKeyManager):
        """
        Initialize TwelveData Downloader

        Args:
            config: Foreman configuration
            api_key_manager: API key manager for rotation
        """
        self.config = config
        self.api_key_manager = api_key_manager
        self.base_url = 'https://api.twelvedata.com'

        # Statistics
        self.downloads_successful = 0
        self.downloads_failed = 0
        self.total_data_points = 0

        logger.info("TwelveDataDownloader initialized")

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
        # For TwelveData, we need to format ticker differently
        # Remove suffix and use exchange parameter instead
        base_ticker = ticker
        exchange = self._get_exchange_from_suffix(exchange_suffix)

        full_ticker = f"{base_ticker}{exchange_suffix}"
        logger.info(f"Downloading data for {full_ticker} from TwelveData")

        # Try download with retry
        max_retries = self.config.retry.max_retries
        backoff = self.config.retry.initial_backoff_seconds

        for attempt in range(max_retries):
            try:
                # Get API key from manager
                api_key = self.api_key_manager.get_available_key(
                    wait_if_needed=True,
                    max_wait_seconds=300
                )

                if not api_key:
                    logger.error(f"No API key available for {full_ticker}")
                    break

                # Download data
                df, metadata = self._download_from_twelvedata(
                    base_ticker,
                    exchange,
                    api_key
                )

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

                logger.warning(f"Download attempt {attempt+1}/{max_retries} failed for {full_ticker}")

                if attempt < max_retries - 1:
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

        logger.error(f"Failed to download data for {full_ticker} after {max_retries} attempts")
        self.downloads_failed += 1

        return None, {}

    def _get_exchange_from_suffix(self, suffix: str) -> Optional[str]:
        """
        Map exchange suffix to TwelveData exchange code

        Args:
            suffix: Exchange suffix (e.g., '.DE', '.L')

        Returns:
            TwelveData exchange code or None for US stocks
        """
        # TwelveData exchange mapping
        exchange_map = {
            '': None,           # US stocks (NYSE, NASDAQ)
            '.TO': 'TSX',       # Toronto
            '.L': 'LSE',        # London
            '.SW': 'SIX',       # Switzerland
            '.DE': 'XETRA',     # Germany
            '.PA': 'EURONEXT',  # Paris
            '.MI': 'MIL',       # Milan
            '.MC': 'BME',       # Madrid
            '.LS': 'ELI',       # Lisbon
            '.ST': 'STO',       # Stockholm
            '.IR': 'ISE',       # Ireland
        }

        return exchange_map.get(suffix)

    def _download_from_twelvedata(
        self,
        ticker: str,
        exchange: Optional[str],
        api_key: str
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Perform actual TwelveData API call

        Args:
            ticker: Base ticker symbol
            exchange: Exchange code (or None for US)
            api_key: TwelveData API key

        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        try:
            # TwelveData time_series endpoint
            # outputsize=5000 gives maximum history (up to 12 years on free tier)
            endpoint = f"{self.base_url}/time_series"

            params = {
                'symbol': ticker,
                'interval': '1day',
                'outputsize': 5000,  # Maximum available history
                'apikey': api_key,
                'format': 'JSON'
            }

            # Add exchange if specified
            if exchange:
                params['exchange'] = exchange

            logger.debug(f"Calling TwelveData API for {ticker} (exchange: {exchange})")

            response = requests.get(
                endpoint,
                params=params,
                timeout=self.config.retry.request_timeout_seconds
            )

            response.raise_for_status()

            data = response.json()

            # Check for errors
            if 'status' in data and data['status'] == 'error':
                logger.error(f"TwelveData error for {ticker}: {data.get('message', 'Unknown error')}")
                return None, {}

            # Check for rate limit
            if 'code' in data and data['code'] == 429:
                logger.warning(f"TwelveData rate limit hit for {ticker}")
                return None, {}

            # Extract time series data
            if 'values' not in data:
                logger.error(f"No time series data found for {ticker}")
                logger.debug(f"Response keys: {data.keys()}")
                return None, {}

            time_series = data['values']
            metadata = data.get('meta', {})

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

    def _parse_time_series(self, time_series: list) -> pd.DataFrame:
        """
        Parse TwelveData time series data into DataFrame

        Args:
            time_series: List of time series records from API

        Returns:
            DataFrame with OHLCV columns
        """
        # Convert to DataFrame
        df = pd.DataFrame(time_series)

        # Rename columns to standard format
        df = df.rename(columns={
            'datetime': 'date'
        })

        # Convert types
        df['date'] = pd.to_datetime(df['date'])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(int)

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
        logger.info("TwelveData Downloader Statistics")
        logger.info("=" * 60)
        logger.info(f"Successful downloads: {stats['downloads_successful']}")
        logger.info(f"Failed downloads: {stats['downloads_failed']}")
        logger.info(f"Success rate: {stats['success_rate']*100:.1f}%")
        logger.info(f"Total data points: {stats['total_data_points']}")

        if stats['downloads_successful'] > 0:
            avg_points = stats['total_data_points'] / stats['downloads_successful']
            logger.info(f"Average data points per ticker: {avg_points:.0f}")

        logger.info("=" * 60)
