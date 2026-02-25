"""
YFinance Downloader - Free, Unlimited Historical Data

Downloads historical stock data from Yahoo Finance using yfinance library.
No API key required, unlimited requests, excellent reliability.

Advantages:
- FREE and UNLIMITED (no API keys needed!)
- 20+ years of historical data
- Excellent international exchange support
- Very reliable and widely used
- No rate limits to manage
"""

import logging
import yfinance as yf
import pandas as pd
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class YFinanceDownloader:
    """
    Downloads historical stock data from Yahoo Finance

    Features:
    - Full historical data (max available)
    - No API keys required
    - No rate limits
    - Excellent data quality
    - International exchange support
    """

    def __init__(self, config):
        """
        Initialize YFinance Downloader

        Args:
            config: Foreman configuration
        """
        self.config = config

        # Statistics
        self.downloads_successful = 0
        self.downloads_failed = 0
        self.total_data_points = 0

        logger.info("YFinanceDownloader initialized (FREE, unlimited)")

    def download_ticker_data(
        self,
        ticker: str,
        exchange_suffix: str = '',
        period: str = 'max'
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Download historical data for a single ticker

        Args:
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix (e.g., '.DE', '.L')
            period: Data period ('max' for all available history)

        Returns:
            Tuple of (DataFrame with OHLCV data, metadata dict)
        """
        full_ticker = f"{ticker}{exchange_suffix}"

        logger.info(f"Downloading data for {full_ticker} from Yahoo Finance")

        try:
            # Create Ticker object
            ticker_obj = yf.Ticker(full_ticker)

            # Download historical data
            df = ticker_obj.history(
                period=period,
                interval='1d',
                auto_adjust=False,  # Keep raw prices
                back_adjust=False
            )

            if df is None or len(df) == 0:
                logger.warning(f"No data found for {full_ticker}")
                self.downloads_failed += 1
                return None, {}

            # Rename columns to standard format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Reset index to make date a column
            df = df.reset_index()
            df = df.rename(columns={'Date': 'date'})

            # Remove timezone info for consistency
            if pd.api.types.is_datetime64tz_dtype(df['date']):
                df['date'] = df['date'].dt.tz_localize(None)

            # Set date as index again
            df.set_index('date', inplace=True)

            # Validate data quality
            if self._validate_data(df, full_ticker):
                logger.info(
                    f"Successfully downloaded {len(df)} data points for {full_ticker} "
                    f"(from {df.index.min().date()} to {df.index.max().date()})"
                )

                self.downloads_successful += 1
                self.total_data_points += len(df)

                # Get metadata
                metadata = {
                    'ticker': full_ticker,
                    'source': 'Yahoo Finance',
                    'download_date': datetime.now().isoformat(),
                    'data_points': len(df),
                    'start_date': str(df.index.min().date()),
                    'end_date': str(df.index.max().date())
                }

                return df, metadata
            else:
                logger.warning(f"Data validation failed for {full_ticker}")
                self.downloads_failed += 1
                return None, {}

        except Exception as e:
            logger.error(f"Error downloading {full_ticker}: {e}")
            self.downloads_failed += 1
            return None, {}

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
        required = set(['open', 'high', 'low', 'close', 'volume'])
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
        logger.info("YFinance Downloader Statistics")
        logger.info("=" * 60)
        logger.info(f"Successful downloads: {stats['downloads_successful']}")
        logger.info(f"Failed downloads: {stats['downloads_failed']}")
        logger.info(f"Success rate: {stats['success_rate']*100:.1f}%")
        logger.info(f"Total data points: {stats['total_data_points']}")

        if stats['downloads_successful'] > 0:
            avg_points = stats['total_data_points'] / stats['downloads_successful']
            logger.info(f"Average data points per ticker: {avg_points:.0f}")

        logger.info("=" * 60)
