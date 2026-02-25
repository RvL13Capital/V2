"""
Async Data Downloader - Modern asynchronous data acquisition with aiohttp.

High-performance parallel ticker downloads with:
- Connection pooling via aiohttp ClientSession
- Concurrent downloads with asyncio.gather()
- Semaphore-based rate limiting
- Retry logic with exponential backoff
- Support for multiple data sources (yfinance, Alpha Vantage, etc.)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass

import aiohttp
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a single ticker download."""
    ticker: str
    success: bool
    data: Optional[pd.DataFrame] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


class AsyncDownloader:
    """
    Modern async data downloader with connection pooling.

    Features:
    - Parallel ticker downloads with asyncio.gather()
    - Connection pooling (reuses HTTP connections)
    - Rate limiting via semaphore
    - Automatic retry with exponential backoff
    - Statistics tracking

    Example:
        downloader = AsyncDownloader(max_concurrent=10)
        results = await downloader.download_tickers(['AAPL', 'MSFT', 'GOOGL'])

        # Sync fallback for compatibility
        results = downloader.download_tickers_sync(['AAPL'])
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        """
        Initialize async downloader.

        Args:
            max_concurrent: Maximum concurrent downloads
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_backoff: Initial backoff time for retries (exponential)
        """
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Statistics
        self._downloads_successful = 0
        self._downloads_failed = 0
        self._total_duration = 0.0

        # Semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(
            f"AsyncDownloader initialized "
            f"(max_concurrent={max_concurrent}, timeout={timeout_seconds}s)"
        )

    async def download_tickers(
        self,
        tickers: List[str],
        source: str = 'yfinance',
        **kwargs
    ) -> List[DownloadResult]:
        """
        Download multiple tickers in parallel.

        Args:
            tickers: List of ticker symbols
            source: Data source ('yfinance' or 'alphavantage')
            **kwargs: Additional parameters for the data source

        Returns:
            List of DownloadResult objects
        """
        logger.info(f"Starting parallel download of {len(tickers)} tickers from {source}")

        # Create aiohttp session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,  # Max connections
            ttl_dns_cache=300,  # DNS cache TTL
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create tasks for all tickers
            tasks = [
                self._download_single_ticker(session, ticker, source, **kwargs)
                for ticker in tickers
            ]

            # Execute all tasks in parallel
            start_time = datetime.now()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = (datetime.now() - start_time).total_seconds()

            # Process results (handle exceptions)
            processed_results = []
            for ticker, result in zip(tickers, results):
                if isinstance(result, Exception):
                    logger.error(f"Error downloading {ticker}: {result}")
                    processed_results.append(DownloadResult(
                        ticker=ticker,
                        success=False,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)

            # Update statistics
            successful = sum(1 for r in processed_results if r.success)
            failed = len(processed_results) - successful

            logger.info(
                f"Completed parallel download: {successful} successful, "
                f"{failed} failed in {duration:.2f}s"
            )

            return processed_results

    async def _download_single_ticker(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        source: str,
        **kwargs
    ) -> DownloadResult:
        """
        Download data for a single ticker with rate limiting and retry.

        Args:
            session: aiohttp ClientSession
            ticker: Ticker symbol
            source: Data source
            **kwargs: Additional parameters

        Returns:
            DownloadResult
        """
        # Acquire semaphore to limit concurrent requests
        async with self._semaphore:
            start_time = datetime.now()

            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    # Delegate to source-specific method
                    if source == 'yfinance':
                        df, metadata = await self._download_yfinance(session, ticker, **kwargs)
                    elif source == 'alphavantage':
                        df, metadata = await self._download_alphavantage(session, ticker, **kwargs)
                    else:
                        raise ValueError(f"Unknown source: {source}")

                    # Success
                    if df is not None and len(df) > 0:
                        duration = (datetime.now() - start_time).total_seconds()

                        self._downloads_successful += 1
                        self._total_duration += duration

                        logger.debug(
                            f"{ticker}: Downloaded {len(df)} data points "
                            f"in {duration:.2f}s"
                        )

                        return DownloadResult(
                            ticker=ticker,
                            success=True,
                            data=df,
                            metadata=metadata,
                            duration_seconds=duration
                        )

                    # No data returned
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_backoff * (2 ** attempt)
                        logger.warning(
                            f"{ticker}: No data (attempt {attempt+1}/{self.max_retries}), "
                            f"retrying in {wait_time:.1f}s"
                        )
                        await asyncio.sleep(wait_time)

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_backoff * (2 ** attempt)
                        logger.warning(
                            f"{ticker}: Error (attempt {attempt+1}/{self.max_retries}): {e}, "
                            f"retrying in {wait_time:.1f}s"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"{ticker}: Failed after {self.max_retries} attempts: {e}")
                        self._downloads_failed += 1

                        return DownloadResult(
                            ticker=ticker,
                            success=False,
                            error=str(e),
                            duration_seconds=(datetime.now() - start_time).total_seconds()
                        )

            # All retries failed
            self._downloads_failed += 1
            return DownloadResult(
                ticker=ticker,
                success=False,
                error="Max retries exceeded",
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )

    async def _download_yfinance(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        **kwargs
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Download from Yahoo Finance (yfinance).

        Note: yfinance library is synchronous, so we run it in executor.
        For true async, we would need to implement direct Yahoo Finance API calls.

        Args:
            session: aiohttp session (not used, kept for interface consistency)
            ticker: Ticker symbol
            **kwargs: Additional parameters

        Returns:
            Tuple of (DataFrame, metadata)
        """
        import yfinance as yf

        # Run yfinance in thread executor (it's synchronous)
        loop = asyncio.get_event_loop()
        df, metadata = await loop.run_in_executor(
            None,
            self._sync_yfinance_download,
            ticker,
            kwargs
        )

        return df, metadata

    def _sync_yfinance_download(
        self,
        ticker: str,
        kwargs: Dict
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Synchronous yfinance download (runs in executor).

        Args:
            ticker: Ticker symbol
            kwargs: Additional parameters

        Returns:
            Tuple of (DataFrame, metadata)
        """
        import yfinance as yf

        try:
            ticker_obj = yf.Ticker(ticker)

            period = kwargs.get('period', 'max')
            interval = kwargs.get('interval', '1d')

            df = ticker_obj.history(
                period=period,
                interval=interval,
                auto_adjust=False,
                back_adjust=False
            )

            if df is None or len(df) == 0:
                return None, {}

            # Standardize column names
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Reset index
            df = df.reset_index()
            df = df.rename(columns={'Date': 'date'})

            # Remove timezone
            if pd.api.types.is_datetime64tz_dtype(df['date']):
                df['date'] = df['date'].dt.tz_localize(None)

            df.set_index('date', inplace=True)

            metadata = {
                'ticker': ticker,
                'source': 'Yahoo Finance',
                'download_date': datetime.now().isoformat(),
                'data_points': len(df),
                'start_date': str(df.index.min().date()) if len(df) > 0 else None,
                'end_date': str(df.index.max().date()) if len(df) > 0 else None,
            }

            return df, metadata

        except Exception as e:
            logger.error(f"yfinance error for {ticker}: {e}")
            return None, {}

    async def _download_alphavantage(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        **kwargs
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Download from Alpha Vantage API using aiohttp.

        Args:
            session: aiohttp session
            ticker: Ticker symbol
            **kwargs: Must include 'api_key'

        Returns:
            Tuple of (DataFrame, metadata)
        """
        api_key = kwargs.get('api_key')
        if not api_key:
            raise ValueError("api_key required for Alpha Vantage")

        base_url = kwargs.get('base_url', 'https://www.alphavantage.co/query')

        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'outputsize': 'full',
            'apikey': api_key,
            'datatype': 'json'
        }

        async with session.get(base_url, params=params) as response:
            response.raise_for_status()
            data = await response.json()

            # Check for errors
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage error: {data['Error Message']}")

            if 'Note' in data:
                raise ValueError(f"Rate limit: {data['Note']}")

            if 'Time Series (Daily)' not in data:
                raise ValueError("No time series data in response")

            time_series = data['Time Series (Daily)']

            # Parse to DataFrame
            records = []
            for date_str, values in time_series.items():
                record = {
                    'date': pd.to_datetime(date_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(float(values['6. volume']))
                }
                records.append(record)

            df = pd.DataFrame(records)
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            metadata = {
                'ticker': ticker,
                'source': 'Alpha Vantage',
                'download_date': datetime.now().isoformat(),
                'data_points': len(df),
                'start_date': str(df.index.min().date()) if len(df) > 0 else None,
                'end_date': str(df.index.max().date()) if len(df) > 0 else None,
            }

            return df, metadata

    def download_tickers_sync(
        self,
        tickers: List[str],
        source: str = 'yfinance',
        **kwargs
    ) -> List[DownloadResult]:
        """
        Synchronous wrapper for download_tickers (compatibility).

        Args:
            tickers: List of ticker symbols
            source: Data source
            **kwargs: Additional parameters

        Returns:
            List of DownloadResult objects
        """
        # Run async method in new event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.download_tickers(tickers, source, **kwargs)
        )

    def get_statistics(self) -> Dict:
        """Get download statistics."""
        total = self._downloads_successful + self._downloads_failed

        return {
            'downloads_successful': self._downloads_successful,
            'downloads_failed': self._downloads_failed,
            'total_downloads': total,
            'success_rate': self._downloads_successful / max(1, total),
            'total_duration_seconds': self._total_duration,
            'avg_duration_seconds': self._total_duration / max(1, self._downloads_successful),
        }

    def print_statistics(self):
        """Print download statistics."""
        stats = self.get_statistics()

        logger.info("=" * 60)
        logger.info("AsyncDownloader Statistics")
        logger.info("=" * 60)
        logger.info(f"Successful downloads: {stats['downloads_successful']}")
        logger.info(f"Failed downloads: {stats['downloads_failed']}")
        logger.info(f"Success rate: {stats['success_rate']*100:.1f}%")
        logger.info(f"Total duration: {stats['total_duration_seconds']:.2f}s")
        logger.info(f"Avg duration per ticker: {stats['avg_duration_seconds']:.2f}s")
        logger.info("=" * 60)
