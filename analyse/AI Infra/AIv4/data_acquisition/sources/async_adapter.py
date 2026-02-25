"""
Async Adapter - Wraps synchronous downloaders for async compatibility.

Provides async interface for existing synchronous data sources:
- YFinanceDownloader (Yahoo Finance)
- TwelveDataDownloader (Twelve Data API)
- AlphaVantage (via DataDownloader)

Uses asyncio.run_in_executor() to run sync code in thread pool,
enabling parallel downloads without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class AsyncYFinanceAdapter:
    """
    Async adapter for YFinanceDownloader.

    Wraps synchronous yfinance calls to run in executor for async compatibility.
    """

    def __init__(self, config, max_workers: int = 10):
        """
        Initialize async adapter.

        Args:
            config: Foreman configuration
            max_workers: Max thread pool workers
        """
        self.config = config
        self.max_workers = max_workers

        # Lazy import to avoid circular dependencies
        from data_acquisition.sources_orig.yfinance_downloader import YFinanceDownloader
        self.downloader = YFinanceDownloader(config)

        # Thread pool for running sync code
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        logger.info(f"AsyncYFinanceAdapter initialized (max_workers={max_workers})")

    async def download_ticker(
        self,
        ticker: str,
        exchange_suffix: str = '',
        period: str = 'max'
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Download ticker data asynchronously.

        Args:
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix
            period: Data period ('max' for all available)

        Returns:
            Tuple of (DataFrame, metadata)
        """
        loop = asyncio.get_event_loop()

        # Run sync download in executor
        df, metadata = await loop.run_in_executor(
            self._executor,
            self.downloader.download_ticker_data,
            ticker,
            exchange_suffix,
            period
        )

        return df, metadata

    async def download_multiple(
        self,
        tickers: List[Tuple[str, str]],
        period: str = 'max'
    ) -> List[Tuple[str, Optional[pd.DataFrame], Dict]]:
        """
        Download multiple tickers in parallel.

        Args:
            tickers: List of (ticker, exchange_suffix) tuples
            period: Data period

        Returns:
            List of (ticker, DataFrame, metadata) tuples
        """
        logger.info(f"Downloading {len(tickers)} tickers in parallel")

        # Create tasks for all tickers
        tasks = [
            self.download_ticker(ticker, exchange_suffix, period)
            for ticker, exchange_suffix in tickers
        ]

        # Execute in parallel
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = (datetime.now() - start_time).total_seconds()

        # Process results
        processed = []
        for (ticker, exchange_suffix), result in zip(tickers, results):
            full_ticker = f"{ticker}{exchange_suffix}"

            if isinstance(result, Exception):
                logger.error(f"{full_ticker}: Error - {result}")
                processed.append((full_ticker, None, {}))
            else:
                df, metadata = result
                processed.append((full_ticker, df, metadata))

        successful = sum(1 for _, df, _ in processed if df is not None)
        logger.info(
            f"Downloaded {successful}/{len(tickers)} tickers successfully "
            f"in {duration:.2f}s"
        )

        return processed

    def get_statistics(self) -> Dict:
        """Get downloader statistics."""
        return self.downloader.get_statistics()

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


class AsyncAlphaVantageAdapter:
    """
    Async adapter for Alpha Vantage DataDownloader.

    Wraps synchronous Alpha Vantage API calls to run in executor.
    """

    def __init__(self, config, api_key_manager, max_workers: int = 4):
        """
        Initialize async adapter.

        Args:
            config: Foreman configuration
            api_key_manager: API key manager for rotation
            max_workers: Max thread pool workers (lower for API rate limits)
        """
        self.config = config
        self.api_key_manager = api_key_manager
        self.max_workers = max_workers

        # Lazy import
        from data_acquisition.sources_orig.data_downloader import DataDownloader
        self.downloader = DataDownloader(config, api_key_manager)

        # Thread pool (smaller for API rate limits)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        logger.info(f"AsyncAlphaVantageAdapter initialized (max_workers={max_workers})")

    async def download_ticker(
        self,
        ticker: str,
        exchange_suffix: str = ''
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Download ticker data asynchronously.

        Args:
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix

        Returns:
            Tuple of (DataFrame, metadata)
        """
        loop = asyncio.get_event_loop()

        # Run sync download in executor
        df, metadata = await loop.run_in_executor(
            self._executor,
            self.downloader.download_ticker_data,
            ticker,
            exchange_suffix
        )

        return df, metadata

    async def download_multiple(
        self,
        tickers: List[Tuple[str, str]]
    ) -> List[Tuple[str, Optional[pd.DataFrame], Dict]]:
        """
        Download multiple tickers in parallel (respecting rate limits).

        Args:
            tickers: List of (ticker, exchange_suffix) tuples

        Returns:
            List of (ticker, DataFrame, metadata) tuples
        """
        logger.info(f"Downloading {len(tickers)} tickers via Alpha Vantage")

        # Create tasks
        tasks = [
            self.download_ticker(ticker, exchange_suffix)
            for ticker, exchange_suffix in tickers
        ]

        # Execute with rate limiting handled by APIKeyManager
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = (datetime.now() - start_time).total_seconds()

        # Process results
        processed = []
        for (ticker, exchange_suffix), result in zip(tickers, results):
            full_ticker = f"{ticker}{exchange_suffix}"

            if isinstance(result, Exception):
                logger.error(f"{full_ticker}: Error - {result}")
                processed.append((full_ticker, None, {}))
            else:
                df, metadata = result
                processed.append((full_ticker, df, metadata))

        successful = sum(1 for _, df, _ in processed if df is not None)
        logger.info(
            f"Downloaded {successful}/{len(tickers)} tickers successfully "
            f"in {duration:.2f}s"
        )

        return processed

    def get_statistics(self) -> Dict:
        """Get downloader statistics."""
        return self.downloader.get_statistics()

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


class AsyncTwelveDataAdapter:
    """
    Async adapter for TwelveDataDownloader.

    Wraps synchronous Twelve Data API calls to run in executor.
    """

    def __init__(self, config, max_workers: int = 8):
        """
        Initialize async adapter.

        Args:
            config: Foreman configuration
            max_workers: Max thread pool workers
        """
        self.config = config
        self.max_workers = max_workers

        # Lazy import
        try:
            from data_acquisition.sources_orig.twelvedata_downloader import TwelveDataDownloader
            self.downloader = TwelveDataDownloader(config)
        except ImportError:
            logger.warning("TwelveDataDownloader not available")
            self.downloader = None

        # Thread pool
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        logger.info(f"AsyncTwelveDataAdapter initialized (max_workers={max_workers})")

    async def download_ticker(
        self,
        ticker: str,
        exchange_suffix: str = ''
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Download ticker data asynchronously.

        Args:
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix

        Returns:
            Tuple of (DataFrame, metadata)
        """
        if not self.downloader:
            raise RuntimeError("TwelveDataDownloader not available")

        loop = asyncio.get_event_loop()

        # Run sync download in executor
        df, metadata = await loop.run_in_executor(
            self._executor,
            self.downloader.download_ticker_data,
            ticker,
            exchange_suffix
        )

        return df, metadata

    async def download_multiple(
        self,
        tickers: List[Tuple[str, str]]
    ) -> List[Tuple[str, Optional[pd.DataFrame], Dict]]:
        """
        Download multiple tickers in parallel.

        Args:
            tickers: List of (ticker, exchange_suffix) tuples

        Returns:
            List of (ticker, DataFrame, metadata) tuples
        """
        if not self.downloader:
            raise RuntimeError("TwelveDataDownloader not available")

        logger.info(f"Downloading {len(tickers)} tickers via Twelve Data")

        # Create tasks
        tasks = [
            self.download_ticker(ticker, exchange_suffix)
            for ticker, exchange_suffix in tickers
        ]

        # Execute in parallel
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = (datetime.now() - start_time).total_seconds()

        # Process results
        processed = []
        for (ticker, exchange_suffix), result in zip(tickers, results):
            full_ticker = f"{ticker}{exchange_suffix}"

            if isinstance(result, Exception):
                logger.error(f"{full_ticker}: Error - {result}")
                processed.append((full_ticker, None, {}))
            else:
                df, metadata = result
                processed.append((full_ticker, df, metadata))

        successful = sum(1 for _, df, _ in processed if df is not None)
        logger.info(
            f"Downloaded {successful}/{len(tickers)} tickers successfully "
            f"in {duration:.2f}s"
        )

        return processed

    def get_statistics(self) -> Dict:
        """Get downloader statistics."""
        if self.downloader:
            return self.downloader.get_statistics()
        return {}

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


class UnifiedAsyncDownloader:
    """
    Unified async downloader that automatically selects best available source.

    Provides single interface for all data sources with automatic fallback:
    1. YFinance (free, unlimited, preferred)
    2. Twelve Data (API key required)
    3. Alpha Vantage (API key required, rate limited)
    """

    def __init__(
        self,
        config,
        api_key_manager=None,
        prefer_source: str = 'yfinance',
        max_workers: int = 10
    ):
        """
        Initialize unified downloader.

        Args:
            config: Foreman configuration
            api_key_manager: API key manager (for Alpha Vantage)
            prefer_source: Preferred source ('yfinance', 'twelvedata', 'alphavantage')
            max_workers: Max parallel workers
        """
        self.config = config
        self.prefer_source = prefer_source

        # Initialize adapters
        self.yfinance = AsyncYFinanceAdapter(config, max_workers=max_workers)

        if api_key_manager:
            self.alphavantage = AsyncAlphaVantageAdapter(
                config, api_key_manager, max_workers=max(4, max_workers // 2)
            )
        else:
            self.alphavantage = None

        try:
            self.twelvedata = AsyncTwelveDataAdapter(config, max_workers=max_workers)
        except:
            self.twelvedata = None

        logger.info(
            f"UnifiedAsyncDownloader initialized "
            f"(prefer_source={prefer_source}, max_workers={max_workers})"
        )

    async def download_ticker(
        self,
        ticker: str,
        exchange_suffix: str = '',
        source: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Download ticker data from preferred or specified source.

        Args:
            ticker: Ticker symbol
            exchange_suffix: Exchange suffix
            source: Override preferred source (optional)

        Returns:
            Tuple of (DataFrame, metadata)
        """
        source = source or self.prefer_source

        if source == 'yfinance':
            return await self.yfinance.download_ticker(ticker, exchange_suffix)
        elif source == 'alphavantage' and self.alphavantage:
            return await self.alphavantage.download_ticker(ticker, exchange_suffix)
        elif source == 'twelvedata' and self.twelvedata:
            return await self.twelvedata.download_ticker(ticker, exchange_suffix)
        else:
            # Fallback to yfinance
            logger.warning(f"Source {source} not available, falling back to yfinance")
            return await self.yfinance.download_ticker(ticker, exchange_suffix)

    async def download_multiple(
        self,
        tickers: List[Tuple[str, str]],
        source: Optional[str] = None
    ) -> List[Tuple[str, Optional[pd.DataFrame], Dict]]:
        """
        Download multiple tickers in parallel.

        Args:
            tickers: List of (ticker, exchange_suffix) tuples
            source: Override preferred source (optional)

        Returns:
            List of (ticker, DataFrame, metadata) tuples
        """
        source = source or self.prefer_source

        if source == 'yfinance':
            return await self.yfinance.download_multiple(tickers)
        elif source == 'alphavantage' and self.alphavantage:
            return await self.alphavantage.download_multiple(tickers)
        elif source == 'twelvedata' and self.twelvedata:
            return await self.twelvedata.download_multiple(tickers)
        else:
            # Fallback to yfinance
            logger.warning(f"Source {source} not available, falling back to yfinance")
            return await self.yfinance.download_multiple(tickers)

    def get_statistics(self, source: Optional[str] = None) -> Dict:
        """Get statistics for specified or all sources."""
        if source:
            if source == 'yfinance':
                return self.yfinance.get_statistics()
            elif source == 'alphavantage' and self.alphavantage:
                return self.alphavantage.get_statistics()
            elif source == 'twelvedata' and self.twelvedata:
                return self.twelvedata.get_statistics()

        # Return all statistics
        stats = {
            'yfinance': self.yfinance.get_statistics(),
        }

        if self.alphavantage:
            stats['alphavantage'] = self.alphavantage.get_statistics()

        if self.twelvedata:
            stats['twelvedata'] = self.twelvedata.get_statistics()

        return stats
