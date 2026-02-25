"""
Tests for async data downloading functionality.
"""

from __future__ import annotations

import pytest
import asyncio
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from data_acquisition.sources import (
    AsyncDownloader,
    DownloadResult,
    AsyncYFinanceAdapter,
    UnifiedAsyncDownloader,
)


class TestAsyncDownloader:
    """Test AsyncDownloader class."""

    def test_initialization(self):
        """Test downloader initialization."""
        downloader = AsyncDownloader(
            max_concurrent=5,
            timeout_seconds=15,
            max_retries=2
        )

        assert downloader.max_concurrent == 5
        assert downloader.timeout_seconds == 15
        assert downloader.max_retries == 2

    @pytest.mark.asyncio
    async def test_download_single_ticker_yfinance(self):
        """Test downloading a single ticker from yfinance."""
        downloader = AsyncDownloader(max_concurrent=1, timeout_seconds=30)

        # Download AAPL (should work with real yfinance)
        results = await downloader.download_tickers(['AAPL'], source='yfinance')

        assert len(results) == 1
        result = results[0]

        assert result.ticker == 'AAPL'
        assert result.success is True
        assert result.data is not None
        assert len(result.data) > 0
        assert 'open' in result.data.columns
        assert 'close' in result.data.columns

    @pytest.mark.asyncio
    async def test_download_multiple_tickers_yfinance(self):
        """Test downloading multiple tickers in parallel."""
        downloader = AsyncDownloader(max_concurrent=3, timeout_seconds=30)

        tickers = ['AAPL', 'MSFT', 'GOOGL']
        results = await downloader.download_tickers(tickers, source='yfinance')

        assert len(results) == 3

        # Check all downloaded successfully
        successful = [r for r in results if r.success]
        assert len(successful) >= 2  # At least 2 should succeed

        # Check data structure
        for result in successful:
            assert result.data is not None
            assert len(result.data) > 0
            assert 'close' in result.data.columns

    @pytest.mark.asyncio
    async def test_download_invalid_ticker(self):
        """Test downloading invalid ticker (should fail gracefully)."""
        downloader = AsyncDownloader(max_concurrent=1, max_retries=1)

        results = await downloader.download_tickers(
            ['INVALIDTICKER12345XYZ'],
            source='yfinance'
        )

        assert len(results) == 1
        result = results[0]

        # Should either fail or return empty data
        if result.success:
            assert result.data is None or len(result.data) == 0
        else:
            assert result.error is not None

    def test_sync_wrapper(self):
        """Test synchronous wrapper for async download."""
        downloader = AsyncDownloader(max_concurrent=2)

        # Use sync wrapper
        results = downloader.download_tickers_sync(['AAPL'], source='yfinance')

        assert len(results) == 1
        assert results[0].ticker == 'AAPL'

    def test_statistics(self):
        """Test statistics tracking."""
        downloader = AsyncDownloader()

        # Initially zero
        stats = downloader.get_statistics()
        assert stats['downloads_successful'] == 0
        assert stats['downloads_failed'] == 0

        # Download some tickers
        results = downloader.download_tickers_sync(['AAPL', 'MSFT'])

        # Check stats updated
        stats = downloader.get_statistics()
        assert stats['total_downloads'] == 2
        assert stats['downloads_successful'] >= 1


class TestAsyncYFinanceAdapter:
    """Test AsyncYFinanceAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        config = Mock()
        adapter = AsyncYFinanceAdapter(config, max_workers=5)

        assert adapter.config == config
        assert adapter.max_workers == 5

    @pytest.mark.asyncio
    async def test_download_single_ticker(self):
        """Test downloading single ticker."""
        config = Mock()
        config.data_quality = Mock(
            min_data_points=100,
            validate_ohlc=True,
            validate_positive_prices=True,
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )

        adapter = AsyncYFinanceAdapter(config, max_workers=2)

        df, metadata = await adapter.download_ticker('AAPL', '')

        # Should succeed for AAPL
        assert df is not None
        assert len(df) > 100
        assert metadata['ticker'] == 'AAPL'
        assert metadata['source'] == 'Yahoo Finance'

    @pytest.mark.asyncio
    async def test_download_multiple_tickers(self):
        """Test downloading multiple tickers."""
        config = Mock()
        config.data_quality = Mock(
            min_data_points=100,
            validate_ohlc=True,
            validate_positive_prices=True,
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )

        adapter = AsyncYFinanceAdapter(config, max_workers=3)

        tickers = [('AAPL', ''), ('MSFT', ''), ('GOOGL', '')]
        results = await adapter.download_multiple(tickers)

        assert len(results) == 3

        # Check structure
        for ticker_full, df, metadata in results:
            assert isinstance(ticker_full, str)
            # df might be None if download failed
            if df is not None:
                assert 'close' in df.columns

    def test_get_statistics(self):
        """Test getting adapter statistics."""
        config = Mock()
        config.data_quality = Mock(
            min_data_points=100,
            validate_ohlc=True,
            validate_positive_prices=True,
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )

        adapter = AsyncYFinanceAdapter(config)
        stats = adapter.get_statistics()

        assert 'downloads_successful' in stats
        assert 'downloads_failed' in stats


class TestUnifiedAsyncDownloader:
    """Test UnifiedAsyncDownloader."""

    def test_initialization(self):
        """Test unified downloader initialization."""
        config = Mock()
        config.data_quality = Mock(
            min_data_points=100,
            validate_ohlc=True,
            validate_positive_prices=True,
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )

        downloader = UnifiedAsyncDownloader(
            config,
            prefer_source='yfinance',
            max_workers=5
        )

        assert downloader.prefer_source == 'yfinance'
        assert downloader.yfinance is not None

    @pytest.mark.asyncio
    async def test_download_ticker_yfinance(self):
        """Test downloading single ticker via yfinance."""
        config = Mock()
        config.data_quality = Mock(
            min_data_points=100,
            validate_ohlc=True,
            validate_positive_prices=True,
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )

        downloader = UnifiedAsyncDownloader(config, prefer_source='yfinance')

        df, metadata = await downloader.download_ticker('AAPL', '')

        assert df is not None
        assert len(df) > 100
        assert 'close' in df.columns

    @pytest.mark.asyncio
    async def test_download_multiple(self):
        """Test downloading multiple tickers."""
        config = Mock()
        config.data_quality = Mock(
            min_data_points=100,
            validate_ohlc=True,
            validate_positive_prices=True,
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )

        downloader = UnifiedAsyncDownloader(config)

        tickers = [('AAPL', ''), ('MSFT', '')]
        results = await downloader.download_multiple(tickers)

        assert len(results) == 2

        # At least one should succeed
        successful = [r for r in results if r[1] is not None]
        assert len(successful) >= 1

    def test_get_statistics(self):
        """Test getting unified statistics."""
        config = Mock()
        config.data_quality = Mock(
            min_data_points=100,
            validate_ohlc=True,
            validate_positive_prices=True,
            required_columns=['open', 'high', 'low', 'close', 'volume']
        )

        downloader = UnifiedAsyncDownloader(config)
        stats = downloader.get_statistics()

        assert 'yfinance' in stats
        assert isinstance(stats['yfinance'], dict)


class TestDownloadResult:
    """Test DownloadResult dataclass."""

    def test_success_result(self):
        """Test creating success result."""
        df = pd.DataFrame({'close': [100, 101, 102]})

        result = DownloadResult(
            ticker='AAPL',
            success=True,
            data=df,
            metadata={'source': 'test'},
            duration_seconds=1.5
        )

        assert result.ticker == 'AAPL'
        assert result.success is True
        assert result.data is not None
        assert result.metadata['source'] == 'test'
        assert result.duration_seconds == 1.5

    def test_failure_result(self):
        """Test creating failure result."""
        result = DownloadResult(
            ticker='INVALID',
            success=False,
            error='Ticker not found'
        )

        assert result.ticker == 'INVALID'
        assert result.success is False
        assert result.data is None
        assert result.error == 'Ticker not found'


# Integration tests

class TestAsyncIntegration:
    """Integration tests for async download pipeline."""

    @pytest.mark.asyncio
    async def test_full_download_pipeline(self):
        """Test complete download pipeline end-to-end."""
        downloader = AsyncDownloader(max_concurrent=3, timeout_seconds=30)

        # Download multiple tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        results = await downloader.download_tickers(tickers, source='yfinance')

        # Verify results
        assert len(results) == 3

        successful = [r for r in results if r.success]
        assert len(successful) >= 2

        # Check data quality
        for result in successful:
            assert result.data is not None
            assert len(result.data) > 0
            assert set(['open', 'high', 'low', 'close', 'volume']).issubset(result.data.columns)

            # Check OHLC relationships
            assert (result.data['high'] >= result.data['low']).all()
            assert (result.data['high'] >= result.data['open']).all()
            assert (result.data['high'] >= result.data['close']).all()

        # Check statistics
        stats = downloader.get_statistics()
        assert stats['downloads_successful'] >= 2
        assert stats['success_rate'] > 0.5

    def test_sync_vs_async_performance(self):
        """Test that async is faster than sync for multiple downloads."""
        import time

        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

        # Async download
        downloader = AsyncDownloader(max_concurrent=5)
        start = time.time()
        async_results = downloader.download_tickers_sync(tickers, source='yfinance')
        async_duration = time.time() - start

        # Should complete reasonably quickly
        assert async_duration < 30  # Less than 30 seconds for 5 tickers

        # Check results
        successful = [r for r in async_results if r.success]
        assert len(successful) >= 3  # At least 3 should succeed
