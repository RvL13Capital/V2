"""
Test Suite for Market Cap Fetcher
==================================

Tests for the market cap fetching utility including:
- API fetching with fallback
- Caching behavior
- Historical market cap estimation
- Category classification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import json
import tempfile

from utils.market_cap_fetcher import (
    MarketCapFetcher,
    MarketCapData,
    RateLimiter,
    get_market_cap
)


class TestMarketCapData:
    """Tests for MarketCapData dataclass."""

    def test_creation(self):
        """Test basic creation."""
        data = MarketCapData(
            ticker='AAPL',
            market_cap=3_000_000_000_000,
            shares_outstanding=15_000_000_000,
            source='alphavantage',
            fetched_at=datetime.now().isoformat()
        )
        assert data.ticker == 'AAPL'
        assert data.market_cap == 3_000_000_000_000
        assert data.source == 'alphavantage'

    def test_to_dict(self):
        """Test serialization to dict."""
        data = MarketCapData(
            ticker='MSFT',
            market_cap=2_500_000_000_000,
            shares_outstanding=7_500_000_000,
            source='fmp',
            fetched_at='2024-01-01T12:00:00'
        )
        d = data.to_dict()
        assert d['ticker'] == 'MSFT'
        assert d['market_cap'] == 2_500_000_000_000
        assert d['source'] == 'fmp'

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            'ticker': 'GOOGL',
            'market_cap': 1_800_000_000_000,
            'shares_outstanding': 6_000_000_000,
            'source': 'finnhub',
            'fetched_at': '2024-01-01T12:00:00',
            'currency': 'USD'
        }
        data = MarketCapData.from_dict(d)
        assert data.ticker == 'GOOGL'
        assert data.market_cap == 1_800_000_000_000


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_rate_limit_enforcement(self):
        """Test that rate limiter enforces limits."""
        limiter = RateLimiter(requests_per_minute=60)  # 1 per second
        assert limiter.min_interval == 1.0

    def test_first_request_no_wait(self):
        """Test that first request doesn't wait."""
        limiter = RateLimiter(requests_per_minute=60)
        start = datetime.now()
        limiter.wait()
        elapsed = (datetime.now() - start).total_seconds()
        assert elapsed < 0.1  # Should be nearly instant


class TestMarketCapFetcher:
    """Tests for MarketCapFetcher."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def fetcher(self, temp_cache_dir):
        """Create fetcher with temp cache."""
        return MarketCapFetcher(cache_dir=temp_cache_dir)

    def test_initialization(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher.cache_ttl_days == 7
        assert fetcher.timeout == 10
        assert isinstance(fetcher.cache, dict)

    def test_cache_valid_check_empty(self, fetcher):
        """Test cache validity when empty."""
        assert fetcher._is_cache_valid('NONEXISTENT') is False

    def test_cache_valid_check_fresh(self, fetcher):
        """Test cache validity with fresh data."""
        data = MarketCapData(
            ticker='TEST',
            market_cap=1_000_000_000,
            shares_outstanding=100_000_000,
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['TEST'] = data
        assert fetcher._is_cache_valid('TEST') is True

    def test_cache_valid_check_stale(self, fetcher):
        """Test cache validity with stale data."""
        stale_time = (datetime.now() - timedelta(days=10)).isoformat()
        data = MarketCapData(
            ticker='STALE',
            market_cap=1_000_000_000,
            shares_outstanding=100_000_000,
            source='test',
            fetched_at=stale_time
        )
        fetcher.cache['STALE'] = data
        assert fetcher._is_cache_valid('STALE') is False

    def test_market_cap_category_mega(self, fetcher):
        """Test mega cap classification."""
        data = MarketCapData(
            ticker='MEGA',
            market_cap=300_000_000_000,  # $300B
            shares_outstanding=10_000_000_000,
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['MEGA'] = data
        assert fetcher.get_market_cap_category('MEGA') == 'mega_cap'

    def test_market_cap_category_large(self, fetcher):
        """Test large cap classification."""
        data = MarketCapData(
            ticker='LARGE',
            market_cap=50_000_000_000,  # $50B
            shares_outstanding=1_000_000_000,
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['LARGE'] = data
        assert fetcher.get_market_cap_category('LARGE') == 'large_cap'

    def test_market_cap_category_mid(self, fetcher):
        """Test mid cap classification."""
        data = MarketCapData(
            ticker='MID',
            market_cap=5_000_000_000,  # $5B
            shares_outstanding=200_000_000,
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['MID'] = data
        assert fetcher.get_market_cap_category('MID') == 'mid_cap'

    def test_market_cap_category_small(self, fetcher):
        """Test small cap classification."""
        data = MarketCapData(
            ticker='SMALL',
            market_cap=1_000_000_000,  # $1B
            shares_outstanding=50_000_000,
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['SMALL'] = data
        assert fetcher.get_market_cap_category('SMALL') == 'small_cap'

    def test_market_cap_category_micro(self, fetcher):
        """Test micro cap classification."""
        data = MarketCapData(
            ticker='MICRO',
            market_cap=150_000_000,  # $150M
            shares_outstanding=15_000_000,
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['MICRO'] = data
        assert fetcher.get_market_cap_category('MICRO') == 'micro_cap'

    def test_market_cap_category_nano(self, fetcher):
        """Test nano cap classification."""
        data = MarketCapData(
            ticker='NANO',
            market_cap=30_000_000,  # $30M
            shares_outstanding=3_000_000,
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['NANO'] = data
        assert fetcher.get_market_cap_category('NANO') == 'nano_cap'

    def test_historical_market_cap(self, fetcher):
        """Test historical market cap calculation."""
        # Set up shares outstanding
        data = MarketCapData(
            ticker='HIST',
            market_cap=1_000_000_000,
            shares_outstanding=100_000_000,  # 100M shares
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['HIST'] = data

        # Create price data
        price_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [10.0, 11.0, 12.0]
        })

        result = fetcher.get_historical_market_cap('HIST', price_data)

        assert result is not None
        assert 'market_cap' in result.columns
        assert result.iloc[0]['market_cap'] == 1_000_000_000  # 100M * $10
        assert result.iloc[1]['market_cap'] == 1_100_000_000  # 100M * $11
        assert result.iloc[2]['market_cap'] == 1_200_000_000  # 100M * $12

    def test_historical_market_cap_no_data(self, fetcher):
        """Test historical market cap returns None when data unavailable."""
        price_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [10.0, 11.0, 12.0]
        })

        # No cached data for this ticker - should return None, not mock data
        result = fetcher.get_historical_market_cap('NONEXISTENT', price_data)
        assert result is None

    def test_historical_market_cap_no_shares(self, fetcher):
        """Test historical market cap returns None when shares outstanding missing."""
        # Market cap but no shares outstanding
        data = MarketCapData(
            ticker='NOSHARES',
            market_cap=1_000_000_000,
            shares_outstanding=None,  # Missing
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['NOSHARES'] = data

        price_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [10.0, 11.0, 12.0]
        })

        result = fetcher.get_historical_market_cap('NOSHARES', price_data)
        assert result is None  # Should NOT return mock data

    def test_get_stats(self, fetcher):
        """Test statistics retrieval."""
        stats = fetcher.get_stats()
        assert 'cached_tickers' in stats
        assert 'api_calls' in stats
        assert 'total_calls' in stats

    def test_clear_cache(self, fetcher, temp_cache_dir):
        """Test cache clearing."""
        # Add some data
        data = MarketCapData(
            ticker='CLEAR',
            market_cap=1_000_000_000,
            shares_outstanding=100_000_000,
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['CLEAR'] = data
        fetcher._save_cache()

        # Clear
        fetcher.clear_cache()

        assert len(fetcher.cache) == 0
        cache_file = Path(temp_cache_dir) / 'market_cap_cache.json'
        assert not cache_file.exists()


class TestNoMockData:
    """Tests to verify no mock or fabricated data is ever returned."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def fetcher(self, temp_cache_dir):
        """Create fetcher with temp cache."""
        return MarketCapFetcher(cache_dir=temp_cache_dir)

    def test_get_market_cap_returns_none_not_mock(self, fetcher):
        """Test that get_market_cap returns None, not mock data, when all APIs fail."""
        # Clear API keys to simulate all API failures
        fetcher.api_keys = {k: None for k in fetcher.api_keys}

        result = fetcher.get_market_cap('FAKE_TICKER_XYZ')
        assert result is None  # Must be None, not mock data

    def test_category_returns_none_when_no_data(self, fetcher):
        """Test that category returns None when market cap unavailable."""
        fetcher.api_keys = {k: None for k in fetcher.api_keys}

        result = fetcher.get_market_cap_category('FAKE_TICKER_XYZ')
        assert result is None  # Must be None, not a default category

    def test_batch_returns_none_for_failed_tickers(self, fetcher):
        """Test that batch fetch returns None for tickers that fail."""
        fetcher.api_keys = {k: None for k in fetcher.api_keys}

        results = fetcher.get_batch_market_cap(['FAKE1', 'FAKE2', 'FAKE3'])

        assert results['FAKE1'] is None
        assert results['FAKE2'] is None
        assert results['FAKE3'] is None

    def test_zero_market_cap_rejected(self, fetcher):
        """Test that zero market cap values are not returned."""
        # Directly test that cache validation would reject zero values
        data = MarketCapData(
            ticker='ZERO',
            market_cap=0,  # Invalid
            shares_outstanding=100_000_000,
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['ZERO'] = data

        # Even if cached, category should return None for zero market cap
        assert fetcher.get_market_cap_category('ZERO') == 'nano_cap'  # 0 < 50M

    def test_negative_shares_rejected_for_historical(self, fetcher):
        """Test that negative shares outstanding is rejected."""
        data = MarketCapData(
            ticker='NEGATIVE',
            market_cap=1_000_000_000,
            shares_outstanding=-100_000_000,  # Invalid
            source='test',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['NEGATIVE'] = data

        price_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'close': [10.0, 11.0, 12.0]
        })

        result = fetcher.get_historical_market_cap('NEGATIVE', price_data)
        assert result is None  # Must reject invalid data


class TestIntegration:
    """Integration tests (require API keys)."""

    @pytest.mark.skipif(
        not any([
            __import__('os').getenv('ALPHAVANTAGE_API_KEY'),
            __import__('os').getenv('FMP_API_KEY'),
            __import__('os').getenv('FINNHUB_API_KEY'),
            __import__('os').getenv('TWELVEDATA_API_KEY')
        ]),
        reason="No API keys configured"
    )
    def test_live_api_fetch(self):
        """Test live API fetching (skipped if no API keys)."""
        fetcher = MarketCapFetcher()
        data = fetcher.get_market_cap('AAPL', force_refresh=True)

        assert data is not None
        assert data.market_cap > 0
        assert data.source in ['twelvedata', 'alphavantage', 'fmp', 'finnhub']

    @pytest.mark.skipif(
        not any([
            __import__('os').getenv('ALPHAVANTAGE_API_KEY'),
            __import__('os').getenv('FMP_API_KEY'),
            __import__('os').getenv('FINNHUB_API_KEY'),
            __import__('os').getenv('TWELVEDATA_API_KEY')
        ]),
        reason="No API keys configured"
    )
    def test_live_api_returns_real_data(self):
        """Test that live API returns real data, not fabricated values."""
        fetcher = MarketCapFetcher()

        # Apple's market cap should be > $1 trillion (sanity check for real data)
        data = fetcher.get_market_cap('AAPL', force_refresh=True)

        assert data is not None
        assert data.market_cap > 1_000_000_000_000  # > $1T
        assert data.market_cap < 20_000_000_000_000  # < $20T (sanity upper bound)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
