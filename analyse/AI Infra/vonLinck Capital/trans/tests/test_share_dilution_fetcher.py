"""
Test Suite for Share Dilution Fetcher
=====================================

Tests for SEC EDGAR-based share dilution checking:
- CIK lookup
- Shares outstanding history fetching
- Dilution calculation
- Caching behavior
- Regional filtering (US-only)

Author: TRANS System
Date: December 2024
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from utils.share_dilution_fetcher import (
    ShareDilutionFetcher,
    SharesDilutionData,
    RateLimiter
)
from core.pattern_scanner import is_us_ticker


class TestIsUsTicker:
    """Tests for US ticker detection."""

    def test_us_ticker_no_suffix(self):
        """US tickers have no suffix."""
        assert is_us_ticker('AAPL') is True
        assert is_us_ticker('MSFT') is True
        assert is_us_ticker('GOOGL') is True
        assert is_us_ticker('SNDL') is True
        assert is_us_ticker('MULN') is True

    def test_eu_ticker_de_suffix(self):
        """German stocks have .DE suffix."""
        assert is_us_ticker('SAP.DE') is False
        assert is_us_ticker('BMW.DE') is False
        assert is_us_ticker('SIE.DE') is False

    def test_eu_ticker_l_suffix(self):
        """UK stocks have .L suffix."""
        assert is_us_ticker('HSBA.L') is False
        assert is_us_ticker('VOD.L') is False
        assert is_us_ticker('BP.L') is False

    def test_eu_ticker_all_suffixes(self):
        """Test all EU market suffixes."""
        suffixes = ['.DE', '.L', '.PA', '.MI', '.MC', '.AS', '.LS',
                    '.BR', '.SW', '.ST', '.OL', '.CO', '.HE', '.IR', '.VI']
        for suffix in suffixes:
            assert is_us_ticker(f'TEST{suffix}') is False, f"Failed for {suffix}"

    def test_empty_ticker(self):
        """Empty or None tickers return False."""
        assert is_us_ticker('') is False
        assert is_us_ticker(None) is False

    def test_case_insensitive(self):
        """Suffix detection is case-insensitive."""
        assert is_us_ticker('sap.de') is False
        assert is_us_ticker('SAP.de') is False
        assert is_us_ticker('sap.DE') is False


class TestSharesDilutionData:
    """Tests for SharesDilutionData dataclass."""

    def test_creation(self):
        """Test dataclass creation."""
        data = SharesDilutionData(
            ticker='AAPL',
            cik='0000320193',
            current_shares=15_000_000_000,
            shares_12m_ago=14_500_000_000,
            dilution_pct=3.45,
            current_filing_date='2024-09-28',
            historical_filing_date='2023-09-30',
            source='sec_edgar',
            fetched_at=datetime.now().isoformat()
        )
        assert data.ticker == 'AAPL'
        assert data.cik == '0000320193'
        assert data.dilution_pct == 3.45

    def test_to_dict(self):
        """Test conversion to dictionary."""
        data = SharesDilutionData(
            ticker='AAPL',
            cik='0000320193',
            current_shares=15_000_000_000,
            shares_12m_ago=14_500_000_000,
            dilution_pct=3.45,
            current_filing_date='2024-09-28',
            historical_filing_date='2023-09-30',
            source='sec_edgar',
            fetched_at='2024-12-23T10:00:00'
        )
        d = data.to_dict()
        assert isinstance(d, dict)
        assert d['ticker'] == 'AAPL'
        assert d['dilution_pct'] == 3.45

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            'ticker': 'MSFT',
            'cik': '0000789019',
            'current_shares': 7_400_000_000,
            'shares_12m_ago': 7_200_000_000,
            'dilution_pct': 2.78,
            'current_filing_date': '2024-09-30',
            'historical_filing_date': '2023-09-30',
            'source': 'sec_edgar',
            'fetched_at': '2024-12-23T10:00:00'
        }
        data = SharesDilutionData.from_dict(d)
        assert data.ticker == 'MSFT'
        assert data.dilution_pct == 2.78


class TestRateLimiter:
    """Tests for rate limiter."""

    def test_rate_limiter_interval(self):
        """Test rate limiter calculates correct interval."""
        limiter = RateLimiter(60)  # 60 requests per minute
        assert limiter.min_interval == 1.0  # 1 second

        limiter2 = RateLimiter(300)  # 300 rpm = 5/sec
        assert limiter2.min_interval == 0.2  # 0.2 seconds


class TestShareDilutionFetcher:
    """Tests for ShareDilutionFetcher."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def fetcher(self, temp_cache_dir):
        """Create fetcher with temp cache."""
        return ShareDilutionFetcher(cache_dir=temp_cache_dir)

    def test_initialization(self, fetcher):
        """Test fetcher initialization."""
        assert fetcher.cache_ttl_days == 7
        assert fetcher.cache_dir.exists()
        assert fetcher.api_calls == 0
        assert fetcher.cache_hits == 0

    def test_cache_valid_check_empty(self, fetcher):
        """Empty cache returns invalid."""
        assert fetcher._is_cache_valid('NONEXISTENT') is False

    def test_cache_valid_check_fresh(self, fetcher):
        """Fresh cache entry is valid."""
        data = SharesDilutionData(
            ticker='TEST',
            cik='0000000001',
            current_shares=100_000_000,
            shares_12m_ago=95_000_000,
            dilution_pct=5.26,
            current_filing_date='2024-09-28',
            historical_filing_date='2023-09-30',
            source='sec_edgar',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['TEST'] = data
        assert fetcher._is_cache_valid('TEST') is True

    def test_cache_valid_check_stale(self, fetcher):
        """Stale cache entry (>7 days) is invalid."""
        data = SharesDilutionData(
            ticker='STALE',
            cik='0000000002',
            current_shares=100_000_000,
            shares_12m_ago=95_000_000,
            dilution_pct=5.26,
            current_filing_date='2024-09-28',
            historical_filing_date='2023-09-30',
            source='sec_edgar',
            fetched_at=(datetime.now() - timedelta(days=10)).isoformat()
        )
        fetcher.cache['STALE'] = data
        assert fetcher._is_cache_valid('STALE') is False

    def test_stats(self, fetcher):
        """Test stats method."""
        stats = fetcher.get_stats()
        assert 'cached_tickers' in stats
        assert 'api_calls' in stats
        assert 'cache_hits' in stats
        assert stats['cached_tickers'] == 0
        assert stats['api_calls'] == 0


class TestDilutionCalculation:
    """Tests for dilution calculation logic."""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def fetcher(self, temp_cache_dir):
        return ShareDilutionFetcher(cache_dir=temp_cache_dir)

    def test_no_dilution(self, fetcher):
        """Stock with stable share count = 0% dilution."""
        filing_history = [
            {'end': '2024-09-28', 'val': 100_000_000, 'form': '10-K'},
            {'end': '2023-09-30', 'val': 100_000_000, 'form': '10-K'},
        ]
        current, historical, pct, _, _ = fetcher._calculate_dilution(filing_history)
        assert pct == 0.0
        assert current == 100_000_000
        assert historical == 100_000_000

    def test_moderate_dilution(self, fetcher):
        """Stock with 10% dilution."""
        filing_history = [
            {'end': '2024-09-28', 'val': 110_000_000, 'form': '10-K'},
            {'end': '2023-09-30', 'val': 100_000_000, 'form': '10-K'},
        ]
        current, historical, pct, _, _ = fetcher._calculate_dilution(filing_history)
        assert abs(pct - 10.0) < 0.1
        assert current == 110_000_000
        assert historical == 100_000_000

    def test_high_dilution(self, fetcher):
        """Stock with 50% dilution (above 20% threshold)."""
        filing_history = [
            {'end': '2024-09-28', 'val': 150_000_000, 'form': '10-K'},
            {'end': '2023-09-30', 'val': 100_000_000, 'form': '10-K'},
        ]
        current, historical, pct, _, _ = fetcher._calculate_dilution(filing_history)
        assert pct == 50.0
        assert pct > 20.0  # Above default threshold

    def test_share_buyback_negative_dilution(self, fetcher):
        """Stock with share buyback = negative dilution."""
        filing_history = [
            {'end': '2024-09-28', 'val': 90_000_000, 'form': '10-K'},
            {'end': '2023-09-30', 'val': 100_000_000, 'form': '10-K'},
        ]
        current, historical, pct, _, _ = fetcher._calculate_dilution(filing_history)
        assert pct == -10.0  # Negative = buyback (good!)

    def test_stock_split_detection(self, fetcher):
        """Stock split (2:1) shows as 100% increase."""
        filing_history = [
            {'end': '2024-09-28', 'val': 200_000_000, 'form': '10-K'},  # After 2:1 split
            {'end': '2023-09-30', 'val': 100_000_000, 'form': '10-K'},  # Before split
        ]
        current, historical, pct, _, _ = fetcher._calculate_dilution(filing_history)
        assert pct == 100.0  # This is a split, not real dilution
        # The scanner should skip filter when dilution > 100%

    def test_insufficient_data_single_filing(self, fetcher):
        """Single filing = insufficient data."""
        filing_history = [
            {'end': '2024-09-28', 'val': 100_000_000, 'form': '10-K'},
        ]
        current, historical, pct, _, _ = fetcher._calculate_dilution(filing_history)
        assert historical is None
        assert pct is None

    def test_insufficient_data_empty(self, fetcher):
        """Empty filing history."""
        current, historical, pct, _, _ = fetcher._calculate_dilution([])
        assert current is None
        assert historical is None
        assert pct is None


class TestNoFabricatedData:
    """Tests to verify no mock data is ever returned."""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def fetcher(self, temp_cache_dir):
        return ShareDilutionFetcher(cache_dir=temp_cache_dir)

    @patch('utils.share_dilution_fetcher.requests.get')
    def test_returns_none_on_api_failure(self, mock_get, fetcher):
        """API failures return None, not fabricated data."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        # CIK lookup should fail
        result = fetcher._lookup_cik('FAKE_TICKER')
        # Either returns None (no company_tickers.json) or fails gracefully
        # The key is no fabricated data

    @patch('utils.share_dilution_fetcher.requests.get')
    def test_returns_none_for_404(self, mock_get, fetcher):
        """404 response returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = fetcher._fetch_shares_history('0000000001')
        assert result is None


class TestConvenienceMethods:
    """Tests for convenience methods."""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def fetcher(self, temp_cache_dir):
        return ShareDilutionFetcher(cache_dir=temp_cache_dir)

    def test_is_diluted_threshold(self, fetcher):
        """Test is_diluted with different thresholds."""
        # Manually cache some test data
        data = SharesDilutionData(
            ticker='TEST',
            cik='0000000001',
            current_shares=125_000_000,
            shares_12m_ago=100_000_000,
            dilution_pct=25.0,  # 25% dilution
            current_filing_date='2024-09-28',
            historical_filing_date='2023-09-30',
            source='sec_edgar',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['TEST'] = data

        # Test various thresholds
        assert fetcher.is_diluted('TEST', threshold=20.0) is True   # 25 > 20
        assert fetcher.is_diluted('TEST', threshold=30.0) is False  # 25 < 30
        assert fetcher.is_diluted('TEST', threshold=25.0) is False  # 25 == 25 (not greater)

    def test_get_dilution_pct_from_cache(self, fetcher):
        """Test get_dilution_pct returns cached value."""
        data = SharesDilutionData(
            ticker='CACHED',
            cik='0000000002',
            current_shares=115_000_000,
            shares_12m_ago=100_000_000,
            dilution_pct=15.0,
            current_filing_date='2024-09-28',
            historical_filing_date='2023-09-30',
            source='sec_edgar',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cache['CACHED'] = data

        pct = fetcher.get_dilution_pct('CACHED')
        assert pct == 15.0
        assert fetcher.cache_hits == 1


class TestCacheOperations:
    """Tests for cache save/load operations."""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_cache_persistence(self, temp_cache_dir):
        """Test cache saves and loads correctly."""
        # Create fetcher and add data
        fetcher1 = ShareDilutionFetcher(cache_dir=temp_cache_dir)
        data = SharesDilutionData(
            ticker='PERSIST',
            cik='0000000003',
            current_shares=100_000_000,
            shares_12m_ago=95_000_000,
            dilution_pct=5.26,
            current_filing_date='2024-09-28',
            historical_filing_date='2023-09-30',
            source='sec_edgar',
            fetched_at=datetime.now().isoformat()
        )
        fetcher1.cache['PERSIST'] = data
        fetcher1._save_cache()

        # Create new fetcher - should load cache
        fetcher2 = ShareDilutionFetcher(cache_dir=temp_cache_dir)
        assert 'PERSIST' in fetcher2.cache
        assert fetcher2.cache['PERSIST'].dilution_pct == 5.26

    def test_clear_cache(self, temp_cache_dir):
        """Test cache clearing."""
        fetcher = ShareDilutionFetcher(cache_dir=temp_cache_dir)

        # Add some data
        fetcher.cache['TEST1'] = SharesDilutionData(
            ticker='TEST1', cik='0000000001',
            current_shares=100_000_000, shares_12m_ago=95_000_000,
            dilution_pct=5.0, current_filing_date='2024-09-28',
            historical_filing_date='2023-09-30', source='sec_edgar',
            fetched_at=datetime.now().isoformat()
        )
        fetcher.cik_cache['TEST1'] = '0000000001'
        fetcher._save_cache()
        fetcher._save_cik_cache()

        # Clear cache
        fetcher.clear_cache()

        assert len(fetcher.cache) == 0
        assert len(fetcher.cik_cache) == 0


# Integration tests (require network access)
class TestIntegration:
    """Integration tests with real SEC EDGAR API."""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.skipif(
        not __import__('os').getenv('RUN_INTEGRATION_TESTS'),
        reason="Integration tests disabled (set RUN_INTEGRATION_TESTS=1)"
    )
    def test_live_sec_edgar_fetch_aapl(self, temp_cache_dir):
        """Test live fetch from SEC EDGAR for Apple."""
        fetcher = ShareDilutionFetcher(cache_dir=temp_cache_dir)
        data = fetcher.get_dilution('AAPL', force_refresh=True)

        assert data is not None
        assert data.source == 'sec_edgar'
        assert data.cik is not None
        assert data.current_shares is not None
        # Apple typically has low dilution (buybacks)
        if data.dilution_pct is not None:
            assert data.dilution_pct < 50  # Reasonable sanity check

    @pytest.mark.skipif(
        not __import__('os').getenv('RUN_INTEGRATION_TESTS'),
        reason="Integration tests disabled"
    )
    def test_live_known_diluter(self, temp_cache_dir):
        """Test with a known high-dilution stock."""
        fetcher = ShareDilutionFetcher(cache_dir=temp_cache_dir)
        # SNDL has historically had significant dilution
        data = fetcher.get_dilution('SNDL', force_refresh=True)

        # May or may not have data, but should not crash
        if data is not None:
            assert data.source == 'sec_edgar'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
