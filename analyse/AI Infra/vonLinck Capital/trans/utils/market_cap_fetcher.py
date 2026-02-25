"""
Market Cap Fetcher for TRANS System
====================================

Fetches market cap data from multiple API sources with intelligent fallback
and caching to minimize API calls.

Supported APIs (in order of preference):
1. TwelveData - statistics endpoint (shares outstanding)
2. Alpha Vantage - OVERVIEW endpoint (market cap)
3. FMP - company profile (market cap)
4. Finnhub - company profile (shares outstanding)

Features:
- Multi-source fallback
- Disk caching with configurable TTL
- Rate limiting per API
- Historical market cap estimation (shares × price)

Usage:
    from utils.market_cap_fetcher import MarketCapFetcher

    fetcher = MarketCapFetcher()

    # Get current market cap
    market_cap = fetcher.get_market_cap('AAPL')

    # Get historical market cap (estimate)
    historical = fetcher.get_historical_market_cap('AAPL', price_data)

Author: TRANS System
Date: December 2024
"""

import os
import json
import time
import logging
import requests
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Try to import aiohttp for async fetching (optional)
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MarketCapData:
    """Container for market cap information."""
    ticker: str
    market_cap: Optional[float]  # In dollars
    shares_outstanding: Optional[float]  # Number of shares
    source: str  # Which API provided the data
    fetched_at: str  # ISO timestamp
    currency: str = 'USD'
    security_type: str = 'EQUITY'  # EQUITY, MUTUALFUND, ETF, etc.
    name: Optional[str] = None  # Company name

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketCapData':
        # Handle legacy cache entries without security_type
        if 'security_type' not in data:
            data['security_type'] = 'EQUITY'
        if 'name' not in data:
            data['name'] = None
        return cls(**data)

    def is_equity(self) -> bool:
        """Check if this is an equity (not a fund/ETF)."""
        return self.security_type in ['EQUITY', 'Common Stock', '']


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0.0

    def wait(self):
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_request = time.time()


class MarketCapFetcher:
    """
    Multi-source market cap fetcher with caching and rate limiting.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_ttl_days: int = 7,
        timeout: int = 10,
        cache_only: bool = False
    ):
        """
        Initialize market cap fetcher.

        Args:
            cache_dir: Directory for caching market cap data
            cache_ttl_days: Cache time-to-live in days
            timeout: Request timeout in seconds
            cache_only: Skip all API calls, use cached data only (for PIT estimation)
        """
        self.cache_only = cache_only
        # API configuration from environment
        self.api_keys = {
            'twelvedata': os.getenv('TWELVEDATA_API_KEY'),
            'alphavantage': os.getenv('ALPHAVANTAGE_API_KEY'),
            'fmp': os.getenv('FMP_API_KEY'),
            'finnhub': os.getenv('FINNHUB_API_KEY')
        }

        # Rate limiters (based on free tier limits)
        self.rate_limiters = {
            'twelvedata': RateLimiter(int(os.getenv('TWELVEDATA_RPM', 8))),
            'alphavantage': RateLimiter(int(os.getenv('ALPHAVANTAGE_RPM', 5))),
            'fmp': RateLimiter(int(os.getenv('FMP_RPM', 5))),
            'finnhub': RateLimiter(int(os.getenv('FINNHUB_RPM', 60)))
        }

        # Cache setup
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent / 'data' / 'cache' / 'market_cap'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_ttl_days = cache_ttl_days
        self.timeout = timeout

        # Load existing cache metadata
        self.cache_file = self.cache_dir / 'market_cap_cache.json'
        self.cache = self._load_cache()

        # Track API usage
        self.api_calls = {source: 0 for source in self.api_keys}

        if self.cache_only:
            logger.info(f"MarketCapFetcher initialized in CACHE-ONLY mode (no API calls)")
        else:
            logger.info(f"MarketCapFetcher initialized with {len([k for k, v in self.api_keys.items() if v])} API sources")

    def _load_cache(self) -> Dict[str, MarketCapData]:
        """Load cached market cap data."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    return {k: MarketCapData.from_dict(v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.cache.items()}
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached data is still valid."""
        if ticker not in self.cache:
            return False

        cached = self.cache[ticker]
        fetched_at = datetime.fromisoformat(cached.fetched_at)
        age = datetime.now() - fetched_at

        return age < timedelta(days=self.cache_ttl_days)

    def get_market_cap(
        self,
        ticker: str,
        force_refresh: bool = False
    ) -> Optional[MarketCapData]:
        """
        Get market cap for a ticker.

        Tries multiple API sources with fallback.

        Args:
            ticker: Stock ticker symbol
            force_refresh: Force API call even if cached

        Returns:
            MarketCapData object or None if unavailable
        """
        # Check cache first
        if not force_refresh and self._is_cache_valid(ticker):
            logger.debug(f"{ticker}: Cache hit")
            return self.cache[ticker]

        # If cache_only mode, don't call APIs - return None for cache misses
        if self.cache_only:
            logger.debug(f"{ticker}: Cache miss, cache_only mode - skipping API calls")
            return None

        # Try each API source in order
        fetchers = [
            ('twelvedata', self._fetch_twelvedata),
            ('alphavantage', self._fetch_alphavantage),
            ('fmp', self._fetch_fmp),
            ('finnhub', self._fetch_finnhub)
        ]

        for source, fetcher in fetchers:
            if not self.api_keys.get(source):
                continue

            try:
                data = fetcher(ticker)
                if data is not None:
                    # Update cache
                    self.cache[ticker] = data
                    self._save_cache()
                    self.api_calls[source] += 1
                    logger.info(f"{ticker}: Market cap fetched from {source}")
                    return data

            except Exception as e:
                logger.debug(f"{ticker}: {source} failed - {e}")
                continue

        logger.warning(f"{ticker}: No market cap data available from any source")
        return None

    def _fetch_twelvedata(self, ticker: str) -> Optional[MarketCapData]:
        """Fetch from TwelveData statistics endpoint."""
        api_key = self.api_keys['twelvedata']
        if not api_key:
            return None

        self.rate_limiters['twelvedata'].wait()

        url = f"https://api.twelvedata.com/statistics"
        params = {
            'symbol': ticker,
            'apikey': api_key
        }

        response = requests.get(url, params=params, timeout=self.timeout)

        if response.status_code != 200:
            return None

        data = response.json()

        if 'status' in data and data['status'] == 'error':
            return None

        # TwelveData returns statistics including shares outstanding
        stats = data.get('statistics', {})
        shares = stats.get('shares_outstanding')
        market_cap = stats.get('market_capitalization')

        # Only return if we have a valid, non-zero market cap from the API
        if market_cap:
            mc_value = float(market_cap)
            if mc_value > 0:
                return MarketCapData(
                    ticker=ticker,
                    market_cap=mc_value,
                    shares_outstanding=float(shares) if shares else None,
                    source='twelvedata',
                    fetched_at=datetime.now().isoformat()
                )

        return None

    def _fetch_alphavantage(self, ticker: str) -> Optional[MarketCapData]:
        """Fetch from Alpha Vantage OVERVIEW endpoint."""
        api_key = self.api_keys['alphavantage']
        if not api_key:
            return None

        self.rate_limiters['alphavantage'].wait()

        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': api_key
        }

        response = requests.get(url, params=params, timeout=self.timeout)

        if response.status_code != 200:
            return None

        data = response.json()

        if 'Error Message' in data or 'Note' in data:
            return None

        market_cap = data.get('MarketCapitalization')
        shares = data.get('SharesOutstanding')

        # Only return if we have a valid, non-zero market cap from the API
        if market_cap:
            mc_value = float(market_cap)
            if mc_value > 0:
                return MarketCapData(
                    ticker=ticker,
                    market_cap=mc_value,
                    shares_outstanding=float(shares) if shares else None,
                    source='alphavantage',
                    fetched_at=datetime.now().isoformat()
                )

        return None

    def _fetch_fmp(self, ticker: str) -> Optional[MarketCapData]:
        """Fetch from Financial Modeling Prep profile endpoint."""
        api_key = self.api_keys['fmp']
        if not api_key:
            return None

        self.rate_limiters['fmp'].wait()

        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
        params = {'apikey': api_key}

        response = requests.get(url, params=params, timeout=self.timeout)

        if response.status_code != 200:
            return None

        data = response.json()

        if not data or isinstance(data, dict) and 'Error Message' in data:
            return None

        # FMP returns a list
        if isinstance(data, list) and len(data) > 0:
            profile = data[0]
            market_cap = profile.get('mktCap')

            # Only return if we have a valid, non-zero market cap from the API
            if market_cap:
                mc_value = float(market_cap)
                if mc_value > 0:
                    shares = profile.get('sharesOutstanding')
                    return MarketCapData(
                        ticker=ticker,
                        market_cap=mc_value,
                        shares_outstanding=float(shares) if shares else None,
                        source='fmp',
                        fetched_at=datetime.now().isoformat()
                    )

        return None

    def _fetch_finnhub(self, ticker: str) -> Optional[MarketCapData]:
        """Fetch from Finnhub company profile endpoint."""
        api_key = self.api_keys['finnhub']
        if not api_key:
            return None

        self.rate_limiters['finnhub'].wait()

        url = f"https://finnhub.io/api/v1/stock/profile2"
        params = {
            'symbol': ticker,
            'token': api_key
        }

        response = requests.get(url, params=params, timeout=self.timeout)

        if response.status_code != 200:
            return None

        data = response.json()

        if not data or 'shareOutstanding' not in data:
            return None

        shares = data.get('shareOutstanding')
        market_cap = data.get('marketCapitalization')

        # Finnhub returns market cap in millions
        if market_cap:
            # Convert from millions to actual value
            market_cap = float(market_cap) * 1_000_000
        else:
            # No market cap available - do NOT return data without it
            return None

        # Only return if we have a valid, non-zero market cap
        if market_cap and market_cap > 0:
            return MarketCapData(
                ticker=ticker,
                market_cap=market_cap,
                shares_outstanding=float(shares) * 1_000_000 if shares else None,  # shares in millions
                source='finnhub',
                fetched_at=datetime.now().isoformat()
            )

        return None

    def get_historical_market_cap(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        shares_outstanding: Optional[float] = None
    ) -> Optional[pd.DataFrame]:
        """
        Estimate historical market cap based on shares outstanding and price.

        Uses adjusted close when available for accurate historical calculation that
        accounts for stock splits, dividends, and other corporate actions.

        Args:
            ticker: Stock ticker symbol
            price_data: DataFrame with 'date' and 'close' columns (optionally 'adj_close')
            shares_outstanding: Number of shares (fetched if not provided)

        Returns:
            DataFrame with 'date', 'close', 'market_cap' columns, or None if data unavailable.
            NEVER returns mock or estimated data - only real API data.
        """
        # Get shares outstanding if not provided
        if shares_outstanding is None:
            data = self.get_market_cap(ticker)
            if data is None:
                logger.warning(f"{ticker}: No market cap data available from any API source")
                return None
            if data.shares_outstanding is None:
                logger.warning(f"{ticker}: Market cap available but shares outstanding missing - cannot calculate historical")
                return None
            shares_outstanding = data.shares_outstanding

        # Validate shares_outstanding is a real value
        if shares_outstanding <= 0:
            logger.warning(f"{ticker}: Invalid shares outstanding value: {shares_outstanding}")
            return None

        # Calculate historical market cap using real API data
        result = price_data.copy()
        result['shares_outstanding'] = shares_outstanding

        # Use adjusted close if available (accounts for splits/dividends), otherwise use close
        if 'adj_close' in result.columns:
            result['market_cap'] = result['adj_close'] * shares_outstanding
            logger.info(f"{ticker}: Calculated historical market cap for {len(result)} days using adj_close (split-adjusted)")
        else:
            result['market_cap'] = result['close'] * shares_outstanding
            logger.info(f"{ticker}: Calculated historical market cap for {len(result)} days using close (warning: not split-adjusted)")

        return result

    def get_pit_market_cap(
        self,
        ticker: str,
        price_on_date: float,
        reference_date: Optional[str] = None
    ) -> Tuple[Optional[float], str]:
        """
        Get point-in-time market cap using cached shares × price.

        This method calculates historical market cap without making API calls,
        using the shares_outstanding from cache and the price on a specific date.
        This avoids look-ahead bias when analyzing historical patterns.

        Args:
            ticker: Stock ticker symbol
            price_on_date: Close price on the pattern date
            reference_date: Date reference for logging (optional)

        Returns:
            Tuple of (market_cap_value, source) where source is:
            - 'shares_x_price': Calculated from shares_outstanding × price
            - 'price_fallback': Estimated category from price alone (no shares data)
            - 'unavailable': No data available
        """
        # Try to get shares from cache (don't call API)
        if ticker in self.cache and self.cache[ticker].shares_outstanding:
            shares = self.cache[ticker].shares_outstanding
            if shares > 0:
                market_cap = shares * price_on_date
                logger.debug(f"{ticker}: PIT market cap = {shares:,.0f} shares × ${price_on_date:.2f} = ${market_cap:,.0f}")
                return (market_cap, 'shares_x_price')

        # Fallback: No shares data available, return None with price_fallback source
        # The caller can use price-based category estimation
        logger.debug(f"{ticker}: No cached shares_outstanding, using price-based fallback")
        return (None, 'price_fallback')

    def get_batch_market_cap(
        self,
        tickers: list,
        force_refresh: bool = False
    ) -> Dict[str, Optional[MarketCapData]]:
        """
        Get market cap for multiple tickers (sequential).

        Args:
            tickers: List of ticker symbols
            force_refresh: Force API calls even if cached

        Returns:
            Dictionary mapping ticker to MarketCapData
        """
        results = {}

        for ticker in tickers:
            results[ticker] = self.get_market_cap(ticker, force_refresh)

        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"Batch fetch: {successful}/{len(tickers)} successful")

        return results

    def get_batch_market_cap_parallel(
        self,
        tickers: List[str],
        max_workers: int = 4,
        force_refresh: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Optional[MarketCapData]]:
        """
        Get market cap for multiple tickers using parallel threads.

        OPTIMIZATION: Uses ThreadPoolExecutor to fetch from cache in parallel,
        then batches API calls by source to minimize rate limit waiting.

        Args:
            tickers: List of ticker symbols
            max_workers: Maximum parallel workers (default: 4)
            force_refresh: Force API calls even if cached
            progress_callback: Optional callback(completed, total, ticker)

        Returns:
            Dictionary mapping ticker to MarketCapData
        """
        results = {}
        cache_hits = []
        cache_misses = []

        # Phase 1: Check cache in parallel (no rate limiting needed)
        logger.info(f"Checking cache for {len(tickers)} tickers...")
        start_time = time.time()

        for ticker in tickers:
            if not force_refresh and self._is_cache_valid(ticker):
                cache_hits.append(ticker)
                results[ticker] = self.cache[ticker]
            else:
                cache_misses.append(ticker)

        cache_time = time.time() - start_time
        logger.info(f"Cache check: {len(cache_hits)} hits, {len(cache_misses)} misses ({cache_time:.2f}s)")

        # Phase 2: Fetch missing tickers from API (with rate limiting)
        if cache_misses:
            logger.info(f"Fetching {len(cache_misses)} tickers from API...")
            api_start = time.time()

            # Use ThreadPoolExecutor for parallel API calls
            # Note: Rate limiting is still applied per-source
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(self._fetch_with_fallback, ticker): ticker
                    for ticker in cache_misses
                }

                completed = len(cache_hits)
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        data = future.result()
                        results[ticker] = data

                        if data is not None:
                            # Update cache
                            self.cache[ticker] = data

                    except Exception as e:
                        logger.debug(f"{ticker}: Fetch failed: {e}")
                        results[ticker] = None

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, len(tickers), ticker)

            # Save cache after batch
            self._save_cache()

            api_time = time.time() - api_start
            logger.info(f"API fetch completed in {api_time:.2f}s")

        successful = sum(1 for v in results.values() if v is not None)
        total_time = time.time() - start_time
        logger.info(f"Parallel batch fetch: {successful}/{len(tickers)} successful ({total_time:.2f}s)")

        return results

    def _fetch_with_fallback(self, ticker: str) -> Optional[MarketCapData]:
        """
        Fetch market cap with fallback through all sources.

        Separated from get_market_cap to avoid cache operations in threads.
        """
        fetchers = [
            ('twelvedata', self._fetch_twelvedata),
            ('alphavantage', self._fetch_alphavantage),
            ('fmp', self._fetch_fmp),
            ('finnhub', self._fetch_finnhub)
        ]

        for source, fetcher in fetchers:
            if not self.api_keys.get(source):
                continue

            try:
                data = fetcher(ticker)
                if data is not None:
                    self.api_calls[source] += 1
                    logger.debug(f"{ticker}: Fetched from {source}")
                    return data

            except Exception as e:
                logger.debug(f"{ticker}: {source} failed - {e}")
                continue

        return None

    async def get_batch_market_cap_async(
        self,
        tickers: List[str],
        max_concurrent: int = 10,
        force_refresh: bool = False
    ) -> Dict[str, Optional[MarketCapData]]:
        """
        Get market cap for multiple tickers using async HTTP.

        OPTIMIZATION: Uses aiohttp for true async I/O, bypassing rate limiters
        for cached data and batching API requests efficiently.

        Requires: pip install aiohttp

        Args:
            tickers: List of ticker symbols
            max_concurrent: Maximum concurrent requests (default: 10)
            force_refresh: Force API calls even if cached

        Returns:
            Dictionary mapping ticker to MarketCapData
        """
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, falling back to parallel fetch")
            return self.get_batch_market_cap_parallel(tickers, force_refresh=force_refresh)

        results = {}
        cache_misses = []

        # Phase 1: Check cache (sync, fast)
        for ticker in tickers:
            if not force_refresh and self._is_cache_valid(ticker):
                results[ticker] = self.cache[ticker]
            else:
                cache_misses.append(ticker)

        logger.info(f"Async fetch: {len(results)} cached, {len(cache_misses)} to fetch")

        if not cache_misses:
            return results

        # Phase 2: Async fetch from API
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_one(session: 'aiohttp.ClientSession', ticker: str) -> tuple:
            async with semaphore:
                data = await self._async_fetch_with_fallback(session, ticker)
                return ticker, data

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            tasks = [fetch_one(session, ticker) for ticker in cache_misses]
            fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in fetch_results:
                if isinstance(result, Exception):
                    logger.debug(f"Async fetch exception: {result}")
                    continue

                ticker, data = result
                results[ticker] = data
                if data is not None:
                    self.cache[ticker] = data

        # Save cache
        self._save_cache()

        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"Async batch fetch: {successful}/{len(tickers)} successful")

        return results

    async def _async_fetch_with_fallback(
        self,
        session: 'aiohttp.ClientSession',
        ticker: str
    ) -> Optional[MarketCapData]:
        """Async version of fetch with fallback."""
        # Try Finnhub first (highest rate limit)
        if self.api_keys.get('finnhub'):
            try:
                data = await self._async_fetch_finnhub(session, ticker)
                if data is not None:
                    return data
            except Exception as e:
                logger.debug(f"{ticker}: Finnhub async failed: {e}")

        # Try TwelveData
        if self.api_keys.get('twelvedata'):
            try:
                data = await self._async_fetch_twelvedata(session, ticker)
                if data is not None:
                    return data
            except Exception as e:
                logger.debug(f"{ticker}: TwelveData async failed: {e}")

        # Try FMP
        if self.api_keys.get('fmp'):
            try:
                data = await self._async_fetch_fmp(session, ticker)
                if data is not None:
                    return data
            except Exception as e:
                logger.debug(f"{ticker}: FMP async failed: {e}")

        return None

    async def _async_fetch_finnhub(
        self,
        session: 'aiohttp.ClientSession',
        ticker: str
    ) -> Optional[MarketCapData]:
        """Async Finnhub fetch."""
        api_key = self.api_keys['finnhub']
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={api_key}"

        async with session.get(url) as response:
            if response.status != 200:
                return None

            data = await response.json()

            if not data or 'shareOutstanding' not in data:
                return None

            market_cap = data.get('marketCapitalization')
            if market_cap:
                market_cap = float(market_cap) * 1_000_000
            else:
                return None

            if market_cap and market_cap > 0:
                shares = data.get('shareOutstanding')
                return MarketCapData(
                    ticker=ticker,
                    market_cap=market_cap,
                    shares_outstanding=float(shares) * 1_000_000 if shares else None,
                    source='finnhub_async',
                    fetched_at=datetime.now().isoformat()
                )

        return None

    async def _async_fetch_twelvedata(
        self,
        session: 'aiohttp.ClientSession',
        ticker: str
    ) -> Optional[MarketCapData]:
        """Async TwelveData fetch."""
        api_key = self.api_keys['twelvedata']
        url = f"https://api.twelvedata.com/statistics?symbol={ticker}&apikey={api_key}"

        async with session.get(url) as response:
            if response.status != 200:
                return None

            data = await response.json()

            if 'status' in data and data['status'] == 'error':
                return None

            stats = data.get('statistics', {})
            market_cap = stats.get('market_capitalization')

            if market_cap:
                mc_value = float(market_cap)
                if mc_value > 0:
                    shares = stats.get('shares_outstanding')
                    return MarketCapData(
                        ticker=ticker,
                        market_cap=mc_value,
                        shares_outstanding=float(shares) if shares else None,
                        source='twelvedata_async',
                        fetched_at=datetime.now().isoformat()
                    )

        return None

    async def _async_fetch_fmp(
        self,
        session: 'aiohttp.ClientSession',
        ticker: str
    ) -> Optional[MarketCapData]:
        """Async FMP fetch."""
        api_key = self.api_keys['fmp']
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"

        async with session.get(url) as response:
            if response.status != 200:
                return None

            data = await response.json()

            if not data or (isinstance(data, dict) and 'Error Message' in data):
                return None

            if isinstance(data, list) and len(data) > 0:
                profile = data[0]
                market_cap = profile.get('mktCap')

                if market_cap:
                    mc_value = float(market_cap)
                    if mc_value > 0:
                        shares = profile.get('sharesOutstanding')
                        return MarketCapData(
                            ticker=ticker,
                            market_cap=mc_value,
                            shares_outstanding=float(shares) if shares else None,
                            source='fmp_async',
                            fetched_at=datetime.now().isoformat()
                        )

        return None

    def prefetch_market_caps(
        self,
        tickers: List[str],
        use_async: bool = True,
        max_workers: int = 4,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Optional[MarketCapData]]:
        """
        Pre-fetch market caps for a list of tickers before processing.

        RECOMMENDED: Call this before scan_universe() to warm the cache
        and avoid per-ticker API rate limiting delays.

        Args:
            tickers: List of ticker symbols to pre-fetch
            use_async: Use async fetching if aiohttp available (default: True)
            max_workers: Max parallel workers for non-async mode
            progress_callback: Optional callback(completed, total, ticker)

        Returns:
            Dictionary mapping ticker to MarketCapData

        Example:
            fetcher = MarketCapFetcher()
            fetcher.prefetch_market_caps(tickers)  # Warm cache
            # Now scan_ticker() calls will hit cache instead of API
        """
        logger.info(f"Pre-fetching market caps for {len(tickers)} tickers...")
        start_time = time.time()

        if use_async and AIOHTTP_AVAILABLE:
            # Run async in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            results = loop.run_until_complete(
                self.get_batch_market_cap_async(tickers, max_concurrent=max_workers * 2)
            )
        else:
            results = self.get_batch_market_cap_parallel(
                tickers,
                max_workers=max_workers,
                progress_callback=progress_callback
            )

        elapsed = time.time() - start_time
        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"Pre-fetch complete: {successful}/{len(tickers)} in {elapsed:.2f}s "
                   f"({len(tickers)/elapsed:.1f} tickers/sec)")

        return results

    @staticmethod
    def categorize_market_cap_value(market_cap: float) -> str:
        """
        Categorize a market cap value into standard categories.

        Categories:
        - Mega Cap: > $200B
        - Large Cap: $10B - $200B
        - Mid Cap: $2B - $10B
        - Small Cap: $300M - $2B
        - Micro Cap: $50M - $300M
        - Nano Cap: < $50M

        Args:
            market_cap: Market cap value in dollars

        Returns:
            Category string (mega_cap, large_cap, mid_cap, small_cap, micro_cap, nano_cap)
        """
        if market_cap >= 200_000_000_000:
            return 'mega_cap'
        elif market_cap >= 10_000_000_000:
            return 'large_cap'
        elif market_cap >= 2_000_000_000:
            return 'mid_cap'
        elif market_cap >= 300_000_000:
            return 'small_cap'
        elif market_cap >= 50_000_000:
            return 'micro_cap'
        else:
            return 'nano_cap'

    def get_market_cap_category(
        self,
        ticker: str
    ) -> Optional[str]:
        """
        Categorize stock by current market cap.

        Categories:
        - Mega Cap: > $200B
        - Large Cap: $10B - $200B
        - Mid Cap: $2B - $10B
        - Small Cap: $300M - $2B
        - Micro Cap: $50M - $300M
        - Nano Cap: < $50M

        Args:
            ticker: Stock ticker symbol

        Returns:
            Category string or None
        """
        data = self.get_market_cap(ticker)
        if data is None or data.market_cap is None:
            return None

        return self.categorize_market_cap_value(data.market_cap)

    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics."""
        return {
            'cached_tickers': len(self.cache),
            'api_calls': self.api_calls.copy(),
            'total_calls': sum(self.api_calls.values()),
            'cache_file': str(self.cache_file)
        }

    def clear_cache(self):
        """Clear all cached data."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Market cap cache cleared")

    def get_equity_tickers(self) -> List[str]:
        """Get list of tickers that are confirmed equities from cache."""
        return [
            ticker for ticker, data in self.cache.items()
            if data.is_equity()
        ]

    def get_non_equity_tickers(self) -> List[str]:
        """Get list of tickers that are NOT equities (funds, ETFs, etc.)."""
        return [
            ticker for ticker, data in self.cache.items()
            if not data.is_equity()
        ]

    def filter_to_equities(self, tickers: List[str]) -> List[str]:
        """
        Filter a list of tickers to only include equities.

        Uses cached security_type data. Tickers not in cache are included
        (assume equity until proven otherwise).

        Args:
            tickers: List of ticker symbols

        Returns:
            Filtered list containing only equities
        """
        result = []
        excluded = []

        for ticker in tickers:
            if ticker in self.cache:
                if self.cache[ticker].is_equity():
                    result.append(ticker)
                else:
                    excluded.append((ticker, self.cache[ticker].security_type))
            else:
                # Not in cache - include by default
                result.append(ticker)

        if excluded:
            logger.info(f"Filtered out {len(excluded)} non-equity tickers")
            for t, st in excluded[:10]:
                logger.debug(f"  {t}: {st}")

        return result


# Convenience function for simple usage
def get_market_cap(ticker: str) -> Optional[float]:
    """
    Simple function to get market cap for a single ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Market cap in dollars or None
    """
    fetcher = MarketCapFetcher()
    data = fetcher.get_market_cap(ticker)
    return data.market_cap if data else None


if __name__ == "__main__":
    # Test the market cap fetcher
    import sys
    from dotenv import load_dotenv

    # Load environment variables
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)

    logging.basicConfig(level=logging.INFO)

    fetcher = MarketCapFetcher()

    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'TSLA']

    print("\n" + "="*60)
    print("MARKET CAP FETCHER TEST")
    print("="*60)

    for ticker in test_tickers:
        print(f"\nFetching {ticker}...")
        data = fetcher.get_market_cap(ticker)

        if data:
            mc_billions = data.market_cap / 1_000_000_000 if data.market_cap else 0
            print(f"  Market Cap: ${mc_billions:.2f}B")
            print(f"  Source: {data.source}")
            print(f"  Category: {fetcher.get_market_cap_category(ticker)}")
        else:
            print(f"  No data available")

    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    stats = fetcher.get_stats()
    print(f"Cached tickers: {stats['cached_tickers']}")
    print(f"API calls: {stats['api_calls']}")
    print(f"Total calls: {stats['total_calls']}")
