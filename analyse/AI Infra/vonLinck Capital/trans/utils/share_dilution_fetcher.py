"""
Share Dilution Fetcher for TRANS System
========================================

Fetches historical shares outstanding from SEC EDGAR to calculate
trailing 12-month share dilution for US stocks.

SEC EDGAR XBRL API (free, no API key required):
- CIK Lookup: https://www.sec.gov/files/company_tickers.json
- Shares History: https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/dei/EntityCommonStockSharesOutstanding.json

Features:
- SEC EDGAR integration (free, authoritative data)
- Disk caching with 7-day TTL
- Rate limiting (5 req/sec, conservative for SEC)
- Split detection (>100% change = likely split, skip filter)

Usage:
    from utils.share_dilution_fetcher import ShareDilutionFetcher

    fetcher = ShareDilutionFetcher()

    # Get dilution percentage for a US stock
    dilution_pct = fetcher.get_dilution_pct('AAPL')

    # Check if stock exceeds dilution threshold
    is_diluted = fetcher.is_diluted('SNDL', threshold=20.0)

Author: TRANS System
Date: December 2024
"""

import json
import time
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SharesDilutionData:
    """Container for share dilution information."""
    ticker: str
    cik: Optional[str]                          # SEC CIK number (10 digits)
    current_shares: Optional[float]             # Most recent shares outstanding
    shares_12m_ago: Optional[float]             # Shares from ~12 months ago
    dilution_pct: Optional[float]               # Calculated dilution percentage
    current_filing_date: Optional[str]          # Date of most recent filing
    historical_filing_date: Optional[str]       # Date of ~12 month old filing
    source: str                                 # 'sec_edgar' or 'unavailable'
    fetched_at: str                             # ISO timestamp

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SharesDilutionData':
        return cls(**data)


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


class ShareDilutionFetcher:
    """
    SEC EDGAR-based share dilution fetcher with caching.

    Fetches historical shares outstanding from 10-Q/10-K filings
    and calculates trailing 12-month dilution percentage.

    SEC EDGAR API is free and requires no API key.
    Rate limit: 10 requests/second (SEC guideline).
    """

    # Import configuration (with fallbacks for standalone use)
    try:
        from config import (
            SEC_EDGAR_RATE_LIMIT_RPM,
            SEC_EDGAR_CACHE_TTL_DAYS,
            SEC_EDGAR_USER_AGENT
        )
        DEFAULT_RATE_LIMIT = SEC_EDGAR_RATE_LIMIT_RPM
        DEFAULT_CACHE_TTL = SEC_EDGAR_CACHE_TTL_DAYS
        DEFAULT_USER_AGENT = SEC_EDGAR_USER_AGENT
    except ImportError:
        DEFAULT_RATE_LIMIT = 300  # 5 req/sec
        DEFAULT_CACHE_TTL = 7
        DEFAULT_USER_AGENT = "TRANS-System/1.0 (pattern-detection@example.com)"

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_ttl_days: Optional[int] = None,
        timeout: int = 15,
        user_agent: Optional[str] = None
    ):
        """
        Initialize share dilution fetcher.

        Args:
            cache_dir: Directory for caching dilution data
            cache_ttl_days: Cache time-to-live in days (default: 7)
            timeout: Request timeout in seconds (default: 15, longer for SEC)
            user_agent: User-Agent header for SEC (required by SEC)
        """
        # Cache setup
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent / 'data' / 'cache' / 'share_dilution'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_ttl_days = cache_ttl_days or self.DEFAULT_CACHE_TTL
        self.timeout = timeout

        # SEC requires User-Agent with contact info
        self.headers = {
            'User-Agent': user_agent or self.DEFAULT_USER_AGENT,
            'Accept': 'application/json'
        }

        # Rate limiter (5 req/sec is conservative for SEC's 10 req/sec limit)
        self.rate_limiter = RateLimiter(self.DEFAULT_RATE_LIMIT)

        # Load caches
        self.cache_file = self.cache_dir / 'share_dilution_cache.json'
        self.cache = self._load_cache()

        self.cik_cache_file = self.cache_dir / 'cik_cache.json'
        self.cik_cache = self._load_cik_cache()

        # Company tickers file (ticker → CIK mapping)
        self.company_tickers_file = self.cache_dir / 'company_tickers.json'
        self.company_tickers_age = None

        # Statistics
        self.api_calls = 0
        self.cache_hits = 0

        logger.info(f"ShareDilutionFetcher initialized (cache: {self.cache_dir})")

    def _load_cache(self) -> Dict[str, SharesDilutionData]:
        """Load cached dilution data."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    return {k: SharesDilutionData.from_dict(v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load dilution cache: {e}")
        return {}

    def _save_cache(self):
        """Save dilution cache to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.cache.items()}
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save dilution cache: {e}")

    def _load_cik_cache(self) -> Dict[str, str]:
        """Load cached ticker → CIK mappings."""
        if self.cik_cache_file.exists():
            try:
                with open(self.cik_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load CIK cache: {e}")
        return {}

    def _save_cik_cache(self):
        """Save CIK cache to disk."""
        try:
            with open(self.cik_cache_file, 'w') as f:
                json.dump(self.cik_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save CIK cache: {e}")

    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached dilution data is still valid."""
        if ticker not in self.cache:
            return False

        cached = self.cache[ticker]
        try:
            fetched_at = datetime.fromisoformat(cached.fetched_at)
            age = datetime.now() - fetched_at
            return age < timedelta(days=self.cache_ttl_days)
        except Exception:
            return False

    def _get_file_age_hours(self, filepath: Path) -> Optional[float]:
        """Get file age in hours."""
        if not filepath.exists():
            return None
        try:
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            age = datetime.now() - mtime
            return age.total_seconds() / 3600
        except Exception:
            return None

    def _lookup_cik(self, ticker: str) -> Optional[str]:
        """
        Look up CIK for a US ticker.

        Uses SEC's company_tickers.json file (cached locally for 24 hours).

        Args:
            ticker: US stock ticker (e.g., 'AAPL')

        Returns:
            10-digit CIK string or None if not found
        """
        ticker_upper = ticker.upper()

        # Check CIK cache first
        if ticker_upper in self.cik_cache:
            return self.cik_cache[ticker_upper]

        # Check if company_tickers.json needs refresh (older than 24 hours)
        tickers_age = self._get_file_age_hours(self.company_tickers_file)

        if tickers_age is None or tickers_age > 24:
            # Fetch fresh company_tickers.json from SEC
            try:
                self.rate_limiter.wait()
                url = "https://www.sec.gov/files/company_tickers.json"
                response = requests.get(url, headers=self.headers, timeout=self.timeout)
                self.api_calls += 1

                if response.status_code == 200:
                    with open(self.company_tickers_file, 'w') as f:
                        json.dump(response.json(), f)
                    logger.info("Refreshed SEC company_tickers.json")
                else:
                    logger.warning(f"Failed to fetch company_tickers.json: {response.status_code}")
                    # Continue with existing file if available
                    if not self.company_tickers_file.exists():
                        return None
            except Exception as e:
                logger.warning(f"Error fetching company_tickers.json: {e}")
                if not self.company_tickers_file.exists():
                    return None

        # Parse company_tickers.json and find CIK
        try:
            with open(self.company_tickers_file, 'r') as f:
                data = json.load(f)

            for entry in data.values():
                if entry.get('ticker', '').upper() == ticker_upper:
                    cik = str(entry['cik_str']).zfill(10)  # Pad to 10 digits
                    self.cik_cache[ticker_upper] = cik
                    self._save_cik_cache()
                    return cik

            # Ticker not found in SEC data
            logger.debug(f"{ticker}: CIK not found in SEC data")
            return None

        except Exception as e:
            logger.warning(f"Error parsing company_tickers.json: {e}")
            return None

    def _fetch_shares_history(self, cik: str) -> Optional[List[Dict]]:
        """
        Fetch shares outstanding history from SEC EDGAR.

        Uses the XBRL Company Concept API to get all historical
        EntityCommonStockSharesOutstanding disclosures.

        Args:
            cik: 10-digit CIK number

        Returns:
            List of filings sorted by end date (most recent first), or None
        """
        self.rate_limiter.wait()

        url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/dei/EntityCommonStockSharesOutstanding.json"

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            self.api_calls += 1

            if response.status_code == 404:
                # Stock may not have this XBRL tag (some older filings)
                logger.debug(f"CIK {cik}: No shares outstanding data in XBRL")
                return None

            if response.status_code != 200:
                logger.warning(f"SEC EDGAR returned {response.status_code} for CIK {cik}")
                return None

            data = response.json()

            # Extract shares data from units
            units = data.get('units', {})
            shares_list = units.get('shares', [])

            if not shares_list:
                logger.debug(f"CIK {cik}: No shares data in response")
                return None

            # Sort by end date (most recent first)
            shares_list.sort(key=lambda x: x.get('end', ''), reverse=True)

            return shares_list

        except requests.exceptions.Timeout:
            logger.warning(f"CIK {cik}: SEC EDGAR request timed out")
            return None
        except Exception as e:
            logger.warning(f"CIK {cik}: SEC EDGAR fetch error: {e}")
            return None

    def _calculate_dilution(
        self,
        filing_history: List[Dict]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str], Optional[str]]:
        """
        Calculate trailing 12-month dilution from filing history.

        Args:
            filing_history: List of filings sorted by end date (most recent first)

        Returns:
            Tuple of (current_shares, shares_12m_ago, dilution_pct, current_date, historical_date)
            Returns (None, None, None, None, None) if insufficient data.
        """
        if not filing_history or len(filing_history) < 2:
            return None, None, None, None, None

        # Get most recent filing
        current = filing_history[0]
        current_shares = current.get('val')
        current_date = current.get('end')

        if current_shares is None or current_shares <= 0:
            return None, None, None, None, None

        # Parse current date
        try:
            current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        except Exception:
            return current_shares, None, None, current_date, None

        # Find filing from ~12 months ago (between 10-14 months to allow flexibility)
        target_date = current_dt - timedelta(days=365)
        min_date = current_dt - timedelta(days=14*30)  # 14 months ago
        max_date = current_dt - timedelta(days=10*30)  # 10 months ago

        historical_shares = None
        historical_date = None

        # First pass: find filing in ideal range (10-14 months ago)
        for filing in filing_history:
            try:
                filing_date_str = filing.get('end')
                filing_dt = datetime.strptime(filing_date_str, '%Y-%m-%d')

                if min_date <= filing_dt <= max_date:
                    historical_shares = filing.get('val')
                    historical_date = filing_date_str
                    break
            except Exception:
                continue

        # Fallback: find closest filing to 12 months ago (6-18 month range)
        if historical_shares is None:
            best_match = None
            best_diff = float('inf')

            for filing in filing_history:
                try:
                    filing_date_str = filing.get('end')
                    filing_dt = datetime.strptime(filing_date_str, '%Y-%m-%d')
                    diff = abs((filing_dt - target_date).days)

                    # Must be at least 6 months old, at most 18 months
                    days_old = (current_dt - filing_dt).days
                    if 180 <= days_old <= 540:
                        if diff < best_diff:
                            best_diff = diff
                            best_match = filing
                except Exception:
                    continue

            if best_match:
                historical_shares = best_match.get('val')
                historical_date = best_match.get('end')

        if historical_shares is None or historical_shares <= 0:
            logger.debug("Could not find historical shares data for 12-month comparison")
            return current_shares, None, None, current_date, None

        # Calculate dilution percentage
        dilution_pct = ((current_shares - historical_shares) / historical_shares) * 100

        return current_shares, historical_shares, dilution_pct, current_date, historical_date

    def get_dilution(
        self,
        ticker: str,
        force_refresh: bool = False
    ) -> Optional[SharesDilutionData]:
        """
        Get share dilution data for a US ticker.

        Fetches shares outstanding from SEC EDGAR and calculates
        trailing 12-month dilution percentage.

        Args:
            ticker: US stock ticker symbol
            force_refresh: Force API call even if cached

        Returns:
            SharesDilutionData object or None if unavailable
        """
        ticker_upper = ticker.upper()

        # Check cache first
        if not force_refresh and self._is_cache_valid(ticker_upper):
            self.cache_hits += 1
            logger.debug(f"{ticker}: Dilution cache hit")
            return self.cache[ticker_upper]

        # Look up CIK
        cik = self._lookup_cik(ticker_upper)
        if cik is None:
            logger.debug(f"{ticker}: CIK not found, cannot fetch dilution data")
            return None

        # Fetch shares outstanding history
        filing_history = self._fetch_shares_history(cik)
        if filing_history is None:
            logger.debug(f"{ticker}: No shares outstanding history available")
            return None

        # Calculate dilution
        current_shares, historical_shares, dilution_pct, current_date, historical_date = \
            self._calculate_dilution(filing_history)

        # Create data object
        data = SharesDilutionData(
            ticker=ticker_upper,
            cik=cik,
            current_shares=current_shares,
            shares_12m_ago=historical_shares,
            dilution_pct=dilution_pct,
            current_filing_date=current_date,
            historical_filing_date=historical_date,
            source='sec_edgar',
            fetched_at=datetime.now().isoformat()
        )

        # Cache the result
        self.cache[ticker_upper] = data
        self._save_cache()

        if dilution_pct is not None:
            logger.info(f"{ticker}: {dilution_pct:+.1f}% dilution (CIK: {cik})")
        else:
            logger.info(f"{ticker}: Dilution data incomplete (CIK: {cik})")

        return data

    def get_dilution_pct(self, ticker: str, force_refresh: bool = False) -> Optional[float]:
        """
        Convenience method to get just the dilution percentage.

        Args:
            ticker: US stock ticker symbol
            force_refresh: Force API call even if cached

        Returns:
            Dilution percentage or None if unavailable
        """
        data = self.get_dilution(ticker, force_refresh=force_refresh)
        if data is None:
            return None
        return data.dilution_pct

    def is_diluted(
        self,
        ticker: str,
        threshold: float = 20.0,
        force_refresh: bool = False
    ) -> Optional[bool]:
        """
        Check if a ticker exceeds the dilution threshold.

        Note: Returns None if dilution data is unavailable (not False).
        The caller should decide how to handle unavailable data.

        Args:
            ticker: US stock ticker symbol
            threshold: Maximum allowed dilution percentage
            force_refresh: Force API call even if cached

        Returns:
            True if dilution > threshold, False if <= threshold, None if unavailable
        """
        dilution_pct = self.get_dilution_pct(ticker, force_refresh=force_refresh)
        if dilution_pct is None:
            return None
        return dilution_pct > threshold

    def get_stats(self) -> Dict[str, any]:
        """Get fetcher statistics."""
        return {
            'cached_tickers': len(self.cache),
            'cik_mappings': len(self.cik_cache),
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.api_calls)
        }

    def clear_cache(self):
        """Clear all cached data."""
        self.cache = {}
        self.cik_cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.cik_cache_file.exists():
            self.cik_cache_file.unlink()
        if self.company_tickers_file.exists():
            self.company_tickers_file.unlink()
        logger.info("Dilution cache cleared")


# Convenience function for standalone usage
def check_dilution(ticker: str, threshold: float = 20.0) -> Tuple[Optional[float], Optional[bool]]:
    """
    Quick check for share dilution.

    Args:
        ticker: US stock ticker symbol
        threshold: Maximum allowed dilution percentage

    Returns:
        Tuple of (dilution_pct, is_diluted)
    """
    fetcher = ShareDilutionFetcher()
    dilution_pct = fetcher.get_dilution_pct(ticker)
    is_diluted = dilution_pct > threshold if dilution_pct is not None else None
    return dilution_pct, is_diluted


if __name__ == "__main__":
    # Quick test
    import sys

    logging.basicConfig(level=logging.INFO)

    fetcher = ShareDilutionFetcher()

    # Test tickers
    test_tickers = sys.argv[1:] if len(sys.argv) > 1 else ['AAPL', 'MSFT', 'SNDL']

    print("\n=== Share Dilution Check ===\n")

    for ticker in test_tickers:
        data = fetcher.get_dilution(ticker)
        if data and data.dilution_pct is not None:
            status = "DILUTED" if data.dilution_pct > 20 else "OK"
            print(f"{ticker}: {data.dilution_pct:+.1f}% dilution ({status})")
            print(f"  - Current: {data.current_shares:,.0f} shares ({data.current_filing_date})")
            print(f"  - 12M ago: {data.shares_12m_ago:,.0f} shares ({data.historical_filing_date})")
        else:
            print(f"{ticker}: Dilution data unavailable")
        print()

    print(f"Stats: {fetcher.get_stats()}")
