"""
Exchange Mapper - EODHD Integration for Exchange/Ticker Normalization

Retrieves exchange listings and normalizes ticker symbols across different
data providers (Yahoo Finance, FMP, Alpha Vantage, EODHD).
"""

import logging
import requests
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from config.foreman_config import ForemanConfig, ExchangeDefinition


logger = logging.getLogger(__name__)


@dataclass
class TickerInfo:
    """Information about a ticker from exchange"""
    ticker: str
    name: str
    exchange_code: str
    country: str
    type: str  # Common Stock, ETF, etc.
    currency: Optional[str] = None
    isin: Optional[str] = None


class ExchangeMapper:
    """
    Maps ticker symbols across different data providers

    Uses EODHD API to get comprehensive exchange listings and
    normalizes ticker formats for different APIs.
    """

    def __init__(self, config: ForemanConfig):
        """
        Initialize Exchange Mapper

        Args:
            config: Foreman configuration
        """
        self.config = config
        self.api_key = config.api.eodhd_api_key
        self.base_url = config.api.eodhd_base_url

        # Cache of exchange listings
        self._exchange_cache: Dict[str, List[TickerInfo]] = {}

        # Ticker normalization maps
        self._ticker_to_exchange: Dict[str, str] = {}

        logger.info(f"ExchangeMapper initialized with {len(config.exchanges.exchanges)} exchanges")

    def get_exchange_tickers(self, exchange_code: str, force_refresh: bool = False) -> List[TickerInfo]:
        """
        Get all tickers for a specific exchange from EODHD

        Args:
            exchange_code: Exchange code (e.g., 'US', 'LSE', 'XETRA')
            force_refresh: Force refresh cache

        Returns:
            List of ticker information
        """
        # Check cache
        if exchange_code in self._exchange_cache and not force_refresh:
            logger.debug(f"Returning cached tickers for {exchange_code} ({len(self._exchange_cache[exchange_code])} tickers)")
            return self._exchange_cache[exchange_code]

        logger.info(f"Fetching ticker list for exchange: {exchange_code}")

        # Get EODHD suffix for this exchange
        exchange_def = self.config.exchanges.get_exchange_by_code(exchange_code)
        if not exchange_def:
            logger.error(f"Exchange {exchange_code} not found in configuration")
            return []

        # EODHD uses specific exchange codes (e.g., 'US', 'LSE', 'XETRA')
        eodhd_code = exchange_def.eodhd_suffix.lstrip('.')

        try:
            # EODHD Exchange Symbol List API
            url = f"{self.base_url}/exchange-symbol-list/{eodhd_code}"
            params = {
                'api_token': self.api_key,
                'fmt': 'json'
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Parse ticker data
            tickers = []
            for item in data:
                # Filter for common stocks (skip ETFs, funds, etc. unless needed)
                ticker_type = item.get('Type', '')

                ticker_info = TickerInfo(
                    ticker=item.get('Code', ''),
                    name=item.get('Name', ''),
                    exchange_code=exchange_code,
                    country=exchange_def.country,
                    type=ticker_type,
                    currency=item.get('Currency'),
                    isin=item.get('Isin')
                )

                tickers.append(ticker_info)
                self._ticker_to_exchange[ticker_info.ticker] = exchange_code

            # Cache results
            self._exchange_cache[exchange_code] = tickers

            logger.info(f"Retrieved {len(tickers)} tickers for {exchange_code}")

            return tickers

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching tickers for {exchange_code}: {e}")
            return []

        except Exception as e:
            logger.error(f"Unexpected error fetching tickers for {exchange_code}: {e}")
            return []

    def get_all_exchange_tickers(self, exchange_codes: Optional[List[str]] = None) -> Dict[str, List[TickerInfo]]:
        """
        Get tickers for multiple exchanges

        Args:
            exchange_codes: List of exchange codes, or None for all configured exchanges

        Returns:
            Dictionary mapping exchange code to list of tickers
        """
        if exchange_codes is None:
            exchange_codes = self.config.exchanges.get_all_codes()

        results = {}

        for exchange_code in exchange_codes:
            tickers = self.get_exchange_tickers(exchange_code)
            results[exchange_code] = tickers

        total_tickers = sum(len(t) for t in results.values())
        logger.info(f"Retrieved {total_tickers} total tickers across {len(results)} exchanges")

        return results

    def normalize_ticker_for_fmp(self, ticker: str, exchange_code: str) -> str:
        """
        Normalize ticker for Financial Modeling Prep API

        Args:
            ticker: Base ticker symbol
            exchange_code: Exchange code

        Returns:
            Normalized ticker with FMP suffix (e.g., 'AAPL', 'ADS.DE')
        """
        exchange_def = self.config.exchanges.get_exchange_by_code(exchange_code)
        if not exchange_def:
            logger.warning(f"Unknown exchange: {exchange_code}, returning ticker as-is")
            return ticker

        # FMP format: TICKER.SUFFIX (e.g., 'ADS.DE' for Adidas on XETRA)
        suffix = exchange_def.fmp_suffix
        if suffix:
            return f"{ticker}{suffix}"

        return ticker

    def normalize_ticker_for_alpha_vantage(self, ticker: str, exchange_code: str) -> str:
        """
        Normalize ticker for Alpha Vantage API

        Args:
            ticker: Base ticker symbol
            exchange_code: Exchange code

        Returns:
            Normalized ticker (Alpha Vantage uses different format for some exchanges)
        """
        exchange_def = self.config.exchanges.get_exchange_by_code(exchange_code)
        if not exchange_def:
            return ticker

        # Alpha Vantage uses specific format for international stocks
        # For most exchanges, it's TICKER.EXCHANGE (e.g., 'ADS.XETRA')
        # US stocks don't need suffix
        if exchange_code in ['NYSE', 'NASDAQ']:
            return ticker

        # For international stocks, use exchange code directly
        return f"{ticker}.{exchange_code}"

    def normalize_ticker_for_yahoo(self, ticker: str, exchange_code: str) -> str:
        """
        Normalize ticker for Yahoo Finance

        Args:
            ticker: Base ticker symbol
            exchange_code: Exchange code

        Returns:
            Normalized ticker with Yahoo suffix (e.g., 'ADS.DE', 'BARC.L')
        """
        exchange_def = self.config.exchanges.get_exchange_by_code(exchange_code)
        if not exchange_def:
            return ticker

        suffix = exchange_def.yahoo_suffix
        if suffix:
            return f"{ticker}{suffix}"

        return ticker

    def get_exchange_for_ticker(self, ticker: str) -> Optional[str]:
        """
        Get exchange code for a ticker

        Args:
            ticker: Ticker symbol

        Returns:
            Exchange code or None if not found
        """
        return self._ticker_to_exchange.get(ticker)

    def filter_common_stocks(self, tickers: List[TickerInfo]) -> List[TickerInfo]:
        """
        Filter to only common stocks (exclude ETFs, funds, etc.)

        Args:
            tickers: List of ticker info

        Returns:
            Filtered list containing only common stocks
        """
        common_stock_types = {'Common Stock', 'Ordinary Share', 'Equity'}

        filtered = [
            t for t in tickers
            if t.type in common_stock_types or 'Stock' in t.type
        ]

        logger.info(f"Filtered {len(tickers)} tickers to {len(filtered)} common stocks")

        return filtered

    def get_tickers_for_screening(
        self,
        exchange_codes: Optional[List[str]] = None,
        common_stocks_only: bool = True
    ) -> Dict[str, Set[str]]:
        """
        Get base ticker symbols ready for market cap screening

        Args:
            exchange_codes: List of exchanges to include, or None for all
            common_stocks_only: Filter to common stocks only

        Returns:
            Dictionary mapping exchange code to set of base ticker symbols
        """
        all_tickers = self.get_all_exchange_tickers(exchange_codes)

        results = {}

        for exchange_code, tickers in all_tickers.items():
            if common_stocks_only:
                tickers = self.filter_common_stocks(tickers)

            # Extract just the ticker symbols
            ticker_set = {t.ticker for t in tickers if t.ticker}
            results[exchange_code] = ticker_set

        return results

    def get_statistics(self) -> Dict:
        """Get statistics about cached exchanges"""
        total_tickers = sum(len(tickers) for tickers in self._exchange_cache.values())

        return {
            'exchanges_cached': len(self._exchange_cache),
            'total_tickers': total_tickers,
            'tickers_per_exchange': {
                exchange: len(tickers)
                for exchange, tickers in self._exchange_cache.items()
            }
        }

    def print_statistics(self):
        """Print exchange mapper statistics"""
        stats = self.get_statistics()

        logger.info("=" * 60)
        logger.info("Exchange Mapper Statistics")
        logger.info("=" * 60)
        logger.info(f"Exchanges cached: {stats['exchanges_cached']}")
        logger.info(f"Total tickers: {stats['total_tickers']}")
        logger.info("")

        for exchange, count in stats['tickers_per_exchange'].items():
            logger.info(f"  {exchange}: {count} tickers")

        logger.info("=" * 60)
