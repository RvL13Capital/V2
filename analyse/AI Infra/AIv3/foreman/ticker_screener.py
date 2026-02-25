"""
Ticker Screener - FMP Market Capitalization Filtering

Uses Financial Modeling Prep Stock Screener API to identify stocks
within target market cap ranges across international exchanges.
"""

import logging
import requests
import random
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from config.foreman_config import ForemanConfig


logger = logging.getLogger(__name__)


@dataclass
class ScreenedStock:
    """Stock that passed market cap screening"""
    ticker: str
    exchange: str
    market_cap: float
    price: Optional[float] = None
    volume: Optional[int] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    is_primary_range: bool = True  # True if in primary range, False if secondary


class TickerScreener:
    """
    Screens stocks by market capitalization using FMP API

    Primary range: $200M - $2B
    Secondary range: $2B - $4B (with sampling)
    """

    def __init__(self, config: ForemanConfig):
        """
        Initialize Ticker Screener

        Args:
            config: Foreman configuration
        """
        self.config = config
        self.api_key = config.api.fmp_api_key
        self.base_url = config.api.fmp_base_url

        # Market cap ranges (in millions)
        self.primary_min, self.primary_max = config.market_cap.get_primary_range_millions()
        self.secondary_min, self.secondary_max = config.market_cap.get_secondary_range_millions()
        self.secondary_sample_pct = config.market_cap.secondary_sample_percentage

        logger.info(
            f"TickerScreener initialized: "
            f"Primary ${self.primary_min}M-${self.primary_max}M, "
            f"Secondary ${self.secondary_min}M-${self.secondary_max}M ({self.secondary_sample_pct*100}% sample)"
        )

    def screen_by_market_cap(
        self,
        exchange: str,
        min_market_cap_millions: int,
        max_market_cap_millions: int,
        limit: int = 10000
    ) -> List[Dict]:
        """
        Screen stocks on a specific exchange by market cap

        Args:
            exchange: Exchange code for FMP (e.g., 'NYSE', 'XETRA')
            min_market_cap_millions: Minimum market cap in millions USD
            max_market_cap_millions: Maximum market cap in millions USD
            limit: Maximum number of results

        Returns:
            List of stock data dictionaries
        """
        logger.info(
            f"Screening {exchange}: ${min_market_cap_millions}M - ${max_market_cap_millions}M"
        )

        try:
            # FMP Stock Screener API
            url = f"{self.base_url}/stock-screener"

            params = {
                'marketCapMoreThan': min_market_cap_millions * 1_000_000,
                'marketCapLowerThan': max_market_cap_millions * 1_000_000,
                'exchange': exchange,
                'limit': limit,
                'apikey': self.api_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not isinstance(data, list):
                logger.warning(f"Unexpected response format for {exchange}: {type(data)}")
                return []

            logger.info(f"Found {len(data)} stocks on {exchange} in range ${min_market_cap_millions}M-${max_market_cap_millions}M")

            return data

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"FMP API rate limit exceeded for {exchange}")
            else:
                logger.error(f"HTTP error screening {exchange}: {e}")
            return []

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error screening {exchange}: {e}")
            return []

        except Exception as e:
            logger.error(f"Unexpected error screening {exchange}: {e}")
            return []

    def screen_exchange(
        self,
        exchange_code: str,
        include_secondary: bool = True
    ) -> List[ScreenedStock]:
        """
        Screen a single exchange for both primary and secondary ranges

        Args:
            exchange_code: Exchange code (e.g., 'NYSE', 'XETRA')
            include_secondary: Include secondary range stocks

        Returns:
            List of screened stocks
        """
        results = []

        # Screen primary range
        primary_data = self.screen_by_market_cap(
            exchange=exchange_code,
            min_market_cap_millions=self.primary_min,
            max_market_cap_millions=self.primary_max
        )

        # Parse primary range stocks
        for stock_data in primary_data:
            stock = self._parse_stock_data(stock_data, exchange_code, is_primary_range=True)
            if stock:
                results.append(stock)

        logger.info(f"{exchange_code} primary range: {len(results)} stocks")

        # Screen secondary range if requested
        if include_secondary:
            secondary_data = self.screen_by_market_cap(
                exchange=exchange_code,
                min_market_cap_millions=self.secondary_min,
                max_market_cap_millions=self.secondary_max
            )

            # Parse and sample secondary range
            secondary_stocks = []
            for stock_data in secondary_data:
                stock = self._parse_stock_data(stock_data, exchange_code, is_primary_range=False)
                if stock:
                    secondary_stocks.append(stock)

            # Sample from secondary range
            num_to_sample = int(len(secondary_stocks) * self.secondary_sample_pct)
            if num_to_sample > 0 and len(secondary_stocks) > 0:
                sampled = random.sample(secondary_stocks, min(num_to_sample, len(secondary_stocks)))
                results.extend(sampled)
                logger.info(
                    f"{exchange_code} secondary range: {len(secondary_stocks)} stocks, "
                    f"sampled {len(sampled)} ({self.secondary_sample_pct*100}%)"
                )

        return results

    def screen_all_exchanges(
        self,
        exchange_codes: Optional[List[str]] = None,
        include_secondary: bool = True
    ) -> Dict[str, List[ScreenedStock]]:
        """
        Screen multiple exchanges

        Args:
            exchange_codes: List of exchange codes, or None for all configured
            include_secondary: Include secondary range stocks

        Returns:
            Dictionary mapping exchange code to list of screened stocks
        """
        if exchange_codes is None:
            exchange_codes = self.config.exchanges.get_all_codes()

        results = {}

        for exchange_code in exchange_codes:
            stocks = self.screen_exchange(exchange_code, include_secondary=include_secondary)
            results[exchange_code] = stocks

        total_stocks = sum(len(stocks) for stocks in results.values())
        logger.info(f"Total screened stocks across {len(results)} exchanges: {total_stocks}")

        return results

    def get_all_screened_tickers(
        self,
        exchange_codes: Optional[List[str]] = None,
        include_secondary: bool = True
    ) -> List[ScreenedStock]:
        """
        Get flat list of all screened stocks

        Args:
            exchange_codes: List of exchange codes, or None for all
            include_secondary: Include secondary range stocks

        Returns:
            List of all screened stocks
        """
        results_by_exchange = self.screen_all_exchanges(exchange_codes, include_secondary)

        all_stocks = []
        for stocks in results_by_exchange.values():
            all_stocks.extend(stocks)

        return all_stocks

    def _parse_stock_data(
        self,
        stock_data: Dict,
        exchange: str,
        is_primary_range: bool
    ) -> Optional[ScreenedStock]:
        """
        Parse stock data from FMP API response

        Args:
            stock_data: Raw stock data from API
            exchange: Exchange code
            is_primary_range: Whether stock is in primary or secondary range

        Returns:
            ScreenedStock object or None if parsing fails
        """
        try:
            ticker = stock_data.get('symbol', '')
            if not ticker:
                return None

            market_cap = stock_data.get('marketCap')
            if market_cap is None:
                return None

            stock = ScreenedStock(
                ticker=ticker,
                exchange=exchange,
                market_cap=float(market_cap),
                price=stock_data.get('price'),
                volume=stock_data.get('volume'),
                sector=stock_data.get('sector'),
                industry=stock_data.get('industry'),
                country=stock_data.get('country'),
                is_primary_range=is_primary_range
            )

            return stock

        except Exception as e:
            logger.debug(f"Error parsing stock data: {e}")
            return None

    def get_statistics(self, screened_stocks: List[ScreenedStock]) -> Dict:
        """
        Get statistics about screened stocks

        Args:
            screened_stocks: List of screened stocks

        Returns:
            Statistics dictionary
        """
        if not screened_stocks:
            return {
                'total': 0,
                'primary_range': 0,
                'secondary_range': 0,
                'by_exchange': {},
                'by_sector': {},
                'avg_market_cap': 0,
                'min_market_cap': 0,
                'max_market_cap': 0
            }

        primary = [s for s in screened_stocks if s.is_primary_range]
        secondary = [s for s in screened_stocks if not s.is_primary_range]

        market_caps = [s.market_cap for s in screened_stocks]

        # By exchange
        by_exchange = {}
        for stock in screened_stocks:
            by_exchange[stock.exchange] = by_exchange.get(stock.exchange, 0) + 1

        # By sector
        by_sector = {}
        for stock in screened_stocks:
            if stock.sector:
                by_sector[stock.sector] = by_sector.get(stock.sector, 0) + 1

        return {
            'total': len(screened_stocks),
            'primary_range': len(primary),
            'secondary_range': len(secondary),
            'by_exchange': by_exchange,
            'by_sector': by_sector,
            'avg_market_cap': sum(market_caps) / len(market_caps),
            'min_market_cap': min(market_caps),
            'max_market_cap': max(market_caps)
        }

    def print_statistics(self, screened_stocks: List[ScreenedStock]):
        """Print screening statistics"""
        stats = self.get_statistics(screened_stocks)

        logger.info("=" * 60)
        logger.info("Ticker Screening Statistics")
        logger.info("=" * 60)
        logger.info(f"Total stocks: {stats['total']}")
        logger.info(f"  Primary range ($200M-$2B): {stats['primary_range']}")
        logger.info(f"  Secondary range ($2B-$4B): {stats['secondary_range']}")
        logger.info("")
        logger.info(f"Average market cap: ${stats['avg_market_cap']/1e9:.2f}B")
        logger.info(f"Market cap range: ${stats['min_market_cap']/1e9:.2f}B - ${stats['max_market_cap']/1e9:.2f}B")
        logger.info("")
        logger.info("By Exchange:")
        for exchange, count in sorted(stats['by_exchange'].items(), key=lambda x: -x[1]):
            logger.info(f"  {exchange}: {count} stocks")
        logger.info("")
        logger.info("By Sector (top 10):")
        for sector, count in sorted(stats['by_sector'].items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {sector}: {count} stocks")
        logger.info("=" * 60)
