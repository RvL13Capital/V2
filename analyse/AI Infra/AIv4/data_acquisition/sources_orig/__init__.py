"""
Foreman Service - International Stock Data Acquisition System

This module provides automated collection of international stock data
filtered by market capitalization, with efficient storage in GCS.

Main Components:
- ForemanService: Main orchestration class
- TickerScreener: Market cap filtering via FMP API
- ExchangeMapper: Exchange/ticker normalization via EODHD
- DataDownloader: Historical data retrieval via Alpha Vantage
- APIKeyManager: 4-key rotation with rate limit management
- StorageManager: CSV/Parquet/GCS storage operations
"""

from .foreman_service import ForemanService
from .ticker_screener import TickerScreener
from .exchange_mapper import ExchangeMapper
from .data_downloader import DataDownloader
from .api_key_manager import APIKeyManager
from .storage_manager import StorageManager

__all__ = [
    'ForemanService',
    'TickerScreener',
    'ExchangeMapper',
    'DataDownloader',
    'APIKeyManager',
    'StorageManager',
]

__version__ = '1.0.0'
