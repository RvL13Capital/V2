"""
Modern async data acquisition sources.
"""

from .async_downloader import AsyncDownloader, DownloadResult
from .async_adapter import (
    AsyncYFinanceAdapter,
    AsyncAlphaVantageAdapter,
    AsyncTwelveDataAdapter,
    UnifiedAsyncDownloader,
)

__all__ = [
    'AsyncDownloader',
    'DownloadResult',
    'AsyncYFinanceAdapter',
    'AsyncAlphaVantageAdapter',
    'AsyncTwelveDataAdapter',
    'UnifiedAsyncDownloader',
]
