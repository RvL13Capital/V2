"""
Ticker Cache Manager for TRANS Temporal Architecture
====================================================

Provides multi-level caching for ticker data to optimize performance:
- Level 1: Memory cache (LRU) for frequently accessed data
- Level 2: Disk cache (parquet) for persistent storage
- Level 3: GCS/source data (fallback)

Features:
- Automatic cache invalidation based on TTL
- Memory-efficient DataFrame storage
- Thread-safe operations
- Performance metrics tracking
- Batch loading support

Usage:
    from utils.cache_manager import TickerCacheManager

    cache = TickerCacheManager(memory_size=100, cache_dir="data/cache")

    # Load ticker with caching
    df = cache.get_ticker('AAPL', start_date='2024-01-01')

    # Batch load multiple tickers
    data = cache.load_batch(['AAPL', 'MSFT', 'GOOGL'])
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime, timedelta
import logging
import pickle
import hashlib
import json
import time
from collections import OrderedDict
import threading

# Import performance monitoring
from .performance import timeit, PerformanceMonitor, optimize_dataframe

logger = logging.getLogger(__name__)
monitor = PerformanceMonitor()


class DataFrameCache:
    """
    Memory-efficient DataFrame cache with TTL support.

    Stores DataFrames with metadata for validation and expiry.
    """

    def __init__(self, maxsize: int = 100, ttl_hours: int = 24):
        """
        Initialize DataFrame cache.

        Args:
            maxsize: Maximum number of DataFrames to cache
            ttl_hours: Time-to-live in hours
        """
        self.maxsize = maxsize
        self.ttl_seconds = ttl_hours * 3600
        self.cache: OrderedDict[str, Tuple[pd.DataFrame, float, Dict]] = OrderedDict()
        self.lock = threading.Lock()

    def _make_key(self, ticker: str, start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> str:
        """Generate cache key from parameters."""
        key_parts = [ticker]
        if start_date:
            key_parts.append(f"start_{start_date}")
        if end_date:
            key_parts.append(f"end_{end_date}")
        return "_".join(key_parts)

    def get(self, ticker: str, start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get DataFrame from cache if valid.

        Returns None if not found or expired.
        """
        key = self._make_key(ticker, start_date, end_date)

        with self.lock:
            if key not in self.cache:
                monitor.record_cache_miss("memory_cache")
                return None

            df, timestamp, metadata = self.cache[key]

            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                monitor.record_cache_miss("memory_cache")
                logger.debug(f"Cache expired for {ticker}")
                return None

            # Move to end (LRU)
            self.cache.move_to_end(key)
            monitor.record_cache_hit("memory_cache")

            # Return copy to prevent modification
            return df.copy()

    def put(self, ticker: str, df: pd.DataFrame,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> None:
        """
        Store DataFrame in cache.

        Automatically optimizes DataFrame memory usage.
        """
        key = self._make_key(ticker, start_date, end_date)

        # Optimize memory usage
        df_optimized = optimize_dataframe(df.copy())

        metadata = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'rows': len(df),
            'columns': list(df.columns),
            'memory_mb': df_optimized.memory_usage().sum() / 1024**2
        }

        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.maxsize:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"Evicted {oldest_key} from memory cache")

            self.cache[key] = (df_optimized, time.time(), metadata)
            logger.debug(f"Cached {ticker} ({metadata['memory_mb']:.1f}MB)")

    def clear(self) -> None:
        """Clear all cached data."""
        with self.lock:
            self.cache.clear()
            logger.info("Memory cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            total_memory = sum(
                meta['memory_mb']
                for _, _, meta in self.cache.values()
            )

            return {
                'items': len(self.cache),
                'capacity': self.maxsize,
                'total_memory_mb': total_memory,
                'tickers': list(set(
                    meta['ticker']
                    for _, _, meta in self.cache.values()
                ))
            }


class TickerCacheManager:
    """
    Multi-level cache manager for ticker data.

    Provides seamless caching with fallback to disk and source data.
    """

    def __init__(
        self,
        memory_size: int = 100,
        cache_dir: Optional[Union[str, Path]] = None,
        ttl_hours: int = 24,
        data_loader: Optional[object] = None
    ):
        """
        Initialize cache manager.

        Args:
            memory_size: Max DataFrames in memory cache
            cache_dir: Directory for disk cache
            ttl_hours: Cache time-to-live in hours
            data_loader: Optional data loader instance
        """
        # Memory cache (Level 1)
        self.memory_cache = DataFrameCache(maxsize=memory_size, ttl_hours=ttl_hours)

        # Disk cache directory (Level 2)
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path("data/cache/tickers")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file for disk cache
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.disk_metadata = self._load_disk_metadata()

        # Data loader for fallback (Level 3)
        self.data_loader = data_loader

        # Performance tracking
        self.access_count = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.source_loads = 0

    def _load_disk_metadata(self) -> Dict:
        """Load disk cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}

    def _save_disk_metadata(self) -> None:
        """Save disk cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.disk_metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def _get_disk_cache_path(self, ticker: str) -> Path:
        """Get path for ticker disk cache."""
        return self.cache_dir / f"{ticker}.parquet"

    @timeit(name="cache.get_ticker")
    def get_ticker(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get ticker data with multi-level caching.

        Search order:
        1. Memory cache (fastest)
        2. Disk cache (fast)
        3. Data loader/source (slowest)

        Args:
            ticker: Ticker symbol
            start_date: Start date for filtering
            end_date: End date for filtering
            force_refresh: Force reload from source

        Returns:
            DataFrame with ticker data or None if not found
        """
        self.access_count += 1

        # Convert dates to strings for caching
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')

        # Level 1: Check memory cache
        if not force_refresh:
            df = self.memory_cache.get(ticker, start_date, end_date)
            if df is not None:
                self.memory_hits += 1
                logger.debug(f"{ticker}: Memory cache hit")
                return df

        # Level 2: Check disk cache
        if not force_refresh:
            df = self._load_from_disk(ticker)
            if df is not None:
                # Filter by date range if needed
                if start_date or end_date:
                    df = self._filter_dates(df, start_date, end_date)

                # Store in memory cache
                self.memory_cache.put(ticker, df, start_date, end_date)
                self.disk_hits += 1
                logger.debug(f"{ticker}: Disk cache hit")
                return df

        # Level 3: Load from source
        df = self._load_from_source(ticker, start_date, end_date)
        if df is not None:
            # Cache for future use
            self._save_to_disk(ticker, df)
            self.memory_cache.put(ticker, df, start_date, end_date)
            self.source_loads += 1
            logger.info(f"{ticker}: Loaded from source")
            return df

        logger.warning(f"{ticker}: No data found")
        return None

    def _load_from_disk(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load ticker data from disk cache."""
        cache_path = self._get_disk_cache_path(ticker)

        if not cache_path.exists():
            return None

        try:
            # Check metadata for expiry
            if ticker in self.disk_metadata:
                cache_time = datetime.fromisoformat(self.disk_metadata[ticker]['cached_at'])
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600

                # Check if expired (default 7 days for disk cache)
                if age_hours > 7 * 24:
                    logger.debug(f"{ticker}: Disk cache expired")
                    cache_path.unlink()
                    del self.disk_metadata[ticker]
                    self._save_disk_metadata()
                    return None

            # Load from disk
            df = pd.read_parquet(cache_path)

            # Ensure date column is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            monitor.record_cache_hit("disk_cache")
            return df

        except Exception as e:
            logger.warning(f"Failed to load {ticker} from disk cache: {e}")
            monitor.record_cache_miss("disk_cache")
            return None

    def _save_to_disk(self, ticker: str, df: pd.DataFrame) -> None:
        """Save ticker data to disk cache."""
        try:
            cache_path = self._get_disk_cache_path(ticker)

            # Save as parquet (efficient format)
            df.to_parquet(cache_path, index=False)

            # Update metadata
            self.disk_metadata[ticker] = {
                'cached_at': datetime.now().isoformat(),
                'rows': len(df),
                'columns': list(df.columns),
                'file_size_mb': cache_path.stat().st_size / 1024**2
            }
            self._save_disk_metadata()

            logger.debug(f"{ticker}: Saved to disk cache")

        except Exception as e:
            logger.warning(f"Failed to save {ticker} to disk cache: {e}")

    def _load_from_source(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load ticker data from source (data loader or GCS)."""
        if self.data_loader is None:
            logger.warning("No data loader configured")
            return None

        try:
            # Try to use the data loader's load_ticker method
            if hasattr(self.data_loader, 'load_ticker'):
                df = self.data_loader.load_ticker(
                    ticker,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=False  # We handle caching
                )
            else:
                logger.warning(f"Data loader doesn't have load_ticker method")
                return None

            monitor.record_cache_miss("source_load")
            return df

        except Exception as e:
            logger.error(f"Failed to load {ticker} from source: {e}")
            return None

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if 'date' not in df.columns:
            return df

        df = df.copy()

        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]

        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        return df

    @timeit(name="cache.load_batch")
    def load_batch(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        parallel: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple tickers efficiently.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for all tickers
            end_date: End date for all tickers
            parallel: Use parallel loading (future feature)

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}

        for ticker in tickers:
            df = self.get_ticker(ticker, start_date, end_date)
            if df is not None:
                results[ticker] = df

        logger.info(f"Loaded {len(results)}/{len(tickers)} tickers")
        return results

    def preload(self, tickers: List[str]) -> None:
        """
        Preload tickers into cache for faster access.

        Args:
            tickers: List of tickers to preload
        """
        logger.info(f"Preloading {len(tickers)} tickers...")

        for ticker in tickers:
            self.get_ticker(ticker)

        logger.info("Preloading complete")

    def clear_memory(self) -> None:
        """Clear memory cache only."""
        self.memory_cache.clear()

    def clear_disk(self) -> None:
        """Clear disk cache."""
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()

        self.disk_metadata = {}
        self._save_disk_metadata()
        logger.info("Disk cache cleared")

    def clear_all(self) -> None:
        """Clear all caches."""
        self.clear_memory()
        self.clear_disk()

    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()

        # Calculate hit rates
        total_hits = self.memory_hits + self.disk_hits
        hit_rate = total_hits / self.access_count if self.access_count > 0 else 0

        # Disk cache stats
        disk_files = list(self.cache_dir.glob("*.parquet"))
        disk_size_mb = sum(f.stat().st_size for f in disk_files) / 1024**2

        return {
            'access_count': self.access_count,
            'memory_hits': self.memory_hits,
            'disk_hits': self.disk_hits,
            'source_loads': self.source_loads,
            'hit_rate': hit_rate,
            'memory_cache': memory_stats,
            'disk_cache': {
                'files': len(disk_files),
                'size_mb': disk_size_mb,
                'tickers': list(self.disk_metadata.keys())
            }
        }

    def print_stats(self) -> None:
        """Print formatted cache statistics."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("CACHE STATISTICS")
        print("="*60)

        print(f"Total Accesses: {stats['access_count']}")
        print(f"Hit Rate: {stats['hit_rate']:.1%}")
        print(f"  Memory Hits: {stats['memory_hits']}")
        print(f"  Disk Hits: {stats['disk_hits']}")
        print(f"  Source Loads: {stats['source_loads']}")

        print(f"\nMemory Cache:")
        print(f"  Items: {stats['memory_cache']['items']}/{stats['memory_cache']['capacity']}")
        print(f"  Memory: {stats['memory_cache']['total_memory_mb']:.1f}MB")

        print(f"\nDisk Cache:")
        print(f"  Files: {stats['disk_cache']['files']}")
        print(f"  Size: {stats['disk_cache']['size_mb']:.1f}MB")

        print("="*60)


if __name__ == "__main__":
    # Example usage and testing
    cache = TickerCacheManager(memory_size=10)

    # Test basic operations
    print("Testing cache manager...")

    # Simulate loading
    for ticker in ['AAPL', 'MSFT', 'GOOGL']:
        print(f"\nLoading {ticker}...")
        df = cache.get_ticker(ticker)
        if df is not None:
            print(f"  Loaded {len(df)} rows")

    # Load again to test cache
    print("\nLoading again (should hit cache)...")
    for ticker in ['AAPL', 'MSFT', 'GOOGL']:
        df = cache.get_ticker(ticker)

    # Print statistics
    cache.print_stats()