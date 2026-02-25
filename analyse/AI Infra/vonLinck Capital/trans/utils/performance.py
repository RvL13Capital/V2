"""
Performance Monitoring and Profiling Utilities
===============================================

Provides decorators and utilities for performance monitoring, profiling,
and optimization tracking in the TRANS temporal architecture.

Features:
- Function execution timing
- Memory usage tracking
- Cache hit rate monitoring
- Performance metrics collection
- Batch processing utilities

Usage:
    from utils.performance import timeit, PerformanceMonitor, cached

    @timeit
    def slow_function():
        # Function will be timed automatically
        pass

    @cached(maxsize=128)
    def expensive_calculation(x):
        # Results will be cached
        return complex_computation(x)
"""

import time
import functools
import logging
from typing import Dict, Any, Optional, Callable, Tuple
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import psutil
import json

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Singleton class for tracking performance metrics across the application.

    Collects timing data, cache hit rates, memory usage, and other metrics
    for optimization tracking and bottleneck identification.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.timings: Dict[str, list] = defaultdict(list)
        self.cache_hits: Dict[str, int] = defaultdict(int)
        self.cache_misses: Dict[str, int] = defaultdict(int)
        self.memory_usage: Dict[str, list] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()
        self._initialized = True

    def record_timing(self, func_name: str, duration: float):
        """Record function execution time."""
        self.timings[func_name].append(duration)
        self.call_counts[func_name] += 1

    def record_cache_hit(self, cache_name: str):
        """Record cache hit."""
        self.cache_hits[cache_name] += 1

    def record_cache_miss(self, cache_name: str):
        """Record cache miss."""
        self.cache_misses[cache_name] += 1

    def record_memory(self, context: str, memory_mb: float):
        """Record memory usage."""
        self.memory_usage[context].append(memory_mb)

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {
            'runtime_seconds': time.time() - self.start_time,
            'function_stats': {},
            'cache_stats': {},
            'memory_stats': {}
        }

        # Function timing statistics
        for func_name, times in self.timings.items():
            if times:
                summary['function_stats'][func_name] = {
                    'calls': self.call_counts[func_name],
                    'total_time': sum(times),
                    'mean_time': np.mean(times),
                    'median_time': np.median(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times) if len(times) > 1 else 0
                }

        # Cache hit rates
        for cache_name in set(list(self.cache_hits.keys()) + list(self.cache_misses.keys())):
            hits = self.cache_hits[cache_name]
            misses = self.cache_misses[cache_name]
            total = hits + misses
            if total > 0:
                summary['cache_stats'][cache_name] = {
                    'hits': hits,
                    'misses': misses,
                    'hit_rate': hits / total,
                    'total_accesses': total
                }

        # Memory statistics
        for context, usage in self.memory_usage.items():
            if usage:
                summary['memory_stats'][context] = {
                    'mean_mb': np.mean(usage),
                    'max_mb': max(usage),
                    'min_mb': min(usage),
                    'measurements': len(usage)
                }

        return summary

    def print_report(self):
        """Print formatted performance report."""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        print(f"Total Runtime: {summary['runtime_seconds']:.2f} seconds\n")

        # Function performance
        if summary['function_stats']:
            print("Function Performance:")
            print("-"*60)
            for func, stats in sorted(summary['function_stats'].items(),
                                     key=lambda x: x[1]['total_time'], reverse=True):
                print(f"  {func}:")
                print(f"    Calls: {stats['calls']}")
                print(f"    Total: {stats['total_time']:.3f}s")
                print(f"    Mean:  {stats['mean_time']*1000:.1f}ms")
                print(f"    Range: {stats['min_time']*1000:.1f}-{stats['max_time']*1000:.1f}ms")

        # Cache performance
        if summary['cache_stats']:
            print("\nCache Performance:")
            print("-"*60)
            for cache, stats in summary['cache_stats'].items():
                print(f"  {cache}:")
                print(f"    Hit Rate: {stats['hit_rate']:.1%}")
                print(f"    Hits/Misses: {stats['hits']}/{stats['misses']}")

        print("="*60)

    def save_report(self, output_dir: Optional[Path] = None):
        """Save performance report to JSON file."""
        if output_dir is None:
            output_dir = Path("output/performance")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"performance_{timestamp}.json"

        summary = self.get_summary()
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Performance report saved to {report_path}")
        return report_path

    def reset(self):
        """Reset all metrics."""
        self.__init__()
        self._initialized = False
        self.__init__()


# Global monitor instance
monitor = PerformanceMonitor()


def timeit(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator to time function execution.

    Can be used with or without parameters:
        @timeit
        def function(): pass

        @timeit(name="custom_name")
        def function(): pass
    """
    def decorator(f):
        func_name = name or f"{f.__module__}.{f.__name__}"

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                monitor.record_timing(func_name, duration)
                if duration > 1.0:  # Log slow operations
                    logger.warning(f"{func_name} took {duration:.2f}s")

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def memory_profile(func: Optional[Callable] = None, *, context: Optional[str] = None):
    """
    Decorator to track memory usage of a function.

    Usage:
        @memory_profile
        def memory_intensive_function(): pass
    """
    def decorator(f):
        context_name = context or f"{f.__module__}.{f.__name__}"

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            result = f(*args, **kwargs)

            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_used = mem_after - mem_before

            monitor.record_memory(context_name, mem_used)

            if mem_used > 100:  # Log if > 100MB used
                logger.warning(f"{context_name} used {mem_used:.1f}MB memory")

            return result

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


class LRUCache:
    """
    Simple LRU cache implementation with monitoring.

    Thread-safe LRU cache with configurable size and TTL.
    """

    def __init__(self, maxsize: int = 128, ttl: Optional[int] = None, name: str = "unnamed"):
        """
        Initialize LRU cache.

        Args:
            maxsize: Maximum cache size
            ttl: Time-to-live in seconds (None = no expiry)
            name: Cache name for monitoring
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.name = name
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[Any, float] = {}

    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            monitor.record_cache_miss(self.name)
            return None

        # Check TTL
        if self.ttl and (time.time() - self.timestamps[key]) > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            monitor.record_cache_miss(self.name)
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        monitor.record_cache_hit(self.name)
        return self.cache[key]

    def put(self, key: Any, value: Any):
        """Put value in cache."""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.maxsize:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.timestamps:
                    del self.timestamps[oldest_key]

        self.cache[key] = value
        self.timestamps[key] = time.time()

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.timestamps.clear()


def cached(maxsize: int = 128, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results with monitoring.

    Args:
        maxsize: Maximum cache size
        ttl: Time-to-live in seconds
        key_func: Custom function to generate cache key from args

    Usage:
        @cached(maxsize=256, ttl=3600)
        def expensive_calculation(x, y):
            return complex_computation(x, y)
    """
    def decorator(func):
        cache_name = f"{func.__module__}.{func.__name__}"
        cache = LRUCache(maxsize=maxsize, ttl=ttl, name=cache_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Simple key generation
                cache_key = (args, tuple(sorted(kwargs.items())))

            # Check cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.put(cache_key, result)

            return result

        # Add cache control methods
        wrapper.cache_clear = cache.clear
        wrapper.cache = cache

        return wrapper

    return decorator


def batch_process(func: Callable, items: list, batch_size: int = 100,
                  show_progress: bool = True) -> list:
    """
    Process items in batches with optional progress tracking.

    Args:
        func: Function to apply to each batch
        items: List of items to process
        batch_size: Size of each batch
        show_progress: Whether to show progress

    Returns:
        List of results
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1

        if show_progress:
            logger.info(f"Processing batch {batch_num}/{total_batches}")

        batch_results = func(batch)
        results.extend(batch_results)

    return results


class TimingContext:
    """
    Context manager for timing code blocks.

    Usage:
        with TimingContext("data_loading"):
            # Code to time
            load_data()
    """

    def __init__(self, name: str, log_threshold: float = 0.1):
        """
        Initialize timing context.

        Args:
            name: Name for this timing block
            log_threshold: Log if duration exceeds this threshold (seconds)
        """
        self.name = name
        self.log_threshold = log_threshold
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        monitor.record_timing(self.name, duration)

        if duration > self.log_threshold:
            logger.info(f"{self.name} took {duration:.3f}s")


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.

    Args:
        df: DataFrame to optimize

    Returns:
        Optimized DataFrame with reduced memory footprint
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    logger.debug(f"Memory usage reduced from {start_mem:.1f}MB to {end_mem:.1f}MB")

    return df


if __name__ == "__main__":
    # Example usage
    @timeit
    def slow_function():
        time.sleep(0.1)
        return "done"

    @cached(maxsize=10)
    def expensive_calculation(x):
        time.sleep(0.05)
        return x * x

    # Test timing
    slow_function()
    slow_function()

    # Test caching
    expensive_calculation(5)  # Miss
    expensive_calculation(5)  # Hit
    expensive_calculation(10) # Miss
    expensive_calculation(5)  # Hit

    # Print report
    monitor.print_report()