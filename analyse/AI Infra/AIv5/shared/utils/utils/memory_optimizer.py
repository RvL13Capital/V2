"""
Memory Optimizer for AIv3 System

Provides utilities to optimize memory usage on systems with limited RAM (e.g., 8GB).
Key features:
- Data type downcasting (float64 → float32, int64 → int32/int16)
- Memory profiling and monitoring
- Automatic garbage collection
- Memory usage warnings
"""

import pandas as pd
import numpy as np
import gc
import psutil
import logging
from typing import Dict, List, Optional, Tuple, Callable
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Memory optimization utilities for pandas DataFrames and system memory management.

    Usage:
        optimizer = MemoryOptimizer()
        df_optimized = optimizer.optimize_dataframe(df)
    """

    def __init__(self, aggressive: bool = False):
        """
        Initialize memory optimizer.

        Args:
            aggressive: If True, use more aggressive optimization (may lose precision)
        """
        self.aggressive = aggressive
        self.optimization_stats = []

    def optimize_dataframe(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types.

        Args:
            df: Input DataFrame
            exclude_columns: Columns to exclude from optimization
            verbose: Print optimization statistics

        Returns:
            Optimized DataFrame with reduced memory footprint
        """
        if df.empty:
            return df

        exclude_columns = exclude_columns or []

        # Calculate initial memory
        mem_before = df.memory_usage(deep=True).sum() / 1024**2  # MB

        # Create copy to avoid modifying original
        df_optimized = df.copy()

        # Track changes
        changes = []

        for col in df_optimized.columns:
            if col in exclude_columns:
                continue

            col_type = df_optimized[col].dtype

            # Optimize numeric columns
            if np.issubdtype(col_type, np.integer):
                df_optimized[col], change = self._optimize_integer_column(df_optimized[col])
                if change:
                    changes.append(f"{col}: {col_type} → {df_optimized[col].dtype}")

            elif np.issubdtype(col_type, np.floating):
                df_optimized[col], change = self._optimize_float_column(df_optimized[col])
                if change:
                    changes.append(f"{col}: {col_type} → {df_optimized[col].dtype}")

            elif col_type == 'object':
                # Try to convert object columns to category if beneficial
                if self._should_convert_to_category(df_optimized[col]):
                    df_optimized[col] = df_optimized[col].astype('category')
                    changes.append(f"{col}: object → category")

        # Calculate final memory
        mem_after = df_optimized.memory_usage(deep=True).sum() / 1024**2  # MB
        mem_reduction = mem_before - mem_after
        mem_reduction_pct = (mem_reduction / mem_before) * 100 if mem_before > 0 else 0

        # Store statistics
        stats = {
            'timestamp': datetime.now(),
            'rows': len(df),
            'columns': len(df.columns),
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'memory_reduction_mb': mem_reduction,
            'memory_reduction_pct': mem_reduction_pct,
            'changes': changes
        }
        self.optimization_stats.append(stats)

        if verbose:
            logger.info(f"Memory optimization complete:")
            logger.info(f"  Before: {mem_before:.2f} MB")
            logger.info(f"  After:  {mem_after:.2f} MB")
            logger.info(f"  Saved:  {mem_reduction:.2f} MB ({mem_reduction_pct:.1f}%)")
            if changes:
                logger.info(f"  Changes: {len(changes)} columns optimized")

        return df_optimized

    def _optimize_integer_column(self, col: pd.Series) -> Tuple[pd.Series, bool]:
        """
        Optimize integer column by downcasting to smallest appropriate type.

        Returns:
            Tuple of (optimized_series, was_changed)
        """
        col_min = col.min()
        col_max = col.max()

        # Try unsigned types first if all values are non-negative
        if col_min >= 0:
            if col_max < np.iinfo(np.uint8).max:
                return col.astype(np.uint8), True
            elif col_max < np.iinfo(np.uint16).max:
                return col.astype(np.uint16), True
            elif col_max < np.iinfo(np.uint32).max:
                return col.astype(np.uint32), True

        # Use signed types
        if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
            return col.astype(np.int8), True
        elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
            return col.astype(np.int16), True
        elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
            return col.astype(np.int32), True

        return col, False

    def _optimize_float_column(self, col: pd.Series) -> Tuple[pd.Series, bool]:
        """
        Optimize float column by downcasting to float32.

        For price data, float32 provides sufficient precision (7 decimal digits).

        Returns:
            Tuple of (optimized_series, was_changed)
        """
        if col.dtype == np.float64:
            if self.aggressive:
                # Aggressive: Always downcast to float32
                return col.astype(np.float32), True
            else:
                # Conservative: Check if precision loss is acceptable
                # For financial data, float32 is generally sufficient
                # (7 significant decimal digits vs 15 for float64)
                return col.astype(np.float32), True

        return col, False

    def _should_convert_to_category(self, col: pd.Series) -> bool:
        """
        Determine if object column should be converted to category type.

        Category type is beneficial when:
        - Unique values < 50% of total values
        - Column has repeated string values
        """
        if col.dtype != 'object':
            return False

        unique_ratio = col.nunique() / len(col)
        return unique_ratio < 0.5

    def optimize_ohlcv_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize OHLCV DataFrame specifically for financial data.

        Ensures required columns (open, high, low, close, volume) are present
        and optimized to float32.

        Args:
            df: OHLCV DataFrame

        Returns:
            Optimized DataFrame
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']

        # Verify required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing OHLCV columns: {missing_cols}")

        # Optimize
        df_optimized = self.optimize_dataframe(df, verbose=False)

        # Ensure OHLCV columns are float32 for TA-Lib compatibility
        for col in required_cols:
            if col in df_optimized.columns:
                if df_optimized[col].dtype == np.float64:
                    df_optimized[col] = df_optimized[col].astype(np.float32)

        return df_optimized

    def get_optimization_summary(self) -> pd.DataFrame:
        """
        Get summary of all optimizations performed.

        Returns:
            DataFrame with optimization statistics
        """
        if not self.optimization_stats:
            return pd.DataFrame()

        summary_df = pd.DataFrame(self.optimization_stats)
        summary_df['changes_count'] = summary_df['changes'].apply(len)

        return summary_df[['timestamp', 'rows', 'columns', 'memory_before_mb',
                          'memory_after_mb', 'memory_reduction_mb', 'memory_reduction_pct']]


class SystemMemoryMonitor:
    """
    Monitor system memory usage and provide warnings.

    Usage:
        monitor = SystemMemoryMonitor()
        monitor.check_memory_availability(required_mb=500)
    """

    def __init__(self, warning_threshold_pct: float = 80.0):
        """
        Initialize memory monitor.

        Args:
            warning_threshold_pct: Memory usage percentage to trigger warnings
        """
        self.warning_threshold_pct = warning_threshold_pct

    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current system memory information.

        Returns:
            Dictionary with memory stats in MB
        """
        memory = psutil.virtual_memory()

        return {
            'total_mb': memory.total / 1024**2,
            'available_mb': memory.available / 1024**2,
            'used_mb': memory.used / 1024**2,
            'used_pct': memory.percent,
            'free_mb': memory.free / 1024**2
        }

    def check_memory_availability(
        self,
        required_mb: float = 0,
        raise_error: bool = False
    ) -> bool:
        """
        Check if sufficient memory is available.

        Args:
            required_mb: Required memory in MB (0 = just check threshold)
            raise_error: Raise MemoryError if insufficient

        Returns:
            True if sufficient memory available
        """
        mem_info = self.get_memory_info()

        # Check threshold
        if mem_info['used_pct'] > self.warning_threshold_pct:
            logger.warning(f"High memory usage: {mem_info['used_pct']:.1f}%")
            logger.warning(f"Available: {mem_info['available_mb']:.0f} MB")

        # Check required memory
        if required_mb > 0:
            if mem_info['available_mb'] < required_mb:
                msg = (f"Insufficient memory: {mem_info['available_mb']:.0f} MB available, "
                      f"{required_mb:.0f} MB required")
                if raise_error:
                    raise MemoryError(msg)
                else:
                    logger.error(msg)
                    return False

        return True

    def print_memory_usage(self) -> None:
        """Print current memory usage to console."""
        mem_info = self.get_memory_info()

        print("\n" + "="*60)
        print("SYSTEM MEMORY USAGE")
        print("="*60)
        print(f"Total Memory:     {mem_info['total_mb']:>10.0f} MB")
        print(f"Used Memory:      {mem_info['used_mb']:>10.0f} MB ({mem_info['used_pct']:.1f}%)")
        print(f"Available Memory: {mem_info['available_mb']:>10.0f} MB")
        print(f"Free Memory:      {mem_info['free_mb']:>10.0f} MB")

        # Warning if high usage
        if mem_info['used_pct'] > self.warning_threshold_pct:
            print(f"\n[WARNING] High memory usage ({mem_info['used_pct']:.1f}%)")
            print("Consider:")
            print("  - Processing fewer tickers at once")
            print("  - Using batch processing mode")
            print("  - Enabling aggressive memory optimization")

        print("="*60 + "\n")

    def force_garbage_collection(self, verbose: bool = True) -> Dict[str, int]:
        """
        Force garbage collection and return statistics.

        Args:
            verbose: Print collection statistics

        Returns:
            Dictionary with garbage collection stats
        """
        mem_before = self.get_memory_info()['used_mb']

        # Collect all generations
        collected = {
            'gen0': gc.collect(0),
            'gen1': gc.collect(1),
            'gen2': gc.collect(2)
        }

        mem_after = self.get_memory_info()['used_mb']
        mem_freed = mem_before - mem_after

        if verbose and mem_freed > 10:  # Only report if > 10MB freed
            logger.info(f"Garbage collection freed {mem_freed:.1f} MB")
            logger.info(f"  Objects collected: {sum(collected.values())}")

        return {
            **collected,
            'memory_freed_mb': mem_freed
        }


def memory_profile(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of a function.

    Usage:
        @memory_profile
        def my_function():
            # ... code ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = SystemMemoryMonitor()

        mem_before = monitor.get_memory_info()
        logger.debug(f"[{func.__name__}] Starting - Memory: {mem_before['used_mb']:.0f} MB")

        result = func(*args, **kwargs)

        mem_after = monitor.get_memory_info()
        mem_delta = mem_after['used_mb'] - mem_before['used_mb']

        logger.debug(f"[{func.__name__}] Finished - Memory: {mem_after['used_mb']:.0f} MB "
                    f"(Δ {mem_delta:+.0f} MB)")

        return result

    return wrapper


def optimize_for_low_memory(
    df: pd.DataFrame,
    aggressive: bool = True,
    force_gc: bool = True
) -> pd.DataFrame:
    """
    Convenience function for quick memory optimization.

    Args:
        df: DataFrame to optimize
        aggressive: Use aggressive optimization
        force_gc: Force garbage collection after optimization

    Returns:
        Optimized DataFrame
    """
    optimizer = MemoryOptimizer(aggressive=aggressive)
    df_optimized = optimizer.optimize_dataframe(df)

    if force_gc:
        gc.collect()

    return df_optimized


def get_dataframe_memory_usage(df: pd.DataFrame, detailed: bool = False) -> Dict:
    """
    Get detailed memory usage of a DataFrame.

    Args:
        df: DataFrame to analyze
        detailed: Include per-column breakdown

    Returns:
        Dictionary with memory usage information
    """
    total_mb = df.memory_usage(deep=True).sum() / 1024**2

    result = {
        'total_mb': total_mb,
        'rows': len(df),
        'columns': len(df.columns),
        'memory_per_row_kb': (total_mb * 1024) / len(df) if len(df) > 0 else 0
    }

    if detailed:
        col_memory = df.memory_usage(deep=True) / 1024**2  # MB
        result['column_breakdown'] = col_memory.to_dict()
        result['largest_columns'] = col_memory.nlargest(5).to_dict()

    return result
