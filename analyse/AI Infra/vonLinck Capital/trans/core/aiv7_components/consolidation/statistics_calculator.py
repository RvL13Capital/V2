"""
Temporal Statistics Calculator - Statistical Calculations with Caching
======================================================================

Provides statistical calculations with temporal safety and caching.
All calculations use only historical data (lookback from current point).
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import hashlib
import logging
from sklearn.linear_model import HuberRegressor

# Import performance utilities from parent module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from utils.performance import LRUCache

logger = logging.getLogger(__name__)


class TemporalStatisticsCalculator:
    """
    Calculate statistics with temporal safety and caching.

    All methods ensure no future data access and cache results
    for performance optimization.
    """

    def __init__(self, cache_size: int = 256, cache_ttl: int = 3600):
        """
        Initialize calculator with cache.

        Args:
            cache_size: Maximum cache entries
            cache_ttl: Cache time-to-live in seconds
        """
        self.slope_cache = LRUCache(
            maxsize=cache_size,
            ttl=cache_ttl,
            name="statistics_calculator"
        )

    def calculate_slope(
        self,
        series: pd.Series,
        end_idx: Optional[int] = None,
        use_robust: bool = True
    ) -> float:
        """
        Calculate linear regression slope with caching.

        Strategy:
        1. Check cache first (5-10x speedup for repeated calculations)
        2. Try fast OLS calculation first
        3. Use Huber regression only if outliers detected
        4. Cache result for future use

        Args:
            series: Time series values
            end_idx: End index (exclusive) for temporal safety
            use_robust: Use robust regression for outliers

        Returns:
            Slope (rise/run), or 0.0 if insufficient data
        """
        # Apply temporal boundary if specified
        if end_idx is not None:
            series = series[:end_idx]

        if len(series) < 2:
            return 0.0

        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return 0.0

        # Create cache key from values
        cache_key = self._make_cache_key(clean_series.values, use_robust)

        # Check cache first
        cached_slope = self.slope_cache.get(cache_key)
        if cached_slope is not None:
            return cached_slope

        # Calculate slope
        if use_robust:
            slope = self._calculate_robust_slope(clean_series.values)
        else:
            slope = self._calculate_ols_slope(clean_series.values)

        # Cache the result
        self.slope_cache.put(cache_key, slope)

        return slope

    def _calculate_ols_slope(self, values: np.ndarray) -> float:
        """
        Calculate simple OLS slope.

        Args:
            values: Clean array of values

        Returns:
            OLS slope
        """
        x = np.arange(len(values))
        y = values

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

    def _calculate_robust_slope(self, values: np.ndarray) -> float:
        """
        Calculate robust slope using Huber regression.

        Falls back to OLS if outliers not detected.

        Args:
            values: Clean array of values

        Returns:
            Robust slope
        """
        # First try OLS
        ols_slope = self._calculate_ols_slope(values)

        # Check for outliers
        x = np.arange(len(values))
        y_pred = np.mean(values) + ols_slope * (x - np.mean(x))
        residuals = values - y_pred
        residual_std = np.std(residuals)

        # If no significant outliers, use OLS
        if residual_std == 0 or not np.any(np.abs(residuals) > 2.5 * residual_std):
            return ols_slope

        # Use Huber regression for outlier resistance
        try:
            x_reshaped = x.reshape(-1, 1)
            huber = HuberRegressor(epsilon=1.35, max_iter=100)
            huber.fit(x_reshaped, values)
            return float(huber.coef_[0])
        except Exception as e:
            logger.debug(f"Huber regression failed, using OLS: {e}")
            return ols_slope

    def calculate_moving_statistics(
        self,
        series: pd.Series,
        window: int,
        end_idx: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate moving statistics over window.

        Args:
            series: Time series data
            window: Window size
            end_idx: End index for temporal boundary

        Returns:
            Dictionary with mean, std, min, max
        """
        # Apply temporal boundary
        if end_idx is not None:
            series = series[:end_idx]

        # Get window data
        if len(series) < window:
            window_data = series
        else:
            window_data = series.iloc[-window:]

        # Remove NaN values
        clean_data = window_data.dropna()

        if len(clean_data) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }

        return {
            'mean': float(clean_data.mean()),
            'std': float(clean_data.std()) if len(clean_data) > 1 else 0.0,
            'min': float(clean_data.min()),
            'max': float(clean_data.max()),
            'median': float(clean_data.median())
        }

    def calculate_compression_ratio(
        self,
        recent_value: float,
        baseline_value: float,
        min_baseline: float = 0.001
    ) -> float:
        """
        Calculate compression ratio safely.

        Args:
            recent_value: Recent period value
            baseline_value: Baseline period value
            min_baseline: Minimum baseline to avoid division issues

        Returns:
            Compression ratio (< 1.0 indicates compression)
        """
        safe_baseline = max(baseline_value, min_baseline)
        return recent_value / safe_baseline

    def calculate_percentile(
        self,
        series: pd.Series,
        value: float,
        end_idx: Optional[int] = None
    ) -> float:
        """
        Calculate percentile rank of value in series.

        Args:
            series: Historical series
            value: Value to rank
            end_idx: End index for temporal boundary

        Returns:
            Percentile rank (0-100)
        """
        # Apply temporal boundary
        if end_idx is not None:
            series = series[:end_idx]

        clean_series = series.dropna()
        if len(clean_series) == 0:
            return 50.0

        # Count values less than or equal to the given value
        rank = (clean_series <= value).sum()
        percentile = (rank / len(clean_series)) * 100

        return float(percentile)

    def calculate_volatility(
        self,
        prices: pd.Series,
        window: int = 20,
        end_idx: Optional[int] = None
    ) -> float:
        """
        Calculate price volatility (standard deviation of returns).

        Args:
            prices: Price series
            window: Window for calculation
            end_idx: End index for temporal boundary

        Returns:
            Volatility value
        """
        # Apply temporal boundary
        if end_idx is not None:
            prices = prices[:end_idx]

        if len(prices) < 2:
            return 0.0

        # Get window data
        if len(prices) < window:
            window_prices = prices
        else:
            window_prices = prices.iloc[-window:]

        # Calculate returns
        returns = window_prices.pct_change().dropna()

        if len(returns) == 0:
            return 0.0

        return float(returns.std())

    def calculate_trend_strength(
        self,
        series: pd.Series,
        end_idx: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Calculate trend strength using R-squared of linear regression.

        Args:
            series: Time series data
            end_idx: End index for temporal boundary

        Returns:
            Tuple of (slope, r_squared)
        """
        # Apply temporal boundary
        if end_idx is not None:
            series = series[:end_idx]

        clean_series = series.dropna()
        if len(clean_series) < 2:
            return 0.0, 0.0

        x = np.arange(len(clean_series))
        y = clean_series.values

        # Calculate slope and R-squared
        slope = self._calculate_ols_slope(y)

        # Calculate R-squared
        y_mean = np.mean(y)
        y_pred = y_mean + slope * (x - np.mean(x))

        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)

        if ss_tot == 0:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        return float(slope), float(r_squared)

    def _make_cache_key(
        self,
        values: np.ndarray,
        use_robust: bool
    ) -> str:
        """
        Create cache key from values and parameters.

        Args:
            values: Array values
            use_robust: Robust regression flag

        Returns:
            Hash key for caching
        """
        # Combine values and parameters
        key_data = values.tobytes() + str(use_robust).encode()
        return hashlib.md5(key_data).hexdigest()

    def clear_cache(self) -> None:
        """Clear the slope cache."""
        self.slope_cache.clear()
        logger.info("Statistics cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.slope_cache.get_stats()

    def validate_temporal_access(
        self,
        series_length: int,
        access_idx: int,
        context: str = ""
    ) -> bool:
        """
        Validate that data access is temporally safe.

        Args:
            series_length: Total length of series
            access_idx: Index being accessed
            context: Context for error message

        Returns:
            True if valid, raises error otherwise
        """
        if access_idx > series_length:
            raise ValueError(
                f"Temporal violation in {context}: "
                f"Accessing index {access_idx} in series of length {series_length}"
            )
        return True