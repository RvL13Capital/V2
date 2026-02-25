"""
Temporal Feature Extractor - Lookback-Only Feature Engineering
==============================================================

Extracts technical features with strict temporal constraints.
Only accesses historical data (lookback from current point).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureWindows:
    """Configuration for feature extraction windows."""
    recent_window: int = 20  # Last 20 days
    baseline_start: int = 50  # 50 days ago
    baseline_end: int = 20  # to 20 days ago


class TemporalFeatureExtractor:
    """
    Extract features with temporal safety guarantees.

    All methods enforce lookback-only data access to prevent
    future data leakage in pattern detection and training.
    """

    def __init__(self, windows: Optional[FeatureWindows] = None):
        """
        Initialize feature extractor.

        Args:
            windows: Window configuration for feature extraction
        """
        self.windows = windows or FeatureWindows()

    def extract_features_at_point(
        self,
        df: pd.DataFrame,
        current_idx: int,
        pattern_start_idx: int
    ) -> Dict[str, Any]:
        """
        Extract all features at specific point in time.

        CRITICAL: Only looks BACKWARD from current_idx.
        - Recent window: [current_idx-20 : current_idx]
        - Baseline window: [current_idx-50 : current_idx-20]
        - NEVER accesses data > current_idx

        Args:
            df: Full DataFrame (but only historical portion used)
            current_idx: Current position in data (present)
            pattern_start_idx: When pattern started (for duration features)

        Returns:
            Dictionary of extracted features
        """
        # Temporal safety check
        if current_idx >= len(df):
            raise ValueError(
                f"current_idx {current_idx} exceeds data length {len(df)}. "
                "Cannot look into future."
            )

        # Extract different feature groups
        features = {}

        # 1. Recent window features (last 20 days)
        recent_features = self._extract_recent_features(df, current_idx)
        features.update(recent_features)

        # 2. Baseline window features (50-20 days ago)
        baseline_features = self._extract_baseline_features(df, current_idx)
        features.update(baseline_features)

        # 3. Compression metrics (recent vs baseline)
        compression_features = self._calculate_compression_metrics(
            recent_features,
            baseline_features
        )
        features.update(compression_features)

        # 4. Pattern duration features
        duration_features = self._extract_duration_features(
            current_idx,
            pattern_start_idx
        )
        features.update(duration_features)

        # 5. Price position features (if boundaries available)
        # These will be added by the boundary manager

        return features

    def _extract_recent_features(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> Dict[str, float]:
        """
        Extract features from recent window (last 20 days).

        Args:
            df: Price data DataFrame
            current_idx: Current position

        Returns:
            Recent window features
        """
        # Define window boundaries (LOOKBACK ONLY)
        window_start = max(0, current_idx - self.windows.recent_window + 1)
        window_end = current_idx + 1  # Inclusive of current

        # Slice historical data only
        window_data = df.iloc[window_start:window_end]

        if len(window_data) < 2:
            # Not enough data for features
            return self._empty_recent_features()

        features = {}

        # BBW features (if available)
        if 'bbw_20' in window_data.columns:
            bbw_series = window_data['bbw_20'].dropna()
            if len(bbw_series) > 0:
                features['avg_bbw_20d'] = float(bbw_series.mean())
                features['bbw_std_20d'] = float(bbw_series.std())
                features['bbw_slope_20d'] = self._calculate_slope(bbw_series)
            else:
                features['avg_bbw_20d'] = 0.0
                features['bbw_std_20d'] = 0.0
                features['bbw_slope_20d'] = 0.0

        # ADX features
        if 'adx' in window_data.columns:
            adx_series = window_data['adx'].dropna()
            if len(adx_series) > 0:
                features['avg_adx_20d'] = float(adx_series.mean())
                features['adx_slope_20d'] = self._calculate_slope(adx_series)
            else:
                features['avg_adx_20d'] = 0.0
                features['adx_slope_20d'] = 0.0

        # Volume features
        if 'volume_ratio_20' in window_data.columns:
            vol_series = window_data['volume_ratio_20'].dropna()
            if len(vol_series) > 0:
                features['avg_volume_ratio_20d'] = float(vol_series.mean())
                features['volume_std_20d'] = float(vol_series.std())
                features['volume_slope_20d'] = self._calculate_slope(vol_series)
            else:
                features['avg_volume_ratio_20d'] = 0.0
                features['volume_std_20d'] = 0.0
                features['volume_slope_20d'] = 0.0

        # Range features
        if 'range_ratio_20' in window_data.columns:
            range_series = window_data['range_ratio_20'].dropna()
            if len(range_series) > 0:
                features['avg_range_ratio_20d'] = float(range_series.mean())
                features['range_slope_20d'] = self._calculate_slope(range_series)
            else:
                features['avg_range_ratio_20d'] = 0.0
                features['range_slope_20d'] = 0.0

        # Price volatility
        if 'close' in window_data.columns:
            close_series = window_data['close'].dropna()
            if len(close_series) > 1:
                returns = close_series.pct_change().dropna()
                features['price_volatility_20d'] = float(returns.std())
            else:
                features['price_volatility_20d'] = 0.0

        return features

    def _extract_baseline_features(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> Dict[str, float]:
        """
        Extract features from baseline window (50-20 days ago).

        Used for comparison to detect compression/expansion.

        Args:
            df: Price data DataFrame
            current_idx: Current position

        Returns:
            Baseline window features
        """
        # Define baseline window (HISTORICAL ONLY)
        baseline_start = max(0, current_idx - self.windows.baseline_start + 1)
        baseline_end = max(0, current_idx - self.windows.baseline_end + 1)

        if baseline_end <= baseline_start:
            return self._empty_baseline_features()

        # Slice historical baseline data
        baseline_data = df.iloc[baseline_start:baseline_end]

        if len(baseline_data) < 2:
            return self._empty_baseline_features()

        features = {}

        # BBW baseline
        if 'bbw_20' in baseline_data.columns:
            bbw_baseline = baseline_data['bbw_20'].dropna()
            if len(bbw_baseline) > 0:
                features['baseline_bbw_avg'] = float(bbw_baseline.mean())
                features['baseline_bbw_std'] = float(bbw_baseline.std())
            else:
                features['baseline_bbw_avg'] = 0.0
                features['baseline_bbw_std'] = 0.0

        # ADX baseline
        if 'adx' in baseline_data.columns:
            adx_baseline = baseline_data['adx'].dropna()
            if len(adx_baseline) > 0:
                features['baseline_adx_avg'] = float(adx_baseline.mean())
            else:
                features['baseline_adx_avg'] = 0.0

        # Volume baseline
        if 'volume_ratio_20' in baseline_data.columns:
            vol_baseline = baseline_data['volume_ratio_20'].dropna()
            if len(vol_baseline) > 0:
                features['baseline_volume_avg'] = float(vol_baseline.mean())
                features['baseline_volume_std'] = float(vol_baseline.std())
            else:
                features['baseline_volume_avg'] = 0.0
                features['baseline_volume_std'] = 0.0

        # Range baseline
        if 'range_ratio_20' in baseline_data.columns:
            range_baseline = baseline_data['range_ratio_20'].dropna()
            if len(range_baseline) > 0:
                features['baseline_range_avg'] = float(range_baseline.mean())
            else:
                features['baseline_range_avg'] = 0.0

        # Baseline volatility
        if 'close' in baseline_data.columns:
            close_baseline = baseline_data['close'].dropna()
            if len(close_baseline) > 1:
                baseline_returns = close_baseline.pct_change().dropna()
                features['baseline_volatility'] = float(baseline_returns.std())
            else:
                features['baseline_volatility'] = 0.0

        return features

    def _calculate_compression_metrics(
        self,
        recent_features: Dict[str, float],
        baseline_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate compression ratios (recent vs baseline).

        Compression < 1.0 indicates contraction.

        Args:
            recent_features: Features from recent window
            baseline_features: Features from baseline window

        Returns:
            Compression metric features
        """
        features = {}

        # BBW compression
        if 'avg_bbw_20d' in recent_features and 'baseline_bbw_avg' in baseline_features:
            baseline_bbw = baseline_features.get('baseline_bbw_avg', 1.0)
            if baseline_bbw > 0:
                features['bbw_compression_ratio'] = (
                    recent_features['avg_bbw_20d'] / baseline_bbw
                )
            else:
                features['bbw_compression_ratio'] = 1.0

        # Volume compression
        if 'avg_volume_ratio_20d' in recent_features and 'baseline_volume_avg' in baseline_features:
            baseline_vol = baseline_features.get('baseline_volume_avg', 1.0)
            if baseline_vol > 0:
                features['volume_compression_ratio'] = (
                    recent_features['avg_volume_ratio_20d'] / baseline_vol
                )
            else:
                features['volume_compression_ratio'] = 1.0

        # Range compression
        if 'avg_range_ratio_20d' in recent_features and 'baseline_range_avg' in baseline_features:
            baseline_range = baseline_features.get('baseline_range_avg', 1.0)
            if baseline_range > 0:
                features['range_compression_ratio'] = (
                    recent_features['avg_range_ratio_20d'] / baseline_range
                )
            else:
                features['range_compression_ratio'] = 1.0

        # Volatility stability
        if 'price_volatility_20d' in recent_features and 'baseline_volatility' in baseline_features:
            baseline_vol = baseline_features.get('baseline_volatility', 1.0)
            if baseline_vol > 0:
                features['volatility_stability_ratio'] = (
                    recent_features['price_volatility_20d'] / baseline_vol
                )
            else:
                features['volatility_stability_ratio'] = 1.0

        # Overall compression (geometric mean of ratios)
        ratios = [
            features.get('bbw_compression_ratio', 1.0),
            features.get('volume_compression_ratio', 1.0),
            features.get('range_compression_ratio', 1.0),
            features.get('volatility_stability_ratio', 1.0)
        ]
        features['overall_compression'] = float(np.prod(ratios) ** (1.0 / len(ratios)))

        return features

    def _extract_duration_features(
        self,
        current_idx: int,
        pattern_start_idx: int
    ) -> Dict[str, int]:
        """
        Extract pattern duration features.

        Args:
            current_idx: Current position
            pattern_start_idx: Pattern start position

        Returns:
            Duration-based features
        """
        return {
            'pattern_duration_days': current_idx - pattern_start_idx,
            'pattern_start_idx': pattern_start_idx,
            'current_idx': current_idx
        }

    def _calculate_slope(self, series: pd.Series) -> float:
        """
        Calculate simple linear regression slope.

        Args:
            series: Time series values

        Returns:
            Slope value
        """
        if len(series) < 2:
            return 0.0

        x = np.arange(len(series))
        y = series.values

        # Simple OLS slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)

    def _empty_recent_features(self) -> Dict[str, float]:
        """Return empty recent features with zero values."""
        return {
            'avg_bbw_20d': 0.0,
            'bbw_std_20d': 0.0,
            'bbw_slope_20d': 0.0,
            'avg_adx_20d': 0.0,
            'adx_slope_20d': 0.0,
            'avg_volume_ratio_20d': 0.0,
            'volume_std_20d': 0.0,
            'volume_slope_20d': 0.0,
            'avg_range_ratio_20d': 0.0,
            'range_slope_20d': 0.0,
            'price_volatility_20d': 0.0
        }

    def _empty_baseline_features(self) -> Dict[str, float]:
        """Return empty baseline features with zero values."""
        return {
            'baseline_bbw_avg': 0.0,
            'baseline_bbw_std': 0.0,
            'baseline_adx_avg': 0.0,
            'baseline_volume_avg': 0.0,
            'baseline_volume_std': 0.0,
            'baseline_range_avg': 0.0,
            'baseline_volatility': 0.0
        }

    def validate_temporal_integrity(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> bool:
        """
        Validate that feature extraction maintains temporal integrity.

        Ensures no future data access occurs.

        Args:
            df: Data DataFrame
            current_idx: Current index position

        Returns:
            True if temporally valid, False otherwise
        """
        # Check that current_idx doesn't exceed data bounds
        if current_idx >= len(df):
            logger.error(
                f"Temporal violation: current_idx {current_idx} "
                f">= data length {len(df)}"
            )
            return False

        # Check that windows don't extend into future
        window_start = max(0, current_idx - self.windows.baseline_start + 1)
        if window_start > current_idx:
            logger.error(
                f"Temporal violation: window_start {window_start} "
                f"> current_idx {current_idx}"
            )
            return False

        return True