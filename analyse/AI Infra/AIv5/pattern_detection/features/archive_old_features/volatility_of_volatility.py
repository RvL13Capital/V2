"""
Volatility of Volatility (Vol-of-Vol) Feature Extraction

This module calculates features related to the dynamics of Bollinger Band Width (BBW),
helping identify patterns with smooth, consistent volatility contraction vs choppy
volatility behavior.

Key Insight:
A steady, smooth contraction of volatility (BBW) is a stronger signal than
a choppy, erratic decline in volatility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class VolatilityOfVolatilityCalculator:
    """
    Calculate Volatility-of-Volatility features for pattern analysis.

    Focuses on BBW dynamics to identify smooth consolidations.
    """

    def __init__(
        self,
        bbw_periods: List[int] = None,
        vol_of_vol_windows: List[int] = None
    ):
        """
        Initialize calculator.

        Args:
            bbw_periods: Periods for BBW calculation
            vol_of_vol_windows: Windows for vol-of-vol metrics
        """
        self.bbw_periods = bbw_periods or [20]
        self.vol_of_vol_windows = vol_of_vol_windows or [5, 10, 20]

    def calculate_bbw(
        self,
        df: pd.DataFrame,
        period: int = 20,
        num_std: float = 2.0
    ) -> pd.Series:
        """
        Calculate Bollinger Band Width.

        Args:
            df: DataFrame with 'close' prices
            period: Moving average period
            num_std: Number of standard deviations

        Returns:
            Series with BBW values
        """
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()

        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)

        bbw = ((upper_band - lower_band) / sma) * 100

        return bbw

    def calculate_vol_of_vol_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate all volatility-of-volatility features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with vol-of-vol features added
        """
        result = df.copy()

        # Calculate BBW for different periods
        for period in self.bbw_periods:
            bbw_col = f'bbw_{period}'
            result[bbw_col] = self.calculate_bbw(result, period=period)

            # Calculate vol-of-vol metrics for each window
            for window in self.vol_of_vol_windows:
                prefix = f'bbw{period}_w{window}'

                # 1. BBW Volatility (standard deviation of BBW)
                result[f'{prefix}_volatility'] = (
                    result[bbw_col].rolling(window=window).std()
                )

                # 2. BBW Smoothness (inverse of coefficient of variation)
                bbw_mean = result[bbw_col].rolling(window=window).mean()
                bbw_std = result[bbw_col].rolling(window=window).std()
                cv = bbw_std / (bbw_mean + 1e-10)
                result[f'{prefix}_smoothness'] = 1.0 / (cv + 1.0)

                # 3. BBW Slope (linear regression slope)
                result[f'{prefix}_slope'] = (
                    result[bbw_col].rolling(window=window).apply(
                        lambda x: self._calculate_slope(x),
                        raw=False
                    )
                )

                # 4. BBW Acceleration (second derivative)
                slope_col = f'{prefix}_slope'
                result[f'{prefix}_acceleration'] = (
                    result[slope_col].diff()
                )

                # 5. BBW Consistency Score
                # Measures how many days BBW is decreasing
                bbw_decreasing = (result[bbw_col].diff() < 0).astype(int)
                result[f'{prefix}_consistency'] = (
                    bbw_decreasing.rolling(window=window).mean()
                )

                # 6. BBW Range (max - min over window)
                bbw_max = result[bbw_col].rolling(window=window).max()
                bbw_min = result[bbw_col].rolling(window=window).min()
                result[f'{prefix}_range'] = bbw_max - bbw_min

                # 7. BBW Position (current BBW vs window range)
                result[f'{prefix}_position'] = (
                    (result[bbw_col] - bbw_min) / (bbw_max - bbw_min + 1e-10)
                )

        # Calculate regime shift indicators
        result = self._calculate_regime_shifts(result)

        # Calculate contraction quality score
        result = self._calculate_contraction_quality(result)

        return result

    def _calculate_slope(self, x: pd.Series) -> float:
        """Calculate linear regression slope."""
        if len(x) < 2 or x.isna().all():
            return 0.0

        x_clean = x.dropna()
        if len(x_clean) < 2:
            return 0.0

        indices = np.arange(len(x_clean))
        slope, _ = np.polyfit(indices, x_clean.values, 1)

        return slope

    def _calculate_regime_shifts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regime shifts in BBW behavior.

        A regime shift occurs when BBW dynamics change significantly.
        """
        result = df.copy()

        for period in self.bbw_periods:
            bbw_col = f'bbw_{period}'

            if bbw_col not in result.columns:
                continue

            # Calculate rolling statistics
            rolling_mean = result[bbw_col].rolling(window=20).mean()
            rolling_std = result[bbw_col].rolling(window=20).std()

            # Z-score of BBW
            z_score = (result[bbw_col] - rolling_mean) / (rolling_std + 1e-10)

            # Regime shift detected when z-score > 2
            result[f'bbw{period}_regime_shift'] = (
                (abs(z_score) > 2.0).astype(int)
            )

            # Regime type: contraction vs expansion
            result[f'bbw{period}_regime_type'] = np.where(
                result[bbw_col] < rolling_mean,
                -1,  # Contraction regime
                1    # Expansion regime
            )

            # Days since last regime shift
            shift_mask = result[f'bbw{period}_regime_shift'] == 1
            result[f'bbw{period}_days_since_shift'] = 0

            days_counter = 0
            for idx in range(len(result)):
                if shift_mask.iloc[idx]:
                    days_counter = 0
                else:
                    days_counter += 1
                result.iloc[idx, result.columns.get_loc(f'bbw{period}_days_since_shift')] = days_counter

        return result

    def _calculate_contraction_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall contraction quality score.

        Higher score = smoother, more consistent contraction
        """
        result = df.copy()

        for period in self.bbw_periods:
            bbw_col = f'bbw_{period}'

            if bbw_col not in result.columns:
                continue

            # Components of quality score
            scores = []

            # 1. Smoothness (from consistency)
            consistency_col = f'bbw{period}_w10_consistency'
            if consistency_col in result.columns:
                scores.append(result[consistency_col])

            # 2. Steady decline (negative slope)
            slope_col = f'bbw{period}_w10_slope'
            if slope_col in result.columns:
                # Normalize slope to 0-1 range (more negative = better)
                slope_score = np.clip(-result[slope_col] * 100, 0, 1)
                scores.append(slope_score)

            # 3. Low volatility of BBW
            vol_col = f'bbw{period}_w10_volatility'
            if vol_col in result.columns:
                # Normalize: lower volatility = higher score
                max_vol = result[vol_col].quantile(0.95)
                vol_score = 1.0 - np.clip(result[vol_col] / max_vol, 0, 1)
                scores.append(vol_score)

            # 4. No recent regime shifts
            regime_col = f'bbw{period}_days_since_shift'
            if regime_col in result.columns:
                # More days without shift = better
                regime_score = np.clip(result[regime_col] / 20, 0, 1)
                scores.append(regime_score)

            # Combine scores
            if scores:
                quality_score = np.mean(scores, axis=0)
                result[f'bbw{period}_contraction_quality'] = quality_score

                # Classify quality
                result[f'bbw{period}_quality_high'] = (
                    (quality_score > 0.7).astype(int)
                )
                result[f'bbw{period}_quality_medium'] = (
                    ((quality_score > 0.5) & (quality_score <= 0.7)).astype(int)
                )
                result[f'bbw{period}_quality_low'] = (
                    (quality_score <= 0.5).astype(int)
                )

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names generated."""
        features = []

        for period in self.bbw_periods:
            features.append(f'bbw_{period}')

            for window in self.vol_of_vol_windows:
                prefix = f'bbw{period}_w{window}'
                features.extend([
                    f'{prefix}_volatility',
                    f'{prefix}_smoothness',
                    f'{prefix}_slope',
                    f'{prefix}_acceleration',
                    f'{prefix}_consistency',
                    f'{prefix}_range',
                    f'{prefix}_position',
                ])

            # Regime shift features
            features.extend([
                f'bbw{period}_regime_shift',
                f'bbw{period}_regime_type',
                f'bbw{period}_days_since_shift',
                f'bbw{period}_contraction_quality',
                f'bbw{period}_quality_high',
                f'bbw{period}_quality_medium',
                f'bbw{period}_quality_low',
            ])

        return features

    def analyze_contraction_pattern(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int
    ) -> Dict[str, float]:
        """
        Analyze the quality of a specific consolidation period.

        Args:
            df: DataFrame with BBW and vol-of-vol features
            start_idx: Start index of consolidation
            end_idx: End index of consolidation

        Returns:
            Dictionary with analysis results
        """
        period_data = df.iloc[start_idx:end_idx + 1]

        if len(period_data) < 5:
            return {'error': 'insufficient_data'}

        analysis = {}

        # Get BBW data
        bbw_col = 'bbw_20'
        if bbw_col in period_data.columns:
            bbw_values = period_data[bbw_col].values

            # Overall trend
            slope, _ = np.polyfit(np.arange(len(bbw_values)), bbw_values, 1)
            analysis['bbw_slope'] = slope
            analysis['is_contracting'] = slope < 0

            # Smoothness
            analysis['bbw_std'] = np.std(bbw_values)
            analysis['bbw_mean'] = np.mean(bbw_values)
            analysis['coefficient_of_variation'] = (
                analysis['bbw_std'] / (analysis['bbw_mean'] + 1e-10)
            )

            # Consistency
            daily_changes = np.diff(bbw_values)
            decreasing_days = (daily_changes < 0).sum()
            analysis['consistency_pct'] = decreasing_days / len(daily_changes)

            # Quality classification
            if (slope < -0.01 and
                analysis['coefficient_of_variation'] < 0.15 and
                analysis['consistency_pct'] > 0.60):
                analysis['quality'] = 'high'
            elif (slope < 0 and
                  analysis['coefficient_of_variation'] < 0.25 and
                  analysis['consistency_pct'] > 0.50):
                analysis['quality'] = 'medium'
            else:
                analysis['quality'] = 'low'

        return analysis
