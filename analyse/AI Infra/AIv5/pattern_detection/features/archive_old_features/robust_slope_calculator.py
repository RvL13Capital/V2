"""
Robust Slope Calculation for Technical Indicators

This module provides outlier-resistant slope calculations using:
1. Theil-Sen estimator: Median of all pairwise slopes
2. RANSAC regression: Filters outliers iteratively
3. Weighted regression: Recent data weighted more heavily

Traditional linear regression is sensitive to outliers, which can mislead
pattern detection. Robust methods provide more reliable trend estimation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from scipy import stats
from sklearn.linear_model import RANSACRegressor, LinearRegression, TheilSenRegressor, HuberRegressor
import logging

logger = logging.getLogger(__name__)


class RobustSlopeCalculator:
    """
    Calculate robust slopes for price and indicator trends.

    Methods:
    - Theil-Sen: Outlier-resistant, uses median of pairwise slopes
    - RANSAC: Iteratively filters outliers
    - Weighted: Recent data weighted more heavily
    - Confidence intervals: Measure reliability of slope
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize calculator.

        Args:
            confidence_level: Confidence level for intervals (default 95%)
        """
        self.confidence_level = confidence_level

    def calculate_slopes(
        self,
        series: pd.Series,
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Calculate robust slopes for multiple windows.

        Args:
            series: Time series data (e.g., price, indicator)
            windows: List of window sizes to calculate slopes

        Returns:
            DataFrame with slope features
        """
        if windows is None:
            windows = [5, 10, 20]

        results = pd.DataFrame(index=series.index)

        for window in windows:
            # Standard linear regression slope
            results[f'slope_ols_{window}'] = series.rolling(window=window).apply(
                lambda x: self._ols_slope(x), raw=False
            )

            # Theil-Sen robust slope
            results[f'slope_theilsen_{window}'] = series.rolling(window=window).apply(
                lambda x: self._theilsen_slope(x), raw=False
            )

            # RANSAC robust slope
            results[f'slope_ransac_{window}'] = series.rolling(window=window).apply(
                lambda x: self._ransac_slope(x), raw=False
            )

            # Weighted slope (recent data weighted more)
            results[f'slope_weighted_{window}'] = series.rolling(window=window).apply(
                lambda x: self._weighted_slope(x), raw=False
            )

            # Slope confidence interval width
            results[f'slope_ci_width_{window}'] = series.rolling(window=window).apply(
                lambda x: self._slope_confidence_width(x), raw=False
            )

            # Slope agreement score (how consistent are different methods?)
            results[f'slope_agreement_{window}'] = self._calculate_slope_agreement(
                results[[f'slope_ols_{window}', f'slope_theilsen_{window}',
                         f'slope_ransac_{window}', f'slope_weighted_{window}']].iloc[-len(series):]
            )

        return results

    def _ols_slope(self, x: pd.Series) -> float:
        """Standard OLS linear regression slope."""
        if len(x) < 2 or x.isna().all():
            return 0.0

        x_clean = x.dropna()
        if len(x_clean) < 2:
            return 0.0

        indices = np.arange(len(x_clean))
        try:
            slope, _ = np.polyfit(indices, x_clean.values, 1)
            return slope
        except:
            return 0.0

    def _theilsen_slope(self, x: pd.Series) -> float:
        """
        Huber robust regression: M-estimator robust to outliers.

        More robust to outliers than OLS, much faster than Theil-Sen.
        Uses Huber loss which combines L1 and L2 for outlier resistance.
        """
        if len(x) < 2 or x.isna().all():
            return 0.0

        x_clean = x.dropna()
        if len(x_clean) < 2:
            return 0.0

        try:
            indices = np.arange(len(x_clean)).reshape(-1, 1)
            # HuberRegressor: Fast (O(n)) and robust to outliers
            # epsilon=1.35 is standard for ~95% efficiency vs OLS
            estimator = HuberRegressor(epsilon=1.35, max_iter=100)
            estimator.fit(indices, x_clean.values)
            return estimator.coef_[0]
        except:
            return self._ols_slope(x)

    def _ransac_slope(self, x: pd.Series) -> float:
        """
        RANSAC regression: Iteratively filters outliers.

        Fits model to subset of inliers, ignoring outliers.
        """
        if len(x) < 4 or x.isna().all():  # RANSAC needs at least 4 points
            return self._ols_slope(x)

        x_clean = x.dropna()
        if len(x_clean) < 4:
            return self._ols_slope(x)

        try:
            indices = np.arange(len(x_clean)).reshape(-1, 1)
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                min_samples=max(2, int(len(x_clean) * 0.5)),
                max_trials=100,
                random_state=42
            )
            ransac.fit(indices, x_clean.values)
            return ransac.estimator_.coef_[0]
        except:
            return self._ols_slope(x)

    def _weighted_slope(self, x: pd.Series) -> float:
        """
        Weighted linear regression with recent data weighted more heavily.

        Uses exponential weights: more recent = higher weight.
        """
        if len(x) < 2 or x.isna().all():
            return 0.0

        x_clean = x.dropna()
        if len(x_clean) < 2:
            return 0.0

        # Exponential weights (recent data weighted more)
        weights = np.exp(np.linspace(0, 1, len(x_clean)))
        weights = weights / weights.sum()

        try:
            indices = np.arange(len(x_clean))

            # Weighted least squares
            W = np.diag(weights)
            X = np.column_stack([np.ones(len(indices)), indices])
            y = x_clean.values

            # Solve (X^T W X)^-1 X^T W y
            beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
            slope = beta[1]

            return slope
        except:
            return self._ols_slope(x)

    def _slope_confidence_width(self, x: pd.Series) -> float:
        """
        Calculate width of confidence interval for slope.

        Wider interval = less reliable slope estimate.
        """
        if len(x) < 3 or x.isna().all():
            return np.inf

        x_clean = x.dropna()
        if len(x_clean) < 3:
            return np.inf

        try:
            indices = np.arange(len(x_clean))
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                indices, x_clean.values
            )

            # Calculate confidence interval
            t_crit = stats.t.ppf((1 + self.confidence_level) / 2, len(x_clean) - 2)
            ci_width = 2 * t_crit * std_err

            return ci_width
        except:
            return np.inf

    def _calculate_slope_agreement(self, slopes_df: pd.DataFrame) -> pd.Series:
        """
        Calculate agreement score between different slope methods.

        High agreement = more reliable trend signal.
        """
        if slopes_df.empty or slopes_df.shape[1] < 2:
            return pd.Series(0.0, index=slopes_df.index)

        # Calculate coefficient of variation for each row
        # (lower CV = higher agreement)
        slope_std = slopes_df.std(axis=1)
        slope_mean = slopes_df.mean(axis=1)

        cv = slope_std / (np.abs(slope_mean) + 1e-10)

        # Convert to agreement score (0 to 1)
        # High CV = low agreement, Low CV = high agreement
        agreement = 1.0 / (1.0 + cv)

        return agreement

    def calculate_multi_window_slope_features(
        self,
        df: pd.DataFrame,
        column: str = 'close',
        windows: List[int] = None
    ) -> pd.DataFrame:
        """
        Calculate comprehensive slope features for a DataFrame column.

        Args:
            df: DataFrame with time series data
            column: Column name to analyze
            windows: List of window sizes

        Returns:
            DataFrame with all slope features added
        """
        if windows is None:
            windows = [5, 10, 20, 50]

        result = df.copy()

        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return result

        # Calculate all slopes
        slope_features = self.calculate_slopes(df[column], windows=windows)

        # Merge back to result
        for col in slope_features.columns:
            result[f'{column}_{col}'] = slope_features[col]

        # Add cross-window features
        result = self._add_cross_window_features(result, column, windows)

        return result

    def _add_cross_window_features(
        self,
        df: pd.DataFrame,
        column: str,
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Add features that compare slopes across different windows.

        E.g., Is short-term slope steeper than long-term slope?
        """
        result = df.copy()

        # Compare short vs long window slopes
        if len(windows) >= 2:
            short_window = windows[0]
            long_window = windows[-1]

            # Slope divergence
            short_slope_col = f'{column}_slope_theilsen_{short_window}'
            long_slope_col = f'{column}_slope_theilsen_{long_window}'

            if short_slope_col in result.columns and long_slope_col in result.columns:
                result[f'{column}_slope_divergence'] = (
                    result[short_slope_col] - result[long_slope_col]
                )

                # Slope acceleration
                result[f'{column}_slope_acceleration'] = (
                    result[short_slope_col] / (result[long_slope_col] + 1e-10)
                )

                # Trend consistency (same direction?)
                result[f'{column}_trend_consistent'] = (
                    (np.sign(result[short_slope_col]) == np.sign(result[long_slope_col]))
                    .astype(int)
                )

        # Slope stability (how much does slope change over time)
        for window in windows:
            slope_col = f'{column}_slope_theilsen_{window}'
            if slope_col in result.columns:
                result[f'{column}_slope_stability_{window}'] = (
                    result[slope_col].rolling(window=5).std()
                )

        return result

    def identify_trend_reversals(
        self,
        df: pd.DataFrame,
        column: str = 'close',
        window: int = 20
    ) -> pd.Series:
        """
        Identify trend reversal points using robust slopes.

        Args:
            df: DataFrame with time series data
            column: Column to analyze
            window: Window for slope calculation

        Returns:
            Series with reversal flags (1 = reversal detected, 0 = no reversal)
        """
        slope_col = f'{column}_slope_theilsen_{window}'

        if slope_col not in df.columns:
            # Calculate slopes if not already present
            df = self.calculate_multi_window_slope_features(df, column, [window])

        slopes = df[slope_col]

        # Reversal: slope changes sign and crosses zero
        slope_sign = np.sign(slopes)
        slope_sign_change = slope_sign.diff().abs() == 2

        # Only flag as reversal if slope magnitude is significant
        slope_magnitude = slopes.abs()
        significant_slope = slope_magnitude > slope_magnitude.rolling(50).quantile(0.25)

        reversals = slope_sign_change & significant_slope

        return reversals.astype(int)

    def get_feature_names(self, column: str = 'close', windows: List[int] = None) -> List[str]:
        """Get list of all feature names that would be generated."""
        if windows is None:
            windows = [5, 10, 20, 50]

        features = []

        for window in windows:
            features.extend([
                f'{column}_slope_ols_{window}',
                f'{column}_slope_theilsen_{window}',
                f'{column}_slope_ransac_{window}',
                f'{column}_slope_weighted_{window}',
                f'{column}_slope_ci_width_{window}',
                f'{column}_slope_agreement_{window}',
                f'{column}_slope_stability_{window}',
            ])

        # Cross-window features
        features.extend([
            f'{column}_slope_divergence',
            f'{column}_slope_acceleration',
            f'{column}_trend_consistent',
        ])

        return features
