"""
Vectorized Feature Calculator for Temporal Model

Optimized feature calculation using vectorized NumPy operations.
Replaces sequential loops with O(1) complexity operations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VectorizedFeatureCalculator:
    """Optimized feature calculation using vectorized operations

    This class provides vectorized implementations of all feature calculations
    required for the temporal model, replacing sequential loops with efficient
    NumPy operations.
    """

    def __init__(self):
        """Initialize the calculator"""
        self.min_periods = 20  # Minimum periods for rolling calculations

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all temporal features using vectorized operations

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all 10 features calculated

        Feature layout (10 total):
        - [0-4]: Market data (open, high, low, close, volume) - from input
        - [5-7]: Technical indicators (bbw_20, adx, volume_ratio_20)
        - [8-9]: Boundary slopes (added separately via add_boundaries_from_pattern)

        DISABLED (2026-01-18): Composite scores (vol_dryup_ratio, var_score, nes_score, lpf_score)
        """
        # Validate input
        self._validate_input(df)

        # Technical indicators (vectorized)
        df = self._calculate_technical_indicators(df)

        # DISABLED (2026-01-18): Composite scores removed
        # df = self._calculate_composite_scores(df)

        return df

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input dataframe has required columns"""
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using vectorized operations

        Calculates features [5-7]:
        - bbw_20: Bollinger Band Width (20-period)
        - adx: Average Directional Index
        - volume_ratio_20: Volume relative to 20-day average
        """
        # BBW (Bollinger Band Width)
        df['bbw_20'] = self._vectorized_bbw(df['close'].values)

        # ADX (Average Directional Index)
        df['adx'] = self._vectorized_adx(
            df['high'].values,
            df['low'].values,
            df['close'].values
        )

        # Volume Ratio
        df['volume_ratio_20'] = self._vectorized_volume_ratio(df['volume'].values)

        # DISABLED (2026-01-18): vol_dryup_ratio moved to composite (now disabled)
        # df['vol_dryup_ratio'] = self._vectorized_vol_dryup_ratio(df['volume'].values)

        return df

    def _calculate_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite scores using vectorized operations

        DISABLED (2026-01-18): Composite scores removed from feature set.
        This method is kept for backwards compatibility but does nothing.

        Previously calculated:
        - vol_dryup_ratio: Supply Vacuum Signal
        - var_score: Volume Accumulation Profile
        - nes_score: Energy Concentration Index
        - lpf_score: Liquidity Flow Pressure
        """
        # DISABLED - composite features removed
        return df

        # Legacy code (disabled):
        # # CCI Score - Compression Intensity
        # df['cci_score'] = self._vectorized_cci(
        #     df['high'].values,
        #     df['low'].values,
        #     df['close'].values
        # )
        #
        # # VAR Score - Volume Accumulation
        # df['var_score'] = self._vectorized_var(
        #     df['volume'].values,
        #     df['close'].values
        # )
        #
        # # NES Score - Energy Concentration
        # df['nes_score'] = self._vectorized_nes(
        #     df['close'].values,
        #     df['volume'].values
        # )
        #
        # # LPF Score - Liquidity Flow Pressure
        # df['lpf_score'] = self._vectorized_lpf(
        #     df['volume'].values,
        #     df['close'].values
        # )

        return df

    def _vectorized_bbw(self, close: np.ndarray, period: int = 20) -> np.ndarray:
        """Vectorized Bollinger Band Width calculation

        BBW = (Upper Band - Lower Band) / Middle Band

        CRITICAL FIX (2026-01-06): Removed .bfill() which caused look-ahead bias.
        Now keeps NaN for insufficient warmup periods. Sequences with NaN in BBW
        must be filtered downstream in pattern_detector.py.

        CRITICAL FIX (2026-01-31): Added closed='left' to ensure strictly historic calculation.
        With closed='left', the rolling window at index i includes data from [i-period, i),
        excluding the current observation. This prevents any forward-looking bias.

        Warmup requirement: Needs 20+ days of data for valid BBW calculation.
        Days 0-19 will have NaN values when calculated on short windows.
        """
        # Use pandas for rolling window operations
        # CRITICAL: closed='left' ensures we use only PRIOR data (excluding current day)
        close_series = pd.Series(close)
        sma = close_series.rolling(window=period, min_periods=period, closed='left').mean()
        std = close_series.rolling(window=period, min_periods=period, closed='left').std()

        # CRITICAL: DO NOT backfill - leave NaN for insufficient warmup
        # This prevents look-ahead bias where Day 0 "borrows" volatility from Day 1+
        # Sequences with NaN will be filtered downstream
        # std = std.bfill()  # REMOVED - look-ahead bias

        # Bollinger Bands
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)

        # BBW calculation with zero division protection
        bbw = np.where(
            sma != 0,
            (upper_band - lower_band) / sma,
            0
        )

        return bbw

    def _vectorized_adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Vectorized ADX calculation

        Calculates Average Directional Index using vectorized operations.
        """
        # Calculate True Range
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))

        # Stack and take maximum (vectorized)
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        tr[0] = high_low[0]  # First element doesn't have previous close

        # Calculate directional movements
        dm_plus = np.where(
            (high - np.roll(high, 1)) > (np.roll(low, 1) - low),
            np.maximum(high - np.roll(high, 1), 0),
            0
        )
        dm_minus = np.where(
            (np.roll(low, 1) - low) > (high - np.roll(high, 1)),
            np.maximum(np.roll(low, 1) - low, 0),
            0
        )

        # Smooth using exponential moving average
        tr_smooth = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False).mean().values

        # Calculate DI+ and DI-
        di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
        di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)

        # Calculate DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values

        return adx

    def _vectorized_volume_ratio(
        self,
        volume: np.ndarray,
        period: int = 20
    ) -> np.ndarray:
        """Vectorized volume ratio calculation

        Volume Ratio = Current Volume / Average Volume (of PRIOR period)

        CRITICAL FIX (2026-01-31): Added closed='left' to ensure strictly historic calculation.
        With closed='left', average volume at index i is computed from [i-period, i),
        excluding the current observation. This prevents forward-looking bias.
        """
        volume_series = pd.Series(volume)
        # CRITICAL: closed='left' ensures we compare against PRIOR average only
        avg_volume = volume_series.rolling(window=period, min_periods=period, closed='left').mean()

        # Avoid division by zero
        volume_ratio = np.where(
            avg_volume > 0,
            volume / avg_volume,
            1.0
        )

        return volume_ratio

    def _vectorized_vol_dryup_ratio(
        self,
        volume: np.ndarray,
        short_period: int = 3,
        long_period: int = 20
    ) -> np.ndarray:
        """Vectorized Volume Dry-Up Ratio calculation

        Formula: Mean(Volume_Last_3_Days) / Mean(Volume_Last_20_Days)

        This indicator detects volume contraction - a classic precursor to breakouts.

        CRITICAL FIX (2026-01-31): Added closed='left' to ensure strictly historic calculation.
        Both short and long period means are calculated using only PRIOR data,
        excluding the current observation. This prevents forward-looking bias.

        Signal Interpretation:
        - Ratio < 0.30: Volume is drying up (only 30% of average) → IMMINENT MOVE
        - Ratio < 0.40: Volume contraction beginning
        - Ratio 0.40-0.80: Normal range
        - Ratio > 0.80: Volume stable or increasing

        Trading Logic:
        When Vol_DryUp_Ratio < 0.3 AND Attention_Score (Days 15-19) is High
        → IMMINENT MOVE (breakout or breakdown expected)

        Why This Works:
        - Volume dry-up in consolidation = institutions accumulated, retail lost interest
        - Low volume makes it easier for smart money to push price
        - When volume returns, explosive moves happen

        Args:
            volume: Volume array
            short_period: Recent volume window (default: 3 days)
            long_period: Baseline volume window (default: 20 days)

        Returns:
            Array of vol_dryup_ratio values (0.0 to 2.0+ range)
        """
        volume_series = pd.Series(volume)

        # Calculate rolling means with closed='left' for strictly historic calculation
        # CRITICAL: closed='left' excludes current observation from window
        vol_short = volume_series.rolling(window=short_period, min_periods=short_period, closed='left').mean()
        vol_long = volume_series.rolling(window=long_period, min_periods=long_period, closed='left').mean()

        # Calculate ratio with zero division protection
        vol_dryup_ratio = np.where(
            vol_long > 0,
            vol_short / vol_long,
            1.0  # Neutral value when no baseline
        )

        return np.asarray(vol_dryup_ratio)

    def _vectorized_cci(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 20
    ) -> np.ndarray:
        """Vectorized CCI (Commodity Channel Index) calculation

        CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
        """
        # Typical Price
        typical_price = (high + low + close) / 3

        # Simple Moving Average
        tp_series = pd.Series(typical_price)
        sma = tp_series.rolling(window=period, min_periods=1).mean()

        # Mean Absolute Deviation
        mad = tp_series.rolling(window=period, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))),
            raw=True
        )

        # CCI calculation
        cci = (typical_price - sma) / (0.015 * mad + 1e-10)

        # Normalize to score between -1 and 1
        cci_score = np.tanh(cci / 100)

        return np.asarray(cci_score)

    def _vectorized_var(
        self,
        volume: np.ndarray,
        close: np.ndarray,
        period: int = 20
    ) -> np.ndarray:
        """Vectorized VAR (Volume Accumulation Rate) score

        Measures the rate of volume accumulation relative to price movement.

        CRITICAL FIX (2026-01-06): Removed .bfill() which caused look-ahead bias.
        Now keeps NaN for insufficient warmup periods. Sequences with NaN in VAR
        must be filtered downstream in pattern_detector.py.
        """
        # Calculate volume-weighted price
        vwap = (volume * close).cumsum() / volume.cumsum()

        # Volume accumulation
        volume_series = pd.Series(volume)
        volume_ma = volume_series.rolling(window=period, min_periods=1).mean()
        # CRITICAL: DO NOT backfill - leave NaN for insufficient warmup
        volume_std = volume_series.rolling(window=period, min_periods=2).std()
        # volume_std = volume_std.bfill().fillna(0.0)  # REMOVED - look-ahead bias

        # Normalized volume
        volume_zscore = np.where(
            volume_std > 0,
            (volume - volume_ma) / volume_std,
            0
        )

        # Price momentum
        close_series = pd.Series(close)
        price_change = close_series.pct_change(periods=period).fillna(0)

        # VAR score combines volume and price signals
        var_score = np.tanh(volume_zscore * (1 + price_change))

        return np.asarray(var_score)

    def _vectorized_nes(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 10
    ) -> np.ndarray:
        """Vectorized NES (Normalized Energy Score)

        Measures the concentration of price energy (volatility * volume).
        """
        # Calculate returns
        close_series = pd.Series(close)
        returns = close_series.pct_change().fillna(0)

        # Energy = squared returns * volume (vectorized)
        energy = returns.values ** 2 * volume

        # Normalize energy
        energy_series = pd.Series(energy)
        energy_ma = energy_series.rolling(window=period, min_periods=1).mean()
        energy_std = energy_series.rolling(window=period, min_periods=1).std()

        # NES score
        nes_score = np.where(
            energy_std > 0,
            (energy - energy_ma) / energy_std,
            0
        )

        # Bound between -1 and 1
        nes_score = np.tanh(nes_score)

        return nes_score

    def _vectorized_lpf(
        self,
        volume: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Vectorized LPF (Liquidity Flow Pressure) score

        Measures the pressure of liquidity flow based on volume and price direction.
        """
        # Price direction
        close_series = pd.Series(close)
        price_change = close_series.diff().fillna(0)
        price_direction = np.sign(price_change)

        # Directional volume
        directional_volume = volume * price_direction

        # Cumulative flow
        cumulative_flow = pd.Series(directional_volume).rolling(
            window=period,
            min_periods=1
        ).sum()

        # Total volume
        total_volume = pd.Series(volume).rolling(
            window=period,
            min_periods=1
        ).sum()

        # LPF score
        lpf_score = np.where(
            total_volume > 0,
            cumulative_flow / total_volume,
            0
        )

        return lpf_score

    def add_boundaries_from_pattern(
        self,
        df: pd.DataFrame,
        upper_boundary: float,
        lower_boundary: float
    ) -> pd.DataFrame:
        """Add dynamic boundary convergence features to dataframe

        GEOMETRY FIX (2026-01-18): Replaced static box coordinates with
        dynamic convergence slopes. Model now sees Triangle, not Rectangle.

        New features (indices 12-13):
        - upper_slope: Rolling regression slope of highs, normalized by box height
        - lower_slope: Rolling regression slope of lows, normalized by box height

        Triangle Geometry Interpretation:
        - Symmetric triangle: upper_slope < 0, lower_slope > 0 (converging)
        - Ascending triangle: upper_slope ≈ 0, lower_slope > 0
        - Descending triangle: upper_slope < 0, lower_slope ≈ 0
        - Rectangle/box: both ≈ 0 (parallel boundaries)
        - Wedge: both same sign (diverging or parallel drift)

        Args:
            df: DataFrame with OHLCV features
            upper_boundary: Original upper boundary (for normalization reference)
            lower_boundary: Original lower boundary (for normalization reference)

        Returns:
            DataFrame with upper_slope and lower_slope columns
        """
        box_height = upper_boundary - lower_boundary
        if box_height <= 0:
            box_height = 1e-8  # Prevent division by zero

        # Calculate dynamic boundary slopes
        upper_slope, lower_slope = self._calculate_boundary_slopes(
            high=df['high'].values,
            low=df['low'].values,
            box_height=box_height,
            lookback=5  # 5-day rolling regression
        )

        df['upper_boundary'] = upper_slope
        df['lower_boundary'] = lower_slope
        return df

    def _calculate_boundary_slopes(
        self,
        high: np.ndarray,
        low: np.ndarray,
        box_height: float,
        lookback: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate rolling linear regression slopes for boundary convergence

        Uses OLS regression on rolling windows to measure boundary direction:
        - Positive slope = boundary rising
        - Negative slope = boundary falling
        - Near zero = horizontal boundary

        Slopes are normalized by box_height / lookback to make them:
        1. Scale-invariant (comparable across price levels)
        2. Interpretable (slope of ~1.0 = boundary moving full box height in lookback days)

        Args:
            high: Array of high prices
            low: Array of low prices
            box_height: Original consolidation box height for normalization
            lookback: Rolling window size for regression (default: 5 days)

        Returns:
            Tuple of (upper_slope, lower_slope) arrays with same length as input
        """
        n = len(high)
        upper_slope = np.zeros(n)
        lower_slope = np.zeros(n)

        # Normalization factor: slope per day relative to box height
        # A slope of 1.0 means boundary moves full box height in `lookback` days
        norm_factor = box_height / lookback

        # Precompute regression coefficients for efficiency
        # For y = mx + b, m = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
        x = np.arange(lookback, dtype=np.float64)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        for t in range(n):
            if t < lookback - 1:
                # Not enough history - use available data
                window_size = t + 1
                if window_size < 2:
                    # Single point - no slope
                    upper_slope[t] = 0.0
                    lower_slope[t] = 0.0
                    continue

                # Shorter window regression
                x_short = np.arange(window_size, dtype=np.float64)
                x_mean_short = x_short.mean()
                x_var_short = ((x_short - x_mean_short) ** 2).sum()

                high_window = high[0:t+1]
                low_window = low[0:t+1]

                high_mean = high_window.mean()
                low_mean = low_window.mean()

                if x_var_short > 0:
                    upper_slope[t] = np.sum((x_short - x_mean_short) * (high_window - high_mean)) / x_var_short
                    lower_slope[t] = np.sum((x_short - x_mean_short) * (low_window - low_mean)) / x_var_short
                else:
                    upper_slope[t] = 0.0
                    lower_slope[t] = 0.0
            else:
                # Full lookback window
                high_window = high[t-lookback+1:t+1]
                low_window = low[t-lookback+1:t+1]

                high_mean = high_window.mean()
                low_mean = low_window.mean()

                if x_var > 0:
                    upper_slope[t] = np.sum((x - x_mean) * (high_window - high_mean)) / x_var
                    lower_slope[t] = np.sum((x - x_mean) * (low_window - low_mean)) / x_var
                else:
                    upper_slope[t] = 0.0
                    lower_slope[t] = 0.0

        # Normalize by box height to make scale-invariant
        if norm_factor > 0:
            upper_slope = upper_slope / norm_factor
            lower_slope = lower_slope / norm_factor

        # Clip extreme values for numerical stability
        # Range [-3, 3] covers slopes up to 3x box height movement in lookback period
        upper_slope = np.clip(upper_slope, -3.0, 3.0)
        lower_slope = np.clip(lower_slope, -3.0, 3.0)

        return upper_slope, lower_slope

    def normalize_features(
        self,
        features: np.ndarray,
        method: str = 'standardize'
    ) -> Tuple[np.ndarray, dict]:
        """Normalize features using vectorized operations

        Args:
            features: Feature array to normalize
            method: 'standardize' for z-score, 'minmax' for min-max scaling

        Returns:
            Normalized features and normalization parameters
        """
        if method == 'standardize':
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std = np.where(std == 0, 1, std)  # Avoid division by zero
            normalized = (features - mean) / std
            params = {'mean': mean, 'std': std}

        elif method == 'minmax':
            min_val = np.min(features, axis=0)
            max_val = np.max(features, axis=0)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (features - min_val) / range_val
            params = {'min': min_val, 'max': max_val}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized, params

    # =========================================================================
    # DOLLAR BAR SPECIFIC FEATURES
    # =========================================================================

    def calculate_dollar_bar_features(
        self,
        df: pd.DataFrame,
        threshold: float
    ) -> pd.DataFrame:
        """Calculate features specific to dollar bars

        Dollar bars have different characteristics than time bars:
        - Variable duration (some bars span hours, others days)
        - Consistent dollar volume (by construction)
        - Activity intensity varies with market conditions

        Args:
            df: DataFrame with dollar bar data. Expected columns:
                - dollar_volume: Dollar volume for this bar
                - bar_duration_hours: Time span of bar (optional)
                - num_days: Number of calendar days in bar (optional)
            threshold: Dollar volume threshold used to create bars

        Returns:
            DataFrame with added dollar bar features:
            - bars_per_day: Activity intensity (1 / num_days)
            - time_between_bars: log1p(bar_duration_hours)
            - volume_vs_threshold: dollar_volume / threshold

        Note:
            For dollar bars, the standard volume_ratio_20 feature is
            meaningless (volume is ~constant by construction). These
            features replace it with activity-based metrics.
        """
        if 'dollar_volume' not in df.columns:
            logger.warning("dollar_volume column missing - cannot calculate dollar bar features")
            return df

        df = df.copy()

        # bars_per_day: Activity intensity
        # High value = high activity (multiple bars per day)
        # Low value = low activity (bar spans multiple days)
        if 'num_days' in df.columns:
            df['bars_per_day'] = 1.0 / df['num_days'].replace(0, 1)
        else:
            # Estimate from duration if num_days not available
            if 'bar_duration_hours' in df.columns:
                df['bars_per_day'] = 24.0 / df['bar_duration_hours'].replace(0, 24)
            else:
                df['bars_per_day'] = 1.0  # Default: 1 bar per day

        # time_between_bars: Dormancy detection (log scale)
        # Low value = fast-paced trading
        # High value = dormant periods (long gaps between bars)
        if 'bar_duration_hours' in df.columns:
            df['time_between_bars'] = np.log1p(df['bar_duration_hours'])
        else:
            df['time_between_bars'] = 0.0  # Default: no dormancy signal

        # volume_vs_threshold: Volume deviation from expected
        # ~1.0 = bar closed right at threshold
        # >1.0 = bar had extra volume (gap open, etc.)
        # <1.0 = partial bar (shouldn't happen normally)
        if threshold > 0:
            df['volume_vs_threshold'] = df['dollar_volume'] / threshold
        else:
            df['volume_vs_threshold'] = 1.0

        return df

    def adapt_bbw_for_dollar_bars(
        self,
        close: np.ndarray,
        bars_per_day: np.ndarray,
        period: int = 20
    ) -> np.ndarray:
        """Adapted BBW calculation for dollar bars

        Standard BBW uses fixed time periods, but dollar bars have variable
        duration. This adaptation weights the calculation by activity level.

        For high-activity periods (many bars/day), BBW responds faster.
        For low-activity periods, BBW uses longer effective lookback.

        Args:
            close: Close prices for dollar bars
            bars_per_day: Activity intensity for each bar
            period: Base period for BBW (default: 20 bars)

        Returns:
            BBW values adjusted for activity level
        """
        # Calculate standard BBW
        standard_bbw = self._vectorized_bbw(close, period)

        # Weight by activity: faster response during active periods
        # When bars_per_day > 1, we see more bars than calendar days
        # When bars_per_day < 1, each bar spans multiple days
        activity_weight = np.clip(bars_per_day, 0.2, 5.0)

        # Blend with activity-adjusted version
        # High activity = use shorter period (more responsive)
        # Low activity = use longer period (more stable)
        close_series = pd.Series(close)

        # Calculate multi-period BBWs
        bbw_short = self._vectorized_bbw(close, max(5, period // 2))
        bbw_long = self._vectorized_bbw(close, min(40, period * 2))

        # Blend based on activity
        # activity_weight > 1: favor short-period BBW
        # activity_weight < 1: favor long-period BBW
        normalized_weight = (activity_weight - 0.2) / (5.0 - 0.2)  # 0 to 1
        adapted_bbw = bbw_long + normalized_weight * (bbw_short - bbw_long)

        return np.asarray(adapted_bbw)

    def adapt_volume_ratio_for_dollar_bars(
        self,
        dollar_volume: np.ndarray,
        bars_per_day: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """Adapted volume ratio for dollar bars

        Traditional volume_ratio_20 is meaningless for dollar bars since
        volume is constant by construction. Instead, we measure:
        - Activity relative to historical activity
        - Volume deviation from threshold

        Args:
            dollar_volume: Dollar volume for each bar
            bars_per_day: Activity intensity for each bar
            threshold: Dollar volume threshold

        Returns:
            Activity-adjusted volume signal
        """
        # Calculate rolling activity ratio
        bars_per_day_series = pd.Series(bars_per_day)
        avg_activity = bars_per_day_series.rolling(window=20, min_periods=5).mean()

        # Activity ratio: current activity vs recent average
        activity_ratio = np.where(
            avg_activity > 0,
            bars_per_day / avg_activity,
            1.0
        )

        # Volume deviation from threshold
        if threshold > 0:
            volume_deviation = dollar_volume / threshold
        else:
            volume_deviation = np.ones_like(dollar_volume)

        # Combined signal: activity * deviation
        # High activity + high volume = strong signal
        combined_ratio = np.clip(activity_ratio * volume_deviation, 0.1, 10.0)

        return np.asarray(combined_ratio)