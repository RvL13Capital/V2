"""
Microstructural Price Dynamics Features

Extracts fine-grained price patterns from daily OHLC bars that reveal
intrabar price behavior and market microstructure dynamics.

While true microstructure analysis requires tick data, daily OHLC bars
contain rich information about:
- Intrabar price movements (shadows, bodies, gaps)
- Price efficiency and directional strength
- Volume-weighted metrics
- Order flow proxies

These features are particularly valuable for consolidation pattern analysis,
as they reveal accumulation/distribution and breakout precursors.

Feature Categories:
1. OHLC Pattern Features (candle patterns)
2. Intrabar Momentum & Reversals
3. Volume-Weighted Metrics
4. Price Efficiency Measures
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MicrostructuralFeatureCalculator:
    """
    Calculate microstructural features from OHLC data.

    These features capture fine-grained price dynamics that occur within
    each daily bar, providing insights into market participant behavior.
    """

    def __init__(self):
        """Initialize calculator."""
        pass

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all microstructural features.

        Args:
            df: DataFrame with OHLC columns (open, high, low, close, volume)

        Returns:
            DataFrame with all microstructural features added
        """
        result = df.copy()

        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return result

        # Category 1: OHLC Pattern Features
        result = self._add_ohlc_pattern_features(result)

        # Category 2: Intrabar Momentum
        result = self._add_intrabar_momentum_features(result)

        # Category 3: Volume-Weighted Metrics
        result = self._add_volume_weighted_features(result)

        # Category 4: Price Efficiency
        result = self._add_price_efficiency_features(result)

        logger.debug(f"Added {len([c for c in result.columns if c not in df.columns])} microstructural features")

        return result

    def _add_ohlc_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add OHLC candle pattern features.

        These features describe the shape and characteristics of each daily bar.
        """
        result = df.copy()

        # Basic bar measurements
        result['bar_range'] = result['high'] - result['low']
        result['bar_body'] = result['close'] - result['open']
        result['bar_body_abs'] = result['bar_body'].abs()

        # Avoid division by zero
        bar_range_safe = result['bar_range'].replace(0, np.nan)

        # Body-to-range ratio (0 = doji, 1 = marubozu)
        result['body_to_range_ratio'] = result['bar_body_abs'] / bar_range_safe

        # Shadow measurements
        result['upper_shadow'] = result['high'] - result[['open', 'close']].max(axis=1)
        result['lower_shadow'] = result[['open', 'close']].min(axis=1) - result['low']

        # Shadow ratios
        result['upper_shadow_ratio'] = result['upper_shadow'] / bar_range_safe
        result['lower_shadow_ratio'] = result['lower_shadow'] / bar_range_safe

        # Shadow balance (positive = upper > lower)
        result['shadow_balance'] = (result['upper_shadow'] - result['lower_shadow']) / bar_range_safe

        # Price position within bar (0 = at low, 1 = at high)
        result['close_position_in_range'] = (
            (result['close'] - result['low']) / bar_range_safe
        )
        result['open_position_in_range'] = (
            (result['open'] - result['low']) / bar_range_safe
        )

        # Candle type classification
        result['is_bullish_candle'] = (result['close'] > result['open']).astype(int)
        result['is_bearish_candle'] = (result['close'] < result['open']).astype(int)
        result['is_doji'] = (result['body_to_range_ratio'] < 0.1).astype(int)

        # Hammer/shooting star patterns
        # Hammer: small body at top, long lower shadow
        result['is_hammer_like'] = (
            (result['body_to_range_ratio'] < 0.3) &
            (result['lower_shadow_ratio'] > 0.6) &
            (result['close_position_in_range'] > 0.7)
        ).astype(int)

        # Shooting star: small body at bottom, long upper shadow
        result['is_shooting_star_like'] = (
            (result['body_to_range_ratio'] < 0.3) &
            (result['upper_shadow_ratio'] > 0.6) &
            (result['close_position_in_range'] < 0.3)
        ).astype(int)

        # Engulfing patterns (requires previous bar)
        prev_body = result['bar_body'].shift(1)
        prev_body_abs = result['bar_body_abs'].shift(1)

        result['is_bullish_engulfing'] = (
            (result['bar_body'] > 0) &  # Current bullish
            (prev_body < 0) &  # Previous bearish
            (result['bar_body_abs'] > prev_body_abs * 1.2)  # Current body 20% larger
        ).astype(int)

        result['is_bearish_engulfing'] = (
            (result['bar_body'] < 0) &  # Current bearish
            (prev_body > 0) &  # Previous bullish
            (result['bar_body_abs'] > prev_body_abs * 1.2)
        ).astype(int)

        # Range expansion/contraction
        avg_range_20 = result['bar_range'].rolling(20).mean()
        result['range_expansion_ratio'] = result['bar_range'] / avg_range_20

        # Consecutive candle patterns
        result['consecutive_bullish_candles'] = self._count_consecutive(
            result['is_bullish_candle'], window=5
        )
        result['consecutive_bearish_candles'] = self._count_consecutive(
            result['is_bearish_candle'], window=5
        )

        return result

    def _add_intrabar_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add intrabar momentum and reversal features.

        These features capture the dynamics of price movement within each bar.
        """
        result = df.copy()

        prev_close = result['close'].shift(1)

        # Opening gap (gap between previous close and current open)
        result['opening_gap'] = (result['open'] - prev_close) / prev_close.replace(0, np.nan)
        result['opening_gap_abs'] = result['opening_gap'].abs()

        # Gap classification
        result['is_gap_up'] = (result['opening_gap'] > 0.005).astype(int)  # >0.5% gap up
        result['is_gap_down'] = (result['opening_gap'] < -0.005).astype(int)  # >0.5% gap down

        # Intrabar price reversal strength
        # Measures how much the price reversed from open to close relative to range
        bar_range_safe = result['bar_range'].replace(0, np.nan)

        # Did price reverse from open direction?
        max_adverse = result[['open', 'close']].min(axis=1) - result['low']
        max_favorable = result['high'] - result[['open', 'close']].max(axis=1)

        result['intrabar_reversal_strength'] = (
            (max_adverse + max_favorable) / bar_range_safe
        )

        # Price efficiency within bar (straight line vs actual path)
        # Efficiency = |close - open| / (high - low)
        result['intrabar_price_efficiency'] = result['body_to_range_ratio']

        # Momentum persistence (does the bar continue previous momentum?)
        prev_momentum = (result['close'] - prev_close) / prev_close.replace(0, np.nan)
        current_momentum = (result['close'] - result['open']) / result['open'].replace(0, np.nan)

        result['momentum_persistence'] = (
            (np.sign(prev_momentum) == np.sign(current_momentum)).astype(int)
        )

        # Momentum acceleration
        result['momentum_acceleration'] = current_momentum / (prev_momentum.replace(0, np.nan))

        # Breakout detection (close outside previous bar's range)
        prev_high = result['high'].shift(1)
        prev_low = result['low'].shift(1)

        result['breaks_above_prev_high'] = (result['close'] > prev_high).astype(int)
        result['breaks_below_prev_low'] = (result['close'] < prev_low).astype(int)

        # Inside bar (current range contained within previous range)
        result['is_inside_bar'] = (
            (result['high'] <= prev_high) &
            (result['low'] >= prev_low)
        ).astype(int)

        # Outside bar (current range engulfs previous range)
        result['is_outside_bar'] = (
            (result['high'] > prev_high) &
            (result['low'] < prev_low)
        ).astype(int)

        return result

    def _add_volume_weighted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-weighted price metrics.

        These features weight price by volume to better reflect actual trading activity.
        """
        result = df.copy()

        # Typical price (HLCC/4) - common VWAP proxy
        result['typical_price'] = (result['high'] + result['low'] + 2 * result['close']) / 4

        # VWAP proxy (typical price weighted by volume over rolling window)
        for window in [5, 10, 20]:
            result[f'vwap_proxy_{window}d'] = (
                (result['typical_price'] * result['volume']).rolling(window).sum() /
                result['volume'].rolling(window).sum()
            )

            # Distance from VWAP
            result[f'distance_from_vwap_{window}d'] = (
                (result['close'] - result[f'vwap_proxy_{window}d']) /
                result[f'vwap_proxy_{window}d'].replace(0, np.nan)
            )

        # Volume-adjusted range (larger volume = more significant range)
        volume_normalized = result['volume'] / result['volume'].rolling(20).mean()
        result['volume_adjusted_range'] = result['bar_range'] * np.sqrt(volume_normalized)

        # Volume surprise on small body (accumulation/distribution signal)
        # High volume on small body suggests hidden buying/selling
        result['volume_on_small_body'] = (
            (result['body_to_range_ratio'] < 0.3) &
            (volume_normalized > 1.5)
        ).astype(int)

        # Volume-weighted directional pressure
        # Positive body + high volume = buying pressure
        # Negative body + high volume = selling pressure
        result['buying_pressure'] = (
            result['bar_body'].clip(lower=0) * volume_normalized
        )
        result['selling_pressure'] = (
            result['bar_body'].clip(upper=0).abs() * volume_normalized
        )

        # Net pressure
        result['net_volume_pressure'] = result['buying_pressure'] - result['selling_pressure']

        # Volume profile within bar (where in the range did most trading occur?)
        # Close near high + high volume = buying at ask
        # Close near low + high volume = selling at bid
        result['volume_at_high'] = (
            result['close_position_in_range'] * volume_normalized
        )
        result['volume_at_low'] = (
            (1 - result['close_position_in_range']) * volume_normalized
        )

        return result

    def _add_price_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price efficiency and fractal features.

        These features measure how efficiently price moves toward its destination.
        """
        result = df.copy()

        # Multi-period efficiency measures
        for window in [5, 10, 20]:
            # Net price change over window
            price_change = result['close'] - result['close'].shift(window)

            # Sum of absolute bar-to-bar changes (actual path traveled)
            abs_changes = result['close'].diff().abs().rolling(window).sum()

            # Efficiency = straight-line distance / actual path
            # 1.0 = perfect efficiency (straight line), 0.0 = no net movement
            result[f'price_efficiency_{window}d'] = (
                price_change.abs() / abs_changes.replace(0, np.nan)
            )

        # Range retracement (how much of previous bar's range is retraced?)
        prev_range = result['bar_range'].shift(1)
        overlap_high = result[['high', 'high']].shift(1).min(axis=1)
        overlap_low = result[['low', 'low']].shift(1).max(axis=1)
        overlap_range = (overlap_high - overlap_low).clip(lower=0)

        result['range_retracement_ratio'] = overlap_range / prev_range.replace(0, np.nan)

        # Volatility efficiency (is volatility productive or chaotic?)
        for window in [5, 10]:
            returns = result['close'].pct_change()
            cum_return = returns.rolling(window).sum()
            volatility = returns.rolling(window).std()

            result[f'volatility_efficiency_{window}d'] = (
                cum_return.abs() / (volatility * np.sqrt(window))
            ).replace([np.inf, -np.inf], np.nan)

        # Fractal dimension proxy (complexity of price path)
        # Higher = more complex/chaotic, Lower = smoother trend
        for window in [10, 20]:
            result[f'price_complexity_{window}d'] = self._calculate_price_complexity(
                result['close'], window
            )

        # Bid-ask bounce proxy (alternating up/down suggests low volume chop)
        price_direction = np.sign(result['close'].diff())
        result['direction_alternation_5d'] = (
            (price_direction != price_direction.shift(1)).rolling(5).sum() / 5
        )

        # Smooth vs choppy movement
        for window in [5, 10]:
            # Count direction changes
            direction_changes = (price_direction != price_direction.shift(1)).rolling(window).sum()
            result[f'price_smoothness_{window}d'] = 1.0 - (direction_changes / window)

        return result

    def _count_consecutive(self, series: pd.Series, window: int) -> pd.Series:
        """
        Count consecutive occurrences of a condition.

        Args:
            series: Boolean series
            window: Maximum window to count

        Returns:
            Series with consecutive counts
        """
        # Create groups where value changes
        groups = (series != series.shift(1)).cumsum()

        # Count within each group
        consecutive = series.groupby(groups).cumsum()

        # Limit to window
        return consecutive.clip(upper=window)

    def _calculate_price_complexity(self, price_series: pd.Series, window: int) -> pd.Series:
        """
        Calculate price path complexity (fractal dimension proxy).

        Higher values indicate more complex/choppy price action.

        Args:
            price_series: Price series
            window: Window for calculation

        Returns:
            Series with complexity scores
        """
        def complexity_score(x):
            if len(x) < 3:
                return np.nan

            # Number of local minima/maxima
            changes = np.diff(np.sign(np.diff(x)))
            turning_points = np.sum(changes != 0)

            # Normalize by window
            return turning_points / (len(x) - 2)

        return price_series.rolling(window).apply(complexity_score, raw=True)

    def get_feature_names(self) -> List[str]:
        """
        Get list of all microstructural feature names.

        Returns:
            List of feature names
        """
        features = [
            # OHLC Patterns
            'bar_range', 'bar_body', 'bar_body_abs',
            'body_to_range_ratio',
            'upper_shadow', 'lower_shadow',
            'upper_shadow_ratio', 'lower_shadow_ratio', 'shadow_balance',
            'close_position_in_range', 'open_position_in_range',
            'is_bullish_candle', 'is_bearish_candle', 'is_doji',
            'is_hammer_like', 'is_shooting_star_like',
            'is_bullish_engulfing', 'is_bearish_engulfing',
            'range_expansion_ratio',
            'consecutive_bullish_candles', 'consecutive_bearish_candles',

            # Intrabar Momentum
            'opening_gap', 'opening_gap_abs',
            'is_gap_up', 'is_gap_down',
            'intrabar_reversal_strength', 'intrabar_price_efficiency',
            'momentum_persistence', 'momentum_acceleration',
            'breaks_above_prev_high', 'breaks_below_prev_low',
            'is_inside_bar', 'is_outside_bar',

            # Volume-Weighted
            'typical_price',
            'vwap_proxy_5d', 'vwap_proxy_10d', 'vwap_proxy_20d',
            'distance_from_vwap_5d', 'distance_from_vwap_10d', 'distance_from_vwap_20d',
            'volume_adjusted_range',
            'volume_on_small_body',
            'buying_pressure', 'selling_pressure', 'net_volume_pressure',
            'volume_at_high', 'volume_at_low',

            # Price Efficiency
            'price_efficiency_5d', 'price_efficiency_10d', 'price_efficiency_20d',
            'range_retracement_ratio',
            'volatility_efficiency_5d', 'volatility_efficiency_10d',
            'price_complexity_10d', 'price_complexity_20d',
            'direction_alternation_5d',
            'price_smoothness_5d', 'price_smoothness_10d',
        ]

        return features

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get microstructural features organized by category.

        Returns:
            Dictionary mapping category to feature names
        """
        return {
            'ohlc_patterns': [
                'bar_range', 'bar_body', 'bar_body_abs',
                'body_to_range_ratio',
                'upper_shadow', 'lower_shadow',
                'upper_shadow_ratio', 'lower_shadow_ratio', 'shadow_balance',
                'close_position_in_range', 'open_position_in_range',
                'is_bullish_candle', 'is_bearish_candle', 'is_doji',
                'is_hammer_like', 'is_shooting_star_like',
                'is_bullish_engulfing', 'is_bearish_engulfing',
                'range_expansion_ratio',
                'consecutive_bullish_candles', 'consecutive_bearish_candles',
            ],
            'intrabar_momentum': [
                'opening_gap', 'opening_gap_abs',
                'is_gap_up', 'is_gap_down',
                'intrabar_reversal_strength', 'intrabar_price_efficiency',
                'momentum_persistence', 'momentum_acceleration',
                'breaks_above_prev_high', 'breaks_below_prev_low',
                'is_inside_bar', 'is_outside_bar',
            ],
            'volume_weighted': [
                'typical_price',
                'vwap_proxy_5d', 'vwap_proxy_10d', 'vwap_proxy_20d',
                'distance_from_vwap_5d', 'distance_from_vwap_10d', 'distance_from_vwap_20d',
                'volume_adjusted_range',
                'volume_on_small_body',
                'buying_pressure', 'selling_pressure', 'net_volume_pressure',
                'volume_at_high', 'volume_at_low',
            ],
            'price_efficiency': [
                'price_efficiency_5d', 'price_efficiency_10d', 'price_efficiency_20d',
                'range_retracement_ratio',
                'volatility_efficiency_5d', 'volatility_efficiency_10d',
                'price_complexity_10d', 'price_complexity_20d',
                'direction_alternation_5d',
                'price_smoothness_5d', 'price_smoothness_10d',
            ]
        }
