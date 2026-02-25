"""
Time-to-Peak Feature Extraction for Pattern Analysis

This module calculates features related to the timing and velocity of price movements,
helping distinguish quick explosive moves from slow grinds.

Key Features:
- days_to_max_gain: How many days to reach peak price
- gain_velocity: Gain per day (explosive vs slow)
- peak_timing: Early/Mid/Late classification
- sustainability: How well gains were maintained
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TimeToPeakCalculator:
    """
    Calculate time-to-peak metrics for pattern outcomes.

    These features help identify patterns with quick, explosive moves
    vs patterns with slow grinds that are more vulnerable to reversals.
    """

    def __init__(self, lookforward_days: int = 100):
        """
        Initialize calculator.

        Args:
            lookforward_days: Days to look forward for outcome evaluation
        """
        self.lookforward_days = lookforward_days

    def calculate_time_to_peak_features(
        self,
        df: pd.DataFrame,
        pattern_start_idx: int
    ) -> Dict[str, float]:
        """
        Calculate all time-to-peak features for a pattern.

        Args:
            df: DataFrame with OHLCV data
            pattern_start_idx: Index where pattern starts

        Returns:
            Dictionary with time-to-peak features
        """
        if pattern_start_idx + self.lookforward_days > len(df):
            return self._empty_features()

        # Get future price data
        future_end = min(pattern_start_idx + self.lookforward_days, len(df))
        future_data = df.iloc[pattern_start_idx:future_end]

        if len(future_data) < 2:
            return self._empty_features()

        # Calculate basic metrics
        breakout_price = df.iloc[pattern_start_idx]['close']
        future_prices = future_data['close'].values

        # Find maximum price and its index
        max_price_idx = np.argmax(future_prices)
        max_price = future_prices[max_price_idx]

        # Find minimum price and its index
        min_price_idx = np.argmin(future_prices)
        min_price = future_prices[min_price_idx]

        # Calculate gains
        max_gain = (max_price - breakout_price) / breakout_price
        max_loss = (min_price - breakout_price) / breakout_price
        end_price = future_prices[-1]
        end_gain = (end_price - breakout_price) / breakout_price

        # Time to peak metrics
        days_to_max_gain = max_price_idx
        days_to_min_price = min_price_idx
        total_days = len(future_prices) - 1

        # Velocity metrics
        if days_to_max_gain > 0:
            gain_velocity = max_gain / days_to_max_gain
            avg_daily_gain = gain_velocity
        else:
            gain_velocity = 0.0
            avg_daily_gain = 0.0

        # Peak timing classification
        if total_days > 0:
            peak_timing_ratio = days_to_max_gain / total_days
            if peak_timing_ratio < 0.33:
                peak_timing_category = 'early'
            elif peak_timing_ratio < 0.66:
                peak_timing_category = 'mid'
            else:
                peak_timing_category = 'late'
        else:
            peak_timing_ratio = 0.0
            peak_timing_category = 'unknown'

        # Sustainability metrics
        if max_gain > 0:
            sustainability = end_gain / max_gain
            gain_retention = max(0, min(1, sustainability))
        else:
            sustainability = 0.0
            gain_retention = 0.0

        # Drawdown after peak
        if max_price_idx < len(future_prices) - 1:
            prices_after_peak = future_prices[max_price_idx:]
            min_after_peak = prices_after_peak.min()
            drawdown_from_peak = (min_after_peak - max_price) / max_price
        else:
            drawdown_from_peak = 0.0

        # Recovery metrics
        recovery_strength = 0.0
        if drawdown_from_peak < -0.05:  # If there was a >5% drawdown
            # Check if price recovered
            if end_price > max_price * 0.95:  # Within 5% of peak
                recovery_strength = 1.0
            elif end_price > max_price * 0.90:
                recovery_strength = 0.5

        # Momentum consistency (how smooth was the rise)
        if days_to_max_gain > 1:
            prices_to_peak = future_prices[:days_to_max_gain + 1]
            daily_returns = np.diff(prices_to_peak) / prices_to_peak[:-1]

            # Positive return ratio
            positive_days = (daily_returns > 0).sum()
            momentum_consistency = positive_days / len(daily_returns) if len(daily_returns) > 0 else 0

            # Return volatility
            return_volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
        else:
            momentum_consistency = 0.0
            return_volatility = 0.0

        # Explosive move detection
        is_explosive = (
            gain_velocity > 0.03 and  # >3% per day average
            days_to_max_gain <= 30 and  # Reached peak quickly
            max_gain >= 0.40  # Significant total gain
        )

        # Slow grind detection
        is_slow_grind = (
            days_to_max_gain > 60 and  # Took a long time
            max_gain >= 0.40  # But still hit gain target
        )

        features = {
            # Core timing features
            'days_to_max_gain': float(days_to_max_gain),
            'days_to_min_price': float(days_to_min_price),
            'time_to_peak_ratio': peak_timing_ratio,
            'peak_timing_category_early': 1.0 if peak_timing_category == 'early' else 0.0,
            'peak_timing_category_mid': 1.0 if peak_timing_category == 'mid' else 0.0,
            'peak_timing_category_late': 1.0 if peak_timing_category == 'late' else 0.0,

            # Velocity features
            'gain_velocity': gain_velocity,
            'avg_daily_gain': avg_daily_gain,

            # Sustainability features
            'sustainability': sustainability,
            'gain_retention': gain_retention,
            'drawdown_from_peak': drawdown_from_peak,
            'recovery_strength': recovery_strength,

            # Momentum features
            'momentum_consistency': momentum_consistency,
            'return_volatility': return_volatility,

            # Pattern classification
            'is_explosive_move': 1.0 if is_explosive else 0.0,
            'is_slow_grind': 1.0 if is_slow_grind else 0.0,

            # Additional metrics
            'max_gain': max_gain,
            'max_loss': max_loss,
            'end_gain': end_gain,
        }

        return features

    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dictionary."""
        return {
            'days_to_max_gain': 0.0,
            'days_to_min_price': 0.0,
            'time_to_peak_ratio': 0.0,
            'peak_timing_category_early': 0.0,
            'peak_timing_category_mid': 0.0,
            'peak_timing_category_late': 0.0,
            'gain_velocity': 0.0,
            'avg_daily_gain': 0.0,
            'sustainability': 0.0,
            'gain_retention': 0.0,
            'drawdown_from_peak': 0.0,
            'recovery_strength': 0.0,
            'momentum_consistency': 0.0,
            'return_volatility': 0.0,
            'is_explosive_move': 0.0,
            'is_slow_grind': 0.0,
            'max_gain': 0.0,
            'max_loss': 0.0,
            'end_gain': 0.0,
        }

    def classify_move_quality(
        self,
        features: Dict[str, float]
    ) -> Tuple[str, float]:
        """
        Classify the quality of the move based on time-to-peak features.

        Args:
            features: Dictionary from calculate_time_to_peak_features

        Returns:
            (quality_category, quality_score)

        Quality Categories:
        - 'explosive_high_quality': Fast, sustainable, high gain
        - 'explosive_reversal': Fast gain but gave it back
        - 'slow_grind': Took too long to reach peak
        - 'failed': Never achieved significant gain
        """
        max_gain = features.get('max_gain', 0.0)
        gain_velocity = features.get('gain_velocity', 0.0)
        sustainability = features.get('sustainability', 0.0)
        days_to_max = features.get('days_to_max_gain', 100)

        # Explosive high quality: Fast + sustained
        if gain_velocity > 0.025 and max_gain >= 0.40 and sustainability > 0.70:
            return 'explosive_high_quality', 1.0

        # Explosive but reversed: Fast but not sustained
        if gain_velocity > 0.025 and max_gain >= 0.40 and sustainability < 0.50:
            return 'explosive_reversal', 0.5

        # Slow grind: Reached target but took too long
        if days_to_max > 60 and max_gain >= 0.40:
            return 'slow_grind', 0.3

        # Moderate quality: Decent velocity and sustainability
        if gain_velocity > 0.015 and max_gain >= 0.25 and sustainability > 0.60:
            return 'moderate_quality', 0.6

        # Failed pattern
        if max_gain < 0.15:
            return 'failed', 0.0

        return 'mixed', 0.4

    def get_feature_names(self) -> list:
        """Get list of all feature names generated."""
        return [
            'days_to_max_gain',
            'days_to_min_price',
            'time_to_peak_ratio',
            'peak_timing_category_early',
            'peak_timing_category_mid',
            'peak_timing_category_late',
            'gain_velocity',
            'avg_daily_gain',
            'sustainability',
            'gain_retention',
            'drawdown_from_peak',
            'recovery_strength',
            'momentum_consistency',
            'return_volatility',
            'is_explosive_move',
            'is_slow_grind',
            'max_gain',
            'max_loss',
            'end_gain',
        ]
