"""
Consolidation-based breakout classification system with K0-K5 categories.
Only classifies stocks currently in consolidation patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BreakoutClass(Enum):
    """Breakout classification for consolidation patterns."""
    K4_EXCEPTIONAL = 4   # 75%+ gain or 100%+ before crash
    K3_STRONG = 3        # 35-75% gain
    K2_QUALITY = 2       # 15-35% gain
    K1_MINIMAL = 1       # 5-15% gain
    K0_STAGNANT = 0      # -5% to +5% (sideways)
    K5_FAILED = 5        # Crash or significant loss
    NO_CONSOLIDATION = -1  # Not in consolidation (not classified)


class ConsolidationBreakoutClassifier:
    """
    Classify ONLY stocks currently in consolidation patterns.
    Predicts future breakout outcome based on current consolidation characteristics.
    """

    def __init__(self,
                 lookforward_days: int = 100,
                 consolidation_min_days: int = 10,
                 bbw_threshold: float = 0.03,
                 volume_threshold: float = 0.7,
                 range_threshold: float = 0.65):
        """
        Initialize consolidation-based classifier.

        Args:
            lookforward_days: Days to look forward for outcomes
            consolidation_min_days: Minimum days in consolidation
            bbw_threshold: Bollinger Band Width threshold for consolidation
            volume_threshold: Volume ratio threshold for consolidation
            range_threshold: Range ratio threshold for consolidation
        """
        self.lookforward_days = lookforward_days
        self.consolidation_min_days = consolidation_min_days
        self.bbw_threshold = bbw_threshold
        self.volume_threshold = volume_threshold
        self.range_threshold = range_threshold

        # Strategic values for expected value calculation
        self.class_values = {
            BreakoutClass.K4_EXCEPTIONAL: 10.0,
            BreakoutClass.K3_STRONG: 3.0,
            BreakoutClass.K2_QUALITY: 1.0,
            BreakoutClass.K1_MINIMAL: -0.2,
            BreakoutClass.K0_STAGNANT: -2.0,
            BreakoutClass.K5_FAILED: -10.0,
            BreakoutClass.NO_CONSOLIDATION: 0.0
        }

        self.consolidation_stats = {}

    def detect_consolidation(self,
                            df: pd.DataFrame,
                            current_idx: int = -1) -> Tuple[bool, Dict]:
        """
        Detect if stock is currently in consolidation.

        Args:
            df: DataFrame with price and indicator data
            current_idx: Index to check (-1 for latest)

        Returns:
            Is in consolidation, consolidation metadata
        """
        if current_idx == -1:
            current_idx = len(df) - 1

        # Need minimum history
        if current_idx < self.consolidation_min_days:
            return False, {'reason': 'insufficient_history'}

        # Get recent data window
        window_start = max(0, current_idx - self.consolidation_min_days)
        window_data = df.iloc[window_start:current_idx + 1]

        # Check consolidation criteria
        consolidation_checks = {}

        # 1. Bollinger Band Width check
        if 'bbw' in df.columns:
            current_bbw = df.iloc[current_idx]['bbw']
            bbw_percentile = df['bbw'].iloc[:current_idx].quantile(0.3)
            consolidation_checks['bbw'] = current_bbw < min(self.bbw_threshold, bbw_percentile)
        else:
            consolidation_checks['bbw'] = False

        # 2. Volume check
        if 'volume_ratio' in df.columns:
            avg_volume_ratio = window_data['volume_ratio'].mean()
            consolidation_checks['volume'] = avg_volume_ratio < self.volume_threshold
        else:
            consolidation_checks['volume'] = False

        # 3. Range check
        if 'range_ratio' in df.columns:
            avg_range_ratio = window_data['range_ratio'].mean()
            consolidation_checks['range'] = avg_range_ratio < self.range_threshold
        else:
            consolidation_checks['range'] = False

        # 4. Check pattern detection methods if available
        pattern_methods = ['method1_bollinger', 'method2_range_based',
                          'method3_volume_weighted', 'method4_atr_based']
        pattern_count = 0
        for method in pattern_methods:
            if method in df.columns:
                if df.iloc[current_idx][method]:
                    pattern_count += 1

        consolidation_checks['patterns'] = pattern_count >= 2

        # Determine if in consolidation (need at least 3 criteria met)
        criteria_met = sum(consolidation_checks.values())
        is_consolidation = criteria_met >= 3

        # Calculate consolidation strength
        consolidation_strength = criteria_met / len(consolidation_checks)

        metadata = {
            'checks': consolidation_checks,
            'criteria_met': criteria_met,
            'strength': consolidation_strength,
            'pattern_count': pattern_count,
            'current_bbw': df.iloc[current_idx].get('bbw', None),
            'avg_volume_ratio': window_data.get('volume_ratio', pd.Series()).mean(),
            'avg_range_ratio': window_data.get('range_ratio', pd.Series()).mean()
        }

        return is_consolidation, metadata

    def classify_if_consolidating(self,
                                 df: pd.DataFrame,
                                 symbol: str,
                                 current_idx: int = -1) -> Tuple[BreakoutClass, Dict]:
        """
        Classify ONLY if currently in consolidation, otherwise return NO_CONSOLIDATION.

        Args:
            df: DataFrame with price and indicator data
            symbol: Stock symbol
            current_idx: Current position (-1 for latest)

        Returns:
            Classification and metadata
        """
        if current_idx == -1:
            current_idx = len(df) - 1

        # First check if in consolidation
        is_consolidation, consol_metadata = self.detect_consolidation(df, current_idx)

        if not is_consolidation:
            return BreakoutClass.NO_CONSOLIDATION, {
                'symbol': symbol,
                'reason': 'not_in_consolidation',
                'consolidation_metadata': consol_metadata
            }

        # Stock IS in consolidation - now classify expected breakout
        # For training: use future data if available
        # For prediction: use model features

        if current_idx + self.lookforward_days < len(df):
            # Training mode: we have future data
            classification, class_metadata = self._classify_with_future_data(
                df, current_idx
            )
        else:
            # Prediction mode: no future data, would use ML model
            classification = BreakoutClass.K0_STAGNANT  # Default
            class_metadata = {'mode': 'prediction', 'reason': 'needs_ml_model'}

        # Combine metadata
        metadata = {
            'symbol': symbol,
            'is_consolidation': True,
            'consolidation_strength': consol_metadata['strength'],
            'pattern_count': consol_metadata['pattern_count'],
            'classification': classification.name,
            'strategic_value': self.class_values[classification],
            **class_metadata
        }

        return classification, metadata

    def _classify_with_future_data(self,
                                  df: pd.DataFrame,
                                  current_idx: int) -> Tuple[BreakoutClass, Dict]:
        """
        Classify using future data (for training labels).

        Args:
            df: DataFrame with price data
            current_idx: Current position

        Returns:
            Classification and metadata
        """
        # Get future prices
        future_end = min(current_idx + self.lookforward_days, len(df))
        future_data = df.iloc[current_idx:future_end]

        if len(future_data) < 2:
            return BreakoutClass.K0_STAGNANT, {'reason': 'insufficient_future_data'}

        # Calculate gains/losses
        breakout_price = df.iloc[current_idx]['close']
        future_prices = future_data['close'].values

        max_price = future_prices.max()
        min_price = future_prices.min()
        end_price = future_prices[-1]

        max_gain = (max_price - breakout_price) / breakout_price
        max_loss = (min_price - breakout_price) / breakout_price
        end_gain = (end_price - breakout_price) / breakout_price

        # Check for crash
        crash_occurred = max_loss <= -0.30

        # Find max gain before crash
        max_gain_before_crash = 0
        if crash_occurred:
            crash_idx = np.where(future_prices <= breakout_price * 0.70)[0]
            if len(crash_idx) > 0:
                prices_before_crash = future_prices[:crash_idx[0]]
                if len(prices_before_crash) > 0:
                    max_gain_before_crash = (prices_before_crash.max() - breakout_price) / breakout_price

        # Apply classification logic
        if crash_occurred and max_gain_before_crash >= 1.0:
            classification = BreakoutClass.K4_EXCEPTIONAL
            reason = f"100%+ gain before crash"
        elif crash_occurred:
            classification = BreakoutClass.K5_FAILED
            reason = f"Crash without 100% gain"
        elif max_gain >= 0.75:
            classification = BreakoutClass.K4_EXCEPTIONAL
            reason = f"Exceptional gain ({max_gain:.1%})"
        elif max_gain >= 0.35:
            classification = BreakoutClass.K3_STRONG
            reason = f"Strong gain ({max_gain:.1%})"
        elif end_gain >= 0.15:
            classification = BreakoutClass.K2_QUALITY
            reason = f"Quality gain ({end_gain:.1%})"
        elif end_gain >= 0.05:
            classification = BreakoutClass.K1_MINIMAL
            reason = f"Minimal gain ({end_gain:.1%})"
        elif end_gain >= -0.05:
            classification = BreakoutClass.K0_STAGNANT
            reason = f"Sideways ({end_gain:.1%})"
        else:
            classification = BreakoutClass.K5_FAILED
            reason = f"Failed ({end_gain:.1%})"

        metadata = {
            'max_gain': max_gain,
            'max_loss': max_loss,
            'end_gain': end_gain,
            'crash_occurred': crash_occurred,
            'max_gain_before_crash': max_gain_before_crash if crash_occurred else None,
            'reason': reason
        }

        return classification, metadata

    def process_cloud_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process cloud analysis data - classify ONLY current consolidations.

        Args:
            df: DataFrame from cloud analysis

        Returns:
            DataFrame with classifications for consolidating stocks only
        """
        df = df.copy()

        # Initialize classification columns
        df['in_consolidation'] = False
        df['consolidation_strength'] = 0.0
        df['breakout_class'] = BreakoutClass.NO_CONSOLIDATION.name
        df['strategic_value'] = 0.0
        df['expected_outcome'] = ''

        # Group by symbol and check latest data point for each
        if 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')

                # Check if currently (latest data point) in consolidation
                latest_idx = len(symbol_data) - 1
                classification, metadata = self.classify_if_consolidating(
                    symbol_data, symbol, latest_idx
                )

                # Update latest row for this symbol
                latest_timestamp = symbol_data.iloc[-1]['timestamp']
                mask = (df['symbol'] == symbol) & (df['timestamp'] == latest_timestamp)

                if classification != BreakoutClass.NO_CONSOLIDATION:
                    df.loc[mask, 'in_consolidation'] = True
                    df.loc[mask, 'consolidation_strength'] = metadata.get('consolidation_strength', 0)
                    df.loc[mask, 'breakout_class'] = classification.name
                    df.loc[mask, 'strategic_value'] = self.class_values[classification]
                    df.loc[mask, 'expected_outcome'] = metadata.get('reason', '')

                    logger.info(f"{symbol}: In consolidation - Expected: {classification.name}")

        # Summary statistics
        consolidating = df[df['in_consolidation'] == True]
        self.consolidation_stats = {
            'total_symbols': df['symbol'].nunique() if 'symbol' in df.columns else 0,
            'consolidating_symbols': len(consolidating['symbol'].unique()) if len(consolidating) > 0 else 0,
            'consolidation_rate': len(consolidating) / len(df) if len(df) > 0 else 0,
            'class_distribution': consolidating['breakout_class'].value_counts().to_dict() if len(consolidating) > 0 else {},
            'avg_strategic_value': consolidating['strategic_value'].mean() if len(consolidating) > 0 else 0
        }

        return df

    def get_current_consolidations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get only the stocks currently in consolidation.

        Args:
            df: Processed DataFrame

        Returns:
            DataFrame with only consolidating stocks
        """
        if 'in_consolidation' not in df.columns:
            df = self.process_cloud_data(df)

        consolidating = df[df['in_consolidation'] == True].copy()

        if len(consolidating) > 0:
            # Sort by strategic value (highest expected value first)
            consolidating = consolidating.sort_values('strategic_value', ascending=False)

            print(f"\n{'='*60}")
            print(f"STOCKS CURRENTLY IN CONSOLIDATION: {len(consolidating['symbol'].unique())}")
            print('='*60)

            for _, row in consolidating.head(10).iterrows():
                print(f"\n{row['symbol']}:")
                print(f"  Expected Outcome: {row['breakout_class']}")
                print(f"  Strategic Value: {row['strategic_value']:.1f}")
                print(f"  Consolidation Strength: {row['consolidation_strength']:.1%}")
                print(f"  Current Price: ${row.get('close', 0):.2f}")
                print(f"  BBW: {row.get('bbw', 0):.3f}")

        return consolidating

    def get_statistics(self) -> Dict:
        """Get consolidation and classification statistics."""
        return self.consolidation_stats