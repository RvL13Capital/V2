"""
Feature Interaction Generator for AIv3

Creates explicit interaction features between key indicators to capture
non-linear relationships and complex patterns.

Key Interaction Types:
1. Volatility × Trend (BBW × ADX)
2. Volume × Momentum (Volume × Price change)
3. Volatility × Volume (BBW × Volume ratio)
4. Trend × Momentum (ADX × RSI)
5. Range × Volume (Daily range × Volume)

These interactions help the model understand:
- High volatility in trending vs. consolidating markets
- Volume confirmation of price movements
- Accumulation patterns (low volatility + high volume)
- Breakout precursors (compression + volume buildup)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureInteractionGenerator:
    """
    Generates interaction features between key indicators.

    Philosophy:
    - Multiplication captures joint effects (e.g., volatility * trend)
    - Division/ratio captures relative strength
    - Difference captures divergence
    - Domain knowledge guides which interactions to create
    """

    def __init__(
        self,
        enable_volatility_interactions: bool = True,
        enable_volume_interactions: bool = True,
        enable_momentum_interactions: bool = True,
        enable_microstructural_interactions: bool = True,
        enable_polynomial_features: bool = False,  # Squared, cubed terms
        max_polynomial_degree: int = 2
    ):
        """
        Initialize interaction generator.

        Args:
            enable_volatility_interactions: Create volatility-based interactions
            enable_volume_interactions: Create volume-based interactions
            enable_momentum_interactions: Create momentum-based interactions
            enable_microstructural_interactions: Create micro-structure interactions
            enable_polynomial_features: Add polynomial terms (x^2, x^3)
            max_polynomial_degree: Maximum polynomial degree (2 or 3)
        """
        self.enable_volatility_interactions = enable_volatility_interactions
        self.enable_volume_interactions = enable_volume_interactions
        self.enable_momentum_interactions = enable_momentum_interactions
        self.enable_microstructural_interactions = enable_microstructural_interactions
        self.enable_polynomial_features = enable_polynomial_features
        self.max_polynomial_degree = max_polynomial_degree

        self.interaction_names = []

    def generate_all_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all interaction features.

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with added interaction features
        """
        result = df.copy()
        self.interaction_names = []

        # 1. Volatility interactions
        if self.enable_volatility_interactions:
            result = self._add_volatility_interactions(result)

        # 2. Volume interactions
        if self.enable_volume_interactions:
            result = self._add_volume_interactions(result)

        # 3. Momentum interactions
        if self.enable_momentum_interactions:
            result = self._add_momentum_interactions(result)

        # 4. Microstructural interactions
        if self.enable_microstructural_interactions:
            result = self._add_microstructural_interactions(result)

        # 5. Polynomial features (optional)
        if self.enable_polynomial_features:
            result = self._add_polynomial_features(result)

        # 6. Advanced composite interactions
        result = self._add_composite_interactions(result)

        logger.info(f"Generated {len(self.interaction_names)} interaction features")

        return result

    def _add_volatility_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility × Other Features.

        Key insights:
        - Low volatility + low trend = consolidation (coiling for breakout)
        - High volatility + high trend = explosive move (ride the trend)
        - Low volatility + high volume = accumulation (bullish)
        """
        result = df.copy()

        # BBW × ADX: Volatility-Trend interaction
        if 'bbw_20' in result.columns and 'adx' in result.columns:
            # Normalized interaction
            result['volatility_x_trend'] = result['bbw_20'] * result['adx'] / 100
            self.interaction_names.append('volatility_x_trend')

            # Consolidation detector: Low BBW + Low ADX
            result['consolidation_strength'] = (
                (1.0 - np.clip(result['bbw_20'], 0, 1)) *
                (1.0 - np.clip(result['adx'] / 50, 0, 1))
            )
            self.interaction_names.append('consolidation_strength')

            # Explosive move detector: High BBW + High ADX
            result['explosive_strength'] = (
                np.clip(result['bbw_20'], 0, 1) *
                np.clip(result['adx'] / 50, 0, 1)
            )
            self.interaction_names.append('explosive_strength')

        # BBW × Volume ratio: Volatility-Volume interaction
        if 'bbw_20' in result.columns and 'volume_ratio_20' in result.columns:
            # Accumulation pattern: Low BBW + High Volume
            result['accumulation_pattern'] = (
                (1.0 - np.clip(result['bbw_20'], 0, 1)) *
                np.clip(result['volume_ratio_20'], 0, 2)
            )
            self.interaction_names.append('accumulation_pattern')

            # Distribution pattern: High BBW + High Volume
            result['distribution_pattern'] = (
                np.clip(result['bbw_20'], 0, 1) *
                np.clip(result['volume_ratio_20'], 0, 2)
            )
            self.interaction_names.append('distribution_pattern')

        # BBW slope × ADX: Volatility change + Trend
        if 'bbw_20' in result.columns and 'adx' in result.columns:
            bbw_slope = result['bbw_20'].diff(5) / result['bbw_20'].shift(5)

            # Compression with low trend = setup for breakout
            result['compression_setup'] = (
                np.where(bbw_slope < 0, -bbw_slope, 0) *
                (1.0 - np.clip(result['adx'] / 50, 0, 1))
            )
            self.interaction_names.append('compression_setup')

        # ATR × BBW: True range volatility interaction
        if 'atr_14' in result.columns and 'bbw_20' in result.columns:
            result['true_volatility_composite'] = (
                result['atr_14'] / result['close'] * result['bbw_20']
            )
            self.interaction_names.append('true_volatility_composite')

        return result

    def _add_volume_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume × Other Features.

        Key insights:
        - Volume × Price change = momentum confirmation
        - Volume × Range = buying/selling pressure
        - Volume trend × Price trend = trend strength
        """
        result = df.copy()

        # Volume × Price momentum
        if 'volume_ratio_20' in result.columns and 'close' in result.columns:
            price_momentum_5d = result['close'].pct_change(5)

            # Strong volume with strong price = momentum
            result['volume_momentum_strength'] = (
                np.clip(result['volume_ratio_20'], 0, 2) *
                np.abs(price_momentum_5d) * 10
            )
            self.interaction_names.append('volume_momentum_strength')

            # Volume divergence: High volume, weak price
            result['volume_price_divergence'] = np.where(
                (result['volume_ratio_20'] > 1.5) & (np.abs(price_momentum_5d) < 0.05),
                result['volume_ratio_20'] - 1.0,
                0.0
            )
            self.interaction_names.append('volume_price_divergence')

        # Volume × RSI: Overbought/oversold with volume confirmation
        if 'volume_ratio_20' in result.columns and 'rsi_14' in result.columns:
            # High volume at oversold = bullish
            result['oversold_volume_signal'] = np.where(
                result['rsi_14'] < 30,
                np.clip(result['volume_ratio_20'], 0, 2),
                0.0
            )
            self.interaction_names.append('oversold_volume_signal')

            # High volume at overbought = bearish
            result['overbought_volume_signal'] = np.where(
                result['rsi_14'] > 70,
                np.clip(result['volume_ratio_20'], 0, 2),
                0.0
            )
            self.interaction_names.append('overbought_volume_signal')

        # Volume × Daily range: Buying/selling pressure
        if 'volume_ratio_20' in result.columns and all(c in result.columns for c in ['high', 'low', 'close']):
            daily_range_pct = (result['high'] - result['low']) / result['close']

            # High volume + wide range = strong movement
            result['volume_range_intensity'] = (
                np.clip(result['volume_ratio_20'], 0, 2) *
                np.clip(daily_range_pct * 20, 0, 2)
            )
            self.interaction_names.append('volume_range_intensity')

        # Volume accumulation × Price position
        if 'volume_ratio_20' in result.columns and 'close' in result.columns:
            # Price position in recent range
            high_20 = result['high'].rolling(20).max()
            low_20 = result['low'].rolling(20).min()
            price_position = (result['close'] - low_20) / (high_20 - low_20 + 1e-10)

            # Volume at top of range (resistance)
            result['volume_at_resistance'] = (
                np.clip(result['volume_ratio_20'], 0, 2) *
                np.where(price_position > 0.9, price_position, 0)
            )
            self.interaction_names.append('volume_at_resistance')

            # Volume at bottom of range (support)
            result['volume_at_support'] = (
                np.clip(result['volume_ratio_20'], 0, 2) *
                np.where(price_position < 0.1, 1 - price_position, 0)
            )
            self.interaction_names.append('volume_at_support')

        return result

    def _add_momentum_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum × Other Features.

        Key insights:
        - RSI × ADX = momentum in trending market
        - RSI × Volume = confirmed momentum
        - Price momentum × Volatility = explosive potential
        """
        result = df.copy()

        # RSI × ADX: Momentum-Trend interaction
        if 'rsi_14' in result.columns and 'adx' in result.columns:
            # Strong momentum in strong trend
            result['momentum_trend_alignment'] = (
                np.abs(result['rsi_14'] - 50) / 50 *  # Momentum strength
                np.clip(result['adx'] / 50, 0, 1)      # Trend strength
            )
            self.interaction_names.append('momentum_trend_alignment')

        # Price momentum × BBW: Momentum during compression
        if 'close' in result.columns and 'bbw_20' in result.columns:
            price_momentum_10d = result['close'].pct_change(10)

            # Momentum building during compression
            result['compressed_momentum'] = (
                np.abs(price_momentum_10d) * 10 *
                (1.0 - np.clip(result['bbw_20'], 0, 1))
            )
            self.interaction_names.append('compressed_momentum')

        # RSI × Volume trend: Momentum with volume confirmation
        if 'rsi_14' in result.columns and 'volume' in result.columns:
            volume_trend_5d = result['volume'].pct_change(5)

            # Bullish: High RSI + Rising volume
            result['bullish_momentum_confirmed'] = np.where(
                (result['rsi_14'] > 50) & (volume_trend_5d > 0),
                (result['rsi_14'] - 50) / 50 * np.clip(volume_trend_5d * 2, 0, 1),
                0.0
            )
            self.interaction_names.append('bullish_momentum_confirmed')

        # Multi-timeframe momentum alignment
        if 'close' in result.columns:
            mom_5d = result['close'].pct_change(5)
            mom_10d = result['close'].pct_change(10)
            mom_20d = result['close'].pct_change(20)

            # All timeframes aligned (same direction)
            result['momentum_alignment_score'] = (
                ((mom_5d > 0) & (mom_10d > 0) & (mom_20d > 0)).astype(float) +
                ((mom_5d < 0) & (mom_10d < 0) & (mom_20d < 0)).astype(float) * -1
            )
            self.interaction_names.append('momentum_alignment_score')

        return result

    def _add_microstructural_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Microstructural × Other Features.

        Key insights:
        - Candle patterns × Volume = pattern confirmation
        - Body/shadow ratio × Volatility = price action quality
        """
        result = df.copy()

        # Body/Shadow ratio × Volume
        if all(c in result.columns for c in ['body_ratio', 'volume_ratio_20']):
            # Strong candle body + high volume = conviction
            result['conviction_signal'] = (
                np.clip(result['body_ratio'], 0, 1) *
                np.clip(result['volume_ratio_20'], 0, 2)
            )
            self.interaction_names.append('conviction_signal')

        # Candle complexity × Volatility
        if 'candle_complexity_score' in result.columns and 'bbw_20' in result.columns:
            # Complex candles during low volatility = indecision before breakout
            result['indecision_before_breakout'] = (
                np.clip(result['candle_complexity_score'], 0, 1) *
                (1.0 - np.clip(result['bbw_20'], 0, 1))
            )
            self.interaction_names.append('indecision_before_breakout')

        # Bar range × Volume (pressure)
        if 'bar_range_pct' in result.columns and 'volume_ratio_20' in result.columns:
            result['bar_pressure'] = (
                np.clip(result['bar_range_pct'] * 10, 0, 2) *
                np.clip(result['volume_ratio_20'], 0, 2)
            )
            self.interaction_names.append('bar_pressure')

        return result

    def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add polynomial terms for key features.

        Captures non-linear relationships (e.g., extreme RSI has exponential impact).
        """
        result = df.copy()

        # Key features to apply polynomial transformation
        polynomial_features = []

        # Add BBW if exists
        if 'bbw_20' in result.columns:
            polynomial_features.append('bbw_20')

        # Add RSI if exists
        if 'rsi_14' in result.columns:
            polynomial_features.append('rsi_14')

        # Add ADX if exists
        if 'adx' in result.columns:
            polynomial_features.append('adx')

        # Generate polynomial terms
        for feature in polynomial_features:
            if self.max_polynomial_degree >= 2:
                result[f'{feature}_squared'] = result[feature] ** 2
                self.interaction_names.append(f'{feature}_squared')

            if self.max_polynomial_degree >= 3:
                result[f'{feature}_cubed'] = result[feature] ** 3
                self.interaction_names.append(f'{feature}_cubed')

        return result

    def _add_composite_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced composite interactions combining 3+ features.

        These capture complex multi-factor scenarios.
        """
        result = df.copy()

        # Breakout setup score: Low BBW + Low ADX + Rising Volume
        if all(c in result.columns for c in ['bbw_20', 'adx', 'volume_ratio_20']):
            result['breakout_setup_score'] = (
                (1.0 - np.clip(result['bbw_20'], 0, 1)) *  # Compression
                (1.0 - np.clip(result['adx'] / 50, 0, 1)) *  # Low trend
                np.clip(result['volume_ratio_20'], 0, 2)    # Volume building
            )
            self.interaction_names.append('breakout_setup_score')

        # Trend exhaustion: High RSI + High ADX + Declining Volume
        if all(c in result.columns for c in ['rsi_14', 'adx', 'volume' ]):
            volume_trend = result['volume'].pct_change(5)

            result['trend_exhaustion_score'] = (
                np.where(result['rsi_14'] > 70, (result['rsi_14'] - 70) / 30, 0) *
                np.clip(result['adx'] / 50, 0, 1) *
                np.where(volume_trend < 0, -volume_trend, 0)
            )
            self.interaction_names.append('trend_exhaustion_score')

        # Accumulation score: Low volatility + High volume + Price near support
        if all(c in result.columns for c in ['bbw_20', 'volume_ratio_20', 'close', 'low']):
            low_20 = result['low'].rolling(20).min()
            distance_to_support = (result['close'] - low_20) / result['close']

            result['accumulation_score'] = (
                (1.0 - np.clip(result['bbw_20'], 0, 1)) *
                np.clip(result['volume_ratio_20'], 0, 2) *
                (1.0 - np.clip(distance_to_support * 10, 0, 1))
            )
            self.interaction_names.append('accumulation_score')

        return result

    def get_interaction_names(self) -> List[str]:
        """Get list of all generated interaction feature names."""
        return self.interaction_names

    def get_interaction_groups(self) -> Dict[str, List[str]]:
        """
        Group interactions by type for analysis.

        Returns:
            Dictionary mapping interaction type to feature names
        """
        groups = {
            'volatility_interactions': [],
            'volume_interactions': [],
            'momentum_interactions': [],
            'microstructural_interactions': [],
            'polynomial': [],
            'composite': []
        }

        for name in self.interaction_names:
            if 'volatility' in name or 'compression' in name or 'consolidation' in name:
                groups['volatility_interactions'].append(name)
            elif 'volume' in name:
                groups['volume_interactions'].append(name)
            elif 'momentum' in name or 'rsi' in name:
                groups['momentum_interactions'].append(name)
            elif any(x in name for x in ['body_', 'bar_', 'candle', 'shadow']):
                groups['microstructural_interactions'].append(name)
            elif 'squared' in name or 'cubed' in name:
                groups['polynomial'].append(name)
            elif 'score' in name or 'setup' in name or 'exhaustion' in name:
                groups['composite'].append(name)

        return groups

    def analyze_interaction_importance(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Analyze correlation of interactions with target.

        Args:
            df: DataFrame with interaction features
            target: Target variable
            top_n: Number of top interactions to return

        Returns:
            DataFrame with interaction importance scores
        """
        correlations = []

        for feature in self.interaction_names:
            if feature in df.columns:
                # Calculate correlation with target
                corr = df[feature].corr(target)
                correlations.append({
                    'feature': feature,
                    'correlation': abs(corr),
                    'sign': 'positive' if corr > 0 else 'negative'
                })

        # Sort by absolute correlation
        importance_df = pd.DataFrame(correlations)
        if not importance_df.empty:
            importance_df = importance_df.sort_values('correlation', ascending=False).head(top_n)

        return importance_df
