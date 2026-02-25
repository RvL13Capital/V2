"""
Enhanced Tabular Feature Extraction for AIv3

This module extends the base tabular features from AIv2 with:
1. Robust slope calculations (Theil-Sen, RANSAC)
2. Volatility-of-volatility (BBW dynamics)
3. Advanced volume accumulation patterns
4. Consolidation quality scoring
5. Feature interactions (NEW v3.6) - Explicit interactions between features

Feature Interactions (v3.6):
- Captures non-linear relationships (e.g., volatility × trend)
- Domain-guided combinations (e.g., compression + volume = accumulation)
- Multi-factor composite scores (e.g., breakout setup score)
- Polynomial features (optional: x², x³)

Combines all AIv3 feature innovations into a single interface.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Add parent directory to path to import from base system
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import base features from AIv2
from features.tabular_features import TabularFeatureExtractor

# Import AIv3 enhancements
from .robust_slope_calculator import RobustSlopeCalculator
from .volatility_of_volatility import VolatilityOfVolatilityCalculator
from .microstructural_features import MicrostructuralFeatureCalculator
from .feature_interactions import FeatureInteractionGenerator

# Import AIv4 EBP features
from .ebp_features import EBPFeatureCalculator

logger = logging.getLogger(__name__)


class EnhancedTabularFeatureExtractor:
    """
    Enhanced feature extractor combining AIv2 base features with AIv3 improvements.

    New AIv3 Features:
    - Robust slopes for better trend detection
    - Volatility-of-volatility for consolidation quality
    - Advanced volume patterns
    - Price-volume divergence metrics

    New AIv4 Features:
    - EBP (Explosive Breakout Predictor) composite indicator
    - 5-component scoring: CCI, VAR, NES, LPF, TSF
    """

    def __init__(
        self,
        feature_groups: Optional[List[str]] = None,
        use_robust_slopes: bool = True,
        use_vol_of_vol: bool = True,
        use_advanced_volume: bool = True,
        use_microstructural: bool = True,
        use_feature_interactions: bool = True,
        use_ebp: bool = True,
        enable_polynomial_features: bool = False
    ):
        """
        Initialize enhanced feature extractor.

        Args:
            feature_groups: Groups of features to extract
            use_robust_slopes: Use Theil-Sen and RANSAC slopes
            use_vol_of_vol: Use volatility-of-volatility features
            use_advanced_volume: Use advanced volume patterns
            use_microstructural: Use microstructural price dynamics features
            use_feature_interactions: Generate explicit feature interactions (NEW)
            use_ebp: Use EBP (Explosive Breakout Predictor) features (AIv4)
            enable_polynomial_features: Add polynomial terms (x^2, x^3)
        """
        # Initialize base AIv2 feature extractor
        self.base_extractor = TabularFeatureExtractor(
            feature_groups=feature_groups
        )

        # Initialize AIv3 enhancement modules
        self.use_robust_slopes = use_robust_slopes
        self.use_vol_of_vol = use_vol_of_vol
        self.use_advanced_volume = use_advanced_volume
        self.use_microstructural = use_microstructural
        self.use_feature_interactions = use_feature_interactions
        self.use_ebp = use_ebp

        if use_robust_slopes:
            self.slope_calculator = RobustSlopeCalculator()

        if use_vol_of_vol:
            self.vol_of_vol_calculator = VolatilityOfVolatilityCalculator()

        if use_microstructural:
            self.microstructural_calculator = MicrostructuralFeatureCalculator()

        if use_feature_interactions:
            self.interaction_generator = FeatureInteractionGenerator(
                enable_polynomial_features=enable_polynomial_features
            )

        if use_ebp:
            self.ebp_calculator = EBPFeatureCalculator()

        self.feature_names = []

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features (base + enhancements).

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with all features
        """
        # Start with base AIv2 features
        features = self.base_extractor.extract_all_features(df)

        # Add AIv3 enhancements
        if self.use_robust_slopes:
            features = self._add_robust_slope_features(features)

        if self.use_vol_of_vol:
            features = self._add_vol_of_vol_features(features)

        if self.use_advanced_volume:
            features = self._add_advanced_volume_features(features)

        if self.use_microstructural:
            features = self._add_microstructural_features(features)

        # Add consolidation quality score
        features = self._add_consolidation_quality_score(features)

        # Add EBP features (AIv4)
        # Must be added AFTER consolidation quality and volume features are computed
        if self.use_ebp:
            features = self._add_ebp_features(features)

        # Add feature interactions (NEW - v3.6)
        # Must be added AFTER all base features are computed
        if self.use_feature_interactions:
            features = self._add_feature_interactions(features)

        # Store feature names
        self.feature_names = [col for col in features.columns
                             if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol']]

        return features

    def _add_robust_slope_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add robust slope calculations for key indicators."""
        result = df.copy()

        # Apply robust slopes to key indicators
        indicators = ['close', 'volume']

        # Add BBW if it exists
        if 'bbw_20' in df.columns or 'bb_width_20' in df.columns:
            bbw_col = 'bbw_20' if 'bbw_20' in df.columns else 'bb_width_20'
            indicators.append(bbw_col)

        # Add ADX if it exists
        if 'adx' in df.columns:
            indicators.append('adx')

        for indicator in indicators:
            if indicator in result.columns:
                result = self.slope_calculator.calculate_multi_window_slope_features(
                    result,
                    column=indicator,
                    windows=[5, 10, 20]
                )

        return result

    def _add_vol_of_vol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-of-volatility features."""
        result = self.vol_of_vol_calculator.calculate_vol_of_vol_features(df)
        return result

    def _add_advanced_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced volume accumulation patterns.

        These features capture multi-day volume sequences that
        predict explosive moves better than single-day volume.
        """
        result = df.copy()

        # Volume accumulation sequences (3-5 days, 5-10 days)
        result['volume_ratio'] = result['volume'] / result['volume'].rolling(20).mean()

        # 3-5 day accumulation
        for window in [3, 5]:
            # Average volume ratio over window
            result[f'vol_accum_{window}d'] = (
                result['volume_ratio'].rolling(window=window).mean()
            )

            # Consecutive days above average
            above_avg = (result['volume_ratio'] > 1.0).astype(int)
            result[f'consec_vol_above_avg_{window}d'] = (
                above_avg.rolling(window=window).sum()
            )

            # Volume momentum
            result[f'vol_momentum_{window}d'] = (
                result['volume'].pct_change(window)
            )

        # 5-10 day accumulation patterns
        for window in [7, 10]:
            result[f'vol_strength_{window}d'] = (
                result['volume_ratio'].rolling(window=window).apply(
                    lambda x: (x > 1.0).sum() / len(x), raw=False
                )
            )

        # Price-volume divergence
        # Strong volume with weak price action
        result['price_change_5d'] = result['close'].pct_change(5)
        result['volume_change_5d'] = result['volume'].pct_change(5)

        # Divergence: High volume increase with low price increase
        result['pv_divergence_bullish'] = np.where(
            (result['volume_change_5d'] > 0.50) &  # Volume up 50%+
            (result['price_change_5d'] < 0.10),     # Price up <10%
            1.0,
            0.0
        )

        # Accumulation/Distribution with volume confirmation
        if 'ad' in result.columns:
            result['ad_volume_confirmed'] = (
                (result['ad'] > result['ad'].shift(5)) &
                (result['volume_ratio'] > 1.2)
            ).astype(int)

        # Volume trend strength
        for window in [5, 10, 20]:
            # Calculate if volume is trending up
            vol_slope = result['volume'].rolling(window=window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0,
                raw=False
            )
            result[f'vol_trend_{window}d'] = np.where(vol_slope > 0, 1, -1)

        return result

    def _add_microstructural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add microstructural price dynamics features.

        These features capture intrabar price patterns and market microstructure
        from OHLC data, providing insights into accumulation/distribution and
        breakout precursors.
        """
        result = self.microstructural_calculator.calculate_all_features(df)
        return result

    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add explicit feature interactions.

        Captures non-linear relationships between features:
        - Volatility × Trend (BBW × ADX)
        - Volume × Momentum
        - Compression + Volume = Accumulation
        - Multi-factor composite scores
        """
        result = self.interaction_generator.generate_all_interactions(df)
        logger.info(f"Added {len(self.interaction_generator.get_interaction_names())} interaction features")
        return result

    def _add_consolidation_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add consolidation quality composite score.

        Combines multiple indicators to assess how "tight" and
        "high-quality" a consolidation pattern is.
        """
        result = df.copy()

        quality_components = []

        # Component 1: BBW contraction quality
        if 'bbw20_contraction_quality' in result.columns:
            quality_components.append(result['bbw20_contraction_quality'])

        # Component 2: Volume behavior (low volume = good)
        if 'volume_ratio' in result.columns:
            # Lower volume ratio = better consolidation
            vol_quality = 1.0 - np.clip(result['volume_ratio'] / 2.0, 0, 1)
            quality_components.append(vol_quality)

        # Component 3: Range compression
        if 'high' in result.columns and 'low' in result.columns:
            daily_range = (result['high'] - result['low']) / result['close']
            range_20d_avg = daily_range.rolling(20).mean()

            # Current range vs 20-day average
            range_ratio = daily_range / (range_20d_avg + 1e-10)

            # Lower ratio = tighter consolidation
            range_quality = 1.0 - np.clip(range_ratio, 0, 1)
            quality_components.append(range_quality)

        # Component 4: Price stability
        if 'close' in result.columns:
            price_std = result['close'].rolling(10).std()
            price_mean = result['close'].rolling(10).mean()
            price_cv = price_std / (price_mean + 1e-10)

            # Lower CV = more stable price
            price_stability = 1.0 / (1.0 + price_cv * 10)
            quality_components.append(price_stability)

        # Component 5: ADX (low trending = good consolidation)
        if 'adx' in result.columns:
            # ADX < 32 is ideal consolidation
            adx_quality = np.clip(1.0 - result['adx'] / 50, 0, 1)
            quality_components.append(adx_quality)

        # Combine all components
        if quality_components:
            quality_df = pd.DataFrame(quality_components).T
            result['consolidation_quality_score'] = quality_df.mean(axis=1)

            # Classify quality levels
            result['consolidation_quality_high'] = (
                (result['consolidation_quality_score'] > 0.7).astype(int)
            )
            result['consolidation_quality_medium'] = (
                ((result['consolidation_quality_score'] > 0.5) &
                 (result['consolidation_quality_score'] <= 0.7)).astype(int)
            )
            result['consolidation_quality_low'] = (
                (result['consolidation_quality_score'] <= 0.5).astype(int)
            )
        else:
            result['consolidation_quality_score'] = 0.0
            result['consolidation_quality_high'] = 0
            result['consolidation_quality_medium'] = 0
            result['consolidation_quality_low'] = 0

        return result

    def _add_ebp_features(self, df: pd.DataFrame, pattern_start_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Add Explosive Breakout Predictor (EBP) features.

        This adds 19 new features including:
        - CCI (Consolidation Compression Index) components
        - VAR (Volume Accumulation Ratio) components
        - NES (Narrative Energy Score) components
        - LPF (Liquidity Pressure Factor) components
        - TSF (Time Scaling Factor) components
        - EBP composite score

        Args:
            df: DataFrame with OHLCV and technical indicators
            pattern_start_idx: Optional index where pattern started (for TSF calculation)

        Returns:
            DataFrame with EBP features added
        """
        result = self.ebp_calculator.calculate_all_ebp_features(df, pattern_start_idx)
        return result

    def extract_pattern_features(
        self,
        df: pd.DataFrame,
        pattern_start_idx: int,
        pattern_end_idx: int
    ) -> Dict[str, float]:
        """
        Extract features specific to a consolidation pattern.

        Args:
            df: DataFrame with full data
            pattern_start_idx: Pattern start index
            pattern_end_idx: Pattern end index

        Returns:
            Dictionary with pattern-specific features
        """
        pattern_data = df.iloc[pattern_start_idx:pattern_end_idx + 1]

        if len(pattern_data) < 5:
            return {}

        features = {}

        # Pattern duration
        features['pattern_duration_days'] = len(pattern_data)

        # BBW statistics during pattern
        if 'bbw_20' in pattern_data.columns:
            features['pattern_bbw_mean'] = pattern_data['bbw_20'].mean()
            features['pattern_bbw_min'] = pattern_data['bbw_20'].min()
            features['pattern_bbw_trend'] = self._calculate_simple_slope(
                pattern_data['bbw_20'].values
            )

        # Volume statistics
        if 'volume_ratio' in pattern_data.columns:
            features['pattern_vol_ratio_mean'] = pattern_data['volume_ratio'].mean()
            features['pattern_vol_ratio_max'] = pattern_data['volume_ratio'].max()
            features['pattern_low_volume_days'] = (
                (pattern_data['volume_ratio'] < 0.70).sum()
            )

        # Price range compression
        if all(col in pattern_data.columns for col in ['high', 'low', 'close']):
            pattern_high = pattern_data['high'].max()
            pattern_low = pattern_data['low'].min()
            pattern_range_pct = (pattern_high - pattern_low) / pattern_low

            features['pattern_range_pct'] = pattern_range_pct

            # Daily range statistics
            daily_ranges = (pattern_data['high'] - pattern_data['low']) / pattern_data['close']
            features['pattern_avg_daily_range'] = daily_ranges.mean()
            features['pattern_range_compression'] = daily_ranges.std()

        # Consolidation quality during pattern
        if 'consolidation_quality_score' in pattern_data.columns:
            features['pattern_quality_mean'] = pattern_data['consolidation_quality_score'].mean()
            features['pattern_quality_min'] = pattern_data['consolidation_quality_score'].min()

        # Boundary test count
        if all(col in pattern_data.columns for col in ['high', 'low']):
            pattern_high = pattern_data['high'].max()
            pattern_low = pattern_data['low'].min()

            # Count touches of upper/lower boundaries
            upper_touches = (pattern_data['high'] >= pattern_high * 0.995).sum()
            lower_touches = (pattern_data['low'] <= pattern_low * 1.005).sum()

            features['pattern_upper_boundary_tests'] = upper_touches
            features['pattern_lower_boundary_tests'] = lower_touches
            features['pattern_total_boundary_tests'] = upper_touches + lower_touches

        return features

    def _calculate_simple_slope(self, values: np.ndarray) -> float:
        """Simple slope calculation helper."""
        if len(values) < 2:
            return 0.0

        try:
            indices = np.arange(len(values))
            slope, _ = np.polyfit(indices, values, 1)
            return slope
        except:
            return 0.0

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.feature_names

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Group features by category for interpretability.

        Returns:
            Dictionary mapping category names to feature lists
        """
        groups = {
            'robust_slopes': [],
            'vol_of_vol': [],
            'volume_accumulation': [],
            'microstructural': [],
            'consolidation_quality': [],
            'feature_interactions': [],  # NEW
            'base_features': []
        }

        for feature in self.feature_names:
            if any(x in feature for x in ['slope_theilsen', 'slope_ransac', 'slope_weighted']):
                groups['robust_slopes'].append(feature)
            elif any(x in feature for x in ['bbw', 'volatility', 'smoothness', 'acceleration']):
                groups['vol_of_vol'].append(feature)
            elif any(x in feature for x in ['vol_accum', 'vol_strength', 'consec_vol', 'pv_divergence']):
                groups['volume_accumulation'].append(feature)
            elif any(x in feature for x in ['bar_', 'body_', 'shadow_', 'candle', 'gap_', 'intrabar_',
                                            'vwap_', 'pressure', 'efficiency', 'complexity', 'smoothness']):
                groups['microstructural'].append(feature)
            elif 'quality' in feature or 'consolidation' in feature:
                groups['consolidation_quality'].append(feature)
            # NEW: Interaction features
            elif any(x in feature for x in ['_x_', 'interaction', 'divergence', 'alignment',
                                            'setup_score', 'exhaustion', 'accumulation_pattern',
                                            'conviction', 'squared', 'cubed']):
                groups['feature_interactions'].append(feature)
            else:
                groups['base_features'].append(feature)

        return groups
