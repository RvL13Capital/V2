"""
Canonical Feature Extractor - Single Source of Truth for AIv5
==============================================================

This is THE ONLY feature extractor for AIv5. All training, validation,
and production use this exact same code.

Features (69 total):
- 26 core pattern features
- 19 EBP composite features
- 24 derived/metadata features

CRITICAL: This extractor ensures training/validation consistency.
          DO NOT use any other feature extraction methods.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import required modules
from shared.indicators.technical import (
    calculate_bbw,
    calculate_adx,
    calculate_volume_ratio,
    get_bbw_percentile,
    calculate_ad_line
)
from pattern_detection.features.ebp_features_incremental import IncrementalEBPCalculator

logger = logging.getLogger(__name__)


class CanonicalFeatureExtractor:
    """
    Single unified feature extractor ensuring training/validation consistency.

    This replaces:
    - enhanced_tabular_features.py
    - The inline calculations in src/pipeline/01_scan_patterns.py
    - The temporal_safe feature extraction

    Usage:
        extractor = CanonicalFeatureExtractor()
        features_dict = extractor.extract_snapshot_features(
            df_ohlcv,
            snapshot_date,
            pattern
        )
    """

    # Define the exact 69 feature names in canonical order
    FEATURE_NAMES = [
        # === Core Pattern Features (26) ===
        # Pattern lifecycle
        'days_since_activation',
        'days_in_pattern',
        'consolidation_channel_width_pct',
        'consolidation_period_days',

        # Current price/volume state
        'current_price',
        'current_high',
        'current_low',
        'current_volume',

        # Current indicators
        'current_bbw_20',
        'current_bbw_percentile',
        'current_adx',
        'current_volume_ratio_20',
        'current_range_ratio',

        # Baseline statistics
        'baseline_bbw_avg',
        'baseline_volume_avg',
        'baseline_volatility',

        # Compression metrics
        'bbw_compression_ratio',
        'volume_compression_ratio',
        'volatility_compression_ratio',

        # Trend indicators
        'bbw_slope_20d',
        'adx_slope_20d',

        # Pattern quality
        'consolidation_quality_score',

        # Price position relative to boundaries
        'price_position_in_range',
        'price_distance_from_upper_pct',
        'price_distance_from_lower_pct',
        'distance_from_power_pct',

        # === EBP Features (19) ===
        # CCI (Consolidation Compression Index)
        'cci_bbw_compression',
        'cci_atr_compression',
        'cci_days_factor',
        'cci_score',

        # VAR (Volume Accumulation Ratio)
        'var_raw',
        'var_score',

        # NES (Narrative Energy Score)
        'nes_inactive_mass',
        'nes_wavelet_energy',
        'nes_rsa_proxy',
        'nes_score',

        # LPF (Liquidity Pressure Factor)
        'lpf_bid_pressure',
        'lpf_volume_pressure',
        'lpf_fta_proxy',
        'lpf_score',

        # TSF (Time Scaling Factor)
        'tsf_days_in_consolidation',
        'tsf_score',

        # EBP Composite
        'ebp_raw',
        'ebp_composite',
        'ebp_signal',

        # === Derived Features (24) ===
        # Recent window statistics
        'avg_range_20d',
        'bbw_std_20d',
        'price_volatility_20d',

        # Pattern boundaries (metadata)
        'start_price',
        'upper_boundary',
        'lower_boundary',
        'power_boundary',

        # Volume analysis
        'volume_trend_10d',
        'volume_spike',
        'volume_vs_50d',

        # Price dynamics
        'daily_volatility',
        'high_low_spread',
        'close_vs_vwap',

        # Moving averages
        'ma_5',
        'ma_10',
        'ma_20',

        # Price vs MAs
        'price_vs_ma5',
        'price_vs_ma20',

        # Momentum
        'price_momentum_20d',

        # Additional pattern metrics
        'days_since_bbw_low',
        'days_since_volume_spike',
        'channel_narrowing_pct',

        # Additional metrics for 69 total
        'bbw_acceleration',
        'days_in_qualification'
    ]

    def __init__(self):
        """Initialize the canonical feature extractor."""
        # Initialize EBP calculator (incremental version for performance)
        self.ebp_calculator = IncrementalEBPCalculator()
        self.feature_count = len(self.FEATURE_NAMES)
        logger.info(f"CanonicalFeatureExtractor initialized with {self.feature_count} features")

    def extract_snapshot_features(
        self,
        df: pd.DataFrame,
        snapshot_date: pd.Timestamp,
        pattern: object,  # ConsolidationPattern object
        snapshot_idx: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Extract ALL 69 features for a snapshot.

        Args:
            df: OHLCV DataFrame with calculated indicators (temporal-safe: only dates <= snapshot_date)
            snapshot_date: The date of the snapshot
            pattern: ConsolidationPattern object with boundaries and metadata
            snapshot_idx: Optional index in df for the snapshot date (for optimization)

        Returns:
            Dictionary with exactly 69 features matching FEATURE_NAMES
        """
        # Get snapshot index if not provided
        if snapshot_idx is None:
            if snapshot_date in df.index:
                snapshot_idx = df.index.get_loc(snapshot_date)
            else:
                raise ValueError(f"Snapshot date {snapshot_date} not found in DataFrame")

        # Ensure we only use data up to snapshot_date (temporal safety)
        df_safe = df.iloc[:snapshot_idx + 1].copy()
        row = df_safe.iloc[-1]  # Current snapshot row

        features = {}

        # === 1. Core Pattern Features (26) ===
        features['days_since_activation'] = (snapshot_date - pattern.activation_date).days if pattern.activation_date else 0
        features['days_in_pattern'] = (snapshot_date - pattern.start_date).days

        # Consolidation metrics
        if pattern.upper_boundary and pattern.lower_boundary:
            channel_width = pattern.upper_boundary - pattern.lower_boundary
            features['consolidation_channel_width_pct'] = (channel_width / pattern.lower_boundary) * 100
        else:
            features['consolidation_channel_width_pct'] = 0.0

        features['consolidation_period_days'] = features['days_since_activation']

        # Current state
        features['current_price'] = row['close']
        features['current_high'] = row['high']
        features['current_low'] = row['low']
        features['current_volume'] = row['volume']

        # Current indicators (should be pre-calculated in df)
        features['current_bbw_20'] = row.get('bbw_20', 0.0)
        features['current_bbw_percentile'] = row.get('bbw_percentile', 0.0)
        features['current_adx'] = row.get('adx', 0.0)
        features['current_volume_ratio_20'] = row.get('volume_ratio_20', 1.0)
        features['current_range_ratio'] = row.get('range_ratio', 1.0)

        # Baseline statistics (pre-calculated)
        features['baseline_bbw_avg'] = row.get('baseline_bbw_avg', 0.0)
        features['baseline_volume_avg'] = row.get('baseline_volume_avg', 0.0)
        features['baseline_volatility'] = row.get('baseline_volatility', 0.0)

        # Compression ratios
        features['bbw_compression_ratio'] = row.get('bbw_compression_ratio', 0.0)
        features['volume_compression_ratio'] = row.get('volume_compression_ratio', 0.0)
        features['volatility_compression_ratio'] = row.get('volatility_compression_ratio', 0.0)

        # Trend slopes
        features['bbw_slope_20d'] = row.get('bbw_slope_20d', 0.0)
        features['adx_slope_20d'] = row.get('adx_slope_20d', 0.0)

        # Quality score
        features['consolidation_quality_score'] = row.get('consolidation_quality_score', 0.0)

        # Price position relative to boundaries
        current_price = row['close']
        if pattern.upper_boundary and pattern.lower_boundary:
            range_width = pattern.upper_boundary - pattern.lower_boundary
            features['price_position_in_range'] = (current_price - pattern.lower_boundary) / (range_width + 1e-10)
            features['price_distance_from_upper_pct'] = ((pattern.upper_boundary - current_price) / (current_price + 1e-10)) * 100
            features['price_distance_from_lower_pct'] = ((current_price - pattern.lower_boundary) / (current_price + 1e-10)) * 100
            features['distance_from_power_pct'] = ((pattern.power_boundary - current_price) / (current_price + 1e-10)) * 100
        else:
            features['price_position_in_range'] = 0.0
            features['price_distance_from_upper_pct'] = 0.0
            features['price_distance_from_lower_pct'] = 0.0
            features['distance_from_power_pct'] = 0.0

        # === 2. EBP Features (19) ===
        # Get EBP features from the incremental calculator
        ebp_features = self._extract_ebp_features(df_safe, pattern)
        features.update(ebp_features)

        # === 3. Derived Features (24) ===
        # Recent window statistics
        features['avg_range_20d'] = row.get('avg_range_20d', 0.0)
        features['bbw_std_20d'] = row.get('bbw_std_20d', 0.0)
        features['price_volatility_20d'] = row.get('price_volatility_20d', 0.0)

        # Pattern boundaries
        features['start_price'] = pattern.start_price if pattern.start_price else row['close']
        features['upper_boundary'] = pattern.upper_boundary if pattern.upper_boundary else 0.0
        features['lower_boundary'] = pattern.lower_boundary if pattern.lower_boundary else 0.0
        features['power_boundary'] = pattern.power_boundary if pattern.power_boundary else 0.0

        # Volume analysis
        features['volume_trend_10d'] = self._calculate_trend(df_safe['volume'], 10)
        features['volume_spike'] = self._calculate_volume_spike(df_safe)
        features['volume_vs_50d'] = self._calculate_volume_vs_ma(df_safe, 50)

        # Price dynamics
        features['daily_volatility'] = row.get('daily_range', 0.0) / row['close'] if row['close'] > 0 else 0.0
        features['high_low_spread'] = (row['high'] - row['low']) / row['close'] if row['close'] > 0 else 0.0
        features['close_vs_vwap'] = self._calculate_close_vs_vwap(row)

        # Moving averages
        features['ma_5'] = df_safe['close'].rolling(5, min_periods=1).mean().iloc[-1]
        features['ma_10'] = df_safe['close'].rolling(10, min_periods=1).mean().iloc[-1]
        features['ma_20'] = df_safe['close'].rolling(20, min_periods=1).mean().iloc[-1]

        # Price vs MAs
        features['price_vs_ma5'] = (row['close'] - features['ma_5']) / features['ma_5'] if features['ma_5'] > 0 else 0.0
        features['price_vs_ma20'] = (row['close'] - features['ma_20']) / features['ma_20'] if features['ma_20'] > 0 else 0.0

        # Momentum
        features['price_momentum_20d'] = self._calculate_momentum(df_safe['close'], 20)

        # Additional pattern metrics
        features['days_since_bbw_low'] = self._calculate_days_since_low(df_safe, 'bbw_20')
        features['days_since_volume_spike'] = self._calculate_days_since_spike(df_safe)
        features['channel_narrowing_pct'] = self._calculate_channel_narrowing(df_safe, pattern)

        # Additional metrics for 69 total
        features['bbw_acceleration'] = self._calculate_bbw_acceleration(df_safe)
        features['days_in_qualification'] = pattern.days_qualifying if hasattr(pattern, 'days_qualifying') else 10

        # Fill NaN values with 0
        features = {k: (v if pd.notna(v) else 0.0) for k, v in features.items()}

        # Validate we have exactly 69 features
        if len(features) != self.feature_count:
            missing = set(self.FEATURE_NAMES) - set(features.keys())
            extra = set(features.keys()) - set(self.FEATURE_NAMES)
            if missing:
                logger.warning(f"Missing features: {missing}")
            if extra:
                logger.warning(f"Extra features: {extra}")
            raise ValueError(f"Expected {self.feature_count} features, got {len(features)}")

        # Return features in canonical order
        return {name: features[name] for name in self.FEATURE_NAMES}

    def _extract_ebp_features(self, df: pd.DataFrame, pattern: object) -> Dict[str, float]:
        """
        Extract EBP features using the incremental calculator.

        Returns dict with 19 EBP features.
        """
        # Reset calculator for new pattern
        self.ebp_calculator.reset()

        # Process data up to snapshot date
        for idx in range(len(df)):
            current_row = df.iloc[idx].to_dict()
            prev_row = df.iloc[idx-1].to_dict() if idx > 0 else None

            # Update EBP calculator
            self.ebp_calculator.update_and_calculate(
                current_row,
                prev_row,
                pattern_start_idx=0,  # Relative to df start
                current_idx=idx
            )

        # Get final EBP features
        ebp_features = self.ebp_calculator.get_current_features()

        # Ensure we have all 19 EBP features
        required_ebp = [
            'cci_bbw_compression', 'cci_atr_compression', 'cci_days_factor', 'cci_score',
            'var_raw', 'var_score',
            'nes_inactive_mass', 'nes_wavelet_energy', 'nes_rsa_proxy', 'nes_score',
            'lpf_bid_pressure', 'lpf_volume_pressure', 'lpf_fta_proxy', 'lpf_score',
            'tsf_days_in_consolidation', 'tsf_score',
            'ebp_raw', 'ebp_composite', 'ebp_signal'
        ]

        # Fill missing with 0
        for feat in required_ebp:
            if feat not in ebp_features:
                ebp_features[feat] = 0.0

        # Convert ebp_signal to numeric if it's a string
        if 'ebp_signal' in ebp_features and isinstance(ebp_features['ebp_signal'], str):
            signal_map = {'WEAK': 1.0, 'MODERATE': 2.0, 'GOOD': 3.0, 'STRONG': 4.0, 'EXCEPTIONAL': 5.0}
            ebp_features['ebp_signal'] = signal_map.get(ebp_features['ebp_signal'], 0.0)

        return ebp_features

    def _calculate_trend(self, series: pd.Series, window: int) -> float:
        """Calculate trend slope over window."""
        if len(series) < window:
            return 0.0

        recent = series.tail(window)
        x = np.arange(len(recent))
        try:
            slope = np.polyfit(x, recent, 1)[0]
            return slope / recent.mean() if recent.mean() != 0 else 0.0
        except:
            return 0.0

    def _calculate_volume_spike(self, df: pd.DataFrame) -> float:
        """Calculate if current volume is a spike vs 20-day average."""
        if 'volume' not in df.columns or len(df) < 20:
            return 0.0

        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume'].iloc[-20:].mean()
        return (current_vol / avg_vol) if avg_vol > 0 else 0.0

    def _calculate_volume_vs_ma(self, df: pd.DataFrame, period: int) -> float:
        """Calculate current volume vs moving average."""
        if 'volume' not in df.columns or len(df) < period:
            return 1.0

        current_vol = df['volume'].iloc[-1]
        ma_vol = df['volume'].rolling(period, min_periods=1).mean().iloc[-1]
        return (current_vol / ma_vol) if ma_vol > 0 else 1.0

    def _calculate_close_vs_vwap(self, row: pd.Series) -> float:
        """Calculate close vs VWAP approximation."""
        vwap = (row['high'] + row['low'] + row['close']) / 3
        return ((row['close'] - vwap) / vwap) if vwap > 0 else 0.0

    def _calculate_momentum(self, series: pd.Series, period: int) -> float:
        """Calculate price momentum over period."""
        if len(series) < period + 1:
            return 0.0

        current = series.iloc[-1]
        past = series.iloc[-period-1]
        return ((current - past) / past) if past > 0 else 0.0

    def _calculate_days_since_low(self, df: pd.DataFrame, column: str) -> int:
        """Calculate days since column was at minimum."""
        if column not in df.columns or len(df) < 20:
            return 0

        recent = df[column].tail(60)
        min_idx = recent.idxmin()
        days_since = len(df) - df.index.get_loc(min_idx) - 1
        return max(0, days_since)

    def _calculate_days_since_spike(self, df: pd.DataFrame) -> int:
        """Calculate days since last volume spike (>2x average)."""
        if 'volume' not in df.columns or len(df) < 20:
            return 0

        for i in range(len(df)-1, max(0, len(df)-60), -1):
            current_vol = df['volume'].iloc[i]
            avg_vol = df['volume'].iloc[max(0, i-20):i].mean()
            if avg_vol > 0 and current_vol > 2 * avg_vol:
                return len(df) - i - 1

        return 60  # No spike found in last 60 days

    def _calculate_channel_narrowing(self, df: pd.DataFrame, pattern: object) -> float:
        """Calculate if channel is narrowing over time."""
        if not pattern.upper_boundary or not pattern.lower_boundary:
            return 0.0

        # Current channel width
        current_width = pattern.upper_boundary - pattern.lower_boundary

        # Initial channel width (approximate from volatility)
        if len(df) > 20:
            initial_vol = df['close'].iloc[:20].std()
            initial_width = initial_vol * 2  # Approximate
            if initial_width > 0:
                return ((initial_width - current_width) / initial_width) * 100

        return 0.0

    def _calculate_bbw_acceleration(self, df: pd.DataFrame) -> float:
        """Calculate BBW acceleration (second derivative)."""
        if 'bbw_20' not in df.columns or len(df) < 10:
            return 0.0

        # Calculate first derivative (change in BBW)
        bbw_diff = df['bbw_20'].diff()

        # Calculate second derivative (acceleration)
        if len(bbw_diff) >= 2:
            # Use last 5 days for smoothing
            recent_accel = bbw_diff.tail(5).diff().mean()
            return recent_accel if pd.notna(recent_accel) else 0.0

        return 0.0

    def get_feature_names(self) -> List[str]:
        """Return ordered list of all 69 feature names."""
        return self.FEATURE_NAMES.copy()

    def validate_features(self, features: Dict[str, float]) -> bool:
        """
        Validate that features dictionary contains all required features.

        Args:
            features: Dictionary of features to validate

        Returns:
            True if valid, raises ValueError if not
        """
        missing = set(self.FEATURE_NAMES) - set(features.keys())
        extra = set(features.keys()) - set(self.FEATURE_NAMES)

        if missing:
            raise ValueError(f"Missing required features: {missing}")
        if extra:
            raise ValueError(f"Unexpected extra features: {extra}")

        return True