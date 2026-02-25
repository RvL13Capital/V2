"""
Explosive Breakout Predictor (EBP) Feature Extraction

This module implements the EBP composite indicator for detecting explosive breakout
patterns from consolidations. Based on epidemiological modeling concepts and
multi-factor composite scoring.

EBP Formula:
    EBP = (CCI × VAR × NES × LPF) ^ (1/TSF)

Components:
    - CCI: Consolidation Compression Index (volatility contraction quality)
    - VAR: Volume Accumulation Ratio (smart money accumulation)
    - NES: Narrative Energy Score (inactive mass + wavelet energy)
    - LPF: Liquidity Pressure Factor (bid pressure × volume pressure)
    - TSF: Time Scaling Factor (consolidation duration weighting)

Implementation Notes:
    - Uses OHLCV data only (no fundamental data required)
    - Simplified wavelet energy using rolling std at multiple timeframes
    - Sector RS approximated using SPY benchmark
    - Float turnover approximated using volume ratios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class EBPFeatureCalculator:
    """
    Calculate Explosive Breakout Predictor (EBP) features.

    Provides both individual components (CCI, VAR, NES, LPF, TSF) and
    the composite EBP score for pattern ranking and ML feature engineering.
    """

    def __init__(
        self,
        bbw_lookback: int = 60,
        atr_lookback: int = 60,
        var_window: int = 20,
        volume_baseline: int = 250,
        wavelet_windows: List[int] = None,
        liquidity_windows: List[int] = None
    ):
        """
        Initialize EBP calculator.

        Args:
            bbw_lookback: Lookback period for BBW max calculation
            atr_lookback: Lookback period for ATR max calculation
            var_window: Window for Volume Accumulation Ratio
            volume_baseline: Baseline period for inactive mass calculation
            wavelet_windows: Windows for wavelet energy approximation [5, 10, 20]
            liquidity_windows: Windows for liquidity pressure [5, 50]
        """
        self.bbw_lookback = bbw_lookback
        self.atr_lookback = atr_lookback
        self.var_window = var_window
        self.volume_baseline = volume_baseline
        self.wavelet_windows = wavelet_windows or [5, 10, 20]
        self.liquidity_windows = liquidity_windows or [5, 50]

    def calculate_all_ebp_features(
        self,
        df: pd.DataFrame,
        pattern_start_idx: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate all EBP features.

        Args:
            df: DataFrame with OHLCV data and technical indicators
            pattern_start_idx: Index where pattern started (for days_in_consolidation)

        Returns:
            DataFrame with EBP features added
        """
        result = df.copy()

        # Calculate individual components
        result = self._calculate_cci(result)
        result = self._calculate_var(result)
        result = self._calculate_nes(result)
        result = self._calculate_lpf(result)
        result = self._calculate_tsf(result, pattern_start_idx)

        # Calculate composite EBP score
        result = self._calculate_ebp_composite(result)

        return result

    def _calculate_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Consolidation Compression Index (CCI).

        Formula:
            CCI = (1 - BBW_current/BBW_max60) × (1 - ATR20/ATR60) × sqrt(days_in_range/20)

        Approximations:
            - days_in_range: Estimated using rolling count of low volatility days
            - BBW from existing 'bbw_20' column or calculated
            - ATR from price range data
        """
        result = df.copy()

        # Get or calculate BBW
        if 'bbw_20' in df.columns:
            bbw_current = df['bbw_20']
        elif 'bb_width_20' in df.columns:
            bbw_current = df['bb_width_20']
        else:
            # Calculate BBW if not present
            sma = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            bbw_current = ((std * 2 * 2) / sma) * 100

        # BBW compression component
        bbw_max = bbw_current.rolling(window=self.bbw_lookback).max()
        bbw_compression = 1 - (bbw_current / (bbw_max + 1e-10))
        bbw_compression = bbw_compression.clip(0, 1)

        # ATR compression component
        true_range = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)

        atr_20 = true_range.rolling(window=20).mean()
        atr_60 = true_range.rolling(window=60).mean()
        atr_compression = 1 - (atr_20 / (atr_60 + 1e-10))
        atr_compression = atr_compression.clip(0, 1)

        # Days in range component (approximate)
        # Count consecutive days with BBW below median
        bbw_median = bbw_current.rolling(window=60).median()
        below_median = (bbw_current < bbw_median).astype(int)

        # OPTIMIZED: Vectorized consecutive days calculation (was O(n) loop, now O(1))
        # Create groups that change when below_median value changes
        groups = (below_median != below_median.shift()).cumsum()
        # Within each group, cumsum gives consecutive count, then multiply by below_median to zero out groups where it's 0
        days_in_range = below_median.groupby(groups).cumsum() * below_median

        days_factor = np.sqrt(days_in_range / 20.0).clip(0, 2.0)

        # Composite CCI
        result['cci_bbw_compression'] = bbw_compression
        result['cci_atr_compression'] = atr_compression
        result['cci_days_factor'] = days_factor
        result['cci_score'] = bbw_compression * atr_compression * days_factor

        return result

    def _calculate_var(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Accumulation Ratio (VAR).

        Formula:
            VAR = Σ(Volume_up × Price_change_up) / Σ(Volume_down × |Price_change_down|)
            VAR_normalized = VAR / VAR_MA20

        Measures smart money accumulation vs distribution.
        """
        result = df.copy()

        # Calculate price changes
        price_change = df['close'].pct_change()

        # Separate up and down days
        volume_up = np.where(price_change > 0, df['volume'] * price_change, 0)
        volume_down = np.where(price_change < 0, df['volume'] * abs(price_change), 0)

        # Rolling sums for VAR calculation (preserve index)
        sum_volume_up = pd.Series(volume_up, index=df.index).rolling(window=self.var_window).sum()
        sum_volume_down = pd.Series(volume_down, index=df.index).rolling(window=self.var_window).sum()

        # VAR ratio
        var_raw = sum_volume_up / (sum_volume_down + 1e-10)

        # Normalize by 20-day moving average
        var_ma20 = var_raw.rolling(window=20).mean()
        var_normalized = var_raw / (var_ma20 + 1e-10)

        # Clip to reasonable range (0.5 to 3.0)
        var_normalized = var_normalized.clip(0.5, 3.0)

        result['var_raw'] = var_raw
        result['var_score'] = var_normalized

        return result

    def _calculate_nes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Narrative Energy Score (NES).

        Formula:
            IM = (1 - Volume_current/Volume_avg250) × MarketCap
            WE = wavelet_energy_5d + wavelet_energy_10d + wavelet_energy_20d
            RSA = (Stock_RS - Sector_RS) / Sector_StdDev
            NES = (IM^0.3 × WE^0.4 × (1 + RSA)^0.3)

        Approximations (OHLCV only):
            - MarketCap: Approximated as Price × Volume (relative measure)
            - Wavelet Energy: Approximated using rolling std (volatility at multiple timeframes)
            - Sector_RS: Ignored (set RSA = 0) or use simple momentum as proxy
        """
        result = df.copy()

        # Inactive Mass (IM) - Simplified without market cap
        volume_avg = df['volume'].rolling(window=self.volume_baseline).mean()
        volume_ratio = df['volume'] / (volume_avg + 1e-10)
        inactive_mass_factor = (1 - volume_ratio.clip(0, 2)).clip(0, 1)

        # Approximate market cap as price * volume (relative measure)
        pseudo_market_cap = df['close'] * df['volume']
        inactive_mass = inactive_mass_factor * pseudo_market_cap

        # OPTIMIZED: Normalize to 0-1 range using rank (was slow percentileofscore)
        inactive_mass_normalized = inactive_mass.rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5,
            raw=True
        )

        # Wavelet Energy (WE) - Approximated using multi-timeframe volatility
        wavelet_energies = []
        for window in self.wavelet_windows:
            # Use rolling std as proxy for wavelet energy at this frequency
            energy = df['close'].pct_change().rolling(window=window).std()
            energy_normalized = energy / (energy.rolling(window=50).mean() + 1e-10)
            wavelet_energies.append(energy_normalized)

        wavelet_energy_total = sum(wavelet_energies) / len(wavelet_energies)

        # Relative Strength Anomaly (RSA) - Simplified
        # Use momentum as proxy (no sector data available)
        momentum_20 = df['close'].pct_change(20)
        momentum_mean = momentum_20.rolling(window=100).mean()
        momentum_std = momentum_20.rolling(window=100).std()
        rsa_proxy = (momentum_20 - momentum_mean) / (momentum_std + 1e-10)
        rsa_proxy = rsa_proxy.clip(-3, 3)  # Clip to ±3 sigma

        # Composite NES
        # NES = IM^0.3 × WE^0.4 × (1 + RSA)^0.3
        im_component = np.power(inactive_mass_normalized.clip(1e-10, 1), 0.3)
        we_component = np.power(wavelet_energy_total.clip(0.1, 10), 0.4)
        rsa_component = np.power((1 + rsa_proxy).clip(0.1, 10), 0.3)

        result['nes_inactive_mass'] = inactive_mass_normalized
        result['nes_wavelet_energy'] = wavelet_energy_total
        result['nes_rsa_proxy'] = rsa_proxy
        result['nes_score'] = im_component * we_component * rsa_component

        return result

    def _calculate_lpf(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Liquidity Pressure Factor (LPF).

        Formula:
            BP = (Close - Low) / (High - Low)  # Bid Pressure
            VP = Volume_5d / Volume_50d         # Volume Pressure
            FTA = (Turnover_5d - Turnover_20d) / Turnover_20d  # Float Turnover Acceleration
            LPF = BP × VP × (1 + FTA)

        Approximations:
            - FTA: Use volume acceleration as proxy for float turnover
        """
        result = df.copy()

        # Bid Pressure (BP)
        daily_range = df['high'] - df['low']
        bid_pressure = np.where(
            daily_range > 1e-10,
            (df['close'] - df['low']) / daily_range,
            0.5  # Neutral if no range
        )

        # Smooth bid pressure (preserve index)
        bp_smoothed = pd.Series(bid_pressure, index=df.index).rolling(window=5).mean()

        # Volume Pressure (VP)
        volume_5d = df['volume'].rolling(window=self.liquidity_windows[0]).mean()
        volume_50d = df['volume'].rolling(window=self.liquidity_windows[1]).mean()
        volume_pressure = volume_5d / (volume_50d + 1e-10)

        # Float Turnover Acceleration (FTA) - Approximated
        # Use volume acceleration as proxy
        volume_20d = df['volume'].rolling(window=20).mean()
        fta_proxy = (volume_5d - volume_20d) / (volume_20d + 1e-10)
        fta_proxy = fta_proxy.clip(-1, 3)  # Clip to reasonable range

        # Composite LPF
        result['lpf_bid_pressure'] = bp_smoothed
        result['lpf_volume_pressure'] = volume_pressure
        result['lpf_fta_proxy'] = fta_proxy
        result['lpf_score'] = bp_smoothed * volume_pressure * (1 + fta_proxy)

        return result

    def _calculate_tsf(
        self,
        df: pd.DataFrame,
        pattern_start_idx: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate Time Scaling Factor (TSF).

        Formula:
            TSF = 1 + sqrt(days_in_consolidation / 30)

        Note: If pattern_start_idx is not provided, estimate using
        consecutive low-volatility days.
        """
        result = df.copy()

        if pattern_start_idx is not None:
            # Calculate exact days in consolidation
            days_in_consolidation = pd.Series(0, index=df.index)
            for i in range(pattern_start_idx, len(df)):
                days_in_consolidation.iloc[i] = i - pattern_start_idx
        else:
            # Estimate using consecutive low-BBW days (from CCI calculation)
            if 'cci_days_factor' in df.columns:
                # Use the days count from CCI calculation
                days_in_consolidation = (df['cci_days_factor'] ** 2) * 20
            else:
                # Fallback: use a simple estimate
                days_in_consolidation = pd.Series(0, index=df.index)

        # Calculate TSF
        tsf = 1 + np.sqrt(days_in_consolidation / 30.0)

        result['tsf_days_in_consolidation'] = days_in_consolidation
        result['tsf_score'] = tsf

        return result

    def _calculate_ebp_composite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite EBP score.

        Formula:
            EBP = (CCI × VAR × NES × LPF) ^ (1/TSF)

        All components should be normalized to similar scales (0.1 to 3.0 typically).
        """
        result = df.copy()

        # Check that all components exist
        required_cols = ['cci_score', 'var_score', 'nes_score', 'lpf_score', 'tsf_score']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Not all EBP components calculated. Setting EBP to NaN.")
            result['ebp_composite'] = np.nan
            return result

        # Calculate composite
        # Product of components
        product = (
            df['cci_score'].clip(0.01, 10) *
            df['var_score'].clip(0.1, 10) *
            df['nes_score'].clip(0.1, 10) *
            df['lpf_score'].clip(0.1, 10)
        )

        # Apply time scaling factor (root)
        tsf_safe = df['tsf_score'].clip(1.0, 10)
        ebp = np.power(product.clip(1e-10, 1000), 1 / tsf_safe)

        # OPTIMIZED: Normalize to 0-1 range using rank (was slow percentileofscore)
        # rank(pct=True) returns percentile rank (0-1) within rolling window
        ebp_normalized = ebp.rolling(window=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 10 else 0.5,
            raw=True
        )

        result['ebp_raw'] = ebp
        result['ebp_composite'] = ebp_normalized

        # Create signal categories
        result['ebp_signal'] = pd.cut(
            ebp_normalized,
            bins=[0, 0.5, 0.7, 0.85, 0.95, 1.0],
            labels=['WEAK', 'MODERATE', 'GOOD', 'STRONG', 'EXCEPTIONAL']
        )

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of all EBP feature names."""
        return [
            # CCI components
            'cci_bbw_compression',
            'cci_atr_compression',
            'cci_days_factor',
            'cci_score',

            # VAR components
            'var_raw',
            'var_score',

            # NES components
            'nes_inactive_mass',
            'nes_wavelet_energy',
            'nes_rsa_proxy',
            'nes_score',

            # LPF components
            'lpf_bid_pressure',
            'lpf_volume_pressure',
            'lpf_fta_proxy',
            'lpf_score',

            # TSF components
            'tsf_days_in_consolidation',
            'tsf_score',

            # Composite
            'ebp_raw',
            'ebp_composite',
            'ebp_signal'
        ]
