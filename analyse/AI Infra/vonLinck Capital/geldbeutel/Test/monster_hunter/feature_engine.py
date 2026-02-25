"""
feature_engine.py

Transforms raw OHLCV data into "Tension" features for the Coil strategy.
Identifies "forgotten" micro-cap stocks with high potential energy (compression).

Philosophy: We are mean-reversion traders. Buy dead, sell alive.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoilDetector:
    """
    Identifies 'The Coil': A period of extreme tension in micro-cap stocks.

    Coil Conditions (ALL must be true):
    1. COMPRESSION: Price range below threshold over lookback window
       - Price >= $1.00: Max 6% range (tight coil)
       - Price < $1.00: Max 15% range (account for wide spreads)
    2. DORMANT: ADX < 20 (trend is dead)
    3. DEAD VOLUME: RVOL < 0.5 (volume must be DEAD)

    The RVOL < 0.5 requirement is CRITICAL: It confirms that the wide penny stock
    range is just bid/ask spread noise, NOT active selling pressure.
    """

    def __init__(self, window: int = 40, max_adx: float = 20.0, max_rvol: float = 0.5):
        """
        Args:
            window: Lookback period for range calculation (default 40 days)
            max_adx: Maximum ADX for "dormant" classification (default 20)
            max_rvol: Maximum RVOL for "dead volume" (default 0.5)
        """
        self.window = window
        self.max_adx = max_adx
        self.max_rvol = max_rvol
        # Price-aware range thresholds
        self.max_range_high = 0.06  # 6% for stocks >= $1.00
        self.max_range_low = 0.15   # 15% for stocks < $1.00

    def check_coil(self, df: pd.DataFrame, idx: int) -> Tuple[bool, Dict]:
        """
        Checks if the stock is currently in a valid Coil at index 'idx'.

        Args:
            df: DataFrame with columns: high, low, close, volume, adx, rvol_30d
            idx: Index position to check (typically current day)

        Returns:
            (is_coiled, metrics_dict)
            - is_coiled: True if all coil conditions are met
            - metrics_dict: Contains range_pct, adx, rvol, support_level, max_range_used
        """
        if idx < self.window:
            return False, {}

        # Get the lookback window ENDING at idx
        subset = df.iloc[idx - self.window + 1: idx + 1]

        if subset.empty or len(subset) < self.window:
            return False, {}

        # Current price for threshold selection
        current_close = df.iloc[idx]['close']
        if current_close <= 0:
            return False, {}

        # 1. CALCULATE PRICE RANGE COMPRESSION
        period_high = subset['high'].max()
        period_low = subset['low'].min()
        range_pct = (period_high - period_low) / current_close

        # Price-aware threshold
        if current_close >= 1.00:
            max_range = self.max_range_high  # 6% for stocks >= $1
        else:
            max_range = self.max_range_low   # 15% for penny stocks

        # 2. GET ADX (assumes pre-calculated in df)
        current_adx = df.iloc[idx].get('adx', 100)  # Default high = no trade

        # 3. GET RVOL (assumes pre-calculated in df)
        current_rvol = df.iloc[idx].get('rvol_30d', 100)  # Default high = no trade

        # Build metrics dict
        metrics = {
            'range_pct': range_pct,
            'max_range_used': max_range,
            'adx': current_adx,
            'rvol': current_rvol,
            'support_level': period_low,  # Bottom of the coil - entry target
            'resistance_level': period_high,
            'current_price': current_close
        }

        # COIL CONDITIONS (ALL must be true)
        is_compressed = range_pct <= max_range
        is_dormant = current_adx <= self.max_adx
        is_dead = current_rvol <= self.max_rvol

        is_coiled = is_compressed and is_dormant and is_dead

        return is_coiled, metrics

    def get_support_level(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        """
        Returns the support level (bottom of the coil range) for limit order placement.
        """
        if idx < self.window:
            return None

        subset = df.iloc[idx - self.window + 1: idx + 1]
        if subset.empty:
            return None

        return float(subset['low'].min())


def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculates Average Directional Index (ADX) manually.
    No external dependencies (no ta-lib required).

    ADX measures trend strength (0-100):
    - ADX < 20: No trend (dormant) - WHAT WE WANT
    - ADX 20-40: Weak trend
    - ADX > 40: Strong trend

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        window: Smoothing period (default 14)

    Returns:
        pd.Series of ADX values
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # 1. TRUE RANGE (TR)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed TR (Wilder's smoothing)
    atr = tr.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    # 2. DIRECTIONAL MOVEMENT (+DM and -DM)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # +DM: Up move is positive and greater than down move
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    # -DM: Down move is positive and greater than up move
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # 3. SMOOTHED DIRECTIONAL MOVEMENT (Wilder's smoothing)
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1/window, min_periods=window, adjust=False
    ).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1/window, min_periods=window, adjust=False
    ).mean()

    # 4. DIRECTIONAL INDICATORS (+DI and -DI)
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    # 5. DIRECTIONAL INDEX (DX)
    dx_sum = plus_di + minus_di
    dx_sum = dx_sum.replace(0, np.nan)  # Avoid division by zero
    dx = 100 * abs(plus_di - minus_di) / dx_sum

    # 6. ADX (Smoothed DX)
    adx = dx.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    # Fill NaN with neutral value (50 = moderate, won't pass our <20 filter)
    return adx.fillna(50)


def calculate_rvol(volume_series: pd.Series, window: int = 30) -> pd.Series:
    """
    RVOL = Current Volume / Average Volume over window.

    For Coil strategy, we want RVOL < 0.5 (volume is DEAD).
    """
    avg_vol = volume_series.rolling(window=window).mean()
    rvol = volume_series / avg_vol.replace(0, np.nan)
    return rvol.fillna(1.0)  # Default to 1.0 (average) if NaN


def calculate_dollar_volume(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Calculate rolling average dollar volume.
    Used for liquidity filtering ($10k-$100k for truly dormant stocks).
    """
    dollar_vol = df['close'] * df['volume']
    return dollar_vol.rolling(window=window).mean()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pipeline to attach Coil detection features.

    Adds columns:
    - dollar_vol: Daily dollar volume
    - dollar_vol_30d: 30-day average dollar volume
    - rvol_30d: Relative volume (current / 30-day avg)
    - adx: Average Directional Index (14-day)
    """
    df = df.copy()

    # Dollar volume for liquidity filtering
    df['dollar_vol'] = df['close'] * df['volume']
    df['dollar_vol_30d'] = calculate_dollar_volume(df, window=30)

    # RVOL - must be < 0.5 for coil
    df['rvol_30d'] = calculate_rvol(df['volume'], window=30)

    # ADX - must be < 20 for coil
    df['adx'] = calculate_adx(df, window=14)

    return df


def validate_coil_candidate(row: pd.Series, min_dollar_vol: float = 10000, max_dollar_vol: float = 100000) -> Tuple[bool, list]:
    """
    Hard filters for Coil candidates.

    For truly dormant micro-caps:
    - Dollar Vol (30d avg) between $10k and $100k
    - Price > $0.01 (not completely worthless)
    """
    reasons = []

    # Liquidity floor
    dollar_vol = row.get('dollar_vol_30d', 0)
    if dollar_vol < min_dollar_vol:
        reasons.append(f"Liquidity < ${min_dollar_vol/1000:.0f}k")

    # Liquidity ceiling (too liquid = not a true micro-cap)
    if dollar_vol > max_dollar_vol:
        reasons.append(f"Liquidity > ${max_dollar_vol/1000:.0f}k (too liquid)")

    # Price floor
    if row.get('close', 0) < 0.01:
        reasons.append("Price < $0.01")

    return (len(reasons) == 0), reasons
