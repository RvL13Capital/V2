"""
Technical Indicators for AIv3 System

Calculates technical indicators used in pattern detection:
- BBW (Bollinger Band Width) - volatility measurement
- ADX (Average Directional Index) - trend strength
- RSI (Relative Strength Index) - momentum
- Volume indicators - accumulation/distribution patterns
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: ta-lib not installed. Using pure Python implementations.")
    print("For better performance, install ta-lib: pip install ta-lib")

logger = logging.getLogger(__name__)


def calculate_bbw(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    column: str = 'close'
) -> pd.Series:
    """
    Calculate Bollinger Band Width (BBW).

    BBW measures volatility contraction - key indicator for consolidation patterns.
    Lower BBW = tighter consolidation = potential breakout setup.

    Args:
        df: DataFrame with price data
        period: Moving average period (default 20)
        num_std: Number of standard deviations (default 2.0)
        column: Price column to use (default 'close')

    Returns:
        Series with BBW values
    """
    if TALIB_AVAILABLE:
        try:
            # Ensure float64 for TA-Lib (eliminates warnings, 2-5x faster)
            close_values = np.array(df[column].values, dtype=np.float64)
            upper, middle, lower = talib.BBANDS(
                close_values,
                timeperiod=period,
                nbdevup=num_std,
                nbdevdn=num_std,
                matype=0
            )
            bbw = ((upper - lower) / middle) * 100
            return pd.Series(bbw, index=df.index, name=f'bbw_{period}')
        except Exception as e:
            logger.warning(f"TA-Lib BBW calculation failed: {e}. Using pure Python.")

    # Pure Python implementation
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()

    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)

    bbw = ((upper_band - lower_band) / sma) * 100

    return bbw.rename(f'bbw_{period}')


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).

    ADX measures trend strength. Low ADX (<32) indicates weak trending,
    which is ideal for consolidation patterns before breakouts.

    Args:
        df: DataFrame with high, low, close columns
        period: ADX period (default 14)

    Returns:
        Series with ADX values
    """
    if TALIB_AVAILABLE:
        try:
            # Ensure float64 for TA-Lib (eliminates warnings, faster)
            high_values = np.array(df['high'].values, dtype=np.float64)
            low_values = np.array(df['low'].values, dtype=np.float64)
            close_values = np.array(df['close'].values, dtype=np.float64)
            adx = talib.ADX(
                high_values,
                low_values,
                close_values,
                timeperiod=period
            )
            return pd.Series(adx, index=df.index, name='adx')
        except Exception as e:
            logger.warning(f"TA-Lib ADX calculation failed: {e}. Using pure Python.")

    # Pure Python implementation
    high = df['high']
    low = df['low']
    close = df['close']

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Smooth with Wilder's smoothing
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx.rename('adx')


def calculate_rsi(
    df: pd.DataFrame,
    period: int = 14,
    column: str = 'close'
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures momentum. Values 30-70 often indicate consolidation zones.

    Args:
        df: DataFrame with price data
        period: RSI period (default 14)
        column: Price column to use (default 'close')

    Returns:
        Series with RSI values
    """
    if TALIB_AVAILABLE:
        try:
            # Ensure float64 for TA-Lib (eliminates warnings, faster)
            close_values = np.array(df[column].values, dtype=np.float64)
            rsi = talib.RSI(close_values, timeperiod=period)
            return pd.Series(rsi, index=df.index, name=f'rsi_{period}')
        except Exception as e:
            logger.warning(f"TA-Lib RSI calculation failed: {e}. Using pure Python.")

    # Pure Python implementation
    delta = df[column].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi.rename(f'rsi_{period}')


def calculate_volume_ratio(
    df: pd.DataFrame,
    period: int = 20
) -> pd.Series:
    """
    Calculate volume ratio relative to moving average.

    Volume ratio < 0.35 indicates low volume consolidation.

    Args:
        df: DataFrame with volume column
        period: Moving average period (default 20)

    Returns:
        Series with volume ratio values
    """
    volume_ma = df['volume'].rolling(window=period).mean()
    volume_ratio = df['volume'] / (volume_ma + 1e-10)

    return volume_ratio.rename(f'volume_ratio_{period}')


def calculate_daily_range(
    df: pd.DataFrame,
    normalize: bool = True
) -> pd.Series:
    """
    Calculate daily price range.

    Args:
        df: DataFrame with high, low, close columns
        normalize: If True, normalize by close price (default True)

    Returns:
        Series with daily range values
    """
    daily_range = df['high'] - df['low']

    if normalize:
        daily_range = daily_range / df['close']

    return daily_range.rename('daily_range')


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    ATR measures volatility. Declining ATR indicates consolidation.

    Args:
        df: DataFrame with high, low, close columns
        period: ATR period (default 14)

    Returns:
        Series with ATR values
    """
    if TALIB_AVAILABLE:
        try:
            # Ensure float64 for TA-Lib (eliminates warnings, faster)
            high_values = np.array(df['high'].values, dtype=np.float64)
            low_values = np.array(df['low'].values, dtype=np.float64)
            close_values = np.array(df['close'].values, dtype=np.float64)
            atr = talib.ATR(
                high_values,
                low_values,
                close_values,
                timeperiod=period
            )
            return pd.Series(atr, index=df.index, name=f'atr_{period}')
        except Exception as e:
            logger.warning(f"TA-Lib ATR calculation failed: {e}. Using pure Python.")

    # Pure Python implementation
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    return atr.rename(f'atr_{period}')


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    OBV tracks volume flow and can indicate accumulation.

    Args:
        df: DataFrame with close and volume columns

    Returns:
        Series with OBV values
    """
    if TALIB_AVAILABLE:
        try:
            # Ensure float64 for TA-Lib (eliminates warnings, faster)
            close_values = np.array(df['close'].values, dtype=np.float64)
            volume_values = np.array(df['volume'].values, dtype=np.float64)
            obv = talib.OBV(close_values, volume_values)
            return pd.Series(obv, index=df.index, name='obv')
        except Exception as e:
            logger.warning(f"TA-Lib OBV calculation failed: {e}. Using pure Python.")

    # Pure Python implementation
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv.rename('obv')


def calculate_all_indicators(
    df: pd.DataFrame,
    bbw_period: int = 20,
    adx_period: int = 14,
    rsi_period: int = 14,
    volume_period: int = 20,
    atr_period: int = 14
) -> pd.DataFrame:
    """
    Calculate all technical indicators needed for pattern detection.

    Args:
        df: DataFrame with OHLCV data
        bbw_period: BBW period
        adx_period: ADX period
        rsi_period: RSI period
        volume_period: Volume MA period
        atr_period: ATR period

    Returns:
        DataFrame with all indicators added
    """
    result = df.copy()

    # Bollinger Band Width (critical for consolidation detection)
    result[f'bbw_{bbw_period}'] = calculate_bbw(result, period=bbw_period)

    # ADX (trend strength)
    result['adx'] = calculate_adx(result, period=adx_period)

    # RSI (momentum)
    result[f'rsi_{rsi_period}'] = calculate_rsi(result, period=rsi_period)

    # Volume indicators
    result[f'volume_ratio_{volume_period}'] = calculate_volume_ratio(result, period=volume_period)

    # ATR (volatility)
    result[f'atr_{atr_period}'] = calculate_atr(result, period=atr_period)

    # Daily range
    result['daily_range'] = calculate_daily_range(result, normalize=True)
    result['daily_range_avg'] = result['daily_range'].rolling(window=20).mean()

    # OBV (volume flow)
    result['obv'] = calculate_obv(result)

    # Additional derived indicators
    result['bbw_percentile'] = result[f'bbw_{bbw_period}'].rolling(window=100).apply(
        lambda x: (x.iloc[-1] < x).sum() / len(x) if len(x) > 0 else 0.5,
        raw=False
    )

    result['volume_percentile'] = result['volume'].rolling(window=100).apply(
        lambda x: (x.iloc[-1] < x).sum() / len(x) if len(x) > 0 else 0.5,
        raw=False
    )

    return result


def get_bbw_percentile(df: pd.DataFrame, window: int = 100) -> pd.Series:
    """
    Get BBW percentile rank over rolling window.

    Args:
        df: DataFrame with bbw column
        window: Rolling window for percentile calculation

    Returns:
        Series with BBW percentile values
    """
    bbw_col = [col for col in df.columns if col.startswith('bbw_')][0]

    percentile = df[bbw_col].rolling(window=window).apply(
        lambda x: (x.iloc[-1] < x).sum() / len(x) if len(x) > 0 else 0.5,
        raw=False
    )

    return percentile.rename('bbw_percentile')


def check_consolidation_criteria(
    df: pd.DataFrame,
    bbw_percentile_threshold: float = 0.30,
    adx_threshold: float = 32.0,
    volume_ratio_threshold: float = 0.35,
    range_ratio_threshold: float = 0.65
) -> pd.Series:
    """
    Check if current bar meets consolidation criteria.

    Consolidation criteria (all must be true):
    - BBW < 30th percentile (volatility contraction)
    - ADX < 32 (low trending)
    - Volume < 35% of 20-day average (low volume)
    - Daily Range < 65% of 20-day average (tight range)

    Args:
        df: DataFrame with indicators
        bbw_percentile_threshold: BBW percentile threshold
        adx_threshold: ADX threshold
        volume_ratio_threshold: Volume ratio threshold
        range_ratio_threshold: Range ratio threshold

    Returns:
        Boolean Series indicating consolidation
    """
    # Get BBW percentile
    if 'bbw_percentile' not in df.columns:
        df['bbw_percentile'] = get_bbw_percentile(df)

    # Get volume ratio
    volume_col = [col for col in df.columns if 'volume_ratio' in col]
    if volume_col:
        volume_ratio = df[volume_col[0]]
    else:
        volume_ratio = calculate_volume_ratio(df)

    # Get range ratio
    if 'daily_range' not in df.columns or 'daily_range_avg' not in df.columns:
        df['daily_range'] = calculate_daily_range(df)
        df['daily_range_avg'] = df['daily_range'].rolling(20).mean()

    range_ratio = df['daily_range'] / (df['daily_range_avg'] + 1e-10)

    # Check all criteria
    consolidating = (
        (df['bbw_percentile'] < bbw_percentile_threshold) &
        (df['adx'] < adx_threshold) &
        (volume_ratio < volume_ratio_threshold) &
        (range_ratio < range_ratio_threshold)
    )

    return consolidating.rename('is_consolidating')
