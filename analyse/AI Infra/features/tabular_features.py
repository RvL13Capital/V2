"""
Tabular feature extraction for financial time series.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import talib
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class TabularFeatureExtractor:
    """Extract tabular features from OHLCV data."""

    def __init__(self,
                 feature_groups: Optional[List[str]] = None,
                 periods: Optional[Dict[str, List[int]]] = None):
        """
        Initialize feature extractor.

        Args:
            feature_groups: Groups of features to extract
            periods: Period settings for indicators
        """
        self.feature_groups = feature_groups or [
            'price', 'volume', 'volatility', 'momentum',
            'trend', 'pattern', 'statistical'
        ]

        self.periods = periods or {
            'short': [5, 10, 14],
            'medium': [20, 30, 50],
            'long': [100, 200]
        }

        self.feature_names = []

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all feature groups.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with all features
        """
        features = df.copy()

        # Convert to float64 for TA-Lib compatibility
        # TA-Lib requires double precision arrays
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in features.columns:
                features[col] = features[col].astype(np.float64)

        for group in self.feature_groups:
            if group == 'price':
                features = self._extract_price_features(features)
            elif group == 'volume':
                features = self._extract_volume_features(features)
            elif group == 'volatility':
                features = self._extract_volatility_features(features)
            elif group == 'momentum':
                features = self._extract_momentum_features(features)
            elif group == 'trend':
                features = self._extract_trend_features(features)
            elif group == 'pattern':
                features = self._extract_pattern_features(features)
            elif group == 'statistical':
                features = self._extract_statistical_features(features)

        # Store feature names
        self.feature_names = [col for col in features.columns
                             if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]

        return features

    def _extract_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract price-based features."""
        # Price ratios
        df['high_low_ratio'] = df['high'] / (df['low'] + 1e-10)
        df['close_open_ratio'] = df['close'] / (df['open'] + 1e-10)

        # Price position
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Moving averages
        for period in self.periods['short'] + self.periods['medium']:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_to_sma_{period}'] = df['close'] / (df[f'sma_{period}'] + 1e-10)

        # Weighted averages
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        return df

    def _extract_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volume-based features."""
        # Volume ratios
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

        # On-Balance Volume
        df['obv'] = talib.OBV(df['close'].values, df['volume'].values)

        # Volume-Price Trend
        df['vpt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
        df['vpt_cumsum'] = df['vpt'].cumsum()

        # Accumulation/Distribution
        df['ad'] = talib.AD(df['high'].values, df['low'].values,
                           df['close'].values, df['volume'].values)

        # Chaikin Money Flow
        df['cmf'] = talib.ADOSC(df['high'].values, df['low'].values,
                                df['close'].values, df['volume'].values)

        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_sma_{period}'] + 1e-10)

        return df

    def _extract_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility features."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Historical volatility
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std() * np.sqrt(252)
            df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(window=50).mean()

        # ATR (Average True Range)
        df['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['atr_ratio'] = df['atr_14'] / df['close']

        # Bollinger Bands
        for period in [20, 50]:
            upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=period)
            df[f'bb_upper_{period}'] = upper
            df[f'bb_lower_{period}'] = lower
            df[f'bb_width_{period}'] = (upper - lower) / middle
            df[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower + 1e-10)

        # Keltner Channels
        df['kc_upper'] = df['ema_20'] + 2 * df['atr_14']
        df['kc_lower'] = df['ema_20'] - 2 * df['atr_14']
        df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'] + 1e-10)

        return df

    def _extract_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract momentum indicators."""
        # RSI
        for period in [14, 21]:
            df[f'rsi_{period}'] = talib.RSI(df['close'].values, timeperiod=period)
            df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
            df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)

        # MACD
        macd, signal, hist = talib.MACD(df['close'].values)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        df['macd_cross'] = np.where(macd > signal, 1, -1)

        # Stochastic
        k, d = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
        df['stoch_k'] = k
        df['stoch_d'] = d
        df['stoch_cross'] = np.where(k > d, 1, -1)

        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values)

        # CCI
        df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values)

        # MFI
        df['mfi'] = talib.MFI(df['high'].values, df['low'].values,
                             df['close'].values, df['volume'].values)

        # Rate of Change
        for period in [10, 20]:
            df[f'roc_{period}'] = talib.ROC(df['close'].values, timeperiod=period)

        return df

    def _extract_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract trend indicators."""
        # ADX
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values)
        df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values)
        df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values)
        df['adx_trend_strength'] = df['adx'] / 100

        # Aroon
        aroon_up, aroon_down = talib.AROON(df['high'].values, df['low'].values)
        df['aroon_up'] = aroon_up
        df['aroon_down'] = aroon_down
        df['aroon_oscillator'] = aroon_up - aroon_down

        # Parabolic SAR
        df['sar'] = talib.SAR(df['high'].values, df['low'].values)
        df['sar_position'] = np.where(df['close'] > df['sar'], 1, -1)

        # Supertrend (simplified)
        df['supertrend'] = df['close'].rolling(window=10).mean() + 2 * df['atr_14']
        df['supertrend_signal'] = np.where(df['close'] > df['supertrend'], 1, -1)

        # Linear regression slope
        for period in [10, 20, 50]:
            df[f'trend_slope_{period}'] = df['close'].rolling(window=period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan
            )

        return df

    def _extract_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract candlestick patterns."""
        # Basic patterns
        df['doji'] = talib.CDLDOJI(df['open'].values, df['high'].values,
                                   df['low'].values, df['close'].values)
        df['hammer'] = talib.CDLHAMMER(df['open'].values, df['high'].values,
                                       df['low'].values, df['close'].values)
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'].values, df['high'].values,
                                                    df['low'].values, df['close'].values)
        df['engulfing'] = talib.CDLENGULFING(df['open'].values, df['high'].values,
                                             df['low'].values, df['close'].values)

        # Pivot points
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance_1'] = 2 * df['pivot'] - df['low']
        df['support_1'] = 2 * df['pivot'] - df['high']

        # Price patterns
        df['higher_high'] = ((df['high'] > df['high'].shift(1)) &
                            (df['high'].shift(1) > df['high'].shift(2))).astype(int)
        df['lower_low'] = ((df['low'] < df['low'].shift(1)) &
                          (df['low'].shift(1) < df['low'].shift(2))).astype(int)

        # Breakout detection
        df['breakout_up'] = (df['close'] > df['high'].rolling(window=20).max().shift(1)).astype(int)
        df['breakout_down'] = (df['close'] < df['low'].rolling(window=20).min().shift(1)).astype(int)

        return df

    def _extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features."""
        # Rolling statistics
        for period in [10, 20, 50]:
            # Basic stats
            df[f'rolling_mean_{period}'] = df['close'].rolling(window=period).mean()
            df[f'rolling_std_{period}'] = df['close'].rolling(window=period).std()
            df[f'rolling_skew_{period}'] = df['returns'].rolling(window=period).skew()
            df[f'rolling_kurt_{period}'] = df['returns'].rolling(window=period).kurt()

            # Quantiles
            df[f'rolling_q25_{period}'] = df['close'].rolling(window=period).quantile(0.25)
            df[f'rolling_q75_{period}'] = df['close'].rolling(window=period).quantile(0.75)
            df[f'iqr_{period}'] = df[f'rolling_q75_{period}'] - df[f'rolling_q25_{period}']

            # Z-score
            df[f'zscore_{period}'] = (df['close'] - df[f'rolling_mean_{period}']) / (df[f'rolling_std_{period}'] + 1e-10)

        # Autocorrelation
        df['autocorr_1'] = df['returns'].rolling(window=20).apply(lambda x: x.autocorr(lag=1))
        df['autocorr_5'] = df['returns'].rolling(window=20).apply(lambda x: x.autocorr(lag=5))

        # Hurst exponent (simplified)
        def hurst_exponent(ts):
            lags = range(2, min(20, len(ts)//2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0

        df['hurst'] = df['close'].rolling(window=50).apply(hurst_exponent)

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

    def select_top_features(self,
                           df: pd.DataFrame,
                           y: np.ndarray,
                           n_features: int = 50) -> List[str]:
        """
        Select top features based on importance.

        Args:
            df: Feature DataFrame
            y: Target labels
            n_features: Number of features to select

        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import mutual_info_classif

        # Get feature columns
        feature_cols = [col for col in df.columns
                       if col not in ['timestamp', 'symbol']]

        X = df[feature_cols].fillna(0).values

        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y)

        # Get top features
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)

        top_features = feature_importance.head(n_features)['feature'].tolist()

        return top_features