"""
Enhanced Feature Engineering v3 with Advanced Pattern Recognition
==================================================================
Adds attention-based features, anomaly scores, and character ratings
for improved K4 pattern detection.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineerV3:
    """
    Feature Engineering v3 with:
    - Market microstructure features
    - Attention-weighted temporal features
    - Anomaly detection scores
    - Pattern character ratings
    - Volatility regime indicators
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_importance = {}
        self.attention_weights = None

    def extract_all_features(self, df: pd.DataFrame, pattern: Dict,
                            snapshot_date: pd.Timestamp) -> pd.DataFrame:
        """Extract comprehensive v3 feature set."""

        features = {}

        # Core features
        features.update(self._extract_core_features(df, pattern, snapshot_date))

        # Market microstructure features
        features.update(self._extract_microstructure_features(df, snapshot_date))

        # Attention-weighted temporal features
        features.update(self._extract_attention_features(df, snapshot_date))

        # Anomaly detection scores
        features.update(self._extract_anomaly_scores(df, snapshot_date))

        # Character rating features
        features.update(self._extract_character_ratings(df, pattern, snapshot_date))

        # Volatility regime features
        features.update(self._extract_volatility_regime(df, snapshot_date))

        # Volume profile features
        features.update(self._extract_volume_profile(df, snapshot_date))

        # Price action quality features
        features.update(self._extract_price_action_quality(df, snapshot_date))

        # Pattern maturity features
        features.update(self._extract_pattern_maturity(df, pattern, snapshot_date))

        # Market context features
        features.update(self._extract_market_context(df, snapshot_date))

        return pd.DataFrame([features])

    def _extract_core_features(self, df: pd.DataFrame, pattern: Dict,
                              snapshot_date: pd.Timestamp) -> Dict:
        """Extract core pattern features."""

        # Handle both date as column and date as index
        if 'date' in df.columns:
            window = df[pd.to_datetime(df['date']) <= snapshot_date].tail(100)
        else:
            df.index = pd.to_datetime(df.index)
            # Handle both date as column and date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            window = df[df['date'] <= pd.to_datetime(snapshot_date)]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            window = df[df.index <= pd.to_datetime(snapshot_date)].tail(100)
        if len(window) < 20:
            return self._get_default_core_features()

        latest = window.iloc[-1]

        return {
            # Pattern boundaries
            'upper_boundary': pattern.get('upper_boundary', 0),
            'lower_boundary': pattern.get('lower_boundary', 0),
            'power_boundary': pattern.get('power_boundary', 0),

            # Price position
            'price_to_upper_pct': ((latest['close'] - pattern['upper_boundary']) /
                                   pattern['upper_boundary'] * 100),
            'price_to_lower_pct': ((latest['close'] - pattern['lower_boundary']) /
                                   pattern['lower_boundary'] * 100),
            'price_position_in_range': ((latest['close'] - pattern['lower_boundary']) /
                                        (pattern['upper_boundary'] - pattern['lower_boundary'])),

            # Pattern metrics
            'days_in_pattern': pattern.get('days_in_pattern', 0),
            'days_qualifying': pattern.get('days_qualifying', 0),
            'days_active': pattern.get('days_active', 0),
            'range_width_pct': ((pattern['upper_boundary'] - pattern['lower_boundary']) /
                               pattern['lower_boundary'] * 100),

            # Recent price action
            'close_price': latest['close'],
            'volume': latest['volume'],
            'daily_return': window['close'].pct_change().iloc[-1] * 100,
            '5d_return': (window['close'].iloc[-1] / window['close'].iloc[-6] - 1) * 100 if len(window) >= 6 else 0,
            '20d_return': (window['close'].iloc[-1] / window['close'].iloc[-21] - 1) * 100 if len(window) >= 21 else 0,
        }

    def _extract_microstructure_features(self, df: pd.DataFrame,
                                        snapshot_date: pd.Timestamp) -> Dict:
        """Extract market microstructure features."""

        # Handle both date as column and date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            window = df[df['date'] <= pd.to_datetime(snapshot_date)]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            window = df[df.index <= pd.to_datetime(snapshot_date)].tail(50)
        if len(window) < 20:
            return self._get_default_microstructure_features()

        features = {}

        # Intraday volatility
        features['intraday_volatility'] = ((window['high'] - window['low']) / window['close']).mean()
        features['close_to_high_ratio'] = ((window['close'] - window['low']) /
                                           (window['high'] - window['low'])).mean()

        # Volume-weighted metrics
        vwap = (window['close'] * window['volume']).sum() / window['volume'].sum()
        features['price_to_vwap_ratio'] = window['close'].iloc[-1] / vwap

        # Bid-ask spread proxy (using high-low)
        features['spread_proxy'] = ((window['high'] - window['low']) / window['close']).rolling(5).mean().iloc[-1]

        # Trade intensity
        features['volume_intensity'] = window['volume'].iloc[-5:].sum() / window['volume'].iloc[-20:].sum()

        # Price efficiency (deviation from random walk)
        returns = window['close'].pct_change().dropna()
        features['price_efficiency'] = abs(returns.autocorr()) if len(returns) > 10 else 0

        # Microstructure noise
        features['noise_ratio'] = returns.std() / abs(returns.mean()) if returns.mean() != 0 else 999

        # Order flow imbalance proxy
        up_days = window[window['close'] > window['open']]
        down_days = window[window['close'] < window['open']]
        features['order_flow_imbalance'] = (len(up_days) - len(down_days)) / len(window)

        return features

    def _extract_attention_features(self, df: pd.DataFrame,
                                   snapshot_date: pd.Timestamp) -> Dict:
        """Extract attention-weighted temporal features."""

        # Handle both date as column and date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            window = df[df['date'] <= pd.to_datetime(snapshot_date)]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            window = df[df.index <= pd.to_datetime(snapshot_date)].tail(60)
        if len(window) < 30:
            return self._get_default_attention_features()

        features = {}

        # Calculate attention weights (more recent = higher weight)
        n = len(window)
        positions = np.arange(n)

        # Exponential attention weights
        exp_weights = np.exp(positions / 10)
        exp_weights = exp_weights / exp_weights.sum()

        # Linear attention weights
        linear_weights = positions / positions.sum()

        # Sigmoid attention weights
        sigmoid_weights = 1 / (1 + np.exp(-(positions - n/2) / 5))
        sigmoid_weights = sigmoid_weights / sigmoid_weights.sum()

        # Store best weights for later use
        self.attention_weights = exp_weights

        # Attention-weighted price features
        features['attention_weighted_return'] = (window['close'].pct_change().fillna(0) * exp_weights).sum() * 100
        features['attention_weighted_volume'] = (window['volume'] * exp_weights).sum()
        features['attention_weighted_volatility'] = np.sqrt((((window['close'].pct_change().fillna(0) -
                                                               window['close'].pct_change().mean()) ** 2) * exp_weights).sum())

        # Temporal attention scores
        price_changes = window['close'].pct_change().fillna(0)
        features['recent_attention_score'] = (abs(price_changes) * exp_weights).sum()
        features['volume_attention_score'] = ((window['volume'] / window['volume'].mean() - 1) * exp_weights).sum()

        # Attention-based momentum
        features['attention_momentum'] = (price_changes * exp_weights * np.sign(price_changes.mean())).sum()

        # Focus periods (high attention areas)
        high_attention_mask = exp_weights > np.percentile(exp_weights, 75)
        features['high_attention_return'] = price_changes[high_attention_mask].sum() * 100
        features['high_attention_volume_ratio'] = window['volume'][high_attention_mask].mean() / window['volume'].mean()

        return features

    def _extract_anomaly_scores(self, df: pd.DataFrame,
                               snapshot_date: pd.Timestamp) -> Dict:
        """Calculate anomaly detection scores."""

        # Handle both date as column and date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            window = df[df['date'] <= pd.to_datetime(snapshot_date)]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            window = df[df.index <= pd.to_datetime(snapshot_date)].tail(100)
        if len(window) < 50:
            return self._get_default_anomaly_scores()

        features = {}

        # Volume anomaly score
        volume_zscore = (window['volume'].iloc[-1] - window['volume'].mean()) / window['volume'].std()
        features['volume_anomaly_score'] = abs(volume_zscore)
        features['volume_anomaly_direction'] = np.sign(volume_zscore)

        # Price anomaly score (using returns)
        returns = window['close'].pct_change().dropna()
        return_zscore = (returns.iloc[-1] - returns.mean()) / returns.std()
        features['price_anomaly_score'] = abs(return_zscore)
        features['price_anomaly_direction'] = np.sign(return_zscore)

        # Volatility anomaly
        volatility = window['high'] - window['low']
        vol_zscore = (volatility.iloc[-1] - volatility.mean()) / volatility.std()
        features['volatility_anomaly_score'] = abs(vol_zscore)

        # Pattern anomaly (using Isolation Forest concept)
        recent_patterns = pd.DataFrame({
            'return': returns.rolling(5).mean(),
            'volume': window['volume'].rolling(5).mean(),
            'volatility': volatility.rolling(5).mean()
        }).dropna()

        if len(recent_patterns) > 10:
            # Calculate Mahalanobis distance for anomaly detection
            mean = recent_patterns.mean()
            cov = recent_patterns.cov()
            inv_cov = np.linalg.pinv(cov)

            latest_pattern = recent_patterns.iloc[-1]
            diff = latest_pattern - mean
            mahal_dist = np.sqrt(diff.values @ inv_cov @ diff.values.T)
            features['pattern_anomaly_score'] = mahal_dist
        else:
            features['pattern_anomaly_score'] = 0

        # Composite anomaly score
        features['composite_anomaly_score'] = (features['volume_anomaly_score'] * 0.3 +
                                               features['price_anomaly_score'] * 0.3 +
                                               features['volatility_anomaly_score'] * 0.2 +
                                               features['pattern_anomaly_score'] * 0.2)

        # Anomaly persistence (how long anomalous behavior continues)
        anomaly_threshold = 2.0  # z-score threshold
        recent_anomalies = abs(stats.zscore(returns.tail(10))) > anomaly_threshold
        features['anomaly_persistence'] = recent_anomalies.sum() / len(recent_anomalies)

        return features

    def _extract_character_ratings(self, df: pd.DataFrame, pattern: Dict,
                                  snapshot_date: pd.Timestamp) -> Dict:
        """Calculate pattern character ratings."""

        # Handle both date as column and date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            window = df[df['date'] <= pd.to_datetime(snapshot_date)]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            window = df[df.index <= pd.to_datetime(snapshot_date)].tail(50)
        if len(window) < 30:
            return self._get_default_character_ratings()

        features = {}

        # Compression character (0-10)
        bbw = self._calculate_bbw(window)
        compression_score = max(0, min(10, (1 - bbw / 100) * 10))
        features['compression_character'] = compression_score

        # Momentum character (0-10)
        returns = window['close'].pct_change().dropna()
        momentum_score = max(0, min(10, (returns.tail(10).mean() / returns.std()) * 5 + 5))
        features['momentum_character'] = momentum_score

        # Volume character (0-10)
        volume_trend = np.polyfit(range(len(window)), window['volume'], 1)[0]
        volume_score = max(0, min(10, (volume_trend / window['volume'].mean()) * 50 + 5))
        features['volume_character'] = volume_score

        # Stability character (0-10)
        stability_score = max(0, min(10, (1 / (returns.std() * 100)) * 5))
        features['stability_character'] = stability_score

        # Breakout potential character (0-10)
        distance_to_upper = (pattern['upper_boundary'] - window['close'].iloc[-1]) / pattern['upper_boundary']
        recent_high = window['high'].tail(10).max()
        approach_score = (recent_high - pattern['lower_boundary']) / (pattern['upper_boundary'] - pattern['lower_boundary'])
        breakout_score = max(0, min(10, (approach_score * 10) - distance_to_upper * 5))
        features['breakout_character'] = breakout_score

        # Energy accumulation character (0-10)
        energy = (window['volume'] * abs(window['close'].pct_change())).fillna(0)
        energy_ma = energy.rolling(20).mean()
        energy_score = max(0, min(10, (energy.iloc[-1] / energy_ma.iloc[-1]) * 5)) if energy_ma.iloc[-1] > 0 else 5
        features['energy_character'] = energy_score

        # Overall character rating (weighted average)
        features['overall_character_rating'] = (
            compression_score * 0.25 +
            momentum_score * 0.20 +
            volume_score * 0.15 +
            stability_score * 0.10 +
            breakout_score * 0.20 +
            energy_score * 0.10
        )

        # Character category
        if features['overall_character_rating'] >= 8:
            features['character_category'] = 4  # Exceptional
        elif features['overall_character_rating'] >= 6:
            features['character_category'] = 3  # Strong
        elif features['overall_character_rating'] >= 4:
            features['character_category'] = 2  # Moderate
        elif features['overall_character_rating'] >= 2:
            features['character_category'] = 1  # Weak
        else:
            features['character_category'] = 0  # Poor

        return features

    def _extract_volatility_regime(self, df: pd.DataFrame,
                                  snapshot_date: pd.Timestamp) -> Dict:
        """Extract volatility regime features."""

        # Handle both date as column and date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            window = df[df['date'] <= pd.to_datetime(snapshot_date)]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            window = df[df.index <= pd.to_datetime(snapshot_date)].tail(100)
        if len(window) < 50:
            return self._get_default_volatility_regime()

        features = {}

        # Calculate various volatility measures
        returns = window['close'].pct_change().dropna()

        # Realized volatility
        features['realized_volatility_5d'] = returns.tail(5).std() * np.sqrt(252)
        features['realized_volatility_20d'] = returns.tail(20).std() * np.sqrt(252)
        features['realized_volatility_50d'] = returns.std() * np.sqrt(252)

        # Volatility ratio
        features['volatility_ratio_5_20'] = features['realized_volatility_5d'] / features['realized_volatility_20d']
        features['volatility_ratio_20_50'] = features['realized_volatility_20d'] / features['realized_volatility_50d']

        # Volatility trend
        vol_series = returns.rolling(10).std()
        vol_trend = np.polyfit(range(len(vol_series.dropna())), vol_series.dropna(), 1)[0]
        features['volatility_trend'] = vol_trend * 1000

        # Volatility regime (low/medium/high)
        if features['realized_volatility_20d'] < 0.15:
            features['volatility_regime'] = 0  # Low
        elif features['realized_volatility_20d'] < 0.30:
            features['volatility_regime'] = 1  # Medium
        else:
            features['volatility_regime'] = 2  # High

        # Volatility clustering (GARCH effect)
        squared_returns = returns ** 2
        features['volatility_clustering'] = squared_returns.autocorr()

        return features

    def _extract_volume_profile(self, df: pd.DataFrame,
                               snapshot_date: pd.Timestamp) -> Dict:
        """Extract volume profile features."""

        # Handle both date as column and date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            window = df[df['date'] <= pd.to_datetime(snapshot_date)]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            window = df[df.index <= pd.to_datetime(snapshot_date)].tail(50)
        if len(window) < 30:
            return self._get_default_volume_profile()

        features = {}

        # Volume distribution
        features['volume_mean'] = window['volume'].mean()
        features['volume_std'] = window['volume'].std()
        features['volume_skew'] = window['volume'].skew()
        features['volume_kurtosis'] = window['volume'].kurtosis()

        # Volume concentration
        top_5_volume = window.nlargest(5, 'volume')['volume'].sum()
        features['volume_concentration'] = top_5_volume / window['volume'].sum()

        # Price-volume correlation
        features['price_volume_corr'] = window['close'].corr(window['volume'])

        # Volume at price levels
        price_levels = pd.qcut(window['close'], 5, labels=False)
        volume_by_level = window.groupby(price_levels)['volume'].mean()
        features['volume_at_lows'] = volume_by_level.iloc[0] / features['volume_mean']
        features['volume_at_highs'] = volume_by_level.iloc[-1] / features['volume_mean']

        # Volume momentum
        volume_ma_5 = window['volume'].rolling(5).mean()
        volume_ma_20 = window['volume'].rolling(20).mean()
        features['volume_momentum'] = (volume_ma_5.iloc[-1] / volume_ma_20.iloc[-1]) if volume_ma_20.iloc[-1] > 0 else 1

        return features

    def _extract_price_action_quality(self, df: pd.DataFrame,
                                     snapshot_date: pd.Timestamp) -> Dict:
        """Extract price action quality features."""

        # Handle both date as column and date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            window = df[df['date'] <= pd.to_datetime(snapshot_date)]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            window = df[df.index <= pd.to_datetime(snapshot_date)].tail(50)
        if len(window) < 30:
            return self._get_default_price_action_quality()

        features = {}

        # Trend quality
        closes = window['close'].values
        x = np.arange(len(closes))
        slope, intercept = np.polyfit(x, closes, 1)
        predicted = slope * x + intercept
        r_squared = 1 - (np.sum((closes - predicted) ** 2) / np.sum((closes - closes.mean()) ** 2))
        features['trend_quality_r2'] = r_squared
        features['trend_slope_normalized'] = slope / closes.mean()

        # Smoothness (inverse of choppiness)
        daily_ranges = window['high'] - window['low']
        total_range = window['high'].max() - window['low'].min()
        features['price_smoothness'] = 1 - (daily_ranges.sum() / total_range) if total_range > 0 else 0

        # Support/Resistance quality
        peaks, _ = find_peaks(window['high'].values, distance=5)
        troughs, _ = find_peaks(-window['low'].values, distance=5)
        features['resistance_touches'] = len(peaks)
        features['support_touches'] = len(troughs)
        features['sr_balance'] = abs(len(peaks) - len(troughs)) / (len(peaks) + len(troughs)) if (len(peaks) + len(troughs)) > 0 else 0

        # Price acceptance (time spent at levels)
        price_bins = pd.qcut(window['close'], 10, labels=False, duplicates='drop')
        time_at_level = price_bins.value_counts(normalize=True)
        features['price_acceptance_entropy'] = -np.sum(time_at_level * np.log(time_at_level + 1e-10))

        return features

    def _extract_pattern_maturity(self, df: pd.DataFrame, pattern: Dict,
                                 snapshot_date: pd.Timestamp) -> Dict:
        """Extract pattern maturity features."""

        features = {}

        # Time-based maturity
        days_in_pattern = pattern.get('days_in_pattern', 0)
        features['pattern_age_score'] = min(days_in_pattern / 30, 1.0)  # Normalized to 30 days

        # Maturity stages
        if days_in_pattern < 10:
            features['maturity_stage'] = 0  # Early
        elif days_in_pattern < 20:
            features['maturity_stage'] = 1  # Developing
        elif days_in_pattern < 40:
            features['maturity_stage'] = 2  # Mature
        else:
            features['maturity_stage'] = 3  # Late

        # Pattern tightening
        # Handle both date as column and date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            window = df[df['date'] <= pd.to_datetime(snapshot_date)]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            window = df[df.index <= pd.to_datetime(snapshot_date)].tail(days_in_pattern) if days_in_pattern > 0 else df.tail(20)

        if len(window) > 10:
            early_range = (window.iloc[:len(window)//2]['high'] - window.iloc[:len(window)//2]['low']).mean()
            recent_range = (window.iloc[len(window)//2:]['high'] - window.iloc[len(window)//2:]['low']).mean()
            features['pattern_tightening'] = 1 - (recent_range / early_range) if early_range > 0 else 0
        else:
            features['pattern_tightening'] = 0

        # Boundary test frequency
        upper_tests = ((window['high'] >= pattern['upper_boundary'] * 0.98).sum() if 'upper_boundary' in pattern else 0)
        lower_tests = ((window['low'] <= pattern['lower_boundary'] * 1.02).sum() if 'lower_boundary' in pattern else 0)
        features['boundary_test_frequency'] = (upper_tests + lower_tests) / len(window)

        # Pattern readiness score
        features['pattern_readiness'] = (
            features['pattern_age_score'] * 0.3 +
            features['pattern_tightening'] * 0.4 +
            min(features['boundary_test_frequency'] * 2, 1.0) * 0.3
        )

        return features

    def _extract_market_context(self, df: pd.DataFrame,
                               snapshot_date: pd.Timestamp) -> Dict:
        """Extract market context features."""

        # Handle both date as column and date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            window = df[df['date'] <= pd.to_datetime(snapshot_date)]
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            window = df[df.index <= pd.to_datetime(snapshot_date)].tail(100)
        if len(window) < 50:
            return self._get_default_market_context()

        features = {}

        # Relative strength
        returns = window['close'].pct_change().dropna()
        cumulative_return = (1 + returns).prod() - 1
        features['relative_strength'] = cumulative_return

        # Market regime detection
        sma_20 = window['close'].rolling(20).mean()
        sma_50 = window['close'].rolling(50).mean()

        if len(sma_50.dropna()) > 0:
            current_price = window['close'].iloc[-1]
            features['above_sma20'] = 1 if current_price > sma_20.iloc[-1] else 0
            features['above_sma50'] = 1 if current_price > sma_50.iloc[-1] else 0
            features['sma20_sma50_spread'] = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        else:
            features['above_sma20'] = 0
            features['above_sma50'] = 0
            features['sma20_sma50_spread'] = 0

        # Market breadth proxy (using volume distribution)
        up_volume = window[window['close'] > window['open']]['volume'].sum()
        down_volume = window[window['close'] < window['open']]['volume'].sum()
        features['volume_breadth'] = (up_volume - down_volume) / (up_volume + down_volume) if (up_volume + down_volume) > 0 else 0

        # Sector momentum proxy (using price momentum)
        features['short_momentum'] = returns.tail(10).mean() * 252
        features['medium_momentum'] = returns.tail(30).mean() * 252 if len(returns) >= 30 else features['short_momentum']

        return features

    def _calculate_bbw(self, window: pd.DataFrame) -> float:
        """Calculate Bollinger Band Width."""
        sma = window['close'].rolling(20).mean()
        std = window['close'].rolling(20).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        bbw = ((upper_band - lower_band) / sma * 100).iloc[-1]
        return bbw if not np.isnan(bbw) else 100

    def _get_default_core_features(self) -> Dict:
        """Return default core features."""
        return {f: 0 for f in [
            'upper_boundary', 'lower_boundary', 'power_boundary',
            'price_to_upper_pct', 'price_to_lower_pct', 'price_position_in_range',
            'days_in_pattern', 'days_qualifying', 'days_active', 'range_width_pct',
            'close_price', 'volume', 'daily_return', '5d_return', '20d_return'
        ]}

    def _get_default_microstructure_features(self) -> Dict:
        """Return default microstructure features."""
        return {f: 0 for f in [
            'intraday_volatility', 'close_to_high_ratio', 'price_to_vwap_ratio',
            'spread_proxy', 'volume_intensity', 'price_efficiency',
            'noise_ratio', 'order_flow_imbalance'
        ]}

    def _get_default_attention_features(self) -> Dict:
        """Return default attention features."""
        return {f: 0 for f in [
            'attention_weighted_return', 'attention_weighted_volume',
            'attention_weighted_volatility', 'recent_attention_score',
            'volume_attention_score', 'attention_momentum',
            'high_attention_return', 'high_attention_volume_ratio'
        ]}

    def _get_default_anomaly_scores(self) -> Dict:
        """Return default anomaly scores."""
        return {f: 0 for f in [
            'volume_anomaly_score', 'volume_anomaly_direction',
            'price_anomaly_score', 'price_anomaly_direction',
            'volatility_anomaly_score', 'pattern_anomaly_score',
            'composite_anomaly_score', 'anomaly_persistence'
        ]}

    def _get_default_character_ratings(self) -> Dict:
        """Return default character ratings."""
        return {
            'compression_character': 5, 'momentum_character': 5,
            'volume_character': 5, 'stability_character': 5,
            'breakout_character': 5, 'energy_character': 5,
            'overall_character_rating': 5, 'character_category': 2
        }

    def _get_default_volatility_regime(self) -> Dict:
        """Return default volatility regime features."""
        return {f: 0 for f in [
            'realized_volatility_5d', 'realized_volatility_20d', 'realized_volatility_50d',
            'volatility_ratio_5_20', 'volatility_ratio_20_50', 'volatility_trend',
            'volatility_regime', 'volatility_clustering'
        ]}

    def _get_default_volume_profile(self) -> Dict:
        """Return default volume profile features."""
        return {f: 0 for f in [
            'volume_mean', 'volume_std', 'volume_skew', 'volume_kurtosis',
            'volume_concentration', 'price_volume_corr',
            'volume_at_lows', 'volume_at_highs', 'volume_momentum'
        ]}

    def _get_default_price_action_quality(self) -> Dict:
        """Return default price action quality features."""
        return {f: 0 for f in [
            'trend_quality_r2', 'trend_slope_normalized', 'price_smoothness',
            'resistance_touches', 'support_touches', 'sr_balance',
            'price_acceptance_entropy'
        ]}

    def _get_default_market_context(self) -> Dict:
        """Return default market context features."""
        return {f: 0 for f in [
            'relative_strength', 'above_sma20', 'above_sma50',
            'sma20_sma50_spread', 'volume_breadth',
            'short_momentum', 'medium_momentum'
        ]}

    def get_feature_importance(self) -> Dict:
        """Return tracked feature importance scores."""
        return self.feature_importance

    def update_feature_importance(self, importance_dict: Dict):
        """Update feature importance tracking."""
        self.feature_importance.update(importance_dict)


if __name__ == "__main__":
    # Test the feature engineering
    print("Advanced Feature Engineering v3 initialized")
    print("Features include:")
    print("- Market microstructure (8 features)")
    print("- Attention-weighted temporal (8 features)")
    print("- Anomaly detection scores (8 features)")
    print("- Character ratings (8 features)")
    print("- Volatility regime (8 features)")
    print("- Volume profile (9 features)")
    print("- Price action quality (7 features)")
    print("- Pattern maturity (5 features)")
    print("- Market context (7 features)")
    print("\nTotal: 83+ advanced features")