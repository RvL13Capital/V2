"""
Pattern Metrics Calculator for Consolidation Analysis
Calculates detailed metrics for consolidation patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PatternMetrics:
    """Calculate comprehensive metrics for consolidation patterns"""
    
    def __init__(self):
        self.metrics_cache = {}
        
    def calculate_pattern_metrics(self, 
                                 price_data: pd.DataFrame,
                                 start_date: datetime,
                                 end_date: datetime,
                                 upper_boundary: float,
                                 lower_boundary: float,
                                 power_boundary: float) -> Dict:
        """Calculate all metrics for a single pattern"""
        
        # Filter data to pattern period
        pattern_data = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)].copy()
        
        if pattern_data.empty:
            return {}
        
        metrics = {
            'duration_days': (end_date - start_date).days,
            'boundary_width': upper_boundary - lower_boundary,
            'boundary_width_pct': ((upper_boundary - lower_boundary) / lower_boundary) * 100,
            'power_buffer': power_boundary - upper_boundary,
            'power_buffer_pct': ((power_boundary - upper_boundary) / upper_boundary) * 100
        }
        
        # Price position metrics
        metrics.update(self._calculate_price_position_metrics(pattern_data, upper_boundary, lower_boundary))
        
        # Volume metrics
        metrics.update(self._calculate_volume_metrics(pattern_data))
        
        # Volatility metrics
        metrics.update(self._calculate_volatility_metrics(pattern_data))
        
        # Touch metrics
        metrics.update(self._calculate_boundary_touch_metrics(pattern_data, upper_boundary, lower_boundary))
        
        # Consolidation quality metrics
        metrics.update(self._calculate_quality_metrics(pattern_data, upper_boundary, lower_boundary))
        
        return metrics
    
    def _calculate_price_position_metrics(self, data: pd.DataFrame, upper: float, lower: float) -> Dict:
        """Calculate price position within consolidation range"""
        metrics = {}
        
        # Average position in range (0 = at lower, 1 = at upper)
        avg_position = np.mean((data['close'] - lower) / (upper - lower))
        metrics['avg_price_position'] = avg_position
        
        # Time spent in different zones
        range_height = upper - lower
        upper_zone = upper - (range_height * 0.25)
        lower_zone = lower + (range_height * 0.25)
        
        metrics['time_in_upper_25pct'] = len(data[data['close'] >= upper_zone]) / len(data)
        metrics['time_in_lower_25pct'] = len(data[data['close'] <= lower_zone]) / len(data)
        metrics['time_in_middle_50pct'] = len(data[(data['close'] > lower_zone) & (data['close'] < upper_zone)]) / len(data)
        
        # Distance from boundaries at end
        final_close = data.iloc[-1]['close']
        metrics['final_distance_from_upper'] = upper - final_close
        metrics['final_distance_from_lower'] = final_close - lower
        metrics['final_position_in_range'] = (final_close - lower) / (upper - lower)
        
        return metrics
    
    def _calculate_volume_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate volume-based metrics"""
        metrics = {}
        
        if 'volume' not in data.columns:
            return metrics
        
        # Volume trends
        volumes = data['volume'].values
        if len(volumes) > 1:
            # Linear regression for volume trend
            x = np.arange(len(volumes))
            coeffs = np.polyfit(x, volumes, 1)
            metrics['volume_trend_slope'] = coeffs[0]
            metrics['volume_trend_normalized'] = coeffs[0] / np.mean(volumes) if np.mean(volumes) > 0 else 0
        
        # Volume statistics
        metrics['avg_volume'] = np.mean(volumes)
        metrics['volume_std'] = np.std(volumes)
        metrics['volume_cv'] = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
        
        # Volume contraction
        if len(data) >= 20:
            recent_avg = np.mean(volumes[-10:])
            early_avg = np.mean(volumes[:10])
            metrics['volume_contraction_ratio'] = recent_avg / early_avg if early_avg > 0 else 0
        
        # Volume spikes
        volume_mean = np.mean(volumes)
        volume_std = np.std(volumes)
        spike_threshold = volume_mean + (2 * volume_std)
        metrics['volume_spike_count'] = len(volumes[volumes > spike_threshold])
        metrics['volume_spike_ratio'] = len(volumes[volumes > spike_threshold]) / len(volumes)
        
        return metrics
    
    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate volatility-based metrics"""
        metrics = {}
        
        # Daily returns
        returns = data['close'].pct_change().dropna()
        
        if len(returns) > 0:
            metrics['return_mean'] = np.mean(returns)
            metrics['return_std'] = np.std(returns)
            metrics['return_skew'] = returns.skew()
            metrics['return_kurtosis'] = returns.kurtosis()
        
        # Range volatility
        daily_ranges = (data['high'] - data['low']) / data['low'] * 100
        metrics['avg_daily_range_pct'] = np.mean(daily_ranges)
        metrics['range_volatility'] = np.std(daily_ranges)
        
        # True Range
        if len(data) > 1:
            true_ranges = []
            for i in range(1, len(data)):
                high_low = data.iloc[i]['high'] - data.iloc[i]['low']
                high_close = abs(data.iloc[i]['high'] - data.iloc[i-1]['close'])
                low_close = abs(data.iloc[i]['low'] - data.iloc[i-1]['close'])
                true_ranges.append(max(high_low, high_close, low_close))
            
            metrics['avg_true_range'] = np.mean(true_ranges)
            metrics['atr_pct'] = (np.mean(true_ranges) / data['close'].mean()) * 100
        
        # Volatility trend
        if len(returns) >= 10:
            window_size = 5
            rolling_std = returns.rolling(window=window_size).std()
            rolling_std_clean = rolling_std.dropna()
            if len(rolling_std_clean) > 1:
                x = np.arange(len(rolling_std_clean))
                coeffs = np.polyfit(x, rolling_std_clean.values, 1)
                metrics['volatility_trend'] = coeffs[0]
                metrics['volatility_contracting'] = coeffs[0] < 0
        
        return metrics
    
    def _calculate_boundary_touch_metrics(self, data: pd.DataFrame, upper: float, lower: float) -> Dict:
        """Calculate how often price touches boundaries"""
        metrics = {}
        
        # Define touch as within 0.5% of boundary
        upper_touch_threshold = upper * 0.995
        lower_touch_threshold = lower * 1.005
        
        # Count touches
        upper_touches = 0
        lower_touches = 0
        
        for _, row in data.iterrows():
            if row['high'] >= upper_touch_threshold:
                upper_touches += 1
            if row['low'] <= lower_touch_threshold:
                lower_touches += 1
        
        metrics['upper_boundary_touches'] = upper_touches
        metrics['lower_boundary_touches'] = lower_touches
        metrics['total_boundary_touches'] = upper_touches + lower_touches
        metrics['touch_ratio'] = (upper_touches + lower_touches) / len(data)
        metrics['touch_balance'] = upper_touches / (upper_touches + lower_touches) if (upper_touches + lower_touches) > 0 else 0.5
        
        # Days since last touch
        last_upper_touch = None
        last_lower_touch = None
        
        for i in range(len(data) - 1, -1, -1):
            if data.iloc[i]['high'] >= upper_touch_threshold and last_upper_touch is None:
                last_upper_touch = i
            if data.iloc[i]['low'] <= lower_touch_threshold and last_lower_touch is None:
                last_lower_touch = i
            if last_upper_touch is not None and last_lower_touch is not None:
                break
        
        metrics['days_since_upper_touch'] = len(data) - 1 - last_upper_touch if last_upper_touch is not None else len(data)
        metrics['days_since_lower_touch'] = len(data) - 1 - last_lower_touch if last_lower_touch is not None else len(data)
        
        return metrics
    
    def _calculate_quality_metrics(self, data: pd.DataFrame, upper: float, lower: float) -> Dict:
        """Calculate consolidation quality metrics"""
        metrics = {}
        
        range_height = upper - lower
        
        # Tightness score (inverse of range width)
        metrics['tightness_score'] = 1 / (range_height / lower * 100) if lower > 0 else 0
        
        # Consistency score (how well price stays within range)
        violations = 0
        for _, row in data.iterrows():
            if row['high'] > upper * 1.01 or row['low'] < lower * 0.99:
                violations += 1
        
        metrics['consistency_score'] = 1 - (violations / len(data))
        
        # Compression score (decreasing volatility over time)
        if len(data) >= 10:
            first_half = data.iloc[:len(data)//2]
            second_half = data.iloc[len(data)//2:]
            
            first_half_volatility = np.std(first_half['close'].pct_change().dropna())
            second_half_volatility = np.std(second_half['close'].pct_change().dropna())
            
            metrics['compression_ratio'] = second_half_volatility / first_half_volatility if first_half_volatility > 0 else 1
            metrics['is_compressing'] = metrics['compression_ratio'] < 0.9
        
        # Balance score (equal time near upper and lower boundaries)
        mid_point = (upper + lower) / 2
        above_mid = len(data[data['close'] > mid_point])
        below_mid = len(data[data['close'] <= mid_point])
        
        metrics['balance_score'] = 1 - abs(above_mid - below_mid) / len(data)
        
        # Overall quality score
        quality_components = []
        if 'tightness_score' in metrics:
            quality_components.append(min(metrics['tightness_score'] * 10, 1))  # Normalize
        if 'consistency_score' in metrics:
            quality_components.append(metrics['consistency_score'])
        if 'balance_score' in metrics:
            quality_components.append(metrics['balance_score'])
        if 'compression_ratio' in metrics:
            quality_components.append(1 - min(metrics['compression_ratio'], 1))
        
        metrics['overall_quality_score'] = np.mean(quality_components) if quality_components else 0
        
        return metrics
    
    def calculate_post_breakout_metrics(self,
                                       price_data: pd.DataFrame,
                                       breakout_date: datetime,
                                       breakout_price: float,
                                       evaluation_days: List[int] = [5, 10, 20, 30, 50, 75, 100]) -> Dict:
        """Calculate metrics for post-breakout performance"""
        
        metrics = {}
        
        for days in evaluation_days:
            target_date = breakout_date + timedelta(days=days)
            
            # Find nearest trading day
            future_data = price_data[price_data.index >= target_date]
            
            if not future_data.empty:
                target_price = future_data.iloc[0]['close']
                
                # Calculate performance
                performance = ((target_price - breakout_price) / breakout_price) * 100
                
                metrics[f'performance_{days}d'] = performance
                metrics[f'price_{days}d'] = target_price
                
                # Calculate max/min in period
                period_data = price_data[(price_data.index >= breakout_date) & 
                                        (price_data.index <= target_date)]
                
                if not period_data.empty:
                    metrics[f'max_gain_{days}d'] = ((period_data['high'].max() - breakout_price) / breakout_price) * 100
                    metrics[f'max_loss_{days}d'] = ((period_data['low'].min() - breakout_price) / breakout_price) * 100
                    metrics[f'volatility_{days}d'] = np.std(period_data['close'].pct_change().dropna())
        
        return metrics
    
    def calculate_momentum_indicators(self, price_data: pd.DataFrame, end_date: datetime) -> Dict:
        """Calculate momentum indicators at pattern end"""
        
        # Get data up to end date
        data = price_data[price_data.index <= end_date].copy()
        
        if len(data) < 20:
            return {}
        
        metrics = {}
        
        # RSI
        close_prices = data['close'].values
        deltas = np.diff(close_prices)
        seed = deltas[:14]
        up = seed[seed >= 0].sum() / 14
        down = -seed[seed < 0].sum() / 14
        
        if down != 0:
            rs = up / down
            metrics['rsi'] = 100 - (100 / (1 + rs))
        else:
            metrics['rsi'] = 100
        
        # MACD
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        metrics['macd'] = macd.iloc[-1]
        metrics['macd_signal'] = signal.iloc[-1]
        metrics['macd_histogram'] = macd.iloc[-1] - signal.iloc[-1]
        
        # Stochastic
        low_14 = data['low'].rolling(window=14).min()
        high_14 = data['high'].rolling(window=14).max()
        
        k_percent = 100 * ((data['close'] - low_14) / (high_14 - low_14))
        metrics['stochastic_k'] = k_percent.iloc[-1]
        
        # Price relative to moving averages
        metrics['price_vs_sma20'] = (data['close'].iloc[-1] / data['close'].rolling(20).mean().iloc[-1] - 1) * 100
        if len(data) >= 50:
            metrics['price_vs_sma50'] = (data['close'].iloc[-1] / data['close'].rolling(50).mean().iloc[-1] - 1) * 100
        
        return metrics
    
    def calculate_gap_analysis(self, price_data: pd.DataFrame, start_date: datetime, end_date: datetime) -> Dict:
        """Analyze gaps in price data during consolidation"""
        
        pattern_data = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)].copy()
        
        if len(pattern_data) < 2:
            return {}
        
        metrics = {
            'gap_count': 0,
            'gap_up_count': 0,
            'gap_down_count': 0,
            'total_gap_size': 0,
            'avg_gap_size': 0,
            'max_gap_size': 0,
            'gaps_filled': 0
        }
        
        gaps = []
        
        for i in range(1, len(pattern_data)):
            prev_close = pattern_data.iloc[i-1]['close']
            curr_open = pattern_data.iloc[i]['open']
            
            gap_size = ((curr_open - prev_close) / prev_close) * 100
            
            # Consider it a gap if > 1%
            if abs(gap_size) > 1:
                metrics['gap_count'] += 1
                gaps.append(gap_size)
                
                if gap_size > 0:
                    metrics['gap_up_count'] += 1
                else:
                    metrics['gap_down_count'] += 1
                
                # Check if gap was filled
                if gap_size > 0:  # Gap up
                    if pattern_data.iloc[i]['low'] <= prev_close:
                        metrics['gaps_filled'] += 1
                else:  # Gap down
                    if pattern_data.iloc[i]['high'] >= prev_close:
                        metrics['gaps_filled'] += 1
        
        if gaps:
            metrics['total_gap_size'] = sum(abs(g) for g in gaps)
            metrics['avg_gap_size'] = np.mean([abs(g) for g in gaps])
            metrics['max_gap_size'] = max(abs(g) for g in gaps)
            metrics['gap_fill_rate'] = metrics['gaps_filled'] / metrics['gap_count']
        
        return metrics