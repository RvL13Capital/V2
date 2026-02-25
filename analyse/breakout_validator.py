"""
Breakout Validator for Consolidation Patterns
Identifies fake-outs, validates breakouts, and tracks re-entry patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class BreakoutDirection(Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"

class BreakoutQuality(Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    FAILED = "failed"
    FAKEOUT = "fakeout"

class BreakoutValidator:
    """Validates and analyzes breakout patterns"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_breakout(self,
                         price_data: pd.DataFrame,
                         breakout_date: datetime,
                         breakout_price: float,
                         upper_boundary: float,
                         lower_boundary: float,
                         power_boundary: float,
                         direction: str = None) -> Dict:
        """Comprehensive breakout validation"""
        
        # Determine breakout direction if not provided
        if direction is None:
            direction = self._determine_breakout_direction(
                breakout_price, upper_boundary, lower_boundary, power_boundary
            )
        
        validation = {
            'breakout_date': breakout_date,
            'breakout_price': breakout_price,
            'direction': direction,
            'is_valid': False,
            'is_fakeout': False,
            'quality': BreakoutQuality.FAILED.value,
            'confirmation_days': 0,
            'retest_count': 0,
            'sustainability_score': 0
        }
        
        # Get post-breakout data
        post_breakout = price_data[price_data.index >= breakout_date].copy()
        
        if len(post_breakout) < 2:
            return validation
        
        # Check for immediate confirmation
        validation.update(self._check_confirmation(
            post_breakout, breakout_price, upper_boundary, lower_boundary, direction
        ))
        
        # Check for fakeout
        validation.update(self._check_fakeout(
            post_breakout, upper_boundary, lower_boundary, direction
        ))
        
        # Assess breakout quality
        validation['quality'] = self._assess_breakout_quality(
            post_breakout, breakout_price, direction, validation['is_fakeout']
        )
        
        # Calculate sustainability
        validation['sustainability_score'] = self._calculate_sustainability(
            post_breakout, breakout_price, direction
        )
        
        # Check for retests
        validation['retest_count'] = self._count_retests(
            post_breakout, upper_boundary, lower_boundary, power_boundary, direction
        )
        
        # Determine overall validity
        validation['is_valid'] = (
            not validation['is_fakeout'] and
            validation['quality'] in [BreakoutQuality.STRONG.value, BreakoutQuality.MODERATE.value]
        )
        
        return validation
    
    def _determine_breakout_direction(self, price: float, upper: float, lower: float, power: float) -> str:
        """Determine breakout direction based on price and boundaries"""
        if price >= power:
            return BreakoutDirection.UP.value
        elif price <= lower:
            return BreakoutDirection.DOWN.value
        else:
            return BreakoutDirection.NONE.value
    
    def _check_confirmation(self, 
                           post_data: pd.DataFrame,
                           breakout_price: float,
                           upper: float,
                           lower: float,
                           direction: str) -> Dict:
        """Check for breakout confirmation"""
        
        confirmation = {
            'confirmation_days': 0,
            'confirmed': False,
            'confirmation_strength': 0
        }
        
        if direction == BreakoutDirection.UP.value:
            # For upward breakout, check if price stays above upper boundary
            consecutive_days = 0
            total_above = 0
            
            for i in range(min(10, len(post_data))):
                if post_data.iloc[i]['close'] > upper:
                    consecutive_days += 1
                    total_above += 1
                    if consecutive_days >= 3:
                        confirmation['confirmed'] = True
                        confirmation['confirmation_days'] = consecutive_days
                        break
                else:
                    consecutive_days = 0
            
            confirmation['confirmation_strength'] = total_above / min(10, len(post_data))
            
        elif direction == BreakoutDirection.DOWN.value:
            # For downward breakout, check if price stays below lower boundary
            consecutive_days = 0
            total_below = 0
            
            for i in range(min(10, len(post_data))):
                if post_data.iloc[i]['close'] < lower:
                    consecutive_days += 1
                    total_below += 1
                    if consecutive_days >= 3:
                        confirmation['confirmed'] = True
                        confirmation['confirmation_days'] = consecutive_days
                        break
                else:
                    consecutive_days = 0
            
            confirmation['confirmation_strength'] = total_below / min(10, len(post_data))
        
        return confirmation
    
    def _check_fakeout(self,
                      post_data: pd.DataFrame,
                      upper: float,
                      lower: float,
                      direction: str) -> Dict:
        """Check if breakout is a fakeout"""
        
        fakeout_info = {
            'is_fakeout': False,
            'fakeout_days': 0,
            'return_to_range_day': None,
            'fakeout_type': None
        }
        
        # Check first 10 days for return to consolidation range
        check_days = min(10, len(post_data))
        
        for i in range(1, check_days):  # Start from day 1, not 0 (breakout day)
            close = post_data.iloc[i]['close']
            
            if direction == BreakoutDirection.UP.value:
                # Check if price returns below upper boundary
                if close < upper * 0.995:  # Small buffer
                    fakeout_info['is_fakeout'] = True
                    fakeout_info['fakeout_days'] = i
                    fakeout_info['return_to_range_day'] = post_data.index[i]
                    
                    # Determine fakeout type
                    if close < lower:
                        fakeout_info['fakeout_type'] = 'reversal'  # Complete reversal
                    elif close < (upper + lower) / 2:
                        fakeout_info['fakeout_type'] = 'deep_pullback'
                    else:
                        fakeout_info['fakeout_type'] = 'shallow_pullback'
                    break
                    
            elif direction == BreakoutDirection.DOWN.value:
                # Check if price returns above lower boundary
                if close > lower * 1.005:  # Small buffer
                    fakeout_info['is_fakeout'] = True
                    fakeout_info['fakeout_days'] = i
                    fakeout_info['return_to_range_day'] = post_data.index[i]
                    
                    # Determine fakeout type
                    if close > upper:
                        fakeout_info['fakeout_type'] = 'reversal'
                    elif close > (upper + lower) / 2:
                        fakeout_info['fakeout_type'] = 'deep_bounce'
                    else:
                        fakeout_info['fakeout_type'] = 'shallow_bounce'
                    break
        
        return fakeout_info
    
    def _assess_breakout_quality(self,
                                post_data: pd.DataFrame,
                                breakout_price: float,
                                direction: str,
                                is_fakeout: bool) -> str:
        """Assess the quality of the breakout"""
        
        if is_fakeout:
            return BreakoutQuality.FAKEOUT.value
        
        if len(post_data) < 5:
            return BreakoutQuality.WEAK.value
        
        # Calculate metrics for quality assessment
        first_5_days = post_data.iloc[:5]
        
        if direction == BreakoutDirection.UP.value:
            # Calculate average gain in first 5 days
            avg_gain = ((first_5_days['close'].mean() - breakout_price) / breakout_price) * 100
            max_gain = ((first_5_days['high'].max() - breakout_price) / breakout_price) * 100
            
            # Check volume (if available)
            volume_increase = 1
            if 'volume' in post_data.columns and 'volume' in first_5_days.columns:
                pre_breakout_volume = post_data.iloc[0]['volume']
                post_breakout_volume = first_5_days['volume'].mean()
                if pre_breakout_volume > 0:
                    volume_increase = post_breakout_volume / pre_breakout_volume
            
            # Determine quality
            if avg_gain > 5 and max_gain > 7 and volume_increase > 1.5:
                return BreakoutQuality.STRONG.value
            elif avg_gain > 2 and max_gain > 4:
                return BreakoutQuality.MODERATE.value
            elif avg_gain > 0:
                return BreakoutQuality.WEAK.value
            else:
                return BreakoutQuality.FAILED.value
                
        elif direction == BreakoutDirection.DOWN.value:
            # Calculate average loss in first 5 days
            avg_loss = ((breakout_price - first_5_days['close'].mean()) / breakout_price) * 100
            max_loss = ((breakout_price - first_5_days['low'].min()) / breakout_price) * 100
            
            # Determine quality
            if avg_loss > 5 and max_loss > 7:
                return BreakoutQuality.STRONG.value
            elif avg_loss > 2 and max_loss > 4:
                return BreakoutQuality.MODERATE.value
            elif avg_loss > 0:
                return BreakoutQuality.WEAK.value
            else:
                return BreakoutQuality.FAILED.value
        
        return BreakoutQuality.WEAK.value
    
    def _calculate_sustainability(self,
                                 post_data: pd.DataFrame,
                                 breakout_price: float,
                                 direction: str) -> float:
        """Calculate breakout sustainability score (0-1)"""
        
        if len(post_data) < 20:
            return 0
        
        sustainability_factors = []
        
        # Factor 1: Trend continuation
        first_20_days = post_data.iloc[:20]
        
        if direction == BreakoutDirection.UP.value:
            # Check how many days close above breakout price
            days_above = len(first_20_days[first_20_days['close'] > breakout_price])
            sustainability_factors.append(days_above / 20)
            
            # Check if making higher highs
            highs = first_20_days['high'].values
            higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
            sustainability_factors.append(higher_highs / (len(highs) - 1))
            
        elif direction == BreakoutDirection.DOWN.value:
            # Check how many days close below breakout price
            days_below = len(first_20_days[first_20_days['close'] < breakout_price])
            sustainability_factors.append(days_below / 20)
            
            # Check if making lower lows
            lows = first_20_days['low'].values
            lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
            sustainability_factors.append(lower_lows / (len(lows) - 1))
        
        # Factor 2: Volatility (lower is better for sustainability)
        returns = first_20_days['close'].pct_change().dropna()
        if len(returns) > 0:
            volatility_score = 1 / (1 + np.std(returns) * 10)  # Normalize
            sustainability_factors.append(volatility_score)
        
        # Factor 3: Distance from breakout (should maintain or increase)
        final_close = first_20_days.iloc[-1]['close']
        if direction == BreakoutDirection.UP.value:
            distance_score = min(((final_close - breakout_price) / breakout_price) * 10, 1)
        else:
            distance_score = min(((breakout_price - final_close) / breakout_price) * 10, 1)
        sustainability_factors.append(max(0, distance_score))
        
        return np.mean(sustainability_factors) if sustainability_factors else 0
    
    def _count_retests(self,
                      post_data: pd.DataFrame,
                      upper: float,
                      lower: float,
                      power: float,
                      direction: str) -> int:
        """Count number of successful retests of breakout level"""
        
        retest_count = 0
        
        if len(post_data) < 5:
            return 0
        
        if direction == BreakoutDirection.UP.value:
            # Look for retests of upper boundary or power boundary
            retest_level = upper
            
            for i in range(5, min(50, len(post_data))):
                # Check if price comes back to test the level
                if post_data.iloc[i]['low'] <= retest_level * 1.01:  # Within 1% of level
                    # Check if it bounces (next day closes higher)
                    if i + 1 < len(post_data):
                        if post_data.iloc[i + 1]['close'] > post_data.iloc[i]['close']:
                            retest_count += 1
                            
        elif direction == BreakoutDirection.DOWN.value:
            # Look for retests of lower boundary
            retest_level = lower
            
            for i in range(5, min(50, len(post_data))):
                # Check if price comes back to test the level
                if post_data.iloc[i]['high'] >= retest_level * 0.99:  # Within 1% of level
                    # Check if it gets rejected (next day closes lower)
                    if i + 1 < len(post_data):
                        if post_data.iloc[i + 1]['close'] < post_data.iloc[i]['close']:
                            retest_count += 1
        
        return retest_count
    
    def analyze_breakout_patterns(self, patterns: List[Dict]) -> pd.DataFrame:
        """Analyze multiple breakout patterns and return summary statistics"""
        
        results = []
        
        for pattern in patterns:
            # Extract pattern information
            result = {
                'ticker': pattern.get('ticker'),
                'breakout_date': pattern.get('breakout_date'),
                'direction': pattern.get('direction'),
                'quality': pattern.get('quality'),
                'is_fakeout': pattern.get('is_fakeout', False),
                'sustainability_score': pattern.get('sustainability_score', 0),
                'retest_count': pattern.get('retest_count', 0)
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Calculate summary statistics
        if not df.empty:
            summary = {
                'total_breakouts': len(df),
                'up_breakouts': len(df[df['direction'] == BreakoutDirection.UP.value]),
                'down_breakouts': len(df[df['direction'] == BreakoutDirection.DOWN.value]),
                'fakeout_rate': len(df[df['is_fakeout'] == True]) / len(df),
                'strong_quality_rate': len(df[df['quality'] == BreakoutQuality.STRONG.value]) / len(df),
                'avg_sustainability': df['sustainability_score'].mean(),
                'avg_retests': df['retest_count'].mean()
            }
            
            logger.info(f"Breakout Analysis Summary: {summary}")
        
        return df
    
    def identify_re_entry_patterns(self,
                                  price_data: pd.DataFrame,
                                  original_breakout_date: datetime,
                                  upper_boundary: float,
                                  lower_boundary: float) -> List[Dict]:
        """Identify if price re-enters consolidation range after breakout"""
        
        re_entries = []
        
        # Get data after breakout
        post_breakout = price_data[price_data.index > original_breakout_date].copy()
        
        if len(post_breakout) < 2:
            return re_entries
        
        # Track if we're outside or inside the range
        currently_outside = True
        last_exit_date = original_breakout_date
        
        for i in range(len(post_breakout)):
            close = post_breakout.iloc[i]['close']
            date = post_breakout.index[i]
            
            # Check if price is within original consolidation range
            in_range = lower <= close <= upper
            
            if currently_outside and in_range:
                # Re-entry detected
                days_outside = (date - last_exit_date).days
                re_entries.append({
                    're_entry_date': date,
                    're_entry_price': close,
                    'days_outside_range': days_outside,
                    'previous_exit_date': last_exit_date
                })
                currently_outside = False
                
            elif not currently_outside and not in_range:
                # Exit from range
                currently_outside = True
                last_exit_date = date
        
        return re_entries