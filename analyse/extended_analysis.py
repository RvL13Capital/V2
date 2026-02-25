"""
Extended Analysis Module with Comprehensive Metrics
Calculates detailed performance metrics for consolidation patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ExtendedPatternAnalyzer:
    """Extended analysis for consolidation patterns"""
    
    def __init__(self, patterns: List[Dict], price_data: Dict[str, pd.DataFrame] = None):
        self.patterns = patterns
        self.price_data = price_data or {}
        self.df = pd.DataFrame(patterns) if patterns else pd.DataFrame()
        
    def calculate_time_to_targets(self) -> Dict:
        """Calculate time to reach different gain targets"""
        
        targets = {
            'target_5pct': [],
            'target_10pct': [],
            'target_15pct': [],
            'target_25pct': [],
            'target_50pct': [],
            'target_max': []
        }
        
        for pattern in self.patterns:
            if 'max_gain' in pattern and pattern['max_gain'] > 0:
                # Simulate days to targets (simplified without actual price data)
                # In production, this would use actual price data
                max_gain = pattern['max_gain']
                
                # Only use real data - skip if days_to_target data not available
                # For now, only record the max target if we have actual data
                if 'days_to_max' in pattern and pattern['days_to_max'] is not None:
                    targets['target_max'].append(pattern['days_to_max'])
                    
                    # If we have detailed target data, use it
                    if 'days_to_5pct' in pattern and pattern['days_to_5pct'] is not None and max_gain >= 5:
                        targets['target_5pct'].append(pattern['days_to_5pct'])
                    if 'days_to_10pct' in pattern and pattern['days_to_10pct'] is not None and max_gain >= 10:
                        targets['target_10pct'].append(pattern['days_to_10pct'])
                    if 'days_to_15pct' in pattern and pattern['days_to_15pct'] is not None and max_gain >= 15:
                        targets['target_15pct'].append(pattern['days_to_15pct'])
                    if 'days_to_25pct' in pattern and pattern['days_to_25pct'] is not None and max_gain >= 25:
                        targets['target_25pct'].append(pattern['days_to_25pct'])
                    if 'days_to_50pct' in pattern and pattern['days_to_50pct'] is not None and max_gain >= 50:
                        targets['target_50pct'].append(pattern['days_to_50pct'])
        
        # Calculate statistics
        results = {}
        for target, days_list in targets.items():
            if days_list:
                results[target] = {
                    'count': len(days_list),
                    'avg_days': np.mean(days_list),
                    'median_days': np.median(days_list),
                    'min_days': np.min(days_list),
                    'max_days': np.max(days_list),
                    'std_days': np.std(days_list)
                }
            else:
                results[target] = {
                    'count': 0,
                    'avg_days': 0,
                    'median_days': 0,
                    'min_days': 0,
                    'max_days': 0,
                    'std_days': 0
                }
        
        return results
    
    def analyze_consolidation_duration_by_outcome(self) -> Dict:
        """Analyze consolidation duration differentiated by outcome"""
        
        if self.df.empty:
            return {}
        
        results = {}
        
        # Group by outcome class
        for outcome in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
            outcome_df = self.df[self.df['outcome_class'] == outcome]
            
            if not outcome_df.empty:
                durations = outcome_df['duration'].values
                
                results[outcome] = {
                    'count': len(durations),
                    'avg_duration': np.mean(durations),
                    'median_duration': np.median(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'std_duration': np.std(durations),
                    'distribution': self._get_duration_distribution(durations)
                }
                
                # Special analysis for successful vs failed
                if outcome in ['K2', 'K3', 'K4']:
                    results[outcome]['category'] = 'successful'
                elif outcome == 'K5':
                    results[outcome]['category'] = 'failed'
                else:
                    results[outcome]['category'] = 'neutral'
        
        # Overall statistics
        successful = self.df[self.df['outcome_class'].isin(['K2', 'K3', 'K4'])]
        failed = self.df[self.df['outcome_class'] == 'K5']
        
        results['successful_overall'] = {
            'count': len(successful),
            'avg_duration': successful['duration'].mean() if not successful.empty else 0,
            'median_duration': successful['duration'].median() if not successful.empty else 0
        }
        
        results['failed_overall'] = {
            'count': len(failed),
            'avg_duration': failed['duration'].mean() if not failed.empty else 0,
            'median_duration': failed['duration'].median() if not failed.empty else 0
        }
        
        return results
    
    def _get_duration_distribution(self, durations: np.ndarray) -> Dict:
        """Get duration distribution in buckets"""
        buckets = {
            '10-15': 0,
            '16-20': 0,
            '21-30': 0,
            '31-40': 0,
            '41-50': 0,
            '51-75': 0,
            '76-100': 0,
            '>100': 0
        }
        
        for d in durations:
            if d <= 15:
                buckets['10-15'] += 1
            elif d <= 20:
                buckets['16-20'] += 1
            elif d <= 30:
                buckets['21-30'] += 1
            elif d <= 40:
                buckets['31-40'] += 1
            elif d <= 50:
                buckets['41-50'] += 1
            elif d <= 75:
                buckets['51-75'] += 1
            elif d <= 100:
                buckets['76-100'] += 1
            else:
                buckets['>100'] += 1
        
        return buckets
    
    def analyze_false_breakouts(self) -> Dict:
        """Analyze upward breakouts that later broke down"""
        
        if self.df.empty:
            return {}
        
        # Patterns that initially broke upward but ended as K5 (failed)
        # This requires more detailed breakout direction data
        
        results = {
            'total_upward_breakouts': 0,
            'false_upward_breakouts': 0,
            'true_upward_breakouts': 0,
            'false_breakout_rate': 0,
            'characteristics': {}
        }
        
        # Count upward breakouts
        if 'breakout_direction' in self.df.columns:
            upward = self.df[self.df['breakout_direction'] == 'up']
            results['total_upward_breakouts'] = len(upward)
            
            # False breakouts (went up but ended in K5)
            false_breakouts = upward[upward['outcome_class'] == 'K5']
            results['false_upward_breakouts'] = len(false_breakouts)
            
            # True breakouts
            true_breakouts = upward[upward['outcome_class'].isin(['K2', 'K3', 'K4'])]
            results['true_upward_breakouts'] = len(true_breakouts)
            
            # False breakout rate
            if results['total_upward_breakouts'] > 0:
                results['false_breakout_rate'] = (
                    results['false_upward_breakouts'] / results['total_upward_breakouts'] * 100
                )
            
            # Characteristics of false breakouts
            if not false_breakouts.empty:
                results['characteristics'] = {
                    'avg_duration': false_breakouts['duration'].mean(),
                    'avg_boundary_width': false_breakouts['boundary_width_pct'].mean() if 'boundary_width_pct' in false_breakouts.columns else 0,
                    'avg_initial_gain': false_breakouts['max_gain'].mean() if false_breakouts['max_gain'].notna().any() else 0
                }
        
        return results
    
    def analyze_by_detection_method(self) -> Dict:
        """Analyze patterns by detection method"""
        
        if self.df.empty or 'detection_method' not in self.df.columns:
            return {}
        
        results = {}
        
        for method in self.df['detection_method'].unique():
            method_df = self.df[self.df['detection_method'] == method]
            
            # Outcome distribution
            outcome_dist = method_df['outcome_class'].value_counts().to_dict()
            
            # Calculate positive vs negative
            positive = method_df[method_df['outcome_class'].isin(['K2', 'K3', 'K4'])]
            negative = method_df[method_df['outcome_class'].isin(['K0', 'K1', 'K5'])]
            
            results[method] = {
                'total_patterns': len(method_df),
                'outcome_distribution': outcome_dist,
                'positive_patterns': len(positive),
                'negative_patterns': len(negative),
                'success_rate': len(positive) / len(method_df) * 100 if len(method_df) > 0 else 0,
                'avg_max_gain': method_df['max_gain'].mean(),
                'median_max_gain': method_df['max_gain'].median(),
                'exceptional_rate': len(method_df[method_df['outcome_class'] == 'K4']) / len(method_df) * 100 if len(method_df) > 0 else 0,
                'failure_rate': len(method_df[method_df['outcome_class'] == 'K5']) / len(method_df) * 100 if len(method_df) > 0 else 0
            }
        
        return results
    
    def get_pattern_quality_metrics(self) -> Dict:
        """Calculate pattern quality metrics"""
        
        if self.df.empty:
            return {}
        
        metrics = {
            'high_quality_patterns': {},
            'low_quality_patterns': {},
            'quality_indicators': {}
        }
        
        # High quality patterns (K3, K4)
        high_quality = self.df[self.df['outcome_class'].isin(['K3', 'K4'])]
        if not high_quality.empty:
            metrics['high_quality_patterns'] = {
                'count': len(high_quality),
                'avg_duration': high_quality['duration'].mean(),
                'avg_boundary_width': high_quality['boundary_width_pct'].mean() if 'boundary_width_pct' in high_quality.columns else 0,
                'avg_gain': high_quality['max_gain'].mean(),
                'tickers': high_quality['ticker'].value_counts().head(10).to_dict()
            }
        
        # Low quality patterns (K0, K5)
        low_quality = self.df[self.df['outcome_class'].isin(['K0', 'K5'])]
        if not low_quality.empty:
            metrics['low_quality_patterns'] = {
                'count': len(low_quality),
                'avg_duration': low_quality['duration'].mean(),
                'avg_boundary_width': low_quality['boundary_width_pct'].mean() if 'boundary_width_pct' in low_quality.columns else 0,
                'avg_loss': low_quality[low_quality['max_gain'] < 0]['max_gain'].mean() if any(low_quality['max_gain'] < 0) else 0
            }
        
        # Quality indicators
        if 'volume_contraction' in self.df.columns:
            successful = self.df[self.df['outcome_class'].isin(['K2', 'K3', 'K4'])]
            failed = self.df[self.df['outcome_class'] == 'K5']
            
            metrics['quality_indicators'] = {
                'successful_volume_contraction': successful['volume_contraction'].mean() if not successful.empty else 0,
                'failed_volume_contraction': failed['volume_contraction'].mean() if not failed.empty else 0,
                'successful_avg_range': successful['avg_range'].mean() if 'avg_range' in successful.columns and not successful.empty else 0,
                'failed_avg_range': failed['avg_range'].mean() if 'avg_range' in failed.columns and not failed.empty else 0
            }
        
        return metrics
    
    def get_example_patterns(self, n: int = 5) -> List[Dict]:
        """Get example patterns for detailed inspection"""
        
        if self.df.empty:
            return []
        
        examples = []
        
        # Get examples from different outcome classes
        for outcome in ['K4', 'K3', 'K2', 'K5', 'K0']:
            outcome_df = self.df[self.df['outcome_class'] == outcome]
            if not outcome_df.empty:
                # Get one example from this outcome class
                example = outcome_df.sample(n=min(1, len(outcome_df))).iloc[0].to_dict()
                examples.append(example)
                
                if len(examples) >= n:
                    break
        
        # If we need more examples, get random ones
        if len(examples) < n:
            remaining = n - len(examples)
            additional = self.df.sample(n=min(remaining, len(self.df)))
            for _, row in additional.iterrows():
                examples.append(row.to_dict())
        
        return examples
    
    def calculate_risk_reward_metrics(self) -> Dict:
        """Calculate risk/reward metrics"""
        
        if self.df.empty:
            return {}
        
        # Separate gains and losses
        gains = self.df[self.df['max_gain'] > 0]['max_gain'].values
        losses = self.df[self.df['max_gain'] < 0]['max_gain'].abs().values
        
        metrics = {
            'avg_gain': np.mean(gains) if len(gains) > 0 else 0,
            'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
            'win_rate': len(gains) / len(self.df) * 100 if len(self.df) > 0 else 0,
            'risk_reward_ratio': np.mean(gains) / np.mean(losses) if len(losses) > 0 and np.mean(losses) > 0 else 0,
            'profit_factor': np.sum(gains) / np.sum(losses) if len(losses) > 0 and np.sum(losses) > 0 else 0,
            'expectancy': (len(gains) / len(self.df) * np.mean(gains) - len(losses) / len(self.df) * np.mean(losses)) if len(self.df) > 0 else 0
        }
        
        # Sharpe ratio approximation
        if len(self.df) > 0:
            all_returns = self.df['max_gain'].values
            metrics['sharpe_ratio'] = np.mean(all_returns) / np.std(all_returns) if np.std(all_returns) > 0 else 0
        
        return metrics
    
    def calculate_strategic_value_metrics(self) -> Dict:
        """Calculate strategic value metrics based on the K0-K5 classification system"""
        
        if self.df.empty:
            return {}
        
        # Strategic value mapping
        value_map = {
            'K4': 10,    # Exceptional: >75% gain
            'K3': 3,     # Strong: 35-75% gain
            'K2': 1,     # Quality: 15-35% gain
            'K1': -0.2,  # Minimal: 5-15% gain
            'K0': -2,    # Stagnant: <5% gain
            'K5': -10    # Failed: Breakdown
        }
        
        # Calculate strategic values if not already present
        if 'strategic_value' not in self.df.columns:
            self.df['strategic_value'] = self.df['outcome_class'].map(value_map)
        
        metrics = {
            'total_strategic_value': self.df['strategic_value'].sum(),
            'avg_strategic_value': self.df['strategic_value'].mean(),
            'positive_value_patterns': len(self.df[self.df['strategic_value'] > 0]),
            'negative_value_patterns': len(self.df[self.df['strategic_value'] < 0]),
            'value_by_class': {},
            'expected_value': 0,
            'high_value_patterns': []
        }
        
        # Value distribution by class
        for outcome_class in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
            class_patterns = self.df[self.df['outcome_class'] == outcome_class]
            if not class_patterns.empty:
                metrics['value_by_class'][outcome_class] = {
                    'count': len(class_patterns),
                    'total_value': class_patterns['strategic_value'].sum(),
                    'percentage': len(class_patterns) / len(self.df) * 100,
                    'unit_value': value_map[outcome_class]
                }
        
        # Calculate expected value (probability-weighted)
        for outcome_class, value in value_map.items():
            prob = len(self.df[self.df['outcome_class'] == outcome_class]) / len(self.df) if len(self.df) > 0 else 0
            metrics['expected_value'] += prob * value
        
        # Identify high-value patterns (K3 and K4)
        high_value = self.df[self.df['outcome_class'].isin(['K3', 'K4'])].copy()
        if not high_value.empty:
            metrics['high_value_patterns'] = {
                'count': len(high_value),
                'percentage': len(high_value) / len(self.df) * 100,
                'avg_gain': high_value['max_gain'].mean(),
                'total_value': high_value['strategic_value'].sum(),
                'top_tickers': high_value['ticker'].value_counts().head(5).to_dict() if 'ticker' in high_value.columns else {}
            }
        
        # Risk analysis - patterns with negative value
        failed_patterns = self.df[self.df['outcome_class'] == 'K5']
        if not failed_patterns.empty:
            metrics['failure_analysis'] = {
                'count': len(failed_patterns),
                'percentage': len(failed_patterns) / len(self.df) * 100,
                'total_negative_value': failed_patterns['strategic_value'].sum(),
                'avg_loss': failed_patterns['max_gain'].mean() if 'max_gain' in failed_patterns.columns else 0
            }
        
        return metrics