"""
Comprehensive Consolidation Pattern Analyzer
Analyzes patterns across all detection methods for the AIv3 System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConsolidationPattern:
    """Represents a detected consolidation pattern"""
    ticker: str
    start_date: datetime
    end_date: datetime
    detection_method: str
    upper_boundary: float
    lower_boundary: float
    power_boundary: float
    qualification_metrics: Dict[str, float] = field(default_factory=dict)
    breakout_date: Optional[datetime] = None
    breakout_direction: Optional[str] = None  # 'up', 'down', or None
    breakout_price: Optional[float] = None
    outcome_class: Optional[str] = None  # K0-K5
    max_gain: Optional[float] = None
    days_to_max: Optional[int] = None
    
class ConsolidationAnalyzer:
    """Main analyzer for consolidation patterns"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.patterns: List[ConsolidationPattern] = []
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.analysis_results: Dict[str, Any] = {}
        
    def load_patterns(self, patterns_file: str = None) -> None:
        """Load historical patterns from file or GCS"""
        if patterns_file:
            logger.info(f"Loading patterns from {patterns_file}")
            with open(patterns_file, 'r') as f:
                data = json.load(f)
                for pattern_dict in data:
                    pattern = self._dict_to_pattern(pattern_dict)
                    self.patterns.append(pattern)
        logger.info(f"Loaded {len(self.patterns)} patterns")
        
    def _dict_to_pattern(self, data: Dict) -> ConsolidationPattern:
        """Convert dictionary to ConsolidationPattern object"""
        return ConsolidationPattern(
            ticker=data['ticker'],
            start_date=pd.to_datetime(data['start_date']),
            end_date=pd.to_datetime(data['end_date']),
            detection_method=data.get('detection_method', 'stateful'),
            upper_boundary=data['upper_boundary'],
            lower_boundary=data['lower_boundary'],
            power_boundary=data.get('power_boundary', data['upper_boundary'] * 1.005),
            qualification_metrics=data.get('qualification_metrics', {}),
            breakout_date=pd.to_datetime(data.get('breakout_date')) if data.get('breakout_date') else None,
            breakout_direction=data.get('breakout_direction'),
            breakout_price=data.get('breakout_price'),
            outcome_class=data.get('outcome_class'),
            max_gain=data.get('max_gain'),
            days_to_max=data.get('days_to_max')
        )
    
    def analyze_consolidation_duration(self) -> Dict[str, Dict]:
        """Analyze consolidation duration statistics by detection method"""
        duration_stats = {}
        
        for method in set(p.detection_method for p in self.patterns):
            method_patterns = [p for p in self.patterns if p.detection_method == method]
            durations = [(p.end_date - p.start_date).days for p in method_patterns]
            
            if durations:
                duration_stats[method] = {
                    'count': len(durations),
                    'mean': np.mean(durations),
                    'median': np.median(durations),
                    'std': np.std(durations),
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'q25': np.percentile(durations, 25),
                    'q75': np.percentile(durations, 75),
                    'distribution': self._get_duration_distribution(durations)
                }
                
        self.analysis_results['duration_stats'] = duration_stats
        return duration_stats
    
    def _get_duration_distribution(self, durations: List[int]) -> Dict[str, int]:
        """Get duration distribution in buckets"""
        buckets = {
            '10-15 days': 0,
            '16-20 days': 0,
            '21-30 days': 0,
            '31-50 days': 0,
            '51-75 days': 0,
            '76-100 days': 0,
            '>100 days': 0
        }
        
        for d in durations:
            if d <= 15:
                buckets['10-15 days'] += 1
            elif d <= 20:
                buckets['16-20 days'] += 1
            elif d <= 30:
                buckets['21-30 days'] += 1
            elif d <= 50:
                buckets['31-50 days'] += 1
            elif d <= 75:
                buckets['51-75 days'] += 1
            elif d <= 100:
                buckets['76-100 days'] += 1
            else:
                buckets['>100 days'] += 1
                
        return buckets
    
    def analyze_breakout_outcomes(self) -> Dict[str, Dict]:
        """Analyze breakout directions and outcomes"""
        outcome_stats = {}
        
        for method in set(p.detection_method for p in self.patterns):
            method_patterns = [p for p in self.patterns if p.detection_method == method and p.breakout_direction]
            
            if method_patterns:
                up_breakouts = [p for p in method_patterns if p.breakout_direction == 'up']
                down_breakouts = [p for p in method_patterns if p.breakout_direction == 'down']
                
                outcome_stats[method] = {
                    'total_breakouts': len(method_patterns),
                    'up_breakouts': len(up_breakouts),
                    'down_breakouts': len(down_breakouts),
                    'up_ratio': len(up_breakouts) / len(method_patterns) if method_patterns else 0,
                    'outcome_distribution': self._get_outcome_distribution(method_patterns),
                    'avg_max_gain': np.mean([p.max_gain for p in method_patterns if p.max_gain]) if any(p.max_gain for p in method_patterns) else 0,
                    'success_rate': self._calculate_success_rate(method_patterns)
                }
                
        self.analysis_results['outcome_stats'] = outcome_stats
        return outcome_stats
    
    def _get_outcome_distribution(self, patterns: List[ConsolidationPattern]) -> Dict[str, int]:
        """Get distribution of outcome classes"""
        distribution = {
            'K0': 0,  # Stagnant
            'K1': 0,  # Minimal
            'K2': 0,  # Quality
            'K3': 0,  # Strong
            'K4': 0,  # Exceptional
            'K5': 0   # Failed
        }
        
        for p in patterns:
            if p.outcome_class and p.outcome_class in distribution:
                distribution[p.outcome_class] += 1
                
        return distribution
    
    def _calculate_success_rate(self, patterns: List[ConsolidationPattern]) -> float:
        """Calculate success rate (K2, K3, K4 outcomes)"""
        if not patterns:
            return 0
            
        successful = [p for p in patterns if p.outcome_class in ['K2', 'K3', 'K4']]
        return len(successful) / len(patterns)
    
    def analyze_qualification_metrics(self) -> Dict[str, Dict]:
        """Analyze qualification phase metrics"""
        metrics_analysis = {}
        
        # Collect all unique metrics
        all_metrics = set()
        for p in self.patterns:
            all_metrics.update(p.qualification_metrics.keys())
        
        for metric in all_metrics:
            values_by_outcome = {
                'K0': [], 'K1': [], 'K2': [], 
                'K3': [], 'K4': [], 'K5': []
            }
            
            for p in self.patterns:
                if metric in p.qualification_metrics and p.outcome_class:
                    values_by_outcome[p.outcome_class].append(p.qualification_metrics[metric])
            
            # Calculate statistics for each outcome class
            metric_stats = {}
            for outcome, values in values_by_outcome.items():
                if values:
                    metric_stats[outcome] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
            
            if metric_stats:
                metrics_analysis[metric] = metric_stats
        
        self.analysis_results['qualification_metrics'] = metrics_analysis
        return metrics_analysis
    
    def analyze_post_breakout_performance(self, days: List[int] = [5, 10, 20, 30, 50, 75, 100]) -> Dict:
        """Analyze price performance at specific intervals post-breakout"""
        performance_stats = {day: {'gains': [], 'losses': []} for day in days}
        
        for pattern in self.patterns:
            if not pattern.breakout_date or not pattern.breakout_price:
                continue
                
            ticker = pattern.ticker
            if ticker not in self.price_data:
                continue
                
            price_df = self.price_data[ticker]
            
            for day in days:
                target_date = pattern.breakout_date + timedelta(days=day)
                
                # Find price at target date
                if target_date in price_df.index:
                    target_price = price_df.loc[target_date, 'close']
                    performance = ((target_price - pattern.breakout_price) / pattern.breakout_price) * 100
                    
                    if performance >= 0:
                        performance_stats[day]['gains'].append(performance)
                    else:
                        performance_stats[day]['losses'].append(abs(performance))
        
        # Calculate statistics
        for day in days:
            gains = performance_stats[day]['gains']
            losses = performance_stats[day]['losses']
            
            performance_stats[day] = {
                'avg_gain': np.mean(gains) if gains else 0,
                'avg_loss': np.mean(losses) if losses else 0,
                'win_rate': len(gains) / (len(gains) + len(losses)) if (gains or losses) else 0,
                'max_gain': np.max(gains) if gains else 0,
                'max_loss': np.max(losses) if losses else 0,
                'total_patterns': len(gains) + len(losses)
            }
        
        self.analysis_results['post_breakout_performance'] = performance_stats
        return performance_stats
    
    def identify_optimal_patterns(self) -> List[Dict]:
        """Identify characteristics of most successful patterns"""
        successful_patterns = [p for p in self.patterns if p.outcome_class in ['K3', 'K4']]
        
        if not successful_patterns:
            return []
        
        optimal_characteristics = []
        
        # Analyze by duration
        durations = [(p.end_date - p.start_date).days for p in successful_patterns]
        optimal_duration = {
            'metric': 'consolidation_duration',
            'optimal_range': (np.percentile(durations, 25), np.percentile(durations, 75)),
            'mean': np.mean(durations),
            'median': np.median(durations)
        }
        optimal_characteristics.append(optimal_duration)
        
        # Analyze by qualification metrics
        for metric in ['avg_bbw', 'avg_adx', 'avg_volume_ratio', 'avg_range_ratio']:
            values = [p.qualification_metrics.get(metric, 0) for p in successful_patterns if metric in p.qualification_metrics]
            if values:
                optimal_characteristics.append({
                    'metric': metric,
                    'optimal_range': (np.percentile(values, 25), np.percentile(values, 75)),
                    'mean': np.mean(values),
                    'median': np.median(values)
                })
        
        # Analyze by boundary width
        boundary_widths = [(p.upper_boundary - p.lower_boundary) / p.lower_boundary * 100 for p in successful_patterns]
        optimal_characteristics.append({
            'metric': 'boundary_width_percent',
            'optimal_range': (np.percentile(boundary_widths, 25), np.percentile(boundary_widths, 75)),
            'mean': np.mean(boundary_widths),
            'median': np.median(boundary_widths)
        })
        
        self.analysis_results['optimal_patterns'] = optimal_characteristics
        return optimal_characteristics
    
    def compare_detection_methods(self) -> pd.DataFrame:
        """Compare performance across different detection methods"""
        methods = list(set(p.detection_method for p in self.patterns))
        comparison_data = []
        
        for method in methods:
            method_patterns = [p for p in self.patterns if p.detection_method == method]
            
            if not method_patterns:
                continue
                
            # Calculate various metrics
            durations = [(p.end_date - p.start_date).days for p in method_patterns]
            success_rate = len([p for p in method_patterns if p.outcome_class in ['K2', 'K3', 'K4']]) / len(method_patterns)
            exceptional_rate = len([p for p in method_patterns if p.outcome_class == 'K4']) / len(method_patterns)
            failure_rate = len([p for p in method_patterns if p.outcome_class == 'K5']) / len(method_patterns)
            
            comparison_data.append({
                'method': method,
                'total_patterns': len(method_patterns),
                'avg_duration': np.mean(durations),
                'success_rate': success_rate,
                'exceptional_rate': exceptional_rate,
                'failure_rate': failure_rate,
                'avg_max_gain': np.mean([p.max_gain for p in method_patterns if p.max_gain]) if any(p.max_gain for p in method_patterns) else 0
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        self.analysis_results['method_comparison'] = comparison_df
        return comparison_df
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        summary = {
            'total_patterns': len(self.patterns),
            'unique_tickers': len(set(p.ticker for p in self.patterns)),
            'date_range': {
                'earliest': min(p.start_date for p in self.patterns).strftime('%Y-%m-%d') if self.patterns else None,
                'latest': max(p.end_date for p in self.patterns).strftime('%Y-%m-%d') if self.patterns else None
            },
            'detection_methods': list(set(p.detection_method for p in self.patterns)),
            'overall_success_rate': len([p for p in self.patterns if p.outcome_class in ['K2', 'K3', 'K4']]) / len(self.patterns) if self.patterns else 0,
            'exceptional_patterns': len([p for p in self.patterns if p.outcome_class == 'K4']),
            'failed_patterns': len([p for p in self.patterns if p.outcome_class == 'K5']),
            'key_findings': self._generate_key_findings()
        }
        
        self.analysis_results['summary'] = summary
        return summary
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from analysis"""
        findings = []
        
        if 'duration_stats' in self.analysis_results:
            # Find method with highest success rate
            best_method = None
            best_rate = 0
            for method, stats in self.analysis_results.get('outcome_stats', {}).items():
                if stats['success_rate'] > best_rate:
                    best_rate = stats['success_rate']
                    best_method = method
            
            if best_method:
                findings.append(f"Best performing detection method: {best_method} with {best_rate:.1%} success rate")
        
        if 'optimal_patterns' in self.analysis_results:
            optimal = self.analysis_results['optimal_patterns']
            for char in optimal[:3]:  # Top 3 characteristics
                findings.append(f"Optimal {char['metric']}: {char['mean']:.2f} (median: {char['median']:.2f})")
        
        return findings
    
    def export_results(self, output_dir: str = './analysis_output') -> None:
        """Export all analysis results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export summary as JSON
        with open(output_path / 'analysis_summary.json', 'w') as f:
            json.dump(self.analysis_results.get('summary', {}), f, indent=2, default=str)
        
        # Export method comparison as CSV
        if 'method_comparison' in self.analysis_results:
            self.analysis_results['method_comparison'].to_csv(output_path / 'method_comparison.csv', index=False)
        
        # Export all results as comprehensive JSON
        with open(output_path / 'full_analysis_results.json', 'w') as f:
            # Convert dataframes to dict for JSON serialization
            results_copy = self.analysis_results.copy()
            if 'method_comparison' in results_copy:
                results_copy['method_comparison'] = results_copy['method_comparison'].to_dict('records')
            json.dump(results_copy, f, indent=2, default=str)
        
        logger.info(f"Results exported to {output_path}")