"""
Statistical Analysis Module for Consolidation Patterns
Performs deep statistical analysis and comparisons across detection methods
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class StatisticalResult:
    """Container for statistical test results"""
    statistic: float
    p_value: float
    significant: bool
    effect_size: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    interpretation: str = ""

class StatisticalAnalyzer:
    """Comprehensive statistical analysis for consolidation patterns"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.analysis_results = {}
        
    def compare_methods_performance(self, patterns_by_method: Dict[str, List]) -> Dict:
        """Compare performance across different detection methods"""
        
        comparison_results = {
            'method_counts': {},
            'success_rates': {},
            'outcome_distributions': {},
            'statistical_tests': {},
            'best_method': None
        }
        
        # Prepare data for each method
        method_data = {}
        for method, patterns in patterns_by_method.items():
            # Extract outcomes
            outcomes = [p.get('outcome_class', 'K0') for p in patterns]
            gains = [p.get('max_gain', 0) for p in patterns if p.get('max_gain')]
            
            method_data[method] = {
                'patterns': patterns,
                'outcomes': outcomes,
                'gains': gains,
                'success_count': sum(1 for o in outcomes if o in ['K2', 'K3', 'K4']),
                'total_count': len(patterns)
            }
            
            comparison_results['method_counts'][method] = len(patterns)
            comparison_results['success_rates'][method] = (
                method_data[method]['success_count'] / method_data[method]['total_count']
                if method_data[method]['total_count'] > 0 else 0
            )
        
        # Statistical tests between methods
        if len(method_data) >= 2:
            comparison_results['statistical_tests'] = self._perform_method_comparisons(method_data)
        
        # Find best method
        if comparison_results['success_rates']:
            comparison_results['best_method'] = max(
                comparison_results['success_rates'].items(),
                key=lambda x: x[1]
            )
        
        self.analysis_results['method_comparison'] = comparison_results
        return comparison_results
    
    def _perform_method_comparisons(self, method_data: Dict) -> Dict:
        """Perform statistical tests between methods"""
        
        tests = {}
        methods = list(method_data.keys())
        
        # Pairwise comparisons
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                
                # Chi-square test for outcome distributions
                outcomes1 = method_data[method1]['outcomes']
                outcomes2 = method_data[method2]['outcomes']
                
                if outcomes1 and outcomes2:
                    chi2_result = self._chi_square_test(outcomes1, outcomes2)
                    
                    # T-test for gains
                    gains1 = method_data[method1]['gains']
                    gains2 = method_data[method2]['gains']
                    
                    t_test_result = None
                    if len(gains1) > 1 and len(gains2) > 1:
                        t_test_result = self._t_test(gains1, gains2)
                    
                    tests[f"{method1}_vs_{method2}"] = {
                        'chi_square': chi2_result,
                        't_test': t_test_result
                    }
        
        return tests
    
    def _chi_square_test(self, outcomes1: List, outcomes2: List) -> StatisticalResult:
        """Perform chi-square test on outcome distributions"""
        
        # Create contingency table
        outcome_classes = ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']
        
        counts1 = [outcomes1.count(oc) for oc in outcome_classes]
        counts2 = [outcomes2.count(oc) for oc in outcome_classes]
        
        contingency_table = np.array([counts1, counts2])
        
        # Perform test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Calculate Cramér's V for effect size
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0
        
        return StatisticalResult(
            statistic=chi2,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=cramers_v,
            interpretation=self._interpret_effect_size(cramers_v, 'cramers_v')
        )
    
    def _t_test(self, group1: List, group2: List) -> StatisticalResult:
        """Perform independent samples t-test"""
        
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Calculate Cohen's d for effect size
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Calculate confidence interval for mean difference
        se = pooled_std * np.sqrt(1/n1 + 1/n2)
        margin = 1.96 * se  # 95% CI
        ci = (mean1 - mean2 - margin, mean1 - mean2 + margin)
        
        return StatisticalResult(
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=cohens_d,
            confidence_interval=ci,
            interpretation=self._interpret_effect_size(cohens_d, 'cohens_d')
        )
    
    def _interpret_effect_size(self, effect_size: float, metric: str) -> str:
        """Interpret effect size based on metric type"""
        
        abs_effect = abs(effect_size)
        
        if metric == 'cohens_d':
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        elif metric == 'cramers_v':
            if abs_effect < 0.1:
                return "negligible"
            elif abs_effect < 0.3:
                return "small"
            elif abs_effect < 0.5:
                return "medium"
            else:
                return "large"
        else:
            return "unknown"
    
    def analyze_pattern_characteristics(self, patterns: List[Dict]) -> Dict:
        """Analyze relationships between pattern characteristics and outcomes"""
        
        if not patterns:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(patterns)
        
        # Ensure necessary columns exist
        required_cols = ['outcome_class', 'duration', 'boundary_width_pct', 'avg_bbw', 'avg_adx']
        existing_cols = [col for col in required_cols if col in df.columns]
        
        if not existing_cols:
            return {}
        
        analysis = {
            'correlations': {},
            'optimal_ranges': {},
            'feature_importance': {}
        }
        
        # Calculate correlations with success
        if 'outcome_class' in df.columns:
            # Create success binary variable
            df['success'] = df['outcome_class'].isin(['K2', 'K3', 'K4']).astype(int)
            df['exceptional'] = df['outcome_class'].isin(['K4']).astype(int)
            
            for col in existing_cols:
                if col != 'outcome_class' and col in df.columns:
                    # Calculate point-biserial correlation
                    if df[col].notna().sum() > 1:
                        corr, p_value = stats.pointbiserialr(df['success'], df[col].fillna(0))
                        analysis['correlations'][col] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'significant': p_value < self.significance_level
                        }
            
            # Find optimal ranges for successful patterns
            successful_patterns = df[df['success'] == 1]
            
            for col in existing_cols:
                if col != 'outcome_class' and col in successful_patterns.columns:
                    values = successful_patterns[col].dropna()
                    if len(values) > 0:
                        analysis['optimal_ranges'][col] = {
                            'mean': values.mean(),
                            'median': values.median(),
                            'q25': values.quantile(0.25),
                            'q75': values.quantile(0.75),
                            'optimal_range': (values.quantile(0.25), values.quantile(0.75))
                        }
        
        self.analysis_results['pattern_characteristics'] = analysis
        return analysis
    
    def analyze_temporal_patterns(self, patterns: List[Dict]) -> Dict:
        """Analyze temporal patterns and seasonality"""
        
        if not patterns:
            return {}
        
        temporal_analysis = {
            'monthly_distribution': {},
            'quarterly_distribution': {},
            'day_of_week_distribution': {},
            'seasonal_effects': {}
        }
        
        # Extract dates and outcomes
        dates = []
        outcomes = []
        
        for pattern in patterns:
            if 'start_date' in pattern and 'outcome_class' in pattern:
                dates.append(pd.to_datetime(pattern['start_date']))
                outcomes.append(pattern['outcome_class'])
        
        if not dates:
            return temporal_analysis
        
        df = pd.DataFrame({'date': dates, 'outcome': outcomes})
        df['success'] = df['outcome'].isin(['K2', 'K3', 'K4']).astype(int)
        
        # Monthly analysis
        df['month'] = df['date'].dt.month
        monthly_stats = df.groupby('month').agg({
            'success': ['count', 'mean'],
            'outcome': lambda x: x.value_counts().to_dict()
        })
        temporal_analysis['monthly_distribution'] = monthly_stats.to_dict()
        
        # Quarterly analysis
        df['quarter'] = df['date'].dt.quarter
        quarterly_stats = df.groupby('quarter').agg({
            'success': ['count', 'mean'],
            'outcome': lambda x: x.value_counts().to_dict()
        })
        temporal_analysis['quarterly_distribution'] = quarterly_stats.to_dict()
        
        # Day of week analysis
        df['day_of_week'] = df['date'].dt.dayofweek
        dow_stats = df.groupby('day_of_week').agg({
            'success': ['count', 'mean']
        })
        temporal_analysis['day_of_week_distribution'] = dow_stats.to_dict()
        
        # Test for seasonal effects
        if len(df) > 30:
            # ANOVA test for monthly differences
            monthly_groups = [group['success'].values for name, group in df.groupby('month') if len(group) > 1]
            if len(monthly_groups) > 2:
                f_stat, p_value = stats.f_oneway(*monthly_groups)
                temporal_analysis['seasonal_effects']['monthly_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level
                }
        
        self.analysis_results['temporal_patterns'] = temporal_analysis
        return temporal_analysis
    
    def analyze_market_regime_effects(self, patterns: List[Dict], market_data: pd.DataFrame = None) -> Dict:
        """Analyze how market conditions affect pattern outcomes"""
        
        regime_analysis = {
            'volatility_regimes': {},
            'trend_regimes': {},
            'volume_regimes': {}
        }
        
        if not patterns or market_data is None:
            return regime_analysis
        
        # Classify market regimes
        for pattern in patterns:
            if 'start_date' not in pattern:
                continue
                
            start_date = pd.to_datetime(pattern['start_date'])
            
            # Get market data around pattern start
            market_window = market_data[
                (market_data.index >= start_date - pd.Timedelta(days=30)) &
                (market_data.index <= start_date)
            ]
            
            if market_window.empty:
                continue
            
            # Classify volatility regime
            returns = market_window['close'].pct_change().dropna()
            volatility = returns.std()
            
            if volatility < returns.quantile(0.33):
                vol_regime = 'low'
            elif volatility < returns.quantile(0.67):
                vol_regime = 'medium'
            else:
                vol_regime = 'high'
            
            pattern['volatility_regime'] = vol_regime
            
            # Classify trend regime
            trend = (market_window['close'].iloc[-1] - market_window['close'].iloc[0]) / market_window['close'].iloc[0]
            
            if trend < -0.05:
                trend_regime = 'bearish'
            elif trend > 0.05:
                trend_regime = 'bullish'
            else:
                trend_regime = 'neutral'
            
            pattern['trend_regime'] = trend_regime
        
        # Analyze outcomes by regime
        df = pd.DataFrame(patterns)
        
        if 'volatility_regime' in df.columns:
            vol_outcomes = df.groupby('volatility_regime')['outcome_class'].value_counts()
            regime_analysis['volatility_regimes'] = vol_outcomes.to_dict()
        
        if 'trend_regime' in df.columns:
            trend_outcomes = df.groupby('trend_regime')['outcome_class'].value_counts()
            regime_analysis['trend_regimes'] = trend_outcomes.to_dict()
        
        self.analysis_results['market_regime_effects'] = regime_analysis
        return regime_analysis
    
    def calculate_risk_metrics(self, patterns: List[Dict]) -> Dict:
        """Calculate risk metrics for patterns"""
        
        risk_metrics = {
            'failure_rate': 0,
            'max_drawdown_stats': {},
            'risk_reward_ratios': {},
            'win_loss_ratios': {}
        }
        
        if not patterns:
            return risk_metrics
        
        # Calculate failure rate
        total = len(patterns)
        failures = sum(1 for p in patterns if p.get('outcome_class') == 'K5')
        risk_metrics['failure_rate'] = failures / total if total > 0 else 0
        
        # Calculate drawdowns
        drawdowns = []
        for pattern in patterns:
            if 'max_drawdown' in pattern:
                drawdowns.append(pattern['max_drawdown'])
        
        if drawdowns:
            risk_metrics['max_drawdown_stats'] = {
                'mean': np.mean(drawdowns),
                'median': np.median(drawdowns),
                'std': np.std(drawdowns),
                'worst': min(drawdowns),
                'best': max(drawdowns)
            }
        
        # Calculate risk-reward ratios
        gains = [p.get('max_gain', 0) for p in patterns if p.get('max_gain', 0) > 0]
        losses = [abs(p.get('max_loss', 0)) for p in patterns if p.get('max_loss', 0) < 0]
        
        if gains and losses:
            risk_metrics['risk_reward_ratios'] = {
                'average_gain': np.mean(gains),
                'average_loss': np.mean(losses),
                'ratio': np.mean(gains) / np.mean(losses) if np.mean(losses) > 0 else float('inf'),
                'kelly_criterion': self._calculate_kelly_criterion(patterns)
            }
        
        # Win/loss ratios
        wins = sum(1 for p in patterns if p.get('outcome_class') in ['K2', 'K3', 'K4'])
        losses = sum(1 for p in patterns if p.get('outcome_class') in ['K0', 'K1', 'K5'])
        
        risk_metrics['win_loss_ratios'] = {
            'wins': wins,
            'losses': losses,
            'win_rate': wins / total if total > 0 else 0,
            'win_loss_ratio': wins / losses if losses > 0 else float('inf')
        }
        
        self.analysis_results['risk_metrics'] = risk_metrics
        return risk_metrics
    
    def _calculate_kelly_criterion(self, patterns: List[Dict]) -> float:
        """Calculate Kelly Criterion for position sizing"""
        
        wins = []
        losses = []
        
        for pattern in patterns:
            gain = pattern.get('max_gain', 0)
            if gain > 0:
                wins.append(gain / 100)  # Convert to decimal
            elif gain < 0:
                losses.append(abs(gain) / 100)
        
        if not wins or not losses:
            return 0
        
        p = len(wins) / (len(wins) + len(losses))  # Win probability
        b = np.mean(wins) / np.mean(losses)  # Win/loss ratio
        
        # Kelly formula: f = p - q/b, where q = 1-p
        kelly = p - (1 - p) / b
        
        # Cap at 25% for safety
        return min(max(kelly, 0), 0.25)
    
    def generate_statistical_summary(self) -> Dict:
        """Generate comprehensive statistical summary"""
        
        summary = {
            'total_analyses': len(self.analysis_results),
            'key_findings': [],
            'recommendations': []
        }
        
        # Extract key findings
        if 'method_comparison' in self.analysis_results:
            best_method = self.analysis_results['method_comparison'].get('best_method')
            if best_method:
                summary['key_findings'].append(
                    f"Best performing method: {best_method[0]} with {best_method[1]:.1%} success rate"
                )
        
        if 'pattern_characteristics' in self.analysis_results:
            correlations = self.analysis_results['pattern_characteristics'].get('correlations', {})
            significant_factors = [
                f"{factor}: r={data['correlation']:.3f}"
                for factor, data in correlations.items()
                if data.get('significant', False)
            ]
            if significant_factors:
                summary['key_findings'].append(
                    f"Significant factors: {', '.join(significant_factors)}"
                )
        
        if 'risk_metrics' in self.analysis_results:
            risk = self.analysis_results['risk_metrics']
            if 'kelly_criterion' in risk.get('risk_reward_ratios', {}):
                kelly = risk['risk_reward_ratios']['kelly_criterion']
                summary['recommendations'].append(
                    f"Optimal position size (Kelly): {kelly:.1%}"
                )
        
        self.analysis_results['summary'] = summary
        return summary