"""
Comprehensive Backtesting System for AIv3
Integrates Walk-Forward Validation with Extended Performance Metrics
Provides complete, publication-ready backtesting reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import json

from walk_forward_validator import WalkForwardValidator, WalkForwardWindow, WindowPerformance
from extended_performance_metrics import PerformanceMetricsCalculator, ExtendedMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveBacktester:
    """
    Complete backtesting system combining:
    - Walk-forward validation (temporal integrity)
    - Extended performance metrics (financial rigor)
    - Comprehensive reporting (professional quality)
    """

    def __init__(self,
                 initial_train_years: float = 2.0,
                 retrain_frequency_days: int = 90,
                 test_window_days: int = 100,
                 min_train_samples: int = 500,
                 risk_free_rate: float = 0.02):
        """
        Initialize comprehensive backtester

        Args:
            initial_train_years: Years for initial training (default: 2.0)
            retrain_frequency_days: Days between retraining (default: 90)
            test_window_days: Days for each test window (default: 100)
            min_train_samples: Minimum patterns for training (default: 500)
            risk_free_rate: Annual risk-free rate (default: 2%)
        """

        self.walk_forward = WalkForwardValidator(
            initial_train_days=int(initial_train_years * 365),
            retrain_frequency_days=retrain_frequency_days,
            test_window_days=test_window_days,
            min_train_samples=min_train_samples
        )

        self.metrics_calculator = PerformanceMetricsCalculator(
            risk_free_rate=risk_free_rate
        )

        self.results = {}
        self.config = {
            'initial_train_years': initial_train_years,
            'retrain_frequency_days': retrain_frequency_days,
            'test_window_days': test_window_days,
            'min_train_samples': min_train_samples,
            'risk_free_rate': risk_free_rate
        }

    def run_comprehensive_backtest(self,
                                   df: pd.DataFrame,
                                   model_class,
                                   feature_columns: List[str],
                                   target_column: str = 'outcome_class',
                                   date_column: str = 'start_date',
                                   model_params: Dict = None) -> Dict[str, Any]:
        """
        Run complete backtest with walk-forward validation and extended metrics

        Args:
            df: DataFrame with historical patterns
            model_class: Model class to train
            feature_columns: List of feature column names
            target_column: Target column name
            date_column: Date column name
            model_params: Optional model parameters

        Returns:
            Dictionary with comprehensive results
        """

        logger.info("="*100)
        logger.info("COMPREHENSIVE BACKTEST - AIv3 PATTERN DETECTION SYSTEM")
        logger.info("="*100)
        logger.info(f"Total Patterns: {len(df):,}")
        logger.info(f"Date Range: {df[date_column].min():%Y-%m-%d} to {df[date_column].max():%Y-%m-%d}")
        logger.info(f"Features: {len(feature_columns)}")
        logger.info("")

        # Step 1: Run Walk-Forward Validation
        logger.info("Step 1/3: Running Walk-Forward Validation...")
        window_performances = self.walk_forward.validate_model(
            df=df,
            model_class=model_class,
            feature_columns=feature_columns,
            target_column=target_column,
            date_column=date_column,
            model_params=model_params
        )

        # Step 2: Calculate Overall Extended Metrics
        logger.info("\nStep 2/3: Calculating Extended Performance Metrics...")

        # Combine all test period predictions
        all_predictions = []
        for window in self.walk_forward.windows:
            test_data = df[
                (df[date_column] >= window.test_start) &
                (df[date_column] < window.test_end)
            ].copy()
            all_predictions.append(test_data)

        combined_predictions = pd.concat(all_predictions, ignore_index=True)

        overall_metrics = self.metrics_calculator.calculate_all_metrics(
            predictions=combined_predictions,
            outcome_column=target_column
        )

        # Step 3: Calculate Per-Window Extended Metrics
        logger.info("\nStep 3/3: Calculating Per-Window Metrics...")
        window_extended_metrics = []

        for window in self.walk_forward.windows:
            test_data = df[
                (df[date_column] >= window.test_start) &
                (df[date_column] < window.test_end)
            ].copy()

            if len(test_data) > 0:
                window_metrics = self.metrics_calculator.calculate_all_metrics(
                    predictions=test_data,
                    outcome_column=target_column
                )
                window_extended_metrics.append(window_metrics)

        # Compile results
        self.results = {
            'config': self.config,
            'data_summary': {
                'total_patterns': len(df),
                'total_tickers': df['ticker'].nunique() if 'ticker' in df.columns else 0,
                'date_range': {
                    'start': df[date_column].min(),
                    'end': df[date_column].max(),
                    'days': (df[date_column].max() - df[date_column].min()).days
                },
                'outcome_distribution': df[target_column].value_counts().to_dict()
            },
            'walk_forward': {
                'num_windows': len(window_performances),
                'windows': [w.__dict__ for w in self.walk_forward.windows],
                'performances': [p.to_dict() for p in window_performances]
            },
            'overall_metrics': overall_metrics.to_dict(),
            'per_window_metrics': [m.to_dict() for m in window_extended_metrics],
            'timestamp': datetime.now().isoformat()
        }

        logger.info("\n✓ Backtest complete!")
        return self.results

    def generate_publication_report(self) -> str:
        """
        Generate publication-quality backtest report

        Returns:
            Formatted report string
        """

        if not self.results:
            return "No backtest results available. Run backtest first."

        report = []

        # Header
        report.append("="*120)
        report.append("AIv3 PATTERN DETECTION SYSTEM - COMPREHENSIVE BACKTEST REPORT")
        report.append("="*120)
        report.append(f"Report Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        report.append(f"Backtest Date: {self.results['timestamp']}")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-"*120)

        data_sum = self.results['data_summary']
        overall = self.results['overall_metrics']
        wf = self.results['walk_forward']

        report.append(f"Testing Period: {data_sum['date_range']['start']:%Y-%m-%d} to {data_sum['date_range']['end']:%Y-%m-%d} "
                     f"({data_sum['date_range']['days']} days)")
        report.append(f"Total Patterns Analyzed: {data_sum['total_patterns']:,}")
        report.append(f"Walk-Forward Windows: {wf['num_windows']}")
        report.append(f"Retraining Frequency: Every {self.config['retrain_frequency_days']} days")
        report.append("")

        report.append("Key Performance Indicators:")
        report.append(f"  • Overall Win Rate: {overall['win_rate']:.1%}")
        report.append(f"  • Sharpe Ratio: {overall['sharpe_ratio']:.2f}")
        report.append(f"  • Profit Factor: {overall['profit_factor']:.2f}")
        report.append(f"  • Maximum Drawdown: {overall['max_drawdown']:.1%}")
        report.append(f"  • Total Return: {overall['total_return']:+.1%}")
        report.append(f"  • Expected Value per Pattern: {overall['ev_per_pattern']:+.2f}")
        report.append("")

        # Pattern Outcome Analysis
        report.append("PATTERN OUTCOME DISTRIBUTION")
        report.append("-"*120)

        if 'outcome_distribution' in data_sum:
            total = sum(data_sum['outcome_distribution'].values())
            report.append(f"{'Outcome':<15} {'Count':<10} {'Percentage':<12} {'Strategic Value':<15} {'Description'}")
            report.append("-"*120)

            outcome_info = {
                'K4': ('+10', 'Exceptional (>75% gain)'),
                'K3': ('+3', 'Strong (35-75% gain)'),
                'K2': ('+1', 'Quality (15-35% gain)'),
                'K1': ('-0.2', 'Minimal (5-15% gain)'),
                'K0': ('-2', 'Stagnant (<5% gain)'),
                'K5': ('-10', 'Failed (breakdown)')
            }

            for outcome in ['K4', 'K3', 'K2', 'K1', 'K0', 'K5']:
                count = data_sum['outcome_distribution'].get(outcome, 0)
                pct = count / total * 100 if total > 0 else 0
                value, desc = outcome_info[outcome]
                report.append(f"{outcome:<15} {count:<10,} {pct:>10.1f}% {value:>14} {desc}")

        report.append("")

        # Walk-Forward Validation Results
        report.append("WALK-FORWARD VALIDATION RESULTS")
        report.append("-"*120)
        report.append(f"Methodology: {self.config['initial_train_years']:.1f}-year initial training, "
                     f"{self.config['retrain_frequency_days']}-day retraining frequency")
        report.append("")

        # Performance consistency across windows
        wf_perfs = wf['performances']
        report.append("Performance Consistency Analysis:")
        report.append(f"{'Metric':<25} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12} {'Stability'}")
        report.append("-"*120)

        metrics_to_analyze = [
            ('win_rate', 'Win Rate', '%'),
            ('sharpe_ratio', 'Sharpe Ratio', ''),
            ('profit_factor', 'Profit Factor', ''),
            ('max_drawdown', 'Max Drawdown', '%')
        ]

        for key, name, unit in metrics_to_analyze:
            values = [p[key] for p in wf_perfs]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)

            # Stability score (lower std relative to mean = more stable)
            stability = "High" if std_val / abs(mean_val) < 0.3 else "Medium" if std_val / abs(mean_val) < 0.5 else "Low"

            if unit == '%':
                report.append(f"{name:<25} {mean_val*100:>10.1f}% {std_val*100:>10.1f}% "
                            f"{min_val*100:>10.1f}% {max_val*100:>10.1f}% {stability}")
            else:
                report.append(f"{name:<25} {mean_val:>11.2f} {std_val:>11.2f} "
                            f"{min_val:>11.2f} {max_val:>11.2f} {stability}")

        report.append("")

        # Detailed per-window results
        report.append("Per-Window Performance Detail:")
        report.append(f"{'Window':<10} {'Test Period':<30} {'Win Rate':<12} {'Sharpe':<10} {'PF':<10} {'Max DD':<10}")
        report.append("-"*120)

        for i, perf in enumerate(wf_perfs):
            report.append(f"{i+1:<10} {perf['test_period']:<30} {perf['win_rate']*100:>10.1f}% "
                         f"{perf['sharpe_ratio']:>9.2f} {perf['profit_factor']:>9.2f} {perf['max_drawdown']*100:>9.1f}%")

        report.append("")

        # Extended Performance Metrics
        report.append("EXTENDED PERFORMANCE METRICS")
        report.append("-"*120)

        report.append("Return Metrics:")
        report.append(f"  • Total Return: {overall['total_return']:+.2%}")
        report.append(f"  • Average Return per Trade: {overall['avg_return']:+.2%}")
        report.append(f"  • Average Win: {overall['avg_win']:+.2%}")
        report.append(f"  • Average Loss: {overall['avg_loss']:+.2%}")
        report.append(f"  • Win/Loss Ratio: {overall['win_loss_ratio']:.2f}")
        report.append(f"  • Largest Win: {overall['largest_win']:+.2%}")
        report.append(f"  • Largest Loss: {overall['largest_loss']:+.2%}")
        report.append("")

        report.append("Risk-Adjusted Performance:")
        report.append(f"  • Sharpe Ratio: {overall['sharpe_ratio']:.3f}")
        report.append(f"  • Sortino Ratio: {overall['sortino_ratio']:.3f}")
        report.append(f"  • Calmar Ratio: {overall['calmar_ratio']:.3f}")
        report.append("")

        report.append("Drawdown Analysis:")
        report.append(f"  • Maximum Drawdown: {overall['max_drawdown']:.2%}")
        report.append(f"  • Max Drawdown Duration: {overall['max_drawdown_duration']} trades")
        report.append(f"  • Average Drawdown: {overall['avg_drawdown']:.2%}")
        report.append(f"  • Recovery Factor: {overall['recovery_factor']:.2f}")
        report.append("")

        report.append("Profit Metrics:")
        report.append(f"  • Profit Factor: {overall['profit_factor']:.3f}")
        report.append(f"  • Expectancy per Trade: {overall['expectancy']:+.4f}")
        report.append(f"  • Expected Value per Pattern: {overall['ev_per_pattern']:+.3f}")
        report.append("")

        report.append("Risk Metrics:")
        report.append(f"  • Value at Risk (95%): {overall['value_at_risk_95']:.2%}")
        report.append(f"  • Conditional VaR (95%): {overall['conditional_var_95']:.2%}")
        report.append(f"  • Max Consecutive Losses: {overall['max_consecutive_losses']}")
        report.append("")

        # Pattern Quality Analysis
        report.append("PATTERN QUALITY ANALYSIS")
        report.append("-"*120)
        report.append(f"K4 Exceptional Rate (>75% gains): {overall['k4_exceptional_rate']:.1%}")
        report.append(f"K3 Strong Rate (35-75% gains): {overall['k3_strong_rate']:.1%}")
        report.append(f"Combined K3+K4 Success Rate: {overall['k3_strong_rate'] + overall['k4_exceptional_rate']:.1%}")
        report.append(f"K5 Failure Rate (Breakdowns): {overall['k5_failure_rate']:.1%}")
        report.append("")

        # Strategy Assessment
        report.append("STRATEGY QUALITY ASSESSMENT")
        report.append("-"*120)

        quality_score = self._calculate_strategy_score(overall)
        report.append(f"Overall Quality Score: {quality_score}/10")
        report.append("")

        if quality_score >= 8:
            assessment = "EXCELLENT - Strategy demonstrates strong, consistent performance"
        elif quality_score >= 6:
            assessment = "GOOD - Strategy shows promise with room for optimization"
        elif quality_score >= 4:
            assessment = "FAIR - Strategy needs improvement before production deployment"
        else:
            assessment = "POOR - Strategy requires significant rework"

        report.append(f"Assessment: {assessment}")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-"*120)
        recommendations = self._generate_recommendations(overall, wf_perfs)
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")

        report.append("")
        report.append("="*120)
        report.append("END OF REPORT")
        report.append("="*120)

        return "\n".join(report)

    def _calculate_strategy_score(self, metrics: Dict) -> int:
        """Calculate overall strategy quality score (0-10)"""

        score = 0

        # Win rate (0-2 points)
        if metrics['win_rate'] > 0.30:
            score += 2
        elif metrics['win_rate'] > 0.20:
            score += 1

        # Sharpe ratio (0-2 points)
        if metrics['sharpe_ratio'] > 1.5:
            score += 2
        elif metrics['sharpe_ratio'] > 1.0:
            score += 1

        # Profit factor (0-2 points)
        if metrics['profit_factor'] > 2.0:
            score += 2
        elif metrics['profit_factor'] > 1.5:
            score += 1

        # Max drawdown (0-2 points)
        if metrics['max_drawdown'] > -0.15:
            score += 2
        elif metrics['max_drawdown'] > -0.25:
            score += 1

        # Expectancy (0-2 points)
        if metrics['expectancy'] > 0.05:
            score += 2
        elif metrics['expectancy'] > 0.02:
            score += 1

        return score

    def _generate_recommendations(self, overall: Dict, wf_perfs: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on results"""

        recommendations = []

        # Win rate recommendations
        if overall['win_rate'] < 0.20:
            recommendations.append("Win rate is below target (20%). Consider stricter pattern selection criteria or additional feature engineering.")

        # Sharpe ratio recommendations
        if overall['sharpe_ratio'] < 1.0:
            recommendations.append("Sharpe ratio below 1.0 indicates suboptimal risk-adjusted returns. Review position sizing and risk management.")

        # Profit factor recommendations
        if overall['profit_factor'] < 1.5:
            recommendations.append("Profit factor could be improved. Focus on reducing losing trade sizes or increasing average wins.")

        # Drawdown recommendations
        if overall['max_drawdown'] < -0.25:
            recommendations.append("Maximum drawdown exceeds 25%. Implement tighter stop losses or reduce position sizes during drawdown periods.")

        # Consistency recommendations
        win_rates = [p['win_rate'] for p in wf_perfs]
        if np.std(win_rates) / np.mean(win_rates) > 0.5:
            recommendations.append("High variability in win rates across windows suggests model instability. Consider ensemble methods or more robust features.")

        # K5 failure recommendations
        if overall['k5_failure_rate'] > 0.15:
            recommendations.append("K5 failure rate above 15%. Improve failure pattern detection to avoid false signals.")

        # Expectancy recommendations
        if overall['expectancy'] < 0:
            recommendations.append("Negative expectancy indicates unprofitable strategy. Do not deploy to production without major improvements.")

        # If good performance
        if len(recommendations) == 0:
            recommendations.append("Strategy shows strong performance across all key metrics. Consider paper trading before live deployment.")
            recommendations.append("Monitor performance closely for regime changes and be prepared to retrain model quarterly.")

        return recommendations

    def save_results(self, output_dir: str = 'output/comprehensive_backtest'):
        """Save all results to files"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save comprehensive report
        report = self.generate_publication_report()
        report_file = output_path / f'backtest_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_file}")

        # Save detailed results as JSON
        json_file = output_path / f'backtest_results_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results JSON saved to {json_file}")

        # Save performance summary as CSV
        summary_data = {
            'metric': [],
            'value': []
        }

        for key, value in self.results['overall_metrics'].items():
            summary_data['metric'].append(key)
            summary_data['value'].append(value)

        summary_df = pd.DataFrame(summary_data)
        csv_file = output_path / f'performance_summary_{timestamp}.csv'
        summary_df.to_csv(csv_file, index=False)
        logger.info(f"Performance CSV saved to {csv_file}")

        # Save walk-forward details
        wf_df = pd.DataFrame(self.results['walk_forward']['performances'])
        wf_file = output_path / f'walk_forward_details_{timestamp}.csv'
        wf_df.to_csv(wf_file, index=False)
        logger.info(f"Walk-forward CSV saved to {wf_file}")


if __name__ == "__main__":
    logger.info("Comprehensive Backtesting System for AIv3")
    logger.info("This module integrates walk-forward validation with extended performance metrics")
    logger.info("\nUsage:")
    logger.info("  backtester = ComprehensiveBacktester()")
    logger.info("  results = backtester.run_comprehensive_backtest(df, model_class, features)")
    logger.info("  report = backtester.generate_publication_report()")
    logger.info("  backtester.save_results()")
