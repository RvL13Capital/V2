"""
Performance Metrics for Backtesting
====================================

Comprehensive metrics for evaluating pattern detection performance
across full historical dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from config import OutcomeClass, STRATEGIC_VALUES, calculate_expected_value

logger = logging.getLogger(__name__)


@dataclass
class PatternStatistics:
    """Statistics for a set of patterns"""

    total_patterns: int = 0
    completed_patterns: int = 0
    active_patterns: int = 0
    failed_patterns: int = 0

    # Outcome distribution (K0-K5)
    outcome_distribution: Dict[int, int] = None

    # Success metrics
    success_rate_K2_plus: float = 0.0  # 15%+ gain
    success_rate_K3_plus: float = 0.0  # 35%+ gain
    success_rate_K4: float = 0.0        # 75%+ gain (exceptional)
    failure_rate_K5: float = 0.0        # Breakdown

    # Gain metrics
    avg_max_gain: float = 0.0
    median_max_gain: float = 0.0
    std_max_gain: float = 0.0

    # Strategic value metrics
    avg_strategic_value: float = 0.0
    total_strategic_value: float = 0.0

    # Duration metrics
    avg_days_in_pattern: float = 0.0
    avg_days_qualifying: float = 0.0
    avg_days_active: float = 0.0

    def __post_init__(self):
        if self.outcome_distribution is None:
            self.outcome_distribution = {}

    def __str__(self) -> str:
        return f"""
Pattern Statistics:
  Total Patterns: {self.total_patterns}
  Completed: {self.completed_patterns}
  Active: {self.active_patterns}
  Failed: {self.failed_patterns}

Success Rates:
  K2+ (15%+ gain): {self.success_rate_K2_plus:.1f}%
  K3+ (35%+ gain): {self.success_rate_K3_plus:.1f}%
  K4 (75%+ gain): {self.success_rate_K4:.1f}%
  K5 (Breakdown): {self.failure_rate_K5:.1f}%

Gain Metrics:
  Average: {self.avg_max_gain:.2f}%
  Median: {self.median_max_gain:.2f}%
  Std Dev: {self.std_max_gain:.2f}%

Strategic Value:
  Average per Pattern: {self.avg_strategic_value:.2f}
  Total Value: {self.total_strategic_value:.2f}

Duration:
  Avg Days in Pattern: {self.avg_days_in_pattern:.1f}
  Avg Days Qualifying: {self.avg_days_qualifying:.1f}
  Avg Days Active: {self.avg_days_active:.1f}
        """


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for backtesting results
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_statistics(self, patterns_df: pd.DataFrame,
                            completed_df: Optional[pd.DataFrame] = None,
                            active_df: Optional[pd.DataFrame] = None,
                            failed_df: Optional[pd.DataFrame] = None) -> PatternStatistics:
        """
        Calculate comprehensive statistics from pattern DataFrames

        Args:
            patterns_df: All patterns found
            completed_df: Completed patterns with outcomes
            active_df: Still-active patterns
            failed_df: Failed/broken down patterns

        Returns:
            PatternStatistics object
        """
        stats = PatternStatistics()

        # Count patterns
        stats.total_patterns = len(patterns_df)
        stats.completed_patterns = len(completed_df) if completed_df is not None else 0
        stats.active_patterns = len(active_df) if active_df is not None else 0
        stats.failed_patterns = len(failed_df) if failed_df is not None else 0

        # If no completed patterns, return basic stats
        if completed_df is None or len(completed_df) == 0:
            return stats

        # Outcome distribution
        if 'outcome_class' in completed_df.columns:
            stats.outcome_distribution = completed_df['outcome_class'].value_counts().to_dict()

            # Calculate success rates
            total = len(completed_df)
            stats.success_rate_K2_plus = (completed_df['outcome_class'] >= 2).sum() / total * 100
            stats.success_rate_K3_plus = (completed_df['outcome_class'] >= 3).sum() / total * 100
            stats.success_rate_K4 = (completed_df['outcome_class'] == 4).sum() / total * 100
            stats.failure_rate_K5 = (completed_df['outcome_class'] == 5).sum() / total * 100

        # Gain metrics
        if 'max_gain_pct' in completed_df.columns:
            stats.avg_max_gain = completed_df['max_gain_pct'].mean()
            stats.median_max_gain = completed_df['max_gain_pct'].median()
            stats.std_max_gain = completed_df['max_gain_pct'].std()

        # Strategic value metrics
        if 'strategic_value' in completed_df.columns:
            stats.avg_strategic_value = completed_df['strategic_value'].mean()
            stats.total_strategic_value = completed_df['strategic_value'].sum()

        # Duration metrics
        if 'days_in_pattern' in completed_df.columns:
            stats.avg_days_in_pattern = completed_df['days_in_pattern'].mean()

        if 'days_qualifying' in completed_df.columns:
            stats.avg_days_qualifying = completed_df['days_qualifying'].mean()

        if 'days_active' in completed_df.columns:
            stats.avg_days_active = completed_df['days_active'].mean()

        return stats

    def calculate_ev_correlation(self, predictions_df: pd.DataFrame) -> float:
        """
        Calculate correlation between predicted EV and actual strategic value

        Args:
            predictions_df: DataFrame with 'predicted_ev' and 'actual_value' columns

        Returns:
            Pearson correlation coefficient
        """
        if len(predictions_df) < 10:
            return 0.0

        if 'predicted_ev' not in predictions_df.columns or 'actual_value' not in predictions_df.columns:
            return 0.0

        correlation = predictions_df['predicted_ev'].corr(predictions_df['actual_value'])
        return correlation

    def calculate_signal_quality(self, predictions_df: pd.DataFrame,
                                 ev_threshold: float = 5.0) -> Dict[str, float]:
        """
        Calculate quality metrics for high-EV signals

        Args:
            predictions_df: DataFrame with predictions and outcomes
            ev_threshold: Minimum EV to consider as "signal"

        Returns:
            Dictionary of signal quality metrics
        """
        if len(predictions_df) == 0:
            return {}

        # Filter to high-EV signals
        signals = predictions_df[predictions_df['predicted_ev'] >= ev_threshold]

        if len(signals) == 0:
            return {
                'signal_count': 0,
                'signal_rate': 0.0
            }

        metrics = {
            'signal_count': len(signals),
            'signal_rate': len(signals) / len(predictions_df) * 100,
            'signal_avg_actual_gain': signals['max_gain_pct'].mean() if 'max_gain_pct' in signals.columns else 0.0,
            'signal_K3_plus_rate': (signals['outcome_class'] >= 3).mean() * 100 if 'outcome_class' in signals.columns else 0.0,
            'signal_K4_rate': (signals['outcome_class'] == 4).mean() * 100 if 'outcome_class' in signals.columns else 0.0,
            'signal_failure_rate': (signals['outcome_class'] == 5).mean() * 100 if 'outcome_class' in signals.columns else 0.0
        }

        return metrics

    def calculate_monthly_performance(self, patterns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics by month

        Args:
            patterns_df: DataFrame with patterns and outcomes

        Returns:
            DataFrame with monthly aggregated metrics
        """
        if len(patterns_df) == 0:
            return pd.DataFrame()

        df = patterns_df.copy()
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['year_month'] = df['start_date'].dt.to_period('M')

        monthly_metrics = df.groupby('year_month').agg({
            'ticker': 'count',
            'max_gain_pct': ['mean', 'median'],
            'strategic_value': ['mean', 'sum'],
            'days_in_pattern': 'mean'
        }).round(2)

        monthly_metrics.columns = ['_'.join(col).strip() for col in monthly_metrics.columns.values]
        monthly_metrics = monthly_metrics.rename(columns={'ticker_count': 'pattern_count'})

        return monthly_metrics

    def calculate_ticker_performance(self, patterns_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Calculate performance by ticker (top N performers)

        Args:
            patterns_df: DataFrame with patterns and outcomes
            top_n: Number of top tickers to return

        Returns:
            DataFrame with ticker performance metrics
        """
        if len(patterns_df) == 0:
            return pd.DataFrame()

        ticker_metrics = patterns_df.groupby('ticker').agg({
            'ticker': 'count',
            'max_gain_pct': ['mean', 'max'],
            'strategic_value': ['mean', 'sum'],
            'outcome_class': lambda x: (x >= 3).mean() * 100  # K3+ rate
        }).round(2)

        ticker_metrics.columns = ['pattern_count', 'avg_gain', 'max_gain', 'avg_value', 'total_value', 'K3_plus_rate']

        # Sort by total strategic value
        ticker_metrics = ticker_metrics.sort_values('total_value', ascending=False).head(top_n)

        return ticker_metrics

    def generate_performance_report(self, stats: PatternStatistics,
                                   monthly_perf: Optional[pd.DataFrame] = None,
                                   ticker_perf: Optional[pd.DataFrame] = None) -> str:
        """
        Generate comprehensive performance report

        Args:
            stats: Pattern statistics
            monthly_perf: Monthly performance DataFrame
            ticker_perf: Ticker performance DataFrame

        Returns:
            Formatted report string
        """
        report = []

        report.append("=" * 70)
        report.append("PERFORMANCE REPORT - FULL HISTORICAL BACKTEST")
        report.append("=" * 70)
        report.append("")

        # Pattern statistics
        report.append(str(stats))
        report.append("")

        # Monthly performance
        if monthly_perf is not None and len(monthly_perf) > 0:
            report.append("=" * 70)
            report.append("MONTHLY PERFORMANCE")
            report.append("=" * 70)
            report.append(monthly_perf.to_string())
            report.append("")

        # Ticker performance
        if ticker_perf is not None and len(ticker_perf) > 0:
            report.append("=" * 70)
            report.append("TOP PERFORMING TICKERS")
            report.append("=" * 70)
            report.append(ticker_perf.to_string())
            report.append("")

        return "\n".join(report)

    def calculate_risk_metrics(self, patterns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk-adjusted performance metrics

        Args:
            patterns_df: DataFrame with patterns and outcomes

        Returns:
            Dictionary of risk metrics
        """
        if len(patterns_df) == 0 or 'max_gain_pct' not in patterns_df.columns:
            return {}

        gains = patterns_df['max_gain_pct']

        metrics = {
            'sharpe_ratio': self._calculate_sharpe_ratio(gains),
            'max_drawdown': self._calculate_max_drawdown(gains),
            'win_rate': (gains > 0).mean() * 100,
            'loss_rate': (gains < 0).mean() * 100,
            'avg_win': gains[gains > 0].mean() if (gains > 0).any() else 0.0,
            'avg_loss': gains[gains < 0].mean() if (gains < 0).any() else 0.0,
            'profit_factor': self._calculate_profit_factor(gains)
        }

        return metrics

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_return = returns.mean() - risk_free_rate
        return excess_return / returns.std()

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns / 100).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100

        return drawdown.min()

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor (total wins / total losses)"""
        total_wins = returns[returns > 0].sum()
        total_losses = abs(returns[returns < 0].sum())

        if total_losses == 0:
            return float('inf') if total_wins > 0 else 0.0

        return total_wins / total_losses
