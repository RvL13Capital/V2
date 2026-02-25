"""
Extended Performance Metrics for AIv3 System
Provides comprehensive financial metrics beyond basic win rate
Includes: Profit Factor, Sharpe Ratio, Maximum Drawdown, and more
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtendedMetrics:
    """Container for extended performance metrics"""

    # Basic metrics
    total_patterns: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Return metrics
    total_return: float
    avg_return: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    recovery_factor: float

    # Profit metrics
    profit_factor: float
    expectancy: float
    win_loss_ratio: float

    # Pattern-specific metrics
    k4_exceptional_rate: float  # >75% gains
    k3_strong_rate: float  # 35-75% gains
    k5_failure_rate: float  # Breakdowns
    ev_per_pattern: float  # Expected value per pattern

    # Risk metrics
    value_at_risk_95: float  # VaR at 95% confidence
    conditional_var_95: float  # CVaR (Expected Shortfall)
    max_consecutive_losses: int

    # Time-based metrics
    avg_holding_period: float
    profit_per_day: float

    def to_dict(self):
        return asdict(self)


class PerformanceMetricsCalculator:
    """
    Calculates comprehensive performance metrics for trading strategies
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize calculator

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation (default: 2%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(self,
                             predictions: pd.DataFrame,
                             outcome_column: str = 'outcome_class',
                             gain_column: str = 'max_gain',
                             duration_column: str = 'days_to_max') -> ExtendedMetrics:
        """
        Calculate all performance metrics from pattern predictions

        Args:
            predictions: DataFrame with predictions and outcomes
            outcome_column: Column name for outcome classes (K0-K5)
            gain_column: Column name for actual gains/losses
            duration_column: Column name for holding period

        Returns:
            ExtendedMetrics object
        """

        # Prepare data
        df = predictions.copy()

        # Ensure we have required columns
        required_cols = [outcome_column]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate returns based on outcome classes
        returns = self._calculate_returns(df, outcome_column, gain_column)

        # Basic metrics
        total_patterns = len(df)
        total_trades = len(returns)
        winning_trades = (returns > 0).sum()
        losing_trades = (returns < 0).sum()
        win_rate = winning_trades / max(total_trades, 1)

        # Return metrics
        total_return = returns.sum()
        avg_return = returns.mean()
        avg_win = returns[returns > 0].mean() if winning_trades > 0 else 0.0
        avg_loss = returns[returns < 0].mean() if losing_trades > 0 else 0.0
        largest_win = returns.max() if len(returns) > 0 else 0.0
        largest_loss = returns.min() if len(returns) > 0 else 0.0

        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)

        # Drawdown metrics
        dd_metrics = self._calculate_drawdown_metrics(returns)

        # Calmar Ratio
        calmar_ratio = (avg_return * 252) / abs(dd_metrics['max_drawdown']) if dd_metrics['max_drawdown'] != 0 else 0.0

        # Profit metrics
        profit_factor = self._calculate_profit_factor(returns)
        expectancy = self._calculate_expectancy(returns)
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Pattern-specific metrics
        k4_rate = (df[outcome_column] == 'K4').mean() if outcome_column in df.columns else 0.0
        k3_rate = (df[outcome_column] == 'K3').mean() if outcome_column in df.columns else 0.0
        k5_rate = (df[outcome_column] == 'K5').mean() if outcome_column in df.columns else 0.0

        # Expected value per pattern
        ev_per_pattern = self._calculate_ev_per_pattern(df, outcome_column)

        # Risk metrics
        var_95 = self._calculate_value_at_risk(returns, confidence=0.95)
        cvar_95 = self._calculate_conditional_var(returns, confidence=0.95)
        max_consecutive_losses = self._calculate_max_consecutive_losses(returns)

        # Time-based metrics
        if duration_column in df.columns:
            avg_holding_period = df[duration_column].mean()
            profit_per_day = avg_return / avg_holding_period if avg_holding_period > 0 else 0.0
        else:
            avg_holding_period = 0.0
            profit_per_day = 0.0

        # Recovery factor
        recovery_factor = total_return / abs(dd_metrics['max_drawdown']) if dd_metrics['max_drawdown'] != 0 else 0.0

        return ExtendedMetrics(
            total_patterns=total_patterns,
            total_trades=total_trades,
            winning_trades=int(winning_trades),
            losing_trades=int(losing_trades),
            win_rate=win_rate,
            total_return=total_return,
            avg_return=avg_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=dd_metrics['max_drawdown'],
            max_drawdown_duration=dd_metrics['max_drawdown_duration'],
            avg_drawdown=dd_metrics['avg_drawdown'],
            recovery_factor=recovery_factor,
            profit_factor=profit_factor,
            expectancy=expectancy,
            win_loss_ratio=win_loss_ratio,
            k4_exceptional_rate=k4_rate,
            k3_strong_rate=k3_rate,
            k5_failure_rate=k5_rate,
            ev_per_pattern=ev_per_pattern,
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            max_consecutive_losses=max_consecutive_losses,
            avg_holding_period=avg_holding_period,
            profit_per_day=profit_per_day
        )

    def _calculate_returns(self,
                          df: pd.DataFrame,
                          outcome_column: str,
                          gain_column: str) -> np.ndarray:
        """
        Calculate returns based on outcome classes

        Uses actual gains if available, otherwise uses class-based estimates
        """

        if gain_column in df.columns:
            # Use actual gains/losses
            returns = df[gain_column].values
        else:
            # Estimate based on outcome classes
            outcome_to_return = {
                'K4': 0.75,   # >75% gain (use conservative 75%)
                'K3': 0.50,   # 35-75% gain (use midpoint)
                'K2': 0.25,   # 15-35% gain (use midpoint)
                'K1': 0.10,   # 5-15% gain (use midpoint)
                'K0': 0.00,   # <5% gain (stagnant)
                'K5': -0.10   # Breakdown (stop loss at -10%)
            }

            returns = df[outcome_column].map(outcome_to_return).fillna(0).values

        return returns

    def _calculate_sharpe_ratio(self, returns: np.ndarray, annualize: bool = True) -> float:
        """
        Calculate Sharpe Ratio

        Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns

        Args:
            returns: Array of returns
            annualize: Whether to annualize the result

        Returns:
            Sharpe ratio
        """

        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Daily risk-free rate
        daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1

        sharpe = (mean_return - daily_rf) / std_return

        # Annualize (assuming ~252 trading days)
        if annualize:
            sharpe *= np.sqrt(252)

        return sharpe

    def _calculate_sortino_ratio(self, returns: np.ndarray, annualize: bool = True) -> float:
        """
        Calculate Sortino Ratio (like Sharpe but only considers downside volatility)

        Sortino = (Mean Return - Risk-Free Rate) / Downside Deviation
        """

        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)

        # Daily risk-free rate
        daily_rf = (1 + self.risk_free_rate) ** (1/252) - 1

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < daily_rf]
        if len(downside_returns) == 0:
            return float('inf')  # No downside

        downside_dev = np.std(downside_returns, ddof=1)

        if downside_dev == 0:
            return 0.0

        sortino = (mean_return - daily_rf) / downside_dev

        if annualize:
            sortino *= np.sqrt(252)

        return sortino

    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive drawdown metrics

        Returns:
            Dictionary with max_drawdown, max_drawdown_duration, avg_drawdown
        """

        if len(returns) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'avg_drawdown': 0.0
            }

        # Calculate cumulative returns
        cumulative = np.cumsum(returns)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)

        # Calculate drawdown
        drawdown = cumulative - running_max

        # Maximum drawdown
        max_drawdown = np.min(drawdown)

        # Maximum drawdown duration (in number of trades)
        max_dd_duration = 0
        current_dd_duration = 0

        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0

        # Average drawdown (of all negative drawdowns)
        negative_dds = drawdown[drawdown < 0]
        avg_drawdown = np.mean(negative_dds) if len(negative_dds) > 0 else 0.0

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'avg_drawdown': avg_drawdown
        }

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """
        Calculate Profit Factor

        Profit Factor = Gross Profit / Gross Loss

        A value > 1 indicates profitable strategy
        """

        if len(returns) == 0:
            return 0.0

        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _calculate_expectancy(self, returns: np.ndarray) -> float:
        """
        Calculate expectancy (expected return per trade)

        Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
        """

        if len(returns) == 0:
            return 0.0

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = len(wins) / len(returns)
        loss_rate = len(losses) / len(returns)

        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0

        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        return expectancy

    def _calculate_ev_per_pattern(self, df: pd.DataFrame, outcome_column: str) -> float:
        """
        Calculate Expected Value per pattern using strategic value system

        K4 (Exceptional): >75% gain → Value: +10
        K3 (Strong): 35-75% gain → Value: +3
        K2 (Quality): 15-35% gain → Value: +1
        K1 (Minimal): 5-15% gain → Value: -0.2
        K0 (Stagnant): <5% gain → Value: -2
        K5 (Failed): Breakdown → Value: -10
        """

        strategic_values = {
            'K4': 10.0,
            'K3': 3.0,
            'K2': 1.0,
            'K1': -0.2,
            'K0': -2.0,
            'K5': -10.0
        }

        if outcome_column not in df.columns:
            return 0.0

        # Calculate EV
        ev = 0.0
        for outcome, value in strategic_values.items():
            count = (df[outcome_column] == outcome).sum()
            ev += (count / len(df)) * value

        return ev

    def _calculate_value_at_risk(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)

        VaR answers: "What is the maximum loss we can expect with X% confidence?"

        Args:
            returns: Array of returns
            confidence: Confidence level (default: 0.95 for 95%)

        Returns:
            VaR value (negative number indicating potential loss)
        """

        if len(returns) == 0:
            return 0.0

        # Calculate VaR at specified confidence level
        var = np.percentile(returns, (1 - confidence) * 100)

        return var

    def _calculate_conditional_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall

        CVaR answers: "If we exceed VaR, what's the expected loss?"

        Args:
            returns: Array of returns
            confidence: Confidence level (default: 0.95)

        Returns:
            CVaR value
        """

        if len(returns) == 0:
            return 0.0

        var = self._calculate_value_at_risk(returns, confidence)

        # CVaR is the average of all returns worse than VaR
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return var

        cvar = np.mean(tail_returns)

        return cvar

    def _calculate_max_consecutive_losses(self, returns: np.ndarray) -> int:
        """Calculate maximum consecutive losing trades"""

        if len(returns) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def generate_metrics_report(self, metrics: ExtendedMetrics) -> str:
        """Generate formatted report of all metrics"""

        report = []
        report.append("="*100)
        report.append("EXTENDED PERFORMANCE METRICS REPORT")
        report.append("="*100)
        report.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        report.append("")

        # Basic Statistics
        report.append("BASIC STATISTICS")
        report.append("-"*100)
        report.append(f"Total Patterns: {metrics.total_patterns:,}")
        report.append(f"Total Trades: {metrics.total_trades:,}")
        report.append(f"Winning Trades: {metrics.winning_trades:,} ({metrics.winning_trades/max(metrics.total_trades,1):.1%})")
        report.append(f"Losing Trades: {metrics.losing_trades:,} ({metrics.losing_trades/max(metrics.total_trades,1):.1%})")
        report.append(f"Win Rate: {metrics.win_rate:.1%}")
        report.append("")

        # Return Metrics
        report.append("RETURN METRICS")
        report.append("-"*100)
        report.append(f"Total Return: {metrics.total_return:+.2%}")
        report.append(f"Average Return per Trade: {metrics.avg_return:+.2%}")
        report.append(f"Average Win: {metrics.avg_win:+.2%}")
        report.append(f"Average Loss: {metrics.avg_loss:+.2%}")
        report.append(f"Largest Win: {metrics.largest_win:+.2%}")
        report.append(f"Largest Loss: {metrics.largest_loss:+.2%}")
        report.append(f"Win/Loss Ratio: {metrics.win_loss_ratio:.2f}")
        report.append("")

        # Risk-Adjusted Metrics
        report.append("RISK-ADJUSTED PERFORMANCE")
        report.append("-"*100)
        report.append(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        report.append(f"Sortino Ratio: {metrics.sortino_ratio:.3f}")
        report.append(f"Calmar Ratio: {metrics.calmar_ratio:.3f}")
        report.append("")

        # Drawdown Metrics
        report.append("DRAWDOWN ANALYSIS")
        report.append("-"*100)
        report.append(f"Maximum Drawdown: {metrics.max_drawdown:.2%}")
        report.append(f"Max Drawdown Duration: {metrics.max_drawdown_duration} trades")
        report.append(f"Average Drawdown: {metrics.avg_drawdown:.2%}")
        report.append(f"Recovery Factor: {metrics.recovery_factor:.2f}")
        report.append("")

        # Profit Metrics
        report.append("PROFIT METRICS")
        report.append("-"*100)
        report.append(f"Profit Factor: {metrics.profit_factor:.3f}")
        report.append(f"Expectancy per Trade: {metrics.expectancy:+.3f}")
        report.append(f"Expected Value per Pattern: {metrics.ev_per_pattern:+.3f}")
        report.append("")

        # Pattern-Specific Metrics
        report.append("PATTERN OUTCOME DISTRIBUTION")
        report.append("-"*100)
        report.append(f"K4 Exceptional Rate (>75% gains): {metrics.k4_exceptional_rate:.1%}")
        report.append(f"K3 Strong Rate (35-75% gains): {metrics.k3_strong_rate:.1%}")
        report.append(f"K5 Failure Rate (Breakdowns): {metrics.k5_failure_rate:.1%}")
        report.append("")

        # Risk Metrics
        report.append("RISK METRICS")
        report.append("-"*100)
        report.append(f"Value at Risk (95%): {metrics.value_at_risk_95:.2%}")
        report.append(f"Conditional VaR (95%): {metrics.conditional_var_95:.2%}")
        report.append(f"Max Consecutive Losses: {metrics.max_consecutive_losses}")
        report.append("")

        # Time-Based Metrics
        if metrics.avg_holding_period > 0:
            report.append("TIME-BASED METRICS")
            report.append("-"*100)
            report.append(f"Average Holding Period: {metrics.avg_holding_period:.1f} days")
            report.append(f"Profit per Day: {metrics.profit_per_day:+.3%}")
            report.append("")

        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-"*100)

        # Quality assessment
        quality_score = self._assess_strategy_quality(metrics)
        report.append(f"Strategy Quality Score: {quality_score}/10")
        report.append("")
        report.append("Key Strengths:" if quality_score >= 7 else "Areas for Improvement:")

        if metrics.sharpe_ratio > 1.0:
            report.append("  ✓ Strong risk-adjusted returns (Sharpe > 1.0)")
        if metrics.profit_factor > 1.5:
            report.append("  ✓ Excellent profit factor (>1.5)")
        if metrics.max_drawdown > -0.20:
            report.append("  ✓ Controlled drawdown (<20%)")
        if metrics.win_rate > 0.25:
            report.append("  ✓ High win rate (>25%)")

        report.append("")
        report.append("="*100)

        return "\n".join(report)

    def _assess_strategy_quality(self, metrics: ExtendedMetrics) -> int:
        """Assess overall strategy quality on a 0-10 scale"""

        score = 0

        # Win rate (0-2 points)
        if metrics.win_rate > 0.30:
            score += 2
        elif metrics.win_rate > 0.20:
            score += 1

        # Sharpe ratio (0-2 points)
        if metrics.sharpe_ratio > 1.5:
            score += 2
        elif metrics.sharpe_ratio > 1.0:
            score += 1

        # Profit factor (0-2 points)
        if metrics.profit_factor > 2.0:
            score += 2
        elif metrics.profit_factor > 1.5:
            score += 1

        # Max drawdown (0-2 points)
        if metrics.max_drawdown > -0.15:
            score += 2
        elif metrics.max_drawdown > -0.25:
            score += 1

        # Expectancy (0-2 points)
        if metrics.expectancy > 0.05:
            score += 2
        elif metrics.expectancy > 0.02:
            score += 1

        return score


if __name__ == "__main__":
    # Example usage
    logger.info("Extended Performance Metrics Calculator")
    logger.info("This module provides comprehensive financial metrics for trading strategies")

    # Create example data
    np.random.seed(42)
    example_outcomes = np.random.choice(['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                                       size=100,
                                       p=[0.20, 0.20, 0.25, 0.20, 0.10, 0.05])

    example_df = pd.DataFrame({
        'outcome_class': example_outcomes,
        'days_to_max': np.random.randint(5, 100, size=100)
    })

    # Calculate metrics
    calculator = PerformanceMetricsCalculator(risk_free_rate=0.02)
    metrics = calculator.calculate_all_metrics(example_df)

    # Generate report
    report = calculator.generate_metrics_report(metrics)
    print(report)
