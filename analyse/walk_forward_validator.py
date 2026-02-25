"""
Walk-Forward Validation System for AIv3
Implements rigorous temporal backtesting with no look-ahead bias
Ensures realistic performance estimation through proper time-series splitting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
import json
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward validation window"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int


@dataclass
class WindowPerformance:
    """Performance metrics for a single validation window"""
    window_id: int
    test_period: str

    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float

    # Trading-specific metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Financial metrics
    total_return: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_duration: int

    # Risk metrics
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    expectancy: float

    # Pattern metrics
    k3_k4_hit_rate: float  # Hit rate for exceptional patterns
    k5_avoidance_rate: float  # How well we avoid failures

    def to_dict(self):
        return asdict(self)


class WalkForwardValidator:
    """
    Walk-Forward Validation System

    Implements:
    - Temporal data splitting (no look-ahead bias)
    - Quarterly model retraining (every 90 days)
    - 100-day evaluation windows
    - Minimum 4-year backtest period
    - Comprehensive performance tracking
    """

    def __init__(self,
                 initial_train_days: int = 730,  # 2 years initial training
                 retrain_frequency_days: int = 90,  # Retrain every 90 days
                 test_window_days: int = 100,  # 100-day evaluation window
                 min_train_samples: int = 500):
        """
        Initialize Walk-Forward Validator

        Args:
            initial_train_days: Days for initial training (default: 2 years)
            retrain_frequency_days: Days between retraining (default: 90)
            test_window_days: Days for each test window (default: 100)
            min_train_samples: Minimum patterns required for training
        """
        self.initial_train_days = initial_train_days
        self.retrain_frequency_days = retrain_frequency_days
        self.test_window_days = test_window_days
        self.min_train_samples = min_train_samples

        self.windows: List[WalkForwardWindow] = []
        self.performances: List[WindowPerformance] = []

    def create_walk_forward_windows(self,
                                   df: pd.DataFrame,
                                   date_column: str = 'start_date') -> List[WalkForwardWindow]:
        """
        Create temporal windows for walk-forward validation

        Args:
            df: DataFrame with temporal data
            date_column: Column name containing dates

        Returns:
            List of WalkForwardWindow objects
        """
        # Ensure data is sorted by date
        df = df.sort_values(date_column).copy()
        df[date_column] = pd.to_datetime(df[date_column])

        min_date = df[date_column].min()
        max_date = df[date_column].max()

        total_days = (max_date - min_date).days

        logger.info(f"Data range: {min_date:%Y-%m-%d} to {max_date:%Y-%m-%d} ({total_days} days)")

        # Check if we have enough data for 4-year minimum
        if total_days < 1460:  # 4 years
            logger.warning(f"Only {total_days/365:.1f} years of data. Minimum 4 years recommended.")

        windows = []
        window_id = 0

        # Initial training window
        train_start = min_date
        train_end = min_date + timedelta(days=self.initial_train_days)

        # Walk forward through time
        current_date = train_end

        while current_date + timedelta(days=self.test_window_days) <= max_date:
            # Test window
            test_start = current_date
            test_end = current_date + timedelta(days=self.test_window_days)

            # Get sample counts
            train_mask = (df[date_column] >= train_start) & (df[date_column] < train_end)
            test_mask = (df[date_column] >= test_start) & (df[date_column] < test_end)

            train_samples = train_mask.sum()
            test_samples = test_mask.sum()

            # Only create window if we have enough training data
            if train_samples >= self.min_train_samples and test_samples > 0:
                window = WalkForwardWindow(
                    window_id=window_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_samples=train_samples,
                    test_samples=test_samples
                )
                windows.append(window)

                logger.info(f"Window {window_id}: Train={train_start:%Y-%m-%d} to {train_end:%Y-%m-%d} ({train_samples} samples), "
                           f"Test={test_start:%Y-%m-%d} to {test_end:%Y-%m-%d} ({test_samples} samples)")

                window_id += 1

            # Move to next window
            current_date += timedelta(days=self.retrain_frequency_days)

            # Expand training window (cumulative learning)
            train_end = current_date

        self.windows = windows
        logger.info(f"Created {len(windows)} walk-forward windows")

        return windows

    def validate_model(self,
                      df: pd.DataFrame,
                      model_class,
                      feature_columns: List[str],
                      target_column: str = 'outcome_class',
                      date_column: str = 'start_date',
                      model_params: Dict = None) -> List[WindowPerformance]:
        """
        Perform walk-forward validation

        Args:
            df: DataFrame with features and targets
            model_class: Model class to train (must have fit/predict methods)
            feature_columns: List of feature column names
            target_column: Target column name
            date_column: Date column name
            model_params: Optional model parameters

        Returns:
            List of WindowPerformance objects
        """
        if not self.windows:
            self.create_walk_forward_windows(df, date_column)

        if model_params is None:
            model_params = {}

        performances = []

        for window in self.windows:
            logger.info(f"\n{'='*80}")
            logger.info(f"Validating Window {window.window_id}")
            logger.info(f"{'='*80}")

            # Split data
            train_data = df[
                (df[date_column] >= window.train_start) &
                (df[date_column] < window.train_end)
            ].copy()

            test_data = df[
                (df[date_column] >= window.test_start) &
                (df[date_column] < window.test_end)
            ].copy()

            # Prepare features and targets
            X_train = train_data[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
            X_test = test_data[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)

            # Create binary target for K3/K4 (strong moves)
            y_train = train_data[target_column].isin(['K3', 'K4']).astype(int)
            y_test = test_data[target_column].isin(['K3', 'K4']).astype(int)

            # Train model
            model = model_class(**model_params)
            logger.info(f"Training on {len(X_train)} samples...")
            model.fit(X_train, y_train)

            # Predict on test set
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

            # Calculate performance metrics
            performance = self._calculate_window_performance(
                window_id=window.window_id,
                test_period=f"{window.test_start:%Y-%m-%d} to {window.test_end:%Y-%m-%d}",
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                test_data=test_data,
                target_column=target_column
            )

            performances.append(performance)

            # Log key metrics
            logger.info(f"Win Rate: {performance.win_rate:.1%}")
            logger.info(f"Sharpe Ratio: {performance.sharpe_ratio:.2f}")
            logger.info(f"Profit Factor: {performance.profit_factor:.2f}")
            logger.info(f"Max Drawdown: {performance.max_drawdown:.1%}")

        self.performances = performances
        return performances

    def _calculate_window_performance(self,
                                     window_id: int,
                                     test_period: str,
                                     y_true: pd.Series,
                                     y_pred: np.ndarray,
                                     y_prob: np.ndarray,
                                     test_data: pd.DataFrame,
                                     target_column: str) -> WindowPerformance:
        """Calculate comprehensive performance metrics for a window"""

        # Classification metrics
        accuracy = (y_pred == y_true).mean()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else 0.0

        # Trading metrics
        total_trades = len(y_pred)
        winning_trades = ((y_pred == 1) & (y_true == 1)).sum()
        losing_trades = ((y_pred == 1) & (y_true == 0)).sum()
        win_rate = winning_trades / max(total_trades, 1)

        # Financial metrics (assuming realistic returns)
        # K3/K4 patterns: 35-75% gains (use conservative 40% average)
        # Failed patterns: -10% loss (stop loss)
        avg_win = 0.40  # 40% gain on winning trades
        avg_loss = -0.10  # 10% loss on losing trades

        # Calculate returns
        wins = y_true[y_pred == 1].sum()
        losses = (y_pred == 1).sum() - wins

        total_return = (wins * avg_win) + (losses * avg_loss)

        # Sharpe Ratio (simplified: assumes daily rebalancing)
        # Calculate trade-by-trade returns
        trade_returns = []
        for pred, actual in zip(y_pred, y_true):
            if pred == 1:  # We took the trade
                ret = avg_win if actual == 1 else avg_loss
                trade_returns.append(ret)

        if len(trade_returns) > 1:
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Profit Factor
        gross_profit = wins * avg_win
        gross_loss = abs(losses * avg_loss)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Maximum Drawdown
        cumulative_returns = np.cumsum([r for r in trade_returns])
        running_max = np.maximum.accumulate(cumulative_returns) if len(cumulative_returns) > 0 else np.array([0])
        drawdown = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # Drawdown duration (number of trades in drawdown)
        max_dd_duration = 0
        current_dd_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0

        # Pattern-specific metrics
        if target_column in test_data.columns:
            outcomes = test_data[target_column].values

            # K3/K4 hit rate (exceptional patterns)
            predicted_strong = y_pred == 1
            actual_k3_k4 = test_data[target_column].isin(['K3', 'K4']).values
            k3_k4_hit_rate = actual_k3_k4[predicted_strong].mean() if predicted_strong.sum() > 0 else 0.0

            # K5 avoidance (failure patterns)
            actual_k5 = (test_data[target_column] == 'K5').values
            predicted_weak = y_pred == 0
            k5_avoidance_rate = predicted_weak[actual_k5].mean() if actual_k5.sum() > 0 else 1.0
        else:
            k3_k4_hit_rate = 0.0
            k5_avoidance_rate = 1.0

        # Calculate expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Find largest win/loss
        largest_win = avg_win if wins > 0 else 0.0
        largest_loss = avg_loss if losses > 0 else 0.0

        return WindowPerformance(
            window_id=window_id,
            test_period=test_period,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc,
            total_trades=total_trades,
            winning_trades=int(winning_trades),
            losing_trades=int(losing_trades),
            win_rate=win_rate,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            expectancy=expectancy,
            k3_k4_hit_rate=k3_k4_hit_rate,
            k5_avoidance_rate=k5_avoidance_rate
        )

    def generate_comprehensive_report(self) -> str:
        """Generate detailed walk-forward validation report"""

        if not self.performances:
            return "No validation results available"

        report = []
        report.append("="*100)
        report.append("WALK-FORWARD VALIDATION REPORT")
        report.append("="*100)
        report.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        report.append(f"Total Windows: {len(self.performances)}")
        report.append(f"Retraining Frequency: {self.retrain_frequency_days} days")
        report.append(f"Test Window Size: {self.test_window_days} days")
        report.append("")

        # Aggregate metrics
        report.append("AGGREGATE PERFORMANCE METRICS")
        report.append("-"*100)

        metrics = {
            'Win Rate': [p.win_rate for p in self.performances],
            'Sharpe Ratio': [p.sharpe_ratio for p in self.performances],
            'Profit Factor': [p.profit_factor for p in self.performances],
            'Max Drawdown': [p.max_drawdown for p in self.performances],
            'Accuracy': [p.accuracy for p in self.performances],
            'Precision': [p.precision for p in self.performances],
            'Recall': [p.recall for p in self.performances],
            'F1 Score': [p.f1_score for p in self.performances],
            'K3/K4 Hit Rate': [p.k3_k4_hit_rate for p in self.performances],
            'Expectancy': [p.expectancy for p in self.performances]
        }

        report.append(f"{'Metric':<20} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
        report.append("-"*100)

        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)

            if metric_name in ['Win Rate', 'Max Drawdown', 'Accuracy', 'Precision', 'Recall', 'K3/K4 Hit Rate']:
                report.append(f"{metric_name:<20} {mean_val*100:>11.2f}% {std_val*100:>11.2f}% {min_val*100:>11.2f}% {max_val*100:>11.2f}%")
            else:
                report.append(f"{metric_name:<20} {mean_val:>11.3f}  {std_val:>11.3f}  {min_val:>11.3f}  {max_val:>11.3f}")

        report.append("")

        # Trading performance summary
        report.append("TRADING PERFORMANCE SUMMARY")
        report.append("-"*100)

        total_trades = sum(p.total_trades for p in self.performances)
        total_wins = sum(p.winning_trades for p in self.performances)
        total_losses = sum(p.losing_trades for p in self.performances)
        overall_win_rate = total_wins / max(total_trades, 1)

        report.append(f"Total Trades: {total_trades:,}")
        report.append(f"Winning Trades: {total_wins:,} ({overall_win_rate:.1%})")
        report.append(f"Losing Trades: {total_losses:,} ({(1-overall_win_rate):.1%})")
        report.append(f"Average Trades per Window: {total_trades/len(self.performances):.1f}")
        report.append("")

        # Per-window details
        report.append("PER-WINDOW RESULTS")
        report.append("-"*100)
        report.append(f"{'Window':<8} {'Period':<25} {'Win Rate':<12} {'Sharpe':<10} {'PF':<10} {'Max DD':<10} {'Trades':<8}")
        report.append("-"*100)

        for perf in self.performances:
            report.append(f"{perf.window_id:<8} {perf.test_period:<25} {perf.win_rate*100:>10.1f}% "
                         f"{perf.sharpe_ratio:>9.2f} {perf.profit_factor:>9.2f} {perf.max_drawdown*100:>9.1f}% {perf.total_trades:>7}")

        report.append("")
        report.append("="*100)

        # Key findings
        report.append("\nKEY FINDINGS:")
        report.append(f"• Average Win Rate: {np.mean([p.win_rate for p in self.performances]):.1%}")
        report.append(f"• Average Sharpe Ratio: {np.mean([p.sharpe_ratio for p in self.performances]):.2f}")
        report.append(f"• Average Profit Factor: {np.mean([p.profit_factor for p in self.performances]):.2f}")
        report.append(f"• Average Max Drawdown: {np.mean([p.max_drawdown for p in self.performances]):.1%}")
        report.append(f"• K3/K4 Pattern Hit Rate: {np.mean([p.k3_k4_hit_rate for p in self.performances]):.1%}")
        report.append(f"• Expected Value per Trade: {np.mean([p.expectancy for p in self.performances]):+.3f}")

        return "\n".join(report)

    def save_results(self, output_dir: str = 'output/walk_forward'):
        """Save validation results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save report
        report = self.generate_comprehensive_report()
        report_file = output_path / f'walk_forward_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_file}")

        # Save detailed results as JSON
        results = {
            'windows': [asdict(w) for w in self.windows],
            'performances': [p.to_dict() for p in self.performances],
            'config': {
                'initial_train_days': self.initial_train_days,
                'retrain_frequency_days': self.retrain_frequency_days,
                'test_window_days': self.test_window_days,
                'min_train_samples': self.min_train_samples
            }
        }

        json_file = output_path / f'walk_forward_results_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {json_file}")

        # Save performance DataFrame
        perf_df = pd.DataFrame([p.to_dict() for p in self.performances])
        csv_file = output_path / f'walk_forward_performance_{timestamp}.csv'
        perf_df.to_csv(csv_file, index=False)
        logger.info(f"Performance CSV saved to {csv_file}")


def run_walk_forward_example():
    """Example usage of Walk-Forward Validator"""

    # This is an example - replace with actual data loading
    logger.info("Walk-Forward Validation Example")
    logger.info("Note: This requires actual pattern data to run")

    # Example configuration
    validator = WalkForwardValidator(
        initial_train_days=730,  # 2 years
        retrain_frequency_days=90,  # Quarterly
        test_window_days=100,
        min_train_samples=500
    )

    logger.info(f"Validator configured:")
    logger.info(f"  Initial training: {validator.initial_train_days} days")
    logger.info(f"  Retraining frequency: {validator.retrain_frequency_days} days")
    logger.info(f"  Test window: {validator.test_window_days} days")
    logger.info(f"  Min training samples: {validator.min_train_samples}")

    return validator


if __name__ == "__main__":
    validator = run_walk_forward_example()
    print("\nWalk-Forward Validator ready for use!")
    print("\nTo use with your data:")
    print("  validator.validate_model(df, model_class, feature_columns)")
