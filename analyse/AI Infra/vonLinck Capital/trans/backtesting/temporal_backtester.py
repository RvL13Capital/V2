"""
Temporal Backtester with Full Historical Scan
==============================================

Comprehensive backtesting system that:
1. Scans ENTIRE historical dataset (not just recent patterns)
2. Detects ALL consolidation patterns from start_date to end_date
3. Tracks patterns through full 100-day outcome window
4. Uses walk-forward validation (no look-ahead bias)
5. Calculates comprehensive performance metrics

Key Feature: FULL HISTORICAL COVERAGE
- Does NOT limit to recent patterns only
- Scans every possible consolidation in the date range
- Labels all completed patterns with actual outcomes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from tqdm import tqdm

from core.aiv7_components import ConsolidationTracker, PatternPhase
from core.aiv7_components.data_loader import DataLoader
from core.path_dependent_labeler import PathDependentLabelerV17
from config import (
    MIN_PATTERN_DURATION,
    STRATEGIC_VALUES,
    calculate_expected_value
)
from config.constants import NUM_CLASSES, CLASS_NAMES

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting run"""

    # Date range for FULL historical scan
    start_date: str  # Scan from this date
    end_date: str    # Scan to this date

    # Ticker universe
    tickers: List[str]  # List of tickers to scan

    # Pattern parameters
    min_pattern_duration: int = 10  # Minimum days in qualification
    outcome_window: int = 100  # Days to wait for outcome after pattern completion

    # Walk-forward validation
    train_window_years: int = 2  # Years of training data
    test_window_months: int = 3  # Months of test data
    retrain_frequency_days: int = 90  # Retrain model every N days

    # Output configuration
    output_dir: str = "output/backtest"
    save_all_patterns: bool = True  # Save all detected patterns, not just recent

    # Performance
    max_workers: int = 4  # Parallel processing workers
    verbose: bool = True


@dataclass
class BacktestResults:
    """Results from backtesting run"""

    config: BacktestConfig

    # ALL patterns found across ENTIRE history
    all_patterns: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Completed patterns with outcomes (for model training/evaluation)
    completed_patterns: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Active patterns (still in consolidation)
    active_patterns: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Failed patterns (broke down)
    failed_patterns: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Performance metrics
    metrics: Dict = field(default_factory=dict)

    # Statistics by time period
    temporal_stats: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Execution metadata
    execution_time_seconds: float = 0.0
    total_tickers_scanned: int = 0
    total_patterns_found: int = 0

    def summary(self) -> str:
        """Generate summary report"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║     FULL HISTORICAL BACKTEST RESULTS                         ║
╚══════════════════════════════════════════════════════════════╝

Date Range: {self.config.start_date} → {self.config.end_date}
Execution Time: {self.execution_time_seconds:.1f} seconds

PATTERN DETECTION (FULL HISTORY):
  Total Tickers Scanned: {self.total_tickers_scanned}
  Total Patterns Found: {self.total_patterns_found}

  Completed Patterns: {len(self.completed_patterns)}
  Active Patterns: {len(self.active_patterns)}
  Failed Patterns: {len(self.failed_patterns)}

OUTCOME DISTRIBUTION (Completed Patterns):
{self._outcome_distribution()}

PERFORMANCE METRICS:
{self._format_metrics()}
        """

    def _outcome_distribution(self) -> str:
        """Format outcome distribution for V17 3-class system"""
        if len(self.completed_patterns) == 0:
            return "  No completed patterns"

        outcome_counts = self.completed_patterns['outcome_class'].value_counts().sort_index()
        lines = []

        # V17 distribution (3 classes: Danger, Noise, Target)
        v17_names = {0: "DANGER (Breakdown)", 1: "NOISE (Base case)", 2: "TARGET (Home Run)"}
        for outcome_id in range(NUM_CLASSES):
            count = outcome_counts.get(outcome_id, 0)
            pct = (count / len(self.completed_patterns)) * 100
            lines.append(f"  {v17_names[outcome_id]}: {count} ({pct:.1f}%)")

        return "\n".join(lines)

    def _format_metrics(self) -> str:
        """Format performance metrics"""
        if not self.metrics:
            return "  No metrics available"

        lines = []
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class TemporalBacktester:
    """
    Backtester with FULL historical pattern detection

    Key Feature: Scans entire historical dataset, not just recent patterns.
    Ensures comprehensive pattern coverage across all time periods.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_loader = DataLoader()

        # Storage for ALL patterns found (historical + recent)
        self.all_patterns: List[Dict] = []

        # Initialize V17 labeler (3-class: Danger, Noise, Target)
        self.labeler_v17 = PathDependentLabelerV17()
        logger.info("Using Path-Dependent Labeling V17 (3-class)")

        logger.info("Initialized TemporalBacktester")
        logger.info(f"Full historical scan: {config.start_date} → {config.end_date}")

    def run(self) -> BacktestResults:
        """
        Run full historical backtest

        Steps:
        1. Load historical data for all tickers
        2. Scan ENTIRE history for consolidation patterns
        3. Label completed patterns with actual outcomes
        4. Calculate comprehensive performance metrics
        5. Generate temporal statistics
        """
        start_time = datetime.now()

        logger.info("=" * 70)
        logger.info("STARTING FULL HISTORICAL BACKTEST")
        logger.info("=" * 70)

        # Step 1: Scan all tickers for patterns (FULL HISTORY)
        logger.info(f"Step 1: Scanning {len(self.config.tickers)} tickers...")
        all_ticker_patterns = self._scan_all_tickers()

        # Step 2: Convert to DataFrame
        logger.info(f"Step 2: Processing {len(all_ticker_patterns)} detected patterns...")
        patterns_df = pd.DataFrame(all_ticker_patterns)

        if len(patterns_df) == 0:
            logger.warning("No patterns found in historical data!")
            return BacktestResults(
                config=self.config,
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                total_tickers_scanned=len(self.config.tickers),
                total_patterns_found=0
            )

        # Step 3: Separate patterns by status
        logger.info("Step 3: Categorizing patterns...")
        completed_patterns, active_patterns, failed_patterns = self._categorize_patterns(patterns_df)

        # Step 4: Label completed patterns with outcomes
        logger.info(f"Step 4: Labeling {len(completed_patterns)} completed patterns with outcomes...")
        labeled_patterns = self._label_outcomes(completed_patterns)

        # Step 5: Calculate performance metrics
        logger.info("Step 5: Calculating performance metrics...")
        metrics = self._calculate_metrics(labeled_patterns)

        # Step 6: Generate temporal statistics
        logger.info("Step 6: Generating temporal statistics...")
        temporal_stats = self._calculate_temporal_stats(patterns_df)

        execution_time = (datetime.now() - start_time).total_seconds()

        results = BacktestResults(
            config=self.config,
            all_patterns=patterns_df,
            completed_patterns=labeled_patterns,
            active_patterns=active_patterns,
            failed_patterns=failed_patterns,
            metrics=metrics,
            temporal_stats=temporal_stats,
            execution_time_seconds=execution_time,
            total_tickers_scanned=len(self.config.tickers),
            total_patterns_found=len(patterns_df)
        )

        logger.info("=" * 70)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 70)
        logger.info(results.summary())

        return results

    def _scan_all_tickers(self) -> List[Dict]:
        """
        Scan all tickers for consolidation patterns across FULL history

        Critical: This scans ENTIRE historical range, not just recent data
        """
        all_patterns = []

        iterator = tqdm(self.config.tickers, desc="Scanning tickers") if self.config.verbose else self.config.tickers

        for ticker in iterator:
            try:
                # Load FULL historical data for this ticker
                ticker_data = self._load_ticker_data(ticker)

                if ticker_data is None or len(ticker_data) < 100:
                    logger.warning(f"Insufficient data for {ticker}, skipping")
                    continue

                # Scan ENTIRE history for patterns
                ticker_patterns = self._scan_ticker_history(ticker, ticker_data)
                all_patterns.extend(ticker_patterns)

                if self.config.verbose and len(ticker_patterns) > 0:
                    logger.info(f"{ticker}: Found {len(ticker_patterns)} patterns across full history")

            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue

        return all_patterns

    def _load_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load full historical data for ticker"""
        try:
            # Load data from start_date to end_date + outcome_window
            # (need extra data to calculate outcomes)
            extended_end = pd.to_datetime(self.config.end_date) + timedelta(days=self.config.outcome_window)

            df = self.data_loader.load_ticker_data(
                ticker=ticker,
                start_date=self.config.start_date,
                end_date=extended_end.strftime('%Y-%m-%d')
            )

            return df

        except Exception as e:
            logger.error(f"Failed to load data for {ticker}: {e}")
            return None

    def _scan_ticker_history(self, ticker: str, data: pd.DataFrame) -> List[Dict]:
        """
        Scan single ticker across ENTIRE historical range

        Uses ConsolidationTracker to detect all patterns from start to end
        """
        tracker = ConsolidationTracker(ticker=ticker)
        patterns_found = []

        # Iterate through ENTIRE dataset day by day
        # This ensures we find ALL patterns, not just recent ones
        for i in range(len(data)):
            current_date = data.index[i]

            # Only start tracking after start_date
            if current_date < pd.to_datetime(self.config.start_date):
                continue

            # Stop tracking at end_date
            if current_date > pd.to_datetime(self.config.end_date):
                break

            # Get data up to current date (temporal integrity)
            historical_data = data.iloc[:i+1].copy()

            # Update tracker with new day of data
            tracker.update(historical_data)

            # Check if pattern completed or failed
            if tracker.state.phase == PatternPhase.COMPLETED:
                pattern_info = self._extract_pattern_info(tracker, ticker, data, i)
                patterns_found.append(pattern_info)

                # Reset tracker to look for next pattern
                tracker = ConsolidationTracker(ticker=ticker)

            elif tracker.state.phase == PatternPhase.FAILED:
                pattern_info = self._extract_pattern_info(tracker, ticker, data, i)
                patterns_found.append(pattern_info)

                # Reset tracker
                tracker = ConsolidationTracker(ticker=ticker)

        # Capture active pattern if still qualifying/active at end
        if tracker.state.phase in [PatternPhase.QUALIFYING, PatternPhase.ACTIVE]:
            pattern_info = self._extract_pattern_info(tracker, ticker, data, len(data)-1)
            patterns_found.append(pattern_info)

        return patterns_found

    def _extract_pattern_info(self, tracker: ConsolidationTracker, ticker: str,
                              data: pd.DataFrame, current_idx: int) -> Dict:
        """Extract pattern information from tracker"""
        state = tracker.state

        pattern_info = {
            'ticker': ticker,
            'phase': state.phase.value,
            'start_date': state.start_date,
            'end_date': data.index[current_idx],
            'days_in_pattern': state.days_in_pattern,
            'days_qualifying': state.days_qualifying,
            'days_active': state.days_active,
            'upper_boundary': state.upper_boundary,
            'lower_boundary': state.lower_boundary,
            'start_price': state.start_price,
            'current_price': data.iloc[current_idx]['close'],
            'avg_bbw': np.mean(state.bbw_history) if state.bbw_history else None,
            'avg_adx': np.mean(state.adx_history) if state.adx_history else None,
            'avg_volume_ratio': np.mean(state.volume_ratio_history) if state.volume_ratio_history else None,
        }

        return pattern_info

    def _categorize_patterns(self, patterns_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Separate patterns by completion status"""
        completed = patterns_df[patterns_df['phase'] == 'COMPLETED'].copy()
        active = patterns_df[patterns_df['phase'].isin(['QUALIFYING', 'ACTIVE'])].copy()
        failed = patterns_df[patterns_df['phase'] == 'FAILED'].copy()

        return completed, active, failed

    def _label_outcomes(self, completed_patterns: pd.DataFrame) -> pd.DataFrame:
        """
        Label completed patterns with V17 3-class outcome labels.

        V17 Classes:
          - 0: Danger (stop loss hit)
          - 1: Noise (neither target nor stop)
          - 2: Target (profitable trade)
        """
        if len(completed_patterns) == 0:
            return completed_patterns

        labeled = completed_patterns.copy()
        labeled['outcome_class'] = -1
        labeled['max_gain_pct'] = 0.0
        labeled['outcome_date'] = None
        labeled['strategic_value'] = 0.0
        labeled['risk_unit'] = 0.0
        labeled['labeling_version'] = 'v17'

        for idx, row in tqdm(labeled.iterrows(), total=len(labeled), desc="Labeling outcomes"):
            try:
                # V17 Path-Dependent Labeling
                label = self._label_pattern_v17(row)

                # Skip grey zone patterns
                if label is None or label == -1:
                    continue

                labeled.at[idx, 'outcome_class'] = label
                labeled.at[idx, 'strategic_value'] = STRATEGIC_VALUES.get(label, 0)

            except Exception as e:
                logger.warning(f"Failed to label outcome for {row['ticker']} at {row['end_date']}: {e}")
                continue

        # Remove patterns without valid outcomes (exclude grey zone = -1)
        labeled = labeled[(labeled['outcome_class'] >= 0) & (labeled['outcome_class'] != -1)].copy()

        return labeled

    def _label_pattern_v17(self, pattern_row: pd.Series) -> Optional[int]:
        """
        Label a pattern using v17 path-dependent logic

        Args:
            pattern_row: Row from patterns DataFrame

        Returns:
            Label (0, 1, 2) or -1 for grey zone, None for invalid
        """
        try:
            # Load full historical data for the pattern
            ticker = pattern_row['ticker']
            pattern_end = pd.to_datetime(pattern_row['end_date'])

            # Need data from before pattern for warmup + outcome window
            data_start = pattern_end - timedelta(days=200)  # Extra buffer
            data_end = pattern_end + timedelta(days=self.config.outcome_window + 10)

            full_data = self.data_loader.load_ticker_data(
                ticker=ticker,
                start_date=data_start.strftime('%Y-%m-%d'),
                end_date=data_end.strftime('%Y-%m-%d')
            )

            if full_data is None or len(full_data) < 230:  # Need min for warmup + outcome
                return None

            # Find pattern end index in the data
            pattern_end_idx = None
            for i, date in enumerate(full_data.index):
                if date.date() == pattern_end.date():
                    pattern_end_idx = i
                    break

            if pattern_end_idx is None:
                return None

            # Get pattern boundaries
            pattern_boundaries = {
                'upper': pattern_row.get('upper_boundary'),
                'lower': pattern_row.get('lower_boundary')
            }

            # Use v17 labeler
            label = self.labeler_v17.label_pattern(
                full_data=full_data,
                pattern_end_idx=pattern_end_idx,
                pattern_boundaries=pattern_boundaries
            )

            return label

        except Exception as e:
            logger.warning(f"V17 labeling failed for {ticker}: {e}")
            return None

    def _calculate_metrics(self, labeled_patterns: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics for V17 3-class system."""
        if len(labeled_patterns) == 0:
            return {}

        metrics = {}

        # V17 metrics (3-class system: Danger, Noise, Target)
        outcome_counts = labeled_patterns['outcome_class'].value_counts()

        # Class distribution
        for outcome_id in range(NUM_CLASSES):
            count = outcome_counts.get(outcome_id, 0)
            pct = (count / len(labeled_patterns)) * 100
            class_name = CLASS_NAMES[outcome_id]
            metrics[f'outcome_{class_name}_count'] = int(count)
            metrics[f'outcome_{class_name}_pct'] = pct

        # Success metrics
        metrics['target_rate'] = (labeled_patterns['outcome_class'] == 2).mean() * 100
        metrics['danger_rate'] = (labeled_patterns['outcome_class'] == 0).mean() * 100
        metrics['noise_rate'] = (labeled_patterns['outcome_class'] == 1).mean() * 100

        # Risk/reward ratio (Target to Danger)
        target_count = (labeled_patterns['outcome_class'] == 2).sum()
        danger_count = (labeled_patterns['outcome_class'] == 0).sum()
        if danger_count > 0:
            metrics['target_to_danger_ratio'] = target_count / danger_count
        else:
            metrics['target_to_danger_ratio'] = float('inf') if target_count > 0 else 0

        # Common metrics
        metrics['avg_strategic_value'] = labeled_patterns['strategic_value'].mean()
        metrics['total_strategic_value'] = labeled_patterns['strategic_value'].sum()

        # Pattern duration metrics (if available)
        if 'days_in_pattern' in labeled_patterns.columns:
            metrics['avg_days_in_pattern'] = labeled_patterns['days_in_pattern'].mean()
        if 'days_qualifying' in labeled_patterns.columns:
            metrics['avg_days_qualifying'] = labeled_patterns['days_qualifying'].mean()

        return metrics

    def _calculate_temporal_stats(self, patterns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistics by time period (monthly/quarterly)"""
        if len(patterns_df) == 0:
            return pd.DataFrame()

        patterns_df = patterns_df.copy()
        patterns_df['start_date'] = pd.to_datetime(patterns_df['start_date'])
        patterns_df['year_month'] = patterns_df['start_date'].dt.to_period('M')

        # Group by month
        monthly_stats = patterns_df.groupby('year_month').agg({
            'ticker': 'count',
            'days_in_pattern': 'mean',
            'avg_bbw': 'mean',
            'avg_adx': 'mean'
        }).rename(columns={'ticker': 'pattern_count'})

        return monthly_stats

    def save_results(self, results: BacktestResults, output_dir: Optional[str] = None):
        """Save backtest results to disk"""
        if output_dir is None:
            output_dir = self.config.output_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save all patterns (FULL HISTORY)
        if self.config.save_all_patterns:
            all_patterns_path = output_path / "all_patterns_full_history.parquet"
            results.all_patterns.to_parquet(all_patterns_path)
            logger.info(f"Saved all patterns to {all_patterns_path}")

        # Save completed patterns with outcomes
        completed_path = output_path / "completed_patterns_labeled.parquet"
        results.completed_patterns.to_parquet(completed_path)
        logger.info(f"Saved completed patterns to {completed_path}")

        # Save metrics
        metrics_path = output_path / "backtest_metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            json.dump(results.metrics, f, indent=2, default=str)
        logger.info(f"Saved metrics to {metrics_path}")

        # Save summary report
        summary_path = output_path / "backtest_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(results.summary())
        logger.info(f"Saved summary to {summary_path}")
