"""
Pattern Detection Service - Consolidation Pattern Scanning

Provides high-level pattern detection orchestration:
- StatefulDetector initialization with config settings
- Pattern scanning across multiple tickers
- Results formatting and statistics
- Integration with DataService for data loading
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

from config import SystemConfig
from core import StatefulDetector, ConsolidationPattern

logger = logging.getLogger(__name__)


class PatternDetectionService:
    """
    High-level service for pattern detection operations.

    Consolidates pattern detection logic from main.py and scan_existing_data.py.

    Usage:
        config = SystemConfig()
        service = PatternDetectionService(config)

        # Scan patterns
        results = service.scan_patterns(ticker_data)

        # Get formatted results
        patterns_df = service.get_patterns_dataframe()
        stats = service.get_statistics()
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize pattern detection service.

        Args:
            config: SystemConfig instance (creates default if None)
        """
        self.config = config or SystemConfig()

        # Initialize StatefulDetector with config settings
        self.detector = StatefulDetector(
            bbw_percentile_threshold=self.config.consolidation.bbw_percentile_threshold,
            adx_threshold=self.config.consolidation.adx_threshold,
            volume_ratio_threshold=self.config.consolidation.volume_ratio_threshold,
            range_ratio_threshold=self.config.consolidation.range_ratio_threshold,
            qualifying_days=self.config.consolidation.qualifying_days,
            max_pattern_days=self.config.consolidation.max_pattern_days
        )

        logger.info("PatternDetectionService initialized")
        logger.info(f"Consolidation criteria: BBW<{self.config.consolidation.bbw_percentile_threshold:.0%}, "
                   f"ADX<{self.config.consolidation.adx_threshold}, "
                   f"Vol<{self.config.consolidation.volume_ratio_threshold:.0%}, "
                   f"Range<{self.config.consolidation.range_ratio_threshold:.0%}")
        logger.info(f"Qualification: {self.config.consolidation.qualifying_days} days, "
                   f"Max duration: {self.config.consolidation.max_pattern_days} days")

    def scan_patterns(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, List[ConsolidationPattern]]:
        """
        Scan for consolidation patterns across multiple tickers.

        Args:
            ticker_data: Dictionary mapping ticker -> DataFrame with OHLCV data

        Returns:
            Dictionary mapping ticker -> list of detected patterns
        """
        if not ticker_data:
            logger.warning("No ticker data provided for pattern scanning")
            return {}

        logger.info(f"Running pattern detection on {len(ticker_data)} tickers...")

        # Detect patterns using StatefulDetector
        pattern_results = self.detector.detect_patterns_multiple(ticker_data)

        # Log summary
        total_patterns = sum(len(patterns) for patterns in pattern_results.values())
        tickers_with_patterns = sum(1 for patterns in pattern_results.values() if patterns)

        logger.info(f"Pattern detection complete: {total_patterns} patterns found across "
                   f"{tickers_with_patterns} tickers")

        return pattern_results

    def get_patterns_dataframe(self) -> pd.DataFrame:
        """
        Get all detected patterns as a DataFrame.

        Returns:
            DataFrame with pattern details (ticker, dates, metrics, outcomes)
        """
        return self.detector.patterns_to_dataframe()

    def get_statistics(self) -> Dict:
        """
        Get pattern detection statistics.

        Returns:
            Dictionary with statistics:
            - total_tickers_tracked: Number of tickers scanned
            - total_completed_patterns: Total patterns found
            - active_patterns: Currently active patterns
            - qualifying_patterns: Patterns in qualification phase
            - breakouts: Patterns that broke out
            - breakdowns: Patterns that broke down
            - timeouts: Patterns that timed out
            - breakout_rate: Percentage of breakouts
            - avg_pattern_duration_days: Average pattern duration
            - avg_breakout_gain: Average gain on breakouts
        """
        return self.detector.get_statistics()

    def filter_patterns(
        self,
        patterns_df: Optional[pd.DataFrame] = None,
        phase: Optional[str] = None,
        min_days: Optional[int] = None,
        max_days: Optional[int] = None,
        breakout_direction: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter patterns based on criteria.

        Args:
            patterns_df: DataFrame to filter (uses internal if None)
            phase: Filter by phase ('ACTIVE', 'COMPLETED', 'FAILED', 'QUALIFYING')
            min_days: Minimum days in pattern
            max_days: Maximum days in pattern
            breakout_direction: Filter by breakout direction ('UP', 'DOWN')

        Returns:
            Filtered DataFrame
        """
        if patterns_df is None:
            patterns_df = self.get_patterns_dataframe()

        if patterns_df.empty:
            return patterns_df

        # Apply filters
        filtered = patterns_df.copy()

        if phase:
            filtered = filtered[filtered['phase'] == phase]

        if min_days is not None:
            filtered = filtered[filtered['days_in_pattern'] >= min_days]

        if max_days is not None:
            filtered = filtered[filtered['days_in_pattern'] <= max_days]

        if breakout_direction:
            filtered = filtered[filtered['breakout_direction'] == breakout_direction]

        logger.info(f"Filtered patterns: {len(patterns_df)} -> {len(filtered)}")

        return filtered

    def get_active_patterns(self) -> pd.DataFrame:
        """
        Get currently active patterns (in consolidation, not yet broken out).

        Returns:
            DataFrame with active patterns
        """
        patterns_df = self.get_patterns_dataframe()

        if patterns_df.empty:
            return patterns_df

        active = patterns_df[patterns_df['phase'] == 'ACTIVE'].copy()

        logger.info(f"Found {len(active)} active patterns")

        return active

    def get_completed_patterns(self, breakout_only: bool = False) -> pd.DataFrame:
        """
        Get completed patterns (broke out or broke down).

        Args:
            breakout_only: If True, return only successful breakouts (not breakdowns)

        Returns:
            DataFrame with completed patterns
        """
        patterns_df = self.get_patterns_dataframe()

        if patterns_df.empty:
            return patterns_df

        if breakout_only:
            completed = patterns_df[
                (patterns_df['phase'] == 'COMPLETED') &
                (patterns_df['breakout_direction'] == 'UP')
            ].copy()
            logger.info(f"Found {len(completed)} successful breakout patterns")
        else:
            completed = patterns_df[
                patterns_df['phase'].isin(['COMPLETED', 'FAILED'])
            ].copy()
            logger.info(f"Found {len(completed)} completed patterns (breakouts + breakdowns)")

        return completed

    def format_summary_report(self) -> str:
        """
        Format a text summary report of pattern detection results.

        Returns:
            Formatted text report
        """
        stats = self.get_statistics()
        patterns_df = self.get_patterns_dataframe()

        report = []
        report.append("=" * 70)
        report.append("PATTERN DETECTION SUMMARY")
        report.append("=" * 70)

        # Overall statistics
        report.append(f"\nTickers Scanned:        {stats['total_tickers_tracked']}")
        report.append(f"Total Patterns Found:   {stats['total_completed_patterns']}")
        report.append(f"Active Patterns:        {stats['active_patterns']}")
        report.append(f"Qualifying Patterns:    {stats['qualifying_patterns']}")

        # Outcomes
        report.append(f"\nPattern Outcomes:")
        report.append(f"  Breakouts:            {stats['breakouts']}")
        report.append(f"  Breakdowns:           {stats['breakdowns']}")
        report.append(f"  Timeouts:             {stats['timeouts']}")
        report.append(f"  Breakout Rate:        {stats['breakout_rate']:.1%}")

        # Metrics
        report.append(f"\nPattern Metrics:")
        report.append(f"  Avg Duration:         {stats['avg_pattern_duration_days']:.1f} days")
        report.append(f"  Avg Breakout Gain:    {stats['avg_breakout_gain']:.1%}")

        # Top patterns by gain
        if not patterns_df.empty and 'max_gain' in patterns_df.columns:
            breakouts = patterns_df[
                (patterns_df['breakout_direction'] == 'UP') &
                (patterns_df['max_gain'].notna())
            ].copy()

            if not breakouts.empty:
                top_5 = breakouts.nlargest(5, 'max_gain')
                report.append(f"\nTop 5 Breakouts:")
                for idx, row in top_5.iterrows():
                    report.append(f"  {row['ticker']}: {row['max_gain']:.1%} "
                                f"({row['days_in_pattern']:.0f} days, "
                                f"started {row['start_date'].strftime('%Y-%m-%d')})")

        report.append("=" * 70)

        return "\n".join(report)

    def save_patterns(self, output_file: str, include_active: bool = True) -> None:
        """
        Save detected patterns to CSV file.

        Args:
            output_file: Output CSV file path
            include_active: Include active patterns (not yet completed)
        """
        patterns_df = self.get_patterns_dataframe()

        if patterns_df.empty:
            logger.warning("No patterns to save")
            return

        if not include_active:
            patterns_df = patterns_df[patterns_df['phase'].isin(['COMPLETED', 'FAILED'])]
            logger.info(f"Saving {len(patterns_df)} completed patterns (excluding active)")

        patterns_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(patterns_df)} patterns to {output_file}")

    def get_pattern_by_ticker_date(
        self,
        ticker: str,
        start_date: datetime
    ) -> Optional[ConsolidationPattern]:
        """
        Get a specific pattern by ticker and start date.

        Args:
            ticker: Ticker symbol
            start_date: Pattern start date

        Returns:
            ConsolidationPattern or None if not found
        """
        patterns_df = self.get_patterns_dataframe()

        if patterns_df.empty:
            return None

        matches = patterns_df[
            (patterns_df['ticker'] == ticker) &
            (patterns_df['start_date'] == start_date)
        ]

        if matches.empty:
            return None

        # Return the first match (should be unique by ticker + start_date)
        return matches.iloc[0].to_dict()
