"""
Stateful Detector - Manages Multiple Consolidation Trackers

Orchestrates pattern detection across multiple tickers while ensuring
temporal integrity. Processes data day-by-day, never using future information.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

from .consolidation_tracker import ConsolidationTracker, ConsolidationPattern, PatternPhase
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from shared.indicators.technical import calculate_all_indicators

logger = logging.getLogger(__name__)


class StatefulDetector:
    """
    Stateful pattern detector managing multiple tickers.

    Key Features:
    - Maintains separate ConsolidationTracker for each ticker
    - Processes data chronologically (day-by-day)
    - Ensures no look-ahead bias
    - Tracks active and completed patterns
    """

    def __init__(
        self,
        bbw_percentile_threshold: float = 0.30,
        adx_threshold: float = 32.0,
        volume_ratio_threshold: float = 0.35,
        range_ratio_threshold: float = 0.65,
        qualifying_days: int = 10,
        max_pattern_days: int = 90
    ):
        """
        Initialize stateful detector.

        Args:
            bbw_percentile_threshold: BBW percentile threshold
            adx_threshold: ADX threshold
            volume_ratio_threshold: Volume ratio threshold
            range_ratio_threshold: Range ratio threshold
            qualifying_days: Days to qualify pattern
            max_pattern_days: Max pattern duration
        """
        self.bbw_percentile_threshold = bbw_percentile_threshold
        self.adx_threshold = adx_threshold
        self.volume_ratio_threshold = volume_ratio_threshold
        self.range_ratio_threshold = range_ratio_threshold
        self.qualifying_days = qualifying_days
        self.max_pattern_days = max_pattern_days

        self.trackers: Dict[str, ConsolidationTracker] = {}
        self.all_patterns: List[ConsolidationPattern] = []

    def detect_patterns(
        self,
        ticker: str,
        df: pd.DataFrame
    ) -> List[ConsolidationPattern]:
        """
        Detect consolidation patterns for a ticker.

        CRITICAL: Processes data chronologically to maintain temporal integrity.

        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data (must be indexed by date)

        Returns:
            List of detected ConsolidationPattern objects
        """
        logger.info(f"Detecting patterns for {ticker} ({len(df)} bars)")

        # Calculate all indicators
        df_with_indicators = calculate_all_indicators(df)

        # Initialize tracker for this ticker
        if ticker not in self.trackers:
            self.trackers[ticker] = ConsolidationTracker(
                ticker=ticker,
                bbw_percentile_threshold=self.bbw_percentile_threshold,
                adx_threshold=self.adx_threshold,
                volume_ratio_threshold=self.volume_ratio_threshold,
                range_ratio_threshold=self.range_ratio_threshold,
                qualifying_days=self.qualifying_days,
                max_pattern_days=self.max_pattern_days
            )

        tracker = self.trackers[ticker]

        # Set full DataFrame for dynamic feature extraction
        tracker.set_data(df_with_indicators)

        # Process day-by-day (temporal order maintained)
        for idx, (date, row) in enumerate(df_with_indicators.iterrows()):
            # Extract price data
            price_data = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }

            # Extract indicators
            indicators = {
                'bbw_20': row.get('bbw_20', np.nan),
                'bbw_percentile': row.get('bbw_percentile', np.nan),
                'adx': row.get('adx', np.nan),
                'volume_ratio_20': row.get('volume_ratio_20', np.nan),
                'rsi_14': row.get('rsi_14', np.nan),
                'atr_14': row.get('atr_14', np.nan)
            }

            # Calculate range ratio
            if 'daily_range' in row and 'daily_range_avg' in row:
                indicators['range_ratio'] = row['daily_range'] / (row['daily_range_avg'] + 1e-10)
            else:
                indicators['range_ratio'] = 1.0

            # Update tracker
            event = tracker.update(date, idx, price_data, indicators)

            if event:
                logger.debug(f"{ticker} @ {date}: {event}")

        # Get completed patterns
        patterns = tracker.get_completed_patterns()
        self.all_patterns.extend(patterns)

        logger.info(f"Found {len(patterns)} completed patterns for {ticker}")

        return patterns

    def detect_patterns_multiple(
        self,
        ticker_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[ConsolidationPattern]]:
        """
        Detect patterns for multiple tickers.

        Args:
            ticker_data: Dictionary mapping ticker -> DataFrame

        Returns:
            Dictionary mapping ticker -> list of patterns
        """
        results = {}

        for ticker, df in ticker_data.items():
            try:
                patterns = self.detect_patterns(ticker, df)
                results[ticker] = patterns
            except Exception as e:
                logger.error(f"Failed to detect patterns for {ticker}: {e}")
                results[ticker] = []

        total_patterns = sum(len(p) for p in results.values())
        logger.info(f"Total patterns detected: {total_patterns} across {len(ticker_data)} tickers")

        return results

    def get_active_patterns(self) -> List[ConsolidationPattern]:
        """
        Get all currently active patterns.

        Returns:
            List of active ConsolidationPattern objects
        """
        active_patterns = []

        for ticker, tracker in self.trackers.items():
            current_pattern = tracker.get_current_pattern()
            if current_pattern and tracker.phase == PatternPhase.ACTIVE:
                active_patterns.append(current_pattern)

        return active_patterns

    def get_qualifying_patterns(self) -> List[ConsolidationPattern]:
        """
        Get all patterns currently in qualification phase.

        Returns:
            List of qualifying ConsolidationPattern objects
        """
        qualifying_patterns = []

        for ticker, tracker in self.trackers.items():
            current_pattern = tracker.get_current_pattern()
            if current_pattern and tracker.phase == PatternPhase.QUALIFYING:
                qualifying_patterns.append(current_pattern)

        return qualifying_patterns

    def get_all_completed_patterns(self) -> List[ConsolidationPattern]:
        """
        Get all completed patterns from all tickers.

        Returns:
            List of all completed patterns
        """
        return self.all_patterns

    def patterns_to_dataframe(
        self,
        patterns: Optional[List[ConsolidationPattern]] = None,
        expand_snapshots: bool = True
    ) -> pd.DataFrame:
        """
        Convert patterns to DataFrame with optional snapshot expansion.

        Args:
            patterns: List of patterns (uses all completed if None)
            expand_snapshots: If True, expands patterns into multiple rows
                            (one per feature snapshot). If False, uses single
                            row per pattern (legacy mode).

        Returns:
            DataFrame with pattern data

        Note:
            With expand_snapshots=True (default):
            - 500 patterns → 3,000+ rows (multiple samples per pattern)
            - Better training data (no position bias, proportional sampling)

            With expand_snapshots=False:
            - 500 patterns → 500 rows (one per pattern)
            - Legacy mode (not recommended for training)
        """
        if patterns is None:
            patterns = self.all_patterns

        if not patterns:
            return pd.DataFrame()

        records = []

        if expand_snapshots:
            # NEW: Expand patterns into multiple training samples
            # Each pattern contributes ~6 samples (proportional to duration)
            for pattern in patterns:
                samples = pattern.to_training_samples()  # Returns list of dicts
                records.extend(samples)  # Flatten into single list

            logger.info(f"Expanded {len(patterns)} patterns into {len(records)} training samples")
            logger.info(f"Average {len(records)/len(patterns):.1f} samples per pattern")
        else:
            # OLD: Single row per pattern (legacy mode)
            records = [p.to_dict() for p in patterns]
            logger.info(f"Created {len(records)} rows (1 per pattern, no expansion)")

        df = pd.DataFrame(records)

        return df

    def get_statistics(self) -> Dict:
        """
        Get detection statistics.

        Returns:
            Dictionary with statistics
        """
        total_patterns = len(self.all_patterns)
        active_patterns = len(self.get_active_patterns())
        qualifying_patterns = len(self.get_qualifying_patterns())

        # Count by outcome
        breakouts = sum(1 for p in self.all_patterns if p.breakout_direction == 'UP')
        breakdowns = sum(1 for p in self.all_patterns if p.breakout_direction == 'DOWN')
        timeouts = sum(1 for p in self.all_patterns if p.breakout_direction == 'TIMEOUT')

        # Average pattern duration
        avg_duration = np.mean([p.days_in_pattern for p in self.all_patterns]) if self.all_patterns else 0

        # Average gain for breakouts
        breakout_patterns = [p for p in self.all_patterns if p.breakout_direction == 'UP']
        avg_breakout_gain = np.mean([p.max_gain for p in breakout_patterns]) if breakout_patterns else 0

        stats = {
            'total_tickers_tracked': len(self.trackers),
            'total_completed_patterns': total_patterns,
            'active_patterns': active_patterns,
            'qualifying_patterns': qualifying_patterns,
            'breakouts': breakouts,
            'breakdowns': breakdowns,
            'timeouts': timeouts,
            'breakout_rate': breakouts / total_patterns if total_patterns > 0 else 0,
            'avg_pattern_duration_days': avg_duration,
            'avg_breakout_gain': avg_breakout_gain
        }

        return stats

    def reset(self) -> None:
        """Reset detector state."""
        self.trackers = {}
        self.all_patterns = []
        logger.info("Detector state reset")
