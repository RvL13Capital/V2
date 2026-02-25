"""
Protocols for Pattern Detection - Dependency Injection Interfaces

Defines abstract interfaces for pattern detection components to enable:
- Loose coupling between components
- Easy testing with mocks
- Flexibility to swap implementations
- Clear contracts for behavior
"""

from __future__ import annotations

from typing import Protocol, List, Optional, Dict
from datetime import datetime
import pandas as pd

from pattern_detection.models import ConsolidationPattern, PatternPhase


class PatternTracker(Protocol):
    """
    Protocol for tracking a single ticker's consolidation pattern state.

    Implementations should maintain state machine logic for pattern lifecycle:
    NONE → QUALIFYING → ACTIVE → COMPLETED/FAILED → NONE (reset)
    """

    @property
    def ticker(self) -> str:
        """Ticker symbol being tracked."""
        ...

    @property
    def current_pattern(self) -> Optional[ConsolidationPattern]:
        """Currently active pattern, if any."""
        ...

    @property
    def completed_patterns(self) -> List[ConsolidationPattern]:
        """List of completed/failed patterns for this ticker."""
        ...

    def process_day(
        self,
        date: datetime,
        idx: int,
        row: pd.Series,
        indicators: Dict,
    ) -> Optional[ConsolidationPattern]:
        """
        Process a single day of data for pattern detection.

        Args:
            date: Date being processed
            idx: Index in the dataframe
            row: Price data for the day (OHLCV)
            indicators: Pre-calculated technical indicators (BBW, ADX, etc.)

        Returns:
            ConsolidationPattern if state changed (activation, completion), else None
        """
        ...

    def reset(self) -> None:
        """Reset tracker to initial state (for new pattern detection)."""
        ...


class PatternDetector(Protocol):
    """
    Protocol for detecting patterns across multiple tickers.

    Implementations should manage multiple PatternTracker instances and
    coordinate pattern detection across a portfolio of tickers.
    """

    def scan_ticker(
        self,
        ticker: str,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[ConsolidationPattern]:
        """
        Scan a single ticker for consolidation patterns.

        Args:
            ticker: Ticker symbol to scan
            data: OHLCV price data with technical indicators
            start_date: Optional start date for scanning
            end_date: Optional end date for scanning

        Returns:
            List of detected patterns (can be empty)
        """
        ...

    def scan_multiple(
        self,
        tickers: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, List[ConsolidationPattern]]:
        """
        Scan multiple tickers for patterns.

        Args:
            tickers: List of ticker symbols
            data_dict: Dictionary mapping ticker → OHLCV dataframe
            start_date: Optional start date for scanning
            end_date: Optional end date for scanning

        Returns:
            Dictionary mapping ticker → list of patterns
        """
        ...

    def get_active_patterns(self) -> List[ConsolidationPattern]:
        """
        Get all currently active patterns across all tickers.

        Returns:
            List of patterns in ACTIVE phase
        """
        ...

    def get_completed_patterns(self) -> List[ConsolidationPattern]:
        """
        Get all completed/failed patterns across all tickers.

        Returns:
            List of patterns in COMPLETED or FAILED phase
        """
        ...


class IndicatorCalculator(Protocol):
    """
    Protocol for calculating technical indicators.

    Implementations should provide efficient indicator calculation
    for pattern detection (BBW, ADX, volume ratios, etc.).
    """

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        lookback_period: int = 20,
    ) -> pd.DataFrame:
        """
        Calculate all required technical indicators.

        Args:
            data: OHLCV price data
            lookback_period: Lookback period for indicators (default: 20)

        Returns:
            DataFrame with added indicator columns
        """
        ...

    def calculate_bbw(
        self,
        data: pd.DataFrame,
        period: int = 20,
        std_dev: int = 2,
    ) -> pd.Series:
        """
        Calculate Bollinger Band Width (BBW).

        Args:
            data: Price data with 'close' column
            period: Moving average period
            std_dev: Number of standard deviations

        Returns:
            Series of BBW values
        """
        ...

    def calculate_adx(
        self,
        data: pd.DataFrame,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).

        Args:
            data: OHLCV price data
            period: ADX period

        Returns:
            Series of ADX values
        """
        ...


class PatternLabeler(Protocol):
    """
    Protocol for labeling patterns with outcomes.

    Implementations should assign labels (K0-K5) based on
    pattern outcomes after the lookforward period.
    """

    def label_patterns(
        self,
        patterns: List[ConsolidationPattern],
        price_data: Dict[str, pd.DataFrame],
        lookforward_days: int = 100,
    ) -> List[ConsolidationPattern]:
        """
        Label patterns with outcomes based on future price action.

        Args:
            patterns: List of completed patterns to label
            price_data: Dictionary of ticker → price dataframes
            lookforward_days: Days to look forward for outcome

        Returns:
            Patterns with outcome labels (K0-K5) assigned
        """
        ...

    def calculate_max_gain(
        self,
        pattern: ConsolidationPattern,
        price_data: pd.DataFrame,
        lookforward_days: int = 100,
    ) -> float:
        """
        Calculate maximum gain achieved after pattern breakout.

        Args:
            pattern: Pattern to evaluate
            price_data: Price data for the ticker
            lookforward_days: Days to look forward

        Returns:
            Maximum percentage gain (e.g., 0.45 for 45% gain)
        """
        ...
