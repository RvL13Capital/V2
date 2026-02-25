"""
Survivorship Bias Detection and Correction Module
===================================================

Handles detection and labeling of patterns affected by survivorship bias:
1. Delisted stocks - patterns near delisting events
2. IPO filtering - patterns too close to listing date
3. Trading halts - extended gaps indicating regulatory action
4. Survivorship status labeling - tracking for validation splits

Usage:
    from utils.survivorship_checker import SurvivorshipChecker

    checker = SurvivorshipChecker(delisted_tickers_file="data/delisted_tickers.csv")
    patterns_df = checker.label_survivorship_status(patterns_df)
    report = checker.generate_survivorship_report(patterns_df)

Note:
    Delisted stocks should be INCLUDED in training (they have real outcomes),
    but need to be tracked for proper survivorship-aware validation splits.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Set
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SurvivorshipChecker:
    """
    Check and label patterns for survivorship bias.

    Survivorship bias occurs when analysis only includes stocks that "survived"
    to the present day, excluding delisted stocks that may have failed.

    This class:
    - Identifies patterns near delisting events
    - Filters patterns too close to IPO dates
    - Detects trading halts (potential regulatory issues)
    - Labels patterns for survivorship-aware validation
    """

    # Default thresholds
    DEFAULT_DELISTING_PROXIMITY_DAYS = 90    # Days before delisting to flag
    DEFAULT_IPO_COOLDOWN_DAYS = 252          # Min trading days before first pattern
    DEFAULT_HALT_THRESHOLD_DAYS = 5          # Gap days indicating potential halt
    DEFAULT_WEEKEND_SKIP_DAYS = 3            # Normal weekend gap (Fri-Mon)

    def __init__(
        self,
        delisted_tickers_file: Optional[str] = None,
        delisting_proximity_days: int = DEFAULT_DELISTING_PROXIMITY_DAYS,
        ipo_cooldown_days: int = DEFAULT_IPO_COOLDOWN_DAYS,
        halt_threshold_days: int = DEFAULT_HALT_THRESHOLD_DAYS,
    ):
        """
        Initialize SurvivorshipChecker.

        Args:
            delisted_tickers_file: Path to CSV with delisted ticker info
                                   Expected columns: ticker, delisting_date, reason
            delisting_proximity_days: Days before delisting to flag patterns
            ipo_cooldown_days: Minimum trading days required before first pattern
            halt_threshold_days: Gap days threshold to detect trading halts
        """
        self.delisting_proximity_days = delisting_proximity_days
        self.ipo_cooldown_days = ipo_cooldown_days
        self.halt_threshold_days = halt_threshold_days

        # Load delisted tickers if file provided
        self.delisted_tickers: Dict[str, Dict[str, Any]] = {}
        if delisted_tickers_file:
            self._load_delisted_tickers(delisted_tickers_file)

    def _load_delisted_tickers(self, file_path: str) -> None:
        """
        Load delisted tickers from CSV file.

        Expected CSV columns:
        - ticker: Stock symbol
        - delisting_date: Date of delisting
        - reason: (optional) Reason for delisting
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Delisted tickers file not found: {file_path}")
            return

        try:
            df = pd.read_csv(path)

            if 'ticker' not in df.columns or 'delisting_date' not in df.columns:
                logger.warning("Delisted file missing required columns (ticker, delisting_date)")
                return

            df['delisting_date'] = pd.to_datetime(df['delisting_date'])

            for _, row in df.iterrows():
                ticker = row['ticker']
                self.delisted_tickers[ticker] = {
                    'delisting_date': row['delisting_date'],
                    'reason': row.get('reason', 'unknown')
                }

            logger.info(f"Loaded {len(self.delisted_tickers)} delisted tickers from {file_path}")

        except Exception as e:
            logger.error(f"Error loading delisted tickers: {e}")

    def add_delisted_ticker(
        self,
        ticker: str,
        delisting_date: datetime,
        reason: str = 'unknown'
    ) -> None:
        """
        Add a delisted ticker manually.

        Args:
            ticker: Stock symbol
            delisting_date: Date of delisting
            reason: Reason for delisting
        """
        self.delisted_tickers[ticker] = {
            'delisting_date': pd.to_datetime(delisting_date),
            'reason': reason
        }

    def check_delisting_proximity(
        self,
        ticker: str,
        pattern_end_date: datetime
    ) -> Dict[str, Any]:
        """
        Check if pattern occurred near delisting date.

        Args:
            ticker: Stock symbol
            pattern_end_date: End date of the pattern

        Returns:
            Dictionary with:
            - is_delisted: Whether ticker was delisted
            - near_delisting: Whether pattern is within proximity threshold
            - days_to_delisting: Days from pattern to delisting (if applicable)
            - delisting_date: Date of delisting (if applicable)
            - reason: Delisting reason (if applicable)
            - outcome_complete: Whether outcome window completes before delisting
        """
        result = {
            'is_delisted': False,
            'near_delisting': False,
            'days_to_delisting': None,
            'delisting_date': None,
            'reason': None,
            'outcome_complete': True
        }

        if ticker not in self.delisted_tickers:
            return result

        delisting_info = self.delisted_tickers[ticker]
        delisting_date = delisting_info['delisting_date']
        pattern_end = pd.to_datetime(pattern_end_date)

        result['is_delisted'] = True
        result['delisting_date'] = delisting_date
        result['reason'] = delisting_info['reason']

        # Calculate days from pattern end to delisting
        days_to_delisting = (delisting_date - pattern_end).days
        result['days_to_delisting'] = days_to_delisting

        # Check if pattern is near delisting
        if 0 < days_to_delisting <= self.delisting_proximity_days:
            result['near_delisting'] = True

        # Check if 100-day outcome window completes before delisting
        # (using standard outcome window from constants)
        outcome_window_days = 100
        if days_to_delisting < outcome_window_days:
            result['outcome_complete'] = False

        return result

    def check_ipo_filter(
        self,
        ticker_df: pd.DataFrame,
        pattern_date: datetime,
        date_col: str = 'date'
    ) -> Dict[str, Any]:
        """
        Check if pattern has sufficient history (not too close to IPO).

        Patterns in the first year post-IPO should be flagged or rejected
        due to:
        - Insufficient data for indicator calculation
        - Abnormal price behavior typical of new listings
        - Lock-up expiration volatility

        Args:
            ticker_df: DataFrame with ticker's historical data
            pattern_date: Date of the pattern
            date_col: Column name for date (uses index if not found)

        Returns:
            Dictionary with:
            - has_sufficient_history: Whether enough history exists
            - trading_days_available: Number of trading days before pattern
            - first_date: First available date in data
            - days_required: Minimum required days
        """
        result = {
            'has_sufficient_history': False,
            'trading_days_available': 0,
            'first_date': None,
            'days_required': self.ipo_cooldown_days
        }

        if ticker_df is None or len(ticker_df) == 0:
            return result

        # Get dates
        if date_col in ticker_df.columns:
            dates = pd.to_datetime(ticker_df[date_col])
        elif isinstance(ticker_df.index, pd.DatetimeIndex):
            dates = ticker_df.index
        else:
            logger.warning("Cannot determine dates for IPO filter check")
            return result

        pattern_dt = pd.to_datetime(pattern_date)
        first_date = dates.min()

        result['first_date'] = first_date

        # Count trading days before pattern
        trading_days_before = (dates < pattern_dt).sum()
        result['trading_days_available'] = trading_days_before

        # Check if sufficient
        result['has_sufficient_history'] = trading_days_before >= self.ipo_cooldown_days

        return result

    def check_trading_halt(
        self,
        ticker_df: pd.DataFrame,
        date_col: str = 'date'
    ) -> Dict[str, Any]:
        """
        Identify trading halts (gaps > threshold days).

        Trading halts may indicate:
        - SEC investigation
        - Pending material news
        - Market-wide circuit breakers
        - Exchange-specific halts

        Args:
            ticker_df: DataFrame with ticker's historical data
            date_col: Column name for date (uses index if not found)

        Returns:
            Dictionary with:
            - has_halts: Whether any halts detected
            - halt_periods: List of (start_date, end_date, gap_days) tuples
            - max_gap_days: Maximum gap detected
            - total_halt_days: Sum of all halt periods
        """
        result = {
            'has_halts': False,
            'halt_periods': [],
            'max_gap_days': 0,
            'total_halt_days': 0
        }

        if ticker_df is None or len(ticker_df) < 2:
            return result

        # Get dates
        if date_col in ticker_df.columns:
            dates = pd.to_datetime(ticker_df[date_col]).sort_values()
        elif isinstance(ticker_df.index, pd.DatetimeIndex):
            dates = ticker_df.index.sort_values()
        else:
            logger.warning("Cannot determine dates for halt check")
            return result

        # Calculate gaps between consecutive trading days
        date_diffs = dates.to_series().diff()

        # Find gaps exceeding threshold
        # Note: Normal weekend is ~3 days (Fri to Mon), holidays can add more
        # We look for gaps significantly beyond normal
        for i, gap in enumerate(date_diffs):
            if pd.isna(gap):
                continue

            gap_days = gap.days

            # Adjust threshold for weekends/holidays
            # A gap of > halt_threshold trading days suggests potential halt
            # Conservative: we look for gaps > 2 weeks (10+ business days)
            if gap_days > (self.halt_threshold_days * 2 + 4):  # Account for weekends
                result['halt_periods'].append({
                    'start_date': dates.iloc[i - 1] if i > 0 else None,
                    'end_date': dates.iloc[i],
                    'gap_days': gap_days
                })
                result['total_halt_days'] += gap_days

        result['has_halts'] = len(result['halt_periods']) > 0

        if result['halt_periods']:
            result['max_gap_days'] = max(p['gap_days'] for p in result['halt_periods'])

        return result

    def label_survivorship_status(
        self,
        patterns_df: pd.DataFrame,
        ticker_col: str = 'ticker',
        date_col: str = 'pattern_end_date',
        data_end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Add survivorship status columns to patterns DataFrame.

        Adds columns:
        - survived: True if ticker is still active at data end date
        - near_delisting: True if pattern is within proximity of delisting
        - days_to_delisting: Days from pattern to delisting (if applicable)
        - delisting_reason: Reason for delisting (if applicable)

        Args:
            patterns_df: DataFrame with pattern data
            ticker_col: Column name for ticker symbol
            date_col: Column name for pattern date
            data_end_date: Reference date for survival status (defaults to today)

        Returns:
            DataFrame with added survivorship columns
        """
        if len(patterns_df) == 0:
            return patterns_df

        df = patterns_df.copy()

        # Default data end date to today
        if data_end_date is None:
            data_end_date = pd.Timestamp.now()
        else:
            data_end_date = pd.to_datetime(data_end_date)

        # Initialize columns
        df['survived'] = True
        df['near_delisting'] = False
        df['days_to_delisting'] = np.nan
        df['delisting_reason'] = None

        # Check each pattern
        for idx, row in df.iterrows():
            ticker = row[ticker_col]
            pattern_date = pd.to_datetime(row[date_col])

            result = self.check_delisting_proximity(ticker, pattern_date)

            if result['is_delisted']:
                df.at[idx, 'survived'] = False
                df.at[idx, 'near_delisting'] = result['near_delisting']
                df.at[idx, 'days_to_delisting'] = result['days_to_delisting']
                df.at[idx, 'delisting_reason'] = result['reason']

        # Log summary
        n_delisted = (~df['survived']).sum()
        n_near_delisting = df['near_delisting'].sum()

        logger.info(f"Survivorship labeling complete:")
        logger.info(f"  Total patterns: {len(df)}")
        logger.info(f"  From delisted stocks: {n_delisted} ({100*n_delisted/len(df):.1f}%)")
        logger.info(f"  Near delisting event: {n_near_delisting} ({100*n_near_delisting/len(df):.1f}%)")

        return df

    def generate_survivorship_report(
        self,
        patterns_df: pd.DataFrame,
        ticker_col: str = 'ticker',
        label_col: str = 'label',
        target_label: int = 2
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive survivorship bias report.

        Args:
            patterns_df: DataFrame with survivorship-labeled patterns
            ticker_col: Column name for ticker symbol
            label_col: Column name for outcome label
            target_label: Value indicating Target outcome

        Returns:
            Dictionary with survivorship statistics and analysis
        """
        if 'survived' not in patterns_df.columns:
            logger.warning("DataFrame missing 'survived' column. Run label_survivorship_status first.")
            return {'error': 'Missing survivorship labels'}

        report = {
            'total_patterns': len(patterns_df),
            'summary': {},
            'target_rate_comparison': {},
            'delisting_reasons': {},
            'recommendations': []
        }

        # Overall statistics
        survived = patterns_df['survived']
        report['summary'] = {
            'survived_patterns': survived.sum(),
            'delisted_patterns': (~survived).sum(),
            'survived_pct': 100 * survived.mean(),
            'near_delisting_count': patterns_df['near_delisting'].sum() if 'near_delisting' in patterns_df.columns else 0
        }

        # Target rate comparison: survived vs delisted
        if label_col in patterns_df.columns:
            survived_df = patterns_df[survived]
            delisted_df = patterns_df[~survived]

            survived_target_rate = (survived_df[label_col] == target_label).mean() if len(survived_df) > 0 else 0
            delisted_target_rate = (delisted_df[label_col] == target_label).mean() if len(delisted_df) > 0 else 0

            report['target_rate_comparison'] = {
                'survived_target_rate': survived_target_rate,
                'delisted_target_rate': delisted_target_rate,
                'difference': survived_target_rate - delisted_target_rate,
                'potential_bias': survived_target_rate > delisted_target_rate
            }

            # If survived has higher target rate, survivorship bias likely
            if survived_target_rate > delisted_target_rate * 1.2:  # 20% higher
                report['recommendations'].append(
                    "WARNING: Survived stocks show significantly higher Target rate "
                    f"({survived_target_rate:.1%} vs {delisted_target_rate:.1%}). "
                    "Consider stratified sampling in validation."
                )

        # Delisting reasons breakdown
        if 'delisting_reason' in patterns_df.columns:
            reason_counts = patterns_df[~survived]['delisting_reason'].value_counts().to_dict()
            report['delisting_reasons'] = reason_counts

        # Recommendations
        delisted_pct = report['summary']['delisted_patterns'] / max(1, len(patterns_df))
        if delisted_pct < 0.05:
            report['recommendations'].append(
                "Low representation of delisted stocks (<5%). "
                "Consider acquiring more delisted ticker data for robust validation."
            )

        if report['summary']['near_delisting_count'] > 0:
            report['recommendations'].append(
                f"{report['summary']['near_delisting_count']} patterns detected near delisting. "
                "These should be analyzed separately for outcome validity."
            )

        return report

    def filter_patterns_for_validation(
        self,
        patterns_df: pd.DataFrame,
        exclude_near_delisting: bool = True,
        require_outcome_complete: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split patterns into clean (for training) and flagged (for separate analysis).

        Args:
            patterns_df: DataFrame with survivorship-labeled patterns
            exclude_near_delisting: Whether to exclude near-delisting patterns
            require_outcome_complete: Whether to exclude incomplete outcomes

        Returns:
            Tuple of (clean_patterns, flagged_patterns) DataFrames
        """
        if 'survived' not in patterns_df.columns:
            logger.warning("DataFrame missing survivorship labels. Returning original data.")
            return patterns_df, pd.DataFrame()

        flagged_mask = pd.Series(False, index=patterns_df.index)

        if exclude_near_delisting and 'near_delisting' in patterns_df.columns:
            flagged_mask |= patterns_df['near_delisting']

        # Note: outcome_complete would need to be calculated separately

        clean_patterns = patterns_df[~flagged_mask]
        flagged_patterns = patterns_df[flagged_mask]

        logger.info(f"Pattern filtering: {len(clean_patterns)} clean, {len(flagged_patterns)} flagged")

        return clean_patterns, flagged_patterns


def check_survivorship_quick(
    patterns_df: pd.DataFrame,
    delisted_file: Optional[str] = None,
    ticker_col: str = 'ticker',
    date_col: str = 'pattern_end_date'
) -> Dict[str, Any]:
    """
    Quick survivorship check without creating checker instance.

    Args:
        patterns_df: DataFrame with pattern data
        delisted_file: Optional path to delisted tickers file
        ticker_col: Column name for ticker symbol
        date_col: Column name for pattern date

    Returns:
        Dictionary with basic survivorship statistics
    """
    checker = SurvivorshipChecker(delisted_tickers_file=delisted_file)

    labeled_df = checker.label_survivorship_status(
        patterns_df,
        ticker_col=ticker_col,
        date_col=date_col
    )

    report = checker.generate_survivorship_report(labeled_df, ticker_col=ticker_col)

    return report
