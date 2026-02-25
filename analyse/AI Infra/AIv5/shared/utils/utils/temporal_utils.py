"""
Temporal Utilities for Walk-Forward Validation

Provides utilities for ensuring temporal integrity in time series analysis:
- Chronological sorting and validation
- Temporal train/test splits
- Future data leakage detection
- Walk-forward window generation
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def ensure_temporal_order(
    df: pd.DataFrame,
    date_column: str = 'date',
    validate: bool = True
) -> pd.DataFrame:
    """
    Ensure DataFrame is chronologically sorted.

    Args:
        df: Input DataFrame
        date_column: Name of date column
        validate: Whether to validate ordering (raises error if already sorted incorrectly)

    Returns:
        Chronologically sorted DataFrame

    Raises:
        ValueError: If date column missing or contains invalid dates
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame. "
                        f"Available columns: {list(df.columns)}")

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        logger.info(f"Converting '{date_column}' to datetime")
        df[date_column] = pd.to_datetime(df[date_column])

    # Check for NaT values
    nat_count = df[date_column].isna().sum()
    if nat_count > 0:
        raise ValueError(f"Found {nat_count} invalid/missing dates in '{date_column}'")

    # Check current ordering
    is_sorted = df[date_column].is_monotonic_increasing

    if validate and not is_sorted:
        # Calculate how many rows are out of order
        sorted_df = df.sort_values(date_column)
        mismatches = (df.index != sorted_df.index).sum()
        logger.warning(f"Data not chronologically sorted! {mismatches} rows out of order.")

    # Sort by date
    df_sorted = df.sort_values(date_column).reset_index(drop=True)

    logger.info(f"Data sorted by '{date_column}': {len(df_sorted)} rows, "
               f"range {df_sorted[date_column].min()} to {df_sorted[date_column].max()}")

    return df_sorted


def split_temporal_data(
    df: pd.DataFrame,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    train_ratio: float = 0.8,
    date_column: str = 'date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (earlier = train, later = test).

    Args:
        df: Input DataFrame (must be chronologically sorted)
        train_size: Absolute number of samples for training (overrides train_ratio)
        test_size: Absolute number of samples for testing
        train_ratio: Fraction of data for training (default 0.8)
        date_column: Name of date column for logging

    Returns:
        Tuple of (train_df, test_df)

    Raises:
        ValueError: If data not sorted or invalid split parameters
    """
    # Ensure chronological order
    if date_column in df.columns:
        if not df[date_column].is_monotonic_increasing:
            raise ValueError(f"Data must be chronologically sorted by '{date_column}' before splitting")

    n_samples = len(df)

    # Determine split point
    if train_size is not None:
        split_idx = train_size
    elif test_size is not None:
        split_idx = n_samples - test_size
    else:
        split_idx = int(n_samples * train_ratio)

    # Validate split
    if split_idx <= 0 or split_idx >= n_samples:
        raise ValueError(f"Invalid split: split_idx={split_idx}, n_samples={n_samples}")

    # Split data
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if date_column in df.columns:
        logger.info(f"Temporal split: Train={len(train_df)} "
                   f"({train_df[date_column].min()} to {train_df[date_column].max()}), "
                   f"Test={len(test_df)} "
                   f"({test_df[date_column].min()} to {test_df[date_column].max()})")
    else:
        logger.info(f"Temporal split: Train={len(train_df)}, Test={len(test_df)}")

    return train_df, test_df


def validate_no_future_leakage(
    train_dates: pd.Series,
    test_dates: pd.Series,
    raise_error: bool = True
) -> bool:
    """
    Validate that no test dates occur before train dates (future leakage check).

    Args:
        train_dates: Training set dates
        test_dates: Test set dates
        raise_error: Whether to raise error if leakage detected

    Returns:
        True if no leakage detected, False otherwise

    Raises:
        ValueError: If future leakage detected and raise_error=True
    """
    if len(train_dates) == 0 or len(test_dates) == 0:
        logger.warning("Empty date series provided for leakage validation")
        return True

    train_max = train_dates.max()
    test_min = test_dates.min()

    if test_min < train_max:
        leakage_days = (train_max - test_min).days
        error_msg = (f"FUTURE DATA LEAKAGE DETECTED! "
                    f"Test data starts {test_min} but train data ends {train_max}. "
                    f"Overlap: {leakage_days} days")

        if raise_error:
            raise ValueError(error_msg)
        else:
            logger.error(error_msg)
            return False

    logger.info(f"✓ No future leakage: Train ends {train_max}, Test starts {test_min}")
    return True


def generate_walk_forward_windows(
    df: pd.DataFrame,
    initial_train_size: int,
    test_period_size: int,
    step_size: int,
    date_column: str = 'date',
    min_train_size: Optional[int] = None
) -> List[Dict]:
    """
    Generate walk-forward validation windows.

    Args:
        df: Input DataFrame (must be chronologically sorted)
        initial_train_size: Initial training window size (in samples)
        test_period_size: Size of each test period (in samples)
        step_size: How many samples to step forward each iteration
        date_column: Name of date column
        min_train_size: Minimum training size (defaults to initial_train_size)

    Returns:
        List of dictionaries with train/test indices and date ranges

    Example:
        >>> windows = generate_walk_forward_windows(
        ...     df, initial_train_size=500, test_period_size=60, step_size=20
        ... )
        >>> # Window 1: Train[0:500], Test[500:560]
        >>> # Window 2: Train[0:520], Test[520:580]
        >>> # ...
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found")

    if not df[date_column].is_monotonic_increasing:
        raise ValueError("Data must be chronologically sorted")

    min_train_size = min_train_size or initial_train_size
    n_samples = len(df)

    windows = []
    train_end = initial_train_size

    window_num = 1

    while train_end + test_period_size <= n_samples:
        # Define indices
        train_start = 0  # Expanding window (use all historical data)
        # Alternative: train_start = max(0, train_end - max_train_size)  # Rolling window

        test_start = train_end
        test_end = min(test_start + test_period_size, n_samples)

        # Ensure minimum train size
        if train_end - train_start < min_train_size:
            logger.warning(f"Window {window_num}: Train size {train_end - train_start} "
                          f"< min_train_size {min_train_size}, skipping")
            train_end += step_size
            continue

        # Extract date ranges
        train_dates = df[date_column].iloc[train_start:train_end]
        test_dates = df[date_column].iloc[test_start:test_end]

        window = {
            'window_num': window_num,
            'train_start_idx': train_start,
            'train_end_idx': train_end,
            'test_start_idx': test_start,
            'test_end_idx': test_end,
            'train_size': train_end - train_start,
            'test_size': test_end - test_start,
            'train_start_date': train_dates.iloc[0],
            'train_end_date': train_dates.iloc[-1],
            'test_start_date': test_dates.iloc[0],
            'test_end_date': test_dates.iloc[-1]
        }

        windows.append(window)

        logger.debug(f"Window {window_num}: Train[{train_start}:{train_end}] "
                    f"({window['train_start_date']} to {window['train_end_date']}), "
                    f"Test[{test_start}:{test_end}] "
                    f"({window['test_start_date']} to {window['test_end_date']})")

        # Move forward
        train_end += step_size
        window_num += 1

    logger.info(f"Generated {len(windows)} walk-forward windows:")
    logger.info(f"  Initial train size: {initial_train_size}")
    logger.info(f"  Test period size: {test_period_size}")
    logger.info(f"  Step size: {step_size}")

    return windows


def aggregate_walk_forward_predictions(
    predictions_list: List[pd.DataFrame],
    date_column: str = 'date'
) -> pd.DataFrame:
    """
    Aggregate predictions from multiple walk-forward windows.

    Handles overlapping predictions by keeping the first (earliest) prediction.

    Args:
        predictions_list: List of prediction DataFrames from each window
        date_column: Name of date column

    Returns:
        Aggregated DataFrame with unique predictions
    """
    if not predictions_list:
        return pd.DataFrame()

    # Concatenate all predictions
    all_predictions = pd.concat(predictions_list, ignore_index=True)

    # Sort by date
    if date_column in all_predictions.columns:
        all_predictions = all_predictions.sort_values(date_column)

    # Remove duplicates (keep first prediction for each ticker-date)
    if 'ticker' in all_predictions.columns and date_column in all_predictions.columns:
        all_predictions = all_predictions.drop_duplicates(
            subset=['ticker', date_column],
            keep='first'
        )
        logger.info(f"Aggregated predictions: {len(all_predictions)} unique predictions")
    else:
        logger.warning("Could not deduplicate predictions (missing ticker or date column)")

    return all_predictions


def get_temporal_train_test_dates(
    df: pd.DataFrame,
    train_years: float = 2.0,
    test_months: int = 3,
    date_column: str = 'date'
) -> Tuple[datetime, datetime, datetime, datetime]:
    """
    Calculate train/test date boundaries based on time periods.

    Args:
        df: Input DataFrame with date column
        train_years: Number of years for training (from start)
        test_months: Number of months for testing (after training)
        date_column: Name of date column

    Returns:
        Tuple of (train_start, train_end, test_start, test_end)
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found")

    dates = pd.to_datetime(df[date_column])

    train_start = dates.min()
    train_end = train_start + pd.DateOffset(years=train_years)
    test_start = train_end
    test_end = test_start + pd.DateOffset(months=test_months)

    # Ensure test_end doesn't exceed data
    if test_end > dates.max():
        test_end = dates.max()

    logger.info(f"Temporal boundaries:")
    logger.info(f"  Train: {train_start} to {train_end} ({train_years} years)")
    logger.info(f"  Test:  {test_start} to {test_end} ({test_months} months)")

    return train_start, train_end, test_start, test_end


def validate_temporal_integrity(
    df: pd.DataFrame,
    date_column: str = 'date',
    raise_on_error: bool = True
) -> Dict[str, any]:
    """
    Comprehensive temporal integrity validation.

    Checks:
    - Chronological ordering
    - Duplicate dates
    - Date gaps
    - Future dates

    Args:
        df: Input DataFrame
        date_column: Name of date column
        raise_on_error: Whether to raise errors or return issues

    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'issues': [],
        'warnings': []
    }

    # Check date column exists
    if date_column not in df.columns:
        results['is_valid'] = False
        results['issues'].append(f"Date column '{date_column}' not found")
        if raise_on_error:
            raise ValueError(results['issues'][-1])
        return results

    dates = pd.to_datetime(df[date_column])

    # Check chronological order
    if not dates.is_monotonic_increasing:
        results['is_valid'] = False
        results['issues'].append("Dates not in chronological order")
        if raise_on_error:
            raise ValueError("Data must be chronologically sorted")

    # Check for duplicates
    duplicates = dates.duplicated().sum()
    if duplicates > 0:
        results['warnings'].append(f"Found {duplicates} duplicate dates")

    # Check for NaT
    nat_count = dates.isna().sum()
    if nat_count > 0:
        results['is_valid'] = False
        results['issues'].append(f"Found {nat_count} invalid dates (NaT)")
        if raise_on_error:
            raise ValueError(f"Found {nat_count} invalid dates")

    # Check for future dates
    today = pd.Timestamp.now()
    future_dates = (dates > today).sum()
    if future_dates > 0:
        results['warnings'].append(f"Found {future_dates} dates in the future")

    # Check for large gaps
    date_diffs = dates.diff().dropna()
    if len(date_diffs) > 0:
        max_gap = date_diffs.max()
        if max_gap > pd.Timedelta(days=30):
            results['warnings'].append(f"Large date gap detected: {max_gap.days} days")

    logger.info(f"Temporal integrity check: {'✓ PASS' if results['is_valid'] else '✗ FAIL'}")
    if results['issues']:
        logger.error(f"Issues: {', '.join(results['issues'])}")
    if results['warnings']:
        logger.warning(f"Warnings: {', '.join(results['warnings'])}")

    return results
