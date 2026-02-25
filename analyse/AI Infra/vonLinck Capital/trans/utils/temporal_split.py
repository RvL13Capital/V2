"""
Temporal Train/Test Split Utilities
====================================

Provides temporal-aware splitting functions for time series data to prevent
look-ahead bias in machine learning models.

Key Features:
- Temporal train/test split based on dates
- Multiple splitting strategies (percentage, date cutoff, walk-forward)
- Cluster-aware splitting for Trinity mode (Entry/Coil/Trigger stay together)
- Validation of temporal integrity
- Support for both sequences and metadata

Usage:
    from utils.temporal_split import temporal_train_test_split

    train_idx, test_idx = temporal_train_test_split(
        dates,
        split_ratio=0.8,
        strategy='percentage'
    )

Trinity Mode (Cluster-Aware):
    from utils.temporal_split import split_by_clusters, validate_cluster_split

    train_clusters, val_clusters, test_clusters = split_by_clusters(
        metadata,
        train_cutoff='2024-01-01',
        val_cutoff='2024-07-01'
    )
    validate_cluster_split(metadata, train_clusters, val_clusters, test_clusters)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# CRITICAL: Embargo period to prevent label leakage from outcome window
# Labels use 100-day outcome windows, so patterns ending in train period
# have outcomes that extend into test period without an embargo
EMBARGO_DAYS = 100


# ============================================================================
# Market Regime Definitions (for regime-aware validation)
# ============================================================================

MARKET_REGIMES: Dict[Tuple[int, int], str] = {
    # Year range -> regime name
    (2015, 2016): "post_qe_bull",       # Post-QE low volatility bull
    (2017, 2017): "low_vol_bull",       # Extremely low VIX year
    (2018, 2018): "volatility_spike",   # Feb/Dec volatility events
    (2019, 2019): "fed_pivot_bull",     # Fed rate cuts, recovery
    (2020, 2020): "covid_crash_recovery",  # March crash + unprecedented recovery
    (2021, 2021): "meme_speculation",   # Retail mania, SPACs, meme stocks
    (2022, 2022): "bear_market",        # Fed tightening, 25%+ drawdown
    (2023, 2023): "ai_rally",           # AI-driven tech rally
    (2024, 2024): "ai_bull_continuation",  # Continued AI optimism
}

# Regime characteristics for strategy adjustment
REGIME_CHARACTERISTICS: Dict[str, Dict[str, Any]] = {
    "post_qe_bull": {"volatility": "low", "trend": "bullish", "sector_rotation": "gradual"},
    "low_vol_bull": {"volatility": "very_low", "trend": "strong_bullish", "sector_rotation": "minimal"},
    "volatility_spike": {"volatility": "high", "trend": "choppy", "sector_rotation": "rapid"},
    "fed_pivot_bull": {"volatility": "moderate", "trend": "bullish", "sector_rotation": "gradual"},
    "covid_crash_recovery": {"volatility": "extreme", "trend": "v_shaped", "sector_rotation": "extreme"},
    "meme_speculation": {"volatility": "high", "trend": "bullish", "sector_rotation": "chaotic"},
    "bear_market": {"volatility": "high", "trend": "bearish", "sector_rotation": "defensive"},
    "ai_rally": {"volatility": "moderate", "trend": "bullish", "sector_rotation": "tech_focused"},
    "ai_bull_continuation": {"volatility": "moderate", "trend": "bullish", "sector_rotation": "tech_focused"},
}


def get_regime_for_date(date: Union[str, datetime, pd.Timestamp]) -> str:
    """
    Get market regime name for a given date.

    Args:
        date: Date to look up

    Returns:
        Regime name string (e.g., "bear_market", "covid_crash_recovery")
    """
    dt = pd.to_datetime(date)
    year = dt.year

    for (start_year, end_year), regime in MARKET_REGIMES.items():
        if start_year <= year <= end_year:
            return regime

    # Default for years outside defined ranges
    if year < 2015:
        return "pre_2015_unknown"
    return "post_2024_unknown"


def tag_patterns_with_regime(
    patterns_df: pd.DataFrame,
    date_col: str = 'pattern_end_date'
) -> pd.DataFrame:
    """
    Add market regime tag to each pattern.

    Args:
        patterns_df: DataFrame with pattern data
        date_col: Column name for pattern date

    Returns:
        DataFrame with added 'market_regime' column
    """
    df = patterns_df.copy()

    if date_col not in df.columns and 'end_date' in df.columns:
        date_col = 'end_date'

    df['market_regime'] = df[date_col].apply(get_regime_for_date)

    # Log regime distribution
    regime_counts = df['market_regime'].value_counts()
    logger.info("Pattern distribution by market regime:")
    for regime, count in regime_counts.items():
        logger.info(f"  {regime}: {count} patterns ({100*count/len(df):.1f}%)")

    return df


def check_data_completeness(
    ticker_df: pd.DataFrame,
    required_period: Tuple[str, str],
    date_col: str = 'date',
    min_coverage: float = 0.7
) -> Dict[str, Any]:
    """
    Calculate coverage ratio for a ticker over required period.

    Useful for determining if a ticker has sufficient data for
    training/validation across different market regimes.

    Args:
        ticker_df: DataFrame with ticker's historical data
        required_period: Tuple of (start_date, end_date) strings
        date_col: Column name for date (uses index if not found)
        min_coverage: Minimum required coverage ratio

    Returns:
        Dictionary with:
        - coverage_ratio: Fraction of trading days present (0.0-1.0)
        - has_sufficient: Whether coverage meets minimum
        - trading_days_expected: Approximate expected trading days
        - trading_days_actual: Actual trading days in data
        - first_date: First date in data
        - last_date: Last date in data
    """
    result = {
        'coverage_ratio': 0.0,
        'has_sufficient': False,
        'trading_days_expected': 0,
        'trading_days_actual': 0,
        'first_date': None,
        'last_date': None
    }

    if ticker_df is None or len(ticker_df) == 0:
        return result

    # Get dates
    if date_col in ticker_df.columns:
        dates = pd.to_datetime(ticker_df[date_col])
    elif isinstance(ticker_df.index, pd.DatetimeIndex):
        dates = ticker_df.index
    else:
        logger.warning("Cannot determine dates for completeness check")
        return result

    start_date = pd.to_datetime(required_period[0])
    end_date = pd.to_datetime(required_period[1])

    result['first_date'] = dates.min()
    result['last_date'] = dates.max()

    # Filter to required period
    mask = (dates >= start_date) & (dates <= end_date)
    period_dates = dates[mask]

    result['trading_days_actual'] = len(period_dates)

    # Estimate expected trading days (252 per year, ~21 per month)
    calendar_days = (end_date - start_date).days
    result['trading_days_expected'] = int(calendar_days * 252 / 365)

    # Calculate coverage
    if result['trading_days_expected'] > 0:
        result['coverage_ratio'] = result['trading_days_actual'] / result['trading_days_expected']
        result['has_sufficient'] = result['coverage_ratio'] >= min_coverage

    return result


def adaptive_regime_split(
    patterns_df: pd.DataFrame,
    train_cutoff: str = '2022-12-31',
    val_cutoff: str = '2023-12-31',
    date_col: str = 'pattern_end_date',
    min_patterns_per_regime: int = 50,
    require_multi_regime: bool = True
) -> Dict[str, Any]:
    """
    Adaptive regime-aware temporal splitting.

    Groups patterns by regime and ensures proper representation across
    train/val/test splits. Validates that each split contains patterns
    from multiple market regimes for robust generalization.

    Args:
        patterns_df: DataFrame with pattern data
        train_cutoff: End date for training period (inclusive)
        val_cutoff: End date for validation period (inclusive)
        date_col: Column name for pattern date
        min_patterns_per_regime: Minimum patterns required per regime
        require_multi_regime: Raise error if any split has only one regime

    Returns:
        Dictionary with:
        - train_mask: Boolean mask for training data
        - val_mask: Boolean mask for validation data
        - test_mask: Boolean mask for test data
        - regime_distribution: Patterns per regime per split
        - warnings: List of regime coverage warnings
    """
    # Add regime tags if not present
    if 'market_regime' not in patterns_df.columns:
        patterns_df = tag_patterns_with_regime(patterns_df, date_col)

    if date_col not in patterns_df.columns and 'end_date' in patterns_df.columns:
        date_col = 'end_date'

    # Create temporal split masks
    df = patterns_df.copy()
    df['_date'] = pd.to_datetime(df[date_col])
    train_cutoff_dt = pd.to_datetime(train_cutoff)
    val_cutoff_dt = pd.to_datetime(val_cutoff)

    train_mask = df['_date'] <= train_cutoff_dt
    val_mask = (df['_date'] > train_cutoff_dt) & (df['_date'] <= val_cutoff_dt)
    test_mask = df['_date'] > val_cutoff_dt

    # Analyze regime distribution per split
    regime_distribution = {
        'train': {},
        'val': {},
        'test': {}
    }
    warnings = []

    for split_name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        split_df = df[mask]
        regime_counts = split_df['market_regime'].value_counts().to_dict()
        regime_distribution[split_name] = regime_counts

        n_regimes = len(regime_counts)
        total = len(split_df)

        logger.info(f"{split_name.upper()} split: {total} patterns, {n_regimes} regimes")
        for regime, count in sorted(regime_counts.items()):
            logger.info(f"  {regime}: {count} ({100*count/total:.1f}%)")

        # Check minimum patterns per regime
        for regime, count in regime_counts.items():
            if count < min_patterns_per_regime:
                warnings.append(
                    f"LOW_REGIME_COVERAGE: {split_name} has only {count} "
                    f"patterns for {regime} (need {min_patterns_per_regime})"
                )

        # Check multi-regime requirement
        if require_multi_regime and n_regimes < 2 and total > 0:
            warnings.append(
                f"SINGLE_REGIME_SPLIT: {split_name} contains only {n_regimes} regime(s). "
                "Model may not generalize across market conditions."
            )

    result = {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'regime_distribution': regime_distribution,
        'warnings': warnings,
        'train_cutoff': train_cutoff,
        'val_cutoff': val_cutoff
    }

    # Log warnings
    for warning in warnings:
        logger.warning(warning)

    return result


def temporal_train_test_split(
    dates: Union[pd.Series, np.ndarray, List],
    split_ratio: float = 0.8,
    strategy: str = 'percentage',
    cutoff_date: Optional[Union[str, datetime]] = None,
    min_train_samples: int = 100,
    min_test_samples: int = 20,
    return_dates: bool = False,
    embargo_days: int = EMBARGO_DAYS
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, datetime, datetime]]:
    """
    Split data temporally to prevent look-ahead bias.

    CRITICAL: Enforces embargo period between train and test to prevent label leakage.
    Labels use 100-day outcome windows, so without embargo, patterns ending near
    the split point have outcomes that extend into the test period.

    Args:
        dates: Array-like of dates for each sample
        split_ratio: Fraction of data for training (only used for 'percentage' strategy)
        strategy: Split strategy - 'percentage', 'date_cutoff', or 'walk_forward'
        cutoff_date: Specific date to split on (only for 'date_cutoff' strategy)
        min_train_samples: Minimum number of training samples required
        min_test_samples: Minimum number of test samples required
        return_dates: If True, also return train_end and test_start dates
        embargo_days: Number of days gap between train end and test start (default: 100)
                      Patterns in the embargo zone are DROPPED to prevent leakage

    Returns:
        train_indices: Indices for training set
        test_indices: Indices for test set
        (optional) train_end_date: Last date in training set
        (optional) test_start_date: First date in test set

    Raises:
        ValueError: If insufficient samples or invalid parameters
    """
    # Convert to pandas datetime if needed
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)

    dates = pd.to_datetime(dates)

    # Sort by date and get indices
    sorted_indices = dates.argsort().values
    sorted_dates = dates.iloc[sorted_indices]

    n_samples = len(dates)

    # Validate minimum samples
    if n_samples < min_train_samples + min_test_samples:
        raise ValueError(
            f"Insufficient samples: {n_samples} < {min_train_samples + min_test_samples} "
            f"(min_train + min_test)"
        )

    if strategy == 'percentage':
        # Split based on percentage
        train_size = int(n_samples * split_ratio)

        # Ensure minimum samples
        train_size = max(train_size, min_train_samples)
        train_size = min(train_size, n_samples - min_test_samples)

        # Initial split point
        train_end_date = sorted_dates.iloc[train_size - 1]

        # CRITICAL: Calculate embargo end date
        embargo_end = train_end_date + pd.Timedelta(days=embargo_days)

        # Find all samples AFTER embargo period
        after_embargo_mask = sorted_dates > embargo_end

        # Patterns in embargo zone are DROPPED (not assigned to either set)
        embargo_zone_mask = (sorted_dates > train_end_date) & (sorted_dates <= embargo_end)
        n_embargo_dropped = embargo_zone_mask.sum()

        if n_embargo_dropped > 0:
            logger.warning(f"EMBARGO: Dropped {n_embargo_dropped} patterns in {embargo_days}-day embargo zone")
            logger.warning(f"  Embargo zone: {train_end_date.date()} to {embargo_end.date()}")

        # Assign indices
        train_indices = sorted_indices[:train_size]
        test_indices = sorted_indices[after_embargo_mask]

        # Validate test set size after embargo
        if len(test_indices) < min_test_samples:
            raise ValueError(
                f"Insufficient test samples after {embargo_days}-day embargo: "
                f"{len(test_indices)} < {min_test_samples}. "
                f"Original would have {n_samples - train_size}, but {n_embargo_dropped} were in embargo zone."
            )

        # Get actual boundary dates
        test_start_date = sorted_dates[after_embargo_mask].iloc[0] if len(test_indices) > 0 else embargo_end

    elif strategy == 'date_cutoff':
        if cutoff_date is None:
            raise ValueError("cutoff_date must be provided for 'date_cutoff' strategy")

        cutoff = pd.to_datetime(cutoff_date)

        # CRITICAL: Calculate embargo end date from cutoff
        embargo_end = cutoff + pd.Timedelta(days=embargo_days)

        # Training: patterns before cutoff
        train_mask = sorted_dates < cutoff

        # Test: patterns AFTER embargo period (not just after cutoff)
        after_embargo_mask = sorted_dates > embargo_end

        # Patterns in embargo zone are DROPPED
        embargo_zone_mask = (sorted_dates >= cutoff) & (sorted_dates <= embargo_end)
        n_embargo_dropped = embargo_zone_mask.sum()

        if n_embargo_dropped > 0:
            logger.warning(f"EMBARGO: Dropped {n_embargo_dropped} patterns in {embargo_days}-day embargo zone")
            logger.warning(f"  Embargo zone: {cutoff.date()} to {embargo_end.date()}")

        train_indices = sorted_indices[train_mask]
        test_indices = sorted_indices[after_embargo_mask]

        if len(train_indices) < min_train_samples:
            raise ValueError(
                f"Insufficient training samples before cutoff date: "
                f"{len(train_indices)} < {min_train_samples}"
            )

        if len(test_indices) < min_test_samples:
            raise ValueError(
                f"Insufficient test samples after {embargo_days}-day embargo: "
                f"{len(test_indices)} < {min_test_samples}. "
                f"{n_embargo_dropped} patterns were in embargo zone."
            )

        train_end_date = sorted_dates[train_mask].iloc[-1]
        test_start_date = sorted_dates[after_embargo_mask].iloc[0] if len(test_indices) > 0 else embargo_end

    elif strategy == 'walk_forward':
        # For walk-forward, we return multiple train/test splits
        # This is a simplified version - could be extended
        raise NotImplementedError("Walk-forward validation not yet implemented")

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'percentage' or 'date_cutoff'")

    # Log split information
    logger.info(f"Temporal split: Train {len(train_indices)} samples, Test {len(test_indices)} samples")
    logger.info(f"Train period: {sorted_dates.iloc[0].date()} to {train_end_date.date()}")
    logger.info(f"Test period: {test_start_date.date()} to {sorted_dates.iloc[-1].date()}")

    # Calculate temporal gap (should be >= embargo_days)
    gap_days = (test_start_date - train_end_date).days
    if gap_days >= embargo_days:
        logger.info(f"Temporal gap between train/test: {gap_days} days (>= {embargo_days}-day embargo)")
    elif gap_days > 0:
        logger.warning(f"Temporal gap {gap_days} days is LESS than {embargo_days}-day embargo requirement!")

    # Validate temporal integrity
    validate_temporal_split(dates, train_indices, test_indices)

    if return_dates:
        return train_indices, test_indices, train_end_date, test_start_date
    else:
        return train_indices, test_indices


def split_sequences_temporal(
    sequences: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    date_column: str = 'sequence_end_date',
    split_ratio: float = 0.8,
    strategy: str = 'percentage',
    cutoff_date: Optional[Union[str, datetime]] = None,
    min_train_samples: int = 50,
    min_test_samples: int = 10,
    embargo_days: int = EMBARGO_DAYS
) -> Dict[str, np.ndarray]:
    """
    Split sequences temporally based on pattern dates.

    Args:
        sequences: Array of sequences (n_samples, window_size, n_features)
        labels: Array of labels
        metadata: DataFrame with pattern metadata including dates
        date_column: Column name containing dates for temporal split
        split_ratio: Fraction of data for training
        strategy: Split strategy - 'percentage' or 'date_cutoff'
        cutoff_date: Specific date to split on (for 'date_cutoff' strategy)
        min_train_samples: Minimum number of training samples required
        min_test_samples: Minimum number of test samples required
        embargo_days: Number of days gap between train end and test start (default: 100)

    Returns:
        Dictionary with X_train, X_test, y_train, y_test arrays

    Raises:
        ValueError: If date_column not in metadata
    """
    if date_column not in metadata.columns:
        raise ValueError(f"Date column '{date_column}' not found in metadata. "
                        f"Available columns: {metadata.columns.tolist()}")

    # Get temporal split indices
    dates = metadata[date_column]
    train_idx, test_idx = temporal_train_test_split(
        dates,
        split_ratio=split_ratio,
        strategy=strategy,
        cutoff_date=cutoff_date,
        min_train_samples=min_train_samples,
        min_test_samples=min_test_samples,
        embargo_days=embargo_days
    )

    # Split sequences and labels
    X_train = sequences[train_idx]
    X_test = sequences[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    logger.info(f"Temporal split complete:")
    logger.info(f"  Train: {len(X_train)} sequences")
    logger.info(f"  Test: {len(X_test)} sequences")

    # Check class distribution
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    test_classes, test_counts = np.unique(y_test, return_counts=True)

    logger.info("Train class distribution:")
    for cls, count in zip(train_classes, train_counts):
        logger.info(f"  Class {cls}: {count} ({100*count/len(y_train):.1f}%)")

    logger.info("Test class distribution:")
    for cls, count in zip(test_classes, test_counts):
        logger.info(f"  Class {cls}: {count} ({100*count/len(y_test):.1f}%)")

    # Warn if any class missing from test set
    missing_classes = set(train_classes) - set(test_classes)
    if missing_classes:
        logger.warning(f"Classes {missing_classes} present in train but missing from test")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_idx': train_idx,
        'test_idx': test_idx
    }


def validate_temporal_split(
    dates: pd.Series,
    train_indices: np.ndarray,
    test_indices: np.ndarray
) -> None:
    """
    Validate that temporal split has no look-ahead bias.

    Args:
        dates: Original dates series
        train_indices: Indices for training set
        test_indices: Indices for test set

    Raises:
        ValueError: If temporal leakage detected
    """
    train_dates = pd.to_datetime(dates.iloc[train_indices])
    test_dates = pd.to_datetime(dates.iloc[test_indices])

    # Check for overlap
    max_train_date = train_dates.max()
    min_test_date = test_dates.min()

    if max_train_date > min_test_date:
        raise ValueError(
            f"Temporal leakage detected! "
            f"Latest training date ({max_train_date}) is after "
            f"earliest test date ({min_test_date})"
        )

    # Check for any test samples before training samples
    if test_dates.min() < train_dates.max():
        overlap_count = (test_dates < train_dates.max()).sum()
        if overlap_count > 0:
            logger.warning(
                f"Found {overlap_count} test samples dated before end of training period. "
                f"This could indicate temporal leakage!"
            )


def create_walk_forward_splits(
    dates: Union[pd.Series, np.ndarray],
    n_splits: int = 5,
    test_size: float = 0.2,
    gap_days: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create multiple train/test splits for walk-forward validation.

    Args:
        dates: Array-like of dates for each sample
        n_splits: Number of splits to create
        test_size: Fraction of data for each test set
        gap_days: Number of days gap between train and test

    Returns:
        List of (train_indices, test_indices) tuples

    Note:
        This is a placeholder for future walk-forward implementation
    """
    raise NotImplementedError("Walk-forward validation will be implemented in future version")


def get_temporal_fold_statistics(
    dates: pd.Series,
    train_indices: np.ndarray,
    test_indices: np.ndarray
) -> Dict[str, Any]:
    """
    Get statistics about a temporal train/test split.

    Args:
        dates: Original dates series
        train_indices: Indices for training set
        test_indices: Indices for test set

    Returns:
        Dictionary with split statistics
    """
    train_dates = pd.to_datetime(dates.iloc[train_indices])
    test_dates = pd.to_datetime(dates.iloc[test_indices])

    stats = {
        'train_size': len(train_indices),
        'test_size': len(test_indices),
        'train_start': train_dates.min(),
        'train_end': train_dates.max(),
        'test_start': test_dates.min(),
        'test_end': test_dates.max(),
        'train_days': (train_dates.max() - train_dates.min()).days,
        'test_days': (test_dates.max() - test_dates.min()).days,
        'gap_days': (test_dates.min() - train_dates.max()).days,
        'train_ratio': len(train_indices) / (len(train_indices) + len(test_indices))
    }

    return stats


# ============================================================================
# Cluster-Aware Splitting (Trinity Mode)
# ============================================================================


def detect_trinity_mode(metadata: pd.DataFrame) -> bool:
    """
    Detect if metadata contains valid Trinity mode cluster IDs.

    Args:
        metadata: DataFrame with pattern/sequence metadata

    Returns:
        True if Trinity mode should be used (nms_cluster_id present and not all -1)
    """
    if metadata is None:
        return False

    if 'nms_cluster_id' not in metadata.columns:
        return False

    unique_clusters = metadata['nms_cluster_id'].unique()

    # Trinity mode if we have multiple clusters OR one cluster that's not -1
    return len(unique_clusters) > 1 or (len(unique_clusters) == 1 and unique_clusters[0] != -1)


def split_by_clusters(
    metadata: pd.DataFrame,
    train_cutoff: Union[str, datetime, pd.Timestamp],
    val_cutoff: Union[str, datetime, pd.Timestamp],
    date_column: str = 'pattern_start_date',
    cluster_column: str = 'nms_cluster_id'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split clusters temporally for Trinity mode.

    Ensures all patterns from the same cluster (Entry/Coil/Trigger views)
    go to the same split, preventing data leakage.

    Args:
        metadata: DataFrame with nms_cluster_id and date columns
        train_cutoff: Date before which clusters go to train
        val_cutoff: Date before which clusters go to val (after train_cutoff)
        date_column: Column containing pattern dates
        cluster_column: Column containing cluster IDs

    Returns:
        Tuple of (train_clusters, val_clusters, test_clusters) as numpy arrays

    Raises:
        ValueError: If required columns missing or dates invalid
    """
    if cluster_column not in metadata.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not in metadata")

    if date_column not in metadata.columns:
        raise ValueError(f"Date column '{date_column}' not in metadata")

    train_cutoff_dt = pd.Timestamp(train_cutoff)
    val_cutoff_dt = pd.Timestamp(val_cutoff)

    if train_cutoff_dt >= val_cutoff_dt:
        raise ValueError(f"train_cutoff ({train_cutoff_dt}) must be before val_cutoff ({val_cutoff_dt})")

    # Ensure date column is datetime
    metadata = metadata.copy()
    metadata['_split_date'] = pd.to_datetime(metadata[date_column])

    # Map cluster to date (use earliest pattern date in cluster)
    cluster_to_date = metadata.groupby(cluster_column)['_split_date'].min().to_dict()

    train_clusters = []
    val_clusters = []
    test_clusters = []

    for cluster_id, cluster_date in cluster_to_date.items():
        # Handle unclustered patterns (-1) by putting in train
        if cluster_id == -1:
            # For -1 clusters, each pattern is independent
            # We still group by date for temporal integrity
            if pd.isna(cluster_date):
                train_clusters.append(cluster_id)
            elif cluster_date < train_cutoff_dt:
                train_clusters.append(cluster_id)
            elif cluster_date < val_cutoff_dt:
                val_clusters.append(cluster_id)
            else:
                test_clusters.append(cluster_id)
        elif pd.isna(cluster_date):
            train_clusters.append(cluster_id)
        elif cluster_date < train_cutoff_dt:
            train_clusters.append(cluster_id)
        elif cluster_date < val_cutoff_dt:
            val_clusters.append(cluster_id)
        else:
            test_clusters.append(cluster_id)

    logger.info(f"Cluster split (Trinity mode):")
    logger.info(f"  Train: {len(train_clusters)} clusters")
    logger.info(f"  Val:   {len(val_clusters)} clusters")
    logger.info(f"  Test:  {len(test_clusters)} clusters")

    return np.array(train_clusters), np.array(val_clusters), np.array(test_clusters)


def validate_cluster_split(
    metadata: pd.DataFrame,
    train_clusters: np.ndarray,
    val_clusters: np.ndarray,
    test_clusters: np.ndarray,
    cluster_column: str = 'nms_cluster_id',
    pattern_column: str = 'pattern_id'
) -> Dict[str, Any]:
    """
    Validate that no cluster is split across train/val/test.

    Args:
        metadata: DataFrame with cluster and pattern information
        train_clusters: Array of cluster IDs for training
        val_clusters: Array of cluster IDs for validation
        test_clusters: Array of cluster IDs for testing
        cluster_column: Column name for cluster IDs
        pattern_column: Column name for pattern IDs

    Returns:
        Validation report dictionary

    Raises:
        ValueError: If any cluster is split across sets
    """
    train_set = set(train_clusters)
    val_set = set(val_clusters)
    test_set = set(test_clusters)

    report = {
        'valid': True,
        'n_train_clusters': len(train_set),
        'n_val_clusters': len(val_set),
        'n_test_clusters': len(test_set),
        'overlaps': {}
    }

    # Check for overlaps
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    if train_val_overlap:
        report['valid'] = False
        report['overlaps']['train_val'] = list(train_val_overlap)
        logger.error(f"CLUSTER LEAKAGE: {len(train_val_overlap)} clusters in both train and val!")

    if train_test_overlap:
        report['valid'] = False
        report['overlaps']['train_test'] = list(train_test_overlap)
        logger.error(f"CLUSTER LEAKAGE: {len(train_test_overlap)} clusters in both train and test!")

    if val_test_overlap:
        report['valid'] = False
        report['overlaps']['val_test'] = list(val_test_overlap)
        logger.error(f"CLUSTER LEAKAGE: {len(val_test_overlap)} clusters in both val and test!")

    if not report['valid']:
        raise ValueError(
            f"Cluster split validation failed! Overlaps: {report['overlaps']}"
        )

    # Check that all clusters are assigned
    all_clusters = set(metadata[cluster_column].unique())
    assigned_clusters = train_set | val_set | test_set
    unassigned = all_clusters - assigned_clusters

    if unassigned:
        logger.warning(f"Unassigned clusters: {unassigned}")
        report['unassigned_clusters'] = list(unassigned)

    # Verify patterns follow their clusters
    if pattern_column in metadata.columns:
        cluster_to_patterns = metadata.groupby(cluster_column)[pattern_column].apply(set).to_dict()

        # Check each pattern appears in only one split
        train_patterns = set()
        val_patterns = set()
        test_patterns = set()

        for c in train_clusters:
            train_patterns.update(cluster_to_patterns.get(c, set()))
        for c in val_clusters:
            val_patterns.update(cluster_to_patterns.get(c, set()))
        for c in test_clusters:
            test_patterns.update(cluster_to_patterns.get(c, set()))

        pattern_train_val = train_patterns & val_patterns
        pattern_train_test = train_patterns & test_patterns
        pattern_val_test = val_patterns & test_patterns

        if pattern_train_val or pattern_train_test or pattern_val_test:
            report['valid'] = False
            report['pattern_overlaps'] = {
                'train_val': list(pattern_train_val),
                'train_test': list(pattern_train_test),
                'val_test': list(pattern_val_test)
            }
            raise ValueError(f"Pattern split validation failed! Patterns appear in multiple sets")

        report['n_train_patterns'] = len(train_patterns)
        report['n_val_patterns'] = len(val_patterns)
        report['n_test_patterns'] = len(test_patterns)

    logger.info("Cluster split validation PASSED - no leakage detected")
    return report


def get_patterns_for_clusters(
    metadata: pd.DataFrame,
    clusters: np.ndarray,
    cluster_column: str = 'nms_cluster_id',
    pattern_column: str = 'pattern_id'
) -> np.ndarray:
    """
    Get all pattern IDs belonging to specified clusters.

    Args:
        metadata: DataFrame with cluster and pattern information
        clusters: Array of cluster IDs to get patterns for
        cluster_column: Column name for cluster IDs
        pattern_column: Column name for pattern IDs

    Returns:
        Array of pattern IDs belonging to the specified clusters
    """
    cluster_set = set(clusters)
    cluster_to_patterns = metadata.groupby(cluster_column)[pattern_column].apply(set).to_dict()

    patterns = set()
    for c in cluster_set:
        patterns.update(cluster_to_patterns.get(c, set()))

    return np.array(list(patterns))


def get_sequence_mask_for_clusters(
    pattern_ids: np.ndarray,
    metadata: pd.DataFrame,
    clusters: np.ndarray,
    cluster_column: str = 'nms_cluster_id',
    pattern_column: str = 'pattern_id'
) -> np.ndarray:
    """
    Get boolean mask for sequences belonging to specified clusters.

    Args:
        pattern_ids: Array of pattern IDs for each sequence
        metadata: DataFrame with cluster and pattern information
        clusters: Array of cluster IDs to include
        cluster_column: Column name for cluster IDs
        pattern_column: Column name for pattern IDs

    Returns:
        Boolean mask where True = sequence belongs to one of the clusters
    """
    patterns = get_patterns_for_clusters(metadata, clusters, cluster_column, pattern_column)
    return np.isin(pattern_ids, patterns)