"""
Ignition-Aware Pattern Prioritization
======================================

Prioritizes patterns for live trading by scoring their ignition potential.
High-ignition patterns are oversampled during training to catch imminent breakouts.

Scoring Hierarchy:
1. ignition_score (from ignition_detector) - highest weight
2. coil_intensity (from coil features) - medium weight
3. pattern_recency (days since pattern end) - lowest weight

Usage:
    from features.ignition_prioritizer import (
        score_patterns_for_ignition,
        prioritize_and_oversample,
        stratified_oversample
    )

    # Score patterns
    patterns_df = score_patterns_for_ignition(patterns_df, ticker_data_cache)

    # Prioritize and oversample
    patterns_df = prioritize_and_oversample(
        patterns_df,
        oversample_factor=2.0,
        top_pct=0.2
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# Weight configuration for composite priority score
PRIORITY_WEIGHTS = {
    'ignition_score': 0.50,      # Highest priority - imminent breakout signals
    'coil_intensity': 0.35,      # Medium priority - pattern quality
    'recency_score': 0.15,       # Lowest priority - favor recent patterns
}

# Ignition score thresholds
IGNITION_HIGH_THRESHOLD = 0.6   # High ignition (top tier)
IGNITION_MED_THRESHOLD = 0.4    # Medium ignition


def calculate_pattern_ignition_score(
    pattern_row: pd.Series,
    ticker_df: Optional[pd.DataFrame] = None
) -> float:
    """
    Calculate ignition score for a single pattern.

    Uses the last day of the pattern to compute ignition signals:
    - Price position in channel (near upper boundary = higher)
    - Volume spike (>1.5x average = higher)
    - BBW expansion (positive slope = higher)
    - Green bar (close > open = higher)
    - ADX momentum (trending = higher)

    Args:
        pattern_row: Row from patterns DataFrame
        ticker_df: Optional ticker data for detailed calculation

    Returns:
        Ignition score [0, 1]
    """
    # Try to use pre-computed ignition features if available
    if 'ignition_score' in pattern_row.index and pd.notna(pattern_row.get('ignition_score')):
        return float(pattern_row['ignition_score'])

    # Compute from coil features if available
    score = 0.0
    weight_sum = 0.0

    # Price position - low position better for coil, but high position better for ignition
    if 'price_position_at_end' in pattern_row.index and pd.notna(pattern_row.get('price_position_at_end')):
        pos = pattern_row['price_position_at_end']
        # For ignition, we want position near top (imminent breakout)
        # But not too high (might be exhausted move)
        # Optimal: 0.6-0.85 range
        if pos >= 0.6 and pos <= 0.85:
            pos_score = 1.0
        elif pos > 0.85:
            pos_score = 0.7  # Slightly penalize very high position
        else:
            pos_score = pos / 0.6  # Linear ramp up to 0.6
        score += pos_score * 0.30
        weight_sum += 0.30

    # BBW slope - positive slope indicates volatility expansion (breakout imminent)
    if 'bbw_slope_5d' in pattern_row.index and pd.notna(pattern_row.get('bbw_slope_5d')):
        bbw_slope = pattern_row['bbw_slope_5d']
        # Positive slope = expanding volatility = potential breakout
        slope_score = np.clip((bbw_slope + 0.01) / 0.02, 0, 1)  # -0.01 to +0.01 maps to 0-1
        score += slope_score * 0.25
        weight_sum += 0.25

    # Volume trend - increasing volume suggests accumulation
    if 'vol_trend_5d' in pattern_row.index and pd.notna(pattern_row.get('vol_trend_5d')):
        vol_trend = pattern_row['vol_trend_5d']
        # Volume > 1.2x average is bullish
        vol_score = np.clip((vol_trend - 0.8) / 0.7, 0, 1)  # 0.8-1.5 maps to 0-1
        score += vol_score * 0.25
        weight_sum += 0.25

    # Coil intensity - tight coil is better setup
    if 'coil_intensity' in pattern_row.index and pd.notna(pattern_row.get('coil_intensity')):
        coil = pattern_row['coil_intensity']
        score += coil * 0.20
        weight_sum += 0.20

    # Normalize if we have partial features
    if weight_sum > 0:
        return score / weight_sum

    # Fallback: estimate from basic features
    return _estimate_ignition_from_basic(pattern_row)


def _estimate_ignition_from_basic(pattern_row: pd.Series) -> float:
    """Estimate ignition score from basic pattern features."""
    score = 0.5  # Default neutral

    # Box width - tighter is better
    if 'box_width' in pattern_row.index:
        width = pattern_row.get('box_width', 0)
        if width > 0:
            # Assuming typical widths 0.02-0.15, normalize
            width_score = 1.0 - np.clip(width / 0.15, 0, 1)
            score = 0.3 + 0.4 * width_score

    return score


def calculate_recency_score(
    pattern_end_date: datetime,
    reference_date: Optional[datetime] = None,
    decay_days: int = 90
) -> float:
    """
    Calculate recency score - more recent patterns score higher.

    Args:
        pattern_end_date: End date of the pattern
        reference_date: Reference date (default: today)
        decay_days: Days after which score decays to 0.5

    Returns:
        Recency score [0.5, 1.0] where 1.0 = today
    """
    if reference_date is None:
        reference_date = datetime.now()

    if isinstance(pattern_end_date, str):
        pattern_end_date = pd.to_datetime(pattern_end_date)
    if isinstance(reference_date, str):
        reference_date = pd.to_datetime(reference_date)

    days_ago = (reference_date - pattern_end_date).days

    if days_ago <= 0:
        return 1.0

    # Exponential decay with floor at 0.5
    decay_rate = 0.5 / decay_days
    score = 1.0 - (days_ago * decay_rate)
    return max(0.5, min(1.0, score))


def score_patterns_for_ignition(
    patterns_df: pd.DataFrame,
    ticker_data_cache: Optional[Dict[str, pd.DataFrame]] = None,
    reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Score all patterns for ignition potential.

    Adds columns:
    - ignition_score: Immediate breakout potential [0, 1]
    - recency_score: How recent the pattern is [0.5, 1]
    - priority_score: Composite score for prioritization [0, 1]

    Args:
        patterns_df: DataFrame with pattern data
        ticker_data_cache: Optional cache of ticker DataFrames
        reference_date: Reference date for recency calculation

    Returns:
        DataFrame with added score columns
    """
    logger.info("Scoring patterns for ignition potential...")
    patterns_df = patterns_df.copy()

    n_patterns = len(patterns_df)

    # Calculate ignition scores
    ignition_scores = []
    for idx, row in patterns_df.iterrows():
        ticker_df = ticker_data_cache.get(row['ticker']) if ticker_data_cache else None
        score = calculate_pattern_ignition_score(row, ticker_df)
        ignition_scores.append(score)

    patterns_df['ignition_score_computed'] = ignition_scores

    # Use pre-computed if available, else use computed
    if 'ignition_score' not in patterns_df.columns:
        patterns_df['ignition_score'] = patterns_df['ignition_score_computed']
    else:
        patterns_df['ignition_score'] = patterns_df['ignition_score'].fillna(
            patterns_df['ignition_score_computed']
        )

    # Calculate recency scores
    if 'end_date' in patterns_df.columns:
        date_col = 'end_date'
    elif 'pattern_end_date' in patterns_df.columns:
        date_col = 'pattern_end_date'
    else:
        date_col = None

    if date_col:
        patterns_df['recency_score'] = patterns_df[date_col].apply(
            lambda x: calculate_recency_score(x, reference_date)
        )
    else:
        patterns_df['recency_score'] = 0.75  # Default if no date

    # Ensure coil_intensity exists
    if 'coil_intensity' not in patterns_df.columns:
        patterns_df['coil_intensity'] = 0.5  # Default neutral

    # Calculate composite priority score
    patterns_df['priority_score'] = (
        PRIORITY_WEIGHTS['ignition_score'] * patterns_df['ignition_score'] +
        PRIORITY_WEIGHTS['coil_intensity'] * patterns_df['coil_intensity'].fillna(0.5) +
        PRIORITY_WEIGHTS['recency_score'] * patterns_df['recency_score']
    )

    # Log statistics
    high_ignition = (patterns_df['ignition_score'] >= IGNITION_HIGH_THRESHOLD).sum()
    med_ignition = ((patterns_df['ignition_score'] >= IGNITION_MED_THRESHOLD) &
                    (patterns_df['ignition_score'] < IGNITION_HIGH_THRESHOLD)).sum()

    logger.info(f"Ignition scoring complete:")
    logger.info(f"  Total patterns: {n_patterns}")
    logger.info(f"  High ignition (>={IGNITION_HIGH_THRESHOLD}): {high_ignition} ({100*high_ignition/n_patterns:.1f}%)")
    logger.info(f"  Medium ignition ({IGNITION_MED_THRESHOLD}-{IGNITION_HIGH_THRESHOLD}): {med_ignition} ({100*med_ignition/n_patterns:.1f}%)")
    logger.info(f"  Priority score range: [{patterns_df['priority_score'].min():.3f}, {patterns_df['priority_score'].max():.3f}]")

    return patterns_df


def prioritize_and_sort(patterns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort patterns by priority score (descending).

    Priority hierarchy:
    1. ignition_score (50%)
    2. coil_intensity (35%)
    3. recency_score (15%)

    Args:
        patterns_df: DataFrame with priority_score column

    Returns:
        Sorted DataFrame
    """
    if 'priority_score' not in patterns_df.columns:
        logger.warning("priority_score not found, sorting by ignition_score")
        sort_col = 'ignition_score' if 'ignition_score' in patterns_df.columns else None
        if sort_col is None:
            return patterns_df

    return patterns_df.sort_values('priority_score', ascending=False).reset_index(drop=True)


def oversample_high_ignition(
    patterns_df: pd.DataFrame,
    oversample_factor: float = 2.0,
    top_pct: float = 0.2
) -> pd.DataFrame:
    """
    Oversample high-ignition patterns for training optimization.

    Creates additional copies of top patterns to increase their weight
    in the training dataset.

    Args:
        patterns_df: DataFrame with ignition scores
        oversample_factor: How many times to duplicate top patterns (e.g., 2.0 = 2x)
        top_pct: Top percentage of patterns to oversample (e.g., 0.2 = top 20%)

    Returns:
        DataFrame with oversampled high-ignition patterns
    """
    if 'priority_score' not in patterns_df.columns:
        logger.warning("No priority_score column, skipping oversampling")
        return patterns_df

    n_patterns = len(patterns_df)
    top_n = int(n_patterns * top_pct)

    if top_n == 0:
        return patterns_df

    # Sort by priority
    sorted_df = patterns_df.sort_values('priority_score', ascending=False)

    # Get top patterns
    top_patterns = sorted_df.head(top_n)
    rest_patterns = sorted_df.tail(n_patterns - top_n)

    # Calculate how many copies to add
    n_copies = int(oversample_factor - 1)  # -1 because originals already exist

    if n_copies <= 0:
        return patterns_df

    # Create copies with marker
    copies_list = [top_patterns.copy() for _ in range(n_copies)]
    for i, copies in enumerate(copies_list):
        copies['oversample_copy'] = i + 1

    # Original patterns get copy marker 0
    top_patterns = top_patterns.copy()
    top_patterns['oversample_copy'] = 0
    rest_patterns = rest_patterns.copy()
    rest_patterns['oversample_copy'] = 0

    # Combine all
    result = pd.concat([top_patterns] + copies_list + [rest_patterns], ignore_index=True)

    logger.info(f"Oversampling complete:")
    logger.info(f"  Original patterns: {n_patterns}")
    logger.info(f"  Top {top_pct*100:.0f}% ({top_n} patterns) oversampled {oversample_factor}x")
    logger.info(f"  Final patterns: {len(result)}")

    return result


def stratified_oversample(
    patterns_df: pd.DataFrame,
    oversample_factor: float = 2.0,
    top_pct: float = 0.2,
    class_column: str = 'outcome_class',
    target_class: Optional[int] = 2
) -> pd.DataFrame:
    """
    Stratified oversampling that maintains class distribution.

    Only oversamples high-ignition patterns from each class proportionally,
    ensuring the class distribution remains balanced after oversampling.

    Args:
        patterns_df: DataFrame with ignition scores and class labels
        oversample_factor: Oversampling multiplier for top patterns
        top_pct: Top percentage within each class to oversample
        class_column: Column containing class labels
        target_class: If specified, only oversample this class (e.g., 2 for Target)

    Returns:
        DataFrame with stratified oversampling applied
    """
    if class_column not in patterns_df.columns:
        logger.warning(f"Class column '{class_column}' not found, falling back to non-stratified")
        return oversample_high_ignition(patterns_df, oversample_factor, top_pct)

    if 'priority_score' not in patterns_df.columns:
        logger.warning("No priority_score column, skipping oversampling")
        return patterns_df

    # Get original class distribution
    class_dist = patterns_df[class_column].value_counts(normalize=True)
    logger.info(f"Original class distribution:")
    for cls, pct in class_dist.items():
        logger.info(f"  Class {cls}: {pct*100:.1f}%")

    result_parts = []

    for cls in patterns_df[class_column].unique():
        cls_df = patterns_df[patterns_df[class_column] == cls].copy()
        n_cls = len(cls_df)

        # Sort by priority within class
        cls_df = cls_df.sort_values('priority_score', ascending=False)

        # Determine oversampling for this class
        if target_class is not None and cls != target_class:
            # Don't oversample non-target classes
            cls_df['oversample_copy'] = 0
            result_parts.append(cls_df)
            continue

        # Top patterns in this class
        top_n = max(1, int(n_cls * top_pct))
        top_patterns = cls_df.head(top_n)
        rest_patterns = cls_df.tail(n_cls - top_n)

        # Create copies
        n_copies = int(oversample_factor - 1)

        if n_copies > 0:
            copies_list = [top_patterns.copy() for _ in range(n_copies)]
            for i, copies in enumerate(copies_list):
                copies['oversample_copy'] = i + 1

            top_patterns = top_patterns.copy()
            top_patterns['oversample_copy'] = 0
            rest_patterns = rest_patterns.copy()
            rest_patterns['oversample_copy'] = 0

            result_parts.append(top_patterns)
            result_parts.extend(copies_list)
            result_parts.append(rest_patterns)
        else:
            cls_df['oversample_copy'] = 0
            result_parts.append(cls_df)

    result = pd.concat(result_parts, ignore_index=True)

    # Log final distribution
    final_dist = result[class_column].value_counts(normalize=True)
    logger.info(f"Final class distribution after stratified oversampling:")
    for cls, pct in final_dist.items():
        orig_pct = class_dist.get(cls, 0) * 100
        logger.info(f"  Class {cls}: {pct*100:.1f}% (was {orig_pct:.1f}%)")

    logger.info(f"Stratified oversampling complete: {len(patterns_df)} -> {len(result)} patterns")

    return result


def prioritize_and_oversample(
    patterns_df: pd.DataFrame,
    oversample_factor: float = 2.0,
    top_pct: float = 0.2,
    stratified: bool = True,
    class_column: str = 'outcome_class',
    target_class: Optional[int] = 2,
    ticker_data_cache: Optional[Dict] = None,
    reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Complete ignition-aware prioritization pipeline.

    1. Score patterns for ignition potential
    2. Sort by priority score
    3. Oversample high-priority patterns
    4. Optionally use stratified sampling for class balance

    Args:
        patterns_df: Input patterns DataFrame
        oversample_factor: Multiplier for top patterns
        top_pct: Percentage of patterns to oversample
        stratified: Use stratified sampling to maintain class balance
        class_column: Column containing class labels
        target_class: Only oversample this class if stratified
        ticker_data_cache: Cache of ticker DataFrames
        reference_date: Reference date for recency scoring

    Returns:
        Prioritized and oversampled DataFrame
    """
    logger.info("=" * 60)
    logger.info("IGNITION-AWARE PATTERN PRIORITIZATION")
    logger.info("=" * 60)

    # Step 1: Score patterns
    patterns_df = score_patterns_for_ignition(
        patterns_df,
        ticker_data_cache,
        reference_date
    )

    # Step 2: Sort by priority
    patterns_df = prioritize_and_sort(patterns_df)

    # Step 3: Oversample
    if stratified and class_column in patterns_df.columns:
        patterns_df = stratified_oversample(
            patterns_df,
            oversample_factor,
            top_pct,
            class_column,
            target_class
        )
    else:
        patterns_df = oversample_high_ignition(
            patterns_df,
            oversample_factor,
            top_pct
        )

    logger.info("=" * 60)

    return patterns_df


def get_ignition_statistics(patterns_df: pd.DataFrame) -> Dict:
    """
    Get statistics about ignition scoring.

    Args:
        patterns_df: DataFrame with ignition scores

    Returns:
        Dictionary with statistics
    """
    stats = {
        'n_patterns': len(patterns_df),
        'has_ignition_scores': 'ignition_score' in patterns_df.columns,
    }

    if 'ignition_score' in patterns_df.columns:
        scores = patterns_df['ignition_score']
        stats.update({
            'ignition_mean': float(scores.mean()),
            'ignition_std': float(scores.std()),
            'ignition_median': float(scores.median()),
            'n_high_ignition': int((scores >= IGNITION_HIGH_THRESHOLD).sum()),
            'n_med_ignition': int(((scores >= IGNITION_MED_THRESHOLD) &
                                   (scores < IGNITION_HIGH_THRESHOLD)).sum()),
        })

    if 'priority_score' in patterns_df.columns:
        stats['priority_mean'] = float(patterns_df['priority_score'].mean())
        stats['priority_std'] = float(patterns_df['priority_score'].std())

    if 'outcome_class' in patterns_df.columns:
        class_dist = patterns_df['outcome_class'].value_counts().to_dict()
        stats['class_distribution'] = {int(k): int(v) for k, v in class_dist.items()}

    return stats
