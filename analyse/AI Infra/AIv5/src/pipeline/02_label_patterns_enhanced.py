"""
Enhanced First-Breach Pattern Labeling with 15-Day Filter and Boundary Refinement

Key enhancements:
1. Only includes patterns that respected boundaries for 15+ active days
2. On day 15, creates new tighter boundaries based on days 5-15 trading range
3. New boundaries must be inside original boundaries (narrower consolidation)
4. Uses first-breach logic with grace period on the refined boundaries
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Tuple

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.parquet_helper import read_data, write_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = Path('data/raw/gcs_cache')
EVALUATION_DAYS = 100
OUTPUT_DIR = Path('output')
GRACE_PERIOD_PCT = 10  # K5 triggered only if ≥10% below lower_boundary
MIN_ACTIVE_DAYS_BEFORE_REFINEMENT = 15  # Must survive 15 days in active phase
REFINEMENT_WINDOW_START = 5  # Use days 5-15 for new boundaries
REFINEMENT_WINDOW_END = 15


def load_ticker_data(ticker, cache_dir=CACHE_DIR):
    """Load cached ticker data from Parquet (faster) or CSV."""
    # Clean ticker name (remove suffixes like "(1)")
    if ' (' in ticker:
        ticker = ticker.split(' (')[0]

    # Try Parquet first (5x faster than CSV)
    parquet_dir = Path('data_acquisition/storage')
    ticker_file = parquet_dir / f"{ticker}.parquet"
    if ticker_file.exists():
        df = pd.read_parquet(ticker_file)
    else:
        # Fall back to CSV
        ticker_file = cache_dir / f"{ticker}.csv"
        if ticker_file.exists():
            df = pd.read_csv(ticker_file)
        else:
            return None

    # Normalize columns
    df.columns = df.columns.str.lower()
    if 'adj close' in df.columns:
        df = df.rename(columns={'adj close': 'adj_close'})

    # Ensure date index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    # Remove duplicate dates (keep first occurrence)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='first')]

    return df


def check_boundary_respect(window_data, upper_boundary, lower_boundary, num_days=15):
    """
    Check if pattern respected boundaries for specified number of days.
    UPDATED: Changed from 20-day to 15-day filter for better K3/K4 retention
    Returns (respected, days_survived)
    """
    if len(window_data) < num_days:
        return False, len(window_data)

    for day_idx in range(num_days):
        if day_idx >= len(window_data):
            break

        row = window_data.iloc[day_idx]

        # Check if price breached boundaries
        if row['high'] > upper_boundary * 1.001:  # Allow 0.1% tolerance
            return False, day_idx

        # Use grace period for lower boundary
        if row['low'] < lower_boundary * (1 - GRACE_PERIOD_PCT/100):
            return False, day_idx

    return True, num_days


def calculate_refined_boundaries(window_data, original_upper, original_lower,
                                start_day=5, end_day=15):
    """
    Calculate refined boundaries based on trading range of days 5-15.
    New boundaries must be inside original boundaries.
    """
    if len(window_data) < end_day:
        return original_upper, original_lower, False

    # Get the window for refinement (days 5-15)
    refinement_window = window_data.iloc[start_day:end_day]

    # Calculate new boundaries from the window
    new_upper = refinement_window['high'].max()
    new_lower = refinement_window['low'].min()

    # Ensure new boundaries are inside original boundaries (narrower)
    # Upper boundary should be lower than or equal to original
    new_upper = min(new_upper, original_upper)

    # Lower boundary should be higher than or equal to original
    new_lower = max(new_lower, original_lower)

    # Validate that we have a valid channel (upper > lower)
    if new_upper <= new_lower:
        return original_upper, original_lower, False

    # Check if refinement actually narrowed the channel
    original_width = original_upper - original_lower
    new_width = new_upper - new_lower

    if new_width >= original_width:
        # No meaningful refinement
        return original_upper, original_lower, False

    # Calculate narrowing percentage
    narrowing_pct = ((original_width - new_width) / original_width) * 100

    logger.debug(f"  Refined boundaries: {new_lower:.4f} - {new_upper:.4f} (narrowed {narrowing_pct:.1f}%)")

    return new_upper, new_lower, True


def calculate_snapshot_metrics_enhanced(window_data, upper_boundary, lower_boundary,
                                       activation_date, snapshot_date):
    """
    Enhanced metrics calculation with:
    1. 15-day survival filter
    2. Boundary refinement after day 15
    3. First-breach logic on refined boundaries
    """
    if len(window_data) == 0:
        return None

    # First check if pattern survived 15 days without breaching
    respected_15_days, days_survived = check_boundary_respect(
        window_data, upper_boundary, lower_boundary, MIN_ACTIVE_DAYS_BEFORE_REFINEMENT
    )

    if not respected_15_days:
        # Pattern failed before 15 days - exclude it
        return {
            'excluded': True,
            'exclusion_reason': f'breached_boundaries_day_{days_survived}',
            'days_survived': days_survived
        }

    # Pattern survived 15 days - calculate refined boundaries
    refined_upper, refined_lower, refinement_applied = calculate_refined_boundaries(
        window_data, upper_boundary, lower_boundary,
        REFINEMENT_WINDOW_START, REFINEMENT_WINDOW_END
    )

    # Use refined boundaries for classification
    eval_upper = refined_upper if refinement_applied else upper_boundary
    eval_lower = refined_lower if refinement_applied else lower_boundary

    # Now apply first-breach logic on the evaluation window AFTER day 15
    eval_start_day = MIN_ACTIVE_DAYS_BEFORE_REFINEMENT
    eval_window = window_data.iloc[eval_start_day:eval_start_day + EVALUATION_DAYS]

    if len(eval_window) < 10:
        return {
            'excluded': True,
            'exclusion_reason': 'insufficient_data_after_refinement',
            'days_survived': MIN_ACTIVE_DAYS_BEFORE_REFINEMENT
        }

    # Initialize tracking variables
    first_breach_class = None
    first_breach_day = None
    max_gain_from_upper = 0.0
    max_loss_from_lower = 0.0
    day_of_max_gain = None
    day_of_max_loss = None
    reached_anxious = False
    consecutive_below = 0
    max_consecutive_below = 0

    # Process each day in evaluation window
    for day_idx, (date, row) in enumerate(eval_window.iterrows()):
        # Calculate gain from refined upper boundary
        gain_from_upper = ((row['high'] - eval_upper) / eval_upper) * 100

        # Calculate loss from refined lower boundary
        loss_from_lower = ((row['low'] - eval_lower) / eval_lower) * 100

        # Update max metrics
        if gain_from_upper > max_gain_from_upper:
            max_gain_from_upper = gain_from_upper
            day_of_max_gain = day_idx

        if loss_from_lower < max_loss_from_lower:
            max_loss_from_lower = loss_from_lower
            day_of_max_loss = day_idx

        # Track consecutive days below boundary
        is_below = row['close'] < eval_lower
        if is_below:
            consecutive_below += 1
            max_consecutive_below = max(max_consecutive_below, consecutive_below)
        else:
            consecutive_below = 0

        # Check for ANXIOUS state
        if consecutive_below >= 7 and not reached_anxious:
            if loss_from_lower > -GRACE_PERIOD_PCT:
                reached_anxious = True

        # FIRST-BREACH LOGIC with refined boundaries
        if first_breach_class is None:
            # Check K5 (with grace period from refined lower boundary)
            if loss_from_lower <= -GRACE_PERIOD_PCT:
                first_breach_class = 'K5_FAILED'
                first_breach_day = eval_start_day + day_idx
                break

            # Check K4: ≥75% gain from refined upper boundary
            elif gain_from_upper >= 75:
                first_breach_class = 'K4_EXCEPTIONAL'
                first_breach_day = eval_start_day + day_idx
                break

            # Check K3: ≥35% gain from refined upper boundary
            elif gain_from_upper >= 35:
                first_breach_class = 'K3_STRONG'
                first_breach_day = eval_start_day + day_idx
                break

    # If no major breach occurred, classify by best outcome
    if first_breach_class is None:
        if max_gain_from_upper >= 15:
            first_breach_class = 'K2_QUALITY'
        elif max_gain_from_upper >= 5:
            first_breach_class = 'K1_MINIMAL'
        else:
            first_breach_class = 'K0_STAGNANT'

        first_breach_day = eval_start_day + (day_of_max_gain if day_of_max_gain else 0)

    # Calculate channel widths
    original_channel_width = ((upper_boundary - lower_boundary) / lower_boundary) * 100
    refined_channel_width = ((eval_upper - eval_lower) / eval_lower) * 100

    return {
        # Classification
        'outcome_class': first_breach_class,
        'first_breach_day': first_breach_day,

        # Filtering
        'excluded': False,
        'passed_15_day_filter': True,
        'days_survived': days_survived,

        # Refinement
        'refinement_applied': refinement_applied,
        'original_upper_boundary': upper_boundary,
        'original_lower_boundary': lower_boundary,
        'refined_upper_boundary': eval_upper,
        'refined_lower_boundary': eval_lower,
        'original_channel_width_pct': original_channel_width,
        'refined_channel_width_pct': refined_channel_width,
        'channel_narrowing_pct': ((original_channel_width - refined_channel_width) / original_channel_width * 100) if refinement_applied else 0,

        # Metrics (from refined boundaries)
        'max_gain_from_upper_pct': max_gain_from_upper,
        'max_loss_from_lower_pct': max_loss_from_lower,
        'days_to_max_gain': day_of_max_gain,
        'days_to_max_loss': day_of_max_loss,

        # Behavior
        'consecutive_days_below_max': max_consecutive_below,
        'reached_anxious': reached_anxious,
    }


def process_pattern_snapshots_enhanced(ticker, activation_date, pattern_snapshots, ticker_data):
    """
    Process snapshots with enhanced filtering and boundary refinement.
    """
    results = []
    excluded_count = 0

    # Get pattern boundaries (same for all snapshots initially)
    upper_boundary = pattern_snapshots.iloc[0]['upper_boundary']
    lower_boundary = pattern_snapshots.iloc[0]['lower_boundary']

    # Process each snapshot
    for idx, snapshot in pattern_snapshots.iterrows():
        snapshot_date = pd.to_datetime(snapshot['snapshot_date'])

        # Find snapshot date in ticker data
        if snapshot_date not in ticker_data.index:
            future_dates = ticker_data.index[ticker_data.index >= snapshot_date]
            if len(future_dates) == 0:
                continue
            snapshot_date = future_dates[0]

        # Get full evaluation window (20 days + 100 days)
        snapshot_idx = ticker_data.index.get_loc(snapshot_date)
        if isinstance(snapshot_idx, np.ndarray):
            snapshot_idx = np.where(snapshot_idx)[0][0]

        # Need at least 120 days of data (20 for filter + 100 for evaluation)
        window_data = ticker_data.iloc[snapshot_idx:snapshot_idx + MIN_ACTIVE_DAYS_BEFORE_REFINEMENT + EVALUATION_DAYS]

        if len(window_data) < MIN_ACTIVE_DAYS_BEFORE_REFINEMENT + 10:
            continue

        # Calculate enhanced metrics
        metrics = calculate_snapshot_metrics_enhanced(
            window_data, upper_boundary, lower_boundary,
            activation_date, snapshot_date
        )

        if metrics is None:
            continue

        # Check if pattern was excluded
        if metrics.get('excluded', False):
            excluded_count += 1
            continue  # Skip excluded patterns

        # Calculate consolidation period
        activation_dt = pd.to_datetime(activation_date)
        consolidation_period = (snapshot_date - activation_dt).days
        metrics['consolidation_period_days'] = consolidation_period

        # Build result
        result = snapshot.to_dict()
        result.update(metrics)

        results.append(result)

    if excluded_count > 0:
        logger.debug(f"  Excluded {excluded_count} snapshots that breached before day {MIN_ACTIVE_DAYS_BEFORE_REFINEMENT}")

    return results


def label_snapshots_enhanced(snapshots_file, output_file, output_format='parquet'):
    """
    Enhanced labeling with 15-day filter and boundary refinement.
    """
    logger.info(f"\n{'='*70}")
    logger.info("ENHANCED PATTERN LABELING - AIv4")
    logger.info("With 15-Day Filter and Boundary Refinement")
    logger.info(f"{'='*70}\n")

    logger.info("Enhancement Features:")
    logger.info(f"  ✓ Only patterns that respect boundaries for {MIN_ACTIVE_DAYS_BEFORE_REFINEMENT}+ days")
    logger.info(f"  ✓ Boundary refinement on day {MIN_ACTIVE_DAYS_BEFORE_REFINEMENT} using days {REFINEMENT_WINDOW_START}-{REFINEMENT_WINDOW_END}")
    logger.info(f"  ✓ Refined boundaries must be narrower (inside original)")
    logger.info(f"  ✓ First-breach logic with {GRACE_PERIOD_PCT}% grace period")
    logger.info("")

    # Load snapshots
    logger.info(f"Loading snapshots from: {snapshots_file}")
    snapshots_df = read_data(snapshots_file)
    logger.info(f"Loaded {len(snapshots_df):,} snapshots")

    # Verify required columns
    required_cols = ['ticker', 'activation_date', 'snapshot_date', 'upper_boundary', 'lower_boundary']
    missing = [col for col in required_cols if col not in snapshots_df.columns]
    if missing:
        logger.error(f"ERROR: Missing columns: {missing}")
        return

    # Group by pattern
    logger.info(f"\nGrouping snapshots by pattern...")
    grouped = snapshots_df.groupby(['ticker', 'activation_date'])
    unique_patterns = len(grouped)
    logger.info(f"Found {unique_patterns:,} unique patterns")

    # Process each pattern
    all_results = []
    patterns_processed = 0
    patterns_failed = 0
    patterns_excluded = 0
    ticker_cache = {}

    logger.info(f"\nProcessing patterns with enhanced filtering...")
    logger.info("="*50)

    for (ticker, activation_date), pattern_snapshots in grouped:
        patterns_processed += 1

        if patterns_processed % 100 == 0:
            pct_complete = (patterns_processed / unique_patterns) * 100
            logger.info(f"  Progress: {patterns_processed:,}/{unique_patterns:,} patterns ({pct_complete:.1f}%)")

        # Load ticker data (with caching)
        if ticker not in ticker_cache:
            ticker_data = load_ticker_data(ticker)
            if ticker_data is not None:
                ticker_cache[ticker] = ticker_data
        else:
            ticker_data = ticker_cache[ticker]

        if ticker_data is None:
            patterns_failed += 1
            continue

        # Process with enhanced logic
        pattern_results = process_pattern_snapshots_enhanced(
            ticker, activation_date, pattern_snapshots, ticker_data
        )

        if len(pattern_results) == 0:
            patterns_excluded += 1

        all_results.extend(pattern_results)

    # Convert to DataFrame
    logger.info(f"\n{'='*50}")
    logger.info(f"Patterns processed: {patterns_processed:,}")
    logger.info(f"Patterns failed (no data): {patterns_failed:,}")
    logger.info(f"Patterns excluded (breached < 15 days): {patterns_excluded:,}")
    logger.info(f"Total labeled snapshots: {len(all_results):,}")

    if len(all_results) == 0:
        logger.error("ERROR: No snapshots passed the 15-day filter!")
        return

    results_df = pd.DataFrame(all_results)

    # Calculate outcome distribution
    logger.info(f"\nOutcome Distribution (Enhanced):")
    logger.info("-" * 40)
    outcome_counts = results_df['outcome_class'].value_counts()
    for outcome, count in outcome_counts.items():
        pct = (count / len(results_df)) * 100
        logger.info(f"  {outcome:15s}: {count:7,} ({pct:5.1f}%)")

    # Calculate key metrics
    k4_k3_count = len(results_df[results_df['outcome_class'].isin(['K4_EXCEPTIONAL', 'K3_STRONG'])])
    k5_count = len(results_df[results_df['outcome_class'] == 'K5_FAILED'])
    win_rate = (k4_k3_count / len(results_df)) * 100
    failure_rate = (k5_count / len(results_df)) * 100

    # Refinement statistics
    refinement_applied = results_df['refinement_applied'].sum()
    refinement_rate = (refinement_applied / len(results_df)) * 100
    avg_narrowing = results_df[results_df['refinement_applied']]['channel_narrowing_pct'].mean() if refinement_applied > 0 else 0

    logger.info(f"\nKey Metrics:")
    logger.info("-" * 40)
    logger.info(f"  Win Rate (K3+K4): {win_rate:.1f}%")
    logger.info(f"  Failure Rate (K5): {failure_rate:.1f}%")
    logger.info(f"  Survived 20+ days: {len(results_df):,} patterns (100% by design)")
    logger.info(f"  Refinement applied: {refinement_applied:,} ({refinement_rate:.1f}%)")
    if refinement_applied > 0:
        logger.info(f"  Avg channel narrowing: {avg_narrowing:.1f}%")
    logger.info(f"  ANXIOUS state reached: {results_df['reached_anxious'].sum():,} patterns")

    # Quality metrics
    logger.info(f"\nQuality Improvements:")
    logger.info("-" * 40)
    original_patterns = len(snapshots_df)
    filtered_patterns = len(results_df)
    reduction_pct = ((original_patterns - filtered_patterns) / original_patterns) * 100
    logger.info(f"  Original patterns: {original_patterns:,}")
    logger.info(f"  After 15-day filter: {filtered_patterns:,}")
    logger.info(f"  Reduction: {reduction_pct:.1f}% (higher quality)")

    # Save results
    logger.info(f"\nSaving enhanced labeled data to: {output_file}")
    write_data(results_df, output_file, output_format)
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    logger.info(f"\n{'='*70}")
    logger.info("ENHANCED LABELING COMPLETE!")
    logger.info("High-quality patterns with refined boundaries")
    logger.info(f"{'='*70}")

    return results_df


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced pattern labeling with quality filters')
    parser.add_argument('--input', type=str, default='data/processed/patterns_final.parquet',
                      help='Input snapshots file')
    parser.add_argument('--output', type=str, default=None,
                      help='Output file path')
    parser.add_argument('--format', type=str, default='parquet',
                      choices=['parquet', 'csv'],
                      help='Output format')

    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = OUTPUT_DIR / f'patterns_labeled_enhanced_{timestamp}.{args.format}'
    else:
        output_file = Path(args.output)

    # Run enhanced labeling
    label_snapshots_enhanced(args.input, output_file, args.format)