"""
First-Breach Pattern Labeling with Grace Period - AIv4 (CORRECTED)

Implements CORRECT temporal classification logic:
- Uses first-breach logic: whichever threshold is breached FIRST determines classification
- Grace period: K5 only triggered if price falls ≥10% below lower_boundary
- Gains calculated from upper_boundary (not snapshot price)
- Losses calculated from lower_boundary (not snapshot price)
- Efficient rolling window optimization (loads ticker data once per pattern)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.parquet_helper import read_data, write_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = Path('data/raw/gcs_cache')  # Updated path after refactoring
EVALUATION_DAYS = 100
OUTPUT_DIR = Path('output')
GRACE_PERIOD_PCT = 10  # K5 triggered only if ≥10% below lower_boundary


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


def calculate_snapshot_metrics_firstbreach(window_data, upper_boundary, lower_boundary):
    """
    Calculate metrics using CORRECT first-breach logic with grace period.

    Returns dict with classification, metrics, and breach details.
    """
    if len(window_data) == 0:
        return None

    # Initialize tracking variables
    first_breach_class = None
    first_breach_day = None

    # Track all metrics for analysis
    max_gain_from_upper = 0.0
    max_loss_from_lower = 0.0
    day_of_max_gain = None
    day_of_max_loss = None

    # Track boundary behavior
    went_below_boundary = False
    days_to_below_boundary = None
    consecutive_below = 0
    max_consecutive_below = 0
    reached_anxious = False

    # Track which thresholds were reached (for analysis)
    reached_k5_threshold = False
    reached_k4_threshold = False
    reached_k3_threshold = False
    reached_k2_threshold = False
    reached_k1_threshold = False

    # Process each day in the evaluation window
    for day_idx, (date, row) in enumerate(window_data.iterrows()):
        # Calculate gain from upper_boundary
        gain_from_upper = ((row['high'] - upper_boundary) / upper_boundary) * 100

        # Calculate loss from lower_boundary (will be negative)
        loss_from_lower = ((row['low'] - lower_boundary) / lower_boundary) * 100

        # Update max metrics
        if gain_from_upper > max_gain_from_upper:
            max_gain_from_upper = gain_from_upper
            day_of_max_gain = day_idx

        if loss_from_lower < max_loss_from_lower:
            max_loss_from_lower = loss_from_lower
            day_of_max_loss = day_idx

        # Track below boundary behavior
        is_below = row['close'] < lower_boundary
        if is_below:
            if not went_below_boundary:
                went_below_boundary = True
                days_to_below_boundary = day_idx
            consecutive_below += 1
            max_consecutive_below = max(max_consecutive_below, consecutive_below)
        else:
            consecutive_below = 0

        # Check for ANXIOUS state (≥7 consecutive days below, but <10% deep)
        if consecutive_below >= 7 and not reached_anxious:
            # Check depth for current streak
            start_idx = max(0, day_idx - consecutive_below + 1)
            streak_data = window_data.iloc[start_idx:day_idx + 1]
            depths = ((streak_data['low'] - lower_boundary) / lower_boundary) * 100
            if all(depths > -GRACE_PERIOD_PCT):  # All less than 10% deep
                reached_anxious = True

        # FIRST-BREACH LOGIC: Check thresholds in order of severity
        # Only classify if not already classified (first breach wins)

        if first_breach_class is None:
            # Check K5 (with grace period): ≥10% below lower_boundary
            if loss_from_lower <= -GRACE_PERIOD_PCT:
                first_breach_class = 'K5_FAILED'
                first_breach_day = day_idx
                reached_k5_threshold = True
                # K5 locks in immediately - no need to check further days
                break

            # Check K4: ≥75% gain from upper_boundary
            elif gain_from_upper >= 75:
                first_breach_class = 'K4_EXCEPTIONAL'
                first_breach_day = day_idx
                reached_k4_threshold = True
                # K4 locks in immediately
                break

            # Check K3: ≥35% gain from upper_boundary
            elif gain_from_upper >= 35:
                first_breach_class = 'K3_STRONG'
                first_breach_day = day_idx
                reached_k3_threshold = True
                # K3 locks in immediately
                break

    # If no major breach occurred during evaluation, classify by best outcome
    if first_breach_class is None:
        # Use maximum gain achieved to classify
        if max_gain_from_upper >= 15:
            first_breach_class = 'K2_QUALITY'
            reached_k2_threshold = True
        elif max_gain_from_upper >= 5:
            first_breach_class = 'K1_MINIMAL'
            reached_k1_threshold = True
        else:
            first_breach_class = 'K0_STAGNANT'

        # For non-breach classifications, use day of max gain
        first_breach_day = day_of_max_gain if day_of_max_gain is not None else 0

    return {
        # Classification
        'outcome_class': first_breach_class,
        'first_breach_day': first_breach_day,

        # Metrics
        'max_gain_from_upper_pct': max_gain_from_upper,
        'max_loss_from_lower_pct': max_loss_from_lower,
        'days_to_max_gain': day_of_max_gain,
        'days_to_max_loss': day_of_max_loss,

        # Boundary behavior
        'went_below_boundary': went_below_boundary,
        'days_to_below_boundary': days_to_below_boundary,
        'consecutive_days_below_max': max_consecutive_below,
        'reached_anxious': reached_anxious,

        # Threshold tracking (for analysis)
        'reached_k5_threshold': reached_k5_threshold,
        'reached_k4_threshold': reached_k4_threshold,
        'reached_k3_threshold': reached_k3_threshold,
        'reached_k2_threshold': reached_k2_threshold,
        'reached_k1_threshold': reached_k1_threshold,

        # Grace period info
        'grace_period_applied': loss_from_lower > -GRACE_PERIOD_PCT and loss_from_lower < 0
    }


def process_pattern_snapshots(ticker, activation_date, pattern_snapshots, ticker_data):
    """
    Process all snapshots for a single pattern using first-breach logic.

    Returns list of labeled snapshots with correct temporal classification.
    """
    results = []

    # Get pattern boundaries (same for all snapshots)
    upper_boundary = pattern_snapshots.iloc[0]['upper_boundary']
    lower_boundary = pattern_snapshots.iloc[0]['lower_boundary']

    # Calculate channel width
    channel_width_pct = ((upper_boundary - lower_boundary) / lower_boundary) * 100

    # Process each snapshot
    for idx, snapshot in pattern_snapshots.iterrows():
        snapshot_date = pd.to_datetime(snapshot['snapshot_date'])

        # Find snapshot date in ticker data
        if snapshot_date not in ticker_data.index:
            future_dates = ticker_data.index[ticker_data.index >= snapshot_date]
            if len(future_dates) == 0:
                continue
            snapshot_date = future_dates[0]

        # Get 100-day window from snapshot
        snapshot_idx = ticker_data.index.get_loc(snapshot_date)

        # Handle case where get_loc returns boolean array (duplicate dates)
        if isinstance(snapshot_idx, np.ndarray):
            snapshot_idx = np.where(snapshot_idx)[0][0]  # Get first occurrence

        window_data = ticker_data.iloc[snapshot_idx:snapshot_idx + EVALUATION_DAYS]

        if len(window_data) < 10:  # Need sufficient data
            continue

        # Calculate metrics with first-breach logic
        metrics = calculate_snapshot_metrics_firstbreach(window_data, upper_boundary, lower_boundary)

        if metrics is None:
            continue

        # Add pattern-level metrics
        metrics['consolidation_channel_width_pct'] = channel_width_pct

        # Calculate consolidation period
        activation_dt = pd.to_datetime(activation_date)
        consolidation_period = (snapshot_date - activation_dt).days
        metrics['consolidation_period_days'] = consolidation_period

        # Build result (snapshot data + metrics)
        result = snapshot.to_dict()
        result.update(metrics)

        results.append(result)

    return results


def label_snapshots_firstbreach(snapshots_file, output_file, output_format='parquet'):
    """
    Label snapshots using CORRECT first-breach logic with grace period.

    This is the main entry point for relabeling the dataset.
    """
    logger.info(f"\n{'='*70}")
    logger.info("FIRST-BREACH PATTERN LABELING WITH GRACE PERIOD - AIv4")
    logger.info(f"{'='*70}\n")

    logger.info("Classification Logic:")
    logger.info("  - First breach determines classification (temporal priority)")
    logger.info(f"  - K5 grace period: Only triggered if ≥{GRACE_PERIOD_PCT}% below lower_boundary")
    logger.info("  - Gains calculated from upper_boundary")
    logger.info("  - Losses calculated from lower_boundary")
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
    ticker_cache = {}  # Cache loaded ticker data

    logger.info(f"\nProcessing patterns with rolling window optimization...")
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

        # Process all snapshots for this pattern
        pattern_results = process_pattern_snapshots(
            ticker, activation_date, pattern_snapshots, ticker_data
        )

        all_results.extend(pattern_results)

    # Convert to DataFrame
    logger.info(f"\n{'='*50}")
    logger.info(f"Patterns processed: {patterns_processed:,}")
    logger.info(f"Patterns failed (no data): {patterns_failed:,}")
    logger.info(f"Total labeled snapshots: {len(all_results):,}")

    if len(all_results) == 0:
        logger.error("ERROR: No snapshots were successfully labeled!")
        return

    results_df = pd.DataFrame(all_results)

    # Calculate outcome distribution
    logger.info(f"\nOutcome Distribution (First-Breach Logic):")
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

    logger.info(f"\nKey Metrics:")
    logger.info("-" * 40)
    logger.info(f"  Win Rate (K3+K4): {win_rate:.1f}%")
    logger.info(f"  Failure Rate (K5): {failure_rate:.1f}%")
    logger.info(f"  Grace Period Applied: {results_df['grace_period_applied'].sum():,} patterns")
    logger.info(f"  ANXIOUS State Reached: {results_df['reached_anxious'].sum():,} patterns")

    # Save results
    logger.info(f"\nSaving labeled data to: {output_file}")
    write_data(results_df, output_file, output_format)
    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    logger.info(f"\n{'='*70}")
    logger.info("LABELING COMPLETE - First-Breach Logic Applied Successfully!")
    logger.info(f"{'='*70}")

    return results_df


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Label patterns with first-breach logic')
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
        output_file = OUTPUT_DIR / f'patterns_labeled_firstbreach_{timestamp}.{args.format}'
    else:
        output_file = Path(args.output)

    # Run labeling
    label_snapshots_firstbreach(args.input, output_file, args.format)