"""
Correct Snapshot-Forward Labeling for Temporal Integrity

CRITICAL FIX: Each snapshot is labeled based on what happens in the
100 days AFTER that specific snapshot, not based on the pattern's
final outcome.

This ensures:
1. No data leakage (snapshot only knows its past)
2. Realistic training (early snapshots may have different outcomes than late ones)
3. Temporal validation integrity
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Tuple

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.parquet_helper import read_data, write_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = Path('data/raw/gcs_cache')
EVALUATION_DAYS = 100  # Look forward 100 days from each snapshot
GRACE_PERIOD_PCT = 10  # K5 grace period


def load_ticker_data(ticker, cache_dir=CACHE_DIR):
    """Load cached ticker data."""
    # Clean ticker name
    if ' (' in ticker:
        ticker = ticker.split(' (')[0]

    # Try Parquet first
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

    # Remove duplicates
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='first')]

    return df


def calculate_forward_outcome(future_window, upper_boundary, lower_boundary):
    """
    Calculate outcome based on FUTURE data from a specific snapshot.

    This is the CORRECTED version - each snapshot looks at its OWN future,
    not the pattern's final outcome.

    Args:
        future_window: 100 days of data AFTER the snapshot
        upper_boundary: Upper consolidation boundary at snapshot time
        lower_boundary: Lower consolidation boundary at snapshot time

    Returns:
        Dict with outcome class and metrics
    """
    if len(future_window) == 0:
        return None

    # Initialize tracking
    first_breach_class = None
    first_breach_day = None
    max_gain_from_upper = 0.0
    max_loss_from_lower = 0.0
    day_of_max_gain = None
    day_of_max_loss = None

    # Process each day in the future window
    for day_idx, (date, row) in enumerate(future_window.iterrows()):
        # Calculate metrics from boundaries
        gain_from_upper = ((row['high'] - upper_boundary) / upper_boundary) * 100
        loss_from_lower = ((row['low'] - lower_boundary) / lower_boundary) * 100

        # Track max metrics
        if gain_from_upper > max_gain_from_upper:
            max_gain_from_upper = gain_from_upper
            day_of_max_gain = day_idx

        if loss_from_lower < max_loss_from_lower:
            max_loss_from_lower = loss_from_lower
            day_of_max_loss = day_idx

        # First-breach logic (only check if not already classified)
        if first_breach_class is None:
            # Check K5 (with grace period)
            if loss_from_lower <= -GRACE_PERIOD_PCT:
                first_breach_class = 'K5_FAILED'
                first_breach_day = day_idx
                break  # K5 locks in immediately

            # Check K4: ≥75% gain
            elif gain_from_upper >= 75:
                first_breach_class = 'K4_EXCEPTIONAL'
                first_breach_day = day_idx
                break  # K4 locks in immediately

            # Check K3: ≥35% gain
            elif gain_from_upper >= 35:
                first_breach_class = 'K3_STRONG'
                first_breach_day = day_idx
                break  # K3 locks in immediately

    # If no major breach occurred, classify by best outcome
    if first_breach_class is None:
        if max_gain_from_upper >= 15:
            first_breach_class = 'K2_QUALITY'
        elif max_gain_from_upper >= 5:
            first_breach_class = 'K1_MINIMAL'
        else:
            first_breach_class = 'K0_STAGNANT'

        first_breach_day = day_of_max_gain if day_of_max_gain is not None else 0

    return {
        'forward_outcome_class': first_breach_class,
        'forward_breach_day': first_breach_day,
        'forward_max_gain_pct': max_gain_from_upper,
        'forward_max_loss_pct': max_loss_from_lower,
        'forward_days_to_gain': day_of_max_gain,
        'forward_days_to_loss': day_of_max_loss,
        'forward_window_days': len(future_window)
    }


def label_snapshot_forward(snapshot_row, ticker_data):
    """
    Label a single snapshot based on its own future (not pattern's outcome).

    CRITICAL: This is the key fix - we look forward from the snapshot date,
    not from the pattern end date.
    """
    snapshot_date = pd.to_datetime(snapshot_row['snapshot_date'])

    # Get boundaries AT SNAPSHOT TIME
    # Use refined boundaries if available, else original
    if 'refined_upper_boundary' in snapshot_row and pd.notna(snapshot_row['refined_upper_boundary']):
        upper = snapshot_row['refined_upper_boundary']
        lower = snapshot_row['refined_lower_boundary']
    else:
        upper = snapshot_row['upper_boundary']
        lower = snapshot_row['lower_boundary']

    # Find snapshot date in ticker data
    if snapshot_date not in ticker_data.index:
        future_dates = ticker_data.index[ticker_data.index >= snapshot_date]
        if len(future_dates) == 0:
            return None
        snapshot_date = future_dates[0]

    # Get 100-day window FROM THIS SNAPSHOT
    snapshot_idx = ticker_data.index.get_loc(snapshot_date)
    if isinstance(snapshot_idx, np.ndarray):
        snapshot_idx = np.where(snapshot_idx)[0][0]

    future_window = ticker_data.iloc[snapshot_idx:snapshot_idx + EVALUATION_DAYS]

    if len(future_window) < 10:  # Need sufficient data
        return None

    # Calculate outcome based on THIS snapshot's future
    outcome = calculate_forward_outcome(future_window, upper, lower)

    if outcome is None:
        return None

    # Add snapshot info
    result = {
        'ticker': snapshot_row['ticker'],
        'snapshot_date': snapshot_row['snapshot_date'],
        'activation_date': snapshot_row['activation_date'],
        'days_since_activation': snapshot_row.get('days_since_activation', 0)
    }

    # Add all original features (except old labels)
    for col in snapshot_row.index:
        if col not in result and not col.startswith('outcome_') and not col.startswith('max_'):
            result[col] = snapshot_row[col]

    # Add forward-looking labels
    result.update(outcome)

    return result


def process_snapshots_forward(snapshots_df, output_file):
    """
    Process all snapshots with forward-looking labels.

    This is the main entry point for correct snapshot labeling.
    """
    logger.info("\n" + "="*80)
    logger.info("FORWARD-LOOKING SNAPSHOT LABELING (TEMPORAL INTEGRITY FIX)")
    logger.info("="*80)

    logger.info("\nKey Fix: Each snapshot labeled by its OWN next 100 days")
    logger.info("- Early snapshots may be K0 while late snapshots are K4")
    logger.info("- No data leakage - snapshot only knows its past")
    logger.info("- Realistic for temporal validation\n")

    # Group by ticker for efficiency
    grouped = snapshots_df.groupby('ticker')
    n_tickers = len(grouped)

    all_results = []
    ticker_cache = {}

    logger.info(f"Processing {len(snapshots_df):,} snapshots from {n_tickers} tickers...")
    logger.info("-" * 50)

    for ticker_idx, (ticker, ticker_snapshots) in enumerate(grouped):
        if ticker_idx % 10 == 0:
            logger.info(f"  Progress: {ticker_idx}/{n_tickers} tickers ({ticker_idx/n_tickers*100:.1f}%)")

        # Load ticker data (with caching)
        if ticker not in ticker_cache:
            ticker_data = load_ticker_data(ticker)
            if ticker_data is not None:
                ticker_cache[ticker] = ticker_data
        else:
            ticker_data = ticker_cache[ticker]

        if ticker_data is None:
            continue

        # Process each snapshot individually
        for idx, snapshot in ticker_snapshots.iterrows():
            result = label_snapshot_forward(snapshot, ticker_data)
            if result is not None:
                all_results.append(result)

    # Convert to DataFrame
    logger.info(f"\n{'='*50}")
    logger.info(f"Snapshots processed: {len(snapshots_df):,}")
    logger.info(f"Snapshots labeled: {len(all_results):,}")

    if len(all_results) == 0:
        logger.error("ERROR: No snapshots were successfully labeled!")
        return None

    results_df = pd.DataFrame(all_results)

    # Show forward outcome distribution
    logger.info(f"\nForward Outcome Distribution (Each Snapshot's Own Future):")
    logger.info("-" * 50)
    outcome_counts = results_df['forward_outcome_class'].value_counts()
    for outcome, count in outcome_counts.items():
        pct = (count / len(results_df)) * 100
        logger.info(f"  {outcome:15s}: {count:7,} ({pct:5.1f}%)")

    # Compare to old labels if available
    if 'outcome_class' in results_df.columns:
        logger.info("\nLabel Changes (Old → New):")
        changed = results_df[results_df['outcome_class'] != results_df['forward_outcome_class']]
        logger.info(f"  Snapshots with different labels: {len(changed):,} ({len(changed)/len(results_df)*100:.1f}%)")

        # Show some examples
        if len(changed) > 0:
            logger.info("\n  Examples of label changes:")
            for i, row in changed.head(5).iterrows():
                days = row.get('days_since_activation', 'N/A')
                logger.info(f"    Day {days}: {row['outcome_class']} → {row['forward_outcome_class']}")

    # Save results
    logger.info(f"\nSaving forward-labeled data to: {output_file}")
    write_data(results_df, output_file)

    file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    # Key metrics
    k4_count = (results_df['forward_outcome_class'] == 'K4_EXCEPTIONAL').sum()
    k3_count = (results_df['forward_outcome_class'] == 'K3_STRONG').sum()
    win_rate = (k3_count + k4_count) / len(results_df) * 100

    logger.info(f"\nKey Metrics (Forward-Looking):")
    logger.info(f"  K4_EXCEPTIONAL: {k4_count:,} snapshots")
    logger.info(f"  K3_STRONG: {k3_count:,} snapshots")
    logger.info(f"  Win Rate (K3+K4): {win_rate:.2f}%")

    logger.info("\n" + "="*80)
    logger.info("FORWARD-LOOKING LABELING COMPLETE - TEMPORAL INTEGRITY RESTORED")
    logger.info("="*80)

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Label snapshots with forward-looking outcomes (temporal integrity fix)'
    )
    parser.add_argument('--input', type=str, required=True,
                      help='Input snapshots file (enhanced or original)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output file path')
    parser.add_argument('--format', type=str, default='parquet',
                      choices=['parquet', 'csv'],
                      help='Output format')

    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'snapshots_forward_labeled_{timestamp}.{args.format}'
    else:
        output_file = Path(args.output)

    # Load input snapshots
    logger.info(f"Loading snapshots from: {args.input}")
    snapshots_df = read_data(args.input)
    logger.info(f"Loaded {len(snapshots_df):,} snapshots")

    # Process with forward-looking labels
    process_snapshots_forward(snapshots_df, output_file)