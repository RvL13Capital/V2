"""
Snapshot-Forward Labeling with First-Breach K0-K5 Classification
=================================================================

Key Changes:
1. Each SNAPSHOT labeled based on 100 days AFTER snapshot_date (not pattern start)
2. Gains calculated from UPPER_BOUNDARY (not snapshot price)
3. First-breach logic: first threshold crossed determines label
4. Grace period: K5 requires 10%+ below lower_boundary

This means different snapshots from the same pattern can have different labels!
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = Path('data_acquisition/storage')
EVALUATION_DAYS = 100
OUTPUT_DIR = Path('output')
GRACE_PERIOD_PCT = 10  # K5 triggered only if ≥10% below lower_boundary

# K0-K5 thresholds (gains from upper_boundary)
K4_THRESHOLD = 75  # ≥75% above upper
K3_THRESHOLD = 35  # ≥35% above upper
K2_THRESHOLD = 15  # ≥15% above upper
K1_THRESHOLD = 5   # ≥5% above upper
# K0: <5% above upper
# K5: ≥10% below lower


def load_ticker_data(ticker, cache_dir=CACHE_DIR):
    """Load cached ticker data from Parquet."""
    # Clean ticker name
    if ' (' in ticker:
        ticker = ticker.split(' (')[0]

    ticker_file = cache_dir / f"{ticker}.parquet"
    if not ticker_file.exists():
        return None

    df = pd.read_parquet(ticker_file)

    # Normalize columns
    df.columns = df.columns.str.lower()
    if 'adj close' in df.columns:
        df = df.rename(columns={'adj close': 'adj_close'})

    # Ensure date index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    # Remove duplicate dates
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep='first')]

    return df


def calculate_snapshot_forward_label(
    snapshot_date: pd.Timestamp,
    upper_boundary: float,
    lower_boundary: float,
    ticker_data: pd.DataFrame,
    evaluation_days: int = EVALUATION_DAYS
) -> Optional[Dict]:
    """
    Calculate K0-K5 label for a snapshot based on 100 days AFTER snapshot_date.

    Key: Gains calculated from upper_boundary, not snapshot price.

    Args:
        snapshot_date: The date of this specific snapshot
        upper_boundary: Pattern's upper boundary (breakout level)
        lower_boundary: Pattern's lower boundary (breakdown level)
        ticker_data: Full price data for the ticker
        evaluation_days: Days to look forward from snapshot_date

    Returns:
        Dict with label and metrics
    """
    try:
        # Find snapshot date in data
        if snapshot_date not in ticker_data.index:
            nearest_idx = ticker_data.index.searchsorted(snapshot_date)
            if nearest_idx >= len(ticker_data):
                return None
            snapshot_date = ticker_data.index[nearest_idx]

        # Get evaluation window (100 days AFTER this snapshot)
        start_idx = ticker_data.index.get_loc(snapshot_date)
        end_idx = min(start_idx + evaluation_days, len(ticker_data) - 1)

        if end_idx <= start_idx:
            return None  # Not enough forward data

        window = ticker_data.iloc[start_idx:end_idx+1]

        # Track metrics
        snapshot_price = ticker_data.loc[snapshot_date, 'close']
        first_breach_class = None
        first_breach_day = None
        max_gain_from_upper = 0.0
        max_loss_from_lower = 0.0
        day_of_max_gain = None
        day_of_max_loss = None

        # Process each day in evaluation window
        for day_idx, (date, row) in enumerate(window.iterrows()):
            # Calculate gain from UPPER boundary (breakout level)
            gain_from_upper = ((row['high'] - upper_boundary) / upper_boundary) * 100

            # Calculate loss from LOWER boundary (breakdown level)
            loss_from_lower = ((row['low'] - lower_boundary) / lower_boundary) * 100

            # Track maximums
            if gain_from_upper > max_gain_from_upper:
                max_gain_from_upper = gain_from_upper
                day_of_max_gain = day_idx

            if loss_from_lower < max_loss_from_lower:
                max_loss_from_lower = loss_from_lower
                day_of_max_loss = day_idx

            # FIRST-BREACH LOGIC
            if first_breach_class is None:
                # Check K5 (with grace period): ≥10% below lower_boundary
                if loss_from_lower <= -GRACE_PERIOD_PCT:
                    first_breach_class = 'K5_FAILED'
                    first_breach_day = day_idx
                    break  # K5 locks in immediately

                # Check K4: ≥75% gain from upper_boundary
                elif gain_from_upper >= K4_THRESHOLD:
                    first_breach_class = 'K4_EXCEPTIONAL'
                    first_breach_day = day_idx
                    break  # K4 locks in immediately

                # Check K3: ≥35% gain from upper_boundary
                elif gain_from_upper >= K3_THRESHOLD:
                    first_breach_class = 'K3_STRONG'
                    first_breach_day = day_idx
                    break  # K3 locks in immediately

        # If no major breach, classify by best outcome
        if first_breach_class is None:
            if max_gain_from_upper >= K2_THRESHOLD:
                first_breach_class = 'K2_QUALITY'
            elif max_gain_from_upper >= K1_THRESHOLD:
                first_breach_class = 'K1_MINIMAL'
            else:
                first_breach_class = 'K0_STAGNANT'

            first_breach_day = day_of_max_gain if day_of_max_gain is not None else 0

        # Calculate additional metrics
        final_price = window['close'].iloc[-1]
        final_return_from_snapshot = ((final_price - snapshot_price) / snapshot_price) * 100
        final_position_vs_upper = ((final_price - upper_boundary) / upper_boundary) * 100

        return {
            # Classification
            'outcome_class': first_breach_class,
            'first_breach_day': first_breach_day,

            # Core metrics (from boundaries)
            'max_gain_from_upper_pct': max_gain_from_upper,
            'max_loss_from_lower_pct': max_loss_from_lower,
            'days_to_max_gain': day_of_max_gain,
            'days_to_max_loss': day_of_max_loss,

            # Snapshot-specific metrics
            'snapshot_price': snapshot_price,
            'final_return_from_snapshot_pct': final_return_from_snapshot,
            'final_position_vs_upper_pct': final_position_vs_upper,

            # Evaluation info
            'evaluation_days_actual': end_idx - start_idx,
            'window_start_date': window.index[0],
            'window_end_date': window.index[-1]
        }

    except Exception as e:
        logger.warning(f"Error labeling snapshot at {snapshot_date}: {e}")
        return None


def label_snapshots_forward(input_file):
    """
    Main function to label pattern snapshots with forward-looking K0-K5 classification.

    Each snapshot evaluated based on 100 days after its own date, not pattern start.
    """
    logger.info("="*70)
    logger.info("SNAPSHOT-FORWARD LABELING (K0-K5)")
    logger.info("="*70)
    logger.info(f"Evaluation: {EVALUATION_DAYS} days after each snapshot")
    logger.info(f"Gains calculated from: upper_boundary")
    logger.info(f"K5 grace period: {GRACE_PERIOD_PCT}% below lower_boundary")
    logger.info("")

    # Load patterns
    if Path(input_file).suffix == '.parquet':
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)

    logger.info(f"Loaded {len(df)} snapshots")

    # Check required columns
    required_cols = ['ticker', 'snapshot_date', 'upper_boundary', 'lower_boundary']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return None

    # Process by ticker for efficiency
    ticker_groups = df.groupby('ticker')
    total_tickers = len(ticker_groups)

    labeled_snapshots = []
    class_counts = {f'K{i}': 0 for i in range(6)}
    class_counts['K5'] = 0  # Rename for clarity
    skipped_count = 0

    for idx, (ticker, group) in enumerate(ticker_groups, 1):
        if idx % 10 == 0:
            k_summary = ' | '.join([f"{k}: {v}" for k, v in class_counts.items()])
            logger.info(f"Progress: {idx}/{total_tickers} tickers | {k_summary}")

        # Load ticker data once
        ticker_data = load_ticker_data(ticker)
        if ticker_data is None:
            logger.debug(f"No data for {ticker}")
            skipped_count += len(group)
            continue

        # Label each snapshot independently
        for _, snapshot in group.iterrows():
            snapshot_date = pd.to_datetime(snapshot['snapshot_date'])
            upper_boundary = snapshot['upper_boundary']
            lower_boundary = snapshot['lower_boundary']

            # Get label based on 100 days after THIS snapshot
            label_info = calculate_snapshot_forward_label(
                snapshot_date,
                upper_boundary,
                lower_boundary,
                ticker_data,
                EVALUATION_DAYS
            )

            if label_info:
                # Add label to snapshot
                labeled_snapshot = snapshot.to_dict()
                labeled_snapshot.update(label_info)
                labeled_snapshots.append(labeled_snapshot)

                # Update counts
                if 'K5' in label_info['outcome_class']:
                    class_counts['K5'] += 1
                elif 'K4' in label_info['outcome_class']:
                    class_counts['K4'] += 1
                elif 'K3' in label_info['outcome_class']:
                    class_counts['K3'] += 1
                elif 'K2' in label_info['outcome_class']:
                    class_counts['K2'] += 1
                elif 'K1' in label_info['outcome_class']:
                    class_counts['K1'] += 1
                else:  # K0
                    class_counts['K0'] += 1
            else:
                skipped_count += 1

    # Create output DataFrame
    result_df = pd.DataFrame(labeled_snapshots)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f'snapshots_labeled_forward_{timestamp}.parquet'
    result_df.to_parquet(output_file, index=False)

    # Display summary
    logger.info("")
    logger.info("="*70)
    logger.info("LABELING COMPLETE")
    logger.info("="*70)
    logger.info(f"Total snapshots: {len(df)}")
    logger.info(f"Labeled snapshots: {len(result_df)}")
    logger.info(f"Skipped (no data): {skipped_count}")
    logger.info("")

    if len(result_df) > 0:
        logger.info("Class Distribution:")
        total_labeled = len(result_df)
        for class_name in ['K5', 'K4', 'K3', 'K2', 'K1', 'K0']:
            count = class_counts[class_name]
            pct = count / total_labeled * 100 if total_labeled > 0 else 0
            logger.info(f"  {class_name}: {count:7d} ({pct:5.2f}%)")

        logger.info("")
        logger.info("Performance Metrics:")
        logger.info(f"  Avg max gain from upper: {result_df['max_gain_from_upper_pct'].mean():+6.1f}%")
        logger.info(f"  Avg max loss from lower: {result_df['max_loss_from_lower_pct'].mean():+6.1f}%")

        # Metrics by class
        logger.info("")
        logger.info("Average Max Gain by Class:")
        for class_name in ['K4_EXCEPTIONAL', 'K3_STRONG', 'K2_QUALITY', 'K1_MINIMAL', 'K0_STAGNANT', 'K5_FAILED']:
            class_df = result_df[result_df['outcome_class'] == class_name]
            if len(class_df) > 0:
                avg_gain = class_df['max_gain_from_upper_pct'].mean()
                logger.info(f"  {class_name:15s}: {avg_gain:+6.1f}%")

    logger.info("")
    logger.info(f"Output saved to: {output_file}")

    # Additional analysis - show pattern evolution
    if not result_df.empty and 'start_date' in result_df.columns:
        logger.info("")
        logger.info("Pattern Evolution Analysis:")

        # Group by pattern (same start_date and ticker)
        pattern_groups = result_df.groupby(['ticker', 'start_date'])

        evolution_examples = []
        for (ticker, start_date), pattern in pattern_groups:
            if len(pattern) >= 3:  # Patterns with multiple snapshots
                snapshots = pattern.sort_values('snapshot_date')
                classes = snapshots['outcome_class'].values
                # Check if classification changes over time
                if len(set(classes)) > 1:
                    evolution_examples.append({
                        'ticker': ticker,
                        'start_date': start_date,
                        'snapshot_count': len(snapshots),
                        'classes': ' → '.join(classes[:5])  # First 5 snapshots
                    })

        if evolution_examples:
            logger.info("Examples of patterns with changing labels:")
            for ex in evolution_examples[:5]:  # Show first 5
                logger.info(f"  {ex['ticker']}: {ex['classes']}")

    return result_df


if __name__ == "__main__":
    # Find most recent pattern file
    pattern_files = list(OUTPUT_DIR.glob("patterns_enhanced_*.csv"))
    if not pattern_files:
        logger.error("No pattern files found. Run 01_scan_patterns.py first.")
        sys.exit(1)

    latest_file = max(pattern_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using pattern file: {latest_file}")

    # Run labeling
    labeled_df = label_snapshots_forward(latest_file)