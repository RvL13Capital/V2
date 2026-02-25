"""
Snapshot-Forward Labeling with Race Logic (K5 vs Best Positive)
================================================================

Key Logic: It's a RACE between K5 and positive outcomes
- If K4/K3/K2/K1 reached BEFORE K5 → That's the label (K5 later doesn't change it)
- If K5 happens FIRST → Label is K5
- Among positives, track the BEST achieved (K3 can become K4)

Each snapshot labeled based on 100 days AFTER snapshot_date.
Gains calculated from UPPER_BOUNDARY (not snapshot price).
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


def calculate_snapshot_race_label(
    snapshot_date: pd.Timestamp,
    upper_boundary: float,
    lower_boundary: float,
    ticker_data: pd.DataFrame,
    evaluation_days: int = EVALUATION_DAYS
) -> Optional[Dict]:
    """
    Calculate K0-K5 label using RACE logic: K5 vs Best Positive Outcome.

    Key: Whichever happens first between K5 and positive outcomes wins.
    Among positives, track the best achieved (K3 can upgrade to K4).

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

        # Track best positive outcome achieved
        best_positive_outcome = None
        best_positive_day = None

        # Track if K5 happened
        k5_occurred = False
        k5_day = None

        # Track maximums for reporting
        max_gain_from_upper = 0.0
        max_loss_from_lower = 0.0
        day_of_max_gain = None
        day_of_max_loss = None

        # Process each day - it's a RACE!
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

            # RACE LOGIC: Check for K5 first (terminal event)
            if loss_from_lower <= -GRACE_PERIOD_PCT and not k5_occurred:
                k5_occurred = True
                k5_day = day_idx

                # K5 happened - did we achieve something positive before it?
                if best_positive_outcome:
                    # Yes! The positive outcome wins the race
                    # Continue scanning to track max values but outcome is decided
                    pass
                else:
                    # No positive outcome before K5 - this is K5_FAILED
                    # Set as potential outcome but keep scanning for max values
                    pass

            # Track best positive outcome (only K4/K3 can prevent K5)
            if not k5_occurred or (k5_occurred and best_positive_outcome in ['K4_EXCEPTIONAL', 'K3_STRONG']):
                # Either K5 hasn't happened, or we already had K4/K3 before K5
                if gain_from_upper >= K4_THRESHOLD:
                    if not best_positive_outcome or best_positive_outcome != 'K4_EXCEPTIONAL':
                        best_positive_outcome = 'K4_EXCEPTIONAL'
                        best_positive_day = day_idx
                elif gain_from_upper >= K3_THRESHOLD:
                    if best_positive_outcome not in ['K4_EXCEPTIONAL', 'K3_STRONG']:
                        best_positive_outcome = 'K3_STRONG'
                        best_positive_day = day_idx
                elif gain_from_upper >= K2_THRESHOLD:
                    # Track K2 but it won't prevent K5
                    if best_positive_outcome not in ['K4_EXCEPTIONAL', 'K3_STRONG', 'K2_QUALITY']:
                        best_positive_outcome = 'K2_QUALITY'
                        best_positive_day = day_idx
                elif gain_from_upper >= K1_THRESHOLD:
                    # Track K1 but it won't prevent K5
                    if best_positive_outcome not in ['K4_EXCEPTIONAL', 'K3_STRONG', 'K2_QUALITY', 'K1_MINIMAL']:
                        best_positive_outcome = 'K1_MINIMAL'
                        best_positive_day = day_idx

        # Determine final classification
        if k5_occurred and not best_positive_outcome:
            # K5 happened first, no positive outcome before it
            outcome_class = 'K5_FAILED'
            breach_day = k5_day
        elif best_positive_outcome:
            # Positive outcome achieved (either before K5 or no K5)
            outcome_class = best_positive_outcome
            breach_day = best_positive_day
        else:
            # No K5, no significant positive outcome
            outcome_class = 'K0_STAGNANT'
            breach_day = 0

        # Calculate additional metrics
        final_price = window['close'].iloc[-1]
        final_return_from_snapshot = ((final_price - snapshot_price) / snapshot_price) * 100
        final_position_vs_upper = ((final_price - upper_boundary) / upper_boundary) * 100

        return {
            # Classification
            'outcome_class': outcome_class,
            'breach_day': breach_day,
            'k5_occurred': k5_occurred,
            'k5_day': k5_day,
            'best_positive_before_k5': best_positive_outcome if k5_occurred else None,

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


def label_snapshots_race(input_file):
    """
    Main function to label pattern snapshots with race-based K0-K5 classification.

    It's a race: K5 vs Best Positive Outcome. Whichever happens first wins.
    """
    logger.info("="*70)
    logger.info("SNAPSHOT LABELING - RACE LOGIC (K5 vs Best Positive)")
    logger.info("="*70)
    logger.info(f"Evaluation: {EVALUATION_DAYS} days after each snapshot")
    logger.info(f"Gains calculated from: upper_boundary")
    logger.info(f"K5 grace period: {GRACE_PERIOD_PCT}% below lower_boundary")
    logger.info("Logic: If K4/K3 achieved before K5, that's the label")
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
    race_wins = {'positive_won': 0, 'k5_won': 0}

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
            label_info = calculate_snapshot_race_label(
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
                    if label_info['best_positive_before_k5']:
                        race_wins['positive_won'] += 1
                    else:
                        race_wins['k5_won'] += 1
                elif 'K4' in label_info['outcome_class']:
                    class_counts['K4'] += 1
                    if label_info['k5_occurred']:
                        race_wins['positive_won'] += 1
                elif 'K3' in label_info['outcome_class']:
                    class_counts['K3'] += 1
                    if label_info['k5_occurred']:
                        race_wins['positive_won'] += 1
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
    output_file = OUTPUT_DIR / f'snapshots_labeled_race_{timestamp}.parquet'
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
        logger.info("Race Statistics (K5 vs Positive):")
        logger.info(f"  Cases where K5 occurred: {result_df['k5_occurred'].sum()}")
        logger.info(f"  - Positive won race: {race_wins['positive_won']}")
        logger.info(f"  - K5 won race: {race_wins['k5_won']}")

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
    labeled_df = label_snapshots_race(latest_file)