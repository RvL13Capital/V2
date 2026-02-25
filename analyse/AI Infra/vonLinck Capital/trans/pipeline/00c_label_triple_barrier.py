"""
Step 0c: Label Pattern Outcomes using Triple Barrier Method
============================================================

Implements Marcos Lopez de Prado's Triple Barrier Method for labeling.
Alternative to the structural risk labeling (00b_label_outcomes.py).

TRIPLE BARRIER FRAMEWORK (Lopez de Prado, 2018):
    1. Upper Barrier: Profit-taking (return >= +threshold)
    2. Lower Barrier: Stop-loss (return <= -threshold)
    3. Vertical Barrier: Time limit (150 days max holding)

KEY DIFFERENCES FROM STRUCTURAL RISK LABELING:
    - Uses RELATIVE returns (percentage change) instead of absolute prices
    - Barriers can be volatility-scaled (adapt to market conditions)
    - 150-day labeling window (vs 10-60 dynamic or 40 fixed)
    - 250-day feature window with strict temporal integrity
    - All rolling calculations use closed='left'

OUTCOME CLASSES (150-day window):
    - Class 2 (TARGET): Upper barrier hit first
    - Class 0 (DANGER): Lower barrier hit first
    - Class 1 (NOISE): Vertical barrier (timeout, no clear outcome)

TEMPORAL INTEGRITY:
    - All features calculated using ONLY data before pattern end date
    - Rolling windows use closed='left' to prevent look-ahead bias
    - Strict train/test splits with no temporal overlap

Usage:
    python 00c_label_triple_barrier.py --input output/candidate_patterns.parquet
    python 00c_label_triple_barrier.py --input output/candidate_patterns.parquet --volatility-scaling
    python 00c_label_triple_barrier.py --input output/candidate_patterns.parquet --upper-barrier 0.05 --lower-barrier 0.03
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.aiv7_components import DataLoader
from utils.logging_config import setup_pipeline_logging
from utils.triple_barrier_labeler import (
    TripleBarrierLabeler,
    create_strict_temporal_split,
    validate_temporal_integrity,
    FEATURE_WINDOW_SIZE,
    VERTICAL_BARRIER_DAYS,
    DEFAULT_UPPER_BARRIER,
    DEFAULT_LOWER_BARRIER,
)
from config import (
    DEFAULT_OUTPUT_DIR,
    INDICATOR_WARMUP_DAYS,
)
from config.constants import (
    TRIPLE_BARRIER_FEATURE_WINDOW,
    TRIPLE_BARRIER_VERTICAL_DAYS,
    TRIPLE_BARRIER_UPPER_PCT,
    TRIPLE_BARRIER_LOWER_PCT,
    TRIPLE_BARRIER_VOL_SCALING,
    TRIPLE_BARRIER_VOL_MULTIPLIER,
    TRIPLE_BARRIER_VOL_LOOKBACK,
    TRIPLE_BARRIER_MIN_DATA_DAYS,
)

# Setup centralized logging
logger = setup_pipeline_logging('00c_label_triple_barrier')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Label pattern outcomes using Triple Barrier Method (Lopez de Prado)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Triple Barrier Framework (Lopez de Prado, 2018):
    Upper Barrier: Return >= +threshold (profit-taking)
    Lower Barrier: Return <= -threshold (stop-loss)
    Vertical Barrier: Time limit expires (timeout)

Outcome Classes (150-day window):
    Class 2 (TARGET): Upper barrier hit first
    Class 0 (DANGER): Lower barrier hit first
    Class 1 (NOISE): Vertical barrier (no clear outcome)

Examples:
    python 00c_label_triple_barrier.py --input output/candidate_patterns.parquet
    python 00c_label_triple_barrier.py --input output/candidate_patterns.parquet --volatility-scaling
    python 00c_label_triple_barrier.py --input output/candidate_patterns.parquet --upper-barrier 0.05 --lower-barrier 0.03
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to candidate_patterns.parquet'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for labeled patterns (default: labeled_patterns_triple_barrier.parquet)'
    )
    parser.add_argument(
        '--reference-date',
        type=str,
        default=None,
        help='Reference date for ripeness check (default: today). Format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--vertical-barrier-days',
        type=int,
        default=TRIPLE_BARRIER_VERTICAL_DAYS,
        help=f'Maximum holding period in days (default: {TRIPLE_BARRIER_VERTICAL_DAYS})'
    )
    parser.add_argument(
        '--upper-barrier',
        type=float,
        default=TRIPLE_BARRIER_UPPER_PCT,
        help=f'Upper barrier threshold as decimal (default: {TRIPLE_BARRIER_UPPER_PCT} = {TRIPLE_BARRIER_UPPER_PCT*100}%%)'
    )
    parser.add_argument(
        '--lower-barrier',
        type=float,
        default=TRIPLE_BARRIER_LOWER_PCT,
        help=f'Lower barrier threshold as decimal (default: {TRIPLE_BARRIER_LOWER_PCT} = {TRIPLE_BARRIER_LOWER_PCT*100}%%)'
    )
    parser.add_argument(
        '--volatility-scaling',
        action='store_true',
        default=TRIPLE_BARRIER_VOL_SCALING,
        help='Scale barriers by realized volatility (default: %(default)s)'
    )
    parser.add_argument(
        '--no-volatility-scaling',
        action='store_true',
        help='Disable volatility scaling (use fixed barriers)'
    )
    parser.add_argument(
        '--volatility-multiplier',
        type=float,
        default=TRIPLE_BARRIER_VOL_MULTIPLIER,
        help=f'Volatility multiplier for scaled barriers (default: {TRIPLE_BARRIER_VOL_MULTIPLIER})'
    )
    parser.add_argument(
        '--feature-window',
        type=int,
        default=TRIPLE_BARRIER_FEATURE_WINDOW,
        help=f'Feature calculation window in days (default: {TRIPLE_BARRIER_FEATURE_WINDOW})'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be labeled without writing output'
    )
    parser.add_argument(
        '--force-relabel',
        action='store_true',
        help='Re-label patterns that already have outcome_class'
    )
    parser.add_argument(
        '--calculate-features',
        action='store_true',
        default=True,
        help='Calculate historic features for each pattern (default: True)'
    )
    parser.add_argument(
        '--no-features',
        action='store_true',
        help='Skip feature calculation (faster labeling)'
    )
    parser.add_argument(
        '--create-splits',
        action='store_true',
        help='Create strict temporal train/val/test splits'
    )
    parser.add_argument(
        '--train-end',
        type=str,
        default=None,
        help='Train split end date (YYYY-MM-DD). Required if --create-splits'
    )
    parser.add_argument(
        '--val-end',
        type=str,
        default=None,
        help='Validation split end date (YYYY-MM-DD). Required if --create-splits'
    )
    parser.add_argument(
        '--gap-days',
        type=int,
        default=0,
        help='Gap days between splits to prevent look-ahead (default: 0)'
    )

    # Dynamic barrier arguments
    parser.add_argument(
        '--dynamic-barriers',
        action='store_true',
        default=False,
        help='Use dynamic barriers based on market cap tier and market regime. '
             'Micro-caps get wider stops (+4%) and higher targets (+8%). '
             'Bullish regime extends targets, bearish widens stops.'
    )
    parser.add_argument(
        '--market-cap-col',
        type=str,
        default=None,
        help='Column name for market cap at pattern time (for dynamic barriers)'
    )
    parser.add_argument(
        '--vix-col',
        type=str,
        default=None,
        help='Column name for VIX at pattern time (for dynamic regime detection)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Step 0c: Triple Barrier Labeling (Lopez de Prado Method)")
    logger.info("=" * 60)

    # Handle volatility scaling flag
    volatility_scaling = args.volatility_scaling
    if args.no_volatility_scaling:
        volatility_scaling = False

    # Handle feature calculation flag
    calculate_features = args.calculate_features
    if args.no_features:
        calculate_features = False

    # Parse reference date
    if args.reference_date:
        reference_date = pd.to_datetime(args.reference_date)
        logger.info(f"Reference date: {reference_date.date()} (from argument)")
    else:
        reference_date = pd.Timestamp.now()
        logger.info(f"Reference date: {reference_date.date()} (today)")

    # Load candidate patterns
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    logger.info(f"Loading candidates from: {input_path}")
    df_candidates = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df_candidates):,} candidate patterns")

    # Filter for patterns that need labeling
    if args.force_relabel:
        patterns_to_label = df_candidates
        logger.warning("Force relabeling ALL patterns (--force-relabel)")
    else:
        if 'outcome_class' in df_candidates.columns:
            mask_unlabeled = df_candidates['outcome_class'].isna()
            patterns_to_label = df_candidates[mask_unlabeled]
            logger.info(f"Found {len(patterns_to_label):,} unlabeled patterns")
        else:
            patterns_to_label = df_candidates
            logger.info(f"No outcome_class column - labeling all {len(patterns_to_label):,} patterns")

    if len(patterns_to_label) == 0:
        logger.info("No patterns to label!")
        return 0

    if args.dry_run:
        logger.info("\n[DRY RUN] Would label:")
        logger.info(f"  - Total patterns: {len(patterns_to_label):,}")
        logger.info(f"  - Tickers: {patterns_to_label['ticker'].nunique()} unique")
        logger.info(f"  - Vertical barrier: {args.vertical_barrier_days} days")
        logger.info(f"  - Upper barrier: +{args.upper_barrier*100:.1f}%")
        logger.info(f"  - Lower barrier: -{args.lower_barrier*100:.1f}%")
        logger.info(f"  - Volatility scaling: {volatility_scaling}")
        return 0

    # Log labeling parameters
    logger.info("\n" + "=" * 50)
    logger.info("TRIPLE BARRIER LABELING PARAMETERS (V24)")
    logger.info("=" * 50)
    logger.info(f"Vertical Barrier: {args.vertical_barrier_days} days")

    if args.dynamic_barriers:
        logger.info(f"DYNAMIC BARRIERS: Enabled")
        logger.info(f"  - Barriers adjust per market cap tier and regime")
        logger.info(f"  - Micro-caps: +8% target, -4% stop (base)")
        logger.info(f"  - Small-caps: +5% target, -2.5% stop (base)")
        logger.info(f"  - Bullish regime: extends targets, tightens stops")
        logger.info(f"  - Bearish regime: reduces targets, widens stops")
        if args.market_cap_col:
            logger.info(f"  - Market cap column: {args.market_cap_col}")
        else:
            logger.info(f"  - Market cap: ADV-based classification")
        if args.vix_col:
            logger.info(f"  - VIX column: {args.vix_col}")
        else:
            logger.info(f"  - Regime: Trend-based detection")
    else:
        logger.info(f"Upper Barrier: +{args.upper_barrier*100:.1f}%")
        logger.info(f"Lower Barrier: -{args.lower_barrier*100:.1f}%")

    logger.info(f"Volatility Scaling: {volatility_scaling}")
    if volatility_scaling:
        logger.info(f"  Volatility Multiplier: {args.volatility_multiplier}x")
        logger.info(f"  Volatility Lookback: {TRIPLE_BARRIER_VOL_LOOKBACK} days")
    logger.info(f"Feature Window: {args.feature_window} days")
    logger.info(f"Calculate Features: {calculate_features}")
    logger.info("-" * 50)
    logger.info("TEMPORAL INTEGRITY:")
    logger.info("  All rolling windows use closed='left'")
    logger.info("  Features computed BEFORE pattern end date only")
    logger.info("=" * 50 + "\n")

    # Initialize Triple Barrier Labeler
    labeler = TripleBarrierLabeler(
        vertical_barrier_days=args.vertical_barrier_days,
        upper_barrier_pct=args.upper_barrier,
        lower_barrier_pct=args.lower_barrier,
        volatility_scaling=volatility_scaling,
        volatility_multiplier=args.volatility_multiplier,
        volatility_lookback=TRIPLE_BARRIER_VOL_LOOKBACK,
        feature_window_size=args.feature_window,
        min_data_days=TRIPLE_BARRIER_MIN_DATA_DAYS,
        use_dynamic_barriers=args.dynamic_barriers
    )

    # Initialize data loader
    data_loader = DataLoader()

    def price_loader_func(ticker, start_date, end_date):
        """Load price data for a ticker."""
        try:
            df = data_loader.load_ticker(
                ticker,
                start_date=start_date,
                end_date=end_date,
                validate=False
            )
            if df is not None and not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
            return df
        except Exception as e:
            logger.warning(f"{ticker}: Data load failed: {e}")
            return None

    # Label patterns
    logger.info("Labeling patterns using Triple Barrier Method...")
    df_labeled = labeler.label_patterns(
        patterns_df=patterns_to_label,
        price_loader_func=price_loader_func,
        ticker_col='ticker',
        date_col='end_date',
        reference_date=reference_date,
        calculate_features=calculate_features,
        market_cap_col=args.market_cap_col,
        vix_col=args.vix_col
    )

    # Combine with already-labeled patterns (if not force relabeling)
    if not args.force_relabel and 'outcome_class' in df_candidates.columns:
        already_labeled = df_candidates[~df_candidates['outcome_class'].isna()]
        if len(already_labeled) > 0 and len(df_labeled) > 0:
            df_labeled = pd.concat([already_labeled, df_labeled], ignore_index=True)
        elif len(already_labeled) > 0:
            df_labeled = already_labeled

    # Save outputs
    output_dir = Path(args.output).parent if args.output else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        labeled_path = Path(args.output)
    else:
        labeled_path = output_dir / 'labeled_patterns_triple_barrier.parquet'

    if len(df_labeled) > 0:
        df_labeled.to_parquet(labeled_path, index=False)
        logger.info(f"\nSaved {len(df_labeled):,} labeled patterns to {labeled_path}")

    # Create temporal splits if requested
    if args.create_splits:
        if args.train_end is None or args.val_end is None:
            logger.error("--create-splits requires --train-end and --val-end")
            return 1

        logger.info("\n" + "=" * 50)
        logger.info("Creating Strict Temporal Splits")
        logger.info("=" * 50)

        # Filter to successfully labeled patterns
        df_valid = df_labeled[df_labeled['label'].notna()].copy()

        train_df, val_df, test_df = create_strict_temporal_split(
            df=df_valid,
            date_col='end_date',
            train_end=args.train_end,
            val_end=args.val_end,
            gap_days=args.gap_days
        )

        # Validate temporal integrity
        validate_temporal_integrity(train_df, val_df, test_df, date_col='end_date')

        # Save splits
        train_path = output_dir / 'train_patterns.parquet'
        val_path = output_dir / 'val_patterns.parquet'
        test_path = output_dir / 'test_patterns.parquet'

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        logger.info(f"\nSaved splits:")
        logger.info(f"  Train: {train_path} ({len(train_df):,} patterns)")
        logger.info(f"  Val: {val_path} ({len(val_df):,} patterns)")
        logger.info(f"  Test: {test_path} ({len(test_df):,} patterns)")

    # Save labeling summary
    summary_path = output_dir / f"labeling_summary_triple_barrier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Calculate statistics
    labeled_mask = df_labeled['label'].notna() if 'label' in df_labeled.columns else pd.Series([False] * len(df_labeled))
    n_labeled = labeled_mask.sum()

    with open(summary_path, 'w') as f:
        f.write("Triple Barrier Labeling Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Reference Date: {reference_date.date()}\n")
        f.write(f"Input: {input_path}\n\n")

        f.write("TRIPLE BARRIER FRAMEWORK (Lopez de Prado)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Vertical Barrier: {args.vertical_barrier_days} days\n")
        f.write(f"Upper Barrier: +{args.upper_barrier*100:.1f}%\n")
        f.write(f"Lower Barrier: -{args.lower_barrier*100:.1f}%\n")
        f.write(f"Volatility Scaling: {volatility_scaling}\n")
        if volatility_scaling:
            f.write(f"  Multiplier: {args.volatility_multiplier}x\n")
        f.write(f"Feature Window: {args.feature_window} days\n\n")

        f.write("TEMPORAL INTEGRITY\n")
        f.write("-" * 40 + "\n")
        f.write("All rolling windows use closed='left'\n")
        f.write("Features computed BEFORE pattern end date only\n\n")

        f.write("RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Candidates loaded: {len(df_candidates):,}\n")
        f.write(f"Successfully labeled: {n_labeled:,}\n")

        if n_labeled > 0 and 'label' in df_labeled.columns:
            labeled_df = df_labeled[labeled_mask]
            label_counts = labeled_df['label'].value_counts().to_dict()
            f.write(f"\nOutcome Distribution:\n")
            f.write(f"  Class 0 (Danger): {label_counts.get(0, 0):,} ({100*label_counts.get(0, 0)/n_labeled:.1f}%)\n")
            f.write(f"  Class 1 (Noise): {label_counts.get(1, 0):,} ({100*label_counts.get(1, 0)/n_labeled:.1f}%)\n")
            f.write(f"  Class 2 (Target): {label_counts.get(2, 0):,} ({100*label_counts.get(2, 0)/n_labeled:.1f}%)\n")

            if 'max_return' in labeled_df.columns:
                f.write(f"\nReturn Statistics:\n")
                f.write(f"  Max Return (mean): {labeled_df['max_return'].mean()*100:.2f}%\n")
                f.write(f"  Max Return (median): {labeled_df['max_return'].median()*100:.2f}%\n")
                f.write(f"  Min Return (mean): {labeled_df['min_return'].mean()*100:.2f}%\n")

            if 'barrier_hit_day' in labeled_df.columns:
                f.write(f"\nBarrier Hit Statistics:\n")
                f.write(f"  Mean hit day: {labeled_df['barrier_hit_day'].mean():.1f}\n")
                f.write(f"  Median hit day: {labeled_df['barrier_hit_day'].median():.1f}\n")

        f.write(f"\nOutput: {labeled_path}\n")

    logger.info(f"Saved summary to {summary_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Triple Barrier Labeling Complete")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
