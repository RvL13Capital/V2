"""
Step 0: Detect Consolidation Patterns (CANDIDATE REGISTRY)
===========================================================

Scans tickers for consolidation patterns using the ConsolidationTracker
state machine. This is the first step in the TRANS pipeline.

TEMPORAL INTEGRITY:
    This script outputs CANDIDATES ONLY - patterns WITHOUT outcome labels.
    Outcome labeling is deferred to 00b_label_outcomes.py which runs
    100+ days after pattern detection to ensure NO look-ahead bias.

    The separation into two registries guarantees that the model cannot
    "see" future price data during training feature engineering.

Input:
    - Ticker list (file, comma-separated, or "ALL" for GCS bucket)
    - Date range for scanning

Output:
    - candidate_patterns.parquet: Detected patterns (NO outcome_class)
      Fields: ticker, pattern_id, start_date, end_date, boundaries, etc.
    - Scan statistics and timing information

    NOTE: outcome_class will be NULL until 00b_label_outcomes.py is run.

Runtime: ~5-30 minutes depending on universe size

Usage:
    # Scan specific tickers (outputs candidates only - DEFAULT)
    python 00_detect_patterns.py --tickers AAPL,MSFT,GOOGL --start-date 2020-01-01

    # Scan from ticker file
    python 00_detect_patterns.py --tickers tickers.txt --start-date 2020-01-01

    # Scan with parallel workers
    python 00_detect_patterns.py --tickers tickers.txt --workers 8

    # Quick test with limit
    python 00_detect_patterns.py --tickers tickers.txt --limit 10

    # LEGACY MODE (with labeling - only for backtesting known historical data)
    # WARNING: This mode has look-ahead bias - only use for backtesting!
    python 00_detect_patterns.py --tickers tickers.txt --with-labeling
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pattern_scanner import ConsolidationPatternScanner, UniverseScanResult
from core.aiv7_components import DataLoader
from utils.logging_config import setup_pipeline_logging
from config import (
    MIN_DATA_LENGTH,
    MIN_PATTERN_DURATION,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_LOG_DIR
)

# Setup centralized logging
logger = setup_pipeline_logging('00_detect_patterns')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detect consolidation patterns in stock data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Scan specific tickers
    python 00_detect_patterns.py --tickers AAPL,MSFT,GOOGL

    # Scan from file
    python 00_detect_patterns.py --tickers universe.txt --start-date 2020-01-01

    # Full scan with parallel processing
    python 00_detect_patterns.py --tickers universe.txt --workers 8 --start-date 2020-01-01
        """
    )

    parser.add_argument(
        '--tickers',
        type=str,
        required=True,
        help='Comma-separated tickers, path to ticker file, or "ALL" for GCS'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,  # Will be set based on --with-labeling flag
        help='Output file path (default: candidate_patterns.parquet or detected_patterns.parquet)'
    )
    parser.add_argument(
        '--with-labeling',
        action='store_true',
        default=False,
        help='LEGACY MODE: Include outcome labels (LOOK-AHEAD RISK - only for backtesting!)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of tickers (for testing)'
    )
    parser.add_argument(
        '--min-data-days',
        type=int,
        default=MIN_DATA_LENGTH,
        help=f'Minimum data days required (default: {MIN_DATA_LENGTH})'
    )

    # Detection thresholds (legacy - percentile based)
    parser.add_argument(
        '--bbw-threshold',
        type=float,
        default=0.30,
        help='BBW percentile threshold (default: 0.30). LEGACY: use --tightness-zscore instead'
    )
    parser.add_argument(
        '--adx-threshold',
        type=float,
        default=32.0,
        help='ADX threshold (default: 32.0)'
    )
    parser.add_argument(
        '--volume-threshold',
        type=float,
        default=0.35,
        help='Volume ratio threshold (default: 0.35)'
    )
    parser.add_argument(
        '--range-threshold',
        type=float,
        default=0.65,
        help='Range ratio threshold (default: 0.65)'
    )

    # Adaptive detection thresholds (Jan 2026)
    parser.add_argument(
        '--tightness-zscore',
        type=float,
        default=None,
        help='Max Z-Score for BBW (relative tightness). E.g., -1.0 = 1 std dev tighter than avg. '
             'When set, overrides --bbw-threshold with adaptive measurement.'
    )
    parser.add_argument(
        '--min-float-turnover',
        type=float,
        default=None,
        help='Minimum 20d float turnover (e.g., 0.10 = 10%% of float traded). '
             'Requires accumulation activity to detect pattern.'
    )
    parser.add_argument(
        '--local-only',
        action='store_true',
        default=False,
        help='Use local data only (no GCS). Enables multiprocessing with --workers > 1'
    )

    # Weekly qualification mode (Jan 2026)
    parser.add_argument(
        '--use-weekly-qualification',
        action='store_true',
        default=False,
        help='Use WEEKLY candles for 10-week qualification period (default: daily 10-day). '
             'Finds longer-term consolidation patterns (~2.5 months). '
             'Requires ~4 years of historical data for SMA_200 calculation.'
    )

    # Performance optimization flags
    parser.add_argument(
        '--fast-validation',
        action='store_true',
        default=False,
        help='Skip expensive mock data detection in validation (~50-100ms savings per ticker)'
    )
    parser.add_argument(
        '--skip-market-cap-prefetch',
        action='store_true',
        default=False,
        help='Skip batch market cap pre-fetching (use if market caps already cached)'
    )
    parser.add_argument(
        '--disable-market-cap',
        action='store_true',
        default=False,
        help='Disable market cap fetching entirely (uses price-based fallback)'
    )
    parser.add_argument(
        '--skip-market-cap-api',
        action='store_true',
        default=False,
        help='Skip market cap API calls, use cached shares × price for point-in-time (PIT) '
             'estimation. Much faster for EU stocks and avoids look-ahead bias.'
    )

    return parser.parse_args()


def load_tickers(ticker_arg: str) -> list:
    """
    Load ticker list from various sources.

    Args:
        ticker_arg: Comma-separated string, file path, or "ALL"

    Returns:
        List of ticker symbols
    """
    # Check if it's a file path
    ticker_path = Path(ticker_arg)
    if ticker_path.exists():
        logger.info(f"Loading tickers from file: {ticker_path}")
        with open(ticker_path, 'r') as f:
            content = f.read()

        # Handle both newline and comma separated
        if ',' in content:
            tickers = [t.strip().upper() for t in content.split(',')]
        else:
            tickers = [t.strip().upper() for t in content.split('\n')]

        # Filter empty strings
        tickers = [t for t in tickers if t]
        logger.info(f"Loaded {len(tickers)} tickers from file")
        return tickers

    # Check if it's "ALL" (scan GCS bucket)
    if ticker_arg.upper() == 'ALL':
        logger.info("Discovering tickers from GCS bucket...")
        data_loader = DataLoader()

        # List all files in bucket
        if data_loader.bucket is not None:
            blobs = data_loader.bucket.list_blobs()
            tickers = []
            for blob in blobs:
                name = blob.name
                if name.endswith('.parquet') or name.endswith('.csv'):
                    ticker = name.replace('.parquet', '').replace('.csv', '')
                    tickers.append(ticker.upper())

            tickers = sorted(set(tickers))
            logger.info(f"Discovered {len(tickers)} tickers from GCS")
            return tickers
        else:
            # Fallback to local cache
            cache_dir = data_loader.cache_dir
            tickers = []
            for f in cache_dir.glob('*.parquet'):
                tickers.append(f.stem.upper())
            for f in cache_dir.glob('*.csv'):
                tickers.append(f.stem.upper())

            tickers = sorted(set(tickers))
            logger.info(f"Discovered {len(tickers)} tickers from local cache")
            return tickers

    # Assume comma-separated string
    tickers = [t.strip().upper() for t in ticker_arg.split(',')]
    tickers = [t for t in tickers if t]
    logger.info(f"Parsed {len(tickers)} tickers from argument")
    return tickers


def create_progress_bar(total: int):
    """Create a tqdm progress bar."""
    return tqdm(
        total=total,
        desc="Scanning",
        unit="ticker",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )


def main():
    args = parse_args()

    # Determine mode and output path
    candidate_only = not args.with_labeling

    if args.output is None:
        if candidate_only:
            args.output = f'{DEFAULT_OUTPUT_DIR}/candidate_patterns.parquet'
        else:
            args.output = f'{DEFAULT_OUTPUT_DIR}/detected_patterns.parquet'

    logger.info("=" * 60)
    if candidate_only:
        logger.info("Step 0: Detect Consolidation Patterns (CANDIDATE REGISTRY)")
        logger.info("         Output: CANDIDATES ONLY (no outcome_class)")
        logger.info("         Run 00b_label_outcomes.py to add labels after 100 days")
    else:
        logger.info("Step 0: Detect Consolidation Patterns (LEGACY MODE)")
        logger.info("         WARNING: LOOK-AHEAD BIAS - labels included at detection time")
        logger.info("         Only use for backtesting on historical data!")
    logger.info("=" * 60)

    # Load tickers
    tickers = load_tickers(args.tickers)

    if not tickers:
        logger.error("No tickers found!")
        return

    # Apply limit for testing
    if args.limit:
        tickers = tickers[:args.limit]
        logger.info(f"Limited to {len(tickers)} tickers for testing")

    logger.info(f"Scanning {len(tickers)} tickers")
    if args.start_date:
        logger.info(f"Start date: {args.start_date}")
    if args.end_date:
        logger.info(f"End date: {args.end_date}")

    # Log optimization flags
    if args.fast_validation:
        logger.info("OPTIMIZATION: Fast validation enabled (skipping mock data detection)")
    if args.skip_market_cap_prefetch:
        logger.info("OPTIMIZATION: Market cap pre-fetch disabled")
    if args.disable_market_cap:
        logger.info("OPTIMIZATION: Market cap fetching disabled (using price-based fallback)")

    # Log adaptive thresholds if set
    if args.tightness_zscore is not None:
        logger.info(f"ADAPTIVE: Using Z-score tightness threshold: {args.tightness_zscore}")
    if args.min_float_turnover is not None:
        logger.info(f"ADAPTIVE: Requiring min float turnover: {args.min_float_turnover:.1%}")

    # Log weekly qualification mode
    if args.use_weekly_qualification:
        logger.info("=" * 60)
        logger.info("WEEKLY QUALIFICATION MODE ENABLED")
        logger.info("  - Qualification period: 10 weeks (~2.5 months)")
        logger.info("  - Requires: ~4 years of historical data")
        logger.info("  - Expected: 65-85% fewer patterns (higher quality)")
        logger.info("=" * 60)

    # Log skip-market-cap-api mode
    if args.skip_market_cap_api:
        logger.info("=" * 60)
        logger.info("POINT-IN-TIME MARKET CAP MODE ENABLED")
        logger.info("  - No API calls (uses cached shares × price)")
        logger.info("  - Faster processing for EU stocks")
        logger.info("  - Avoids look-ahead bias for historical data")
        logger.info("=" * 60)

    # Initialize scanner
    scanner = ConsolidationPatternScanner(
        bbw_percentile_threshold=args.bbw_threshold,
        adx_threshold=args.adx_threshold,
        volume_ratio_threshold=args.volume_threshold,
        range_ratio_threshold=args.range_threshold,
        min_data_days=args.min_data_days,
        candidate_only=candidate_only,
        disable_gcs=args.local_only,
        enable_market_cap=not args.disable_market_cap,
        fast_validation=args.fast_validation,
        # Adaptive thresholds (Jan 2026)
        tightness_zscore=args.tightness_zscore,
        min_float_turnover=args.min_float_turnover,
        # Weekly qualification mode (Jan 2026)
        use_weekly_qualification=args.use_weekly_qualification,
        # Point-in-time market cap (Jan 2026)
        skip_market_cap_api=args.skip_market_cap_api
    )

    # Create progress tracking
    pbar = create_progress_bar(len(tickers))

    def progress_callback(completed, total, ticker, result):
        pbar.update(1)
        if result.success and result.patterns_found > 0:
            pbar.set_postfix({
                'ticker': ticker,
                'patterns': result.patterns_found
            })

    # Run scan
    logger.info(f"Starting scan with {args.workers} worker(s)...")
    result = scanner.scan_universe(
        tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        workers=args.workers,
        progress_callback=progress_callback,
        prefetch_market_caps=not args.skip_market_cap_prefetch
    )

    pbar.close()

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Scan Results")
    logger.info("=" * 60)
    logger.info(f"Total tickers:      {result.total_tickers}")
    logger.info(f"Successful:         {result.successful_tickers}")
    logger.info(f"Failed:             {result.failed_tickers}")
    logger.info(f"Total patterns:     {result.total_patterns}")
    logger.info(f"Patterns/ticker:    {result.patterns_per_ticker:.2f}")
    logger.info(f"Total time:         {result.total_time_seconds:.1f}s")
    logger.info(f"Time/ticker:        {result.total_time_seconds / len(tickers) * 1000:.1f}ms")

    # Outcome class distribution (only in legacy mode with labeling)
    if result.all_patterns:
        outcome_counts = {}
        for pattern in result.all_patterns:
            outcome = pattern.get('outcome_class')
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        if candidate_only:
            # In candidate mode, all outcomes should be None
            none_count = outcome_counts.get(None, 0)
            logger.info(f"\nCandidate Registry: {none_count:,} patterns (outcome_class = NULL)")
            logger.info("  Run 00b_label_outcomes.py to assign outcome classes")
        else:
            # Legacy mode - show actual distribution
            logger.info("\nOutcome Class Distribution:")
            for k in sorted([x for x in outcome_counts.keys() if x is not None]):
                count = outcome_counts[k]
                pct = 100 * count / len(result.all_patterns)
                logger.info(f"  K{k}: {count:,} ({pct:.1f}%)")

    # Log errors if any
    if result.errors:
        logger.info(f"\nErrors ({len(result.errors)} tickers):")
        for ticker, error in list(result.errors.items())[:10]:
            logger.info(f"  {ticker}: {error}")
        if len(result.errors) > 10:
            logger.info(f"  ... and {len(result.errors) - 10} more")

    # Save results
    if result.all_patterns:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scanner.save_patterns(result.all_patterns, output_path)

        logger.info(f"\nSaved {len(result.all_patterns)} patterns to {output_path}")

        # Also save scan summary
        summary_path = output_path.parent / f"scan_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Consolidation Pattern Scan Summary\n")
            f.write(f"{'=' * 40}\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Mode: {'CANDIDATE REGISTRY' if candidate_only else 'LEGACY (with labeling)'}\n")
            qualification_mode = "WEEKLY (10-week)" if args.use_weekly_qualification else "DAILY (10-day)"
            f.write(f"Qualification: {qualification_mode}\n")
            f.write(f"Tickers scanned: {result.total_tickers}\n")
            f.write(f"Successful: {result.successful_tickers}\n")
            f.write(f"Failed: {result.failed_tickers}\n")
            f.write(f"Patterns found: {result.total_patterns}\n")
            f.write(f"Total time: {result.total_time_seconds:.1f}s\n")
            f.write(f"\nOutput: {output_path}\n")
            if candidate_only:
                f.write(f"\nNOTE: outcome_class is NULL - run 00b_label_outcomes.py to add labels\n")
            if args.use_weekly_qualification:
                f.write(f"\nWEEKLY MODE: Patterns use 10-week qualification with wider boundaries.\n")
                f.write(f"end_date_daily column contains actual daily date for outcome labeling.\n")

        logger.info(f"Saved scan summary to {summary_path}")
    else:
        logger.warning("No patterns found!")

    logger.info("\n" + "=" * 60)
    logger.info("Pattern Detection Complete")
    logger.info("=" * 60)

    # Return exit code
    return 0 if result.successful_tickers > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
