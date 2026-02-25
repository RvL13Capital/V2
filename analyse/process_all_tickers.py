"""
Process ALL tickers from GCS using BigQuery with optimizations
Designed for 3000+ tickers while staying within free tier
"""

import os
import argparse
import logging
from datetime import datetime, timedelta
import time
import sys

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json'

from cloud_market_analysis import CloudMarketAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'process_all_tickers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Process all tickers from GCS')
    parser.add_argument('--limit', type=int, help='Limit number of tickers (for testing)')
    parser.add_argument('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=datetime.now().strftime('%Y-%m-%d'),
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--batch-size', type=int, default=500,
                       help='Number of tickers per batch (default: 500)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode with 100 tickers')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = CloudMarketAnalyzer(
        project_id='ignition-ki-csv-storage',
        credentials_path=r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json',
        use_bigquery=True
    )

    # Check current quota usage
    usage = analyzer.get_usage_report()
    logger.info("="*80)
    logger.info("CURRENT QUOTA STATUS")
    logger.info("="*80)
    logger.info(f"Used today: {usage['bigquery']['bytes_processed_today_gb']:.2f} GB")
    logger.info(f"Daily limit: {usage['bigquery']['daily_limit_gb']:.2f} GB")
    logger.info(f"Percentage used: {usage['bigquery']['percentage_used']:.1f}%")
    logger.info(f"Remaining: {usage['remaining_quota']['bytes_gb']:.2f} GB")
    logger.info("="*80)

    # Get all available tickers
    logger.info("Loading all available tickers from GCS...")
    all_tickers = analyzer.load_tickers_from_gcs()

    if args.test_mode:
        all_tickers = all_tickers[:100]
        logger.info("[TEST MODE] Processing first 100 tickers only")
    elif args.limit:
        all_tickers = all_tickers[:args.limit]
        logger.info(f"[LIMITED MODE] Processing first {args.limit} tickers")

    total_tickers = len(all_tickers)
    logger.info(f"Found {total_tickers} tickers to process")

    # Estimate data size and check if it fits in remaining quota
    estimated_gb_per_ticker = 0.00007  # ~70KB per ticker (conservative)
    total_estimated_gb = total_tickers * estimated_gb_per_ticker

    logger.info(f"Estimated total data size: {total_estimated_gb:.2f} GB")

    if total_estimated_gb > usage['remaining_quota']['bytes_gb']:
        logger.warning("Data exceeds remaining daily quota!")
        logger.info("Will process in batches with fallback to local processing if needed")

    # Process in batches
    batch_size = args.batch_size
    num_batches = (total_tickers + batch_size - 1) // batch_size

    logger.info(f"Processing in {num_batches} batches of up to {batch_size} tickers each")
    logger.info("="*80)

    all_results = []
    failed_tickers = []

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_tickers)
        batch_tickers = all_tickers[start_idx:end_idx]

        logger.info(f"\nBATCH {batch_num + 1}/{num_batches}")
        logger.info(f"Processing tickers {start_idx + 1} to {end_idx} ({len(batch_tickers)} tickers)")
        logger.info("-"*40)

        try:
            # Try BigQuery first, with automatic fallback to local
            batch_start = time.time()
            results = analyzer.run_hybrid_analysis(
                tickers=batch_tickers,
                use_cloud_priority=True,
                fallback_to_local=True
            )

            if results is not None and not results.empty:
                all_results.append(results)
                logger.info(f"Batch {batch_num + 1} completed: {len(results)} rows")
            else:
                logger.warning(f"Batch {batch_num + 1} returned no results")
                failed_tickers.extend(batch_tickers)

            batch_time = time.time() - batch_start
            logger.info(f"Batch processing time: {batch_time:.1f} seconds")

            # Check quota after each batch
            usage = analyzer.get_usage_report()
            logger.info(f"Quota used: {usage['bigquery']['bytes_processed_today_gb']:.2f} GB "
                       f"({usage['bigquery']['percentage_used']:.1f}%)")

            # If approaching quota limit, warn user
            if usage['bigquery']['percentage_used'] > 90:
                logger.warning("Approaching daily quota limit (>90% used)")
                logger.info("Consider continuing tomorrow or using local processing")

        except Exception as e:
            logger.error(f"Error processing batch {batch_num + 1}: {e}")
            failed_tickers.extend(batch_tickers)
            continue

        # Small delay between batches to avoid rate limiting
        if batch_num < num_batches - 1:
            time.sleep(1)

    # Combine all results
    logger.info("\n" + "="*80)
    logger.info("COMBINING RESULTS")
    logger.info("="*80)

    if all_results:
        import pandas as pd
        final_results = pd.concat(all_results, ignore_index=True)

        # Save to parquet
        output_file = f"all_tickers_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        final_results.to_parquet(output_file)

        logger.info(f"[SUCCESS] Results saved to {output_file}")
        logger.info(f"Total rows: {len(final_results):,}")
        logger.info(f"Total tickers processed: {final_results['ticker'].nunique():,}")

        # Summary statistics
        if 'consolidation' in final_results.columns:
            consolidation_count = final_results['consolidation'].sum()
            logger.info(f"Consolidation patterns found: {consolidation_count:,}")

        # Save summary
        summary = {
            'total_tickers_requested': total_tickers,
            'tickers_processed': final_results['ticker'].nunique(),
            'total_rows': len(final_results),
            'failed_tickers': failed_tickers,
            'processing_date': datetime.now().isoformat(),
            'start_date': args.start_date,
            'end_date': args.end_date
        }

        import json
        summary_file = f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_file}")

    else:
        logger.error("No results generated!")

    if failed_tickers:
        logger.warning(f"\n{len(failed_tickers)} tickers failed to process:")
        logger.warning(f"Failed tickers: {failed_tickers[:10]}..." if len(failed_tickers) > 10
                      else f"Failed tickers: {failed_tickers}")

    # Final quota report
    final_usage = analyzer.get_usage_report()
    logger.info("\n" + "="*80)
    logger.info("FINAL QUOTA USAGE")
    logger.info("="*80)
    logger.info(f"Total processed today: {final_usage['bigquery']['bytes_processed_today_gb']:.2f} GB")
    logger.info(f"Percentage of daily limit: {final_usage['bigquery']['percentage_used']:.1f}%")
    logger.info(f"Remaining today: {final_usage['remaining_quota']['bytes_gb']:.2f} GB")
    logger.info("="*80)


if __name__ == "__main__":
    main()