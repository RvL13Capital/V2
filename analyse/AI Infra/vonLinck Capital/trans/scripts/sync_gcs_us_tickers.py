"""
Sync US Tickers from GCS
========================

Downloads US stock data from GCS bucket to local cache.
Includes delisted stocks that yfinance cannot provide.

Usage:
    python scripts/sync_gcs_us_tickers.py
    python scripts/sync_gcs_us_tickers.py --limit 100
    python scripts/sync_gcs_us_tickers.py --tickers data/us_delisted_in_gcs.txt
"""

import os
import sys
import re
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(PROJECT_ROOT / 'gcs_credentials.json')

from google.cloud import storage


def is_us_ticker(ticker: str) -> bool:
    """Check if ticker looks like a US stock (no country suffix)."""
    # Has country suffix = not US
    suffixes = ['.KL', '.DE', '.L', '.ST', '.CO', '.PA', '.AS', '.BR', '.MI', '.MC', '.TO', '.AX', '.HK', '.SI', '.T']
    if any(ticker.upper().endswith(s) for s in suffixes):
        return False
    # Pure alphanumeric 1-5 chars = likely US
    return bool(re.match(r'^[A-Z]{1,5}$', ticker.upper()))


def sync_gcs_to_local(
    ticker_filter: list = None,
    output_dir: Path = None,
    limit: int = None,
    skip_existing: bool = True
) -> dict:
    """
    Download US stock data from GCS to local cache.

    Args:
        ticker_filter: List of tickers to download (None = all US tickers)
        output_dir: Output directory
        limit: Max tickers to download
        skip_existing: Skip if local file exists

    Returns:
        Dict with download stats
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'raw_us'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to GCS
    client = storage.Client(project='ignition-ki-csv-storage')
    bucket = client.bucket('ignition-ki-csv-data-2025-user123')

    # List all files in tickers/
    logger.info("Listing GCS tickers...")
    blobs = list(bucket.list_blobs(prefix='tickers/'))

    # Filter to US tickers
    us_blobs = []
    for blob in blobs:
        # Extract ticker from path: tickers/AAPL.parquet or tickers/AAPL.csv
        name = blob.name.replace('tickers/', '')
        ticker = name.split('.')[0]

        if is_us_ticker(ticker):
            # Apply ticker filter if provided
            if ticker_filter is None or ticker.upper() in ticker_filter:
                us_blobs.append((ticker.upper(), blob))

    logger.info(f"Found {len(us_blobs)} US tickers in GCS")

    # Filter existing
    if skip_existing:
        existing = {f.stem.upper() for f in output_dir.glob("*.parquet")}
        us_blobs = [(t, b) for t, b in us_blobs if t not in existing]
        logger.info(f"After skipping existing: {len(us_blobs)} to download")

    # Apply limit
    if limit:
        us_blobs = us_blobs[:limit]
        logger.info(f"Limited to {len(us_blobs)} tickers")

    # Download
    results = {'success': 0, 'failed': 0, 'skipped': 0}

    for ticker, blob in tqdm(us_blobs, desc="Downloading"):
        try:
            # Download to temp, then convert to parquet if CSV
            if blob.name.endswith('.parquet'):
                output_path = output_dir / f"{ticker}.parquet"
                blob.download_to_filename(str(output_path))
            else:
                # CSV - download and convert to parquet
                import io
                content = blob.download_as_bytes()
                df = pd.read_csv(io.BytesIO(content))

                # Standardize columns
                df.columns = [c.lower() for c in df.columns]
                if 'date' not in df.columns and 'datetime' in df.columns:
                    df = df.rename(columns={'datetime': 'date'})

                # Ensure date is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

                # Save as parquet
                output_path = output_dir / f"{ticker}.parquet"
                df.to_parquet(output_path, index=False)

            results['success'] += 1

        except Exception as e:
            logger.debug(f"{ticker}: {e}")
            results['failed'] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description='Sync US Tickers from GCS')
    parser.add_argument('--tickers', type=str, help='Path to ticker list file')
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'data' / 'raw_us'))
    parser.add_argument('--limit', type=int, help='Max tickers to download')
    parser.add_argument('--no-skip', action='store_true', help='Re-download existing files')

    args = parser.parse_args()

    # Load ticker filter if provided
    ticker_filter = None
    if args.tickers:
        ticker_path = Path(args.tickers)
        if ticker_path.exists():
            with open(ticker_path, 'r') as f:
                ticker_filter = set([line.strip().upper() for line in f if line.strip()])
            logger.info(f"Loaded {len(ticker_filter)} tickers from {ticker_path}")

    # Run sync
    results = sync_gcs_to_local(
        ticker_filter=ticker_filter,
        output_dir=Path(args.output_dir),
        limit=args.limit,
        skip_existing=not args.no_skip
    )

    # Summary
    print("\n" + "=" * 50)
    print("GCS SYNC COMPLETE")
    print("=" * 50)
    print(f"  Success: {results['success']}")
    print(f"  Failed:  {results['failed']}")
    print("=" * 50)


if __name__ == '__main__':
    main()
