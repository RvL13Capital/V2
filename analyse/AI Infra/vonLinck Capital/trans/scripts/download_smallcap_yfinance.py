#!/usr/bin/env python3
"""
Download US Small-Cap Stock Data using yfinance
================================================

Uses Finnhub for stock screening and yfinance for historical data download.
yfinance is free and has excellent coverage of small/micro-cap stocks.

Market Cap Filters:
    - Micro Cap: < $300M
    - Small Cap: $300M - $2B
    - Low Mid Cap: $2B - $5B

Usage:
    python scripts/download_smallcap_yfinance.py --limit 500 --min-history 2015
    python scripts/download_smallcap_yfinance.py --cap-category micro --limit 200
    python scripts/download_smallcap_yfinance.py --dry-run

Jan 2026 - Created for expanding US ticker coverage
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Set, Dict, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Keys
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

# Market cap categories (in USD)
CAP_CATEGORIES = {
    'micro': (0, 300_000_000),              # < $300M
    'small': (300_000_000, 2_000_000_000),  # $300M - $2B
    'low_mid': (2_000_000_000, 5_000_000_000),  # $2B - $5B
}


def get_existing_tickers_gcs() -> Set[str]:
    """Get list of US tickers already in GCS."""
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\AI Infra\AIv6\gcs_credentials.json'
        from google.cloud import storage
        import re

        client = storage.Client(project='ignition-ki-csv-storage')
        bucket = client.bucket('ignition-ki-csv-data-2025-user123')

        blobs = list(bucket.list_blobs(prefix='tickers/'))
        us_tickers = set()

        for blob in blobs:
            name = blob.name.replace('tickers/', '').replace('.parquet', '').replace('.csv', '')
            if not name or '/' in name or re.search(r'_\d{8}$', name):
                continue
            if '.' not in name:
                us_tickers.add(name.upper())

        logger.info(f"Found {len(us_tickers)} existing US tickers in GCS")
        return us_tickers
    except Exception as e:
        logger.warning(f"Could not connect to GCS: {e}")
        return set()


def get_us_symbols_finnhub() -> List[Dict]:
    """Get list of US stock symbols from Finnhub."""
    if not FINNHUB_API_KEY:
        logger.error("FINNHUB_API_KEY not configured")
        return []

    try:
        url = f'https://finnhub.io/api/v1/stock/symbol?exchange=US&token={FINNHUB_API_KEY}'
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Filter to common stocks only (exclude warrants, units, preferred)
        filtered = []
        for s in data:
            symbol = s.get('symbol', '')
            stype = s.get('type', '')

            # Skip if not common stock
            if stype not in ['Common Stock', 'EQS', '']:
                continue

            # Skip warrants, units, preferred (typically have special characters)
            if any(x in symbol for x in ['-', '.', '/']) or len(symbol) > 5:
                continue

            # Skip OTC/pink sheets (typically have longer symbols or special patterns)
            if symbol.endswith('F') or symbol.endswith('Y'):
                continue

            filtered.append(s)

        logger.info(f"Finnhub: {len(filtered)} common stocks (filtered from {len(data)})")
        return filtered

    except Exception as e:
        logger.error(f"Finnhub error: {e}")
        return []


def get_market_cap_yfinance(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    """Get market cap for a ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap')
        sector = info.get('sector', 'Unknown')
        return market_cap, sector
    except Exception:
        return None, None


def download_history_yfinance(ticker: str, start_date: str = '2010-01-01') -> Optional[pd.DataFrame]:
    """Download historical data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, auto_adjust=True)

        if df.empty:
            return None

        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Remove timezone info
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        # Add ticker column
        df['ticker'] = ticker

        # Select only needed columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]

        return df

    except Exception as e:
        logger.debug(f"yfinance error for {ticker}: {e}")
        return None


def upload_to_gcs(df: pd.DataFrame, ticker: str) -> bool:
    """Upload ticker data to GCS."""
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\AI Infra\AIv6\gcs_credentials.json'
        from google.cloud import storage

        client = storage.Client(project='ignition-ki-csv-storage')
        bucket = client.bucket('ignition-ki-csv-data-2025-user123')

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        blob = bucket.blob(f'tickers/{ticker}.parquet')
        blob.upload_from_file(buffer, content_type='application/octet-stream')

        return True
    except Exception as e:
        logger.error(f"GCS upload failed for {ticker}: {e}")
        return False


def screen_tickers_batch(
    tickers: List[str],
    cap_min: float,
    cap_max: float,
    batch_size: int = 50
) -> List[Dict]:
    """Screen tickers for market cap in batches."""
    valid_tickers = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        logger.info(f"Screening batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}...")

        for ticker in batch:
            market_cap, sector = get_market_cap_yfinance(ticker)

            if market_cap is None:
                continue

            if cap_min <= market_cap <= cap_max:
                valid_tickers.append({
                    'ticker': ticker,
                    'market_cap': market_cap,
                    'sector': sector
                })

        # Brief pause to avoid rate limiting
        time.sleep(0.5)

    return valid_tickers


def main():
    parser = argparse.ArgumentParser(description='Download US small-cap stock data using yfinance')
    parser.add_argument('--cap-category', type=str, default='all',
                        choices=['micro', 'small', 'low_mid', 'all'],
                        help='Market cap category to download')
    parser.add_argument('--limit', type=int, default=500,
                        help='Maximum number of new tickers to download')
    parser.add_argument('--min-history', type=int, default=2015,
                        help='Minimum year for historical data')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without downloading')
    parser.add_argument('--skip-gcs-check', action='store_true',
                        help='Skip checking existing GCS tickers')
    parser.add_argument('--screen-batch-size', type=int, default=100,
                        help='Batch size for market cap screening')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel download workers')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("US SMALL-CAP DATA DOWNLOADER (yfinance)")
    logger.info("=" * 60)
    logger.info(f"Cap category: {args.cap_category}")
    logger.info(f"Min history: {args.min_history}")
    logger.info(f"Limit: {args.limit}")

    # Get existing tickers
    if args.skip_gcs_check:
        existing_tickers = set()
    else:
        existing_tickers = get_existing_tickers_gcs()

    # Get all US symbols from Finnhub
    all_symbols = get_us_symbols_finnhub()
    if not all_symbols:
        logger.error("No symbols retrieved from Finnhub")
        return 1

    # Filter out existing tickers
    new_symbols = [s for s in all_symbols if s['symbol'] not in existing_tickers]
    logger.info(f"New symbols to screen: {len(new_symbols)}")

    # Determine market cap range
    if args.cap_category == 'all':
        cap_min = 0
        cap_max = CAP_CATEGORIES['low_mid'][1]
    else:
        cap_min, cap_max = CAP_CATEGORIES[args.cap_category]

    logger.info(f"Market cap range: ${cap_min/1e6:.0f}M - ${cap_max/1e6:.0f}M")

    # Screen for market cap (sample first to speed up)
    # Take 3x the limit to account for failed downloads
    symbols_to_screen = [s['symbol'] for s in new_symbols[:args.limit * 3]]

    logger.info(f"\nScreening {len(symbols_to_screen)} tickers for market cap...")
    valid_tickers = screen_tickers_batch(
        symbols_to_screen,
        cap_min,
        cap_max,
        batch_size=args.screen_batch_size
    )

    logger.info(f"Found {len(valid_tickers)} tickers in target market cap range")

    # Sort by market cap (largest small caps first - more liquid)
    valid_tickers.sort(key=lambda x: x['market_cap'], reverse=True)

    # Apply limit
    tickers_to_download = valid_tickers[:args.limit]

    if args.dry_run:
        logger.info("\n[DRY RUN] Would download these tickers:")
        for i, t in enumerate(tickers_to_download[:50]):
            cap_str = f"${t['market_cap']/1e6:,.0f}M"
            logger.info(f"  {i+1:3}. {t['ticker']:6} - {cap_str:>12} - {t['sector']}")
        if len(tickers_to_download) > 50:
            logger.info(f"  ... and {len(tickers_to_download) - 50} more")
        return 0

    # Download data
    successful = []
    failed = []
    insufficient_history = []

    logger.info(f"\nDownloading {len(tickers_to_download)} tickers...")
    logger.info("-" * 60)

    start_date = f"{args.min_history}-01-01"

    for i, t in enumerate(tickers_to_download):
        ticker = t['ticker']
        logger.info(f"[{i+1}/{len(tickers_to_download)}] {ticker} (${t['market_cap']/1e6:,.0f}M)...")

        df = download_history_yfinance(ticker, start_date)

        if df is None or df.empty:
            failed.append(ticker)
            logger.info(f"  [FAILED] No data available")
            continue

        min_date = df['date'].min()
        max_date = df['date'].max()

        if min_date.year > args.min_history:
            insufficient_history.append((ticker, min_date.year))
            logger.info(f"  [SKIP] Only has data from {min_date.year}")
            continue

        # Upload to GCS
        if upload_to_gcs(df, ticker):
            successful.append(ticker)
            logger.info(f"  [OK] {len(df)} days ({min_date.date()} to {max_date.date()})")
        else:
            failed.append(ticker)
            logger.info(f"  [FAILED] GCS upload error")

        # Small delay to be nice to yfinance
        time.sleep(0.3)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Insufficient history: {len(insufficient_history)}")

    if successful:
        logger.info(f"\nNew tickers added to GCS:")
        for ticker in successful[:30]:
            logger.info(f"  {ticker}")
        if len(successful) > 30:
            logger.info(f"  ... and {len(successful) - 30} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
