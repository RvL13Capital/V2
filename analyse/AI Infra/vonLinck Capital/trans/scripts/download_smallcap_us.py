#!/usr/bin/env python3
"""
Download US Micro/Small/Low-Mid Cap Stock Data
===============================================

Uses FMP API to screen for small-cap stocks, then downloads historical data
from TwelveData or AlphaVantage.

Market Cap Filters:
    - Micro Cap: < $300M
    - Small Cap: $300M - $2B
    - Low Mid Cap: $2B - $5B

Usage:
    python scripts/download_smallcap_us.py --limit 500 --min-history 2015
    python scripts/download_smallcap_us.py --cap-category micro --limit 200
    python scripts/download_smallcap_us.py --dry-run  # Show what would be downloaded

Jan 2026 - Created for expanding US ticker coverage
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Set, Dict
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Keys
FMP_API_KEY = os.getenv('FMP_API_KEY')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')

# Rate limits
TWELVEDATA_DELAY = 8  # seconds between requests (8 req/min)
ALPHAVANTAGE_DELAY = 12  # seconds between requests (5 req/min)
FMP_DELAY = 0.5  # FMP is more generous

# Market cap categories (in USD)
CAP_CATEGORIES = {
    'micro': (0, 300_000_000),           # < $300M
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
            if '.' not in name:  # US tickers don't have dots
                us_tickers.add(name.upper())

        logger.info(f"Found {len(us_tickers)} existing US tickers in GCS")
        return us_tickers
    except Exception as e:
        logger.warning(f"Could not connect to GCS: {e}")
        return set()


def get_smallcap_stocks_fmp(
    cap_category: str = 'all',
    min_volume: int = 50000,
    exchanges: List[str] = ['NYSE', 'NASDAQ', 'AMEX']
) -> pd.DataFrame:
    """
    Get list of small-cap stocks from FMP API.

    Args:
        cap_category: 'micro', 'small', 'low_mid', or 'all'
        min_volume: Minimum average volume
        exchanges: List of exchanges to include

    Returns:
        DataFrame with ticker, name, market_cap, exchange
    """
    if not FMP_API_KEY:
        logger.error("FMP_API_KEY not configured")
        return pd.DataFrame()

    logger.info(f"Fetching stock list from FMP (category: {cap_category})...")

    all_stocks = []

    # Get stock list from FMP
    try:
        # Use stock screener endpoint
        url = f"https://financialmodelingprep.com/api/v3/stock-screener"

        if cap_category == 'all':
            # Get all small caps (micro + small + low_mid)
            max_cap = CAP_CATEGORIES['low_mid'][1]
            min_cap = 0
        else:
            min_cap, max_cap = CAP_CATEGORIES.get(cap_category, (0, 5_000_000_000))

        params = {
            'marketCapMoreThan': min_cap,
            'marketCapLowerThan': max_cap,
            'volumeMoreThan': min_volume,
            'exchange': ','.join(exchanges),
            'isEtf': 'false',
            'isActivelyTrading': 'true',
            'country': 'US',
            'limit': 10000,
            'apikey': FMP_API_KEY
        }

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            all_stocks.extend(data)
            logger.info(f"Found {len(data)} stocks from screener")

    except Exception as e:
        logger.error(f"FMP screener error: {e}")

    if not all_stocks:
        # Fallback: use tradeable stocks list
        logger.info("Trying tradeable stocks list as fallback...")
        try:
            url = f"https://financialmodelingprep.com/api/v3/available-traded/list"
            params = {'apikey': FMP_API_KEY}
            response = requests.get(url, params=params, timeout=60)
            data = response.json()

            if isinstance(data, list):
                # Filter to US exchanges only
                us_stocks = [s for s in data if s.get('exchangeShortName') in exchanges]
                all_stocks.extend(us_stocks)
                logger.info(f"Found {len(us_stocks)} US stocks from tradeable list")
        except Exception as e:
            logger.error(f"FMP tradeable list error: {e}")

    if not all_stocks:
        return pd.DataFrame()

    df = pd.DataFrame(all_stocks)

    # Standardize columns
    col_map = {
        'symbol': 'ticker',
        'companyName': 'name',
        'marketCap': 'market_cap',
        'exchangeShortName': 'exchange',
        'sector': 'sector',
        'industry': 'industry'
    }

    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Filter out funds, ETFs, preferred shares
    if 'ticker' in df.columns:
        df = df[~df['ticker'].str.contains(r'[-\^]', na=False)]
        df = df[~df['ticker'].str.match(r'^[A-Z]{5,}$', na=False)]  # Skip 5+ letter tickers (often funds)
        df = df[df['ticker'].str.match(r'^[A-Z]{1,5}$', na=False)]  # 1-5 uppercase letters only

    # Filter by market cap if we have the data
    if 'market_cap' in df.columns and cap_category != 'all':
        min_cap, max_cap = CAP_CATEGORIES.get(cap_category, (0, 5_000_000_000))
        df = df[(df['market_cap'] >= min_cap) & (df['market_cap'] <= max_cap)]

    logger.info(f"After filtering: {len(df)} stocks")

    return df


def download_twelvedata(ticker: str, outputsize: int = 5000) -> Optional[pd.DataFrame]:
    """Download data from TwelveData API."""
    if not TWELVEDATA_API_KEY:
        return None

    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': ticker,
            'interval': '1day',
            'outputsize': outputsize,
            'apikey': TWELVEDATA_API_KEY,
            'format': 'JSON'
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'status' in data and data['status'] == 'error':
            return None

        if 'values' not in data:
            return None

        df = pd.DataFrame(data['values'])
        df['date'] = pd.to_datetime(df['datetime'])
        df = df.drop('datetime', axis=1)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        df = df.sort_values('date').reset_index(drop=True)
        df['ticker'] = ticker

        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]

    except Exception as e:
        logger.debug(f"TwelveData error for {ticker}: {e}")
        return None


def download_alphavantage(ticker: str) -> Optional[pd.DataFrame]:
    """Download data from AlphaVantage API."""
    if not ALPHAVANTAGE_API_KEY:
        return None

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'outputsize': 'full',
            'apikey': ALPHAVANTAGE_API_KEY
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'Error Message' in data or 'Note' in data:
            return None

        if 'Time Series (Daily)' not in data:
            return None

        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.reset_index()
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        df = df.sort_values('date').reset_index(drop=True)
        df['ticker'] = ticker

        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]

    except Exception as e:
        logger.debug(f"AlphaVantage error for {ticker}: {e}")
        return None


def upload_to_gcs(df: pd.DataFrame, ticker: str) -> bool:
    """Upload ticker data to GCS."""
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\AI Infra\AIv6\gcs_credentials.json'
        from google.cloud import storage

        client = storage.Client(project='ignition-ki-csv-storage')
        bucket = client.bucket('ignition-ki-csv-data-2025-user123')

        # Save to parquet in memory
        import io
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        # Upload
        blob = bucket.blob(f'tickers/{ticker}.parquet')
        blob.upload_from_file(buffer, content_type='application/octet-stream')

        return True
    except Exception as e:
        logger.error(f"GCS upload failed for {ticker}: {e}")
        return False


def download_ticker(ticker: str, min_history_year: int = 2015) -> Optional[pd.DataFrame]:
    """
    Download historical data for a ticker.

    Args:
        ticker: Stock symbol
        min_history_year: Minimum year for data coverage

    Returns:
        DataFrame if successful and meets history requirement, None otherwise
    """
    df = None

    # Try TwelveData first (better for small caps)
    logger.debug(f"Trying TwelveData for {ticker}...")
    df = download_twelvedata(ticker)
    time.sleep(TWELVEDATA_DELAY)

    # Check if we have enough history
    if df is not None and not df.empty:
        min_date = df['date'].min()
        if min_date.year <= min_history_year:
            return df
        else:
            logger.debug(f"{ticker}: TwelveData only has data from {min_date.year}")

    # Try AlphaVantage as fallback
    logger.debug(f"Trying AlphaVantage for {ticker}...")
    df = download_alphavantage(ticker)
    time.sleep(ALPHAVANTAGE_DELAY)

    if df is not None and not df.empty:
        min_date = df['date'].min()
        if min_date.year <= min_history_year:
            return df
        else:
            logger.debug(f"{ticker}: AlphaVantage only has data from {min_date.year}")

    return None


def main():
    parser = argparse.ArgumentParser(description='Download US small-cap stock data')
    parser.add_argument('--cap-category', type=str, default='all',
                        choices=['micro', 'small', 'low_mid', 'all'],
                        help='Market cap category to download')
    parser.add_argument('--limit', type=int, default=500,
                        help='Maximum number of new tickers to download')
    parser.add_argument('--min-history', type=int, default=2015,
                        help='Minimum year for historical data')
    parser.add_argument('--min-volume', type=int, default=50000,
                        help='Minimum average daily volume')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without downloading')
    parser.add_argument('--skip-gcs-check', action='store_true',
                        help='Skip checking existing GCS tickers')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("US SMALL-CAP DATA DOWNLOADER")
    logger.info("=" * 60)
    logger.info(f"Cap category: {args.cap_category}")
    logger.info(f"Min history: {args.min_history}")
    logger.info(f"Limit: {args.limit}")
    logger.info(f"Min volume: {args.min_volume:,}")

    # Get existing tickers
    if args.skip_gcs_check:
        existing_tickers = set()
    else:
        existing_tickers = get_existing_tickers_gcs()

    # Get small-cap stocks from FMP
    stocks_df = get_smallcap_stocks_fmp(
        cap_category=args.cap_category,
        min_volume=args.min_volume
    )

    if stocks_df.empty:
        logger.error("No stocks found from FMP")
        return 1

    # Filter out existing tickers
    new_tickers = stocks_df[~stocks_df['ticker'].isin(existing_tickers)]
    logger.info(f"New tickers to download: {len(new_tickers)}")

    # Sort by market cap (largest first for small caps)
    if 'market_cap' in new_tickers.columns:
        new_tickers = new_tickers.sort_values('market_cap', ascending=False)

    # Apply limit
    tickers_to_download = new_tickers['ticker'].head(args.limit).tolist()

    if args.dry_run:
        logger.info("\n[DRY RUN] Would download these tickers:")
        for i, ticker in enumerate(tickers_to_download[:50]):
            row = new_tickers[new_tickers['ticker'] == ticker].iloc[0]
            cap = row.get('market_cap', 0)
            name = row.get('name', 'Unknown')[:30]
            logger.info(f"  {i+1:3}. {ticker:6} - {name:30} (${cap/1e6:,.0f}M)")
        if len(tickers_to_download) > 50:
            logger.info(f"  ... and {len(tickers_to_download) - 50} more")
        return 0

    # Download data
    successful = []
    failed = []
    insufficient_history = []

    logger.info(f"\nDownloading {len(tickers_to_download)} tickers...")
    logger.info("-" * 60)

    for i, ticker in enumerate(tickers_to_download):
        logger.info(f"[{i+1}/{len(tickers_to_download)}] {ticker}...")

        df = download_ticker(ticker, args.min_history)

        if df is None:
            failed.append(ticker)
            logger.info(f"  [FAILED] No data available")
            continue

        min_date = df['date'].min()
        max_date = df['date'].max()

        if min_date.year > args.min_history:
            insufficient_history.append((ticker, min_date.year))
            logger.info(f"  [SKIP] Only has data from {min_date.year} (need {args.min_history})")
            continue

        # Upload to GCS
        if upload_to_gcs(df, ticker):
            successful.append(ticker)
            logger.info(f"  [OK] {len(df)} days ({min_date.date()} to {max_date.date()})")
        else:
            failed.append(ticker)
            logger.info(f"  [FAILED] GCS upload error")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Insufficient history: {len(insufficient_history)}")

    if successful:
        logger.info(f"\nSuccessful tickers: {', '.join(successful[:20])}")
        if len(successful) > 20:
            logger.info(f"  ... and {len(successful) - 20} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
