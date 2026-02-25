"""
SEC EDGAR Ticker Fetcher
========================

Fetches all US public company tickers from SEC EDGAR.
This is the authoritative source for all SEC-registered companies.

Data Sources:
    - https://www.sec.gov/files/company_tickers.json
    - https://www.sec.gov/files/company_tickers_exchange.json

Usage:
    python scripts/fetch_edgar_tickers.py
    python scripts/fetch_edgar_tickers.py --exchange NASDAQ --output data/nasdaq_tickers.txt
    python scripts/fetch_edgar_tickers.py --filter-otc --output data/listed_only.txt
"""

import os
import sys
import json
import argparse
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# SEC EDGAR URLs
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

# Required headers for SEC API
HEADERS = {
    "User-Agent": "TRAnS Research contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}


def fetch_company_tickers() -> pd.DataFrame:
    """
    Fetch basic company tickers from SEC EDGAR.

    Returns DataFrame with columns: cik, ticker, title
    """
    logger.info(f"Fetching tickers from {TICKERS_URL}")

    response = requests.get(TICKERS_URL, headers=HEADERS)
    response.raise_for_status()

    data = response.json()

    # Convert to DataFrame
    records = []
    for idx, info in data.items():
        records.append({
            'cik': str(info['cik_str']).zfill(10),
            'ticker': info['ticker'],
            'title': info['title']
        })

    df = pd.DataFrame(records)
    logger.info(f"Fetched {len(df)} tickers")

    return df


def fetch_company_tickers_exchange() -> pd.DataFrame:
    """
    Fetch company tickers with exchange information from SEC EDGAR.

    Returns DataFrame with columns: cik, name, ticker, exchange
    """
    logger.info(f"Fetching tickers with exchange from {TICKERS_EXCHANGE_URL}")

    response = requests.get(TICKERS_EXCHANGE_URL, headers=HEADERS)
    response.raise_for_status()

    data = response.json()

    # The structure is: {"data": [[cik, name, ticker, exchange], ...]}
    if 'data' in data:
        df = pd.DataFrame(data['data'], columns=['cik', 'name', 'ticker', 'exchange'])
    else:
        # Fallback structure
        df = pd.DataFrame(data)

    # Pad CIK to 10 digits
    df['cik'] = df['cik'].astype(str).str.zfill(10)

    logger.info(f"Fetched {len(df)} tickers with exchange info")

    return df


def filter_by_exchange(df: pd.DataFrame, exchanges: List[str]) -> pd.DataFrame:
    """Filter tickers by exchange."""
    exchanges_upper = [e.upper() for e in exchanges]
    mask = df['exchange'].str.upper().isin(exchanges_upper)
    filtered = df[mask].copy()
    logger.info(f"Filtered to {len(filtered)} tickers on exchanges: {exchanges}")
    return filtered


def filter_exclude_otc(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude OTC/Pink Sheet stocks."""
    # Common OTC exchange codes
    otc_codes = ['OTC', 'OTCBB', 'PINK', 'GREY', 'OTC PINK', 'OTCQB', 'OTCQX']
    mask = ~df['exchange'].str.upper().isin(otc_codes)
    filtered = df[mask].copy()
    logger.info(f"Excluded OTC: {len(df)} -> {len(filtered)} tickers")
    return filtered


def get_exchange_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Get breakdown of tickers by exchange."""
    return df.groupby('exchange').size().sort_values(ascending=False).reset_index(name='count')


def save_ticker_list(tickers: List[str], output_path: Path):
    """Save ticker list to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(tickers))
    logger.info(f"Saved {len(tickers)} tickers to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fetch SEC EDGAR Company Tickers')
    parser.add_argument('--output', type=str, default=str(DATA_DIR / 'edgar_tickers.txt'),
                        help='Output file path')
    parser.add_argument('--exchange', type=str, nargs='+',
                        help='Filter by exchange (e.g., NASDAQ NYSE)')
    parser.add_argument('--filter-otc', action='store_true',
                        help='Exclude OTC/Pink Sheet stocks')
    parser.add_argument('--save-full', action='store_true',
                        help='Also save full DataFrame as parquet')
    parser.add_argument('--show-exchanges', action='store_true',
                        help='Show exchange breakdown and exit')

    args = parser.parse_args()

    # Fetch data with exchange info
    try:
        df = fetch_company_tickers_exchange()
    except Exception as e:
        logger.warning(f"Exchange API failed ({e}), falling back to basic tickers")
        df = fetch_company_tickers()
        df['exchange'] = 'UNKNOWN'

    # Show exchange breakdown if requested
    if args.show_exchanges:
        print("\n" + "=" * 50)
        print("EXCHANGE BREAKDOWN")
        print("=" * 50)
        breakdown = get_exchange_breakdown(df)
        for _, row in breakdown.iterrows():
            print(f"  {row['exchange']:15s} {row['count']:>6,}")
        print("=" * 50)
        print(f"  TOTAL: {len(df):,}")
        return

    # Apply filters
    if args.filter_otc:
        df = filter_exclude_otc(df)

    if args.exchange:
        df = filter_by_exchange(df, args.exchange)

    # Get unique tickers
    tickers = df['ticker'].dropna().unique().tolist()
    tickers = [t for t in tickers if t and len(t) <= 5]  # Filter valid tickers
    tickers = sorted(set(tickers))

    # Save ticker list
    output_path = Path(args.output)
    save_ticker_list(tickers, output_path)

    # Save full data if requested
    if args.save_full:
        parquet_path = output_path.with_suffix('.parquet')
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved full data to {parquet_path}")

    # Summary
    print("\n" + "=" * 50)
    print("SEC EDGAR TICKER FETCH COMPLETE")
    print("=" * 50)
    print(f"  Total companies: {len(df):,}")
    print(f"  Valid tickers:   {len(tickers):,}")
    print(f"  Output file:     {output_path}")
    print("=" * 50)

    # Show sample
    print("\nSample tickers:")
    for t in tickers[:10]:
        print(f"  {t}")
    print("  ...")


if __name__ == '__main__':
    main()
