#!/usr/bin/env python3
"""
Fetch market caps for remaining EU tickers that have price data.
Focus on likely equities (filter out ETFs, ADRs, etc.)
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
import yfinance as yf

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_FILE = DATA_DIR / "market_cap_cache.json"

# Rate limiting
DELAY_BETWEEN_TICKERS = 1.5
DELAY_ON_FAILURE = 3.0
MAX_CONSECUTIVE_FAILURES = 15
BACKOFF_DELAY = 60
BATCH_SIZE = 100  # Save and report every N tickers

EU_EXTENSIONS = ['.L', '.DE', '.PA', '.MI', '.AS', '.SW', '.ST', '.OL', '.VI', '.HE', '.BR', '.IR', '.MC', '.LS', '.CO']


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def is_likely_equity(ticker: str) -> bool:
    """Filter out non-equities."""
    # Numeric prefix usually means ETF/cross-listing/fund
    if ticker[0].isdigit():
        return False

    base = ticker.split('.')[0].upper()

    # Skip known patterns
    skip_patterns = ['XACT', 'UCIT', 'MSCI', 'FTSE', 'SP5', 'ETC', 'IWDA', 'VWCE', 'CSPX']
    for pat in skip_patterns:
        if pat in base:
            return False

    # Skip very short with numbers (ETFs like 3LDE, 2LDE)
    if len(base) <= 4 and any(c.isdigit() for c in base):
        return False

    return True


def fetch_market_cap(ticker: str) -> tuple:
    """Fetch market cap for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        mktcap = info.get('marketCap')
        name = info.get('shortName') or info.get('longName')
        currency = info.get('currency')
        if mktcap and mktcap > 0:
            return mktcap, name, currency
        return None, name, currency
    except Exception as e:
        return None, None, str(e)[:50]


def main():
    print("=" * 60)
    print("REMAINING EU MARKET CAP FETCHER")
    print("=" * 60)

    # Get tickers with price data
    parquet_tickers = set(f.stem for f in RAW_DIR.glob('*.parquet'))
    csv_tickers = set(f.stem for f in RAW_DIR.glob('*.csv'))
    price_tickers = parquet_tickers | csv_tickers

    # Filter to EU tickers
    eu_tickers = [t for t in price_tickers if any(t.endswith(ext) for ext in EU_EXTENSIONS)]

    # Load cache
    cache = load_cache()

    # Filter to missing, likely equities
    to_fetch = []
    for ticker in eu_tickers:
        if ticker not in cache:
            if is_likely_equity(ticker):
                to_fetch.append(ticker)

    # Sort for consistent ordering
    to_fetch.sort()

    print(f"EU tickers with price data: {len(eu_tickers)}")
    print(f"Already in cache: {sum(1 for t in eu_tickers if t in cache)}")
    print(f"Filtered out (non-equity patterns): {len([t for t in eu_tickers if t not in cache and not is_likely_equity(t)])}")
    print(f"To fetch: {len(to_fetch)}")
    print()

    if not to_fetch:
        print("Nothing to fetch!")
        return

    # Fetch
    successful = 0
    failed = 0
    consecutive_failures = 0

    for i, ticker in enumerate(to_fetch):
        print(f"[{i+1}/{len(to_fetch)}] {ticker}...", end=" ", flush=True)

        mktcap, name, extra = fetch_market_cap(ticker)

        if mktcap:
            cache[ticker] = {
                "market_cap": mktcap,
                "name": name,
                "currency": extra,
                "fetched_at": datetime.now().isoformat(),
                "source": "yfinance"
            }
            successful += 1
            consecutive_failures = 0
            print(f"OK - {name[:30] if name else 'N/A'} - ${mktcap:,.0f}")
            time.sleep(DELAY_BETWEEN_TICKERS)
        else:
            failed += 1
            consecutive_failures += 1
            if name:
                print(f"NO MKTCAP - {name[:30]}")
            else:
                print(f"FAILED - {extra}")

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n{MAX_CONSECUTIVE_FAILURES} consecutive failures - backing off {BACKOFF_DELAY}s...")
                save_cache(cache)
                time.sleep(BACKOFF_DELAY)
                consecutive_failures = 0
            else:
                time.sleep(DELAY_ON_FAILURE)

        # Save periodically
        if (i + 1) % BATCH_SIZE == 0:
            save_cache(cache)
            print(f"\n--- Checkpoint: {successful} OK, {failed} failed, {len(cache)} total ---\n")

    save_cache(cache)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "N/A")
    print(f"Total in cache: {len(cache)}")

    eu_count = sum(1 for t in cache if any(t.endswith(ext) for ext in EU_EXTENSIONS))
    print(f"EU tickers in cache: {eu_count}")


if __name__ == "__main__":
    main()
