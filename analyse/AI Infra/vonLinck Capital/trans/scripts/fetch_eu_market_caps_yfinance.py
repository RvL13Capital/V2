"""
Fetch Market Cap Data for EU Tickers using yfinance
====================================================

Fetches shares_outstanding for all EU tickers in data/raw using yfinance.
Results are saved to the market cap cache for use by detect_eu_patterns.py.

Usage:
    python scripts/fetch_eu_market_caps_yfinance.py
    python scripts/fetch_eu_market_caps_yfinance.py --limit 50
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# EU market suffixes
EU_SUFFIXES = ['.DE', '.L', '.PA', '.MI', '.MC', '.AS', '.LS', '.BR', '.SW', '.ST', '.OL', '.CO', '.HE', '.IR', '.VI']


def get_eu_tickers(data_dir: Path) -> list:
    """Get all EU tickers from data/raw directory."""
    tickers = set()

    for f in data_dir.glob('*.parquet'):
        ticker = f.stem
        if any(ticker.endswith(s) for s in EU_SUFFIXES):
            tickers.add(ticker)

    for f in data_dir.glob('*.csv'):
        ticker = f.stem
        if any(ticker.endswith(s) for s in EU_SUFFIXES):
            tickers.add(ticker)

    return sorted(tickers)


def fetch_market_cap(ticker: str) -> dict:
    """Fetch market cap data for a single ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        shares = info.get('sharesOutstanding')
        market_cap = info.get('marketCap')

        if shares and market_cap:
            return {
                'ticker': ticker,
                'market_cap': float(market_cap),
                'shares_outstanding': float(shares),
                'source': 'yfinance',
                'fetched_at': datetime.now().isoformat(),
                'currency': info.get('currency', 'EUR')
            }
    except Exception as e:
        logger.debug(f"{ticker}: Error - {e}")

    return None


def load_cache(cache_file: Path) -> dict:
    """Load existing cache."""
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return {}


def save_cache(cache: dict, cache_file: Path):
    """Save cache to disk."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")


def main():
    parser = argparse.ArgumentParser(description='Fetch market cap for EU tickers using yfinance')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of tickers')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Data directory')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    parser.add_argument('--force', action='store_true', help='Force refresh even if cached')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_file = Path('data/cache/market_cap/market_cap_cache.json')

    logger.info("=" * 60)
    logger.info("Fetching Market Cap for EU Tickers (yfinance)")
    logger.info("=" * 60)

    # Get EU tickers
    tickers = get_eu_tickers(data_dir)
    if args.limit:
        tickers = tickers[:args.limit]

    logger.info(f"Found {len(tickers)} EU tickers")

    # Count by market
    for suffix in EU_SUFFIXES:
        count = len([t for t in tickers if t.endswith(suffix)])
        if count > 0:
            logger.info(f"  {suffix}: {count}")

    # Load existing cache
    cache = load_cache(cache_file)
    logger.info(f"\nExisting cache: {len(cache)} tickers")

    # Filter to only uncached tickers (unless force)
    if not args.force:
        tickers_to_fetch = [t for t in tickers if t not in cache]
        logger.info(f"Already in cache (EU): {len(tickers) - len(tickers_to_fetch)}")
        logger.info(f"Need to fetch: {len(tickers_to_fetch)}")
    else:
        tickers_to_fetch = tickers
        logger.info(f"Force refresh: fetching all {len(tickers_to_fetch)}")

    if not tickers_to_fetch:
        logger.info("All tickers already cached!")
        return

    # Estimate time (~1-2 sec per ticker with parallelization)
    est_seconds = len(tickers_to_fetch) / args.workers * 1.5
    logger.info(f"\nEstimated time: {est_seconds/60:.1f} minutes (with {args.workers} workers)")
    logger.info("Starting fetch...\n")

    # Fetch market caps in parallel
    start_time = time.time()
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_market_cap, t): t for t in tickers_to_fetch}

        for future in tqdm(as_completed(futures), total=len(tickers_to_fetch), desc="Fetching"):
            ticker = futures[future]
            try:
                result = future.result()
                if result:
                    cache[ticker] = result
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.debug(f"{ticker}: Error - {e}")
                fail_count += 1

            # Save cache periodically (every 50 tickers)
            if (success_count + fail_count) % 50 == 0:
                save_cache(cache, cache_file)

    # Final save
    save_cache(cache, cache_file)

    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Fetched: {len(tickers_to_fetch)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed/No data: {fail_count}")
    logger.info(f"Time: {total_time/60:.1f} minutes")

    # Final cache stats
    eu_in_cache = sum(1 for t in tickers if t in cache)
    eu_with_shares = sum(1 for t in tickers if t in cache and cache[t].get('shares_outstanding'))
    logger.info(f"\nEU tickers in cache: {eu_in_cache}/{len(tickers)}")
    logger.info(f"EU tickers with shares_outstanding: {eu_with_shares}/{len(tickers)}")
    logger.info(f"Cache saved to: {cache_file}")


if __name__ == '__main__':
    main()
