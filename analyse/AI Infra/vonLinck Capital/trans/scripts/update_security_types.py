#!/usr/bin/env python3
"""
Update Security Types in Market Cap Cache
==========================================

This script:
1. Reads the existing market cap cache
2. Fetches quoteType from yfinance for each ticker
3. Updates the cache with security_type field
4. Saves equity-only ticker lists

Usage:
    python scripts/update_security_types.py --region eu
    python scripts/update_security_types.py --region us
    python scripts/update_security_types.py --region all

Author: TRANS System
Date: January 2026
"""

import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_FILE = PROJECT_ROOT / 'data' / 'market_cap_cache.json'
CACHE_FILE_V2 = PROJECT_ROOT / 'data' / 'cache' / 'market_cap' / 'market_cap_cache.json'


def load_cache(cache_path: Path) -> Dict:
    """Load existing cache."""
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict, cache_path: Path):
    """Save updated cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    logger.info(f"Saved cache to {cache_path}")


def get_security_type(ticker: str, max_retries: int = 2) -> Optional[str]:
    """
    Get security type (quoteType) from yfinance.

    Returns: 'EQUITY', 'MUTUALFUND', 'ETF', etc. or None on failure
    """
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info
            quote_type = info.get('quoteType', None)
            if quote_type:
                return quote_type
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            continue
    return None


def batch_get_security_types(
    tickers: List[str],
    batch_size: int = 10,
    delay_between_batches: float = 1.0,
    max_workers: int = 4,
    cache: Optional[Dict] = None,
    cache_path: Optional[Path] = None,
    save_every: int = 100
) -> Dict[str, str]:
    """
    Get security types for multiple tickers with rate limiting.

    Supports incremental saving to prevent data loss on rate limits.

    Returns: Dict mapping ticker -> security_type
    """
    results = {}
    total = len(tickers)
    processed = 0
    last_save = 0

    for i in range(0, total, batch_size):
        batch = tickers[i:i + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(get_security_type, ticker): ticker
                for ticker in batch
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        results[ticker] = result
                        # Update cache incrementally
                        if cache is not None and ticker in cache:
                            cache[ticker]['security_type'] = result
                except Exception as e:
                    logger.debug(f"{ticker}: Error - {e}")

        processed += len(batch)
        if processed % 100 == 0 or processed == total:
            logger.info(f"Progress: {processed}/{total} ({len(results)} types found)")

        # Incremental save every N tickers
        if cache is not None and cache_path is not None:
            if processed - last_save >= save_every:
                save_cache(cache, cache_path)
                logger.info(f"Incremental save at {processed} tickers")
                last_save = processed

        # Rate limit between batches
        if i + batch_size < total:
            time.sleep(delay_between_batches)

    return results


def update_cache_with_security_types(
    cache: Dict,
    security_types: Dict[str, str]
) -> Dict:
    """Update cache entries with security_type field."""
    updated = 0

    for ticker, sec_type in security_types.items():
        if ticker in cache:
            cache[ticker]['security_type'] = sec_type
            updated += 1
        else:
            # Create minimal entry for tickers not in cache
            cache[ticker] = {
                'market_cap': None,
                'security_type': sec_type,
                'fetched_at': datetime.now().isoformat(),
                'source': 'security_type_scan'
            }
            updated += 1

    logger.info(f"Updated {updated} cache entries with security_type")
    return cache


def get_tickers_for_region(region: str) -> Set[str]:
    """Get tickers for a region from ticker files."""
    tickers = set()

    if region in ['eu', 'all']:
        eu_file = PROJECT_ROOT / 'data' / 'eu_tickers.txt'
        if eu_file.exists():
            with open(eu_file, 'r') as f:
                tickers.update(t.strip() for t in f if t.strip())

    if region in ['us', 'all']:
        us_file = PROJECT_ROOT / 'data' / 'us_tickers.txt'
        if us_file.exists():
            with open(us_file, 'r') as f:
                tickers.update(t.strip() for t in f if t.strip())

    return tickers


def filter_equity_tickers(cache: Dict, tickers: List[str]) -> List[str]:
    """Filter tickers to only include equities."""
    equity_types = {'EQUITY', 'Common Stock', ''}

    equities = []
    non_equities = []
    unknown = []

    for ticker in tickers:
        if ticker in cache and 'security_type' in cache[ticker]:
            sec_type = cache[ticker]['security_type']
            if sec_type in equity_types:
                equities.append(ticker)
            else:
                non_equities.append((ticker, sec_type))
        else:
            unknown.append(ticker)

    logger.info(f"Equities: {len(equities)}")
    logger.info(f"Non-equities: {len(non_equities)}")
    logger.info(f"Unknown (not in cache): {len(unknown)}")

    if non_equities:
        logger.info("Non-equity breakdown:")
        type_counts = {}
        for t, st in non_equities:
            type_counts[st] = type_counts.get(st, 0) + 1
        for st, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {st}: {count}")

    return equities


def save_equity_tickers(equities: List[str], region: str):
    """Save equity-only ticker list."""
    output_file = PROJECT_ROOT / 'data' / f'{region}_tickers_equity_only.txt'
    with open(output_file, 'w') as f:
        for ticker in sorted(equities):
            f.write(ticker + '\n')
    logger.info(f"Saved {len(equities)} equity tickers to {output_file}")


def save_non_equity_list(cache: Dict, region: str, tickers: Set[str]):
    """Save list of non-equity tickers for reference."""
    equity_types = {'EQUITY', 'Common Stock', '', None}

    non_equities = []
    for ticker in tickers:
        if ticker in cache and 'security_type' in cache[ticker]:
            sec_type = cache[ticker]['security_type']
            if sec_type not in equity_types:
                non_equities.append((ticker, sec_type))

    if non_equities:
        output_file = PROJECT_ROOT / 'data' / f'{region}_non_equity_tickers.txt'
        with open(output_file, 'w') as f:
            f.write("# Non-equity tickers (funds, ETFs, etc.)\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total: {len(non_equities)}\n\n")
            for ticker, sec_type in sorted(non_equities):
                f.write(f"{ticker},{sec_type}\n")
        logger.info(f"Saved {len(non_equities)} non-equity tickers to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Update security types in market cap cache')
    parser.add_argument('--region', choices=['eu', 'us', 'all'], default='eu',
                        help='Region to process (default: eu)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for API calls (default: 10)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between batches in seconds (default: 1.0)')
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip fetching, just filter existing cache')
    parser.add_argument('--only-missing', action='store_true',
                        help='Only fetch security_type for tickers missing it')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("UPDATE SECURITY TYPES IN MARKET CAP CACHE")
    logger.info("=" * 60)

    # Load cache
    cache = load_cache(CACHE_FILE)
    logger.info(f"Loaded {len(cache)} entries from cache")

    # Get tickers for region
    tickers = get_tickers_for_region(args.region)
    logger.info(f"Found {len(tickers)} tickers for region '{args.region}'")

    if not args.skip_fetch:
        # Determine which tickers need security_type
        if args.only_missing:
            tickers_to_check = [
                t for t in tickers
                if t not in cache or 'security_type' not in cache.get(t, {})
            ]
        else:
            tickers_to_check = list(tickers)

        logger.info(f"Fetching security_type for {len(tickers_to_check)} tickers...")

        if tickers_to_check:
            # Fetch security types with incremental saving
            security_types = batch_get_security_types(
                tickers_to_check,
                batch_size=args.batch_size,
                delay_between_batches=args.delay,
                cache=cache,
                cache_path=CACHE_FILE,
                save_every=100
            )

            logger.info(f"Successfully fetched {len(security_types)} security types")

            # Update cache with any remaining types not yet saved
            cache = update_cache_with_security_types(cache, security_types)

            # Final save
            save_cache(cache, CACHE_FILE)

    # Filter to equities
    logger.info("\nFiltering to equity-only tickers...")
    equities = filter_equity_tickers(cache, list(tickers))

    # Save equity-only list
    save_equity_tickers(equities, args.region)

    # Save non-equity list for reference
    save_non_equity_list(cache, args.region, tickers)

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total tickers: {len(tickers)}")
    logger.info(f"Equity tickers: {len(equities)}")
    logger.info(f"Filtered out: {len(tickers) - len(equities)}")


if __name__ == '__main__':
    main()
