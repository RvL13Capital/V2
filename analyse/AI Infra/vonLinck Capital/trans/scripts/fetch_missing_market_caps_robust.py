#!/usr/bin/env python3
"""
Robust market cap fetcher for missing EU tickers using yfinance.
Features:
- Rate limiting to avoid blocks
- Progress saving for resume capability
- Batch processing with delays
- Filters obvious non-equity instruments
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
import yfinance as yf

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MISSING_FILE = DATA_DIR / "missing_market_caps.txt"
CACHE_FILE = DATA_DIR / "market_cap_cache.json"
PROGRESS_FILE = DATA_DIR / "fetch_progress.json"

# Rate limiting config
BATCH_SIZE = 50  # Tickers per batch
DELAY_BETWEEN_TICKERS = 0.5  # seconds
DELAY_BETWEEN_BATCHES = 30  # seconds
MAX_CONSECUTIVE_FAILURES = 20  # Stop if too many failures (likely rate limited)

# Patterns to skip (ETFs, leveraged products, funds, etc.)
SKIP_PATTERNS = [
    '0P',      # Funds/ETFs (Morningstar IDs)
    '3L',      # 3x Leveraged
    '3S',      # 3x Short
    '2L',      # 2x Leveraged
    '2S',      # 2x Short
    'XACT',    # ETFs
    'UCITS',   # UCITS funds
]


def load_cache() -> dict:
    """Load existing market cap cache."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    """Save market cap cache."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def load_progress() -> dict:
    """Load fetch progress for resume."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"processed": [], "last_index": 0}


def save_progress(progress: dict):
    """Save fetch progress."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def should_skip_ticker(ticker: str) -> bool:
    """Check if ticker should be skipped (non-equity)."""
    ticker_upper = ticker.upper()
    for pattern in SKIP_PATTERNS:
        if ticker_upper.startswith(pattern):
            return True
    # Skip tickers with very long numeric prefixes (usually funds)
    if len(ticker) > 10 and ticker[:8].replace('.', '').isdigit():
        return True
    return False


def fetch_market_cap(ticker: str) -> tuple:
    """
    Fetch market cap for a single ticker.
    Returns: (market_cap, name, currency) or (None, None, None) if failed
    """
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
    print("ROBUST MARKET CAP FETCHER")
    print("=" * 60)

    # Load missing tickers
    if not MISSING_FILE.exists():
        print(f"ERROR: Missing file not found: {MISSING_FILE}")
        return

    with open(MISSING_FILE, 'r') as f:
        all_tickers = [line.strip() for line in f if line.strip()]

    print(f"Total missing tickers: {len(all_tickers)}")

    # Load cache and progress
    cache = load_cache()
    progress = load_progress()

    # Filter out already processed and skip patterns
    tickers_to_process = []
    skipped_patterns = 0
    already_in_cache = 0

    for ticker in all_tickers:
        if ticker in cache:
            already_in_cache += 1
            continue
        if ticker in progress.get("processed", []):
            continue
        if should_skip_ticker(ticker):
            skipped_patterns += 1
            continue
        tickers_to_process.append(ticker)

    print(f"Already in cache: {already_in_cache}")
    print(f"Skipped (non-equity patterns): {skipped_patterns}")
    print(f"Tickers to process: {len(tickers_to_process)}")
    print()

    if not tickers_to_process:
        print("No tickers to process!")
        return

    # Process in batches
    successful = 0
    failed = 0
    consecutive_failures = 0
    start_time = datetime.now()

    for batch_idx in range(0, len(tickers_to_process), BATCH_SIZE):
        batch = tickers_to_process[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(tickers_to_process) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} tickers) ---")

        for ticker in batch:
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
                print(f"  OK: {ticker} - {name} - ${mktcap:,.0f}")
            else:
                failed += 1
                consecutive_failures += 1
                if name:  # Got some info but no market cap
                    print(f"  --: {ticker} - {name} (no mktcap)")
                else:
                    print(f"  XX: {ticker} - {extra}")

            # Track processed
            progress["processed"].append(ticker)

            # Check for rate limiting
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n!!! {MAX_CONSECUTIVE_FAILURES} consecutive failures - likely rate limited !!!")
                print("Saving progress and exiting. Run again later to resume.")
                save_cache(cache)
                save_progress(progress)
                return

            time.sleep(DELAY_BETWEEN_TICKERS)

        # Save progress after each batch
        save_cache(cache)
        save_progress(progress)

        # Batch summary
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (successful + failed) / elapsed * 60 if elapsed > 0 else 0
        print(f"  Batch complete. Total: {successful} OK, {failed} failed ({rate:.1f}/min)")

        # Delay between batches
        if batch_idx + BATCH_SIZE < len(tickers_to_process):
            print(f"  Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%" if (successful+failed) > 0 else "N/A")
    print(f"Total in cache: {len(cache)}")

    # Count EU tickers in cache
    eu_count = sum(1 for t in cache if any(t.endswith(ext) for ext in ['.L', '.DE', '.PA', '.MI', '.AS', '.SW', '.ST', '.OL', '.VI', '.HE', '.BR', '.IR', '.MC', '.LS']))
    print(f"EU tickers in cache: {eu_count}")

    # Clear progress file on successful completion
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("Progress file cleared.")


if __name__ == "__main__":
    main()
