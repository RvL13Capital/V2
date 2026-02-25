#!/usr/bin/env python3
"""
Fetch market caps for LSE cross-listings (0XXX.L) by finding primary listing symbols.
These are foreign stocks trading on LSE's International Order Book (IOB).
"""

import json
import time
import re
from pathlib import Path
import yfinance as yf

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_FILE = DATA_DIR / "market_cap_cache.json"

# Known mappings for common cross-listings
KNOWN_MAPPINGS = {
    # Format: '0XXX.L': 'PRIMARY_SYMBOL'
}


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def extract_company_name(info: dict) -> str:
    """Extract clean company name from yfinance info."""
    name = info.get('shortName') or info.get('longName') or ''
    # Remove suffixes like "ORD SHS", "ADR", etc.
    name = re.sub(r'\s+(ORD|ADR|SHS|ORDINARY|SHARES|REPRESENTING|INC|CORP|LTD|PLC|AG|SA|NV|AB).*$', '', name, flags=re.IGNORECASE)
    return name.strip()


def find_primary_symbol(iob_ticker: str, info: dict) -> str:
    """Try to find the primary listing symbol for a cross-listed stock."""

    # Check known mappings first
    if iob_ticker in KNOWN_MAPPINGS:
        return KNOWN_MAPPINGS[iob_ticker]

    # Try to extract from info
    # Sometimes yfinance provides underlying symbol
    underlying = info.get('underlyingSymbol')
    if underlying and underlying != iob_ticker:
        return underlying

    # Try common US exchanges based on company name patterns
    name = extract_company_name(info)
    if not name:
        return None

    # Build potential symbols to try
    # 1. First word of company name (common pattern)
    words = name.upper().split()
    potential_symbols = []

    if words:
        # Try first word
        potential_symbols.append(words[0][:4])
        # Try first letters of first few words
        if len(words) >= 2:
            potential_symbols.append(''.join(w[0] for w in words[:3]))

    return potential_symbols


def fetch_market_cap_from_primary(primary_symbol: str) -> tuple:
    """Fetch market cap from primary listing."""
    try:
        stock = yf.Ticker(primary_symbol)
        info = stock.info
        mc = info.get('marketCap')
        name = info.get('shortName') or info.get('longName')
        currency = info.get('currency')
        if mc and mc > 0:
            return mc, name, currency, primary_symbol
        return None, None, None, None
    except:
        return None, None, None, None


def main():
    print("=" * 60)
    print("CROSS-LISTING MARKET CAP FETCHER")
    print("=" * 60)

    cache = load_cache()

    # Find cross-listed tickers in parquet files
    parquet_files = list(RAW_DIR.glob('*.parquet'))
    cross_listings = [f.stem for f in parquet_files
                      if f.stem.startswith('0') and f.stem.endswith('.L')
                      and f.stem not in cache]

    print(f"Cross-listings to process: {len(cross_listings)}")
    print()

    successful = 0
    failed = 0

    for i, ticker in enumerate(cross_listings):
        print(f"[{i+1}/{len(cross_listings)}] {ticker}...", end=" ", flush=True)

        try:
            # First get info from IOB listing
            iob_info = yf.Ticker(ticker).info
            company_name = extract_company_name(iob_info)

            if not company_name:
                print("NO NAME")
                failed += 1
                time.sleep(1)
                continue

            # Try to find primary symbol
            potential_symbols = find_primary_symbol(ticker, iob_info)

            if isinstance(potential_symbols, str):
                potential_symbols = [potential_symbols]
            elif not potential_symbols:
                potential_symbols = []

            # Also try searching by company name - extract likely ticker
            # Many US stocks: first word or abbreviation
            words = company_name.upper().split()
            if words:
                # Add more potential symbols
                potential_symbols.extend([
                    words[0][:5],  # First 5 chars of first word
                    words[0][:4],  # First 4 chars
                    words[0][:3],  # First 3 chars
                ])
                if len(words) >= 2:
                    potential_symbols.append(words[0][:2] + words[1][:2])  # Combo

            # Remove duplicates
            potential_symbols = list(dict.fromkeys(potential_symbols))

            # Try each potential symbol
            mc, name, currency, found_symbol = None, None, None, None
            for sym in potential_symbols[:5]:  # Limit attempts
                mc, name, currency, found_symbol = fetch_market_cap_from_primary(sym)
                if mc:
                    break
                time.sleep(0.3)

            if mc:
                cache[ticker] = {
                    "market_cap": mc,
                    "name": name,
                    "currency": currency,
                    "source": "yfinance",
                    "primary_symbol": found_symbol,
                    "is_cross_listing": True
                }
                successful += 1
                print(f"OK via {found_symbol} - {name[:25]} - ${mc:,.0f}")
            else:
                print(f"FAILED - tried {potential_symbols[:3]}")
                failed += 1

        except Exception as e:
            print(f"ERROR: {str(e)[:30]}")
            failed += 1

        time.sleep(1)

        # Save periodically
        if (i + 1) % 20 == 0:
            save_cache(cache)

    save_cache(cache)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total in cache: {len(cache)}")


if __name__ == "__main__":
    main()
