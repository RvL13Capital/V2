#!/usr/bin/env python3
"""
Low-Memory Bulk Data Fetcher
============================

Streaming architecture - processes one ticker at a time, writes immediately,
discards data after save. Memory usage: O(1) regardless of universe size.

Usage:
    python scripts/fetch_universe_data.py --region us --workers 5
    python scripts/fetch_universe_data.py --region eu --workers 5
"""

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance")
    sys.exit(1)


def download_one(ticker: str, start: str, out_dir: Path) -> tuple:
    """Download single ticker. Returns (ticker, status, rows)."""
    csv_path = out_dir / f"{ticker}.csv"

    # Skip existing
    if csv_path.exists():
        return (ticker, 'skip', 0)

    try:
        time.sleep(0.2)  # Rate limit (200ms between requests)
        df = yf.download(ticker, start=start, progress=False, auto_adjust=False, threads=False)

        if df.empty or len(df) < 100:
            return (ticker, 'fail', 0)

        # Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
        })

        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if 'adj_close' in df.columns:
            cols.append('adj_close')
        df = df[cols]
        df['ticker'] = ticker

        # Write immediately, then discard
        df.to_csv(csv_path, index=False)
        rows = len(df)
        del df  # Free memory

        return (ticker, 'ok', rows)
    except:
        return (ticker, 'fail', 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', choices=['us', 'eu', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--start-date', default='2018-01-01')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    out_dir = Path('data/raw')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tickers
    tickers = []
    if args.region in ['us', 'all'] and Path('data/us_tickers.txt').exists():
        tickers += Path('data/us_tickers.txt').read_text().strip().split('\n')
    if args.region in ['eu', 'all'] and Path('data/eu_tickers.txt').exists():
        tickers += Path('data/eu_tickers.txt').read_text().strip().split('\n')

    tickers = [t.strip() for t in tickers if t.strip()]
    if args.limit:
        tickers = tickers[:args.limit]

    print(f"Fetching {len(tickers)} tickers, {args.workers} workers")

    ok = skip = fail = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_one, t, args.start_date, out_dir): t for t in tickers}

        for i, f in enumerate(as_completed(futures)):
            ticker, status, rows = f.result()

            if status == 'ok':
                ok += 1
            elif status == 'skip':
                skip += 1
            else:
                fail += 1

            # Progress every 100
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(tickers) - i - 1) / rate / 60
                print(f"{i+1}/{len(tickers)} | OK:{ok} Skip:{skip} Fail:{fail} | {rate:.1f}/s | ETA:{eta:.0f}m")

    print(f"\nDone! OK:{ok} Skip:{skip} Fail:{fail}")
    print(f"Files: {len(list(out_dir.glob('*.csv')))} CSV")


if __name__ == '__main__':
    main()
