#!/usr/bin/env python3
"""
Download market data with adjusted close prices using yfinance.

This script downloads historical price data including adj_close (split-adjusted)
which is critical for accurate historical market cap calculations.

Usage:
    python scripts/download_with_adj_close.py --tickers AAPL,MSFT,GOOGL
    python scripts/download_with_adj_close.py --tickers-file tickers_6yr_300.txt
    python scripts/download_with_adj_close.py --ticker AAPL --start-date 2018-01-01
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("WARNING: yfinance not installed. Install with: pip install yfinance")


def download_ticker_with_adj_close(
    ticker: str,
    start_date: str = '2015-01-01',
    end_date: Optional[str] = None,
    output_dir: str = 'data/raw'
) -> bool:
    """
    Download ticker data with adjusted close from Yahoo Finance.

    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), default=today
        output_dir: Output directory path

    Returns:
        True if successful, False otherwise
    """
    if not YFINANCE_AVAILABLE:
        print(f"[FAILED] {ticker}: yfinance not available")
        return False

    try:
        # Download data using yfinance
        print(f"\nDownloading {ticker} from Yahoo Finance...")
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False  # Keep both Close and Adj Close
        )

        if df.empty:
            print(f"[FAILED] {ticker}: No data returned")
            return False

        # Flatten MultiIndex columns if present (yfinance returns MultiIndex for single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Reset index to make date a column
        df = df.reset_index()

        # Rename columns to lowercase standard format
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        })

        # Select only required columns
        df = df[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Save to CSV (preserving adj_close)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        csv_path = output_path / f"{ticker}.csv"
        df.to_csv(csv_path, index=False)

        print(f"[SUCCESS] {ticker}: Saved {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()})")
        print(f"          Columns: {', '.join(map(str, df.columns))}")
        print(f"          File: {csv_path}")

        return True

    except Exception as e:
        print(f"[FAILED] {ticker}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download market data with adjusted close prices")
    parser.add_argument('--ticker', type=str, help='Single ticker to download')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers')
    parser.add_argument('--tickers-file', type=str, help='File containing list of tickers (one per line)')
    parser.add_argument('--start-date', type=str, default='2015-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD), default=today')
    parser.add_argument('--output-dir', type=str, default='data/raw', help='Output directory')

    args = parser.parse_args()

    # Determine ticker list
    tickers = []

    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    elif args.tickers_file:
        with open(args.tickers_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        print("ERROR: Must specify --ticker, --tickers, or --tickers-file")
        sys.exit(1)

    if not YFINANCE_AVAILABLE:
        print("ERROR: yfinance not installed")
        print("Install with: pip install yfinance")
        sys.exit(1)

    print("=" * 80)
    print("Download Market Data with Adjusted Close (Yahoo Finance)")
    print("=" * 80)
    print(f"Tickers: {len(tickers)}")
    print(f"Date Range: {args.start_date} to {args.end_date or 'today'}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 80)

    # Download each ticker
    successful = []
    failed = []

    for ticker in tickers:
        if download_ticker_with_adj_close(ticker, args.start_date, args.end_date, args.output_dir):
            successful.append(ticker)
        else:
            failed.append(ticker)

    # Summary
    print("\n" + "=" * 80)
    print("Download Summary")
    print("=" * 80)
    print(f"Successful: {len(successful)}/{len(tickers)}")
    if successful:
        print(f"  {', '.join(successful[:10])}")
        if len(successful) > 10:
            print(f"  ... and {len(successful) - 10} more")

    if failed:
        print(f"\nFailed: {len(failed)}")
        print(f"  {', '.join(failed[:10])}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    print("\n[OK] Download complete!")
    print("These files now have adj_close column for accurate historical market cap calculations.")


if __name__ == '__main__':
    main()
