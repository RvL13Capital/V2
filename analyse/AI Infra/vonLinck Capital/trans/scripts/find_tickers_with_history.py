"""
Find tickers with sufficient historical data for grid search.

This script scans the GCS bucket for tickers with at least 6 years of data
(from 2018-01-01 or earlier to present).

Usage:
    python scripts/find_tickers_with_history.py --min-years 6 --output tickers_6yr.txt --limit 300
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import DataLoader
from dotenv import load_dotenv

load_dotenv()


def check_ticker_data_range(ticker: str, data_loader: DataLoader, min_years: int = 6) -> Tuple[bool, str]:
    """
    Check if a ticker has sufficient historical data.

    Args:
        ticker: Stock ticker symbol
        data_loader: DataLoader instance
        min_years: Minimum years of data required

    Returns:
        Tuple of (has_sufficient_data: bool, reason: str)
    """
    try:
        # Try to load data
        df = data_loader.load_ticker(ticker)

        if df is None or len(df) == 0:
            return False, "No data available"

        # DataLoader returns df with date as INDEX, not column
        # Check if date is in index
        if isinstance(df.index, pd.DatetimeIndex):
            min_date = df.index.min()
            max_date = df.index.max()
        elif 'date' in df.columns:
            # Fallback: if date is a column
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            min_date = df['date'].min()
            max_date = df['date'].max()
        else:
            return False, "No date information found"

        # Calculate years of data
        years_of_data = (max_date - min_date).days / 365.25

        # Check if we have data starting from min_years ago or earlier
        cutoff_date = datetime.now() - timedelta(days=min_years * 365)

        if min_date > cutoff_date:
            return False, f"Data starts {min_date.strftime('%Y-%m-%d')} (need {cutoff_date.strftime('%Y-%m-%d')} or earlier)"

        if years_of_data < min_years:
            return False, f"Only {years_of_data:.1f} years of data"

        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        return True, f"{years_of_data:.1f} years ({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})"

    except Exception as e:
        return False, f"Error loading: {str(e)}"


def find_tickers_with_history(
    min_years: int = 6,
    limit: int = 300,
    output_file: str = "tickers_6yr.txt",
    verbose: bool = True
) -> List[str]:
    """
    Find tickers with sufficient historical data.

    Args:
        min_years: Minimum years of data required
        limit: Maximum number of tickers to find
        output_file: Output file path
        verbose: Print progress

    Returns:
        List of ticker symbols with sufficient data
    """
    data_loader = DataLoader()

    # Get list of available tickers from local data directory
    if verbose:
        print(f"Scanning local data directory for tickers with {min_years}+ years of data...")
        print(f"Target: {limit} tickers")
        print("=" * 80)

    try:
        # List all CSV and Parquet files in data/raw directory
        data_dir = Path('data/raw')
        if not data_dir.exists():
            print(f"Error: Data directory {data_dir} does not exist")
            return []

        csv_files = list(data_dir.glob('*.csv'))
        parquet_files = list(data_dir.glob('*.parquet'))
        all_files = csv_files + parquet_files

        # Extract ticker symbols from filenames (remove .csv or .parquet extension)
        available_tickers = [f.stem for f in all_files]

        if not available_tickers:
            print(f"Error: No data files found in {data_dir}")
            return []

    except Exception as e:
        print(f"Error listing tickers: {e}")
        return []

    if verbose:
        print(f"Found {len(available_tickers)} total tickers in data/raw/")
        print("Checking data availability...\n")

    valid_tickers = []
    checked = 0

    for ticker in available_tickers:
        checked += 1

        has_data, reason = check_ticker_data_range(ticker, data_loader, min_years)

        if has_data:
            valid_tickers.append(ticker)
            if verbose:
                print(f"[OK] {ticker:8s} - {reason}")

            # Stop if we've reached the limit
            if len(valid_tickers) >= limit:
                if verbose:
                    print(f"\n[OK] Found {limit} tickers! Stopping search.")
                break
        else:
            if verbose and checked % 10 == 0:
                print(f"Checked {checked}/{len(available_tickers)} tickers, found {len(valid_tickers)} valid")

    if verbose:
        print("\n" + "=" * 80)
        print(f"Summary:")
        print(f"  Total tickers checked: {checked}")
        print(f"  Valid tickers found: {len(valid_tickers)}")
        print(f"  Success rate: {len(valid_tickers)/checked*100:.1f}%")

    # Save to file
    if valid_tickers:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for ticker in valid_tickers:
                f.write(f"{ticker}\n")

        if verbose:
            print(f"\n[OK] Saved {len(valid_tickers)} tickers to {output_file}")

    return valid_tickers


def main():
    parser = argparse.ArgumentParser(description="Find tickers with sufficient historical data")
    parser.add_argument('--min-years', type=int, default=6,
                        help='Minimum years of data required (default: 6)')
    parser.add_argument('--limit', type=int, default=300,
                        help='Maximum number of tickers to find (default: 300)')
    parser.add_argument('--output', type=str, default='tickers_6yr.txt',
                        help='Output file path (default: tickers_6yr.txt)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    tickers = find_tickers_with_history(
        min_years=args.min_years,
        limit=args.limit,
        output_file=args.output,
        verbose=not args.quiet
    )

    if not tickers:
        print("No tickers found with sufficient data.")
        sys.exit(1)


if __name__ == '__main__':
    main()
