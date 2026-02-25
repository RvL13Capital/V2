"""
Debug cached ticker data to understand date issues
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Check a few problematic tickers
problematic_tickers = ['0023.KL', '0037.KL', '0096.KL']
cache_dir = Path('data/unused_tickers_cache')

print("=" * 60)
print("CACHED TICKER DATA ANALYSIS")
print("=" * 60)

for ticker in problematic_tickers:
    cache_file = cache_dir / f"{ticker}.parquet"

    print(f"\n{ticker}:")
    if not cache_file.exists():
        print(f"  [ERROR] Cache file not found")
        continue

    try:
        df = pd.read_parquet(cache_file)
        print(f"  [OK] Loaded: {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")

        # Check date column
        if 'date' in df.columns:
            print(f"  Date column type: {df['date'].dtype}")
            null_dates = df['date'].isna().sum()
            print(f"  Null dates: {null_dates}")
            if null_dates > 0:
                print(f"  [WARNING] Found {null_dates} null dates!")

            # Check date range
            if len(df) > 0 and null_dates < len(df):
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

                # Check for specific dates from metadata
                test_dates = [
                    pd.Timestamp('2004-12-31', tz='UTC'),  # 0023.KL snapshot
                    pd.Timestamp('2008-12-03', tz='UTC'),  # 0037.KL snapshot
                ]

                for test_date in test_dates:
                    if test_date >= df['date'].min() and test_date <= df['date'].max():
                        matching = df[df['date'] == test_date]
                        if len(matching) == 0:
                            print(f"  [WARNING] Date {test_date.date()} not found in data!")
                        else:
                            print(f"  [OK] Date {test_date.date()} found")

        elif df.index.name == 'date':
            print(f"  Date as index type: {df.index.dtype}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
        else:
            print(f"  [ERROR] No date column found!")

        # Show first few rows
        print(f"  First 3 rows:")
        print(df.head(3).to_string())

    except Exception as e:
        print(f"  [ERROR] Error loading: {str(e)}")

# Compare with a US ticker that should work
print("\n" + "=" * 60)
print("COMPARISON: US TICKER (AAGC)")
print("=" * 60)

us_cache = cache_dir / "AAGC.parquet"
if us_cache.exists():
    df = pd.read_parquet(us_cache)
    print(f"  Loaded: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    if 'date' in df.columns:
        print(f"  Date column type: {df['date'].dtype}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  First 3 rows:")
    print(df.head(3).to_string())