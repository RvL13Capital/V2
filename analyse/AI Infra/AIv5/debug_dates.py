"""
Debug script to investigate date handling issues in pattern metadata
"""

import pandas as pd
import numpy as np

# Load pattern metadata
df = pd.read_parquet('output/unused_patterns/pattern_features_historical.parquet')

print("=" * 60)
print("PATTERN METADATA ANALYSIS")
print("=" * 60)
print(f"Total patterns: {len(df)}")

# Check column types
print("\nColumn data types:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

# Check date columns
date_cols = ['qualification_start', 'active_start', 'end_date', 'snapshot_date']
print("\nDate column analysis:")
for col in date_cols:
    if col in df.columns:
        null_count = df[col].isna().sum()
        print(f"\n{col}:")
        print(f"  Null count: {null_count} ({null_count/len(df)*100:.1f}%)")
        if null_count < len(df):
            print(f"  Sample values: {df[col].dropna().head(3).tolist()}")
            print(f"  Dtype: {df[col].dtype}")

# Check tickers with issues
print("\n" + "=" * 60)
print("TICKER ANALYSIS")
print("=" * 60)

# Group by ticker suffix
df['ticker_suffix'] = df['ticker'].str.extract(r'\.([A-Z]+)$', expand=False).fillna('US')
suffix_counts = df['ticker_suffix'].value_counts()
print("\nTicker suffixes:")
for suffix, count in suffix_counts.head(10).items():
    print(f"  {suffix}: {count} patterns")

# Check if certain ticker types have more date issues
print("\nSample problematic tickers:")
problematic_tickers = ['0023.KL', '0037.KL', '0096.KL', '0118.KL', '0127.KL', '0131.KL',
                      '0163.KL', '0QLT.L', '0RP6.L', '1201.KL', '1589.KL', '2038.KL']

for ticker in problematic_tickers[:5]:
    ticker_data = df[df['ticker'] == ticker]
    if not ticker_data.empty:
        print(f"\n{ticker}:")
        for col in date_cols:
            if col in ticker_data.columns:
                val = ticker_data[col].iloc[0]
                print(f"  {col}: {val} (type: {type(val).__name__})")

# Check a US ticker for comparison
us_tickers = df[df['ticker_suffix'] == 'US']['ticker'].head(5)
if len(us_tickers) > 0:
    print(f"\nComparison - US ticker ({us_tickers.iloc[0]}):")
    us_data = df[df['ticker'] == us_tickers.iloc[0]]
    for col in date_cols:
        if col in us_data.columns:
            val = us_data[col].iloc[0]
            print(f"  {col}: {val} (type: {type(val).__name__})")