"""
Build US Ticker List from GCS
Filters to clean US tickers only (no international, no subfolders, no date suffixes)
"""
import os
import re
from pathlib import Path

# Set credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\AI Infra\AIv6\gcs_credentials.json'

from google.cloud import storage

def is_clean_us_ticker(name: str) -> bool:
    """Check if ticker is a clean US ticker."""
    # Remove tickers/ prefix and file extension
    ticker = name.replace('tickers/', '')
    ticker = ticker.replace('.parquet', '').replace('.csv', '')

    # Exclude international tickers (have dots like .KL, .L, .HK)
    if '.' in ticker:
        return False

    # Exclude subfolders (have /)
    if '/' in ticker:
        return False

    # Exclude date suffixes (like _20250928)
    if re.search(r'_\d{8}$', ticker):
        return False

    # Exclude empty or underscore-prefixed
    if not ticker or ticker.startswith('_'):
        return False

    # Exclude fund tickers (5+ characters ending in X) - optional
    # if len(ticker) >= 5 and ticker.endswith('X'):
    #     return False

    return True

def main():
    print("Connecting to GCS...")
    client = storage.Client(project='ignition-ki-csv-storage')
    bucket = client.bucket('ignition-ki-csv-data-2025-user123')

    print("Listing all files in tickers/...")
    blobs = list(bucket.list_blobs(prefix='tickers/'))
    print(f"Total files: {len(blobs)}")

    # Extract unique clean US tickers
    us_tickers = set()
    for blob in blobs:
        if is_clean_us_ticker(blob.name):
            ticker = blob.name.replace('tickers/', '')
            ticker = ticker.replace('.parquet', '').replace('.csv', '')
            us_tickers.add(ticker)

    # Sort alphabetically
    us_tickers = sorted(us_tickers)

    print(f"\nClean US tickers found: {len(us_tickers)}")
    print(f"Sample: {us_tickers[:20]}")

    # Save to file
    output_path = Path(__file__).parent.parent / 'output' / 'us_tickers.txt'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for ticker in us_tickers:
            f.write(f"{ticker}\n")

    print(f"\nSaved to: {output_path}")
    print(f"Total tickers: {len(us_tickers)}")

if __name__ == '__main__':
    main()
