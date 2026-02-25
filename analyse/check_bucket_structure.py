"""
Check the structure of the GCS bucket and list all available tickers
"""

import os
from google.cloud import storage
from collections import defaultdict

# Set credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json'

def analyze_bucket():
    """Analyze bucket structure and list all tickers"""

    client = storage.Client(project='ignition-ki-csv-storage')
    bucket_name = 'ignition-ki-csv-data-2025-user123'
    bucket = client.bucket(bucket_name)

    print("="*80)
    print(f"ANALYZING BUCKET: {bucket_name}")
    print("="*80)

    # Categorize files
    root_files = []
    ticker_files = defaultdict(list)
    other_files = []

    blobs = client.list_blobs(bucket_name)

    for blob in blobs:
        name = blob.name

        # Skip directories
        if name.endswith('/'):
            continue

        # Categorize by location
        if name.startswith('tickers/'):
            # File in tickers/ subdirectory
            ticker_name = name.replace('tickers/', '')
            if ticker_name.endswith('.csv'):
                ticker = ticker_name.replace('.csv', '').replace('_full_history', '').split('_')[0]
                ticker_files[ticker].append(name)
        elif '/' not in name and name.endswith('.csv'):
            # CSV file in root
            if '_full_history.csv' in name:
                ticker = name.replace('_full_history.csv', '')
                root_files.append((ticker, name))
            else:
                ticker = name.replace('.csv', '').split('_')[0]
                if ticker.upper() == ticker and len(ticker) <= 5:  # Likely a ticker symbol
                    root_files.append((ticker, name))
                else:
                    other_files.append(name)
        else:
            other_files.append(name)

    # Report findings
    print("\nFILES IN ROOT DIRECTORY:")
    print("-" * 40)
    if root_files:
        for ticker, filename in sorted(root_files)[:10]:
            print(f"  {ticker:6s} -> {filename}")
        if len(root_files) > 10:
            print(f"  ... and {len(root_files) - 10} more")
        print(f"\nTotal in root: {len(root_files)} ticker files")
    else:
        print("  No ticker files found in root")

    print("\nFILES IN tickers/ DIRECTORY:")
    print("-" * 40)
    if ticker_files:
        ticker_list = sorted(ticker_files.keys())
        for ticker in ticker_list[:10]:
            files = ticker_files[ticker]
            print(f"  {ticker:6s} -> {', '.join(files)}")
        if len(ticker_list) > 10:
            print(f"  ... and {len(ticker_list) - 10} more tickers")
        print(f"\nTotal in tickers/: {len(ticker_list)} unique tickers")
    else:
        print("  No files found in tickers/ directory")

    print("\nOTHER FILES:")
    print("-" * 40)
    if other_files:
        for f in other_files[:5]:
            print(f"  {f}")
        if len(other_files) > 5:
            print(f"  ... and {len(other_files) - 5} more")
    else:
        print("  No other files")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    all_tickers = set()
    for ticker, _ in root_files:
        all_tickers.add(ticker)
    all_tickers.update(ticker_files.keys())

    print(f"Total unique tickers found: {len(all_tickers)}")
    print(f"  - In root directory: {len(root_files)}")
    print(f"  - In tickers/ directory: {len(ticker_files)}")

    # Sample tickers
    print(f"\nSample tickers: {sorted(list(all_tickers))[:20]}")

    return sorted(list(all_tickers))

def test_data_loading():
    """Test loading data from different locations"""

    print("\n" + "="*80)
    print("TESTING DATA LOADING")
    print("="*80)

    from cloud_market_analysis import CloudMarketAnalyzer

    analyzer = CloudMarketAnalyzer(
        project_id='ignition-ki-csv-storage',
        credentials_path=r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json',
        use_bigquery=False  # Just test GCS loading
    )

    # Test loading tickers
    print("\nLoading tickers from GCS...")
    tickers = analyzer.load_tickers_from_gcs()
    print(f"CloudMarketAnalyzer found {len(tickers)} tickers")

    if tickers:
        print(f"First 10: {tickers[:10]}")

        # Test loading data for first ticker
        test_ticker = tickers[0]
        print(f"\nTesting data load for {test_ticker}...")

        result = analyzer._process_single_ticker_optimized(test_ticker)
        if result is not None:
            print(f"  [SUCCESS] Loaded {len(result)} rows for {test_ticker}")
            print(f"  Date range: {result['date'].min()} to {result['date'].max()}")
        else:
            print(f"  [FAILED] Could not load data for {test_ticker}")

if __name__ == "__main__":
    # Analyze bucket structure
    all_tickers = analyze_bucket()

    # Test data loading
    test_data_loading()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)