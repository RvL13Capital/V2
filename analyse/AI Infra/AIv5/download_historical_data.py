"""
Download Historical Data for K4 Pattern Detection
==================================================

This script downloads historical OHLCV data for all tickers in our universe
using YFinance (free, unlimited) and stores them in GCS.

Features:
- Parallel downloading (10+ tickers simultaneously)
- GCS caching to avoid re-downloading
- Progress tracking and error handling
- 9 years of historical data (2015-2024)
"""

import os
import sys
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from google.cloud import storage
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcs_credentials.json'
PROJECT_ID = os.getenv('PROJECT_ID', 'ignition-ki-csv-storage')
BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'ignition-ki-csv-data-2025-user123')
LOCAL_CACHE_DIR = Path('data/historical_cache')
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Download parameters
START_DATE = '2015-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
MAX_WORKERS = 10  # Number of parallel downloads

# Initialize GCS client
try:
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)
    USE_GCS = True
    print(f"[OK] Connected to GCS bucket: {BUCKET_NAME}")
except Exception as e:
    print(f"[WARNING] GCS not available: {e}")
    print("[INFO] Using local cache only")
    USE_GCS = False
    bucket = None

def check_gcs_exists(ticker: str) -> bool:
    """Check if ticker data already exists in GCS."""
    if not USE_GCS:
        return False

    blob_name = f"historical_data/{ticker}.parquet"
    blob = bucket.blob(blob_name)
    return blob.exists()

def save_to_gcs(ticker: str, df: pd.DataFrame) -> bool:
    """Save DataFrame to GCS."""
    if not USE_GCS:
        return False

    try:
        blob_name = f"historical_data/{ticker}.parquet"
        blob = bucket.blob(blob_name)

        # Convert DataFrame to parquet bytes
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=True)
        buffer.seek(0)

        # Upload to GCS
        blob.upload_from_file(buffer, content_type='application/octet-stream')
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to save {ticker} to GCS: {e}")
        return False

def load_from_gcs(ticker: str) -> Optional[pd.DataFrame]:
    """Load DataFrame from GCS."""
    if not USE_GCS:
        return None

    try:
        blob_name = f"historical_data/{ticker}.parquet"
        blob = bucket.blob(blob_name)

        # Download from GCS
        content = blob.download_as_bytes()
        buffer = io.BytesIO(content)
        df = pd.read_parquet(buffer)
        return df
    except Exception as e:
        return None

def download_ticker_data(ticker: str, skip_existing: bool = True) -> Tuple[str, bool, str]:
    """
    Download historical data for a single ticker.

    Args:
        ticker: Ticker symbol
        skip_existing: Skip if data already exists

    Returns:
        Tuple of (ticker, success, message)
    """
    try:
        # Check local cache first
        local_file = LOCAL_CACHE_DIR / f"{ticker}.parquet"
        if local_file.exists() and skip_existing:
            return (ticker, True, "cached_local")

        # Check GCS cache
        if USE_GCS and skip_existing and check_gcs_exists(ticker):
            return (ticker, True, "cached_gcs")

        # Download from YFinance
        stock = yf.Ticker(ticker)
        df = stock.history(start=START_DATE, end=END_DATE, auto_adjust=True)

        # Check if we got data
        if df.empty or len(df) < 250:  # At least 1 year of data
            return (ticker, False, "insufficient_data")

        # Add ticker column
        df['ticker'] = ticker

        # Reset index to have date as column
        df.reset_index(inplace=True)

        # Save locally
        df.to_parquet(local_file, index=False)

        # Save to GCS
        if USE_GCS:
            save_to_gcs(ticker, df)

        return (ticker, True, f"downloaded_{len(df)}_days")

    except Exception as e:
        return (ticker, False, f"error: {str(e)[:50]}")

def download_batch(tickers: List[str], max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    """
    Download data for multiple tickers in parallel.

    Args:
        tickers: List of ticker symbols
        max_workers: Number of parallel downloads

    Returns:
        DataFrame with download results
    """
    results = []
    total = len(tickers)

    print(f"\n[Download] Starting parallel download of {total} tickers...")
    print(f"  Workers: {max_workers}")
    print(f"  Date range: {START_DATE} to {END_DATE}")
    print(f"  Cache: Local + {'GCS' if USE_GCS else 'Local only'}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = {executor.submit(download_ticker_data, ticker): ticker
                  for ticker in tickers}

        # Process completed downloads
        completed = 0
        success_count = 0
        cached_count = 0
        failed_count = 0

        for future in as_completed(futures):
            ticker, success, message = future.result()
            completed += 1

            # Track statistics
            if success:
                success_count += 1
                if 'cached' in message:
                    cached_count += 1
            else:
                failed_count += 1

            # Save result
            results.append({
                'ticker': ticker,
                'success': success,
                'message': message,
                'timestamp': datetime.now()
            })

            # Progress update every 10%
            if completed % max(1, total // 10) == 0 or completed == total:
                pct = completed / total * 100
                print(f"  Progress: {completed}/{total} ({pct:.1f}%) | "
                      f"Success: {success_count} | Cached: {cached_count} | Failed: {failed_count}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def main():
    """Main function to download all historical data."""
    print("=" * 70)
    print("HISTORICAL DATA DOWNLOAD FOR K4 PATTERN DETECTION")
    print("=" * 70)

    # 1. Load ticker universe
    universe_file = Path('output/ticker_universe/us_microcap_universe.csv')
    if not universe_file.exists():
        print("[ERROR] Ticker universe not found. Run build_ticker_universe.py first.")
        sys.exit(1)

    ticker_df = pd.read_csv(universe_file)
    tickers = ticker_df['ticker'].tolist()

    print(f"\n[Universe] Loaded {len(tickers)} tickers")
    print(f"  Market cap range: ${ticker_df['market_cap_millions'].min():.1f}M - "
          f"${ticker_df['market_cap_millions'].max():.1f}M")

    # 2. Download data in batches
    batch_size = 100
    all_results = []

    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch_tickers)} tickers...")
        batch_results = download_batch(batch_tickers)
        all_results.append(batch_results)

    # 3. Combine results
    results_df = pd.concat(all_results, ignore_index=True)

    # 4. Save download report
    report_file = Path('output/download_report.csv')
    results_df.to_csv(report_file, index=False)

    # 5. Display summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    success_df = results_df[results_df['success']]
    failed_df = results_df[~results_df['success']]

    print(f"Total tickers processed: {len(results_df)}")
    print(f"Successfully downloaded: {len(success_df)} ({len(success_df)/len(results_df)*100:.1f}%)")
    print(f"Failed downloads: {len(failed_df)}")

    # Show failure reasons
    if len(failed_df) > 0:
        print("\nFailure reasons:")
        for reason, count in failed_df['message'].value_counts().head(10).items():
            print(f"  {reason}: {count}")

    # Calculate data coverage
    downloaded_files = list(LOCAL_CACHE_DIR.glob("*.parquet"))
    print(f"\n[Data Coverage]")
    print(f"  Local files: {len(downloaded_files)}")
    if USE_GCS:
        print(f"  GCS bucket: {BUCKET_NAME}")

    # Sample data quality check
    if len(downloaded_files) > 0:
        sample_file = downloaded_files[0]
        sample_df = pd.read_parquet(sample_file)
        print(f"\n[Sample Data] {sample_file.stem}")
        print(f"  Date range: {sample_df['Date'].min():%Y-%m-%d} to {sample_df['Date'].max():%Y-%m-%d}")
        print(f"  Trading days: {len(sample_df)}")
        print(f"  Years of data: {(sample_df['Date'].max() - sample_df['Date'].min()).days / 365:.1f}")

    print(f"\nDownload report saved to: {report_file}")
    print("\n[Next Step] Run pattern detection: python detect_patterns_batch.py")

    return results_df

if __name__ == "__main__":
    results = main()