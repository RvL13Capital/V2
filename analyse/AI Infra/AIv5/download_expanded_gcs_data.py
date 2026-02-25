"""
Download Expanded Historical Data from GCS for Training
========================================================
Downloads a larger set of ticker data from GCS bucket to find more K4 patterns
and improve model training with expanded dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Set up environment variables for GCS
os.environ['PROJECT_ID'] = 'ignition-ki-csv-storage'
os.environ['GCS_BUCKET_NAME'] = 'ignition-ki-csv-data-2025-user123'

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_acquisition.storage.storage_manager import StorageManager
from data_acquisition.sources.yfinance_downloader import YFinanceDownloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class GCSDataExpander:
    """
    Downloads and manages expanded ticker dataset from GCS.
    """

    def __init__(self, local_cache_dir: str = 'data/expanded_gcs_cache'):
        self.storage_manager = StorageManager()
        self.downloader = YFinanceDownloader()
        self.cache_dir = Path(local_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ticker_dir = self.cache_dir / 'tickers'
        self.ticker_dir.mkdir(parents=True, exist_ok=True)

    def list_available_tickers_gcs(self, prefix: str = 'ticker/') -> List[str]:
        """
        List all available tickers in GCS bucket.
        """
        try:
            from google.cloud import storage

            client = storage.Client(project=os.environ['PROJECT_ID'])
            bucket = client.bucket(os.environ['GCS_BUCKET_NAME'])

            # List all blobs with the ticker prefix
            blobs = bucket.list_blobs(prefix=prefix)

            tickers = []
            for blob in blobs:
                # Extract ticker symbol from path
                # Format: ticker/AAPL.parquet or ticker/AAPL.csv
                if '/' in blob.name:
                    filename = blob.name.split('/')[-1]
                    if filename.endswith(('.parquet', '.csv')):
                        ticker = filename.replace('.parquet', '').replace('.csv', '')
                        tickers.append(ticker)

            return sorted(list(set(tickers)))

        except Exception as e:
            logger.error(f"Error listing GCS tickers: {e}")
            logger.info("Falling back to local storage manager")
            return self._get_tickers_from_storage_manager()

    def _get_tickers_from_storage_manager(self) -> List[str]:
        """
        Get tickers using storage manager as fallback.
        """
        # Try to get a comprehensive ticker list
        ticker_candidates = []

        # Common ticker lists
        sp500_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'JPM', 'JNJ', 'V', 'UNH', 'HD', 'MA', 'PG', 'DIS', 'BAC', 'XOM',
            'CVX', 'ABBV', 'PFE', 'KO', 'WMT', 'AVGO', 'MRK', 'PEP', 'TMO',
            'COST', 'ABT', 'ORCL', 'ACN', 'NKE', 'LLY', 'MCD', 'ADBE', 'CRM'
        ]

        # Micro/small cap examples
        small_cap = [
            'SOUN', 'MULN', 'FFIE', 'NKLA', 'RIDE', 'GOEV', 'WKHS', 'LCID',
            'RIVN', 'FSR', 'PTRA', 'ARVL', 'REE', 'CHPT', 'BLNK', 'EVGO'
        ]

        ticker_candidates.extend(sp500_tickers)
        ticker_candidates.extend(small_cap)

        return ticker_candidates

    def download_ticker_data(self, ticker: str, start_date: str = '2018-01-01',
                           end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Download ticker data from GCS or fallback to yfinance.
        """
        # Check local cache first
        cache_file = self.ticker_dir / f"{ticker}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.debug(f"Loaded {ticker} from cache")
                return df
            except:
                pass

        # Try storage manager (GCS)
        try:
            df = self.storage_manager.load_ticker_data(ticker, use_cache=True)
            if df is not None and len(df) > 100:  # Ensure sufficient data
                # Save to cache
                df.to_parquet(cache_file)
                return df
        except Exception as e:
            logger.debug(f"Storage manager failed for {ticker}: {e}")

        # Fallback to yfinance
        try:
            df = self.downloader.download(ticker, start_date, end_date)
            if df is not None and len(df) > 100:
                # Save to cache
                df.to_parquet(cache_file)
                return df
        except Exception as e:
            logger.debug(f"YFinance failed for {ticker}: {e}")

        return None

    def download_expanded_dataset(self, num_tickers: int = 1000,
                                 start_date: str = '2018-01-01',
                                 max_workers: int = 10) -> pd.DataFrame:
        """
        Download expanded dataset with many tickers.
        """
        logger.info("="*70)
        logger.info("DOWNLOADING EXPANDED TICKER DATASET FROM GCS")
        logger.info("="*70)

        # Get ticker list
        logger.info("Getting available tickers...")
        all_tickers = self.list_available_tickers_gcs()

        if not all_tickers:
            logger.warning("No tickers from GCS, using fallback list")
            all_tickers = self._get_tickers_from_storage_manager()

        # Limit to requested number
        tickers_to_download = all_tickers[:num_tickers]
        logger.info(f"Will download {len(tickers_to_download)} tickers")

        # Download in parallel
        successful_downloads = []
        failed_tickers = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.download_ticker_data, ticker, start_date): ticker
                for ticker in tickers_to_download
            }

            with tqdm(total=len(futures), desc="Downloading tickers") as pbar:
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        df = future.result()
                        if df is not None:
                            successful_downloads.append(ticker)
                        else:
                            failed_tickers.append(ticker)
                    except Exception as e:
                        logger.debug(f"Error downloading {ticker}: {e}")
                        failed_tickers.append(ticker)
                    pbar.update(1)

        logger.info(f"\nDownload complete:")
        logger.info(f"  Successful: {len(successful_downloads)}")
        logger.info(f"  Failed: {len(failed_tickers)}")

        # Create summary
        summary = pd.DataFrame({
            'ticker': successful_downloads,
            'status': 'downloaded',
            'data_points': [
                len(pd.read_parquet(self.ticker_dir / f"{t}.parquet"))
                for t in successful_downloads
            ]
        })

        # Save summary
        summary_file = self.cache_dir / f'download_summary_{datetime.now():%Y%m%d_%H%M%S}.csv'
        summary.to_csv(summary_file, index=False)
        logger.info(f"\nSummary saved to: {summary_file}")

        return summary

    def get_ticker_statistics(self) -> pd.DataFrame:
        """
        Get statistics about downloaded tickers.
        """
        ticker_files = list(self.ticker_dir.glob("*.parquet"))

        stats = []
        for file in ticker_files:
            try:
                df = pd.read_parquet(file)
                ticker = file.stem

                # Calculate statistics
                date_range = df.index.max() - df.index.min()
                avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
                price_range = df['close'].max() - df['close'].min() if 'close' in df.columns else 0

                stats.append({
                    'ticker': ticker,
                    'data_points': len(df),
                    'date_range_days': date_range.days if hasattr(date_range, 'days') else 0,
                    'avg_volume': avg_volume,
                    'price_range': price_range,
                    'min_date': df.index.min(),
                    'max_date': df.index.max()
                })
            except:
                continue

        return pd.DataFrame(stats)

    def prepare_for_pattern_detection(self) -> List[str]:
        """
        Prepare downloaded data for pattern detection.
        """
        # Get all downloaded tickers
        ticker_files = list(self.ticker_dir.glob("*.parquet"))
        valid_tickers = []

        for file in ticker_files:
            ticker = file.stem
            try:
                df = pd.read_parquet(file)
                # Check if ticker has enough data
                if len(df) >= 252:  # At least 1 year of data
                    valid_tickers.append(ticker)
            except:
                continue

        logger.info(f"\nTickers ready for pattern detection: {len(valid_tickers)}")
        return valid_tickers


def main():
    """
    Main function to download expanded GCS data.
    """
    # Initialize expander
    expander = GCSDataExpander()

    # User input
    print("\n" + "="*70)
    print("GCS DATA EXPANSION FOR TRAINING")
    print("="*70)

    num_tickers = input("\nHow many tickers to download? (default 500): ").strip()
    num_tickers = int(num_tickers) if num_tickers else 500

    start_date = input("Start date (YYYY-MM-DD, default 2018-01-01): ").strip()
    start_date = start_date if start_date else '2018-01-01'

    print(f"\nDownloading {num_tickers} tickers from {start_date}...")

    # Download expanded dataset
    summary = expander.download_expanded_dataset(
        num_tickers=num_tickers,
        start_date=start_date,
        max_workers=10
    )

    # Show statistics
    stats = expander.get_ticker_statistics()
    if not stats.empty:
        print("\n" + "="*70)
        print("DOWNLOAD STATISTICS")
        print("="*70)
        print(f"Total tickers downloaded: {len(stats)}")
        print(f"Average data points per ticker: {stats['data_points'].mean():.0f}")
        print(f"Total data points: {stats['data_points'].sum():,}")
        print(f"Date range: {stats['min_date'].min()} to {stats['max_date'].max()}")

        # Identify high-quality tickers
        quality_tickers = stats[
            (stats['data_points'] >= 1000) &
            (stats['avg_volume'] > 1000000)
        ]
        print(f"\nHigh-quality tickers (1000+ days, 1M+ volume): {len(quality_tickers)}")

    # Prepare for pattern detection
    valid_tickers = expander.prepare_for_pattern_detection()

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Run pattern detection on expanded dataset:")
    print(f"   python src/pipeline/01_scan_patterns_batch.py --tickers {len(valid_tickers)}")
    print("\n2. Label patterns with race logic:")
    print("   python src/pipeline/02_label_snapshots_race.py")
    print("\n3. Retrain models with larger dataset:")
    print("   python src/pipeline/03_train_enhanced.py")

    # Save ticker list for pattern detection
    ticker_list_file = Path('output') / f'expanded_tickers_{datetime.now():%Y%m%d_%H%M%S}.txt'
    ticker_list_file.parent.mkdir(exist_ok=True)
    with open(ticker_list_file, 'w') as f:
        f.write('\n'.join(valid_tickers))
    print(f"\nTicker list saved to: {ticker_list_file}")

    return valid_tickers


if __name__ == "__main__":
    tickers = main()