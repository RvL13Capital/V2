"""
Direct GCS Ticker Download from ignition-ki-csv-data-2025-user123/tickers
==========================================================================
Downloads ticker data directly from the GCS bucket's tickers folder.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

# Configure environment for GCS
os.environ['PROJECT_ID'] = 'ignition-ki-csv-storage'
os.environ['GCS_BUCKET_NAME'] = 'ignition-ki-csv-data-2025-user123'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectGCSDownloader:
    """
    Downloads ticker data directly from GCS bucket's tickers folder.
    """

    def __init__(self):
        self.cache_dir = Path('data/gcs_tickers_massive')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ticker_dir = self.cache_dir / 'tickers'
        self.ticker_dir.mkdir(parents=True, exist_ok=True)
        self.successful_tickers = []
        self.failed_tickers = []

    def setup_gcs_client(self):
        """
        Setup GCS client with proper authentication.
        """
        try:
            from google.cloud import storage
            from google.auth import default
            import google.auth.exceptions

            # Try different authentication methods
            client = None

            # Method 1: Try default credentials
            try:
                credentials, project = default()
                client = storage.Client(
                    project=os.environ.get('PROJECT_ID'),
                    credentials=credentials
                )
                logger.info("Connected with default credentials")
            except google.auth.exceptions.DefaultCredentialsError:
                pass

            # Method 2: Try application default credentials
            if not client:
                try:
                    client = storage.Client(project=os.environ.get('PROJECT_ID'))
                    logger.info("Connected with application default credentials")
                except:
                    pass

            # Method 3: Try anonymous client for public bucket
            if not client:
                try:
                    client = storage.Client.create_anonymous_client()
                    logger.info("Connected as anonymous client")
                except:
                    pass

            if not client:
                logger.error("Failed to create GCS client")
                return None

            self.gcs_client = client
            self.bucket = client.bucket(os.environ['GCS_BUCKET_NAME'])
            return True

        except ImportError:
            logger.info("Installing google-cloud-storage...")
            os.system("pip install google-cloud-storage")
            return self.setup_gcs_client()
        except Exception as e:
            logger.error(f"Failed to setup GCS client: {e}")
            return None

    def list_tickers_in_bucket(self, prefix='tickers/', max_tickers=10000):
        """
        List all tickers in the GCS bucket's tickers folder.
        """
        logger.info(f"Listing tickers in gs://{os.environ['GCS_BUCKET_NAME']}/{prefix}")

        try:
            # List all blobs in tickers folder
            blobs = self.bucket.list_blobs(prefix=prefix, max_results=max_tickers)

            ticker_list = []
            for blob in blobs:
                # Extract ticker from path: tickers/AAPL.parquet or tickers/AAPL.csv
                if '/' in blob.name:
                    filename = blob.name.split('/')[-1]
                    if filename and (filename.endswith('.parquet') or filename.endswith('.csv')):
                        ticker = filename.replace('.parquet', '').replace('.csv', '')
                        if ticker and not ticker.startswith('.'):
                            ticker_list.append((ticker, blob.name, blob.size))

            logger.info(f"Found {len(ticker_list)} tickers in GCS bucket")

            # Sort by size (larger files likely have more data)
            ticker_list.sort(key=lambda x: x[2], reverse=True)

            return ticker_list

        except Exception as e:
            logger.error(f"Error listing tickers: {e}")
            return []

    def download_ticker_blob(self, ticker, blob_path):
        """
        Download a single ticker from GCS.
        """
        local_file = self.ticker_dir / f"{ticker}.parquet"

        # Skip if already downloaded
        if local_file.exists():
            try:
                df = pd.read_parquet(local_file)
                if len(df) > 100:
                    return ticker, True, len(df)
            except:
                pass

        try:
            blob = self.bucket.blob(blob_path)

            # Download to bytes
            content = blob.download_as_bytes()

            # Parse data
            if blob_path.endswith('.parquet'):
                df = pd.read_parquet(pd.io.common.BytesIO(content))
            else:  # CSV
                df = pd.read_csv(pd.io.common.BytesIO(content))

            # Validate and save
            if len(df) > 100:
                df.to_parquet(local_file)
                return ticker, True, len(df)
            else:
                return ticker, False, 0

        except Exception as e:
            logger.debug(f"Failed to download {ticker}: {e}")
            return ticker, False, 0

    def download_massive_dataset(self, target_tickers=5000, max_workers=30):
        """
        Download massive dataset from GCS.
        """
        print("\n" + "="*80)
        print("MASSIVE GCS TICKER DOWNLOAD")
        print("="*80)
        print(f"Target: {target_tickers:,} tickers from tickers/ folder")
        print(f"Bucket: gs://{os.environ['GCS_BUCKET_NAME']}/tickers/")
        print("="*80)

        # Setup GCS
        if not self.setup_gcs_client():
            logger.error("Failed to setup GCS client")

            # Try using gsutil command line as fallback
            print("\nTrying gsutil command line tool...")
            return self.download_with_gsutil(target_tickers)

        # List available tickers
        ticker_list = self.list_tickers_in_bucket(max_tickers=target_tickers * 2)

        if not ticker_list:
            logger.error("No tickers found in bucket")
            return self.download_with_gsutil(target_tickers)

        # Limit to target number
        ticker_list = ticker_list[:target_tickers]

        print(f"\nDownloading {len(ticker_list)} tickers...")
        print(f"Using {max_workers} parallel workers")

        # Download in parallel
        stats = {
            'successful': 0,
            'failed': 0,
            'total_rows': 0
        }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.download_ticker_blob, ticker, blob_path): ticker
                for ticker, blob_path, _ in ticker_list
            }

            with tqdm(total=len(futures), desc="Downloading") as pbar:
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        ticker, success, rows = future.result(timeout=30)
                        if success:
                            stats['successful'] += 1
                            stats['total_rows'] += rows
                            self.successful_tickers.append(ticker)
                        else:
                            stats['failed'] += 1
                            self.failed_tickers.append(ticker)
                    except:
                        stats['failed'] += 1
                        self.failed_tickers.append(ticker)

                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': stats['successful'],
                        'Failed': stats['failed']
                    })

        self.print_summary(stats)
        return stats

    def download_with_gsutil(self, target_tickers):
        """
        Fallback method using gsutil command line.
        """
        print("\n" + "="*80)
        print("USING GSUTIL COMMAND LINE")
        print("="*80)

        try:
            # Create download script
            script = f"""
# Download tickers using gsutil
gsutil -m cp -r gs://{os.environ['GCS_BUCKET_NAME']}/tickers/* {self.ticker_dir}/
"""

            script_file = Path('download_tickers.sh')
            script_file.write_text(script)

            print(f"Executing: gsutil -m cp gs://{os.environ['GCS_BUCKET_NAME']}/tickers/* ...")

            # Run gsutil
            import subprocess
            result = subprocess.run(
                f'gsutil -m cp -n gs://{os.environ["GCS_BUCKET_NAME"]}/tickers/*.parquet {self.ticker_dir}/',
                shell=True,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                # Count downloaded files
                downloaded = list(self.ticker_dir.glob('*.parquet'))
                print(f"\nDownloaded {len(downloaded)} ticker files")

                stats = {
                    'successful': len(downloaded),
                    'failed': 0,
                    'total_rows': len(downloaded) * 1000  # Estimate
                }

                self.print_summary(stats)
                return stats
            else:
                print(f"gsutil failed: {result.stderr}")
                return {'successful': 0, 'failed': 1, 'total_rows': 0}

        except Exception as e:
            print(f"gsutil error: {e}")
            return {'successful': 0, 'failed': 1, 'total_rows': 0}

    def print_summary(self, stats):
        """
        Print download summary.
        """
        print("\n" + "="*80)
        print("DOWNLOAD COMPLETE")
        print("="*80)
        print(f"Successful downloads: {stats['successful']:,}")
        print(f"Failed downloads: {stats['failed']}")
        print(f"Total data rows: {stats['total_rows']:,}")

        if stats['successful'] > 0:
            print(f"Average rows per ticker: {stats['total_rows']//stats['successful']:,}")

            # Save summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'successful_tickers': self.successful_tickers,
                'failed_tickers': self.failed_tickers,
                'stats': stats
            }

            summary_file = self.cache_dir / f'download_summary_{datetime.now():%Y%m%d_%H%M%S}.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"\nSummary saved to: {summary_file}")

            # Expected patterns
            expected_patterns = stats['successful'] * 30
            expected_k4 = expected_patterns * 0.065  # 6.5% K4 rate from previous scan

            print("\n" + "="*80)
            print("EXPECTED RESULTS")
            print("="*80)
            print(f"Expected pattern snapshots: {expected_patterns:,}")
            print(f"Expected K4 patterns: {expected_k4:,.0f}")
            print(f"Total K4s after combining: ~{expected_k4 + 1419:,.0f}")

            print("\n" + "="*80)
            print("NEXT STEPS")
            print("="*80)
            print("1. Run pattern detection on downloaded tickers:")
            print(f"   python detect_patterns_batch.py --input {self.ticker_dir}")
            print("\n2. Extract V3 features from patterns:")
            print("   python feature_engineering_v3.py")
            print("\n3. Train attention ensemble with massive dataset:")
            print("   python train_attention_ensemble.py")


def main():
    """
    Main function for massive GCS download.
    """
    downloader = DirectGCSDownloader()

    # User input
    print("\n" + "="*80)
    print("MASSIVE GCS TICKER DOWNLOAD")
    print("="*80)
    print("\nDownload options:")
    print("1. Quick test (100 tickers)")
    print("2. Medium (500 tickers)")
    print("3. Large (1000 tickers)")
    print("4. Massive (5000 tickers)")
    print("5. Ultra (10000 tickers)")
    print("6. Custom number")

    choice = input("\nSelect option (1-6, default 4): ").strip() or '4'

    if choice == '1':
        target = 100
    elif choice == '2':
        target = 500
    elif choice == '3':
        target = 1000
    elif choice == '4':
        target = 5000
    elif choice == '5':
        target = 10000
    else:
        target = int(input("Enter number of tickers: "))

    workers = min(50, max(20, target // 100))

    print(f"\nDownloading {target:,} tickers with {workers} workers...")

    # Run download
    stats = downloader.download_massive_dataset(target_tickers=target, max_workers=workers)

    return stats


if __name__ == "__main__":
    stats = main()