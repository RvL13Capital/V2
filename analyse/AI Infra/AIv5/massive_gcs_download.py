"""
Massive GCS Data Download for Training Set Expansion
=====================================================
Downloads thousands of tickers from GCS to massively expand the training dataset
and find many more K4 patterns.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import gc

# Configure environment for GCS
os.environ['PROJECT_ID'] = 'ignition-ki-csv-storage'
os.environ['GCS_BUCKET_NAME'] = 'ignition-ki-csv-data-2025-user123'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''  # Will use default auth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class MassiveGCSDownloader:
    """
    Downloads massive amounts of ticker data from GCS for training expansion.
    """

    def __init__(self):
        self.cache_dir = Path('data/massive_gcs_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ticker_dir = self.cache_dir / 'tickers'
        self.ticker_dir.mkdir(parents=True, exist_ok=True)
        self.failed_tickers = []
        self.successful_tickers = []

    def setup_gcs_client(self):
        """
        Setup GCS client with proper authentication.
        """
        try:
            from google.cloud import storage
            from google.auth import default

            # Try default credentials first
            try:
                credentials, project = default()
                client = storage.Client(
                    project=os.environ.get('PROJECT_ID', 'ignition-ki-csv-storage'),
                    credentials=credentials
                )
            except:
                # Fallback to anonymous for public buckets
                client = storage.Client.create_anonymous_client()

            self.gcs_client = client
            self.bucket = client.bucket(os.environ['GCS_BUCKET_NAME'])
            return True

        except ImportError:
            logger.warning("Google Cloud Storage not installed. Installing...")
            os.system("pip install google-cloud-storage")
            return self.setup_gcs_client()
        except Exception as e:
            logger.error(f"Failed to setup GCS client: {e}")
            return False

    def list_all_gcs_tickers(self, prefix='ticker/', limit=None):
        """
        List ALL available tickers in GCS bucket.
        """
        logger.info("Listing all available tickers in GCS...")

        try:
            # List all blobs
            blobs = self.bucket.list_blobs(prefix=prefix, max_results=limit)

            tickers = []
            for blob in blobs:
                if blob.name.endswith(('.parquet', '.csv')):
                    # Extract ticker from path: ticker/AAPL.parquet
                    parts = blob.name.split('/')
                    if len(parts) >= 2:
                        ticker = parts[-1].replace('.parquet', '').replace('.csv', '')
                        if ticker and not ticker.startswith('.'):
                            tickers.append((ticker, blob.name))

            logger.info(f"Found {len(tickers)} tickers in GCS")
            return tickers

        except Exception as e:
            logger.error(f"Error listing GCS tickers: {e}")
            # Fallback to manual list
            return self.get_fallback_ticker_list()

    def get_fallback_ticker_list(self):
        """
        Comprehensive fallback ticker list if GCS listing fails.
        """
        # Comprehensive list including S&P 500, Russell 2000, and popular stocks
        ticker_list = [
            # Major indices components
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'JPM', 'JNJ', 'V', 'UNH', 'HD', 'MA', 'PG', 'DIS', 'BAC', 'XOM',
            'CVX', 'ABBV', 'PFE', 'KO', 'WMT', 'AVGO', 'MRK', 'PEP', 'TMO',
            'COST', 'ABT', 'ORCL', 'ACN', 'NKE', 'LLY', 'MCD', 'ADBE', 'CRM',
            'AMD', 'INTC', 'CSCO', 'CMCSA', 'VZ', 'TXN', 'NFLX', 'T', 'HON',

            # Small/Mid caps with high volatility
            'GME', 'AMC', 'BB', 'NOK', 'BBBY', 'PLTR', 'SOFI', 'WISH', 'CLOV',
            'SPCE', 'DKNG', 'PENN', 'TLRY', 'CGC', 'ACB', 'SNDL', 'OGI', 'HEXO',
            'RIOT', 'MARA', 'BTBT', 'MSTR', 'SQ', 'PYPL', 'ROKU', 'SNAP', 'PINS',
            'UBER', 'LYFT', 'ABNB', 'DASH', 'SNOW', 'U', 'RBLX', 'COIN', 'HOOD',

            # EV and clean energy
            'LCID', 'RIVN', 'NIO', 'LI', 'XPEV', 'FSR', 'GOEV', 'RIDE', 'WKHS',
            'NKLA', 'HYLN', 'PTRA', 'ARVL', 'REE', 'CHPT', 'BLNK', 'EVGO', 'VLDR',
            'LAZR', 'AEVA', 'OUST', 'PLUG', 'FCEL', 'BE', 'BLDP', 'ENPH', 'SEDG',

            # Biotech and pharma penny stocks
            'SAVA', 'ANVS', 'BNGO', 'OCGN', 'VXRT', 'INO', 'SRNE', 'GERN', 'ATHX',
            'KMPH', 'CTXR', 'ATNF', 'OPGN', 'XERS', 'DARE', 'EVFM', 'TTOO', 'TNXP',
            'ADMP', 'AGTC', 'BCRX', 'CRBP', 'IMGN', 'MGEN', 'NTLA', 'CRSP', 'EDIT',

            # Tech penny stocks
            'GNUS', 'IDEX', 'SOLO', 'AYRO', 'WATT', 'MARK', 'INPX', 'SHIP', 'TOPS',
            'CTRM', 'CASTOR', 'GLBS', 'PSHG', 'SINO', 'CPSH', 'EDRY', 'ESEA', 'GRIN',

            # Cannabis stocks
            'TLRY', 'CGC', 'ACB', 'CRON', 'SNDL', 'OGI', 'HEXO', 'VFF', 'GRWG',
            'CURLF', 'TCNNF', 'GTBIF', 'CRLBF', 'AYRWF', 'HRVSF', 'IIPR', 'SMG',
        ]

        # Add variations and additional tickers
        additional = []
        for base in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
            for i in range(3):
                additional.append(f"{base}{chr(65+i)}")  # AA, AB, AC, etc.

        ticker_list.extend(additional[:200])  # Add 200 more

        return [(ticker, f'ticker/{ticker}.parquet') for ticker in ticker_list]

    def download_blob(self, blob_name, ticker):
        """
        Download a single blob from GCS.
        """
        local_file = self.ticker_dir / f"{ticker}.parquet"

        # Skip if already downloaded
        if local_file.exists():
            try:
                df = pd.read_parquet(local_file)
                if len(df) > 100:  # Valid data
                    return ticker, True, len(df)
            except:
                pass

        try:
            blob = self.bucket.blob(blob_name)

            # Download to memory first
            content = blob.download_as_bytes()

            # Try to parse as parquet
            try:
                df = pd.read_parquet(pd.io.common.BytesIO(content))
            except:
                # Try CSV if parquet fails
                df = pd.read_csv(pd.io.common.BytesIO(content))

            # Validate data
            if len(df) > 100:
                # Save locally
                df.to_parquet(local_file)
                return ticker, True, len(df)
            else:
                return ticker, False, 0

        except Exception as e:
            logger.debug(f"Failed to download {ticker}: {e}")
            return ticker, False, 0

    def download_from_yfinance(self, ticker):
        """
        Fallback to yfinance for missing tickers.
        """
        try:
            import yfinance as yf

            local_file = self.ticker_dir / f"{ticker}.parquet"
            if local_file.exists():
                return ticker, True, 0

            # Download 5 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5*365)

            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if len(df) > 100:
                df.to_parquet(local_file)
                return ticker, True, len(df)
            else:
                return ticker, False, 0

        except:
            return ticker, False, 0

    def download_massive_dataset(self, target_tickers=5000, max_workers=20):
        """
        Download massive dataset with target number of tickers.
        """
        print("\n" + "="*80)
        print("MASSIVE GCS DOWNLOAD FOR TRAINING EXPANSION")
        print("="*80)
        print(f"Target: {target_tickers:,} tickers")
        print(f"Expected patterns: {target_tickers * 30:,}+")
        print(f"Expected K4 patterns: {target_tickers * 30 * 0.004:,.0f}+ (vs current 450)")
        print("="*80)

        # Setup GCS
        if not self.setup_gcs_client():
            logger.warning("Using fallback download methods")

        # Get ticker list
        logger.info("Getting ticker list from GCS...")
        ticker_list = self.list_all_gcs_tickers(limit=target_tickers * 2)

        if len(ticker_list) < target_tickers:
            logger.info(f"Only {len(ticker_list)} tickers available in GCS")
            # Augment with yfinance
            fallback_list = self.get_fallback_ticker_list()
            ticker_list.extend(fallback_list[:(target_tickers - len(ticker_list))])

        ticker_list = ticker_list[:target_tickers]
        logger.info(f"Will attempt to download {len(ticker_list)} tickers")

        # Download in parallel
        stats = {
            'successful': 0,
            'failed': 0,
            'total_rows': 0
        }

        print(f"\nDownloading with {max_workers} parallel workers...")
        print("This may take 30-60 minutes for 5000 tickers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all downloads
            futures = {}
            for ticker, blob_name in ticker_list[:target_tickers]:
                if 'ticker/' in blob_name:
                    future = executor.submit(self.download_blob, blob_name, ticker)
                else:
                    future = executor.submit(self.download_from_yfinance, ticker)
                futures[future] = ticker

            # Process results with progress bar
            with tqdm(total=len(futures), desc="Downloading tickers") as pbar:
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

        # Save summary
        self.save_download_summary(stats)

        return stats

    def save_download_summary(self, stats):
        """
        Save download summary and statistics.
        """
        print("\n" + "="*80)
        print("DOWNLOAD COMPLETE!")
        print("="*80)
        print(f"Successful downloads: {stats['successful']:,}")
        print(f"Failed downloads: {stats['failed']}")
        print(f"Total data rows: {stats['total_rows']:,}")
        print(f"Average rows per ticker: {stats['total_rows']/max(stats['successful'],1):,.0f}")

        # Save summary
        summary = pd.DataFrame({
            'ticker': self.successful_tickers,
            'status': 'downloaded'
        })

        summary_file = self.cache_dir / f'download_summary_{datetime.now():%Y%m%d_%H%M%S}.csv'
        summary.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")

        # Save ticker list for processing
        ticker_file = self.cache_dir / 'massive_ticker_list.txt'
        with open(ticker_file, 'w') as f:
            f.write('\n'.join(self.successful_tickers))
        print(f"Ticker list saved to: {ticker_file}")

        print("\n" + "="*80)
        print("EXPECTED TRAINING IMPROVEMENT")
        print("="*80)

        expected_patterns = stats['successful'] * 30
        expected_k4 = expected_patterns * 0.004

        print(f"Expected pattern snapshots: {expected_patterns:,}")
        print(f"Expected K4 patterns: {expected_k4:,.0f}")
        print(f"Improvement over current: {expected_k4/450:.1f}x more K4 patterns")

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("1. Run pattern detection on massive dataset:")
        print("   python run_massive_pattern_scan.py")
        print("\n2. Label with race logic:")
        print("   python src/pipeline/02_label_snapshots_race.py --batch")
        print("\n3. Train with massive dataset:")
        print("   python train_attention_ensemble.py")

        return summary


def main():
    """
    Main function for massive GCS download.
    """
    downloader = MassiveGCSDownloader()

    # User input
    print("\n" + "="*80)
    print("MASSIVE TRAINING SET EXPANSION")
    print("="*80)

    print("\nRecommended options:")
    print("1. Quick (1000 tickers) - 10-15 minutes")
    print("2. Standard (2500 tickers) - 25-35 minutes")
    print("3. Large (5000 tickers) - 50-70 minutes")
    print("4. Massive (10000 tickers) - 2-3 hours")
    print("5. Custom number")

    choice = input("\nSelect option (1-5, default 2): ").strip() or '2'

    if choice == '1':
        target = 1000
    elif choice == '2':
        target = 2500
    elif choice == '3':
        target = 5000
    elif choice == '4':
        target = 10000
    else:
        target = int(input("Enter number of tickers: "))

    workers = min(30, max(10, target // 100))  # Scale workers with dataset size

    print(f"\nDownloading {target:,} tickers with {workers} parallel workers...")
    print("Starting download...")

    # Run download
    stats = downloader.download_massive_dataset(target_tickers=target, max_workers=workers)

    return stats


if __name__ == "__main__":
    stats = main()