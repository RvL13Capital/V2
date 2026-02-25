"""
Fast GCS-based analysis without BigQuery table imports
Reads directly from GCS and processes locally with optimizations
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
import io
import time
from typing import List, Optional, Dict
import pyarrow.parquet as pq
import pyarrow as pa

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastGCSAnalyzer:
    """Direct GCS processing without BigQuery overhead"""

    def __init__(self, project_id: str = 'ignition-ki-csv-storage'):
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)
        self.bucket_name = 'ignition-ki-csv-data-2025-user123'
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def get_all_tickers(self) -> List[str]:
        """Get all available tickers from GCS"""
        tickers = set()

        # Check root directory
        for blob in self.storage_client.list_blobs(self.bucket_name):
            if blob.name.endswith('_full_history.csv') and '/' not in blob.name:
                ticker = blob.name.replace('_full_history.csv', '')
                tickers.add(ticker)

        # Check tickers/ directory
        for blob in self.storage_client.list_blobs(self.bucket_name, prefix='tickers/'):
            if blob.name.endswith('.csv'):
                ticker = blob.name.replace('tickers/', '').replace('.csv', '').split('_')[0]
                if ticker.upper() == ticker and len(ticker) <= 10:
                    tickers.add(ticker)

        return sorted(list(tickers))

    def process_ticker(self, ticker: str) -> Optional[Dict]:
        """Process single ticker and return consolidation analysis"""
        try:
            # Find the file
            possible_paths = [
                f"{ticker}_full_history.csv",
                f"tickers/{ticker}_full_history.csv",
                f"tickers/{ticker}.csv",
                f"{ticker}.csv"
            ]

            blob = None
            for path in possible_paths:
                test_blob = self.bucket.blob(path)
                if test_blob.exists():
                    blob = test_blob
                    break

            if not blob:
                return None

            # Read CSV
            csv_data = blob.download_as_text()
            df = pd.read_csv(io.StringIO(csv_data))

            # Standardize columns
            df.columns = df.columns.str.lower()
            if 'date' not in df.columns and 'timestamp' in df.columns:
                df.rename(columns={'timestamp': 'date'}, inplace=True)

            # Parse dates with proper format handling
            # Try different date formats
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
            except:
                try:
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce', utc=True)
                except:
                    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce', utc=True)

            # Skip if not enough data
            if len(df) < 20:
                return None

            # Calculate indicators
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['bbw'] = (2 * df['std20']) / df['ma20']
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma20']
            df['range'] = df['high'] - df['low']
            df['range_ma20'] = df['range'].rolling(window=20).mean()
            df['range_ratio'] = df['range'] / df['range_ma20']

            # Detect consolidation patterns
            df['consolidation'] = (
                (df['bbw'] < df['bbw'].quantile(0.3)) &
                (df['volume_ratio'] < 0.5) &
                (df['range_ratio'] < 0.65)
            )

            # Get latest data
            latest = df.iloc[-1] if len(df) > 0 else None

            # Count consolidation days in last 30 days
            last_30 = df.tail(30)
            consolidation_days = last_30['consolidation'].sum() if len(last_30) > 0 else 0

            return {
                'ticker': ticker,
                'date': latest['date'] if latest is not None else None,
                'price': latest['close'] if latest is not None else None,
                'bbw': latest['bbw'] if latest is not None else None,
                'volume_ratio': latest['volume_ratio'] if latest is not None else None,
                'consolidation': latest['consolidation'] if latest is not None else False,
                'consolidation_days_30d': int(consolidation_days),
                'total_days': len(df),
                'data_quality': 'good' if len(df) > 100 else 'limited'
            }

        except Exception as e:
            logger.debug(f"Error processing {ticker}: {e}")
            return None

    def process_batch(self, tickers: List[str], max_workers: int = 5) -> pd.DataFrame:
        """Process batch of tickers in parallel"""
        results = []

        # Reduce workers to avoid connection pool issues
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(self.process_ticker, ticker): ticker
                              for ticker in tickers}

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result(timeout=10)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.debug(f"Failed to process {ticker}: {e}")

        return pd.DataFrame(results)

    def analyze_all_tickers(self, limit: Optional[int] = None,
                           batch_size: int = 100) -> pd.DataFrame:
        """Analyze all tickers with batching"""

        # Get all tickers
        all_tickers = self.get_all_tickers()

        if limit:
            all_tickers = all_tickers[:limit]

        logger.info(f"Processing {len(all_tickers)} tickers")

        all_results = []

        # Process in batches
        for i in range(0, len(all_tickers), batch_size):
            batch = all_tickers[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} tickers")

            batch_start = time.time()
            batch_results = self.process_batch(batch)
            batch_time = time.time() - batch_start

            if not batch_results.empty:
                all_results.append(batch_results)
                logger.info(f"  Processed {len(batch_results)} tickers in {batch_time:.1f}s")
                logger.info(f"  Found {batch_results['consolidation'].sum()} in consolidation")

        # Combine results
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)

            # Sort by consolidation strength
            final_df['consolidation_score'] = (
                final_df['consolidation'].astype(int) * 10 +
                final_df['consolidation_days_30d'] / 3
            )
            final_df = final_df.sort_values('consolidation_score', ascending=False)

            return final_df

        return pd.DataFrame()


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Fast GCS Analysis')
    parser.add_argument('--limit', type=int, help='Limit number of tickers')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--output', default='gcs_analysis_results.parquet',
                       help='Output file')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = FastGCSAnalyzer()

    # Run analysis
    start_time = time.time()
    results = analyzer.analyze_all_tickers(
        limit=args.limit,
        batch_size=args.batch_size
    )

    total_time = time.time() - start_time

    if not results.empty:
        # Save results
        results.to_parquet(args.output)

        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total tickers processed: {len(results)}")
        print(f"Tickers in consolidation: {results['consolidation'].sum()}")
        print(f"Processing time: {total_time:.1f} seconds")
        print(f"Results saved to: {args.output}")

        # Show top consolidation candidates
        print("\nTop 10 Consolidation Candidates:")
        print("-"*40)
        top_10 = results[results['consolidation'] == True].head(10)
        for _, row in top_10.iterrows():
            print(f"{row['ticker']:6s} - {row['consolidation_days_30d']} days in consolidation")
    else:
        print("No results generated")


if __name__ == "__main__":
    main()