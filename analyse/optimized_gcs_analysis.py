"""
Optimized GCS Analysis with Connection Pool Management
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from google.cloud import storage
import io
import time
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Set up credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\gcs-key.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedGCSAnalyzer:
    """Optimized GCS processing with proper connection management"""

    def __init__(self):
        self.storage_client = storage.Client(project='ignition-ki-csv-storage')
        self.bucket_name = 'ignition-ki-csv-data-2025-user123'
        self.bucket = self.storage_client.bucket(self.bucket_name)
        # Cache for blob listings to reduce API calls
        self._blob_cache = {}
        self._build_blob_cache()

    def _build_blob_cache(self):
        """Build cache of all blob paths"""
        logger.info("Building file cache...")
        for blob in self.storage_client.list_blobs(self.bucket_name):
            self._blob_cache[blob.name] = True
        logger.info(f"Cached {len(self._blob_cache)} files")

    def _find_ticker_file(self, ticker: str) -> Optional[str]:
        """Find ticker file from cache"""
        possible_paths = [
            f"{ticker}_full_history.csv",
            f"tickers/{ticker}_full_history.csv",
            f"tickers/{ticker}.csv",
            f"{ticker}.csv"
        ]

        for path in possible_paths:
            if path in self._blob_cache:
                return path
        return None

    def get_all_tickers(self) -> List[str]:
        """Get all tickers from cache"""
        tickers = set()

        for path in self._blob_cache.keys():
            if path.endswith('.csv'):
                if path.startswith('tickers/'):
                    ticker = path.replace('tickers/', '').replace('.csv', '').split('_')[0]
                elif '/' not in path:
                    ticker = path.replace('_full_history.csv', '').replace('.csv', '').split('_')[0]
                else:
                    continue

                if ticker.upper() == ticker and len(ticker) <= 10 and ticker.isalpha():
                    tickers.add(ticker)

        return sorted(list(tickers))

    def process_ticker_batch(self, tickers: List[str]) -> List[Dict]:
        """Process a batch of tickers sequentially to avoid connection issues"""
        results = []

        for ticker in tickers:
            try:
                # Find file path
                file_path = self._find_ticker_file(ticker)
                if not file_path:
                    continue

                # Download and process
                blob = self.bucket.blob(file_path)
                csv_data = blob.download_as_text()
                df = pd.read_csv(io.StringIO(csv_data))

                # Standardize columns
                df.columns = df.columns.str.lower()
                if 'date' not in df.columns and 'timestamp' in df.columns:
                    df.rename(columns={'timestamp': 'date'}, inplace=True)

                # Parse dates safely
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                except:
                    pass

                # Skip if not enough data
                if len(df) < 20:
                    continue

                # Calculate indicators
                df['ma20'] = df['close'].rolling(window=20, min_periods=20).mean()
                df['std20'] = df['close'].rolling(window=20, min_periods=20).std()

                # Safe division
                with np.errstate(divide='ignore', invalid='ignore'):
                    df['bbw'] = np.where(df['ma20'] > 0, (2 * df['std20']) / df['ma20'], np.nan)
                    df['volume_ma20'] = df['volume'].rolling(window=20, min_periods=20).mean()
                    df['volume_ratio'] = np.where(df['volume_ma20'] > 0,
                                                 df['volume'] / df['volume_ma20'], np.nan)

                df['range'] = df['high'] - df['low']
                df['range_ma20'] = df['range'].rolling(window=20, min_periods=20).mean()

                with np.errstate(divide='ignore', invalid='ignore'):
                    df['range_ratio'] = np.where(df['range_ma20'] > 0,
                                                df['range'] / df['range_ma20'], np.nan)

                # Detect consolidation
                df['consolidation'] = False
                valid_rows = df.dropna(subset=['bbw', 'volume_ratio', 'range_ratio'])

                if len(valid_rows) > 0:
                    bbw_30 = valid_rows['bbw'].quantile(0.3)
                    consolidation_mask = (
                        (valid_rows['bbw'] < bbw_30) &
                        (valid_rows['volume_ratio'] < 0.5) &
                        (valid_rows['range_ratio'] < 0.65)
                    )
                    df.loc[valid_rows.index, 'consolidation'] = consolidation_mask

                # Get summary
                latest = df.iloc[-1] if len(df) > 0 else None
                last_30 = df.tail(30)

                result = {
                    'ticker': ticker,
                    'latest_date': latest['date'] if latest is not None else None,
                    'price': float(latest['close']) if latest is not None else None,
                    'bbw': float(latest['bbw']) if latest is not None and not pd.isna(latest['bbw']) else None,
                    'volume_ratio': float(latest['volume_ratio']) if latest is not None and not pd.isna(latest['volume_ratio']) else None,
                    'range_ratio': float(latest['range_ratio']) if latest is not None and not pd.isna(latest['range_ratio']) else None,
                    'consolidation': bool(latest['consolidation']) if latest is not None else False,
                    'consolidation_days_30d': int(last_30['consolidation'].sum()),
                    'total_days': len(df)
                }

                results.append(result)

            except Exception as e:
                logger.debug(f"Error processing {ticker}: {e}")
                continue

        return results

    def analyze_all(self, limit: Optional[int] = None,
                   batch_size: int = 50) -> pd.DataFrame:
        """Analyze all tickers"""

        # Get tickers
        all_tickers = self.get_all_tickers()

        if limit:
            all_tickers = all_tickers[:limit]

        logger.info(f"Processing {len(all_tickers)} tickers in batches of {batch_size}")

        all_results = []
        total_batches = (len(all_tickers) + batch_size - 1) // batch_size

        for i in range(0, len(all_tickers), batch_size):
            batch = all_tickers[i:i+batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"Batch {batch_num}/{total_batches}: Processing {len(batch)} tickers...")

            batch_start = time.time()
            batch_results = self.process_ticker_batch(batch)
            batch_time = time.time() - batch_start

            all_results.extend(batch_results)

            consolidation_count = sum(1 for r in batch_results if r.get('consolidation', False))
            logger.info(f"  Completed in {batch_time:.1f}s - Found {consolidation_count} in consolidation")

            # Small delay between batches
            if i + batch_size < len(all_tickers):
                time.sleep(0.1)

        # Create DataFrame
        df = pd.DataFrame(all_results)

        if not df.empty:
            # Calculate consolidation score
            df['consolidation_score'] = (
                df['consolidation'].astype(int) * 10 +
                df['consolidation_days_30d'] / 3
            )

            # Sort by score
            df = df.sort_values('consolidation_score', ascending=False)

        return df


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Limit tickers')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--output', default='optimized_analysis.parquet')

    args = parser.parse_args()

    # Run analysis
    analyzer = OptimizedGCSAnalyzer()

    start_time = time.time()
    results = analyzer.analyze_all(limit=args.limit, batch_size=args.batch_size)
    total_time = time.time() - start_time

    if not results.empty:
        # Save results
        results.to_parquet(args.output)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Tickers processed: {len(results)}")
        print(f"In consolidation: {results['consolidation'].sum()}")
        print(f"Processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Speed: {len(results)/total_time:.1f} tickers/second")
        print(f"Results saved to: {args.output}")

        # Show top candidates
        print("\nTop 10 Consolidation Candidates:")
        print("-"*40)
        top = results[results['consolidation'] == True].head(10)
        if not top.empty:
            for _, row in top.iterrows():
                print(f"{row['ticker']:6s} - Price: ${row['price']:7.2f} - "
                     f"{row['consolidation_days_30d']} days in consolidation")
    else:
        print("No results generated")


if __name__ == "__main__":
    main()