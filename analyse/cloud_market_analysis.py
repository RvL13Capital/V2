"""
Cloud-based Market Analysis using Google Cloud Services
Optimized for free tier limits while maximizing performance
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Try to import Google Cloud dependencies
BIGQUERY_AVAILABLE = False
GCS_AVAILABLE = False
PYARROW_AVAILABLE = False

try:
    from google.cloud import bigquery
    from google.cloud.exceptions import GoogleCloudError
    BIGQUERY_AVAILABLE = True
except ImportError:
    logging.warning("google-cloud-bigquery not installed. Cloud features will be limited.")
    logging.warning("Install with: pip install google-cloud-bigquery")

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    logging.warning("google-cloud-storage not installed. GCS features will be limited.")
    logging.warning("Install with: pip install google-cloud-storage")

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    logging.warning("pyarrow not installed. Parquet support will be limited.")
    logging.warning("Install with: pip install pyarrow")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CloudMarketAnalyzer:
    """
    Cloud-optimized market analyzer using BigQuery and GCS
    Designed to stay within free tier limits
    """

    # Free tier limits
    BIGQUERY_FREE_TIER = {
        'monthly_bytes_processed': 1_000_000_000_000,  # 1 TB per month
        'monthly_table_operations': 1000,
        'daily_bytes_processed': 33_000_000_000,  # ~33 GB per day (1TB/30)
        'max_query_bytes': 10_000_000_000  # 10 GB per query for safety
    }

    GCS_FREE_TIER = {
        'monthly_storage_gb': 5,
        'monthly_operations': 50000,
        'monthly_egress_gb': 1
    }

    def __init__(self, project_id: str, credentials_path: str,
                 dataset_name: str = 'market_analysis',
                 use_bigquery: bool = True):
        """
        Initialize cloud analyzer

        Args:
            project_id: GCP project ID
            credentials_path: Path to service account credentials
            dataset_name: BigQuery dataset name
            use_bigquery: Whether to use BigQuery (faster but uses quota)
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.use_bigquery = use_bigquery and BIGQUERY_AVAILABLE

        # Initialize clients
        if GCS_AVAILABLE:
            self.storage_client = storage.Client(project=project_id)
            self.bucket_name = "ignition-ki-csv-data-2025-user123"
            self.bucket = self.storage_client.bucket(self.bucket_name)
        else:
            logger.warning("GCS not available. Using local file fallback.")
            self.storage_client = None
            self.bucket = None

        if self.use_bigquery and BIGQUERY_AVAILABLE:
            self.bq_client = bigquery.Client(project=project_id)
            self._setup_bigquery_dataset()
        elif use_bigquery and not BIGQUERY_AVAILABLE:
            logger.warning("BigQuery requested but not available. Install with: pip install google-cloud-bigquery")
            self.bq_client = None
        else:
            self.bq_client = None

        # Track usage to stay within limits
        self.usage_tracker = {
            'bytes_processed_today': 0,
            'bytes_processed_month': 0,
            'operations_today': 0,
            'last_reset': datetime.now()
        }

    def load_market_data_to_bigquery(self, tickers: List[str]) -> bool:
        """
        Load market data from GCS to BigQuery for the specified tickers
        """
        if not self.use_bigquery or not tickers:
            return False

        try:
            # Create market_data table if it doesn't exist
            table_id = f"{self.project_id}.{self.dataset_name}.market_data"

            # Check which tickers already exist in BigQuery
            existing_tickers = []
            try:
                check_query = f"""
                SELECT DISTINCT ticker
                FROM `{table_id}`
                WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
                """
                result = self.bq_client.query(check_query).result()
                existing_tickers = [row.ticker for row in result]
                logger.info(f"Found {len(existing_tickers)} tickers already in BigQuery")
            except Exception as e:
                # Table doesn't exist yet
                logger.info("Table doesn't exist yet, will create it")

            # Define schema for market data
            schema = [
                bigquery.SchemaField("ticker", "STRING"),
                bigquery.SchemaField("date", "DATE"),
                bigquery.SchemaField("open", "FLOAT64"),
                bigquery.SchemaField("high", "FLOAT64"),
                bigquery.SchemaField("low", "FLOAT64"),
                bigquery.SchemaField("close", "FLOAT64"),
                bigquery.SchemaField("volume", "INT64"),
            ]

            table = bigquery.Table(table_id, schema=schema)

            # Create table if it doesn't exist
            try:
                table = self.bq_client.create_table(table)
                logger.info(f"Created table {table_id}")
            except Exception as e:
                if "Already Exists" not in str(e):
                    logger.error(f"Error creating table: {e}")
                    return False

            # Only load tickers that aren't already in BigQuery
            tickers_to_load = [t for t in tickers if t not in existing_tickers]
            if not tickers_to_load:
                logger.info("All tickers already loaded in BigQuery")
                return True

            logger.info(f"Loading {len(tickers_to_load)} new tickers to BigQuery")

            # Load data for each ticker from GCS
            for ticker in tickers_to_load:
                try:
                    # Check multiple possible locations
                    blob = None
                    blob_name = None

                    # Try root directory first
                    possible_paths = [
                        f"{ticker}_full_history.csv",  # Root with full_history
                        f"tickers/{ticker}_full_history.csv",  # tickers/ subdirectory
                        f"tickers/{ticker}.csv",  # tickers/ without suffix
                        f"{ticker}.csv"  # Root without suffix
                    ]

                    for path in possible_paths:
                        test_blob = self.bucket.blob(path)
                        if test_blob.exists():
                            blob = test_blob
                            blob_name = path
                            logger.debug(f"Found {ticker} at {path}")
                            break

                    if blob and blob.exists():
                        # Download and read CSV
                        csv_data = blob.download_as_text()
                        import io
                        df = pd.read_csv(io.StringIO(csv_data))

                        # Add ticker column
                        df['ticker'] = ticker

                        # Ensure columns are in correct format (handle various naming conventions)
                        df.columns = df.columns.str.lower()
                        if 'date' not in df.columns and 'timestamp' in df.columns:
                            df.rename(columns={'timestamp': 'date'}, inplace=True)

                        # Convert date column to datetime
                        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.date

                        # Select only required columns and ensure correct types
                        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
                        df = df[required_cols]

                        # Ensure numeric columns are float/int
                        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in numeric_cols:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        # Upload to BigQuery
                        job_config = bigquery.LoadJobConfig(
                            write_disposition="WRITE_APPEND",
                            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
                        )

                        job = self.bq_client.load_table_from_dataframe(
                            df, table_id, job_config=job_config
                        )
                        job.result()  # Wait for the job to complete

                        logger.info(f"Loaded {ticker} data to BigQuery")
                    else:
                        logger.warning(f"No data found for ticker {ticker} in GCS")

                except Exception as e:
                    logger.error(f"Error loading {ticker} data: {e}")
                    continue

            return True

        except Exception as e:
            logger.error(f"Error loading market data to BigQuery: {e}")
            return False

    def load_tickers_from_gcs(self, num_stocks: Optional[int] = None) -> List[str]:
        """
        Load available tickers from GCS bucket
        """
        tickers = []

        if not GCS_AVAILABLE or self.bucket is None:
            logger.error("GCS not available. Cannot load tickers from cloud.")
            return tickers

        try:
            # Check both locations: root directory AND tickers/ subdirectory
            # First check root directory for _full_history.csv files
            blobs = self.storage_client.list_blobs(self.bucket_name)

            for blob in blobs:
                # Files in root with _full_history.csv suffix
                if blob.name.endswith('_full_history.csv') and '/' not in blob.name:
                    ticker = blob.name.replace('_full_history.csv', '')
                    if ticker and ticker not in tickers:
                        tickers.append(ticker)

            # Also check tickers/ subdirectory
            blobs_in_tickers = self.storage_client.list_blobs(self.bucket_name, prefix='tickers/')

            for blob in blobs_in_tickers:
                # Extract ticker from path like "tickers/AAPL_full_history.csv"
                if blob.name.endswith('_full_history.csv'):
                    ticker = blob.name.replace('tickers/', '').replace('_full_history.csv', '')
                    if ticker and ticker not in tickers:
                        tickers.append(ticker)
                # Also check for files without _full_history suffix in tickers/
                elif blob.name.startswith('tickers/') and blob.name.endswith('.csv'):
                    ticker = blob.name.replace('tickers/', '').replace('.csv', '')
                    # Remove any date suffixes if present
                    if '_' in ticker:
                        ticker = ticker.split('_')[0]
                    if ticker and ticker not in tickers and ticker.upper() == ticker:
                        tickers.append(ticker)

            logger.info(f"Found {len(tickers)} unique tickers across bucket")

            # Sort tickers for consistency
            tickers.sort()

            # Limit to requested number if specified
            if num_stocks and num_stocks < len(tickers):
                tickers = tickers[:num_stocks]

            logger.info(f"Loaded {len(tickers)} tickers from GCS")
            return tickers

        except Exception as e:
            logger.error(f"Error loading tickers from GCS: {e}")
            return []

    def _setup_bigquery_dataset(self):
        """Create BigQuery dataset if it doesn't exist"""
        dataset_id = f"{self.project_id}.{self.dataset_name}"
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"

        try:
            dataset = self.bq_client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {dataset_id}")
        except Exception as e:
            if "Already Exists" not in str(e):
                logger.error(f"Error creating dataset: {e}")

    def _check_quota_limits(self, estimated_bytes: int) -> bool:
        """Check if operation would exceed free tier limits"""
        if not self.use_bigquery:
            return True

        # Reset daily counter if needed
        if (datetime.now() - self.usage_tracker['last_reset']).days >= 1:
            self.usage_tracker['bytes_processed_today'] = 0
            self.usage_tracker['operations_today'] = 0
            self.usage_tracker['last_reset'] = datetime.now()

        # Check limits
        if self.usage_tracker['bytes_processed_today'] + estimated_bytes > self.BIGQUERY_FREE_TIER['daily_bytes_processed']:
            logger.warning(f"Would exceed daily BigQuery limit. Processed today: {self.usage_tracker['bytes_processed_today']/1e9:.2f} GB")
            return False

        if estimated_bytes > self.BIGQUERY_FREE_TIER['max_query_bytes']:
            logger.warning(f"Query too large: {estimated_bytes/1e9:.2f} GB > 10 GB limit")
            return False

        return True

    def upload_to_bigquery_streaming(self, df: pd.DataFrame, table_name: str,
                                    chunk_size: int = 50000) -> bool:
        """
        Stream data to BigQuery in chunks to avoid memory issues
        """
        if not self.use_bigquery:
            return False

        table_id = f"{self.project_id}.{self.dataset_name}.{table_name}"

        # Define schema based on DataFrame
        schema = self._generate_bq_schema(df)

        # Create table if it doesn't exist
        table = bigquery.Table(table_id, schema=schema)
        try:
            table = self.bq_client.create_table(table)
            logger.info(f"Created table {table_id}")
        except Exception as e:
            if "Already Exists" not in str(e):
                logger.error(f"Error creating table: {e}")
                return False

        # Stream data in chunks
        total_rows = len(df)
        logger.info(f"Streaming {total_rows} rows to BigQuery in chunks of {chunk_size}")

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx]

            # Convert chunk to list of dicts
            rows_to_insert = chunk.to_dict('records')

            # Insert rows
            errors = self.bq_client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Error inserting rows: {errors}")
                return False

            logger.info(f"Inserted {end_idx}/{total_rows} rows ({(end_idx/total_rows)*100:.1f}%)")

            # Update usage tracker
            self.usage_tracker['bytes_processed_today'] += len(chunk) * 1000  # Rough estimate
            self.usage_tracker['operations_today'] += 1

        return True

    def process_with_bigquery(self, tickers: List[str],
                            start_date: str, end_date: str) -> pd.DataFrame:
        """
        Process data using BigQuery SQL for massive speedup
        """
        if not self.use_bigquery:
            logger.info("BigQuery disabled, falling back to local processing")
            return None

        # Check if tickers list is empty
        if not tickers:
            logger.error("No tickers provided for BigQuery processing")
            return pd.DataFrame()

        # Ensure market data is loaded to BigQuery
        logger.info(f"Ensuring market data is available in BigQuery for {len(tickers)} tickers...")
        load_success = self.load_market_data_to_bigquery(tickers)
        if not load_success:
            logger.warning("Failed to load some data to BigQuery")

        # Check if we have quota
        estimated_bytes = len(tickers) * 365 * 1000  # Rough estimate
        if not self._check_quota_limits(estimated_bytes):
            logger.warning("Quota limits would be exceeded, falling back to local processing")
            return None

        # Build optimized SQL query for consolidation detection
        query = f"""
        WITH price_data_raw AS (
            SELECT
                ticker,
                date,
                close,
                high,
                low,
                volume,
                open,
                LAG(close) OVER (PARTITION BY ticker ORDER BY date) as prev_close
            FROM `{self.project_id}.{self.dataset_name}.market_data`
            WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
                AND date BETWEEN '{start_date}' AND '{end_date}'
        ),
        price_data AS (
            SELECT
                ticker,
                date,
                close as price,
                high,
                low,
                volume,
                -- Calculate 20-day moving averages
                AVG(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20,
                STDDEV(close) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as std20,
                AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_volume_20d,
                AVG(high - low) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as avg_range_20d,
                -- Calculate ATR (simplified without nested window functions)
                AVG(GREATEST(high - low,
                           ABS(high - IFNULL(prev_close, close)),
                           ABS(low - IFNULL(prev_close, close))))
                    OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) as atr14
            FROM price_data_raw
        ),
        indicators AS (
            SELECT
                *,
                -- Bollinger Band Width
                CASE WHEN ma20 > 0 THEN (2 * std20) / ma20 ELSE NULL END as bbw,
                -- Volume ratio
                CASE WHEN avg_volume_20d > 0 THEN volume / avg_volume_20d ELSE NULL END as volume_ratio,
                -- Range ratio
                CASE WHEN avg_range_20d > 0 THEN (high - low) / avg_range_20d ELSE NULL END as range_ratio,
                -- ATR percentage
                CASE WHEN price > 0 THEN atr14 / price ELSE NULL END as atr_pct
            FROM price_data
        ),
        consolidation_signals AS (
            SELECT
                *,
                -- Method 1: Bollinger Band Width
                CASE WHEN bbw < PERCENTILE_CONT(bbw, 0.3) OVER (PARTITION BY ticker) THEN TRUE ELSE FALSE END as method1_bollinger,
                -- Method 2: Range-based
                CASE WHEN range_ratio < 0.65 AND atr_pct < 0.04 THEN TRUE ELSE FALSE END as method2_range_based,
                -- Method 3: Volume-weighted
                CASE WHEN volume_ratio < 0.35 THEN TRUE ELSE FALSE END as method3_volume_weighted,
                -- Method 4: ATR-based
                CASE WHEN atr_pct < PERCENTILE_CONT(atr_pct, 0.3) OVER (PARTITION BY ticker) THEN TRUE ELSE FALSE END as method4_atr_based
            FROM indicators
        )
        SELECT
            *,
            -- Calculate future performance (if historical data available)
            MAX(price) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) / price - 1 as max_gain_20d,
            MIN(price) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 20 FOLLOWING) / price - 1 as max_loss_20d,
            MAX(price) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 40 FOLLOWING) / price - 1 as max_gain_40d,
            MIN(price) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 1 FOLLOWING AND 40 FOLLOWING) / price - 1 as max_loss_40d
        FROM consolidation_signals
        ORDER BY ticker, date
        """

        try:
            # Run query
            logger.info("Executing BigQuery analysis...")
            query_job = self.bq_client.query(query)
            results = query_job.result()

            # Convert to DataFrame - handle Storage API permissions gracefully
            df = None
            try:
                # Try using BigQuery Storage API for better performance
                df = results.to_dataframe(create_bqstorage_client=True)
            except Exception as storage_error:
                if 'permission' in str(storage_error).lower() or 'readsessions' in str(storage_error).lower():
                    logger.warning("BigQuery Storage API not available, using REST API instead")
                    # Re-run the query for REST API (iterator already consumed)
                    query_job = self.bq_client.query(query)
                    results = query_job.result()
                    df = results.to_dataframe(create_bqstorage_client=False)
                else:
                    # For other errors, try REST API as fallback
                    logger.warning(f"Storage API error: {storage_error}, falling back to REST API")
                    query_job = self.bq_client.query(query)
                    results = query_job.result()
                    df = results.to_dataframe(create_bqstorage_client=False)

            # Update usage
            self.usage_tracker['bytes_processed_today'] += query_job.total_bytes_processed
            logger.info(f"Query processed {query_job.total_bytes_processed/1e9:.2f} GB")
            logger.info(f"Query returned {len(df)} rows for {df['ticker'].nunique() if not df.empty else 0} tickers")

            return df

        except Exception as e:
            logger.error(f"BigQuery error: {e}")
            return None

    def process_locally_optimized(self, tickers: List[str],
                                 batch_size: int = 50) -> pd.DataFrame:
        """
        Process data locally with optimizations
        Uses parallel processing and efficient memory management
        """
        logger.info(f"Processing {len(tickers)} tickers locally in batches of {batch_size}")

        all_results = []

        # Process in batches to manage memory
        for batch_start in range(0, len(tickers), batch_size):
            batch_end = min(batch_start + batch_size, len(tickers))
            batch_tickers = tickers[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//batch_size + 1}: {len(batch_tickers)} tickers")

            # Parallel processing for each batch
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for ticker in batch_tickers:
                    future = executor.submit(self._process_single_ticker_optimized, ticker)
                    futures.append(future)

                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing ticker: {e}")

            # Clear memory between batches
            import gc
            gc.collect()

            logger.info(f"Completed batch {batch_start//batch_size + 1}")

        # Combine all results
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()

    def _process_single_ticker_optimized(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Process single ticker with memory optimization
        """
        try:
            # Check multiple possible locations
            blob = None
            possible_paths = [
                f"{ticker}_full_history.csv",  # Root with full_history
                f"tickers/{ticker}_full_history.csv",  # tickers/ subdirectory
                f"tickers/{ticker}.csv",  # tickers/ without suffix
                f"{ticker}.csv"  # Root without suffix
            ]

            for path in possible_paths:
                test_blob = self.bucket.blob(path)
                if test_blob.exists():
                    blob = test_blob
                    logger.debug(f"Found {ticker} at {path}")
                    break

            if not blob or not blob.exists():
                logger.warning(f"No data found for {ticker}")
                return None

            # Read CSV
            import io
            csv_data = blob.download_as_text()
            df = pd.read_csv(io.StringIO(csv_data))

            # Standardize column names
            df.columns = df.columns.str.lower()
            if 'date' not in df.columns and 'timestamp' in df.columns:
                df.rename(columns={'timestamp': 'date'}, inplace=True)

            # Parse dates with UTC to avoid mixed timezone warning
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

            # Add ticker column
            df['ticker'] = ticker

            # Calculate basic consolidation indicators
            df['price'] = df['close']
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['bbw'] = (2 * df['std20']) / df['ma20']
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma20']
            df['range'] = df['high'] - df['low']
            df['range_ma20'] = df['range'].rolling(window=20).mean()
            df['range_ratio'] = df['range'] / df['range_ma20']

            # Simple consolidation detection
            df['consolidation'] = (
                (df['bbw'] < df['bbw'].quantile(0.3)) &
                (df['volume_ratio'] < 0.5) &
                (df['range_ratio'] < 0.65)
            )

            return df

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return None

    def _generate_bq_schema(self, df: pd.DataFrame) -> List:
        """Generate BigQuery schema from DataFrame"""
        schema = []
        for col, dtype in df.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                field_type = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                field_type = "FLOAT"
            elif pd.api.types.is_bool_dtype(dtype):
                field_type = "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                field_type = "TIMESTAMP"
            else:
                field_type = "STRING"

            schema.append(bigquery.SchemaField(col, field_type))

        return schema

    def run_hybrid_analysis(self, tickers: List[str],
                           use_cloud_priority: bool = True,
                           fallback_to_local: bool = True) -> pd.DataFrame:
        """
        Run analysis with intelligent cloud/local hybrid approach

        Args:
            tickers: List of tickers to analyze
            use_cloud_priority: Try cloud first if within limits
            fallback_to_local: Fall back to local if cloud fails/exceeds limits
        """
        start_time = time.time()

        # Estimate processing size
        estimated_size = len(tickers) * 365 * 1000  # Rough estimate
        logger.info(f"Estimated data size: {estimated_size/1e9:.2f} GB")

        results = None

        # Try cloud processing first if enabled and within limits
        if use_cloud_priority and self.use_bigquery:
            if self._check_quota_limits(estimated_size):
                logger.info("Using BigQuery for fast cloud processing...")
                results = self.process_with_bigquery(
                    tickers,
                    start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                    end_date=datetime.now().strftime('%Y-%m-%d')
                )

                if results is not None:
                    elapsed = time.time() - start_time
                    logger.info(f"Cloud processing completed in {elapsed:.1f} seconds")
                    return results

        # Fallback to optimized local processing
        if fallback_to_local and results is None:
            logger.info("Using optimized local processing...")
            results = self.process_locally_optimized(tickers)
            elapsed = time.time() - start_time
            logger.info(f"Local processing completed in {elapsed:.1f} seconds")

        return results

    def get_usage_report(self) -> Dict:
        """Get current usage statistics"""
        return {
            'bigquery': {
                'bytes_processed_today_gb': self.usage_tracker['bytes_processed_today'] / 1e9,
                'daily_limit_gb': self.BIGQUERY_FREE_TIER['daily_bytes_processed'] / 1e9,
                'percentage_used': (self.usage_tracker['bytes_processed_today'] /
                                  self.BIGQUERY_FREE_TIER['daily_bytes_processed']) * 100,
                'operations_today': self.usage_tracker['operations_today']
            },
            'remaining_quota': {
                'bytes_gb': (self.BIGQUERY_FREE_TIER['daily_bytes_processed'] -
                           self.usage_tracker['bytes_processed_today']) / 1e9,
                'can_process_tickers': int((self.BIGQUERY_FREE_TIER['daily_bytes_processed'] -
                                          self.usage_tracker['bytes_processed_today']) / (365 * 1000))
            }
        }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Cloud-optimized market analysis')
    parser.add_argument('--tickers', nargs='+', help='List of tickers to analyze')
    parser.add_argument('--num-stocks', type=int, help='Number of stocks to analyze')
    parser.add_argument('--use-bigquery', action='store_true',
                       help='Use BigQuery for faster processing (uses quota)')
    parser.add_argument('--cloud-only', action='store_true',
                       help='Only use cloud processing, fail if quota exceeded')
    parser.add_argument('--local-only', action='store_true',
                       help='Only use local processing')
    parser.add_argument('--check-usage', action='store_true',
                       help='Check current usage and quotas')

    args = parser.parse_args()

    # Set up credentials
    credentials_path = r"C:\Users\Pfenn\Downloads\ignition-ki-csv-storage-e7bb9d0fd1d0.json"
    project_id = "ignition-ki-csv-storage"

    # Initialize analyzer
    use_bigquery = args.use_bigquery and not args.local_only
    analyzer = CloudMarketAnalyzer(
        project_id=project_id,
        credentials_path=credentials_path,
        use_bigquery=use_bigquery
    )

    # Check usage if requested
    if args.check_usage:
        usage = analyzer.get_usage_report()
        print("\n" + "="*60)
        print("CLOUD USAGE REPORT")
        print("="*60)
        print(f"BigQuery Daily Usage: {usage['bigquery']['bytes_processed_today_gb']:.2f} GB / "
              f"{usage['bigquery']['daily_limit_gb']:.2f} GB "
              f"({usage['bigquery']['percentage_used']:.1f}%)")
        print(f"Remaining Quota: {usage['remaining_quota']['bytes_gb']:.2f} GB")
        print(f"Can Process: ~{usage['remaining_quota']['can_process_tickers']} more tickers today")
        print("="*60)
        return

    # Determine processing mode
    if args.cloud_only:
        use_cloud_priority = True
        fallback_to_local = False
    elif args.local_only:
        use_cloud_priority = False
        fallback_to_local = True
    else:
        use_cloud_priority = True
        fallback_to_local = True

    # Get tickers
    if args.tickers:
        tickers = args.tickers
    elif args.num_stocks:
        # Load available tickers from GCS and select first N
        logger.info(f"Loading {args.num_stocks} tickers from GCS...")
        tickers = analyzer.load_tickers_from_gcs(num_stocks=args.num_stocks)
        if not tickers:
            logger.error("Failed to load tickers from GCS. Please check your GCS configuration.")
            return
    else:
        logger.error("Please specify --tickers or --num-stocks")
        return

    # Run analysis
    logger.info(f"Starting analysis for {len(tickers)} tickers...")
    results = analyzer.run_hybrid_analysis(
        tickers=tickers,
        use_cloud_priority=use_cloud_priority,
        fallback_to_local=fallback_to_local
    )

    if results is not None and not results.empty:
        # Save results
        output_file = f"cloud_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        results.to_parquet(output_file)
        logger.info(f"Results saved to {output_file}")

        # Show usage after processing
        usage = analyzer.get_usage_report()
        print(f"\nProcessing complete. Used {usage['bigquery']['bytes_processed_today_gb']:.2f} GB of quota")
    else:
        logger.error("No results generated")


if __name__ == "__main__":
    main()