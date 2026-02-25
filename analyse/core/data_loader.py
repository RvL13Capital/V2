"""
Unified Data Loader Module for AIv3 System
Handles all data loading from GCS, local files, and caching
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from google.cloud import storage
from google.oauth2 import service_account
import logging
from functools import lru_cache
import pickle

from .config import get_config

logger = logging.getLogger(__name__)


class DataCache:
    """Simple file-based cache for data"""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if exists and not expired"""
        cache_file = self.cache_dir / f"{key}.pkl"
        meta_file = self.cache_dir / f"{key}.meta"

        if cache_file.exists() and meta_file.exists():
            # Check expiry
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                cached_time = datetime.fromisoformat(meta['timestamp'])
                if datetime.now() - cached_time < timedelta(hours=24):
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
        return None

    def set(self, key: str, data: pd.DataFrame):
        """Cache data with timestamp"""
        cache_file = self.cache_dir / f"{key}.pkl"
        meta_file = self.cache_dir / f"{key}.meta"

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

        with open(meta_file, 'w') as f:
            json.dump({'timestamp': datetime.now().isoformat()}, f)


class UnifiedDataLoader:
    """Unified data loader for all data sources"""

    def __init__(self, use_cache: bool = True):
        self.config = get_config()
        self.cache = DataCache() if use_cache else None
        self.gcs_client = None
        self.bucket = None
        self._initialize_gcs()

    def _initialize_gcs(self):
        """Initialize GCS client"""
        try:
            import requests
            # Disable SSL verification warnings (temporary workaround)
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            # Set environment to help with SSL issues
            os.environ['GOOGLE_AUTH_DISABLE_MTLS'] = '1'

            credentials = service_account.Credentials.from_service_account_file(
                self.config.gcs.credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )

            # Create a custom session with retry logic for SSL errors
            from google.auth.transport.requests import AuthorizedSession
            session = AuthorizedSession(credentials)

            # Increase timeout and add retry logic
            import google.api_core.retry as retries

            self.gcs_client = storage.Client(
                credentials=credentials,
                project=self.config.gcs.project_id
            )

            self.bucket = self.gcs_client.bucket(self.config.gcs.bucket_name)
            logger.info(f"Connected to GCS bucket: {self.config.gcs.bucket_name}")

        except Exception as e:
            logger.warning(f"GCS initialization failed: {e}. Will use local data only.")

    def load_ticker_data(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = "auto"
    ) -> pd.DataFrame:
        """
        Load ticker data from GCS or local file

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            source: 'gcs', 'local', or 'auto'

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_key = f"{ticker}_{source}"
        if self.cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Using cached data for {ticker}")
                return self._filter_dates(cached_data, start_date, end_date)

        # Try loading based on source
        data = None
        if source in ['gcs', 'auto']:
            data = self._load_from_gcs(ticker)

        if data is None and source in ['local', 'auto']:
            data = self._load_from_local(ticker)

        if data is None:
            logger.warning(f"No data found for {ticker}")
            return pd.DataFrame()

        # Process and cache
        data = self._process_ticker_data(data)

        if self.cache:
            self.cache.set(cache_key, data)

        return self._filter_dates(data, start_date, end_date)

    def _load_from_gcs(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load ticker data from GCS"""
        if not self.bucket:
            return None

        try:
            blob_path = f"tickers/{ticker}.csv"
            blob = self.bucket.blob(blob_path)

            if blob.exists():
                csv_content = blob.download_as_text()
                df = pd.read_csv(pd.io.common.StringIO(csv_content))
                logger.debug(f"Loaded {ticker} from GCS")
                return df
        except Exception as e:
            logger.error(f"Error loading {ticker} from GCS: {e}")

        return None

    def _load_from_local(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load ticker data from local file"""
        local_paths = [
            Path(f"./data/{ticker}.csv"),
            Path(f"./tickers/{ticker}.csv"),
            Path(f"./historical_data/{ticker}.csv"),
        ]

        for path in local_paths:
            if path.exists():
                df = pd.read_csv(path)
                logger.debug(f"Loaded {ticker} from {path}")
                return df

        return None

    def _process_ticker_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and standardize ticker data"""
        # Standardize column names
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }

        df = df.rename(columns=column_mapping)

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df.set_index('date')

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column {col}, filling with zeros")
                df[col] = 0

        return df

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Filter dataframe by date range"""
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        return df

    def load_multiple_tickers(
        self,
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            parallel: Whether to load in parallel

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}

        if parallel and len(tickers) > 5:
            # Use parallel loading for large ticker lists
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.config.analysis.parallel_workers) as executor:
                future_to_ticker = {
                    executor.submit(self.load_ticker_data, ticker, start_date, end_date): ticker
                    for ticker in tickers
                }

                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        data = future.result()
                        if not data.empty:
                            results[ticker] = data
                    except Exception as e:
                        logger.error(f"Error loading {ticker}: {e}")
        else:
            # Sequential loading
            for ticker in tickers:
                data = self.load_ticker_data(ticker, start_date, end_date)
                if not data.empty:
                    results[ticker] = data

        logger.info(f"Loaded data for {len(results)}/{len(tickers)} tickers")
        return results

    def load_patterns(
        self,
        source: str = "historical_patterns.parquet",
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Load pattern data

        Args:
            source: Path to pattern file or GCS blob name
            filters: Optional filters to apply

        Returns:
            DataFrame with pattern data
        """
        # Check if local file
        if Path(source).exists():
            if source.endswith('.parquet'):
                df = pd.read_parquet(source)
            elif source.endswith('.json'):
                with open(source, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:
                df = pd.read_csv(source)

            logger.info(f"Loaded {len(df)} patterns from {source}")

        # Try GCS
        elif self.bucket:
            try:
                blob = self.bucket.blob(source)
                if blob.exists():
                    content = blob.download_as_text()
                    if source.endswith('.json'):
                        data = json.loads(content)
                        df = pd.DataFrame(data)
                    else:
                        df = pd.read_csv(pd.io.common.StringIO(content))
                    logger.info(f"Loaded {len(df)} patterns from GCS")
                else:
                    logger.warning(f"Pattern file {source} not found")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error loading patterns: {e}")
                return pd.DataFrame()
        else:
            logger.warning("No pattern source available")
            return pd.DataFrame()

        # Apply filters
        if filters:
            for key, value in filters.items():
                if key in df.columns:
                    if isinstance(value, (list, tuple)):
                        df = df[df[key].isin(value)]
                    else:
                        df = df[df[key] == value]

        return df

    def list_available_tickers(self, source: str = "auto") -> List[str]:
        """Get list of available tickers"""
        tickers = set()

        # Check GCS
        if source in ['gcs', 'auto'] and self.bucket:
            try:
                blobs = self.bucket.list_blobs(prefix="tickers/")
                gcs_tickers = [
                    blob.name.replace("tickers/", "").replace(".csv", "")
                    for blob in blobs
                    if blob.name.endswith('.csv')
                ]
                tickers.update(gcs_tickers)
                logger.info(f"Found {len(gcs_tickers)} tickers in GCS")
            except Exception as e:
                logger.error(f"Error listing GCS tickers: {e}")

        # Check local
        if source in ['local', 'auto']:
            local_dirs = [Path("./data"), Path("./tickers")]
            for directory in local_dirs:
                if directory.exists():
                    local_tickers = [
                        f.stem for f in directory.glob("*.csv")
                    ]
                    tickers.update(local_tickers)

        return sorted(list(tickers))

    def save_results(
        self,
        data: Union[pd.DataFrame, Dict, List],
        filename: str,
        format: str = "auto"
    ):
        """
        Save analysis results

        Args:
            data: Data to save
            filename: Output filename
            format: 'parquet', 'json', 'csv', or 'auto'
        """
        # Ensure output directory exists
        self.config.analysis.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if filename already contains the output directory
        from pathlib import Path
        filename_path = Path(filename)
        if filename_path.parts and filename_path.parts[0] == str(self.config.analysis.output_dir):
            # Filename already contains output directory, use as is
            output_path = Path(filename)
        else:
            # Append to output directory
            output_path = self.config.analysis.output_dir / filename

        if format == "auto":
            if filename.endswith('.parquet'):
                format = "parquet"
            elif filename.endswith('.json'):
                format = "json"
            else:
                format = "csv"

        try:
            if format == "parquet" and isinstance(data, pd.DataFrame):
                data.to_parquet(output_path)
            elif format == "json":
                with open(output_path, 'w') as f:
                    if isinstance(data, pd.DataFrame):
                        json.dump(data.to_dict('records'), f, indent=2)
                    else:
                        json.dump(data, f, indent=2)
            elif format == "csv" and isinstance(data, pd.DataFrame):
                data.to_csv(output_path)
            else:
                # Fallback to pickle
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)

            logger.info(f"Saved results to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None


# Convenience function
def get_data_loader() -> UnifiedDataLoader:
    """Get data loader instance"""
    return UnifiedDataLoader()