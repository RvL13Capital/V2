"""
Storage Manager - CSV/Parquet/GCS Storage Operations

Handles dual format storage (CSV + Parquet) and uploads to Google Cloud Storage.
"""

import os
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from config.foreman_config import ForemanConfig


logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages storage of stock data in multiple formats

    Features:
    - Dual format storage (CSV + Parquet)
    - Local caching
    - GCS upload with retry
    - Error handling and logging
    """

    def __init__(self, config: ForemanConfig):
        """
        Initialize Storage Manager

        Args:
            config: Foreman configuration
        """
        self.config = config

        # GCS client
        try:
            self.gcs_client = storage.Client(project=config.storage.project_id)
            self.bucket = self.gcs_client.bucket(config.storage.bucket_name)
            logger.info(
                f"GCS client initialized: {config.storage.project_id}/{config.storage.bucket_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            self.gcs_client = None
            self.bucket = None

        # Local cache directory
        if config.storage.enable_local_cache:
            self.cache_dir = Path(config.storage.local_cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Local cache directory: {self.cache_dir}")
        else:
            self.cache_dir = None

        # Statistics
        self.files_saved_csv = 0
        self.files_saved_parquet = 0
        self.files_uploaded_gcs = 0
        self.errors = 0

    def save_ticker_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        exchange_suffix: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save ticker data in both CSV and Parquet formats

        Args:
            df: DataFrame with OHLCV data
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix (e.g., '.DE')
            metadata: Optional metadata to save alongside data

        Returns:
            True if all operations successful
        """
        if df is None or len(df) == 0:
            logger.warning(f"Cannot save empty data for {ticker}{exchange_suffix}")
            return False

        full_ticker = f"{ticker}{exchange_suffix}"
        success = True

        # Reset index to make date a column (required for storage)
        df_to_save = df.reset_index()

        # Ensure date column is named correctly
        if 'date' not in df_to_save.columns:
            df_to_save.columns = ['date'] + list(df_to_save.columns[1:])

        # Save to local cache
        if self.cache_dir:
            success = success and self._save_local(df_to_save, ticker, exchange_suffix)

        # Upload to GCS
        if self.bucket:
            success = success and self._upload_to_gcs(df_to_save, ticker, exchange_suffix)

        return success

    def _save_local(
        self,
        df: pd.DataFrame,
        ticker: str,
        exchange_suffix: str
    ) -> bool:
        """
        Save data to local cache in CSV and Parquet formats

        Args:
            df: DataFrame to save
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix

        Returns:
            True if successful
        """
        success = True

        # Save CSV
        if self.config.storage.save_csv:
            csv_path = self.cache_dir / f"{ticker}{exchange_suffix}.csv"

            try:
                df.to_csv(csv_path, index=False)
                self.files_saved_csv += 1
                logger.debug(f"Saved CSV: {csv_path}")
            except Exception as e:
                logger.error(f"Error saving CSV for {ticker}{exchange_suffix}: {e}")
                self.errors += 1
                success = False

        # Save Parquet
        if self.config.storage.save_parquet:
            parquet_path = self.cache_dir / f"{ticker}{exchange_suffix}.parquet"

            try:
                df.to_parquet(parquet_path, index=False, engine='pyarrow')
                self.files_saved_parquet += 1
                logger.debug(f"Saved Parquet: {parquet_path}")
            except Exception as e:
                logger.error(f"Error saving Parquet for {ticker}{exchange_suffix}: {e}")
                self.errors += 1
                success = False

        return success

    def _upload_to_gcs(
        self,
        df: pd.DataFrame,
        ticker: str,
        exchange_suffix: str
    ) -> bool:
        """
        Upload data to Google Cloud Storage

        Args:
            df: DataFrame to upload
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix

        Returns:
            True if successful
        """
        if not self.bucket:
            logger.warning("GCS client not initialized, skipping upload")
            return False

        success = True
        full_ticker = f"{ticker}{exchange_suffix}"

        # Upload CSV
        if self.config.storage.save_csv:
            success = success and self._upload_file(
                df, full_ticker, format='csv'
            )

        # Upload Parquet
        if self.config.storage.save_parquet:
            success = success and self._upload_file(
                df, full_ticker, format='parquet'
            )

        return success

    def _upload_file(
        self,
        df: pd.DataFrame,
        full_ticker: str,
        format: str
    ) -> bool:
        """
        Upload a single file to GCS

        Args:
            df: DataFrame to upload
            full_ticker: Full ticker with exchange suffix
            format: 'csv' or 'parquet'

        Returns:
            True if successful
        """
        try:
            # Create blob path
            blob_path = f"{self.config.storage.tickers_folder}{full_ticker}.{format}"
            blob = self.bucket.blob(blob_path)

            # Convert DataFrame to bytes
            if format == 'csv':
                data_bytes = df.to_csv(index=False).encode('utf-8')
            elif format == 'parquet':
                # For parquet, write to bytes buffer
                import io
                buffer = io.BytesIO()
                df.to_parquet(buffer, index=False, engine='pyarrow')
                data_bytes = buffer.getvalue()
            else:
                logger.error(f"Unsupported format: {format}")
                return False

            # Upload to GCS
            blob.upload_from_string(data_bytes)

            self.files_uploaded_gcs += 1
            logger.info(f"Uploaded to GCS: gs://{self.bucket.name}/{blob_path}")

            return True

        except GoogleCloudError as e:
            logger.error(f"GCS error uploading {full_ticker}.{format}: {e}")
            self.errors += 1
            return False

        except Exception as e:
            logger.error(f"Error uploading {full_ticker}.{format}: {e}")
            self.errors += 1
            return False

    def file_exists_in_gcs(self, ticker: str, exchange_suffix: str, format: str = 'csv') -> bool:
        """
        Check if file already exists in GCS

        Args:
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix
            format: File format ('csv' or 'parquet')

        Returns:
            True if file exists
        """
        if not self.bucket:
            return False

        try:
            blob_path = f"{self.config.storage.tickers_folder}{ticker}{exchange_suffix}.{format}"
            blob = self.bucket.blob(blob_path)
            return blob.exists()

        except Exception as e:
            logger.debug(f"Error checking file existence: {e}")
            return False

    def file_exists_locally(self, ticker: str, exchange_suffix: str, format: str = 'csv') -> bool:
        """
        Check if file exists in local cache

        Args:
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix
            format: File format

        Returns:
            True if file exists
        """
        if not self.cache_dir:
            return False

        file_path = self.cache_dir / f"{ticker}{exchange_suffix}.{format}"
        return file_path.exists()

    def load_from_local_cache(
        self,
        ticker: str,
        exchange_suffix: str,
        format: str = 'csv'
    ) -> Optional[pd.DataFrame]:
        """
        Load data from local cache

        Args:
            ticker: Base ticker symbol
            exchange_suffix: Exchange suffix
            format: File format

        Returns:
            DataFrame or None if not found
        """
        if not self.cache_dir:
            return None

        file_path = self.cache_dir / f"{ticker}{exchange_suffix}.{format}"

        if not file_path.exists():
            return None

        try:
            if format == 'csv':
                df = pd.read_csv(file_path, parse_dates=['date'])
            elif format == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                logger.error(f"Unsupported format: {format}")
                return None

            logger.debug(f"Loaded from local cache: {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None

    def get_statistics(self) -> Dict:
        """Get storage statistics"""
        return {
            'files_saved_csv': self.files_saved_csv,
            'files_saved_parquet': self.files_saved_parquet,
            'files_uploaded_gcs': self.files_uploaded_gcs,
            'errors': self.errors
        }

    def print_statistics(self):
        """Print storage statistics"""
        stats = self.get_statistics()

        logger.info("=" * 60)
        logger.info("Storage Manager Statistics")
        logger.info("=" * 60)
        logger.info(f"CSV files saved: {stats['files_saved_csv']}")
        logger.info(f"Parquet files saved: {stats['files_saved_parquet']}")
        logger.info(f"Files uploaded to GCS: {stats['files_uploaded_gcs']}")
        logger.info(f"Errors: {stats['errors']}")

        if self.cache_dir:
            logger.info(f"Local cache: {self.cache_dir}")

        if self.bucket:
            logger.info(f"GCS bucket: gs://{self.bucket.name}/{self.config.storage.tickers_folder}")

        logger.info("=" * 60)
