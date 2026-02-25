"""
Standalone Data Loader for Temporal Architecture
=================================================

Simplified data loader that works with:
1. Local parquet/CSV files
2. Google Cloud Storage (GCS) - optional

This is a standalone version extracted from AIv7's full DataLoader.
Includes comprehensive data validation for temporal integrity.
"""

import os
from pathlib import Path

# Load .env file BEFORE any GCS imports - override=True forces .env to take precedence
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent.parent / '.env'
if _env_path.exists():
    load_dotenv(_env_path, override=True)

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Tuple
from datetime import datetime
import logging

# Import exceptions for validation
from ..exceptions import (
    DataIntegrityError,
    TemporalConsistencyError,
    ValidationError
)

# Import configuration constants
from config import (
    MIN_DATA_LENGTH,
    MAX_TEMPORAL_GAP_DAYS,
    MIN_VOLUME,
    MAX_PRICE_RATIO
)

# Try to import GCS support (optional)
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load stock price data from local cache or GCS.

    Simplified standalone version for temporal architecture.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        bucket_name: Optional[str] = None,
        local_cache_dir: Optional[str] = None,
        disable_gcs: bool = False
    ):
        """
        Initialize data loader.

        Args:
            project_id: GCS project ID (optional)
            bucket_name: GCS bucket name (optional)
            local_cache_dir: Local cache directory
            disable_gcs: If True, skip GCS initialization (enables multiprocessing)
        """
        self.project_id = project_id or os.getenv('PROJECT_ID', 'ignition-ki-csv-storage')
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME', 'ignition-ki-csv-data-2025-user123')
        self.disable_gcs = disable_gcs

        # Setup local cache
        if local_cache_dir:
            self.cache_dir = Path(local_cache_dir)
        else:
            # Default to data/raw relative to project root
            self.cache_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GCS client if available and not disabled
        self.gcs_client = None
        self.bucket = None

        if disable_gcs:
            logger.info("GCS disabled. Using local cache only (multiprocessing-safe).")
        elif GCS_AVAILABLE:
            try:
                self.gcs_client = storage.Client(project=self.project_id)
                self.bucket = self.gcs_client.bucket(self.bucket_name)
                logger.info(f"GCS client initialized: {self.project_id}/{self.bucket_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS client: {e}")
                logger.warning("Will use local cache only")
        else:
            logger.info("GCS not available. Using local cache only.")

    def load_ticker(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True,
        validate: bool = True,
        fast_validation: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data for a ticker with optional validation.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime)
            use_cache: Use local cache if available
            validate: Perform comprehensive data validation
            fast_validation: Skip expensive mock data detection (default False)
                            Use True for faster scanning when data quality is known

        Returns:
            DataFrame with OHLCV data or None if not found

        Raises:
            DataIntegrityError: If data validation fails
            TemporalConsistencyError: If temporal issues detected
        """
        # Convert dates
        if start_date:
            start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)

        # Try to load from local cache
        df = None

        if use_cache:
            df = self._load_from_cache(ticker)

        # If not in cache and GCS available, try GCS
        if df is None and self.bucket is not None:
            df = self._load_from_gcs(ticker)

            # Cache it locally
            if df is not None:
                self._save_to_cache(ticker, df)

        if df is None:
            logger.warning(f"{ticker}: No data found in cache or GCS")
            return None

        # Basic column validation
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValidationError(f"{ticker}: Missing required columns")

        # Ensure date column is datetime and timezone-naive for consistent comparisons
        df['date'] = pd.to_datetime(df['date'])
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)

        # Filter by date range
        if start_date is not None:
            # Ensure start_date is also timezone-naive
            if hasattr(start_date, 'tz') and start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            df = df[df['date'] >= start_date]
        if end_date is not None:
            # Ensure end_date is also timezone-naive
            if hasattr(end_date, 'tz') and end_date.tz is not None:
                end_date = end_date.tz_localize(None)
            df = df[df['date'] <= end_date]

        if len(df) == 0:
            logger.warning(f"{ticker}: No data in date range")
            return None

        # Perform comprehensive validation if requested
        if validate:
            self.validate_temporal_data(df, ticker=ticker, fast_mode=fast_validation)

        # Set index to date for better temporal handling
        df = df.set_index('date').sort_index()

        return df

    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load from local cache (parquet or CSV)."""
        # Try parquet first
        parquet_file = self.cache_dir / f"{ticker}.parquet"
        if parquet_file.exists():
            try:
                df = pd.read_parquet(parquet_file)

                # Normalize columns (handle DatetimeIndex, capitalized columns)
                df = self._normalize_columns(df, ticker)

                logger.debug(f"{ticker}: Loaded from parquet cache")
                return df
            except Exception as e:
                logger.warning(f"{ticker}: Failed to load parquet: {e}")

        # Try CSV
        csv_file = self.cache_dir / f"{ticker}.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)

                # Normalize columns (handle capitalized columns, extra columns)
                df = self._normalize_columns(df, ticker)

                # Ensure date column is datetime (after normalization)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

                logger.debug(f"{ticker}: Loaded from CSV cache")
                return df
            except Exception as e:
                logger.warning(f"{ticker}: Failed to load CSV: {e}")

        return None

    def _normalize_columns(self, df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
        """
        Normalize column names to standard lowercase format.

        Handles:
        - Capitalized columns (Date, Open, High, Low, Close, Volume)
        - DatetimeIndex in Parquet files
        - Extra columns like 'Adj Close'

        Args:
            df: DataFrame to normalize
            ticker: Ticker name for logging

        Returns:
            Normalized DataFrame with standard columns
        """
        # Handle DatetimeIndex in Parquet files
        if 'date' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                # Rename index column to 'date'
                if 'datetime' in df.columns:
                    df = df.rename(columns={'datetime': 'date'})
                elif 'index' in df.columns:
                    df = df.rename(columns={'index': 'date'})
                elif 'Date' in df.columns:
                    df = df.rename(columns={'Date': 'date'})
                else:
                    # First column is likely the date after reset_index
                    first_col = df.columns[0]
                    if pd.api.types.is_datetime64_any_dtype(df[first_col]):
                        df = df.rename(columns={first_col: 'date'})

        # Map common column name variations to standard lowercase names
        column_map = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            'adjClose': 'adj_close'  # Alternative naming convention
        }

        df = df.rename(columns=column_map)

        # Keep standard columns (including adj_close for accurate historical market cap)
        standard_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        existing_standard = [c for c in standard_cols if c in df.columns]

        if len(existing_standard) < len(standard_cols):
            missing = [c for c in standard_cols if c not in df.columns]
            logger.warning(f"{ticker}: Missing columns after normalization: {missing}")

        # Keep only standard columns that exist
        df = df[existing_standard]

        return df

    def _load_from_gcs(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load from GCS bucket (from tickers/ subdirectory)."""
        if self.bucket is None:
            return None

        # Try parquet first (in tickers/ subdirectory)
        parquet_blob = f"tickers/{ticker}.parquet"
        try:
            blob = self.bucket.blob(parquet_blob)
            if blob.exists():
                temp_file = self.cache_dir / f"{ticker}_temp.parquet"
                blob.download_to_filename(str(temp_file))
                df = pd.read_parquet(temp_file)
                temp_file.unlink()  # Delete temp file

                # Normalize columns (handle DatetimeIndex, capitalized columns)
                df = self._normalize_columns(df, ticker)

                logger.info(f"{ticker}: Downloaded from GCS (parquet)")
                return df
        except Exception as e:
            logger.debug(f"{ticker}: Failed to load parquet from GCS: {e}")

        # Try CSV (in tickers/ subdirectory)
        csv_blob = f"tickers/{ticker}.csv"
        try:
            blob = self.bucket.blob(csv_blob)
            if blob.exists():
                temp_file = self.cache_dir / f"{ticker}_temp.csv"
                blob.download_to_filename(str(temp_file))
                df = pd.read_csv(temp_file)

                # Normalize columns (handle capitalized columns, extra columns)
                df = self._normalize_columns(df, ticker)

                # Ensure date column is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

                temp_file.unlink()  # Delete temp file
                logger.info(f"{ticker}: Downloaded from GCS (CSV)")
                return df
        except Exception as e:
            logger.debug(f"{ticker}: Failed to load CSV from GCS: {e}")

        return None

    def _save_to_cache(self, ticker: str, df: pd.DataFrame):
        """Save DataFrame to local cache (parquet format)."""
        try:
            cache_file = self.cache_dir / f"{ticker}.parquet"
            df.to_parquet(cache_file, index=False)
            logger.debug(f"{ticker}: Saved to cache")
        except Exception as e:
            logger.warning(f"{ticker}: Failed to save to cache: {e}")

    def validate_temporal_data(
        self,
        df: pd.DataFrame,
        min_length: Optional[int] = None,
        ticker: Optional[str] = None,
        fast_mode: bool = False
    ) -> None:
        """
        Comprehensive temporal data validation.

        Ensures:
        - No mock/synthetic data (skipped in fast_mode)
        - Temporal integrity maintained
        - OHLCV relationships valid
        - Sufficient data length

        Args:
            df: DataFrame to validate
            min_length: Minimum required data length (default: MIN_DATA_LENGTH)
            ticker: Ticker name for error messages
            fast_mode: Skip expensive mock data detection (default False)
                      Saves ~50-100ms per ticker for large datasets

        Raises:
            DataIntegrityError: If data integrity violations detected
            TemporalConsistencyError: If temporal issues detected
            ValidationError: If validation fails
        """
        if min_length is None:
            min_length = MIN_DATA_LENGTH

        ticker_prefix = f"{ticker}: " if ticker else ""
        errors = []

        # Check minimum length
        if len(df) < min_length:
            errors.append(f"Insufficient data: {len(df)} rows (need {min_length}+)")

        # Check required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")

        # Check for nulls
        if df[required].isnull().any().any():
            null_counts = df[required].isnull().sum()
            null_cols = null_counts[null_counts > 0].to_dict()
            errors.append(f"Missing values: {null_cols}")

        # OHLCV sanity checks (always run - fast)
        ohlcv_errors = self._validate_ohlcv_relationships(df)
        errors.extend(ohlcv_errors)

        # Check for mock data indicators (EXPENSIVE - skip in fast mode)
        # This involves multiple array operations and pattern matching
        if not fast_mode:
            mock_indicators = self._detect_mock_data(df)
            if mock_indicators:
                errors.append(f"Suspected mock data: {', '.join(mock_indicators)}")

        # Temporal integrity checks (if date column or datetime index exists)
        if 'date' in df.columns or isinstance(df.index, pd.DatetimeIndex):
            temporal_errors = self._validate_temporal_integrity(df)
            errors.extend(temporal_errors)

        # Raise appropriate exception if errors found
        if errors:
            error_msg = ticker_prefix + "; ".join(errors)
            if any("temporal" in e.lower() or "chronological" in e.lower() for e in errors):
                raise TemporalConsistencyError(error_msg)
            elif any("mock" in e.lower() for e in errors):
                raise DataIntegrityError(error_msg)
            else:
                raise ValidationError(error_msg)

        logger.debug(f"{ticker_prefix}Data validation passed" + (" (fast mode)" if fast_mode else ""))

    def _validate_ohlcv_relationships(self, df: pd.DataFrame) -> List[str]:
        """
        Validate OHLCV data relationships.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # High must be >= Low
        if (df['high'] < df['low']).any():
            count = (df['high'] < df['low']).sum()
            errors.append(f"Invalid OHLCV: high < low ({count} rows)")

        # High must be >= Open and Close
        if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
            errors.append("Invalid OHLCV: high not highest value")

        # Low must be <= Open and Close
        if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
            errors.append("Invalid OHLCV: low not lowest value")

        # Volume must be non-negative
        if (df['volume'] < MIN_VOLUME).any():
            count = (df['volume'] < MIN_VOLUME).sum()
            errors.append(f"Negative/zero volume detected ({count} rows)")

        # Check for unrealistic price ratios
        price_ratio = df['high'] / df['low']
        if (price_ratio > MAX_PRICE_RATIO).any():
            count = (price_ratio > MAX_PRICE_RATIO).sum()
            max_ratio = price_ratio.max()
            errors.append(f"Unrealistic price ratio: {max_ratio:.2f} ({count} rows)")

        return errors

    def _detect_mock_data(self, df: pd.DataFrame) -> List[str]:
        """
        Detect indicators of mock or synthetic data.

        Returns:
            List of mock data indicators
        """
        indicators = []

        # Check for constant values
        if df['volume'].std() == 0:
            indicators.append("constant volume")

        if df['close'].std() == 0:
            indicators.append("constant price")

        # Check for perfectly linear price movement
        if len(df) > 10:
            close_diff = df['close'].diff().dropna()
            if close_diff.std() < 1e-10:  # Near-zero variance in price changes
                indicators.append("perfectly linear price movement")

        # Check for suspiciously round numbers
        if len(df) > 10:
            # Check if all prices are round numbers
            decimals = df['close'].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
            if decimals.max() == 0:  # All integers
                indicators.append("all prices are integers")

            # Check if volume is always round thousands/millions
            if (df['volume'] % 1000 == 0).all():
                indicators.append("volume always round thousands")

        # Check for repeating patterns
        if len(df) > 20:
            # Look for exact repeating sequences in price
            price_pattern = df['close'].values[:10]
            pattern_repeats = 0
            for i in range(10, len(df) - 10, 10):
                if np.array_equal(df['close'].values[i:i+10], price_pattern):
                    pattern_repeats += 1

            if pattern_repeats > 1:
                indicators.append("repeating price patterns")

        return indicators

    def _validate_temporal_integrity(self, df: pd.DataFrame) -> List[str]:
        """
        Validate temporal ordering and consistency.

        Returns:
            List of temporal validation errors
        """
        errors = []

        # Get datetime index or column
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        elif 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
        else:
            return errors  # No date information to validate

        # Check for duplicate timestamps
        if dates.duplicated().any():
            count = dates.duplicated().sum()
            errors.append(f"Duplicate timestamps ({count} duplicates)")

        # Check chronological order
        if not dates.is_monotonic_increasing:
            errors.append("Data not in chronological order")

        # Check for unrealistic gaps
        if len(dates) > 1:
            gaps = pd.Series(dates).diff()
            max_gap = gaps.max()

            # Exclude NaT values
            if pd.notna(max_gap):
                if max_gap > pd.Timedelta(days=MAX_TEMPORAL_GAP_DAYS):
                    errors.append(f"Large temporal gap: {max_gap.days} days")

                # Check for negative gaps (time going backwards)
                if (gaps < pd.Timedelta(0)).any():
                    errors.append("Temporal inconsistency: time going backwards")

        # Check for weekend/holiday gaps (expected for stock data)
        if len(dates) > 5:
            # This is informational, not an error
            weekend_gaps = 0
            for gap in pd.Series(dates).diff().dropna():
                if gap == pd.Timedelta(days=3):  # Weekend gap (Friday to Monday)
                    weekend_gaps += 1

            if weekend_gaps > 0:
                logger.debug(f"Found {weekend_gaps} weekend gaps (expected)")

        return errors

    def validate_multiple_tickers(
        self,
        tickers: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Validate data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for validation
            end_date: End date for validation

        Returns:
            Tuple of (valid_tickers, invalid_tickers)
        """
        valid_tickers = []
        invalid_tickers = []

        for ticker in tickers:
            try:
                df = self.load_ticker(ticker, start_date, end_date, validate=True)
                if df is not None:
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
            except (DataIntegrityError, TemporalConsistencyError, ValidationError) as e:
                logger.warning(f"{ticker}: Validation failed: {e}")
                invalid_tickers.append(ticker)
            except Exception as e:
                logger.error(f"{ticker}: Unexpected error: {e}")
                invalid_tickers.append(ticker)

        logger.info(f"Validation complete: {len(valid_tickers)} valid, {len(invalid_tickers)} invalid")

        return valid_tickers, invalid_tickers


if __name__ == "__main__":
    # Test data loader
    logging.basicConfig(level=logging.INFO)

    loader = DataLoader()
    df = loader.load_ticker('AAPL', start_date='2024-01-01')

    if df is not None:
        print(f"Loaded {len(df)} rows")
        print(df.head())
    else:
        print("Failed to load data")
