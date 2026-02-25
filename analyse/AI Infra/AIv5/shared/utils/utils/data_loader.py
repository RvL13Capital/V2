"""
Data Loader with GCS Support for AIv3 System

Loads OHLCV price data from Google Cloud Storage buckets.
Supports caching to local disk for faster subsequent loads.

MEMORY OPTIMIZATION (v3.1):
- Uses Parquet format with compression for caching (10x smaller than CSV)
- Automatically downcasts float64 → float32 (50% memory reduction)
- Supports selective column loading
- Integrates memory profiling and monitoring

AUTOMATIC PARQUET CONVERSION (v3.5):
- Prioritizes Parquet files over CSV when loading from GCS
- Automatically converts CSV → Parquet and uploads to GCS when Parquet not available
- Reduces storage costs and speeds up future loads (5-10x compression)
- Preserves CSV files for backward compatibility
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta
import logging
import gc

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Warning: google-cloud-storage not installed. GCS features disabled.")
    print("Install with: pip install google-cloud-storage")

# Import memory optimizer
from .memory_optimizer import MemoryOptimizer, SystemMemoryMonitor, get_dataframe_memory_usage

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load stock price data from GCS buckets with local caching.

    Environment Variables:
        PROJECT_ID: GCS project ID (default: "ignition-ki-csv-storage")
        GCS_BUCKET_NAME: GCS bucket name (default: "ignition-ki-csv-data-2025-user123")
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        bucket_name: Optional[str] = None,
        local_cache_dir: Optional[str] = None,
        use_memory_optimization: bool = True,
        aggressive_optimization: bool = False,
        cache_format: str = 'parquet',  # 'parquet' or 'csv'
        min_price: float = 0.05  # Minimum price threshold
    ):
        """
        Initialize data loader.

        Args:
            project_id: GCS project ID (falls back to env var)
            bucket_name: GCS bucket name (falls back to env var)
            local_cache_dir: Local cache directory for downloaded data
            use_memory_optimization: Apply memory optimization (float32, etc.)
            aggressive_optimization: Use aggressive memory optimization
            cache_format: Cache format ('parquet' recommended, or 'csv' for legacy)
            min_price: Minimum price threshold (exclude data where price < min_price)
        """
        self.project_id = project_id or os.getenv('PROJECT_ID', 'ignition-ki-csv-storage')
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME', 'ignition-ki-csv-data-2025-user123')
        self.min_price = min_price

        # Memory optimization settings
        self.use_memory_optimization = use_memory_optimization
        self.aggressive_optimization = aggressive_optimization
        self.cache_format = cache_format

        # Initialize memory utilities
        if use_memory_optimization:
            self.memory_optimizer = MemoryOptimizer(aggressive=aggressive_optimization)
            self.memory_monitor = SystemMemoryMonitor()
            logger.info("Memory optimization enabled")
        else:
            self.memory_optimizer = None
            self.memory_monitor = None

        # Setup local cache
        if local_cache_dir:
            self.cache_dir = Path(local_cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent / 'data' / 'raw'

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GCS client if available
        self.gcs_client = None
        self.bucket = None

        if GCS_AVAILABLE:
            try:
                self.gcs_client = storage.Client(project=self.project_id)
                self.bucket = self.gcs_client.bucket(self.bucket_name)
                logger.info(f"GCS client initialized: {self.project_id}/{self.bucket_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS client: {e}")
                logger.warning("Will use local cache only")
        else:
            logger.warning("GCS not available. Using local cache only.")

    def load_ticker_data(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a ticker with memory optimization.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime)
            use_cache: Use local cache if available
            force_refresh: Force download from GCS even if cached
            columns: Specific columns to load (None = load all)

        Returns:
            DataFrame with OHLCV data indexed by date (memory optimized)
        """
        # Convert dates
        if start_date:
            start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)

        # Check cache first (try parquet, then csv)
        cache_file_parquet = self.cache_dir / f"{ticker}.parquet"
        cache_file_csv = self.cache_dir / f"{ticker}.csv"

        # Determine cache file to use
        if self.cache_format == 'parquet':
            cache_file = cache_file_parquet
            fallback_cache = cache_file_csv
        else:
            cache_file = cache_file_csv
            fallback_cache = cache_file_parquet

        if use_cache and not force_refresh:
            # Try primary cache format
            if cache_file.exists():
                logger.debug(f"Loading {ticker} from cache: {cache_file}")
                df = self._load_from_cache(cache_file, start_date, end_date, columns)
                if not df.empty:
                    return df

            # Try fallback format
            if fallback_cache.exists():
                logger.debug(f"Loading {ticker} from fallback cache: {fallback_cache}")
                df = self._load_from_cache(fallback_cache, start_date, end_date, columns)
                if not df.empty:
                    # Convert to preferred format
                    self._save_to_cache(df, ticker)
                    return df

        # Download from GCS
        if self.gcs_client and self.bucket:
            logger.info(f"Downloading {ticker} from GCS...")
            df = self._load_from_gcs(ticker)

            if not df.empty:
                # Optimize memory before caching
                if self.use_memory_optimization:
                    df = self.memory_optimizer.optimize_ohlcv_dataframe(df)

                # Filter by minimum price threshold
                df = self._filter_by_min_price(df, ticker)

                # Save to cache (after price filtering)
                self._save_to_cache(df, ticker)
                logger.debug(f"Cached {ticker} to {cache_file}")

                # Filter by date range
                if start_date or end_date:
                    df = self._filter_by_date(df, start_date, end_date)

                # Select columns if specified
                if columns:
                    df = self._select_columns(df, columns)

                return df

        # If GCS fails or not available, check cache anyway
        if cache_file.exists():
            logger.warning(f"GCS failed, using cached data for {ticker}")
            return self._load_from_cache(cache_file, start_date, end_date, columns)

        if fallback_cache.exists():
            logger.warning(f"GCS failed, using fallback cached data for {ticker}")
            return self._load_from_cache(fallback_cache, start_date, end_date, columns)

        logger.error(f"No data found for {ticker}")
        return pd.DataFrame()

    def _load_from_cache(
        self,
        cache_file: Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load data from local cache file (CSV or Parquet).

        Args:
            cache_file: Path to cache file
            start_date: Filter start date
            end_date: Filter end date
            columns: Specific columns to load (Parquet only)

        Returns:
            Loaded DataFrame (memory optimized)
        """
        try:
            # Load based on file extension
            if cache_file.suffix == '.parquet':
                # Parquet supports selective column loading
                df = pd.read_parquet(cache_file, columns=columns)

                # If date is already the index, reset it to column for normalization
                if isinstance(df.index, pd.DatetimeIndex) and df.index.name in ['date', 'Date', 'DATE']:
                    df = df.reset_index()
            else:
                # CSV loading - don't assume date is index
                df = pd.read_csv(cache_file)

            # Normalize columns (handles any naming variations)
            ticker = cache_file.stem
            df = self._normalize_columns(df, ticker)

            if df.empty:
                logger.warning(f"Failed to normalize cache file: {cache_file}")
                return pd.DataFrame()

            # Validate required columns (should have them after normalization)
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Cache file missing required columns after normalization: {cache_file}")
                return pd.DataFrame()

            # Apply memory optimization
            if self.use_memory_optimization and self.memory_optimizer:
                df = self.memory_optimizer.optimize_ohlcv_dataframe(df)
            else:
                # Ensure float32 for TA-Lib compatibility (even without full optimization)
                for col in required_cols:
                    if col in df.columns:
                        if df[col].dtype == np.float64:
                            df[col] = df[col].astype(np.float32)

            # Filter by minimum price threshold
            ticker = cache_file.stem  # Extract ticker from filename
            df = self._filter_by_min_price(df, ticker)

            # Filter by date
            if start_date or end_date:
                df = self._filter_by_date(df, start_date, end_date)

            return df
        except Exception as e:
            logger.error(f"Failed to load from cache {cache_file}: {e}")
            return pd.DataFrame()

    def _load_from_gcs(self, ticker: str) -> pd.DataFrame:
        """
        Load data from GCS bucket with memory optimization.

        Priority: Parquet files first (faster, smaller), then CSV.
        If CSV is found but Parquet doesn't exist, converts and uploads Parquet to GCS.

        Returns DataFrame with float32 types for memory efficiency.
        """
        if not self.bucket:
            return pd.DataFrame()

        # PRIORITY 1: Try Parquet files first (preferred format)
        parquet_paths = [
            f"tickers/{ticker}.parquet",
            f"{ticker}.parquet",
            f"daily/{ticker}.parquet",
            f"data/{ticker}.parquet",
        ]

        # PRIORITY 2: CSV files (fallback, will be converted to Parquet)
        csv_paths = [
            f"tickers/{ticker}.csv",
            f"tickers/{ticker}/data.csv",
            f"tickers/{ticker}/twelvedata.csv",
            f"{ticker}.csv",
            f"{ticker}_full_history.csv",  # NEW: Handle _full_history suffix
            f"daily/{ticker}.csv",
            f"data/{ticker}.csv",
        ]

        # Combine: Parquet first, then CSV
        possible_paths = parquet_paths + csv_paths

        loaded_from_csv = False
        csv_source_path = None

        for gcs_path in possible_paths:
            try:
                blob = self.bucket.blob(gcs_path)

                if not blob.exists():
                    continue

                # Download to memory
                data = blob.download_as_bytes()

                # Parse based on file type
                if gcs_path.endswith('.csv'):
                    from io import BytesIO
                    # Don't assume 'date' column exists - normalization will handle it
                    df = pd.read_csv(BytesIO(data))
                    loaded_from_csv = True
                    csv_source_path = gcs_path
                elif gcs_path.endswith('.parquet'):
                    from io import BytesIO
                    df = pd.read_parquet(BytesIO(data))

                    # If date is already the index, reset it to column for normalization
                    if isinstance(df.index, pd.DatetimeIndex) and df.index.name in ['date', 'Date', 'DATE']:
                        df = df.reset_index()

                    loaded_from_csv = False
                else:
                    continue

                # Intelligent column detection and normalization
                df = self._normalize_columns(df, ticker)

                if df.empty:
                    logger.debug(f"Failed to normalize columns for {gcs_path}")
                    continue

                # Validate required columns (should have them after normalization)
                required_cols = ['open', 'high', 'low', 'close', 'volume']

                # DEBUG: Log validation status
                columns_ok = all(col in df.columns for col in required_cols)
                index_ok = isinstance(df.index, pd.DatetimeIndex)
                logger.debug(f"{gcs_path}: columns_ok={columns_ok} (found={list(df.columns)}), index_ok={index_ok} (type={type(df.index).__name__})")

                if columns_ok and index_ok:
                    # Apply memory optimization
                    if self.use_memory_optimization and self.memory_optimizer:
                        df = self.memory_optimizer.optimize_ohlcv_dataframe(df)
                    else:
                        # At minimum, use float32 for OHLCV columns
                        for col in required_cols:
                            df[col] = df[col].astype(np.float32)

                    logger.info(f"Loaded {ticker} from GCS: {gcs_path} ({len(df)} rows)")

                    # If loaded from CSV, convert to Parquet and upload to GCS
                    if loaded_from_csv:
                        self._convert_and_upload_parquet(df, ticker, csv_source_path)

                    return df.sort_index()
                else:
                    logger.warning(f"File {gcs_path} missing required columns")

            except Exception as e:
                logger.debug(f"Failed to load {gcs_path}: {e}")
                continue

        logger.warning(f"No valid data found in GCS for {ticker}")
        return pd.DataFrame()

    def _normalize_columns(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Intelligently detect and normalize column names with MAXIMUM FLEXIBILITY.

        Handles extensive variations including:
        - Date: date, Date, DATE, timestamp, datetime, Time, Datum, t, dt, index, etc.
        - Open: open, Open, OPEN, o, O, öffnung, opening, etc.
        - High: high, High, HIGH, h, H, hoch, highest, etc.
        - Low: low, Low, LOW, l, L, tief, lowest, etc.
        - Close: close, Close, CLOSE, c, C, schluss, closing, adj_close, etc.
        - Volume: volume, Volume, VOLUME, vol, v, V, volumen, etc.

        Uses fuzzy matching and partial string matching for maximum compatibility.

        Args:
            df: DataFrame with potentially non-standard column names
            ticker: Ticker symbol (for logging)

        Returns:
            DataFrame with standardized lowercase column names and date index
        """
        if df.empty:
            return df

        # EXPANDED column mapping: variations -> standard name (case-insensitive)
        column_map = {
            # Date variations (will become index) - MASSIVELY EXPANDED
            'date': [
                # English variations
                'date', 'Date', 'DATE', 'timestamp', 'Timestamp', 'TIMESTAMP',
                'datetime', 'DateTime', 'DATETIME', 'time', 'Time', 'TIME',
                't', 'T', 'dt', 'DT', 'dates', 'Dates', 'DATES',
                # German variations
                'Datum', 'datum', 'DATUM', 'Zeit', 'zeit', 'ZEIT',
                # Other common variations
                'index', 'Index', 'INDEX', 'day', 'Day', 'DAY',
                'trading_date', 'TradingDate', 'TRADING_DATE',
                'period', 'Period', 'PERIOD', 'fecha', 'Fecha', 'FECHA',
                # Unnamed index column
                'Unnamed: 0', 'unnamed: 0'
            ],
            # OHLCV variations - MASSIVELY EXPANDED
            'open': [
                # Standard
                'open', 'Open', 'OPEN', 'opening', 'Opening', 'OPENING',
                # Abbreviations
                'o', 'O', 'open_price', 'Open_Price', 'OPEN_PRICE',
                # With underscores/prefixes
                'open_', '_open', 'price_open', 'Price_Open',
                # German
                'öffnung', 'Öffnung', 'ÖFFNUNG', 'eröffnung', 'Eröffnung',
                # Adjusted
                'adj_open', 'Adj_Open', 'ADJ_OPEN', 'adjusted_open'
            ],
            'high': [
                # Standard
                'high', 'High', 'HIGH', 'highest', 'Highest', 'HIGHEST',
                # Abbreviations
                'h', 'H', 'high_price', 'High_Price', 'HIGH_PRICE',
                # With underscores/prefixes
                'high_', '_high', 'price_high', 'Price_High',
                # German
                'hoch', 'Hoch', 'HOCH', 'höchst', 'Höchst', 'HÖCHST',
                # Adjusted
                'adj_high', 'Adj_High', 'ADJ_HIGH', 'adjusted_high'
            ],
            'low': [
                # Standard
                'low', 'Low', 'LOW', 'lowest', 'Lowest', 'LOWEST',
                # Abbreviations
                'l', 'L', 'low_price', 'Low_Price', 'LOW_PRICE',
                # With underscores/prefixes
                'low_', '_low', 'price_low', 'Price_Low',
                # German
                'tief', 'Tief', 'TIEF', 'tiefst', 'Tiefst', 'TIEFST',
                # Adjusted
                'adj_low', 'Adj_Low', 'ADJ_LOW', 'adjusted_low'
            ],
            'close': [
                # Standard
                'close', 'Close', 'CLOSE', 'closing', 'Closing', 'CLOSING',
                # Abbreviations
                'c', 'C', 'close_price', 'Close_Price', 'CLOSE_PRICE',
                # With underscores/prefixes
                'close_', '_close', 'price_close', 'Price_Close',
                # German
                'schluss', 'Schluss', 'SCHLUSS', 'schlusskurs', 'Schlusskurs',
                # Adjusted variations (IMPORTANT - many datasets use adjusted close)
                'adj_close', 'Adj_Close', 'ADJ_CLOSE', 'Adj Close', 'Adj. Close',
                'adjusted_close', 'Adjusted_Close', 'ADJUSTED_CLOSE',
                'adjclose', 'AdjClose', 'ADJCLOSE'
            ],
            'volume': [
                # Standard
                'volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'VOL',
                # Abbreviations
                'v', 'V', 'trading_volume', 'Trading_Volume', 'TRADING_VOLUME',
                # With underscores/prefixes
                'volume_', '_volume', 'vol_', '_vol',
                # German
                'volumen', 'Volumen', 'VOLUMEN', 'handelsvolumen', 'Handelsvolumen',
                # Other variations
                'shares', 'Shares', 'SHARES', 'quantity', 'Quantity', 'QUANTITY'
            ]
        }

        # Find and rename columns (case-insensitive matching)
        rename_map = {}
        found_columns = set()

        # First pass: exact matching
        for standard_name, variations in column_map.items():
            for col in df.columns:
                if col in variations:
                    rename_map[col] = standard_name
                    found_columns.add(standard_name)
                    break

        # Second pass: fuzzy matching for missing columns (case-insensitive partial match)
        if len(found_columns) < 6:  # Missing some required columns
            for standard_name, variations in column_map.items():
                if standard_name in found_columns:
                    continue  # Already found

                for col in df.columns:
                    col_lower = str(col).lower().strip()

                    # Partial matching: if column contains any of the variations
                    for var in variations:
                        var_lower = str(var).lower().strip()
                        if var_lower in col_lower or col_lower in var_lower:
                            rename_map[col] = standard_name
                            found_columns.add(standard_name)
                            logger.info(f"{ticker}: Fuzzy matched '{col}' → '{standard_name}' (via '{var}')")
                            break
                    if standard_name in found_columns:
                        break

        # Check if we found all required columns
        required = {'open', 'high', 'low', 'close', 'volume', 'date'}
        missing = required - found_columns

        if missing:
            logger.warning(f"{ticker}: Missing columns {missing}. Available: {list(df.columns)}")
            logger.info(f"{ticker}: Found columns: {found_columns}")
            logger.info(f"{ticker}: Column mapping attempted: {rename_map}")
            return pd.DataFrame()

        # Rename columns
        df = df.rename(columns=rename_map)

        # Set date as index with FLEXIBLE parsing
        if 'date' in df.columns:
            try:
                # Try multiple date parsing strategies with UTC conversion to handle mixed timezones
                df['date'] = pd.to_datetime(
                    df['date'],
                    utc=True,  # Convert to UTC to handle mixed timezones
                    errors='coerce'  # Convert invalid dates to NaT
                )

                # Drop rows with invalid dates
                initial_len = len(df)
                df = df.dropna(subset=['date'])
                if len(df) < initial_len:
                    logger.info(f"{ticker}: Dropped {initial_len - len(df)} rows with invalid dates")

                # Convert timezone-aware to timezone-naive (removes timezone info)
                if df['date'].dt.tz is not None:
                    df['date'] = df['date'].dt.tz_localize(None)

                df = df.set_index('date')
                df = df.sort_index()

                logger.debug(f"{ticker}: Successfully parsed date column. Date range: {df.index.min()} to {df.index.max()}")
            except Exception as e:
                logger.warning(f"{ticker}: Failed to parse date column: {e}")
                return pd.DataFrame()
        else:
            logger.warning(f"{ticker}: Date column not found after mapping")
            return pd.DataFrame()

        # Keep only standard columns (remove extras)
        standard_cols = ['open', 'high', 'low', 'close', 'volume']
        extra_cols = [col for col in df.columns if col not in standard_cols]
        if extra_cols:
            df = df[standard_cols]

        logger.debug(f"{ticker}: Normalization complete. Columns: {list(df.columns)}, Rows: {len(df)}")

        return df

    def _convert_and_upload_parquet(
        self,
        df: pd.DataFrame,
        ticker: str,
        csv_source_path: str
    ) -> bool:
        """
        Convert DataFrame to Parquet and upload to GCS.

        Args:
            df: DataFrame to convert (already memory-optimized)
            ticker: Ticker symbol
            csv_source_path: Original CSV path in GCS (used to determine upload location)

        Returns:
            True if upload successful, False otherwise
        """
        if not self.bucket:
            return False

        try:
            # Determine Parquet path based on CSV source path
            # Example: tickers/AAPL.csv → tickers/AAPL.parquet
            if csv_source_path.endswith('.csv'):
                parquet_path = csv_source_path[:-4] + '.parquet'
            else:
                # Fallback: use standard path
                parquet_path = f"tickers/{ticker}.parquet"

            # Check if Parquet already exists in GCS
            parquet_blob = self.bucket.blob(parquet_path)
            if parquet_blob.exists():
                logger.debug(f"Parquet already exists in GCS: {parquet_path}")
                return False

            # Convert to Parquet in memory
            from io import BytesIO
            parquet_buffer = BytesIO()

            # Write Parquet with compression
            df.to_parquet(
                parquet_buffer,
                compression='snappy',
                index=True
            )

            parquet_buffer.seek(0)
            parquet_data = parquet_buffer.read()

            # Calculate size reduction
            csv_blob = self.bucket.blob(csv_source_path)
            csv_size_mb = csv_blob.size / (1024 * 1024)
            parquet_size_mb = len(parquet_data) / (1024 * 1024)
            compression_ratio = csv_size_mb / parquet_size_mb if parquet_size_mb > 0 else 0

            # Upload to GCS
            parquet_blob.upload_from_string(
                parquet_data,
                content_type='application/octet-stream'
            )

            logger.info(
                f"✓ Converted {ticker} CSV→Parquet and uploaded to GCS\n"
                f"  Path: {parquet_path}\n"
                f"  Size: {csv_size_mb:.2f}MB → {parquet_size_mb:.2f}MB "
                f"({compression_ratio:.1f}x compression)"
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to upload Parquet for {ticker}: {e}")
            return False

    def _save_to_cache(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Save DataFrame to cache in configured format.

        Args:
            df: DataFrame to cache
            ticker: Ticker symbol
        """
        try:
            if self.cache_format == 'parquet':
                cache_file = self.cache_dir / f"{ticker}.parquet"
                df.to_parquet(
                    cache_file,
                    compression='snappy',  # Fast compression
                    index=True
                )
            else:
                cache_file = self.cache_dir / f"{ticker}.csv"
                df.to_csv(cache_file, index=True)

            logger.debug(f"Saved {ticker} to cache: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save {ticker} to cache: {e}")

    def _select_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Select specific columns from DataFrame.

        Args:
            df: Input DataFrame
            columns: Columns to select

        Returns:
            DataFrame with selected columns
        """
        available_cols = [col for col in columns if col in df.columns]
        if not available_cols:
            logger.warning(f"None of the requested columns found: {columns}")
            return df

        return df[available_cols]

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Filter dataframe by date range."""
        if df.empty:
            return df

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def _filter_by_min_price(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Filter out data where any OHLC price is below minimum threshold.

        This is important for micro/small cap stocks to exclude penny stocks
        that may be delisted or in severe distress.

        Args:
            df: DataFrame with OHLC data
            ticker: Ticker symbol (for logging)

        Returns:
            Filtered DataFrame with rows where all prices >= min_price
        """
        if df.empty or self.min_price <= 0:
            return df

        original_len = len(df)

        # Check if any OHLC price is below threshold
        price_cols = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_cols if col in df.columns]

        if not available_price_cols:
            return df

        # Keep rows where ALL price columns are >= min_price
        mask = pd.Series(True, index=df.index)
        for col in available_price_cols:
            mask &= (df[col] >= self.min_price)

        df_filtered = df[mask].copy()

        filtered_count = original_len - len(df_filtered)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} rows from {ticker} where price < ${self.min_price:.2f}")

        return df_filtered

    def load_multiple_tickers(
        self,
        tickers: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True,
        min_data_days: Optional[int] = None,
        batch_size: Optional[int] = None,
        force_gc_between_batches: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple tickers with memory optimization.

        MEMORY OPTIMIZATION:
        - Processes tickers in batches to limit memory usage
        - Forces garbage collection between batches
        - Monitors memory usage and warns if approaching limits

        Args:
            tickers: List of ticker symbols
            start_date: Start date (None = load all available history)
            end_date: End date (None = load until most recent)
            use_cache: Use local cache
            min_data_days: Minimum days of data required (e.g., 504 for 2 years)
            batch_size: Process tickers in batches (None = no batching, 10 recommended for 8GB RAM)
            force_gc_between_batches: Force garbage collection between batches

        Returns:
            Dictionary mapping ticker -> DataFrame (memory optimized)
        """
        data = {}
        excluded_count = 0
        insufficient_history = []

        # Memory monitoring
        if self.memory_monitor:
            self.memory_monitor.print_memory_usage()

        # Process in batches if specified
        if batch_size:
            ticker_batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
            logger.info(f"Processing {len(tickers)} tickers in {len(ticker_batches)} batches of {batch_size}")
        else:
            ticker_batches = [tickers]

        for batch_idx, batch in enumerate(ticker_batches):
            if len(ticker_batches) > 1:
                logger.info(f"\nProcessing batch {batch_idx + 1}/{len(ticker_batches)}...")

            for ticker in batch:
                logger.info(f"Loading data for {ticker}...")

                # Load complete history first (no date filtering)
                df = self.load_ticker_data(ticker, start_date=None, end_date=None, use_cache=use_cache)

                if df.empty:
                    logger.warning(f"No data loaded for {ticker}")
                    excluded_count += 1
                    continue

                # Check minimum data requirement
                if min_data_days and len(df) < min_data_days:
                    days_available = len(df)
                    years_available = days_available / 252
                    logger.info(f"Excluding {ticker}: Only {days_available} days ({years_available:.1f} years) of data "
                               f"(minimum {min_data_days} required)")
                    insufficient_history.append({
                        'ticker': ticker,
                        'days': days_available,
                        'years': years_available
                    })
                    excluded_count += 1
                    continue

                # Apply date filtering if requested (after checking minimum)
                if start_date or end_date:
                    df = self._filter_by_date(df,
                                             pd.to_datetime(start_date) if start_date else None,
                                             pd.to_datetime(end_date) if end_date else None)

                if not df.empty:
                    data_span_years = (df.index.max() - df.index.min()).days / 365.25
                    logger.info(f"Loaded {ticker}: {len(df)} bars spanning {data_span_years:.1f} years "
                               f"({df.index.min().date()} to {df.index.max().date()})")
                    data[ticker] = df
                else:
                    logger.warning(f"No data for {ticker} after date filtering")
                    excluded_count += 1

            # Garbage collection between batches
            if force_gc_between_batches and batch_idx < len(ticker_batches) - 1:
                if self.memory_monitor:
                    collected = self.memory_monitor.force_garbage_collection(verbose=True)
                else:
                    gc.collect()

        # Summary statistics
        included_count = len(data)
        logger.info(f"\n{'='*70}")
        logger.info(f"DATA LOADING SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total tickers processed:     {len(tickers)}")
        logger.info(f"Successfully loaded:         {included_count} ({included_count/len(tickers)*100:.1f}%)")
        logger.info(f"Excluded (no data/filters):  {excluded_count}")

        if min_data_days:
            logger.info(f"Minimum history required:    {min_data_days} days ({min_data_days/252:.1f} years)")
            logger.info(f"Excluded (insufficient):     {len(insufficient_history)}")

        if data:
            avg_bars = np.mean([len(df) for df in data.values()])
            avg_years = avg_bars / 252
            logger.info(f"Average bars per ticker:     {avg_bars:.0f} ({avg_years:.1f} years)")

            # Memory usage for loaded data
            if self.memory_monitor:
                total_mem_mb = sum(get_dataframe_memory_usage(df)['total_mb'] for df in data.values())
                logger.info(f"Total data memory usage:     {total_mem_mb:.1f} MB")
                logger.info(f"Average per ticker:          {total_mem_mb/len(data):.1f} MB")

        logger.info(f"{'='*70}\n")

        # Final memory check
        if self.memory_monitor:
            self.memory_monitor.print_memory_usage()

        return data

    def get_available_tickers(self, from_gcs: bool = True) -> List[str]:
        """
        Get list of available tickers from cache or GCS.

        Args:
            from_gcs: If True, list from GCS; if False, list from local cache

        Returns:
            List of ticker symbols
        """
        tickers = []

        if from_gcs and self.gcs_client and self.bucket:
            # List tickers from GCS
            logger.info("Listing tickers from GCS...")
            try:
                # SCAN 1: tickers/ directory - PROPER ticker name extraction
                blobs_tickers = self.bucket.list_blobs(prefix='tickers/')

                for blob in blobs_tickers:
                    # Skip directories
                    if blob.name.endswith('/'):
                        continue

                    # Extract ticker from path
                    # Handle: tickers/AAPL.csv or tickers/0001.KL.csv or tickers/AAPL/data.csv
                    parts = blob.name.split('/')

                    if len(parts) >= 2:
                        # If file is tickers/AAPL.csv or tickers/0001.KL.csv
                        if parts[-1].endswith(('.csv', '.parquet')):
                            # CORRECT extraction: Remove file extension completely
                            # "0001.KL.csv" -> "0001.KL" (not "0001"!)
                            filename = parts[-1]
                            if filename.endswith('.csv'):
                                ticker = filename[:-4]  # Remove .csv
                            elif filename.endswith('.parquet'):
                                ticker = filename[:-8]  # Remove .parquet
                            else:
                                continue

                            # Skip timestamped files like "twelvedata_20250825"
                            if not any(x in ticker for x in ['twelvedata', 'data_', 'prices_']):
                                if ticker not in tickers:
                                    tickers.append(ticker)
                        # If subdirectory like tickers/AAPL/data.csv
                        elif len(parts) >= 3:
                            ticker = parts[1]
                            if ticker not in tickers:
                                tickers.append(ticker)

                # SCAN 2: Root directory (for files like ADNT_full_history.csv)
                blobs_root = self.bucket.list_blobs(max_results=10000)  # Limit to prevent huge lists

                for blob in blobs_root:
                    filename = blob.name

                    # Skip directories and files in subdirectories
                    if '/' in filename or filename.endswith('/'):
                        continue

                    # Handle _full_history.csv pattern
                    if filename.endswith('_full_history.csv'):
                        ticker = filename.replace('_full_history.csv', '')
                        if ticker not in tickers:
                            tickers.append(ticker)

                    # Also handle plain CSV files in root
                    elif filename.endswith('.csv'):
                        ticker = filename.replace('.csv', '')
                        # Skip common non-ticker files
                        if not any(x in ticker for x in ['twelvedata', 'data_', 'prices_', 'metadata']):
                            if ticker not in tickers:
                                tickers.append(ticker)

                logger.info(f"Found {len(tickers)} tickers in GCS")

            except Exception as e:
                logger.error(f"Error listing GCS tickers: {e}")
                logger.info("Falling back to local cache...")
                from_gcs = False

        # Fall back to local cache if GCS fails or not requested
        if not from_gcs or not tickers:
            for file in self.cache_dir.glob('*.csv'):
                tickers.append(file.stem)

            for file in self.cache_dir.glob('*.parquet'):
                if file.stem not in tickers:
                    tickers.append(file.stem)

            logger.info(f"Found {len(tickers)} tickers in local cache")

        return sorted(tickers)

    def get_cache_info(self, ticker: str) -> Optional[Dict]:
        """
        Get information about cached ticker data.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with cache info or None if not cached
        """
        cache_file = self.cache_dir / f"{ticker}.csv"

        if not cache_file.exists():
            return None

        try:
            df = pd.read_csv(cache_file, parse_dates=['date'], index_col='date')

            return {
                'ticker': ticker,
                'file_path': str(cache_file),
                'file_size_mb': cache_file.stat().st_size / (1024 * 1024),
                'start_date': df.index.min(),
                'end_date': df.index.max(),
                'num_rows': len(df),
                'columns': list(df.columns)
            }
        except Exception as e:
            logger.error(f"Failed to get cache info for {ticker}: {e}")
            return None

    def clear_cache(self, ticker: Optional[str] = None, format: Optional[str] = None) -> None:
        """
        Clear cached data.

        Args:
            ticker: Specific ticker to clear, or None to clear all
            format: Cache format to clear ('csv', 'parquet', or None for both)
        """
        if ticker:
            files_to_clear = []
            if format is None or format == 'csv':
                csv_file = self.cache_dir / f"{ticker}.csv"
                if csv_file.exists():
                    files_to_clear.append(csv_file)

            if format is None or format == 'parquet':
                parquet_file = self.cache_dir / f"{ticker}.parquet"
                if parquet_file.exists():
                    files_to_clear.append(parquet_file)

            for file in files_to_clear:
                file.unlink()
                logger.info(f"Cleared cache: {file}")

            if not files_to_clear:
                logger.warning(f"No cache files found for {ticker}")
        else:
            cleared_count = 0
            if format is None or format == 'csv':
                for file in self.cache_dir.glob('*.csv'):
                    file.unlink()
                    cleared_count += 1

            if format is None or format == 'parquet':
                for file in self.cache_dir.glob('*.parquet'):
                    file.unlink()
                    cleared_count += 1

            logger.info(f"Cleared {cleared_count} cache files")

    def get_cache_size_mb(self) -> float:
        """
        Get total size of cache directory in MB.

        Returns:
            Total cache size in MB
        """
        total_size = 0
        for file in self.cache_dir.glob('*'):
            if file.is_file():
                total_size += file.stat().st_size

        return total_size / (1024 ** 2)


# Convenience function for quick loading
def load_ticker_data(
    ticker: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None
) -> pd.DataFrame:
    """
    Quick function to load ticker data.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD or datetime)
        end_date: End date (YYYY-MM-DD or datetime)

    Returns:
        DataFrame with OHLCV data
    """
    loader = DataLoader()
    return loader.load_ticker_data(ticker, start_date, end_date)
