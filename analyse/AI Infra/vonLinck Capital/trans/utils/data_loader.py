"""
Data Loader Wrapper
===================

Provides a simple interface to load ticker data from various sources.
Wraps the core DataLoader for easy importing.

Usage:
    from utils.data_loader import load_ticker_data

    df = load_ticker_data('AAPL', start_date='2024-01-01')
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import the core DataLoader
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.aiv7_components.data_loader import DataLoader
    DATALOADER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import core DataLoader: {e}")
    DATALOADER_AVAILABLE = False
    DataLoader = None


# Global data loader instance (lazy loaded)
_data_loader: Optional['DataLoader'] = None


def get_data_loader() -> Optional['DataLoader']:
    """Get or create the global DataLoader instance."""
    global _data_loader

    if not DATALOADER_AVAILABLE:
        return None

    if _data_loader is None:
        try:
            _data_loader = DataLoader()
        except Exception as e:
            logger.warning(f"Could not initialize DataLoader: {e}")
            return None

    return _data_loader


def load_ticker_data(
    ticker: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    use_cache: bool = True,
    validate: bool = False
) -> Optional[pd.DataFrame]:
    """
    Load OHLCV data for a ticker.

    Tries multiple sources:
    1. Local cache (data/raw/)
    2. GCS bucket (if configured)

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD or datetime)
        end_date: End date (YYYY-MM-DD or datetime)
        use_cache: Use local cache if available
        validate: Perform data validation

    Returns:
        DataFrame with OHLCV data or None if not found
    """
    return _load_daily_data(ticker, start_date, end_date, use_cache, validate)


def _load_daily_data(
    ticker: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    use_cache: bool = True,
    validate: bool = False
) -> Optional[pd.DataFrame]:
    """
    Load daily OHLCV data for a ticker (internal function).

    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
        use_cache: Use local cache if available
        validate: Perform data validation

    Returns:
        DataFrame with daily OHLCV data or None if not found
    """
    loader = get_data_loader()

    if loader is not None:
        try:
            df = loader.load_ticker(
                ticker,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
                validate=validate
            )
            if df is not None:
                # Ensure date column exists
                if 'date' not in df.columns:
                    df = df.reset_index()
                return df
        except Exception as e:
            logger.debug(f"{ticker}: DataLoader failed: {e}")

    # Fallback: Try local files directly
    return _load_local(ticker, start_date, end_date)


def _load_local(
    ticker: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None
) -> Optional[pd.DataFrame]:
    """
    Fallback: Load directly from local data directory.
    """
    # Try multiple possible locations
    base_paths = [
        Path(__file__).parent.parent / 'data' / 'raw',
        Path(__file__).parent.parent.parent / 'data' / 'raw',
        Path(os.getcwd()) / 'data' / 'raw',
    ]

    for base_path in base_paths:
        if not base_path.exists():
            continue

        # Try parquet
        parquet_path = base_path / f"{ticker}.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                return _normalize_and_filter(df, start_date, end_date)
            except Exception as e:
                logger.debug(f"Failed to load {parquet_path}: {e}")

        # Try CSV
        csv_path = base_path / f"{ticker}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                return _normalize_and_filter(df, start_date, end_date)
            except Exception as e:
                logger.debug(f"Failed to load {csv_path}: {e}")

    return None


def _normalize_and_filter(
    df: pd.DataFrame,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None
) -> pd.DataFrame:
    """
    Normalize column names and filter by date range.
    """
    # Handle DatetimeIndex
    if 'date' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            # Find the date column
            for col in ['datetime', 'index', 'Date']:
                if col in df.columns:
                    df = df.rename(columns={col: 'date'})
                    break
            else:
                # First column is likely date
                first_col = df.columns[0]
                if pd.api.types.is_datetime64_any_dtype(df[first_col]):
                    df = df.rename(columns={first_col: 'date'})

    # Normalize column names
    column_map = {
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Adj Close': 'adj_close',  # Preserve adjusted close for accurate historical market cap
        'adjClose': 'adj_close'     # Alternative naming convention
    }
    df = df.rename(columns=column_map)

    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Keep standard columns (including adj_close for split-adjusted market cap calculations)
    standard_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
    existing_cols = [c for c in standard_cols if c in df.columns]
    df = df[existing_cols]

    # Filter by date range
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df['date'] >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df['date'] <= end_date]

    return df.sort_values('date').reset_index(drop=True)


if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)

    df = load_ticker_data('AAPL', start_date='2024-01-01')

    if df is not None:
        print(f"Loaded {len(df)} rows")
        print(df.head())
    else:
        print("No data found")
