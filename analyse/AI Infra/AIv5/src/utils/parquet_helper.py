"""
Parquet I/O Helper

Provides fast Parquet reading/writing with CSV fallback.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import pyarrow
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("pyarrow not available - falling back to CSV")


def read_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read data from Parquet or CSV.

    Auto-detects format based on extension.
    Falls back to CSV if Parquet unavailable.

    Args:
        file_path: Path to file (.parquet or .csv)
        **kwargs: Additional arguments for pd.read_csv or pd.read_parquet

    Returns:
        DataFrame
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check extension
    if file_path.suffix.lower() in ['.parquet', '.pq']:
        if PARQUET_AVAILABLE:
            return pd.read_parquet(file_path, **kwargs)
        else:
            # Look for CSV version
            csv_path = file_path.with_suffix('.csv')
            if csv_path.exists():
                logger.info(f"Parquet not available, reading CSV: {csv_path}")
                return pd.read_csv(csv_path, **kwargs)
            else:
                raise ImportError("pyarrow not installed and no CSV fallback found")
    else:
        # CSV
        return pd.read_csv(file_path, **kwargs)


def write_data(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    format: str = 'auto',
    compression: Optional[str] = 'snappy',
    **kwargs
) -> Path:
    """
    Write DataFrame to Parquet or CSV.

    Args:
        df: DataFrame to write
        file_path: Output path
        format: 'parquet', 'csv', or 'auto' (uses extension)
        compression: Compression for Parquet ('snappy', 'gzip', 'zstd', None)
        **kwargs: Additional arguments

    Returns:
        Path to written file
    """
    file_path = Path(file_path)

    # Auto-detect format
    if format == 'auto':
        if file_path.suffix.lower() in ['.parquet', '.pq']:
            format = 'parquet'
        else:
            format = 'csv'

    # Write
    if format == 'parquet':
        if PARQUET_AVAILABLE:
            df.to_parquet(file_path, compression=compression, **kwargs)
            logger.info(f"Wrote Parquet: {file_path} ({len(df):,} rows)")
        else:
            # Fallback to CSV
            csv_path = file_path.with_suffix('.csv')
            logger.warning(f"Parquet not available, writing CSV: {csv_path}")
            df.to_csv(csv_path, index=False, **kwargs)
            return csv_path
    else:
        # CSV
        df.to_csv(file_path, index=False, **kwargs)
        logger.info(f"Wrote CSV: {file_path} ({len(df):,} rows)")

    return file_path


def convert_csv_to_parquet(
    csv_path: Union[str, Path],
    parquet_path: Optional[Union[str, Path]] = None,
    compression: str = 'snappy',
    delete_csv: bool = False
) -> Path:
    """
    Convert CSV to Parquet.

    Args:
        csv_path: Input CSV file
        parquet_path: Output Parquet file (default: same name with .parquet)
        compression: Compression method
        delete_csv: Delete CSV after successful conversion

    Returns:
        Path to Parquet file
    """
    if not PARQUET_AVAILABLE:
        raise ImportError("pyarrow required for Parquet conversion")

    csv_path = Path(csv_path)

    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet')
    else:
        parquet_path = Path(parquet_path)

    logger.info(f"Converting {csv_path} to Parquet...")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Write Parquet
    df.to_parquet(parquet_path, compression=compression)

    # Check sizes
    csv_size = csv_path.stat().st_size
    parquet_size = parquet_path.stat().st_size
    reduction = (1 - parquet_size / csv_size) * 100

    logger.info(f"Conversion complete:")
    logger.info(f"  CSV:     {csv_size:,} bytes")
    logger.info(f"  Parquet: {parquet_size:,} bytes")
    logger.info(f"  Reduction: {reduction:.1f}%")

    if delete_csv:
        csv_path.unlink()
        logger.info(f"Deleted CSV: {csv_path}")

    return parquet_path


def get_file_info(file_path: Union[str, Path]) -> dict:
    """
    Get file information.

    Returns dict with:
    - format: 'parquet' or 'csv'
    - size_bytes: File size
    - rows: Number of rows (if readable)
    - columns: Number of columns (if readable)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {'exists': False}

    info = {
        'exists': True,
        'path': str(file_path),
        'size_bytes': file_path.stat().st_size,
    }

    # Detect format
    if file_path.suffix.lower() in ['.parquet', '.pq']:
        info['format'] = 'parquet'
    else:
        info['format'] = 'csv'

    # Try to get row/column counts
    try:
        df = read_data(file_path)
        info['rows'] = len(df)
        info['columns'] = len(df.columns)
    except Exception as e:
        logger.debug(f"Could not read file for stats: {e}")

    return info
