"""
Parquet file utilities for efficient data loading and processing.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ParquetDataLoader:
    """Efficient Parquet file loader with schema validation and batch processing."""

    def __init__(self,
                 data_path: Union[str, Path],
                 schema: Optional[pa.Schema] = None,
                 batch_size: int = 10000,
                 n_jobs: int = 4):
        """
        Initialize Parquet data loader.

        Args:
            data_path: Path to Parquet files or directory
            schema: Expected PyArrow schema for validation
            batch_size: Size of batches for processing
            n_jobs: Number of parallel workers
        """
        self.data_path = Path(data_path)
        self.schema = schema or self._default_schema()
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    @staticmethod
    def _default_schema() -> pa.Schema:
        """Define default schema for financial time series data."""
        return pa.schema([
            ('timestamp', pa.timestamp('ns')),
            ('open', pa.float64()),
            ('high', pa.float64()),
            ('low', pa.float64()),
            ('close', pa.float64()),
            ('volume', pa.float64()),
            ('symbol', pa.string()),
        ])

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame against expected schema.

        Args:
            df: DataFrame to validate

        Returns:
            True if schema is valid
        """
        try:
            table = pa.Table.from_pandas(df)

            # Check required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing required column: {col}")
                    return False

            # Validate data types
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logger.warning(f"Column {col} is not numeric")
                    return False

            return True

        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            return False

    def load_single_file(self,
                        file_path: Path,
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load a single Parquet file.

        Args:
            file_path: Path to Parquet file
            columns: Specific columns to load

        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_parquet(file_path, columns=columns)

            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')

            if self.validate_schema(df):
                return df
            else:
                logger.warning(f"Schema validation failed for {file_path}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return pd.DataFrame()

    def load_multiple_files(self,
                           pattern: str = "*.parquet",
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load multiple Parquet files in parallel.

        Args:
            pattern: Glob pattern for file selection
            columns: Specific columns to load

        Returns:
            Combined DataFrame
        """
        if self.data_path.is_dir():
            files = list(self.data_path.glob(pattern))
        else:
            files = [self.data_path]

        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return pd.DataFrame()

        logger.info(f"Loading {len(files)} Parquet files...")

        dfs = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(self.load_single_file, f, columns): f
                for f in files
            }

            for future in tqdm(as_completed(futures), total=len(files)):
                df = future.result()
                if not df.empty:
                    dfs.append(df)

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)

            # Remove duplicates based on timestamp and symbol
            if 'symbol' in combined_df.columns and 'timestamp' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(
                    subset=['timestamp', 'symbol'],
                    keep='last'
                )

            return combined_df.sort_values('timestamp')
        else:
            return pd.DataFrame()

    def load_partitioned_dataset(self,
                                 filters: Optional[List[Tuple]] = None) -> pd.DataFrame:
        """
        Load partitioned Parquet dataset.

        Args:
            filters: PyArrow filters for partition pruning

        Returns:
            Filtered DataFrame
        """
        try:
            dataset = pq.ParquetDataset(
                self.data_path,
                filters=filters,
                validate_schema=True
            )

            table = dataset.read()
            df = table.to_pandas()

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')

            return df

        except Exception as e:
            logger.error(f"Error loading partitioned dataset: {str(e)}")
            return pd.DataFrame()

    def stream_batches(self,
                      file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Stream Parquet file in batches for memory efficiency.

        Args:
            file_path: Specific file to stream

        Yields:
            DataFrame batches
        """
        if file_path is None:
            if self.data_path.is_file():
                file_path = self.data_path
            else:
                files = list(self.data_path.glob("*.parquet"))
                if files:
                    file_path = files[0]
                else:
                    logger.error("No Parquet files found")
                    return

        try:
            parquet_file = pq.ParquetFile(file_path)

            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                df = batch.to_pandas()

                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                yield df

        except Exception as e:
            logger.error(f"Error streaming {file_path}: {str(e)}")


class ParquetDataWriter:
    """Efficient Parquet file writer with compression and partitioning."""

    def __init__(self,
                 output_path: Union[str, Path],
                 compression: str = 'snappy',
                 partition_cols: Optional[List[str]] = None):
        """
        Initialize Parquet data writer.

        Args:
            output_path: Output directory for Parquet files
            compression: Compression algorithm (snappy, gzip, brotli, lz4)
            partition_cols: Columns to use for partitioning
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.partition_cols = partition_cols

    def write_single_file(self,
                         df: pd.DataFrame,
                         filename: str,
                         schema: Optional[pa.Schema] = None) -> bool:
        """
        Write DataFrame to single Parquet file.

        Args:
            df: DataFrame to write
            filename: Output filename
            schema: Optional PyArrow schema

        Returns:
            Success status
        """
        try:
            output_file = self.output_path / filename

            if not filename.endswith('.parquet'):
                output_file = self.output_path / f"{filename}.parquet"

            df.to_parquet(
                output_file,
                engine='pyarrow',
                compression=self.compression,
                index=False
            )

            logger.info(f"Wrote {len(df)} rows to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error writing Parquet file: {str(e)}")
            return False

    def write_partitioned_dataset(self,
                                 df: pd.DataFrame,
                                 partition_cols: Optional[List[str]] = None) -> bool:
        """
        Write DataFrame as partitioned dataset.

        Args:
            df: DataFrame to write
            partition_cols: Columns for partitioning

        Returns:
            Success status
        """
        try:
            partition_cols = partition_cols or self.partition_cols

            if not partition_cols:
                logger.warning("No partition columns specified")
                return self.write_single_file(df, "data.parquet")

            table = pa.Table.from_pandas(df)

            pq.write_to_dataset(
                table,
                root_path=self.output_path,
                partition_cols=partition_cols,
                compression=self.compression,
                existing_data_behavior='overwrite_or_ignore'
            )

            logger.info(f"Wrote partitioned dataset with {len(df)} rows")
            return True

        except Exception as e:
            logger.error(f"Error writing partitioned dataset: {str(e)}")
            return False


def optimize_parquet_file(input_path: Path,
                         output_path: Path,
                         row_group_size: int = 50000) -> bool:
    """
    Optimize Parquet file for better query performance.

    Args:
        input_path: Input Parquet file
        output_path: Output path for optimized file
        row_group_size: Target row group size

    Returns:
        Success status
    """
    try:
        # Read the original file
        df = pd.read_parquet(input_path)

        # Sort by timestamp for better compression
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        # Write with optimized settings
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            row_group_size=row_group_size,
            index=False
        )

        # Compare file sizes
        original_size = input_path.stat().st_size / (1024 * 1024)  # MB
        optimized_size = output_path.stat().st_size / (1024 * 1024)  # MB

        logger.info(f"Optimization complete: {original_size:.2f}MB -> {optimized_size:.2f}MB")
        return True

    except Exception as e:
        logger.error(f"Error optimizing Parquet file: {str(e)}")
        return False


def get_parquet_metadata(file_path: Path) -> Dict:
    """
    Get metadata from Parquet file.

    Args:
        file_path: Path to Parquet file

    Returns:
        Dictionary with metadata
    """
    try:
        parquet_file = pq.ParquetFile(file_path)

        metadata = {
            'num_rows': parquet_file.metadata.num_rows,
            'num_columns': parquet_file.metadata.num_columns,
            'num_row_groups': parquet_file.metadata.num_row_groups,
            'format_version': parquet_file.metadata.format_version,
            'created_by': parquet_file.metadata.created_by,
            'schema': parquet_file.schema.to_arrow_schema().names,
            'size_mb': file_path.stat().st_size / (1024 * 1024)
        }

        return metadata

    except Exception as e:
        logger.error(f"Error reading metadata: {str(e)}")
        return {}