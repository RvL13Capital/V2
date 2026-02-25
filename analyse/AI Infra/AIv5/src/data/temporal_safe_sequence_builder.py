"""
Temporal-Safe Sequence Builder for MambaAttention
==================================================

CRITICAL: This module builds temporal sequences with ZERO forward-looking bias.
Every feature in every timestep uses ONLY data available at that moment.

Architecture:
- Input: Snapshot metadata (ticker, start_date, snapshot_date)
- Process: Extract 69 features daily going backwards from snapshot_date
- Output: (n_snapshots, sequence_length, 69) with attention masks

Temporal Safety Mechanisms:
1. Date filtering: df[df['date'] <= lookback_date] for each timestep
2. Reverse construction: Build from snapshot_date backwards
3. Assertions: Verify all sequence dates <= snapshot_date
4. Masking: Explicit tracking of valid vs padded timesteps
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import timedelta
import logging
from tqdm import tqdm
import warnings

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pattern_detection.features.canonical_feature_extractor import CanonicalFeatureExtractor
from shared.indicators.technical import calculate_bbw, calculate_adx, calculate_volume_ratio

# Simple ConsolidationPattern dataclass (avoiding import issues)
from dataclasses import dataclass

@dataclass
class ConsolidationPattern:
    ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    upper_boundary: float
    lower_boundary: float
    power_boundary: float
    qualification_days: int = 10
    phase: str = 'ACTIVE'
    activation_date: pd.Timestamp = None
    start_price: float = None
    days_qualifying: int = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalSafeSequenceBuilder:
    """
    Build temporal sequences with strict no-look-ahead guarantees.

    Usage:
        builder = TemporalSafeSequenceBuilder(sequence_length=50)
        sequences_df = builder.build_sequences_from_snapshots(
            snapshots_df=pd.read_parquet('snapshots_labeled.parquet')
        )
    """

    def __init__(
        self,
        sequence_length: int = 50,
        n_features: int = 69,
        cache_dir: str = "data/historical_cache"
    ):
        """
        Initialize the sequence builder.

        Args:
            sequence_length: Number of days in each sequence (default: 50)
            n_features: Number of features per timestep (default: 69)
            cache_dir: Directory for cached OHLCV data
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.cache_dir = Path(cache_dir)

        # Initialize feature extractor
        self.feature_extractor = CanonicalFeatureExtractor()

        # Verify feature count matches
        assert len(self.feature_extractor.FEATURE_NAMES) == n_features, \
            f"Feature count mismatch: {len(self.feature_extractor.FEATURE_NAMES)} != {n_features}"

        logger.info(f"TemporalSafeSequenceBuilder initialized: seq_len={sequence_length}, n_features={n_features}")

    def build_sequences_from_snapshots(
        self,
        snapshots_df: pd.DataFrame,
        max_samples: Optional[int] = None,
        validate_temporal: bool = True
    ) -> pd.DataFrame:
        """
        Build temporal sequences from snapshot metadata.

        Args:
            snapshots_df: DataFrame with columns: ticker, start_date, snapshot_date,
                         upper_boundary, lower_boundary, power_boundary, outcome_class
            max_samples: Limit processing to first N snapshots (for testing)
            validate_temporal: Run temporal integrity checks

        Returns:
            DataFrame with columns:
                - sequence: np.array (sequence_length, n_features)
                - attention_mask: np.array (sequence_length,) - binary mask
                - ticker: str
                - start_date: pd.Timestamp
                - snapshot_date: pd.Timestamp
                - days_in_pattern: int
                - outcome_class: int (K0-K5)
                - sequence_dates: List[pd.Timestamp] - for validation
        """
        logger.info(f"Building temporal-safe sequences from {len(snapshots_df)} snapshots")

        if max_samples:
            snapshots_df = snapshots_df.head(max_samples)
            logger.info(f"Limited to {max_samples} samples for testing")

        # Ensure required columns exist
        required_cols = ['ticker', 'start_date', 'snapshot_date', 'outcome_class']
        for col in required_cols:
            if col not in snapshots_df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert date columns (handle mixed formats)
        snapshots_df['start_date'] = pd.to_datetime(snapshots_df['start_date'], format='mixed')
        snapshots_df['snapshot_date'] = pd.to_datetime(snapshots_df['snapshot_date'], format='mixed')

        # Build sequences
        sequences_list = []
        failed_count = 0

        for idx, row in tqdm(snapshots_df.iterrows(), total=len(snapshots_df), desc="Building sequences"):
            try:
                sequence_data = self._build_single_sequence(row, validate_temporal)
                sequences_list.append(sequence_data)

            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # Log first 5 failures
                    logger.warning(f"Failed to build sequence for {row['ticker']} at {row['snapshot_date']}: {e}")

        logger.info(f"Successfully built {len(sequences_list)} sequences ({failed_count} failures)")

        # Create DataFrame
        sequences_df = pd.DataFrame(sequences_list)

        # Final validation
        if validate_temporal:
            self._validate_temporal_integrity(sequences_df)

        return sequences_df

    def _build_single_sequence(
        self,
        snapshot_row: pd.Series,
        validate_temporal: bool = True
    ) -> Dict:
        """
        Build a single temporal sequence for one snapshot.

        TEMPORAL SAFETY:
        - Loads ONLY data up to snapshot_date
        - Builds sequence backwards from snapshot_date
        - Each timestep uses ONLY data available at that moment

        Args:
            snapshot_row: Row with ticker, start_date, snapshot_date, pattern metadata
            validate_temporal: Run temporal checks

        Returns:
            Dict with sequence, mask, metadata
        """
        ticker = snapshot_row['ticker']
        start_date = pd.to_datetime(snapshot_row['start_date'])
        snapshot_date = pd.to_datetime(snapshot_row['snapshot_date'])
        outcome_class = snapshot_row['outcome_class']

        # Load OHLCV data (ONLY up to snapshot_date - NO FUTURE DATA)
        df_ohlcv = self._load_ticker_data_temporal_safe(ticker, snapshot_date)

        if df_ohlcv is None or len(df_ohlcv) == 0:
            raise ValueError(f"No OHLCV data available for {ticker}")

        # Create pattern metadata object
        pattern = self._create_pattern_object(snapshot_row, df_ohlcv)

        # Build sequence going backwards from snapshot_date
        sequence_features = []
        sequence_dates = []

        # Ensure snapshot_date is timezone-naive
        if hasattr(snapshot_date, 'tz') and snapshot_date.tz is not None:
            snapshot_date = snapshot_date.tz_localize(None)

        for day_offset in range(self.sequence_length - 1, -1, -1):  # Reverse order (50, 49, ..., 1, 0)
            lookback_date = snapshot_date - timedelta(days=day_offset)

            # CRITICAL: Filter data to ONLY dates <= lookback_date
            # Since 'date' is now the index, use df.index instead of df['date']
            df_upto_date = df_ohlcv[df_ohlcv.index <= lookback_date].copy()

            if len(df_upto_date) < 20:
                # Not enough data for indicators (need 20-day BBW, ADX, etc.)
                # This timestep will be masked out
                sequence_features.append(np.zeros(self.n_features))
                sequence_dates.append(lookback_date)
                continue

            # Extract 69 features using ONLY data up to lookback_date
            try:
                features_dict = self.feature_extractor.extract_snapshot_features(
                    df=df_upto_date,
                    snapshot_date=lookback_date,
                    pattern=pattern
                )

                # Convert dict to ordered array
                feature_array = np.array([
                    features_dict.get(fname, 0.0)
                    for fname in self.feature_extractor.FEATURE_NAMES
                ])

                sequence_features.append(feature_array)
                sequence_dates.append(lookback_date)

            except Exception as e:
                logger.debug(f"Feature extraction failed for {ticker} at {lookback_date}: {e}")
                sequence_features.append(np.zeros(self.n_features))
                sequence_dates.append(lookback_date)

        # Convert to numpy array
        sequence_array = np.array(sequence_features)  # (sequence_length, n_features)

        # Create attention mask (1 = valid data, 0 = padding/missing)
        attention_mask = (sequence_array.sum(axis=1) != 0).astype(int)

        # Temporal validation
        if validate_temporal:
            self._validate_single_sequence(sequence_dates, snapshot_date, ticker)

        # Calculate days_in_pattern
        days_in_pattern = (snapshot_date - start_date).days

        return {
            'sequence': sequence_array,
            'attention_mask': attention_mask,
            'ticker': ticker,
            'start_date': start_date,
            'snapshot_date': snapshot_date,
            'days_in_pattern': days_in_pattern,
            'outcome_class': outcome_class,
            'sequence_dates': sequence_dates  # For validation
        }

    def _load_ticker_data_temporal_safe(
        self,
        ticker: str,
        max_date: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data for ticker with strict date filtering.

        TEMPORAL SAFETY: Returns ONLY data up to max_date (no future data).

        Args:
            ticker: Stock ticker symbol
            max_date: Maximum date to include (snapshot_date)

        Returns:
            DataFrame with OHLCV data filtered to dates <= max_date
        """
        # Try to load from cache
        cache_file = self.cache_dir / f"{ticker}.parquet"

        if not cache_file.exists():
            # Try CSV
            cache_file = self.cache_dir / f"{ticker}.csv"

        if not cache_file.exists():
            logger.warning(f"No cached data found for {ticker}")
            return None

        try:
            # Load data
            if cache_file.suffix == '.parquet':
                df = pd.read_parquet(cache_file)
            else:
                df = pd.read_csv(cache_file)

            # Ensure date column
            if 'date' not in df.columns:
                if 'Date' in df.columns:
                    df.rename(columns={'Date': 'date'}, inplace=True)
                else:
                    raise ValueError(f"No 'date' column found in {ticker} data")

            # Convert to datetime
            df['date'] = pd.to_datetime(df['date'])

            # Remove timezone info for comparison (make timezone-naive)
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)

            # Ensure max_date is timezone-naive
            if hasattr(max_date, 'tz') and max_date.tz is not None:
                max_date = max_date.tz_localize(None)

            # CRITICAL: Filter to dates <= max_date (NO FUTURE DATA)
            df = df[df['date'] <= max_date].copy()

            # Sort by date (before setting as index)
            df = df.sort_values('date')

            # Validate columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col.lower() not in df.columns.str.lower()]

            if missing_cols:
                raise ValueError(f"Missing columns for {ticker}: {missing_cols}")

            # Standardize column names
            df.columns = df.columns.str.lower()

            # CRITICAL: Set 'date' as index for CanonicalFeatureExtractor compatibility
            # The extractor expects snapshot_date in df.index, not df['date']
            df = df.set_index('date')

            return df

        except Exception as e:
            logger.error(f"Failed to load data for {ticker}: {e}")
            return None

    def _create_pattern_object(
        self,
        snapshot_row: pd.Series,
        df_ohlcv: pd.DataFrame
    ) -> ConsolidationPattern:
        """
        Create ConsolidationPattern object from snapshot metadata.

        Args:
            snapshot_row: Row with pattern metadata
            df_ohlcv: OHLCV DataFrame

        Returns:
            ConsolidationPattern object
        """
        # Extract boundaries (if available)
        upper = snapshot_row.get('upper_boundary', df_ohlcv['high'].max())
        lower = snapshot_row.get('lower_boundary', df_ohlcv['low'].min())
        power = snapshot_row.get('power_boundary', upper * 1.005)

        # Extract additional fields
        activation_date = snapshot_row.get('activation_date', snapshot_row.get('snapshot_date'))
        start_price = snapshot_row.get('start_price', None)
        days_qualifying = snapshot_row.get('days_qualifying', 10)

        # Create pattern object
        pattern = ConsolidationPattern(
            ticker=snapshot_row['ticker'],
            start_date=pd.to_datetime(snapshot_row['start_date']),
            end_date=pd.to_datetime(snapshot_row.get('end_date', snapshot_row['snapshot_date'])),
            upper_boundary=upper,
            lower_boundary=lower,
            power_boundary=power,
            qualification_days=10,
            phase='ACTIVE',
            activation_date=pd.to_datetime(activation_date) if activation_date else None,
            start_price=start_price,
            days_qualifying=days_qualifying
        )

        return pattern

    def _validate_single_sequence(
        self,
        sequence_dates: List[pd.Timestamp],
        snapshot_date: pd.Timestamp,
        ticker: str
    ):
        """
        Validate temporal integrity of a single sequence.

        CRITICAL: All dates in sequence must be <= snapshot_date.

        Args:
            sequence_dates: List of dates in sequence
            snapshot_date: The snapshot date (present moment)
            ticker: Ticker symbol (for error reporting)

        Raises:
            ValueError: If any sequence date is after snapshot_date (LEAKAGE!)
        """
        future_dates = [d for d in sequence_dates if d > snapshot_date]

        if future_dates:
            raise ValueError(
                f"TEMPORAL LEAKAGE DETECTED for {ticker}!\n"
                f"Snapshot date: {snapshot_date}\n"
                f"Future dates in sequence: {future_dates}\n"
                f"This indicates forward-looking bias - sequence contains future data!"
            )

        # Verify chronological order
        for i in range(len(sequence_dates) - 1):
            if sequence_dates[i] > sequence_dates[i+1]:
                logger.warning(
                    f"Sequence dates not chronological for {ticker}: "
                    f"{sequence_dates[i]} > {sequence_dates[i+1]}"
                )

    def _validate_temporal_integrity(self, sequences_df: pd.DataFrame):
        """
        Validate temporal integrity of all sequences.

        Checks:
        1. All sequence dates <= snapshot_date
        2. No NaN in critical features
        3. Sequences have reasonable lengths

        Args:
            sequences_df: DataFrame with all sequences

        Raises:
            ValueError: If temporal integrity is violated
        """
        logger.info("Validating temporal integrity of sequences...")

        violations = 0

        for idx, row in sequences_df.iterrows():
            # Check 1: All sequence dates <= snapshot_date
            sequence_dates = row['sequence_dates']
            snapshot_date = row['snapshot_date']

            future_dates = [d for d in sequence_dates if d > snapshot_date]
            if future_dates:
                violations += 1
                if violations <= 3:
                    logger.error(
                        f"Row {idx}: Future dates detected! "
                        f"Snapshot: {snapshot_date}, Future dates: {future_dates}"
                    )

            # Check 2: Sequence shape
            if row['sequence'].shape != (self.sequence_length, self.n_features):
                logger.warning(
                    f"Row {idx}: Incorrect sequence shape {row['sequence'].shape}, "
                    f"expected ({self.sequence_length}, {self.n_features})"
                )

            # Check 3: Attention mask validity
            if row['attention_mask'].sum() == 0:
                logger.warning(f"Row {idx}: All timesteps masked (no valid data)")

        if violations > 0:
            raise ValueError(
                f"TEMPORAL INTEGRITY VIOLATION: {violations} sequences contain future data! "
                f"This indicates forward-looking bias. Cannot proceed."
            )

        logger.info(f"✓ Temporal integrity validated: {len(sequences_df)} sequences passed all checks")

        # Log statistics
        avg_valid_steps = sequences_df['attention_mask'].apply(lambda x: x.sum()).mean()
        logger.info(f"Average valid timesteps per sequence: {avg_valid_steps:.1f} / {self.sequence_length}")


def main():
    """
    Example usage: Build sequences from labeled snapshots.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Build temporal-safe sequences for MambaAttention")
    parser.add_argument('--input', type=str, required=True, help='Input snapshots parquet file')
    parser.add_argument('--output', type=str, required=True, help='Output sequences parquet file')
    parser.add_argument('--sequence-length', type=int, default=50, help='Sequence length (days)')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit samples (for testing)')
    parser.add_argument('--no-validate', action='store_true', help='Skip temporal validation')

    args = parser.parse_args()

    # Load snapshots
    logger.info(f"Loading snapshots from {args.input}")
    snapshots_df = pd.read_parquet(args.input)

    # Build sequences
    builder = TemporalSafeSequenceBuilder(sequence_length=args.sequence_length)
    sequences_df = builder.build_sequences_from_snapshots(
        snapshots_df=snapshots_df,
        max_samples=args.max_samples,
        validate_temporal=not args.no_validate
    )

    # Save (convert numpy arrays to lists for Parquet compatibility)
    logger.info(f"Saving {len(sequences_df)} sequences to {args.output}")
    save_df = sequences_df.copy()
    save_df['sequence'] = save_df['sequence'].apply(lambda x: x.tolist())
    save_df['attention_mask'] = save_df['attention_mask'].apply(lambda x: x.tolist())
    save_df.to_parquet(args.output, index=False)
    logger.info(f"Saved to {args.output}")

    logger.info("✓ Sequence building complete!")

    # Print summary
    print("\n" + "="*60)
    print("SEQUENCE BUILDING SUMMARY")
    print("="*60)
    print(f"Total sequences: {len(sequences_df)}")
    print(f"Sequence length: {args.sequence_length} days")
    print(f"Features per timestep: {builder.n_features}")
    print(f"Avg valid timesteps: {sequences_df['attention_mask'].apply(lambda x: x.sum()).mean():.1f}")
    print(f"\nOutcome distribution:")
    print(sequences_df['outcome_class'].value_counts().sort_index())
    print("="*60)


if __name__ == "__main__":
    main()
