"""
Pattern Detector for Temporal Architecture
==========================================

Generates temporal sequences from consolidation patterns.

Key Features:
- Sliding windows (20 timesteps) for temporal model input
- Each pattern produces N - 20 + 1 sequences
- Temporal integrity maintained (no look-ahead bias)
- Vectorized feature extraction for performance
- Integration with centralized configuration

Usage:
    detector = TemporalPatternDetector()
    sequences, labels = detector.process_ticker(df, ticker)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Union
from datetime import datetime, timedelta, date
import logging

# Import ConsolidationTracker directly
from .consolidation_tracker import ConsolidationTracker

# Import updated constants (CRITICAL: Use 5.0R target + 5% stop buffer)
from config.constants import (
    RISK_MULTIPLIER_TARGET,
    RISK_MULTIPLIER_GREY,
    STOP_BUFFER_PERCENT,
    INDICATOR_WARMUP_DAYS  # ADX/BBW need warmup before values are valid
)

# Import other components
from .aiv7_components import (
    ConsolidationPattern,
    PatternPhase
)

# Import configurations and utilities
from config import (
    TemporalFeatureConfig,
    SequenceGenerationConfig,
    PatternPhase as ConfigPatternPhase,
    TEMPORAL_WINDOW_SIZE,
    MIN_PATTERN_DURATION
)
from features import VectorizedFeatureCalculator
from .exceptions import (
    SequenceGenerationError,
    TemporalConsistencyError,
    ValidationError,
    ConfigError
)

logger = logging.getLogger(__name__)


class TemporalPatternDetector:
    """
    Detects consolidation patterns and generates temporal sequences.

    Uses ConsolidationTracker for pattern detection and generates
    sliding window sequences for temporal model training with proper
    temporal integrity and vectorized feature extraction.
    """

    def __init__(
        self,
        feature_config: Optional[TemporalFeatureConfig] = None,
        sequence_config: Optional[SequenceGenerationConfig] = None,
        random_seed: Optional[int] = None,
        mode: 'ProcessingMode' = None,  # Will be imported below
        robust_scaling_path: Optional[str] = None,  # Path to robust_scaling_params.json
        skip_composite_normalization: bool = False  # If True, save RAW composite values
    ):
        """
        Initialize with configuration objects.

        Args:
            feature_config: Feature configuration (uses default if None)
            sequence_config: Sequence generation config (uses default if None)
            random_seed: Random seed for reproducibility
            mode: ProcessingMode.TRAINING (strict: only labeled terminal patterns) or
                  ProcessingMode.INFERENCE (permissive: includes active/qualifying patterns)
                  Defaults to TRAINING for backward compatibility
            robust_scaling_path: Path to JSON file with global Median/IQR for composite features.
                                 If provided, uses global robust scaling instead of per-window whitening.
            skip_composite_normalization: If True, skip ALL normalization of composite features
                                         (indices 8-11). Use during sequence generation to save
                                         RAW values, then apply global robust scaling during training.
                                         This fixes the chicken-and-egg problem where robust_params
                                         can't be computed before sequences exist.
        """
        from config.constants import ProcessingMode
        import json

        self.mode = mode if mode is not None else ProcessingMode.TRAINING
        self.feature_config = feature_config or TemporalFeatureConfig()
        self.sequence_config = sequence_config or SequenceGenerationConfig()

        # Load robust scaling parameters for global normalization (if provided)
        self.robust_params = None
        if robust_scaling_path:
            try:
                with open(robust_scaling_path, 'r') as f:
                    self.robust_params = json.load(f)
                logger.info(f"Loaded global robust scaling params from {robust_scaling_path}")
            except Exception as e:
                logger.warning(f"Could not load robust scaling params: {e}. Using per-window whitening.")

        # Skip composite normalization flag (for sequence generation)
        self.skip_composite_normalization = skip_composite_normalization
        if skip_composite_normalization:
            logger.info("Composite normalization DISABLED - saving RAW values for later global scaling")

        # Validate feature configuration
        # UPDATED (2026-01-18): 14 → 10 features (composite features disabled)
        if self.feature_config.feature_count != 10:
            raise ConfigError(f"Expected 10 features, got {self.feature_config.feature_count}")

        # Initialize components
        # Legacy ConsolidationTracker requires ticker but we'll set it later per ticker
        self.tracker = ConsolidationTracker(ticker="")
        self.feature_calculator = VectorizedFeatureCalculator()

        # Initialize v17 labeler for path-dependent classification
        from core.path_dependent_labeler import PathDependentLabelerV17

        # CRITICAL FIX: Use updated constants (5.0R target, 5% stop buffer)
        # Disable dynamic profiles to ensure static constants are used
        self.labeler_v17 = PathDependentLabelerV17(
            indicator_warmup=30,
            indicator_stable=100,
            outcome_window=100,
            risk_multiplier_target=RISK_MULTIPLIER_TARGET,  # 5.0R (filters noise winners)
            risk_multiplier_grey=RISK_MULTIPLIER_GREY,      # 2.5R
            stop_buffer=STOP_BUFFER_PERCENT,                 # 5% (prevents false danger labels)
            use_dynamic_profiles=False  # Use static constants instead of market cap profiles
        )

        # MEMORY FIX: Cache DataLoader to avoid creating 23k+ GCS clients
        self._cached_data_loader = None

        logger.info(f"TemporalPatternDetector initialized in {self.mode.value.upper()} mode with window_size={self.sequence_config.window_size}")

    def process_ticker(
        self,
        df: pd.DataFrame,
        ticker: str,
        generate_sequences: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process a single ticker and generate temporal sequences.

        Args:
            df: DataFrame with OHLCV data and indicators
            ticker: Ticker symbol
            generate_sequences: If True, generate sliding windows

        Returns:
            sequences: Array of shape (n_windows, window_size, n_features)
            labels: Array of pattern labels (K0-K5) for each sequence

        Raises:
            TemporalConsistencyError: If temporal integrity is violated
            ValidationError: If data validation fails
        """
        # Validate temporal integrity
        self._validate_temporal_integrity(df)

        # Reset tracker for new ticker
        self.tracker = ConsolidationTracker(ticker=ticker)

        # Store full DataFrame for v17 labeler access
        self.full_data_df = df
        self.current_ticker = ticker

        detected_patterns = []

        # Track pattern through all data points
        for idx, row in df.iterrows():
            try:
                pattern, phase = self.tracker.track_point(
                    date=row.get('date', idx),
                    close=row['close'],
                    high=row['high'],
                    low=row['low'],
                    volume=row['volume'],
                    bbw=row.get('bbw', 0),
                    adx=row.get('adx', 0),
                    current_idx=idx
                )

                # Pattern collection: mode-aware filtering
                # TRAINING: Only terminal patterns (have definitive outcomes)
                # INFERENCE: Include active/qualifying patterns (for live prediction)
                if pattern is not None:
                    from config.constants import ProcessingMode
                    if self.mode == ProcessingMode.TRAINING:
                        # TRAINING: Only terminal patterns
                        if pattern.is_terminal():
                            detected_patterns.append(pattern)
                    else:
                        # INFERENCE: Include active/qualifying patterns
                        if pattern.phase in [PatternPhase.ACTIVE, PatternPhase.QUALIFYING] or pattern.is_terminal():
                            detected_patterns.append(pattern)

            except Exception as e:
                logger.warning(f"Error tracking pattern at index {idx}: {e}")
                continue

        if not generate_sequences or len(detected_patterns) == 0:
            logger.info(f"{ticker}: No patterns detected")
            return None, None

        # Generate temporal sequences from patterns
        all_sequences = []
        all_labels = []

        for pattern in detected_patterns:
            # FLAW FIX #11: Don't skip unripe patterns in INFERENCE mode
            # In TRAINING: only process terminal patterns
            # In INFERENCE: process ALL patterns (including ACTIVE/QUALIFYING)
            from config.constants import ProcessingMode, LABEL_GREY_ZONE, LABEL_PLACEHOLDER_UNRIPE
            if self.mode == ProcessingMode.TRAINING and not pattern.is_terminal():
                continue  # Skip non-terminal patterns in TRAINING mode

            try:
                # Extract data for this pattern WITH WARMUP PREFIX
                # CRITICAL FIX: Include INDICATOR_WARMUP_DAYS before pattern start so
                # ADX/BBW have valid historical context. First ~14 days of ADX are
                # meaningless without warmup data.
                warmup_start_idx = max(0, pattern.start_idx - INDICATOR_WARMUP_DAYS)
                warmup_offset = pattern.start_idx - warmup_start_idx  # How many warmup rows
                pattern_data = df.iloc[warmup_start_idx:pattern.end_idx + 1]

                # Pattern portion (excluding warmup)
                pattern_length = pattern.end_idx - pattern.start_idx + 1
                if pattern_length >= self.sequence_config.min_pattern_duration:
                    sequences = self._generate_sliding_windows(pattern_data, pattern, warmup_offset)

                    if sequences is not None and len(sequences) > 0:
                        # Get label (may be -1 for grey zone, -2 for unripe, 0-2 for terminal)
                        label = self._get_pattern_label(pattern)

                        # FLAW FIX #13: Filter grey zones AND placeholders in TRAINING mode (defensive)
                        if self.mode == ProcessingMode.TRAINING:
                            if label == LABEL_GREY_ZONE:
                                logger.debug(f"Skipping grey zone pattern in TRAINING mode")
                                continue
                            if label == LABEL_PLACEHOLDER_UNRIPE:
                                logger.warning(f"Placeholder label in TRAINING mode - skipping")
                                continue

                        # FLAW FIX #12: Filter grey zones in BOTH modes
                        # Grey zones are ambiguous - not useful even for inference
                        if label == LABEL_GREY_ZONE:
                            logger.debug(f"Skipping grey zone pattern (ambiguous)")
                            continue

                        all_sequences.append(sequences)
                        # All sequences from same pattern get same label
                        labels = np.full(len(sequences), label)
                        all_labels.append(labels)

            except Exception as e:
                logger.error(f"Error generating sequences for pattern: {e}")
                continue

        if len(all_sequences) == 0:
            logger.info(f"{ticker}: No valid sequences generated")
            return None, None

        # Concatenate all sequences
        sequences = np.concatenate(all_sequences, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        logger.info(f"{ticker}: Generated {len(sequences)} sequences from {len(detected_patterns)} patterns")

        return sequences, labels

    def process_ticker_with_dates(
        self,
        df: pd.DataFrame,
        ticker: str,
        generate_sequences: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """
        Process a single ticker and generate temporal sequences with date information.

        This method extends process_ticker to also return temporal metadata for each
        sequence, which is critical for preventing look-ahead bias in train/test splits.

        Args:
            df: DataFrame with OHLCV data and indicators
            ticker: Ticker symbol
            generate_sequences: If True, generate sliding windows

        Returns:
            sequences: Array of shape (n_windows, window_size, n_features)
            labels: Array of pattern labels (K0-K5) for each sequence
            pattern_dates: Dictionary with date information for each sequence:
                - 'start_dates': Pattern start date for each sequence
                - 'end_dates': Pattern end date for each sequence
                - 'sequence_end_dates': Specific end date of each sequence window

        Raises:
            TemporalConsistencyError: If temporal integrity is violated
            ValidationError: If data validation fails
        """
        # Validate temporal integrity
        self._validate_temporal_integrity(df)

        # Reset tracker for new ticker
        self.tracker = ConsolidationTracker(ticker=ticker)

        detected_patterns = []

        # Track pattern through all data points
        for idx, row in df.iterrows():
            try:
                pattern, phase = self.tracker.track_point(
                    date=row.get('date', idx),
                    close=row['close'],
                    high=row['high'],
                    low=row['low'],
                    volume=row['volume'],
                    bbw=row.get('bbw', 0),
                    adx=row.get('adx', 0),
                    current_idx=idx
                )

                # Pattern collection: mode-aware filtering
                # TRAINING: Only terminal patterns (have definitive outcomes)
                # INFERENCE: Include active/qualifying patterns (for live prediction)
                if pattern is not None:
                    from config.constants import ProcessingMode
                    if self.mode == ProcessingMode.TRAINING:
                        # TRAINING: Only terminal patterns
                        if pattern.is_terminal():
                            detected_patterns.append(pattern)
                    else:
                        # INFERENCE: Include active/qualifying patterns
                        if pattern.phase in [PatternPhase.ACTIVE, PatternPhase.QUALIFYING] or pattern.is_terminal():
                            detected_patterns.append(pattern)

            except Exception as e:
                logger.warning(f"Error tracking pattern at index {idx}: {e}")
                continue

        if not generate_sequences or len(detected_patterns) == 0:
            logger.info(f"{ticker}: No patterns detected")
            return None, None, None

        # Generate temporal sequences from patterns
        all_sequences = []
        all_labels = []
        all_start_dates = []
        all_end_dates = []
        all_sequence_end_dates = []

        for pattern in detected_patterns:
            if pattern.is_terminal():
                try:
                    # Extract data for this pattern WITH WARMUP PREFIX
                    # CRITICAL FIX: Include INDICATOR_WARMUP_DAYS before pattern start
                    warmup_start_idx = max(0, pattern.start_idx - INDICATOR_WARMUP_DAYS)
                    warmup_offset = pattern.start_idx - warmup_start_idx
                    pattern_data = df.iloc[warmup_start_idx:pattern.end_idx + 1]

                    # Pattern portion (excluding warmup)
                    pattern_length = pattern.end_idx - pattern.start_idx + 1
                    if pattern_length >= self.sequence_config.min_pattern_duration:
                        sequences = self._generate_sliding_windows(pattern_data, pattern, warmup_offset)

                        if sequences is not None and len(sequences) > 0:
                            all_sequences.append(sequences)
                            # All sequences from same pattern get same label
                            labels = np.full(len(sequences), self._get_pattern_label(pattern))
                            all_labels.append(labels)

                            # Generate date information for each sequence
                            n_sequences = len(sequences)

                            # Pattern dates (same for all sequences from this pattern)
                            start_dates = [pattern.start_date] * n_sequences
                            end_dates = [pattern.end_date] * n_sequences

                            # Calculate specific end date for each sequence window
                            # Each sequence is offset by stride from the previous
                            sequence_end_dates = []
                            for i in range(n_sequences):
                                # Calculate the index of the last element in this sequence window
                                seq_end_idx = pattern.start_idx + i * self.sequence_config.stride + self.sequence_config.window_size - 1
                                # Make sure we don't exceed pattern bounds
                                seq_end_idx = min(seq_end_idx, pattern.end_idx)
                                # Get the date for this index
                                if isinstance(df.index, pd.DatetimeIndex):
                                    seq_end_date = df.index[seq_end_idx]
                                else:
                                    # If we have a date column, use it
                                    seq_end_date = df.iloc[seq_end_idx].get('date', seq_end_idx)
                                sequence_end_dates.append(seq_end_date)

                            all_start_dates.extend(start_dates)
                            all_end_dates.extend(end_dates)
                            all_sequence_end_dates.extend(sequence_end_dates)

                except Exception as e:
                    logger.error(f"Error generating sequences for pattern: {e}")
                    continue

        if len(all_sequences) == 0:
            logger.info(f"{ticker}: No valid sequences generated")
            return None, None, None

        # Concatenate all sequences
        sequences = np.concatenate(all_sequences, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Create pattern dates dictionary
        pattern_dates = {
            'start_dates': all_start_dates,
            'end_dates': all_end_dates,
            'sequence_end_dates': all_sequence_end_dates
        }

        logger.info(f"{ticker}: Generated {len(sequences)} sequences from {len(detected_patterns)} patterns")

        return sequences, labels, pattern_dates

    def generate_sequences_for_pattern(
        self,
        ticker: str,
        pattern_start: Union[date, datetime],
        pattern_end: Union[date, datetime],
        df: Optional[pd.DataFrame] = None
    ) -> Optional[np.ndarray]:
        """
        Generate temporal sequences for a specific pattern date range.

        This method is used by the API to generate sequences for prediction
        on individual patterns without reprocessing the entire ticker history.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
            pattern_start: Pattern start date (date or datetime object)
            pattern_end: Pattern end date (date or datetime object)
            df: Optional pre-loaded DataFrame (for testing or when data already available)

        Returns:
            Array of sequences with shape (n_sequences, window_size, n_features)
            where:
            - n_sequences: Number of sliding windows extracted
            - window_size: 20 timesteps (configurable)
            - n_features: 14 temporal features

            Returns None if:
            - Data cannot be loaded
            - Pattern too short (< 20 days)
            - Feature calculation fails

        Raises:
            ValidationError: If inputs are invalid or data is malformed
            TemporalConsistencyError: If data has temporal gaps
            DataIntegrityError: If required columns are missing

        Example:
            >>> from datetime import date
            >>> detector = TemporalPatternDetector()
            >>> sequences = detector.generate_sequences_for_pattern(
            ...     ticker="AAPL",
            ...     pattern_start=date(2023, 1, 15),
            ...     pattern_end=date(2023, 2, 15)
            ... )
            >>> if sequences is not None:
            ...     print(f"Generated {len(sequences)} sequences")
            ...     print(f"Shape: {sequences.shape}")  # (11, 20, 14)

        Notes:
            - Loads 130 extra days before pattern_start for indicator calculation (CRITICAL for feature stability)
            - Uses 95th/5th percentile for boundaries (wick-proof)
            - Returns sequences from sliding window (stride=1 by default)
            - Each sequence is 20 consecutive timesteps
            - All 14 temporal features are calculated automatically
            - Buffer period MUST match training pipeline to prevent feature distribution mismatch

        Performance:
            - Typical: 100-500ms for 30-day pattern
            - Depends on: data availability, pattern length, feature calculation
        """
        try:
            # Convert date to datetime if needed (for timedelta operations)
            if isinstance(pattern_start, date) and not isinstance(pattern_start, datetime):
                pattern_start = datetime.combine(pattern_start, datetime.min.time())
            if isinstance(pattern_end, date) and not isinstance(pattern_end, datetime):
                pattern_end = datetime.combine(pattern_end, datetime.max.time())

            logger.info(f"Generating sequences for {ticker}: {pattern_start.date()} to {pattern_end.date()}")

            # Validate inputs
            if not ticker or not isinstance(ticker, str):
                raise ValidationError(f"Invalid ticker: {ticker}")

            if pattern_start >= pattern_end:
                raise ValidationError(
                    f"Invalid date range: start ({pattern_start.date()}) >= end ({pattern_end.date()})"
                )

            # Check reasonable date range (use window_size - 1 because date difference is exclusive)
            # e.g., 20 days of data spans from day 0 to day 19, so difference = 19
            date_range = (pattern_end - pattern_start).days
            if date_range < (self.sequence_config.window_size - 1):
                logger.warning(f"Pattern too short: {date_range} days < minimum {self.sequence_config.window_size - 1}")
                return None  # Return None as documented

            if date_range > 365:
                logger.warning(f"Long pattern detected: {date_range} days")

            if pattern_end > datetime.now():
                raise ValidationError(f"Future date not allowed: {pattern_end.date()}")

            # Load ticker data (or use provided DataFrame)
            if df is None:
                from .aiv7_components.data_loader import DataLoader
                from config.constants import INDICATOR_STABLE_DAYS  # INDICATOR_WARMUP_DAYS already imported at module level

                # MEMORY FIX: Reuse cached DataLoader to avoid creating 23k+ GCS clients
                if self._cached_data_loader is None:
                    self._cached_data_loader = DataLoader()
                loader = self._cached_data_loader

                # CRITICAL: Load with sufficient buffer for indicator calculation
                # MUST match training pipeline warm-up requirements to prevent feature distribution mismatch
                # Training uses 100+ days lookback → inference MUST use the same
                buffer_days = INDICATOR_WARMUP_DAYS + INDICATOR_STABLE_DAYS  # 130 days (30 + 100)
                start_with_buffer = pattern_start - timedelta(days=buffer_days)
                logger.debug(f"Loading data with {buffer_days}-day buffer: {start_with_buffer.date()} to {pattern_end.date()}")

                # For short patterns, skip the 50-day minimum validation
                # (patterns are typically 28-42 days, buffer adds ~130 days = ~140-160 trading days)
                df = loader.load_ticker(
                    ticker=ticker,
                    start_date=start_with_buffer,
                    end_date=pattern_end,
                    validate=False  # Skip MIN_DATA_LENGTH check for short patterns
                )

                # Validate loaded data
                if df is None or len(df) == 0:
                    logger.warning(f"No data loaded for {ticker}")
                    return None  # Return None as documented, don't raise exception
            else:
                # DataFrame provided - use it directly
                logger.debug(f"Using provided DataFrame with {len(df)} rows")

            logger.debug(f"Loaded {len(df)} rows for {ticker}")

            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")

            # Ensure date index
            if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                raise ValidationError("DataFrame must have DatetimeIndex or 'date' column")

            # FIX: Normalize timezone - ensure both index and pattern dates are tz-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # Also ensure pattern dates are tz-naive
            pattern_start = pd.Timestamp(pattern_start)
            pattern_end = pd.Timestamp(pattern_end)
            if pattern_start.tzinfo is not None:
                pattern_start = pattern_start.tz_convert('UTC').tz_localize(None)
            if pattern_end.tzinfo is not None:
                pattern_end = pattern_end.tz_convert('UTC').tz_localize(None)

            # Filter to pattern date range WITH WARMUP PREFIX
            # CRITICAL FIX: Include INDICATOR_WARMUP_DAYS before pattern start
            warmup_start = pattern_start - pd.Timedelta(days=int(INDICATOR_WARMUP_DAYS * 1.5))  # Extra buffer for weekends
            pattern_data_with_warmup = df.loc[warmup_start:pattern_end]

            # Calculate actual warmup offset (rows before pattern_start)
            pattern_mask = pattern_data_with_warmup.index >= pattern_start
            warmup_offset = (~pattern_mask).sum()
            pattern_data = df.loc[pattern_start:pattern_end]  # Pure pattern data for boundary calc
            logger.debug(f"Pattern data: {len(pattern_data)} rows, warmup: {warmup_offset} rows")

            if len(pattern_data) < self.sequence_config.window_size:
                logger.warning(f"Pattern data too short: {len(pattern_data)} < {self.sequence_config.window_size}")
                return None  # Return None as documented

            # Create a dummy pattern object with boundaries from the data
            # Calculate boundaries as 95th/5th percentiles (same as scanner)
            # IMPORTANT: Calculate boundaries from pattern data only, not warmup
            upper_boundary = pattern_data['high'].quantile(0.95)
            lower_boundary = pattern_data['low'].quantile(0.05)
            logger.debug(f"Boundaries: upper={upper_boundary:.2f}, lower={lower_boundary:.2f}")

            # Create minimal pattern object for feature calculation
            from .aiv7_components import ConsolidationPattern, PatternPhase
            start_price = pattern_data.iloc[0]['close'] if len(pattern_data) > 0 else 0.0
            dummy_pattern = ConsolidationPattern(
                ticker=ticker,
                start_date=pattern_start,
                end_date=pattern_end,
                start_idx=warmup_offset,  # Offset accounts for warmup
                end_idx=warmup_offset + len(pattern_data) - 1,
                start_price=start_price,
                upper_boundary=upper_boundary,
                lower_boundary=lower_boundary,
                phase=PatternPhase.COMPLETED
            )

            # Generate sequences using existing method
            # CRITICAL: Use pattern_data_with_warmup so features have full history
            logger.debug(f"Generating sliding windows for {len(pattern_data_with_warmup)} timesteps (warmup={warmup_offset})")
            sequences = self._generate_sliding_windows(pattern_data_with_warmup, dummy_pattern, warmup_offset)

            if sequences is not None:
                logger.info(f"✓ Generated {len(sequences)} sequences (shape: {sequences.shape}) for {ticker}")
            else:
                logger.warning(f"✗ Failed to generate sequences (returned None) for {ticker}")

            return sequences

        except ValidationError:
            # Re-raise validation errors for proper handling upstream
            raise
        except Exception as e:
            logger.error(f"✗ Failed to generate sequences for {ticker}: {e}", exc_info=True)
            return None

    def _validate_temporal_integrity(self, df: pd.DataFrame) -> None:
        """
        Validate temporal integrity of the data.

        Args:
            df: DataFrame to validate

        Raises:
            TemporalConsistencyError: If temporal issues detected
        """
        # Check if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            # Check for duplicates
            if df.index.duplicated().any():
                raise TemporalConsistencyError("Duplicate timestamps detected")

            # Check chronological order
            if not df.index.is_monotonic_increasing:
                raise TemporalConsistencyError("Data not in chronological order")

            # Check for large gaps
            if self.sequence_config.max_temporal_gap_days:
                gaps = df.index.to_series().diff()
                max_gap = gaps.max()
                if max_gap > pd.Timedelta(days=self.sequence_config.max_temporal_gap_days):
                    raise TemporalConsistencyError(f"Large temporal gap detected: {max_gap}")

        # Check data length
        if len(df) < self.sequence_config.min_pattern_duration:
            raise ValidationError(f"Insufficient data: {len(df)} < {self.sequence_config.min_pattern_duration}")

    def _generate_sliding_windows(
        self,
        pattern_data: pd.DataFrame,
        pattern: ConsolidationPattern,
        warmup_offset: int = 0
    ) -> Optional[np.ndarray]:
        """
        Generate sliding windows from pattern data using vectorized operations.

        Protocol:
        - Each window contains window_size consecutive timesteps
        - Stride defined by sequence_config (default 1)
        - Temporal integrity maintained
        - Windows start AFTER warmup_offset to ensure valid indicator values

        Args:
            pattern_data: DataFrame with pattern data (may include warmup prefix)
            pattern: ConsolidationPattern object
            warmup_offset: Number of rows at start used for indicator warmup (skip these)

        Returns:
            Array of shape (n_windows, window_size, n_features)
        """
        n_timesteps = len(pattern_data)
        valid_timesteps = n_timesteps - warmup_offset  # Timesteps after warmup

        # Validate minimum length (must have enough data AFTER warmup)
        if valid_timesteps < self.sequence_config.window_size:
            logger.debug(f"Pattern too short after warmup: {valid_timesteps} < {self.sequence_config.window_size}")
            return None

        # Calculate features using vectorized calculator
        # CRITICAL: Calculate on FULL data (including warmup) so indicators have history
        pattern_data = self._prepare_pattern_features(pattern_data, pattern)

        # Validate all required features present
        missing = self.feature_config.validate_dataframe_columns(pattern_data)
        if missing:
            logger.warning(f"Missing features: {missing}")
            return None

        # Extract feature columns in correct order
        feature_data = pattern_data[self.feature_config.all_features].values

        # Calculate number of windows with stride
        # CRITICAL: Start windows AFTER warmup_offset, not at index 0
        n_windows = (valid_timesteps - self.sequence_config.window_size) // self.sequence_config.stride + 1

        if n_windows <= 0:
            logger.debug(f"No valid windows after warmup offset {warmup_offset}")
            return None

        # Vectorized window extraction - START AFTER WARMUP
        windows = np.array([
            feature_data[warmup_offset + i:warmup_offset + i + self.sequence_config.window_size]
            for i in range(0, n_windows * self.sequence_config.stride, self.sequence_config.stride)
        ])

        # CRITICAL FIX: Relativize OHLC features to make model price-agnostic
        # Formula: (Price_t / Price_0) - 1
        # This ensures a $2 stock and $200 stock with identical patterns produce identical sequences
        windows = self._relativize_price_features(windows)

        # WINDOW-ANCHOR FIX: Normalize volume using 6-month median (not day-0)
        # This ensures LSTM sees "Wake Up" magnitude relative to dormant history,
        # not just relative to arbitrary pattern start volume.
        # Formula: log(volume_t / vol_6m_median)
        vol_6m_ref = None
        if '_vol_6m_median' in pattern_data.columns:
            # Extract 6-month median at the START of each window
            vol_6m_values = pattern_data['_vol_6m_median'].values
            vol_6m_ref = np.array([
                vol_6m_values[warmup_offset + i]
                for i in range(0, n_windows * self.sequence_config.stride, self.sequence_config.stride)
            ])
            logger.debug(f"Using 6-month median for volume normalization (median of medians: {np.median(vol_6m_ref):.0f})")

        windows = self._normalize_volume(windows, volume_ma_50=vol_6m_ref)

        # Normalize composite scores (indices 8-11)
        # Uses global robust scaling if robust_params provided, else per-window whitening
        windows = self._normalize_composite_scores(windows, robust_params=self.robust_params)

        # CRITICAL FIX (2026-01-06): Filter windows with NaN in critical features
        # After removing .bfill() from indicator calculations, early windows may have NaN
        # in BBW (index 5) and VAR (index 9) due to insufficient warmup data
        # These must be filtered to prevent training on invalid data
        BBW_IDX = 5
        VAR_IDX = 9
        critical_feature_indices = [BBW_IDX, VAR_IDX]

        # Check for NaN in critical features across all timesteps
        nan_mask = np.isnan(windows[:, :, critical_feature_indices]).any(axis=(1, 2))
        n_nan_windows = nan_mask.sum()

        if n_nan_windows > 0:
            logger.debug(f"Filtered {n_nan_windows}/{len(windows)} windows with NaN in BBW/VAR")
            windows = windows[~nan_mask]

        if len(windows) == 0:
            logger.debug("All windows filtered due to NaN in critical features")
            return None

        return windows

    def _relativize_price_features(self, windows: np.ndarray) -> np.ndarray:
        """
        Relativize OHLC and boundary features to make model price-agnostic.

        Converts absolute prices to relative changes from day 0:
            Relative_t = (Price_t / Price_0) - 1

        This fixes the "Global Price Fallacy": A $2 micro-cap and a $20 small-cap
        with identical consolidation patterns should produce identical sequences.

        Features relativized:
        - Indices 0-3: open, high, low, close (OHLC)
        - Indices 12-13: upper_boundary, lower_boundary

        Features NOT relativized (handled separately):
        - Index 4: volume (normalized via log ratio in _normalize_volume())
        - Indices 5-7: technical indicators (already bounded: BBW, ADX, volume_ratio)
        - Indices 8-9: boundary slopes (upper_slope, lower_slope)

        Args:
            windows: Array of shape (n_windows, window_size, n_features)
                     where n_features = 10 (composite features disabled)

        Returns:
            Relativized windows with same shape

        Example:
            Before: Day 0 close = $50, Day 5 close = $55
            After:  Day 0 close = 0.0, Day 5 close = 0.10 (10% gain)
        """
        # OHLC indices: 0=open, 1=high, 2=low, 3=close
        ohlc_indices = [0, 1, 2, 3]

        # GEOMETRY FIX (2026-01-18): Boundary features (indices 12-13) are now
        # normalized convergence slopes, NOT absolute prices. Do NOT relativize them.
        #
        # Old (broken): Static box coordinates → variance = 0 → model sees Rectangle
        # New (fixed): Dynamic slope features → variance > 0 → model sees Triangle
        #
        # Upper/lower slope features are already:
        # 1. Scale-invariant (normalized by box_height / lookback)
        # 2. Time-varying (calculated per timestep via rolling regression)
        # 3. Bounded (clipped to [-3, 3] range)

        # Only relativize OHLC price features
        price_indices = ohlc_indices  # Removed: boundary_indices = [12, 13]

        # Reference price: close at timestep 0 (index 3)
        # Shape: (n_windows, 1, 1) for broadcasting
        # CRITICAL: Use .copy() to prevent aliasing - slicing creates a view that shares memory
        reference_close = windows[:, 0:1, 3:4].copy()  # (n_windows, 1, 1)

        # Avoid division by zero
        reference_close = np.where(reference_close == 0, 1e-8, reference_close)

        # Relativize: (Price_t / Price_0) - 1
        # Vectorized across all windows and timesteps
        for idx in price_indices:
            windows[:, :, idx] = (windows[:, :, idx:idx+1] / reference_close)[:, :, 0] - 1

        return windows

    def _normalize_volume(
        self,
        windows: np.ndarray,
        volume_ma_50: np.ndarray = None
    ) -> np.ndarray:
        """
        Normalize volume using log ratio to reference value.

        Three modes (in priority order):
        1. 6-MONTH MEDIAN REFERENCE (preferred - Window-Anchor Fix):
           - Formula: log(volume_t / vol_6m_median)
           - LSTM sees "Wake Up" magnitude relative to dormant history
           - Passed via volume_ma_50 parameter (reused for any reference)

        2. 50-DAY MA REFERENCE: Uses stable moving average
           - Formula: log(volume_t / 50_day_MA_volume)
           - Cross-sectional comparability (same MA interpretation across stocks)
           - Requires volume_ma_50 parameter

        3. DAY-0 REFERENCE (legacy fallback): Uses volume at timestep 0
           - Formula: log(volume_t / volume_0)
           - Pattern-specific, arbitrary reference point
           - Used as fallback when no reference provided

        Interpretation (same for all modes):
        - 0.0 = same volume as reference
        - +0.69 = volume doubled (log(2))
        - -0.69 = volume halved (log(0.5))
        - +2.3 = 10x volume spike (log(10))
        - -2.3 = volume dropped to 10% (log(0.1))

        Output range: Typically [-3, +5] covering 0.05x to 150x volume changes
        Extreme values are clipped to [-4, +6] for numerical stability.

        Args:
            windows: Array of shape (n_windows, window_size, n_features)
                     where volume is at index 4
            volume_ma_50: Optional array of shape (n_windows,) or (n_windows, 1)
                         containing reference volume for each window
                         (can be 6-month median, 50-day MA, or any stable reference)

        Returns:
            Windows with volume normalized, same shape

        Note:
            - Zero or negative volumes are handled with epsilon (1e-8)
            - This preserves temporal dynamics (volume trends visible)
            - Does NOT use window mean/std (which would leak future info within window)
        """
        VOLUME_IDX = 4
        EPS = 1e-8

        if volume_ma_50 is not None:
            # 50-DAY MA REFERENCE (preferred)
            # Ensure shape (n_windows, 1) for broadcasting
            if volume_ma_50.ndim == 1:
                volume_ref = volume_ma_50[:, np.newaxis].copy()  # (n_windows, 1)
            else:
                volume_ref = volume_ma_50.copy()
        else:
            # DAY-0 REFERENCE (legacy fallback)
            # CRITICAL: Use .copy() to prevent aliasing
            volume_ref = windows[:, 0, VOLUME_IDX:VOLUME_IDX+1].copy()  # (n_windows, 1)

        # Handle zero volumes with epsilon
        volume_ref = np.maximum(volume_ref, EPS)

        # Extract all volume values: (n_windows, 20)
        volume_all = windows[:, :, VOLUME_IDX]

        # Handle zero/negative volumes
        volume_all = np.maximum(volume_all, EPS)

        # Compute log ratio: log(volume_t / volume_ref)
        volume_normalized = np.log(volume_all / volume_ref)

        # Clip extreme values for numerical stability
        # [-4, +6] covers ~0.02x to ~400x volume changes
        volume_normalized = np.clip(volume_normalized, -4, 6)

        # Write back to windows array
        windows[:, :, VOLUME_IDX] = volume_normalized

        return windows

    def _normalize_composite_scores(
        self,
        windows: np.ndarray,
        robust_params: dict = None
    ) -> np.ndarray:
        """
        Apply normalization to composite score features.

        DISABLED (2026-01-18): Composite features removed from feature set.
        This method now returns windows unchanged for backwards compatibility.

        Previously normalized (now disabled):
        - Index 8: vol_dryup_ratio
        - Index 9: var_score
        - Index 10: nes_score
        - Index 11: lpf_score

        Args:
            windows: Array of shape (n_windows, window_size, n_features)
            robust_params: Unused (kept for API compatibility)

        Returns:
            Windows unchanged (composite features disabled)
        """
        # DISABLED (2026-01-18): Composite features removed
        # Return windows unchanged
        return windows

        # Legacy code below (disabled):
        # SKIP MODE: For sequence generation, save RAW values
        # This fixes the chicken-and-egg problem where robust_params need raw data to compute
        if self.skip_composite_normalization:
            return windows

        # Composite score indices (from config/temporal_features.py)
        composite_indices = []  # DISABLED: was [8, 9, 10, 11]

        if robust_params is not None:
            # GLOBAL ROBUST SCALING: Use training-set Median/IQR
            for idx in composite_indices:
                median_key = f'feat_{idx}_median'
                iqr_key = f'feat_{idx}_iqr'

                if median_key in robust_params and iqr_key in robust_params:
                    median = robust_params[median_key]
                    iqr = robust_params[iqr_key]

                    # Apply robust scaling: (value - median) / IQR
                    if iqr > 1e-8:
                        windows[:, :, idx] = (windows[:, :, idx] - median) / iqr
                    # If IQR is too small, leave values unchanged
        else:
            # LEGACY PER-WINDOW WHITENING: Each sequence normalized independently
            composite_features = windows[:, :, composite_indices]

            # Compute mean and std per sequence per feature (across 20 timesteps)
            means = composite_features.mean(axis=1, keepdims=True)  # (n_windows, 1, 4)
            stds = composite_features.std(axis=1, keepdims=True)    # (n_windows, 1, 4)

            # Normalize where std > threshold (avoid division by zero)
            valid_mask = stds > 1e-8
            normalized = np.where(valid_mask, (composite_features - means) / stds, composite_features)

            # Write back to windows array
            windows[:, :, composite_indices] = normalized

        return windows

    def _prepare_pattern_features(
        self,
        pattern_data: pd.DataFrame,
        pattern: ConsolidationPattern
    ) -> pd.DataFrame:
        """
        Prepare all 14 temporal features using vectorized operations.

        Uses the feature configuration to ensure correct features are calculated
        in the right order. All operations are vectorized for performance.

        Args:
            pattern_data: DataFrame with pattern OHLCV data
            pattern: ConsolidationPattern object with boundaries

        Returns:
            DataFrame with all 14 features calculated
        """
        df = pattern_data.copy()

        # Calculate technical indicators using vectorized calculator
        df = self.feature_calculator.calculate_all_features(df)

        # Add structural boundaries from pattern
        if pattern.upper_boundary and pattern.lower_boundary:
            df = self.feature_calculator.add_boundaries_from_pattern(
                df,
                upper_boundary=pattern.upper_boundary,
                lower_boundary=pattern.lower_boundary
            )
        else:
            # Use high/low as fallback boundaries
            df['upper_boundary'] = df['high'].rolling(20, min_periods=1).max()
            df['lower_boundary'] = df['low'].rolling(20, min_periods=1).min()

        # Ensure all required features are present
        for feature in self.feature_config.all_features:
            if feature not in df.columns:
                if feature in self.feature_config.market_features:
                    # Market features should already be present
                    if feature not in df.columns:
                        raise ValidationError(f"Missing market feature: {feature}")
                elif feature in self.feature_config.composite_features:
                    # Composite features should be calculated by feature_calculator
                    logger.warning(f"Missing composite feature {feature}, using default")
                    df[feature] = 0.0

        return df

    def _get_pattern_label(self, pattern: ConsolidationPattern) -> int:
        """
        Get label for pattern.

        Returns:
            int: Label (0=Danger, 1=Noise, 2=Target, -1=Grey, -2=Unripe)

        Training mode: Only terminal patterns with definitive outcomes (0, 1, 2)
        Inference mode: Includes placeholder (-2) for unripe patterns
        """
        from config.constants import LABEL_PLACEHOLDER_UNRIPE, LABEL_GREY_ZONE, ProcessingMode

        # Unripe patterns (active/qualifying) - no outcome yet
        if pattern.phase in [PatternPhase.ACTIVE, PatternPhase.QUALIFYING]:
            if self.mode == ProcessingMode.INFERENCE:
                return LABEL_PLACEHOLDER_UNRIPE  # -2
            else:
                # FLAW FIX #7: Log warning instead of crash
                logger.warning(
                    f"Unripe pattern {pattern.ticker} reached labeling in TRAINING mode. "
                    f"This should have been filtered earlier. Skipping."
                )
                return LABEL_PLACEHOLDER_UNRIPE  # Return placeholder, will be filtered

        # Terminal patterns - use PathDependentLabelerV17
        if not pattern.is_terminal():
            logger.warning(f"Non-terminal pattern {pattern.ticker} in labeling - defaulting to Noise")
            return 1

        # Validate boundaries exist
        if pattern.upper_boundary is None or pattern.lower_boundary is None:
            logger.warning(f"Pattern {pattern.ticker} missing boundaries - cannot label")
            return None

        # Call v17 labeler with full data
        try:
            label = self.labeler_v17.label_pattern(
                full_data=self.full_data_df,  # Full DataFrame from process_ticker
                pattern_end_idx=pattern.end_idx,
                pattern_boundaries={
                    'upper': pattern.upper_boundary,
                    'lower': pattern.lower_boundary
                }
            )

            # Handle return values
            if label is None:
                logger.warning(
                    f"V17 labeler returned None for {pattern.ticker} "
                    f"(insufficient data or integrity failure)"
                )
                return None

            if label == LABEL_GREY_ZONE:
                # Grey zones will be filtered in sequence generation loop (line 197-199)
                logger.debug(f"Pattern {pattern.ticker} labeled as grey zone (-1)")
                return label

            # Valid labels: 0 (Danger), 1 (Noise), 2 (Target)
            return label

        except Exception as e:
            logger.error(f"V17 labeler error for {pattern.ticker}: {e}", exc_info=True)
            return None

