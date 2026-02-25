"""
Bull/Bear Market Phase Classifier
==================================

Simple moving average-based bull/bear market classifier with hysteresis logic.
Transforms index data into a queryable reference book for checking market phases
on any date with guaranteed temporal integrity (no forward-looking bias).

Key Features:
- Compute-once, O(1) lookup architecture
- Hysteresis to avoid false signals in sideways markets
- Dual storage: CSV (backup) + database (queries)
- Configurable MA periods (default 50/200)
- Strict temporal integrity guaranteed

Usage:
    from utils.bull_bear_classifier import get_classifier

    classifier = get_classifier()
    phase = classifier.get_phase('2020-03-15')
    print(f"Phase: {'Bull' if phase.phase == 1 else 'Bear'}")
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union, Optional, Dict, List, Any
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MarketPhaseData:
    """Container for market phase information at a specific date."""

    date: str  # ISO format (YYYY-MM-DD)
    phase: Optional[int]  # 1 = Bull, 0 = Bear, None = Insufficient data
    fast_ma: Optional[float]  # 50-day MA value (or custom period)
    slow_ma: Optional[float]  # 200-day MA value (or custom period)
    crossover_date: Optional[str]  # Last crossover date (ISO format)
    days_in_phase: int  # Days since last crossover

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketPhaseData':
        """Create from dictionary."""
        return cls(**data)

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.phase is None:
            return f"{self.date}: Insufficient data for classification"
        phase_name = "BULL" if self.phase == 1 else "BEAR"
        return (f"{self.date}: {phase_name} Market "
                f"(50MA: {self.fast_ma:.2f}, 200MA: {self.slow_ma:.2f}, "
                f"Days: {self.days_in_phase})")


class BullBearClassifier:
    """
    Bull/Bear market phase classifier based on MA crossovers.

    Architecture:
    1. Load index data once on initialization
    2. Calculate fast/slow MAs (default 50/200)
    3. Apply hysteresis logic with state tracking (no forward-looking bias)
    4. Build regime_map dictionary: {date_str: MarketPhaseData}
    5. Persist to CSV and optionally to database

    The compute-once architecture ensures O(1) lookups after initialization.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        index_symbol: str = 'USA500IDXUSD',
        fast_period: int = 50,
        slow_period: int = 200,
        hysteresis_pct: float = 0.05,
        cache_dir: Optional[Path] = None,
        enable_db_storage: bool = False
    ):
        """
        Initialize classifier - computes all phases upfront.

        Args:
            data_path: Path to index CSV file (Time, Open, High, Low, Close, Volume)
            index_symbol: Index identifier (e.g., 'USA500IDXUSD')
            fast_period: Fast MA period (default: 50 days)
            slow_period: Slow MA period (default: 200 days)
            hysteresis_pct: Hysteresis buffer percentage (default: 0.05 = 5%)
            cache_dir: Directory for CSV storage (default: data/market_phases)
            enable_db_storage: Whether to store in database (default: False)
        """
        self.data_path = Path(data_path)
        self.index_symbol = index_symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.hysteresis_pct = hysteresis_pct
        self.enable_db_storage = enable_db_storage

        # Set cache directory
        if cache_dir is None:
            self.cache_dir = Path(__file__).parent.parent / 'data' / 'market_phases'
        else:
            self.cache_dir = Path(cache_dir)

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # CSV file path
        self.csv_path = self.cache_dir / f"{index_symbol}_phases.csv"
        self.metadata_path = self.cache_dir / 'metadata.json'

        logger.info(f"Initializing BullBearClassifier for {index_symbol}...")
        logger.info(f"Parameters: fast_period={fast_period}, slow_period={slow_period}, hysteresis={hysteresis_pct}")

        # Load and compute
        try:
            df = self._load_and_compute()
        except FileNotFoundError:
            logger.error(f"Data file {data_path} not found!")
            raise

        # Build regime map for O(1) lookups
        self.regime_map = self._build_regime_map(df)

        # Store metadata
        self.last_date = df['date'].max()
        last_phase_data = self.regime_map.get(self.last_date.strftime('%Y-%m-%d'))
        self.last_phase = last_phase_data.phase if last_phase_data else None
        self.total_days = len(df)

        # Save to CSV
        self._save_to_csv(df)

        # Optionally save to database
        if self.enable_db_storage:
            self._save_to_database(df)

        # Save metadata
        self._save_metadata(df)

        logger.info(f"Initialization complete. {len(self.regime_map)} trading days loaded.")
        if self.last_phase is not None:
            phase_name = 'BULL' if self.last_phase == 1 else 'BEAR'
            logger.info(f"Last date ({self.last_date.date()}): {phase_name} market")
        else:
            logger.info(f"Last date ({self.last_date.date()}): Insufficient data for classification")

    def _load_and_compute(self) -> pd.DataFrame:
        """
        Load CSV data and compute MAs with hysteresis logic.

        Returns:
            DataFrame with columns: date, phase, fast_ma, slow_ma, crossover_date, days_in_phase
        """
        # Load CSV (handles both comma and tab-delimited files)
        logger.info(f"Loading data from {self.data_path}...")

        # Try to detect delimiter
        try:
            df = pd.read_csv(self.data_path)
            # Check if it's actually tab-delimited being read as comma-delimited
            if len(df.columns) == 1 and '\t' in df.columns[0]:
                # Re-read as tab-delimited
                df = pd.read_csv(self.data_path, sep='\t')
        except Exception:
            # Try tab-delimited
            df = pd.read_csv(self.data_path, sep='\t')

        # Parse date column (try 'Time' first, then 'Date')
        if 'Time' in df.columns:
            df['date'] = pd.to_datetime(df['Time'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
        else:
            raise ValueError(f"CSV must have 'Time' or 'Date' column. Found columns: {df.columns.tolist()}")

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} days of data from {df['date'].min().date()} to {df['date'].max().date()}")

        # Calculate moving averages
        logger.info(f"Calculating {self.fast_period}-day and {self.slow_period}-day moving averages...")
        closes = df['Close'].values
        df['fast_ma'] = df['Close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['Close'].rolling(window=self.slow_period).mean()

        # Apply hysteresis logic
        logger.info(f"Applying hysteresis logic ({self.hysteresis_pct * 100:.1f}% buffer)...")
        df = self._apply_hysteresis_logic(df)

        return df

    def _apply_hysteresis_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply hysteresis to avoid false signals in sideways markets.

        Logic:
        - Start with current_regime = None (unknown)
        - For crossover from bear→bull: fast_ma > slow_ma × (1 + hysteresis_pct)
        - For crossover from bull→bear: fast_ma < slow_ma × (1 - hysteresis_pct)
        - Track state transitions and crossover dates
        - Loop forward through time (strict temporal integrity)

        Args:
            df: DataFrame with 'date', 'fast_ma', 'slow_ma' columns

        Returns:
            DataFrame with added columns: phase, crossover_date, days_in_phase
        """
        results = []
        current_regime = None  # Start unknown (None)
        last_crossover_date = None
        transition_count = 0

        for idx, row in df.iterrows():
            date = row['date']
            fast_ma = row['fast_ma']
            slow_ma = row['slow_ma']

            # Skip if MAs not available (first slow_period days)
            if pd.isna(fast_ma) or pd.isna(slow_ma):
                results.append({
                    'date': date,
                    'phase': None,
                    'fast_ma': None,
                    'slow_ma': None,
                    'crossover_date': None,
                    'days_in_phase': 0
                })
                continue

            # Initialize regime on first valid MA
            if current_regime is None:
                current_regime = 1 if fast_ma > slow_ma else 0
                last_crossover_date = date
                logger.debug(f"Initial regime at {date.date()}: {'BULL' if current_regime == 1 else 'BEAR'}")

            # Check for crossover with hysteresis
            previous_regime = current_regime

            if current_regime == 0:  # Currently bear
                # Crossover to bull requires fast_ma > slow_ma * (1 + hysteresis)
                bull_threshold = slow_ma * (1 + self.hysteresis_pct)
                if fast_ma > bull_threshold:
                    current_regime = 1
                    last_crossover_date = date
                    transition_count += 1
                    logger.debug(f"Transition {transition_count}: BEAR → BULL at {date.date()} "
                               f"(fast_ma={fast_ma:.2f} > {bull_threshold:.2f})")
            else:  # Currently bull
                # Crossover to bear requires fast_ma < slow_ma * (1 - hysteresis)
                bear_threshold = slow_ma * (1 - self.hysteresis_pct)
                if fast_ma < bear_threshold:
                    current_regime = 0
                    last_crossover_date = date
                    transition_count += 1
                    logger.debug(f"Transition {transition_count}: BULL → BEAR at {date.date()} "
                               f"(fast_ma={fast_ma:.2f} < {bear_threshold:.2f})")

            # Calculate days in phase
            days_in_phase = (date - last_crossover_date).days if last_crossover_date else 0

            results.append({
                'date': date,
                'phase': current_regime,
                'fast_ma': float(fast_ma),
                'slow_ma': float(slow_ma),
                'crossover_date': last_crossover_date,
                'days_in_phase': days_in_phase
            })

        logger.info(f"Detected {transition_count} phase transitions")

        result_df = pd.DataFrame(results)
        return result_df

    def _build_regime_map(self, df: pd.DataFrame) -> Dict[str, MarketPhaseData]:
        """
        Build regime map dictionary for O(1) lookups.

        Args:
            df: DataFrame with phase data

        Returns:
            Dictionary mapping date strings to MarketPhaseData objects
        """
        regime_map = {}

        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            crossover_str = row['crossover_date'].strftime('%Y-%m-%d') if pd.notna(row['crossover_date']) else None

            regime_map[date_str] = MarketPhaseData(
                date=date_str,
                phase=int(row['phase']) if pd.notna(row['phase']) else None,
                fast_ma=float(row['fast_ma']) if pd.notna(row['fast_ma']) else None,
                slow_ma=float(row['slow_ma']) if pd.notna(row['slow_ma']) else None,
                crossover_date=crossover_str,
                days_in_phase=int(row['days_in_phase'])
            )

        return regime_map

    def _save_to_csv(self, df: pd.DataFrame):
        """Save phases to CSV file."""
        logger.info(f"Saving to CSV: {self.csv_path}")

        # Prepare DataFrame for CSV
        csv_df = df.copy()
        csv_df['date'] = csv_df['date'].dt.strftime('%Y-%m-%d')
        csv_df['crossover_date'] = csv_df['crossover_date'].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None
        )

        # Add configuration columns
        csv_df['fast_period'] = self.fast_period
        csv_df['slow_period'] = self.slow_period
        csv_df['hysteresis_pct'] = self.hysteresis_pct

        # Save to CSV
        csv_df.to_csv(self.csv_path, index=False)
        logger.info(f"✓ Saved {len(csv_df)} rows to CSV")

    def _save_to_database(self, df: pd.DataFrame):
        """Save phases to database (if enabled)."""
        try:
            from database.connection import get_db_session
            from database.models import MarketPhase

            logger.info("Saving to database...")

            with get_db_session() as session:
                # Delete existing data for this index
                session.query(MarketPhase).filter_by(index_symbol=self.index_symbol).delete()

                # Prepare records
                records = []
                for _, row in df.iterrows():
                    if pd.notna(row['phase']):  # Only save valid phases
                        records.append(MarketPhase(
                            index_symbol=self.index_symbol,
                            date=row['date'],
                            phase=int(row['phase']),
                            fast_ma=float(row['fast_ma']),
                            slow_ma=float(row['slow_ma']),
                            fast_period=self.fast_period,
                            slow_period=self.slow_period,
                            crossover_date=row['crossover_date'] if pd.notna(row['crossover_date']) else None,
                            days_in_phase=int(row['days_in_phase']),
                            hysteresis_pct=self.hysteresis_pct
                        ))

                # Batch insert
                session.bulk_save_objects(records)
                session.commit()

                logger.info(f"✓ Saved {len(records)} rows to database")

        except ImportError:
            logger.warning("Database models not available, skipping database save")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")

    def _save_metadata(self, df: pd.DataFrame):
        """Save metadata JSON file."""
        # Calculate statistics
        valid_phases = df[df['phase'].notna()]
        bull_days = len(valid_phases[valid_phases['phase'] == 1])
        bear_days = len(valid_phases[valid_phases['phase'] == 0])

        # Count transitions
        transitions = (valid_phases['crossover_date'].shift(-1) != valid_phases['crossover_date']).sum()

        metadata = {
            'index_symbol': self.index_symbol,
            'data_source': str(self.data_path.name),
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'hysteresis_pct': self.hysteresis_pct,
            'last_computed': datetime.now().isoformat(),
            'total_days': self.total_days,
            'valid_days': len(valid_phases),
            'bull_days': int(bull_days),
            'bear_days': int(bear_days),
            'transitions': int(transitions),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Saved metadata to {self.metadata_path}")

    def get_phase(self, date_input: Union[str, datetime]) -> Optional[MarketPhaseData]:
        """
        Get market phase for a specific date (O(1) lookup).

        Args:
            date_input: Date as string ('YYYY-MM-DD') or datetime object

        Returns:
            MarketPhaseData object or None if date not found
        """
        # Normalize date to string
        if isinstance(date_input, datetime):
            date_key = date_input.strftime('%Y-%m-%d')
        else:
            date_key = pd.to_datetime(date_input).strftime('%Y-%m-%d')

        return self.regime_map.get(date_key)

    def get_phase_range(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Get phases for a date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with phase data for the date range
        """
        # Normalize dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Filter regime map
        results = []
        for date_str, phase_data in self.regime_map.items():
            date = pd.to_datetime(date_str)
            if start <= date <= end:
                results.append(phase_data.to_dict())

        return pd.DataFrame(results).sort_values('date')

    def get_transitions(self, min_days: int = 0) -> List[Dict[str, Any]]:
        """
        Get all bull/bear transitions with durations.

        Args:
            min_days: Minimum days in phase to include (default: 0 = all)

        Returns:
            List of dictionaries with transition information
        """
        transitions = []
        prev_phase = None
        prev_date = None

        for date_str in sorted(self.regime_map.keys()):
            phase_data = self.regime_map[date_str]

            if phase_data.phase is None:
                continue

            # Detect transition
            if prev_phase is not None and phase_data.phase != prev_phase:
                duration = (pd.to_datetime(date_str) - pd.to_datetime(prev_date)).days

                if duration >= min_days:
                    transitions.append({
                        'date': date_str,
                        'from': 'Bull' if prev_phase == 1 else 'Bear',
                        'to': 'Bull' if phase_data.phase == 1 else 'Bear',
                        'duration_days': duration,
                        'prev_date': prev_date
                    })

            prev_phase = phase_data.phase
            prev_date = date_str

        return transitions

    def get_stats(self) -> Dict[str, Any]:
        """
        Get classifier statistics.

        Returns:
            Dictionary with statistics (bull days, bear days, transitions, etc.)
        """
        valid_phases = [p for p in self.regime_map.values() if p.phase is not None]
        bull_phases = [p for p in valid_phases if p.phase == 1]
        bear_phases = [p for p in valid_phases if p.phase == 0]

        transitions = self.get_transitions()
        bull_transitions = [t for t in transitions if t['to'] == 'Bull']
        bear_transitions = [t for t in transitions if t['to'] == 'Bear']

        return {
            'index_symbol': self.index_symbol,
            'total_days': self.total_days,
            'valid_days': len(valid_phases),
            'bull_days': len(bull_phases),
            'bear_days': len(bear_phases),
            'bull_pct': len(bull_phases) / len(valid_phases) * 100 if valid_phases else 0,
            'bear_pct': len(bear_phases) / len(valid_phases) * 100 if valid_phases else 0,
            'transitions': len(transitions),
            'bull_periods': len(bull_transitions),
            'bear_periods': len(bear_transitions),
            'avg_bull_duration': np.mean([t['duration_days'] for t in bull_transitions]) if bull_transitions else 0,
            'avg_bear_duration': np.mean([t['duration_days'] for t in bear_transitions]) if bear_transitions else 0,
            'config': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'hysteresis_pct': self.hysteresis_pct
            }
        }

    def rebuild(self, force: bool = False):
        """
        Recompute all phases (use when index data updates).

        Args:
            force: Force rebuild even if data hasn't changed
        """
        logger.info("Rebuilding classifier...")
        self.__init__(
            data_path=self.data_path,
            index_symbol=self.index_symbol,
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            hysteresis_pct=self.hysteresis_pct,
            cache_dir=self.cache_dir,
            enable_db_storage=self.enable_db_storage
        )
        logger.info("✓ Rebuild complete")


# Global singleton instance
_global_classifier: Optional[BullBearClassifier] = None


def get_classifier(
    index_symbol: str = 'USA500IDXUSD',
    force_rebuild: bool = False,
    **kwargs
) -> BullBearClassifier:
    """
    Get or create global classifier instance (singleton pattern).

    This avoids recomputing on every call. Use force_rebuild=True
    to reload if index data has been updated.

    Args:
        index_symbol: Index identifier (default: 'USA500IDXUSD')
        force_rebuild: Force rebuild even if instance exists
        **kwargs: Additional arguments passed to BullBearClassifier

    Returns:
        BullBearClassifier instance
    """
    global _global_classifier

    if _global_classifier is None or force_rebuild:
        # Determine data path
        data_path = Path(__file__).parent.parent / f'{index_symbol}_D1.csv'

        if not data_path.exists():
            raise FileNotFoundError(f"Index data not found: {data_path}")

        _global_classifier = BullBearClassifier(
            data_path=data_path,
            index_symbol=index_symbol,
            **kwargs
        )

    return _global_classifier
