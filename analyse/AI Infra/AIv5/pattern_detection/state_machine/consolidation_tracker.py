"""
Consolidation Tracker - State Machine for Pattern Detection

Tracks individual tickers through consolidation pattern lifecycle:
NONE → QUALIFYING (10 days) → ACTIVE → NONE (reset after completion)

After each pattern completes (breakout/breakdown/timeout), the tracker resets
to NONE to allow detection of multiple patterns per ticker.

Critical feature: Maintains temporal integrity - never uses future data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PatternPhase(Enum):
    """Pattern lifecycle phases."""
    NONE = "NONE"  # Not in consolidation
    QUALIFYING = "QUALIFYING"  # First 10 days - testing consolidation criteria
    ACTIVE = "ACTIVE"  # After 10 days - active consolidation pattern
    COMPLETED = "COMPLETED"  # Broke out successfully
    FAILED = "FAILED"  # Broke down


class ConsolidationPattern:
    """Represents a detected consolidation pattern."""

    def __init__(
        self,
        ticker: str,
        start_date: datetime,
        start_idx: int,
        start_price: float
    ):
        """
        Initialize pattern.

        Args:
            ticker: Stock ticker symbol
            start_date: Pattern start date
            start_idx: Pattern start index in data
            start_price: Price at pattern start
        """
        self.ticker = ticker
        self.start_date = start_date
        self.start_idx = start_idx
        self.start_price = start_price

        self.end_date = None
        self.end_idx = None
        self.end_price = None

        self.upper_boundary = None
        self.lower_boundary = None
        self.power_boundary = None  # upper * 1.005
        self.boundary_range_pct = None  # (upper - lower) / lower

        # Temporal tracking (LEAK-FREE)
        self.days_since_activation = 0  # Days since pattern became ACTIVE (leak-free)
        self.total_days_in_pattern = 0  # Total duration until completion (LEAKED - analysis only!)
        self.days_qualifying = 0

        # Activation tracking
        self.activation_date = None
        self.activation_idx = None

        # Track highs/lows during qualification period (days 1-10)
        self.qualification_highs = []
        self.qualification_lows = []

        self.phase = PatternPhase.QUALIFYING

        # Feature snapshots for training (MULTIPLE samples per pattern)
        # Each snapshot = features from a random day during ACTIVE phase
        # Longer patterns → more snapshots → proportional contribution to training
        self.feature_snapshots = []

        # ========================================
        # RECENT WINDOW (Last 20 days) - Updated daily during ACTIVE phase
        # ========================================
        self.avg_bbw_20d = None  # Average BBW last 20 days
        self.avg_adx_20d = None  # Average ADX last 20 days
        self.avg_volume_ratio_20d = None  # Volume vs baseline
        self.avg_range_ratio_20d = None  # Range vs baseline

        # Recent trends (20-day slopes)
        self.bbw_slope_20d = None  # BBW tightening (-) or widening (+)
        self.volume_slope_20d = None  # Volume declining (-) or rising (+)
        self.adx_slope_20d = None  # Trend weakening (-) or strengthening (+)

        # Recent stability
        self.bbw_std_20d = None  # BBW volatility (lower = more stable)
        self.volume_std_20d = None  # Volume consistency

        # ========================================
        # BASELINE WINDOW (Days -50 to -20) - Updated daily
        # ========================================
        self.baseline_bbw_avg = None  # BBW average 20-50 days ago
        self.baseline_adx_avg = None  # ADX average 20-50 days ago
        self.baseline_volume_avg = None  # Volume average 20-50 days ago
        self.baseline_range_avg = None  # Range average 20-50 days ago
        self.baseline_volatility = None  # ATR 20-50 days ago
        self.baseline_bbw_std = None  # BBW stability 20-50 days ago
        self.baseline_volume_std = None  # Volume stability 20-50 days ago

        # ========================================
        # RELATIVE COMPRESSION METRICS (Recent vs Baseline)
        # ========================================
        self.bbw_compression_ratio = None  # Recent / Baseline (< 1.0 = tighter now)
        self.volume_compression_ratio = None  # Recent / Baseline (< 1.0 = quieter now)
        self.range_compression_ratio = None  # Recent / Baseline (< 1.0 = tighter range)
        self.volatility_stability_ratio = None  # Recent_std / Baseline_std (< 1.0 = more stable)
        self.overall_compression = None  # Combined compression score (0.0-1.0)

        # ========================================
        # PRICE POSITION FEATURES - Updated daily
        # ========================================
        self.price_position_in_range = None  # 0.0 (lower) to 1.0 (upper)
        self.price_distance_from_upper_pct = None  # % from breakout
        self.price_distance_from_lower_pct = None  # % from breakdown

        # Legacy fields (kept for backward compatibility, but NOT used for training)
        self.avg_bbw = None  # Old static avg (use avg_bbw_20d instead)
        self.avg_volume_ratio = None  # Old static avg (use avg_volume_ratio_20d instead)
        self.avg_adx = None  # Old static avg (use avg_adx_20d instead)
        self.range_compression = None  # Old static (use range_compression_ratio instead)

        # Outcome
        self.breakout_date = None
        self.breakout_direction = None  # 'UP' or 'DOWN'
        self.max_gain = None

    def to_dict(self) -> Dict:
        """Convert to dictionary with all features."""
        return {
            # Basic info
            'ticker': self.ticker,
            'start_date': self.start_date,
            'start_idx': self.start_idx,
            'start_price': self.start_price,
            'end_date': self.end_date,
            'end_idx': self.end_idx,
            'end_price': self.end_price,
            'phase': self.phase.value,

            # Temporal (LEAK-FREE)
            'days_since_activation': self.days_since_activation,
            'total_days_in_pattern': self.total_days_in_pattern,  # ⚠️ LEAKED - analysis only!

            # Activation info
            'activation_date': self.activation_date,
            'activation_idx': self.activation_idx,

            # Static boundaries
            'upper_boundary': self.upper_boundary,
            'lower_boundary': self.lower_boundary,
            'power_boundary': self.power_boundary,
            'boundary_range_pct': self.boundary_range_pct,

            # Recent window (20 days)
            'avg_bbw_20d': self.avg_bbw_20d,
            'avg_adx_20d': self.avg_adx_20d,
            'avg_volume_ratio_20d': self.avg_volume_ratio_20d,
            'avg_range_ratio_20d': self.avg_range_ratio_20d,
            'bbw_slope_20d': self.bbw_slope_20d,
            'volume_slope_20d': self.volume_slope_20d,
            'adx_slope_20d': self.adx_slope_20d,
            'bbw_std_20d': self.bbw_std_20d,
            'volume_std_20d': self.volume_std_20d,

            # Baseline window (Days -50 to -20)
            'baseline_bbw_avg': self.baseline_bbw_avg,
            'baseline_adx_avg': self.baseline_adx_avg,
            'baseline_volume_avg': self.baseline_volume_avg,
            'baseline_range_avg': self.baseline_range_avg,
            'baseline_volatility': self.baseline_volatility,
            'baseline_bbw_std': self.baseline_bbw_std,
            'baseline_volume_std': self.baseline_volume_std,

            # Compression metrics
            'bbw_compression_ratio': self.bbw_compression_ratio,
            'volume_compression_ratio': self.volume_compression_ratio,
            'range_compression_ratio': self.range_compression_ratio,
            'volatility_stability_ratio': self.volatility_stability_ratio,
            'overall_compression': self.overall_compression,

            # Price position
            'price_position_in_range': self.price_position_in_range,
            'price_distance_from_upper_pct': self.price_distance_from_upper_pct,
            'price_distance_from_lower_pct': self.price_distance_from_lower_pct,

            # Legacy fields (backward compatibility - NOT for training)
            'avg_bbw': self.avg_bbw,
            'avg_volume_ratio': self.avg_volume_ratio,
            'avg_adx': self.avg_adx,
            'range_compression': self.range_compression,

            # Outcome
            'breakout_date': self.breakout_date,
            'breakout_direction': self.breakout_direction,
            'max_gain': self.max_gain
        }

    def _calculate_slope(self, values: pd.Series) -> float:
        """
        Calculate linear regression slope of a time series.

        Args:
            values: Time series values

        Returns:
            Slope (rise/run), or 0.0 if insufficient data
        """
        if len(values) < 2:
            return 0.0

        # Remove NaN values
        clean_values = values.dropna()
        if len(clean_values) < 2:
            return 0.0

        # Linear regression: y = mx + b
        x = np.arange(len(clean_values))
        y = clean_values.values

        # Calculate slope using least squares
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope

    def extract_pattern_features(
        self,
        date: datetime,
        idx: int,
        price_data: pd.DataFrame,
        indicators: Dict[str, float]
    ) -> None:
        """
        Extract leak-free features comparing recent 20 days vs baseline 20-50 days ago.

        This method extracts dynamic features that change daily during the ACTIVE phase:
        - Recent window (last 20 days): Current consolidation behavior
        - Baseline window (days -50 to -20): Comparison baseline
        - Compression ratios: How much tighter/quieter is recent vs baseline

        Critical: Uses only past data up to current day (no look-ahead bias).

        Args:
            date: Current date being analyzed
            idx: Current index in price_data
            price_data: Full price history DataFrame
            indicators: Current day's technical indicators
        """
        # =====================================================================
        # RECENT WINDOW: Last 20 days (including today)
        # =====================================================================
        recent_start = max(0, idx - 19)
        recent_window = price_data.iloc[recent_start:idx+1]

        if len(recent_window) >= 10:  # Need minimum data
            # Average indicators over recent 20 days
            if 'bbw_20' in recent_window.columns:
                self.avg_bbw_20d = recent_window['bbw_20'].mean()
                self.bbw_slope_20d = self._calculate_slope(recent_window['bbw_20'])
                self.bbw_std_20d = recent_window['bbw_20'].std()

            if 'adx' in recent_window.columns:
                self.avg_adx_20d = recent_window['adx'].mean()
                self.adx_slope_20d = self._calculate_slope(recent_window['adx'])

            if 'volume_ratio_20' in recent_window.columns:
                self.avg_volume_ratio_20d = recent_window['volume_ratio_20'].mean()
                self.volume_slope_20d = self._calculate_slope(recent_window['volume_ratio_20'])

            if 'daily_range' in recent_window.columns:
                self.avg_range_20d = recent_window['daily_range'].mean()
                self.range_slope_20d = self._calculate_slope(recent_window['daily_range'])

            # Price volatility (standard deviation of close prices)
            if 'close' in recent_window.columns:
                self.price_volatility_20d = recent_window['close'].std()

            # Volume characteristics
            if 'volume' in recent_window.columns:
                self.volume_std_20d = recent_window['volume'].std()

                # Volume slope: linear trend of volume over 20 days (already calculated above)

        # =====================================================================
        # BASELINE WINDOW: Days -50 to -20 (30-day window for comparison)
        # =====================================================================
        baseline_start = max(0, idx - 50)
        baseline_end = max(0, idx - 20)
        baseline_window = price_data.iloc[baseline_start:baseline_end]

        if len(baseline_window) >= 10:  # Need minimum data
            # Baseline averages
            if 'bbw_20' in baseline_window.columns:
                self.baseline_bbw_avg = baseline_window['bbw_20'].mean()
                self.baseline_bbw_std = baseline_window['bbw_20'].std()

            if 'volume' in baseline_window.columns:
                self.baseline_volume_avg = baseline_window['volume'].mean()

            if 'close' in baseline_window.columns:
                self.baseline_volatility = baseline_window['close'].std()

            if 'daily_range' in baseline_window.columns:
                self.baseline_range_avg = baseline_window['daily_range'].mean()

        # =====================================================================
        # COMPRESSION RATIOS: Recent / Baseline
        # =====================================================================
        # Ratio < 1.0 = compression (tightening)
        # Ratio > 1.0 = expansion (widening)

        if self.baseline_bbw_avg and self.baseline_bbw_avg > 0:
            self.bbw_compression_ratio = self.avg_bbw_20d / self.baseline_bbw_avg if self.avg_bbw_20d else None

        if self.baseline_volume_avg and self.baseline_volume_avg > 0:
            self.volume_compression_ratio = self.avg_volume_ratio_20d / (self.baseline_volume_avg / self.baseline_volume_avg) if self.avg_volume_ratio_20d else None

        if self.baseline_volatility and self.baseline_volatility > 0:
            self.volatility_stability_ratio = self.price_volatility_20d / self.baseline_volatility if self.price_volatility_20d else None

        if self.baseline_range_avg and self.baseline_range_avg > 0:
            self.range_compression_ratio = self.avg_range_20d / self.baseline_range_avg if self.avg_range_20d else None

        # Overall compression score (average of all ratios)
        compression_ratios = [
            r for r in [
                self.bbw_compression_ratio,
                self.volume_compression_ratio,
                self.volatility_stability_ratio,
                self.range_compression_ratio
            ] if r is not None
        ]

        if compression_ratios:
            self.overall_compression = np.mean(compression_ratios)

        # =====================================================================
        # PRICE POSITION: Where is price within the pattern range?
        # =====================================================================
        if self.upper_boundary and self.lower_boundary:
            current_price = indicators.get('close', 0)

            # Price position: 0.0 = at lower boundary, 1.0 = at upper boundary
            range_size = self.upper_boundary - self.lower_boundary
            if range_size > 0:
                self.price_position_in_range = (current_price - self.lower_boundary) / range_size

            # Distance from upper boundary (as percentage)
            if self.upper_boundary > 0:
                self.price_distance_from_upper_pct = (self.upper_boundary - current_price) / self.upper_boundary

            # Distance from lower boundary (as percentage)
            if self.lower_boundary > 0:
                self.price_distance_from_lower_pct = (current_price - self.lower_boundary) / self.lower_boundary

        # =====================================================================
        # DAYS SINCE ACTIVATION: Leak-free temporal tracking
        # =====================================================================
        # This is incremented daily during ACTIVE phase
        # (No update needed here - handled in _check_active_pattern)

    def save_feature_snapshot(self) -> Dict:
        """
        Save current feature values as a snapshot.

        Used to preserve features from the day BEFORE pattern exits,
        preventing data leakage from breakout/breakdown day.

        Returns:
            Dictionary with all dynamic feature values
        """
        return {
            'days_since_activation': self.days_since_activation,
            'total_days_in_pattern': self.total_days_in_pattern,

            # Recent window features
            'avg_bbw_20d': self.avg_bbw_20d,
            'avg_adx_20d': self.avg_adx_20d,
            'avg_volume_ratio_20d': self.avg_volume_ratio_20d,
            'avg_range_20d': self.avg_range_20d,
            'bbw_slope_20d': self.bbw_slope_20d,
            'adx_slope_20d': self.adx_slope_20d,
            'volume_slope_20d': self.volume_slope_20d,
            'range_slope_20d': self.range_slope_20d,
            'bbw_std_20d': self.bbw_std_20d,
            'volume_std_20d': self.volume_std_20d,
            'price_volatility_20d': self.price_volatility_20d,
            'volume_slope_20d': self.volume_slope_20d,

            # Baseline window features
            'baseline_bbw_avg': self.baseline_bbw_avg,
            'baseline_bbw_std': self.baseline_bbw_std,
            'baseline_volume_avg': self.baseline_volume_avg,
            'baseline_volatility': self.baseline_volatility,
            'baseline_range_avg': self.baseline_range_avg,

            # Compression ratios
            'bbw_compression_ratio': self.bbw_compression_ratio,
            'volume_compression_ratio': self.volume_compression_ratio,
            'volatility_stability_ratio': self.volatility_stability_ratio,
            'range_compression_ratio': self.range_compression_ratio,
            'overall_compression': self.overall_compression,

            # Price position
            'price_position_in_range': self.price_position_in_range,
            'price_distance_from_upper_pct': self.price_distance_from_upper_pct,
            'price_distance_from_lower_pct': self.price_distance_from_lower_pct
        }

    def restore_feature_snapshot(self, snapshot: Dict) -> None:
        """
        Restore feature values from a snapshot.

        Used to restore features from the day BEFORE pattern exit,
        ensuring no data leakage from the exit day.

        Args:
            snapshot: Dictionary with feature values to restore
        """
        self.days_since_activation = snapshot['days_since_activation']
        self.total_days_in_pattern = snapshot['total_days_in_pattern']

        # Recent window features
        self.avg_bbw_20d = snapshot['avg_bbw_20d']
        self.avg_adx_20d = snapshot['avg_adx_20d']
        self.avg_volume_ratio_20d = snapshot['avg_volume_ratio_20d']
        self.avg_range_20d = snapshot['avg_range_20d']
        self.bbw_slope_20d = snapshot['bbw_slope_20d']
        self.adx_slope_20d = snapshot['adx_slope_20d']
        self.volume_slope_20d = snapshot['volume_slope_20d']
        self.range_slope_20d = snapshot['range_slope_20d']
        self.bbw_std_20d = snapshot['bbw_std_20d']
        self.volume_std_20d = snapshot['volume_std_20d']
        self.price_volatility_20d = snapshot['price_volatility_20d']
        self.volume_slope_20d = snapshot['volume_slope_20d']

        # Baseline window features
        self.baseline_bbw_avg = snapshot['baseline_bbw_avg']
        self.baseline_bbw_std = snapshot['baseline_bbw_std']
        self.baseline_volume_avg = snapshot['baseline_volume_avg']
        self.baseline_volatility = snapshot['baseline_volatility']
        self.baseline_range_avg = snapshot['baseline_range_avg']

        # Compression ratios
        self.bbw_compression_ratio = snapshot['bbw_compression_ratio']
        self.volume_compression_ratio = snapshot['volume_compression_ratio']
        self.volatility_stability_ratio = snapshot['volatility_stability_ratio']
        self.range_compression_ratio = snapshot['range_compression_ratio']
        self.overall_compression = snapshot['overall_compression']

        # Price position
        self.price_position_in_range = snapshot['price_position_in_range']
        self.price_distance_from_upper_pct = snapshot['price_distance_from_upper_pct']
        self.price_distance_from_lower_pct = snapshot['price_distance_from_lower_pct']

    def to_training_samples(self) -> List[Dict]:
        """
        Convert pattern to multiple training samples (one per feature snapshot).

        Each snapshot becomes a separate training sample with its own outcome
        evaluated from snapshot_date + 100 days (not pattern end_date).
        This prevents position bias and creates proportional training data
        (longer patterns contribute more samples).

        Returns:
            List of dictionaries, one per snapshot, each with features only
            (outcomes calculated during labeling phase)
        """
        training_samples = []

        # If no snapshots collected (e.g., pattern too short), use final features
        if not self.feature_snapshots:
            # Fallback: single sample with final features
            training_samples.append(self.to_dict())
            return training_samples

        # Create one training sample per snapshot
        for snapshot in self.feature_snapshots:
            sample = {
                # Pattern identification
                'ticker': self.ticker,
                'pattern_id': f"{self.ticker}_{self.start_date}",
                'start_date': self.start_date,
                'end_date': self.end_date,
                'snapshot_date': snapshot.get('snapshot_date'),

                # Pattern boundaries (FIXED)
                'upper_boundary': self.upper_boundary,
                'lower_boundary': self.lower_boundary,
                'power_boundary': self.power_boundary,

                # Temporal (from snapshot)
                'days_since_activation': snapshot['days_since_activation'],
                # NOTE: total_days_in_pattern is NOT included (leaked)

                # Recent window features (from snapshot)
                'avg_bbw_20d': snapshot['avg_bbw_20d'],
                'avg_adx_20d': snapshot['avg_adx_20d'],
                'avg_volume_ratio_20d': snapshot['avg_volume_ratio_20d'],
                'avg_range_20d': snapshot['avg_range_20d'],
                'bbw_slope_20d': snapshot['bbw_slope_20d'],
                'adx_slope_20d': snapshot['adx_slope_20d'],
                'volume_slope_20d': snapshot['volume_slope_20d'],
                'range_slope_20d': snapshot['range_slope_20d'],
                'bbw_std_20d': snapshot['bbw_std_20d'],
                'volume_std_20d': snapshot['volume_std_20d'],
                'price_volatility_20d': snapshot['price_volatility_20d'],
                'volume_slope_20d': snapshot['volume_slope_20d'],

                # Baseline window features (from snapshot)
                'baseline_bbw_avg': snapshot['baseline_bbw_avg'],
                'baseline_bbw_std': snapshot['baseline_bbw_std'],
                'baseline_volume_avg': snapshot['baseline_volume_avg'],
                'baseline_volatility': snapshot['baseline_volatility'],
                'baseline_range_avg': snapshot['baseline_range_avg'],

                # Compression ratios (from snapshot)
                'bbw_compression_ratio': snapshot['bbw_compression_ratio'],
                'volume_compression_ratio': snapshot['volume_compression_ratio'],
                'volatility_stability_ratio': snapshot['volatility_stability_ratio'],
                'range_compression_ratio': snapshot['range_compression_ratio'],
                'overall_compression': snapshot['overall_compression'],

                # Price position (from snapshot)
                'price_position_in_range': snapshot['price_position_in_range'],
                'price_distance_from_upper_pct': snapshot['price_distance_from_upper_pct'],
                'price_distance_from_lower_pct': snapshot['price_distance_from_lower_pct'],

                # Legacy features
                'avg_bbw': self.avg_bbw,
                'avg_volume_ratio': self.avg_volume_ratio,
                'avg_adx': self.avg_adx,
                'range_compression': self.range_compression

                # NOTE: Outcomes are NOT included here - they will be calculated
                # per-snapshot during labeling, each with its own 100-day window
                # starting from snapshot_date (not pattern end_date)
            }

            training_samples.append(sample)

        return training_samples


class ConsolidationTracker:
    """
    Tracks consolidation patterns for a single ticker using state machine.

    State Transitions:
    NONE → QUALIFYING: Meets consolidation criteria
    QUALIFYING → ACTIVE: 10 days of meeting criteria
    QUALIFYING → NONE: Fails to meet criteria before day 10
    ACTIVE → NONE: Breaks above power boundary (breakout - pattern saved)
    ACTIVE → NONE: Breaks below lower boundary (breakdown - pattern saved)
    ACTIVE → NONE: Exceeds max duration (timeout - pattern saved)

    NOTE: After any pattern completion (breakout/breakdown/timeout), the tracker
    resets to NONE to allow detection of new patterns on the same ticker.
    """

    def __init__(
        self,
        ticker: str,
        bbw_percentile_threshold: float = 0.30,
        adx_threshold: float = 32.0,
        volume_ratio_threshold: float = 0.35,
        range_ratio_threshold: float = 0.65,
        qualifying_days: int = 10,
        max_pattern_days: int = 90
    ):
        """
        Initialize tracker.

        Args:
            ticker: Stock ticker symbol
            bbw_percentile_threshold: BBW percentile threshold (default 0.30)
            adx_threshold: ADX threshold (default 32)
            volume_ratio_threshold: Volume ratio threshold (default 0.35)
            range_ratio_threshold: Range ratio threshold (default 0.65)
            qualifying_days: Days to qualify pattern (default 10)
            max_pattern_days: Max pattern duration (default 90)
        """
        self.ticker = ticker
        self.bbw_percentile_threshold = bbw_percentile_threshold
        self.adx_threshold = adx_threshold
        self.volume_ratio_threshold = volume_ratio_threshold
        self.range_ratio_threshold = range_ratio_threshold
        self.qualifying_days = qualifying_days
        self.max_pattern_days = max_pattern_days

        self.current_pattern: Optional[ConsolidationPattern] = None
        self.completed_patterns = []

        self.phase = PatternPhase.NONE

        # Store full DataFrame for feature extraction (set via set_data())
        self._full_data: Optional[pd.DataFrame] = None

    def set_data(self, df: pd.DataFrame) -> None:
        """
        Set the full DataFrame for feature extraction.

        This must be called before processing daily updates to enable
        dynamic feature extraction that looks back 50 days.

        Args:
            df: Full DataFrame with price data and indicators
        """
        self._full_data = df

    def update(
        self,
        date: datetime,
        idx: int,
        price_data: Dict[str, float],
        indicators: Dict[str, float]
    ) -> Optional[str]:
        """
        Update tracker with new bar (called day-by-day).

        Args:
            date: Current date
            idx: Current index in data
            price_data: Dict with OHLCV values
            indicators: Dict with technical indicators

        Returns:
            Event string if pattern state changed, None otherwise
        """
        event = None

        if self.phase == PatternPhase.NONE:
            event = self._check_start_qualification(date, idx, price_data, indicators)

        elif self.phase == PatternPhase.QUALIFYING:
            event = self._check_qualification_progress(date, idx, price_data, indicators)

        elif self.phase == PatternPhase.ACTIVE:
            event = self._check_active_pattern(date, idx, price_data, indicators)

        return event

    def _check_start_qualification(
        self,
        date: datetime,
        idx: int,
        price_data: Dict[str, float],
        indicators: Dict[str, float]
    ) -> Optional[str]:
        """Check if consolidation pattern is starting."""
        if self._meets_consolidation_criteria(indicators):
            # Start new pattern
            self.current_pattern = ConsolidationPattern(
                ticker=self.ticker,
                start_date=date,
                start_idx=idx,
                start_price=price_data['close']
            )
            self.phase = PatternPhase.QUALIFYING
            logger.debug(f"{self.ticker}: Started QUALIFYING at {date}")
            return "QUALIFICATION_STARTED"

        return None

    def _check_qualification_progress(
        self,
        date: datetime,
        idx: int,
        price_data: Dict[str, float],
        indicators: Dict[str, float]
    ) -> Optional[str]:
        """Check qualification progress."""
        if not self.current_pattern:
            return None

        self.current_pattern.days_qualifying += 1

        # Track highs/lows during qualification period
        self.current_pattern.qualification_highs.append(price_data['high'])
        self.current_pattern.qualification_lows.append(price_data['low'])

        # Check if still meets criteria
        if not self._meets_consolidation_criteria(indicators):
            # Failed qualification
            logger.debug(f"{self.ticker}: Failed QUALIFYING at {date} " +
                        f"(day {self.current_pattern.days_qualifying})")
            self.current_pattern = None
            self.phase = PatternPhase.NONE
            return "QUALIFICATION_FAILED"

        # Check if qualified (10 days)
        if self.current_pattern.days_qualifying >= self.qualifying_days:
            # Transition to ACTIVE
            self._activate_pattern(date, idx, price_data, indicators)
            self.phase = PatternPhase.ACTIVE
            logger.info(f"{self.ticker}: Pattern ACTIVE at {date}")
            return "PATTERN_ACTIVATED"

        return None

    def _activate_pattern(
        self,
        date: datetime,
        idx: int,
        price_data: Dict[str, float],
        indicators: Dict[str, float]
    ) -> None:
        """Activate pattern after qualification."""
        if not self.current_pattern:
            return

        # Set boundaries from ENTIRE qualification period (10 days)
        # These boundaries are FIXED once set and never change during ACTIVE phase
        self.current_pattern.upper_boundary = max(self.current_pattern.qualification_highs)
        self.current_pattern.lower_boundary = min(self.current_pattern.qualification_lows)
        self.current_pattern.power_boundary = self.current_pattern.upper_boundary * 1.005

        # Calculate average metrics during qualification
        self.current_pattern.avg_bbw = indicators.get('bbw_20', 0)
        self.current_pattern.avg_volume_ratio = indicators.get('volume_ratio_20', 0)
        self.current_pattern.avg_adx = indicators.get('adx', 0)

        # Set activation date and index (FIX: was missing, causing all snapshots to have days_since_activation=0)
        self.current_pattern.activation_date = date
        self.current_pattern.activation_idx = idx

        # Initialize leak-free temporal tracking
        self.current_pattern.days_since_activation = 0  # Leak-free: always known
        self.current_pattern.total_days_in_pattern = self.current_pattern.days_qualifying  # Leaked: for analysis only

        # Extract initial dynamic features (if full data available)
        if self._full_data is not None:
            indicators_with_close = dict(indicators)
            indicators_with_close['close'] = price_data['close']
            self.current_pattern.extract_pattern_features(
                date=date,
                idx=idx,
                price_data=self._full_data,
                indicators=indicators_with_close
            )

    def _check_active_pattern(
        self,
        date: datetime,
        idx: int,
        price_data: Dict[str, float],
        indicators: Dict[str, float]
    ) -> Optional[str]:
        """
        Check active pattern for breakout/breakdown.

        CRITICAL: To prevent data leakage and position bias, we randomly sample
        feature snapshots during the ACTIVE phase. This creates multiple training
        samples per pattern with diverse price positions.
        """
        if not self.current_pattern:
            return None

        # =====================================================================
        # STEP 1: Increment temporal counters for TODAY
        # =====================================================================
        self.current_pattern.days_since_activation += 1  # Leak-free: days since ACTIVE
        self.current_pattern.total_days_in_pattern += 1  # Leaked: total duration (for analysis only)

        # =====================================================================
        # STEP 2: Extract dynamic features for TODAY
        # =====================================================================
        if self._full_data is not None:
            indicators_with_close = dict(indicators)
            indicators_with_close['close'] = price_data['close']
            self.current_pattern.extract_pattern_features(
                date=date,
                idx=idx,
                price_data=self._full_data,
                indicators=indicators_with_close
            )

        # =====================================================================
        # STEP 3: Check exit conditions (with TODAY's price data)
        # =====================================================================
        # Boundaries are FIXED - set once during activation, never updated
        # This ensures we detect true breakouts outside the consolidation range

        # Check for breakout (above power boundary)
        if price_data['close'] > self.current_pattern.power_boundary:
            self._complete_pattern(date, idx, price_data, 'UP')
            # Reset to NONE to allow detection of new patterns
            self.phase = PatternPhase.NONE
            logger.info(f"{self.ticker}: BREAKOUT at {date} - pattern completed, reset to NONE")
            return "BREAKOUT"

        # Check for breakdown (below lower boundary)
        if price_data['close'] < self.current_pattern.lower_boundary:
            self._complete_pattern(date, idx, price_data, 'DOWN')
            # Reset to NONE to allow detection of new patterns
            self.phase = PatternPhase.NONE
            logger.info(f"{self.ticker}: BREAKDOWN at {date} - pattern completed, reset to NONE")
            return "BREAKDOWN"

        # Check for max duration (use total_days_in_pattern for timeout check)
        if self.current_pattern.total_days_in_pattern > self.max_pattern_days:
            self._complete_pattern(date, idx, price_data, 'TIMEOUT')
            self.phase = PatternPhase.NONE
            logger.info(f"{self.ticker}: Pattern TIMEOUT at {date}")
            return "TIMEOUT"

        # =====================================================================
        # STEP 4: Random sampling for training data (pattern continues)
        # =====================================================================
        # Randomly save feature snapshot with probability 0.20 (20% chance)
        # This creates ~1 sample per 5 days on average
        # Longer patterns → more samples (proportional contribution)

        import random
        if random.random() < 0.20:  # 20% sampling probability
            snapshot = self.current_pattern.save_feature_snapshot()
            snapshot['snapshot_date'] = date
            snapshot['snapshot_idx'] = idx
            self.current_pattern.feature_snapshots.append(snapshot)
            logger.debug(f"{self.ticker}: Saved feature snapshot on day {self.current_pattern.days_since_activation}")

        return None

    def _complete_pattern(
        self,
        date: datetime,
        idx: int,
        price_data: Dict[str, float],
        direction: str
    ) -> None:
        """Complete current pattern."""
        if not self.current_pattern:
            return

        self.current_pattern.end_date = date
        self.current_pattern.end_idx = idx
        self.current_pattern.end_price = price_data['close']
        self.current_pattern.breakout_date = date
        self.current_pattern.breakout_direction = direction

        # Calculate max gain
        gain = (self.current_pattern.end_price - self.current_pattern.start_price) / self.current_pattern.start_price
        self.current_pattern.max_gain = gain

        # Store completed pattern
        self.completed_patterns.append(self.current_pattern)
        self.current_pattern = None

    def _meets_consolidation_criteria(self, indicators: Dict[str, float]) -> bool:
        """
        Check if current bar meets consolidation criteria.

        Criteria (BBW MANDATORY + 2 of 3 others):
        - BBW percentile < threshold (volatility contraction) - REQUIRED
        - ADX < threshold (low trending)
        - Volume < threshold (low volume - typical of consolidations)
        - Daily range < threshold (tight trading range)

        CRITICAL FIX (2025-11-03): BBW is now mandatory to ensure all patterns
        have compressed volatility. The old "3 of 4" rule allowed patterns to
        qualify without BBW compression, resulting in validation patterns with
        mean BBW of 119.8 (vs training 12.3) - these were NOT real consolidations.
        """
        bbw_percentile = indicators.get('bbw_percentile', 1.0)
        adx = indicators.get('adx', 100.0)
        volume_ratio = indicators.get('volume_ratio_20', 1.0)
        range_ratio = indicators.get('range_ratio', 1.0)

        # BBW compression is MANDATORY
        bbw_met = bbw_percentile < self.bbw_percentile_threshold

        # Check other criteria
        other_criteria_met = [
            adx < self.adx_threshold,
            volume_ratio < self.volume_ratio_threshold,
            range_ratio < self.range_ratio_threshold
        ]

        # Require BBW + at least 2 of the other 3 criteria
        num_other_criteria_met = sum(other_criteria_met)
        return bbw_met and (num_other_criteria_met >= 2)

    def get_current_pattern(self) -> Optional[ConsolidationPattern]:
        """Get current active pattern."""
        return self.current_pattern

    def get_completed_patterns(self) -> list:
        """Get all completed patterns."""
        return self.completed_patterns

    def get_state(self) -> Dict:
        """Get current state."""
        return {
            'ticker': self.ticker,
            'phase': self.phase.value,
            'current_pattern': self.current_pattern.to_dict() if self.current_pattern else None,
            'num_completed_patterns': len(self.completed_patterns)
        }
