"""
Temporal Training Data Generator - ML Sample Generation
=======================================================

Generates training samples from completed patterns.
Ensures temporal integrity by only creating labels after outcomes are known.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import random
import logging
from .pattern_state import PatternState, PatternPhase

logger = logging.getLogger(__name__)


class TemporalTrainingDataGenerator:
    """
    Generate training samples with temporal safety.

    Training samples are created ONLY after pattern completion
    to ensure labels are based on actual outcomes (no look-ahead).
    """

    def __init__(
        self,
        samples_per_pattern_day: float = 0.5,
        min_samples: int = 5,
        max_samples: int = 30,
        random_seed: Optional[int] = None
    ):
        """
        Initialize training data generator.

        Args:
            samples_per_pattern_day: Samples per day pattern was active
            min_samples: Minimum samples per pattern
            max_samples: Maximum samples per pattern
            random_seed: Random seed for reproducibility
        """
        self.samples_per_pattern_day = samples_per_pattern_day
        self.min_samples = min_samples
        self.max_samples = max_samples

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def generate_samples(
        self,
        pattern: PatternState,
        only_after_completion: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate training samples from pattern.

        CRITICAL: Samples are only generated after pattern completes
        to ensure we know the actual outcome for labeling.

        Args:
            pattern: Completed pattern state
            only_after_completion: Only generate if pattern is complete

        Returns:
            List of training samples with features and labels
        """
        # Temporal safety: Only generate samples after completion
        if only_after_completion and not pattern.is_terminal():
            logger.debug(
                f"Pattern not complete ({pattern.phase.value}), "
                "no training samples generated"
            )
            return []

        # Must have feature snapshots from ACTIVE phase
        if not pattern.feature_snapshots:
            logger.warning(
                f"No feature snapshots available for {pattern.ticker}"
            )
            return []

        # Determine number of samples to generate
        num_samples = self._calculate_num_samples(pattern)

        # Select snapshots for training
        selected_snapshots = self._select_snapshots(
            pattern.feature_snapshots,
            num_samples
        )

        # Generate samples with labels
        samples = []
        for snapshot in selected_snapshots:
            sample = self._create_training_sample(pattern, snapshot)
            if sample is not None:
                samples.append(sample)

        logger.info(
            f"Generated {len(samples)} training samples from {pattern.ticker} "
            f"pattern ({pattern.phase.value})"
        )

        return samples

    def _calculate_num_samples(self, pattern: PatternState) -> int:
        """
        Calculate number of samples to generate.

        Longer patterns contribute more samples (proportional representation).

        Args:
            pattern: Pattern state

        Returns:
            Number of samples to generate
        """
        # Base on days pattern was active
        days_active = pattern.days_since_activation

        # Calculate target samples
        target_samples = int(days_active * self.samples_per_pattern_day)

        # Apply bounds
        num_samples = max(self.min_samples, min(self.max_samples, target_samples))

        # Ensure we don't exceed available snapshots
        num_samples = min(num_samples, len(pattern.feature_snapshots))

        return num_samples

    def _select_snapshots(
        self,
        snapshots: List[Dict[str, Any]],
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """
        Select snapshots for training.

        Uses random sampling for diversity.

        Args:
            snapshots: Available feature snapshots
            num_samples: Number to select

        Returns:
            Selected snapshots
        """
        if len(snapshots) <= num_samples:
            # Use all snapshots
            return snapshots

        # Random sample without replacement
        return random.sample(snapshots, num_samples)

    def _create_training_sample(
        self,
        pattern: PatternState,
        snapshot: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create single training sample with features and label.

        Args:
            pattern: Completed pattern
            snapshot: Feature snapshot from ACTIVE phase

        Returns:
            Training sample dictionary
        """
        # Validate pattern has outcome information
        if pattern.breakout_direction is None:
            logger.warning("Pattern missing breakout direction, skipping sample")
            return None

        # Create sample
        sample = {
            # Metadata (not for training)
            'ticker': pattern.ticker,
            'pattern_start_date': pattern.start_date.isoformat()
                if pattern.start_date else None,
            'pattern_end_date': pattern.end_date.isoformat()
                if pattern.end_date else None,
            'snapshot_date': snapshot.get('snapshot_date'),

            # Features (for training)
            'features': self._extract_training_features(snapshot),

            # Label (based on outcome)
            'label': self._determine_label(pattern),

            # Additional outcome info (for analysis)
            'outcome_direction': pattern.breakout_direction,
            'days_to_outcome': pattern.days_since_activation,
            'boundary_range_pct': pattern.boundary_range_pct
        }

        return sample

    def _extract_training_features(
        self,
        snapshot: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract features for training from snapshot.

        Args:
            snapshot: Feature snapshot

        Returns:
            Training feature dictionary
        """
        # Core features for training
        training_features = {
            # Recent window features
            'avg_bbw_20d': snapshot.get('avg_bbw_20d', 0.0),
            'bbw_std_20d': snapshot.get('bbw_std_20d', 0.0),
            'bbw_slope_20d': snapshot.get('bbw_slope_20d', 0.0),
            'avg_adx_20d': snapshot.get('avg_adx_20d', 0.0),
            'adx_slope_20d': snapshot.get('adx_slope_20d', 0.0),
            'avg_volume_ratio_20d': snapshot.get('avg_volume_ratio_20d', 0.0),
            'volume_std_20d': snapshot.get('volume_std_20d', 0.0),
            'volume_slope_20d': snapshot.get('volume_slope_20d', 0.0),
            'avg_range_ratio_20d': snapshot.get('avg_range_ratio_20d', 0.0),
            'range_slope_20d': snapshot.get('range_slope_20d', 0.0),
            'price_volatility_20d': snapshot.get('price_volatility_20d', 0.0),

            # Compression metrics
            'bbw_compression_ratio': snapshot.get('bbw_compression_ratio', 1.0),
            'volume_compression_ratio': snapshot.get('volume_compression_ratio', 1.0),
            'range_compression_ratio': snapshot.get('range_compression_ratio', 1.0),
            'volatility_stability_ratio': snapshot.get('volatility_stability_ratio', 1.0),
            'overall_compression': snapshot.get('overall_compression', 1.0),

            # Price position
            'price_position_in_range': snapshot.get('price_position_in_range', 0.5),
            'price_distance_from_upper_pct': snapshot.get('price_distance_from_upper_pct', 0.0),
            'price_distance_from_lower_pct': snapshot.get('price_distance_from_lower_pct', 0.0),

            # Pattern duration
            'days_since_activation': snapshot.get('days_since_activation', 0)
        }

        return training_features

    def _determine_label(self, pattern: PatternState) -> str:
        """
        Determine training label based on pattern outcome.

        Args:
            pattern: Completed pattern

        Returns:
            Label string ('BREAKOUT', 'BREAKDOWN', etc.)
        """
        # Based on breakout direction
        if pattern.breakout_direction == 'UP':
            return 'BREAKOUT'
        elif pattern.breakout_direction == 'DOWN':
            return 'BREAKDOWN'
        else:
            # Should not happen for completed patterns
            return 'UNKNOWN'

    def create_batch_samples(
        self,
        patterns: List[PatternState],
        only_completed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate training samples from multiple patterns.

        Args:
            patterns: List of pattern states
            only_completed: Only use completed patterns

        Returns:
            Combined list of training samples
        """
        all_samples = []

        for pattern in patterns:
            if only_completed and not pattern.is_terminal():
                continue

            samples = self.generate_samples(pattern, only_completed)
            all_samples.extend(samples)

        logger.info(
            f"Generated {len(all_samples)} total samples from "
            f"{len(patterns)} patterns"
        )

        return all_samples

    def to_dataframe(
        self,
        samples: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Convert samples to DataFrame format for ML.

        Args:
            samples: List of training samples

        Returns:
            Tuple of (features_df, labels_series)
        """
        if not samples:
            return pd.DataFrame(), pd.Series(dtype=str)

        # Extract features and labels
        features_list = []
        labels_list = []

        for sample in samples:
            features_list.append(sample['features'])
            labels_list.append(sample['label'])

        # Create DataFrames
        features_df = pd.DataFrame(features_list)
        labels_series = pd.Series(labels_list)

        return features_df, labels_series

    def validate_temporal_integrity(
        self,
        pattern: PatternState,
        sample: Dict[str, Any]
    ) -> bool:
        """
        Validate that sample maintains temporal integrity.

        Args:
            pattern: Pattern state
            sample: Training sample

        Returns:
            True if temporally valid
        """
        # Check pattern is complete
        if not pattern.is_terminal():
            logger.error(
                "Temporal violation: Creating training sample from "
                f"incomplete pattern ({pattern.phase.value})"
            )
            return False

        # Check snapshot was taken during ACTIVE phase
        snapshot_date_str = sample.get('snapshot_date')
        if snapshot_date_str and pattern.activated_at and pattern.completed_at:
            snapshot_date = datetime.fromisoformat(snapshot_date_str)

            if snapshot_date < pattern.activated_at:
                logger.error(
                    "Temporal violation: Snapshot before activation"
                )
                return False

            if snapshot_date > pattern.completed_at:
                logger.error(
                    "Temporal violation: Snapshot after completion"
                )
                return False

        return True