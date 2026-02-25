"""
Temporal Augmentations for Illiquid Market Microstructure

PHYSICS-AWARE AUGMENTATION POLICY:
- BAN: Additive noise (Gaussian/OU) on volume features
  - Zero Volume = supply exhaustion signal, NOT missing data
  - Adding noise to "0" destroys the scarcity information

- ALLOWED: Time Warping (drop/stretch timesteps)
  - Preserves zero-volume days as they occur in real data
  - Models must handle variable pattern durations

- ALLOWED: Feature Masking
  - Masks entire timesteps or feature groups
  - Forces model to learn robust representations
  - Does NOT interpolate/generate fake values
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Union

# Import TemporalFeatureConfig for feature indices (eliminates hardcoded magic numbers)
from config.temporal_features import TemporalFeatureConfig

# Singleton instance for temporal feature configuration
_TEMPORAL_CONFIG = TemporalFeatureConfig()


class TemporalAugmentation:
    """
    Base class for temporal augmentations.

    All augmentations preserve the physics of illiquid markets:
    - Zero volume remains zero (supply exhaustion signal)
    - Price gaps remain intact (feature, not noise)
    - No synthetic data generation
    """

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying the augmentation
        """
        self.p = p

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single sequence.

        Args:
            sequence: Shape (T, F) where T=timesteps, F=features

        Returns:
            Augmented sequence with same shape
        """
        if np.random.random() > self.p:
            return sequence
        return self._apply(sequence)

    def _apply(self, sequence: np.ndarray) -> np.ndarray:
        """Override in subclasses."""
        raise NotImplementedError


class TimeWarping(TemporalAugmentation):
    """
    Time Warping: Stretch or compress segments of the temporal sequence.

    Physics-preserving properties:
    - Segments with zero volume stay zero (no interpolation)
    - Creates realistic variations in pattern duration
    - Simulates faster/slower consolidation phases

    Implementation:
    - Select random anchor points
    - Stretch/compress segments between anchors
    - Resample to original length using nearest-neighbor (not interpolation!)
    """

    def __init__(
        self,
        p: float = 0.5,
        warp_factor_range: Tuple[float, float] = (0.8, 1.2),
        n_anchors: int = 3
    ):
        """
        Args:
            p: Probability of applying warping
            warp_factor_range: (min, max) warp factors for segments
            n_anchors: Number of anchor points for warping
        """
        super().__init__(p)
        self.warp_factor_range = warp_factor_range
        self.n_anchors = n_anchors

    def _apply(self, sequence: np.ndarray) -> np.ndarray:
        """Apply time warping.

        Uses nearest-neighbor resampling to preserve discrete values
        (especially zero volumes).
        """
        T, F = sequence.shape

        # Create anchor points (including start and end)
        anchors = np.sort(np.random.choice(range(1, T - 1), size=self.n_anchors, replace=False))
        anchors = np.concatenate([[0], anchors, [T - 1]])

        # Generate warp factors for each segment
        n_segments = len(anchors) - 1
        warp_factors = np.random.uniform(
            self.warp_factor_range[0],
            self.warp_factor_range[1],
            size=n_segments
        )

        # Build warped time indices
        warped_times = []
        for i in range(n_segments):
            start_t = anchors[i]
            end_t = anchors[i + 1]
            segment_len = end_t - start_t

            # Warped length for this segment
            warped_len = int(segment_len * warp_factors[i])
            warped_len = max(1, warped_len)  # At least 1 timestep

            # Original indices for this segment
            original_indices = np.linspace(start_t, end_t, warped_len, endpoint=False)
            warped_times.extend(original_indices)

        # Append final timestep
        warped_times.append(T - 1)
        warped_times = np.array(warped_times)

        # Resample to original length using NEAREST NEIGHBOR (physics-preserving)
        # This ensures zero-volume stays zero, no interpolation artifacts
        target_times = np.linspace(0, len(warped_times) - 1, T)
        nearest_indices = np.round(target_times).astype(int)
        nearest_indices = np.clip(nearest_indices, 0, len(warped_times) - 1)

        # Map back to original sequence indices
        original_indices = np.round(warped_times[nearest_indices]).astype(int)
        original_indices = np.clip(original_indices, 0, T - 1)

        return sequence[original_indices]


class TimestepDropout(TemporalAugmentation):
    """
    Timestep Dropout: Randomly drop timesteps and contract sequence.

    Physics-preserving properties:
    - Simulates missing trading days (holidays, halts)
    - No fake data generated for dropped days
    - Models must handle irregular sampling

    Implementation:
    - Randomly select timesteps to drop
    - Contract sequence (shift remaining timesteps)
    - Pad with last valid value to maintain shape
    """

    def __init__(
        self,
        p: float = 0.5,
        drop_rate: float = 0.1,
        max_consecutive: int = 2
    ):
        """
        Args:
            p: Probability of applying dropout
            drop_rate: Fraction of timesteps to drop
            max_consecutive: Maximum consecutive timesteps to drop
        """
        super().__init__(p)
        self.drop_rate = drop_rate
        self.max_consecutive = max_consecutive

    def _apply(self, sequence: np.ndarray) -> np.ndarray:
        """Apply timestep dropout."""
        T, F = sequence.shape
        n_drop = int(T * self.drop_rate)

        if n_drop == 0:
            return sequence

        # Select timesteps to drop (avoid first and last)
        drop_candidates = list(range(1, T - 1))
        np.random.shuffle(drop_candidates)

        dropped = set()
        for t in drop_candidates:
            if len(dropped) >= n_drop:
                break

            # Check consecutive constraint
            consecutive = sum(1 for i in range(t - self.max_consecutive, t + self.max_consecutive + 1)
                            if i in dropped)
            if consecutive < self.max_consecutive:
                dropped.add(t)

        # Keep non-dropped timesteps
        keep_indices = [t for t in range(T) if t not in dropped]
        contracted = sequence[keep_indices]

        # Pad to original length by repeating last value
        if len(contracted) < T:
            padding = np.repeat(contracted[-1:], T - len(contracted), axis=0)
            contracted = np.concatenate([contracted, padding], axis=0)

        return contracted


class FeatureMasking(TemporalAugmentation):
    """
    Feature Masking: Mask specific feature groups or timesteps.

    Physics-preserving properties:
    - Masks to zero or neutral value (not random noise)
    - Forces model to rely on redundant signals
    - Simulates data quality issues in illiquid markets

    Mask strategies:
    - timestep_mask: Mask entire timesteps
    - feature_group_mask: Mask specific feature groups
    - random_mask: Random rectangular patches

    Feature indices are loaded from TemporalFeatureConfig (single source of truth).
    """

    # Feature group definitions - loaded from TemporalFeatureConfig
    FEATURE_GROUPS = {
        'ohlc': _TEMPORAL_CONFIG.market_feature_indices[:4],  # Open, High, Low, Close
        'volume': [_TEMPORAL_CONFIG.market_feature_indices[4]],  # Volume (NEVER add noise!)
        'technical': _TEMPORAL_CONFIG.technical_feature_indices,  # BBW, ADX, Volume Ratio
        'boundary': _TEMPORAL_CONFIG.boundary_feature_indices  # Upper/Lower slopes
    }

    # Volume features - PROTECTED from noise augmentation
    # Uses TemporalFeatureConfig for single source of truth
    VOLUME_FEATURES = _TEMPORAL_CONFIG.volume_group_indices  # Raw volume and volume ratio

    def __init__(
        self,
        p: float = 0.5,
        mask_ratio: float = 0.15,
        mask_value: float = 0.0,
        mask_strategy: str = 'random',
        protect_volume: bool = True
    ):
        """
        Args:
            p: Probability of applying masking
            mask_ratio: Fraction of values to mask
            mask_value: Value to use for masked positions
            mask_strategy: 'random', 'timestep', 'feature_group'
            protect_volume: If True, never mask volume features
        """
        super().__init__(p)
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.mask_strategy = mask_strategy
        self.protect_volume = protect_volume

    def _apply(self, sequence: np.ndarray) -> np.ndarray:
        """Apply feature masking."""
        T, F = sequence.shape
        masked = sequence.copy()

        if self.mask_strategy == 'timestep':
            # Mask entire timesteps
            n_mask = max(1, int(T * self.mask_ratio))
            mask_t = np.random.choice(T, size=n_mask, replace=False)

            if self.protect_volume:
                # Mask non-volume features only
                non_volume = [i for i in range(F) if i not in self.VOLUME_FEATURES]
                masked[mask_t][:, non_volume] = self.mask_value
            else:
                masked[mask_t] = self.mask_value

        elif self.mask_strategy == 'feature_group':
            # Mask entire feature groups
            maskable_groups = list(self.FEATURE_GROUPS.keys())
            if self.protect_volume:
                maskable_groups = [g for g in maskable_groups if g not in ['volume']]

            n_groups = max(1, int(len(maskable_groups) * self.mask_ratio))
            groups_to_mask = np.random.choice(maskable_groups, size=n_groups, replace=False)

            for group in groups_to_mask:
                feature_indices = self.FEATURE_GROUPS[group]
                if self.protect_volume:
                    feature_indices = [i for i in feature_indices if i not in self.VOLUME_FEATURES]
                masked[:, feature_indices] = self.mask_value

        else:  # random
            # Random masking with protected volume features
            n_mask = int(T * F * self.mask_ratio)

            # Create mask candidates
            if self.protect_volume:
                maskable_features = [i for i in range(F) if i not in self.VOLUME_FEATURES]
            else:
                maskable_features = list(range(F))

            if maskable_features:
                # Random positions to mask
                mask_t = np.random.randint(0, T, size=n_mask)
                mask_f = np.random.choice(maskable_features, size=n_mask)
                masked[mask_t, mask_f] = self.mask_value

        return masked


class CutMix(TemporalAugmentation):
    """
    CutMix for sequences: Swap segments between two sequences.

    Physics-preserving properties:
    - Swaps actual data, no synthetic generation
    - Creates realistic pattern variations
    - Label mixing reflects actual segment proportions

    NOTE: Requires batch-level application (two sequences needed).
    When applied to single sequence, acts as segment shuffle.
    """

    def __init__(
        self,
        p: float = 0.5,
        cut_ratio_range: Tuple[float, float] = (0.2, 0.5)
    ):
        """
        Args:
            p: Probability of applying CutMix
            cut_ratio_range: (min, max) fraction of sequence to swap
        """
        super().__init__(p)
        self.cut_ratio_range = cut_ratio_range

    def _apply(self, sequence: np.ndarray) -> np.ndarray:
        """Single-sequence fallback: segment shuffle."""
        T, F = sequence.shape

        cut_ratio = np.random.uniform(*self.cut_ratio_range)
        cut_len = int(T * cut_ratio)

        # Random cut positions
        cut_start = np.random.randint(0, T - cut_len)
        cut_end = cut_start + cut_len

        # Shift cut segment to beginning or end
        if np.random.random() < 0.5:
            # Move to beginning
            result = np.concatenate([
                sequence[cut_start:cut_end],
                sequence[:cut_start],
                sequence[cut_end:]
            ], axis=0)
        else:
            # Move to end
            result = np.concatenate([
                sequence[:cut_start],
                sequence[cut_end:],
                sequence[cut_start:cut_end]
            ], axis=0)

        return result

    def apply_batch(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray,
        label1: int,
        label2: int
    ) -> Tuple[np.ndarray, float]:
        """
        Apply CutMix between two sequences.

        Args:
            seq1, seq2: Sequences shape (T, F)
            label1, label2: Integer labels

        Returns:
            mixed_sequence: CutMix result
            label_weight: Weight for label1 (1 - weight for label2)
        """
        if np.random.random() > self.p:
            return seq1, 1.0

        T, F = seq1.shape

        cut_ratio = np.random.uniform(*self.cut_ratio_range)
        cut_len = int(T * cut_ratio)
        cut_start = np.random.randint(0, T - cut_len)
        cut_end = cut_start + cut_len

        # Mix sequences
        mixed = seq1.copy()
        mixed[cut_start:cut_end] = seq2[cut_start:cut_end]

        # Label weight proportional to seq1's contribution
        label_weight = 1.0 - cut_ratio

        return mixed, label_weight


class Compose:
    """Compose multiple augmentations."""

    def __init__(self, augmentations: List[TemporalAugmentation]):
        self.augmentations = augmentations

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        for aug in self.augmentations:
            sequence = aug(sequence)
        return sequence


class PhysicsAwareAugmentor:
    """
    High-level augmentation API that enforces physics constraints.

    BANNED operations:
    - Additive Gaussian/OU noise on volume features
    - Interpolation that creates fake volume values

    ALLOWED operations:
    - Time Warping (stretch/compress)
    - Feature Masking (to zero/neutral)
    - Timestep Dropout (drop, not interpolate)
    - CutMix (swap real segments)
    """

    def __init__(
        self,
        time_warp_p: float = 0.3,
        time_warp_range: Tuple[float, float] = (0.8, 1.2),
        timestep_dropout_p: float = 0.2,
        timestep_dropout_rate: float = 0.1,
        feature_mask_p: float = 0.2,
        feature_mask_ratio: float = 0.15,
        protect_volume: bool = True  # Always True in production
    ):
        """
        Create physics-aware augmentor.

        Args:
            time_warp_p: Probability of time warping
            time_warp_range: (min, max) warp factors
            timestep_dropout_p: Probability of timestep dropout
            timestep_dropout_rate: Fraction of timesteps to drop
            feature_mask_p: Probability of feature masking
            feature_mask_ratio: Fraction of features to mask
            protect_volume: ALWAYS True - prevents noise on volume
        """
        self.protect_volume = protect_volume

        augmentations = []

        if time_warp_p > 0:
            augmentations.append(TimeWarping(
                p=time_warp_p,
                warp_factor_range=time_warp_range
            ))

        if timestep_dropout_p > 0:
            augmentations.append(TimestepDropout(
                p=timestep_dropout_p,
                drop_rate=timestep_dropout_rate
            ))

        if feature_mask_p > 0:
            augmentations.append(FeatureMasking(
                p=feature_mask_p,
                mask_ratio=feature_mask_ratio,
                protect_volume=protect_volume
            ))

        self.augmentations = Compose(augmentations) if augmentations else None

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """Apply physics-aware augmentation."""
        if self.augmentations is None:
            return sequence
        return self.augmentations(sequence)

    @staticmethod
    def get_default_config() -> Dict:
        """Get default production configuration."""
        return {
            'time_warp_p': 0.3,
            'time_warp_range': (0.8, 1.2),
            'timestep_dropout_p': 0.2,
            'timestep_dropout_rate': 0.1,
            'feature_mask_p': 0.2,
            'feature_mask_ratio': 0.15,
            'protect_volume': True  # NEVER change to False
        }


# =============================================================================
# BANNED AUGMENTATIONS (DO NOT USE)
# =============================================================================
# The following are intentionally NOT implemented:
#
# class GaussianNoise(TemporalAugmentation):
#     """BANNED: Destroys zero-volume supply exhaustion signal."""
#     pass
#
# class OUProcessNoise(TemporalAugmentation):
#     """BANNED: Ornstein-Uhlenbeck noise destroys volume semantics."""
#     pass
#
# class LinearInterpolation(TemporalAugmentation):
#     """BANNED: Creates fake volume values where none existed."""
#     pass
# =============================================================================
