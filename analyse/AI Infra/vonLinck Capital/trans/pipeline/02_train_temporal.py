import os
import sys

# CRITICAL: Disable PyTorch dynamo/inductor BEFORE importing torch
# Fixes PyTorch 2.7.x bug: "cannot import name 'check_supported_striding'"
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import argparse
import logging
import platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Model and Loss
from models.temporal_hybrid_unified import HybridFeatureNetwork, create_v20_model  # Unified V18+V20
from models.asymmetric_loss import AsymmetricLoss  # <--- Added Import
from losses.coil_focal_loss import (  # Jan 2026
    CoilAwareFocalLoss,
    AsymmetricCoilFocalLoss,
    RankMatchCoilLoss,
    get_coil_intensity_from_context,
    get_volume_shock_from_context  # Still used for metrics tracking
)
from config.constants import USE_GRN_CONTEXT, USE_PROBABILITY_CALIBRATION, STRATEGIC_VALUES
from config.context_features import (
    NUM_CONTEXT_FEATURES,
    CONTEXT_FEATURES,
    CONTEXT_FEATURE_DEFAULTS,
    CONTEXT_FEATURE_RANGES,
    VOLUME_RATIO_INDICES,  # Centralized - no more hardcoding!
    log_diff_batch  # For future use in feature extraction
)
from utils.temporal_split import (
    detect_trinity_mode,
    split_by_clusters,
    validate_cluster_split,
    get_patterns_for_clusters
)
from utils.data_integrity import (
    DataIntegrityChecker,
    DataIntegrityError,
    assert_no_duplicates,
    assert_temporal_split
)
from models.inference_wrapper import InferenceWrapper, PREDICTION_TEMPERATURE
from augmentations import PhysicsAwareAugmentor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_temporal.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def apply_window_jitter(seq: np.ndarray, jitter_range: tuple = (-3, 2)) -> np.ndarray:
    """
    Apply random window jittering to a sequence.

    Forces the model to recognize patterns regardless of whether the breakout
    occurs at index 18, 19, or 20 - utilizing RoPE layers correctly.

    Args:
        seq: Shape (T, F) temporal sequence
        jitter_range: (min_jitter, max_jitter) for random shift
            - Negative jitter: window starts earlier, pad at beginning with first value
            - Positive jitter: window starts later, pad at end with last value

    Returns:
        Jittered sequence of same shape (T, F)
    """
    import random
    jitter = random.randint(jitter_range[0], jitter_range[1])
    if jitter == 0:
        return seq

    T = seq.shape[0]  # timesteps (typically 20)

    if jitter < 0:
        # Window starts earlier (e.g., jitter=-3 means conceptually starting at "day -3")
        # We don't have those early days, so pad the beginning with edge values
        # Take seq[0:T-|jitter|] and prepend |jitter| copies of seq[0]
        n_pad = abs(jitter)
        core = seq[0:T-n_pad]  # T-|jitter| elements from start
        pad = np.tile(seq[0:1], (n_pad, 1))  # repeat first row
        return np.concatenate([pad, core], axis=0)
    else:
        # Window starts later (e.g., jitter=2 means starting at "day 2")
        # We'll run past the end, so pad the end with edge values
        # Take seq[jitter:T] and append jitter copies of seq[-1]
        core = seq[jitter:T]  # T-jitter elements from position jitter
        pad = np.tile(seq[-1:], (jitter, 1))  # repeat last row
        return np.concatenate([core, pad], axis=0)


class TemporalSequenceDataset(Dataset):
    """
    Standard in-memory dataset with optional physics-aware augmentation and window jittering.

    Args:
        sequences: Shape (N, T, F) temporal sequences
        labels: Shape (N,) integer labels
        context: Optional shape (N, C) context features
        augmentor: Optional PhysicsAwareAugmentor for training-time augmentation
        use_window_jitter: Enable random window shifting (-3 to +2 days)
        jitter_range: Tuple (min, max) for window jitter (default: (-3, 2))
    """
    def __init__(self, sequences, labels, context=None, augmentor=None,
                 use_window_jitter: bool = False, jitter_range: tuple = (-3, 2)):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.context = torch.FloatTensor(context) if context is not None else None
        self.augmentor = augmentor  # Physics-aware augmentation (TimeWarping, Masking)
        self.use_window_jitter = use_window_jitter
        self.jitter_range = jitter_range

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Convert to numpy for augmentations
        seq_np = seq.numpy()

        # Apply window jittering first (training only)
        if self.use_window_jitter:
            seq_np = apply_window_jitter(seq_np, self.jitter_range)

        # Apply physics-aware augmentation (training only)
        if self.augmentor is not None:
            seq_np = self.augmentor(seq_np)

        # Convert back to tensor
        if self.use_window_jitter or self.augmentor is not None:
            seq = torch.from_numpy(seq_np).float()

        if self.context is not None:
            return seq, self.labels[idx], self.context[idx]
        return seq, self.labels[idx]


class LazyHDF5Dataset(Dataset):
    """
    Memory-efficient dataset that reads from HDF5 on demand.

    Instead of loading all data into memory, this dataset stores the file path
    and indices, only reading specific samples when __getitem__ is called.
    Applies robust scaling (median/IQR) on-the-fly.

    CRITICAL (Jan 2026 Fix): File handle is lazy-opened in __getitem__, NOT __init__.
    Opening h5py.File in __init__ makes the dataset unpickleable, causing DataLoader
    workers to fail silently (GPU starvation, 0% utilization during load).

    Args:
        h5_path: Path to HDF5 file containing 'sequences', 'labels', 'context'
        indices: Array of indices into the HDF5 datasets for this split
        train_median: Median for robust scaling (computed from training set)
        train_iqr: IQR for robust scaling (computed from training set)
        context_median: Median for context robust scaling (computed from training set)
        context_iqr: IQR for context robust scaling (computed from training set)
        context_log1p_indices: Indices of context features to apply log1p (volume ratios)
        augmentor: Optional PhysicsAwareAugmentor for training-time augmentation
        use_window_jitter: Enable random window shifting (-3 to +2 days)
        jitter_range: Tuple (min, max) for window jitter (default: (-3, 2))
    """
    def __init__(self, h5_path: str, indices: np.ndarray,
                 train_median: np.ndarray = None, train_iqr: np.ndarray = None,
                 context_median: np.ndarray = None, context_iqr: np.ndarray = None,
                 context_log1p_indices: list = None,
                 augmentor=None, use_window_jitter: bool = False,
                 jitter_range: tuple = (-3, 2)):
        self.h5_path = h5_path
        self.indices = np.asarray(indices)
        self.train_median = train_median
        self.train_iqr = train_iqr
        self.context_median = context_median
        self.context_iqr = context_iqr
        self.context_log1p_indices = context_log1p_indices or []
        self.augmentor = augmentor  # Physics-aware augmentation (TimeWarping, Masking)
        self.use_window_jitter = use_window_jitter
        self.jitter_range = jitter_range

        # LAZY OPEN: Initialize as None, open in __getitem__ on first access
        # RATIONALE: h5py.File handles cannot be pickled, breaking DataLoader workers
        self.h5_file = None
        self.sequences = None
        self.labels = None
        self.context = None
        self._has_context = None  # Cached flag for context existence

        # Cache length from indices (doesn't require file open)
        self._len = len(self.indices)

    def _ensure_open(self):
        """Lazy-open HDF5 file on first access (per-worker)."""
        if self.h5_file is None:
            import h5py
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.sequences = self.h5_file['sequences']
            self.labels = self.h5_file['labels']
            self._has_context = 'context' in self.h5_file
            self.context = self.h5_file['context'] if self._has_context else None

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # Lazy-open file handle (safe for multiprocessing - each worker opens its own)
        self._ensure_open()

        # Map local index to HDF5 index
        h5_idx = self.indices[idx]

        # Read single sample from HDF5
        seq = self.sequences[h5_idx].astype(np.float32)
        label = int(self.labels[h5_idx])

        # Apply robust scaling on-the-fly (median/IQR from training set)
        if self.train_median is not None and self.train_iqr is not None:
            seq = (seq - self.train_median) / self.train_iqr
            # Squeeze in case broadcasting added dimensions (1, T, F) -> (T, F)
            seq = np.squeeze(seq)

        # Apply window jittering (training only)
        if self.use_window_jitter:
            seq = apply_window_jitter(seq, self.jitter_range)

        # Apply physics-aware augmentation (training only)
        if self.augmentor is not None:
            seq = self.augmentor(seq)

        # Convert to tensors
        seq_tensor = torch.from_numpy(seq)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.context is not None:
            ctx = self.context[h5_idx].astype(np.float32)

            # NOTE (Jan 2026): Log1p transformation is NO LONGER APPLIED here.
            # Volume ratio features are now pre-computed with log_diff() in
            # 01_generate_sequences.py, which already produces log-space values.
            # The context_log1p_indices list should be empty [].
            # Keeping the loop for backwards compatibility with older checkpoints.
            for idx in self.context_log1p_indices:
                if idx < len(ctx):
                    ctx[idx] = np.log1p(max(ctx[idx], 0))

            # Apply robust scaling to context
            if self.context_median is not None and self.context_iqr is not None:
                ctx = (ctx - self.context_median) / self.context_iqr

            ctx_tensor = torch.from_numpy(ctx)
            return seq_tensor, label_tensor, ctx_tensor

        return seq_tensor, label_tensor

    def close(self):
        """Close the HDF5 file handle."""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()

    def __del__(self):
        self.close()


def get_optimal_workers(requested: int) -> int:
    """
    Determine optimal number of DataLoader workers.

    Args:
        requested: User-requested workers (0=auto, -1=disable)

    Returns:
        Optimal worker count for the platform
    """
    if requested == -1:
        return 0  # Explicitly disabled

    if requested > 0:
        return requested  # User specified

    # Auto-detection (requested == 0)
    is_windows = platform.system() == 'Windows'
    if is_windows:
        # Windows has issues with multiprocessing in DataLoader
        # 2 workers is safe, 4+ can cause issues
        return 2
    else:
        # Linux/Mac: Use up to 4 workers
        return min(4, os.cpu_count() or 1)


def train_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # CRITICAL FIX (Jan 2026): Resolve wildcards on Windows FIRST
    # Windows does NOT expand wildcards automatically like Unix shells
    import glob as glob_module
    sequences_path = args.sequences
    if '*' in sequences_path or '?' in sequences_path:
        matched_files = sorted(glob_module.glob(sequences_path))
        if not matched_files:
            raise FileNotFoundError(f"No files matched pattern: {sequences_path}")
        sequences_path = matched_files[-1]  # Use most recent (sorted by name)
        logger.info(f"Wildcard resolved: {args.sequences} -> {sequences_path}")

    # AUTO-DETECT metadata path early (needed for pattern_ids loading)
    import re
    metadata_path = args.metadata
    if not metadata_path and sequences_path.endswith('.h5'):
        seq_basename = os.path.basename(sequences_path)
        timestamp_match = re.search(r'sequences_(\d{8}_\d{6})\.h5', seq_basename)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            auto_metadata_path = os.path.join(
                os.path.dirname(sequences_path),
                f'metadata_{timestamp}.parquet'
            )
            if os.path.exists(auto_metadata_path):
                metadata_path = auto_metadata_path
                logger.info(f"AUTO-DETECTED metadata: {metadata_path}")

    # 1. Load Data (Supports both .npy and .h5 formats)

    logger.info(f"Loading sequences from {sequences_path}")
    try:
        # Check if HDF5 format (produced by --claude-safe or --skip-npy-export)
        if sequences_path.endswith('.h5'):
            import h5py
            logger.info("Loading from HDF5 format (memory-efficient)")
            with h5py.File(sequences_path, 'r') as f:
                sequences = f['sequences'][:]
                labels = f['labels'][:]
                context = f['context'][:] if 'context' in f else None
            logger.info(f"Loaded from HDF5: sequences={sequences.shape}, labels={labels.shape}")
        else:
            # NPY format requires separate labels file
            if not args.labels:
                raise ValueError("--labels is required when using .npy format. "
                                "Use .h5 files (which contain labels) or provide --labels path.")
            sequences = np.load(sequences_path)
            logger.info(f"Loading labels from {args.labels}")
            labels = np.load(args.labels)

        # Load context features for Branch B (GRN)
        # CONTROLLED BY: config.constants.USE_GRN_CONTEXT
        use_context = False
        if not sequences_path.endswith('.h5'):  # Only load separately if not already from HDF5
            context = None
        if not USE_GRN_CONTEXT:
            logger.info("GRN Context Branch DISABLED (USE_GRN_CONTEXT=False) - pure Temporal/Spatial mode")
        elif context is not None:
            logger.info(f"Context shape: {context.shape}")
            use_context = True
        elif args.context and os.path.exists(args.context):
            logger.info(f"Loading context features from {args.context}")
            context = np.load(args.context)
            logger.info(f"Context shape: {context.shape}")
            use_context = True
        else:
            logger.info("No context features provided - Branch B will use neutral defaults")

        # Context feature dimension validation (8-feature standard)
        if use_context and context is not None:
            if context.shape[-1] != NUM_CONTEXT_FEATURES:
                raise ValueError(
                    f"Context has {context.shape[-1]} features, expected {NUM_CONTEXT_FEATURES}. "
                    f"Regenerate sequences with 01_generate_sequences.py to include all 8 context features."
                )
            logger.info(f"[OK] Context validation passed: {context.shape[-1]} features")

        # Apply context feature exclusion (Jan 2026 - remove overfitting features)
        if use_context and context is not None and args.exclude_context_features:
            exclude_indices = [int(x.strip()) for x in args.exclude_context_features.split(',')]
            logger.info(f"Excluding context features at indices: {exclude_indices}")

            # Get default values for excluded features
            for idx in exclude_indices:
                if 0 <= idx < context.shape[-1]:
                    # Set to feature's default value (from CONTEXT_FEATURE_DEFAULTS)
                    feature_name = CONTEXT_FEATURES[idx] if idx < len(CONTEXT_FEATURES) else None
                    default_val = CONTEXT_FEATURE_DEFAULTS.get(feature_name, 0.0) if feature_name else 0.0
                    context[:, idx] = default_val
                    logger.info(f"  Feature {idx} ({feature_name}): set to default {default_val}")
                else:
                    logger.warning(f"  Feature index {idx} out of range (0-{context.shape[-1]-1})")

        # CRITICAL: Load pattern_ids for pattern-aware splitting
        # Prefer metadata (always aligned) over pattern_ids.npy (may be stale)
        sequences_dir = os.path.dirname(sequences_path)
        pattern_ids_path = os.path.join(sequences_dir, 'pattern_ids.npy')
        pattern_ids = None
        n_sequences = len(sequences)

        # Try metadata first (guaranteed to be aligned with sequences)
        if metadata_path and os.path.exists(metadata_path):
            logger.info(f"Loading pattern IDs from metadata")
            meta_temp = pd.read_parquet(metadata_path)
            if 'pattern_id' in meta_temp.columns:
                if len(meta_temp) == n_sequences:
                    pattern_ids = meta_temp['pattern_id'].values
                    logger.info(f"Extracted {len(pattern_ids)} pattern IDs from metadata (aligned)")
                else:
                    logger.warning(f"Metadata has {len(meta_temp)} rows but {n_sequences} sequences")
            else:
                logger.warning("Metadata missing 'pattern_id' column")

        # Fall back to pattern_ids.npy only if metadata didn't work
        if pattern_ids is None and os.path.exists(pattern_ids_path):
            logger.info(f"Trying pattern_ids.npy from {pattern_ids_path}")
            loaded_ids = np.load(pattern_ids_path, allow_pickle=True)
            if len(loaded_ids) == n_sequences:
                pattern_ids = loaded_ids
                logger.info(f"Loaded {len(pattern_ids)} pattern IDs from .npy (aligned)")
            else:
                logger.warning(f"pattern_ids.npy has {len(loaded_ids)} entries but {n_sequences} sequences (stale file)")

        if pattern_ids is None:
            raise FileNotFoundError(
                f"Cannot load aligned pattern_ids. Sequences: {n_sequences}. "
                f"Metadata: {metadata_path}, pattern_ids.npy exists: {os.path.exists(pattern_ids_path)}"
            )

        # Sequence shape validation - Accept (N, 20, 10) or (N, 20, 14) for backward compatibility
        logger.info(f"Loaded sequence shape: {sequences.shape}")
        valid_shapes = [(20, 10), (20, 14)]
        if sequences.shape[1:] not in valid_shapes:
            logger.error(f"Unexpected shape! Expected (N, 20, 10) or (N, 20, 14), got {sequences.shape}")
            logger.error("Run 01_generate_sequences.py to regenerate clean sequences")
            return
        logger.info(f"[OK] Shape validation passed: {sequences.shape}")

        logger.info(f"Loaded {len(sequences)} sequences from {len(np.unique(pattern_ids))} unique patterns")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Load metadata for temporal split (REQUIRED unless --allow-random-split)
    # metadata_path was auto-detected at the start of train_model()
    metadata = None
    if metadata_path:
        logger.info(f"Loading metadata from {metadata_path}")
        metadata = pd.read_parquet(metadata_path)

        # ================================================================
        # MANDATORY DATA INTEGRITY CHECK (Jan 2026)
        # Prevents 74x duplication bug and look-ahead bias
        # ================================================================
        logger.info("=" * 60)
        logger.info("DATA INTEGRITY VALIDATION")
        logger.info("=" * 60)

        checker = DataIntegrityChecker(
            metadata,
            date_col='pattern_end_date' if 'pattern_end_date' in metadata.columns else 'end_date',
            strict=True  # Raise error on critical issues
        )

        # Check 1: Duplication (CRITICAL - caused 74x inflation bug)
        dup_passed, dup_details = checker.check_duplication()
        if dup_passed:
            logger.info(f"[PASS] Duplication: {dup_details['unique_patterns']:,} unique patterns")
        else:
            logger.error(f"[FAIL] Duplication: {dup_details['duplication_ratio']:.1f}x ratio!")
            logger.error(f"       Rows: {dup_details['total_rows']:,}, Unique: {dup_details['unique_patterns']:,}")
            logger.error("       Run deduplication before training!")
            raise DataIntegrityError(
                f"Data has {dup_details['duplication_ratio']:.1f}x duplication. "
                f"Deduplicate metadata before training."
            )

        # Check 2: Statistical power
        power_passed, power_details = checker.check_statistical_power()
        if power_passed:
            logger.info(f"[PASS] Statistical Power: {power_details['target_events']:,} target events")
        else:
            logger.warning(f"[WARN] Low power: only {power_details['target_events']:,} target events")

        # Check 3: Feature leakage (if context features provided)
        if context is not None:
            leakage_cols = ['breakout_class', 'label_3R', 'label_4R', 'max_r_achieved']
            leakage_found = [c for c in leakage_cols if c in metadata.columns]
            if leakage_found:
                logger.warning(f"[WARN] Potential leakage columns in metadata: {leakage_found}")
                logger.warning("       Ensure these are NOT used as features!")

        # Check 4: Metadata-HDF5 Label Alignment (Jan 2026 fix)
        # CRITICAL: Ensures metadata rows are in same order as HDF5 sequences
        # Previously caused all predictions to map to index 0 after deduplication
        label_col = 'outcome_class' if 'outcome_class' in metadata.columns else 'label'
        if label_col in metadata.columns and len(metadata) == len(labels):
            metadata_labels = metadata[label_col].values.astype(int)
            h5_labels = labels.astype(int)
            mismatches = np.sum(metadata_labels != h5_labels)
            if mismatches == 0:
                logger.info(f"[PASS] Metadata-HDF5 Alignment: {len(labels):,} labels match perfectly")
            else:
                mismatch_pct = 100 * mismatches / len(labels)
                logger.error(f"[FAIL] Metadata-HDF5 Alignment: {mismatches:,} mismatches ({mismatch_pct:.1f}%)")
                logger.error("       Metadata and HDF5 are out of sync!")
                logger.error("       Regenerate sequences with 01_generate_sequences.py")
                raise DataIntegrityError(
                    f"Metadata labels do not match HDF5 labels: {mismatches:,} mismatches. "
                    f"Regenerate sequences to fix alignment."
                )
        elif len(metadata) != len(labels):
            logger.error(f"[FAIL] Metadata-HDF5 Alignment: Row count mismatch!")
            logger.error(f"       Metadata: {len(metadata):,} rows, HDF5: {len(labels):,} sequences")
            raise DataIntegrityError(
                f"Metadata has {len(metadata)} rows but HDF5 has {len(labels)} sequences. "
                f"Regenerate sequences to fix alignment."
            )
        else:
            logger.warning(f"[SKIP] Metadata-HDF5 Alignment: '{label_col}' column not found in metadata")

        logger.info("=" * 60)

    # Early-closed pattern filtering (ablation study)
    if getattr(args, 'exclude_early_closed', False) and metadata is not None:
        if 'labeling_method' in metadata.columns:
            before_count = len(metadata)
            early_closed_mask = metadata['labeling_method'] == 'early_closed'
            n_early = early_closed_mask.sum()

            if n_early > 0:
                # Filter metadata to full_window patterns only
                metadata = metadata[~early_closed_mask].reset_index(drop=True)

                # Create mask for sequences (each sequence has pattern_id in metadata)
                valid_pattern_ids = set(metadata['pattern_id'].unique())
                seq_mask = np.array([pid in valid_pattern_ids for pid in pattern_ids])

                # Apply filter to sequences and labels
                sequences = sequences[seq_mask]
                labels = labels[seq_mask]
                pattern_ids = pattern_ids[seq_mask]
                if context is not None:
                    context = context[seq_mask]

                logger.info(f"\n*** EARLY-CLOSED FILTER APPLIED (--exclude-early-closed) ***")
                logger.info(f"  Removed {n_early:,} early-closed patterns")
                logger.info(f"  Remaining: {len(metadata):,} full-window patterns")
                logger.info(f"  Sequences: {before_count} -> {len(sequences)}")
        else:
            logger.warning("--exclude-early-closed: 'labeling_method' column not in metadata, skipping filter")

    # 2. PATTERN-AWARE Train/Val/Test Split
    # CRITICAL: Split by Pattern ID to prevent data leakage
    # Each pattern generates ~11 sequences (sliding windows) - these MUST stay together
    #
    # ENFORCED TEMPORAL SPLIT (2025-12-24):
    # Random split allows overlapping windows from the same event to leak across
    # train/val sets. Temporal split ensures all patterns before cutoff date go to
    # train, and patterns after go to val/test. This prevents data leakage.
    #
    # TRINITY MODE (Jan 2026): When nms_cluster_id is present, split by CLUSTER
    # instead of pattern_id. This ensures all views (Entry/Coil/Trigger) of the
    # same consolidation event stay together in the same split.

    # Detect Trinity mode: nms_cluster_id present and not all -1
    # Also enabled by --cluster-aware-split flag
    # Can be DISABLED by --disable-trinity flag (for ablation studies)
    use_trinity_split = False
    cluster_ids = None
    force_cluster_split = getattr(args, 'cluster_aware_split', False)
    disable_trinity = getattr(args, 'disable_trinity', False)

    if disable_trinity:
        logger.info("\n*** TRINITY MODE DISABLED (--disable-trinity) ***")
        logger.info("  Using pattern-level temporal splitting instead of cluster-level")
        logger.info("  Note: This may allow Entry/Coil/Trigger from same event in different splits")
        use_trinity_split = False
    elif force_cluster_split:
        logger.info("\n*** CLUSTER-AWARE SPLIT FORCED (--cluster-aware-split) ***")
        if metadata is None or 'nms_cluster_id' not in metadata.columns:
            logger.error("--cluster-aware-split requires metadata with nms_cluster_id column!")
            raise ValueError("Missing nms_cluster_id column for cluster-aware split")
        use_trinity_split = True
    elif detect_trinity_mode(metadata):
        use_trinity_split = True
        logger.info("\n*** TRINITY MODE AUTO-DETECTED ***")

    if use_trinity_split:
        unique_clusters = metadata['nms_cluster_id'].unique()
        # Map each sequence to its cluster via pattern_id
        pattern_to_cluster = metadata.groupby('pattern_id')['nms_cluster_id'].first().to_dict()
        cluster_ids = np.array([pattern_to_cluster.get(pid, -1) for pid in pattern_ids])
        logger.info(f"  Unique clusters: {len(unique_clusters):,}")
        logger.info(f"  Patterns per cluster: ~{len(pattern_ids) / len(unique_clusters):.1f}")
        logger.info("  Splitting by CLUSTER to prevent Entry/Coil/Trigger leakage")

    unique_patterns = np.unique(pattern_ids)
    num_patterns = len(unique_patterns)

    logger.info("=" * 80)
    logger.info("PATTERN-AWARE SPLITTING (prevents data leakage)")
    logger.info("=" * 80)
    logger.info(f"Total patterns: {num_patterns}")
    logger.info(f"Total sequences: {len(sequences)}")
    logger.info(f"Avg sequences/pattern: {len(sequences) / num_patterns:.1f}")

    # ENFORCE temporal split unless explicitly allowed to use random
    allow_random = getattr(args, 'allow_random_split', False)
    use_bear_aware = getattr(args, 'bear_aware_split', False)

    if metadata is None and not allow_random:
        logger.error("=" * 80)
        logger.error("TEMPORAL SPLIT REQUIRED - Metadata not provided!")
        logger.error("=" * 80)
        logger.error("Random split causes data leakage: overlapping sliding windows")
        logger.error("from the same pattern can end up in both train and val sets.")
        logger.error("")
        logger.error("Options:")
        logger.error("  1. Provide --metadata path/to/metadata.parquet (RECOMMENDED)")
        logger.error("  2. Use --allow-random-split to bypass (NOT recommended)")
        logger.error("=" * 80)
        return

    use_temporal = metadata is not None

    if use_temporal and metadata is not None:
        # TEMPORAL SPLIT: Split by pattern_start_date (no look-ahead bias)
        logger.info("\n*** USING TEMPORAL SPLIT (date-based, no shuffle) ***")

        # Define temporal boundaries
        if use_bear_aware:
            # BEAR-AWARE SPLIT: Ensures validation set contains bear market patterns
            # Train: 2020-2022 (includes 2022 bear patterns for learning)
            # Val:   2023 (includes bear->bull transition for validation)
            # Test:  2024+ (bull market, but model validated on bear)
            train_cutoff_date = '2023-01-01'
            val_cutoff_date = '2024-01-01'
            logger.info("*** BEAR-AWARE SPLIT ENABLED ***")
            logger.info("  This ensures validation includes bear market patterns (2023)")
        else:
            train_cutoff_date = args.train_cutoff
            val_cutoff_date = args.val_cutoff

        logger.info(f"  Train: < {train_cutoff_date}")
        logger.info(f"  Val:   {train_cutoff_date} to {val_cutoff_date}")
        logger.info(f"  Test:  >= {val_cutoff_date}")

        # Get pattern dates from metadata
        # Create pattern_id to date mapping
        if 'pattern_start_date' in metadata.columns:
            metadata['pattern_date'] = pd.to_datetime(metadata['pattern_start_date'])
        elif 'pattern_end_date' in metadata.columns:
            metadata['pattern_date'] = pd.to_datetime(metadata['pattern_end_date'])
        else:
            raise ValueError("Metadata must have 'pattern_start_date' or 'pattern_end_date' column")

        train_cutoff_dt = pd.Timestamp(train_cutoff_date)
        val_cutoff_dt = pd.Timestamp(val_cutoff_date)

        if use_trinity_split:
            # TRINITY MODE: Split by CLUSTER to keep Entry/Coil/Trigger together
            # Map cluster_id to date (use earliest pattern's date in cluster)
            cluster_to_date = metadata.groupby('nms_cluster_id')['pattern_date'].min().to_dict()
            unique_clusters = np.array(list(cluster_to_date.keys()))

            # Assign CLUSTERS to splits based on date
            train_clusters = []
            val_clusters = []
            test_clusters = []

            for cluster in unique_clusters:
                cluster_date = cluster_to_date.get(cluster)
                if cluster_date is None or cluster == -1:
                    train_clusters.append(cluster)
                elif cluster_date < train_cutoff_dt:
                    train_clusters.append(cluster)
                elif cluster_date < val_cutoff_dt:
                    val_clusters.append(cluster)
                else:
                    test_clusters.append(cluster)

            train_clusters = np.array(train_clusters)
            val_clusters = np.array(val_clusters)
            test_clusters = np.array(test_clusters)

            logger.info(f"\nCluster split (Trinity mode):")
            logger.info(f"  Train: {len(train_clusters)} clusters")
            logger.info(f"  Val:   {len(val_clusters)} clusters")
            logger.info(f"  Test:  {len(test_clusters)} clusters")

            # For compatibility, also track pattern splits (derived from clusters)
            cluster_to_patterns = metadata.groupby('nms_cluster_id')['pattern_id'].apply(set).to_dict()
            train_patterns = np.array([p for c in train_clusters for p in cluster_to_patterns.get(c, [])])
            val_patterns = np.array([p for c in val_clusters for p in cluster_to_patterns.get(c, [])])
            test_patterns = np.array([p for c in test_clusters for p in cluster_to_patterns.get(c, [])])

        else:
            # STANDARD MODE: Split by pattern_id
            # Map pattern_id to its date (use first occurrence)
            pattern_to_date = metadata.groupby('pattern_id')['pattern_date'].first().to_dict()

            # Assign patterns to splits based on date
            train_patterns = []
            val_patterns = []
            test_patterns = []

            for pattern in unique_patterns:
                pattern_date = pattern_to_date.get(pattern)
                if pattern_date is None:
                    # If pattern not in metadata, skip or assign to train
                    train_patterns.append(pattern)
                elif pattern_date < train_cutoff_dt:
                    train_patterns.append(pattern)
                elif pattern_date < val_cutoff_dt:
                    val_patterns.append(pattern)
                else:
                    test_patterns.append(pattern)

            train_patterns = np.array(train_patterns)
            val_patterns = np.array(val_patterns)
            test_patterns = np.array(test_patterns)

    else:
        # RANDOM SPLIT: Explicitly allowed via --allow-random-split
        logger.warning("=" * 80)
        logger.warning("WARNING: USING RANDOM SPLIT (--allow-random-split)")
        logger.warning("=" * 80)
        logger.warning("This may cause data leakage if patterns have overlapping windows.")
        logger.warning("Validation metrics may be INFLATED. Use only for debugging/testing.")
        logger.warning("For production models, use --metadata for temporal split.")
        logger.warning("=" * 80)

        np.random.seed(args.seed if hasattr(args, 'seed') else 42)
        shuffled_patterns = np.random.permutation(unique_patterns)

        # Split pattern IDs into train/val/test
        train_split = 1.0 - args.test_split - args.val_split
        train_cutoff = int(num_patterns * train_split)
        val_cutoff = int(num_patterns * (train_split + args.val_split))

        train_patterns = shuffled_patterns[:train_cutoff]
        val_patterns = shuffled_patterns[train_cutoff:val_cutoff]
        test_patterns = shuffled_patterns[val_cutoff:]

    logger.info(f"\nPattern split:")
    logger.info(f"  Train: {len(train_patterns)} patterns ({len(train_patterns)/num_patterns*100:.1f}%)")
    logger.info(f"  Val:   {len(val_patterns)} patterns ({len(val_patterns)/num_patterns*100:.1f}%)")
    logger.info(f"  Test:  {len(test_patterns)} patterns ({len(test_patterns)/num_patterns*100:.1f}%)")

    # Get sequence indices for each split (all sequences from selected patterns)
    train_mask = np.isin(pattern_ids, train_patterns)
    val_mask = np.isin(pattern_ids, val_patterns)
    test_mask = np.isin(pattern_ids, test_patterns)

    X_train = sequences[train_mask]
    y_train = labels[train_mask]
    X_val = sequences[val_mask]
    y_val = labels[val_mask]
    X_test = sequences[test_mask]
    y_test = labels[test_mask]

    # Split context features if available
    if use_context and context is not None:
        C_train = context[train_mask]
        C_val = context[val_mask]
        C_test = context[test_mask]
    else:
        C_train = C_val = C_test = None

    # ================================================================
    # EXTRACT VALIDATION DATES FOR ROLLING DAILY TOP-15% (Jan 2026)
    # ================================================================
    # Used to calculate per-day Top-15% precision instead of global pooling
    # This prevents survivorship bias from hiding poor daily performance
    # ================================================================
    val_dates_array = None
    if use_temporal and metadata is not None:
        date_col = 'pattern_end_date' if 'pattern_end_date' in metadata.columns else 'pattern_date'
        # Create pattern_id -> date mapping
        pattern_to_end_date = metadata.groupby('pattern_id')[date_col].first().to_dict()
        # Get date for each validation sequence (via its pattern_id)
        val_pattern_ids = pattern_ids[val_mask]
        val_dates_array = np.array([
            pattern_to_end_date.get(pid, pd.NaT)
            for pid in val_pattern_ids
        ])
        # Convert to string format for grouping (YYYY-MM-DD)
        val_dates_array = pd.to_datetime(val_dates_array).strftime('%Y-%m-%d').values
        logger.info(f"Extracted {len(val_dates_array)} validation dates for Rolling Daily Top-15%")

    logger.info(f"\nSequence split:")
    logger.info(f"  Train: {len(X_train)} sequences ({len(X_train)/len(sequences)*100:.1f}%)")
    logger.info(f"  Val:   {len(X_val)} sequences ({len(X_val)/len(sequences)*100:.1f}%)")
    logger.info(f"  Test:  {len(X_test)} sequences ({len(X_test)/len(sequences)*100:.1f}%)")

    # Verify no pattern appears in multiple splits
    assert len(set(train_patterns) & set(val_patterns)) == 0, "Train/Val pattern overlap!"
    assert len(set(train_patterns) & set(test_patterns)) == 0, "Train/Test pattern overlap!"
    assert len(set(val_patterns) & set(test_patterns)) == 0, "Val/Test pattern overlap!"
    logger.info(f"\nNo pattern overlap between splits (zero data leakage)")

    # ================================================================
    # TEMPORAL INTEGRITY CHECK (Jan 2026)
    # Verify no look-ahead bias in train/val/test splits
    # ================================================================
    if use_temporal and metadata is not None:
        logger.info("\n*** TEMPORAL INTEGRITY VERIFICATION ***")

        # Create metadata-level masks for integrity check
        meta_train_mask = metadata['pattern_id'].isin(train_patterns)
        meta_val_mask = metadata['pattern_id'].isin(val_patterns)
        meta_test_mask = metadata['pattern_id'].isin(test_patterns)

        # Get date ranges for each split
        train_dates = metadata.loc[meta_train_mask, 'pattern_date']
        val_dates = metadata.loc[meta_val_mask, 'pattern_date']
        test_dates = metadata.loc[meta_test_mask, 'pattern_date']

        train_max = train_dates.max() if len(train_dates) > 0 else None
        val_min = val_dates.min() if len(val_dates) > 0 else None
        val_max = val_dates.max() if len(val_dates) > 0 else None
        test_min = test_dates.min() if len(test_dates) > 0 else None

        logger.info(f"  Train date range: {train_dates.min().date() if len(train_dates) > 0 else 'N/A'} to {train_max.date() if train_max else 'N/A'}")
        logger.info(f"  Val date range:   {val_min.date() if val_min else 'N/A'} to {val_max.date() if val_max else 'N/A'}")
        logger.info(f"  Test date range:  {test_min.date() if test_min else 'N/A'} to {test_dates.max().date() if len(test_dates) > 0 else 'N/A'}")

        # Check for temporal leakage
        leakage_issues = []

        if val_min and train_max and val_min <= train_max:
            val_leakage = (val_dates <= train_max).sum()
            leakage_issues.append(f"Val has {val_leakage} samples <= train max ({train_max.date()})")

        if test_min and val_max and test_min <= val_max:
            test_leakage = (test_dates <= val_max).sum()
            leakage_issues.append(f"Test has {test_leakage} samples <= val max ({val_max.date()})")

        if leakage_issues:
            logger.error("*** TEMPORAL LEAKAGE DETECTED ***")
            for issue in leakage_issues:
                logger.error(f"  {issue}")
            raise DataIntegrityError(
                "Temporal leakage detected in train/val/test split:\n" +
                "\n".join(f"  - {issue}" for issue in leakage_issues)
            )
        else:
            logger.info("[PASS] No temporal leakage detected")

    # Trinity mode: Comprehensive cluster validation using utility function
    if use_trinity_split:
        try:
            validation_report = validate_cluster_split(
                metadata,
                train_clusters,
                val_clusters,
                test_clusters,
                cluster_column='nms_cluster_id',
                pattern_column='pattern_id'
            )
            logger.info(f"Cluster split validation PASSED:")
            logger.info(f"  Train: {validation_report['n_train_clusters']} clusters, {validation_report.get('n_train_patterns', '?')} patterns")
            logger.info(f"  Val:   {validation_report['n_val_clusters']} clusters, {validation_report.get('n_val_patterns', '?')} patterns")
            logger.info(f"  Test:  {validation_report['n_test_clusters']} clusters, {validation_report.get('n_test_patterns', '?')} patterns")
        except ValueError as e:
            logger.error(f"CLUSTER SPLIT VALIDATION FAILED: {e}")
            raise

    # Log market phase distribution per split (if available)
    if metadata is not None and 'market_phase' in metadata.columns:
        logger.info("\nMarket phase distribution per split:")
        for split_name, split_patterns in [('Train', train_patterns), ('Val', val_patterns), ('Test', test_patterns)]:
            split_meta = metadata[metadata['pattern_id'].isin(split_patterns)]
            if len(split_meta) > 0:
                bull_pct = (split_meta['market_phase'] == 1).mean() * 100
                bear_pct = (split_meta['market_phase'] == 0).mean() * 100
                logger.info(f"  {split_name}: Bull={bull_pct:.1f}% Bear={bear_pct:.1f}%")
                if bear_pct == 0 and split_name == 'Val':
                    logger.warning(f"  WARNING: {split_name} has NO bear patterns! Consider --bear-aware-split")
    logger.info("=" * 80)

    # 2.5 Feature Normalization using ROBUST SCALING (CRITICAL - fixes mode collapse from extreme values)
    # Uses pre-computed median/IQR from 01_generate_sequences.py for robustness to outliers
    logger.info("Applying robust scaling (median/IQR)...")

    # Feature groups (10 features total)
    # [0-4]: market (open, high, low, close, volume)
    # [5-7]: technical (bbw, adx, volume_ratio)
    # [8-9]: boundary slopes (upper_slope, lower_slope)
    n_features = X_train.shape[2]  # Should be 10

    # ========================================================================
    # CRITICAL FIX (Jan 2026): Compute robust scaling STRICTLY from X_train
    # ========================================================================
    # PREVIOUS BUG: StreamingSequenceWriter computed stats from ALL sequences
    # (train+val+test), causing test set information leakage via normalization.
    # Impact: 15-25% artificial Sharpe inflation (model knew future volatility).
    #
    # FIX: ALWAYS compute median/IQR from X_train only, NEVER load from file.
    # The global robust_scaling_params_*.json is now DEPRECATED for training.
    # ========================================================================
    logger.info("=" * 80)
    logger.info("COMPUTING ROBUST SCALING PARAMETERS STRICTLY FROM X_TRAIN")
    logger.info("(Global robust_scaling_params_*.json is DEPRECATED - prevents leakage)")
    logger.info("=" * 80)

    # Flatten (N, T, F) -> (N*T, F) to compute global stats per feature
    train_flat = X_train.reshape(-1, n_features)

    # Compute Median and IQR (ignoring NaNs for robustness)
    train_median = np.nanmedian(train_flat, axis=0).reshape(1, 1, n_features).astype(np.float32)
    q75, q25 = np.nanpercentile(train_flat, [75, 25], axis=0)
    train_iqr = (q75 - q25).reshape(1, 1, n_features).astype(np.float32)

    # Handle IQR=0 edge cases (replace with 1.0 to avoid division by zero)
    train_iqr = np.where(train_iqr > 1e-8, train_iqr, 1.0)

    # Log feature statistics BEFORE normalization
    logger.info("Feature statistics BEFORE normalization (train set):")
    for i in range(n_features):
        feat_min = X_train[:, :, i].min()
        feat_max = X_train[:, :, i].max()
        feat_median = train_median[0, 0, i]
        feat_iqr = train_iqr[0, 0, i]
        logger.info(f"  Feature {i}: min={feat_min:.2f}, max={feat_max:.2f}, median={feat_median:.2f}, iqr={feat_iqr:.2f}")

    # Apply robust scaling: (X - median) / IQR
    X_train = (X_train - train_median) / train_iqr
    X_val = (X_val - train_median) / train_iqr    # Use TRAIN statistics for val set
    X_test = (X_test - train_median) / train_iqr  # Use TRAIN statistics for test set

    # Log feature statistics AFTER normalization
    logger.info("Feature statistics AFTER normalization (train set):")
    for i in range(n_features):
        feat_min = X_train[:, :, i].min()
        feat_max = X_train[:, :, i].max()
        feat_mean = X_train[:, :, i].mean()
        feat_std = X_train[:, :, i].std()
        logger.info(f"  Feature {i}: min={feat_min:.2f}, max={feat_max:.2f}, mean={feat_mean:.4f}, std={feat_std:.4f}")

    # ========================================================================
    # CONTEXT FEATURE NORMALIZATION (Jan 2026 - "Lottery Ticket" Patch)
    # ========================================================================
    # VOLUME RATIO FEATURES - NOW PRE-COMPUTED WITH LOG_DIFF
    # ========================================================================
    # FIX (Jan 2026): Volume ratio features are now computed with log_diff()
    # in 01_generate_sequences.py, which already produces log-space values:
    #   log_diff(num, denom) = log1p(num) - log1p(denom)
    #
    # OLD PROBLEM: Raw ratios (e.g., 500x) saturated neural networks
    # OLD FIX: Apply log1p during training
    # NEW FIX: Features computed with log_diff at generation time
    #
    # IMPORTANT: Do NOT apply additional log1p here - values are already
    # in log-space and bounded (roughly -10 to +10).
    #
    # Indices affected (VOLUME_RATIO_INDICES from config):
    #   3: relative_volume     = log_diff(vol_20d, vol_60d)
    #   7: dormancy_shock      = log_diff(vol_20d, vol_252d)
    #   8: vol_dryup_ratio     = log_diff(vol_20d, vol_100d)
    #   10: volume_shock       = log_diff(max_vol_3d, vol_20d)
    #   12: vol_trend_5d       = log_diff(vol_5d, vol_20d)
    #   16: vol_contraction    = log_diff(vol_5d, vol_60d)
    # ========================================================================
    context_median = None
    context_iqr = None

    if C_train is not None and len(C_train) > 0:
        logger.info("=" * 80)
        logger.info("CONTEXT FEATURE NORMALIZATION (log_diff pre-applied at generation)")
        logger.info("=" * 80)

        n_context = C_train.shape[1]
        logger.info(f"Context features: {n_context} total")
        logger.info(f"Volume ratio indices (pre-computed with log_diff): {VOLUME_RATIO_INDICES}")
        logger.info("  -> Skipping log1p transformation (already in log-space)")

        # NOTE: No log1p transformation needed - features already computed with log_diff
        # The log_diff output is bounded and handles zeros gracefully

        # Log context stats BEFORE scaling
        logger.info("Context feature statistics BEFORE robust scaling:")
        for i in range(min(n_context, 18)):  # Cap at 18 features
            feat_min = C_train[:, i].min()
            feat_max = C_train[:, i].max()
            feat_mean = C_train[:, i].mean()
            logger.info(f"  Context {i}: min={feat_min:.2f}, max={feat_max:.2f}, mean={feat_mean:.4f}")

        # Compute robust scaling for context features from C_train
        context_median = np.nanmedian(C_train, axis=0).reshape(1, n_context).astype(np.float32)
        q75_ctx, q25_ctx = np.nanpercentile(C_train, [75, 25], axis=0)
        context_iqr = (q75_ctx - q25_ctx).reshape(1, n_context).astype(np.float32)
        context_iqr = np.where(context_iqr > 1e-8, context_iqr, 1.0)  # Avoid division by zero

        # Apply robust scaling to context features
        C_train = (C_train - context_median) / context_iqr
        C_val = (C_val - context_median) / context_iqr
        C_test = (C_test - context_median) / context_iqr

        # Log context stats AFTER scaling
        logger.info("Context feature statistics AFTER robust scaling:")
        for i in range(min(n_context, 18)):
            feat_min = C_train[:, i].min()
            feat_max = C_train[:, i].max()
            feat_mean = C_train[:, i].mean()
            logger.info(f"  Context {i}: min={feat_min:.2f}, max={feat_max:.2f}, mean={feat_mean:.4f}")

        logger.info("=" * 80)

    # Save normalization parameters for inference (ROBUST FORMAT)
    norm_params = {
        'scaling_type': 'robust',  # Identifies this as robust scaling (not standard)
        'median': train_median.squeeze().tolist(),
        'iqr': train_iqr.squeeze().tolist(),
        # Context normalization params (for inference)
        # NOTE (Jan 2026): context_log1p_indices is now EMPTY because volume ratio
        # features are pre-computed with log_diff in 01_generate_sequences.py.
        # Keeping the key for backwards compatibility but it should be empty [].
        'context_log1p_indices': [],  # EMPTY - log_diff applied at generation time
        'context_median': context_median.squeeze().tolist() if context_median is not None else None,
        'context_iqr': context_iqr.squeeze().tolist() if context_iqr is not None else None
    }
    os.makedirs("output/models", exist_ok=True)
    norm_params_path = f"output/models/norm_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(norm_params_path, 'w') as f:
        json.dump(norm_params, f, indent=2)
    logger.info(f"Saved robust scaling parameters to {norm_params_path}")

    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    dist = dict(zip(unique, counts))
    logger.info(f"Class distribution: {dist}")

    # =========================================================================
    # PHYSICS-AWARE AUGMENTATION (Train-time only)
    # =========================================================================
    # BAN: Additive noise on volume (destroys zero-volume supply exhaustion signal)
    # ALLOWED: Time Warping, Masking (preserves discrete volume values)
    # =========================================================================
    train_augmentor = None
    if getattr(args, 'use_augmentation', False):
        # Parse warp range
        warp_range = tuple(float(x) for x in args.aug_time_warp_range.split(','))

        train_augmentor = PhysicsAwareAugmentor(
            time_warp_p=args.aug_time_warp_p,
            time_warp_range=warp_range,
            timestep_dropout_p=args.aug_dropout_p,
            timestep_dropout_rate=args.aug_dropout_rate,
            feature_mask_p=args.aug_mask_p,
            feature_mask_ratio=args.aug_mask_ratio,
            protect_volume=True  # ALWAYS True - preserves zero-volume signal
        )
        logger.info("\n*** PHYSICS-AWARE AUGMENTATION ENABLED ***")
        logger.info(f"  TimeWarping: p={args.aug_time_warp_p}, range={warp_range}")
        logger.info(f"  TimestepDropout: p={args.aug_dropout_p}, rate={args.aug_dropout_rate}")
        logger.info(f"  FeatureMasking: p={args.aug_mask_p}, ratio={args.aug_mask_ratio}")
        logger.info("  Volume features PROTECTED (zero-volume = supply exhaustion signal)")

    # Window Jittering (forces RoPE utilization - breakouts valid at any timestep)
    if getattr(args, 'use_window_jitter', False):
        logger.info("\n*** WINDOW JITTERING ENABLED ***")
        logger.info(f"  Jitter range: [{args.jitter_min}, {args.jitter_max}] days")
        logger.info("  Forces model to recognize patterns at any position (RoPE utilization)")
        logger.info("  Edge-padding used for boundary handling")

    # Create Datasets (with context if available)
    # Check if lazy loading is enabled and input is HDF5
    use_lazy_loading = getattr(args, 'lazy_loading', False) and sequences_path.endswith('.h5')

    if use_lazy_loading:
        logger.info("Using LazyHDF5Dataset for memory-efficient on-demand loading")

        # Convert masks to indices for lazy dataset
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Compute context normalization for lazy loading (if context available)
        # Load training context samples to compute stats
        lazy_context_median = None
        lazy_context_iqr = None
        # NOTE (Jan 2026): context_log1p_indices is EMPTY - log_diff applied at generation time
        lazy_context_log1p_indices = []  # EMPTY - features pre-computed with log_diff

        import h5py
        with h5py.File(sequences_path, 'r') as h5f:
            if 'context' in h5f:
                logger.info("Computing context normalization for lazy loading...")
                logger.info("  -> Volume ratio features pre-computed with log_diff (no log1p needed)")
                lazy_context = h5f['context'][train_indices].astype(np.float32)
                n_ctx = lazy_context.shape[1]

                # NOTE: No log1p transformation - features already in log-space via log_diff

                # Compute robust scaling stats
                lazy_context_median = np.nanmedian(lazy_context, axis=0).reshape(1, n_ctx).astype(np.float32)
                q75, q25 = np.nanpercentile(lazy_context, [75, 25], axis=0)
                lazy_context_iqr = (q75 - q25).reshape(1, n_ctx).astype(np.float32)
                lazy_context_iqr = np.where(lazy_context_iqr > 1e-8, lazy_context_iqr, 1.0)

                # Squeeze for 1D indexing in LazyHDF5Dataset
                lazy_context_median = lazy_context_median.squeeze()
                lazy_context_iqr = lazy_context_iqr.squeeze()
                del lazy_context  # Free memory
                logger.info(f"  Context normalization computed from {len(train_indices)} training samples")

        # Create lazy datasets with robust scaling params for on-the-fly application
        # NOTE: Augmentor and window jitter passed to train_dataset ONLY (val/test should be unaugmented)
        jitter_range = (args.jitter_min, args.jitter_max) if args.use_window_jitter else None
        train_dataset = LazyHDF5Dataset(
            sequences_path, train_indices,
            train_median=train_median, train_iqr=train_iqr,
            context_median=lazy_context_median, context_iqr=lazy_context_iqr,
            context_log1p_indices=lazy_context_log1p_indices,
            augmentor=train_augmentor,  # Physics-aware augmentation (train only)
            use_window_jitter=args.use_window_jitter,
            jitter_range=jitter_range or (-3, 2)
        )
        val_dataset = LazyHDF5Dataset(
            sequences_path, val_indices,
            train_median=train_median, train_iqr=train_iqr,
            context_median=lazy_context_median, context_iqr=lazy_context_iqr,
            context_log1p_indices=lazy_context_log1p_indices,
            augmentor=None,  # No augmentation for validation
            use_window_jitter=False  # No jittering for validation
        )
        test_dataset = LazyHDF5Dataset(
            sequences_path, test_indices,
            train_median=train_median, train_iqr=train_iqr,
            context_median=lazy_context_median, context_iqr=lazy_context_iqr,
            context_log1p_indices=lazy_context_log1p_indices,
            augmentor=None,  # No augmentation for test
            use_window_jitter=False  # No jittering for test
        )

        logger.info(f"LazyHDF5Dataset created: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    else:
        # Standard in-memory datasets
        # NOTE: Augmentor and window jitter passed to train_dataset ONLY (val/test should be unaugmented)
        jitter_range = (args.jitter_min, args.jitter_max) if args.use_window_jitter else (-3, 2)
        train_dataset = TemporalSequenceDataset(
            X_train, y_train, C_train,
            augmentor=train_augmentor,
            use_window_jitter=args.use_window_jitter,
            jitter_range=jitter_range
        )
        val_dataset = TemporalSequenceDataset(X_val, y_val, C_val, augmentor=None, use_window_jitter=False)
        test_dataset = TemporalSequenceDataset(X_test, y_test, C_test, augmentor=None, use_window_jitter=False)

    # 3. Create DataLoaders with performance optimizations
    num_workers = get_optimal_workers(args.num_workers)
    use_pin_memory = torch.cuda.is_available()

    logger.info(f"DataLoader config: workers={num_workers}, pin_memory={use_pin_memory}, "
                f"persistent_workers={num_workers > 0}")

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': num_workers,
        'pin_memory': use_pin_memory,
        'persistent_workers': num_workers > 0,  # Keep workers alive between epochs
    }

    train_loader = TorchDataLoader(
        train_dataset,
        shuffle=True,  # Shuffle is True for ASL
        **loader_kwargs
    )

    val_loader = TorchDataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = TorchDataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )

    # 4. Initialize Model
    # Context features: 18 static features (retention_rate, trend_position, base_duration, relative_volume, etc.)
    # When use_context=True, enables GRN context branch with gated fusion
    context_dim = NUM_CONTEXT_FEATURES if use_context else 0
    # Dynamically determine input features from sequence shape
    input_features = sequences.shape[2]  # (N, 20, F) -> F features

    # Determine model mode from arguments
    # v18 (default) uses 'v18_full' mode, v20 uses specified ablation_mode
    model_version = getattr(args, 'model_version', 'v18')
    ablation_mode = getattr(args, 'ablation_mode', 'lstm')

    if model_version == 'v20':
        mode = ablation_mode  # concat, lstm, cqa_only, or v18_baseline
    else:
        mode = 'v18_full'

    logger.info("=" * 60)
    logger.info(f"INITIALIZING UNIFIED MODEL (mode={mode})")
    logger.info("=" * 60)

    model = HybridFeatureNetwork(
        mode=mode,
        input_features=input_features,
        sequence_length=20,
        context_features=context_dim,
        num_classes=3,
        fusion_dropout=args.dropout,
        use_conditioned_lstm=args.use_conditioned_lstm
    ).to(device)

    summary = model.get_architecture_summary()
    logger.info(f"  Mode: {summary['mode']}")
    logger.info(f"  Combined dim: {summary['combined_dim']}")
    logger.info(f"  Component dims: {summary['component_dims']}")
    logger.info(f"  Has LSTM: {summary['has_lstm']} ({summary['lstm_type']})")
    logger.info(f"  Has CQA: {summary['has_cqa']}")

    # Apply torch.compile for performance (PyTorch 2.0+)
    if args.compile:
        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if torch_version >= (2, 0):
            logger.info("Applying torch.compile optimization...")
            model = torch.compile(model)
            logger.info("Model compiled successfully (first epoch may be slower due to JIT)")
        else:
            logger.warning(f"torch.compile requires PyTorch 2.0+, you have {torch.__version__}. Skipping.")

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Context features: {'ENABLED (' + str(context_dim) + ' features)' if use_context else 'DISABLED'}")
    if args.use_conditioned_lstm:
        logger.info("Context-Conditioned LSTM: ENABLED (h0/c0 initialized from GRN output)")
        logger.info("  -> LSTM 'knows' the regime before processing the sequence")

    # 5. Define Loss Function (ASL Implementation)

    # Calculate weights for logging/fallback
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights_tensor = torch.FloatTensor(total_samples / (len(class_counts) * class_counts)).to(device)

    # Flag to track if we're using coil-aware loss (need to pass coil_intensity in training loop)
    use_coil_aware_loss = False
    use_volume_weighted = False  # VolumeWeightedLoss removed (Jan 2026) - volume is execution rule only

    if args.use_rank_match:
        # Ranking-Aware Coil Loss (Jan 2026) - Focal + MarginRanking for direct rank optimization
        # Requires 14 context features (coil_intensity at index 13)
        logger.info("Initializing Ranking-Aware Coil Loss (RankMatchCoilLoss)")
        logger.info(f"  -> focal_gamma=2.0, coil_strength_weight={args.coil_strength_weight}")
        logger.info(f"  -> rank_margin={args.rank_margin}, rank_lambda={args.rank_lambda}")
        logger.info(f"  -> danger_margin_mult={args.danger_margin_mult}")
        logger.info("  -> Directly optimizes for Top K precision via margin ranking")

        # Use sqrt weights for class balancing
        sqrt_weights = torch.sqrt(class_weights_tensor)
        logger.info(f"  -> Class weights (sqrt): {sqrt_weights}")

        criterion = RankMatchCoilLoss(
            focal_gamma=2.0,
            coil_weight=args.coil_strength_weight,
            rank_margin=args.rank_margin,
            rank_lambda=args.rank_lambda,
            danger_margin_mult=args.danger_margin_mult,
            class_weights=sqrt_weights
        )
        use_coil_aware_loss = True

    elif args.use_coil_focal:
        # Coil-Aware Focal Loss (Jan 2026) - boosts K2 patterns with strong coil signals
        # Requires 14 context features (coil_intensity at index 13)

        if args.use_asl:
            # COMBINED: Asymmetric + Coil-Aware (best of both worlds)
            logger.info(f"Initializing ASYMMETRIC Coil-Aware Focal Loss")
            logger.info(f"  -> gamma_neg={args.gamma_neg} (hard negative mining)")
            logger.info(f"  -> gamma_pos={args.gamma_pos} (preserve K2 gradients)")
            logger.info(f"  -> coil_strength_weight={args.coil_strength_weight}")
            logger.info("  -> K2 patterns with high coil_intensity will receive amplified gradients")

            # Use sqrt weights for class balancing
            sqrt_weights = torch.sqrt(class_weights_tensor)
            logger.info(f"  -> Class weights (sqrt): {sqrt_weights}")

            criterion = AsymmetricCoilFocalLoss(
                gamma_neg=args.gamma_neg,
                gamma_pos=args.gamma_pos,
                coil_strength_weight=args.coil_strength_weight,
                class_weights=sqrt_weights
            )
        else:
            # Standard Coil-Aware Focal Loss
            logger.info(f"Initializing Coil-Aware Focal Loss (gamma=2.0, coil_strength_weight={args.coil_strength_weight})")
            logger.info("  -> K2 patterns with high coil_intensity will receive amplified gradients")

            # Use sqrt weights for class balancing
            sqrt_weights = torch.sqrt(class_weights_tensor)
            logger.info(f"  -> Class weights (sqrt): {sqrt_weights}")

            criterion = CoilAwareFocalLoss(
                gamma=2.0,
                coil_strength_weight=args.coil_strength_weight,
                class_weights=sqrt_weights
            )
        use_coil_aware_loss = True

    elif args.use_asl:
        # Parse per-class gamma if provided
        gamma_per_class = None
        if args.gamma_per_class:
            gamma_per_class = [float(g) for g in args.gamma_per_class.split(',')]
            logger.info(f"Initializing Asymmetric Loss with per-class gamma: K0={gamma_per_class[0]}, K1={gamma_per_class[1]}, K2={gamma_per_class[2]}")
        else:
            logger.info(f"Initializing Asymmetric Loss (gamma_neg={args.gamma_neg}, gamma_pos={args.gamma_pos})")

        # Parse class weights if provided, otherwise compute from class distribution
        asl_class_weights = None
        if args.class_weights:
            asl_class_weights = [float(w) for w in args.class_weights.split(',')]
            logger.info(f"Using explicit class weights: K0={asl_class_weights[0]}, K1={asl_class_weights[1]}, K2={asl_class_weights[2]}")
        elif args.use_inverse_freq_weights:
            # Compute inverse frequency weights to boost minority class gradients
            asl_class_weights = (total_samples / (len(class_counts) * class_counts)).tolist()
            logger.info(f"Using inverse frequency weights: K0={asl_class_weights[0]:.2f}, K1={asl_class_weights[1]:.2f}, K2={asl_class_weights[2]:.2f}")

        # ASL handles imbalance via gamma + optional class weights + label smoothing
        criterion = AsymmetricLoss(
            gamma_neg=args.gamma_neg,
            gamma_pos=args.gamma_pos,
            clip=args.clip,
            disable_torch_grad_focal_loss=True,
            gamma_per_class=gamma_per_class,
            class_weights=asl_class_weights,
            label_smoothing=args.label_smoothing
        )
        logger.info(f"Label smoothing: {args.label_smoothing}")
    else:
        # Fallback to CrossEntropy (weighted or unweighted)
        if args.no_class_weights:
            logger.info("Using unweighted CrossEntropyLoss")
            criterion = nn.CrossEntropyLoss()
        else:
            # Use sqrt weights for more moderate effect
            sqrt_weights = torch.sqrt(class_weights_tensor)
            logger.info(f"Using CrossEntropyLoss with sqrt weights: {sqrt_weights}")
            criterion = nn.CrossEntropyLoss(weight=sqrt_weights)

    # ==========================================================================
    # NOTE: VolumeWeightedLoss REMOVED (Jan 2026)
    # ==========================================================================
    # Volume is now an EXECUTION RULE, not a training target.
    # Labels are pure price outcomes (Target = +3R hit regardless of volume).
    # Volume confirmation is checked at execution time in 03_predict_temporal.py
    # via the required_breakout_volume field in the execution plan.
    # ==========================================================================

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Log STRATEGIC_VALUES being used (imported from config.constants - single source of truth)
    logger.info(f"STRATEGIC_VALUES (EV calculation): {STRATEGIC_VALUES}")
    logger.info(f"  Danger (K0): {STRATEGIC_VALUES[0]}, Noise (K1): {STRATEGIC_VALUES[1]}, Target (K2): {STRATEGIC_VALUES[2]}")

    # Training Loop
    # Early stopping based on Top-15% Precision (Target rate), NOT global accuracy
    # Logic: Maximizes Profit Factor, not statistical correctness
    best_top15_precision = 0.0
    best_val_acc = 0  # Still track for logging
    patience_counter = 0
    best_model_path = ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("Starting training...")
    logger.info("Early stopping metric: Top-15% Precision (Target rate >= 45%)")

    quiet = getattr(args, 'quiet', False)

    # Verify context tensor shape on first batch (should be 15 features)
    first_batch = next(iter(train_loader))
    if len(first_batch) == 3:
        _, _, first_context = first_batch
        logger.info(f"Context tensor shape verification: {first_context.shape}")
        if first_context.shape[-1] != NUM_CONTEXT_FEATURES:
            logger.error(f"CONTEXT SHAPE MISMATCH! Expected {NUM_CONTEXT_FEATURES} features, got {first_context.shape[-1]}")
            raise ValueError(f"Context features mismatch: expected {NUM_CONTEXT_FEATURES}, got {first_context.shape[-1]}")
        logger.info(f"  [OK] Context has {NUM_CONTEXT_FEATURES} features (relative_strength_cohort at index 14)")
    else:
        logger.warning("No context features in batch - model will use sequences only")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=quiet):
            # Unpack batch - handle both with and without context
            if len(batch) == 3:
                batch_X, batch_y, batch_context = batch
                batch_context = batch_context.to(device)
            else:
                batch_X, batch_y = batch
                batch_context = None

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X, context=batch_context)

            # Compute loss (with coil_intensity for coil-aware loss)
            if use_coil_aware_loss and batch_context is not None:
                coil_intensity = get_coil_intensity_from_context(batch_context)
                loss = criterion(outputs, batch_y, coil_intensity=coil_intensity)
            else:
                loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation (CRITICAL: Use val_loader, NOT test_loader)
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []  # For Top-15% Precision calculation

        # Track attention weights for monitoring
        attention_setup_scores = []  # timesteps 0-2 (setup quality)
        attention_coil_scores = []   # timesteps 15-19 (final coil)
        attention_std_scores = []     # std across all timesteps (flatness check)

        # Track volume_shock for validation metrics (Precision at Volume)
        all_volume_shocks = []

        with torch.no_grad():
            for batch in val_loader:
                # Unpack batch - handle both with and without context
                if len(batch) == 3:
                    batch_X, batch_y, batch_context = batch
                    batch_context = batch_context.to(device)
                else:
                    batch_X, batch_y = batch
                    batch_context = None

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X, context=batch_context)

                # Compute loss (with coil_intensity for coil-aware loss, context for volume-weighted)
                if use_coil_aware_loss and batch_context is not None:
                    coil_intensity = get_coil_intensity_from_context(batch_context)
                    if use_volume_weighted:
                        loss = criterion(outputs, batch_y, context=batch_context, coil_intensity=coil_intensity)
                    else:
                        loss = criterion(outputs, batch_y, coil_intensity=coil_intensity)
                elif use_volume_weighted and batch_context is not None:
                    loss = criterion(outputs, batch_y, context=batch_context)
                else:
                    loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                # Get probabilities for Top-15% calculation
                probs = torch.softmax(outputs, dim=-1)
                all_probs.append(probs.cpu().numpy())

                # Track volume_shock for validation metrics
                if batch_context is not None:
                    vol_shock = get_volume_shock_from_context(batch_context)
                    all_volume_shocks.append(vol_shock.cpu().numpy())
                else:
                    all_volume_shocks.append(np.ones(batch_y.size(0)))  # Neutral default

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

                # Extract attention weights from first batch only (for efficiency)
                if len(attention_setup_scores) == 0:
                    importance = model.get_feature_importance(batch_X, context=batch_context)
                    # temporal_importance shape: (batch, 20)
                    temporal_importance = importance['temporal_importance']

                    # Mean attention for setup (days 0-2)
                    setup_attention = temporal_importance[:, 0:3].mean()
                    attention_setup_scores.append(setup_attention)

                    # Mean attention for coil (days 15-19)
                    coil_attention = temporal_importance[:, 15:20].mean()
                    attention_coil_scores.append(coil_attention)

                    # Std across all timesteps (detect flat attention)
                    attention_std = temporal_importance.std(axis=1).mean()
                    attention_std_scores.append(attention_std)

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # =========================================================================
        # TOP-15% PRECISION (Primary Early Stopping Metric)
        # =========================================================================
        # Sort by EV, take top 15%, measure Target (class 2) rate
        # Logic: Maximizes Profit Factor, not statistical correctness
        # =========================================================================
        all_probs_np = np.concatenate(all_probs, axis=0)  # (N, 3)
        all_labels_np = np.array(all_labels)

        # Calculate EV for each sample: EV = sum(prob_i * value_i)
        ev_scores = (
            all_probs_np[:, 0] * STRATEGIC_VALUES[0] +  # Danger
            all_probs_np[:, 1] * STRATEGIC_VALUES[1] +  # Noise
            all_probs_np[:, 2] * STRATEGIC_VALUES[2]    # Target
        )

        # Sort by EV descending, take top 15%
        n_samples = len(ev_scores)
        n_top15 = max(1, int(n_samples * 0.15))
        top15_indices = np.argsort(ev_scores)[::-1][:n_top15]

        # Calculate Target rate in top 15%
        top15_labels = all_labels_np[top15_indices]
        top15_target_rate = (top15_labels == 2).mean() * 100  # Percentage
        top15_danger_rate = (top15_labels == 0).mean() * 100

        # =========================================================================
        # ROLLING DAILY TOP-15% (Jan 2026 - Alpha Validation)
        # =========================================================================
        # Per-day calculation reflects real trading: you pick Top-15% each day
        # Global pooling hides survivorship bias (some days have 0% success)
        # =========================================================================
        rolling_top15_target = None
        rolling_top15_danger = None
        rolling_n_days = 0
        if val_dates_array is not None and len(val_dates_array) == len(ev_scores):
            daily_target_rates = []
            daily_danger_rates = []
            unique_dates = np.unique(val_dates_array)

            for date in unique_dates:
                date_mask = val_dates_array == date
                day_ev = ev_scores[date_mask]
                day_labels = all_labels_np[date_mask]

                n_day = len(day_ev)
                if n_day < 10:  # Skip days with too few patterns for meaningful Top-15%
                    continue

                n_top15_day = max(1, int(n_day * 0.15))
                top15_day_idx = np.argsort(day_ev)[::-1][:n_top15_day]
                day_top15_labels = day_labels[top15_day_idx]

                target_rate = (day_top15_labels == 2).mean() * 100
                danger_rate = (day_top15_labels == 0).mean() * 100
                daily_target_rates.append(target_rate)
                daily_danger_rates.append(danger_rate)

            if len(daily_target_rates) > 0:
                rolling_top15_target = np.mean(daily_target_rates)
                rolling_top15_danger = np.mean(daily_danger_rates)
                rolling_n_days = len(daily_target_rates)

        # Calculate per-class recall for monitoring
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        k0_recall = report.get('0', {}).get('recall', 0)
        k1_recall = report.get('1', {}).get('recall', 0)
        k2_recall = report.get('2', {}).get('recall', 0)

        # =========================================================================
        # VOLUME-AWARE VALIDATION METRICS (Jan 2026)
        # =========================================================================
        # 1. Precision at Volume: Accuracy of Class 2 predictions when high volume occurs
        # 2. Expected R-Value: (Precision_Class2 * 3.0) - (False_Positives * 1.0)
        # =========================================================================
        all_preds_np = np.array(all_preds)

        # Combine volume_shocks from all batches
        if all_volume_shocks:
            all_volume_np = np.concatenate(all_volume_shocks, axis=0)
        else:
            all_volume_np = np.ones(len(all_labels_np))  # Neutral default

        # Get volume threshold (use CLI arg or default)
        vol_thresh = getattr(args, 'volume_threshold', 1.5)

        # High volume mask (volume_shock >= threshold)
        high_vol_mask = all_volume_np >= vol_thresh

        # Class 2 predictions
        class2_preds_mask = (all_preds_np == 2)
        class2_actual_mask = (all_labels_np == 2)

        # --- Precision at Volume ---
        # Among Class 2 predictions with high volume, what % are actually Class 2?
        class2_high_vol_preds = class2_preds_mask & high_vol_mask
        n_class2_high_vol = class2_high_vol_preds.sum()

        if n_class2_high_vol > 0:
            # True Positives at high volume
            tp_high_vol = (class2_high_vol_preds & class2_actual_mask).sum()
            precision_at_volume = tp_high_vol / n_class2_high_vol * 100
        else:
            precision_at_volume = 0.0

        # --- Expected R-Value ---
        # Formula: (Precision_Class2 * 3.0) - (FP_Rate * 1.0)
        # Where: Precision_Class2 = TP / (TP + FP)
        #        FP_Rate = FP / total_class2_preds
        n_class2_preds = class2_preds_mask.sum()
        if n_class2_preds > 0:
            tp = (class2_preds_mask & class2_actual_mask).sum()
            fp = (class2_preds_mask & ~class2_actual_mask).sum()
            precision_class2 = tp / n_class2_preds
            fp_rate = fp / n_class2_preds
            expected_r_value = (precision_class2 * 3.0) - (fp_rate * 1.0)
        else:
            precision_class2 = 0.0
            fp_rate = 0.0
            expected_r_value = 0.0

        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}% | "
                    f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")
        logger.info(f"Recall: K0(Danger)={k0_recall:.2%} | K1(Noise)={k1_recall:.2%} | K2(Target)={k2_recall:.2%}")
        logger.info(f"*** Top-15% Precision (Global): Target={top15_target_rate:.1f}% | Danger={top15_danger_rate:.1f}% | "
                   f"n={n_top15}/{n_samples} ***")
        if rolling_top15_target is not None:
            logger.info(f"*** Top-15% Precision (Rolling Daily): Target={rolling_top15_target:.1f}% | "
                       f"Danger={rolling_top15_danger:.1f}% | n_days={rolling_n_days} ***")
        logger.info(f"Volume Metrics: Precision@Volume={precision_at_volume:.1f}% (n={n_class2_high_vol}) | "
                   f"Expected R={expected_r_value:.2f}R (Prec={precision_class2:.1%}, FP={fp_rate:.1%})")

        # Log attention weight patterns (Coil Detector monitoring)
        if attention_setup_scores:
            setup_attn = attention_setup_scores[0]
            coil_attn = attention_coil_scores[0]
            attn_std = attention_std_scores[0]

            # Coil/Setup ratio: Should be >= 1.0 (coil period more important than setup)
            coil_setup_ratio = coil_attn / setup_attn if setup_attn > 1e-8 else float('inf')

            logger.info(f"Attention: Setup(0-2)={setup_attn:.4f} | Coil(15-19)={coil_attn:.4f} | "
                       f"Ratio={coil_setup_ratio:.2f} | Std={attn_std:.4f}")

            # Warning ONLY if Coil/Setup ratio < 1.0 (wrong semantic focus)
            # Uniform attention is acceptable if coil focus >= setup focus
            if coil_setup_ratio < 1.0:
                logger.warning(f"[WARN] ATTENTION FOCUS INVERTED (Coil/Setup={coil_setup_ratio:.2f} < 1.0) - "
                              "Model focusing on early setup instead of critical coil period!")

        # Save best model based on TOP-15% PRECISION (not global accuracy)
        # Logic: Maximizes Profit Factor - we care about precision in our top picks
        if top15_target_rate > best_top15_precision:
            best_top15_precision = top15_target_rate
            best_val_acc = val_acc  # Track for reference
            patience_counter = 0
            os.makedirs("output/models", exist_ok=True)

            # Use custom model name if provided
            model_prefix = args.model_name if hasattr(args, 'model_name') and args.model_name else 'best_model'
            best_model_path = f"output/models/{model_prefix}_{timestamp}.pt"

            # Handle compiled models - extract underlying model for saving
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

            # === ATOMIC CHECKPOINT (Self-Contained Deployment) ===
            # Embed ALL transform params so model is truly self-contained
            checkpoint_config = vars(args).copy()
            checkpoint_config['input_features'] = 10  # Save for inference compatibility
            InferenceWrapper.create_atomic_checkpoint(
                model=model_to_save,
                optimizer=optimizer,
                epoch=epoch,
                val_acc=val_acc,
                config=checkpoint_config,
                norm_params=norm_params,
                context_ranges=dict(CONTEXT_FEATURE_RANGES),  # 15 context feature ranges
                calibrator=None,  # Will be added after calibration
                temperature=PREDICTION_TEMPERATURE,
                save_path=best_model_path,
                top15_precision=top15_target_rate,  # Primary metric
                top15_danger_rate=top15_danger_rate
            )
            logger.info(f"*** NEW BEST MODEL *** Top-15% Target={top15_target_rate:.1f}% | "
                       f"Danger={top15_danger_rate:.1f}% | Acc={val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    logger.info("Training complete.")
    logger.info(f"Best Top-15% Precision: {best_top15_precision:.1f}% Target rate")

    # =========================================================================
    # PROBABILITY CALIBRATION (Post-hoc Isotonic Regression)
    # =========================================================================
    # Focal loss distorts predicted probabilities. Isotonic Regression learns
    # a monotonic mapping from raw probs to actual class frequencies.
    # =========================================================================
    if USE_PROBABILITY_CALIBRATION:
        logger.info("\n" + "=" * 60)
        logger.info("FITTING PROBABILITY CALIBRATOR (Isotonic Regression)")
        logger.info("=" * 60)

        try:
            from utils.probability_calibration import ProbabilityCalibrator

            # Load best model for calibration
            if best_model_path and os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                logger.info(f"Loaded best model from {best_model_path}")

            # Get validation set predictions (raw probabilities)
            val_probs = []
            val_true = []

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        batch_X, batch_y, batch_context = batch
                        batch_context = batch_context.to(device)
                    else:
                        batch_X, batch_y = batch
                        batch_context = None

                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    logits = model(batch_X, context=batch_context)
                    probs = torch.softmax(logits, dim=-1)

                    val_probs.append(probs.cpu().numpy())
                    val_true.append(batch_y.cpu().numpy())

            val_probs = np.concatenate(val_probs, axis=0)
            val_true = np.concatenate(val_true, axis=0)

            logger.info(f"Fitting calibrator on {len(val_true)} validation samples")

            # Fit calibrator
            calibrator = ProbabilityCalibrator(n_classes=3)
            calibrator.fit(val_probs, val_true)

            # Save calibrator (separate file for legacy compatibility)
            calibrator_path = f"output/models/calibrator_{timestamp}.pkl"
            calibrator.save(calibrator_path)
            logger.info(f"Saved calibrator to {calibrator_path}")

            # === RE-SAVE ATOMIC CHECKPOINT WITH EMBEDDED CALIBRATOR ===
            if best_model_path and os.path.exists(best_model_path):
                logger.info("Re-saving atomic checkpoint with embedded calibrator...")
                calib_config = vars(args).copy()
                calib_config['input_features'] = 10  # Save for inference compatibility
                InferenceWrapper.create_atomic_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=checkpoint.get('epoch', 0),
                    val_acc=checkpoint.get('val_acc', best_val_acc),
                    config=calib_config,
                    norm_params=norm_params,
                    context_ranges=dict(CONTEXT_FEATURE_RANGES),
                    calibrator=calibrator,  # Now includes calibrator!
                    temperature=PREDICTION_TEMPERATURE,
                    save_path=best_model_path,
                    top15_precision=checkpoint.get('top15_precision', best_top15_precision),
                    top15_danger_rate=checkpoint.get('top15_danger_rate')
                )
                logger.info(f"Atomic checkpoint updated with calibrator: {best_model_path}")

            # Log calibration report
            logger.info("\n" + calibrator.get_calibration_report())

        except ImportError as e:
            logger.warning(f"Calibration skipped: {e}")
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info("Probability calibration DISABLED (USE_PROBABILITY_CALIBRATION=False)")


def train_ensemble(args):
    """
    Train an ensemble of models with varied seeds/architectures.

    Produces:
    - Individual model checkpoints (ensemble_member_00.pt, etc.)
    - Combined ensemble checkpoint (ensemble_combined.pt)
    - Dirichlet-calibrated ensemble for uncertainty estimation

    Args:
        args: Parsed command-line arguments with ensemble_size > 1
    """
    import copy
    from pathlib import Path
    from models.ensemble_wrapper import EnsembleWrapper, EnsembleTrainer
    from utils.dirichlet_calibration import DirichletCalibrator

    n_models = args.ensemble_size
    base_seed = args.ensemble_base_seed
    vary_arch = args.ensemble_vary_arch

    logger.info("\n" + "="*70)
    logger.info(f"ENSEMBLE TRAINING: {n_models} models")
    logger.info("="*70)
    logger.info(f"  Base seed: {base_seed}")
    logger.info(f"  Vary architecture: {vary_arch}")
    logger.info(f"  Dirichlet calibration: {args.use_dirichlet_calibration}")

    # Setup output directory for ensemble
    # FIXED: Must match where train_model saves checkpoints (output/models/)
    output_dir = Path(args.output) if hasattr(args, 'output') and args.output else Path('output/models')
    ensemble_dir = output_dir / 'ensemble'
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    # Architecture variations if enabled
    arch_variations = [
        {'dropout': 0.30, 'hidden_dim': 128},
        {'dropout': 0.40, 'hidden_dim': 128},
        {'dropout': 0.35, 'hidden_dim': 144},
        {'dropout': 0.30, 'hidden_dim': 160},
        {'dropout': 0.45, 'hidden_dim': 128},
    ]

    checkpoint_paths = []
    training_stats = []

    # Train each ensemble member
    for i in range(n_models):
        member_seed = base_seed + i * 1000
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Ensemble Member {i+1}/{n_models}")
        logger.info(f"  Seed: {member_seed}")

        # Create modified args for this member
        member_args = copy.deepcopy(args)
        member_args.seed = member_seed
        member_args.ensemble_size = 1  # Train as single model

        # Apply architecture variation if enabled
        if vary_arch and i < len(arch_variations):
            variation = arch_variations[i]
            member_args.dropout = variation['dropout']
            # hidden_dim would need model modification, so just vary dropout
            logger.info(f"  Dropout: {member_args.dropout}")

        # Modify model name to include member index
        base_name = args.model_name or 'model'
        member_args.model_name = f"{base_name}_ensemble_{i:02d}"

        logger.info(f"{'='*60}\n")

        # Train the member
        train_model(member_args)

        # Find the saved checkpoint
        # The checkpoint is saved as {model_name}_{timestamp}.pt
        import glob
        checkpoint_pattern = str(output_dir / f"{member_args.model_name}_*.pt")
        matches = sorted(glob.glob(checkpoint_pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
        if matches:
            checkpoint_path = Path(matches[0])  # Most recent
            checkpoint_paths.append(checkpoint_path)
            logger.info(f"  Found checkpoint: {checkpoint_path}")
        else:
            # Fallback: look for any file with the model name
            fallback_pattern = str(output_dir / f"*{member_args.model_name}*.pt")
            fallback_matches = sorted(glob.glob(fallback_pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
            if fallback_matches:
                checkpoint_path = Path(fallback_matches[0])
                checkpoint_paths.append(checkpoint_path)
                logger.info(f"  Found checkpoint (fallback): {checkpoint_path}")
            else:
                logger.warning(f"  No checkpoint found for {member_args.model_name}")

    # Create ensemble from trained models
    logger.info(f"\n{'='*70}")
    logger.info("Creating Ensemble")
    logger.info("="*70)

    if len(checkpoint_paths) < 2:
        logger.error("Insufficient checkpoints for ensemble. Need at least 2.")
        return

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        ensemble = EnsembleWrapper.from_checkpoints(checkpoint_paths, device=device)
        logger.info(f"  Loaded {ensemble.n_models} models")

        # Fit Dirichlet calibrator if requested
        if args.use_dirichlet_calibration:
            logger.info("\nFitting Dirichlet calibrator on validation data...")

            # Load validation data for calibration
            # We need to re-load the data that was used for validation
            # NOTE: sequences_path already resolved from wildcards earlier in main()
            calibration_path = Path(sequences_path)

            if calibration_path.suffix == '.h5':
                import h5py
                with h5py.File(calibration_path, 'r') as f:
                    sequences = f['sequences'][:]
                    labels = f['labels'][:]
                    context = f['context'][:] if 'context' in f else None

                # Use last 20% as calibration set (approximating val set)
                n_total = len(sequences)
                n_cal = int(n_total * 0.2)
                cal_sequences = sequences[-n_cal:]
                cal_labels = labels[-n_cal:]
                cal_context = context[-n_cal:] if context is not None else None

                ensemble.fit_calibrator(cal_sequences, cal_labels, cal_context)
                logger.info("  Dirichlet calibration complete")

        # Save combined ensemble checkpoint
        ensemble_path = ensemble_dir / "ensemble_combined.pt"
        ensemble.save_ensemble_checkpoint(ensemble_path)
        logger.info(f"\nSaved ensemble checkpoint: {ensemble_path}")

        # Print ensemble summary
        info = ensemble.get_ensemble_info()
        logger.info("\nEnsemble Summary:")
        logger.info(f"  Models: {info['n_models']}")
        logger.info(f"  Device: {info['device']}")
        logger.info(f"  Dirichlet calibrator: {info['has_calibrator']}")

        if info['calibrator_stats']:
            stats = info['calibrator_stats']
            logger.info(f"  Pre-calibration ECE: {stats.get('pre_ece', 'N/A'):.4f}")
            logger.info(f"  Post-calibration ECE: {stats.get('post_ece', 'N/A'):.4f}")
            logger.info(f"  ECE reduction: {stats.get('ece_reduction', 'N/A'):.1f}%")

    except Exception as e:
        logger.error(f"Ensemble creation failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "="*70)
    logger.info("ENSEMBLE TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"  Individual checkpoints: {len(checkpoint_paths)}")
    logger.info(f"  Ensemble checkpoint: {ensemble_dir / 'ensemble_combined.pt'}")
    logger.info("\nUsage:")
    logger.info("  from models.ensemble_wrapper import EnsembleWrapper")
    logger.info(f"  ensemble = EnsembleWrapper.from_ensemble_checkpoint('{ensemble_dir / 'ensemble_combined.pt'}')")
    logger.info("  results = ensemble.predict_with_uncertainty(sequences, context)")


def main():
    parser = argparse.ArgumentParser(description='Train Temporal Hybrid Model')
    parser.add_argument('--sequences', type=str, required=True,
                       help='Path to sequences file (.npy or .h5). HDF5 files contain labels internally.')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to labels .npy file (not required for .h5 files)')
    parser.add_argument('--context', type=str, default=None,
                       help='Path to context features .npy file (8 features for Branch B GRN)')
    parser.add_argument('--metadata', type=str,
                       help='Path to metadata parquet (REQUIRED for temporal split)')
    parser.add_argument('--allow-random-split', action='store_true',
                       help='Allow random split without metadata (NOT recommended - causes data leakage)')
    parser.add_argument('--bear-aware-split', action='store_true',
                       help='Use bear-aware split: Train<2023, Val=2023 (includes bear), Test=2024+')
    parser.add_argument('--cluster-aware-split', action='store_true',
                       help='Force cluster-aware splitting for Trinity mode (Entry/Coil/Trigger stay together). '
                            'Auto-detected when nms_cluster_id present in metadata.')
    parser.add_argument('--disable-trinity', action='store_true',
                       help='Disable Trinity mode cluster-based splitting. Use pattern-level temporal splitting '
                            'instead. Required for ablation studies when Trinity causes imbalanced splits.')
    parser.add_argument('--train-cutoff', type=str, default='2024-01-01',
                       help='Temporal split: train data before this date (default: 2024-01-01)')
    parser.add_argument('--val-cutoff', type=str, default='2024-07-01',
                       help='Temporal split: val data before this date, test after (default: 2024-07-01)')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Custom model name prefix (e.g., eu_model, us_model)')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Random split: test set size (only with --allow-random-split)')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Random split: val set size (only with --allow-random-split)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Model Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--early-stopping', type=int, default=15)

    # ASL Configuration - PRECISION FOCUS (Hard Negative Mining)
    # Philosophy: Penalize False Positives (predicting Target when actual is Danger) MORE than missing Targets
    # This maximizes Precision at the cost of Recall, which is optimal for live trading
    parser.add_argument('--use-asl', action='store_true', help='Use Asymmetric Loss')
    parser.add_argument('--use-coil-focal', action='store_true',
                       help='Use Coil-Aware Focal Loss (Jan 2026) - boosts K2 patterns with strong coil signals')
    parser.add_argument('--use-rank-match', action='store_true',
                       help='Use Ranking-Aware Coil Loss (Jan 2026) - Focal + MarginRanking for direct rank optimization')
    parser.add_argument('--coil-strength-weight', type=float, default=3.0,
                       help='Coil intensity boost weight for K2 patterns (default: 3.0)')
    parser.add_argument('--rank-margin', type=float, default=0.1,
                       help='Margin for ranking loss (default: 0.1). Target must exceed Noise by this margin.')
    parser.add_argument('--rank-lambda', type=float, default=0.5,
                       help='Weight of ranking loss relative to focal (default: 0.5)')
    parser.add_argument('--danger-margin-mult', type=float, default=2.0,
                       help='Multiplier for Danger margin vs Noise margin (default: 2.0)')
    # Note: VolumeWeightedLoss removed (Jan 2026) - volume is now an execution rule, not a training target
    # Volume metrics (Precision@Volume, Expected R) still tracked for monitoring
    parser.add_argument('--volume-threshold', type=float, default=1.5,
                       help='Volume shock threshold for metrics tracking (default: 1.5). Used for Precision@Volume metric only.')
    parser.add_argument('--gamma-neg', type=float, default=4.0,
                       help='ASL Gamma Negative (Precision: 4.0) - High gamma focuses on hard negatives')
    parser.add_argument('--gamma-pos', type=float, default=0.5,
                       help='ASL Gamma Positive (Precision: 0.5) - Low gamma for easy positives')
    parser.add_argument('--clip', type=float, default=0.05, help='ASL probability clipping')
    parser.add_argument('--label-smoothing', type=float, default=0.01,
                       help='Label smoothing factor (0.0-0.2). Prevents mode collapse on noisy micro-cap data. Default: 0.01')
    parser.add_argument('--gamma-per-class', type=str, default="4.0,2.0,0.5",
                       help='Per-class gamma: K0(Danger),K1(Noise),K2(Target). Precision: "4.0,2.0,0.5" (high gamma on Danger)')
    parser.add_argument('--class-weights', type=str, default="5.0,1.0,1.0",
                       help='Per-class weights: K0(Danger),K1(Noise),K2(Target). Precision: "5.0,1.0,1.0" (heavy Danger penalty)')
    parser.add_argument('--use-inverse-freq-weights', action='store_true',
                       help='Use inverse frequency class weights to boost minority class gradients')
    parser.add_argument('--no-class-weights', action='store_true',
                       help='Disable class weights (use unweighted CrossEntropyLoss)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output (summary only)')
    parser.add_argument('--exclude-context-features', type=str, default=None,
                       help='Comma-separated indices of context features to exclude (e.g., "13" to remove coil_intensity). '
                            'Excluded features are set to their default values.')

    # Performance Optimizations
    parser.add_argument('--num-workers', type=int, default=0,
                       help='DataLoader workers (0=auto: 2 Windows/4 Linux, -1=disable)')
    parser.add_argument('--compile', action='store_true',
                       help='Enable torch.compile optimization (PyTorch 2.0+, 20-50%% speedup)')
    parser.add_argument('--lazy-loading', action='store_true',
                       help='Use LazyHDF5Dataset to read samples on-demand instead of loading all into memory. '
                            'Reduces memory usage for large datasets but may be slower due to I/O.')

    # Architecture Options
    parser.add_argument('--model-version', type=str, default='v18', choices=['v18', 'v20'],
                       help='Model version: v18 (default) or v20 (ablation architecture)')
    parser.add_argument('--ablation-mode', type=str, default='lstm',
                       choices=['concat', 'lstm', 'cqa_only', 'v18_baseline'],
                       help='V20 ablation mode: concat (simplest), lstm (standard), '
                            'cqa_only (no LSTM), v18_baseline (full V18). Only used with --model-version v20')
    parser.add_argument('--use-conditioned-lstm', action='store_true',
                       help='Initialize LSTM hidden state from context embedding (regime prior). V18 only.')

    # Early Labeling Ablation
    parser.add_argument('--exclude-early-closed', action='store_true',
                       help='Exclude early-closed patterns from training (ablation study). '
                            'Only train on patterns with full 100-day window.')

    # Physics-Aware Augmentation (Jan 2026)
    # NOTE: ONLY TimeWarping and Masking are allowed. Additive noise BANNED (destroys zero-volume signal).
    parser.add_argument('--use-augmentation', action='store_true',
                       help='Enable physics-aware augmentation (TimeWarping + Masking). '
                            'BAN: Additive noise on volume - destroys supply exhaustion signal.')
    parser.add_argument('--aug-time-warp-p', type=float, default=0.3,
                       help='Probability of time warping augmentation (default: 0.3)')
    parser.add_argument('--aug-time-warp-range', type=str, default='0.8,1.2',
                       help='Time warp factor range as min,max (default: 0.8,1.2)')
    parser.add_argument('--aug-dropout-p', type=float, default=0.2,
                       help='Probability of timestep dropout augmentation (default: 0.2)')
    parser.add_argument('--aug-dropout-rate', type=float, default=0.1,
                       help='Fraction of timesteps to drop (default: 0.1)')
    parser.add_argument('--aug-mask-p', type=float, default=0.2,
                       help='Probability of feature masking augmentation (default: 0.2)')
    parser.add_argument('--aug-mask-ratio', type=float, default=0.15,
                       help='Fraction of features to mask (default: 0.15)')

    # Window Jittering (Jan 2026)
    parser.add_argument('--use-window-jitter', action='store_true',
                       help='Enable window jittering: randomly shift start index by -3 to +2 days. '
                            'Forces model to recognize patterns at any position (RoPE utilization).')
    parser.add_argument('--jitter-min', type=int, default=-3,
                       help='Minimum window shift (negative = earlier window). Default: -3')
    parser.add_argument('--jitter-max', type=int, default=2,
                       help='Maximum window shift (positive = later window). Default: +2')

    # Ensemble Training (Jan 2026)
    parser.add_argument('--ensemble-size', type=int, default=1,
                       help='Number of models in ensemble (default: 1 = single model). '
                            'Use 3-5 for production ensemble with uncertainty estimates.')
    parser.add_argument('--ensemble-vary-arch', action='store_true',
                       help='Vary architecture (dropout, hidden_dim) across ensemble members')
    parser.add_argument('--use-dirichlet-calibration', action='store_true',
                       help='Use Dirichlet calibration instead of isotonic (better for multi-class)')
    parser.add_argument('--ensemble-base-seed', type=int, default=42,
                       help='Base seed for ensemble (members use base_seed + i*1000)')

    args = parser.parse_args()

    # Quiet mode setup
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
        import warnings
        warnings.filterwarnings('ignore')

    # Ensemble or single model training
    if args.ensemble_size > 1:
        train_ensemble(args)
    else:
        train_model(args)

if __name__ == "__main__":
    main()
