#!/usr/bin/env python3
"""
Validate sequence data integrity BEFORE robust scaling.

This script checks that:
1. Temporal sequences contain no NaN/Inf values
2. Context features contain no NaN/Inf values
3. The log-diff transformation properly handled dormant stocks

Usage:
    python scripts/validate_sequence_integrity.py --sequences output/sequences/eu/sequences.h5
    python scripts/validate_sequence_integrity.py --seq-dir output/sequences/eu

Exit codes:
    0 = Clean (no NaN/Inf)
    1 = Found NaN/Inf values
    2 = File not found / load error
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import feature names from config (single source of truth)
from config.context_features import CONTEXT_FEATURES, VOLUME_RATIO_INDICES
from config.temporal_features import TemporalFeatureConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Feature names from config
_temporal_config = TemporalFeatureConfig()
TEMPORAL_FEATURES = _temporal_config.all_features

# Indices that use LOG-DIFF (from FeatureRegistry - single source of truth)
LOG_DIFF_CONTEXT_INDICES = VOLUME_RATIO_INDICES


def load_sequences(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Load sequences from HDF5 or NPY format.

    Returns:
        (temporal_sequences, context_features, metadata_df) or (None, None, None) on error
    """
    temporal = None
    context = None
    metadata = None

    if path.suffix == '.h5':
        # HDF5 format
        try:
            import h5py
            with h5py.File(path, 'r') as f:
                if 'sequences' in f:
                    temporal = f['sequences'][:]
                if 'context' in f:
                    context = f['context'][:]
                elif 'context_features' in f:
                    context = f['context_features'][:]
            logger.info(f"Loaded HDF5: {path}")

            # Try to load metadata
            meta_path = path.parent / 'metadata.parquet'
            if meta_path.exists():
                metadata = pd.read_parquet(meta_path)

        except Exception as e:
            logger.error(f"Failed to load HDF5 {path}: {e}")
            return None, None, None

    elif path.suffix == '.npy':
        # NPY format - look for companion files
        try:
            temporal = np.load(path)
            logger.info(f"Loaded NPY: {path}")

            # Look for context file
            context_path = path.parent / path.name.replace('sequences', 'context')
            if not context_path.exists():
                context_path = path.parent / 'context_features.npy'
            if context_path.exists():
                context = np.load(context_path)
                logger.info(f"Loaded context: {context_path}")

            # Try to load metadata
            meta_path = path.parent / 'metadata.parquet'
            if meta_path.exists():
                metadata = pd.read_parquet(meta_path)

        except Exception as e:
            logger.error(f"Failed to load NPY {path}: {e}")
            return None, None, None
    else:
        logger.error(f"Unsupported format: {path.suffix}")
        return None, None, None

    return temporal, context, metadata


def find_sequences_file(seq_dir: Path) -> Optional[Path]:
    """Find sequences file in directory."""
    # Priority order
    candidates = [
        seq_dir / 'sequences.h5',
        seq_dir / 'sequences.npy',
    ]

    # Also check for timestamped files
    h5_files = list(seq_dir.glob('sequences_*.h5'))
    npy_files = list(seq_dir.glob('sequences_*.npy'))

    # Sort by modification time (newest first)
    h5_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    npy_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    candidates.extend(h5_files)
    candidates.extend(npy_files)

    for path in candidates:
        if path.exists():
            return path

    return None


def check_array_integrity(
    arr: np.ndarray,
    name: str,
    feature_names: List[str] = None
) -> Dict:
    """
    Check array for NaN/Inf values.

    Returns:
        Dict with integrity stats
    """
    results = {
        'name': name,
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'total_elements': arr.size,
        'nan_count': 0,
        'inf_count': 0,
        'neginf_count': 0,
        'clean': True,
        'problem_features': []
    }

    # Count issues
    results['nan_count'] = int(np.isnan(arr).sum())
    results['inf_count'] = int(np.isinf(arr).sum()) - int(np.isneginf(arr).sum())
    results['neginf_count'] = int(np.isneginf(arr).sum())

    total_issues = results['nan_count'] + results['inf_count'] + results['neginf_count']
    results['clean'] = (total_issues == 0)
    results['issue_pct'] = 100.0 * total_issues / arr.size if arr.size > 0 else 0.0

    # Per-feature analysis if 2D or 3D
    if len(arr.shape) >= 2 and feature_names:
        feature_dim = arr.shape[-1]
        if feature_dim == len(feature_names):
            for i, fname in enumerate(feature_names):
                if len(arr.shape) == 3:
                    feature_data = arr[:, :, i]
                else:
                    feature_data = arr[:, i]

                nan_cnt = int(np.isnan(feature_data).sum())
                inf_cnt = int(np.isinf(feature_data).sum())

                if nan_cnt > 0 or inf_cnt > 0:
                    results['problem_features'].append({
                        'index': i,
                        'name': fname,
                        'nan_count': nan_cnt,
                        'inf_count': inf_cnt,
                        'nan_pct': 100.0 * nan_cnt / feature_data.size
                    })

    return results


def validate_sequences(
    sequences_path: Optional[Path] = None,
    seq_dir: Optional[Path] = None,
    verbose: bool = True
) -> int:
    """
    Validate sequence data integrity.

    Args:
        sequences_path: Direct path to sequences file
        seq_dir: Directory containing sequences
        verbose: Print detailed output

    Returns:
        Exit code (0=clean, 1=issues found, 2=load error)
    """
    # Find sequences file
    if sequences_path is None:
        if seq_dir is None:
            logger.error("Must specify --sequences or --seq-dir")
            return 2
        sequences_path = find_sequences_file(seq_dir)
        if sequences_path is None:
            logger.error(f"No sequences file found in {seq_dir}")
            return 2

    if not sequences_path.exists():
        logger.error(f"File not found: {sequences_path}")
        return 2

    # Load data
    logger.info("=" * 70)
    logger.info("SEQUENCE INTEGRITY VALIDATION (Pre-Scaling Check)")
    logger.info("=" * 70)

    temporal, context, metadata = load_sequences(sequences_path)

    if temporal is None:
        logger.error("Failed to load temporal sequences")
        return 2

    all_clean = True

    # Check temporal sequences
    logger.info("")
    logger.info("-" * 50)
    logger.info("TEMPORAL SEQUENCES")
    logger.info("-" * 50)

    temporal_results = check_array_integrity(temporal, "temporal", TEMPORAL_FEATURES)

    logger.info(f"Shape: {temporal_results['shape']}")
    logger.info(f"Dtype: {temporal_results['dtype']}")
    logger.info(f"Total elements: {temporal_results['total_elements']:,}")

    if temporal_results['clean']:
        logger.info("[OK] No NaN/Inf values found")
    else:
        all_clean = False
        logger.error(f"[FAIL] Found {temporal_results['nan_count']:,} NaN, "
                    f"{temporal_results['inf_count']:,} +Inf, "
                    f"{temporal_results['neginf_count']:,} -Inf")
        logger.error(f"       Issue rate: {temporal_results['issue_pct']:.4f}%")

        if temporal_results['problem_features']:
            logger.error("       Problem features:")
            for pf in temporal_results['problem_features']:
                logger.error(f"         [{pf['index']}] {pf['name']}: "
                           f"{pf['nan_count']} NaN, {pf['inf_count']} Inf "
                           f"({pf['nan_pct']:.2f}%)")

    # Check context features
    if context is not None:
        logger.info("")
        logger.info("-" * 50)
        logger.info("CONTEXT FEATURES")
        logger.info("-" * 50)

        context_results = check_array_integrity(context, "context", CONTEXT_FEATURES)

        logger.info(f"Shape: {context_results['shape']}")
        logger.info(f"Dtype: {context_results['dtype']}")
        logger.info(f"Total elements: {context_results['total_elements']:,}")

        if context_results['clean']:
            logger.info("[OK] No NaN/Inf values found")
        else:
            all_clean = False
            logger.error(f"[FAIL] Found {context_results['nan_count']:,} NaN, "
                        f"{context_results['inf_count']:,} +Inf, "
                        f"{context_results['neginf_count']:,} -Inf")
            logger.error(f"       Issue rate: {context_results['issue_pct']:.4f}%")

            if context_results['problem_features']:
                logger.error("       Problem features:")
                for pf in context_results['problem_features']:
                    is_log_diff = pf['index'] in LOG_DIFF_CONTEXT_INDICES
                    tag = " [LOG-DIFF]" if is_log_diff else ""
                    logger.error(f"         [{pf['index']}] {pf['name']}{tag}: "
                               f"{pf['nan_count']} NaN, {pf['inf_count']} Inf "
                               f"({pf['nan_pct']:.2f}%)")

                # Check if LOG-DIFF features have issues
                log_diff_issues = [pf for pf in context_results['problem_features']
                                  if pf['index'] in LOG_DIFF_CONTEXT_INDICES]
                if log_diff_issues:
                    logger.error("")
                    logger.error("[CRITICAL] LOG-DIFF features have NaN/Inf!")
                    logger.error("           This means log_diff() transformation is not being applied.")
                    logger.error("           Check config/context_features.py log_diff() function.")
    else:
        logger.warning("No context features found in file")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    if all_clean:
        logger.info("[PASS] All sequence data is clean (no NaN/Inf)")
        logger.info("       Safe to proceed with training")
        return 0
    else:
        logger.error("[FAIL] Found NaN/Inf values in sequence data")
        logger.error("       DO NOT proceed with training until fixed")
        logger.error("")
        logger.error("Common causes:")
        logger.error("  1. Volume ratio features not using log_diff()")
        logger.error("  2. Division by zero in feature calculation")
        logger.error("  3. Missing data not properly defaulted")
        logger.error("")
        logger.error("Fix: Regenerate sequences with updated feature extractors")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Validate sequence data integrity (check for NaN/Inf)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check specific HDF5 file
    python scripts/validate_sequence_integrity.py --sequences output/sequences/eu/sequences.h5

    # Check latest file in directory
    python scripts/validate_sequence_integrity.py --seq-dir output/sequences/eu

Exit codes:
    0 = Clean (no NaN/Inf) - safe to train
    1 = Found NaN/Inf values - DO NOT train
    2 = File not found or load error
"""
    )

    parser.add_argument('--sequences', type=Path,
                        help='Path to sequences file (HDF5 or NPY)')
    parser.add_argument('--seq-dir', type=Path,
                        help='Directory containing sequences (finds latest)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    exit_code = validate_sequences(
        sequences_path=args.sequences,
        seq_dir=args.seq_dir,
        verbose=not args.quiet
    )

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
