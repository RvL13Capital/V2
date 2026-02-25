"""
Robust Pipeline Orchestrator
============================
Orchestrates sequence generation and training with strict artifact contracts.

Fixes Fragility by:
1. Bridging the gap: Generates the mandatory 'pattern_ids.npy' between steps.
2. Enforcing contracts: Verifies 'norm_params.json' exists before succeeding.
3. Failing fast: Uses check=True to catch subprocess errors immediately.
4. Linking artifacts: Bundles Model + Norm Params for deterministic inference.

Usage:
    python pipeline/run_robust.py --eu --both
    python pipeline/run_robust.py --us --train --epochs 100
    python pipeline/run_robust.py --eu --generate --verbose
"""

import os
import sys
import argparse
import subprocess
import logging
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# Setup Logging
logging.basicConfig(level=logging.INFO, format='[ORCHESTRATOR] %(message)s')
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when pipeline artifact contracts are violated."""
    pass


def run_step(cmd: list, step_name: str):
    """
    Executes a step in a subprocess to ensure memory isolation.
    Uses check=True to fail fast on non-zero exit codes.
    """
    logger.info(f"STARTING: {step_name}")
    try:
        # Set alloc conf to reduce fragmentation during low-mem runs
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        subprocess.run(cmd, check=True, env=env)
        logger.info(f"COMPLETED: {step_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {step_name} (Exit Code: {e.returncode})")
        sys.exit(e.returncode)


def bridge_metadata_gap(metadata_path: Path, output_dir: Path):
    """
    MANDATORY INTERMEDIATE STEP:
    01_generate produces Parquet metadata.
    02_train strictly requires 'pattern_ids.npy' for the pattern-aware split.
    This function bridges that gap, preventing the Trainer from failing silently.
    """
    logger.info("BRIDGING: Generating pattern_ids.npy from metadata...")
    try:
        df = pd.read_parquet(metadata_path)
        if 'pattern_id' not in df.columns:
            # Fallback if pattern_id missing (e.g., legacy data)
            logger.warning("   pattern_id missing, synthesizing...")
            df['pattern_id'] = df.apply(
                lambda x: f"{x.get('ticker', 'UNKNOWN')}_{x.get('end_date', '0')}",
                axis=1
            )

        pattern_ids = df['pattern_id'].values.astype(str)
        output_path = output_dir / "pattern_ids.npy"
        np.save(output_path, pattern_ids)
        logger.info(f"   Saved {output_path}")
    except Exception as e:
        logger.error(f"   Bridge failed: {e}")
        sys.exit(1)


def find_latest_artifacts(output_dir: Path):
    """
    Locates the most recent sequence files to verify Step 1 success.
    Returns dict with paths or None if not found.
    Supports both .npy and HDF5 formats (--claude-safe skips .npy export).
    """
    try:
        # Check for standard naming first (regional dirs use this)
        standard_seq = output_dir / "sequences.npy"
        standard_labels = output_dir / "labels.npy"
        standard_meta = output_dir / "metadata.parquet"

        if standard_seq.exists() and standard_labels.exists():
            return {
                'seq': standard_seq,
                'labels': standard_labels,
                'meta': standard_meta if standard_meta.exists() else None,
                'format': 'npy'
            }

        # Check for HDF5 format (produced by --claude-safe or --skip-npy-export)
        h5_files = sorted(output_dir.glob("sequences_*.h5"), key=lambda x: x.stat().st_mtime)
        if h5_files:
            latest_h5 = h5_files[-1]
            timestamp = latest_h5.stem.replace("sequences_", "")
            meta_path = output_dir / f"metadata_{timestamp}.parquet"
            return {
                'seq': latest_h5,
                'labels': latest_h5,  # Same file for HDF5
                'meta': meta_path if meta_path.exists() else None,
                'format': 'hdf5',
                'h5': latest_h5
            }

        # Fall back to timestamped .npy files
        seq_files = sorted(output_dir.glob("sequences_*.npy"), key=lambda x: x.stat().st_mtime)
        if not seq_files:
            return None

        latest_seq = seq_files[-1]
        timestamp = latest_seq.stem.replace("sequences_", "")

        # Derive others based on timestamp to ensure consistency
        artifacts = {
            'seq': latest_seq,
            'labels': output_dir / f"labels_{timestamp}.npy",
            'meta': output_dir / f"metadata_{timestamp}.parquet"
        }

        # Validate existence
        for k, v in artifacts.items():
            if v and not v.exists():
                if k == 'meta':
                    artifacts['meta'] = None  # Metadata optional for some flows
                else:
                    raise PipelineError(f"Missing artifact: {v}")

        return artifacts
    except IndexError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Robust Pipeline Orchestrator")
    parser.add_argument("--eu", action="store_true", help="EU Region")
    parser.add_argument("--us", action="store_true", help="US Region")
    parser.add_argument("--generate", action="store_true", help="Run sequence generation")
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--both", action="store_true", help="Run both generation and training")
    parser.add_argument("--bear-aware", action="store_true",
                        help="Use bear-aware split: Train<2023, Val=2023, Test=2024+")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--claude-safe", action="store_true",
                        help="CRITICAL: Use when running in Claude Code terminal to prevent OOM/context overflow")
    args = parser.parse_args()

    # Configuration
    quiet = not args.verbose
    region = 'eu' if args.eu else 'us' if args.us else 'global'

    # Path Setup
    base_dir = Path("output")
    if region == 'eu':
        input_patterns = base_dir / "eu_patterns_expanded.parquet"
        seq_dir = base_dir / "sequences" / "eu"
        model_name = "eu_model"
    elif region == 'us':
        input_patterns = base_dir / "us_patterns.parquet"
        seq_dir = base_dir / "sequences" / "us"
        model_name = "us_model"
    else:
        input_patterns = base_dir / "patterns_v17_clean.parquet"
        seq_dir = base_dir / "sequences"
        model_name = "production_model"

    # Ensure output directories exist
    seq_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: GENERATION
    # ---------------------------------------------------------
    if args.generate or args.both:
        if not input_patterns.exists():
            logger.error(f"Input file missing: {input_patterns}")
            sys.exit(1)

        cmd_gen = [
            sys.executable, "pipeline/01_generate_sequences.py",
            "--input", str(input_patterns),
            "--output-dir", str(seq_dir),
            "--chunk-size", "50"
        ]
        if args.claude_safe:
            cmd_gen.append("--claude-safe")  # Prevents OOM and context overflow
        elif quiet:
            cmd_gen.append("--quiet")

        run_step(cmd_gen, "Sequence Generation")

        # Artifact Resolution
        artifacts = find_latest_artifacts(seq_dir)
        if not artifacts:
            logger.error("Generation reported success but produced no files!")
            sys.exit(1)

        # *** FIX THE GAP ***
        # Create pattern_ids.npy which 02_train requires but 01_gen doesn't make
        if artifacts['meta']:
            bridge_metadata_gap(artifacts['meta'], seq_dir)
        else:
            logger.warning("No metadata found, skipping pattern_ids bridge")

    # ---------------------------------------------------------
    # STEP 2: TRAINING
    # ---------------------------------------------------------
    if args.train or args.both:
        # Resolve artifacts again (in case we skipped generation step)
        artifacts = find_latest_artifacts(seq_dir)
        if not artifacts:
            logger.error("No sequence files found for training.")
            sys.exit(1)

        # Verify Bridge Artifact exists
        pattern_ids_path = seq_dir / "pattern_ids.npy"
        if not pattern_ids_path.exists() and artifacts['meta']:
            bridge_metadata_gap(artifacts['meta'], seq_dir)

        cmd_train = [
            sys.executable, "pipeline/02_train_temporal.py",
            "--sequences", str(artifacts['seq']),
            "--labels", str(artifacts['labels']),
            "--model-name", model_name,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--use-asl",
            "--use-inverse-freq-weights",
            "--temporal-split"
        ]

        # Add metadata if available
        if artifacts['meta'] and artifacts['meta'].exists():
            cmd_train.extend(["--metadata", str(artifacts['meta'])])

        if args.bear_aware:
            cmd_train.append("--bear-aware-split")
        if args.claude_safe:
            cmd_train.append("--quiet")  # Training doesn't have --claude-safe yet, use --quiet
        elif quiet:
            cmd_train.append("--quiet")

        run_step(cmd_train, "Model Training")

        # ---------------------------------------------------------
        # STEP 3: ARTIFACT VALIDATION (Normalization Check)
        # ---------------------------------------------------------
        # 02_train saves normalization params with a random timestamp.
        # We must identify and link them to the model to prevent "Skipped Steps".
        models_dir = base_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Find the most recently created norm_params json
            norm_files = sorted(models_dir.glob("norm_params_*.json"), key=lambda x: x.stat().st_mtime)
            if not norm_files:
                raise IndexError("No norm_params files found")

            latest_norm = norm_files[-1]

            # Create a "Production Link" so Inference always finds it
            # e.g., eu_model.pt -> eu_model_norm.json
            linked_norm = models_dir / f"{model_name}_norm.json"
            shutil.copy(latest_norm, linked_norm)

            logger.info(f"SECURED Normalization Params: {linked_norm.name}")
        except IndexError:
            logger.error("CRITICAL: Training finished but Normalization Params were not found!")
            logger.error("   This implies the normalization step was skipped or failed silently.")
            sys.exit(1)

    # Show usage if no action specified
    if not (args.generate or args.train or args.both):
        print("Usage: python pipeline/run_robust.py [OPTIONS]")
        print("")
        print("Actions:")
        print("  --generate           Generate sequences from patterns")
        print("  --train              Train model on sequences")
        print("  --both               Generate + train")
        print("")
        print("Region:")
        print("  --eu                 EU stocks only")
        print("  --us                 US stocks only")
        print("")
        print("Options:")
        print("  --bear-aware         Use bear-aware split (Val=2023 with bear patterns)")
        print("  --verbose            Show full output")
        print("  --epochs N           Training epochs (default: 50)")
        print("  --batch-size N       Batch size (default: 32)")
        print("")
        print("Examples:")
        print("  python pipeline/run_robust.py --eu --both")
        print("  python pipeline/run_robust.py --us --train --epochs 100")
        print("  python pipeline/run_robust.py --eu --generate --verbose")
        sys.exit(1)


if __name__ == "__main__":
    main()
