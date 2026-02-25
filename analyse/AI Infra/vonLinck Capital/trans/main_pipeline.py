#!/usr/bin/env python3
"""
TRAnS Unified Pipeline Orchestrator
====================================

Comprehensive orchestrator for the TRAnS (Temporal ARchitectural Neural System) pipeline.
Integrates all steps from pattern detection to model training with robust artifact
management and temporal integrity enforcement.

PIPELINE STEPS:
    0. detect   - Pattern detection (candidate registry)
    0b. label   - Structural risk labeling (40-day ripening)
    1. generate - Temporal sequence generation (NMS, physics filter)
    2. train    - Model training (temporal split, CoilFocal loss)

KEY FEATURES:
    - Region-aware configuration (EU/US/GLOBAL)
    - Artifact validation between steps
    - Temporal integrity enforcement (ripening period)
    - Memory-efficient subprocess isolation
    - Both live mode (wait for ripening) and backtest mode (--enable-early-labeling)

USAGE:
    # Full pipeline (EU region) - RECOMMENDED
    python main_pipeline.py --region EU --full-pipeline --apply-nms --apply-physics-filter --mode training --use-coil-focal --epochs 100

    # Individual steps
    python main_pipeline.py --region EU --step detect --tickers data/eu_tickers.txt
    python main_pipeline.py --region EU --step label --enable-early-labeling
    python main_pipeline.py --region EU --step generate --apply-nms --apply-physics-filter --mode training
    python main_pipeline.py --region EU --step train --epochs 100 --use-coil-focal

    # Modes
    python main_pipeline.py --region EU --full-pipeline --dry-run
    python main_pipeline.py --region EU --full-pipeline --claude-safe
    python main_pipeline.py --region EU --full-pipeline --continue-on-error

EXIT CODES:
    0 - Success (all steps completed)
    1 - Error (step failed)
    2 - Partial completion (some steps failed with --continue-on-error)
"""

import os
import sys
import argparse
import subprocess
import logging
import shutil
import json
import time
import traceback
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

OUTCOME_WINDOW_DAYS = 40  # From 00b_label_outcomes.py (structural R-labeling)
ZOMBIE_TIMEOUT_DAYS = 150  # Force Danger if no data after this many days

# Pipeline step definitions
STEP_DETECT = 'detect'
STEP_LABEL = 'label'
STEP_GENERATE = 'generate'
STEP_TRAIN = 'train'

PIPELINE_ORDER = [STEP_DETECT, STEP_LABEL, STEP_GENERATE, STEP_TRAIN]


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with optional file output."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

    return logging.getLogger('main_pipeline')


logger = logging.getLogger('main_pipeline')


# =============================================================================
# REGION CONFIGURATION
# =============================================================================

@dataclass
class RegionConfig:
    """Region-specific configuration for paths and settings."""
    region: str
    baseline_index: Optional[str]
    default_tickers: str
    output_prefix: str
    model_name: str
    seq_dir: Path
    patterns_file: Path
    labeled_patterns_file: Path

    @classmethod
    def from_region(cls, region: str, base_dir: Path = Path('output')) -> 'RegionConfig':
        """Factory method to create region-specific configuration."""
        configs = {
            'EU': {
                'baseline_index': 'EXSA.DE',  # iShares EURO STOXX (no SPY leakage)
                'default_tickers': 'data/eu_tickers.txt',
                'output_prefix': 'eu',
                'model_name': 'eu_model',
                'seq_dir': base_dir / 'sequences' / 'eu',
                'patterns_file': base_dir / 'eu_candidate_patterns.parquet',
                'labeled_patterns_file': base_dir / 'eu_labeled_patterns.parquet',
            },
            'US': {
                'baseline_index': 'RSP',  # Equal-weight S&P 500
                'default_tickers': 'data/us_tickers.txt',
                'output_prefix': 'us',
                'model_name': 'us_model',
                'seq_dir': base_dir / 'sequences' / 'us',
                'patterns_file': base_dir / 'us_candidate_patterns.parquet',
                'labeled_patterns_file': base_dir / 'us_labeled_patterns.parquet',
            },
            'GLOBAL': {
                'baseline_index': None,  # Cohort-based only
                'default_tickers': 'data/all_tickers.txt',
                'output_prefix': 'global',
                'model_name': 'production_model',
                'seq_dir': base_dir / 'sequences',
                'patterns_file': base_dir / 'candidate_patterns.parquet',
                'labeled_patterns_file': base_dir / 'labeled_patterns.parquet',
            }
        }

        if region not in configs:
            raise ValueError(f"Unknown region: {region}. Valid: {list(configs.keys())}")

        cfg = configs[region]
        return cls(
            region=region,
            baseline_index=cfg['baseline_index'],
            default_tickers=cfg['default_tickers'],
            output_prefix=cfg['output_prefix'],
            model_name=cfg['model_name'],
            seq_dir=cfg['seq_dir'],
            patterns_file=cfg['patterns_file'],
            labeled_patterns_file=cfg['labeled_patterns_file'],
        )


# =============================================================================
# ARTIFACT MANAGER
# =============================================================================

class ArtifactManager:
    """
    Handles file discovery, validation, and linking between pipeline steps.

    Key responsibilities:
    - Find latest artifacts (supports both .npy and HDF5 formats)
    - Bridge the pattern_ids.npy gap between generate and train steps
    - Link normalization parameters with trained models
    """

    def __init__(self, config: RegionConfig):
        self.config = config

    def find_latest_artifacts(self, output_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Locates the most recent sequence files.
        Supports both .npy and HDF5 formats (--claude-safe skips .npy export).

        Returns:
            Dict with keys: 'seq', 'labels', 'meta', 'format', 'h5' (if HDF5)
            or None if no artifacts found.
        """
        try:
            # Priority 1: Check for standard naming (regional dirs use this)
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

            # Priority 2: Check for HDF5 format (produced by --claude-safe or --skip-npy-export)
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

            # Priority 3: Fall back to timestamped .npy files
            seq_files = sorted(output_dir.glob("sequences_*.npy"), key=lambda x: x.stat().st_mtime)
            if not seq_files:
                return None

            latest_seq = seq_files[-1]
            timestamp = latest_seq.stem.replace("sequences_", "")

            artifacts = {
                'seq': latest_seq,
                'labels': output_dir / f"labels_{timestamp}.npy",
                'meta': output_dir / f"metadata_{timestamp}.parquet",
                'format': 'npy'
            }

            # Validate labels exist
            if not artifacts['labels'].exists():
                logger.warning(f"Labels file missing: {artifacts['labels']}")
                return None

            # Metadata is optional for some flows
            if artifacts['meta'] and not artifacts['meta'].exists():
                artifacts['meta'] = None

            return artifacts

        except Exception as e:
            logger.error(f"Error finding artifacts: {e}")
            return None

    def bridge_metadata_gap(self, metadata_path: Path, output_dir: Path) -> Path:
        """
        MANDATORY INTERMEDIATE STEP:
        01_generate produces Parquet metadata.
        02_train strictly requires 'pattern_ids.npy' for the pattern-aware split.
        This function bridges that gap, preventing the Trainer from failing silently.

        Args:
            metadata_path: Path to metadata.parquet from sequence generation
            output_dir: Directory to write pattern_ids.npy

        Returns:
            Path to the created pattern_ids.npy file
        """
        logger.info("BRIDGING: Generating pattern_ids.npy from metadata...")

        try:
            df = pd.read_parquet(metadata_path)

            if 'pattern_id' not in df.columns:
                # Synthesize pattern_id from ticker + end_date (fallback for legacy data)
                logger.warning("   pattern_id column missing, synthesizing from ticker + end_date...")
                df['pattern_id'] = df.apply(
                    lambda x: f"{x.get('ticker', 'UNKNOWN')}_{x.get('end_date', '0')}",
                    axis=1
                )

            pattern_ids = df['pattern_id'].values.astype(str)
            output_path = output_dir / "pattern_ids.npy"
            np.save(output_path, pattern_ids)

            logger.info(f"   Saved {len(pattern_ids)} pattern IDs to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"   Bridge failed: {e}")
            raise

    def link_normalization_params(self, models_dir: Path, model_name: str) -> Optional[Path]:
        """
        Link timestamped norm_params_*.json to {model_name}_norm.json.
        Ensures inference always finds the correct normalization parameters.

        Args:
            models_dir: Directory containing model checkpoints
            model_name: Base name for the model (e.g., 'eu_model')

        Returns:
            Path to the linked normalization file, or None if no norm_params found
        """
        try:
            # Find the most recently created norm_params json
            norm_files = sorted(
                models_dir.glob("norm_params_*.json"),
                key=lambda x: x.stat().st_mtime
            )

            if not norm_files:
                logger.warning("No norm_params_*.json files found")
                return None

            latest_norm = norm_files[-1]

            # Create a "Production Link" so inference always finds it
            linked_norm = models_dir / f"{model_name}_norm.json"
            shutil.copy(latest_norm, linked_norm)

            logger.info(f"SECURED Normalization Params: {linked_norm.name}")
            return linked_norm

        except Exception as e:
            logger.error(f"Failed to link normalization params: {e}")
            return None

    def validate_artifacts_for_step(self, step: str) -> Tuple[bool, str]:
        """
        Validate that required artifacts exist for a given step.

        Returns:
            (is_valid, message) tuple
        """
        validations = {
            STEP_DETECT: lambda: (True, "No prerequisite artifacts needed"),
            STEP_LABEL: lambda: self._validate_label_inputs(),
            STEP_GENERATE: lambda: self._validate_generate_inputs(),
            STEP_TRAIN: lambda: self._validate_train_inputs(),
        }

        if step not in validations:
            return False, f"Unknown step: {step}"

        return validations[step]()

    def _validate_label_inputs(self) -> Tuple[bool, str]:
        """Check that candidate patterns exist for labeling."""
        if not self.config.patterns_file.exists():
            return False, f"Missing candidate patterns: {self.config.patterns_file}"
        return True, "Candidate patterns found"

    def _validate_generate_inputs(self) -> Tuple[bool, str]:
        """Check that labeled patterns exist for sequence generation."""
        if not self.config.labeled_patterns_file.exists():
            return False, f"Missing labeled patterns: {self.config.labeled_patterns_file}"
        return True, "Labeled patterns found"

    def _validate_train_inputs(self) -> Tuple[bool, str]:
        """Check that sequences and metadata exist for training."""
        artifacts = self.find_latest_artifacts(self.config.seq_dir)
        if not artifacts:
            return False, f"No sequence artifacts found in {self.config.seq_dir}"
        if not artifacts.get('meta'):
            return False, "Metadata parquet not found (required for temporal split)"
        return True, f"Artifacts found: {artifacts['format']} format"


# =============================================================================
# TEMPORAL INTEGRITY CHECKER
# =============================================================================

class TemporalIntegrityChecker:
    """
    Enforces temporal integrity constraints to prevent look-ahead bias.

    Key responsibilities:
    - Check pattern ripeness (outcome window elapsed)
    - Support both live mode (wait for ripening) and backtest mode (early labeling)
    - Validate no FUTURE_ columns leak into model inputs
    """

    def __init__(self, outcome_window_days: int = OUTCOME_WINDOW_DAYS):
        self.outcome_window_days = outcome_window_days

    def check_labeling_eligibility(
        self,
        patterns_df: pd.DataFrame,
        reference_date: datetime,
        enable_early_labeling: bool = False
    ) -> Dict[str, Any]:
        """
        Check which patterns are eligible for labeling.

        Args:
            patterns_df: DataFrame with pattern data (must have 'end_date' column)
            reference_date: Date to check ripeness against
            enable_early_labeling: Allow early labeling for patterns with definitive outcomes

        Returns:
            Dict with keys: 'ripe' (DataFrame), 'unripe' (DataFrame),
                           'earliest_ripe_date' (datetime)
        """
        if 'end_date' not in patterns_df.columns:
            raise ValueError("patterns_df must have 'end_date' column")

        patterns_df = patterns_df.copy()
        patterns_df['end_date'] = pd.to_datetime(patterns_df['end_date'])

        # Calculate ripeness
        ripe_mask = patterns_df['end_date'].apply(
            lambda x: (reference_date - x).days >= self.outcome_window_days
        )

        ripe_patterns = patterns_df[ripe_mask]
        unripe_patterns = patterns_df[~ripe_mask]

        # Calculate earliest ripening date
        if len(unripe_patterns) > 0:
            earliest_end = unripe_patterns['end_date'].min()
            earliest_ripe = earliest_end + timedelta(days=self.outcome_window_days)
        else:
            earliest_ripe = None

        result = {
            'ripe': ripe_patterns,
            'unripe': unripe_patterns,
            'earliest_ripe_date': earliest_ripe,
            'ripe_count': len(ripe_patterns),
            'unripe_count': len(unripe_patterns),
        }

        # Log summary
        logger.info(f"Temporal Integrity Check:")
        logger.info(f"  Reference date: {reference_date.date()}")
        logger.info(f"  Outcome window: {self.outcome_window_days} days")
        logger.info(f"  Ripe patterns: {result['ripe_count']}")
        logger.info(f"  Unripe patterns: {result['unripe_count']}")

        if enable_early_labeling:
            logger.info("  Early labeling: ENABLED (backtest mode)")
        else:
            logger.info("  Early labeling: DISABLED (live mode)")
            if earliest_ripe:
                logger.info(f"  Earliest ripening: {earliest_ripe.date()}")

        return result

    def validate_no_lookahead(self, metadata_df: pd.DataFrame) -> bool:
        """
        Validate that FUTURE_ columns are not used as model inputs.
        FUTURE_ columns are OK in metadata for analysis, but should not
        be in sequences or context tensors.

        Args:
            metadata_df: DataFrame to check

        Returns:
            True if validation passes (FUTURE_ columns are metadata-only)
        """
        future_cols = [c for c in metadata_df.columns if c.startswith('FUTURE_')]

        if future_cols:
            logger.info(f"FUTURE_ columns detected (OK for metadata): {future_cols}")

        # Check for suspicious column patterns that might indicate leakage
        suspicious_patterns = ['outcome', 'result', 'gain', 'return', 'profit']
        suspicious_cols = [
            c for c in metadata_df.columns
            if any(p in c.lower() for p in suspicious_patterns)
            and not c.startswith('FUTURE_')
            and c != 'outcome_class'  # This is the target, not a feature
        ]

        if suspicious_cols:
            logger.warning(f"Suspicious columns found (potential leakage): {suspicious_cols}")
            logger.warning("Consider prefixing with FUTURE_ if these contain post-pattern data")

        return True


# =============================================================================
# STEP RUNNER
# =============================================================================

class StepRunner:
    """
    Executes pipeline steps in subprocesses with error handling and timeouts.

    Key features:
    - Memory isolation (subprocess prevents memory leaks between steps)
    - Timeout handling
    - Continue-on-error support
    - Output capture for logging
    """

    def __init__(self, continue_on_error: bool = False, timeout_ms: int = 600000):
        self.continue_on_error = continue_on_error
        self.timeout_ms = timeout_ms

    def run_step(
        self,
        cmd: List[str],
        step_name: str,
        dry_run: bool = False
    ) -> Tuple[int, Optional[str]]:
        """
        Execute a step in a subprocess.

        Args:
            cmd: Command and arguments to execute
            step_name: Name of the step (for logging)
            dry_run: If True, print command without executing

        Returns:
            (exit_code, error_message) tuple
            exit_code: 0=success, 1=error, 2=partial (continue_on_error)
        """
        cmd_str = ' '.join(cmd)

        if dry_run:
            logger.info(f"[DRY RUN] {step_name}:")
            logger.info(f"  {cmd_str}")
            return 0, None

        logger.info("=" * 70)
        logger.info(f"STARTING: {step_name}")
        logger.info("=" * 70)
        logger.info(f"Command: {cmd_str}")

        start_time = time.time()

        # Set environment for memory optimization
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        try:
            result = subprocess.run(
                cmd,
                check=True,
                env=env,
                timeout=self.timeout_ms / 1000,  # Convert ms to seconds
                capture_output=False,  # Let output flow to console
            )

            elapsed = time.time() - start_time
            logger.info(f"COMPLETED: {step_name} ({elapsed:.1f}s)")
            return 0, None

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            error_msg = f"{step_name} failed with exit code {e.returncode}"
            logger.error(f"FAILED: {error_msg} ({elapsed:.1f}s)")

            if self.continue_on_error:
                logger.warning(f"Continuing despite error (--continue-on-error)")
                return 2, error_msg
            return 1, error_msg

        except subprocess.TimeoutExpired:
            error_msg = f"{step_name} timed out after {self.timeout_ms}ms"
            logger.error(f"TIMEOUT: {error_msg}")

            if self.continue_on_error:
                return 2, error_msg
            return 1, error_msg

        except Exception as e:
            error_msg = f"{step_name} failed: {str(e)}"
            logger.error(f"ERROR: {error_msg}")
            traceback.print_exc()

            if self.continue_on_error:
                return 2, error_msg
            return 1, error_msg


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class PipelineOrchestrator:
    """
    Main orchestrator for the TRAnS pipeline.

    Manages:
    - Step execution order and dependencies
    - Artifact validation between steps
    - Temporal integrity enforcement
    - Summary generation
    """

    def __init__(
        self,
        config: RegionConfig,
        args: argparse.Namespace
    ):
        self.config = config
        self.args = args
        self.artifact_manager = ArtifactManager(config)
        self.temporal_checker = TemporalIntegrityChecker()
        self.step_runner = StepRunner(
            continue_on_error=getattr(args, 'continue_on_error', False),
            timeout_ms=getattr(args, 'timeout', 600000)
        )

        # Statistics tracking
        self.stats = {}
        self.start_time = None
        self.errors = []

    def run(self, steps: List[str], dry_run: bool = False) -> int:
        """
        Execute the specified pipeline steps.

        Args:
            steps: List of step names to execute
            dry_run: If True, print commands without executing

        Returns:
            Exit code: 0=success, 1=error, 2=partial
        """
        self.start_time = time.time()
        final_exit_code = 0

        logger.info("=" * 70)
        logger.info("TRAnS UNIFIED PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Region: {self.config.region}")
        logger.info(f"Steps: {' -> '.join(steps)}")
        logger.info(f"Dry run: {dry_run}")
        logger.info(f"Continue on error: {self.step_runner.continue_on_error}")
        logger.info("=" * 70)

        # Ensure output directories exist
        if not dry_run:
            self._ensure_directories()

        for step in steps:
            # Validate prerequisites
            if not dry_run:
                is_valid, msg = self.artifact_manager.validate_artifacts_for_step(step)
                if not is_valid and step != STEP_DETECT:
                    logger.error(f"Prerequisite validation failed: {msg}")
                    if not self.step_runner.continue_on_error:
                        return 1
                    final_exit_code = 2
                    self.errors.append(msg)
                    continue

            # Build and execute step command
            cmd = self._build_step_command(step)
            if not cmd:
                logger.error(f"Failed to build command for step: {step}")
                if not self.step_runner.continue_on_error:
                    return 1
                final_exit_code = 2
                continue

            exit_code, error = self.step_runner.run_step(cmd, step, dry_run)

            if exit_code == 1:
                return 1
            elif exit_code == 2:
                final_exit_code = 2
                if error:
                    self.errors.append(error)

            # Post-step processing (artifact bridging, linking)
            if not dry_run and exit_code == 0:
                self._post_step_processing(step)

        # Generate summary
        self._print_summary(dry_run)

        return final_exit_code

    def _ensure_directories(self):
        """Create necessary output directories."""
        dirs = [
            Path('output'),
            Path('output/models'),
            self.config.seq_dir,
            Path('logs'),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _build_step_command(self, step: str) -> Optional[List[str]]:
        """Build the command for a specific step."""
        builders = {
            STEP_DETECT: self._build_detect_cmd,
            STEP_LABEL: self._build_label_cmd,
            STEP_GENERATE: self._build_generate_cmd,
            STEP_TRAIN: self._build_train_cmd,
        }

        if step not in builders:
            logger.error(f"Unknown step: {step}")
            return None

        return builders[step]()

    def _build_detect_cmd(self) -> List[str]:
        """Build command for pattern detection step."""
        cmd = [
            sys.executable, 'pipeline/00_detect_patterns.py',
            '--output', str(self.config.patterns_file),
        ]

        # Ticker source
        tickers = getattr(self.args, 'tickers', None) or self.config.default_tickers
        cmd.extend(['--tickers', str(tickers)])

        # Optional parameters
        if getattr(self.args, 'start_date', None):
            cmd.extend(['--start-date', self.args.start_date])
        if getattr(self.args, 'end_date', None):
            cmd.extend(['--end-date', self.args.end_date])
        if getattr(self.args, 'workers', None):
            cmd.extend(['--workers', str(self.args.workers)])
        if getattr(self.args, 'limit', None):
            cmd.extend(['--limit', str(self.args.limit)])
        if getattr(self.args, 'fast_validation', False):
            cmd.append('--fast-validation')
        if getattr(self.args, 'local_only', False):
            cmd.append('--local-only')

        # Adaptive detection thresholds (Jan 2026)
        if getattr(self.args, 'tightness_zscore', None) is not None:
            cmd.extend(['--tightness-zscore', str(self.args.tightness_zscore)])
        if getattr(self.args, 'min_float_turnover', None) is not None:
            cmd.extend(['--min-float-turnover', str(self.args.min_float_turnover)])

        # Weekly qualification mode (Jan 2026)
        if getattr(self.args, 'use_weekly_qualification', False):
            cmd.append('--use-weekly-qualification')

        # Point-in-time market cap (Jan 2026)
        if getattr(self.args, 'skip_market_cap_api', False):
            cmd.append('--skip-market-cap-api')

        return cmd

    def _build_label_cmd(self) -> List[str]:
        """Build command for labeling step."""
        cmd = [
            sys.executable, 'pipeline/00b_label_outcomes.py',
            '--input', str(self.config.patterns_file),
            '--output', str(self.config.labeled_patterns_file),
        ]

        # Reference date
        if getattr(self.args, 'reference_date', None):
            cmd.extend(['--reference-date', self.args.reference_date])

        # Early labeling (backtest mode)
        if getattr(self.args, 'enable_early_labeling', False):
            cmd.append('--enable-early-labeling')

        if getattr(self.args, 'force_relabel', False):
            cmd.append('--force-relabel')

        return cmd

    def _build_generate_cmd(self) -> List[str]:
        """Build command for sequence generation step."""
        cmd = [
            sys.executable, 'pipeline/01_generate_sequences.py',
            '--input', str(self.config.labeled_patterns_file),
            '--output-dir', str(self.config.seq_dir),
            '--region', self.config.region,
        ]

        # Add baseline index if specified
        if self.config.baseline_index:
            cmd.extend(['--baseline-index', self.config.baseline_index])

        # Mode (training preserves dormant, inference strict)
        mode = getattr(self.args, 'mode', 'training')
        cmd.extend(['--mode', mode])

        # Required filters for production
        if getattr(self.args, 'apply_nms', False):
            cmd.append('--apply-nms')
        if getattr(self.args, 'apply_physics_filter', False):
            cmd.append('--apply-physics-filter')

        # Optional parameters
        if getattr(self.args, 'chunk_size', None):
            cmd.extend(['--metadata-batch-size', str(self.args.chunk_size)])

        # Streaming mode
        if getattr(self.args, 'parallel_streaming', False):
            cmd.append('--parallel-streaming')
            workers = getattr(self.args, 'streaming_workers', 4)
            cmd.extend(['--streaming-workers', str(workers)])

        # Memory efficiency modes
        if getattr(self.args, 'claude_safe', False):
            cmd.append('--claude-safe')
        elif getattr(self.args, 'skip_npy_export', False):
            cmd.append('--skip-npy-export')

        # Additional options
        if getattr(self.args, 'use_event_anchor', False):
            cmd.append('--use-event-anchor')
        if getattr(self.args, 'calculate_coil_features', False):
            cmd.append('--calculate-coil-features')
        if getattr(self.args, 'limit', None):
            cmd.extend(['--limit', str(self.args.limit)])

        # Cross-ticker deduplication (Jan 2026) - NOW MANDATORY
        # Deduplication is always applied in 01_generate_sequences.py
        # These flags are kept for backwards compatibility but dedup runs regardless
        overlap_days = getattr(self.args, 'dedup_overlap_days', 5)
        cmd.extend(['--dedup-overlap-days', str(overlap_days)])

        # Dollar bar support (Jan 2026)
        # Dollar bar experiment failed (99% data loss) - using daily bars only
        # See Section XII of the technical report for details

        return cmd

    def _build_train_cmd(self) -> List[str]:
        """Build command for training step."""
        # Find artifacts
        artifacts = self.artifact_manager.find_latest_artifacts(self.config.seq_dir)
        if not artifacts:
            logger.error(f"No sequence artifacts found in {self.config.seq_dir}")
            return None

        cmd = [
            sys.executable, 'pipeline/02_train_temporal.py',
            '--sequences', str(artifacts['seq']),
            '--model-name', self.config.model_name,
        ]

        # Add labels if NPY format
        if artifacts['format'] == 'npy':
            cmd.extend(['--labels', str(artifacts['labels'])])

        # Add metadata if available
        if artifacts.get('meta'):
            cmd.extend(['--metadata', str(artifacts['meta'])])

        # Training hyperparameters
        epochs = getattr(self.args, 'epochs', 50)
        cmd.extend(['--epochs', str(epochs)])

        batch_size = getattr(self.args, 'batch_size', 32)
        cmd.extend(['--batch-size', str(batch_size)])

        # Loss function configuration
        if getattr(self.args, 'use_coil_focal', False):
            cmd.append('--use-coil-focal')
        if getattr(self.args, 'use_asl', False):
            cmd.append('--use-asl')
        if getattr(self.args, 'use_inverse_freq_weights', False):
            cmd.append('--use-inverse-freq-weights')
        if getattr(self.args, 'use_volume_weighted_loss', False):
            cmd.append('--use-volume-weighted-loss')

        # Split configuration
        if getattr(self.args, 'bear_aware_split', False):
            cmd.append('--bear-aware-split')
        if getattr(self.args, 'disable_trinity', False):
            cmd.append('--disable-trinity')

        # Augmentation
        if getattr(self.args, 'use_augmentation', False):
            cmd.append('--use-augmentation')
        if getattr(self.args, 'use_window_jitter', False):
            cmd.append('--use-window-jitter')

        # Memory efficiency
        if getattr(self.args, 'lazy_loading', False):
            cmd.append('--lazy-loading')
        if getattr(self.args, 'claude_safe', False) or getattr(self.args, 'quiet', False):
            cmd.append('--quiet')

        # Advanced options
        if getattr(self.args, 'use_conditioned_lstm', False):
            cmd.append('--use-conditioned-lstm')
        if getattr(self.args, 'compile', False):
            cmd.append('--compile')

        # Ensemble training
        ensemble_size = getattr(self.args, 'ensemble_size', 1)
        if ensemble_size > 1:
            cmd.extend(['--ensemble-size', str(ensemble_size)])
            if getattr(self.args, 'ensemble_vary_arch', False):
                cmd.append('--ensemble-vary-arch')
            if getattr(self.args, 'use_dirichlet_calibration', False):
                cmd.append('--use-dirichlet-calibration')

        return cmd

    def _post_step_processing(self, step: str):
        """Perform post-step processing like artifact bridging and linking."""
        if step == STEP_GENERATE:
            # Bridge the metadata gap for training
            artifacts = self.artifact_manager.find_latest_artifacts(self.config.seq_dir)
            if artifacts and artifacts.get('meta'):
                try:
                    self.artifact_manager.bridge_metadata_gap(
                        artifacts['meta'],
                        self.config.seq_dir
                    )
                except Exception as e:
                    logger.error(f"Failed to bridge metadata gap: {e}")

        elif step == STEP_TRAIN:
            # Link normalization parameters
            models_dir = Path('output/models')
            self.artifact_manager.link_normalization_params(
                models_dir,
                self.config.model_name
            )

    def _print_summary(self, dry_run: bool):
        """Print pipeline execution summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Region: {self.config.region}")
        logger.info(f"Total time: {elapsed:.1f}s")

        if dry_run:
            logger.info("Mode: DRY RUN (no commands executed)")
        else:
            # Memory usage
            try:
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"Peak memory: {mem_mb:.1f} MB")
            except:
                pass

        if self.errors:
            logger.warning(f"Errors encountered: {len(self.errors)}")
            for err in self.errors:
                logger.warning(f"  - {err}")

        # Artifact summary
        artifacts = self.artifact_manager.find_latest_artifacts(self.config.seq_dir)
        if artifacts:
            logger.info(f"Sequence artifacts: {artifacts['format']} format")
            logger.info(f"  Sequences: {artifacts['seq']}")
            if artifacts.get('meta'):
                logger.info(f"  Metadata: {artifacts['meta']}")

        # Model checkpoint
        models_dir = Path('output/models')
        model_files = list(models_dir.glob(f"{self.config.model_name}_*.pt"))
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Model checkpoint: {latest_model}")

        norm_file = models_dir / f"{self.config.model_name}_norm.json"
        if norm_file.exists():
            logger.info(f"Norm params: {norm_file}")

        logger.info("=" * 70)


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all options."""
    parser = argparse.ArgumentParser(
        description='TRAnS Unified Pipeline Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (EU region) - RECOMMENDED
    python main_pipeline.py --region EU --full-pipeline --apply-nms --apply-physics-filter --mode training --use-coil-focal --epochs 100

    # Individual steps
    python main_pipeline.py --region EU --step detect --tickers data/eu_tickers.txt
    python main_pipeline.py --region EU --step label --enable-early-labeling
    python main_pipeline.py --region EU --step generate --apply-nms --apply-physics-filter --mode training
    python main_pipeline.py --region EU --step train --epochs 100 --use-coil-focal

    # Dry run (preview commands)
    python main_pipeline.py --region EU --full-pipeline --dry-run

    # Production pipeline
    python main_pipeline.py --region EU --full-pipeline --apply-nms --apply-physics-filter --mode training --use-coil-focal --epochs 100 --claude-safe
        """
    )

    # =========================================================================
    # PIPELINE CONTROL
    # =========================================================================
    pipeline_group = parser.add_argument_group('Pipeline Control')
    pipeline_group.add_argument(
        '--region', type=str, default='EU', choices=['EU', 'US', 'GLOBAL'],
        help='Region for configuration (default: EU)'
    )
    pipeline_group.add_argument(
        '--full-pipeline', action='store_true',
        help='Run all pipeline steps (detect -> label -> generate -> train)'
    )
    pipeline_group.add_argument(
        '--step', type=str, choices=[STEP_DETECT, STEP_LABEL, STEP_GENERATE, STEP_TRAIN],
        help='Run a single step'
    )
    pipeline_group.add_argument(
        '--steps', type=str, nargs='+',
        choices=[STEP_DETECT, STEP_LABEL, STEP_GENERATE, STEP_TRAIN],
        help='Run specific steps in order'
    )
    pipeline_group.add_argument(
        '--from-step', type=str, choices=[STEP_DETECT, STEP_LABEL, STEP_GENERATE, STEP_TRAIN],
        help='Start from this step (inclusive)'
    )

    # =========================================================================
    # EXECUTION MODES
    # =========================================================================
    mode_group = parser.add_argument_group('Execution Modes')
    mode_group.add_argument(
        '--dry-run', action='store_true',
        help='Print commands without executing'
    )
    mode_group.add_argument(
        '--continue-on-error', action='store_true',
        help='Continue pipeline even if a step fails'
    )
    mode_group.add_argument(
        '--timeout', type=int, default=600000,
        help='Step timeout in milliseconds (default: 600000 = 10 min)'
    )
    mode_group.add_argument(
        '--claude-safe', action='store_true',
        help='Memory-efficient mode for Claude Code terminal'
    )
    mode_group.add_argument(
        '--quiet', '-q', action='store_true',
        help='Minimal output'
    )
    mode_group.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )

    # =========================================================================
    # PATTERN DETECTION (Step 0)
    # =========================================================================
    detect_group = parser.add_argument_group('Pattern Detection (Step 0)')
    detect_group.add_argument(
        '--tickers', type=str,
        help='Ticker source: file path, comma-separated, or "ALL"'
    )
    detect_group.add_argument(
        '--start-date', type=str,
        help='Start date for scanning (YYYY-MM-DD)'
    )
    detect_group.add_argument(
        '--end-date', type=str,
        help='End date for scanning (YYYY-MM-DD)'
    )
    detect_group.add_argument(
        '--workers', type=int, default=4,
        help='Number of parallel workers (default: 4)'
    )
    detect_group.add_argument(
        '--limit', type=int,
        help='Limit number of tickers (for testing)'
    )
    detect_group.add_argument(
        '--fast-validation', action='store_true',
        help='Skip expensive mock data detection'
    )
    detect_group.add_argument(
        '--local-only', action='store_true',
        help='Use local data only (no GCS)'
    )
    # Adaptive detection thresholds (Jan 2026)
    detect_group.add_argument(
        '--tightness-zscore', type=float, default=None,
        help='Max Z-Score for BBW (relative tightness). E.g., -1.0 = 1 std dev tighter than avg'
    )
    detect_group.add_argument(
        '--min-float-turnover', type=float, default=None,
        help='Minimum 20d float turnover (e.g., 0.10 = 10%% of float traded)'
    )
    # Weekly qualification mode (Jan 2026)
    detect_group.add_argument(
        '--use-weekly-qualification', action='store_true', default=False,
        help='Use WEEKLY candles for 10-week qualification (vs daily 10-day). '
             'Requires ~4 years of data. Finds longer-term consolidations.'
    )
    # Point-in-time market cap (Jan 2026)
    detect_group.add_argument(
        '--skip-market-cap-api', action='store_true', default=False,
        help='Skip market cap API calls, use cached shares × price for point-in-time (PIT) '
             'estimation. Much faster for EU stocks (99%% API failure rate) and avoids '
             'look-ahead bias when analyzing historical 2015-2026 data.'
    )

    # =========================================================================
    # LABELING (Step 0b)
    # =========================================================================
    label_group = parser.add_argument_group('Labeling (Step 0b)')
    label_group.add_argument(
        '--reference-date', type=str,
        help='Reference date for ripeness check (default: today)'
    )
    label_group.add_argument(
        '--enable-early-labeling', action='store_true',
        help='Enable early labeling for backtesting (maintains temporal integrity)'
    )
    label_group.add_argument(
        '--force-relabel', action='store_true',
        help='Re-label patterns that already have outcome_class'
    )

    # =========================================================================
    # SEQUENCE GENERATION (Step 1)
    # =========================================================================
    generate_group = parser.add_argument_group('Sequence Generation (Step 1)')
    generate_group.add_argument(
        '--apply-nms', action='store_true',
        help='Apply NMS filter (71%% overlap reduction) - REQUIRED for production'
    )
    generate_group.add_argument(
        '--apply-physics-filter', action='store_true',
        help='Apply physics filter (Lobster Trap protection) - REQUIRED for production'
    )
    generate_group.add_argument(
        '--chunk-size', type=int, default=50,
        help='Batch size for sequence generation (default: 50)'
    )
    generate_group.add_argument(
        '--parallel-streaming', action='store_true',
        help='Use parallel streaming mode'
    )
    generate_group.add_argument(
        '--streaming-workers', type=int, default=4,
        help='Number of streaming workers (default: 4)'
    )
    generate_group.add_argument(
        '--skip-npy-export', action='store_true',
        help='Skip NPY export (HDF5 only)'
    )
    generate_group.add_argument(
        '--use-event-anchor', action='store_true',
        help='Use event-anchor selection (Spring/Ignition/Compression)'
    )
    generate_group.add_argument(
        '--calculate-coil-features', action='store_true',
        help='Calculate coil intensity features'
    )
    generate_group.add_argument(
        '--mode', type=str, default='training', choices=['training', 'inference'],
        help='Mode: training (preserves dormant) or inference (strict filters)'
    )
    generate_group.add_argument(
        '--dedup-overlap-days', type=int, default=5,
        help='Days threshold for overlapping window deduplication (default: 5). '
             'Deduplication is now mandatory; this only adjusts the overlap window.'
    )

    # =========================================================================
    # TRAINING (Step 2)
    # =========================================================================
    train_group = parser.add_argument_group('Training (Step 2)')
    train_group.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    train_group.add_argument(
        '--batch-size', type=int, default=32,
        help='Training batch size (default: 32)'
    )
    train_group.add_argument(
        '--use-coil-focal', action='store_true',
        help='Use Coil-Aware Focal Loss (recommended)'
    )
    train_group.add_argument(
        '--use-asl', action='store_true',
        help='Use Asymmetric Loss'
    )
    train_group.add_argument(
        '--use-inverse-freq-weights', action='store_true',
        help='Use inverse frequency class weights'
    )
    train_group.add_argument(
        '--use-volume-weighted-loss', action='store_true',
        help='Add volume-weighted loss penalty'
    )
    train_group.add_argument(
        '--bear-aware-split', action='store_true',
        help='Use bear-aware temporal split (Val=2023 with bear patterns)'
    )
    train_group.add_argument(
        '--disable-trinity', action='store_true',
        help='Disable Trinity cluster-based splitting (use pattern-level instead)'
    )
    train_group.add_argument(
        '--use-augmentation', action='store_true',
        help='Enable physics-aware augmentation'
    )
    train_group.add_argument(
        '--use-window-jitter', action='store_true',
        help='Enable window jittering (-3 to +2 days)'
    )
    train_group.add_argument(
        '--lazy-loading', action='store_true',
        help='Use lazy HDF5 loading (memory efficient)'
    )
    train_group.add_argument(
        '--use-conditioned-lstm', action='store_true',
        help='Use context-conditioned LSTM (h0/c0 from GRN)'
    )
    train_group.add_argument(
        '--compile', action='store_true',
        help='Use torch.compile optimization (PyTorch 2.0+)'
    )

    # =========================================================================
    # ENSEMBLE TRAINING
    # =========================================================================
    ensemble_group = parser.add_argument_group('Ensemble Training')
    ensemble_group.add_argument(
        '--ensemble-size', type=int, default=1,
        help='Number of ensemble members (default: 1 = single model)'
    )
    ensemble_group.add_argument(
        '--ensemble-vary-arch', action='store_true',
        help='Vary architecture across ensemble members'
    )
    ensemble_group.add_argument(
        '--use-dirichlet-calibration', action='store_true',
        help='Use Dirichlet calibration for ensemble'
    )

    return parser


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the pipeline."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    Path('logs').mkdir(exist_ok=True)
    logger = setup_logging(
        verbose=args.verbose,
        log_file=log_file if not args.dry_run else None
    )

    # Determine which steps to run
    steps = []

    if args.full_pipeline:
        steps = PIPELINE_ORDER.copy()
    elif args.step:
        steps = [args.step]
    elif args.steps:
        steps = args.steps
    elif args.from_step:
        start_idx = PIPELINE_ORDER.index(args.from_step)
        steps = PIPELINE_ORDER[start_idx:]

    if not steps:
        parser.print_help()
        print("\nError: Specify --full-pipeline, --step, --steps, or --from-step")
        return 1

    # Create configuration
    try:
        config = RegionConfig.from_region(args.region)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Create and run orchestrator
    orchestrator = PipelineOrchestrator(config, args)

    try:
        exit_code = orchestrator.run(steps, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        exit_code = 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        traceback.print_exc()
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
