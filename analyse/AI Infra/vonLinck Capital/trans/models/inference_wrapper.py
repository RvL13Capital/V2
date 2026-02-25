"""
InferenceWrapper: Self-Contained Model Deployment
==================================================

Provides a single-file deployment solution where the model checkpoint
contains all necessary transforms for inference:

- Model weights
- Feature normalization params (median/IQR for robust scaling, or mean/std for legacy)
- Context feature ranges for denormalization
- Probability calibrator (optional)
- Label mapping version (for validation)

Usage:
    wrapper = InferenceWrapper.from_checkpoint('model.pt', device='cuda')
    probs, ev_scores = wrapper.predict(sequences, context_features)

The checkpoint is truly atomic - no external config files needed.

Normalization:
    - Robust scaling (preferred): X_norm = (X - median) / IQR
    - Legacy standard scaling: X_norm = (X - mean) / std
    - The scaling type is auto-detected from norm_params['scaling_type']
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any
import json
import logging

logger = logging.getLogger(__name__)

# Import constants from central config (SINGLE SOURCE OF TRUTH)
from config.constants import (
    STRATEGIC_VALUES,
    LABEL_MAPPING_VERSION,
    LABEL_ID_TO_NAME,
    validate_label_mapping,
    decode_label,
)

PREDICTION_TEMPERATURE = 1.5  # Temperature scaling for calibration


class InferenceWrapper:
    """
    Self-contained inference wrapper that loads model + all transforms.

    All normalization parameters are embedded in the checkpoint,
    making deployment a single-file operation.

    Attributes:
        model: The loaded PyTorch model
        device: Torch device (cpu/cuda)
        norm_params: Normalization params - robust (median/iqr) or standard (mean/std)
        context_ranges: Context feature normalization ranges
        config: Training configuration used
        scaling_type: 'robust' (preferred) or 'standard' (legacy)

    Example:
        >>> wrapper = InferenceWrapper.from_checkpoint('best_model.pt')
        >>> probs, ev = wrapper.predict(sequences)  # (N, 20, 10) -> (N, 3), (N,)
    """

    def __init__(
        self,
        model: nn.Module,
        norm_params: Optional[Dict] = None,
        context_ranges: Optional[Dict] = None,
        config: Optional[Dict] = None,
        device: torch.device = None,
        temperature: float = PREDICTION_TEMPERATURE,
        calibrator: Optional[Any] = None
    ):
        """
        Initialize wrapper with model and scaling parameters.

        Args:
            model: Trained PyTorch model
            norm_params: Dict with either:
                - 'scaling_type': 'robust', 'median' and 'iqr' lists (10 features)
                - 'scaling_type': 'standard' (or missing), 'mean' and 'std' lists (10 features)
            context_ranges: Dict mapping feature names to (min, max) tuples
            config: Training configuration
            device: Torch device
            temperature: Softmax temperature for probability calibration
            calibrator: Optional probability calibrator
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.norm_params = norm_params
        self.context_ranges = context_ranges
        self.config = config or {}
        self.temperature = temperature
        self.calibrator = calibrator

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Pre-compute normalization arrays for efficiency
        self._setup_normalization_arrays()

        # Log scaling type
        scaling_type = 'none'
        if norm_params:
            scaling_type = norm_params.get('scaling_type', 'standard')

        logger.info(f"InferenceWrapper initialized on {self.device}")
        logger.info(f"  norm_params: {'embedded (' + scaling_type + ')' if norm_params else 'none'}")
        logger.info(f"  context_ranges: {'embedded' if context_ranges else 'none'}")
        logger.info(f"  calibrator: {'loaded' if calibrator else 'none'}")

    def _setup_normalization_arrays(self):
        """Pre-compute numpy arrays for efficient normalization."""
        # Determine scaling type (robust preferred, standard for legacy)
        self._scaling_type = None
        self._norm_center = None  # median or mean
        self._norm_scale = None   # iqr or std

        if self.norm_params:
            scaling_type = self.norm_params.get('scaling_type', 'standard')
            self._scaling_type = scaling_type

            if scaling_type == 'robust':
                # Robust scaling: (X - median) / IQR
                self._norm_center = np.array(self.norm_params['median'], dtype=np.float32)
                self._norm_scale = np.array(self.norm_params['iqr'], dtype=np.float32)
            else:
                # Standard scaling: (X - mean) / std
                self._norm_center = np.array(self.norm_params['mean'], dtype=np.float32)
                self._norm_scale = np.array(self.norm_params['std'], dtype=np.float32)

            # Avoid division by zero
            self._norm_scale = np.where(self._norm_scale > 1e-8, self._norm_scale, 1.0)

        # All 10 features get normalization
        self._features_to_normalize = list(range(10))  # [0-9]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = 'cpu',
        model_class: Optional[type] = None
    ) -> 'InferenceWrapper':
        """
        Load wrapper from atomic checkpoint file.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Device to load model on ('cpu', 'cuda', 'cuda:0', etc.)
            model_class: Optional model class override (auto-detected if not provided)

        Returns:
            InferenceWrapper ready for inference

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            KeyError: If checkpoint missing required keys
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Handle device
        if isinstance(device, str):
            device = torch.device(device)

        logger.info(f"Loading atomic checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract components
        config = checkpoint.get('config', {})
        norm_params = checkpoint.get('norm_params')
        context_ranges = checkpoint.get('context_ranges')
        temperature = checkpoint.get('temperature', PREDICTION_TEMPERATURE)
        calibrator_state = checkpoint.get('calibrator_state')

        # =================================================================
        # CRITICAL: Validate label mapping before loading model
        # =================================================================
        # This prevents using a model trained with different class ordering
        checkpoint_label_version = checkpoint.get('label_mapping_version')
        if checkpoint_label_version is not None:
            validate_label_mapping(checkpoint_label_version)
            logger.info(f"  label_mapping: {checkpoint_label_version} [OK]")
        else:
            # Legacy checkpoint without version - warn but allow
            logger.warning(
                "  label_mapping: NOT EMBEDDED (legacy checkpoint). "
                "Assuming 0=Danger, 1=Noise, 2=Target. "
                "Re-train model to embed label mapping version."
            )
            # Still validate internal consistency
            validate_label_mapping()

        # Build model
        if model_class is None:
            # Auto-detect model class from checkpoint
            try:
                from models.temporal_hybrid_unified import HybridFeatureNetwork
                model_class = HybridFeatureNetwork
            except ImportError:
                raise ImportError("Cannot auto-detect model class. Provide model_class parameter.")

        # Initialize model with correct config
        use_grn = config.get('use_grn_context', True)
        use_conditioned = config.get('use_conditioned_lstm', False)
        dropout = config.get('dropout', 0.3)

        # Detect mode from config (V20 checkpoints have ablation_mode, V18 use v18_full)
        ablation_mode = config.get('ablation_mode')
        mode = ablation_mode if ablation_mode else 'v18_full'

        # Unified HybridFeatureNetwork with mode parameter
        model = model_class(
            mode=mode,
            input_features=config.get('input_features', 10),  # 10 features after composite disabled
            context_features=13 if use_grn else 0,
            lstm_hidden=32,
            lstm_num_layers=2,
            num_classes=3,
            lstm_dropout=0.2,
            fusion_dropout=dropout,
            use_conditioned_lstm=use_conditioned
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load calibrator if available
        calibrator = None
        if calibrator_state is not None:
            try:
                from utils.probability_calibration import ProbabilityCalibrator
                calibrator = ProbabilityCalibrator(n_classes=3)
                calibrator.load_state(calibrator_state)
                logger.info("Loaded embedded probability calibrator")
            except Exception as e:
                logger.warning(f"Could not load calibrator: {e}")

        # Log what was loaded
        scaling_type = norm_params.get('scaling_type', 'standard') if norm_params else 'none'
        logger.info(f"Checkpoint loaded:")
        logger.info(f"  epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"  val_acc: {checkpoint.get('val_acc', 'unknown'):.4f}" if isinstance(checkpoint.get('val_acc'), (int, float)) else f"  val_acc: {checkpoint.get('val_acc', 'unknown')}")
        logger.info(f"  norm_params: {'yes (' + scaling_type + ' scaling)' if norm_params else 'no'}")
        logger.info(f"  context_ranges: {'yes' if context_ranges else 'no'}")
        logger.info(f"  calibrator: {'yes' if calibrator else 'no'}")

        return cls(
            model=model,
            norm_params=norm_params,
            context_ranges=context_ranges,
            config=config,
            device=device,
            temperature=temperature,
            calibrator=calibrator
        )

    def normalize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Apply normalization transforms to sequences.

        Uses either robust scaling (median/IQR) or standard scaling (mean/std)
        depending on the norm_params['scaling_type'].

        Args:
            sequences: Input sequences of shape (N, 20, 10)

        Returns:
            Normalized sequences of shape (N, 20, 10)
        """
        sequences = sequences.astype(np.float32).copy()

        # Apply normalization: (X - center) / scale
        # Works for both robust (median/IQR) and standard (mean/std)
        if self._norm_center is not None:
            for feat_idx in self._features_to_normalize:
                if self._norm_scale[feat_idx] > 0:
                    sequences[:, :, feat_idx] = (
                        sequences[:, :, feat_idx] - self._norm_center[feat_idx]
                    ) / self._norm_scale[feat_idx]

        return sequences

    def normalize_context(self, context: np.ndarray) -> np.ndarray:
        """
        Normalize context features using embedded ranges.

        Args:
            context: Context features of shape (N, 13) or dict per sample

        Returns:
            Normalized context of shape (N, 13)
        """
        if self.context_ranges is None:
            return context.astype(np.float32)

        context = context.astype(np.float32).copy()

        # Get feature names in order
        try:
            from config.context_features import CONTEXT_FEATURES
            feature_names = CONTEXT_FEATURES
        except ImportError:
            # Fallback to default order
            feature_names = [
                'float_turnover', 'trend_position', 'base_duration', 'relative_volume',
                'distance_to_high', 'log_float', 'log_dollar_volume', 'relative_strength_spy',
                'price_position_at_end', 'distance_to_danger', 'bbw_slope_5d',
                'vol_trend_5d', 'coil_intensity'
            ]

        for i, feat_name in enumerate(feature_names):
            if feat_name in self.context_ranges:
                min_val, max_val = self.context_ranges[feat_name]
                # Clip and normalize to [0, 1]
                context[:, i] = np.clip(context[:, i], min_val, max_val)
                if max_val > min_val:
                    context[:, i] = (context[:, i] - min_val) / (max_val - min_val)

        return context

    @torch.no_grad()
    def predict(
        self,
        sequences: np.ndarray,
        context: Optional[np.ndarray] = None,
        apply_normalization: bool = True,
        return_raw_logits: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with optional normalization.

        Args:
            sequences: Input sequences of shape (N, 20, 14)
            context: Optional context features of shape (N, 13)
            apply_normalization: Whether to apply normalization transforms
            return_raw_logits: If True, return logits instead of probabilities

        Returns:
            Tuple of:
                - probs: Class probabilities of shape (N, 3) [or logits if return_raw_logits]
                - ev_scores: Expected value scores of shape (N,)
        """
        # Validate input - 10 features (composite features disabled 2026-01-18)
        if sequences.ndim != 3 or sequences.shape[1:] != (20, 10):
            raise ValueError(f"Expected sequences shape (N, 20, 10), got {sequences.shape}")

        # Apply normalization if requested
        if apply_normalization:
            sequences = self.normalize_sequences(sequences)
            if context is not None:
                context = self.normalize_context(context)

        # Convert to tensors
        seq_tensor = torch.from_numpy(sequences).float().to(self.device)

        if context is not None:
            ctx_tensor = torch.from_numpy(context).float().to(self.device)
        else:
            ctx_tensor = None

        # Forward pass
        if ctx_tensor is not None:
            logits = self.model(seq_tensor, ctx_tensor)
        else:
            logits = self.model(seq_tensor)

        if return_raw_logits:
            logits_np = logits.cpu().numpy()
            # Still compute EV from softmax for consistency
            probs = torch.softmax(logits / self.temperature, dim=-1).cpu().numpy()
        else:
            # Apply temperature scaling
            probs = torch.softmax(logits / self.temperature, dim=-1).cpu().numpy()

        # Apply calibration if available
        if self.calibrator is not None and not return_raw_logits:
            probs = self.calibrator.calibrate(probs)

        # Compute Expected Value
        ev_scores = np.zeros(len(probs), dtype=np.float32)
        for cls, value in STRATEGIC_VALUES.items():
            ev_scores += probs[:, cls] * value

        if return_raw_logits:
            return logits.cpu().numpy(), ev_scores

        return probs, ev_scores

    def predict_single(
        self,
        sequence: np.ndarray,
        context: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Predict for a single sequence with detailed output.

        Args:
            sequence: Single sequence of shape (20, 14)
            context: Optional context features of shape (13,)

        Returns:
            Dict with:
                - probs: Class probabilities {0: p0, 1: p1, 2: p2}
                - predicted_class: Argmax class
                - ev_score: Expected value
                - class_names: Human-readable class names
        """
        # Add batch dimension
        sequences = sequence[np.newaxis, :, :]
        if context is not None:
            context = context[np.newaxis, :]

        probs, ev_scores = self.predict(sequences, context)

        predicted_class = int(np.argmax(probs[0]))
        class_names = {0: 'Danger', 1: 'Noise', 2: 'Target'}

        return {
            'probs': {i: float(probs[0, i]) for i in range(3)},
            'predicted_class': predicted_class,
            'predicted_class_name': class_names[predicted_class],
            'ev_score': float(ev_scores[0]),
            'recommendation': self._get_recommendation(ev_scores[0], probs[0])
        }

    def _get_recommendation(self, ev: float, probs: np.ndarray) -> str:
        """Get trading recommendation based on EV and probabilities."""
        danger_prob = probs[0]
        target_prob = probs[2]

        if ev >= 3.0 and danger_prob < 0.2:
            return "STRONG_SIGNAL"
        elif ev >= 2.0 and danger_prob < 0.25:
            return "GOOD_SIGNAL"
        elif ev >= 1.0 and danger_prob < 0.3:
            return "MODERATE_SIGNAL"
        elif ev < 0 or danger_prob > 0.35:
            return "AVOID"
        else:
            return "NEUTRAL"

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about the loaded checkpoint."""
        return {
            'config': self.config,
            'has_norm_params': self.norm_params is not None,
            'has_context_ranges': self.context_ranges is not None,
            'has_calibrator': self.calibrator is not None,
            'temperature': self.temperature,
            'device': str(self.device),
            'model_params': sum(p.numel() for p in self.model.parameters()),
        }

    @staticmethod
    def create_atomic_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_acc: float,
        config: Dict,
        norm_params: Dict,
        context_ranges: Optional[Dict] = None,
        calibrator: Optional[Any] = None,
        temperature: float = PREDICTION_TEMPERATURE,
        save_path: Union[str, Path] = None,
        top15_precision: Optional[float] = None,
        top15_danger_rate: Optional[float] = None
    ) -> Dict:
        """
        Create an atomic checkpoint with all embedded transforms.

        This is the recommended way to save checkpoints for production.

        Args:
            model: Trained model
            optimizer: Optimizer state
            epoch: Current epoch
            val_acc: Validation accuracy
            config: Training configuration
            norm_params: Normalization parameters. Should contain either:
                - 'scaling_type': 'robust', 'median' and 'iqr' lists (preferred)
                - 'scaling_type': 'standard' (or missing), 'mean' and 'std' lists (legacy)
            context_ranges: Context feature ranges (optional, uses defaults if None)
            calibrator: Probability calibrator (optional)
            temperature: Softmax temperature
            save_path: Path to save checkpoint (optional)
            top15_precision: Target rate in top 15% by EV (primary metric)
            top15_danger_rate: Danger rate in top 15% by EV

        Returns:
            Checkpoint dictionary (also saved to save_path if provided)
        """
        # Get context ranges from config if not provided
        if context_ranges is None:
            try:
                from config.context_features import CONTEXT_FEATURE_RANGES
                context_ranges = dict(CONTEXT_FEATURE_RANGES)
            except ImportError:
                logger.warning("Could not import CONTEXT_FEATURE_RANGES, context_ranges will be None")

        # Get calibrator state if available
        calibrator_state = None
        if calibrator is not None:
            try:
                calibrator_state = calibrator.get_state()
            except Exception as e:
                logger.warning(f"Could not get calibrator state: {e}")

        checkpoint = {
            # Core model state
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': config,

            # === ATOMIC SCALERS (Production Deployment) ===
            'norm_params': norm_params,           # Median/IQR (robust) or Mean/Std (legacy) for 10 features
            'context_ranges': context_ranges,     # Min/Max for 14 context features

            # === CALIBRATION ===
            'temperature': temperature,
            'calibrator_state': calibrator_state,

            # === LABEL MAPPING (CRITICAL for inference validation) ===
            'label_mapping_version': LABEL_MAPPING_VERSION,  # Validates 0=Danger, 1=Noise, 2=Target
            'label_id_to_name': LABEL_ID_TO_NAME,           # Explicit mapping for debugging

            # === METADATA ===
            'checkpoint_version': '2.2',          # Atomic checkpoint format version (with label mapping)
            'strategic_values': STRATEGIC_VALUES,

            # === TOP-15% METRICS (Primary Early Stopping Criterion) ===
            'top15_precision': top15_precision,     # Target rate in top 15% by EV
            'top15_danger_rate': top15_danger_rate, # Danger rate in top 15% by EV
        }

        if save_path is not None:
            save_path = Path(save_path)
            torch.save(checkpoint, save_path)
            logger.info(f"Saved atomic checkpoint to {save_path}")
            logger.info(f"  Size: {save_path.stat().st_size / 1024 / 1024:.1f} MB")

        return checkpoint


def load_for_inference(
    checkpoint_path: Union[str, Path],
    device: str = 'cpu'
) -> InferenceWrapper:
    """
    Convenience function to load model for inference.

    Args:
        checkpoint_path: Path to .pt checkpoint
        device: Device to use ('cpu' or 'cuda')

    Returns:
        InferenceWrapper ready for predictions

    Example:
        >>> wrapper = load_for_inference('model.pt', device='cuda')
        >>> probs, ev = wrapper.predict(sequences)
    """
    return InferenceWrapper.from_checkpoint(checkpoint_path, device=device)
