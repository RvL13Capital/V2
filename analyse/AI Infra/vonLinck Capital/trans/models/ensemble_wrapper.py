"""
EnsembleWrapper: Multi-Model Inference with Uncertainty Quantification
======================================================================

Loads and runs inference on an ensemble of models, providing:
- Aggregated probability predictions
- Expected Value with confidence intervals
- Uncertainty metrics (epistemic, aleatoric)
- Position sizing recommendations based on confidence

The ensemble reduces variance compared to single models and provides
uncertainty estimates that help traders size positions appropriately.

Usage:
    ensemble = EnsembleWrapper.from_checkpoints(
        ['model_seed0.pt', 'model_seed1.pt', 'model_seed2.pt'],
        device='cuda'
    )
    results = ensemble.predict_with_uncertainty(sequences, context)
    print(f"EV: {results['ev_mean']:.2f} [{results['ev_ci_lower']:.2f}, {results['ev_ci_upper']:.2f}]")
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import json

from models.inference_wrapper import InferenceWrapper, STRATEGIC_VALUES, PREDICTION_TEMPERATURE
from utils.dirichlet_calibration import (
    DirichletCalibrator,
    EnsembleUncertaintyEstimator,
    compute_confidence_interval_ev
)

logger = logging.getLogger(__name__)


class EnsembleWrapper:
    """
    Ensemble of models for robust predictions with uncertainty estimates.

    Aggregates predictions from multiple models trained with different
    random seeds or architectures to:
    1. Reduce prediction variance (more stable EV)
    2. Quantify uncertainty (confidence intervals)
    3. Detect out-of-distribution inputs (high disagreement)

    Attributes:
        members: List of InferenceWrapper instances
        n_models: Number of ensemble members
        calibrator: Optional Dirichlet calibrator for ensemble
        uncertainty_estimator: Computes confidence intervals
    """

    def __init__(
        self,
        members: List[InferenceWrapper],
        calibrator: Optional[DirichletCalibrator] = None,
        strategic_values: Dict[int, float] = None
    ):
        """
        Initialize ensemble from list of model wrappers.

        Args:
            members: List of InferenceWrapper instances
            calibrator: Optional calibrator for ensemble predictions
            strategic_values: Class values for EV calculation
        """
        if len(members) < 2:
            raise ValueError("Ensemble requires at least 2 members")

        self.members = members
        self.n_models = len(members)
        self.calibrator = calibrator
        self.strategic_values = strategic_values or STRATEGIC_VALUES

        # Use first member's config as reference
        self.device = members[0].device
        self.norm_params = members[0].norm_params
        self.robust_params = members[0].robust_params
        self.config = members[0].config

        # Uncertainty estimator
        self.uncertainty_estimator = EnsembleUncertaintyEstimator(
            n_classes=3,
            strategic_values=self.strategic_values
        )

        logger.info(f"EnsembleWrapper initialized with {self.n_models} models")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Calibrator: {'Dirichlet' if calibrator else 'None'}")

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: List[Union[str, Path]],
        device: str = 'cpu',
        model_class: Optional[type] = None
    ) -> 'EnsembleWrapper':
        """
        Load ensemble from multiple checkpoint files.

        Args:
            checkpoint_paths: List of paths to model checkpoints
            device: Torch device ('cpu' or 'cuda')
            model_class: Model class to instantiate (if not embedded)

        Returns:
            EnsembleWrapper instance
        """
        members = []

        for path in checkpoint_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")

            wrapper = InferenceWrapper.from_checkpoint(path, device, model_class)
            members.append(wrapper)
            logger.info(f"  Loaded: {path.name}")

        return cls(members=members)

    @classmethod
    def from_ensemble_checkpoint(
        cls,
        ensemble_path: Union[str, Path],
        device: str = 'cpu',
        model_class: Optional[type] = None
    ) -> 'EnsembleWrapper':
        """
        Load ensemble from a single combined checkpoint.

        The ensemble checkpoint contains:
        - All model state dicts
        - Shared normalization params
        - Ensemble calibrator

        Args:
            ensemble_path: Path to ensemble checkpoint
            device: Torch device
            model_class: Model class to instantiate

        Returns:
            EnsembleWrapper instance
        """
        ensemble_path = Path(ensemble_path)
        checkpoint = torch.load(ensemble_path, map_location=device, weights_only=False)

        if 'ensemble_members' not in checkpoint:
            raise ValueError("Not an ensemble checkpoint (missing 'ensemble_members')")

        # Load shared parameters
        norm_params = checkpoint.get('norm_params')
        robust_params = checkpoint.get('robust_params')
        context_ranges = checkpoint.get('context_ranges')
        config = checkpoint.get('config', {})

        # Determine model class
        if model_class is None:
            try:
                from models.temporal_hybrid_v18 import HybridFeatureNetwork
                model_class = HybridFeatureNetwork
            except ImportError:
                raise ValueError("model_class must be provided")

        # Load each member
        members = []
        for i, member_state in enumerate(checkpoint['ensemble_members']):
            # Create model instance
            model = model_class(**config.get('model_kwargs', {}))
            model.load_state_dict(member_state['model_state_dict'])

            # Create wrapper
            wrapper = InferenceWrapper(
                model=model,
                norm_params=norm_params,
                robust_params=robust_params,
                context_ranges=context_ranges,
                config=config,
                device=torch.device(device),
                temperature=checkpoint.get('temperature', PREDICTION_TEMPERATURE)
            )
            members.append(wrapper)

        # Load calibrator if present
        calibrator = None
        if 'calibrator_state' in checkpoint and checkpoint['calibrator_state']:
            calibrator = DirichletCalibrator.from_state(checkpoint['calibrator_state'])

        return cls(members=members, calibrator=calibrator)

    def predict(
        self,
        sequences: np.ndarray,
        context: Optional[np.ndarray] = None,
        apply_normalization: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ensemble mean predictions.

        Args:
            sequences: Input sequences (N, 20, 14)
            context: Optional context features (N, 13)
            apply_normalization: Whether to apply normalization

        Returns:
            Tuple of (probs_mean, ev_mean) arrays
        """
        results = self.predict_with_uncertainty(sequences, context, apply_normalization)
        return results['probs_mean'], results['ev_mean']

    def predict_with_uncertainty(
        self,
        sequences: np.ndarray,
        context: Optional[np.ndarray] = None,
        apply_normalization: bool = True,
        confidence_level: float = 0.90
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions with full uncertainty quantification.

        Args:
            sequences: Input sequences (N, 20, 14)
            context: Optional context features (N, 13)
            apply_normalization: Whether to apply normalization
            confidence_level: Confidence level for intervals

        Returns:
            Dict with:
            - probs_mean: Mean probabilities (N, 3)
            - probs_std: Std of probabilities (N, 3)
            - ev_mean: Mean expected value (N,)
            - ev_std: Std of expected value (N,)
            - ev_ci_lower: Lower confidence bound (N,)
            - ev_ci_upper: Upper confidence bound (N,)
            - epistemic_uncertainty: Model disagreement (N,)
            - aleatoric_uncertainty: Data uncertainty (N,)
            - prediction_agreement: Model agreement fraction (N,)
            - recommendations: Trading recommendations (N,)
        """
        n_samples = len(sequences)

        # Collect predictions from all members
        ensemble_probs = np.zeros((self.n_models, n_samples, 3))
        ensemble_logits = np.zeros((self.n_models, n_samples, 3))

        for i, member in enumerate(self.members):
            probs, _ = member.predict(sequences, context, apply_normalization)
            ensemble_probs[i] = probs

            # Get raw logits for calibration
            if hasattr(member, '_get_logits'):
                ensemble_logits[i] = member._get_logits(sequences, context, apply_normalization)
            else:
                # Approximate logits from probs
                ensemble_logits[i] = np.log(probs + 1e-10)

        # Apply Dirichlet calibration if available
        if self.calibrator is not None:
            mean_logits = np.mean(ensemble_logits, axis=0)
            calibrated_probs, cal_uncertainty = self.calibrator.calibrate(mean_logits)

            # Update ensemble mean with calibrated version
            results = self.uncertainty_estimator.estimate_uncertainty(
                ensemble_probs, confidence_level
            )
            results['probs_mean'] = calibrated_probs
            results['calibration_uncertainty'] = cal_uncertainty
        else:
            results = self.uncertainty_estimator.estimate_uncertainty(
                ensemble_probs, confidence_level
            )

        # Add trading recommendations
        results['recommendations'] = self.uncertainty_estimator.get_trading_recommendation(
            results,
            ev_threshold=2.0,
            agreement_threshold=0.6,
            max_uncertainty=0.5
        )

        return results

    def predict_single(
        self,
        sequence: np.ndarray,
        context: Optional[np.ndarray] = None,
        confidence_level: float = 0.90
    ) -> Dict:
        """
        Predict for a single sequence with detailed output.

        Args:
            sequence: Single sequence (20, 14)
            context: Optional context features (13,)
            confidence_level: Confidence level for intervals

        Returns:
            Dict with prediction details and uncertainty
        """
        # Expand to batch
        sequences = sequence[np.newaxis, :, :]
        if context is not None:
            context = context[np.newaxis, :]

        results = self.predict_with_uncertainty(
            sequences, context, confidence_level=confidence_level
        )

        return {
            'probs': results['probs_mean'][0],
            'probs_std': results['probs_std'][0],
            'predicted_class': int(np.argmax(results['probs_mean'][0])),
            'ev_mean': float(results['ev_mean'][0]),
            'ev_std': float(results['ev_std'][0]),
            'ev_ci_lower': float(results['ev_ci_lower'][0]),
            'ev_ci_upper': float(results['ev_ci_upper'][0]),
            'confidence_level': confidence_level,
            'epistemic_uncertainty': float(results['epistemic_uncertainty'][0]),
            'aleatoric_uncertainty': float(results['aleatoric_uncertainty'][0]),
            'prediction_agreement': float(results['prediction_agreement'][0]),
            'recommendation': results['recommendations'][0],
            'n_models': self.n_models
        }

    def fit_calibrator(
        self,
        val_sequences: np.ndarray,
        val_labels: np.ndarray,
        val_context: Optional[np.ndarray] = None
    ) -> 'EnsembleWrapper':
        """
        Fit Dirichlet calibrator on validation data.

        Args:
            val_sequences: Validation sequences
            val_labels: True labels
            val_context: Optional context features

        Returns:
            self for chaining
        """
        logger.info("Fitting Dirichlet calibrator for ensemble...")

        # Get ensemble mean logits
        n_samples = len(val_sequences)
        ensemble_logits = np.zeros((self.n_models, n_samples, 3))

        for i, member in enumerate(self.members):
            probs, _ = member.predict(val_sequences, val_context)
            # Approximate logits
            ensemble_logits[i] = np.log(probs + 1e-10)

        mean_logits = np.mean(ensemble_logits, axis=0)

        # Fit calibrator
        self.calibrator = DirichletCalibrator(n_classes=3)
        self.calibrator.fit(mean_logits, val_labels)

        return self

    def save_ensemble_checkpoint(self, save_path: Union[str, Path]):
        """
        Save ensemble as a single checkpoint file.

        Args:
            save_path: Path to save checkpoint
        """
        save_path = Path(save_path)

        # Collect member states
        member_states = []
        for member in self.members:
            member_states.append({
                'model_state_dict': member.model.state_dict()
            })

        checkpoint = {
            'checkpoint_version': '2.0-ensemble',
            'n_models': self.n_models,
            'ensemble_members': member_states,
            'norm_params': self.norm_params,
            'robust_params': self.robust_params,
            'context_ranges': self.members[0].context_ranges,
            'config': self.config,
            'temperature': self.members[0].temperature,
            'strategic_values': self.strategic_values,
            'calibrator_state': self.calibrator.get_state() if self.calibrator else None
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Saved ensemble checkpoint to {save_path}")

    def get_ensemble_info(self) -> Dict:
        """Get information about the ensemble."""
        return {
            'n_models': self.n_models,
            'device': str(self.device),
            'has_calibrator': self.calibrator is not None,
            'strategic_values': self.strategic_values,
            'norm_params': 'embedded' if self.norm_params else 'none',
            'robust_params': 'embedded' if self.robust_params else 'none',
            'calibrator_stats': self.calibrator.calibration_stats if self.calibrator else None
        }


def train_ensemble(
    train_fn,
    n_models: int = 5,
    base_seed: int = 42,
    **train_kwargs
) -> List[Path]:
    """
    Train an ensemble of models with different seeds.

    This is a helper function that calls the training function
    multiple times with different random seeds.

    Args:
        train_fn: Training function that accepts seed parameter
        n_models: Number of models to train
        base_seed: Starting seed (seeds will be base_seed + i)
        **train_kwargs: Additional arguments for train_fn

    Returns:
        List of checkpoint paths
    """
    checkpoint_paths = []

    for i in range(n_models):
        seed = base_seed + i
        logger.info(f"\n{'='*60}")
        logger.info(f"Training ensemble member {i+1}/{n_models} (seed={seed})")
        logger.info(f"{'='*60}\n")

        # Call training function with seed
        checkpoint_path = train_fn(seed=seed, ensemble_idx=i, **train_kwargs)
        checkpoint_paths.append(checkpoint_path)

    return checkpoint_paths


class EnsembleTrainer:
    """
    Coordinates training of ensemble members.

    Handles:
    - Varied random seeds
    - Optional architecture variations (dropout, hidden dims)
    - Progress tracking
    - Checkpoint management
    """

    def __init__(
        self,
        n_models: int = 5,
        base_seed: int = 42,
        vary_architecture: bool = False,
        output_dir: Path = None
    ):
        """
        Initialize ensemble trainer.

        Args:
            n_models: Number of ensemble members
            base_seed: Starting random seed
            vary_architecture: Whether to vary architecture across members
            output_dir: Directory for saving checkpoints
        """
        self.n_models = n_models
        self.base_seed = base_seed
        self.vary_architecture = vary_architecture
        self.output_dir = Path(output_dir) if output_dir else Path('models/ensemble')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_paths = []
        self.training_stats = []

    def get_architecture_variation(self, member_idx: int) -> Dict:
        """
        Get architecture variations for a member.

        Returns:
            Dict of architecture parameter overrides
        """
        if not self.vary_architecture:
            return {}

        # Vary dropout and hidden dimensions
        variations = [
            {'dropout': 0.3, 'hidden_dim': 128},
            {'dropout': 0.4, 'hidden_dim': 128},
            {'dropout': 0.3, 'hidden_dim': 160},
            {'dropout': 0.35, 'hidden_dim': 144},
            {'dropout': 0.4, 'hidden_dim': 160},
        ]

        return variations[member_idx % len(variations)]

    def get_seed(self, member_idx: int) -> int:
        """Get random seed for a member."""
        return self.base_seed + member_idx * 1000  # Space out seeds

    def get_checkpoint_path(self, member_idx: int) -> Path:
        """Get checkpoint path for a member."""
        return self.output_dir / f"ensemble_member_{member_idx:02d}.pt"

    def create_ensemble(self, device: str = 'cpu') -> EnsembleWrapper:
        """
        Create ensemble wrapper from trained checkpoints.

        Args:
            device: Torch device

        Returns:
            EnsembleWrapper instance
        """
        if len(self.checkpoint_paths) < 2:
            raise ValueError("Need at least 2 trained members")

        return EnsembleWrapper.from_checkpoints(self.checkpoint_paths, device)


# Convenience function for quick ensemble prediction
def ensemble_predict(
    checkpoint_paths: List[Union[str, Path]],
    sequences: np.ndarray,
    context: Optional[np.ndarray] = None,
    device: str = 'cpu',
    confidence_level: float = 0.90
) -> Dict:
    """
    Quick ensemble prediction from checkpoint paths.

    Args:
        checkpoint_paths: List of model checkpoint paths
        sequences: Input sequences
        context: Optional context features
        device: Torch device
        confidence_level: Confidence level for intervals

    Returns:
        Dict with predictions and uncertainty
    """
    ensemble = EnsembleWrapper.from_checkpoints(checkpoint_paths, device)
    return ensemble.predict_with_uncertainty(sequences, context, confidence_level=confidence_level)
