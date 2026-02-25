"""
Dirichlet Calibration for Multi-Class Probability Estimation
=============================================================

Implements Dirichlet calibration which is superior to isotonic/Platt scaling
for multi-class classification because it:

1. Preserves the simplex constraint (probabilities sum to 1)
2. Models uncertainty via concentration parameters
3. Produces well-calibrated probability distributions
4. Enables confidence interval estimation

Reference: Kull et al. "Beyond temperature scaling: Obtaining well-calibrated
multi-class probabilities with Dirichlet calibration" (NeurIPS 2019)

Usage:
    calibrator = DirichletCalibrator(n_classes=3)
    calibrator.fit(logits_val, labels_val)
    calibrated_probs, uncertainty = calibrator.calibrate(logits_test)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, digamma, gammaln
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class DirichletCalibrator:
    """
    Dirichlet calibration for multi-class probability estimation.

    Transforms model logits into well-calibrated probabilities with
    uncertainty estimates via Dirichlet concentration parameters.

    The calibration learns a linear transform: z' = W @ z + b
    where z are the original logits and z' are calibrated logits.

    Attributes:
        n_classes: Number of output classes
        W: Calibration weight matrix (n_classes × n_classes)
        b: Calibration bias vector (n_classes,)
        regularization: L2 regularization strength
        calibration_stats: Fit statistics
    """

    def __init__(
        self,
        n_classes: int = 3,
        regularization: float = 1e-4,
        off_diagonal_regularization: float = 1e-3
    ):
        """
        Initialize Dirichlet calibrator.

        Args:
            n_classes: Number of classes (default 3 for Danger/Noise/Target)
            regularization: L2 regularization on diagonal elements
            off_diagonal_regularization: Extra regularization on off-diagonal
                                         (encourages simpler calibration)
        """
        self.n_classes = n_classes
        self.regularization = regularization
        self.off_diagonal_reg = off_diagonal_regularization

        # Initialize with identity transform
        self.W = np.eye(n_classes, dtype=np.float64)
        self.b = np.zeros(n_classes, dtype=np.float64)

        self.is_fitted = False
        self.calibration_stats = {}

    def _pack_params(self, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Pack W and b into a single parameter vector."""
        return np.concatenate([W.flatten(), b])

    def _unpack_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack parameter vector into W and b."""
        W = params[:self.n_classes**2].reshape(self.n_classes, self.n_classes)
        b = params[self.n_classes**2:]
        return W, b

    def _calibrated_logits(self, logits: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Apply calibration transform: z' = W @ z + b."""
        return logits @ W.T + b

    def _nll_loss(self, params: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Negative log-likelihood loss for calibration.

        Args:
            params: Packed W and b parameters
            logits: Model logits (N, n_classes)
            labels: True labels (N,)

        Returns:
            NLL loss value
        """
        W, b = self._unpack_params(params)

        # Apply calibration
        cal_logits = self._calibrated_logits(logits, W, b)

        # Compute softmax probabilities
        probs = softmax(cal_logits, axis=1)

        # Clip for numerical stability
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

        # NLL: -sum(log(p_true))
        n_samples = len(labels)
        nll = -np.sum(np.log(probs[np.arange(n_samples), labels])) / n_samples

        # L2 regularization
        # Encourage diagonal W close to 1, off-diagonal close to 0
        diag_reg = self.regularization * np.sum((np.diag(W) - 1) ** 2)
        off_diag_mask = ~np.eye(self.n_classes, dtype=bool)
        off_diag_reg = self.off_diagonal_reg * np.sum(W[off_diag_mask] ** 2)
        bias_reg = self.regularization * np.sum(b ** 2)

        return nll + diag_reg + off_diag_reg + bias_reg

    def _nll_gradient(self, params: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Gradient of NLL loss w.r.t. parameters."""
        W, b = self._unpack_params(params)
        n_samples = len(labels)

        # Forward pass
        cal_logits = self._calibrated_logits(logits, W, b)
        probs = softmax(cal_logits, axis=1)

        # Gradient of softmax cross-entropy: p - y_onehot
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(n_samples), labels] = 1
        grad_cal_logits = (probs - y_onehot) / n_samples

        # Gradient w.r.t. W: grad_cal_logits.T @ logits
        grad_W = grad_cal_logits.T @ logits

        # Gradient w.r.t. b: sum(grad_cal_logits, axis=0)
        grad_b = np.sum(grad_cal_logits, axis=0)

        # Add regularization gradients
        grad_W_reg = np.zeros_like(W)
        np.fill_diagonal(grad_W_reg, 2 * self.regularization * (np.diag(W) - 1))
        off_diag_mask = ~np.eye(self.n_classes, dtype=bool)
        grad_W_reg[off_diag_mask] += 2 * self.off_diagonal_reg * W[off_diag_mask]

        grad_b_reg = 2 * self.regularization * b

        return self._pack_params(grad_W + grad_W_reg, grad_b + grad_b_reg)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> 'DirichletCalibrator':
        """
        Fit calibration parameters on validation data.

        Args:
            logits: Model logits (N, n_classes)
            labels: True labels (N,)
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance

        Returns:
            self for chaining
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)

        n_samples = len(labels)
        logger.info(f"Fitting Dirichlet calibrator on {n_samples:,} samples...")

        # Compute pre-calibration metrics
        pre_probs = softmax(logits, axis=1)
        pre_ece = self._expected_calibration_error(pre_probs, labels)

        # Initialize parameters
        params_init = self._pack_params(self.W, self.b)

        # Optimize
        result = minimize(
            fun=self._nll_loss,
            x0=params_init,
            args=(logits, labels),
            method='L-BFGS-B',
            jac=self._nll_gradient,
            options={'maxiter': max_iter, 'ftol': tol}
        )

        if not result.success:
            logger.warning(f"Calibration optimization did not converge: {result.message}")

        self.W, self.b = self._unpack_params(result.x)
        self.is_fitted = True

        # Compute post-calibration metrics
        cal_logits = self._calibrated_logits(logits, self.W, self.b)
        post_probs = softmax(cal_logits, axis=1)
        post_ece = self._expected_calibration_error(post_probs, labels)

        self.calibration_stats = {
            'n_samples': n_samples,
            'pre_ece': float(pre_ece),
            'post_ece': float(post_ece),
            'ece_reduction': float((pre_ece - post_ece) / pre_ece * 100),
            'converged': result.success,
            'iterations': result.nit,
            'W_diagonal': self.W.diagonal().tolist(),
            'b': self.b.tolist()
        }

        logger.info(f"  Pre-calibration ECE: {pre_ece:.4f}")
        logger.info(f"  Post-calibration ECE: {post_ece:.4f}")
        logger.info(f"  ECE reduction: {self.calibration_stats['ece_reduction']:.1f}%")
        logger.info(f"  W diagonal: {np.diag(self.W)}")
        logger.info(f"  b: {self.b}")

        return self

    def _expected_calibration_error(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15
    ) -> float:
        """Compute Expected Calibration Error (ECE)."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(accuracies[in_bin])
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

        return ece

    def calibrate(
        self,
        logits: np.ndarray,
        return_uncertainty: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Apply calibration to get well-calibrated probabilities.

        Args:
            logits: Model logits (N, n_classes)
            return_uncertainty: Whether to compute uncertainty metrics

        Returns:
            Tuple of:
            - calibrated_probs: (N, n_classes) calibrated probabilities
            - uncertainty: Dict with uncertainty metrics (if requested)
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning raw softmax")
            probs = softmax(logits, axis=1)
            return (probs, None) if return_uncertainty else probs

        logits = np.asarray(logits, dtype=np.float64)

        # Apply calibration transform
        cal_logits = self._calibrated_logits(logits, self.W, self.b)
        probs = softmax(cal_logits, axis=1)

        if not return_uncertainty:
            return probs

        # Compute uncertainty metrics
        uncertainty = self._compute_uncertainty(probs, cal_logits)

        return probs, uncertainty

    def _compute_uncertainty(self, probs: np.ndarray, cal_logits: np.ndarray) -> Dict:
        """
        Compute uncertainty metrics from calibrated predictions.

        Returns:
            Dict with:
            - entropy: Predictive entropy (higher = more uncertain)
            - max_prob: Maximum class probability (lower = more uncertain)
            - margin: Difference between top-2 probabilities
            - concentration: Dirichlet concentration (proxy for certainty)
        """
        # Predictive entropy: -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        max_entropy = np.log(self.n_classes)  # Max entropy for uniform
        normalized_entropy = entropy / max_entropy

        # Max probability (confidence)
        max_prob = np.max(probs, axis=1)

        # Margin: difference between top-2 probabilities
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]

        # Dirichlet concentration (approximation)
        # Higher concentration = more certain prediction
        # Use inverse temperature as proxy
        concentration = np.sum(np.exp(cal_logits), axis=1)

        return {
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'max_prob': max_prob,
            'margin': margin,
            'concentration': concentration
        }

    def get_state(self) -> Dict:
        """Get calibrator state for serialization."""
        return {
            'n_classes': self.n_classes,
            'regularization': self.regularization,
            'off_diagonal_reg': self.off_diagonal_reg,
            'W': self.W.tolist(),
            'b': self.b.tolist(),
            'is_fitted': self.is_fitted,
            'calibration_stats': self.calibration_stats
        }

    @classmethod
    def from_state(cls, state: Dict) -> 'DirichletCalibrator':
        """Load calibrator from serialized state."""
        calibrator = cls(
            n_classes=state['n_classes'],
            regularization=state['regularization'],
            off_diagonal_regularization=state.get('off_diagonal_reg', 1e-3)
        )
        calibrator.W = np.array(state['W'], dtype=np.float64)
        calibrator.b = np.array(state['b'], dtype=np.float64)
        calibrator.is_fitted = state['is_fitted']
        calibrator.calibration_stats = state.get('calibration_stats', {})
        return calibrator


class EnsembleUncertaintyEstimator:
    """
    Uncertainty estimation from ensemble predictions.

    Combines predictions from multiple models to estimate:
    - Aleatoric uncertainty (inherent data noise)
    - Epistemic uncertainty (model uncertainty)
    - Total predictive uncertainty

    This enables confidence interval construction for EV scores.
    """

    def __init__(self, n_classes: int = 3, strategic_values: Dict[int, float] = None):
        """
        Initialize uncertainty estimator.

        Args:
            n_classes: Number of output classes
            strategic_values: Class values for EV calculation
        """
        self.n_classes = n_classes
        self.strategic_values = strategic_values or {0: -1.0, 1: -0.1, 2: 5.0}

    def estimate_uncertainty(
        self,
        ensemble_probs: np.ndarray,
        confidence_level: float = 0.90
    ) -> Dict:
        """
        Estimate uncertainty from ensemble predictions.

        Args:
            ensemble_probs: (n_models, N, n_classes) array of probability predictions
            confidence_level: Confidence level for intervals (default 90%)

        Returns:
            Dict with uncertainty estimates:
            - probs_mean: Mean probabilities (N, n_classes)
            - probs_std: Std of probabilities (N, n_classes)
            - ev_mean: Mean expected value (N,)
            - ev_std: Std of expected value (N,)
            - ev_ci_lower: Lower confidence bound (N,)
            - ev_ci_upper: Upper confidence bound (N,)
            - epistemic_uncertainty: Model disagreement (N,)
            - aleatoric_uncertainty: Average model entropy (N,)
            - total_uncertainty: Combined uncertainty (N,)
            - prediction_agreement: Fraction of models agreeing (N,)
        """
        n_models, n_samples, n_classes = ensemble_probs.shape

        # Mean and std of probabilities
        probs_mean = np.mean(ensemble_probs, axis=0)
        probs_std = np.std(ensemble_probs, axis=0)

        # EV for each model
        ev_per_model = np.zeros((n_models, n_samples))
        for cls, value in self.strategic_values.items():
            ev_per_model += ensemble_probs[:, :, cls] * value

        # EV statistics
        ev_mean = np.mean(ev_per_model, axis=0)
        ev_std = np.std(ev_per_model, axis=0)

        # Confidence intervals using percentiles
        alpha = 1 - confidence_level
        ev_ci_lower = np.percentile(ev_per_model, alpha / 2 * 100, axis=0)
        ev_ci_upper = np.percentile(ev_per_model, (1 - alpha / 2) * 100, axis=0)

        # Epistemic uncertainty: variance of mean predictions
        # High when models disagree
        epistemic = np.mean(probs_std ** 2, axis=1)

        # Aleatoric uncertainty: average entropy of individual models
        # High when predictions are uncertain even for single models
        entropies = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-10), axis=2)
        aleatoric = np.mean(entropies, axis=0)

        # Total uncertainty: entropy of mean prediction
        total = -np.sum(probs_mean * np.log(probs_mean + 1e-10), axis=1)

        # Prediction agreement: fraction of models predicting same class
        predictions = np.argmax(ensemble_probs, axis=2)  # (n_models, n_samples)
        modal_prediction = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=n_classes).argmax(),
            axis=0,
            arr=predictions
        )
        agreement = np.mean(predictions == modal_prediction, axis=0)

        return {
            'probs_mean': probs_mean,
            'probs_std': probs_std,
            'ev_mean': ev_mean,
            'ev_std': ev_std,
            'ev_ci_lower': ev_ci_lower,
            'ev_ci_upper': ev_ci_upper,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': total,
            'prediction_agreement': agreement,
            'n_models': n_models,
            'confidence_level': confidence_level
        }

    def get_trading_recommendation(
        self,
        uncertainty_results: Dict,
        ev_threshold: float = 2.0,
        agreement_threshold: float = 0.6,
        max_uncertainty: float = 0.5
    ) -> np.ndarray:
        """
        Generate trading recommendations considering uncertainty.

        Args:
            uncertainty_results: Output from estimate_uncertainty()
            ev_threshold: Minimum EV for signal
            agreement_threshold: Minimum model agreement
            max_uncertainty: Maximum allowed epistemic uncertainty

        Returns:
            Array of recommendation strings
        """
        n_samples = len(uncertainty_results['ev_mean'])
        recommendations = np.empty(n_samples, dtype=object)

        ev_mean = uncertainty_results['ev_mean']
        ev_ci_lower = uncertainty_results['ev_ci_lower']
        agreement = uncertainty_results['prediction_agreement']
        epistemic = uncertainty_results['epistemic_uncertainty']
        probs_mean = uncertainty_results['probs_mean']

        danger_prob = probs_mean[:, 0]

        for i in range(n_samples):
            # Conditions for strong signal
            if (ev_ci_lower[i] >= ev_threshold and
                agreement[i] >= agreement_threshold and
                epistemic[i] <= max_uncertainty and
                danger_prob[i] < 0.25):
                recommendations[i] = "STRONG_SIGNAL"

            # Good signal: mean EV high but some uncertainty
            elif (ev_mean[i] >= ev_threshold and
                  agreement[i] >= 0.5 and
                  danger_prob[i] < 0.30):
                recommendations[i] = "GOOD_SIGNAL"

            # Moderate signal
            elif ev_mean[i] >= 1.0 and danger_prob[i] < 0.35:
                recommendations[i] = "MODERATE_SIGNAL"

            # High uncertainty - be cautious
            elif epistemic[i] > max_uncertainty or agreement[i] < 0.5:
                recommendations[i] = "UNCERTAIN"

            # Negative EV or high danger
            elif ev_mean[i] < 0 or danger_prob[i] > 0.35:
                recommendations[i] = "AVOID"

            else:
                recommendations[i] = "WEAK_SIGNAL"

        return recommendations


def compute_confidence_interval_ev(
    ensemble_probs: np.ndarray,
    strategic_values: Dict[int, float] = None,
    confidence_level: float = 0.90
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to compute EV confidence intervals.

    Args:
        ensemble_probs: (n_models, N, n_classes) ensemble predictions
        strategic_values: Class values for EV (default: Danger=-1, Noise=-0.1, Target=5)
        confidence_level: Confidence level (default 90%)

    Returns:
        Tuple of (ev_mean, ev_lower, ev_upper) arrays
    """
    estimator = EnsembleUncertaintyEstimator(
        n_classes=ensemble_probs.shape[2],
        strategic_values=strategic_values
    )
    results = estimator.estimate_uncertainty(ensemble_probs, confidence_level)
    return results['ev_mean'], results['ev_ci_lower'], results['ev_ci_upper']
