"""
Coil-Aware Focal Loss for TRANS Model (Jan 2026)
=================================================

Focal Loss variant that pays extra attention to K2 patterns with strong coil signals.

The standard focal loss reduces weight on easy examples, but our coil_intensity feature
provides additional signal: patterns with HIGH coil intensity (tight BBW, low position,
low volume) are MORE likely to be true K2 targets.

This loss amplifies gradients for K2 samples with strong coil signals, forcing the
model to learn these high-value patterns.

Key insight from analysis:
- Low position (<0.4): 29.8% K2 rate
- High position (>=0.6): 5.4% K2 rate
- Coil intensity > 0.5: 26.2% K2 rate vs 20.9% for intensity < 0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import FeatureRegistry for context feature indices (eliminates hardcoded magic numbers)
from config.feature_registry import FeatureRegistry


class CoilAwareFocalLoss(nn.Module):
    """
    Focal Loss with coil intensity weighting for K2 (Target) class.

    Standard focal loss: FL(pt) = -(1-pt)^gamma * log(pt)

    Coil-aware extension: For K2 targets, multiply loss by (1 + coil_intensity * coil_weight)
    This forces the model to learn K2 patterns with strong coil signals.

    Args:
        gamma: Focal loss focusing parameter (default 2.0)
        coil_strength_weight: Multiplier for coil intensity boost on K2 (default 3.0)
        class_weights: Optional tensor of class weights [K0, K1, K2]
        alpha: Optional per-class balancing weights (alternative to class_weights)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        coil_strength_weight: float = 3.0,
        class_weights: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.gamma = gamma
        self.coil_weight = coil_strength_weight

        # Use alpha or class_weights (alpha takes precedence)
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        elif class_weights is not None:
            self.register_buffer('alpha', class_weights)
        else:
            self.alpha = None

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        coil_intensity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute coil-aware focal loss.

        Args:
            inputs: Model logits of shape (batch, num_classes)
            targets: Ground truth labels of shape (batch,)
            coil_intensity: Coil intensity scores of shape (batch,), values in [0, 1]
                           Higher = tighter coil = more likely K2

        Returns:
            Scalar loss value
        """
        # Compute cross entropy loss per sample (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # Compute pt (probability of true class)
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Standard focal loss: (1 - pt)^gamma * CE
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # COIL BOOST: Amplify loss for K2 patterns with strong coil signals
        if coil_intensity is not None:
            is_k2 = (targets == 2)
            if is_k2.any():
                # Boost = 1 + (coil_intensity * coil_weight)
                # For coil_intensity=0.8, coil_weight=3.0: boost = 1 + 2.4 = 3.4x
                coil_boost = 1.0 + (coil_intensity * self.coil_weight)
                focal_loss = focal_loss.clone()  # Avoid in-place modification
                focal_loss[is_k2] = focal_loss[is_k2] * coil_boost[is_k2]

        return focal_loss.mean()


class AsymmetricCoilFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss with coil intensity weighting.

    Combines:
    1. Asymmetric loss (different gamma for positive/negative samples)
    2. Focal loss focusing (down-weight easy examples)
    3. Coil intensity boost (amplify K2 patterns with strong coils)

    This is the recommended loss for TRANS model training with coil features.

    Args:
        gamma_neg: Focal parameter for negative (non-target) classes (default 4.0)
        gamma_pos: Focal parameter for positive (target) class (default 1.0)
        coil_strength_weight: Multiplier for coil boost on K2 (default 3.0)
        class_weights: Optional tensor of class weights
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        coil_strength_weight: float = 3.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.coil_weight = coil_strength_weight

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        coil_intensity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute asymmetric coil-aware focal loss.

        The asymmetric design:
        - High gamma_neg (4.0): Strongly down-weight easy negatives (confident non-K2)
        - Low gamma_pos (1.0): Keep focus on K2 patterns even when confident
        """
        probs = F.softmax(inputs, dim=1)

        # One-hot encode targets
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()

        # Compute asymmetric focal weights
        # For positive (target) class: (1-p)^gamma_pos
        # For negative classes: p^gamma_neg
        pos_weight = (1 - probs) ** self.gamma_pos
        neg_weight = probs ** self.gamma_neg

        # Combine: use pos_weight where target=1, neg_weight where target=0
        focal_weight = targets_one_hot * pos_weight + (1 - targets_one_hot) * neg_weight

        # Apply class weights if provided
        if self.class_weights is not None:
            class_weight_expanded = self.class_weights.unsqueeze(0).expand_as(probs)
            focal_weight = focal_weight * class_weight_expanded

        # Binary cross entropy per class
        bce = -targets_one_hot * torch.log(probs + 1e-8) - (1 - targets_one_hot) * torch.log(1 - probs + 1e-8)

        # Focal loss per sample (sum over classes)
        focal_loss = (focal_weight * bce).sum(dim=1)

        # COIL BOOST for K2 patterns
        if coil_intensity is not None:
            is_k2 = (targets == 2)
            if is_k2.any():
                coil_boost = 1.0 + (coil_intensity * self.coil_weight)
                focal_loss = focal_loss.clone()
                focal_loss[is_k2] = focal_loss[is_k2] * coil_boost[is_k2]

        return focal_loss.mean()


class RankMatchCoilLoss(nn.Module):
    """
    Ranking-Aware Coil Loss (Jan 2026)

    Hybrid loss combining:
    1. CoilAwareFocalLoss: Classification with coil intensity boosting
    2. MarginRankingLoss: Explicit ranking constraint

    The ranking component directly optimizes for Precision @ Top K by ensuring
    K2 (Target) patterns produce higher logits than K1 (Noise) and K0 (Danger).

    Key insight: Standard cross-entropy optimizes probability calibration, but
    our metric (Top 15% precision) cares about RANK not probability. This loss
    explicitly penalizes the model when a Target's K2 logit is lower than a
    non-Target's K2 logit.

    Ranking pairs:
    - Target vs Noise: margin = 0.1 (Target should score higher)
    - Target vs Danger: margin = 0.2 (Target should score MUCH higher than Danger)

    Args:
        focal_gamma: Focal loss gamma parameter (default 2.0)
        coil_weight: Coil intensity boost for K2 (default 3.0)
        rank_margin: Margin for ranking loss (default 0.1)
        rank_lambda: Weight of ranking loss relative to focal (default 0.5)
        danger_margin_mult: Multiplier for Danger margin vs Noise margin (default 2.0)
        class_weights: Optional class weights for focal loss
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        coil_weight: float = 3.0,
        rank_margin: float = 0.1,
        rank_lambda: float = 0.5,
        danger_margin_mult: float = 2.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.focal = CoilAwareFocalLoss(
            gamma=focal_gamma,
            coil_strength_weight=coil_weight,
            class_weights=class_weights
        )
        self.rank_margin = rank_margin
        self.rank_lambda = rank_lambda
        self.danger_margin_mult = danger_margin_mult

        # MarginRankingLoss: loss = max(0, -y*(x1-x2) + margin)
        # With y=+1: loss = max(0, -(x1-x2) + margin) = max(0, x2-x1 + margin)
        # So x1 must exceed x2 by at least 'margin' to have zero loss
        self.rank_loss_noise = nn.MarginRankingLoss(margin=rank_margin)
        self.rank_loss_danger = nn.MarginRankingLoss(margin=rank_margin * danger_margin_mult)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        coil_intensity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ranking-aware coil loss.

        Args:
            logits: Model logits of shape (batch, 3) for [K0, K1, K2]
            targets: Ground truth labels of shape (batch,)
            coil_intensity: Coil intensity scores of shape (batch,)

        Returns:
            Scalar loss value
        """
        # 1. Base Focal Loss (Classification)
        cls_loss = self.focal(logits, targets, coil_intensity)

        # 2. Auxiliary Ranking Loss (Topology)
        # Goal: Logit_K2(Target) > Logit_K2(Noise/Danger) + Margin
        target_logits = logits[:, 2]  # K2 class score for all samples

        is_target = (targets == 2)
        is_noise = (targets == 1)
        is_danger = (targets == 0)

        ranking_penalty = torch.tensor(0.0, device=logits.device)
        n_pairs = 0

        # Pair Targets with Noise samples
        if is_target.any() and is_noise.any():
            n_target = is_target.sum().item()
            n_noise = is_noise.sum().item()
            n_pairs_noise = min(n_target, n_noise)

            t_scores = target_logits[is_target][:n_pairs_noise]
            n_scores = target_logits[is_noise][:n_pairs_noise]

            # Target should be ranked higher (+1 flag)
            y = torch.ones(n_pairs_noise, device=logits.device)

            ranking_penalty = ranking_penalty + self.rank_loss_noise(t_scores, n_scores, y)
            n_pairs += n_pairs_noise

        # Pair Targets with Danger samples (with higher margin)
        if is_target.any() and is_danger.any():
            n_target = is_target.sum().item()
            n_danger = is_danger.sum().item()
            n_pairs_danger = min(n_target, n_danger)

            t_scores = target_logits[is_target][:n_pairs_danger]
            d_scores = target_logits[is_danger][:n_pairs_danger]

            # Target should be ranked higher (+1 flag)
            y = torch.ones(n_pairs_danger, device=logits.device)

            ranking_penalty = ranking_penalty + self.rank_loss_danger(t_scores, d_scores, y)
            n_pairs += n_pairs_danger

        # Average if we had pairs from both
        if n_pairs > 0:
            # Normalize by number of pair types (1 or 2)
            num_pair_types = int(is_target.any() and is_noise.any()) + int(is_target.any() and is_danger.any())
            if num_pair_types > 1:
                ranking_penalty = ranking_penalty / num_pair_types

            return cls_loss + (self.rank_lambda * ranking_penalty)

        return cls_loss

    def extra_repr(self) -> str:
        return (f"rank_margin={self.rank_margin}, rank_lambda={self.rank_lambda}, "
                f"danger_margin_mult={self.danger_margin_mult}")


def get_coil_intensity_from_context(context: torch.Tensor) -> torch.Tensor:
    """
    Extract coil_intensity from context tensor using FeatureRegistry.

    coil_intensity is a composite coil quality score:
    - High values (>0.5): Tight BBW, low position, low volume = strong coil
    - Low values (<0.5): Loose pattern, weak setup

    Args:
        context: Context tensor of shape (batch, num_features)

    Returns:
        coil_intensity tensor of shape (batch,)
    """
    # Get index from registry (single source of truth)
    coil_idx = FeatureRegistry.get_index('coil_intensity')

    if context.shape[-1] <= coil_idx:
        # Old context format, return default (zeros)
        return torch.zeros(context.shape[0], device=context.device)

    return context[:, coil_idx]


def get_volume_shock_from_context(context: torch.Tensor) -> torch.Tensor:
    """
    Extract volume_shock from context tensor using FeatureRegistry.

    volume_shock = max(volume[-3:]) / vol_20d_avg
    - High values (>1.5) indicate explosive volume preceding breakout
    - Low values (<1.0) indicate quiet/no volume spike

    Args:
        context: Context tensor of shape (batch, num_features)

    Returns:
        volume_shock tensor of shape (batch,)
    """
    # Get index from registry (single source of truth)
    vol_shock_idx = FeatureRegistry.get_index('volume_shock')

    if context.shape[-1] <= vol_shock_idx:
        # Old context format, return neutral value (1.0)
        return torch.ones(context.shape[0], device=context.device)

    return context[:, vol_shock_idx]


class VolumeWeightedLoss(nn.Module):
    """
    Volume-Weighted Loss for TRANS Model (Jan 2026).

    Penalizes Class 2 (Target) predictions when volume confirmation is weak.
    Goal: Make the model "scared" to predict breakouts without explosive volume.

    Logic:
        If model predicts Class 2 AND volume_shock < volume_threshold:
            loss = base_loss * penalty_multiplier (default: 2x)

    This forces the model to associate Target predictions with strong volume.

    Args:
        base_criterion: Base loss function (e.g., CoilAwareFocalLoss)
        volume_threshold: Minimum volume_shock for valid Class 2 prediction (default: 1.5)
        penalty_multiplier: Loss multiplier for low-volume Class 2 predictions (default: 2.0)
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        volume_threshold: float = 1.5,
        penalty_multiplier: float = 2.0
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.volume_threshold = volume_threshold
        self.penalty_multiplier = penalty_multiplier

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        coil_intensity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with volume-aware penalty.

        Args:
            logits: Model output logits of shape (batch, 3)
            targets: Ground truth labels of shape (batch,)
            context: Context tensor of shape (batch, 18) - required for volume
            coil_intensity: Optional coil intensity for base criterion

        Returns:
            Weighted loss scalar
        """
        # Call base criterion with appropriate arguments
        if coil_intensity is not None:
            base_loss = self.base_criterion(logits, targets, coil_intensity=coil_intensity)
        else:
            base_loss = self.base_criterion(logits, targets)

        # If no context, return base loss
        if context is None:
            return base_loss

        # Get predictions (argmax of logits)
        preds = torch.argmax(logits, dim=-1)

        # Get volume_shock from context
        volume_shock = get_volume_shock_from_context(context)

        # Identify low-volume Class 2 predictions
        # Condition: pred == 2 AND volume_shock < threshold
        is_class2_pred = (preds == 2)
        is_low_volume = (volume_shock < self.volume_threshold)
        needs_penalty = is_class2_pred & is_low_volume

        # Apply penalty by computing per-sample loss and re-weighting
        # This is more precise than multiplying the mean loss
        if needs_penalty.any():
            # Compute per-sample cross-entropy loss
            log_probs = F.log_softmax(logits, dim=-1)
            per_sample_loss = F.nll_loss(log_probs, targets, reduction='none')

            # Create weight mask: penalty_multiplier for low-volume Class 2 preds, 1.0 otherwise
            weights = torch.ones_like(per_sample_loss)
            weights[needs_penalty] = self.penalty_multiplier

            # Weighted mean loss
            weighted_loss = (per_sample_loss * weights).mean()

            # Log statistics (only on first batch to avoid spam)
            n_penalized = needs_penalty.sum().item()
            n_class2 = is_class2_pred.sum().item()
            if n_class2 > 0:
                penalty_rate = n_penalized / n_class2
                # Only log occasionally to avoid spam
                # logger.debug(f"VolumeWeightedLoss: {n_penalized}/{n_class2} Class 2 preds penalized ({penalty_rate:.1%})")

            return weighted_loss

        return base_loss

    def extra_repr(self) -> str:
        return (f"volume_threshold={self.volume_threshold}, "
                f"penalty_multiplier={self.penalty_multiplier}")
