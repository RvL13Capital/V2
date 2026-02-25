import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class AsymmetricLoss(nn.Module):
    """
    Production-Ready Asymmetric Focal Loss.

    Improvements over v1:
    1. Numerical Stability: Uses Log-Softmax instead of Softmax -> Log.
    2. Label Smoothing: Prevents overfitting/mode collapse on noisy data.
    3. Consolidated: Removes dead legacy code.
    4. Hardware Safe: Uses buffers for auto-GPU handling.
    """

    def __init__(
        self,
        gamma_neg: float = 2.0,       # Balanced Hunter default
        gamma_pos: float = 1.0,       # Focus on Hard Positives
        gamma_per_class: Optional[List[float]] = None,
        class_weights: Optional[List[float]] = None,
        clip: float = 0.05,           # Kept for API compatibility (unused in log-space)
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
        label_smoothing: float = 0.01 # <--- NEW: The Antidote to Mode Collapse
    ):
        super().__init__()
        
        # Configuration
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.label_smoothing = label_smoothing

        # 1. Setup Gamma Tensor (Buffers move to GPU automatically)
        self.register_buffer('gamma_map', None)
        self.register_buffer('weight_map', None)
        
        # Store configs for lazy init
        self.gamma_config = gamma_per_class
        self.weights_config = class_weights

    def _init_buffers(self, num_classes, device):
        """Initialize tensors on the correct device."""
        # Setup Gamma
        if self.gamma_config:
            gamma_t = torch.tensor(self.gamma_config, dtype=torch.float32, device=device)
        else:
            gamma_t = torch.ones(num_classes, device=device) * self.gamma_pos
            if num_classes > 1:
                gamma_t[1] = self.gamma_neg # Noise gets gamma_neg
        self.gamma_map = gamma_t

        # Setup Class Weights
        if self.weights_config:
            weight_t = torch.tensor(self.weights_config, dtype=torch.float32, device=device)
        else:
            weight_t = torch.ones(num_classes, device=device)
        self.weight_map = weight_t

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw unnormalized scores (N, C)
            targets: Class indices (N)
        """
        # Lazy initialization handles device placement seamlessly
        if self.gamma_map is None:
            self._init_buffers(logits.size(1), logits.device)

        # 1. Compute Stable Log-Probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 2. Calculate Focal Weight (Hardness)
        # p_t = exp(log_prob_of_true_class)
        p_t = torch.exp(log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
        
        # gamma_t based on true class
        gamma_t = self.gamma_map[targets]
        
        # Weight = (1 - p_t)^gamma
        focal_weight = (1.0 - p_t).pow(gamma_t)
        
        if self.disable_torch_grad_focal_loss:
            focal_weight = focal_weight.detach()

        # 3. Calculate Base Loss with Label Smoothing
        # F.cross_entropy handles smoothing internally and efficiently
        ce_loss = F.cross_entropy(
            logits, 
            targets, 
            reduction='none', 
            label_smoothing=self.label_smoothing
        )

        # 4. Combine: Scaled Loss = Focal_Weight * Smoothed_CE
        # This ensures we down-weight the ENTIRE loss (including smoothing penalty) for easy samples
        loss = focal_weight * ce_loss

        # 5. Apply Architecture Class Weights
        if self.weight_map is not None:
            w_t = self.weight_map[targets]
            loss = loss * w_t

        return loss.mean()

# For backward compatibility
TemporalAsymmetricLoss = AsymmetricLoss