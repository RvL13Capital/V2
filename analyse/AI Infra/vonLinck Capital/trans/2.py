"""
Temporal Hybrid Model V21: The "Geometric-Regime" Standard
==========================================================

The definitive production architecture derived from the V20 Ablation Study.
Strictly implements the CNN-GRN-Concat topology (29.4% Target Rate).

Design Philosophy: "Simplicity is Alpha"
----------------------------------------
1. Geometric Branch (Temporal): Multi-Scale CNN extracts shape (Wicks/Coils).
   RoPE Attention relates these shapes across the window.
2. Regime Branch (Context): Gated Residual Network (GRN) filters static context
   (Float, Relative Strength, Dormancy) to isolate the market regime.
3. Fusion: Direct concatenation. No complex gating or cross-attention.

Performance Profile:
- Params: ~72k (Lightweight)
- Inference: ~1.8x faster than V18
- Robustness: High (No LSTM narrative hallucinations)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

# =============================================================================
# SELF-CONTAINED COMPONENTS (Decoupled from V18/V20)
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Encodes relative positions by rotating the query and key vectors.
    Critical for distinguishing "Setup" (early) from "Trigger" (late) in the window.
    """
    def __init__(self, dim: int, max_seq_len: int = 50):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (Batch, Heads, Seq, Dim)
    # cos, sin: (1, 1, Seq, Dim)
    x1, x2 = x.chunk(2, dim=-1)
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)

class RoPEMultiheadAttention(nn.Module):
    """
    Self-Attention with embedded RoPE.
    Applies rotation to Q/K inside the attention head.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 50):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor):
        # x: (Batch, Seq_Len, Embed_Dim)
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(x)
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out), attn_weights

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) for Static Context.
    Filters noise from context features using a Gating mechanism.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(input_dim, input_dim)
        self.res_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_proj(x) if hasattr(self, 'res_proj') else x
        
        x_norm = self.layernorm(x)
        x_out = F.elu(self.fc1(x_norm))
        x_out = self.dropout(self.fc2(x_out))
        gate = torch.sigmoid(self.gate(x_norm))
        return self.layernorm(x_out * gate + residual) # Post-norm for stability

class TemporalEncoderCNN(nn.Module):
    """
    Multi-Scale CNN Encoder.
    Captures geometry: Micro-structure (k=3) and Macro-structure (k=5).
    """
    def __init__(self, input_features: int, channels: tuple = (32, 64), kernels: tuple = (3, 5)):
        super().__init__()
        self.conv_micro = nn.Conv1d(input_features, channels[0], kernels[0], padding=kernels[0]//2)
        self.conv_macro = nn.Conv1d(input_features, channels[1], kernels[1], padding=kernels[1]//2)
        self.output_dim = sum(channels)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq, Feat) -> Transpose for CNN -> (Batch, Feat, Seq)
        x_t = x.transpose(1, 2)
        micro = F.relu(self.conv_micro(x_t))
        macro = F.relu(self.conv_macro(x_t))
        # Concat: (Batch, C1+C2, Seq) -> Transpose -> (Batch, Seq, C1+C2)
        out = torch.cat([micro, macro], dim=1).transpose(1, 2)
        return self.norm(out)

# =============================================================================
# V21 MODEL ARCHITECTURE
# =============================================================================

class HybridFeatureNetworkV21(nn.Module):
    """
    The V21 Production Standard.
    Structure: CNN(Shape) + RoPE(Structure) + GRN(Regime) -> Fusion MLP.
    """
    def __init__(
        self,
        input_features: int = 10,
        context_features: int = 18,
        sequence_length: int = 20,
        cnn_channels: tuple = (32, 64),
        cnn_kernels: tuple = (3, 5),
        context_hidden: int = 32,
        fusion_hidden: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_features = input_features
        self.context_features = context_features

        # Branch A: Temporal (Geometric)
        self.cnn_encoder = TemporalEncoderCNN(input_features, cnn_channels, cnn_kernels)
        self.cnn_dim = sum(cnn_channels)
        self.structure_attention = RoPEMultiheadAttention(
            embed_dim=self.cnn_dim,
            num_heads=4,
            dropout=0.1,
            max_seq_len=sequence_length * 2
        )

        # Branch B: Context (Regime)
        self.context_projection = nn.Linear(context_features, context_hidden)
        self.context_grn = GatedResidualNetwork(context_hidden, context_hidden, dropout=0.1)

        # Fusion: Concat(Structure[96], Context[32]) = 128
        combined_dim = self.cnn_dim + context_hidden
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden), # Stability
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_hidden // 2, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        # Inference Safety: Default context to zeros if missing
        if context is None: context = torch.zeros(B, self.context_features, device=x.device)

        # 1. Feature Integrity (Dormancy Protection)
        if torch.isnan(x).any(): x = torch.nan_to_num(x, nan=0.0)

        # 2. Branch A: Structure
        cnn_seq = self.cnn_encoder(x) # (B, 20, 96)
        attn_out, _ = self.structure_attention(cnn_seq)
        structure_emb = attn_out.mean(dim=1) # Global Avg Pool -> (B, 96)

        # 3. Branch B: Regime
        ctx_proj = F.relu(self.context_projection(context))
        context_emb = self.context_grn(ctx_proj) # (B, 32)

        # 4. Fusion
        combined = torch.cat([structure_emb, context_emb], dim=1)
        return self.fusion_mlp(combined)

    def get_architecture_summary(self) -> Dict:
        return {'model': 'v21_production', 'params': sum(p.numel() for p in self.parameters())}

    def get_feature_importance(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict:
        with torch.no_grad():
            if context is None: context = torch.zeros(x.size(0), self.context_features, device=x.device)
            cnn_seq = self.cnn_encoder(x)
            _, attn_weights = self.structure_attention(cnn_seq)
            return {'temporal_importance': attn_weights.mean(dim=1).mean(dim=1).cpu().numpy()}
