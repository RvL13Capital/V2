"""
TRAnS V21 "The Modulator"
=========================
Production Architecture.
Topology: CNN(Geometry) + RoPE -> FiLM <- GRN(Regime).
Evaluation Verdict:
- TCN/Embeddings: Rejected (Redundant for T=20).
- FiLM: Accepted (Correctly models Regime as a multiplicative gate).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

# -----------------------------------------------------------------------------
# 1. Positional Physics (RoPE)
# -----------------------------------------------------------------------------
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 100):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)

class RoPEMultiheadAttention(nn.Module):
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
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rope(x)
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), attn

# -----------------------------------------------------------------------------
# 2. Geometric Encoder (Multi-Scale CNN)
# -----------------------------------------------------------------------------
class TemporalEncoderCNN(nn.Module):
    def __init__(self, input_features: int, channels: tuple = (32, 64), kernels: tuple = (3, 5)):
        super().__init__()
        self.conv_micro = nn.Conv1d(input_features, channels[0], kernels[0], padding=kernels[0]//2)
        self.conv_macro = nn.Conv1d(input_features, channels[1], kernels[1], padding=kernels[1]//2)
        self.output_dim = sum(channels)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        micro = F.relu(self.conv_micro(x_t))
        macro = F.relu(self.conv_macro(x_t))
        out = torch.cat([micro, macro], dim=1).transpose(1, 2)
        return self.norm(out)

# -----------------------------------------------------------------------------
# 3. Regime Modulation (GRN + FiLM)
# -----------------------------------------------------------------------------
class GatedResidualNetwork(nn.Module):
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
        return self.layernorm(x_out * gate + residual)

class FiLMGenerator(nn.Module):
    """
    Generates Scale (Gamma) and Shift (Beta) from Context.
    Initializes to Identity (Gamma=1, Beta=0) for stability.
    """
    def __init__(self, context_dim: int, feature_dim: int):
        super().__init__()
        self.fc = nn.Linear(context_dim, 2 * feature_dim)
        
        # Identity Initialization (Gamma=1, Beta=0)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        with torch.no_grad():
            self.fc.bias[:feature_dim].fill_(1.0) # Gamma = 1

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.fc(context)
        gamma, beta = torch.chunk(out, 2, dim=-1)
        return gamma, beta

# -----------------------------------------------------------------------------
# 4. The V21 Model
# -----------------------------------------------------------------------------
class HybridFeatureNetworkV21(nn.Module):
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
        fusion_dropout: float = 0.3
    ):
        super().__init__()
        self.input_features = input_features
        self.context_features = context_features

        # Branch A: Geometry (Structure)
        self.cnn_encoder = TemporalEncoderCNN(input_features, cnn_channels, cnn_kernels)
        self.cnn_dim = sum(cnn_channels)
        self.structure_attention = RoPEMultiheadAttention(
            embed_dim=self.cnn_dim,
            num_heads=4,
            dropout=0.1,
            max_seq_len=sequence_length * 2
        )
        self.structure_norm = nn.LayerNorm(self.cnn_dim)

        # Branch B: Regime (Context)
        self.context_projection = nn.Linear(context_features, context_hidden)
        self.context_grn = GatedResidualNetwork(context_hidden, context_hidden, dropout=0.1)

        # Fusion: FiLM
        self.film = FiLMGenerator(context_dim=context_hidden, feature_dim=self.cnn_dim)

        # Decision Layer
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout / 2),
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
        # Safety
        if context is None:
            context = torch.zeros(B, self.context_features, device=x.device)
        x = torch.nan_to_num(x, nan=0.0)
        
        # Branch A: Geometry
        cnn_seq = self.cnn_encoder(x)
        attn_out, _ = self.structure_attention(cnn_seq)
        structure_emb = attn_out.mean(dim=1) 
        structure_emb = self.structure_norm(structure_emb)

        # Branch B: Regime
        ctx_proj = F.relu(self.context_projection(context))
        context_emb = self.context_grn(ctx_proj) # Filter noise

        # FiLM Modulation
        gamma, beta = self.film(context_emb)
        modulated_structure = (structure_emb * gamma) + beta

        # Decision
        return self.classifier(modulated_structure)

    def get_feature_importance(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict:
        with torch.no_grad():
            if context is None: context = torch.zeros(x.size(0), self.context_features, device=x.device)
            cnn_seq = self.cnn_encoder(torch.nan_to_num(x, nan=0.0))
            _, attn_weights = self.structure_attention(cnn_seq)
            return {'temporal_importance': attn_weights.mean(dim=1).mean(dim=1).cpu().numpy()}
