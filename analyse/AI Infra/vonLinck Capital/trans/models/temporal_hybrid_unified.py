"""
Unified Hybrid Temporal Model: V18 + V20 Consolidated
======================================================

Single unified model class that supports all V18 and V20 behaviors via a `mode` parameter.
This replaces both temporal_hybrid_v18.py and temporal_hybrid_v20.py.

Mode Parameter:
---------------
| Mode          | LSTM                    | CQA | Context-Conditioned | Use Case     |
|---------------|-------------------------|-----|---------------------|--------------|
| v18_full      | ContextConditionedLSTM  | Yes | Yes                 | Production   |
| concat        | None                    | No  | N/A                 | Ablation     |
| lstm          | Standard                | Yes | No                  | Ablation     |
| cqa_only      | None                    | Yes | N/A                 | Ablation     |
| v18_baseline  | ContextConditionedLSTM  | Yes | Yes                 | Ablation alias|

Architecture (v18_full mode):
  Input -> [ContextConditionedLSTM] --> [Narrative State]    <-- Context h0/c0
  Input -> [CNN] -> [Self-Attention] -> [Geometric Structure]
  Context -> [GRN] ---> [CQA] --------> [Context Relevance]
                   └------------------> [Base Probability]
  Output: Concat(Narrative, Structure, Relevance, Context) -> Classifier

Jan 2026 - Unified from V18 + V20 for code consolidation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Literal
import math
import warnings

# Import TemporalFeatureConfig for temporal feature indices (eliminates hardcoded magic numbers)
from config.temporal_features import TemporalFeatureConfig

# Type alias for mode parameter
ModelMode = Literal['v18_full', 'concat', 'lstm', 'cqa_only', 'v18_baseline']

# Singleton instance for temporal feature configuration
_TEMPORAL_CONFIG = TemporalFeatureConfig()


# =============================================================================
# SHARED COMPONENTS (from V18)
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for temporal sequences.

    RoPE encodes position by rotating query/key vectors, allowing the model to:
    1. Learn relative positions naturally through dot-product attention
    2. Generalize to different sequence lengths
    3. Detect valid breakouts at ANY timestep without human bias

    Key advantage over manual bias ramps:
    - No hardcoded assumptions about "when breakouts should occur"
    - Model learns position-dependent patterns from data
    - Day 14 breakout vs Day 19 breakout treated equally by architecture

    Reference: RoFormer (Su et al., 2021) - https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_seq_len: int = 64, base: float = 10000.0):
        """
        Initialize RoPE.

        Args:
            dim: Embedding dimension (must be even for rotation pairs)
            max_seq_len: Maximum sequence length to pre-compute
            base: Base for frequency computation (default: 10000)
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute frequency bands: theta_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Pre-compute sin/cos for efficiency
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Pre-compute sin/cos values for positions 0 to seq_len-1."""
        positions = torch.arange(seq_len).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of x for RoPE application."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to query and key tensors.

        Args:
            q: Query tensor of shape (batch, heads, seq_len, head_dim)
            k: Key tensor of shape (batch, heads, seq_len, head_dim)

        Returns:
            Rotated (q, k) tuple with position information encoded
        """
        seq_len = q.size(2)

        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed


class RoPEMultiheadAttention(nn.Module):
    """
    Multi-head Attention with Rotary Positional Embeddings.

    Replaces nn.MultiheadAttention with RoPE-enabled version that:
    1. Applies RoPE to Q and K before attention computation
    2. Removes need for manual positional biases
    3. Enables position-aware attention without human assumptions
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 64,
        batch_first: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        if need_weights:
            attn_weights_avg = attn_weights.mean(dim=1)
            return out, attn_weights_avg
        return out, None


class GatedResidualNetwork(nn.Module):
    """Standard GRN for processing static context features."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.glu = nn.GLU(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        if input_dim != hidden_dim:
            self.res_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_proj(x)
        x = self.glu(self.dropout(self.fc2(self.elu(self.fc1(x)))))
        return self.norm(x + residual)


class ContextConditionedLSTM(nn.Module):
    """
    LSTM with hidden state initialized from context embedding (GRN output).

    Instead of starting with zero hidden states, this LSTM is "primed" with
    market regime information. The context embedding is projected to create
    the initial hidden (h0) and cell (c0) states.

    Context Features (13 for V18 production):
        Original 8: float_turnover, trend_position, base_duration, relative_volume,
                    distance_to_high, log_float, log_dollar_volume, relative_strength_spy
        Coil 5: price_position_at_end, distance_to_danger, bbw_slope_5d, vol_trend_5d, coil_intensity
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        context_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.h_proj = nn.Linear(context_dim, num_layers * hidden_dim)
        self.c_proj = nn.Linear(context_dim, num_layers * hidden_dim)

        self.h_norm = nn.LayerNorm(hidden_dim)
        self.c_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        context_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        if context_emb is not None:
            h0 = self.h_proj(context_emb)
            c0 = self.c_proj(context_emb)

            h0 = h0.view(batch_size, self.num_layers, self.hidden_dim)
            c0 = c0.view(batch_size, self.num_layers, self.hidden_dim)

            h0 = self.h_norm(h0)
            c0 = self.c_norm(c0)

            h0 = h0.permute(1, 0, 2).contiguous()
            c0 = c0.permute(1, 0, 2).contiguous()

            out, (hn, cn) = self.lstm(x, (h0, c0))
        else:
            out, (hn, cn) = self.lstm(x)

        return out[:, -1, :], out


class ContextQueryAttention(nn.Module):
    """
    Context-Query Attention: Uses Market Context (Query) to search Temporal Sequence (Key/Value).
    """
    def __init__(self, context_dim: int, temporal_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(context_dim, temporal_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=temporal_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(temporal_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_emb: torch.Tensor, temporal_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.q_proj(context_emb).unsqueeze(1)
        attn_out, attn_weights = self.attn(query, temporal_seq, temporal_seq)
        out = query + self.dropout(attn_out)
        out = self.norm(out).squeeze(1)
        return out, attn_weights


class SplitFeatureAttention(nn.Module):
    """
    Split-Head Attention for Illiquid Market Microstructure.

    Domain-aware attention that separates price features from volume/liquidity features
    to prevent price noise (wide spreads) from drowning out clean volume signals.

    Feature indices are loaded from TemporalFeatureConfig (single source of truth).
    """

    def __init__(
        self,
        price_indices: list = None,
        volume_indices: list = None,
        price_cnn_channels: int = 48,
        volume_cnn_channels: int = 48,
        kernel_size: int = 3,
        num_heads_price: int = 2,
        num_heads_volume: int = 2,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 64
    ):
        super().__init__()

        # Use TemporalFeatureConfig as single source of truth for indices
        self.price_indices = price_indices or _TEMPORAL_CONFIG.price_group_indices
        self.volume_indices = volume_indices or _TEMPORAL_CONFIG.volume_group_indices
        self.use_rope = use_rope

        price_dim = len(self.price_indices)
        volume_dim = len(self.volume_indices)

        # Branch A: Price/Geometry CNN + Attention
        self.price_conv = nn.Conv1d(
            price_dim, price_cnn_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.price_norm = nn.LayerNorm(price_cnn_channels)

        if use_rope:
            self.price_attention = RoPEMultiheadAttention(
                embed_dim=price_cnn_channels,
                num_heads=num_heads_price,
                dropout=dropout,
                max_seq_len=max_seq_len,
                batch_first=True
            )
        else:
            self.price_attention = nn.MultiheadAttention(
                embed_dim=price_cnn_channels,
                num_heads=num_heads_price,
                dropout=dropout,
                batch_first=True
            )

        # Branch B: Volume/Liquidity CNN + Attention
        self.volume_conv = nn.Conv1d(
            volume_dim, volume_cnn_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.volume_norm = nn.LayerNorm(volume_cnn_channels)

        if use_rope:
            self.volume_attention = RoPEMultiheadAttention(
                embed_dim=volume_cnn_channels,
                num_heads=num_heads_volume,
                dropout=dropout,
                max_seq_len=max_seq_len,
                batch_first=True
            )
        else:
            self.volume_attention = nn.MultiheadAttention(
                embed_dim=volume_cnn_channels,
                num_heads=num_heads_volume,
                dropout=dropout,
                batch_first=True
            )

        self.output_dim = price_cnn_channels + volume_cnn_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        x_price = x[:, :, self.price_indices]
        x_volume = x[:, :, self.volume_indices]

        price_conv = F.relu(self.price_conv(x_price.transpose(1, 2)))
        price_seq = self.price_norm(price_conv.transpose(1, 2))
        price_attn_out, price_attn_weights = self.price_attention(
            price_seq, price_seq, price_seq
        )
        emb_price = price_attn_out.mean(dim=1)

        volume_conv = F.relu(self.volume_conv(x_volume.transpose(1, 2)))
        volume_seq = self.volume_norm(volume_conv.transpose(1, 2))
        volume_attn_out, volume_attn_weights = self.volume_attention(
            volume_seq, volume_seq, volume_seq
        )
        emb_volume = volume_attn_out.mean(dim=1)

        attn_weights = {
            'price': price_attn_weights,
            'volume': volume_attn_weights
        }

        return emb_price, emb_volume, attn_weights


class TemporalEncoderCNN(nn.Module):
    """
    Shared CNN encoder for temporal features.
    Extracts local patterns (coil shapes, volume spikes) at multiple scales.
    """

    def __init__(
        self,
        input_features: int = 10,
        cnn_channels: tuple = (32, 64),
        kernel_sizes: tuple = (3, 5)
    ):
        super().__init__()

        self.conv_small = nn.Conv1d(
            input_features, cnn_channels[0], kernel_sizes[0],
            padding=kernel_sizes[0] // 2
        )
        self.conv_large = nn.Conv1d(
            input_features, cnn_channels[1], kernel_sizes[1],
            padding=kernel_sizes[1] // 2
        )

        self.output_dim = sum(cnn_channels)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        cnn_out = torch.cat([
            F.relu(self.conv_small(x_t)),
            F.relu(self.conv_large(x_t))
        ], dim=1).transpose(1, 2)
        return self.norm(cnn_out)


# =============================================================================
# UNIFIED HYBRID FEATURE NETWORK
# =============================================================================

class HybridFeatureNetwork(nn.Module):
    """
    Unified Hybrid Temporal Model with mode-based architecture selection.

    Supports all V18 and V20 behaviors through a single `mode` parameter:
    - v18_full (default): Full V18 with ContextConditionedLSTM + CQA (Production)
    - concat: Simplest ablation - CNN + context concatenation only
    - lstm: Standard LSTM (not conditioned) + CQA
    - cqa_only: CNN + CQA, no LSTM
    - v18_baseline: Alias for v18_full (ablation comparison)

    Input Features (10):
        [open, high, low, close, volume, bbw_20, adx, volume_ratio_20,
         upper_boundary, lower_boundary]

    Context Features (13 for production):
        Original 8: [float_turnover, trend_position, base_duration, relative_volume,
                    distance_to_high, log_float, log_dollar_volume, relative_strength_spy]
        Coil 5: [price_position_at_end, distance_to_danger, bbw_slope_5d, vol_trend_5d, coil_intensity]
    """

    VALID_MODES = ('v18_full', 'concat', 'lstm', 'cqa_only', 'v18_baseline')

    def __init__(
        self,
        mode: ModelMode = 'v18_full',
        input_features: int = 10,
        sequence_length: int = 20,
        context_features: int = 13,  # V18 production uses 13 context features
        lstm_hidden: int = 32,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.2,
        cnn_channels: list = None,
        cnn_kernel_sizes: list = None,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        fusion_hidden: int = 128,
        fusion_dropout: float = 0.3,
        num_classes: Optional[int] = 3,
        use_conditioned_lstm: bool = True,  # Only used in v18_full/v18_baseline modes
        use_split_attention: bool = False,
        use_rope: bool = True
    ):
        """
        Initialize the unified hybrid temporal model.

        Args:
            mode: Architecture mode - one of VALID_MODES
            input_features: Number of temporal features per timestep (default: 10)
            sequence_length: Length of temporal sequence (default: 20)
            context_features: Number of static context features (default: 13)
            lstm_hidden: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            lstm_dropout: LSTM dropout rate
            cnn_channels: CNN output channels per kernel
            cnn_kernel_sizes: CNN kernel sizes
            num_attention_heads: Attention heads
            attention_dropout: Attention dropout rate
            fusion_hidden: Fusion MLP hidden dimension
            fusion_dropout: Fusion dropout rate
            num_classes: Output classes (default: 3)
            use_conditioned_lstm: Use context-conditioned LSTM (v18_full mode only)
            use_split_attention: Use domain-aware split attention
            use_rope: Use Rotary Positional Embeddings
        """
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode}")

        if cnn_channels is None:
            cnn_channels = [32, 64]
        if cnn_kernel_sizes is None:
            cnn_kernel_sizes = [3, 5]
        if num_classes is None:
            num_classes = 3

        self.mode = mode
        self.num_classes = num_classes
        self.input_features = input_features
        self.context_features = context_features
        self.use_context = context_features > 0
        self.use_split_attention = use_split_attention
        self.use_rope = use_rope
        self.cnn_output_dim = sum(cnn_channels)

        # Determine if we use conditioned LSTM based on mode
        # v18_full and v18_baseline use conditioned LSTM
        # Other modes use standard or no LSTM
        self._use_conditioned_lstm = (
            mode in ('v18_full', 'v18_baseline') and
            use_conditioned_lstm and
            self.use_context
        )

        # For backward compatibility with V18 attribute access
        self.use_conditioned_lstm = self._use_conditioned_lstm

        # ==========================================
        # LSTM Branch (mode-dependent)
        # ==========================================
        self.lstm = None
        lstm_dim = 0

        if mode == 'v18_full' or mode == 'v18_baseline':
            if self._use_conditioned_lstm:
                self.lstm = ContextConditionedLSTM(
                    input_dim=input_features,
                    hidden_dim=lstm_hidden,
                    num_layers=lstm_num_layers,
                    context_dim=32,  # GRN output dimension
                    dropout=lstm_dropout
                )
            else:
                self.lstm = nn.LSTM(
                    input_size=input_features,
                    hidden_size=lstm_hidden,
                    num_layers=lstm_num_layers,
                    batch_first=True,
                    dropout=lstm_dropout if lstm_num_layers > 1 else 0
                )
            lstm_dim = lstm_hidden
        elif mode == 'lstm':
            # Standard LSTM (not context-conditioned)
            self.lstm = nn.LSTM(
                input_size=input_features,
                hidden_size=lstm_hidden,
                num_layers=lstm_num_layers,
                batch_first=True,
                dropout=lstm_dropout if lstm_num_layers > 1 else 0
            )
            lstm_dim = lstm_hidden
        # concat and cqa_only modes have no LSTM

        self.lstm_output_dim = lstm_dim

        # ==========================================
        # CNN + Attention Branch (all modes)
        # ==========================================
        if self.use_split_attention:
            # Indices loaded from TemporalFeatureConfig (single source of truth)
            self.split_attention = SplitFeatureAttention(
                # price_indices and volume_indices default to _TEMPORAL_CONFIG values
                price_cnn_channels=48,
                volume_cnn_channels=48,
                kernel_size=3,
                num_heads_price=2,
                num_heads_volume=2,
                dropout=attention_dropout,
                use_rope=use_rope,
                max_seq_len=sequence_length * 2
            )
            self.structure_output_dim = self.split_attention.output_dim
            self.conv_3 = None
            self.conv_5 = None
            self.cnn_norm = None
            self.self_attention = None
            self.cnn_encoder = None
        else:
            # Standard CNN + Self-Attention (used in both V18 and V20)
            # Use TemporalEncoderCNN for V20 modes, inline CNN for V18 modes
            if mode in ('concat', 'lstm', 'cqa_only'):
                # V20-style: Use TemporalEncoderCNN
                self.cnn_encoder = TemporalEncoderCNN(
                    input_features=input_features,
                    cnn_channels=tuple(cnn_channels),
                    kernel_sizes=tuple(cnn_kernel_sizes)
                )
                self.conv_3 = None
                self.conv_5 = None
                self.cnn_norm = None
            else:
                # V18-style: Inline CNN
                self.conv_3 = nn.Conv1d(input_features, cnn_channels[0], cnn_kernel_sizes[0], padding=cnn_kernel_sizes[0]//2)
                self.conv_5 = nn.Conv1d(input_features, cnn_channels[1], cnn_kernel_sizes[1], padding=cnn_kernel_sizes[1]//2)
                self.cnn_norm = nn.LayerNorm(self.cnn_output_dim)
                self.cnn_encoder = None

            # Self-Attention with optional RoPE
            if use_rope:
                self.self_attention = RoPEMultiheadAttention(
                    embed_dim=self.cnn_output_dim,
                    num_heads=num_attention_heads,
                    dropout=attention_dropout,
                    max_seq_len=sequence_length * 2,
                    batch_first=True
                )
            else:
                self.self_attention = nn.MultiheadAttention(
                    embed_dim=self.cnn_output_dim,
                    num_heads=num_attention_heads,
                    dropout=attention_dropout,
                    batch_first=True
                )
            self.structure_output_dim = self.cnn_output_dim
            self.split_attention = None

        # ==========================================
        # Context Processing (mode-dependent)
        # ==========================================
        self.context_grn = None
        self.context_query_attn = None
        context_dim = 0
        cqa_dim = 0

        if self.use_context:
            # All modes with context get a GRN
            self.context_grn = GatedResidualNetwork(
                input_dim=context_features,
                hidden_dim=32,
                dropout=0.1
            )
            context_dim = 32

            # CQA is used in all modes except 'concat'
            if mode != 'concat':
                self.context_query_attn = ContextQueryAttention(
                    context_dim=32,
                    temporal_dim=self.cnn_output_dim,
                    num_heads=num_attention_heads,
                    dropout=attention_dropout
                )
                cqa_dim = self.cnn_output_dim

        # For backward compatibility (V18 attribute names)
        self.context_dim = context_dim
        self.guided_dim = cqa_dim

        # ==========================================
        # Fusion & Classification
        # ==========================================
        combined_dim = self.structure_output_dim + context_dim + lstm_dim + cqa_dim

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout / 2),
            nn.Linear(fusion_hidden // 2, num_classes)
        )

        # Store dimensions for architecture summary
        self._combined_dim = combined_dim
        self._component_dims = {
            'structure': self.structure_output_dim,
            'context': context_dim,
            'lstm': lstm_dim,
            'cqa': cqa_dim
        }

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through all branches.

        Args:
            x: (Batch, Seq_Len, Features) temporal sequence
            context: (Batch, Context_Features) static context features

        Returns:
            logits: (Batch, num_classes) class logits
        """
        batch_size = x.size(0)

        # Process context if available
        emb_context = None
        if self.use_context:
            if context is None:
                context = torch.zeros(batch_size, self.context_features, device=x.device)
            emb_context = self.context_grn(context)

        # ==========================================
        # 1. LSTM Branch (mode-dependent)
        # ==========================================
        emb_lstm = None
        if self.lstm is not None:
            if self._use_conditioned_lstm and emb_context is not None:
                emb_lstm, _ = self.lstm(x, emb_context)
            else:
                lstm_out, _ = self.lstm(x)
                emb_lstm = lstm_out[:, -1, :]

        # ==========================================
        # 2. Structure Branch (CNN + Attention)
        # ==========================================
        if self.use_split_attention:
            emb_price, emb_volume, _ = self.split_attention(x)
            emb_structure = torch.cat([emb_price, emb_volume], dim=1)
            cnn_seq = emb_structure.unsqueeze(1).expand(-1, x.size(1), -1)
        elif self.cnn_encoder is not None:
            # V20-style: Use TemporalEncoderCNN
            cnn_seq = self.cnn_encoder(x)
            attn_out, _ = self.self_attention(cnn_seq, cnn_seq, cnn_seq)
            emb_structure = attn_out.mean(dim=1)
        else:
            # V18-style: Inline CNN
            x_conv = x.transpose(1, 2)
            cnn_seq = torch.cat([
                F.relu(self.conv_3(x_conv)),
                F.relu(self.conv_5(x_conv))
            ], dim=1).transpose(1, 2)
            cnn_seq = self.cnn_norm(cnn_seq)
            attn_out, _ = self.self_attention(cnn_seq, cnn_seq, cnn_seq)
            emb_structure = attn_out.mean(dim=1)

        # ==========================================
        # 3. Context-Query Attention (if applicable)
        # ==========================================
        emb_cqa = None
        if self.context_query_attn is not None and emb_context is not None:
            emb_cqa, _ = self.context_query_attn(emb_context, cnn_seq)

        # ==========================================
        # 4. Fusion
        # ==========================================
        components = [emb_structure]
        if emb_context is not None:
            components.append(emb_context)
        if emb_lstm is not None:
            components.append(emb_lstm)
        if emb_cqa is not None:
            components.append(emb_cqa)

        combined = torch.cat(components, dim=1)
        logits = self.fusion(combined)
        return logits

    def get_architecture_summary(self) -> Dict:
        """Return summary of architecture for logging."""
        return {
            'mode': self.mode,
            'ablation_mode': self.mode,  # Backward compatibility with V20
            'combined_dim': self._combined_dim,
            'component_dims': self._component_dims,
            'has_lstm': self.lstm is not None,
            'has_cqa': self.context_query_attn is not None,
            'lstm_type': type(self.lstm).__name__ if self.lstm else 'None',
            'total_params': sum(p.numel() for p in self.parameters()),
            'use_conditioned_lstm': self._use_conditioned_lstm,
            'use_split_attention': self.use_split_attention,
            'use_rope': self.use_rope
        }

    def get_feature_importance(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict:
        """
        Extracts both Structural (Self) and Relevance (Context) attention weights.

        Args:
            x: (Batch, Seq_Len, Features) temporal features
            context: (Batch, Context_Features) static context features

        Returns:
            Dict with temporal_importance, structure_weights, relevance_weights, etc.
        """
        with torch.no_grad():
            if self.use_split_attention:
                _, _, split_attn_weights = self.split_attention(x)
                result = {
                    'price_weights': split_attn_weights['price'].cpu().numpy(),
                    'volume_weights': split_attn_weights['volume'].cpu().numpy(),
                    'temporal_importance': (
                        split_attn_weights['price'].mean(dim=1).cpu().numpy() +
                        split_attn_weights['volume'].mean(dim=1).cpu().numpy()
                    ) / 2,
                    'structure_weights': split_attn_weights['price'].cpu().numpy()
                }

                if self.use_context and context is not None:
                    emb_context = self.context_grn(context)
                    emb_price, emb_volume, _ = self.split_attention(x)
                    emb_structure = torch.cat([emb_price, emb_volume], dim=1)
                    cnn_seq = emb_structure.unsqueeze(1).expand(-1, x.size(1), -1)
                    if self.context_query_attn is not None:
                        _, ctx_attn_weights = self.context_query_attn(emb_context, cnn_seq)
                        result['relevance_weights'] = ctx_attn_weights.squeeze(1).cpu().numpy()
                        result['context_focus_mean'] = result['relevance_weights'].mean(axis=0)
            else:
                # Get CNN sequence
                if self.cnn_encoder is not None:
                    cnn_seq = self.cnn_encoder(x)
                else:
                    x_conv = x.transpose(1, 2)
                    cnn_seq = torch.cat([
                        F.relu(self.conv_3(x_conv)),
                        F.relu(self.conv_5(x_conv))
                    ], dim=1).transpose(1, 2)
                    cnn_seq = self.cnn_norm(cnn_seq)

                _, self_attn = self.self_attention(cnn_seq, cnn_seq, cnn_seq)

                result = {
                    'temporal_importance': self_attn.mean(dim=1).cpu().numpy() if self_attn is not None else None,
                    'structure_weights': self_attn.cpu().numpy() if self_attn is not None else None
                }

                if self.use_context and context is not None and self.context_query_attn is not None:
                    emb_context = self.context_grn(context)
                    _, ctx_attn_weights = self.context_query_attn(emb_context, cnn_seq)
                    result['relevance_weights'] = ctx_attn_weights.squeeze(1).cpu().numpy()
                    result['context_focus_mean'] = result['relevance_weights'].mean(axis=0)
                    # V20 compatibility
                    result['cqa_weights'] = result['relevance_weights']
                    result['cqa_focus_mean'] = result['context_focus_mean']

            return result

    @staticmethod
    def from_checkpoint(
        checkpoint_path: str,
        device: str = 'cpu',
        strict: bool = True
    ) -> 'HybridFeatureNetwork':
        """
        Load model from checkpoint with automatic mode detection.

        Supports both V18 and V20 checkpoint formats.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Device to load model on
            strict: Whether to enforce strict state_dict loading

        Returns:
            HybridFeatureNetwork instance with loaded weights
        """
        import torch
        from pathlib import Path

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        state_dict = checkpoint['model_state_dict']

        # Detect mode from checkpoint
        # V20 checkpoints have 'ablation_mode' in config
        # V18 checkpoints don't have it
        ablation_mode = config.get('ablation_mode')

        if ablation_mode is not None:
            mode = ablation_mode
        else:
            # V18 checkpoint - use v18_full
            mode = 'v18_full'

        # Detect input features from state dict
        if 'lstm.weight_ih_l0' in state_dict:
            input_features = state_dict['lstm.weight_ih_l0'].shape[1]
        elif 'lstm.lstm.weight_ih_l0' in state_dict:
            # ContextConditionedLSTM
            input_features = state_dict['lstm.lstm.weight_ih_l0'].shape[1]
        else:
            input_features = config.get('input_features', 10)

        # Detect context features
        has_context = any('context_grn' in k for k in state_dict.keys())
        if has_context and 'context_grn.fc1.weight' in state_dict:
            context_features = state_dict['context_grn.fc1.weight'].shape[1]
        else:
            context_features = 0

        # Detect if using conditioned LSTM
        use_conditioned_lstm = 'lstm.h_proj.weight' in state_dict

        # Create model
        model = HybridFeatureNetwork(
            mode=mode,
            input_features=input_features,
            context_features=context_features,
            use_conditioned_lstm=use_conditioned_lstm,
            num_classes=config.get('num_classes', 3),
            fusion_dropout=config.get('dropout', 0.3)
        )

        # Load state dict
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        model.eval()

        return model


# =============================================================================
# FACTORY FUNCTION (V20 compatibility)
# =============================================================================

def create_v20_model(
    ablation_mode: str = 'lstm',
    context_features: int = 13,
    **kwargs
) -> HybridFeatureNetwork:
    """
    Factory function to create model with specified ablation mode.

    This is provided for backward compatibility with V20 code.

    Args:
        ablation_mode: 'concat', 'lstm', 'cqa_only', or 'v18_baseline'
        context_features: Number of context features (default: 13)
        **kwargs: Additional arguments passed to HybridFeatureNetwork

    Returns:
        Configured HybridFeatureNetwork instance
    """
    return HybridFeatureNetwork(
        mode=ablation_mode,
        context_features=context_features,
        **kwargs
    )


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED MODEL TEST (V18 + V20)")
    print("=" * 70)

    batch_size = 4
    x = torch.randn(batch_size, 20, 10)
    context = torch.randn(batch_size, 13)

    for mode in HybridFeatureNetwork.VALID_MODES:
        print(f"\n[{mode.upper()}] Testing mode...")

        model = HybridFeatureNetwork(mode=mode, context_features=13)
        summary = model.get_architecture_summary()

        print(f"  Combined dim: {summary['combined_dim']}")
        print(f"  Component dims: {summary['component_dims']}")
        print(f"  Has LSTM: {summary['has_lstm']} ({summary['lstm_type']})")
        print(f"  Has CQA: {summary['has_cqa']}")
        print(f"  Total params: {summary['total_params']:,}")

        logits = model(x, context)
        print(f"  Output shape: {logits.shape}")

        importance = model.get_feature_importance(x, context)
        print(f"  Importance keys: {list(importance.keys())}")

        assert logits.shape == (batch_size, 3), f"Expected (4, 3), got {logits.shape}"

    # Test v18_full with conditioned LSTM
    print("\n[V18_FULL + CONDITIONED LSTM]")
    model_conditioned = HybridFeatureNetwork(
        mode='v18_full',
        context_features=13,
        use_conditioned_lstm=True
    )
    assert model_conditioned._use_conditioned_lstm == True
    assert isinstance(model_conditioned.lstm, ContextConditionedLSTM)
    print(f"  Conditioned LSTM: {type(model_conditioned.lstm).__name__}")

    # Test v18_full without conditioned LSTM (legacy)
    print("\n[V18_FULL + STANDARD LSTM (legacy)]")
    model_standard = HybridFeatureNetwork(
        mode='v18_full',
        context_features=13,
        use_conditioned_lstm=False
    )
    assert model_standard._use_conditioned_lstm == False
    assert isinstance(model_standard.lstm, nn.LSTM)
    print(f"  Standard LSTM: {type(model_standard.lstm).__name__}")

    print("\n" + "=" * 70)
    print("All unified model tests passed!")
    print("=" * 70)

    # Comparison table
    print("\n\nMODE COMPARISON:")
    print("-" * 70)
    print(f"{'Mode':<15} {'Params':>12} {'Has LSTM':>10} {'Has CQA':>10} {'Combined':>10}")
    print("-" * 70)

    for mode in HybridFeatureNetwork.VALID_MODES:
        model = HybridFeatureNetwork(mode=mode, context_features=13)
        summary = model.get_architecture_summary()
        print(f"{mode:<15} {summary['total_params']:>12,} {str(summary['has_lstm']):>10} {str(summary['has_cqa']):>10} {summary['combined_dim']:>10}")

    print("-" * 70)
