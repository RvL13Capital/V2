"""
TRANS Models Module
===================

Exports unified hybrid temporal model and related components.

The primary model class is HybridFeatureNetwork from temporal_hybrid_unified,
which supports multiple architecture modes via the `mode` parameter:
- v18_full (default): Full production architecture
- concat, lstm, cqa_only, v18_baseline: Ablation modes

Jan 2026 - Consolidated from V18 + V20 into unified model
"""

from .temporal_hybrid_unified import (
    HybridFeatureNetwork,
    GatedResidualNetwork,
    RotaryPositionalEmbedding,
    RoPEMultiheadAttention,
    ContextConditionedLSTM,
    ContextQueryAttention,
    SplitFeatureAttention,
    TemporalEncoderCNN,
    create_v20_model,
)

# Backward compatibility alias
HybridTemporalClassifier = HybridFeatureNetwork

from .asymmetric_loss import TemporalAsymmetricLoss

__all__ = [
    # Primary model class
    'HybridFeatureNetwork',
    # Backward compatibility alias
    'HybridTemporalClassifier',
    # Component classes
    'GatedResidualNetwork',
    'RotaryPositionalEmbedding',
    'RoPEMultiheadAttention',
    'ContextConditionedLSTM',
    'ContextQueryAttention',
    'SplitFeatureAttention',
    'TemporalEncoderCNN',
    # Factory function
    'create_v20_model',
    # Loss function
    'TemporalAsymmetricLoss',
]
