"""
Multi-Modal Hybrid Temporal Model V18: Context-Query Attention (CQA)
====================================================================

DEPRECATION NOTICE: This module is deprecated.
Use models.temporal_hybrid_unified instead.

This module now re-exports all classes from temporal_hybrid_unified for
backward compatibility. It will be removed after a 90-day transition period.

Migration:
    # OLD (deprecated):
    from models.temporal_hybrid_v18 import HybridFeatureNetwork

    # NEW (recommended):
    from models.temporal_hybrid_unified import HybridFeatureNetwork

Jan 2026 - Deprecated in favor of unified model
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "models.temporal_hybrid_v18 is deprecated. "
    "Use models.temporal_hybrid_unified instead. "
    "This module will be removed after 90 days.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all classes from unified module for backward compatibility
from models.temporal_hybrid_unified import (
    # Core classes
    HybridFeatureNetwork,
    RotaryPositionalEmbedding,
    RoPEMultiheadAttention,
    GatedResidualNetwork,
    ContextConditionedLSTM,
    ContextQueryAttention,
    SplitFeatureAttention,
    TemporalEncoderCNN,
    # Factory function (V20 compatibility)
    create_v20_model,
)

__all__ = [
    'HybridFeatureNetwork',
    'RotaryPositionalEmbedding',
    'RoPEMultiheadAttention',
    'GatedResidualNetwork',
    'ContextConditionedLSTM',
    'ContextQueryAttention',
    'SplitFeatureAttention',
    'TemporalEncoderCNN',
    'create_v20_model',
]


if __name__ == "__main__":
    print("V18 Deprecation Shim Test")
    print("=" * 50)
    print("This module now imports from temporal_hybrid_unified")
    print()

    # Verify exports work
    import torch

    batch_size = 4
    x = torch.randn(batch_size, 20, 10)
    context = torch.randn(batch_size, 13)

    model = HybridFeatureNetwork(
        input_features=10,
        context_features=13,
        num_classes=3
    )

    logits = model(x, context)
    print(f"Input shape: {x.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Output shape: {logits.shape}")
    print()
    print("V18 shim test passed!")
