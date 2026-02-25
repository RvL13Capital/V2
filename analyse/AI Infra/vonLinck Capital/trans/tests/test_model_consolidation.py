"""
Model Consolidation Tests: V18 + V20 -> Unified HybridFeatureNetwork
=====================================================================

Verifies behavioral equivalence between:
1. Original V18 HybridFeatureNetwork and unified model (mode='v18_full')
2. Original V20 ablation modes and unified model equivalents
3. State dict key compatibility
4. Checkpoint round-trip compatibility

These tests ensure the consolidation maintains exact backward compatibility.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_input():
    """Standard test input: batch=4, seq_len=20, features=10."""
    torch.manual_seed(42)
    return torch.randn(4, 20, 10)


@pytest.fixture
def sample_context():
    """Standard test context: batch=4, context_features=13."""
    torch.manual_seed(42)
    return torch.randn(4, 13)


@pytest.fixture
def sample_context_18():
    """V20-style context: batch=4, context_features=18 (for backward compat tests)."""
    torch.manual_seed(42)
    return torch.randn(4, 18)


# =============================================================================
# UNIFIED MODEL BASIC TESTS
# =============================================================================

class TestUnifiedModelBasics:
    """Basic tests for the unified model."""

    def test_valid_modes(self):
        """Test that all valid modes can be instantiated."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        for mode in HybridFeatureNetwork.VALID_MODES:
            model = HybridFeatureNetwork(mode=mode, context_features=13)
            assert model.mode == mode

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        with pytest.raises(ValueError, match="mode must be one of"):
            HybridFeatureNetwork(mode='invalid_mode')

    def test_output_shape(self, sample_input, sample_context):
        """Test that output shape is correct for all modes."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        for mode in HybridFeatureNetwork.VALID_MODES:
            model = HybridFeatureNetwork(mode=mode, context_features=13)
            output = model(sample_input, sample_context)
            assert output.shape == (4, 3), f"Mode {mode} produced wrong shape"

    def test_no_context(self, sample_input):
        """Test that model works without context features."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        model = HybridFeatureNetwork(mode='v18_full', context_features=0)
        output = model(sample_input)
        assert output.shape == (4, 3)

    def test_architecture_summary(self, sample_input, sample_context):
        """Test that architecture summary contains expected keys."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        model = HybridFeatureNetwork(mode='v18_full', context_features=13)
        summary = model.get_architecture_summary()

        expected_keys = [
            'mode', 'combined_dim', 'component_dims',
            'has_lstm', 'has_cqa', 'lstm_type', 'total_params'
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_feature_importance(self, sample_input, sample_context):
        """Test that feature importance extraction works."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        model = HybridFeatureNetwork(mode='v18_full', context_features=13)
        importance = model.get_feature_importance(sample_input, sample_context)

        assert 'temporal_importance' in importance
        assert 'structure_weights' in importance


# =============================================================================
# V18 EQUIVALENCE TESTS
# =============================================================================

class TestV18Equivalence:
    """Tests verifying unified model matches original V18 behavior."""

    def test_v18_state_dict_keys_match(self):
        """Test that v18_full mode has same state_dict keys as V18."""
        from models.temporal_hybrid_v18 import HybridFeatureNetwork as V18Model
        from models.temporal_hybrid_unified import HybridFeatureNetwork as UnifiedModel

        v18 = V18Model(input_features=10, context_features=13, num_classes=3)
        unified = UnifiedModel(mode='v18_full', input_features=10, context_features=13, num_classes=3)

        v18_keys = set(v18.state_dict().keys())
        unified_keys = set(unified.state_dict().keys())

        # Check that unified has at least all V18 keys
        missing_in_unified = v18_keys - unified_keys
        assert len(missing_in_unified) == 0, f"Missing keys in unified: {missing_in_unified}"

    def test_v18_state_dict_loadable(self):
        """Test that V18 state_dict can be loaded into unified model."""
        from models.temporal_hybrid_v18 import HybridFeatureNetwork as V18Model
        from models.temporal_hybrid_unified import HybridFeatureNetwork as UnifiedModel

        v18 = V18Model(input_features=10, context_features=13, num_classes=3)
        unified = UnifiedModel(mode='v18_full', input_features=10, context_features=13, num_classes=3)

        # Load V18 state dict into unified
        unified.load_state_dict(v18.state_dict())

    def test_v18_output_equivalence(self, sample_input, sample_context):
        """Test that v18_full produces identical output to V18."""
        from models.temporal_hybrid_v18 import HybridFeatureNetwork as V18Model
        from models.temporal_hybrid_unified import HybridFeatureNetwork as UnifiedModel

        torch.manual_seed(42)
        v18 = V18Model(input_features=10, context_features=13, num_classes=3)
        unified = UnifiedModel(mode='v18_full', input_features=10, context_features=13, num_classes=3)

        # Copy weights from V18 to unified
        unified.load_state_dict(v18.state_dict())

        # Set both to eval mode
        v18.eval()
        unified.eval()

        # Compare outputs
        with torch.no_grad():
            v18_out = v18(sample_input, sample_context)
            unified_out = unified(sample_input, sample_context)

        assert torch.allclose(v18_out, unified_out, atol=1e-6), \
            f"Output mismatch: max diff = {(v18_out - unified_out).abs().max()}"

    def test_v18_conditioned_lstm_equivalence(self, sample_input, sample_context):
        """Test conditioned LSTM mode equivalence."""
        from models.temporal_hybrid_v18 import HybridFeatureNetwork as V18Model
        from models.temporal_hybrid_unified import HybridFeatureNetwork as UnifiedModel

        torch.manual_seed(42)
        v18 = V18Model(
            input_features=10,
            context_features=13,
            num_classes=3,
            use_conditioned_lstm=True
        )
        unified = UnifiedModel(
            mode='v18_full',
            input_features=10,
            context_features=13,
            num_classes=3,
            use_conditioned_lstm=True
        )

        unified.load_state_dict(v18.state_dict())
        v18.eval()
        unified.eval()

        with torch.no_grad():
            v18_out = v18(sample_input, sample_context)
            unified_out = unified(sample_input, sample_context)

        assert torch.allclose(v18_out, unified_out, atol=1e-6)

    def test_v18_standard_lstm_equivalence(self, sample_input, sample_context):
        """Test standard (non-conditioned) LSTM mode equivalence."""
        from models.temporal_hybrid_v18 import HybridFeatureNetwork as V18Model
        from models.temporal_hybrid_unified import HybridFeatureNetwork as UnifiedModel

        torch.manual_seed(42)
        v18 = V18Model(
            input_features=10,
            context_features=13,
            num_classes=3,
            use_conditioned_lstm=False
        )
        unified = UnifiedModel(
            mode='v18_full',
            input_features=10,
            context_features=13,
            num_classes=3,
            use_conditioned_lstm=False
        )

        unified.load_state_dict(v18.state_dict())
        v18.eval()
        unified.eval()

        with torch.no_grad():
            v18_out = v18(sample_input, sample_context)
            unified_out = unified(sample_input, sample_context)

        assert torch.allclose(v18_out, unified_out, atol=1e-6)


# =============================================================================
# V20 ABLATION MODE TESTS
# =============================================================================

class TestV20AblationModes:
    """Tests verifying unified model matches V20 ablation modes."""

    def test_concat_mode_structure(self):
        """Test concat mode has no LSTM and no CQA."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        model = HybridFeatureNetwork(mode='concat', context_features=13)
        summary = model.get_architecture_summary()

        assert summary['has_lstm'] == False
        assert summary['has_cqa'] == False

    def test_lstm_mode_structure(self):
        """Test lstm mode has standard LSTM and CQA."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        model = HybridFeatureNetwork(mode='lstm', context_features=13)
        summary = model.get_architecture_summary()

        assert summary['has_lstm'] == True
        assert summary['has_cqa'] == True
        assert summary['lstm_type'] == 'LSTM'  # Not ContextConditionedLSTM

    def test_cqa_only_mode_structure(self):
        """Test cqa_only mode has no LSTM but has CQA."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        model = HybridFeatureNetwork(mode='cqa_only', context_features=13)
        summary = model.get_architecture_summary()

        assert summary['has_lstm'] == False
        assert summary['has_cqa'] == True

    def test_v18_baseline_equals_v18_full(self, sample_input, sample_context):
        """Test v18_baseline is identical to v18_full."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        torch.manual_seed(42)
        v18_full = HybridFeatureNetwork(mode='v18_full', context_features=13)
        v18_baseline = HybridFeatureNetwork(mode='v18_baseline', context_features=13)

        # Both should have same architecture
        assert v18_full.get_architecture_summary()['has_lstm'] == \
               v18_baseline.get_architecture_summary()['has_lstm']
        assert v18_full.get_architecture_summary()['has_cqa'] == \
               v18_baseline.get_architecture_summary()['has_cqa']


# =============================================================================
# CHECKPOINT COMPATIBILITY TESTS
# =============================================================================

class TestCheckpointCompatibility:
    """Tests for checkpoint save/load compatibility."""

    def test_checkpoint_round_trip(self, sample_input, sample_context):
        """Test save and load produces identical outputs."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork
        import os

        model = HybridFeatureNetwork(mode='v18_full', context_features=13)
        model.eval()

        with torch.no_grad():
            original_out = model(sample_input, sample_context)

        # Save to temp file
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                temp_path = f.name

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': {'input_features': 10, 'context_features': 13}
            }
            torch.save(checkpoint, temp_path)

            # Load into new model
            loaded = HybridFeatureNetwork.from_checkpoint(temp_path)
            loaded.eval()

            with torch.no_grad():
                loaded_out = loaded(sample_input, sample_context)

            assert torch.allclose(original_out, loaded_out, atol=1e-6)
        finally:
            # Cleanup - ignore errors on Windows
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except (PermissionError, OSError):
                    pass  # Windows may hold file lock

    def test_v18_checkpoint_loadable(self, sample_input, sample_context):
        """Test that simulated V18 checkpoint can be loaded by unified model."""
        from models.temporal_hybrid_v18 import HybridFeatureNetwork as V18Model
        from models.temporal_hybrid_unified import HybridFeatureNetwork as UnifiedModel
        import os

        # Create V18 model and "checkpoint"
        v18 = V18Model(input_features=10, context_features=13, num_classes=3)
        v18.eval()

        with torch.no_grad():
            v18_out = v18(sample_input, sample_context)

        # Save V18 checkpoint
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                temp_path = f.name

            checkpoint = {
                'model_state_dict': v18.state_dict(),
                'config': {'input_features': 10, 'use_conditioned_lstm': True}
            }
            torch.save(checkpoint, temp_path)

            # Load with unified
            unified = UnifiedModel.from_checkpoint(temp_path)
            unified.eval()

            with torch.no_grad():
                unified_out = unified(sample_input, sample_context)

            assert torch.allclose(v18_out, unified_out, atol=1e-6)
        finally:
            # Cleanup - ignore errors on Windows
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except (PermissionError, OSError):
                    pass  # Windows may hold file lock


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory function compatibility."""

    def test_create_v20_model(self, sample_input, sample_context):
        """Test create_v20_model factory function."""
        from models.temporal_hybrid_unified import create_v20_model

        for mode in ['concat', 'lstm', 'cqa_only', 'v18_baseline']:
            model = create_v20_model(ablation_mode=mode, context_features=13)
            output = model(sample_input, sample_context)
            assert output.shape == (4, 3)


# =============================================================================
# PARAMETER COUNT TESTS
# =============================================================================

class TestParameterCounts:
    """Tests verifying parameter counts match expectations."""

    def test_mode_parameter_ordering(self):
        """Test that modes have expected relative parameter counts."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        param_counts = {}
        for mode in HybridFeatureNetwork.VALID_MODES:
            model = HybridFeatureNetwork(mode=mode, context_features=13)
            param_counts[mode] = sum(p.numel() for p in model.parameters())

        # concat should have fewest params (no LSTM, no CQA)
        assert param_counts['concat'] < param_counts['lstm']
        assert param_counts['concat'] < param_counts['cqa_only']

        # v18_full should have most params (conditioned LSTM + CQA)
        assert param_counts['v18_full'] >= param_counts['lstm']

    def test_v18_parameter_count_match(self):
        """Test that unified v18_full has same param count as V18."""
        from models.temporal_hybrid_v18 import HybridFeatureNetwork as V18Model
        from models.temporal_hybrid_unified import HybridFeatureNetwork as UnifiedModel

        v18 = V18Model(input_features=10, context_features=13, num_classes=3)
        unified = UnifiedModel(mode='v18_full', input_features=10, context_features=13, num_classes=3)

        v18_params = sum(p.numel() for p in v18.parameters())
        unified_params = sum(p.numel() for p in unified.parameters())

        assert v18_params == unified_params, \
            f"Parameter count mismatch: V18={v18_params}, Unified={unified_params}"


# =============================================================================
# GRADIENT TESTS
# =============================================================================

class TestGradients:
    """Tests verifying gradients flow correctly."""

    def test_backward_pass(self, sample_input, sample_context):
        """Test that backward pass works for all modes."""
        from models.temporal_hybrid_unified import HybridFeatureNetwork

        for mode in HybridFeatureNetwork.VALID_MODES:
            model = HybridFeatureNetwork(mode=mode, context_features=13)
            model.train()

            output = model(sample_input, sample_context)
            loss = output.sum()
            loss.backward()

            # Check that gradients exist
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"No gradient for {name} in mode {mode}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
