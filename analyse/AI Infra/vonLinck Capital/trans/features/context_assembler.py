"""
Context Feature Assembler for Branch B (GRN) Input Validation
==============================================================

Provides numpy array validation for context features passed to the
TemporalHybridModel's Branch B (Gated Residual Network).

This complements the dict-based validation in config/context_features.py
by validating the final numpy arrays used in training and inference.

The 18-Feature Standard (Jan 2026):
    0. retention_rate       - (Close-Low)/(High-Low) candle strength
    1. trend_position       - 200-SMA relative position
    2. base_duration        - Consolidation maturity (log-normalized)
    3. relative_volume      - Recent vs historical volume
    4. distance_to_high     - Proximity to 52-week high
    5. log_float            - Log10 of shares outstanding
    6. log_dollar_volume    - Log10 of avg daily dollar volume
    7. dormancy_shock       - Vol spike from dormant state
    8. vol_dryup_ratio      - Volume exhaustion indicator
    9. price_position_at_end - Close position in consolidation box
   10. volume_shock         - Recent volume spike magnitude
   11. bbw_slope_5d         - Bollinger Band width trend
   12. vol_trend_5d         - Recent volume trend
   13. coil_intensity       - Combined coil quality score
   14. relative_strength_cohort - vs universe median (no SPY leakage)
   15. risk_width_pct       - Structural stop distance
   16. vol_contraction_intensity - Volume drying up
   17. obv_divergence       - OBV vs Price divergence

Usage:
    from features.context_assembler import ContextFeatureAssembler

    # Validate context array shape
    ContextFeatureAssembler.validate_shape(context_array)

    # Get default context for missing data
    default_context = ContextFeatureAssembler.get_default_array(batch_size=32)
"""

import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.context_features import (
    CONTEXT_FEATURES,
    NUM_CONTEXT_FEATURES,
    CONTEXT_FEATURE_DEFAULTS,
    CONTEXT_FEATURE_RANGES,
    normalize_context_feature
)


class ContextFeatureAssembler:
    """
    Validates and assembles context feature arrays for Branch B (GRN) input.

    This class enforces the 18-feature standard and provides utilities for:
    - Validating context array shapes before model input
    - Generating default context arrays when data is unavailable
    - Normalizing context features to stable ranges

    All class methods are static to avoid instantiation overhead.
    """

    # Schema constants (re-exported from config for convenience)
    SCHEMA = CONTEXT_FEATURES
    NUM_FEATURES = NUM_CONTEXT_FEATURES  # 18 (Jan 2026 - retention_rate replaced float_turnover)

    @classmethod
    def validate_shape(
        cls,
        context: np.ndarray,
        batch_size: Optional[int] = None,
        raise_on_error: bool = True
    ) -> bool:
        """
        Validate that context array matches the 18-feature schema.

        Args:
            context: Context array of shape (batch_size, 18) or (18,)
            batch_size: Expected batch size (optional, validates if provided)
            raise_on_error: If True, raise ValueError on mismatch; if False, return bool

        Returns:
            True if valid, False if invalid (only when raise_on_error=False)

        Raises:
            ValueError: If context shape doesn't match schema (when raise_on_error=True)

        Examples:
            >>> ctx = np.random.randn(32, 18).astype(np.float32)
            >>> ContextFeatureAssembler.validate_shape(ctx)  # OK
            True

            >>> ctx_wrong = np.random.randn(32, 8).astype(np.float32)
            >>> ContextFeatureAssembler.validate_shape(ctx_wrong)
            ValueError: Context has 8 features, expected 18
        """
        # Handle 1D case
        if context.ndim == 1:
            actual_features = context.shape[0]
        elif context.ndim == 2:
            actual_features = context.shape[1]
        else:
            if raise_on_error:
                raise ValueError(f"Context must be 1D or 2D, got {context.ndim}D")
            return False

        # Validate feature count
        if actual_features != cls.NUM_FEATURES:
            if raise_on_error:
                raise ValueError(
                    f"Context has {actual_features} features, expected {cls.NUM_FEATURES}. "
                    f"Schema: {cls.SCHEMA}"
                )
            return False

        # Validate batch size if provided
        if batch_size is not None and context.ndim == 2:
            if context.shape[0] != batch_size:
                if raise_on_error:
                    raise ValueError(
                        f"Context batch size {context.shape[0]} != expected {batch_size}"
                    )
                return False

        return True

    @classmethod
    def get_default_array(cls, batch_size: int = 1) -> np.ndarray:
        """
        Generate a default context array with neutral values.

        Use this when context data is unavailable (e.g., missing market cap data).
        The default values are designed to be neutral and not bias the model.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            np.ndarray of shape (batch_size, 18) with float32 dtype

        Examples:
            >>> default = ContextFeatureAssembler.get_default_array(batch_size=32)
            >>> default.shape
            (32, 18)
        """
        # Build defaults in schema order
        defaults = np.array([
            CONTEXT_FEATURE_DEFAULTS[feature_name]
            for feature_name in cls.SCHEMA
        ], dtype=np.float32)

        # Tile for batch
        return np.tile(defaults, (batch_size, 1))

    @classmethod
    def normalize_array(cls, context: np.ndarray) -> np.ndarray:
        """
        Normalize context array to stable ranges for neural network input.

        Each feature is clipped to its expected range and scaled to [0, 1].

        Args:
            context: Raw context array of shape (batch_size, 18) or (18,)

        Returns:
            Normalized context array with same shape

        Note:
            This applies the normalization defined in CONTEXT_FEATURE_RANGES.
        """
        cls.validate_shape(context)

        # Handle both 1D and 2D
        if context.ndim == 1:
            normalized = np.zeros_like(context)
            for i, feature_name in enumerate(cls.SCHEMA):
                normalized[i] = normalize_context_feature(feature_name, context[i])
        else:
            normalized = np.zeros_like(context)
            for i, feature_name in enumerate(cls.SCHEMA):
                for j in range(context.shape[0]):
                    normalized[j, i] = normalize_context_feature(feature_name, context[j, i])

        return normalized

    @classmethod
    def get_feature_index(cls, feature_name: str) -> int:
        """
        Get the index of a feature in the schema.

        Args:
            feature_name: Name of the feature (e.g., 'retention_rate')

        Returns:
            Index position (0-17)

        Raises:
            ValueError: If feature_name not in schema
        """
        if feature_name not in cls.SCHEMA:
            raise ValueError(
                f"Unknown feature '{feature_name}'. "
                f"Valid features: {cls.SCHEMA}"
            )
        return cls.SCHEMA.index(feature_name)

    @classmethod
    def describe(cls) -> str:
        """
        Return a human-readable description of the context feature schema.

        Returns:
            Multi-line string describing all 18 features
        """
        lines = ["Context Feature Schema (18 features for Branch B GRN):", ""]
        for i, feature_name in enumerate(cls.SCHEMA):
            default = CONTEXT_FEATURE_DEFAULTS[feature_name]
            range_min, range_max = CONTEXT_FEATURE_RANGES.get(feature_name, (None, None))
            lines.append(f"  {i}. {feature_name}")
            lines.append(f"     Default: {default}, Range: [{range_min}, {range_max}]")
        return "\n".join(lines)


# Convenience function for quick validation
def validate_context(context: np.ndarray, batch_size: Optional[int] = None) -> bool:
    """
    Quick validation function - raises ValueError if invalid.

    Args:
        context: Context array to validate
        batch_size: Expected batch size (optional)

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    return ContextFeatureAssembler.validate_shape(context, batch_size)


if __name__ == "__main__":
    # Self-test
    print(ContextFeatureAssembler.describe())
    print()

    # Test default array
    default = ContextFeatureAssembler.get_default_array(batch_size=4)
    print(f"Default array shape: {default.shape}")
    print(f"Default values: {default[0]}")

    # Test validation
    try:
        wrong_shape = np.random.randn(4, 7).astype(np.float32)
        ContextFeatureAssembler.validate_shape(wrong_shape)
    except ValueError as e:
        print(f"\nValidation caught error: {e}")

    print("\nAll tests passed!")
