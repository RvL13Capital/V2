"""
Temporal Feature Configuration for TRANS Architecture

Defines the 10-dimensional feature structure for temporal sequences:
- 5 market data features
- 3 technical indicators
- 2 structural boundaries

DISABLED (2026-01-18): Composite scores (vol_dryup_ratio, var_score, nes_score, lpf_score)
were removed to simplify the model and reduce noise from derived features.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TemporalFeatureConfig:
    """Configuration for 10-dimensional temporal features

    This configuration defines the exact features used in the temporal model.
    The order of features is critical as it determines the input tensor structure.

    Feature layout (10 total):
    - [0-4]: Market data (open, high, low, close, volume)
    - [5-7]: Technical indicators (bbw_20, adx, volume_ratio_20)
    - [8-9]: Boundary slopes (upper_boundary, lower_boundary)

    DISABLED: Composite scores (indices 8-11 in old layout) were removed.
    """

    # Market data features (5)
    market_features: List[str] = field(default_factory=lambda: [
        'open',
        'high',
        'low',
        'close',
        'volume'
    ])

    # Technical indicators (3)
    technical_features: List[str] = field(default_factory=lambda: [
        'bbw_20',           # Bollinger Band Width (20-period)
        'adx',              # Average Directional Index
        'volume_ratio_20'   # Volume relative to 20-day average
    ])

    # DISABLED (2026-01-18): Composite scores removed
    # These features added noise without improving Top-15% Precision
    # Old indices [8-11]: vol_dryup_ratio, var_score, nes_score, lpf_score
    composite_features: List[str] = field(default_factory=lambda: [
        # DISABLED - empty list
    ])

    # Convergence geometry features (2)
    # GEOMETRY FIX (2026-01-18): Changed from static box coordinates to dynamic slopes
    # Old: constant upper/lower prices → variance = 0 → model sees Rectangle
    # New: rolling regression slopes → variance > 0 → model sees Triangle
    #
    # Column names preserved for backwards compatibility, but semantics changed:
    # - upper_boundary: Now contains upper_slope (regression slope of rolling highs)
    # - lower_boundary: Now contains lower_slope (regression slope of rolling lows)
    #
    # Triangle Geometry Interpretation:
    # - Symmetric triangle: upper_slope < 0, lower_slope > 0 (converging)
    # - Ascending triangle: upper_slope ≈ 0, lower_slope > 0
    # - Descending triangle: upper_slope < 0, lower_slope ≈ 0
    # - Rectangle/box: both ≈ 0 (horizontal boundaries)
    boundary_features: List[str] = field(default_factory=lambda: [
        'upper_boundary',  # Rolling slope of highs, normalized by box_height
        'lower_boundary'   # Rolling slope of lows, normalized by box_height
    ])

    @property
    def all_features(self) -> List[str]:
        """Get complete list of all 10 features in order"""
        return (self.market_features +
                self.technical_features +
                # composite_features DISABLED
                self.boundary_features)

    @property
    def feature_count(self) -> int:
        """Total number of features (should always be 10)"""
        total = len(self.all_features)
        if total != 10:
            raise ValueError(f"Feature count must be 10, got {total}")
        return 10

    @property
    def market_feature_indices(self) -> List[int]:
        """Get indices of market features in the feature vector"""
        return list(range(0, 5))  # [0, 1, 2, 3, 4]

    @property
    def technical_feature_indices(self) -> List[int]:
        """Get indices of technical features in the feature vector"""
        return list(range(5, 8))  # [5, 6, 7]

    @property
    def composite_feature_indices(self) -> List[int]:
        """Get indices of composite features in the feature vector

        DISABLED (2026-01-18): Returns empty list - composite features removed
        """
        return []  # DISABLED

    @property
    def boundary_feature_indices(self) -> List[int]:
        """Get indices of boundary features in the feature vector

        Note: Shifted from [12, 13] to [8, 9] after composite removal
        """
        return list(range(8, 10))  # [8, 9] - was [12, 13]

    # ==========================================
    # Split Attention Feature Groups (2026-01-18)
    # ==========================================
    # Domain-aware grouping for illiquid market microstructure:
    # - Price group: Noisy due to wide spreads, but captures geometry
    # - Volume group: Clean signals of supply exhaustion / accumulation
    #
    # Prevents price noise from drowning out volume signals in attention.
    #
    # UPDATED (2026-01-18): Composite features DISABLED, indices shifted

    @property
    def price_group_indices(self) -> List[int]:
        """Price/Geometry features - handles noisy spreads

        Includes:
        - OHLC (0-3): Price action with spread noise
        - BBW (5): Price volatility width
        - ADX (6): Price trend strength
        - Boundaries (8-9): Price convergence slopes (shifted from 12-13)
        """
        return [0, 1, 2, 3, 5, 6, 8, 9]  # 8 features

    @property
    def volume_group_indices(self) -> List[int]:
        """Volume/Liquidity features - handles clean dry-up signals

        Includes:
        - Volume (4): Raw volume (zero = supply exhaustion signal)
        - Volume Ratio (7): Relative to 20-day average

        DISABLED: Composite scores (vol_dryup, var, nes, lpf) removed
        """
        return [4, 7]  # 2 features (was 6 before composite removal)

    @property
    def price_group_dim(self) -> int:
        """Number of features in price group"""
        return len(self.price_group_indices)  # 8

    @property
    def volume_group_dim(self) -> int:
        """Number of features in volume group"""
        return len(self.volume_group_indices)  # 2 (was 6)

    def validate_dataframe_columns(self, df) -> List[str]:
        """Validate that a dataframe has all required feature columns

        Args:
            df: DataFrame to validate

        Returns:
            List of missing columns (empty if all present)
        """
        missing = []
        for feature in self.all_features:
            if feature not in df.columns:
                missing.append(feature)
        return missing


@dataclass
class SequenceGenerationConfig:
    """Configuration for temporal sequence generation

    Defines how sliding windows are extracted from patterns to create
    temporal sequences for model input.
    """

    # Core sequence parameters
    window_size: int = 20               # Total timesteps per sequence
    qualification_phase: int = 10       # Timesteps 1-10: pattern emergence
    validation_phase: int = 10          # Timesteps 11-20: stability verification
    min_pattern_duration: int = 20     # Minimum duration for first valid window

    # Sliding window parameters
    stride: int = 1                    # Step size for sliding window

    # Temporal integrity parameters
    max_temporal_gap_days: int = 10    # Maximum allowed gap (weekends + holidays OK)
    require_chronological: bool = True  # Enforce chronological ordering

    @property
    def activation_timestep(self) -> int:
        """Timestep where pattern activates (end of qualification)"""
        return self.qualification_phase

    @property
    def min_sequences_per_pattern(self) -> int:
        """Minimum number of sequences from a valid pattern"""
        return 1  # At least one sequence from minimum duration pattern

    def calculate_num_sequences(self, pattern_duration: int) -> int:
        """Calculate number of sequences from a pattern of given duration

        Args:
            pattern_duration: Total timesteps in pattern

        Returns:
            Number of sliding window sequences that can be extracted
        """
        if pattern_duration < self.min_pattern_duration:
            return 0
        return (pattern_duration - self.window_size) // self.stride + 1

    def validate_temporal_window(self, window_size: int) -> bool:
        """Validate that a window size is sufficient for sequence generation

        Args:
            window_size: Size of data window

        Returns:
            True if window is large enough for at least one sequence
        """
        return window_size >= self.min_pattern_duration


@dataclass
class ModelArchitectureConfig:
    """Configuration for the temporal model architecture"""

    # Input dimensions
    # UPDATED (2026-01-18): 14 → 10 after composite feature removal
    input_features: int = 10
    sequence_length: int = 20

    # LSTM branch parameters
    lstm_hidden_size: int = 32
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2

    # CNN branch parameters
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])

    # Attention parameters
    attention_heads: int = 8
    attention_dropout: float = 0.1

    # Fusion layer parameters
    fusion_hidden_size: int = 64
    fusion_dropout: float = 0.3

    # Output parameters
    num_classes: int = 3  # Danger, Noise, Target

    @property
    def total_conv_channels(self) -> int:
        """Total channels from all CNN branches"""
        return sum(self.cnn_channels)

    @property
    def fusion_input_size(self) -> int:
        """Size of concatenated features entering fusion layer"""
        return self.lstm_hidden_size + self.total_conv_channels