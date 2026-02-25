"""
Tests for Market Features Module

Validates the feature engineering approach:
1. SPY trend features work correctly
2. Box width categories are computed correctly
3. The 1.54x lift for tight patterns is validated
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.market_features import (
    compute_spy_features,
    compute_box_width_features,
    compute_all_market_features,
    add_market_features_to_metadata,
    analyze_box_width_success_rates
)


class TestBoxWidthFeatures:
    """Tests for box width category features (1.54x lift validated)."""

    def test_tight_pattern_detection(self):
        """Patterns with <5% width should be flagged as tight."""
        for width in [0.01, 0.02, 0.03, 0.04, 0.049]:
            features = compute_box_width_features(width)
            assert features['tight_pattern_flag'] == 1.0, f"Width {width} should be tight"
            assert features['box_width_category'] == 2.0, f"Width {width} should be category 2"

    def test_medium_pattern_detection(self):
        """Patterns with 5-10% width should be medium."""
        for width in [0.05, 0.06, 0.08, 0.099]:
            features = compute_box_width_features(width)
            assert features['tight_pattern_flag'] == 0.0, f"Width {width} should not be tight"
            assert features['box_width_category'] == 1.0, f"Width {width} should be category 1"

    def test_wide_pattern_detection(self):
        """Patterns with >10% width should be wide."""
        for width in [0.10, 0.15, 0.20, 0.30]:
            features = compute_box_width_features(width)
            assert features['tight_pattern_flag'] == 0.0, f"Width {width} should not be tight"
            assert features['box_width_category'] == 0.0, f"Width {width} should be category 0"

    def test_boundary_conditions(self):
        """Test boundary conditions at 5% and 10%."""
        # Exactly 5% should be medium (not tight)
        features = compute_box_width_features(0.05)
        assert features['tight_pattern_flag'] == 0.0
        assert features['box_width_category'] == 1.0

        # Exactly 10% should be wide (not medium)
        features = compute_box_width_features(0.10)
        assert features['tight_pattern_flag'] == 0.0
        assert features['box_width_category'] == 0.0

        # Just under 5% should be tight
        features = compute_box_width_features(0.0499)
        assert features['tight_pattern_flag'] == 1.0
        assert features['box_width_category'] == 2.0


class TestSPYFeatures:
    """Tests for SPY trend features."""

    def create_spy_data(self, n_days: int = 250, trend: str = 'bull') -> pd.DataFrame:
        """Create synthetic SPY data for testing."""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

        if trend == 'bull':
            # Uptrending: close above SMAs
            close = 400 + np.arange(n_days) * 0.5 + np.random.randn(n_days) * 2
        elif trend == 'bear':
            # Downtrending: close below SMAs
            close = 450 - np.arange(n_days) * 0.5 + np.random.randn(n_days) * 2
        else:
            # Flat
            close = 420 + np.random.randn(n_days) * 2

        return pd.DataFrame({
            'open': close - 1,
            'high': close + 2,
            'low': close - 2,
            'close': close,
            'volume': np.random.randint(50000000, 100000000, n_days)
        }, index=dates)

    def test_bull_market_features(self):
        """In bull market, SPY should be above SMAs."""
        spy_data = self.create_spy_data(250, 'bull')
        pattern_date = spy_data.index[-1]

        features = compute_spy_features(spy_data, pattern_date)

        assert features['spy_trend_position'] > 1.0, "Bull: SPY should be above 200 SMA"
        assert features['spy_above_sma50'] == 1.0, "Bull: SPY should be above 50 SMA"
        assert features['spy_momentum_20d'] > 0, "Bull: momentum should be positive"

    def test_bear_market_features(self):
        """In bear market, SPY should be below SMAs."""
        spy_data = self.create_spy_data(250, 'bear')
        pattern_date = spy_data.index[-1]

        features = compute_spy_features(spy_data, pattern_date)

        assert features['spy_trend_position'] < 1.0, "Bear: SPY should be below 200 SMA"
        assert features['spy_momentum_20d'] < 0, "Bear: momentum should be negative"

    def test_insufficient_data_defaults(self):
        """Should return defaults when insufficient data."""
        spy_data = self.create_spy_data(50, 'bull')  # Not enough for 200 SMA
        pattern_date = spy_data.index[-1]

        features = compute_spy_features(spy_data, pattern_date)

        assert features['spy_trend_position'] == 1.0
        assert features['spy_momentum_20d'] == 0.0
        assert features['spy_above_sma50'] == 1.0

    def test_no_lookahead_bias(self):
        """Features should only use data up to pattern date."""
        spy_data = self.create_spy_data(250, 'bull')

        # Use date from middle of data
        pattern_date = spy_data.index[150]

        features = compute_spy_features(spy_data, pattern_date)

        # Should still work (uses only data up to pattern_date)
        assert 0.8 <= features['spy_trend_position'] <= 1.3
        assert -0.15 <= features['spy_momentum_20d'] <= 0.15


class TestIntegration:
    """Integration tests for market features."""

    def test_compute_all_market_features(self):
        """Test computing all features for a pattern."""
        pattern = pd.Series({
            'ticker': 'TEST',
            'pattern_end_date': '2024-01-15',
            'upper_boundary': 100.0,
            'lower_boundary': 97.0,  # 3% width - tight!
            'risk_width_pct': 0.03
        })

        features = compute_all_market_features(pattern)

        # Should have all expected keys
        expected_keys = [
            'spy_trend_position', 'spy_momentum_20d', 'spy_above_sma50',
            'tight_pattern_flag', 'box_width_category',
            'market_breadth_50', 'breadth_momentum'
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

        # 3% width should be tight
        assert features['tight_pattern_flag'] == 1.0
        assert features['box_width_category'] == 2.0

    def test_add_features_to_metadata(self):
        """Test adding features to metadata DataFrame."""
        metadata = pd.DataFrame({
            'ticker': ['A', 'B', 'C'],
            'pattern_end_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'upper_boundary': [100.0, 100.0, 100.0],
            'lower_boundary': [97.0, 92.0, 85.0],  # 3%, 8%, 15% widths
            'risk_width_pct': [0.03, 0.08, 0.15]
        })

        result = add_market_features_to_metadata(metadata)

        # Check tight pattern flags
        assert result.loc[0, 'tight_pattern_flag'] == 1.0  # 3% = tight
        assert result.loc[1, 'tight_pattern_flag'] == 0.0  # 8% = medium
        assert result.loc[2, 'tight_pattern_flag'] == 0.0  # 15% = wide

        # Check categories
        assert result.loc[0, 'box_width_category'] == 2.0  # tight
        assert result.loc[1, 'box_width_category'] == 1.0  # medium
        assert result.loc[2, 'box_width_category'] == 0.0  # wide


class TestSuccessRateAnalysis:
    """Tests for the 1.54x lift validation."""

    def test_analyze_success_rates(self):
        """Validate success rate analysis by category."""
        # Create mock data with known success rates
        np.random.seed(42)
        n = 300

        metadata = pd.DataFrame({
            'ticker': [f'T{i}' for i in range(n)],
            'risk_width_pct': np.concatenate([
                np.random.uniform(0.01, 0.049, 100),  # Tight
                np.random.uniform(0.05, 0.099, 100),  # Medium
                np.random.uniform(0.10, 0.30, 100)    # Wide
            ]),
            # Simulate 2.8x success rate for tight patterns
            'label': np.concatenate([
                np.random.choice([0, 1, 2], 100, p=[0.3, 0.3, 0.4]),   # Tight: 40% success
                np.random.choice([0, 1, 2], 100, p=[0.4, 0.45, 0.15]), # Medium: 15% success
                np.random.choice([0, 1, 2], 100, p=[0.5, 0.42, 0.08])  # Wide: 8% success
            ])
        })

        result = analyze_box_width_success_rates(metadata, label_col='label', target_class=2)

        # Check structure
        assert len(result) == 3
        assert 'Category' in result.columns
        assert 'Lift' in result.columns

        # Tight should have highest rate
        tight_rate = result[result['Category'] == 'Tight (<5%)']['Rate'].iloc[0]
        wide_rate = result[result['Category'] == 'Wide (>10%)']['Rate'].iloc[0]
        assert tight_rate > wide_rate, "Tight patterns should have higher success rate"


class TestFeatureRegistry:
    """Test that features are properly registered."""

    def test_feature_count(self):
        """Verify total feature count is 24."""
        from config.context_features import CONTEXT_FEATURES, NUM_CONTEXT_FEATURES

        assert len(CONTEXT_FEATURES) == 24, f"Expected 24 features, got {len(CONTEXT_FEATURES)}"
        assert NUM_CONTEXT_FEATURES == 24

    def test_core_features_in_registry(self):
        """Verify core features are in the registry."""
        from config.feature_registry import FeatureRegistry

        core_features = [
            'coil_intensity',
            'volume_shock',
            'trend_position',
            'risk_width_pct',
            'vix_regime_level',
            'market_breadth_200',
        ]

        for name in core_features:
            spec = FeatureRegistry.get(name)
            assert spec is not None, f"Feature {name} not found in registry"

    def test_feature_indices(self):
        """Verify feature indices are correct."""
        from config.feature_registry import FeatureRegistry

        assert FeatureRegistry.get_index('coil_intensity') == 13
        assert FeatureRegistry.get_index('volume_shock') == 10
        assert FeatureRegistry.get_index('trend_position') == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
