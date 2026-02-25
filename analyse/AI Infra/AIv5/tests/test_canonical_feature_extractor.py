"""
Test Suite for Canonical Feature Extractor

Ensures the canonical feature extractor produces exactly 69 features
and maintains consistency across all usage scenarios.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_detection.features.canonical_feature_extractor import CanonicalFeatureExtractor
from pattern_detection.models.pattern import ConsolidationPattern, PatternPhase


def create_sample_data(days=100):
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    np.random.seed(42)

    # Generate realistic price data
    close_prices = 100 + np.cumsum(np.random.randn(days) * 0.5)

    df = pd.DataFrame({
        'open': close_prices + np.random.randn(days) * 0.2,
        'high': close_prices + np.abs(np.random.randn(days) * 0.5),
        'low': close_prices - np.abs(np.random.randn(days) * 0.5),
        'close': close_prices,
        'volume': 1000000 + np.random.randn(days) * 100000
    }, index=dates)

    # Add required indicator columns (normally calculated by calculate_indicators)
    df['bbw_20'] = np.random.uniform(0.01, 0.1, days)
    df['bbw_percentile'] = np.random.uniform(0, 100, days)
    df['adx'] = np.random.uniform(15, 35, days)
    df['volume_ratio_20'] = np.random.uniform(0.5, 1.5, days)
    df['range_ratio'] = np.random.uniform(0.5, 1.5, days)
    df['daily_range'] = df['high'] - df['low']

    # Add enhanced pre-calculated features
    df['avg_range_20d'] = df['daily_range'].rolling(20, min_periods=1).mean()
    df['bbw_std_20d'] = df['bbw_20'].rolling(20, min_periods=1).std()
    df['price_volatility_20d'] = df['close'].rolling(20, min_periods=1).std()

    # Baseline features
    df['baseline_bbw_avg'] = df['bbw_20'].rolling(30, min_periods=1).mean().shift(20)
    df['baseline_volume_avg'] = df['volume_ratio_20'].rolling(30, min_periods=1).mean().shift(20)
    df['baseline_volatility'] = df['close'].rolling(30, min_periods=1).std().shift(20)

    # Compression ratios
    df['bbw_compression_ratio'] = df['bbw_20'] / (df['baseline_bbw_avg'] + 1e-10)
    df['volume_compression_ratio'] = df['volume_ratio_20'] / (df['baseline_volume_avg'] + 1e-10)
    df['volatility_compression_ratio'] = df['price_volatility_20d'] / (df['baseline_volatility'] + 1e-10)

    # Slopes (simplified)
    df['bbw_slope_20d'] = np.random.uniform(-0.001, 0.001, days)
    df['adx_slope_20d'] = np.random.uniform(-0.5, 0.5, days)

    # Quality score
    df['consolidation_quality_score'] = np.random.uniform(0.3, 0.8, days)

    return df


def create_sample_pattern():
    """Create a sample ConsolidationPattern for testing."""
    pattern = ConsolidationPattern(
        ticker='TEST',
        start_date=datetime.now() - timedelta(days=30),
        start_idx=0,
        start_price=100.0,
        phase=PatternPhase.ACTIVE,
        activation_date=datetime.now() - timedelta(days=20),
        activation_idx=10
    )
    # Add boundaries as attributes (these are typically added during activation)
    pattern.upper_boundary = 105.0
    pattern.lower_boundary = 95.0
    pattern.power_boundary = 105.5
    return pattern


class TestCanonicalFeatureExtractor:
    """Test suite for the canonical feature extractor."""

    def test_initialization(self):
        """Test that extractor initializes correctly."""
        extractor = CanonicalFeatureExtractor()
        assert extractor.feature_count == 69
        assert len(extractor.get_feature_names()) == 69

    def test_feature_count(self):
        """Test that extractor produces exactly 69 features."""
        df = create_sample_data(100)
        pattern = create_sample_pattern()
        extractor = CanonicalFeatureExtractor()

        snapshot_date = df.index[-1]
        features = extractor.extract_snapshot_features(df, snapshot_date, pattern)

        assert len(features) == 69, f"Expected 69 features, got {len(features)}"

    def test_feature_names_match(self):
        """Test that extracted features match expected names."""
        df = create_sample_data(100)
        pattern = create_sample_pattern()
        extractor = CanonicalFeatureExtractor()

        snapshot_date = df.index[-1]
        features = extractor.extract_snapshot_features(df, snapshot_date, pattern)

        expected_names = set(extractor.get_feature_names())
        actual_names = set(features.keys())

        assert expected_names == actual_names, f"Feature names mismatch"

    def test_temporal_safety(self):
        """Test that no future data is used in feature extraction."""
        df = create_sample_data(100)
        pattern = create_sample_pattern()
        extractor = CanonicalFeatureExtractor()

        # Extract features at day 50
        snapshot_date = df.index[50]
        snapshot_idx = 50

        # Modify future data
        df.loc[df.index[51:], 'close'] = 999999  # Future prices should not affect features

        features = extractor.extract_snapshot_features(
            df, snapshot_date, pattern, snapshot_idx
        )

        # Features should not contain the modified future values
        assert features['current_price'] != 999999
        assert features['ma_20'] < 200  # Should be based on historical data only

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        df = create_sample_data(100)
        pattern = create_sample_pattern()
        extractor = CanonicalFeatureExtractor()

        # Introduce some NaN values
        df.loc[df.index[-10:], 'bbw_20'] = np.nan

        snapshot_date = df.index[-1]
        features = extractor.extract_snapshot_features(df, snapshot_date, pattern)

        # Check that NaN is replaced with 0
        assert features['current_bbw_20'] == 0.0
        assert not any(pd.isna(v) for v in features.values())

    def test_feature_validation(self):
        """Test feature validation method."""
        extractor = CanonicalFeatureExtractor()

        # Valid features
        valid_features = {name: 0.0 for name in extractor.get_feature_names()}
        assert extractor.validate_features(valid_features) == True

        # Missing features
        invalid_features = {name: 0.0 for name in extractor.get_feature_names()[:-5]}
        with pytest.raises(ValueError, match="Missing required features"):
            extractor.validate_features(invalid_features)

        # Extra features
        extra_features = valid_features.copy()
        extra_features['extra_feature'] = 1.0
        with pytest.raises(ValueError, match="Unexpected extra features"):
            extractor.validate_features(extra_features)

    def test_consistency_across_calls(self):
        """Test that multiple calls produce consistent results."""
        df = create_sample_data(100)
        pattern = create_sample_pattern()
        extractor = CanonicalFeatureExtractor()

        snapshot_date = df.index[-1]

        # Extract features twice
        features1 = extractor.extract_snapshot_features(df, snapshot_date, pattern)
        features2 = extractor.extract_snapshot_features(df, snapshot_date, pattern)

        # Should be identical
        for key in features1:
            if isinstance(features1[key], (int, float)):
                assert abs(features1[key] - features2[key]) < 1e-10, \
                    f"Inconsistent value for feature {key}"
            else:
                assert features1[key] == features2[key], \
                    f"Inconsistent value for feature {key}"

    def test_all_feature_groups_present(self):
        """Test that all feature groups are represented."""
        df = create_sample_data(100)
        pattern = create_sample_pattern()
        extractor = CanonicalFeatureExtractor()

        snapshot_date = df.index[-1]
        features = extractor.extract_snapshot_features(df, snapshot_date, pattern)

        # Check for presence of each feature group
        core_features = ['days_since_activation', 'current_price', 'current_bbw_20']
        ebp_features = ['cci_score', 'var_score', 'nes_score', 'lpf_score', 'ebp_composite']
        derived_features = ['ma_20', 'price_momentum_20d', 'volume_spike']

        for feat in core_features:
            assert feat in features, f"Core feature {feat} missing"

        for feat in ebp_features:
            assert feat in features, f"EBP feature {feat} missing"

        for feat in derived_features:
            assert feat in features, f"Derived feature {feat} missing"

    def test_ebp_features_calculated(self):
        """Test that EBP features are properly calculated."""
        df = create_sample_data(100)
        pattern = create_sample_pattern()
        extractor = CanonicalFeatureExtractor()

        snapshot_date = df.index[-1]
        features = extractor.extract_snapshot_features(df, snapshot_date, pattern)

        # Check that EBP features exist and have reasonable values
        ebp_features = [
            'cci_bbw_compression', 'cci_atr_compression', 'cci_days_factor', 'cci_score',
            'var_raw', 'var_score',
            'nes_inactive_mass', 'nes_wavelet_energy', 'nes_rsa_proxy', 'nes_score',
            'lpf_bid_pressure', 'lpf_volume_pressure', 'lpf_fta_proxy', 'lpf_score',
            'tsf_days_in_consolidation', 'tsf_score',
            'ebp_raw', 'ebp_composite', 'ebp_signal'
        ]

        for feat in ebp_features:
            assert feat in features, f"EBP feature {feat} missing"
            assert isinstance(features[feat], (int, float)), f"EBP feature {feat} has wrong type"


def main():
    """Run tests manually."""
    print("Testing Canonical Feature Extractor...")
    print("=" * 70)

    test = TestCanonicalFeatureExtractor()

    try:
        test.test_initialization()
        print("[OK] Initialization test passed")

        test.test_feature_count()
        print("[OK] Feature count test passed (69 features)")

        test.test_feature_names_match()
        print("[OK] Feature names match test passed")

        test.test_temporal_safety()
        print("[OK] Temporal safety test passed")

        test.test_nan_handling()
        print("[OK] NaN handling test passed")

        test.test_feature_validation()
        print("[OK] Feature validation test passed")

        test.test_consistency_across_calls()
        print("[OK] Consistency test passed")

        test.test_all_feature_groups_present()
        print("[OK] All feature groups present")

        test.test_ebp_features_calculated()
        print("[OK] EBP features calculated correctly")

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("Canonical Feature Extractor is working correctly with 69 features")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)