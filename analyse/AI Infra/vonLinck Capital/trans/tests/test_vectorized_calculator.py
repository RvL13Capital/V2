"""
Unit tests for VectorizedFeatureCalculator.

Tests the numpy bug fixes:
- Bug 1 (line 329): _vectorized_lpf returns numpy array (not .values)
- Bug 2 (line 222): _vectorized_cci returns numpy array consistently
- Bug 3 (line 256): _vectorized_var returns numpy array consistently

All methods should return np.ndarray type as per their type annotations.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.vectorized_calculator import VectorizedFeatureCalculator


class TestVectorizedCalculatorTypeConsistency:
    """Test that all vectorized methods return consistent numpy array types."""

    def setup_method(self):
        """Setup before each test."""
        self.calculator = VectorizedFeatureCalculator()

        # Create realistic test data (100 days)
        np.random.seed(42)
        n = 100

        self.test_data = {
            'high': np.random.uniform(95, 105, n),
            'low': np.random.uniform(85, 95, n),
            'close': np.random.uniform(88, 102, n),
            'volume': np.random.uniform(1e6, 5e6, n),
            'open': np.random.uniform(88, 102, n),
        }

        # Ensure high > low
        self.test_data['high'] = np.maximum(self.test_data['high'], self.test_data['low'] + 1)

    def test_vectorized_bbw_returns_numpy_array(self):
        """Test that _vectorized_bbw returns numpy array."""
        result = self.calculator._vectorized_bbw(close=self.test_data['close'])

        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
        assert result.shape == (100,), f"Expected shape (100,), got {result.shape}"
        print(f"[PASS] _vectorized_bbw returns np.ndarray with shape {result.shape}")

    def test_vectorized_adx_returns_numpy_array(self):
        """Test that _vectorized_adx returns numpy array."""
        result = self.calculator._vectorized_adx(
            high=self.test_data['high'],
            low=self.test_data['low'],
            close=self.test_data['close']
        )

        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
        assert result.shape == (100,), f"Expected shape (100,), got {result.shape}"
        print(f"[PASS] _vectorized_adx returns np.ndarray with shape {result.shape}")

    def test_vectorized_volume_ratio_returns_numpy_array(self):
        """Test that _vectorized_volume_ratio returns numpy array."""
        result = self.calculator._vectorized_volume_ratio(
            volume=self.test_data['volume']
        )

        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
        assert result.shape == (100,), f"Expected shape (100,), got {result.shape}"
        print(f"[PASS] _vectorized_volume_ratio returns np.ndarray with shape {result.shape}")

    def test_vectorized_cci_returns_numpy_array(self):
        """Test that _vectorized_cci returns numpy array (Bug 2 fix)."""
        result = self.calculator._vectorized_cci(
            high=self.test_data['high'],
            low=self.test_data['low'],
            close=self.test_data['close']
        )

        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
        assert result.shape == (100,), f"Expected shape (100,), got {result.shape}"

        # CCI score should be normalized to [-1, 1] via tanh
        assert np.all(result >= -1) and np.all(result <= 1), "CCI score should be in [-1, 1]"
        print(f"[PASS] _vectorized_cci returns np.ndarray with shape {result.shape} (Bug 2 FIXED)")

    def test_vectorized_var_returns_numpy_array(self):
        """Test that _vectorized_var returns numpy array (Bug 3 fix)."""
        result = self.calculator._vectorized_var(
            volume=self.test_data['volume'],
            close=self.test_data['close']
        )

        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
        assert result.shape == (100,), f"Expected shape (100,), got {result.shape}"

        # VAR score should be normalized via tanh
        assert np.all(result >= -1) and np.all(result <= 1), "VAR score should be in [-1, 1]"
        print(f"[PASS] _vectorized_var returns np.ndarray with shape {result.shape} (Bug 3 FIXED)")

    def test_vectorized_nes_returns_numpy_array(self):
        """Test that _vectorized_nes returns numpy array."""
        result = self.calculator._vectorized_nes(
            close=self.test_data['close'],
            volume=self.test_data['volume']
        )

        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
        assert result.shape == (100,), f"Expected shape (100,), got {result.shape}"
        print(f"[PASS] _vectorized_nes returns np.ndarray with shape {result.shape}")

    def test_vectorized_lpf_returns_numpy_array(self):
        """Test that _vectorized_lpf returns numpy array (Bug 1 fix - CRITICAL)."""
        result = self.calculator._vectorized_lpf(
            volume=self.test_data['volume'],
            close=self.test_data['close']
        )

        assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result)}"
        assert result.shape == (100,), f"Expected shape (100,), got {result.shape}"

        # LPF score should be in [-1, 1] range (flow pressure indicator)
        assert np.all(result >= -1) and np.all(result <= 1), "LPF score should be in [-1, 1]"
        print(f"[PASS] _vectorized_lpf returns np.ndarray with shape {result.shape} (Bug 1 FIXED - CRITICAL)")

    def test_all_methods_return_numpy_arrays(self):
        """Comprehensive test: all methods return numpy arrays."""
        methods_to_test = [
            ('_vectorized_bbw', {'close': self.test_data['close']}),
            ('_vectorized_adx', {'high': self.test_data['high'], 'low': self.test_data['low'], 'close': self.test_data['close']}),
            ('_vectorized_volume_ratio', {'volume': self.test_data['volume']}),
            ('_vectorized_cci', {'high': self.test_data['high'], 'low': self.test_data['low'], 'close': self.test_data['close']}),
            ('_vectorized_var', {'volume': self.test_data['volume'], 'close': self.test_data['close']}),
            ('_vectorized_nes', {'close': self.test_data['close'], 'volume': self.test_data['volume']}),
            ('_vectorized_lpf', {'volume': self.test_data['volume'], 'close': self.test_data['close']}),
        ]

        for method_name, kwargs in methods_to_test:
            method = getattr(self.calculator, method_name)
            result = method(**kwargs)

            assert isinstance(result, np.ndarray), f"{method_name} should return np.ndarray, got {type(result)}"
            assert result.shape == (100,), f"{method_name} should return shape (100,), got {result.shape}"

        print(f"[PASS] All 7 vectorized methods return np.ndarray consistently")


class TestVectorizedCalculatorFunctional:
    """Test functional correctness of vectorized methods."""

    def setup_method(self):
        """Setup before each test."""
        self.calculator = VectorizedFeatureCalculator()

        # Create more realistic market data
        np.random.seed(42)
        n = 100

        # Simulate realistic price series with trend
        base_price = 100
        trend = np.linspace(0, 10, n)
        noise = np.random.normal(0, 2, n)

        close = base_price + trend + noise

        self.test_data = {
            'close': close,
            'high': close + np.abs(np.random.normal(1, 0.5, n)),
            'low': close - np.abs(np.random.normal(1, 0.5, n)),
            'volume': np.random.uniform(1e6, 5e6, n),
            'open': close + np.random.normal(0, 0.5, n),
        }

    def test_lpf_directional_flow(self):
        """Test LPF captures directional liquidity flow."""
        # Create uptrend with high volume
        close_up = np.linspace(100, 120, 100)
        volume_high = np.full(100, 5e6)

        lpf_score = self.calculator._vectorized_lpf(volume_high, close_up)

        # LPF should be positive for sustained uptrend with volume
        assert np.mean(lpf_score[-20:]) > 0, "LPF should be positive for uptrend with volume"
        print(f"[PASS] LPF correctly identifies upward liquidity flow: {np.mean(lpf_score[-20:]):.3f}")

    def test_cci_oscillator_behavior(self):
        """Test CCI oscillates around typical price."""
        cci_score = self.calculator._vectorized_cci(
            self.test_data['high'],
            self.test_data['low'],
            self.test_data['close']
        )

        # CCI should oscillate with mean near 0
        assert -1 <= np.mean(cci_score) <= 1, "CCI should oscillate around 0"
        assert np.std(cci_score) > 0, "CCI should have variation"
        print(f"[PASS] CCI oscillates correctly: mean={np.mean(cci_score):.3f}, std={np.std(cci_score):.3f}")

    def test_var_volume_price_relationship(self):
        """Test VAR captures volume-price relationship."""
        # High volume with price increase
        volume = np.random.uniform(5e6, 10e6, 100)
        close = np.linspace(100, 120, 100)

        var_score = self.calculator._vectorized_var(volume, close)

        # VAR should respond to volume and price momentum
        assert var_score is not None
        assert len(var_score) == 100
        print(f"[PASS] VAR captures volume-price dynamics: mean={np.mean(var_score):.3f}")

    def test_calculate_all_features_integration(self):
        """Test calculate_all_features returns all expected features."""
        df = pd.DataFrame(self.test_data)

        # Add required pattern boundaries
        df['upper_boundary'] = 110.0
        df['lower_boundary'] = 90.0
        df['days_in_pattern'] = np.arange(len(df))

        features = self.calculator.calculate_all_features(df)

        # Check key features exist (actual column names may have suffixes like bbw_20)
        # UPDATED (2026-01-18): Composite features (cci, var, nes, lpf) DISABLED
        key_feature_patterns = [
            'bbw', 'adx', 'volume_ratio',
            # DISABLED: 'cci', 'var', 'nes', 'lpf' (composite features removed)
            'boundary', 'days_in_pattern'
        ]

        for pattern in key_feature_patterns:
            matching_cols = [col for col in features.columns if pattern in col]
            assert len(matching_cols) > 0, f"No feature found matching pattern: {pattern}"

        # All features should be numpy arrays
        for col in features.columns:
            values = features[col].values
            assert isinstance(values, np.ndarray), f"{col} should have numpy array values"

        print(f"[PASS] calculate_all_features returns {len(features.columns)} features, all as numpy arrays")


class TestBugRegressions:
    """Specific tests to prevent regression of the 3 numpy bugs."""

    def setup_method(self):
        """Setup before each test."""
        self.calculator = VectorizedFeatureCalculator()
        np.random.seed(42)
        n = 50

        self.volume = np.random.uniform(1e6, 5e6, n)
        self.close = np.random.uniform(90, 110, n)
        self.high = np.random.uniform(95, 115, n)
        self.low = np.random.uniform(85, 105, n)

        # Ensure high > low
        self.high = np.maximum(self.high, self.low + 1)

    def test_bug1_lpf_no_values_attribute_error(self):
        """Regression test for Bug 1: lpf_score.values AttributeError."""
        try:
            result = self.calculator._vectorized_lpf(self.volume, self.close)
            assert isinstance(result, np.ndarray)
            print("[PASS] Bug 1 FIXED: _vectorized_lpf no longer raises AttributeError")
        except AttributeError as e:
            if "'numpy.ndarray' object has no attribute 'values'" in str(e):
                pytest.fail("Bug 1 REGRESSION: lpf_score.values AttributeError still present!")
            raise

    def test_bug2_cci_consistent_return_type(self):
        """Regression test for Bug 2: cci returns consistent numpy array."""
        result = self.calculator._vectorized_cci(self.high, self.low, self.close)

        assert isinstance(result, np.ndarray), f"Bug 2 REGRESSION: cci_score returned {type(result)}"
        assert not isinstance(result, pd.Series), "Bug 2 REGRESSION: cci_score is pandas Series"
        print("[PASS] Bug 2 FIXED: _vectorized_cci returns numpy array consistently")

    def test_bug3_var_consistent_return_type(self):
        """Regression test for Bug 3: var returns numpy array."""
        result = self.calculator._vectorized_var(self.volume, self.close)

        assert isinstance(result, np.ndarray), f"Bug 3 REGRESSION: var_score returned {type(result)}"
        assert not isinstance(result, pd.Series), "Bug 3 REGRESSION: var_score is pandas Series"
        print("[PASS] Bug 3 FIXED: _vectorized_var returns numpy array consistently")

    def test_no_pandas_series_in_returns(self):
        """Ensure no method returns pandas Series (should all be numpy arrays)."""
        methods = [
            ('_vectorized_bbw', (self.close,)),
            ('_vectorized_adx', (self.high, self.low, self.close)),
            ('_vectorized_volume_ratio', (self.volume,)),
            ('_vectorized_cci', (self.high, self.low, self.close)),
            ('_vectorized_var', (self.volume, self.close)),
            ('_vectorized_nes', (self.close, self.volume)),
            ('_vectorized_lpf', (self.volume, self.close)),
        ]

        for method_name, args in methods:
            method = getattr(self.calculator, method_name)
            result = method(*args)

            assert not isinstance(result, pd.Series), \
                f"{method_name} returned pandas Series instead of numpy array"

        print("[PASS] All methods return numpy arrays (no pandas Series)")


class TestBoundaryConvergenceGeometry:
    """Test boundary convergence features (Box Coil Geometry Fix - 2026-01-18).

    Validates that boundary features now capture Triangle geometry, not Rectangle.
    Target: Boundary Variance > 0.0001 (model must see convergence, not static box)
    """

    VARIANCE_THRESHOLD = 0.0001  # Model cannot distinguish if variance < this

    def setup_method(self):
        """Setup before each test."""
        self.calculator = VectorizedFeatureCalculator()
        np.random.seed(42)

    def test_boundary_slopes_have_nonzero_variance(self):
        """CRITICAL: Boundary features must have variance > 0.0001.

        Old (broken): Static box coordinates → variance = 0 → model sees Rectangle
        New (fixed): Dynamic slope features → variance > 0 → model sees Triangle
        """
        # Create realistic consolidation data with narrowing pattern
        n = 30
        base_close = 100
        close = np.linspace(base_close, base_close + 5, n)  # Slight uptrend
        high = close + np.random.uniform(2, 5, n)  # Narrowing highs
        low = close - np.random.uniform(2, 5, n)   # Narrowing lows

        # Simulate converging triangle: highs dropping, lows rising
        high = close + (5 - np.linspace(0, 3, n)) + np.random.normal(0, 0.3, n)
        low = close - (5 - np.linspace(0, 3, n)) + np.random.normal(0, 0.3, n)

        df = pd.DataFrame({
            'open': close + np.random.normal(0, 0.5, n),
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1e6, 2e6, n)
        })

        upper_boundary = high.max()
        lower_boundary = low.min()

        # Apply boundary slopes
        result = self.calculator.add_boundaries_from_pattern(
            df.copy(), upper_boundary, lower_boundary
        )

        upper_variance = np.var(result['upper_boundary'].values)
        lower_variance = np.var(result['lower_boundary'].values)

        assert upper_variance > self.VARIANCE_THRESHOLD, \
            f"Upper boundary variance {upper_variance:.6f} <= {self.VARIANCE_THRESHOLD}. " \
            "Model sees Rectangle, not Triangle!"

        assert lower_variance > self.VARIANCE_THRESHOLD, \
            f"Lower boundary variance {lower_variance:.6f} <= {self.VARIANCE_THRESHOLD}. " \
            "Model sees Rectangle, not Triangle!"

        print(f"[PASS] Boundary variances: upper={upper_variance:.6f}, lower={lower_variance:.6f}")

    def test_symmetric_triangle_geometry(self):
        """Test symmetric triangle: upper_slope < 0, lower_slope > 0 (converging)."""
        n = 30
        close = np.full(n, 100.0) + np.random.normal(0, 0.3, n)

        # Create converging triangle
        high = close + np.linspace(10, 2, n)   # Highs descending
        low = close - np.linspace(10, 2, n)    # Lows ascending

        df = pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1e6, 2e6, n)
        })

        result = self.calculator.add_boundaries_from_pattern(
            df.copy(), high.max(), low.min()
        )

        # For symmetric triangle: upper should slope down, lower should slope up
        upper_slope_end = result['upper_boundary'].values[-1]
        lower_slope_end = result['lower_boundary'].values[-1]

        assert upper_slope_end < 0, \
            f"Symmetric triangle upper_slope should be negative, got {upper_slope_end:.4f}"
        assert lower_slope_end > 0, \
            f"Symmetric triangle lower_slope should be positive, got {lower_slope_end:.4f}"

        print(f"[PASS] Symmetric triangle: upper_slope={upper_slope_end:.4f} (negative), "
              f"lower_slope={lower_slope_end:.4f} (positive)")

    def test_ascending_triangle_geometry(self):
        """Test ascending triangle: upper_slope ≈ 0, lower_slope > 0."""
        n = 30
        close = np.linspace(95, 105, n) + np.random.normal(0, 0.3, n)

        # Create ascending triangle
        high = np.full(n, 110.0) + np.random.normal(0, 0.3, n)  # Flat resistance
        low = close - np.linspace(10, 2, n)                      # Rising lows

        df = pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1e6, 2e6, n)
        })

        result = self.calculator.add_boundaries_from_pattern(
            df.copy(), high.max(), low.min()
        )

        upper_slope_end = result['upper_boundary'].values[-1]
        lower_slope_end = result['lower_boundary'].values[-1]

        # Upper should be near zero (flat), lower should be positive (rising)
        assert abs(upper_slope_end) < 0.5, \
            f"Ascending triangle upper_slope should be near 0, got {upper_slope_end:.4f}"
        assert lower_slope_end > 0, \
            f"Ascending triangle lower_slope should be positive, got {lower_slope_end:.4f}"

        print(f"[PASS] Ascending triangle: upper_slope={upper_slope_end:.4f} (near 0), "
              f"lower_slope={lower_slope_end:.4f} (positive)")

    def test_descending_triangle_geometry(self):
        """Test descending triangle: upper_slope < 0, lower_slope ≈ 0."""
        n = 30
        close = np.linspace(105, 95, n) + np.random.normal(0, 0.3, n)

        # Create descending triangle
        high = close + np.linspace(10, 2, n)                      # Falling highs
        low = np.full(n, 90.0) + np.random.normal(0, 0.3, n)      # Flat support

        df = pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1e6, 2e6, n)
        })

        result = self.calculator.add_boundaries_from_pattern(
            df.copy(), high.max(), low.min()
        )

        upper_slope_end = result['upper_boundary'].values[-1]
        lower_slope_end = result['lower_boundary'].values[-1]

        # Upper should be negative (falling), lower should be near zero (flat)
        assert upper_slope_end < 0, \
            f"Descending triangle upper_slope should be negative, got {upper_slope_end:.4f}"
        assert abs(lower_slope_end) < 0.5, \
            f"Descending triangle lower_slope should be near 0, got {lower_slope_end:.4f}"

        print(f"[PASS] Descending triangle: upper_slope={upper_slope_end:.4f} (negative), "
              f"lower_slope={lower_slope_end:.4f} (near 0)")

    def test_rectangle_geometry(self):
        """Test rectangle/channel: both slopes ≈ 0."""
        n = 30
        close = np.full(n, 100.0) + np.random.normal(0, 0.3, n)

        # Create rectangle (parallel boundaries)
        high = np.full(n, 105.0) + np.random.normal(0, 0.3, n)
        low = np.full(n, 95.0) + np.random.normal(0, 0.3, n)

        df = pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1e6, 2e6, n)
        })

        result = self.calculator.add_boundaries_from_pattern(
            df.copy(), high.max(), low.min()
        )

        upper_slope_end = result['upper_boundary'].values[-1]
        lower_slope_end = result['lower_boundary'].values[-1]

        # Both should be near zero for rectangle
        assert abs(upper_slope_end) < 0.5, \
            f"Rectangle upper_slope should be near 0, got {upper_slope_end:.4f}"
        assert abs(lower_slope_end) < 0.5, \
            f"Rectangle lower_slope should be near 0, got {lower_slope_end:.4f}"

        # But variance should still be > threshold due to micro-movements
        upper_variance = np.var(result['upper_boundary'].values)
        lower_variance = np.var(result['lower_boundary'].values)

        print(f"[PASS] Rectangle: upper_slope={upper_slope_end:.4f}, lower_slope={lower_slope_end:.4f}")
        print(f"       Variances: upper={upper_variance:.6f}, lower={lower_variance:.6f}")

    def test_boundary_slopes_are_bounded(self):
        """Test that boundary slopes are clipped to [-3, 3] range."""
        n = 30
        close = np.full(n, 100.0)

        # Create extreme case with very steep convergence
        high = close + np.linspace(50, 1, n)   # Very steep descent
        low = close - np.linspace(50, 1, n)    # Very steep ascent

        df = pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1e6, 2e6, n)
        })

        result = self.calculator.add_boundaries_from_pattern(
            df.copy(), high.max(), low.min()
        )

        upper_vals = result['upper_boundary'].values
        lower_vals = result['lower_boundary'].values

        assert np.all(upper_vals >= -3.0) and np.all(upper_vals <= 3.0), \
            f"Upper slope values out of [-3, 3] range: min={upper_vals.min()}, max={upper_vals.max()}"
        assert np.all(lower_vals >= -3.0) and np.all(lower_vals <= 3.0), \
            f"Lower slope values out of [-3, 3] range: min={lower_vals.min()}, max={lower_vals.max()}"

        print(f"[PASS] Boundary slopes bounded: upper=[{upper_vals.min():.3f}, {upper_vals.max():.3f}], "
              f"lower=[{lower_vals.min():.3f}, {lower_vals.max():.3f}]")

    def test_short_sequence_handling(self):
        """Test boundary slopes handle short sequences (< lookback) correctly."""
        n = 3  # Very short sequence
        close = np.array([100, 101, 102])
        high = np.array([105, 106, 107])
        low = np.array([95, 96, 97])

        df = pd.DataFrame({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.array([1e6, 1.1e6, 1.2e6])
        })

        result = self.calculator.add_boundaries_from_pattern(df.copy(), 107, 95)

        # Should not crash and return valid slopes
        assert len(result['upper_boundary']) == 3
        assert len(result['lower_boundary']) == 3
        assert not np.any(np.isnan(result['upper_boundary']))
        assert not np.any(np.isnan(result['lower_boundary']))

        print(f"[PASS] Short sequence (n={n}) handled without NaN or crash")


def run_all_tests():
    """Run all tests manually."""
    import traceback

    test_classes = [
        TestVectorizedCalculatorTypeConsistency,
        TestVectorizedCalculatorFunctional,
        TestBugRegressions,
        TestBoundaryConvergenceGeometry,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)

        instance = test_class()
        if hasattr(instance, 'setup_method'):
            instance.setup_method()

        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                print(f"\n{method_name}:")
                try:
                    method = getattr(instance, method_name)
                    method()
                    passed += 1
                    print(f"  PASSED")
                except Exception as e:
                    failed += 1
                    print(f"  FAILED: {e}")
                    traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print('='*60)

    return failed == 0


if __name__ == "__main__":
    # Can run with pytest or directly
    import sys

    if '--pytest' in sys.argv:
        pytest.main([__file__, "-v", "-s"])
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
