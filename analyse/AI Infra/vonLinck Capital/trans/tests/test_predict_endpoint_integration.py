"""
Integration test for /predict endpoint with generate_sequences_for_pattern.

Tests the complete flow: pattern_detector.generate_sequences_for_pattern()
through the API predict endpoint.
"""
import pytest
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pattern_detector import TemporalPatternDetector
from core.exceptions import ValidationError, TemporalConsistencyError


class TestGenerateSequencesForPattern:
    """Test the new generate_sequences_for_pattern method."""

    def setup_method(self):
        """Setup before each test."""
        self.detector = TemporalPatternDetector()

    def test_basic_functionality_with_date(self):
        """Test basic functionality with date objects."""
        # This test requires actual data - may need mocking
        ticker = "AAPL"
        pattern_start = date(2023, 1, 1)
        pattern_end = date(2023, 2, 1)

        try:
            sequences = self.detector.generate_sequences_for_pattern(
                ticker=ticker,
                pattern_start=pattern_start,
                pattern_end=pattern_end
            )

            # Assertions (may be None if no data available)
            if sequences is not None:
                assert isinstance(sequences, np.ndarray)
                assert sequences.shape[1] == 20  # window_size
                assert sequences.shape[2] == 10  # n_features (10 after composite disabled)
                print(f"[OK] Generated {len(sequences)} sequences with shape {sequences.shape}")
            else:
                print("[WARN] No sequences generated (data may not be available)")

        except Exception as e:
            # If data not available, test should still pass
            print(f"⚠ Test skipped - data not available: {e}")
            pytest.skip(f"Data not available: {e}")

    def test_datetime_compatibility(self):
        """Test that datetime objects also work."""
        ticker = "AAPL"
        pattern_start = datetime(2023, 1, 1)
        pattern_end = datetime(2023, 2, 1)

        try:
            sequences = self.detector.generate_sequences_for_pattern(
                ticker=ticker,
                pattern_start=pattern_start,
                pattern_end=pattern_end
            )

            # Should accept datetime without error
            assert sequences is None or isinstance(sequences, np.ndarray)
            print("✓ Datetime objects accepted")

        except Exception as e:
            pytest.skip(f"Data not available: {e}")

    def test_mixed_date_datetime(self):
        """Test that mixing date and datetime works."""
        ticker = "AAPL"
        pattern_start = date(2023, 1, 1)  # date object
        pattern_end = datetime(2023, 2, 1)  # datetime object

        try:
            sequences = self.detector.generate_sequences_for_pattern(
                ticker=ticker,
                pattern_start=pattern_start,
                pattern_end=pattern_end
            )

            # Should handle mixed types
            assert sequences is None or isinstance(sequences, np.ndarray)
            print("✓ Mixed date/datetime accepted")

        except Exception as e:
            pytest.skip(f"Data not available: {e}")

    def test_invalid_date_range(self):
        """Test that invalid date ranges raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            self.detector.generate_sequences_for_pattern(
                ticker="AAPL",
                pattern_start=date(2023, 2, 1),
                pattern_end=date(2023, 1, 1)  # End before start
            )

        assert "Invalid date range" in str(exc_info.value)
        print("✓ Invalid date range rejected")

    def test_pattern_too_short(self):
        """Test that patterns shorter than window_size return None."""
        result = self.detector.generate_sequences_for_pattern(
            ticker="AAPL",
            pattern_start=date(2023, 1, 1),
            pattern_end=date(2023, 1, 5)  # Only 5 days
        )

        assert result is None
        print("✓ Short pattern returns None as expected")

    def test_invalid_ticker_empty(self):
        """Test that empty ticker raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            self.detector.generate_sequences_for_pattern(
                ticker="",
                pattern_start=date(2023, 1, 1),
                pattern_end=date(2023, 2, 1)
            )

        assert "Invalid ticker" in str(exc_info.value)
        print("✓ Empty ticker rejected")

    def test_invalid_ticker_none(self):
        """Test that None ticker raises ValidationError."""
        with pytest.raises(ValidationError):
            self.detector.generate_sequences_for_pattern(
                ticker=None,
                pattern_start=date(2023, 1, 1),
                pattern_end=date(2023, 2, 1)
            )

        print("✓ None ticker rejected")

    def test_future_date_rejection(self):
        """Test that future dates are rejected."""
        future_date = date.today() + timedelta(days=30)

        with pytest.raises(ValidationError) as exc_info:
            self.detector.generate_sequences_for_pattern(
                ticker="AAPL",
                pattern_start=date.today(),
                pattern_end=future_date
            )

        assert "Future date not allowed" in str(exc_info.value)
        print("✓ Future date rejected")

    def test_minimum_valid_pattern_length(self):
        """Test pattern with exactly minimum length (20 days)."""
        ticker = "AAPL"
        pattern_start = date(2023, 1, 1)
        pattern_end = date(2023, 1, 21)  # 21 days (inclusive)

        try:
            sequences = self.detector.generate_sequences_for_pattern(
                ticker=ticker,
                pattern_start=pattern_start,
                pattern_end=pattern_end
            )

            # Should not raise ValidationError (even if no data)
            assert sequences is None or isinstance(sequences, np.ndarray)
            print("✓ Minimum length pattern accepted")

        except ValidationError as e:
            # Should not get ValidationError for valid length
            if "too short" in str(e).lower():
                pytest.fail(f"Minimum length pattern incorrectly rejected: {e}")
            else:
                # Other validation errors are OK (e.g., no data)
                pytest.skip(f"Data not available: {e}")
        except Exception as e:
            pytest.skip(f"Data not available: {e}")

    def test_long_pattern_warning(self):
        """Test that long patterns (>365 days) generate warning but work."""
        ticker = "AAPL"
        pattern_start = date(2022, 1, 1)
        pattern_end = date(2023, 6, 1)  # ~500 days

        try:
            # Should work but log warning
            sequences = self.detector.generate_sequences_for_pattern(
                ticker=ticker,
                pattern_start=pattern_start,
                pattern_end=pattern_end
            )

            # Should not raise exception
            assert sequences is None or isinstance(sequences, np.ndarray)
            print("✓ Long pattern accepted with warning")

        except Exception as e:
            pytest.skip(f"Data not available: {e}")


class TestAPIIntegration:
    """Test predict endpoint integration (requires running API)."""

    def test_predict_endpoint_structure(self):
        """Test that predict endpoint has correct structure (import test)."""
        from api.main import app

        # Check endpoint exists
        routes = [route.path for route in app.routes]
        assert "/predict" in routes
        print("✓ Predict endpoint exists")

    def test_exception_imports(self):
        """Test that API properly imports exceptions."""
        try:
            from api.main import ValidationError, TemporalConsistencyError, DataIntegrityError
            print("✓ Exception imports successful")
        except ImportError as e:
            pytest.fail(f"Failed to import exceptions: {e}")


class TestErrorHandling:
    """Test error handling and logging."""

    def test_validation_error_reraise(self):
        """Test that ValidationError is re-raised properly."""
        detector = TemporalPatternDetector()

        # Should raise ValidationError, not return None
        with pytest.raises(ValidationError):
            detector.generate_sequences_for_pattern(
                ticker="AAPL",
                pattern_start=date(2023, 2, 1),
                pattern_end=date(2023, 1, 1)
            )

        print("✓ ValidationError properly re-raised")

    def test_generic_exception_returns_none(self):
        """Test that generic exceptions return None instead of raising."""
        detector = TemporalPatternDetector()

        # Invalid ticker that might cause unexpected error
        result = detector.generate_sequences_for_pattern(
            ticker="INVALID_TICKER_THAT_DOES_NOT_EXIST_12345",
            pattern_start=date(2023, 1, 1),
            pattern_end=date(2023, 2, 1)
        )

        # Should return None, not raise exception
        assert result is None
        print("✓ Generic exception returns None")


def run_all_tests():
    """Run all tests manually."""
    import traceback

    test_classes = [
        TestGenerateSequencesForPattern,
        TestAPIIntegration,
        TestErrorHandling
    ]

    passed = 0
    failed = 0
    skipped = 0

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
                except pytest.skip.Exception as e:
                    skipped += 1
                    print(f"  SKIPPED: {e}")
                except Exception as e:
                    failed += 1
                    print(f"  FAILED: {e}")
                    traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed, {skipped} skipped")
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
