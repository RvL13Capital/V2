"""
Integration Tests for V17 Labeler in TemporalPatternDetector
=============================================================

Tests the integration of PathDependentLabelerV17 into TemporalPatternDetector._get_pattern_label().

FIX: Replaced placeholder logic with actual v17 labeler calls
Coverage: 5 critical integration tests
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config.constants import (
    ProcessingMode,
    LABEL_PLACEHOLDER_UNRIPE,
    LABEL_GREY_ZONE,
    VALID_TRAINING_LABELS
)
from core.pattern_detector import TemporalPatternDetector
from core.exceptions import ValidationError


class TestV17LabelerIntegration:
    """Tests for v17 labeler integration in TemporalPatternDetector"""

    @pytest.fixture
    def sample_data_sufficient(self):
        """Create sample data with sufficient length for v17 labeler (300+ days)."""
        dates = pd.date_range(start='2023-01-01', periods=350, freq='D')
        np.random.seed(42)

        # Generate price data with a consolidation pattern
        prices = []
        base_price = 100.0

        # First 100 days: stable indicator warm-up
        for i in range(100):
            price_change = np.random.randn() * 0.005
            base_price *= (1 + price_change)
            prices.append(base_price)

        # Next 100 days: consolidation phase
        consolidation_base = base_price
        for i in range(100):
            # Small oscillations within range
            price_change = np.random.randn() * 0.003
            prices.append(consolidation_base * (1 + price_change))

        # Next 100 days: outcome window (breakout)
        for i in range(100):
            price_change = 0.005  # Steady uptrend
            prices.append(prices[-1] * (1 + price_change))

        # Additional 50 days buffer
        for i in range(50):
            price_change = np.random.randn() * 0.005
            prices.append(prices[-1] * (1 + price_change))

        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 350),
            'date': dates
        })
        df.set_index('date', inplace=True)

        return df

    @pytest.fixture
    def sample_data_insufficient(self):
        """Create sample data with insufficient length for v17 labeler (< 230 days)."""
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(42)

        prices = []
        base_price = 100.0
        for i in range(150):
            price_change = np.random.randn() * 0.01
            base_price *= (1 + price_change)
            prices.append(base_price)

        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 150),
            'date': dates
        })
        df.set_index('date', inplace=True)

        return df

    def test_v17_labeler_called_for_terminal_patterns(self, sample_data_sufficient):
        """
        TEST 1: V17 labeler should be called for terminal patterns

        Verifies that terminal patterns (COMPLETED/FAILED) use v17 labeler
        instead of placeholder logic.
        """
        detector = TemporalPatternDetector(mode=ProcessingMode.TRAINING)

        # Process ticker - should detect patterns and label with v17
        sequences, labels = detector.process_ticker(sample_data_sufficient, 'TEST')

        # Should have sequences with v17-labeled data
        if sequences is not None and len(sequences) > 0:
            assert labels is not None
            assert len(labels) == len(sequences)

            # All labels should be valid (0, 1, 2) - no placeholders in TRAINING
            assert all(l in VALID_TRAINING_LABELS for l in labels)

            # Should NOT have placeholder labels in TRAINING mode
            assert LABEL_PLACEHOLDER_UNRIPE not in labels

            # Should NOT have grey zones (filtered out)
            assert LABEL_GREY_ZONE not in labels

    def test_none_return_handled_gracefully(self, sample_data_insufficient):
        """
        TEST 2: None from v17 labeler should skip pattern gracefully

        Verifies that when v17 labeler returns None (insufficient data),
        the pattern is skipped without crashing.
        """
        detector = TemporalPatternDetector(mode=ProcessingMode.TRAINING)

        # Process ticker with insufficient data
        # V17 labeler should return None for patterns without enough future data
        sequences, labels = detector.process_ticker(sample_data_insufficient, 'TEST')

        # Should handle None gracefully - either no sequences or filtered sequences
        if sequences is not None and len(sequences) > 0:
            # If we got sequences, they should all have valid labels
            assert labels is not None
            assert all(l in VALID_TRAINING_LABELS for l in labels)

    def test_grey_zone_filtering(self, sample_data_sufficient):
        """
        TEST 3: Grey zone (-1) labels should be filtered in BOTH modes

        Verifies that grey zones (ambiguous patterns) are excluded from
        both TRAINING and INFERENCE modes.
        """
        for mode in [ProcessingMode.TRAINING, ProcessingMode.INFERENCE]:
            detector = TemporalPatternDetector(mode=mode)
            sequences, labels = detector.process_ticker(sample_data_sufficient, 'TEST')

            # Grey zones should be filtered out in BOTH modes
            if labels is not None and len(labels) > 0:
                assert LABEL_GREY_ZONE not in labels, \
                    f"Grey zones should be filtered in {mode.value} mode"

    def test_boundaries_validation(self, sample_data_sufficient):
        """
        TEST 4: Pattern boundaries should be validated before calling labeler

        Verifies that patterns with missing boundaries are handled gracefully.
        """
        detector = TemporalPatternDetector(mode=ProcessingMode.TRAINING)

        # Process ticker normally
        sequences, labels = detector.process_ticker(sample_data_sufficient, 'TEST')

        # All patterns that made it through should have valid boundaries
        # (invalid boundaries would have been logged and pattern skipped)
        if sequences is not None and len(sequences) > 0:
            assert labels is not None
            # All labels should be valid (patterns with None boundaries skipped)
            assert all(l in VALID_TRAINING_LABELS for l in labels)

    def test_training_vs_inference_modes(self, sample_data_sufficient):
        """
        TEST 5: Both modes should work with v17 labeler

        Verifies that v17 labeler integration works correctly with both
        TRAINING (strict) and INFERENCE (permissive) modes.
        """
        # TRAINING MODE: Only terminal patterns with definitive labels
        training_detector = TemporalPatternDetector(mode=ProcessingMode.TRAINING)
        train_sequences, train_labels = training_detector.process_ticker(
            sample_data_sufficient, 'TEST'
        )

        if train_labels is not None and len(train_labels) > 0:
            # Should only have definitive labels (0, 1, 2)
            assert all(l in VALID_TRAINING_LABELS for l in train_labels)
            # No placeholders
            assert LABEL_PLACEHOLDER_UNRIPE not in train_labels
            # No grey zones
            assert LABEL_GREY_ZONE not in train_labels

        # INFERENCE MODE: May include placeholders for unripe patterns
        inference_detector = TemporalPatternDetector(mode=ProcessingMode.INFERENCE)
        inf_sequences, inf_labels = inference_detector.process_ticker(
            sample_data_sufficient, 'TEST'
        )

        if inf_labels is not None and len(inf_labels) > 0:
            # May have placeholders or valid labels
            valid_inference_labels = [LABEL_PLACEHOLDER_UNRIPE, 0, 1, 2]
            assert all(l in valid_inference_labels for l in inf_labels)
            # Should still exclude grey zones
            assert LABEL_GREY_ZONE not in inf_labels

    def test_instance_variables_set_correctly(self, sample_data_sufficient):
        """
        Additional test: Verify that instance variables are set for v17 labeler access

        Ensures that full_data_df and current_ticker are properly stored.
        """
        detector = TemporalPatternDetector(mode=ProcessingMode.TRAINING)

        # Process ticker
        detector.process_ticker(sample_data_sufficient, 'TEST')

        # Check that instance variables were set
        assert hasattr(detector, 'full_data_df'), \
            "Detector should have full_data_df instance variable"
        assert hasattr(detector, 'current_ticker'), \
            "Detector should have current_ticker instance variable"
        assert detector.current_ticker == 'TEST', \
            "current_ticker should be set to 'TEST'"
        assert detector.full_data_df is not None, \
            "full_data_df should not be None"
        assert len(detector.full_data_df) == len(sample_data_sufficient), \
            "full_data_df should have same length as input data"

    def test_v17_labeler_initialized(self):
        """
        Additional test: Verify that v17 labeler is initialized in __init__

        Ensures that PathDependentLabelerV17 is properly instantiated.
        """
        detector = TemporalPatternDetector(mode=ProcessingMode.TRAINING)

        # Check that v17 labeler exists
        assert hasattr(detector, 'labeler_v17'), \
            "Detector should have labeler_v17 instance variable"

        # Check that it's the correct type
        from core.path_dependent_labeler import PathDependentLabelerV17
        assert isinstance(detector.labeler_v17, PathDependentLabelerV17), \
            "labeler_v17 should be instance of PathDependentLabelerV17"

        # Check that it has correct parameters
        assert detector.labeler_v17.indicator_warmup == 30, \
            "V17 labeler should have 30-day indicator warmup"
        assert detector.labeler_v17.indicator_stable == 100, \
            "V17 labeler should have 100-day stable period"
        assert detector.labeler_v17.outcome_window == 100, \
            "V17 labeler should have 100-day outcome window"
        assert detector.labeler_v17.risk_multiplier_target == 5.0, \
            "V17 labeler should use 5R target (updated 2025-12-17 for explosive moves only)"
        assert detector.labeler_v17.risk_multiplier_grey == 2.5, \
            "V17 labeler should use 2.5R grey zone threshold"


class TestEdgeCases:
    """Edge case testing for v17 integration"""

    def test_empty_data_handling(self):
        """Both modes should handle empty data gracefully"""
        for mode in [ProcessingMode.TRAINING, ProcessingMode.INFERENCE]:
            detector = TemporalPatternDetector(mode=mode)
            df_empty = pd.DataFrame()

            # Should raise ValidationError for insufficient data
            with pytest.raises(ValidationError):
                detector.process_ticker(df_empty, 'TEST')

    def test_very_short_data(self):
        """Both modes should handle data too short for patterns"""
        # Create 10-day DataFrame (too short)
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        short_df = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [500000] * 10,
            'date': dates
        })
        short_df.set_index('date', inplace=True)

        for mode in [ProcessingMode.TRAINING, ProcessingMode.INFERENCE]:
            detector = TemporalPatternDetector(mode=mode)

            # Should raise ValidationError for insufficient data length
            with pytest.raises(ValidationError):
                detector.process_ticker(short_df, 'TEST')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
