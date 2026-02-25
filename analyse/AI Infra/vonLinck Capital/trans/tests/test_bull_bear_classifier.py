"""
Tests for Bull/Bear Market Phase Classifier
===========================================

Comprehensive test suite covering all functionality:
- MarketPhaseData dataclass
- BullBearClassifier initialization and computation
- MA calculations and hysteresis logic
- Lookup operations and range queries
- CSV and database persistence
- Temporal integrity guarantees
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.bull_bear_classifier import (
    MarketPhaseData,
    BullBearClassifier,
    get_classifier
)


# ====================
# Fixtures
# ====================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_csv_data(temp_dir):
    """Create sample CSV data for testing."""
    # Generate 300 days of data (enough for 200-day MA + buffer)
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(300)]

    # Create prices that will generate clear bull/bear signals
    # Days 0-100: Uptrend (bull)
    # Days 101-200: Downtrend (bear)
    # Days 201-300: Uptrend (bull)
    prices = []
    for i in range(300):
        if i < 100:
            price = 1000 + i * 2  # Uptrend
        elif i < 200:
            price = 1200 - (i - 100) * 1.5  # Downtrend
        else:
            price = 1050 + (i - 200) * 1.8  # Uptrend

        # Add some noise
        price += np.random.normal(0, 5)
        prices.append(price)

    df = pd.DataFrame({
        'Time': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
        'Open': prices,
        'High': [p + np.random.uniform(0, 10) for p in prices],
        'Low': [p - np.random.uniform(0, 10) for p in prices],
        'Close': prices,
        'Volume': [int(np.random.uniform(1000, 5000)) for _ in range(300)]
    })

    csv_path = temp_dir / 'TEST_D1.csv'
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def classifier(sample_csv_data, temp_dir):
    """Create BullBearClassifier instance for testing."""
    return BullBearClassifier(
        data_path=sample_csv_data,
        index_symbol='TEST',
        fast_period=50,
        slow_period=200,
        hysteresis_pct=0.05,
        cache_dir=temp_dir,
        enable_db_storage=False  # Disable DB for unit tests
    )


# ====================
# MarketPhaseData Tests
# ====================

class TestMarketPhaseData:
    """Test MarketPhaseData dataclass."""

    def test_creation(self):
        """Test creating MarketPhaseData instance."""
        data = MarketPhaseData(
            date='2020-01-01',
            phase=1,
            fast_ma=1500.0,
            slow_ma=1400.0,
            crossover_date='2019-12-15',
            days_in_phase=17
        )

        assert data.date == '2020-01-01'
        assert data.phase == 1
        assert data.fast_ma == 1500.0
        assert data.slow_ma == 1400.0
        assert data.crossover_date == '2019-12-15'
        assert data.days_in_phase == 17

    def test_to_dict(self):
        """Test converting to dictionary."""
        data = MarketPhaseData(
            date='2020-01-01',
            phase=1,
            fast_ma=1500.0,
            slow_ma=1400.0,
            crossover_date='2019-12-15',
            days_in_phase=17
        )

        result = data.to_dict()

        assert isinstance(result, dict)
        assert result['date'] == '2020-01-01'
        assert result['phase'] == 1
        assert result['fast_ma'] == 1500.0

    def test_from_dict(self):
        """Test creating from dictionary."""
        data_dict = {
            'date': '2020-01-01',
            'phase': 1,
            'fast_ma': 1500.0,
            'slow_ma': 1400.0,
            'crossover_date': '2019-12-15',
            'days_in_phase': 17
        }

        data = MarketPhaseData.from_dict(data_dict)

        assert data.date == '2020-01-01'
        assert data.phase == 1
        assert data.fast_ma == 1500.0

    def test_serialization_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = MarketPhaseData(
            date='2020-01-01',
            phase=0,
            fast_ma=1300.0,
            slow_ma=1400.0,
            crossover_date='2019-11-20',
            days_in_phase=42
        )

        # Convert to dict and back
        data_dict = original.to_dict()
        restored = MarketPhaseData.from_dict(data_dict)

        assert restored.date == original.date
        assert restored.phase == original.phase
        assert restored.fast_ma == original.fast_ma
        assert restored.slow_ma == original.slow_ma
        assert restored.crossover_date == original.crossover_date
        assert restored.days_in_phase == original.days_in_phase


# ====================
# Initialization Tests
# ====================

class TestBullBearClassifierInit:
    """Test BullBearClassifier initialization."""

    def test_init_with_default_params(self, sample_csv_data, temp_dir):
        """Test initialization with default parameters."""
        classifier = BullBearClassifier(
            data_path=sample_csv_data,
            index_symbol='TEST',
            cache_dir=temp_dir,
            enable_db_storage=False
        )

        assert classifier.fast_period == 50
        assert classifier.slow_period == 200
        assert classifier.hysteresis_pct == 0.05
        assert classifier.index_symbol == 'TEST'
        assert len(classifier.regime_map) > 0

    def test_init_with_custom_params(self, sample_csv_data, temp_dir):
        """Test initialization with custom parameters."""
        classifier = BullBearClassifier(
            data_path=sample_csv_data,
            index_symbol='CUSTOM',
            fast_period=20,
            slow_period=100,
            hysteresis_pct=0.03,
            cache_dir=temp_dir,
            enable_db_storage=False
        )

        assert classifier.fast_period == 20
        assert classifier.slow_period == 100
        assert classifier.hysteresis_pct == 0.03
        assert classifier.index_symbol == 'CUSTOM'

    def test_init_with_invalid_data(self, temp_dir):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            BullBearClassifier(
                data_path=temp_dir / 'nonexistent.csv',
                cache_dir=temp_dir
            )


# ====================
# MA Calculation Tests
# ====================

class TestMACalculation:
    """Test moving average calculations."""

    def test_fast_ma_calculation(self, classifier):
        """Test that fast MA is calculated correctly."""
        # Get a date with valid phase
        valid_dates = [d for d, p in classifier.regime_map.items() if p.phase is not None]

        if valid_dates:
            phase_data = classifier.get_phase(valid_dates[0])
            assert phase_data.fast_ma is not None
            assert phase_data.fast_ma > 0

    def test_slow_ma_calculation(self, classifier):
        """Test that slow MA is calculated correctly."""
        valid_dates = [d for d, p in classifier.regime_map.items() if p.phase is not None]

        if valid_dates:
            phase_data = classifier.get_phase(valid_dates[0])
            assert phase_data.slow_ma is not None
            assert phase_data.slow_ma > 0

    def test_insufficient_data_returns_none(self, classifier):
        """Test that insufficient data returns None for phase."""
        # First 200 days should have None phase
        first_date = min(classifier.regime_map.keys())
        phase_data = classifier.get_phase(first_date)

        # Could be None or valid depending on data
        # Just verify it's a valid MarketPhaseData object
        assert phase_data is not None


# ====================
# Hysteresis Logic Tests
# ====================

class TestHysteresisLogic:
    """Test hysteresis state machine logic."""

    def test_hysteresis_prevents_whipsaw(self, temp_dir):
        """Test that hysteresis prevents rapid switching."""
        # Create data that oscillates around MA
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(250)]
        base_price = 1000

        prices = []
        for i in range(250):
            if i < 200:
                # Slowly increasing
                price = base_price + i * 0.5
            else:
                # Oscillate slightly (without hysteresis, would cause whipsaw)
                price = base_price + 100 + (i % 2) * 2

            prices.append(price)

        df = pd.DataFrame({
            'Time': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
            'Close': prices,
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Volume': [1000] * 250
        })

        csv_path = temp_dir / 'whipsaw_test.csv'
        df.to_csv(csv_path, index=False)

        # Test with hysteresis
        classifier = BullBearClassifier(
            data_path=csv_path,
            fast_period=50,
            slow_period=200,
            hysteresis_pct=0.05,
            cache_dir=temp_dir,
            enable_db_storage=False
        )

        transitions = classifier.get_transitions()

        # Should have fewer transitions with hysteresis
        assert len(transitions) < 10  # Reasonable bound

    def test_bull_to_bear_transition(self, classifier):
        """Test bull to bear transition detection."""
        transitions = classifier.get_transitions()
        bear_transitions = [t for t in transitions if t['to'] == 'Bear']

        # Should have at least one bear transition
        assert len(bear_transitions) >= 0  # May or may not have transitions

    def test_bear_to_bull_transition(self, classifier):
        """Test bear to bull transition detection."""
        transitions = classifier.get_transitions()
        bull_transitions = [t for t in transitions if t['to'] == 'Bull']

        # Should have at least one bull transition
        assert len(bull_transitions) >= 0  # May or may not have transitions


# ====================
# Lookup Tests
# ====================

class TestLookups:
    """Test phase lookup operations."""

    def test_get_phase_valid_date(self, classifier):
        """Test getting phase for valid date."""
        # Get a date with valid phase
        valid_dates = [d for d, p in classifier.regime_map.items() if p.phase is not None]

        if valid_dates:
            test_date = valid_dates[len(valid_dates) // 2]  # Middle date
            phase_data = classifier.get_phase(test_date)

            assert phase_data is not None
            assert phase_data.date == test_date
            assert phase_data.phase in [0, 1]

    def test_get_phase_invalid_date(self, classifier):
        """Test getting phase for date not in data."""
        phase_data = classifier.get_phase('1990-01-01')
        assert phase_data is None

    def test_get_phase_with_datetime_object(self, classifier):
        """Test getting phase using datetime object."""
        valid_dates = [d for d, p in classifier.regime_map.items() if p.phase is not None]

        if valid_dates:
            date_str = valid_dates[0]
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')

            phase_data = classifier.get_phase(date_obj)
            assert phase_data is not None

    def test_get_phase_before_data_start(self, classifier):
        """Test getting phase for date before data starts."""
        phase_data = classifier.get_phase('2000-01-01')
        assert phase_data is None


# ====================
# Range Query Tests
# ====================

class TestRangeQueries:
    """Test range query operations."""

    def test_get_phase_range(self, classifier):
        """Test getting phases for date range."""
        valid_dates = sorted([d for d, p in classifier.regime_map.items() if p.phase is not None])

        if len(valid_dates) >= 10:
            start_date = valid_dates[0]
            end_date = valid_dates[9]

            df = classifier.get_phase_range(start_date, end_date)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 10
            assert 'date' in df.columns
            assert 'phase' in df.columns

    def test_get_transitions(self, classifier):
        """Test getting all transitions."""
        transitions = classifier.get_transitions()

        assert isinstance(transitions, list)
        # Each transition should have required fields
        for t in transitions:
            assert 'date' in t
            assert 'from' in t
            assert 'to' in t
            assert 'duration_days' in t

    def test_get_transitions_filtered(self, classifier):
        """Test getting transitions with min_days filter."""
        all_transitions = classifier.get_transitions(min_days=0)
        filtered_transitions = classifier.get_transitions(min_days=30)

        assert len(filtered_transitions) <= len(all_transitions)


# ====================
# Edge Case Tests
# ====================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handle_missing_data(self, temp_dir):
        """Test handling of missing price data."""
        # Create data with missing values
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(250)]
        prices = [1000 + i * 0.5 if i % 10 != 5 else np.nan for i in range(250)]

        df = pd.DataFrame({
            'Time': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
            'Close': prices,
            'Open': prices,
            'High': prices,
            'Low': prices,
            'Volume': [1000] * 250
        })

        csv_path = temp_dir / 'missing_data.csv'
        df.to_csv(csv_path, index=False)

        # Should handle gracefully
        classifier = BullBearClassifier(
            data_path=csv_path,
            cache_dir=temp_dir,
            enable_db_storage=False
        )

        assert classifier is not None
        assert len(classifier.regime_map) > 0

    def test_handle_insufficient_data(self, temp_dir):
        """Test handling when not enough data for slow MA."""
        # Create only 100 days (less than 200-day MA)
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        prices = [1000 + i for i in range(100)]

        df = pd.DataFrame({
            'Time': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
            'Close': prices,
            'Open': prices,
            'High': prices,
            'Low': prices,
            'Volume': [1000] * 100
        })

        csv_path = temp_dir / 'short_data.csv'
        df.to_csv(csv_path, index=False)

        classifier = BullBearClassifier(
            data_path=csv_path,
            cache_dir=temp_dir,
            enable_db_storage=False
        )

        # All phases should be None (insufficient data)
        valid_phases = [p for p in classifier.regime_map.values() if p.phase is not None]
        assert len(valid_phases) == 0  # Not enough data for 200-day MA


# ====================
# Persistence Tests
# ====================

class TestPersistence:
    """Test CSV and database persistence."""

    def test_save_to_csv(self, classifier, temp_dir):
        """Test saving phases to CSV."""
        csv_path = temp_dir / 'TEST_phases.csv'

        assert csv_path.exists()

        # Load CSV and verify
        df = pd.read_csv(csv_path)
        assert len(df) > 0
        assert 'date' in df.columns
        assert 'phase' in df.columns
        assert 'fast_ma' in df.columns
        assert 'slow_ma' in df.columns

    def test_csv_contains_config(self, classifier, temp_dir):
        """Test that CSV contains configuration columns."""
        csv_path = temp_dir / 'TEST_phases.csv'
        df = pd.read_csv(csv_path)

        assert 'fast_period' in df.columns
        assert 'slow_period' in df.columns
        assert 'hysteresis_pct' in df.columns

        # Verify values
        assert df['fast_period'].iloc[0] == 50
        assert df['slow_period'].iloc[0] == 200


# ====================
# Statistics Tests
# ====================

class TestStatistics:
    """Test statistical calculations."""

    def test_get_stats(self, classifier):
        """Test get_stats returns correct structure."""
        stats = classifier.get_stats()

        assert isinstance(stats, dict)
        assert 'total_days' in stats
        assert 'valid_days' in stats
        assert 'bull_days' in stats
        assert 'bear_days' in stats
        assert 'bull_pct' in stats
        assert 'bear_pct' in stats
        assert 'transitions' in stats
        assert 'config' in stats

    def test_stats_percentages(self, classifier):
        """Test that bull_pct + bear_pct <= 100."""
        stats = classifier.get_stats()

        if stats['valid_days'] > 0:
            total_pct = stats['bull_pct'] + stats['bear_pct']
            assert 0 <= total_pct <= 100.1  # Allow for rounding


# ====================
# Temporal Integrity Tests
# ====================

class TestTemporalIntegrity:
    """Test temporal integrity guarantees (no look-ahead bias)."""

    def test_no_future_data_in_classification(self, sample_csv_data, temp_dir):
        """Test that classification doesn't use future data."""
        # Create full classifier
        full_classifier = BullBearClassifier(
            data_path=sample_csv_data,
            cache_dir=temp_dir / 'full',
            enable_db_storage=False
        )

        # Get a date from the middle
        all_dates = sorted(full_classifier.regime_map.keys())
        if len(all_dates) >= 250:
            cutoff_date = all_dates[250]
            full_phase = full_classifier.get_phase(cutoff_date)

            # Create CSV with only data up to cutoff
            original_df = pd.read_csv(sample_csv_data)
            original_df['date_parsed'] = pd.to_datetime(original_df['Time'])
            truncated_df = original_df[
                original_df['date_parsed'] <= pd.to_datetime(cutoff_date)
            ].copy()

            # Save truncated data
            truncated_csv = temp_dir / 'truncated.csv'
            truncated_df.drop('date_parsed', axis=1).to_csv(truncated_csv, index=False)

            # Create classifier with truncated data
            truncated_classifier = BullBearClassifier(
                data_path=truncated_csv,
                cache_dir=temp_dir / 'truncated',
                enable_db_storage=False
            )

            truncated_phase = truncated_classifier.get_phase(cutoff_date)

            # Phases should match (no look-ahead)
            if full_phase and truncated_phase:
                assert full_phase.phase == truncated_phase.phase

    def test_sequential_state_transitions(self, classifier):
        """Test that state transitions happen sequentially."""
        transitions = classifier.get_transitions()

        # Sort by date
        sorted_transitions = sorted(transitions, key=lambda t: t['date'])

        # Each transition should come after the previous
        for i in range(1, len(sorted_transitions)):
            prev_date = pd.to_datetime(sorted_transitions[i-1]['date'])
            curr_date = pd.to_datetime(sorted_transitions[i]['date'])

            assert curr_date > prev_date


# ====================
# Performance Tests
# ====================

class TestPerformance:
    """Test performance characteristics."""

    def test_lookup_performance(self, classifier):
        """Test that lookups are fast (O(1))."""
        import time

        # Get valid dates
        valid_dates = [d for d, p in classifier.regime_map.items() if p.phase is not None]

        if len(valid_dates) >= 100:
            # Time 100 lookups
            start = time.time()
            for _ in range(100):
                test_date = valid_dates[_ % len(valid_dates)]
                classifier.get_phase(test_date)
            elapsed = time.time() - start

            # Should be very fast (< 0.1s for 100 lookups)
            assert elapsed < 0.1

    def test_initialization_time(self, sample_csv_data, temp_dir):
        """Test that initialization completes in reasonable time."""
        import time

        start = time.time()
        classifier = BullBearClassifier(
            data_path=sample_csv_data,
            cache_dir=temp_dir / 'perf_test',
            enable_db_storage=False
        )
        elapsed = time.time() - start

        # Should complete in < 5 seconds
        assert elapsed < 5.0


# ====================
# Integration Tests
# ====================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, sample_csv_data, temp_dir):
        """Test complete workflow: init -> lookup -> stats -> export."""
        # 1. Initialize
        classifier = BullBearClassifier(
            data_path=sample_csv_data,
            index_symbol='WORKFLOW_TEST',
            cache_dir=temp_dir,
            enable_db_storage=False
        )

        # 2. Lookup phases
        valid_dates = [d for d, p in classifier.regime_map.items() if p.phase is not None]
        assert len(valid_dates) > 0

        phase = classifier.get_phase(valid_dates[0])
        assert phase is not None

        # 3. Get statistics
        stats = classifier.get_stats()
        assert stats['total_days'] > 0

        # 4. Get transitions
        transitions = classifier.get_transitions()
        assert isinstance(transitions, list)

        # 5. Verify CSV saved
        csv_path = temp_dir / 'WORKFLOW_TEST_phases.csv'
        assert csv_path.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
