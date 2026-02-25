"""
Tests for Ignition-Aware Pattern Prioritization
================================================

Tests the ignition scoring, prioritization, and oversampling functionality.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.ignition_prioritizer import (
    calculate_pattern_ignition_score,
    calculate_recency_score,
    score_patterns_for_ignition,
    prioritize_and_sort,
    oversample_high_ignition,
    stratified_oversample,
    prioritize_and_oversample,
    get_ignition_statistics,
    IGNITION_HIGH_THRESHOLD,
    IGNITION_MED_THRESHOLD,
    PRIORITY_WEIGHTS,
)


class TestIgnitionScoring:
    """Tests for ignition score calculation"""

    def test_score_with_coil_features(self):
        """Ignition score computed from coil features"""
        pattern = pd.Series({
            'ticker': 'AAPL',
            'price_position_at_end': 0.7,  # Good position
            'bbw_slope_5d': 0.005,  # Slightly expanding
            'vol_trend_5d': 1.3,  # Increasing volume
            'coil_intensity': 0.6,  # Good coil
        })

        score = calculate_pattern_ignition_score(pattern)

        assert 0 <= score <= 1
        assert score > 0.5  # Should be above neutral

    def test_score_uses_precomputed(self):
        """Uses pre-computed ignition_score if available"""
        pattern = pd.Series({
            'ticker': 'AAPL',
            'ignition_score': 0.85,
            'price_position_at_end': 0.1,  # Low - would give different score
        })

        score = calculate_pattern_ignition_score(pattern)

        assert score == 0.85  # Should use pre-computed

    def test_score_fallback_to_basic(self):
        """Falls back to basic features when coil features missing"""
        pattern = pd.Series({
            'ticker': 'AAPL',
            'box_width': 0.05,  # Tight box
        })

        score = calculate_pattern_ignition_score(pattern)

        assert 0 <= score <= 1

    def test_high_position_optimal_range(self):
        """Position in 0.6-0.85 range gives highest score"""
        pattern_optimal = pd.Series({
            'price_position_at_end': 0.75,  # Optimal
        })
        pattern_low = pd.Series({
            'price_position_at_end': 0.3,  # Too low
        })
        pattern_very_high = pd.Series({
            'price_position_at_end': 0.95,  # Too high
        })

        score_optimal = calculate_pattern_ignition_score(pattern_optimal)
        score_low = calculate_pattern_ignition_score(pattern_low)
        score_high = calculate_pattern_ignition_score(pattern_very_high)

        # Optimal should be highest
        assert score_optimal > score_low
        assert score_optimal >= score_high


class TestRecencyScoring:
    """Tests for recency score calculation"""

    def test_today_is_max(self):
        """Pattern ending today gets score 1.0"""
        today = datetime.now()
        score = calculate_recency_score(today, today)

        assert score == 1.0

    def test_decay_over_time(self):
        """Score decays for older patterns"""
        reference = datetime(2024, 6, 1)
        recent = datetime(2024, 5, 25)  # 7 days ago
        old = datetime(2024, 3, 1)  # 90 days ago

        score_recent = calculate_recency_score(recent, reference)
        score_old = calculate_recency_score(old, reference)

        assert score_recent > score_old
        assert score_recent > 0.9  # Recent should be high
        assert score_old >= 0.5  # Old should be at floor

    def test_floor_at_half(self):
        """Score never goes below 0.5"""
        reference = datetime(2024, 6, 1)
        very_old = datetime(2020, 1, 1)  # Years ago

        score = calculate_recency_score(very_old, reference)

        assert score == 0.5


class TestPatternScoring:
    """Tests for batch pattern scoring"""

    def create_test_patterns(self, n=10):
        """Create test patterns DataFrame"""
        base_date = datetime(2024, 1, 1)
        return pd.DataFrame({
            'ticker': [f'TICK{i}' for i in range(n)],
            'outcome_class': [i % 3 for i in range(n)],
            'end_date': [base_date + timedelta(days=i*10) for i in range(n)],
            'price_position_at_end': np.random.uniform(0, 1, n),
            'bbw_slope_5d': np.random.uniform(-0.02, 0.02, n),
            'vol_trend_5d': np.random.uniform(0.5, 2.0, n),
            'coil_intensity': np.random.uniform(0, 1, n),
        })

    def test_scoring_adds_columns(self):
        """Scoring adds required columns"""
        patterns = self.create_test_patterns()
        scored = score_patterns_for_ignition(patterns)

        assert 'ignition_score' in scored.columns or 'ignition_score_computed' in scored.columns
        assert 'recency_score' in scored.columns
        assert 'priority_score' in scored.columns

    def test_priority_score_range(self):
        """Priority score is bounded [0, 1]"""
        patterns = self.create_test_patterns(100)
        scored = score_patterns_for_ignition(patterns)

        assert scored['priority_score'].min() >= 0
        assert scored['priority_score'].max() <= 1

    def test_priority_weighted_combination(self):
        """Priority is weighted combination of components"""
        patterns = pd.DataFrame({
            'ticker': ['A'],
            'end_date': [datetime.now()],
            'price_position_at_end': [0.75],
            'coil_intensity': [0.8],
        })

        scored = score_patterns_for_ignition(patterns)

        # Priority should exist and be reasonable
        assert scored['priority_score'].iloc[0] > 0


class TestPrioritization:
    """Tests for sorting by priority"""

    def test_sort_descending(self):
        """Patterns sorted by priority descending"""
        patterns = pd.DataFrame({
            'ticker': ['A', 'B', 'C'],
            'priority_score': [0.3, 0.9, 0.5],
        })

        sorted_df = prioritize_and_sort(patterns)

        assert sorted_df['priority_score'].iloc[0] == 0.9
        assert sorted_df['priority_score'].iloc[1] == 0.5
        assert sorted_df['priority_score'].iloc[2] == 0.3


class TestOversampling:
    """Tests for oversampling functionality"""

    def test_oversample_increases_count(self):
        """Oversampling increases pattern count"""
        patterns = pd.DataFrame({
            'ticker': [f'T{i}' for i in range(100)],
            'priority_score': np.linspace(0, 1, 100),
        })

        original_count = len(patterns)
        oversampled = oversample_high_ignition(patterns, oversample_factor=2.0, top_pct=0.2)

        # Top 20% (20 patterns) doubled = 20 extra
        assert len(oversampled) == original_count + 20

    def test_oversample_marks_copies(self):
        """Oversampled patterns are marked"""
        patterns = pd.DataFrame({
            'ticker': [f'T{i}' for i in range(50)],
            'priority_score': np.linspace(0, 1, 50),
        })

        oversampled = oversample_high_ignition(patterns, oversample_factor=3.0, top_pct=0.2)

        assert 'oversample_copy' in oversampled.columns
        assert oversampled['oversample_copy'].max() == 2  # 3x = 2 copies

    def test_oversample_factor_one_no_change(self):
        """Oversample factor 1.0 doesn't change count"""
        patterns = pd.DataFrame({
            'ticker': ['A', 'B', 'C'],
            'priority_score': [0.9, 0.5, 0.3],
        })

        oversampled = oversample_high_ignition(patterns, oversample_factor=1.0)

        assert len(oversampled) == 3


class TestStratifiedOversampling:
    """Tests for stratified oversampling"""

    def test_maintains_class_balance(self):
        """Class distribution maintained after stratified oversampling"""
        np.random.seed(42)
        patterns = pd.DataFrame({
            'ticker': [f'T{i}' for i in range(100)],
            'outcome_class': [0]*30 + [1]*50 + [2]*20,  # 30/50/20 distribution
            'priority_score': np.random.uniform(0, 1, 100),
        })

        # Only oversample Target class (2)
        oversampled = stratified_oversample(
            patterns,
            oversample_factor=2.0,
            top_pct=0.2,
            target_class=2
        )

        # Danger and Noise counts should be unchanged
        assert (oversampled['outcome_class'] == 0).sum() == 30
        assert (oversampled['outcome_class'] == 1).sum() == 50
        # Target count should increase (top 20% of 20 = 4 patterns, doubled = +4)
        assert (oversampled['outcome_class'] == 2).sum() > 20

    def test_stratified_all_classes(self):
        """All classes can be oversampled when target_class=None"""
        patterns = pd.DataFrame({
            'ticker': [f'T{i}' for i in range(60)],
            'outcome_class': [0]*20 + [1]*20 + [2]*20,
            'priority_score': np.random.uniform(0, 1, 60),
        })

        oversampled = stratified_oversample(
            patterns,
            oversample_factor=2.0,
            top_pct=0.2,
            target_class=None
        )

        # All classes should have increased counts
        assert len(oversampled) > 60


class TestFullPipeline:
    """End-to-end tests for prioritization pipeline"""

    def test_prioritize_and_oversample_complete(self):
        """Full pipeline runs without error"""
        np.random.seed(42)
        patterns = pd.DataFrame({
            'ticker': [f'T{i}' for i in range(50)],
            'outcome_class': np.random.choice([0, 1, 2], 50, p=[0.2, 0.6, 0.2]),
            'end_date': [datetime(2024, 1, 1) + timedelta(days=i*2) for i in range(50)],
            'price_position_at_end': np.random.uniform(0, 1, 50),
            'coil_intensity': np.random.uniform(0, 1, 50),
        })

        result = prioritize_and_oversample(
            patterns,
            oversample_factor=2.0,
            top_pct=0.2,
            stratified=True
        )

        assert len(result) >= len(patterns)
        assert 'priority_score' in result.columns

    def test_statistics_complete(self):
        """Statistics function returns expected fields"""
        patterns = pd.DataFrame({
            'ticker': ['A', 'B', 'C'],
            'outcome_class': [0, 1, 2],
            'ignition_score': [0.3, 0.5, 0.8],
            'priority_score': [0.4, 0.6, 0.7],
        })

        stats = get_ignition_statistics(patterns)

        assert 'n_patterns' in stats
        assert 'ignition_mean' in stats
        assert 'class_distribution' in stats
        assert stats['n_patterns'] == 3


class TestEdgeCases:
    """Edge case handling"""

    def test_empty_dataframe(self):
        """Handle empty DataFrame gracefully"""
        patterns = pd.DataFrame(columns=['ticker', 'priority_score', 'outcome_class'])
        result = oversample_high_ignition(patterns)

        assert len(result) == 0

    def test_single_pattern(self):
        """Handle single pattern"""
        patterns = pd.DataFrame({
            'ticker': ['A'],
            'priority_score': [0.8],
        })

        result = oversample_high_ignition(patterns, top_pct=0.5)
        assert len(result) >= 1

    def test_missing_priority_column(self):
        """Handle missing priority_score gracefully"""
        patterns = pd.DataFrame({
            'ticker': ['A', 'B'],
        })

        # Should not crash
        result = prioritize_and_sort(patterns)
        assert len(result) == 2

    def test_nan_in_scores(self):
        """Handle NaN values in scores"""
        patterns = pd.DataFrame({
            'ticker': ['A', 'B', 'C'],
            'price_position_at_end': [0.5, np.nan, 0.7],
            'coil_intensity': [np.nan, 0.6, np.nan],
        })

        scored = score_patterns_for_ignition(patterns)

        # Should handle NaN gracefully
        assert not scored['priority_score'].isna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
