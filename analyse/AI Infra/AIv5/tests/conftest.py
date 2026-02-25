"""
Pytest configuration and fixtures for AIv4 tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add AIv4 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import (
    Settings,
    reload_settings,
    ConsolidationCriteria,
    MLConfig,
    DeepLearningConfig
)
from pattern_detection.models import ConsolidationPattern, PatternPhase


@pytest.fixture(scope="session")
def test_settings():
    """Create test settings instance."""
    return reload_settings(
        consolidation=ConsolidationCriteria(
            bbw_percentile_threshold=0.30,
            adx_threshold=32.0
        ),
        ml=MLConfig(n_estimators=10),  # Faster for tests
        deep_learning=DeepLearningConfig(enabled=False),  # Disable DL for fast tests
    )


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

    np.random.seed(42)
    data = {
        'open': 100 + np.random.randn(200) * 5,
        'high': 105 + np.random.randn(200) * 5,
        'low': 95 + np.random.randn(200) * 5,
        'close': 100 + np.random.randn(200) * 5,
        'volume': 1000000 + np.random.randint(-100000, 100000, 200)
    }

    df = pd.DataFrame(data, index=dates)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def sample_pattern():
    """Create a sample consolidation pattern."""
    return ConsolidationPattern(
        ticker="TEST",
        start_date=datetime(2024, 1, 1),
        start_idx=0,
        start_price=100.0,
        phase=PatternPhase.QUALIFYING,
    )


@pytest.fixture
def active_pattern(sample_pattern):
    """Create an active pattern with boundaries set."""
    from pattern_detection.models import PatternBoundaries

    sample_pattern.phase = PatternPhase.ACTIVE
    sample_pattern.activation_date = datetime(2024, 1, 11)
    sample_pattern.activation_idx = 10
    sample_pattern.qualification_highs = [102, 103, 104, 103, 102]
    sample_pattern.qualification_lows = [98, 97, 96, 97, 98]

    sample_pattern.boundaries = PatternBoundaries(
        upper=104.0,
        lower=96.0,
        power=104.52,  # 104 * 1.005
        range_pct=0.0833  # (104-96)/96
    )

    return sample_pattern


@pytest.fixture
def completed_pattern_data():
    """Create data for a completed pattern with snapshots."""
    return {
        'ticker': 'AAPL',
        'pattern_id': 'AAPL_2024-01-01',
        'start_date': datetime(2024, 1, 1),
        'end_date': datetime(2024, 2, 15),
        'breakout_direction': 'UP',
        'max_gain': 0.45,  # 45%
        'snapshots': [
            {
                'snapshot_date': datetime(2024, 1, 15),
                'days_since_activation': 5,
                'price_position_in_range': 0.4,
            },
            {
                'snapshot_date': datetime(2024, 1, 30),
                'days_since_activation': 20,
                'price_position_in_range': 0.6,
            },
        ]
    }


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings after each test."""
    yield
    reload_settings()  # Reset to defaults
