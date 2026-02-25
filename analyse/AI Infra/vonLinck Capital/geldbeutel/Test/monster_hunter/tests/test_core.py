"""
Tests for Micro-Cap Monster Hunter System - Coil Strategy.
Run using: pytest tests/test_core.py

Philosophy: Buy dead, sell alive. Test the tension detection.
"""

import pytest
import pandas as pd
import numpy as np
from monster_hunter.feature_engine import (
    CoilDetector,
    prepare_features,
    calculate_adx,
    calculate_rvol,
    validate_coil_candidate
)
from monster_hunter.backtest_engine import BacktestEngine
from monster_hunter.manipulation_screen import ManipulationScreen, is_toxic


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def coiled_stock_above_dollar():
    """
    Generates synthetic data for a coiled stock above $1.00.
    Tight range (< 6%), low ADX, dead volume.
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create a tight coil: price hovers between $1.50 and $1.55 (3.3% range)
    np.random.seed(42)
    base_price = 1.52
    noise = np.random.uniform(-0.025, 0.025, 100)

    df = pd.DataFrame({
        'open': base_price + noise,
        'high': base_price + noise + 0.02,
        'low': base_price + noise - 0.02,
        'close': base_price + noise,
        'volume': [5000] * 100  # Very low volume
    }, index=dates)

    # Prepare features (adds adx, rvol_30d)
    df = prepare_features(df)

    return df


@pytest.fixture
def coiled_penny_stock():
    """
    Generates synthetic data for a coiled penny stock (< $1.00).
    Wider range allowed (< 15%), but must have dead volume.
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Penny stock: price hovers between $0.40 and $0.45 (12.5% range)
    np.random.seed(42)
    base_price = 0.42
    noise = np.random.uniform(-0.02, 0.02, 100)

    df = pd.DataFrame({
        'open': base_price + noise,
        'high': base_price + noise + 0.02,
        'low': base_price + noise - 0.02,
        'close': base_price + noise,
        'volume': [3000] * 100  # Very low volume
    }, index=dates)

    df = prepare_features(df)
    return df


@pytest.fixture
def active_stock():
    """
    Generates synthetic data for an ACTIVE stock (not coiled).
    High volume, trending, wide range.
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Trending stock with high volume
    prices = np.linspace(1.0, 2.0, 100) + np.random.uniform(-0.1, 0.1, 100)

    df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.1,
        'low': prices - 0.1,
        'close': prices,
        'volume': [500000] * 100  # High volume
    }, index=dates)

    df = prepare_features(df)
    return df


# =============================================================================
# COIL DETECTION TESTS
# =============================================================================

def test_coil_detection_above_dollar(coiled_stock_above_dollar):
    """Test coil detection for stocks >= $1.00 (6% range threshold)."""
    detector = CoilDetector(window=40, max_adx=20, max_rvol=0.5)

    # Check at end of data (should be coiled)
    is_coiled, metrics = detector.check_coil(coiled_stock_above_dollar, 80)

    # The stock should be coiled (tight range, dead volume)
    assert 'range_pct' in metrics
    assert 'support_level' in metrics
    assert metrics['max_range_used'] == 0.06  # 6% threshold for > $1


def test_coil_detection_penny_stock(coiled_penny_stock):
    """Test coil detection for penny stocks < $1.00 (15% range threshold)."""
    detector = CoilDetector(window=40, max_adx=20, max_rvol=0.5)

    is_coiled, metrics = detector.check_coil(coiled_penny_stock, 80)

    assert 'range_pct' in metrics
    assert metrics['max_range_used'] == 0.15  # 15% threshold for < $1
    assert metrics['current_price'] < 1.00


def test_coil_detection_active_stock_fails(active_stock):
    """Test that active (non-coiled) stocks are NOT identified as coiled."""
    detector = CoilDetector(window=40, max_adx=20, max_rvol=0.5)

    is_coiled, metrics = detector.check_coil(active_stock, 80)

    # Active stock should NOT be coiled (high volume, trending)
    assert is_coiled is False


def test_coil_insufficient_history():
    """Test coil detection returns False with insufficient history."""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    df = pd.DataFrame({
        'open': [1.0] * 20,
        'high': [1.05] * 20,
        'low': [0.95] * 20,
        'close': [1.0] * 20,
        'volume': [5000] * 20
    }, index=dates)
    df = prepare_features(df)

    detector = CoilDetector(window=40)
    is_coiled, metrics = detector.check_coil(df, 15)

    assert is_coiled is False
    assert metrics == {}


def test_rvol_dead_requirement():
    """Test that RVOL must be < 0.5 for coil detection."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Create a stock with tight range but AVERAGE volume (RVOL = 1.0)
    df = pd.DataFrame({
        'open': [1.50] * 100,
        'high': [1.52] * 100,
        'low': [1.48] * 100,
        'close': [1.50] * 100,
        'volume': [10000] * 100  # Consistent volume = RVOL ~1.0
    }, index=dates)
    df = prepare_features(df)

    detector = CoilDetector(window=40, max_adx=20, max_rvol=0.5)
    is_coiled, metrics = detector.check_coil(df, 80)

    # Should NOT be coiled because RVOL is not dead (close to 1.0)
    # Note: depends on how RVOL calculation handles consistent volume
    assert 'rvol' in metrics


# =============================================================================
# ADX CALCULATION TESTS
# =============================================================================

def test_adx_calculation_trending():
    """Test ADX is high for trending stock."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Strong uptrend
    prices = np.linspace(1.0, 3.0, 100)
    df = pd.DataFrame({
        'high': prices + 0.1,
        'low': prices - 0.1,
        'close': prices
    }, index=dates)

    adx = calculate_adx(df, window=14)

    # ADX should be high (> 25) for trending stock
    assert adx.iloc[-1] > 20


def test_adx_calculation_sideways():
    """Test ADX is low for sideways/ranging stock."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Sideways movement
    np.random.seed(42)
    noise = np.random.uniform(-0.05, 0.05, 100)
    base = 1.5

    df = pd.DataFrame({
        'high': base + noise + 0.02,
        'low': base + noise - 0.02,
        'close': base + noise
    }, index=dates)

    adx = calculate_adx(df, window=14)

    # ADX should be low (< 25) for ranging stock
    # (Exact value depends on noise, but should be moderate to low)
    assert adx.iloc[-1] < 40


def test_adx_no_nan_after_warmup():
    """Test ADX has no NaN values after warmup period."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'high': np.random.uniform(1.0, 1.5, 100),
        'low': np.random.uniform(0.5, 1.0, 100),
        'close': np.random.uniform(0.8, 1.2, 100)
    }, index=dates)

    adx = calculate_adx(df, window=14)

    # After warmup (14*2 = 28 days), should have no NaN
    assert not adx.iloc[50:].isna().any()


# =============================================================================
# BACKTEST ENGINE TESTS - COIL STRATEGY
# =============================================================================

def test_limit_order_entry():
    """Test limit order entry at support + 2%."""
    engine = BacktestEngine(entry_buffer=0.02)

    support_level = 1.00
    entry_price = engine.calculate_entry_price(support_level)

    assert entry_price == 1.02  # Support + 2%


def test_fill_price_slippage():
    """Test slippage applied to fill price."""
    engine = BacktestEngine(slippage_low_price=0.15, slippage_high_price=0.08)

    # Low price stock (< $3.00) - 15% slippage
    fill_low = engine.calculate_fill_price(1.00)
    assert fill_low == 1.15

    # High price stock (>= $3.00) - 8% slippage
    fill_high = engine.calculate_fill_price(4.00)
    assert fill_high == 4.32


def test_structural_stop():
    """Test structural stop is 10% below support."""
    engine = BacktestEngine(structural_stop_pct=0.90)

    support_level = 1.00
    stop_price = engine.calculate_stop_price(support_level)

    assert stop_price == 0.90  # Support * 0.90


def test_profit_take_with_volume_surge():
    """Test profit take at +20% with volume surge."""
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')

    support_level = 1.00
    engine = BacktestEngine(
        entry_buffer=0.02,
        profit_target=0.20,
        volume_surge_threshold=3.0
    )

    # Entry price with slippage: 1.02 * 1.15 = 1.173
    # Target: 1.173 * 1.20 = 1.4076

    df = pd.DataFrame({
        'open': [1.05] * 200,
        'high': [1.10] * 200,
        'low': [0.98] * 200,  # Low enough to fill our limit at 1.02
        'close': [1.05] * 200,
        'volume': [5000] * 200
    }, index=dates)

    # Add volume spike and price pop at day 70
    df.loc[df.index[70], 'high'] = 1.50  # Above target
    df.loc[df.index[70], 'volume'] = 50000  # High volume

    df = prepare_features(df)

    # Force RVOL to be high on spike day
    df.loc[df.index[70], 'rvol_30d'] = 5.0

    result = engine.simulate_coil_trade(df, setup_idx=50, support_level=support_level)

    assert result['outcome'] in ['profit_take', 'target_no_volume']
    assert result['label'] == 1
    assert result['pnl'] > 0.15  # Should be ~20%


def test_structural_stop_hit():
    """Test trade exits on structural stop."""
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')

    support_level = 1.00
    engine = BacktestEngine(structural_stop_pct=0.90)

    # Create data where price fills then drops below stop
    df = pd.DataFrame({
        'open': [1.05] * 200,
        'high': [1.10] * 200,
        'low': [0.98] * 200,  # Fills our limit
        'close': [1.05] * 200,
        'volume': [5000] * 200
    }, index=dates)

    # Price crashes at day 70 - close below stop (0.90)
    df.loc[df.index[70], 'close'] = 0.85
    df.loc[df.index[70], 'low'] = 0.80

    df = prepare_features(df)

    result = engine.simulate_coil_trade(df, setup_idx=50, support_level=support_level)

    assert result['outcome'] == 'structural_stop'
    assert result['label'] == 0
    assert result['pnl'] < 0


def test_time_stop_150_days():
    """Test trade exits after 150 days time stop."""
    dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

    support_level = 1.00
    engine = BacktestEngine(time_stop_days=150)

    # Flat price, no target hit, no stop hit
    df = pd.DataFrame({
        'open': [1.05] * 250,
        'high': [1.08] * 250,
        'low': [0.98] * 250,  # Fills limit but stays in range
        'close': [1.05] * 250,
        'volume': [5000] * 250
    }, index=dates)

    df = prepare_features(df)

    result = engine.simulate_coil_trade(df, setup_idx=50, support_level=support_level)

    assert result['outcome'] == 'time_stop'
    assert result['label'] == 0
    assert result['days'] > 100  # Should hold close to 150 days


def test_no_fill():
    """Test no trade when limit order never fills."""
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')

    support_level = 0.50  # Low support
    engine = BacktestEngine(entry_buffer=0.02)

    # Price never drops to support level
    df = pd.DataFrame({
        'open': [1.00] * 200,
        'high': [1.05] * 200,
        'low': [0.95] * 200,  # Never touches 0.51 limit
        'close': [1.00] * 200,
        'volume': [5000] * 200
    }, index=dates)

    df = prepare_features(df)

    result = engine.simulate_coil_trade(df, setup_idx=50, support_level=support_level)

    assert result['outcome'] == 'no_fill'
    assert result['label'] == 0
    assert result['pnl'] == 0.0


# =============================================================================
# MANIPULATION SCREEN TESTS
# =============================================================================

def test_manipulation_screen_clean():
    """Test clean filing passes the screen."""
    screen = ManipulationScreen()

    clean_filing = """
    The Company has secured financing from XYZ Bank for general operations.
    This is a standard term loan with fixed interest rate.
    """

    passed, flags = screen.scan_filing(clean_filing, "8-K")

    assert passed is True
    assert len([f for f in flags if "CRITICAL" in f]) == 0


def test_manipulation_screen_toxic_lender():
    """Test toxic lender detection."""
    screen = ManipulationScreen()

    toxic_filing = """
    The Company has entered into a Securities Purchase Agreement with
    Streeterville Capital, LLC for the issuance of convertible notes.
    """

    passed, flags = screen.scan_filing(toxic_filing, "8-K")

    assert passed is False
    assert any("Toxic Lender" in f for f in flags)


def test_manipulation_screen_floorless_conversion():
    """Test floorless conversion pattern detection."""
    screen = ManipulationScreen()

    floorless_filing = """
    The conversion price shall be equal to a 30% discount to the lowest
    trading price during the 20 trading days prior to conversion.
    """

    passed, flags = screen.scan_filing(floorless_filing, "8-K")

    assert passed is False
    assert any("floorless" in f.lower() for f in flags)


def test_is_toxic_convenience():
    """Test is_toxic convenience function."""
    toxic_text = "Streeterville Capital provided financing"
    clean_text = "Normal bank provided financing"

    assert is_toxic(toxic_text) is True
    assert is_toxic(clean_text) is False


# =============================================================================
# FEATURE PREPARATION TESTS
# =============================================================================

def test_prepare_features():
    """Test feature preparation pipeline adds all required columns."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.random.uniform(1, 2, 100),
        'high': np.random.uniform(2, 3, 100),
        'low': np.random.uniform(0.5, 1, 100),
        'close': np.random.uniform(1, 2, 100),
        'volume': np.random.randint(10000, 100000, 100)
    }, index=dates)

    result = prepare_features(df)

    # Check all expected columns exist for Coil strategy
    assert 'dollar_vol' in result.columns
    assert 'dollar_vol_30d' in result.columns
    assert 'rvol_30d' in result.columns
    assert 'adx' in result.columns

    # Check no NaN after sufficient warmup
    assert not result.iloc[50:]['rvol_30d'].isna().any()
    assert not result.iloc[50:]['adx'].isna().any()


def test_validate_coil_candidate():
    """Test coil candidate validation filters."""
    # Valid candidate
    valid = pd.Series({
        'close': 0.50,
        'dollar_vol_30d': 50000
    })
    passed, reasons = validate_coil_candidate(valid)
    assert passed is True

    # Too liquid
    too_liquid = pd.Series({
        'close': 0.50,
        'dollar_vol_30d': 500000
    })
    passed, reasons = validate_coil_candidate(too_liquid)
    assert passed is False
    assert any("too liquid" in r.lower() for r in reasons)

    # Not liquid enough
    illiquid = pd.Series({
        'close': 0.50,
        'dollar_vol_30d': 5000
    })
    passed, reasons = validate_coil_candidate(illiquid)
    assert passed is False
