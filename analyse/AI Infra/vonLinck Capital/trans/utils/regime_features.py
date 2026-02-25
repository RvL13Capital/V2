"""
Regime Features Calculation Module
===================================

Provides calculation functions for regime-aware features that help the model
generalize across different market conditions (bull, bear, high-vol, low-vol).

Features:
    1. vix_regime_level: Current volatility regime (0=calm, 1=crisis)
    2. vix_trend_20d: VIX 20-day change direction
    3. market_breadth_200: % of stocks above 200 SMA
    4. risk_on_indicator: Risk appetite composite (0=risk-off, 1=risk-on)
    5. days_since_regime_change: Time in current bull/bear regime

Usage:
    from utils.regime_features import RegimeFeatureCalculator

    calculator = RegimeFeatureCalculator()
    calculator.load_regime_data('data/market_regime/regime_features.parquet')

    # Get features for a specific date
    features = calculator.get_features(pattern_end_date)

Jan 2026 - Created for cross-regime generalization
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# INDIVIDUAL FEATURE CALCULATIONS
# =============================================================================

def vix_regime_level(vix_value: float) -> float:
    """
    Normalize VIX to regime level (0-1 scale).

    Interpretation:
        0.0-0.25: Low volatility regime (VIX < 15)
        0.25-0.50: Normal volatility regime (VIX 15-20)
        0.50-0.75: Elevated volatility regime (VIX 20-30)
        0.75-1.00: High volatility / Crisis regime (VIX > 30)

    Args:
        vix_value: Current VIX close value

    Returns:
        Normalized VIX regime level (0-1)

    Example:
        >>> vix_regime_level(12)
        0.05
        >>> vix_regime_level(25)
        0.375
        >>> vix_regime_level(40)
        0.75
    """
    if pd.isna(vix_value) or vix_value <= 0:
        return 0.25  # Default to low-vol

    return float(np.clip((vix_value - 10) / 40, 0.0, 1.0))


def vix_trend_20d(vix_current: float, vix_20d_ago: float) -> float:
    """
    VIX change over 20 days, normalized.

    Interpretation:
        < -0.2: Volatility collapsing (bullish for breakouts)
        -0.2 to 0.2: Stable volatility
        > 0.2: Volatility expanding (bearish for breakouts)

    Args:
        vix_current: Current VIX value
        vix_20d_ago: VIX value 20 trading days ago

    Returns:
        Normalized VIX trend (-0.5 to +0.5)

    Example:
        >>> vix_trend_20d(15, 20)  # VIX dropped
        -0.25
        >>> vix_trend_20d(25, 15)  # VIX rose
        0.5  # Clipped to max
    """
    if pd.isna(vix_current) or pd.isna(vix_20d_ago) or vix_20d_ago <= 0:
        return 0.0  # Stable default

    change = (vix_current / vix_20d_ago) - 1.0
    return float(np.clip(change, -0.5, 0.5))


def market_breadth_200(pct_above_sma200: float) -> float:
    """
    Normalize market breadth to 0-1 scale.

    Interpretation:
        < 0.30: Extremely narrow market / bearish (< 30% above SMA200)
        0.30-0.50: Weak breadth
        0.50-0.70: Healthy breadth
        > 0.70: Strong breadth / potential overextension

    Args:
        pct_above_sma200: Percentage of stocks above 200 SMA (0-100)

    Returns:
        Normalized breadth (0-1)

    Example:
        >>> market_breadth_200(65)
        0.65
    """
    if pd.isna(pct_above_sma200):
        return 0.5  # Neutral default

    return float(np.clip(pct_above_sma200 / 100.0, 0.0, 1.0))


def risk_on_indicator(
    spy_return_20d: float,
    tlt_return_20d: float,
    jnk_return_20d: float,
    gld_return_20d: float
) -> float:
    """
    Risk-On / Risk-Off composite indicator.

    Logic:
        Risk-On (positive score):
            - Stocks outperforming bonds (SPY > TLT)
            - High-yield outperforming (credit spreads tightening)
            - Gold underperforming (no flight to safety)

        Risk-Off (negative score):
            - Bonds outperforming stocks
            - High-yield underperforming (credit spreads widening)
            - Gold outperforming (flight to safety)

    Args:
        spy_return_20d: S&P 500 20-day return
        tlt_return_20d: Long-term treasury 20-day return
        jnk_return_20d: High-yield bond 20-day return
        gld_return_20d: Gold 20-day return

    Returns:
        Risk-on indicator (0-1, where 0.5 is neutral)

    Example:
        >>> risk_on_indicator(0.05, -0.02, 0.03, -0.01)
        0.75  # Risk-on
        >>> risk_on_indicator(-0.05, 0.03, -0.02, 0.04)
        0.25  # Risk-off
    """
    # Handle missing values
    if any(pd.isna(x) for x in [spy_return_20d, tlt_return_20d, jnk_return_20d, gld_return_20d]):
        return 0.5  # Neutral default

    equity_vs_bonds = spy_return_20d - tlt_return_20d
    credit_strength = jnk_return_20d - tlt_return_20d
    gold_safe_haven = gld_return_20d - spy_return_20d

    score = equity_vs_bonds + credit_strength - gold_safe_haven
    normalized = np.clip(score, -0.2, 0.2) / 0.4 + 0.5

    return float(normalized)


def days_since_regime_change(
    current_date: pd.Timestamp,
    last_crossover_date: pd.Timestamp
) -> float:
    """
    Log-normalized days since last MA crossover.

    Interpretation:
        < 0.4: Early in regime (< 30 days) - uncertain
        0.4-0.6: Establishing (30-100 days)
        0.6-0.8: Mature regime (100-250 days)
        > 0.8: Extended regime (> 250 days) - potential for reversal

    Args:
        current_date: Current pattern date
        last_crossover_date: Date of last bull/bear crossover

    Returns:
        Log-normalized days (0-1)

    Example:
        >>> days_since_regime_change(pd.Timestamp('2024-06-01'), pd.Timestamp('2024-01-01'))
        0.58  # ~150 days, mature regime
    """
    if pd.isna(current_date) or pd.isna(last_crossover_date):
        return 0.5  # Neutral default

    days = (current_date - last_crossover_date).days
    if days < 0:
        days = 0

    return float(np.log1p(days) / np.log1p(500))


# =============================================================================
# REGIME FEATURE CALCULATOR CLASS
# =============================================================================

class RegimeFeatureCalculator:
    """
    Calculator for regime-aware features.

    Loads pre-computed regime data and provides features for any date.
    Handles missing data gracefully with sensible defaults.

    Usage:
        calculator = RegimeFeatureCalculator()
        calculator.load_regime_data('data/market_regime/regime_features.parquet')
        calculator.load_phase_data('data/market_regime/USA500IDXUSD_phases.csv')

        features = calculator.get_features(pattern_end_date)
        # Returns: {
        #     'vix_regime_level': 0.25,
        #     'vix_trend_20d': 0.05,
        #     'market_breadth_200': 0.65,
        #     'risk_on_indicator': 0.55,
        #     'days_since_regime_change': 0.48
        # }
    """

    def __init__(self):
        """Initialize empty calculator."""
        self.regime_data: Optional[pd.DataFrame] = None
        self.phase_data: Optional[pd.DataFrame] = None
        self.breadth_data: Optional[pd.DataFrame] = None
        self._loaded = False

    def load_regime_data(self, filepath: str) -> bool:
        """
        Load pre-computed regime features (VIX, risk-on).

        Args:
            filepath: Path to regime_features.parquet

        Returns:
            True if loaded successfully
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Regime data not found: {filepath}")
            return False

        try:
            self.regime_data = pd.read_parquet(path)
            self.regime_data['date'] = pd.to_datetime(self.regime_data['date'])
            self.regime_data = self.regime_data.set_index('date').sort_index()
            logger.info(f"Loaded regime data: {len(self.regime_data)} records")
            self._loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load regime data: {e}")
            return False

    def load_phase_data(self, filepath: str) -> bool:
        """
        Load market phase data for days_since_regime_change.

        Args:
            filepath: Path to USA500IDXUSD_phases.csv

        Returns:
            True if loaded successfully
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Phase data not found: {filepath}")
            return False

        try:
            self.phase_data = pd.read_csv(path)
            self.phase_data['date'] = pd.to_datetime(self.phase_data['date'])
            if 'crossover_date' in self.phase_data.columns:
                self.phase_data['crossover_date'] = pd.to_datetime(self.phase_data['crossover_date'])
            self.phase_data = self.phase_data.set_index('date').sort_index()
            logger.info(f"Loaded phase data: {len(self.phase_data)} records")
            return True
        except Exception as e:
            logger.error(f"Failed to load phase data: {e}")
            return False

    def load_breadth_data(self, filepath: str) -> bool:
        """
        Load market breadth data.

        Args:
            filepath: Path to breadth data parquet

        Returns:
            True if loaded successfully
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Breadth data not found: {filepath}")
            return False

        try:
            self.breadth_data = pd.read_parquet(path)
            self.breadth_data['date'] = pd.to_datetime(self.breadth_data['date'])
            self.breadth_data = self.breadth_data.set_index('date').sort_index()
            logger.info(f"Loaded breadth data: {len(self.breadth_data)} records")
            return True
        except Exception as e:
            logger.error(f"Failed to load breadth data: {e}")
            return False

    def get_features(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Get all regime features for a specific date.

        Args:
            date: Date to get features for (pattern_end_date)

        Returns:
            Dictionary with all 5 regime features
        """
        date = pd.to_datetime(date)

        # Default values (neutral/stable)
        features = {
            'vix_regime_level': 0.25,      # Low volatility default
            'vix_trend_20d': 0.0,          # Stable
            'market_breadth_200': 0.5,     # Neutral
            'risk_on_indicator': 0.5,      # Neutral
            'days_since_regime_change': 0.5  # Mid-regime
        }

        # Get VIX and risk-on features from regime data
        if self.regime_data is not None and len(self.regime_data) > 0:
            # Find closest date (use as-of date lookup)
            try:
                idx = self.regime_data.index.get_indexer([date], method='ffill')[0]
                if idx >= 0:
                    row = self.regime_data.iloc[idx]
                    if 'vix_regime_level' in row:
                        features['vix_regime_level'] = float(row['vix_regime_level'])
                    if 'vix_trend_20d' in row:
                        features['vix_trend_20d'] = float(row['vix_trend_20d'])
                    if 'risk_on_indicator' in row:
                        features['risk_on_indicator'] = float(row['risk_on_indicator'])
            except Exception:
                pass

        # Get breadth from breadth data
        if self.breadth_data is not None and len(self.breadth_data) > 0:
            try:
                idx = self.breadth_data.index.get_indexer([date], method='ffill')[0]
                if idx >= 0:
                    row = self.breadth_data.iloc[idx]
                    if 'pct_above_sma200' in row:
                        features['market_breadth_200'] = market_breadth_200(row['pct_above_sma200'])
                    elif 'breadth' in row:
                        features['market_breadth_200'] = market_breadth_200(row['breadth'])
            except Exception:
                pass

        # Get days since regime change from phase data
        if self.phase_data is not None and len(self.phase_data) > 0:
            try:
                idx = self.phase_data.index.get_indexer([date], method='ffill')[0]
                if idx >= 0:
                    row = self.phase_data.iloc[idx]
                    if 'crossover_date' in row and pd.notna(row['crossover_date']):
                        features['days_since_regime_change'] = days_since_regime_change(
                            date, row['crossover_date']
                        )
            except Exception:
                pass

        return features

    def get_features_batch(self, dates: pd.Series) -> pd.DataFrame:
        """
        Get regime features for multiple dates efficiently.

        Args:
            dates: Series of dates

        Returns:
            DataFrame with regime features for each date
        """
        results = []
        for date in dates:
            features = self.get_features(date)
            features['date'] = date
            results.append(features)

        return pd.DataFrame(results)

    @property
    def is_loaded(self) -> bool:
        """Check if any data has been loaded."""
        return self._loaded


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_default_regime_features() -> Dict[str, float]:
    """
    Get default regime features (neutral values).

    Use when regime data is unavailable.

    Returns:
        Dictionary with neutral regime features
    """
    return {
        'vix_regime_level': 0.25,      # Low volatility
        'vix_trend_20d': 0.0,          # Stable
        'market_breadth_200': 0.5,     # Neutral breadth
        'risk_on_indicator': 0.5,      # Neutral risk
        'days_since_regime_change': 0.5  # Mid-regime
    }


def create_regime_calculator(
    regime_path: str = None,
    phase_path: str = None,
    breadth_path: str = None
) -> RegimeFeatureCalculator:
    """
    Factory function to create and load regime calculator.

    Args:
        regime_path: Path to regime_features.parquet
        phase_path: Path to market phase CSV
        breadth_path: Path to breadth data parquet

    Returns:
        Configured RegimeFeatureCalculator
    """
    calculator = RegimeFeatureCalculator()

    # Default paths
    base_dir = Path(__file__).parent.parent / 'data' / 'market_regime'

    if regime_path is None:
        regime_path = base_dir / 'regime_features.parquet'
    if phase_path is None:
        phase_path = base_dir / 'USA500IDXUSD_phases.csv'
    if breadth_path is None:
        breadth_path = base_dir / 'breadth_daily.parquet'

    calculator.load_regime_data(str(regime_path))
    calculator.load_phase_data(str(phase_path))
    calculator.load_breadth_data(str(breadth_path))

    return calculator


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Regime Features Module Test")
    print("=" * 50)

    # Test individual functions
    print("\n1. vix_regime_level:")
    for vix in [10, 15, 20, 25, 30, 40]:
        print(f"   VIX={vix}: {vix_regime_level(vix):.3f}")

    print("\n2. vix_trend_20d:")
    for current, ago in [(15, 20), (20, 20), (25, 15), (30, 15)]:
        print(f"   VIX {ago}->{current}: {vix_trend_20d(current, ago):.3f}")

    print("\n3. risk_on_indicator:")
    print(f"   Risk-On  (+5%, -2%, +3%, -1%): {risk_on_indicator(0.05, -0.02, 0.03, -0.01):.3f}")
    print(f"   Risk-Off (-5%, +3%, -2%, +4%): {risk_on_indicator(-0.05, 0.03, -0.02, 0.04):.3f}")
    print(f"   Neutral  (0%, 0%, 0%, 0%):     {risk_on_indicator(0, 0, 0, 0):.3f}")

    print("\n4. days_since_regime_change:")
    base = pd.Timestamp('2024-06-01')
    for days in [10, 30, 100, 250, 500]:
        crossover = base - timedelta(days=days)
        print(f"   {days} days: {days_since_regime_change(base, crossover):.3f}")

    print("\n5. Default features:")
    print(f"   {get_default_regime_features()}")
