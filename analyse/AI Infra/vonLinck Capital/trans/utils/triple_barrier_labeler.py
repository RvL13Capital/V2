"""
Triple Barrier Labeling Method (Marcos Lopez de Prado)
======================================================

Implements the Triple Barrier Method from "Advances in Financial Machine Learning"
for labeling trading patterns with temporal integrity guarantees.

The Three Barriers:
    1. Upper Barrier (Profit-Taking): Relative return >= +threshold
    2. Lower Barrier (Stop-Loss): Relative return <= -threshold
    3. Vertical Barrier (Time Limit): Maximum holding period expires

Labels are determined by which barrier is touched FIRST:
    - 2 (Target): Upper barrier hit first (profit-taking)
    - 0 (Danger): Lower barrier hit first (stop-loss)
    - 1 (Noise): Vertical barrier hit (timeout, no clear outcome)

Key Features:
    - Uses RELATIVE returns (percentage change), not absolute prices
    - Barriers can be volatility-scaled (using daily volatility estimate)
    - DYNAMIC BARRIERS based on market cap tier and market regime
    - All rolling calculations use closed='left' for strict temporal integrity
    - Compatible with mlfinlab/triple-barrier libraries conceptually

Dynamic Barrier Adjustment (Jan 2026):
    - MICRO-CAPS: Wider stops (more volatile), higher targets (40%+ potential)
    - SMALL-CAPS: Medium stops/targets
    - MID/LARGE-CAPS: Tighter stops, smaller but consistent targets
    - BULLISH REGIME: Tighter stops, higher targets (risk-on)
    - BEARISH REGIME: Wider stops, lower targets (risk-off)

References:
    - Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"
    - Chapter 3: Labeling

Usage:
    from utils.triple_barrier_labeler import TripleBarrierLabeler

    # Fixed barriers
    labeler = TripleBarrierLabeler(
        vertical_barrier_days=150,
        upper_barrier_pct=0.03,  # +3% profit-taking
        lower_barrier_pct=0.02,  # -2% stop-loss
    )

    # Dynamic barriers (recommended)
    labeler = TripleBarrierLabeler(
        vertical_barrier_days=150,
        use_dynamic_barriers=True,  # Adjusts per market cap + regime
    )

    df_labeled = labeler.label_patterns(patterns_df, price_df)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Union, List, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import Enum
import logging

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_config import setup_pipeline_logging

# Import dynamic barrier configuration
try:
    from config.barrier_config import (
        get_barriers_for_pattern,
        get_dynamic_barriers,
        classify_market_cap,
        map_trend_to_regime,
        map_vix_to_regime,
        MarketCapTier,
        MarketRegimeState,
    )
    DYNAMIC_BARRIERS_AVAILABLE = True
except ImportError:
    DYNAMIC_BARRIERS_AVAILABLE = False

logger = setup_pipeline_logging('triple_barrier_labeler')


# =============================================================================
# CONSTANTS
# =============================================================================

# Labeling window configuration
FEATURE_WINDOW_SIZE = 250      # Days of history for feature calculation
VERTICAL_BARRIER_DAYS = 150    # Maximum holding period for label calculation
MIN_DATA_DAYS = 300            # Minimum data required (feature + outcome window)

# Default barrier thresholds (as decimal, not percentage)
DEFAULT_UPPER_BARRIER = 0.03   # +3% profit-taking threshold
DEFAULT_LOWER_BARRIER = 0.02   # -2% stop-loss threshold (positive value, applied as negative)
DEFAULT_VOLATILITY_MULTIPLIER = 2.0  # For volatility-scaled barriers

# Rolling window configuration
VOLATILITY_LOOKBACK = 20       # Days for volatility estimation
VOLATILITY_ANNUALIZATION = 252 # Trading days per year


class BarrierType(Enum):
    """Types of barriers in Triple Barrier Method."""
    UPPER = "upper"      # Profit-taking barrier
    LOWER = "lower"      # Stop-loss barrier
    VERTICAL = "vertical"  # Time barrier (expiration)


@dataclass
class BarrierConfig:
    """Configuration for Triple Barrier Method."""
    vertical_barrier_days: int = VERTICAL_BARRIER_DAYS
    upper_barrier_pct: float = DEFAULT_UPPER_BARRIER
    lower_barrier_pct: float = DEFAULT_LOWER_BARRIER
    volatility_scaling: bool = False
    volatility_multiplier: float = DEFAULT_VOLATILITY_MULTIPLIER
    volatility_lookback: int = VOLATILITY_LOOKBACK
    min_data_days: int = MIN_DATA_DAYS
    feature_window_size: int = FEATURE_WINDOW_SIZE
    use_dynamic_barriers: bool = False  # If True, adjust barriers per market cap + regime


class TripleBarrierLabeler:
    """
    Triple Barrier Method Labeler for Pattern Outcomes.

    Implements Marcos Lopez de Prado's labeling methodology with
    strict temporal integrity guarantees.

    Key Principles:
        1. All features use rolling(window, closed='left') - strictly historic
        2. Labels are calculated using relative returns (percentage change)
        3. Barriers can be fixed, volatility-scaled, or DYNAMIC (market cap + regime)
        4. No information from after the vertical barrier is used

    Dynamic Barriers (use_dynamic_barriers=True):
        - Micro-caps get wider stops (+4%) and higher targets (+8%)
        - Small-caps get medium stops (+2.5%) and targets (+5%)
        - Bullish regime tightens stops, extends targets
        - Bearish regime widens stops, reduces targets

    Attributes:
        config: BarrierConfig with labeling parameters
    """

    def __init__(
        self,
        vertical_barrier_days: int = VERTICAL_BARRIER_DAYS,
        upper_barrier_pct: float = DEFAULT_UPPER_BARRIER,
        lower_barrier_pct: float = DEFAULT_LOWER_BARRIER,
        volatility_scaling: bool = False,
        volatility_multiplier: float = DEFAULT_VOLATILITY_MULTIPLIER,
        volatility_lookback: int = VOLATILITY_LOOKBACK,
        feature_window_size: int = FEATURE_WINDOW_SIZE,
        min_data_days: int = MIN_DATA_DAYS,
        use_dynamic_barriers: bool = False
    ):
        """
        Initialize Triple Barrier Labeler.

        Args:
            vertical_barrier_days: Maximum holding period (default: 150 days)
            upper_barrier_pct: Profit-taking threshold as decimal (default: 0.03 = 3%)
                              Ignored if use_dynamic_barriers=True
            lower_barrier_pct: Stop-loss threshold as decimal (default: 0.02 = 2%)
                              Ignored if use_dynamic_barriers=True
            volatility_scaling: If True, scale barriers by realized volatility
            volatility_multiplier: Multiplier for volatility-scaled barriers
            volatility_lookback: Days for volatility estimation (default: 20)
            feature_window_size: Days of history for features (default: 250)
            min_data_days: Minimum required data days (default: 300)
            use_dynamic_barriers: If True, calculate barriers dynamically based on
                                  market cap tier and market regime at pattern time.
                                  This overrides upper_barrier_pct and lower_barrier_pct.
        """
        # Check if dynamic barriers are available
        if use_dynamic_barriers and not DYNAMIC_BARRIERS_AVAILABLE:
            logger.warning("Dynamic barriers requested but config.barrier_config not available. Using fixed barriers.")
            use_dynamic_barriers = False

        self.config = BarrierConfig(
            vertical_barrier_days=vertical_barrier_days,
            upper_barrier_pct=upper_barrier_pct,
            lower_barrier_pct=lower_barrier_pct,
            volatility_scaling=volatility_scaling,
            volatility_multiplier=volatility_multiplier,
            volatility_lookback=volatility_lookback,
            feature_window_size=feature_window_size,
            min_data_days=min_data_days,
            use_dynamic_barriers=use_dynamic_barriers
        )

        logger.info(f"TripleBarrierLabeler initialized:")
        logger.info(f"  Vertical barrier: {vertical_barrier_days} days")
        if use_dynamic_barriers:
            logger.info(f"  DYNAMIC BARRIERS: Enabled (market cap + regime adjusted)")
            logger.info(f"    - Micro-caps: +8% target, -4% stop (base)")
            logger.info(f"    - Small-caps: +5% target, -2.5% stop (base)")
            logger.info(f"    - Adjusts for bullish/bearish regime")
        else:
            logger.info(f"  Upper barrier: +{upper_barrier_pct*100:.1f}%")
            logger.info(f"  Lower barrier: -{lower_barrier_pct*100:.1f}%")
        logger.info(f"  Volatility scaling: {volatility_scaling}")
        logger.info(f"  Feature window: {feature_window_size} days")

    def calculate_daily_volatility(
        self,
        price_df: pd.DataFrame,
        date: datetime,
        price_col: str = 'close'
    ) -> float:
        """
        Calculate realized daily volatility using STRICTLY HISTORIC data.

        Uses rolling window with closed='left' to ensure no look-ahead bias.

        Args:
            price_df: DataFrame with price data (DatetimeIndex)
            date: Reference date for volatility calculation
            price_col: Column to use for prices

        Returns:
            Daily volatility (standard deviation of log returns)
        """
        # Get data STRICTLY BEFORE the reference date
        # closed='left' means the window excludes the right endpoint
        hist_data = price_df[price_df.index < date].tail(self.config.volatility_lookback)

        if len(hist_data) < 5:  # Minimum for meaningful volatility
            return np.nan

        # Calculate log returns
        log_returns = np.log(hist_data[price_col] / hist_data[price_col].shift(1))

        # Daily volatility (standard deviation of log returns)
        daily_vol = log_returns.std()

        return daily_vol

    def get_barriers(
        self,
        price_df: pd.DataFrame,
        entry_date: datetime,
        entry_price: float,
        price_col: str = 'close',
        volume_col: str = 'volume',
        ticker: str = None,
        market_cap: Optional[float] = None,
        vix: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate the three barriers for a pattern.

        If use_dynamic_barriers=True, barriers are adjusted based on:
        - Market cap tier (nano/micro/small/mid/large) → wider or tighter barriers
        - Market regime (bullish/bearish/crisis) → risk tolerance adjustment

        Args:
            price_df: DataFrame with price data (DatetimeIndex)
            entry_date: Pattern end date (entry point)
            entry_price: Price at entry (typically close on entry_date)
            price_col: Column to use for prices
            volume_col: Column to use for volume (for ADV calculation)
            ticker: Ticker symbol (for logging)
            market_cap: Market cap at pattern time (if known)
            vix: VIX level at pattern time (for regime detection)

        Returns:
            Dictionary with:
                - upper_barrier_pct: Upper barrier as return threshold
                - lower_barrier_pct: Lower barrier as return threshold
                - vertical_barrier_date: Maximum holding date
                - daily_volatility: Volatility used for scaling (if applicable)
                - market_cap_tier: Market cap classification (if dynamic)
                - regime: Market regime (if dynamic)
        """
        # Vertical barrier: entry_date + vertical_barrier_days
        vertical_date = entry_date + timedelta(days=self.config.vertical_barrier_days)

        # Calculate daily volatility (used for both dynamic and vol-scaling)
        daily_vol = self.calculate_daily_volatility(price_df, entry_date, price_col)

        # Dynamic barriers mode (Jan 2026)
        if self.config.use_dynamic_barriers and DYNAMIC_BARRIERS_AVAILABLE:
            # Use the comprehensive barrier calculation from barrier_config
            dynamic_barriers = get_barriers_for_pattern(
                ticker=ticker or 'UNKNOWN',
                pattern_date=pd.Timestamp(entry_date),
                price_df=price_df,
                market_cap=market_cap,
                vix=vix,
                price_col=price_col,
                volume_col=volume_col
            )

            upper_barrier = dynamic_barriers['upper_barrier_pct']
            lower_barrier = -dynamic_barriers['lower_barrier_pct']  # Negative for stop

            return {
                'upper_barrier_pct': upper_barrier,
                'lower_barrier_pct': lower_barrier,
                'vertical_barrier_date': vertical_date,
                'daily_volatility': daily_vol,
                'market_cap_tier': dynamic_barriers.get('market_cap_tier', 'unknown'),
                'regime': dynamic_barriers.get('regime', 'neutral'),
                'barrier_rationale': dynamic_barriers.get('rationale', ''),
                'adv_calculated': dynamic_barriers.get('adv_calculated'),
            }

        # Volatility scaling mode
        if self.config.volatility_scaling:
            if np.isnan(daily_vol) or daily_vol <= 0:
                # Fallback to fixed barriers
                upper_barrier = self.config.upper_barrier_pct
                lower_barrier = -self.config.lower_barrier_pct
            else:
                # Scale barriers by volatility
                upper_barrier = daily_vol * self.config.volatility_multiplier
                lower_barrier = -daily_vol * self.config.volatility_multiplier
        else:
            # Fixed barriers
            upper_barrier = self.config.upper_barrier_pct
            lower_barrier = -self.config.lower_barrier_pct

        return {
            'upper_barrier_pct': upper_barrier,
            'lower_barrier_pct': lower_barrier,
            'vertical_barrier_date': vertical_date,
            'daily_volatility': daily_vol
        }

    def label_single_pattern(
        self,
        price_df: pd.DataFrame,
        entry_date: datetime,
        entry_price: float,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        volume_col: str = 'volume',
        ticker: str = None,
        market_cap: Optional[float] = None,
        vix: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Label a single pattern using Triple Barrier Method.

        Uses relative returns and checks which barrier is hit FIRST.
        If use_dynamic_barriers=True, barriers are adjusted based on market cap and regime.

        Args:
            price_df: DataFrame with OHLCV data (DatetimeIndex)
            entry_date: Pattern end date (entry point)
            entry_price: Price at entry
            price_col: Column for close prices
            high_col: Column for high prices (for intraday barrier checks)
            low_col: Column for low prices (for intraday barrier checks)
            volume_col: Column for volume (for ADV calculation in dynamic mode)
            ticker: Ticker symbol (for logging/metadata)
            market_cap: Market cap at pattern time (for dynamic barriers)
            vix: VIX level at pattern time (for regime detection)

        Returns:
            Dictionary with:
                - label: 0 (Danger), 1 (Noise), 2 (Target)
                - barrier_hit: Which barrier was hit first
                - barrier_hit_date: Date when barrier was hit
                - barrier_hit_day: Day number when barrier was hit (1-indexed)
                - max_return: Maximum return achieved
                - min_return: Minimum return achieved
                - final_return: Return at vertical barrier (if hit)
                - barriers: Barrier configuration used
                - market_cap_tier: (if dynamic) Market cap classification
                - regime: (if dynamic) Market regime at pattern time
        """
        # Get barriers for this pattern (may be dynamic based on market cap + regime)
        barriers = self.get_barriers(
            price_df, entry_date, entry_price, price_col,
            volume_col=volume_col,
            ticker=ticker,
            market_cap=market_cap,
            vix=vix
        )
        upper_barrier = barriers['upper_barrier_pct']
        lower_barrier = barriers['lower_barrier_pct']
        vertical_date = barriers['vertical_barrier_date']

        # Get outcome window data (after entry_date, up to vertical barrier)
        outcome_mask = (price_df.index > entry_date) & (price_df.index <= vertical_date)
        outcome_data = price_df[outcome_mask].copy()

        if len(outcome_data) == 0:
            # No data in outcome window
            return {
                'label': None,
                'barrier_hit': None,
                'barrier_hit_date': None,
                'barrier_hit_day': None,
                'max_return': None,
                'min_return': None,
                'final_return': None,
                'barriers': barriers,
                'error': 'no_outcome_data'
            }

        # Calculate relative returns from entry price
        # Using high/low for intraday barrier checks
        outcome_data['return_high'] = (outcome_data[high_col] - entry_price) / entry_price
        outcome_data['return_low'] = (outcome_data[low_col] - entry_price) / entry_price
        outcome_data['return_close'] = (outcome_data[price_col] - entry_price) / entry_price

        max_return = float(outcome_data['return_high'].max())
        min_return = float(outcome_data['return_low'].min())
        final_return = float(outcome_data['return_close'].iloc[-1]) if len(outcome_data) > 0 else None

        # Find first barrier hit
        # Upper barrier: high return >= upper_barrier
        upper_hit_mask = outcome_data['return_high'] >= upper_barrier
        # Lower barrier: low return <= lower_barrier (lower_barrier is negative)
        lower_hit_mask = outcome_data['return_low'] <= lower_barrier

        # Get first occurrence of each
        first_upper_date = outcome_data[upper_hit_mask].index[0] if upper_hit_mask.any() else None
        first_lower_date = outcome_data[lower_hit_mask].index[0] if lower_hit_mask.any() else None

        # Determine which barrier was hit first
        label = None
        barrier_hit = None
        barrier_hit_date = None
        barrier_hit_day = None

        if first_upper_date is not None and first_lower_date is not None:
            # Both barriers hit - check which was first
            if first_upper_date < first_lower_date:
                label = 2  # Target (upper hit first)
                barrier_hit = BarrierType.UPPER.value
                barrier_hit_date = first_upper_date
            elif first_lower_date < first_upper_date:
                label = 0  # Danger (lower hit first)
                barrier_hit = BarrierType.LOWER.value
                barrier_hit_date = first_lower_date
            else:
                # Same day - pessimistic assumption: lower hit first
                label = 0  # Danger
                barrier_hit = BarrierType.LOWER.value
                barrier_hit_date = first_lower_date
                logger.debug(f"Same-day barrier conflict at {first_lower_date}, applying pessimistic label")

        elif first_upper_date is not None:
            # Only upper barrier hit
            label = 2  # Target
            barrier_hit = BarrierType.UPPER.value
            barrier_hit_date = first_upper_date

        elif first_lower_date is not None:
            # Only lower barrier hit
            label = 0  # Danger
            barrier_hit = BarrierType.LOWER.value
            barrier_hit_date = first_lower_date

        else:
            # Neither barrier hit - vertical barrier (timeout)
            label = 1  # Noise
            barrier_hit = BarrierType.VERTICAL.value
            barrier_hit_date = outcome_data.index[-1]

        # Calculate barrier_hit_day (1-indexed)
        if barrier_hit_date is not None:
            barrier_hit_day = len(outcome_data[outcome_data.index <= barrier_hit_date])

        return {
            'label': label,
            'barrier_hit': barrier_hit,
            'barrier_hit_date': barrier_hit_date,
            'barrier_hit_day': barrier_hit_day,
            'max_return': max_return,
            'min_return': min_return,
            'final_return': final_return,
            'barriers': barriers,
            'error': None
        }

    def calculate_historic_features(
        self,
        price_df: pd.DataFrame,
        reference_date: datetime,
        price_col: str = 'close',
        volume_col: str = 'volume'
    ) -> Dict[str, float]:
        """
        Calculate features using STRICTLY HISTORIC data.

        Uses rolling(window, closed='left') to ensure no look-ahead bias.
        All features are calculated using only data BEFORE reference_date.

        Args:
            price_df: DataFrame with OHLCV data (DatetimeIndex)
            reference_date: Reference date (features use data BEFORE this date)
            price_col: Column for close prices
            volume_col: Column for volume

        Returns:
            Dictionary of feature values
        """
        # Get STRICTLY HISTORIC data (before reference_date)
        hist_data = price_df[price_df.index < reference_date].copy()

        if len(hist_data) < 50:  # Minimum for meaningful features
            return {}

        # Calculate log returns for volatility
        hist_data['log_return'] = np.log(hist_data[price_col] / hist_data[price_col].shift(1))

        features = {}

        # =========================================================================
        # VOLATILITY FEATURES - Using closed='left' for strict temporal integrity
        # =========================================================================

        # 20-day volatility (most recent complete window BEFORE reference)
        vol_20 = hist_data['log_return'].rolling(
            window=20, min_periods=15, closed='left'
        ).std().iloc[-1] if len(hist_data) >= 20 else np.nan
        features['volatility_20d'] = vol_20

        # 50-day volatility
        vol_50 = hist_data['log_return'].rolling(
            window=50, min_periods=30, closed='left'
        ).std().iloc[-1] if len(hist_data) >= 50 else np.nan
        features['volatility_50d'] = vol_50

        # Volatility ratio (short/long)
        if vol_20 and vol_50 and vol_50 > 0:
            features['volatility_ratio'] = vol_20 / vol_50
        else:
            features['volatility_ratio'] = np.nan

        # =========================================================================
        # PRICE FEATURES - Using closed='left'
        # =========================================================================

        # Moving averages
        sma_20 = hist_data[price_col].rolling(
            window=20, min_periods=15, closed='left'
        ).mean().iloc[-1] if len(hist_data) >= 20 else np.nan
        features['sma_20'] = sma_20

        sma_50 = hist_data[price_col].rolling(
            window=50, min_periods=30, closed='left'
        ).mean().iloc[-1] if len(hist_data) >= 50 else np.nan
        features['sma_50'] = sma_50

        sma_200 = hist_data[price_col].rolling(
            window=200, min_periods=150, closed='left'
        ).mean().iloc[-1] if len(hist_data) >= 200 else np.nan
        features['sma_200'] = sma_200

        # Price position relative to SMAs
        current_price = hist_data[price_col].iloc[-1]
        if sma_20 and sma_20 > 0:
            features['price_to_sma20'] = (current_price / sma_20) - 1
        if sma_50 and sma_50 > 0:
            features['price_to_sma50'] = (current_price / sma_50) - 1
        if sma_200 and sma_200 > 0:
            features['price_to_sma200'] = (current_price / sma_200) - 1

        # =========================================================================
        # MOMENTUM FEATURES - Using closed='left'
        # =========================================================================

        # Returns over different periods (calculated from HISTORIC data only)
        if len(hist_data) >= 5:
            features['return_5d'] = (current_price / hist_data[price_col].iloc[-6]) - 1
        if len(hist_data) >= 20:
            features['return_20d'] = (current_price / hist_data[price_col].iloc[-21]) - 1
        if len(hist_data) >= 60:
            features['return_60d'] = (current_price / hist_data[price_col].iloc[-61]) - 1

        # =========================================================================
        # VOLUME FEATURES - Using closed='left'
        # =========================================================================

        if volume_col in hist_data.columns:
            vol_sma_20 = hist_data[volume_col].rolling(
                window=20, min_periods=15, closed='left'
            ).mean().iloc[-1] if len(hist_data) >= 20 else np.nan
            features['volume_sma_20'] = vol_sma_20

            current_vol = hist_data[volume_col].iloc[-1]
            if vol_sma_20 and vol_sma_20 > 0:
                features['volume_ratio'] = current_vol / vol_sma_20

        return features

    def label_patterns(
        self,
        patterns_df: pd.DataFrame,
        price_loader_func,
        ticker_col: str = 'ticker',
        date_col: str = 'end_date',
        reference_date: Optional[datetime] = None,
        calculate_features: bool = True,
        market_cap_col: Optional[str] = None,
        vix_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Label multiple patterns using Triple Barrier Method.

        If use_dynamic_barriers=True, barriers are adjusted per pattern based on:
        - Market cap tier (from market_cap_col or calculated from ADV)
        - Market regime (from vix_col or calculated from price/SMA)

        Args:
            patterns_df: DataFrame with pattern metadata
            price_loader_func: Function(ticker, start_date, end_date) -> DataFrame
                              Returns OHLCV DataFrame with DatetimeIndex
            ticker_col: Column name for ticker symbol
            date_col: Column name for pattern end date
            reference_date: Reference date for ripeness check (default: today)
            calculate_features: If True, also calculate historic features
            market_cap_col: Optional column with market cap at pattern time.
                           If None, ADV-based classification is used.
            vix_col: Optional column with VIX at pattern time.
                    If None, trend-based regime detection is used.

        Returns:
            DataFrame with added labeling columns:
                - label: 0 (Danger), 1 (Noise), 2 (Target)
                - barrier_hit: Which barrier triggered
                - barrier_hit_date: When barrier was hit
                - barrier_hit_day: Day number of barrier hit
                - max_return, min_return, final_return
                - upper_barrier_pct, lower_barrier_pct
                - market_cap_tier: (if dynamic) Classification used
                - regime: (if dynamic) Market regime used
                - labeling_method: 'triple_barrier' or 'triple_barrier_dynamic'
                - labeling_version: 'v24_triple_barrier'
        """
        if reference_date is None:
            reference_date = datetime.now()

        logger.info(f"Labeling {len(patterns_df)} patterns using Triple Barrier Method")
        logger.info(f"Reference date: {reference_date.date()}")

        results = []
        labeled_count = 0
        error_count = 0
        label_counts = {0: 0, 1: 0, 2: 0}

        # Group by ticker for efficiency
        for ticker, ticker_patterns in patterns_df.groupby(ticker_col):
            # Calculate required date range
            min_date = pd.to_datetime(ticker_patterns[date_col].min())
            max_date = pd.to_datetime(ticker_patterns[date_col].max())

            # Need extra history for features and outcome window
            start_date = min_date - timedelta(days=self.config.feature_window_size + 50)
            end_date = max_date + timedelta(days=self.config.vertical_barrier_days + 10)

            # Load price data
            try:
                price_df = price_loader_func(ticker, start_date, end_date)
                if price_df is None or len(price_df) < self.config.min_data_days:
                    logger.warning(f"{ticker}: Insufficient data ({len(price_df) if price_df is not None else 0} days)")
                    for _, row in ticker_patterns.iterrows():
                        result = row.to_dict()
                        result['label'] = None
                        result['error'] = 'insufficient_data'
                        results.append(result)
                        error_count += 1
                    continue

                # Ensure DatetimeIndex
                if not isinstance(price_df.index, pd.DatetimeIndex):
                    if 'date' in price_df.columns:
                        price_df['date'] = pd.to_datetime(price_df['date'])
                        price_df = price_df.set_index('date')
                    else:
                        raise ValueError("Price DataFrame must have DatetimeIndex or 'date' column")

            except Exception as e:
                logger.warning(f"{ticker}: Failed to load price data: {e}")
                for _, row in ticker_patterns.iterrows():
                    result = row.to_dict()
                    result['label'] = None
                    result['error'] = f'data_load_error: {e}'
                    results.append(result)
                    error_count += 1
                continue

            # Use adj_close if available (split-adjusted)
            price_col = 'adj_close' if 'adj_close' in price_df.columns else 'close'

            # Label each pattern for this ticker
            for _, row in ticker_patterns.iterrows():
                result = row.to_dict()
                entry_date = pd.to_datetime(row[date_col])

                # Check if pattern has ripened
                ripe_date = entry_date + timedelta(days=self.config.vertical_barrier_days)
                if ripe_date > reference_date:
                    logger.debug(f"{ticker} {entry_date.date()}: Pattern not yet ripe")
                    result['label'] = None
                    result['error'] = 'not_ripe'
                    results.append(result)
                    error_count += 1
                    continue

                # Check if we have data through outcome window
                max_required_date = entry_date + timedelta(days=self.config.vertical_barrier_days)
                if price_df.index.max() < max_required_date:
                    logger.debug(f"{ticker} {entry_date.date()}: Insufficient future data")
                    result['label'] = None
                    result['error'] = 'insufficient_future_data'
                    results.append(result)
                    error_count += 1
                    continue

                # Get entry price
                if entry_date in price_df.index:
                    entry_price = price_df.loc[entry_date, price_col]
                else:
                    # Find nearest date
                    nearest_idx = price_df.index.get_indexer([entry_date], method='nearest')[0]
                    if nearest_idx < 0:
                        result['label'] = None
                        result['error'] = 'entry_date_not_found'
                        results.append(result)
                        error_count += 1
                        continue
                    entry_price = price_df.iloc[nearest_idx][price_col]

                # Get optional market cap and VIX from pattern row
                pattern_market_cap = row.get(market_cap_col) if market_cap_col and market_cap_col in row else None
                pattern_vix = row.get(vix_col) if vix_col and vix_col in row else None

                # Label the pattern (with dynamic barrier support)
                label_result = self.label_single_pattern(
                    price_df=price_df,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    price_col=price_col,
                    ticker=ticker,
                    market_cap=pattern_market_cap,
                    vix=pattern_vix
                )

                # Update result with labeling info
                result['outcome_class'] = label_result['label']  # Use outcome_class for compatibility
                result['label'] = label_result['label']
                result['barrier_hit'] = label_result['barrier_hit']
                result['barrier_hit_date'] = label_result['barrier_hit_date']
                result['barrier_hit_day'] = label_result['barrier_hit_day']
                result['max_return'] = label_result['max_return']
                result['min_return'] = label_result['min_return']
                result['final_return'] = label_result['final_return']
                result['upper_barrier_pct'] = label_result['barriers']['upper_barrier_pct']
                result['lower_barrier_pct'] = label_result['barriers']['lower_barrier_pct']
                result['daily_volatility'] = label_result['barriers'].get('daily_volatility')

                # Dynamic barrier metadata (if using dynamic barriers)
                if self.config.use_dynamic_barriers:
                    result['market_cap_tier'] = label_result['barriers'].get('market_cap_tier', 'unknown')
                    result['regime_at_pattern'] = label_result['barriers'].get('regime', 'neutral')
                    result['barrier_rationale'] = label_result['barriers'].get('barrier_rationale', '')
                    result['adv_calculated'] = label_result['barriers'].get('adv_calculated')
                    result['labeling_method'] = 'triple_barrier_dynamic'
                else:
                    result['labeling_method'] = 'triple_barrier'

                result['labeling_version'] = 'v24_triple_barrier'
                result['labeled_at'] = datetime.now().isoformat()
                result['error'] = label_result.get('error')

                # Calculate historic features if requested
                if calculate_features and label_result['label'] is not None:
                    features = self.calculate_historic_features(
                        price_df=price_df,
                        reference_date=entry_date,
                        price_col=price_col
                    )
                    for k, v in features.items():
                        result[f'feature_{k}'] = v

                if label_result['label'] is not None:
                    labeled_count += 1
                    label_counts[label_result['label']] += 1
                else:
                    error_count += 1

                results.append(result)

        # Create output DataFrame
        df_labeled = pd.DataFrame(results)

        # Log summary
        logger.info(f"\nLabeling Summary:")
        logger.info(f"  - Successfully labeled: {labeled_count}")
        logger.info(f"  - Errors/Not ready: {error_count}")
        logger.info(f"\nLabel Distribution:")
        logger.info(f"  - Danger (0): {label_counts[0]} ({100*label_counts[0]/max(labeled_count,1):.1f}%)")
        logger.info(f"  - Noise (1): {label_counts[1]} ({100*label_counts[1]/max(labeled_count,1):.1f}%)")
        logger.info(f"  - Target (2): {label_counts[2]} ({100*label_counts[2]/max(labeled_count,1):.1f}%)")

        return df_labeled


# =============================================================================
# STRICT TEMPORAL SPLIT FUNCTIONS
# =============================================================================

def create_strict_temporal_split(
    df: pd.DataFrame,
    date_col: str = 'end_date',
    train_end: str = None,
    val_end: str = None,
    train_pct: float = 0.7,
    val_pct: float = 0.15,
    gap_days: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create strict temporal train/val/test split with NO OVERLAP.

    This function ensures:
        1. Train data comes BEFORE validation data
        2. Validation data comes BEFORE test data
        3. Optional gap between splits to prevent information leakage
        4. No patterns from the same time period appear in multiple splits

    Args:
        df: DataFrame with patterns
        date_col: Column containing pattern dates
        train_end: Explicit train end date (YYYY-MM-DD) or None for automatic
        val_end: Explicit validation end date (YYYY-MM-DD) or None for automatic
        train_pct: Fraction of data for training (if auto)
        val_pct: Fraction of data for validation (if auto)
        gap_days: Days of gap between splits (prevents look-ahead from recent patterns)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.copy()
    df['_date'] = pd.to_datetime(df[date_col])
    df = df.sort_values('_date')

    if train_end is not None and val_end is not None:
        # Use explicit dates
        train_cutoff = pd.to_datetime(train_end)
        val_cutoff = pd.to_datetime(val_end)
    else:
        # Calculate automatically from percentages
        n = len(df)
        train_n = int(n * train_pct)
        val_n = int(n * val_pct)

        train_cutoff = df['_date'].iloc[train_n]
        val_cutoff = df['_date'].iloc[train_n + val_n]

    # Apply gap if specified
    train_gap = train_cutoff + timedelta(days=gap_days)
    val_gap = val_cutoff + timedelta(days=gap_days)

    # Create masks with strict boundaries
    train_mask = df['_date'] <= train_cutoff
    val_mask = (df['_date'] > train_gap) & (df['_date'] <= val_cutoff)
    test_mask = df['_date'] > val_gap

    train_df = df[train_mask].drop(columns=['_date'])
    val_df = df[val_mask].drop(columns=['_date'])
    test_df = df[test_mask].drop(columns=['_date'])

    logger.info(f"Strict Temporal Split:")
    logger.info(f"  Train: {len(train_df)} patterns (through {train_cutoff.date()})")
    logger.info(f"  Val: {len(val_df)} patterns ({train_gap.date()} - {val_cutoff.date()})")
    logger.info(f"  Test: {len(test_df)} patterns (after {val_gap.date()})")

    if gap_days > 0:
        logger.info(f"  Gap: {gap_days} days between splits")

    # Validate no overlap
    train_dates = set(pd.to_datetime(train_df[date_col]))
    val_dates = set(pd.to_datetime(val_df[date_col]))
    test_dates = set(pd.to_datetime(test_df[date_col]))

    overlap_train_val = train_dates & val_dates
    overlap_val_test = val_dates & test_dates
    overlap_train_test = train_dates & test_dates

    if overlap_train_val or overlap_val_test or overlap_train_test:
        raise ValueError(
            f"Temporal split has overlapping dates! "
            f"Train-Val: {len(overlap_train_val)}, "
            f"Val-Test: {len(overlap_val_test)}, "
            f"Train-Test: {len(overlap_train_test)}"
        )

    return train_df, val_df, test_df


def validate_temporal_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = 'end_date'
) -> bool:
    """
    Validate that temporal splits have no look-ahead bias.

    Checks:
        1. All train dates < all val dates
        2. All val dates < all test dates
        3. No overlapping dates between any splits

    Args:
        train_df: Training split
        val_df: Validation split
        test_df: Test split
        date_col: Column containing dates

    Returns:
        True if valid, raises ValueError otherwise
    """
    train_max = pd.to_datetime(train_df[date_col]).max()
    val_min = pd.to_datetime(val_df[date_col]).min()
    val_max = pd.to_datetime(val_df[date_col]).max()
    test_min = pd.to_datetime(test_df[date_col]).min()

    issues = []

    if train_max >= val_min:
        issues.append(f"Train max ({train_max.date()}) >= Val min ({val_min.date()})")

    if val_max >= test_min:
        issues.append(f"Val max ({val_max.date()}) >= Test min ({test_min.date()})")

    if issues:
        raise ValueError(f"Temporal integrity violated: {'; '.join(issues)}")

    logger.info("Temporal integrity validated successfully")
    logger.info(f"  Train: through {train_max.date()}")
    logger.info(f"  Val: {val_min.date()} - {val_max.date()}")
    logger.info(f"  Test: from {test_min.date()}")

    return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_triple_barrier_labeler(
    volatility_scaling: bool = True,
    upper_barrier_pct: float = 0.03,
    lower_barrier_pct: float = 0.02
) -> TripleBarrierLabeler:
    """
    Factory function for creating a Triple Barrier Labeler with common settings.

    Args:
        volatility_scaling: If True, scale barriers by realized volatility
        upper_barrier_pct: Profit-taking threshold (default: 3%)
        lower_barrier_pct: Stop-loss threshold (default: 2%)

    Returns:
        Configured TripleBarrierLabeler instance
    """
    return TripleBarrierLabeler(
        vertical_barrier_days=VERTICAL_BARRIER_DAYS,
        upper_barrier_pct=upper_barrier_pct,
        lower_barrier_pct=lower_barrier_pct,
        volatility_scaling=volatility_scaling,
        volatility_multiplier=DEFAULT_VOLATILITY_MULTIPLIER,
        volatility_lookback=VOLATILITY_LOOKBACK,
        feature_window_size=FEATURE_WINDOW_SIZE,
        min_data_days=MIN_DATA_DAYS
    )


if __name__ == "__main__":
    # Self-test
    print("=" * 60)
    print("Triple Barrier Labeler - Self Test")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    # Simulate price with trend and noise
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    price_df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)

    # Create sample pattern
    entry_date = pd.Timestamp('2020-06-01')
    entry_price = price_df.loc[entry_date, 'close']

    print(f"\nSample Pattern:")
    print(f"  Entry date: {entry_date.date()}")
    print(f"  Entry price: ${entry_price:.2f}")

    # Create labeler
    labeler = TripleBarrierLabeler(
        vertical_barrier_days=150,
        upper_barrier_pct=0.03,
        lower_barrier_pct=0.02,
        volatility_scaling=True
    )

    # Label pattern
    result = labeler.label_single_pattern(
        price_df=price_df,
        entry_date=entry_date,
        entry_price=entry_price
    )

    print(f"\nLabeling Result:")
    print(f"  Label: {result['label']} ({'Target' if result['label']==2 else 'Danger' if result['label']==0 else 'Noise'})")
    print(f"  Barrier hit: {result['barrier_hit']}")
    print(f"  Hit day: {result['barrier_hit_day']}")
    print(f"  Max return: {result['max_return']*100:.2f}%")
    print(f"  Min return: {result['min_return']*100:.2f}%")
    print(f"  Barriers: +{result['barriers']['upper_barrier_pct']*100:.2f}% / {result['barriers']['lower_barrier_pct']*100:.2f}%")

    # Test historic features
    features = labeler.calculate_historic_features(price_df, entry_date)
    print(f"\nHistoric Features (strictly before {entry_date.date()}):")
    for k, v in features.items():
        if v is not None and not np.isnan(v):
            print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Self-test complete!")
