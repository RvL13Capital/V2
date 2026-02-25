"""
Unified Pattern Detection Module for AIv3 System
Consolidates all pattern detection logic into a single, configurable class
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from .config import get_config
from .data_loader import get_data_loader

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Data class for consolidation patterns"""
    ticker: str
    start_date: datetime
    end_date: datetime
    duration_days: int
    upper_boundary: float
    lower_boundary: float
    power_boundary: float
    avg_bbw: float
    avg_volume_ratio: float
    avg_range_ratio: float
    price_range_pct: float
    pattern_quality: float
    # Outcome fields (filled after pattern completion)
    outcome_max_gain: Optional[float] = None
    outcome_max_loss: Optional[float] = None
    outcome_end_gain: Optional[float] = None
    breakout_occurred: Optional[bool] = None
    days_to_breakout: Optional[int] = None
    breakout_date: Optional[datetime] = None
    breakout_volume_spike: Optional[float] = None
    outcome_class: Optional[str] = None


class UnifiedPatternDetector:
    """Unified pattern detector combining all detection methods"""

    def __init__(self):
        self.config = get_config()
        self.data_loader = get_data_loader()
        self.patterns: List[Pattern] = []

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required technical indicators"""
        # Bollinger Bands
        df['sma20'] = df['close'].rolling(20).mean()
        df['std20'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['sma20'] + (2 * df['std20'])
        df['bb_lower'] = df['sma20'] - (2 * df['std20'])
        df['bbw'] = ((df['bb_upper'] - df['bb_lower']) / df['sma20']) * 100

        # Volume metrics
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Range metrics
        df['daily_range'] = (df['high'] - df['low']) / df['low'] * 100
        df['avg_range'] = df['daily_range'].rolling(20).mean()
        df['range_ratio'] = df['daily_range'] / df['avg_range']

        # ADX (simplified)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift()),
            np.abs(df['low'] - df['close'].shift())
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # Directional movement
        df['plus_dm'] = np.where(
            (df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
            np.maximum(df['high'] - df['high'].shift(), 0),
            0
        )
        df['minus_dm'] = np.where(
            (df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
            np.maximum(df['low'].shift() - df['low'], 0),
            0
        )

        df['plus_di'] = (df['plus_dm'].rolling(14).mean() / df['atr']) * 100
        df['minus_di'] = (df['minus_dm'].rolling(14).mean() / df['atr']) * 100
        df['dx'] = (np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])) * 100
        df['adx'] = df['dx'].rolling(14).mean()

        # Clean up
        df = df.fillna(method='ffill').fillna(0)

        return df

    def is_consolidating(self, window: pd.DataFrame) -> Tuple[bool, Dict]:
        """Check if price window shows consolidation pattern"""
        if len(window) < self.config.pattern.min_duration_days:
            return False, {}

        # Calculate metrics
        bbw_percentile = np.percentile(window['bbw'].dropna(), 50)
        avg_volume_ratio = window['volume_ratio'].mean()
        avg_range_ratio = window['range_ratio'].mean()
        avg_adx = window['adx'].mean()

        # Check thresholds
        bbw_check = bbw_percentile < self.config.pattern.bbw_percentile_threshold
        volume_check = avg_volume_ratio < self.config.pattern.volume_ratio_threshold
        range_check = avg_range_ratio < self.config.pattern.range_ratio_threshold
        adx_check = avg_adx < self.config.pattern.adx_threshold

        # Pattern quality score
        quality = sum([bbw_check, volume_check, range_check, adx_check]) / 4

        is_pattern = quality >= 0.6  # At least 60% of criteria met

        metrics = {
            'bbw': bbw_percentile,
            'volume_ratio': avg_volume_ratio,
            'range_ratio': avg_range_ratio,
            'adx': avg_adx,
            'quality': quality
        }

        return is_pattern, metrics

    def detect_patterns(
        self,
        ticker: str,
        df: Optional[pd.DataFrame] = None,
        min_duration: Optional[int] = None,
        max_duration: Optional[int] = None
    ) -> List[Pattern]:
        """
        Detect consolidation patterns in ticker data

        Args:
            ticker: Stock ticker symbol
            df: Optional DataFrame (will load if not provided)
            min_duration: Minimum pattern duration
            max_duration: Maximum pattern duration

        Returns:
            List of detected patterns
        """
        # Load data if not provided
        if df is None:
            df = self.data_loader.load_ticker_data(ticker)

        if df.empty or len(df) < 60:
            logger.warning(f"Insufficient data for {ticker}")
            return []

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Pattern detection parameters
        min_dur = min_duration or self.config.pattern.min_duration_days
        max_dur = max_duration or self.config.pattern.max_duration_days
        step_size = max(5, min_dur // 2)

        patterns = []

        # Sliding window detection
        for i in range(60, len(df) - 30, step_size):
            for duration in range(min_dur, min(max_dur, len(df) - i - 30) + 1, 5):
                window = df.iloc[i:i + duration]

                is_pattern, metrics = self.is_consolidating(window)

                if is_pattern:
                    # Calculate boundaries
                    upper = window['high'].max()
                    lower = window['low'].min()
                    power = upper * self.config.pattern.breakout_multiplier

                    # Calculate pattern metrics
                    price_range_pct = ((upper - lower) / lower) * 100

                    # Create pattern
                    pattern = Pattern(
                        ticker=ticker,
                        start_date=window.index[0],
                        end_date=window.index[-1],
                        duration_days=duration,
                        upper_boundary=upper,
                        lower_boundary=lower,
                        power_boundary=power,
                        avg_bbw=metrics['bbw'],
                        avg_volume_ratio=metrics['volume_ratio'],
                        avg_range_ratio=metrics['range_ratio'],
                        price_range_pct=price_range_pct,
                        pattern_quality=metrics['quality']
                    )

                    # Evaluate outcome if historical data
                    pattern = self._evaluate_outcome(pattern, df, i + duration)

                    patterns.append(pattern)

                    # Skip ahead to avoid overlapping patterns
                    break

        logger.info(f"Found {len(patterns)} patterns for {ticker}")
        return patterns

    def _evaluate_outcome(
        self,
        pattern: Pattern,
        df: pd.DataFrame,
        pattern_end_idx: int
    ) -> Pattern:
        """Evaluate pattern outcome using future data"""
        evaluation_days = self.config.pattern.evaluation_days

        # Get future data
        future_start = pattern_end_idx
        future_end = min(pattern_end_idx + evaluation_days, len(df))

        if future_end <= future_start:
            return pattern

        future_data = df.iloc[future_start:future_end]

        if len(future_data) == 0:
            return pattern

        # Calculate outcomes
        max_price = future_data['high'].max()
        min_price = future_data['low'].min()
        final_price = future_data['close'].iloc[-1] if len(future_data) > 0 else pattern.upper_boundary

        # Calculate gains/losses
        pattern.outcome_max_gain = ((max_price - pattern.upper_boundary) / pattern.upper_boundary) * 100
        pattern.outcome_max_loss = ((min_price - pattern.lower_boundary) / pattern.lower_boundary) * 100
        pattern.outcome_end_gain = ((final_price - pattern.upper_boundary) / pattern.upper_boundary) * 100

        # Check for breakout
        breakout_mask = future_data['close'] > pattern.power_boundary
        if breakout_mask.any():
            pattern.breakout_occurred = True
            breakout_idx = breakout_mask.idxmax()
            pattern.breakout_date = breakout_idx
            pattern.days_to_breakout = (breakout_idx - pattern.end_date).days

            # Volume spike
            breakout_volume = future_data.loc[breakout_idx, 'volume']
            avg_volume = df.iloc[pattern_end_idx - 20:pattern_end_idx]['volume'].mean()
            pattern.breakout_volume_spike = breakout_volume / avg_volume if avg_volume > 0 else 1
        else:
            pattern.breakout_occurred = False

        # Classify outcome
        pattern.outcome_class = self.config.outcome.classify_outcome(pattern.outcome_max_gain)

        return pattern

    def detect_multiple_tickers(
        self,
        tickers: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Pattern]:
        """Detect patterns for multiple tickers"""
        if tickers is None:
            tickers = self.data_loader.list_available_tickers()

        if limit:
            tickers = tickers[:limit]

        all_patterns = []

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {ticker} ({i}/{len(tickers)})")
            try:
                patterns = self.detect_patterns(ticker)
                all_patterns.extend(patterns)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")

        logger.info(f"Total patterns found: {len(all_patterns)}")
        return all_patterns

    def filter_patterns(
        self,
        patterns: List[Pattern],
        min_quality: float = 0.7,
        min_duration: int = 10,
        outcome_class: Optional[str] = None
    ) -> List[Pattern]:
        """Filter patterns based on criteria"""
        filtered = patterns

        if min_quality:
            filtered = [p for p in filtered if p.pattern_quality >= min_quality]

        if min_duration:
            filtered = [p for p in filtered if p.duration_days >= min_duration]

        if outcome_class:
            filtered = [p for p in filtered if p.outcome_class == outcome_class]

        logger.info(f"Filtered {len(patterns)} patterns to {len(filtered)}")
        return filtered

    def patterns_to_dataframe(self, patterns: List[Pattern]) -> pd.DataFrame:
        """Convert patterns to DataFrame"""
        if not patterns:
            return pd.DataFrame()

        data = []
        for p in patterns:
            data.append({
                'ticker': p.ticker,
                'pattern_start_date': p.start_date,
                'pattern_end_date': p.end_date,
                'duration_days': p.duration_days,
                'upper_boundary': p.upper_boundary,
                'lower_boundary': p.lower_boundary,
                'power_boundary': p.power_boundary,
                'avg_bbw': p.avg_bbw,
                'avg_volume_ratio': p.avg_volume_ratio,
                'avg_range_ratio': p.avg_range_ratio,
                'price_range_pct': p.price_range_pct,
                'pattern_quality': p.pattern_quality,
                'outcome_max_gain': p.outcome_max_gain,
                'outcome_max_loss': p.outcome_max_loss,
                'outcome_end_gain': p.outcome_end_gain,
                'breakout_occurred': p.breakout_occurred,
                'days_to_breakout': p.days_to_breakout,
                'breakout_date': p.breakout_date,
                'breakout_volume_spike': p.breakout_volume_spike,
                'outcome_class': p.outcome_class
            })

        return pd.DataFrame(data)