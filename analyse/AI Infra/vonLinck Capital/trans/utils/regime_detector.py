"""
Regime Detector for TRAnS Pattern Trading

Identifies BULL/BEAR/SIDEWAYS market regimes to filter pattern trades.
Patterns are only profitable in BULL regimes.

Usage:
    from utils.regime_detector import RegimeDetector

    detector = RegimeDetector()
    regime = detector.get_current_regime(index_data)

    if regime == 'BULL':
        # Execute pattern trades
    else:
        # Stay flat
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class Regime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeSignal:
    regime: Regime
    confidence: float  # 0-1
    indicators: Dict[str, float]
    recommendation: str


class RegimeDetector:
    """
    Multi-factor regime detector using:
    1. Price vs SMA (trend)
    2. SMA slope (momentum)
    3. Volatility (VIX proxy or realized vol)
    4. Breadth (% above 200 SMA) - optional
    """

    def __init__(
        self,
        sma_short: int = 50,
        sma_long: int = 200,
        vol_window: int = 20,
        vol_threshold_low: float = 15.0,  # Annualized vol %
        vol_threshold_high: float = 25.0,
        slope_window: int = 20,
    ):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.vol_window = vol_window
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high
        self.slope_window = slope_window

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate regime indicators from OHLCV data.

        Args:
            df: DataFrame with 'close' column, DatetimeIndex

        Returns:
            Dict of indicator values
        """
        if len(df) < self.sma_long + self.slope_window:
            return {}

        close = df['close'].values

        # SMAs
        sma_short = pd.Series(close).rolling(self.sma_short).mean().iloc[-1]
        sma_long = pd.Series(close).rolling(self.sma_long).mean().iloc[-1]
        current_price = close[-1]

        # Price position relative to SMAs
        price_vs_sma_long = (current_price / sma_long - 1) * 100
        price_vs_sma_short = (current_price / sma_short - 1) * 100

        # SMA slope (annualized %)
        sma_long_series = pd.Series(close).rolling(self.sma_long).mean()
        sma_slope = (sma_long_series.iloc[-1] / sma_long_series.iloc[-self.slope_window] - 1) * 100 * (252 / self.slope_window)

        # Realized volatility (annualized)
        returns = pd.Series(close).pct_change().dropna()
        realized_vol = returns.iloc[-self.vol_window:].std() * np.sqrt(252) * 100

        # Trend strength: SMA alignment
        sma_alignment = 1 if (current_price > sma_short > sma_long) else \
                       -1 if (current_price < sma_short < sma_long) else 0

        return {
            'price': current_price,
            'sma_short': sma_short,
            'sma_long': sma_long,
            'price_vs_sma_long': price_vs_sma_long,
            'price_vs_sma_short': price_vs_sma_short,
            'sma_slope': sma_slope,
            'realized_vol': realized_vol,
            'sma_alignment': sma_alignment,
        }

    def classify_regime(self, indicators: Dict[str, float]) -> Tuple[Regime, float]:
        """
        Classify regime based on indicators.

        Returns:
            (regime, confidence)
        """
        if not indicators:
            return Regime.UNKNOWN, 0.0

        score = 0.0
        max_score = 0.0

        # Factor 1: Price vs SMA 200 (weight: 30%)
        price_vs_sma = indicators.get('price_vs_sma_long', 0)
        if price_vs_sma > 5:
            score += 0.3
        elif price_vs_sma < -5:
            score -= 0.3
        max_score += 0.3

        # Factor 2: SMA alignment (weight: 25%)
        alignment = indicators.get('sma_alignment', 0)
        score += alignment * 0.25
        max_score += 0.25

        # Factor 3: SMA slope (weight: 25%)
        slope = indicators.get('sma_slope', 0)
        if slope > 10:
            score += 0.25
        elif slope > 0:
            score += 0.125
        elif slope < -10:
            score -= 0.25
        elif slope < 0:
            score -= 0.125
        max_score += 0.25

        # Factor 4: Volatility (weight: 20%)
        vol = indicators.get('realized_vol', 20)
        if vol < self.vol_threshold_low:
            score += 0.2  # Low vol = bull
        elif vol > self.vol_threshold_high:
            score -= 0.2  # High vol = bear
        max_score += 0.2

        # Normalize score to -1 to +1
        normalized = score / max_score if max_score > 0 else 0

        # Classify
        if normalized > 0.3:
            regime = Regime.BULL
            confidence = min(1.0, (normalized - 0.3) / 0.7 + 0.5)
        elif normalized < -0.3:
            regime = Regime.BEAR
            confidence = min(1.0, (-normalized - 0.3) / 0.7 + 0.5)
        else:
            regime = Regime.SIDEWAYS
            confidence = 1.0 - abs(normalized) / 0.3

        return regime, confidence

    def get_regime(self, df: pd.DataFrame) -> RegimeSignal:
        """
        Get current market regime from price data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            RegimeSignal with regime, confidence, and recommendation
        """
        indicators = self.calculate_indicators(df)
        regime, confidence = self.classify_regime(indicators)

        # Trading recommendation
        if regime == Regime.BULL and confidence > 0.6:
            recommendation = "TRADE: Bull regime confirmed. Execute pattern signals."
        elif regime == Regime.BULL:
            recommendation = "TRADE CAUTIOUS: Weak bull. Reduce position sizes."
        elif regime == Regime.BEAR:
            recommendation = "NO TRADE: Bear regime. Stay flat."
        elif regime == Regime.SIDEWAYS:
            recommendation = "NO TRADE: Sideways regime. Patterns unreliable."
        else:
            recommendation = "NO TRADE: Insufficient data."

        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            recommendation=recommendation
        )

    def get_regime_history(self, df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
        """
        Calculate rolling regime history.

        Args:
            df: DataFrame with OHLCV data
            lookback: Number of days to calculate history

        Returns:
            DataFrame with date, regime, confidence
        """
        results = []

        min_required = self.sma_long + self.slope_window

        for i in range(min_required, len(df)):
            subset = df.iloc[:i+1]
            signal = self.get_regime(subset)

            results.append({
                'date': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i,
                'regime': signal.regime.value,
                'confidence': signal.confidence,
                **signal.indicators
            })

        return pd.DataFrame(results)


class VIXRegimeDetector:
    """
    Simpler regime detector using VIX levels.

    VIX < 15: Strong Bull
    VIX 15-20: Bull
    VIX 20-25: Sideways
    VIX 25-30: Weak Bear
    VIX > 30: Bear
    """

    def __init__(
        self,
        bull_threshold: float = 20.0,
        bear_threshold: float = 25.0,
    ):
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def get_regime(self, vix_value: float) -> RegimeSignal:
        """
        Get regime from current VIX value.
        """
        if vix_value < 15:
            regime = Regime.BULL
            confidence = 0.9
            recommendation = "TRADE: Low VIX, strong bull regime."
        elif vix_value < self.bull_threshold:
            regime = Regime.BULL
            confidence = 0.7
            recommendation = "TRADE: Moderate VIX, bull regime."
        elif vix_value < self.bear_threshold:
            regime = Regime.SIDEWAYS
            confidence = 0.6
            recommendation = "CAUTION: Elevated VIX, sideways regime."
        elif vix_value < 30:
            regime = Regime.BEAR
            confidence = 0.7
            recommendation = "NO TRADE: High VIX, bear regime."
        else:
            regime = Regime.BEAR
            confidence = 0.9
            recommendation = "NO TRADE: Very high VIX, strong bear regime."

        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            indicators={'vix': vix_value},
            recommendation=recommendation
        )


class DangerRateRegimeDetector:
    """
    Regime detector based on rolling danger rate from pattern outcomes.

    This is a lagging indicator but directly measures pattern success.
    """

    def __init__(
        self,
        lookback_days: int = 30,
        bull_threshold: float = 0.40,  # Danger rate < 40% = bull
        bear_threshold: float = 0.55,  # Danger rate > 55% = bear
    ):
        self.lookback_days = lookback_days
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold

    def get_regime(self, recent_patterns: pd.DataFrame) -> RegimeSignal:
        """
        Get regime from recent pattern outcomes.

        Args:
            recent_patterns: DataFrame with 'outcome_class' column
                            (0=Danger, 1=Noise, 2=Target)
        """
        if len(recent_patterns) < 10:
            return RegimeSignal(
                regime=Regime.UNKNOWN,
                confidence=0.0,
                indicators={},
                recommendation="NO TRADE: Insufficient recent patterns."
            )

        danger_rate = (recent_patterns['outcome_class'] == 0).mean()
        target_rate = (recent_patterns['outcome_class'] == 2).mean()

        if danger_rate < self.bull_threshold and target_rate > 0.12:
            regime = Regime.BULL
            confidence = min(1.0, (self.bull_threshold - danger_rate) / 0.15 + 0.5)
            recommendation = f"TRADE: Low danger rate ({danger_rate:.1%}). Bull regime."
        elif danger_rate > self.bear_threshold:
            regime = Regime.BEAR
            confidence = min(1.0, (danger_rate - self.bear_threshold) / 0.15 + 0.5)
            recommendation = f"NO TRADE: High danger rate ({danger_rate:.1%}). Bear regime."
        else:
            regime = Regime.SIDEWAYS
            confidence = 0.5
            recommendation = f"CAUTION: Moderate danger rate ({danger_rate:.1%}). Sideways."

        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            indicators={
                'danger_rate': danger_rate,
                'target_rate': target_rate,
                'noise_rate': 1 - danger_rate - target_rate,
                'n_patterns': len(recent_patterns),
            },
            recommendation=recommendation
        )


def test_regime_detector():
    """Test regime detector with sample data."""
    import yfinance as yf

    print("Testing Regime Detector")
    print("=" * 60)

    # Download SPY data
    spy = yf.download("SPY", start="2022-01-01", end="2025-01-01", progress=False)
    spy.columns = [c.lower() for c in spy.columns]

    detector = RegimeDetector()

    # Test at different points
    test_dates = ['2022-06-15', '2022-10-15', '2023-06-15', '2024-06-15']

    for date in test_dates:
        subset = spy.loc[:date]
        signal = detector.get_regime(subset)
        print(f"\n{date}:")
        print(f"  Regime: {signal.regime.value}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Price vs SMA200: {signal.indicators.get('price_vs_sma_long', 0):.1f}%")
        print(f"  Realized Vol: {signal.indicators.get('realized_vol', 0):.1f}%")
        print(f"  Recommendation: {signal.recommendation}")


if __name__ == "__main__":
    test_regime_detector()
