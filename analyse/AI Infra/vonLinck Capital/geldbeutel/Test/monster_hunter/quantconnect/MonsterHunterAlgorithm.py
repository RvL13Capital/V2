# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict
# endregion

@dataclass
class ConsolidationProfile:
    """Captures the quality metrics of a consolidation pattern."""
    duration_days: int
    range_compression_ratio: float
    volume_decay_slope: float
    higher_lows_count: int
    flat_base_score: float
    pivot_volume_ratio: float

    @property
    def quality_score(self) -> float:
        """Composite score for consolidation quality (0-100)."""
        score = 0.0

        # Duration: Sweet spot is 30-90 days
        if 30 <= self.duration_days <= 90: score += 25
        elif 20 <= self.duration_days <= 120: score += 15

        # Volume decay: We want negative slope (drying up)
        if self.volume_decay_slope < -0.05: score += 25
        elif self.volume_decay_slope < 0: score += 15

        # Range compression: Tighter = more coiled
        if self.range_compression_ratio < 0.5: score += 25
        elif self.range_compression_ratio < 0.75: score += 15

        # Higher lows: Accumulation signature
        score += min(self.higher_lows_count * 8, 25)

        return score


class ConsolidationDetector:
    """Vectorized engine to detect and score price consolidations."""

    def __init__(self, min_days=20, max_days=120, range_threshold=0.25):
        self.min_days = min_days
        self.max_days = max_days
        self.range_threshold = range_threshold

    def analyze(self, df: pd.DataFrame, current_idx: int) -> Optional[ConsolidationProfile]:
        # 1. Detect Start of Consolidation (Volatility Drop)
        lookback = 120
        slice_start = max(0, current_idx - lookback - 20)
        subset = df.iloc[slice_start : current_idx + 1].copy()

        if len(subset) < 30: return None

        # Vectorized ATR
        high, low, close = subset['high'].values, subset['low'].values, subset['close'].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr_14 = pd.Series(tr).rolling(14).mean().values

        # Threshold for "quiet" activity
        threshold_val = np.nanquantile(atr_14, self.range_threshold)

        # Walk backward to find start
        consol_start_offset = None
        in_consol = False
        for i in range(len(atr_14) - 2, 13, -1):
            if np.isnan(atr_14[i]): continue
            if atr_14[i] <= threshold_val:
                in_consol = True
                consol_start_offset = i
            elif in_consol:
                break  # Consolidation broken

        if consol_start_offset is None: return None

        # Map back to global df index
        consol_start_idx = slice_start + consol_start_offset
        duration = current_idx - consol_start_idx

        if not (self.min_days <= duration <= self.max_days): return None

        # 2. Calculate Metrics
        # Range Compression (Last week range / First week range)
        start_slice = df.iloc[consol_start_idx : consol_start_idx + 5]
        end_slice = df.iloc[current_idx - 4 : current_idx + 1]

        if start_slice.empty or end_slice.empty: return None

        r_start = (start_slice['high'].max() - start_slice['low'].min())
        r_end = (end_slice['high'].max() - end_slice['low'].min())
        range_comp = r_end / (r_start + 1e-9)

        # Volume Decay Slope (Vectorized Linear Regression)
        vol_series = df.iloc[consol_start_idx : current_idx + 1]['volume'].values
        vol_slope = self._calc_slope(vol_series)

        # Higher Lows
        higher_lows = self._count_higher_lows(df.iloc[consol_start_idx : current_idx + 1]['low'].values)

        # Flat Base Score
        closes = df.iloc[consol_start_idx : current_idx + 1]['close'].values
        flat_score = self._calc_flat_score(closes)

        return ConsolidationProfile(
            duration_days=duration,
            range_compression_ratio=range_comp,
            volume_decay_slope=vol_slope,
            higher_lows_count=higher_lows,
            flat_base_score=flat_score,
            pivot_volume_ratio=0.0  # Calc in algo
        )

    def _calc_slope(self, y):
        # Normalized slope
        y_norm = y / (np.mean(y) + 1e-9)
        x = np.arange(len(y))
        # Vectorized m = Cov(x,y) / Var(x)
        return np.polyfit(x, y_norm, 1)[0]

    def _count_higher_lows(self, lows, window=5):
        if len(lows) < window * 2: return 0
        # Identify local minima
        is_min = np.r_[False, lows[1:] < lows[:-1]] & np.r_[lows[:-1] < lows[1:], False]
        # Simple count of ascending minima
        min_vals = lows[is_min]
        if len(min_vals) < 2: return 0
        return np.sum(min_vals[1:] > min_vals[:-1])

    def _calc_flat_score(self, closes):
        if len(closes) < 5: return 0.0
        cv = np.std(closes) / (np.mean(closes) + 1e-9)
        return max(0.0, 1.0 - (cv / 0.20))


class MonsterHunterAlgorithm(QCAlgorithm):
    """
    Monster Hunter v2: The "Coil" Strategy
    Focuses on volatility compression (Consolidation) + Volume Pivot Breakouts.
    """

    def Initialize(self):
        self.SetStartDate(2014, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        # Strategy Params
        self.PIVOT_LOOKBACK = 40
        self.MIN_PRICE = 0.50
        self.MAX_PRICE = 15.00
        self.MIN_DOLLAR_VOL = 100000  # Stricter liquidity for coil

        # Risk Mgmt
        self.TARGET_MULTIPLE = 2.0
        self.TIME_STOP_DAYS = 60
        self.MAX_POSITIONS = 15
        self.POSITION_SIZE_DOLLARS = 3000

        # Slippage (Brutal Mode)
        self.SLIPPAGE_LOW = 0.15
        self.SLIPPAGE_HIGH = 0.08
        self.SLIPPAGE_CUTOFF = 3.00

        self.detector = ConsolidationDetector()
        self.active_trades = {}

        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)
        self.AddEquity("SPY", Resolution.Daily)

        # Execution Schedule
        self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.AfterMarketOpen("SPY", 30), self.ScanForEntries)
        self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.BeforeMarketClose("SPY", 5), self.ManagePositions)

    def CoarseSelectionFunction(self, coarse):
        # Loose filter
        selected = [x for x in coarse if x.HasFundamentalData and
                   x.Price > self.MIN_PRICE and x.Price < self.MAX_PRICE and
                   x.DollarVolume > self.MIN_DOLLAR_VOL]
        # Return top 200 by DollarVol (Liquid Microcaps)
        return [x.Symbol for x in sorted(selected, key=lambda x: x.DollarVolume)[:200]]

    def ScanForEntries(self):
        if len(self.active_trades) >= self.MAX_POSITIONS: return

        # Get history for ALL symbols in universe to optimize (batch request)
        symbols = [s for s in self.ActiveSecurities.Keys if s != self.Symbol("SPY")]
        if not symbols: return

        # Need ~140 days for 120d consolidation + buffers
        history = self.History(symbols, 150, Resolution.Daily)
        if history.empty: return

        for symbol in symbols:
            if symbol in self.active_trades: continue
            if symbol not in history.index: continue

            df = history.loc[symbol]
            if len(df) < 50: continue

            # 1. Trigger Logic (Volume Pivot)
            # Use data strictly before today (-1) to avoid lookahead
            hist_df = df.iloc[:-1]
            if hist_df.empty: continue

            # Find max volume day in last 40 days
            subset = hist_df.tail(self.PIVOT_LOOKBACK)
            if subset.empty: continue

            max_vol_idx = subset['volume'].argmax()
            pivot_row = subset.iloc[max_vol_idx]
            trigger = pivot_row['high']
            pivot_low = pivot_row['low']

            # 2. Breakout Check (Today's Price)
            # Current bar (partial or complete depending on when Scan runs)
            # In QC Daily resolution, 'Price' is current.
            current_price = self.Securities[symbol].Price
            current_open = self.Securities[symbol].Open

            # Simple Breakout Check
            if current_price <= trigger: continue

            # RVOL Check
            avg_vol = subset['volume'].mean()
            if avg_vol == 0: continue
            # Approximate today's volume projection or use yesterday's close vol?
            # Safe bet: Use 30-day RVOL of YESTERDAY to confirm "percolation"
            rvol = subset['volume'].iloc[-1] / avg_vol
            if rvol < 1.5: continue  # Lower threshold, relying on coil quality

            # 3. THE COIL CHECK (Consolidation Engine)
            # Analyze history ending YESTERDAY
            profile = self.detector.analyze(hist_df.reset_index(drop=True), len(hist_df)-1)

            if not profile: continue
            if profile.quality_score < 50:
                # self.Debug(f"Rejected {symbol}: Score {profile.quality_score:.1f}")
                continue

            # 4. EXECUTION (The Gap Fix)
            # You pay MAX(Trigger, Open) * Slippage
            # This kills the "front-running gap" bug
            base_price = max(trigger, current_open)
            buy_price = self._apply_slippage(base_price)

            # Size & Enter
            shares = int(self.POSITION_SIZE_DOLLARS / buy_price)
            if shares <= 0: continue

            self.MarketOrder(symbol, shares)
            stop_price = self._calc_stop(buy_price, pivot_low)

            self.active_trades[symbol] = {
                'buy_price': buy_price,
                'stop_price': stop_price,
                'target': buy_price * self.TARGET_MULTIPLE,
                'days_held': 0,
                'trigger': trigger,
                'score': profile.quality_score
            }
            self.Debug(f"BUY {symbol} @ {buy_price:.2f} (GapBase: {base_price}) | Score: {profile.quality_score:.0f}")

    def ManagePositions(self):
        # EOD Checks
        to_remove = []

        for symbol, trade in self.active_trades.items():
            if not self.Securities.ContainsKey(symbol): continue

            price = self.Securities[symbol].Price
            trade['days_held'] += 1

            # 1. Failed Breakout (Close < Trigger on Day 1)
            # Strict survival rule
            if trade['days_held'] == 1 and price < trade['trigger']:
                self.Liquidate(symbol, "Failed Breakout")
                to_remove.append(symbol)
                continue

            # 2. Stop Hit
            if price <= trade['stop_price']:
                self.Liquidate(symbol, "Stop Hit")
                to_remove.append(symbol)
                continue

            # 3. Target Hit
            if price >= trade['target']:
                self.Liquidate(symbol, "Target Hit")
                to_remove.append(symbol)
                continue

            # 4. Time Stop
            if trade['days_held'] >= self.TIME_STOP_DAYS:
                self.Liquidate(symbol, "Time Stop")
                to_remove.append(symbol)
                continue

        for s in to_remove:
            del self.active_trades[s]

    def _apply_slippage(self, price):
        rate = self.SLIPPAGE_LOW if price < self.SLIPPAGE_CUTOFF else self.SLIPPAGE_HIGH
        return price * (1 + rate)

    def _calc_stop(self, entry, pivot_low):
        # Dynamic Wide Stop
        vol_stop = entry * 0.75
        struct_stop = pivot_low * 0.98
        return min(vol_stop, struct_stop)
