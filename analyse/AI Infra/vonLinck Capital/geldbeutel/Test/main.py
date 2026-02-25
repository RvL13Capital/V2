# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List
import decimal  # For precise decimal handling in orders
# endregion

# === Enhanced Consolidation Detection (Small Cap Optimized) ===

@dataclass
class ConsolidationProfile:
    """Qualitaets-Metriken der Konsolidierung fuer Small Caps."""
    duration_days: int
    range_compression_ratio: float
    volume_decay_ratio: float
    flat_base_score: float
    quality_score: float
    atr_current: float
    avg_volume: float

class ConsolidationDetector:
    """Erkennung von Coils mit Fokus auf Volumen-Decay fuer Small Caps (weniger extrem)."""
    def __init__(self, min_days=25, max_days=100, range_threshold=0.30, vol_decay_threshold=0.75):  # Lockerer fuer Small Caps
        self.min_days = min_days
        self.max_days = max_days
        self.range_threshold = range_threshold
        self.vol_decay_threshold = vol_decay_threshold

    def analyze(self, df: pd.DataFrame, current_idx: int) -> Optional[ConsolidationProfile]:
        if len(df) < 60: return None

        lookback = 200
        slice_start = max(0, current_idx - lookback)
        subset = df.iloc[slice_start : current_idx + 1].copy()

        if len(subset) < 40: return None

        high, low, close = subset['high'].values, subset['low'].values, subset['close'].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr_series = pd.Series(tr).rolling(14).mean().values
        current_atr = atr_series[-1] if not np.isnan(atr_series[-1]) else 0

        if np.isnan(atr_series).all() or current_atr == 0: return None

        threshold_val = np.nanquantile(atr_series, self.range_threshold)

        consol_start_offset = None
        in_consol = False
        quiet_days = 0

        for i in range(len(atr_series) - 1, 13, -1):
            if np.isnan(atr_series[i]): continue
            if atr_series[i] <= threshold_val * 1.1:  # Lockerer Toleranz fuer Small Caps
                in_consol = True
                quiet_days += 1
                if consol_start_offset is None:
                    consol_start_offset = i
            elif in_consol and quiet_days >= 12:  # Etwas mehr ruhige Tage
                break
            elif in_consol:
                quiet_days = 0

        if consol_start_offset is None or quiet_days < 12: return None

        consol_start_idx = slice_start + consol_start_offset
        duration = current_idx - consol_start_idx

        if not (self.min_days <= duration <= self.max_days): return None

        start_slice = df.iloc[consol_start_idx : consol_start_idx + 5]
        end_slice = df.iloc[current_idx - 4 : current_idx + 1]

        if start_slice.empty or end_slice.empty: return None

        r_start = (start_slice['high'].max() - start_slice['low'].min())
        r_end = (end_slice['high'].max() - end_slice['low'].min())

        if r_start < 1e-9: return None
        range_comp = r_end / r_start

        consol_volume = subset['volume'].iloc[-duration:].mean()
        pre_consol_volume = subset['volume'].iloc[-(duration + 20):-duration].mean() if duration + 20 <= len(subset) else consol_volume
        volume_decay = consol_volume / pre_consol_volume if pre_consol_volume > 0 else 1.0

        closes = df.iloc[consol_start_idx : current_idx + 1]['close'].values
        q75, q25 = np.percentile(closes, [75, 25])
        filtered_closes = closes[(closes >= q25 * 0.9) & (closes <= q75 * 1.1)]
        flat_score = self._calc_flat_score(filtered_closes) if len(filtered_closes) > 5 else 0.0

        avg_volume = subset['volume'].tail(20).mean()

        score = 0
        if 30 <= duration <= 90: score += 20
        if range_comp < 0.65: score += 25  # Etwas lockerer
        elif range_comp < 0.85: score += 15
        if volume_decay < self.vol_decay_threshold: score += 25
        score += flat_score * 15

        if score < 65: return None  # Hoeherer Threshold

        return ConsolidationProfile(
            duration_days=duration,
            range_compression_ratio=range_comp,
            volume_decay_ratio=volume_decay,
            flat_base_score=flat_score,
            quality_score=score,
            atr_current=current_atr,
            avg_volume=avg_volume
        )

    def _calc_flat_score(self, closes):
        if len(closes) < 5: return 0.0
        mean_val = np.mean(closes)
        if mean_val < 1e-9: return 0.0
        cv = np.std(closes) / mean_val
        return max(0.0, 1.0 - (cv / 0.15))  # Etwas weiterer Toleranz

# === Algorithm ===

class MonsterHunterV9(QCAlgorithm):
    """
    V.0.9 'The Small Cap Sniper' (Optimized for 300M-2B Market Cap)

    Upgrades for Small Caps:
    - MarketCap Filter: 300M-2B USD.
    - Wider Price Range: 2-10 USD.
    - Reduced Slippage: 7% for better realism.
    - Trailing: Earlier BE (+8%), Aggressive at +40%.
    - Max Positions: 7 for slight diversification.
    - Liquidity constraints: 100k min avg volume, 5% ADV position limit.
    """

    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetEndDate(2024, 12, 1)
        self.SetCash(100000)

        self.MAX_POSITIONS = 7
        self.RISK_PER_TRADE = 0.012  # 1.2%

        self.MIN_PRICE = 2.00
        self.MAX_PRICE = 10.00
        self.MIN_DOLLAR_VOL = 1000000  # 1 Mio.
        self.MIN_AVG_VOLUME = 100000   # 100k shares/day minimum for small caps

        # FIX: Simple lambda for slippage model (removed incorrect SetMarketPrice call)
        self.SetSecurityInitializer(lambda security:
            security.SetSlippageModel(CustomPercentageSlippageModel(0.07))
        )

        self.detector = ConsolidationDetector()

        self.active_trades = {}
        self.pending_setups = {}  # {symbol: {'stop_price': float, 'expected_qty': int, 'filled_qty': 0, 'breakout_trigger': float, 'profile': obj}}

        self.stop_order_ids = {}  # {symbol: order_id}

        self.UniverseSettings.Resolution = Resolution.Daily
        self.spy_symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # Two-stage universe selection: Coarse -> Fine
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        self.SetWarmUp(250)

        # Cache for pre-screened candidates (updated in Fine selection)
        self.fine_candidates = []

        self.Schedule.On(self.DateRules.EveryDay(self.spy_symbol), self.TimeRules.AfterMarketOpen(self.spy_symbol, 45), self.ScanForEntries)
        self.Schedule.On(self.DateRules.EveryDay(self.spy_symbol), self.TimeRules.BeforeMarketClose(self.spy_symbol, 10), self.ManagePositions)
        self.Schedule.On(self.DateRules.EveryDay(self.spy_symbol), self.TimeRules.EndOfDay(self.spy_symbol), self.EODCleanup)

    def CoarseSelectionFunction(self, coarse):
        # Stage 1: Basic liquidity + price filter (fast, no fundamentals needed yet)
        selected = [
            x for x in coarse
            if x.HasFundamentalData
            and x.Price > self.MIN_PRICE
            and x.Price < self.MAX_PRICE
            and x.DollarVolume > self.MIN_DOLLAR_VOL
        ]
        # Top 200 nach DollarVolume -> weiter zu Fine Selection
        return [x.Symbol for x in sorted(selected, key=lambda x: x.DollarVolume, reverse=True)[:200]]

    def FineSelectionFunction(self, fine):
        # Stage 2: MarketCap + Fundamental filters (more expensive, fewer symbols)
        selected = [
            x for x in fine
            if 300000000 <= x.MarketCap <= 2000000000  # 300M-2B USD Small Cap
            and x.OperationRatios.ROE.Value > 0  # Profitabel
        ]

        # Rank by lowest volatility (proxy: beta) - consolidation candidates
        # Lower beta = potentially in consolidation
        ranked = sorted(selected, key=lambda x: abs(x.OperationRatios.ROE.Value), reverse=True)

        # Top 50 candidates for detailed analysis
        self.fine_candidates = [x.Symbol for x in ranked[:50]]
        return self.fine_candidates

    def ScanForEntries(self):
        open_orders = self.Transactions.GetOpenOrders()
        for order in open_orders:
            if order.CreatedTime.date() < self.Time.date() and order.Direction == OrderDirection.Buy:
                self.Transactions.CancelOrder(order.Id)

        pending_count = len([o for o in open_orders if o.Direction == OrderDirection.Buy])
        if len(self.active_trades) + pending_count >= self.MAX_POSITIONS: return

        # Use fine_candidates instead of all ActiveSecurities (50 vs 500)
        candidates = [s for s in self.fine_candidates if s in self.ActiveSecurities.Keys and s != self.spy_symbol and s not in self.active_trades and s not in self.pending_setups]
        if not candidates: return

        # STAGE 1: Quick pre-screen with 30-day history (fast)
        quick_history = self.History(candidates, 30, Resolution.Daily)
        if quick_history.empty: return

        # Pre-filter: Only symbols with low recent volatility (consolidation signal)
        pre_screened = []
        for symbol in candidates:
            if symbol not in quick_history.index.levels[0]: continue
            qh = quick_history.loc[symbol]
            if len(qh) < 20: continue

            # Quick volatility check: ATR/Price ratio
            high, low, close = qh['high'].values, qh['low'].values, qh['close'].values
            tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
            atr_ratio = np.mean(tr[-14:]) / close[-1] if close[-1] > 0 else 1

            # Low volatility = potential consolidation (ATR < 5% of price)
            if atr_ratio < 0.05:
                pre_screened.append(symbol)

        # STAGE 2: Full history only for top 20 pre-screened candidates
        final_candidates = pre_screened[:20]
        if not final_candidates: return

        history = self.History(final_candidates, 250, Resolution.Daily)
        if history.empty: return

        for symbol in final_candidates:
            if symbol not in history.index.levels[0]: continue
            df = history.loc[symbol].copy()
            if len(df) < 200: continue

            # Liquidity check for small caps
            avg_volume = df['volume'].tail(20).mean()
            if avg_volume < self.MIN_AVG_VOLUME:
                continue

            closes = df['close']
            sma_150 = closes.rolling(150).mean().iloc[-1]
            current_price = closes.iloc[-1]
            if current_price < sma_150: continue

            profile = self.detector.analyze(df.reset_index(drop=True), len(df)-1)
            if not profile or profile.quality_score < 65: continue

            lookback_pivot = 20
            recent_high = df['high'].tail(lookback_pivot).max()
            recent_volume = df['volume'].tail(lookback_pivot).mean()
            if current_price >= recent_high or recent_volume < profile.avg_volume * 1.2: continue

            # Wider breakout buffer for small caps (0.15 ATR instead of 0.1)
            breakout_trigger = recent_high + max(0.02, profile.atr_current * 0.15)

            consol_low = df['low'].tail(profile.duration_days).min()
            # Wider stop buffer for small caps (0.8 ATR instead of 0.5)
            stop_buffer = profile.atr_current * 0.8
            stop_loss = consol_low - stop_buffer

            risk_per_share = breakout_trigger - stop_loss
            if risk_per_share <= profile.atr_current * 0.5: continue

            atr_adjusted_risk = risk_per_share / profile.atr_current
            base_risk = self.Portfolio.TotalPortfolioValue * self.RISK_PER_TRADE
            qty = int(base_risk / risk_per_share)
            qty = int(qty / max(1, atr_adjusted_risk))

            max_qty_capital = int((self.Portfolio.TotalPortfolioValue * 0.05) / breakout_trigger)
            # Liquidity constraint: Don't exceed 5% of average daily volume
            max_qty_liquidity = int(avg_volume * 0.05)
            qty = min(qty, max_qty_capital, max_qty_liquidity)
            if qty < 200: continue

            # Wider limit buffer for small caps (8% instead of 5%)
            limit_price = breakout_trigger * 1.08

            self.pending_setups[symbol] = {
                'stop_price': stop_loss,
                'expected_qty': qty,
                'filled_qty': 0,
                'breakout_trigger': breakout_trigger,
                'profile': profile
            }

            self.StopLimitOrder(symbol, qty, breakout_trigger, limit_price, tag="Entry Breakout")
            self.Debug(f"SMALL CAP SETUP {symbol}: Trigger {breakout_trigger:.3f}, Stop {stop_loss:.3f}, Score {profile.quality_score:.1f}, VolDecay {profile.volume_decay_ratio:.2f}, Qty {qty}")

    def OnOrderEvent(self, orderEvent):
        symbol = orderEvent.Symbol

        if orderEvent.Status in [OrderStatus.Canceled, OrderStatus.Invalid]:
            if symbol in self.pending_setups:
                del self.pending_setups[symbol]
            if symbol in self.stop_order_ids:
                del self.stop_order_ids[symbol]

        if orderEvent.Status == OrderStatus.Filled:

            # ENTRY FILL (Buy) - PARTIAL SAFE
            if orderEvent.Direction == OrderDirection.Buy:
                if symbol in self.pending_setups:
                    setup = self.pending_setups[symbol]
                    fill_qty = orderEvent.FillQuantity
                    setup['filled_qty'] += fill_qty

                    total_filled = setup['filled_qty']
                    current_portfolio_qty = self.Portfolio[symbol].Quantity

                    # SAFETY NET: Adjust or create stop based on CURRENT portfolio qty
                    if symbol in self.stop_order_ids:
                        self.Transactions.CancelOrder(self.stop_order_ids[symbol])

                    stop_order = self.StopMarketOrder(symbol, -current_portfolio_qty, setup['stop_price'], tag="Hard Stop Loss")
                    self.stop_order_ids[symbol] = stop_order.OrderId

                    fill_price = orderEvent.FillPrice
                    if fill_price < setup['breakout_trigger'] * 0.95:
                        self.Debug(f"WARNING {symbol}: Gap-Down {fill_price:.3f}. Tightening.")
                        self.Transactions.CancelOrder(self.stop_order_ids[symbol])
                        emergency_stop = fill_price * 0.98
                        stop_order = self.StopMarketOrder(symbol, -current_portfolio_qty, emergency_stop, tag="Emergency Stop")
                        self.stop_order_ids[symbol] = stop_order.OrderId
                        setup['stop_price'] = emergency_stop

                    self.Debug(f"PARTIAL FILL {symbol}: +{fill_qty} (Total: {total_filled}/{setup['expected_qty']}). Stop adjusted for {current_portfolio_qty}.")

                    if total_filled >= setup['expected_qty']:
                        del self.pending_setups[symbol]

                        self.active_trades[symbol] = {
                            'buy_price': fill_price,
                            'stop_price': setup['stop_price'],
                            'highest_price': fill_price,
                            'taken_partial': False,
                            'entry_time': self.Time,
                            'quantity': current_portfolio_qty,
                            'initial_qty': setup['expected_qty'],
                            'atr': setup['profile'].atr_current
                        }
                        self.Debug(f"SMALL CAP ENTRY COMPLETE {symbol} @ ~{fill_price:.3f}. HARD STOP @ {setup['stop_price']:.3f} for {current_portfolio_qty} shares.")

            # SELL FILLS
            elif orderEvent.Direction == OrderDirection.Sell and symbol in self.active_trades:
                trade = self.active_trades[symbol]
                fill_qty = abs(orderEvent.FillQuantity)
                trade['quantity'] -= fill_qty

                if "Stop" in orderEvent.Tag:
                    del self.active_trades[symbol]
                    if symbol in self.stop_order_ids:
                        del self.stop_order_ids[symbol]
                    self.Debug(f"STOP HIT {symbol} @ {orderEvent.FillPrice:.3f}")
                    return

                if "Partial" in orderEvent.Tag and symbol in self.stop_order_ids:
                    remaining_qty = max(0, self.Portfolio[symbol].Quantity)
                    if remaining_qty > 0:
                        self.Transactions.CancelOrder(self.stop_order_ids[symbol])
                        new_stop = self.StopMarketOrder(symbol, -remaining_qty, trade['stop_price'], tag="Adjusted Hard Stop")
                        self.stop_order_ids[symbol] = new_stop.OrderId
                    else:
                        del self.stop_order_ids[symbol]

                if trade['quantity'] <= 0:
                    del self.active_trades[symbol]
                    if symbol in self.stop_order_ids:
                        del self.stop_order_ids[symbol]
                    self.Debug(f"CLOSED {symbol} @ {orderEvent.FillPrice:.3f}")

    def ManagePositions(self):
        for symbol in list(self.active_trades.keys()):
            if not self.Securities.ContainsKey(symbol): continue

            price = self.Securities[symbol].Price
            if price == 0: continue

            trade = self.active_trades[symbol]

            if price > trade['highest_price']:
                trade['highest_price'] = price

            pnl_pct = (price - trade['buy_price']) / trade['buy_price']
            days_held = (self.Time - trade['entry_time']).days

            # PARTIALS
            if not trade['taken_partial'] and pnl_pct >= 1.0:
                qty_to_sell = int(trade['initial_qty'] * 0.25)
                current_qty = self.Portfolio[symbol].Quantity
                actual_sell = min(qty_to_sell, current_qty)
                if actual_sell > 0:
                    self.MarketOrder(symbol, -actual_sell, tag="Partial Profit +100%")
                    trade['taken_partial'] = True
                    self.Debug(f"PARTIAL +100% {symbol}: Selling {actual_sell}.")

            elif trade['taken_partial'] and pnl_pct >= 2.0:
                remaining_qty = self.Portfolio[symbol].Quantity
                qty_to_sell = int(remaining_qty * 0.33)
                if qty_to_sell > 0:
                    self.MarketOrder(symbol, -qty_to_sell, tag="Partial Profit +200%")
                    self.Debug(f"PARTIAL +200% {symbol}: Selling {qty_to_sell}.")

            # OPTIMIZED TRAILING: Small Cap Adjusted (earlier phases)
            atr = trade.get('atr', 0.05)
            trail_distance = atr * 1.8  # Etwas enger

            # Phase 1: BE + ATR Buffer after +8%
            if pnl_pct > 0.08:
                be_stop = trade['buy_price'] + (atr * 0.5)  # Kleiner Buffer
                if be_stop > trade['stop_price']:
                    self._update_stop_order(symbol, be_stop, "Trailing BE + ATR")
                    trade['stop_price'] = be_stop

            # Phase 2: Initial Trail at +20% (1.2x ATR)
            elif pnl_pct > 0.20:
                new_trail = max(trade['highest_price'] - (atr * 1.2), trade['stop_price'])
                if new_trail > trade['stop_price']:
                    self._update_stop_order(symbol, new_trail, "Trailing Initial")
                    trade['stop_price'] = new_trail

            # Phase 3: Aggressive at +40% (0.8x ATR trail)
            elif pnl_pct > 0.40:
                new_trail = max(price - (atr * 0.8), trade['stop_price'])
                if new_trail > trade['stop_price']:
                    self._update_stop_order(symbol, new_trail, "Trailing Aggressive")
                    trade['stop_price'] = new_trail

            # TIME/ZOMBIE - Keep original thresholds to allow positions to develop
            if days_held > 20 and pnl_pct < 0.05:
                self.Liquidate(symbol, "Zombie Kill (20d <5%)")
                continue

            if days_held > 10 and (trade['highest_price'] / price) < 1.05 and pnl_pct < 0.10:  # Approx no new high
                self.Liquidate(symbol, "Stagnation Kill")
                continue

            if days_held > 120:
                self.Liquidate(symbol, "Max Hold 120d")
                continue

    def _update_stop_order(self, symbol, new_stop_price, tag_suffix):
        if symbol in self.stop_order_ids:
            self.Transactions.CancelOrder(self.stop_order_ids[symbol])
        trade = self.active_trades[symbol]
        qty = trade['quantity']
        if qty > 0:
            new_order = self.StopMarketOrder(symbol, -qty, new_stop_price, tag=f"Trailing {tag_suffix}")
            self.stop_order_ids[symbol] = new_order.OrderId
            self.Debug(f"TRAIL UPDATE {symbol}: New Stop {new_stop_price:.3f} for {qty} shares.")

    def EODCleanup(self):
        for symbol in list(self.stop_order_ids.keys()):
            if symbol not in self.active_trades:
                self.Transactions.CancelOrder(self.stop_order_ids[symbol])
                del self.stop_order_ids[symbol]


# === Helper: Custom Slippage Model ===
class CustomPercentageSlippageModel:
    """Custom slippage model for small cap stocks with wider spreads."""
    def __init__(self, slip_pct):
        self.slip = slip_pct

    # FIX: PascalCase method name required by QuantConnect
    def GetSlippageApproximation(self, asset, order):
        return asset.Price * self.slip
