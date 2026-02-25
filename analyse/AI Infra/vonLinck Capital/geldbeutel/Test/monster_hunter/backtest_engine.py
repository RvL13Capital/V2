"""
backtest_engine.py

Simulation logic for the "Coil" mean-reversion strategy.
We provide liquidity to desperate sellers in forgotten micro-caps,
then sell when volume returns.

Philosophy: Buy dead, sell alive. Sniper execution with limit orders.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtest engine for Coil (mean-reversion) strategy.

    Entry: Limit buy at Support Level + 2% buffer (Stink Bid)
    Exit:
        1. Profit Take: +20% gain AND RVOL > 3.0 (sell the pop)
        2. Structural Stop: Close < Support * 0.90 (10% below)
        3. Time Stop: 150 days max hold
    """

    def __init__(self,
                 slippage_low_price: float = 0.15,
                 slippage_high_price: float = 0.08,
                 slippage_cutoff: float = 3.00,
                 entry_buffer: float = 0.02,
                 profit_target: float = 0.20,
                 volume_surge_threshold: float = 3.0,
                 structural_stop_pct: float = 0.90,
                 time_stop_days: int = 150):
        """
        Args:
            slippage_low_price: Slippage for stocks < $3.00 (default 15%)
            slippage_high_price: Slippage for stocks >= $3.00 (default 8%)
            slippage_cutoff: Price threshold for slippage tiers (default $3.00)
            entry_buffer: Buffer above support for limit order (default 2%)
            profit_target: Profit target for exit (default 20%)
            volume_surge_threshold: RVOL required for profit take (default 3.0)
            structural_stop_pct: Stop as multiplier of support (default 0.90 = 10% below)
            time_stop_days: Maximum holding period (default 150 days)
        """
        self.slippage_low = slippage_low_price
        self.slippage_high = slippage_high_price
        self.slippage_cutoff = slippage_cutoff
        self.entry_buffer = entry_buffer
        self.profit_target = profit_target
        self.volume_surge = volume_surge_threshold
        self.structural_stop = structural_stop_pct
        self.time_stop = time_stop_days

    def calculate_entry_price(self, support_level: float) -> float:
        """
        Calculate limit order price: Support + buffer.
        This is where we place our "stink bid" to provide liquidity.
        """
        return support_level * (1 + self.entry_buffer)

    def calculate_fill_price(self, limit_price: float) -> float:
        """
        Apply slippage to limit price when fill occurs.
        Even limit orders have slippage in illiquid micro-caps.
        """
        if limit_price < self.slippage_cutoff:
            return limit_price * (1 + self.slippage_low)
        else:
            return limit_price * (1 + self.slippage_high)

    def calculate_stop_price(self, support_level: float) -> float:
        """
        Structural stop: 10% below support level.
        Wide stop to absorb micro-cap noise without getting stopped out.
        """
        return support_level * self.structural_stop

    def simulate_coil_trade(self,
                            df: pd.DataFrame,
                            setup_idx: int,
                            support_level: float) -> Dict[str, Any]:
        """
        Simulates a Coil trade starting from the day AFTER setup_idx.

        Entry: Limit order at Support + 2%. We wait for price to come to us.
        Exit: Profit take at +20% with volume surge, structural stop, or time stop.

        Args:
            df: Full OHLCV dataframe with 'rvol_30d' column.
            setup_idx: Index where coil was identified.
            support_level: Bottom of the coil range (40-day low).

        Returns:
            Dict with: label (1/0), outcome, pnl, days, entry_price, exit_price
        """
        # Start checking from the NEXT day
        start_idx = setup_idx + 1

        if start_idx >= len(df):
            return {
                'label': 0,
                'outcome': 'insufficient_data',
                'pnl': 0.0,
                'days': 0
            }

        # Calculate entry parameters
        limit_price = self.calculate_entry_price(support_level)
        stop_price = self.calculate_stop_price(support_level)

        # PHASE 1: WAIT FOR FILL
        # We have a limit order sitting at support + 2%
        # Entry occurs when LOW <= limit_price (order gets filled)

        entry_day_idx = None
        buy_price = None

        # Look for fill within the time window
        max_wait_idx = min(start_idx + self.time_stop, len(df))

        for idx in range(start_idx, max_wait_idx):
            row = df.iloc[idx]

            # Check if our limit order gets filled
            if row['low'] <= limit_price:
                # FILL: We bought at our limit price (+ slippage)
                entry_day_idx = idx
                buy_price = self.calculate_fill_price(limit_price)
                break

        # If no fill within time window, no trade
        if entry_day_idx is None:
            return {
                'label': 0,
                'outcome': 'no_fill',
                'pnl': 0.0,
                'days': 0,
                'limit_price': limit_price,
                'support_level': support_level
            }

        # PHASE 2: HOLD AND MONITOR
        # We're now in the trade. Monitor for exits.

        target_price = buy_price * (1 + self.profit_target)
        days_held = 0

        # Time remaining after entry
        remaining_days = self.time_stop - (entry_day_idx - start_idx)
        end_idx = min(entry_day_idx + remaining_days, len(df))

        for idx in range(entry_day_idx + 1, end_idx):
            row = df.iloc[idx]
            days_held += 1

            current_rvol = row.get('rvol_30d', 0)
            current_close = row['close']
            current_high = row['high']
            current_low = row['low']

            # EXIT 1: PROFIT TAKE (Sell the Pop)
            # Price hits +20% AND volume surge (RVOL > 3.0)
            if current_high >= target_price and current_rvol >= self.volume_surge:
                pnl = (target_price - buy_price) / buy_price
                return {
                    'label': 1,
                    'outcome': 'profit_take',
                    'pnl': pnl,
                    'days': days_held,
                    'entry_price': buy_price,
                    'exit_price': target_price,
                    'exit_rvol': current_rvol
                }

            # EXIT 2: STRUCTURAL STOP
            # Close below support * 0.90 (10% below support)
            if current_close < stop_price:
                pnl = (stop_price - buy_price) / buy_price
                return {
                    'label': 0,
                    'outcome': 'structural_stop',
                    'pnl': pnl,
                    'days': days_held,
                    'entry_price': buy_price,
                    'exit_price': stop_price
                }

            # EXIT 3: PRICE TARGET WITHOUT VOLUME (Optional exit)
            # If we hit +20% but NO volume surge, we still take profit
            # (Don't be greedy - dead stocks don't always get volume on moves)
            if current_high >= target_price:
                pnl = (target_price - buy_price) / buy_price
                return {
                    'label': 1,
                    'outcome': 'target_no_volume',
                    'pnl': pnl,
                    'days': days_held,
                    'entry_price': buy_price,
                    'exit_price': target_price,
                    'exit_rvol': current_rvol
                }

        # EXIT 4: TIME STOP
        # 150 days elapsed, exit at market
        if end_idx > entry_day_idx + 1:
            final_close = df.iloc[end_idx - 1]['close']
            pnl = (final_close - buy_price) / buy_price
            return {
                'label': 0,
                'outcome': 'time_stop',
                'pnl': pnl,
                'days': days_held,
                'entry_price': buy_price,
                'exit_price': final_close
            }
        else:
            return {
                'label': 0,
                'outcome': 'data_ended',
                'pnl': 0.0,
                'days': days_held,
                'entry_price': buy_price
            }

    # Alias for backwards compatibility during transition
    def simulate_trade(self,
                       df: pd.DataFrame,
                       setup_idx: int,
                       trigger_level: float = None,
                       pivot_day_low: float = None,
                       support_level: float = None) -> Dict[str, Any]:
        """
        Wrapper for backwards compatibility.
        If support_level is provided, uses new Coil logic.
        Otherwise falls back to interpreting pivot_day_low as support.
        """
        if support_level is not None:
            return self.simulate_coil_trade(df, setup_idx, support_level)
        elif pivot_day_low is not None:
            # Use pivot_day_low as support level (transition helper)
            return self.simulate_coil_trade(df, setup_idx, pivot_day_low)
        else:
            return {
                'label': 0,
                'outcome': 'no_support_level',
                'pnl': 0.0,
                'days': 0
            }
