"""
Rolling Walk-Forward Optimization Matrix — geldbeutel V2.0
===========================================================
Swing Liquidity Sweep Engine on 1-minute CME NQ futures.

Theory: TTDLS — Thermodynamic Theory of Discrete Liquidity Sweeps.
Blueprint: 2terfix.txt — 5-step V2 architecture.

Architecture changes from V1.0:
    Step 1 — True OOS isolation: Phase 1/2 gated at oos_start_idx inside Numba.
              Phase 3 mapping runs continuously (structural memory accumulates
              through the IS overhang). The blueprint's 'continue' bug is fixed:
              the execution lock is applied BEFORE Phase 3, not via 'continue'
              inside the loop body which would skip structural scanning.

    Step 2 — Structural regime gate: tracks the outcome of the last two macro
              sweeps per side. Two consecutive depth-deaths hard-lock that
              side of the state machine until the opposite side succeeds. No
              indicators. Pure price-physics memory.

    Step 3 — Thermodynamic volume friction: sweep must accumulate volume
              > vol_mult × tod_baseline_vol before a pending signal is
              generated. Real stop-runs require forced-liquidation volume.
              vol_mult is the 7th gene, evolved by the GA.

    Step 4 — Roll date handling: is_roll_date zeros L_star, H_star, and all
              pristine memory. Unadjusted CME data + surgical roll reset.
              Gene bounds rescaled to swing dimension (w/m in 1-min bars).

    Step 5 — Pure Calmar fitness: net/dd with hard feasibility gates.
              min_trades=10 ensures statistical significance on 12-month IS.
              min_avg_hold=500 bars (~8h) eliminates scalp-degenerate genomes.
              True intrabar MAE used for mark-to-market equity.

Performance note:
    Phase 3 runs on every 1-min bar. O(1) deque + O(M) pristine scan.
    Cross-roll clamp prevents pre-roll structure from poisoning current contract.
    True intrabar MAE tracked in IS kernel for accurate max-drawdown.
"""

import numpy as np
import pandas as pd
from numba import njit
from concurrent.futures import ThreadPoolExecutor
import math
import time
import os
import sys

# =====================================================================
# SEALED KERNEL — V2.0
# =====================================================================
@njit(nogil=True, fastmath=True, boundscheck=False)
def swing_liquidity_sweep_engine(
        q_open, q_high, q_low, q_close,
        min_of_day, volume, tod_baseline_vol, is_roll_date,
        initial_capital, oos_start_idx,
        w, m, d_min, d_max, v, beta, vol_mult):
    """
    Thermodynamic swing liquidity sweep engine.

    Args:
        q_open/high/low/close : float64[n]  — tick-quantised 1-min OHLC
        min_of_day            : int64[n]    — minute-of-day (0–1439, US/Eastern)
        volume                : float64[n]  — CME exchange volume per bar
        tod_baseline_vol    : float64[n]  — time-of-day 20-period trailing median volume
        is_roll_date          : bool[n]     — True on first bar of new contract
        initial_capital       : float       — starting equity (compounding)
        oos_start_idx         : int         — index where true OOS begins;
                                             Phase 1/2 locked for t < this value
        w      : int    — macro lookback in 1-min bars (anchor scan window)
        m      : int    — anchor maturity in 1-min bars (min age to be pristine)
        d_min  : float  — minimum sweep depth as % of anchor price
        d_max  : float  — maximum sweep depth before depth-death
        v      : int    — max structural checks allowed in VACUUM before time-death
        beta   : float  — reward asymmetry (R-multiple for take-profit)
        vol_mult: float — sweep volume must exceed vol_mult × median to signal

    Returns:
        equity              : float64[n] — mark-to-market equity curve
        trades              : int        — total closed trades
        total_bars_in_trade : int        — cumulative bars spent in open positions
    """
    n = len(q_close)
    capital = initial_capital
    equity = np.full(n, capital)

    TICK = 0.25
    PT_VAL = 2.00
    COMMISSION = 1.00      # $1.00 per contract round-trip (CME MNQ)
    SLIPPAGE = 0.50        # 2 ticks pessimistic market-order fill
    MIN_RISK_PTS = 2.00    # Minimum stop required; d_min already gates sweep depth
    MAX_MARGIN_PCT = 0.25    # Margin ceiling: max fraction of equity deployed as exchange margin
    MARGIN_PER_MNQ = 1250.0  # CME maintenance margin per MNQ contract ($)

    # Position state
    pos = 0
    entry_px = 0.0
    stop_px = 0.0
    tp_px = 0.0
    trades = 0
    total_bars_in_trade = 0

    # Long sweep state machine
    s_l = 0
    L_star = 0.0
    E_min = 0.0
    tau_l = 0
    sweep_vol_l = 0.0
    sweep_base_l = 0.0   # integrated ToD baseline accumulated during vacuum
    sweep_bars_l = 0

    # Short sweep state machine
    s_s = 0
    H_star = 0.0
    E_max = 0.0
    tau_s = 0
    sweep_vol_s = 0.0
    sweep_base_s = 0.0   # integrated ToD baseline accumulated during vacuum
    sweep_bars_s = 0

    # Pending execution flags
    pending_long = False
    pending_short = False
    target_L_star = 0.0
    target_E_min = 0.0
    target_H_star = 0.0
    target_E_max = 0.0

    # Step 2: Structural Regime Gate
    # 0 = no prior outcome, 1 = reclaim (success), -1 = depth death (failure)
    last_l_outcome = 0
    second_l_outcome = 0
    long_locked = False

    last_s_outcome = 0
    second_s_outcome = 0
    short_locked = False

    w_int = int(w)
    m_int = int(m)

    # Monotonic deque state (ring buffers for O(1) sliding min/max)
    deque_size = w_int + 2
    dq_l       = np.zeros(deque_size, dtype=np.int64)  # LONG: indices of sliding min (q_low)
    dq_head_l  = 0
    dq_tail_l  = 0
    dq_s       = np.zeros(deque_size, dtype=np.int64)  # SHORT: indices of sliding max (q_high)
    dq_head_s  = 0
    dq_tail_s  = 0

    # Cross-roll clamp: structural memory bounded to current contract
    last_roll_idx = -999999

    for t in range(w_int, n):

        # ------------------------------------------------------------
        # ROLL DATE: Close open position, zero structural memory.
        # Unadjusted price gaps at contract rolls create phantom P&L
        # and false pristine levels — must be purged.
        # ------------------------------------------------------------
        if is_roll_date[t]:
            last_roll_idx = t
            if pos != 0 and t >= oos_start_idx:
                # Liquidate on final print of expiring contract (MOC simulation)
                if pos > 0:
                    exit_px = q_close[t - 1] - SLIPPAGE
                    pts = exit_px - entry_px
                else:
                    exit_px = q_close[t - 1] + SLIPPAGE
                    pts = entry_px - exit_px
                capital += (pts * PT_VAL * abs(pos)) - (COMMISSION * abs(pos))
                pos = 0
                trades += 1
            # Zero all structural state
            s_l = 0;    L_star = 0.0; E_min = 0.0; tau_l = 0; sweep_vol_l = 0.0; sweep_base_l = 0.0; sweep_bars_l = 0
            s_s = 0;    H_star = 0.0; E_max = 0.0; tau_s = 0; sweep_vol_s = 0.0; sweep_base_s = 0.0; sweep_bars_s = 0
            pending_long = False
            pending_short = False
            # Purge regime locks — old contract physics don't bind new contract
            long_locked = False;  last_l_outcome = 0; second_l_outcome = 0
            short_locked = False; last_s_outcome = 0; second_s_outcome = 0
            # Flush deques — pre-roll prices are a different contract, stale indices corrupt
            dq_head_l = 0;  dq_tail_l = 0
            dq_head_s = 0;  dq_tail_s = 0
            equity[t] = capital
            continue

        # ------------------------------------------------------------
        # PHASE 1 & 2 — EXECUTION (OOS ONLY: t >= oos_start_idx)
        # During IS overhang (t < oos_start_idx): no capital deployed,
        # state machine accumulates structural memory via Phase 3.
        # ------------------------------------------------------------
        phase3_skip = False

        if t >= oos_start_idx:
            # Default: flat capital. Overwritten by Phase 2 fill/MTM if needed.
            equity[t] = capital

            # --- Phase 1: T+0 Open — resolve pending orders ---
            if pending_long and pending_short:
                # Conflicting signals cancel
                pending_long = False
                pending_short = False

            elif pending_long:
                exec_px = q_open[t]
                # Gap-trap filter: if open has already reclaimed, thesis intact
                if exec_px >= target_L_star:
                    calc_entry = exec_px + SLIPPAGE
                    calc_stop = target_E_min - TICK
                    risk_pts = (calc_entry - calc_stop) + SLIPPAGE
                    if risk_pts >= MIN_RISK_PTS:
                        qty_risk   = int((capital * 0.01) / (risk_pts * PT_VAL))
                        qty_margin = int((capital * MAX_MARGIN_PCT) / MARGIN_PER_MNQ)
                        qty = min(qty_risk, qty_margin)
                        if qty >= 1:
                            pos = qty
                            entry_px = calc_entry
                            stop_px = calc_stop
                            tp_px = np.round(
                                (calc_entry + risk_pts * beta) / TICK
                            ) * TICK
                pending_long = False

            elif pending_short:
                exec_px = q_open[t]
                if exec_px <= target_H_star:
                    calc_entry = exec_px - SLIPPAGE
                    calc_stop = target_E_max + TICK
                    risk_pts = (calc_stop - calc_entry) + SLIPPAGE
                    if risk_pts >= MIN_RISK_PTS:
                        qty_risk   = int((capital * 0.01) / (risk_pts * PT_VAL))
                        qty_margin = int((capital * MAX_MARGIN_PCT) / MARGIN_PER_MNQ)
                        qty = min(qty_risk, qty_margin)
                        if qty >= 1:
                            pos = -qty
                            entry_px = calc_entry
                            stop_px = calc_stop
                            tp_px = np.round(
                                (calc_entry - risk_pts * beta) / TICK
                            ) * TICK
                pending_short = False

            # --- Phase 2: T+0 Intrabar — manage open position ---
            if pos != 0:
                filled = False
                exit_px = 0.0

                if pos > 0:
                    if q_open[t] <= stop_px:
                        exit_px = q_open[t] - SLIPPAGE
                        filled = True
                    elif q_open[t] >= tp_px:
                        exit_px = q_open[t]
                        filled = True
                    elif q_low[t] <= stop_px:
                        exit_px = stop_px - SLIPPAGE
                        filled = True
                    elif q_high[t] >= tp_px + TICK:
                        exit_px = tp_px
                        filled = True
                else:  # pos < 0
                    if q_open[t] >= stop_px:
                        exit_px = q_open[t] + SLIPPAGE
                        filled = True
                    elif q_open[t] <= tp_px:
                        exit_px = q_open[t]
                        filled = True
                    elif q_high[t] >= stop_px:
                        exit_px = stop_px + SLIPPAGE
                        filled = True
                    elif q_low[t] <= tp_px - TICK:
                        exit_px = tp_px
                        filled = True

                if filled:
                    pts = (exit_px - entry_px) if pos > 0 else (entry_px - exit_px)
                    capital += (pts * PT_VAL * abs(pos)) - (COMMISSION * abs(pos))
                    pos = 0
                    trades += 1
                    s_l = 0
                    s_s = 0
                    equity[t] = capital
                    phase3_skip = True  # Skip Phase 3 on exit bar (V1.0 behaviour)
                else:
                    # Step 5: True intrabar MAE for mark-to-market equity
                    unrealized = (
                        (q_low[t] - entry_px) if pos > 0
                        else (entry_px - q_high[t])
                    )
                    equity[t] = capital + (unrealized * abs(pos) * PT_VAL)
                    total_bars_in_trade += 1
                    phase3_skip = True  # Skip Phase 3 while managing position

        else:
            # IS overhang: no execution, equity flat at running capital
            equity[t] = capital

        # --- Continuous Thermodynamic Tracking ---
        # Volume and baseline both accumulate on EVERY 1-min bar while in vacuum.
        # Comparing sweep_vol_l vs sweep_base_l gives integral energy vs. expected
        # energy — eliminates cross-timezone bias from pointwise terminal comparison.
        if s_l == 1:
            sweep_vol_l  += volume[t]
            sweep_base_l += tod_baseline_vol[t]
            sweep_bars_l += 1
        if s_s == 1:
            sweep_vol_s  += volume[t]
            sweep_base_s += tod_baseline_vol[t]
            sweep_bars_s += 1

        # ==========================================================
        # MANDATORY CONTINUOUS DEQUE UPDATE (runs every 1-min bar)
        # Maintains O(1) sliding min/max over [t-w, t-m].
        # Must stay current even during position management.
        # ==========================================================
        target_idx = t - m_int
        if target_idx >= last_roll_idx:
            w_start_f = max(t - w_int, last_roll_idx)

            # LONG deque
            while dq_head_l < dq_tail_l and dq_l[dq_head_l % deque_size] < w_start_f:
                dq_head_l += 1
            while dq_head_l < dq_tail_l and q_low[dq_l[(dq_tail_l - 1) % deque_size]] >= q_low[target_idx]:
                dq_tail_l -= 1
            dq_l[dq_tail_l % deque_size] = target_idx
            dq_tail_l += 1

            # SHORT deque
            while dq_head_s < dq_tail_s and dq_s[dq_head_s % deque_size] < w_start_f:
                dq_head_s += 1
            while dq_head_s < dq_tail_s and q_high[dq_s[(dq_tail_s - 1) % deque_size]] <= q_high[target_idx]:
                dq_tail_s -= 1
            dq_s[dq_tail_s % deque_size] = target_idx
            dq_tail_s += 1

        if phase3_skip:
            continue

        # ------------------------------------------------------------
        # PHASE 3: T+0 Close — STRUCTURAL MAPPING
        # Runs continuously (both IS and OOS) on every 1-min bar.
        # ------------------------------------------------------------
        t_min = min_of_day[t]
        is_rth = (t_min >= 570) and (t_min <= 960)  # 09:30–16:00 ET

        # ---- LONG STATE MACHINE ----

        if s_l == 0 and not long_locked:
            # S=0 HUNTING: deque front = index of the window minimum in O(1).
            if dq_head_l < dq_tail_l:
                min_idx = dq_l[dq_head_l % deque_size]
                min_val = q_low[min_idx]
                if q_low[t] < min_val:
                    is_pristine = True
                    # O(M) scan: only verify the maturity gap
                    scan_start = max(min_idx + 1, t - m_int + 1)
                    for i in range(scan_start, t):
                        if q_low[i] <= min_val:
                            is_pristine = False
                            break
                    if is_pristine and q_close[t - 1] >= min_val:
                        s_l = 1
                        L_star       = min_val
                        E_min        = q_low[t]
                        tau_l        = 0
                        sweep_vol_l  = volume[t]
                        sweep_base_l = tod_baseline_vol[t]
                        sweep_bars_l = 1

        elif s_l == 1:
            # S=1 VACUUM: track depth and time, check for reclaim
            # Volume tracked continuously above — only update E_min here
            if q_low[t] < E_min:
                E_min = q_low[t]
            delta_pct = ((L_star - E_min) / L_star) * 100.0

            if delta_pct > d_max:
                # Depth death: limit bids overrun — hypothesis dies
                # Update Step 2 regime gate
                second_l_outcome = last_l_outcome
                last_l_outcome = -1
                if second_l_outcome == -1:
                    # Two consecutive depth deaths — lock the long side
                    long_locked = True
                    pending_long = False
                s_l = 0

            elif tau_l > v:
                # Time death: too many structural checks without reclaim
                # Neutral outcome — does not update regime gate
                s_l = 0

            elif q_close[t] > L_star:
                # Reclaim: price closes back above the swept level
                # Step 3: Thermodynamic volume friction — integral energy vs. expected energy.
                # Compares total sweep volume against total expected volume for those exact
                # minute buckets, eliminating cross-timezone bias from a pointwise comparison.
                vol_ok = (sweep_base_l <= 0.0) or (sweep_vol_l > sweep_base_l * vol_mult)

                if delta_pct >= d_min and is_rth and vol_ok:
                    pending_long = True
                    target_L_star = L_star
                    target_E_min = E_min
                    s_s = 0  # Cancel any concurrent short setup
                    # Step 2: Reclaim success — update gate, unlock shorts
                    second_l_outcome = last_l_outcome
                    last_l_outcome = 1
                    short_locked = False
                s_l = 0

            else:
                # Failed reclaim: close remains below L_star
                tau_l += 1

        # ---- SHORT STATE MACHINE ----

        if s_s == 0 and not short_locked:
            # S=0 HUNTING: deque front = index of the window maximum in O(1).
            if dq_head_s < dq_tail_s:
                max_idx = dq_s[dq_head_s % deque_size]
                max_val = q_high[max_idx]
                if q_high[t] > max_val:
                    is_pristine = True
                    # O(M) scan: only verify the maturity gap
                    scan_start = max(max_idx + 1, t - m_int + 1)
                    for i in range(scan_start, t):
                        if q_high[i] >= max_val:
                            is_pristine = False
                            break
                    if is_pristine and q_close[t - 1] <= max_val:
                        s_s = 1
                        H_star       = max_val
                        E_max        = q_high[t]
                        tau_s        = 0
                        sweep_vol_s  = volume[t]
                        sweep_base_s = tod_baseline_vol[t]
                        sweep_bars_s = 1

        elif s_s == 1:
            # Volume tracked continuously above — only update E_max here
            if q_high[t] > E_max:
                E_max = q_high[t]
            delta_pct = ((E_max - H_star) / H_star) * 100.0

            if delta_pct > d_max:
                second_s_outcome = last_s_outcome
                last_s_outcome = -1
                if second_s_outcome == -1:
                    short_locked = True
                    pending_short = False
                s_s = 0

            elif tau_s > v:
                s_s = 0

            elif q_close[t] < H_star:
                # Step 3: Thermodynamic volume friction — integral energy vs. expected energy.
                vol_ok = (sweep_base_s <= 0.0) or (sweep_vol_s > sweep_base_s * vol_mult)

                if delta_pct >= d_min and is_rth and vol_ok:
                    pending_short = True
                    target_H_star = H_star
                    target_E_max = E_max
                    s_l = 0  # Cancel concurrent long setup
                    # Step 2: update gate, unlock longs
                    second_s_outcome = last_s_outcome
                    last_s_outcome = 1
                    long_locked = False
                s_s = 0

            else:
                tau_s += 1

    return equity, trades, total_bars_in_trade


# =====================================================================
# IS SCALAR FITNESS KERNEL — zero allocation, inline peak/DD tracking
# Identical simulation logic to swing_liquidity_sweep_engine but:
#   - No equity array allocated (saves ~4MB per IS call)
#   - No oos_start_idx — all bars executable (IS always runs full window)
#   - Returns (trades, final_capital, max_dd, total_bars_in_trade) tuple
# swing_liquidity_sweep_engine is preserved unchanged for OOS evaluation.
# =====================================================================
@njit(nogil=True, fastmath=True, boundscheck=False)
def swing_fitness_kernel(
        q_open, q_high, q_low, q_close,
        min_of_day, volume, tod_baseline_vol, is_roll_date,
        w, m, d_min, d_max, v, beta, vol_mult):
    """
    IS fitness kernel — no equity array. Returns (trades, capital, max_dd, bars_in_trade).
    """
    n = len(q_close)
    capital = 100000.0
    max_peak = capital
    max_dd   = 0.0

    TICK = 0.25
    PT_VAL = 2.00
    COMMISSION = 1.00
    SLIPPAGE = 0.50
    MIN_RISK_PTS = 2.00
    MAX_MARGIN_PCT = 0.25    # Margin ceiling: max fraction of equity deployed as exchange margin
    MARGIN_PER_MNQ = 1250.0  # CME maintenance margin per MNQ contract ($)

    pos = 0
    entry_px = 0.0
    stop_px = 0.0
    tp_px = 0.0
    trades = 0
    total_bars_in_trade = 0

    s_l = 0;  L_star = 0.0; E_min = 0.0; tau_l = 0; sweep_vol_l = 0.0; sweep_base_l = 0.0; sweep_bars_l = 0
    s_s = 0;  H_star = 0.0; E_max = 0.0; tau_s = 0; sweep_vol_s = 0.0; sweep_base_s = 0.0; sweep_bars_s = 0

    pending_long = False;  pending_short = False
    target_L_star = 0.0;   target_E_min = 0.0
    target_H_star = 0.0;   target_E_max = 0.0

    last_l_outcome = 0;  second_l_outcome = 0;  long_locked  = False
    last_s_outcome = 0;  second_s_outcome = 0;  short_locked = False

    w_int = int(w)
    m_int = int(m)

    deque_size = w_int + 2
    dq_l       = np.zeros(deque_size, dtype=np.int64)
    dq_head_l  = 0;  dq_tail_l = 0
    dq_s       = np.zeros(deque_size, dtype=np.int64)
    dq_head_s  = 0;  dq_tail_s = 0

    # Cross-roll clamp
    last_roll_idx = -999999

    for t in range(w_int, n):

        if is_roll_date[t]:
            last_roll_idx = t
            if pos != 0:
                # Liquidate on final print of expiring contract (MOC simulation)
                if pos > 0:
                    exit_px = q_close[t - 1] - SLIPPAGE
                    pts = exit_px - entry_px
                else:
                    exit_px = q_close[t - 1] + SLIPPAGE
                    pts = entry_px - exit_px
                capital += (pts * PT_VAL * abs(pos)) - (COMMISSION * abs(pos))
                if capital > max_peak:
                    max_peak = capital
                dd_cur = max_peak - capital
                if dd_cur > max_dd:
                    max_dd = dd_cur
                pos = 0
                trades += 1
            s_l = 0;   L_star = 0.0; E_min = 0.0; tau_l = 0; sweep_vol_l = 0.0; sweep_base_l = 0.0; sweep_bars_l = 0
            s_s = 0;   H_star = 0.0; E_max = 0.0; tau_s = 0; sweep_vol_s = 0.0; sweep_base_s = 0.0; sweep_bars_s = 0
            pending_long = False;  pending_short = False
            long_locked  = False;  last_l_outcome = 0; second_l_outcome = 0
            short_locked = False;  last_s_outcome = 0; second_s_outcome = 0
            dq_head_l = 0;  dq_tail_l = 0
            dq_head_s = 0;  dq_tail_s = 0
            continue

        phase3_skip = False

        if pending_long and pending_short:
            pending_long = False
            pending_short = False

        elif pending_long:
            exec_px = q_open[t]
            if exec_px >= target_L_star:
                calc_entry = exec_px + SLIPPAGE
                calc_stop  = target_E_min - TICK
                risk_pts   = (calc_entry - calc_stop) + SLIPPAGE
                if risk_pts >= MIN_RISK_PTS:
                    qty_risk   = int((capital * 0.01) / (risk_pts * PT_VAL))
                    qty_margin = int((capital * MAX_MARGIN_PCT) / MARGIN_PER_MNQ)
                    qty = min(qty_risk, qty_margin)
                    if qty >= 1:
                        pos      = qty
                        entry_px = calc_entry
                        stop_px  = calc_stop
                        tp_px    = np.round((calc_entry + risk_pts * beta) / TICK) * TICK
            pending_long = False

        elif pending_short:
            exec_px = q_open[t]
            if exec_px <= target_H_star:
                calc_entry = exec_px - SLIPPAGE
                calc_stop  = target_E_max + TICK
                risk_pts   = (calc_stop - calc_entry) + SLIPPAGE
                if risk_pts >= MIN_RISK_PTS:
                    qty_risk   = int((capital * 0.01) / (risk_pts * PT_VAL))
                    qty_margin = int((capital * MAX_MARGIN_PCT) / MARGIN_PER_MNQ)
                    qty = min(qty_risk, qty_margin)
                    if qty >= 1:
                        pos      = -qty
                        entry_px = calc_entry
                        stop_px  = calc_stop
                        tp_px    = np.round((calc_entry - risk_pts * beta) / TICK) * TICK
            pending_short = False

        if pos != 0:
            filled  = False
            exit_px = 0.0

            if pos > 0:
                if q_open[t] <= stop_px:
                    exit_px = q_open[t] - SLIPPAGE;  filled = True
                elif q_open[t] >= tp_px:
                    exit_px = q_open[t];              filled = True
                elif q_low[t] <= stop_px:
                    exit_px = stop_px - SLIPPAGE;     filled = True
                elif q_high[t] >= tp_px + TICK:
                    exit_px = tp_px;                  filled = True
            else:
                if q_open[t] >= stop_px:
                    exit_px = q_open[t] + SLIPPAGE;  filled = True
                elif q_open[t] <= tp_px:
                    exit_px = q_open[t];              filled = True
                elif q_high[t] >= stop_px:
                    exit_px = stop_px + SLIPPAGE;     filled = True
                elif q_low[t] <= tp_px - TICK:
                    exit_px = tp_px;                  filled = True

            if filled:
                pts = (exit_px - entry_px) if pos > 0 else (entry_px - exit_px)
                capital += (pts * PT_VAL * abs(pos)) - (COMMISSION * abs(pos))
                if capital > max_peak:
                    max_peak = capital
                dd_cur = max_peak - capital
                if dd_cur > max_dd:
                    max_dd = dd_cur
                pos = 0
                trades += 1
                s_l = 0;  s_s = 0
                phase3_skip = True
            else:
                total_bars_in_trade += 1
                phase3_skip = True
                # True Intrabar MAE tracking
                unrealized = (q_low[t] - entry_px) if pos > 0 else (entry_px - q_high[t])
                current_eq = capital + (unrealized * abs(pos) * PT_VAL)
                if current_eq > max_peak:
                    max_peak = current_eq
                dd_cur = max_peak - current_eq
                if dd_cur > max_dd:
                    max_dd = dd_cur

        if s_l == 1:
            sweep_vol_l  += volume[t]
            sweep_base_l += tod_baseline_vol[t]
            sweep_bars_l += 1
        if s_s == 1:
            sweep_vol_s  += volume[t]
            sweep_base_s += tod_baseline_vol[t]
            sweep_bars_s += 1

        # ==========================================================
        # MANDATORY CONTINUOUS DEQUE UPDATE (runs every 1-min bar)
        # ==========================================================
        target_idx = t - m_int
        if target_idx >= last_roll_idx:
            w_start_f = max(t - w_int, last_roll_idx)

            # LONG deque
            while dq_head_l < dq_tail_l and dq_l[dq_head_l % deque_size] < w_start_f:
                dq_head_l += 1
            while dq_head_l < dq_tail_l and q_low[dq_l[(dq_tail_l - 1) % deque_size]] >= q_low[target_idx]:
                dq_tail_l -= 1
            dq_l[dq_tail_l % deque_size] = target_idx
            dq_tail_l += 1

            # SHORT deque
            while dq_head_s < dq_tail_s and dq_s[dq_head_s % deque_size] < w_start_f:
                dq_head_s += 1
            while dq_head_s < dq_tail_s and q_high[dq_s[(dq_tail_s - 1) % deque_size]] <= q_high[target_idx]:
                dq_tail_s -= 1
            dq_s[dq_tail_s % deque_size] = target_idx
            dq_tail_s += 1

        if phase3_skip:
            continue

        # PHASE 3: STRUCTURAL MAPPING (every 1-min bar)
        t_min  = min_of_day[t]
        is_rth = (t_min >= 570) and (t_min <= 960)

        if s_l == 0 and not long_locked:
            if dq_head_l < dq_tail_l:
                min_idx = dq_l[dq_head_l % deque_size]
                min_val = q_low[min_idx]
                if q_low[t] < min_val:
                    is_pristine = True
                    # O(M) scan: only verify the maturity gap
                    scan_start = max(min_idx + 1, t - m_int + 1)
                    for i in range(scan_start, t):
                        if q_low[i] <= min_val:
                            is_pristine = False
                            break
                    if is_pristine and q_close[t - 1] >= min_val:
                        s_l = 1
                        L_star       = min_val
                        E_min        = q_low[t]
                        tau_l        = 0
                        sweep_vol_l  = volume[t]
                        sweep_base_l = tod_baseline_vol[t]
                        sweep_bars_l = 1

        elif s_l == 1:
            if q_low[t] < E_min:
                E_min = q_low[t]
            delta_pct = ((L_star - E_min) / L_star) * 100.0

            if delta_pct > d_max:
                second_l_outcome = last_l_outcome
                last_l_outcome = -1
                if second_l_outcome == -1:
                    long_locked  = True
                    pending_long = False
                s_l = 0

            elif tau_l > v:
                s_l = 0

            elif q_close[t] > L_star:
                vol_ok = (sweep_base_l <= 0.0) or (sweep_vol_l > sweep_base_l * vol_mult)

                if delta_pct >= d_min and is_rth and vol_ok:
                    pending_long  = True
                    target_L_star = L_star
                    target_E_min  = E_min
                    s_s = 0
                    second_l_outcome = last_l_outcome
                    last_l_outcome   = 1
                    short_locked     = False
                s_l = 0

            else:
                tau_l += 1

        if s_s == 0 and not short_locked:
            if dq_head_s < dq_tail_s:
                max_idx = dq_s[dq_head_s % deque_size]
                max_val = q_high[max_idx]
                if q_high[t] > max_val:
                    is_pristine = True
                    # O(M) scan: only verify the maturity gap
                    scan_start = max(max_idx + 1, t - m_int + 1)
                    for i in range(scan_start, t):
                        if q_high[i] >= max_val:
                            is_pristine = False
                            break
                    if is_pristine and q_close[t - 1] <= max_val:
                        s_s = 1
                        H_star       = max_val
                        E_max        = q_high[t]
                        tau_s        = 0
                        sweep_vol_s  = volume[t]
                        sweep_base_s = tod_baseline_vol[t]
                        sweep_bars_s = 1

        elif s_s == 1:
            if q_high[t] > E_max:
                E_max = q_high[t]
            delta_pct = ((E_max - H_star) / H_star) * 100.0

            if delta_pct > d_max:
                second_s_outcome = last_s_outcome
                last_s_outcome = -1
                if second_s_outcome == -1:
                    short_locked  = True
                    pending_short = False
                s_s = 0

            elif tau_s > v:
                s_s = 0

            elif q_close[t] < H_star:
                vol_ok = (sweep_base_s <= 0.0) or (sweep_vol_s > sweep_base_s * vol_mult)

                if delta_pct >= d_min and is_rth and vol_ok:
                    pending_short = True
                    target_H_star = H_star
                    target_E_max  = E_max
                    s_l = 0
                    second_s_outcome = last_s_outcome
                    last_s_outcome   = 1
                    long_locked      = False
                s_s = 0

            else:
                tau_s += 1

    return trades, capital, max_dd, total_bars_in_trade


# =====================================================================
# GENOME EVALUATOR — Step 5 Fitness Function
# =====================================================================
def evaluate_genome(q_open, q_high, q_low, q_close, min_of_day,
                    volume, tod_baseline_vol, is_roll_date,
                    w, m, d_min, d_max, v, beta, vol_mult):
    """
    IS fitness evaluation. oos_start_idx=0 means all bars are treated
    as executable (standard IS backtest with no lock-out).
    """
    if int(w) <= int(m):
        return -999999.0
    if d_min >= d_max:
        return -999999.0
    if v < 2 or beta <= 1.0 or vol_mult <= 0.0:
        return -999999.0

    tr, final_cap, max_dd, total_bars = swing_fitness_kernel(
        q_open, q_high, q_low, q_close,
        min_of_day, volume, tod_baseline_vol, is_roll_date,
        w, m, d_min, d_max, v, beta, vol_mult
    )

    if tr < 10:
        return -999999.0

    avg_hold = float(total_bars) / max(float(tr), 1.0)
    if avg_hold < 500.0:
        return -999999.0

    net = final_cap - 100000.0
    dd  = max_dd if max_dd > 0.0 else 1.0

    if net <= 0.0:
        return float(net - dd)

    # Step 5: Pure Calmar — honest edge per unit risk.
    # Hard gates above (min_trades=10, min_avg_hold=200 bars) define the
    # feasible region; within it, net/dd is the only objective.
    return net / dd


# =====================================================================
# DATA SANITIZER
# =====================================================================
TICK = 0.25

def quantize_and_align_data(df):
    """
    Tick-quantise OHLC and extract per-bar metadata.
    Expects df.index to be a tz-aware DatetimeIndex in US/Eastern.
    """
    q_open  = (df['open'].values  / TICK).round() * TICK
    q_high  = (df['high'].values  / TICK).round() * TICK
    q_low   = (df['low'].values   / TICK).round() * TICK
    q_close = (df['close'].values / TICK).round() * TICK
    min_of_day = (df.index.hour * 60 + df.index.minute).values.astype(np.int64)
    volume          = df['volume'].values.astype(np.float64)
    rolling_med_vol = df['tod_baseline_vol'].values.astype(np.float64)
    is_roll         = df['is_roll_date'].values.astype(np.bool_)
    return q_open, q_high, q_low, q_close, min_of_day, volume, rolling_med_vol, is_roll


# =====================================================================
# 7-GENE GENOME DEFINITION
# =====================================================================
# Gene space (Step 4 swing scaling + vol_mult addition):
#   w      : int,   [2000, 8000]   — 1-min bars lookback (1.4–5.6 days)
#   m      : int,   [400,  2760]   — 1-min bars maturity (6.7h–2 days)
#   d_min  : float, [0.10, 1.50]   — min sweep depth % (20–300 pts at NQ=20k)
#   d_max  : float, [1.00, 10.00]  — max excursion % before depth-death
#   v      : int,   [2,    200]    — structural checks in VACUUM (1-min decision checks)
#   beta   : float, [3.0,  15.0]   — reward asymmetry (absorbs overnight gap risk)
#   vol_mult: float,[0.2,  1.5]    — sweep vol must exceed vol_mult × median

GENE_BOUNDS = {
    'w':        (2000,  8000,  'int'),
    'm':        (400,   2760,  'int'),
    'd_min':    (0.10,  1.50,  'float'),
    'd_max':    (1.00,  10.00, 'float'),
    'v':        (2,     200,   'int'),
    'beta':     (3.0,   15.0,  'float'),
    'vol_mult': (0.2,   1.5,   'float'),
}

GENE_NAMES = list(GENE_BOUNDS.keys())
N_GENES = len(GENE_NAMES)

# Ensemble configuration — deploy N diverse genomes per WFO window OOS
ENSEMBLE_SIZE = 3
DIVERSITY_MIN_GENES = 3        # genome must differ on >= 3 of 7 genes
DIVERSITY_THRESHOLD_PCT = 0.15 # each differing gene >= 15% of its range


def random_genome(rng):
    g = np.zeros(N_GENES)
    for i, name in enumerate(GENE_NAMES):
        lo, hi, gtype = GENE_BOUNDS[name]
        g[i] = rng.integers(lo, hi + 1) if gtype == 'int' else rng.uniform(lo, hi)
    return g


def clip_genome(g):
    for i, name in enumerate(GENE_NAMES):
        lo, hi, gtype = GENE_BOUNDS[name]
        g[i] = np.clip(g[i], lo, hi)
        if gtype == 'int':
            g[i] = round(g[i])
    return g


def decode_genome(g):
    return {
        'w':        int(g[0]),
        'm':        int(g[1]),
        'd_min':    float(g[2]),
        'd_max':    float(g[3]),
        'v':        int(g[4]),
        'beta':     float(g[5]),
        'vol_mult': float(g[6]),
    }


def genome_distance(g1, g2):
    """Count genes differing by >= DIVERSITY_THRESHOLD_PCT of gene range."""
    n_different = 0
    for i, name in enumerate(GENE_NAMES):
        lo, hi, _ = GENE_BOUNDS[name]
        gene_range = hi - lo
        if gene_range == 0:
            continue
        if abs(g1[i] - g2[i]) / gene_range >= DIVERSITY_THRESHOLD_PCT:
            n_different += 1
    return n_different


def is_diverse_from_set(candidate, selected_set,
                        min_genes=DIVERSITY_MIN_GENES):
    """True if candidate differs from ALL selected genomes on >= min_genes."""
    for existing in selected_set:
        if genome_distance(candidate, existing) < min_genes:
            return False
    return True


def crossover_sbx(p1, p2, rng, eta=2.0):
    child = np.zeros(N_GENES)
    for i in range(N_GENES):
        if rng.random() < 0.5:
            child[i] = p1[i]
        else:
            u = rng.random()
            if u <= 0.5:
                bq = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                bq = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            child[i] = 0.5 * ((1 + bq) * p1[i] + (1 - bq) * p2[i])
    return clip_genome(child)


def mutate(g, rng, mutation_rate=0.15):
    child = g.copy()
    for i, name in enumerate(GENE_NAMES):
        if rng.random() < mutation_rate:
            lo, hi, gtype = GENE_BOUNDS[name]
            if gtype == 'int':
                child[i] = rng.integers(lo, hi + 1)
            else:
                delta = (hi - lo) * 0.2
                child[i] += rng.normal(0, delta)
    return clip_genome(child)


def tournament_select(population, fitnesses, rng, k=3):
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = indices[0]
    for idx in indices[1:]:
        if fitnesses[idx] > fitnesses[best_idx]:
            best_idx = idx
    return population[best_idx].copy()


def run_ga(q_open, q_high, q_low, q_close, min_of_day,
           volume, tod_baseline_vol, is_roll_date,
           pop_size=80, generations=50, elite_frac=0.1, seed=42,
           ensemble_size=ENSEMBLE_SIZE):
    """
    Returns: (ensemble_list, best_ever_fitness, gen_log)
      ensemble_list = [(genome_dict, fitness), ...] of length ensemble_size
    """
    rng = np.random.default_rng(seed)
    n_elite = max(2, int(pop_size * elite_frac))
    population = [random_genome(rng) for _ in range(pop_size)]

    best_ever_fitness = -999999.0
    best_ever_genome  = None
    gen_log           = []
    cached_fitnesses  = [None] * n_elite   # elite fitness cache; avoids re-evaluation
    max_workers       = os.cpu_count() or 4

    # Hall-of-fame: accumulate top feasible genomes across ALL generations
    # Capacity must be large enough to retain diverse lower-fitness genomes
    # from early generations that the converged late-generation elite prunes.
    HOF_CAPACITY = max(50, ensemble_size * 15)
    hall_of_fame = []  # list of (genome_array_copy, fitness_float)

    print(f"  [GA] {max_workers} logical cores | pop={pop_size} | gens={generations} | "
          f"elite={n_elite} | ensemble={ensemble_size}", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for gen in range(generations):
            fitnesses = np.zeros(pop_size)
            futures   = {}

            # Dispatch evaluations; elite genomes reuse cached fitness
            for i in range(pop_size):
                if gen > 0 and i < n_elite and cached_fitnesses[i] is not None:
                    fitnesses[i] = cached_fitnesses[i]
                else:
                    futures[i] = executor.submit(
                        evaluate_genome,
                        q_open, q_high, q_low, q_close, min_of_day,
                        volume, tod_baseline_vol, is_roll_date,
                        *population[i]
                    )

            for i, fut in futures.items():
                fitnesses[i] = fut.result()

            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fit = fitnesses[gen_best_idx]
            valid        = fitnesses[fitnesses > -999998.0]
            gen_mean_fit = float(np.mean(valid)) if len(valid) > 0 else -999999.0

            if gen_best_fit > best_ever_fitness:
                best_ever_fitness = gen_best_fit
                best_ever_genome  = population[gen_best_idx].copy()

            # Update hall-of-fame with feasible genomes from this generation
            for idx in range(pop_size):
                if fitnesses[idx] > -999998.0:
                    hall_of_fame.append((population[idx].copy(),
                                        float(fitnesses[idx])))

            # Prune HoF: diversity-aware dedup, keep top HOF_CAPACITY.
            # Genomes differing on < 2 genes (by >= 15% of range) are treated
            # as functional duplicates. This prevents the converged late-gen
            # population from flooding the HoF with near-clones, ensuring
            # diverse early-gen hypotheses survive for ensemble selection.
            hall_of_fame.sort(key=lambda x: x[1], reverse=True)
            deduped = []
            for g, f in hall_of_fame:
                is_dup = False
                for eg, _ in deduped:
                    if genome_distance(g, eg) < 2:   # < 2 meaningful gene diffs
                        is_dup = True
                        break
                if not is_dup:
                    deduped.append((g, f))
                if len(deduped) >= HOF_CAPACITY:
                    break
            hall_of_fame = deduped

            gen_log.append({
                'generation':   gen,
                'best_fitness': gen_best_fit,
                'mean_fitness': gen_mean_fit,
                'best_genome':  decode_genome(population[gen_best_idx]),
            })

            params = decode_genome(population[gen_best_idx])
            print(
                f"  Gen {gen:3d} | Best: {gen_best_fit:10.4f} | Mean: {gen_mean_fit:10.4f} | "
                f"w={params['w']} m={params['m']} "
                f"d=[{params['d_min']:.3f},{params['d_max']:.3f}] "
                f"v={params['v']} beta={params['beta']:.2f} "
                f"vm={params['vol_mult']:.2f}",
                flush=True
            )

            if gen == generations - 1:
                break

            sorted_idx = np.argsort(fitnesses)[::-1]
            new_pop    = [population[sorted_idx[i]].copy() for i in range(n_elite)]

            # Cache elite fitness for next generation
            for i in range(n_elite):
                cached_fitnesses[i] = float(fitnesses[sorted_idx[i]])

            while len(new_pop) < pop_size:
                p1    = tournament_select(population, fitnesses, rng)
                p2    = tournament_select(population, fitnesses, rng)
                child = crossover_sbx(p1, p2, rng)
                child = mutate(child, rng)
                new_pop.append(child)
            population = new_pop

    if best_ever_genome is None:
        best_ever_genome = population[0]

    # ── Select diverse ensemble from hall-of-fame ──
    if not hall_of_fame:
        # All genomes infeasible across all generations
        return [(decode_genome(best_ever_genome), -999999.0)], -999999.0, gen_log

    ensemble = []
    selected_arrays = []

    # Greedy: apex first, then highest-fitness passing diversity filter
    for g_arr, fit in hall_of_fame:
        if len(ensemble) == 0:
            ensemble.append((decode_genome(g_arr), fit))
            selected_arrays.append(g_arr)
        elif len(ensemble) < ensemble_size:
            if is_diverse_from_set(g_arr, selected_arrays):
                ensemble.append((decode_genome(g_arr), fit))
                selected_arrays.append(g_arr)

    # Fallback: if diversity filter yields < ensemble_size, pad with
    # best-fitness non-clone genomes (allow near-clones over running short)
    if len(ensemble) < ensemble_size:
        for g_arr, fit in hall_of_fame:
            if len(ensemble) >= ensemble_size:
                break
            already_in = any(np.allclose(g_arr, s, atol=1e-6)
                            for s in selected_arrays)
            if not already_in:
                ensemble.append((decode_genome(g_arr), fit))
                selected_arrays.append(g_arr)

    # Last resort: duplicate apex to fill remaining slots
    while len(ensemble) < ensemble_size:
        ensemble.append(ensemble[0])

    n_feasible = sum(1 for _, f in ensemble if f > -999998.0)
    print(f"  [GA] Ensemble: {len(ensemble)} genomes "
          f"({n_feasible} feasible, "
          f"filtered from {len(hall_of_fame)} HoF candidates)", flush=True)
    for k, (gd, gf) in enumerate(ensemble):
        print(f"    E{k}: fit={gf:.4f} w={gd['w']} m={gd['m']} "
              f"d=[{gd['d_min']:.3f},{gd['d_max']:.3f}] "
              f"v={gd['v']} beta={gd['beta']:.2f} vm={gd['vol_mult']:.2f}",
              flush=True)

    return ensemble, best_ever_fitness, gen_log


# =====================================================================
# ROLLING WALK-FORWARD OPTIMIZATION — V2.0
# =====================================================================

def run_wfo(df, train_months=24, trade_months=12,
            pop_size=80, generations=50, seed=42):
    """
    Rolling WFO Matrix for swing liquidity sweep engine.

    1. Train on [t - train_months, t) -> GA finds apex genome
    2. Overhang: w bars of pre-OOS data for structural map warm-up
       (Phase 1/2 locked inside Numba until true OOS boundary)
    3. Trade on [t, t + trade_months) -> locked genome, blind execution
    4. Stitch OOS segments into compounding equity curve

    Args:
        df           : DataFrame — full 1-min OHLCV with columns:
                       open, high, low, close, volume,
                       is_roll_date, tod_baseline_vol
                       Index: tz-aware DatetimeIndex (US/Eastern)
        train_months : IS window in months (default 24 — swing system needs 2yr IS)
        trade_months : OOS window in months (default 12 — sufficient for ~16 trades)
    """
    print("=" * 80)
    print("ROLLING WALK-FORWARD OPTIMIZATION MATRIX — geldbeutel V2.0")
    print("=" * 80)
    print(f"Data range:      {df.index[0]} to {df.index[-1]}")
    print(f"Total bars:      {len(df):,}")
    print(f"Train window:    {train_months} months | Trade window: {trade_months} months")
    print(f"GA:              pop={pop_size}, gens={generations}")
    print("=" * 80)

    data_start = df.index[0]
    data_end   = df.index[-1]

    # Build rolling window schedule
    windows = []
    train_start = data_start

    while True:
        train_end   = train_start + pd.DateOffset(months=train_months)
        trade_start = train_end
        trade_end   = trade_start + pd.DateOffset(months=trade_months)

        if trade_end > data_end:
            trade_end = data_end
            if trade_start >= data_end:
                break
            windows.append({
                'train_start': train_start,
                'train_end':   train_end,
                'trade_start': trade_start,
                'trade_end':   trade_end,
            })
            break

        windows.append({
            'train_start': train_start,
            'train_end':   train_end,
            'trade_start': trade_start,
            'trade_end':   trade_end,
        })
        train_start = train_start + pd.DateOffset(months=trade_months)

    print(f"\nTotal WFO windows: {len(windows)}")
    for i, win in enumerate(windows):
        print(f"  W{i:02d}: Train [{win['train_start'].date()} -> {win['train_end'].date()}) "
              f"| OOS [{win['trade_start'].date()} -> {win['trade_end'].date()})")

    oos_segments       = []
    window_results     = []
    compounding_capital = 100000.0

    for i, win in enumerate(windows):
        print(f"\n{'='*80}")
        print(f"WINDOW {i}/{len(windows)-1}: Training "
              f"[{win['train_start'].date()} -> {win['train_end'].date()})")
        print(f"{'='*80}")

        train_df = df[win['train_start']:win['train_end']]
        if len(train_df) < 5000:
            print(f"  SKIP: {len(train_df)} bars in IS window (need 5000+)")
            continue

        q_o, q_h, q_l, q_c, mod, vol, med_vol, roll = quantize_and_align_data(train_df)

        t0 = time.time()
        ensemble, apex_fitness, gen_log = run_ga(
            q_o, q_h, q_l, q_c, mod, vol, med_vol, roll,
            pop_size=pop_size, generations=generations, seed=seed + i
        )
        ga_time = time.time() - t0
        apex_genome = ensemble[0][0]  # backward-compat: first genome is apex

        print(f"\n  APEX GENOME (fitness={apex_fitness:.4f}, {ga_time:.1f}s):")
        for k, val in apex_genome.items():
            print(f"    {k}: {val}")

        # Filter ensemble to feasible genomes only
        feasible_ensemble = [(g, f) for g, f in ensemble if f > -999998.0]

        # Skip OOS if no feasible genomes.
        # Capital passes forward unchanged; window is logged with 0 OOS stats.
        if not feasible_ensemble:
            print("  SKIP OOS: all ensemble genomes infeasible — capital passed forward unchanged.")
            window_results.append({
                'window':          i,
                'train_start':     win['train_start'],
                'trade_start':     win['trade_start'],
                'trade_end':       win['trade_end'],
                'genome':          apex_genome,
                'ga_fitness':      apex_fitness,
                'oos_net':         0.0,
                'oos_dd':          0.0,
                'oos_calmar':      0.0,
                'oos_trades':      0,
                'avg_hold_bars':   0.0,
                'start_capital':   compounding_capital,
                'end_capital':     compounding_capital,
                'ensemble_size_deployed': 0,
                'ensemble_oos_nets': [],
            })
            continue

        # OOS run with integer-indexed overhang:
        #   Use max w across ensemble for structural warm-up
        n_deployed = len(feasible_ensemble)
        max_w = max(g['w'] for g, f in feasible_ensemble)
        trade_start_idx = df.index.searchsorted(win['trade_start'])
        trade_end_idx   = df.index.searchsorted(win['trade_end'])

        required_overhang = max_w * 2
        slice_start = max(0, trade_start_idx - required_overhang)

        trade_df_full = df.iloc[slice_start:trade_end_idx].copy()
        oos_start_idx = trade_start_idx - slice_start

        if len(trade_df_full) < oos_start_idx + 200:
            print(f"  SKIP: Insufficient OOS data ({len(trade_df_full)} bars, "
                  f"oos_start_idx={oos_start_idx})")
            continue

        q_o2, q_h2, q_l2, q_c2, mod2, vol2, med_vol2, roll2 = \
            quantize_and_align_data(trade_df_full)

        # ── Ensemble OOS evaluation ──
        oos_equity_curves = []
        oos_trades_list   = []
        genome_oos_nets   = []
        total_bars_all    = 0

        for k, (genome_dict, ga_fit) in enumerate(feasible_ensemble):
            eq_full, trades_full, bars_in_trade = swing_liquidity_sweep_engine(
                q_o2, q_h2, q_l2, q_c2, mod2, vol2, med_vol2, roll2,
                compounding_capital, oos_start_idx,
                genome_dict['w'],     genome_dict['m'],
                genome_dict['d_min'], genome_dict['d_max'],
                genome_dict['v'],     genome_dict['beta'],
                genome_dict['vol_mult']
            )

            oos_eq = eq_full[oos_start_idx:]
            oos_equity_curves.append(oos_eq)

            # Per-genome OOS trade count (approximate, exclude overhang)
            overhang_eq = eq_full[:oos_start_idx]
            oh_trades = int(np.sum(np.diff(overhang_eq) != 0)) \
                        if len(overhang_eq) > 1 else 0
            oos_tr = trades_full - oh_trades
            oos_trades_list.append(oos_tr)
            total_bars_all += bars_in_trade

            g_net = float(oos_eq[-1]) - compounding_capital
            genome_oos_nets.append(g_net)
            print(f"    E{k}/{n_deployed-1}: net=${g_net:,.2f} "
                  f"| trades={oos_tr} | ga_fit={ga_fit:.4f}")

        # Ensemble equity: element-wise mean of all OOS equity curves
        # Mathematically equivalent to 1/N capital allocation
        ensemble_equity = np.mean(np.array(oos_equity_curves), axis=0)
        oos_dates = trade_df_full.index[oos_start_idx:]

        if len(ensemble_equity) > 0:
            final_cap = float(ensemble_equity[-1])
            oos_net   = final_cap - compounding_capital
            oos_peak  = np.maximum.accumulate(ensemble_equity)
            oos_dd    = float(np.max(oos_peak - ensemble_equity))

            total_oos_trades = sum(oos_trades_list)
            avg_hold = float(total_bars_all) / max(float(total_oos_trades), 1.0)

            segment = pd.Series(ensemble_equity, index=oos_dates)
            oos_segments.append(segment)

            window_results.append({
                'window':          i,
                'train_start':     win['train_start'],
                'trade_start':     win['trade_start'],
                'trade_end':       win['trade_end'],
                'genome':          apex_genome,
                'ga_fitness':      apex_fitness,
                'oos_net':         oos_net,
                'oos_dd':          oos_dd,
                'oos_calmar':      oos_net / oos_dd if oos_dd > 0 else 0.0,
                'oos_trades':      total_oos_trades,
                'avg_hold_bars':   avg_hold,
                'start_capital':   compounding_capital,
                'end_capital':     final_cap,
                'ensemble_size_deployed': n_deployed,
                'ensemble_oos_nets': genome_oos_nets,
            })

            print(f"\n  ENSEMBLE OOS RESULT ({n_deployed} genomes):")
            print(f"    Capital:   ${compounding_capital:,.2f} -> ${final_cap:,.2f}")
            print(f"    Net P&L:   ${oos_net:,.2f}")
            print(f"    Max DD:    ${oos_dd:,.2f}")
            print(f"    Calmar:    {oos_net / oos_dd if oos_dd > 0 else 0:.3f}")
            print(f"    Trades:    {total_oos_trades} (total across {n_deployed} genomes)")
            print(f"    Avg Hold:  {avg_hold:.0f} bars ({avg_hold/60:.1f}h)")
            for k, net in enumerate(genome_oos_nets):
                print(f"    E{k}: net=${net:,.2f}")

            compounding_capital = final_cap

    # Stitch global equity curve
    if oos_segments:
        global_equity = pd.concat(oos_segments)
        global_equity = global_equity[~global_equity.index.duplicated(keep='last')]
        global_equity = global_equity.sort_index()
    else:
        global_equity = pd.Series(dtype=float)

    # Final report
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD OPTIMIZATION COMPLETE — V2.0")
    print(f"{'='*80}")
    if window_results:
        total_net = compounding_capital - 100000.0
        total_pct = (compounding_capital / 100000.0 - 1.0) * 100.0
        if len(global_equity) > 0:
            gp   = np.maximum.accumulate(global_equity.values)
            g_dd = float(np.max(gp - global_equity.values))
        else:
            g_dd = 0.0

        print(f"  Windows executed:   {len(window_results)}")
        print(f"  Final capital:      ${compounding_capital:,.2f}")
        print(f"  Total net P&L:      ${total_net:,.2f} ({total_pct:.1f}%)")
        print(f"  Global max DD:      ${g_dd:,.2f}")
        print(f"  Global Calmar:      {total_net / g_dd if g_dd > 0 else 0:.3f}")
        profitable = sum(1 for wr in window_results if wr['oos_net'] > 0)
        print(f"  Profitable windows: {profitable}/{len(window_results)} "
              f"({profitable / len(window_results) * 100:.0f}%)")
    else:
        print("  No windows executed.")

    return {
        'global_equity':  global_equity,
        'window_results': window_results,
        'final_capital':  compounding_capital,
    }


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == '__main__':
    DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(DATA_DIR, "NQ_CME_1min.parquet")

    if not os.path.exists(parquet_path):
        print(f"ERROR: Data file not found: {parquet_path}")
        print("Run data_loader_v2.py first to prepare the CME 1-min dataset.")
        sys.exit(1)

    print("Loading data...", flush=True)
    df = pd.read_parquet(parquet_path)

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('US/Eastern')

    # Validate required columns
    required = {'open', 'high', 'low', 'close', 'volume', 'is_roll_date', 'tod_baseline_vol'}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns in parquet: {missing}")
        print("Run data_loader_v2.py to regenerate the dataset.")
        sys.exit(1)

    print(f"Loaded {len(df):,} bars: {df.index[0]} to {df.index[-1]}")
    print(f"Roll dates: {int(df['is_roll_date'].sum())}")
    print(f"Volume coverage: {(df['volume'] > 0).mean()*100:.1f}%")

    # Numba warmup
    print("Warming up Numba JIT...", flush=True)
    _n = 5000
    _d = np.ones(_n)
    _m = np.full(_n, 600, dtype=np.int64)
    _v = np.ones(_n) * 1000.0
    _mv = np.ones(_n) * 800.0
    _r = np.zeros(_n, dtype=np.bool_)
    _ = swing_liquidity_sweep_engine(
        _d, _d, _d, _d, _m, _v, _mv, _r,
        100000.0, 0,
        3000, 500, 0.3, 2.0, 20, 5.0, 1.5
    )
    print("JIT compiled.", flush=True)

    # Trim early low-volatility data: NQ < 5000 produces noise-level sweeps
    # at d_min=0.25% (only ~12 points). Start from 2015 when NQ > 4000.
    WFO_START = '2015-01-01'
    if df.index[0] < pd.Timestamp(WFO_START, tz=df.index.tz):
        pre_trim = len(df)
        df = df[WFO_START:]
        print(f"Trimmed {pre_trim - len(df):,} bars before {WFO_START} "
              f"(low-vol NQ era). Remaining: {len(df):,} bars.")

    # Run WFO — 24/12 windows (24mo IS, 12mo OOS)
    results = run_wfo(
        df,
        train_months=24,
        trade_months=12,
        pop_size=80,
        generations=50,
    )

    # Save outputs
    if len(results['global_equity']) > 0:
        out_eq = os.path.join(DATA_DIR, "wfo_equity_curve_v2.parquet")
        results['global_equity'].to_frame('equity').to_parquet(out_eq)
        print(f"\nEquity curve saved -> wfo_equity_curve_v2.parquet")

    if results['window_results']:
        import pandas as pd
        wr_df = pd.DataFrame(results['window_results'])
        # Serialize list columns to string for CSV compatibility
        for col in ['ensemble_oos_nets']:
            if col in wr_df.columns:
                wr_df[col] = wr_df[col].apply(str)
        out_wr = os.path.join(DATA_DIR, "wfo_window_results_v2.csv")
        wr_df.to_csv(out_wr, index=False)
        print(f"Window results saved -> wfo_window_results_v2.csv")
