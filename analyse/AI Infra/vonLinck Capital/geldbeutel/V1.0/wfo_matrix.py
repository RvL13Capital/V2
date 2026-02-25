"""
Rolling Walk-Forward Optimization Matrix
==========================================
Mixed-Integer Genetic Algorithm (MI-GA) + Rolling WFO
around the sealed Liquidity Sweep kernel.

Theory reference: TTDLS Section V — The Evolutionary Imperative.

Train: 6 months of data → GA finds apex genome
Trade: 3 months of unseen data → locked genome trades blind
Overhang: w bars of pre-trade data fed for structural mapping, execution locked

The OOS equity segments are stitched into a single compounding curve.
"""

import numpy as np
import pandas as pd
from numba import njit
import time
import os
import sys

# =====================================================================
# IMPORT THE SEALED KERNEL
# =====================================================================
# Inline to avoid import path issues — the kernel is small and must be
# exactly as audited. No modifications.

@njit
def absolute_liquidity_sweep_engine(q_open, q_high, q_low, q_close, min_of_day,
                                    w, m, d_min, d_max, v, beta):
    n = len(q_close)
    capital = 100000.0
    equity = np.full(n, capital)

    TICK = 0.25
    PT_VAL = 2.00
    COMMISSION = 1.00
    SLIPPAGE = 0.50
    MIN_RISK_PTS = 2.00

    pos = 0
    entry_px = 0.0; stop_px = 0.0; tp_px = 0.0
    trades = 0

    s_l = 0; L_star = 0.0; E_min = 0.0; tau_l = 0
    s_s = 0; H_star = 0.0; E_max = 0.0; tau_s = 0

    pending_long = False; pending_short = False
    target_L_star = 0.0; target_E_min = 0.0
    target_H_star = 0.0; target_E_max = 0.0

    w_int = int(w)
    m_int = int(m)

    for t in range(w_int, n):

        # PHASE 1: T_0 OPEN
        if pending_long and pending_short:
            pending_long = False; pending_short = False
        elif pending_long:
            exec_px = q_open[t]
            if exec_px >= target_L_star:
                calc_entry = exec_px + SLIPPAGE
                calc_stop = target_E_min - TICK
                risk_pts = (calc_entry - calc_stop) + SLIPPAGE
                if risk_pts >= MIN_RISK_PTS:
                    qty = int((capital * 0.01) / (risk_pts * PT_VAL))
                    if qty >= 1:
                        pos = min(qty, 20)
                        entry_px = calc_entry
                        stop_px = calc_stop
                        tp_px = np.round((calc_entry + (risk_pts * beta)) / TICK) * TICK
            pending_long = False
        elif pending_short:
            exec_px = q_open[t]
            if exec_px <= target_H_star:
                calc_entry = exec_px - SLIPPAGE
                calc_stop = target_E_max + TICK
                risk_pts = (calc_stop - calc_entry) + SLIPPAGE
                if risk_pts >= MIN_RISK_PTS:
                    qty = int((capital * 0.01) / (risk_pts * PT_VAL))
                    if qty >= 1:
                        pos = -min(qty, 20)
                        entry_px = calc_entry
                        stop_px = calc_stop
                        tp_px = np.round((calc_entry - (risk_pts * beta)) / TICK) * TICK
            pending_short = False

        # PHASE 2: T_0 INTRABAR
        if pos != 0:
            filled = False
            exit_px = 0.0

            if pos > 0:
                if q_open[t] <= stop_px:
                    exit_px = q_open[t] - SLIPPAGE; filled = True
                elif q_open[t] >= tp_px:
                    exit_px = q_open[t]; filled = True
                elif q_low[t] <= stop_px:
                    exit_px = stop_px - SLIPPAGE; filled = True
                elif q_high[t] >= tp_px + TICK:
                    exit_px = tp_px; filled = True
            elif pos < 0:
                if q_open[t] >= stop_px:
                    exit_px = q_open[t] + SLIPPAGE; filled = True
                elif q_open[t] <= tp_px:
                    exit_px = q_open[t]; filled = True
                elif q_high[t] >= stop_px:
                    exit_px = stop_px + SLIPPAGE; filled = True
                elif q_low[t] <= tp_px - TICK:
                    exit_px = tp_px; filled = True

            if filled:
                pts = (exit_px - entry_px) if pos > 0 else (entry_px - exit_px)
                capital += (pts * PT_VAL * abs(pos)) - (COMMISSION * abs(pos))
                pos = 0; trades += 1
                equity[t] = capital
                s_l = 0; s_s = 0
                continue

            diff = (q_close[t] - entry_px) if pos > 0 else (entry_px - q_close[t])
            equity[t] = capital + (diff * abs(pos) * PT_VAL)
            continue
        else:
            equity[t] = capital

        # PHASE 3: T_0 CLOSE
        t_min = min_of_day[t]
        is_rth = (t_min >= 570) and (t_min <= 960)

        # LONG
        if s_l == 0:
            min_val = q_low[t - w_int]
            min_idx = t - w_int
            for i in range(t - w_int, t - m_int + 1):
                if q_low[i] <= min_val:
                    min_val = q_low[i]; min_idx = i
            is_pristine = True
            for i in range(min_idx + 1, t):
                if q_low[i] <= min_val:
                    is_pristine = False; break
            if is_pristine and q_low[t] < min_val and q_close[t - 1] >= min_val:
                s_l = 1; L_star = min_val; E_min = q_low[t]; tau_l = 0

        if s_l == 1:
            if q_low[t] < E_min: E_min = q_low[t]
            delta_pct = ((L_star - E_min) / L_star) * 100.0
            if delta_pct > d_max: s_l = 0
            elif tau_l > v: s_l = 0
            elif q_close[t] > L_star:
                if delta_pct >= d_min and is_rth:
                    pending_long = True; target_L_star = L_star; target_E_min = E_min; s_s = 0
                s_l = 0
            else:
                tau_l += 1

        # SHORT
        if s_s == 0:
            max_val = q_high[t - w_int]
            max_idx = t - w_int
            for i in range(t - w_int, t - m_int + 1):
                if q_high[i] >= max_val:
                    max_val = q_high[i]; max_idx = i
            is_pristine = True
            for i in range(max_idx + 1, t):
                if q_high[i] >= max_val:
                    is_pristine = False; break
            if is_pristine and q_high[t] > max_val and q_close[t - 1] <= max_val:
                s_s = 1; H_star = max_val; E_max = q_high[t]; tau_s = 0

        if s_s == 1:
            if q_high[t] > E_max: E_max = q_high[t]
            delta_pct = ((E_max - H_star) / H_star) * 100.0
            if delta_pct > d_max: s_s = 0
            elif tau_s > v: s_s = 0
            elif q_close[t] < H_star:
                if delta_pct >= d_min and is_rth:
                    pending_short = True; target_H_star = H_star; target_E_max = E_max; s_l = 0
                s_s = 0
            else:
                tau_s += 1

    return equity, trades


# =====================================================================
# GENOME EVALUATOR (Dual-Gradient Fitness)
# =====================================================================
def evaluate_genome(q_open, q_high, q_low, q_close, min_of_day,
                    w, m, d_min, d_max, v, beta):
    if int(w) <= int(m): return -999999.0
    if d_min >= d_max: return -999999.0
    if v < 1 or beta <= 0.5: return -999999.0

    eq, tr = absolute_liquidity_sweep_engine(
        q_open, q_high, q_low, q_close, min_of_day,
        w, m, d_min, d_max, v, beta
    )

    net = eq[-1] - 100000.0
    if tr < 30: return -999999.0

    peak = np.maximum.accumulate(eq)
    dd = np.max(peak - eq)
    if dd <= 0: dd = 1.0

    if net < 0:
        return net - dd
    return net / dd


# =====================================================================
# DATA SANITIZER
# =====================================================================
TICK = 0.25

def quantize_and_align_data(df):
    """
    Forces data onto MNQ tick grid and extracts minute-of-day.
    Expects df.index to be DatetimeIndex in US/Eastern.
    """
    q_open = (df['open'].values / TICK).round() * TICK
    q_high = (df['high'].values / TICK).round() * TICK
    q_low = (df['low'].values / TICK).round() * TICK
    q_close = (df['close'].values / TICK).round() * TICK
    min_of_day = (df.index.hour * 60 + df.index.minute).values.astype(np.int64)
    return q_open, q_high, q_low, q_close, min_of_day


# =====================================================================
# MIXED-INTEGER GENETIC ALGORITHM
# =====================================================================

# Gene space definition (TTDLS Section II)
# w:     int,   [72, 576]    — 6h to 48h in 5-min bars
# m:     int,   [6, 144]     — 30min to 12h minimum anchor age
# d_min: float, [0.02, 0.50] — 0.02% to 0.50% minimum sweep depth
# d_max: float, [0.10, 2.00] — 0.10% to 2.00% maximum excursion
# v:     int,   [1, 10]      — 1 to 10 bars close-below tolerance
# beta:  float, [1.0, 8.0]   — 1R to 8R reward asymmetry

GENE_BOUNDS = {
    'w':     (72,   576,   'int'),
    'm':     (6,    144,   'int'),
    'd_min': (0.02, 0.50,  'float'),
    'd_max': (0.10, 2.00,  'float'),
    'v':     (1,    10,    'int'),
    'beta':  (1.0,  8.0,   'float'),
}

GENE_NAMES = list(GENE_BOUNDS.keys())
N_GENES = len(GENE_NAMES)


def random_genome(rng):
    """Generate a single random genome respecting types and bounds."""
    g = np.zeros(N_GENES)
    for i, name in enumerate(GENE_NAMES):
        lo, hi, gtype = GENE_BOUNDS[name]
        if gtype == 'int':
            g[i] = rng.integers(lo, hi + 1)
        else:
            g[i] = rng.uniform(lo, hi)
    return g


def clip_genome(g):
    """Clip genome to bounds and snap integers."""
    for i, name in enumerate(GENE_NAMES):
        lo, hi, gtype = GENE_BOUNDS[name]
        g[i] = np.clip(g[i], lo, hi)
        if gtype == 'int':
            g[i] = round(g[i])
    return g


def decode_genome(g):
    """Unpack genome array into named parameters."""
    return {
        'w': int(g[0]),
        'm': int(g[1]),
        'd_min': float(g[2]),
        'd_max': float(g[3]),
        'v': int(g[4]),
        'beta': float(g[5]),
    }


def crossover_sbx(p1, p2, rng, eta=2.0):
    """Simulated Binary Crossover — respects mixed-integer topology."""
    child = np.zeros(N_GENES)
    for i in range(N_GENES):
        if rng.random() < 0.5:
            child[i] = p1[i]
        else:
            u = rng.random()
            if u <= 0.5:
                beta_q = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta_q = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            child[i] = 0.5 * ((1 + beta_q) * p1[i] + (1 - beta_q) * p2[i])
    return clip_genome(child)


def mutate(g, rng, mutation_rate=0.15):
    """Polynomial mutation respecting bounds."""
    child = g.copy()
    for i, name in enumerate(GENE_NAMES):
        if rng.random() < mutation_rate:
            lo, hi, gtype = GENE_BOUNDS[name]
            if gtype == 'int':
                child[i] = rng.integers(lo, hi + 1)
            else:
                # Polynomial mutation
                delta = (hi - lo) * 0.2
                child[i] += rng.normal(0, delta)
    return clip_genome(child)


def tournament_select(population, fitnesses, rng, k=3):
    """Tournament selection — picks best of k random candidates."""
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = indices[0]
    for idx in indices[1:]:
        if fitnesses[idx] > fitnesses[best_idx]:
            best_idx = idx
    return population[best_idx].copy()


def run_ga(q_open, q_high, q_low, q_close, min_of_day,
           pop_size=80, generations=40, elite_frac=0.1, seed=42):
    """
    Mixed-Integer GA over the 6D genome space.
    Returns: (best_genome_dict, best_fitness, generation_log)
    """
    rng = np.random.default_rng(seed)
    n_elite = max(2, int(pop_size * elite_frac))

    # Initialize population
    population = [random_genome(rng) for _ in range(pop_size)]

    best_ever_fitness = -999999.0
    best_ever_genome = None
    gen_log = []

    for gen in range(generations):
        # Evaluate
        fitnesses = np.array([
            evaluate_genome(q_open, q_high, q_low, q_close, min_of_day,
                            *population[i])
            for i in range(pop_size)
        ])

        # Track best
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]
        gen_mean_fit = np.mean(fitnesses[fitnesses > -999998.0]) if np.any(fitnesses > -999998.0) else -999999.0

        if gen_best_fit > best_ever_fitness:
            best_ever_fitness = gen_best_fit
            best_ever_genome = population[gen_best_idx].copy()

        gen_log.append({
            'generation': gen,
            'best_fitness': gen_best_fit,
            'mean_fitness': gen_mean_fit,
            'best_genome': decode_genome(population[gen_best_idx]),
        })

        # Print progress
        params = decode_genome(population[gen_best_idx])
        print(f"  Gen {gen:3d} | Best: {gen_best_fit:10.2f} | Mean: {gen_mean_fit:10.2f} | "
              f"w={params['w']} m={params['m']} d=[{params['d_min']:.3f},{params['d_max']:.3f}] "
              f"v={params['v']} β={params['beta']:.2f}", flush=True)

        # Last generation — skip breeding
        if gen == generations - 1:
            break

        # Elitism
        sorted_indices = np.argsort(fitnesses)[::-1]
        new_pop = [population[sorted_indices[i]].copy() for i in range(n_elite)]

        # Breed
        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitnesses, rng)
            p2 = tournament_select(population, fitnesses, rng)
            child = crossover_sbx(p1, p2, rng)
            child = mutate(child, rng)
            new_pop.append(child)

        population = new_pop

    if best_ever_genome is None:
        # Every genome in every generation was invalid (all -999999).
        # Return a random genome from the final population as fallback.
        # The caller (run_wfo) will see fitness=-999999 and the window
        # will produce 0 trades, which is handled gracefully downstream.
        best_ever_genome = population[0]

    return decode_genome(best_ever_genome), best_ever_fitness, gen_log


# =====================================================================
# ROLLING WALK-FORWARD OPTIMIZATION
# =====================================================================

def run_wfo(df, train_months=6, trade_months=3,
            pop_size=80, generations=40, seed=42):
    """
    Rolling WFO Matrix.

    1. Train on [t - train_months, t) → GA finds apex genome
    2. Trade on [t, t + trade_months) → locked genome on unseen data
    3. Overhang: w bars of pre-trade data for structural mapping
    4. Stitch OOS equity segments into compounding curve

    Args:
        df: DataFrame with 5-min OHLCV, index = DatetimeIndex (US/Eastern)
        train_months: IS training window in months
        trade_months: OOS trading window in months

    Returns:
        results: dict with compounding curve, per-window stats, apex genomes
    """
    print("=" * 80)
    print("ROLLING WALK-FORWARD OPTIMIZATION MATRIX")
    print("=" * 80)
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df):,}")
    print(f"Train window: {train_months} months | Trade window: {trade_months} months")
    print(f"GA: pop={pop_size}, gens={generations}")
    print("=" * 80)

    # Build window schedule
    data_start = df.index[0]
    data_end = df.index[-1]

    windows = []
    train_start = data_start

    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        trade_start = train_end
        trade_end = trade_start + pd.DateOffset(months=trade_months)

        if trade_end > data_end:
            # Clamp to data end for the final truncated window
            trade_end = data_end
            if trade_start >= data_end:
                break
            # Append final window then exit — no further sliding
            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'trade_start': trade_start,
                'trade_end': trade_end,
            })
            break

        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'trade_start': trade_start,
            'trade_end': trade_end,
        })

        train_start = train_start + pd.DateOffset(months=trade_months)

    print(f"\nTotal WFO windows: {len(windows)}")
    for i, w in enumerate(windows):
        print(f"  Window {i}: Train [{w['train_start'].date()} → {w['train_end'].date()}) "
              f"| Trade [{w['trade_start'].date()} → {w['trade_end'].date()})")

    # Execute each window
    oos_segments = []
    window_results = []
    compounding_capital = 100000.0

    for i, win in enumerate(windows):
        print(f"\n{'='*80}")
        print(f"WINDOW {i}/{len(windows)-1}: Training on "
              f"[{win['train_start'].date()} → {win['train_end'].date()})")
        print(f"{'='*80}")

        # Extract training data
        train_df = df[win['train_start']:win['train_end']]
        if len(train_df) < 500:
            print(f"  SKIP: Only {len(train_df)} bars in training window (need 500+)")
            continue

        q_o, q_h, q_l, q_c, mod = quantize_and_align_data(train_df)

        # Run GA
        t0 = time.time()
        apex_genome, apex_fitness, gen_log = run_ga(
            q_o, q_h, q_l, q_c, mod,
            pop_size=pop_size, generations=generations, seed=seed + i
        )
        ga_time = time.time() - t0

        print(f"\n  APEX GENOME (fitness={apex_fitness:.2f}, {ga_time:.1f}s):")
        for k, val in apex_genome.items():
            print(f"    {k}: {val}")

        # Extract OOS data with overhang
        w_bars = apex_genome['w']
        overhang_start = win['trade_start'] - pd.Timedelta(minutes=5 * w_bars)
        trade_df_full = df[overhang_start:win['trade_end']]

        if len(trade_df_full) < w_bars + 100:
            print(f"  SKIP: Insufficient OOS data ({len(trade_df_full)} bars)")
            continue

        q_o2, q_h2, q_l2, q_c2, mod2 = quantize_and_align_data(trade_df_full)

        # Find the index where true OOS starts (after overhang)
        oos_mask = trade_df_full.index >= win['trade_start']
        oos_start_idx = np.argmax(oos_mask)

        # Run kernel on full window (overhang + OOS)
        eq_full, trades_full = absolute_liquidity_sweep_engine(
            q_o2, q_h2, q_l2, q_c2, mod2,
            apex_genome['w'], apex_genome['m'],
            apex_genome['d_min'], apex_genome['d_max'],
            apex_genome['v'], apex_genome['beta']
        )

        # Isolate OOS-only trade count: trades that closed during the overhang
        # inflate trades_full. Approximate OOS trades by counting equity changes
        # in the OOS segment only (each filled bar produces a step change).
        oos_equity_full = eq_full[oos_start_idx:]
        # Count bars where capital actually changed (a trade closed on that bar)
        overhang_equity = eq_full[:oos_start_idx]
        overhang_trades = int(np.sum(np.diff(overhang_equity) != 0)) if len(overhang_equity) > 1 else 0
        oos_trades_only = trades_full - overhang_trades

        # Extract only the OOS equity segment
        oos_equity = eq_full[oos_start_idx:]
        oos_dates = trade_df_full.index[oos_start_idx:]

        # Scale to compounding capital
        if len(oos_equity) > 0:
            scale_factor = compounding_capital / 100000.0
            oos_pnl = (oos_equity - 100000.0) * scale_factor
            oos_equity_scaled = compounding_capital + oos_pnl

            segment = pd.Series(oos_equity_scaled, index=oos_dates)
            oos_segments.append(segment)

            final_cap = oos_equity_scaled[-1]
            oos_net = final_cap - compounding_capital
            oos_peak = np.maximum.accumulate(oos_equity_scaled)
            oos_dd = np.max(oos_peak - oos_equity_scaled)

            window_results.append({
                'window': i,
                'train_start': win['train_start'],
                'trade_start': win['trade_start'],
                'trade_end': win['trade_end'],
                'genome': apex_genome,
                'ga_fitness': apex_fitness,
                'oos_net': oos_net,
                'oos_dd': oos_dd,
                'oos_calmar': oos_net / oos_dd if oos_dd > 0 else 0.0,
                'oos_trades': oos_trades_only,  # OOS-only, overhang excluded
                'start_capital': compounding_capital,
                'end_capital': final_cap,
            })

            print(f"\n  OOS RESULT:")
            print(f"    Capital: ${compounding_capital:,.2f} → ${final_cap:,.2f}")
            print(f"    Net P&L: ${oos_net:,.2f}")
            print(f"    Max DD:  ${oos_dd:,.2f}")
            print(f"    Calmar:  {oos_net / oos_dd if oos_dd > 0 else 0:.2f}")
            print(f"    Trades:  {oos_trades_only} (OOS only)")

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
    print(f"WALK-FORWARD OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    if len(window_results) > 0:
        total_net = compounding_capital - 100000.0
        total_return_pct = (compounding_capital / 100000.0 - 1) * 100
        global_peak = np.maximum.accumulate(global_equity.values)
        global_dd = np.max(global_peak - global_equity.values) if len(global_equity) > 0 else 0

        print(f"  Windows executed: {len(window_results)}")
        print(f"  Final capital:    ${compounding_capital:,.2f}")
        print(f"  Total net P&L:    ${total_net:,.2f} ({total_return_pct:.1f}%)")
        print(f"  Global max DD:    ${global_dd:,.2f}")
        print(f"  Global Calmar:    {total_net / global_dd if global_dd > 0 else 0:.2f}")

        profitable = sum(1 for wr in window_results if wr['oos_net'] > 0)
        print(f"  Profitable windows: {profitable}/{len(window_results)} "
              f"({profitable/len(window_results)*100:.0f}%)")
    else:
        print("  No windows executed.")

    return {
        'global_equity': global_equity,
        'window_results': window_results,
        'final_capital': compounding_capital,
    }


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == '__main__':
    # Load data
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(DATA_DIR, "NQ_CFD_5min_2017_2026.parquet")

    if not os.path.exists(parquet_path):
        print(f"ERROR: Data file not found: {parquet_path}")
        print("Run dukascopy_downloader.py first.")
        sys.exit(1)

    print("Loading data...", flush=True)
    df = pd.read_parquet(parquet_path)

    # Ensure index is timezone-aware (UTC from Dukascopy)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    # Convert to US/Eastern for RTH gating
    df.index = df.index.tz_convert('US/Eastern')

    print(f"Loaded {len(df):,} bars: {df.index[0]} to {df.index[-1]}")

    # Numba warmup (compile once before timing)
    print("Warming up Numba JIT...", flush=True)
    warmup_n = 1000
    dummy = np.ones(warmup_n)
    dummy_mod = np.full(warmup_n, 600, dtype=np.int64)
    _ = absolute_liquidity_sweep_engine(dummy, dummy, dummy, dummy, dummy_mod,
                                        100, 10, 0.1, 0.5, 2, 2.0)
    print("JIT compiled.", flush=True)

    # Run WFO
    results = run_wfo(
        df,
        train_months=6,
        trade_months=3,
        pop_size=80,
        generations=40,
    )

    # Save results
    if len(results['global_equity']) > 0:
        results['global_equity'].to_frame('equity').to_parquet(
            os.path.join(DATA_DIR, "wfo_equity_curve.parquet")
        )
        print(f"\nEquity curve saved to wfo_equity_curve.parquet")

    # Save window results
    if results['window_results']:
        wr_df = pd.DataFrame(results['window_results'])
        wr_df.to_csv(os.path.join(DATA_DIR, "wfo_window_results.csv"), index=False)
        print(f"Window results saved to wfo_window_results.csv")
