"""
Diagnostic: Loose genome trade frequency scan — geldbeutel V2.0
================================================================
Approach A: run swing_fitness_kernel with maximally permissive parameters
on the full 2015-2024 dataset to determine maximum achievable signal frequency.

Also scans d_min from 0.10 to 1.50 to characterize signal elasticity.
"""

import sys, os
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DATA_DIR)

from wfo_matrix_v2 import swing_fitness_kernel, quantize_and_align_data

# ---- Load data ----
parquet_path = os.path.join(DATA_DIR, "NQ_CME_1min.parquet")
print(f"Loading {parquet_path} ...")
df = pd.read_parquet(parquet_path)
print(f"  Full dataset: {len(df):,} bars  ({df.index[0]} -> {df.index[-1]})")

# Trim to 2015-2024 (exclude current partial 2025-2026 OOS)
WFO_START = '2015-01-01'
WFO_END   = '2025-01-01'
df = df[WFO_START:WFO_END]
print(f"  Trimmed to {WFO_START} - {WFO_END}: {len(df):,} bars  (~{len(df)/(252*390):.1f} years)")

# ---- Prepare arrays ----
q_open, q_high, q_low, q_close, min_of_day, volume, tod_baseline_vol, is_roll_date = \
    quantize_and_align_data(df)

YEARS = len(df) / (252 * 390)  # approximate trading years

# ---- JIT warmup ----
print("\nWarming up Numba JIT ...")
_n = 5000
_d = np.ones(_n)
_m = np.full(_n, 600, dtype=np.int64)
_v = np.ones(_n) * 1000.0
_mv = np.ones(_n) * 800.0
_r = np.zeros(_n, dtype=np.bool_)
swing_fitness_kernel(_d, _d, _d, _d, _m, _d, _mv, _r,
                     2000, 400, 0.25, 8.0, 200, 3.0, 0.5)
print("  JIT ready.")

# ---- Approach A: Max-permissive base genome ----
print("\n" + "="*60)
print("APPROACH A: MAXIMUM PERMISSIVE GENOME (d_min=0.10)")
print("="*60)
print(f"  w=2000, m=400, d_min=0.10, d_max=8.0, v=200, beta=3.0, vol_mult=0.5")

tr, final_cap, max_dd, total_bars = swing_fitness_kernel(
    q_open, q_high, q_low, q_close, min_of_day, volume, tod_baseline_vol, is_roll_date,
    2000, 400, 0.10, 8.0, 200, 3.0, 0.5
)
avg_hold = total_bars / max(tr, 1)
print(f"  Trades: {tr}  ({tr/YEARS:.1f}/yr)")
print(f"  Avg hold: {avg_hold:.0f} bars  ({avg_hold/60:.1f}h)")
print(f"  Net P&L: ${final_cap - 100000:.0f}  |  Max DD: ${max_dd:.0f}")

# ---- d_min scan ----
print("\n" + "="*60)
print("D_MIN SCAN (all else: w=2000, m=400, d_max=8.0, v=200, beta=3.0, vol_mult=0.5)")
print("="*60)
print(f"{'d_min':>8} | {'Trades':>8} | {'Trades/yr':>10} | {'Avg hold':>10} | {'Net P&L':>10}")
print("-"*60)

for d_min in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 1.25, 1.50]:
    tr, fc, dd, tb = swing_fitness_kernel(
        q_open, q_high, q_low, q_close, min_of_day, volume, tod_baseline_vol, is_roll_date,
        2000, 400, d_min, 8.0, 200, 3.0, 0.5
    )
    ah = tb / max(tr, 1)
    print(f"  {d_min:>5.2f}  | {tr:>8} | {tr/YEARS:>10.1f} | {ah:>8.0f}b | ${fc-100000:>8.0f}")

# ---- vol_mult scan ----
print("\n" + "="*60)
print("VOL_MULT SCAN (d_min=0.25, all else loose)")
print("="*60)
print(f"{'vol_mult':>10} | {'Trades':>8} | {'Trades/yr':>10} | {'Avg hold':>10}")
print("-"*55)

for vm in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
    tr, fc, dd, tb = swing_fitness_kernel(
        q_open, q_high, q_low, q_close, min_of_day, volume, tod_baseline_vol, is_roll_date,
        2000, 400, 0.25, 8.0, 200, 3.0, vm
    )
    ah = tb / max(tr, 1)
    print(f"  {vm:>6.2f}    | {tr:>8} | {tr/YEARS:>10.1f} | {ah:>8.0f}b")

# ---- w (lookback) scan ----
print("\n" + "="*60)
print("W (LOOKBACK) SCAN (d_min=0.25, vol_mult=0.5, all else loose)")
print("="*60)
print(f"{'w':>8} | {'Trades':>8} | {'Trades/yr':>10} | {'Avg hold':>10}")
print("-"*50)

for w in [2000, 4000, 6000, 8000, 10000, 15000, 20000]:
    tr, fc, dd, tb = swing_fitness_kernel(
        q_open, q_high, q_low, q_close, min_of_day, volume, tod_baseline_vol, is_roll_date,
        w, 400, 0.25, 8.0, 200, 3.0, 0.5
    )
    ah = tb / max(tr, 1)
    print(f"  {w:>6}  | {tr:>8} | {tr/YEARS:>10.1f} | {ah:>8.0f}b")

print("\nDiagnostic complete.")
