# Design: Gene Bounds Tighten + Infeasible Genome Fix — geldbeutel V2.0

**Date**: 2026-02-25
**Status**: Pending approval

---

## Problem

Two WFO runs (old fitness: -36.8%, pure Calmar: -13.5%) both show:
- 6/21 windows fully infeasible (GA never finds a feasible genome)
- IS-OOS inverse correlation (highest IS Calmar → worst OOS)
- OOS signal: 1–6 trades per 6-month window

**Root cause (Approach A diagnostic)**:
At max-permissive parameters (d_min=0.25, all else loose), the system generates ~16 trades/year over 2015–2024. But the GA routinely evolves toward large-w (10k–20k) + high vol_mult (1.5–3.0) combinations. These reduce signal to 3–6 trades/year — below the 10-trade IS gate. Once the entire population returns -999999, the GA is trapped with no fitness gradient to escape.

| Gene at max current bound | Trades/10yr | Trades/yr | vs. IS gate (10) |
|--------------------------|-------------|-----------|-----------------|
| w=20000 | 64 | 6.4 | BELOW |
| vol_mult=3.0 | 26 | 2.6 | BELOW |
| w=8000 (new cap) | 88 | 8.8 | marginal |
| vol_mult=1.5 (new cap) | 94 | 9.4 | marginal |

---

## Changes

### 1. Gene Bounds: w [2000, 20000] → [2000, 8000]
The diagnostic shows w>8000 reduces signal to <9/yr independently. Physical justification: a 1-min bar lookback of 8000 bars = 5.6 days of structural memory. TTDLS sweeps on swing timeframes don't require 14-day history; 5–6 days captures the relevant pristine levels. The CLAUDE.md physical meaning becomes "1.4–5.6 days macro lookback."

### 2. Gene Bounds: vol_mult [0.5, 3.0] → [0.5, 1.5]
At vol_mult=3.0, only 26 trades/10yr = 2.6/yr. The thermodynamic friction gate is conceptually sound, but vol_mult=3x is too restrictive — it filters out real sweeps with moderate (not extreme) volume signatures. Capping at 1.5x allows genuine forced-liquidation detection while keeping signal density viable.

### 3. Fix: Infeasible Genome OOS Skip
Currently, when the GA returns fitness=-999999, the infeasible genome is **still deployed to OOS**. W5 accidentally made +$14,778 this way, distorting the equity curve. Fix: if `apex_fitness == -999999`, skip OOS entirely and pass capital forward unchanged. This is a correctness fix independent of bound changes.

### 4. CLAUDE.md Updates
- Gene bounds table: w and vol_mult rows
- Known Constraints table: updated w description

---

## Expected Impact

- Infeasible windows: 6/21 → estimated 2–3/21 (cannot eliminate without 24/12 windows or lower min_trades)
- OOS trade counts: slight increase (fewer windows trapped in sparse-signal space)
- No change to the kernel, fitness function, or WFO protocol

---

## What This Does NOT Fix
- IS-OOS inverse correlation: bounds tightening doesn't address regime instability
- Statistical significance: 6-month OOS with 1–6 trades is still low
- If infeasibility remains above 3–4 windows: next step is 24/12 WFO windows (Option 2)

---

## Files Changed
1. `wfo_matrix_v2.py` — GENE_BOUNDS dict, comment lines 867–874, OOS skip logic
2. `CLAUDE.md` — gene bounds tables (Step 4 and 7-gene table)
