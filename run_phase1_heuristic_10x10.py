"""
run_phase1_heuristic_10x10.py  (REVERTED to surrogate-based, 2026-08-10)
============================================================================
This file was found (at the start of a hybrid-DRAGON5 ARR session) to be
importing `casmo4_bwr10x10` directly from `bwr10x10_dragon5` instead of
`bwr10x10_surrogate` -- i.e. Phase 1.2 was calling REAL DRAGON5 on
essentially every non-repeat trial (not just incumbent-improving ones),
with a default Dragon5Config (no compute_ppf=True), which crashed with
`TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'`
after ~57 CPU-minutes on the very first real evaluation (out["PPF"] was
None since compute_ppf defaults to False).

Beyond the crash, per-trial real DRAGON5 calls are computationally
infeasible for Phase 1 (each real call costs tens of minutes to over an
hour with the full validated burnup schedule) -- this is exactly the
cost problem the hybrid design (arr_hybrid_dragon5.py,
run_phase2_hybrid_dragon5.py) was built to avoid, by calling real
DRAGON5 only at incumbent-update moments.

This version reverts Phase 1 to use the SURROGATE (bwr10x10_surrogate,
now re-anchored to DRAGON5's own computed values on the 3 calibration
patterns -- see arr_hybrid_dragon5.SurrogateCalibrationState) for every
trial, exactly as run_phase2_heuristic_10x10.py's Phase 1 usage already
does. Phase 1's own final solution then gets its FIRST real DRAGON5
verification automatically when passed into run_phase2_hybrid_dragon5's
`run()` (its very first objective() call always triggers a real check --
see that module's docstring). If you want Phase 1's result verified by
DRAGON5 before Phase 2 even starts, call `dragon5.casmo4_bwr10x10`
directly on this function's returned pattern once, yourself -- do not
wire per-trial calls back into the objective functions below.

Everything else (Heuristics 1-4 / 35-var space, Phase 1.1-then-1.2
split, target_score=0.0 stopping rule) is unchanged from the original
design.
"""

import time

import numpy as np

from bwr10x10_geometry_heuristic import build_grid_heuristic, N_VARS, TYPE_MASK
from bwr10x10_geometry import N_TYPES
from bwr10x10_surrogate import casmo4_bwr10x10, pattern_to_grids, average_enrichment
try:
    from arr_optimizer_categorical_centroid05 import arr_optimize_categorical, x_to_type_idx
except ImportError:
    # (2026-11-03) That module (the original surrogate-only ARR driver) is NOT part of the published repo and is not
    # needed by the greedy Phase 1 / Phase 2 campaign (only the old surrogate-only Phase 1.1/1.2 functions below use it).
    # Import stays optional so constants (E_LO..KINF_HI) and constraint_set1_check can be used without it.
    def _missing(*a, **k):
        raise ImportError("arr_optimizer_categorical_centroid05.py is required only for the legacy surrogate-only "
                          "Phase 1.1/1.2 functions in this file; the greedy campaign does not use them.")
    arr_optimize_categorical = x_to_type_idx = _missing

E_LO, E_HI = 4.25, 4.36
NGAD_LO, NGAD_HI = 16, 18
# k_inf,max upper bound. The source paper's constraint is 1.11 (its own CASMO-4 values for the two
# best patterns are 1.10151 / 1.10446). With the SYM oracle the same patterns give 1.1003 / 1.1110
# (10-step schedule), i.e. the source's own RL_PPF_1_4 pattern would sit 0.1 % ABOVE 1.11. The bound is
# therefore raised by ~1 % (oracle-vs-CASMO offset on the source patterns: +0.6 / -0.1 / -0.6 %; step-size
# sensitivity of k_inf,max ~ +-1 %) so that the source patterns are feasible under the oracle: 1.12.
# LEGACY (gap-less oracle and the paper's own value): 1.11000
KINF_HI = 1.12000
E_WEIGHT = 10.0
KINF_WEIGHT = 10.0
PPF_WEIGHT = 10.0


def _term(v, lo, hi):
    return min(0.0, v - lo) + min(0.0, hi - v)


def _term_upper(v, hi):
    return min(0.0, hi - v)


def constraint_set1_check(pattern):
    enrich, gad, water = pattern_to_grids(pattern)
    E = average_enrichment(enrich, water)
    n_gad = int(np.sum((~water) & (gad > 0)))
    feasible = (E_LO <= E <= E_HI) and (NGAD_LO <= n_gad <= NGAD_HI)
    return E, n_gad, feasible


# ---------------------------------------------------------------------
# Phase 1.1 -- constraint set 1 only (no CASMO4/DRAGON5 at all), on the
# 35-var heuristic space. Unchanged.
# ---------------------------------------------------------------------

def objective_phase1_1(x):
    type_idx = x_to_type_idx(x)
    pattern = build_grid_heuristic(type_idx)
    E, n_gad, _ = constraint_set1_check(pattern)
    e_term = _term(E, E_LO, E_HI)
    ngad_term = _term(n_gad, NGAD_LO, NGAD_HI)
    return (E_WEIGHT * e_term + 1.0 * ngad_term) / (E_WEIGHT + 1.0)


def run_phase1_1(max_trials=20000, seed=0, verbose=True):
    t0 = time.perf_counter()
    x_best, score_best, history, info = arr_optimize_categorical(
        objective_phase1_1, n_positions=N_VARS, n_types=N_TYPES,
        maximize=True, max_trials=max_trials, seed=seed,
        target_score=0.0, target_tol=1e-9, verbose=verbose,
        type_mask=TYPE_MASK,
    )
    elapsed = time.perf_counter() - t0
    type_idx = x_to_type_idx(x_best)
    pattern = build_grid_heuristic(type_idx)
    E, n_gad, feasible = constraint_set1_check(pattern)
    if verbose:
        print(f"\n=== Phase 1.1 (heuristic, 35-var) result (seed={seed}) ===")
        print(f"F1.1 = {score_best:.8f}   E={E:.4f}%   N_GAD={n_gad}   feasible={feasible}")
        print(f"trials={info['n_trials']}  objective evals={info['n_c4c']} "
              f"(all free -- no CASMO4/DRAGON5 needed)  time={elapsed:.4f}s")
    return x_best, score_best, pattern, info, feasible


# ---------------------------------------------------------------------
# Phase 1.2 -- constraint set 2, gated on constraint set 1. Uses the
# SURROGATE (fast) for every trial -- see module docstring for why real
# DRAGON5 must NOT be called per-trial here.
# ---------------------------------------------------------------------

def make_objective_phase1_2(ppf_c=1.6):
    call_counter = {"n": 0}

    def objective_phase1_2(x):
        type_idx = x_to_type_idx(x)
        pattern = build_grid_heuristic(type_idx)
        _, _, cs1_feasible = constraint_set1_check(pattern)
        if not cs1_feasible:
            return -1e9
        call_counter["n"] += 1
        out = casmo4_bwr10x10(pattern)  # surrogate -- fast, no real DRAGON5 here
        kinf_term = _term_upper(out["k_inf_max"], KINF_HI)
        ppf_term = _term_upper(out["PPF"], ppf_c)
        return (KINF_WEIGHT * kinf_term + PPF_WEIGHT * ppf_term) / (KINF_WEIGHT + PPF_WEIGHT)

    return objective_phase1_2, call_counter


def run_phase1_2(init_x, max_trials=20000, seed=0, ppf_c=1.6, verbose=True):
    objective_phase1_2, call_counter = make_objective_phase1_2(ppf_c)
    init_score = objective_phase1_2(init_x)

    t0 = time.perf_counter()
    x_best, score_best, history, info = arr_optimize_categorical(
        objective_phase1_2, n_positions=N_VARS, n_types=N_TYPES,
        maximize=True, max_trials=max_trials, seed=seed,
        target_score=0.0, target_tol=1e-9, verbose=verbose,
        init_x=init_x, init_score=init_score, type_mask=TYPE_MASK,
    )
    elapsed = time.perf_counter() - t0

    type_idx = x_to_type_idx(x_best)
    pattern = build_grid_heuristic(type_idx)
    out = casmo4_bwr10x10(pattern)  # surrogate
    _, _, cs1_feasible = constraint_set1_check(pattern)
    cs2_feasible = out["k_inf_max"] <= KINF_HI and out["PPF"] < ppf_c
    feasible = cs1_feasible and cs2_feasible

    if verbose:
        print(f"\n=== Phase 1.2 (heuristic, 35-var) result (seed={seed}, PPF<{ppf_c}) ===")
        print(f"F1.2 = {score_best:.8f}")
        print(f"E={out['E']:.4f}%  N_GAD={out['N_GAD']}  "
              f"k_inf_max={out['k_inf_max']:.5f}  PPF={out['PPF']:.4f}  "
              f"[surrogate values -- real DRAGON5 verification happens once, "
              f"automatically, when this result is passed into Phase 2]")
        print(f"constraint-set-1 feasible = {cs1_feasible} (must always be True)")
        print(f"constraint-set-2 feasible (surrogate) = {cs2_feasible}")
        print(f"trials={info['n_trials']}  real surrogate calls={call_counter['n']}  time={elapsed:.4f}s")
    return x_best, score_best, pattern, info, call_counter["n"], feasible


def run(max_trials=20000, seed=0, ppf_c=1.6, verbose=True):
    x1, score1, pattern1, info1, feas1 = run_phase1_1(max_trials, seed, verbose)
    if not feas1:
        if verbose:
            print("Phase 1.1 failed to satisfy constraint set 1 within the trial budget.")
        return None
    x2, score2, pattern2, info2, real_c4c_2, feas2 = run_phase1_2(
        x1, max_trials, seed, ppf_c, verbose)
    if not feas2:
        if verbose:
            print("Phase 1.2 failed to satisfy constraint set 2 (surrogate) within the trial budget.")
        return None
    return pattern2, x2, real_c4c_2, feas2


if __name__ == "__main__":
    run()
