"""
run_qubo_hybrid_dragon5.py
=============================
QUBO-based quantum-inspired hybrid ARR, now split into two proper phases
(previous versions of this file had only a single Phase 2 loop that
verified its own random starting point for real on the fly):

  * Phase 1 -- find a REAL-DRAGON5-feasible starting point. QUBO joint
    rounding still guarantees constraint-set-1 (E, N_GAD) and GAD-GAD
    adjacency on every candidate, as always. The objective MINIMIZED is
    a continuous surrogate-predicted penalty for constraint set 2
    (k_inf_max <= kinf_hi, PPF < ppf_c):

        penalty = max(0, (k_inf_pred - kinf_hi)/kinf_hi)
                + max(0, (PPF_pred  - ppf_c) /ppf_c)

    -- 0 exactly when the surrogate believes the candidate is fully
    feasible. Real DRAGON5 is called ONLY when penalty reaches 0
    (mirroring Phase 2's "only verify surrogate-predicted improvements"
    cost discipline, but for feasibility instead of CL). Every real
    call folds into the SAME SurrogateCalibrationState Phase 2 will use
    (recalibrating the linear k_inf_max(E)/CL_proxy(E) models and,
    if refit_ppf, the PPF diffusion model too), regardless of whether
    that call turns out feasible -- an infeasible real result is
    exactly where the surrogate was most wrong, so it's valuable
    calibration data, not just confirmed-good points. Phase 1 continues
    -- NOT until the surrogate merely thinks a point is feasible, but --
    until REAL DRAGON5 itself confirms one.

  * Phase 2 -- maximize CL_proxy starting from Phase 1's confirmed
    real-feasible point. Because that point was ALREADY real-verified
    in Phase 1 (and that same real call already returned CL_proxy, not
    just k_inf_max/PPF), Phase 2 seeds HybridVerifiedObjective's
    incumbent state DIRECTLY from it -- no redundant "verify the start
    point again" DRAGON5 call, unlike earlier versions of this pipeline
    (and unlike the classical run_phase2_hybrid_dragon5.py, which still
    does this).

This was motivated directly by a real failure mode observed this
session: under ppf_c=1.4 (a much narrower feasible region than 1.6,
per the source paper's own Table 7 -- GA found only 1 feasible pattern,
SA found 0), the OLD single-phase design repeatedly spent full ~60-90
minute real DRAGON5 calls verifying candidates the surrogate wrongly
believed were feasible (real PPF came back ~1.52/1.54 against a 1.4
threshold -- nowhere close), because there was no dedicated feasibility-
seeking search phase with its own gradient signal toward constraint set
2 -- only a maximize-CL loop that happened to also gate on feasibility.
"""
import json
import threading
import time

import numpy as np

from bwr10x10_geometry_heuristic import VAR_CELLS, VAR_ALLOWED, N_VARS, build_grid_heuristic
from bwr10x10_geometry import PIN_TYPES, N_TYPES
from bwr10x10_surrogate import GRID_N, pattern_to_grids
import bwr10x10_surrogate as surrogate
from bwr10x10_reference_patterns import REFERENCE
from run_phase1_heuristic_10x10 import KINF_HI, E_LO, E_HI, NGAD_LO, NGAD_HI

# (2026-11-03) Target CL_proxy values for the two PPF problems, from the SYM
# oracle (bwr10x10_dragon5.py: channel box + 0.71 cm half gap + 40% in-channel
# void, PPF at BOL, 10-step schedule, CL = BU/29.436 MW/tU).
# With this lattice the source paper's own panel labels are CONFIRMED (the
# earlier "inverted labels" finding was an artifact of the gap-less lattice):
#   paper's RL_PPF_1_4 pattern: DRAGON5 BOL PPF = 1.357 (<1.4), CL = 1475.7 d
#   paper's RL_PPF_1_6 pattern: DRAGON5 BOL PPF = 1.535 (<1.6), CL = 1488.0 d
# (paper's own CASMO-4 values: PPF 1.367/1.57, CL 1462/1478.)
TARGET_CL_PPF_1_4 = 1475.7  # paper's "RL_PPF_1_4" pattern under the SYM oracle
TARGET_CL_PPF_1_6 = 1488.0  # paper's "RL_PPF_1_6" pattern under the SYM oracle
# LEGACY (gap-less oracle, 2026-08-16): 1465.2 (pattern RL_PPF_1_6) / 1466.9 (pattern RL_PPF_1_4)

import bwr10x10_dragon5 as dragon5
import arr_hybrid_dragon5 as hy
import arr_quantum_inspired as qi

# CHANGE (2026-11-03): the SA sampler is only used by the QUBO code paths
# (Phase 0 pool generation, legacy QUBO phases); the greedy Phase 1 / Phase 2
# loops do not need it, so a missing dimod/dwave install must not block import.
try:
    from dwave.samplers import SimulatedAnnealingSampler
except ImportError:
    try:
        from neal import SimulatedAnnealingSampler
    except ImportError:
        SimulatedAnnealingSampler = None

_, _, WATER_MASK = pattern_to_grids(REFERENCE)
MULTIPLICITIES = qi.compute_multiplicities(VAR_CELLS)
ADJACENCY_PAIRS = qi.build_adjacency_pairs(VAR_CELLS, WATER_MASK, GRID_N)


def _run_dragon5_with_heartbeat(pattern, cfg, phase_label, call_no, heartbeat_sec, verbose):
    """Standalone heartbeat-wrapped real DRAGON5 call. Mirrors
    HybridVerifiedObjective's own private method (arr_hybrid_dragon5.py)
    -- duplicated (not imported) because Phase 1 needs this before any
    HybridVerifiedObjective instance exists."""
    result_box = {}
    stop_event = threading.Event()
    t_start = time.time()

    def _heartbeat():
        while not stop_event.wait(heartbeat_sec):
            elapsed = time.time() - t_start
            print(f"    [{phase_label}] ... still running real DRAGON5 "
                  f"(call #{call_no}), {elapsed:.0f}s elapsed "
                  f"({elapsed/60:.1f} min)", flush=True)

    def _worker():
        try:
            result_box["out"] = dragon5.casmo4_bwr10x10(pattern, cfg)
        except BaseException as e:            # keep the real error (2026-11-07); re-raised below
            result_box["err"] = e

    hb_thread = None
    if heartbeat_sec and verbose:
        hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        hb_thread.start()
    worker_thread = threading.Thread(target=_worker)
    worker_thread.start()
    worker_thread.join()
    stop_event.set()
    if hb_thread is not None:
        hb_thread.join(timeout=1)

    if "err" in result_box:
        raise result_box["err"]
    elapsed = time.time() - t_start
    if verbose:
        print(f"    [{phase_label}] real DRAGON5 call #{call_no} "
              f"finished in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    return result_box["out"]


def _compute_rmsd(c, var_allowed):
    """RMSD of the centroid from its uniform (0.5) reference, restricted
    to allowed (i,k) entries -- mirrors arr_quantum_inspired.move_seed's
    internal calculation exactly, exposed standalone here for the two
    reset-check sites that don't otherwise call move_seed this round."""
    allowed_mask = np.zeros_like(c, dtype=bool)
    for i, allowed in enumerate(var_allowed):
        for k in allowed:
            allowed_mask[i, k] = True
    diffs = c[allowed_mask] - 0.5
    return float(np.sqrt(np.mean(diffs ** 2))) if diffs.size else 0.0


def run_greedy_phase1_real_feasible(calib, kinf_hi, ppf_c, seed=0,
                                     max_rounds=20000, dragon5_cfg=None,
                                     verbose=True, heartbeat_sec=60,
                                     max_dragon5_calls=None, reset_window=20,
                                     save_path="phase1_confirmed_feasible.json"):
    """Phase 1 (2026-08-16, full redesign, replacing the QUBO/SA-based
    run_qubo_phase1_real_feasible above): a classical-ARR-style loop
    using greedy_round() instead of QUBO+simulated annealing.

    Motivation: the QUBO approach's whole day of tuning (E/N_GAD soft
    penalties, no-perturbation, multi-candidate sampling, incumbent
    re-scoring, deterministic rmsd-reset, threshold relaxation) kept
    running into the SAME underlying wall -- one-hot and GAD-adjacency,
    encoded as QUBO PENALTY terms, need large coefficients to enforce
    reliably, and that large scale swamps Phi(x)'s own much smaller
    signal (confirmed directly: even fully-converged centroid gave
    SA's own lowest-energy sample only ~40-45% agreement with the
    incumbent, unmoved by 40x more sweeps or far colder beta_range --
    a structural single-spin-flip barrier issue, not a tunable one).

    This redesign sidesteps the whole problem: one-hot and GAD-
    adjacency are enforced by CONSTRUCTION via greedy_round()'s
    priority-order scan (see its docstring) -- no QUBO, no penalty
    terms, no large coefficients, hence nothing for Phi to compete
    against. What Phi(x) drove before, ptb(i,k) drives directly here --
    this IS classical ARR's own per-position argmax rounding, extended
    only by a priority rule for the GAD-adjacency edge case.

    E, N_GAD, k_inf_max, and PPF are all evaluated as SOFT penalty
    terms (identical shape to the QUBO version's combined_penalty)
    summed into one Phase 1 ARR objective. Real DRAGON5 verification
    triggers when this objective reaches exactly 0.0 (every one of the
    four sub-penalties independently zero, not a loosened tolerance --
    unlike the QUBO version, there is no SA-driven noise here to
    accommodate).

    stuck/reset: stuck counts CONSECUTIVE rounds where this round's
    freshly-rounded type_idx exactly equals the incumbent (x_best) --
    the literal classical-ARR definition. Reset fires probabilistically
    with P = min(1, stuck/reset_window) * rmsd. Unlike the QUBO
    version's SA-based rounding, greedy_round() is a DETERMINISTIC
    function of ptb (itself drawn fresh from c each round) with no
    thermal/annealing barrier standing between a converged centroid and
    the rounding actually reproducing it -- so this literal definition
    is expected to actually fire as c converges, unlike in the QUBO
    version where it was confirmed structurally dead.

    On a real-rejected candidate: the surrogate is recalibrated (as
    always), and per explicit instruction, BOTH the current incumbent
    AND this round's own candidate are re-scored under the freshly
    recalibrated surrogate before the standard incumbent-update
    comparison below decides whether this candidate becomes the new
    best solution -- there is no special-case exclusion for rejected
    candidates here (contrast the QUBO version's fix, which explicitly
    keeps a rejected candidate OUT of incumbency); a rejected candidate
    can still become best solution here if its penalty, freshly
    computed, is genuinely the lowest available.

    max_rounds (default 20000, raised from 2000): logging was changed
    at the same time to make this practical -- see the note just above
    the round loop's opening for why (printing every round at this
    scale would be unusably huge).

    save_path (default "phase1_confirmed_feasible.json"): on success
    (real-feasible confirmed), the result is written to this file
    before returning -- pattern, type_idx, real k_inf_max/PPF/CL_proxy,
    E, round/call counts, seed, ppf_c, and a timestamp. Phase 1 is
    treated as a standalone, complete step here (2026-08-16): it saves
    and returns rather than continuing on into Phase 2's own loop --
    Phase 2's own logic is being reconsidered separately given this
    redesign, so nothing here should assume it runs next. Pass
    save_path=None to skip writing.
    """
    if dragon5_cfg is None:
        dragon5_cfg = dragon5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None)

    rng = np.random.default_rng(seed)
    c = qi.init_centroid(VAR_ALLOWED, N_TYPES)

    penalty_best = float("inf")
    type_idx_best = None
    n_dragon5_calls = 0
    n_dragon5_rejected = 0
    stuck = 0
    log = []
    # Re-call guard (2026-11-03): remember every pattern already sent to
    # DRAGON5 in this Phase 1 run. A pattern whose surrogate penalty is
    # 0 again after a recalibration is NOT sent a second time (its real
    # result is already known); it is treated as non-improving instead.
    # All oracle evaluations are also appended to <save_path>_oracle_log.json
    # immediately, so a run that ends without a feasible start still
    # keeps the real DRAGON5 numbers it paid for.
    evaluated = {}
    n_dup_skipped = 0
    import os as _os
    oracle_log_path = (_os.path.splitext(save_path)[0] + "_oracle_log.json"
                       if save_path else None)

    if verbose:
        print("\n" + "#" * 70, flush=True)
        print("# GREEDY PHASE 1 (deterministic priority-round + surrogate-penalty "
              "ARR loop)", flush=True)
        print("#" * 70, flush=True)

    def combined_penalty(pattern, E, n_gad):
        if E < E_LO:
            pe = (E_LO - E) / E_LO
        elif E > E_HI:
            pe = (E - E_HI) / E_HI
        else:
            pe = 0.0
        if n_gad < NGAD_LO:
            pn = (NGAD_LO - n_gad) / NGAD_LO
        elif n_gad > NGAD_HI:
            pn = (n_gad - NGAD_HI) / NGAD_HI
        else:
            pn = 0.0
        surr_out = surrogate.casmo4_bwr10x10(pattern)
        pk = max(0.0, (surr_out["k_inf_max"] - kinf_hi) / kinf_hi)
        pp = max(0.0, (surr_out["PPF"] - ppf_c) / ppf_c)
        return pe + pn + pk + pp, surr_out

    for round_i in range(1, max_rounds + 1):
        ptb = qi.perturb(c, VAR_ALLOWED, rng, mode="mult")
        type_idx = qi.greedy_round(ptb, VAR_ALLOWED, ADJACENCY_PAIRS, qi.GAD_TYPE_IDS)
        _, detail = qi.check_feasibility(
            type_idx, VAR_ALLOWED, PIN_TYPES, MULTIPLICITIES, ADJACENCY_PAIRS,
            check_e_range=False, check_ngad_range=False)
        E, n_gad = detail["E"], detail["N_GAD"]
        pattern = build_grid_heuristic(np.asarray(type_idx))
        penalty, surr_out = combined_penalty(pattern, E, n_gad)
        rmsd_now = _compute_rmsd(c, VAR_ALLOWED)

        # CHANGE (2026-08-16): removed the unconditional per-round print
        # that used to fire here every single round -- with max_rounds
        # raised to 20000, printing every round produces an unusably
        # huge log dominated by near-duplicate lines (the vast majority
        # of rounds just re-confirm the current plateau). Logging now
        # happens ONLY at genuinely informative events: a new best
        # (below), a RESET (below), or a real DRAGON5 verification
        # (further below) -- exactly the three places the search's
        # actual trajectory changes.

        dup_key = tuple(int(k) for k in type_idx)
        if penalty <= 0.0 and dup_key in evaluated:
            n_dup_skipped += 1
            if verbose and (n_dup_skipped <= 5 or n_dup_skipped % 200 == 0):
                prev = evaluated[dup_key]
                print(f"  [PHASE1] Round {round_i}: surrogate says feasible again for a "
                      f"pattern already sent to DRAGON5 (call #{prev['call']}, real "
                      f"PPF={prev['real_PPF']}) -- NOT re-called "
                      f"[{n_dup_skipped} such skips so far]", flush=True)
            penalty = float("inf")   # never becomes / displaces the incumbent
        elif penalty <= 0.0:
            if max_dragon5_calls is not None and n_dragon5_calls >= max_dragon5_calls:
                raise hy.Dragon5CallBudgetReached(
                    f"max_dragon5_calls={max_dragon5_calls} reached during Phase 1 -- "
                    f"no real-feasible start found yet ({n_dragon5_calls} real calls, "
                    f"{n_dragon5_rejected} rejected)."
                )
            n_dragon5_calls += 1
            if verbose:
                now = time.strftime("%H:%M:%S")
                print(f"  [PHASE1] [Round {round_i}] [DRAGON5 verify #{n_dragon5_calls}] "
                      f"@ {now} -- surrogate predicts feasible (k_inf={surr_out['k_inf_max']:.5f}, "
                      f"PPF={surr_out['PPF']:.3f}) -- running real DRAGON5 "
                      f"(compute_ppf={dragon5_cfg.compute_ppf}, "
                      f"compute_cl={dragon5_cfg.compute_cl}) ...", flush=True)
            real_out = _run_dragon5_with_heartbeat(
                pattern, dragon5_cfg, "PHASE1", n_dragon5_calls, heartbeat_sec, verbose)
            log.append({"round": round_i, "n_dragon5_calls": n_dragon5_calls, "E": E,
                        "real_k_inf_max": real_out["k_inf_max"], "real_PPF": real_out["PPF"],
                        "real_CL_proxy": real_out["CL_proxy"],
                        "type_idx": [int(k) for k in type_idx],
                        "surrogate_PPF": float(surr_out["PPF"]),
                        "surrogate_k_inf_max": float(surr_out["k_inf_max"])})
            evaluated[dup_key] = {"call": n_dragon5_calls, "real_PPF": real_out["PPF"]}
            if oracle_log_path is not None:
                try:
                    with open(oracle_log_path, "w") as _f:
                        json.dump(log, _f, indent=1, default=float)
                except Exception as _e:
                    print(f"    (warning: could not write {oracle_log_path}: {_e})", flush=True)

            if (real_out["k_inf_max"] is not None and real_out["PPF"] is not None
                    and real_out["CL_proxy"] is not None):
                calib.add_verified_point(E, real_out["k_inf_max"], real_out["PPF"],
                                          real_out["CL_proxy"])
                if calib.refit_ppf:
                    calib.register_pattern_for_ppf_refit(pattern, real_out["PPF"])
                    calib._refit_ppf()
                if verbose:
                    print(f"    (folded into surrogate calibration -- refit #{calib.n_refits}, "
                          f"{len(calib.points)} verified points total)", flush=True)

            real_feasible = (real_out["k_inf_max"] is not None and real_out["PPF"] is not None
                              and real_out["k_inf_max"] <= kinf_hi and real_out["PPF"] < ppf_c)
            if real_feasible:
                if verbose:
                    print(f"    -> REAL FEASIBLE. Phase 1 complete: {n_dragon5_calls} real "
                          f"call(s) ({n_dragon5_rejected} rejected), {round_i} rounds total.",
                          flush=True)
                result = {
                    "type_idx": type_idx, "pattern": pattern,
                    "real_k_inf_max": real_out["k_inf_max"], "real_PPF": real_out["PPF"],
                    "real_CL_proxy": real_out["CL_proxy"], "E": E,
                    "q4c": round_i, "n_dragon5_calls": n_dragon5_calls,
                    "n_dragon5_rejected": n_dragon5_rejected, "log": log,
                }
                if save_path is not None:
                    save_record = dict(result)
                    save_record["seed"] = seed
                    save_record["ppf_c"] = ppf_c
                    save_record["kinf_hi"] = kinf_hi
                    save_record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(save_path, "w") as f:
                        json.dump(save_record, f, indent=2)
                    if verbose:
                        print(f"    (saved to {save_path})", flush=True)
                return result
            else:
                n_dragon5_rejected += 1
                if verbose:
                    print(f"    -> DRAGON5 says INFEASIBLE (k_inf_max={real_out['k_inf_max']}, "
                          f"PPF={real_out['PPF']}) -- surrogate was wrong here; recalibrated, "
                          f"re-scoring incumbent and this round's candidate under the updated "
                          f"surrogate before comparing.", flush=True)
                # Re-score BOTH the existing incumbent and this round's
                # own candidate under the just-recalibrated surrogate
                # (per explicit instruction) -- E/N_GAD are structural,
                # unaffected by recalibration, only k_inf/PPF predictions
                # change. No special exclusion for this rejected
                # candidate: it competes for incumbency on equal footing
                # below, same as any other round's candidate.
                penalty, surr_out = combined_penalty(pattern, E, n_gad)
                if type_idx_best is not None:
                    pattern_best = build_grid_heuristic(np.asarray(type_idx_best))
                    _, detail_best = qi.check_feasibility(
                        type_idx_best, VAR_ALLOWED, PIN_TYPES, MULTIPLICITIES,
                        ADJACENCY_PAIRS, check_e_range=False, check_ngad_range=False)
                    old_penalty_best = penalty_best
                    penalty_best, _ = combined_penalty(
                        pattern_best, detail_best["E"], detail_best["N_GAD"])
                    if verbose:
                        print(f"    (incumbent re-scored under recalibrated surrogate: "
                              f"penalty_best {old_penalty_best:.4f} -> {penalty_best:.4f})",
                              flush=True)

        # Standard incumbent update: lower penalty wins, unconditionally
        # (including rejected candidates re-scored just above).
        improved = penalty < penalty_best
        if improved:
            penalty_best = penalty
            type_idx_best = type_idx
            if verbose:
                print(f"  [PHASE1] Round {round_i}: new best penalty={penalty:.4f} "
                      f"(E={E:.4f}, N_GAD={n_gad}, "
                      f"surrogate k_inf={surr_out['k_inf_max']:.5f}, "
                      f"PPF={surr_out['PPF']:.3f})  [rmsd={rmsd_now:.4f}]", flush=True)

        # stuck: literal count of consecutive rounds whose freshly-
        # rounded type_idx exactly equals the (possibly just-updated)
        # incumbent.
        repeat_match = (type_idx_best is not None and type_idx == type_idx_best)
        if repeat_match:
            stuck += 1
        else:
            stuck = 0

        c, alpha, rmsd = (qi.move_seed(c, type_idx_best, VAR_ALLOWED)
                           if type_idx_best is not None else (c, 0.0, 0.0))
        reset_prob = min(1.0, stuck / reset_window) * rmsd
        if rng.random() < reset_prob:
            stuck_before_reset = stuck
            c = qi.init_centroid(VAR_ALLOWED, N_TYPES)
            stuck = 0
            if verbose:
                print(f"  [PHASE1] Round {round_i}: RESET centroid to uniform "
                      f"(stuck was {stuck_before_reset}, rmsd={rmsd:.4f}, "
                      f"P_reset={reset_prob:.4f})", flush=True)

    raise RuntimeError(
        f"Phase 1 exhausted max_rounds={max_rounds} without finding a "
        f"real-DRAGON5-feasible starting point ({n_dragon5_calls} real calls made, "
        f"{n_dragon5_rejected} rejected). Try a different seed, a larger "
        f"max_rounds, or relax ppf_c."
    )


def run_qubo_phase1_real_feasible(calib, kinf_hi, ppf_c, seed=0,
                                   max_qubo_calls=2000, num_reads=400,
                                   num_sweeps=500,
                                   dragon5_cfg=None, verbose=True,
                                   heartbeat_sec=60, max_dragon5_calls=None,
                                   reset_window=20,
                                   surrogate_feasible_threshold=0.02):
    """Phase 1: search (QUBO joint-rounding + surrogate-penalty ARR loop)
    until REAL DRAGON5 confirms a constraint-set-2-feasible candidate.
    Returns a dict with the confirmed pattern/type_idx AND its real
    (2026-08-16: num_reads doubled 200->400 for more per-round diversity.
    num_sweeps was briefly tried at 2000 too, then reverted back to 1000
    -- more sweeps means each individual read cools further and locks in
    more deterministically, which REDUCES diversity across reads, the
    opposite of what doubling num_reads was for. num_sweeps had also
    never actually been threaded through to the sampler at all before
    today; see qubo_round_multi's docstring.)
    k_inf_max/PPF/CL_proxy (Phase 2 needs no further verification of
    this point). Raises hy.Dragon5CallBudgetReached (caught by the
    caller, same ARRStopSignal convention as Phase 2) if the real-call
    budget runs out first.

    surrogate_feasible_threshold (default 0.0010): real DRAGON5
    verification triggers once combined_penalty < this value (strict,
    not <=). Demanding an exact 0.0 doesn't reflect the surrogate's own
    noise floor -- confirmed directly from Phase 0's own held-out
    validation: PPF's RMS relative error is ~2.8% (0.0390 absolute,
    against the 1.4 target), roughly 30x larger than the original
    0.0010 threshold, k_inf_max's is much smaller (~0.21%).
    combined_penalty is a SUM of both quantities' relative penalties,
    so it inherits PPF's error as its dominant term -- demanding
    penalty<0.0010 was demanding roughly 30x more precision than the
    surrogate can actually deliver, observed directly: a real run
    plateaued in the 0.015-0.05 range for 195+ rounds without ever
    triggering a single real DRAGON5 call. Raised in steps by request
    (0.0010 -> 0.03 -> 0.02 -> 0.01 -> 0.02) -- still well under PPF's
    own measured
    error scale (0.0390 absolute)."""
    if dragon5_cfg is None:
        dragon5_cfg = dragon5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None)

    rng = np.random.default_rng(seed)
    sampler = SimulatedAnnealingSampler()
    c = qi.init_centroid(VAR_ALLOWED, N_TYPES)

    penalty_best = float("inf")
    type_idx_best = None
    n_dragon5_calls = 0
    n_dragon5_rejected = 0
    log = []

    if verbose:
        print("\n" + "#" * 70, flush=True)
        print("# QUBO-HYBRID PHASE 1 (surrogate-penalty search for a "
              "REAL-DRAGON5-feasible start)", flush=True)
        print("#" * 70, flush=True)

    def combined_penalty(pattern, E, n_gad):
        # E and N_GAD are now BOTH soft terms here (E: 2026-08-15,
        # N_GAD: 2026-08-16) -- previously hard QUBO constraints (H_E,
        # H_N). Beyond E's own original motivation (SA missing the
        # narrow E window by a hair, discarding otherwise-good
        # candidates), removing N_GAD too fixes a SEPARATE, more severe
        # problem: with E excluded but N_GAD still hard, P1/PGAD had to
        # stay boosted to N_GAD's own coefficient scale (multiplicities
        # up to 24x) to keep one-hot/adjacency correctly enforced -- but
        # that made Phi(x) (whose own coefficients are O(1)) roughly
        # 20000x smaller than P1/PGAD after normalize(), rendering it
        # essentially invisible to SA (confirmed directly: centroid rmsd
        # saturated at 0.5 -- fully converged toward x_best -- yet SA
        # kept sampling close to uniformly among feasible candidates
        # regardless, since Phi supplied no usable signal). With BOTH E
        # and N_GAD excluded, no multiplicity-driven large coefficients
        # remain in the QUBO at all (one-hot and GAD-adjacency are
        # naturally O(1) on their own) -- so P1=PGAD=penalty (the
        # original, simple P>|I| derivation) is sufficient again, and
        # Phi's O(35)-scale signal is no longer swamped (~500x more
        # visible in testing: P1 dropped from 20736 back to 36).
        if E < E_LO:
            pe = (E_LO - E) / E_LO
        elif E > E_HI:
            pe = (E - E_HI) / E_HI
        else:
            pe = 0.0
        if n_gad < NGAD_LO:
            pn = (NGAD_LO - n_gad) / NGAD_LO
        elif n_gad > NGAD_HI:
            pn = (n_gad - NGAD_HI) / NGAD_HI
        else:
            pn = 0.0
        surr_out = surrogate.casmo4_bwr10x10(pattern)
        pk = max(0.0, (surr_out["k_inf_max"] - kinf_hi) / kinf_hi)
        pp = max(0.0, (surr_out["PPF"] - ppf_c) / ppf_c)
        return pe + pn + pk + pp, surr_out

    for q4c in range(1, max_qubo_calls + 1):
        # CHANGE (2026-08-15/16): use qubo_round_multi instead of
        # qubo_round -- evaluates combined_penalty across EVERY distinct
        # feasible sample this round produced (not just the single
        # lowest-QUBO-energy one), since the QUBO's own energy reflects
        # Phi(x) (ARR rounding quality) + constraint satisfaction, which
        # has NO relationship to Phase 1's actual goal (low E/N_GAD/
        # k_inf/PPF penalty) now that E and N_GAD are both soft terms.
        # Confirmed near-zero marginal cost: SA already computes all
        # num_reads samples regardless (dominates round time), and
        # evaluating the surrogate on every feasible one of them added
        # only ~0.01s more in testing (real DRAGON5 is never involved
        # here) -- so this effectively multiplies the candidates explored
        # per round by up to num_reads for close to free.
        results, ptb = qi.qubo_round_multi(
            c, VAR_ALLOWED, PIN_TYPES, ADJACENCY_PAIRS, MULTIPLICITIES, rng,
            sampler=sampler, num_reads=num_reads, num_sweeps=num_sweeps,
            include_e_constraint=False, include_ngad_constraint=False,
            use_perturbation=False)
        feasible_results = [r for r in results if r["feasible"]]

        if not feasible_results:
            rmsd_now = _compute_rmsd(c, VAR_ALLOWED)
            if verbose:
                print(f"  [PHASE1] Q4C {q4c}: 0/{len(results)} distinct samples passed "
                      f"N_GAD/adjacency this round  [rmsd={rmsd_now:.4f}]", flush=True)
            # CHANGE (2026-08-16, simplified further): reset is now a
            # deterministic rmsd threshold, no stuck-counting at all --
            # once the centroid has essentially fully converged
            # (rmsd>0.4995, near its 0.5 theoretical max), there is
            # nowhere further for it to usefully drift, so reset
            # immediately rather than waiting on any stuck/probability
            # bookkeeping. Simpler and matches what was empirically
            # observed: extended plateaus (e.g. 314 consecutive rounds
            # with no improvement) coincided with rmsd sitting at its
            # ceiling, regardless of how many resets had already fired
            # under the old stuck-based formula.
            if rmsd_now > 0.4995:
                c = qi.init_centroid(VAR_ALLOWED, N_TYPES)
                if verbose:
                    print(f"  [PHASE1] Q4C {q4c}: RESET centroid to uniform "
                          f"(rmsd={rmsd_now:.4f}, site=0-feasible)", flush=True)
            continue

        # q_best = the QUBO's OWN lowest-energy sample among all num_reads
        # (not filtered to feasible-only) -- the direct analog of classical
        # ARR's per-position argmax rounding of the perturbed centroid
        # (Section 2 of the project's own algorithm description: "Round:
        # position i selects k*(i)=argmax_k ptb(i,k)"). Kept SEPARATE from
        # the best-of-N-by-Phase-1-penalty candidate used for the
        # incumbent update below -- these answer two different questions.
        # Only used for the q_best==x_best diagnostic shown in the log
        # (2026-08-16: no longer feeds any control-flow decision -- see
        # the deterministic rmsd-threshold reset below instead).
        q_best = results[0]

        # Evaluate combined_penalty for every feasible candidate this
        # round and keep the best (lowest); stop early if a candidate
        # already good enough to trigger real DRAGON5 verification
        # (penalty<=surrogate_feasible_threshold) turns up, no need to
        # keep scoring the rest.
        best_r, best_pen, best_pattern, best_surr = None, float("inf"), None, None
        for r in feasible_results:
            pattern_r = build_grid_heuristic(np.asarray(r["type_idx"]))
            E_r = r["detail"]["E"]
            n_gad_r = r["detail"]["N_GAD"]
            pen_r, surr_r = combined_penalty(pattern_r, E_r, n_gad_r)
            if pen_r < best_pen:
                best_r, best_pen, best_pattern, best_surr = r, pen_r, pattern_r, surr_r
                if best_pen < surrogate_feasible_threshold:
                    break

        step, pattern, penalty, surr_out = best_r, best_pattern, best_pen, best_surr
        E = step["detail"]["E"]
        n_gad = step["detail"]["N_GAD"]
        repeat_match = (type_idx_best is not None and q_best["type_idx"] == type_idx_best)
        rmsd_now = _compute_rmsd(c, VAR_ALLOWED)  # centroid state entering this round
        if verbose:
            print(f"  [PHASE1] Q4C {q4c}: best of {len(feasible_results)} feasible "
                  f"candidates this round -> penalty={penalty:.4f}  "
                  f"[q_best==x_best={repeat_match} rmsd={rmsd_now:.4f}]", flush=True)

        if penalty < surrogate_feasible_threshold:
            if max_dragon5_calls is not None and n_dragon5_calls >= max_dragon5_calls:
                raise hy.Dragon5CallBudgetReached(
                    f"max_dragon5_calls={max_dragon5_calls} reached during Phase 1 -- "
                    f"no real-feasible start found yet ({n_dragon5_calls} real calls, "
                    f"{n_dragon5_rejected} rejected)."
                )
            n_dragon5_calls += 1
            if verbose:
                now = time.strftime("%H:%M:%S")
                print(f"  [PHASE1] [Q4C {q4c}] [DRAGON5 verify #{n_dragon5_calls}] @ {now} "
                      f"-- surrogate predicts feasible (k_inf={surr_out['k_inf_max']:.5f}, "
                      f"PPF={surr_out['PPF']:.3f}) -- running real DRAGON5 "
                      f"(compute_ppf={dragon5_cfg.compute_ppf}, "
                      f"compute_cl={dragon5_cfg.compute_cl}) ...", flush=True)
            real_out = _run_dragon5_with_heartbeat(
                pattern, dragon5_cfg, "PHASE1", n_dragon5_calls, heartbeat_sec, verbose)
            E = surr_out["E"]
            log.append({"q4c": q4c, "n_dragon5_calls": n_dragon5_calls, "E": E,
                        "real_k_inf_max": real_out["k_inf_max"], "real_PPF": real_out["PPF"],
                        "real_CL_proxy": real_out["CL_proxy"]})

            if (real_out["k_inf_max"] is not None and real_out["PPF"] is not None
                    and real_out["CL_proxy"] is not None):
                calib.add_verified_point(E, real_out["k_inf_max"], real_out["PPF"],
                                          real_out["CL_proxy"])
                if calib.refit_ppf:
                    calib.register_pattern_for_ppf_refit(pattern, real_out["PPF"])
                    calib._refit_ppf()
                if verbose:
                    print(f"    (folded into surrogate calibration -- refit #{calib.n_refits}, "
                          f"{len(calib.points)} verified points total)", flush=True)

            real_feasible = (real_out["k_inf_max"] is not None and real_out["PPF"] is not None
                              and real_out["k_inf_max"] <= kinf_hi and real_out["PPF"] < ppf_c)
            if real_feasible:
                if verbose:
                    print(f"    -> REAL FEASIBLE. Phase 1 complete: {n_dragon5_calls} real "
                          f"call(s) ({n_dragon5_rejected} rejected), {q4c} QUBO calls total.",
                          flush=True)
                return {
                    "type_idx": step["type_idx"], "pattern": pattern,
                    "real_k_inf_max": real_out["k_inf_max"], "real_PPF": real_out["PPF"],
                    "real_CL_proxy": real_out["CL_proxy"], "E": E,
                    "q4c": q4c, "n_dragon5_calls": n_dragon5_calls,
                    "n_dragon5_rejected": n_dragon5_rejected, "log": log,
                }
            else:
                n_dragon5_rejected += 1
                rmsd_now = _compute_rmsd(c, VAR_ALLOWED)

                # FIX (2026-08-16): re-evaluate the CURRENT incumbent
                # (type_idx_best) under the surrogate we JUST
                # recalibrated (add_verified_point/register_pattern_for_
                # ppf_refit/_refit_ppf above). penalty_best was computed
                # under the OLD surrogate and is now on a different
                # footing than every subsequent round's candidates
                # (evaluated under the NEW surrogate) -- so future
                # candidates could never beat it even if genuinely
                # better post-recalibration. Confirmed directly: after
                # one rejection+recalibration, 487 consecutive rounds
                # never produced a single "new best", stuck at the
                # pre-recalibration penalty_best=0.0182 the whole time.
                if type_idx_best is not None:
                    pattern_best = build_grid_heuristic(np.asarray(type_idx_best))
                    _, detail_best = qi.check_feasibility(
                        type_idx_best, VAR_ALLOWED, PIN_TYPES, MULTIPLICITIES,
                        ADJACENCY_PAIRS, check_e_range=False, check_ngad_range=False)
                    old_penalty_best = penalty_best
                    penalty_best, _ = combined_penalty(
                        pattern_best, detail_best["E"], detail_best["N_GAD"])
                    if verbose:
                        print(f"    (incumbent re-scored under recalibrated surrogate: "
                              f"penalty_best {old_penalty_best:.4f} -> {penalty_best:.4f})",
                              flush=True)

                if verbose:
                    print(f"    -> DRAGON5 says INFEASIBLE (k_inf_max={real_out['k_inf_max']}, "
                          f"PPF={real_out['PPF']}) -- surrogate was wrong here; recalibrated, "
                          f"continuing Phase 1 search.  [rmsd={rmsd_now:.4f}]", flush=True)
                # FIX (2026-08-15): the OLD code re-scored this same
                # candidate under the just-refit surrogate and let it flow
                # into the ARR bookkeeping below. Since penalty_best starts
                # at +inf, whichever candidate reaches that bookkeeping
                # FIRST always "wins" regardless of how bad it truly is --
                # so a real-rejected candidate could become type_idx_best
                # simply by being early, permanently dragging the ARR
                # seed's centroid toward a point already KNOWN to be
                # real-infeasible. Confirmed directly: a run got stuck for
                # ~2000 QUBO calls hugging the E-boundary right around
                # where the single rejected candidate sat, never
                # triggering a second real DRAGON5 call at all. Fix: treat
                # a real-rejected candidate exactly like a constraint-set-1
                # rejection -- it still recalibrates the surrogate (that
                # part was correct and stays), but it must NOT become the
                # ARR incumbent or influence where the seed drifts next.
                if rmsd_now > 0.4995:
                    c = qi.init_centroid(VAR_ALLOWED, N_TYPES)
                    if verbose:
                        print(f"  [PHASE1] Q4C {q4c}: RESET centroid to uniform "
                              f"(rmsd={rmsd_now:.4f}, site=real-rejected)", flush=True)
                continue

        # Incumbent update: based on the best-of-round candidate's Phase 1
        # penalty (E/k_inf/PPF combined) -- unrelated to the stuck check
        # below. `improved` is captured BEFORE penalty_best is overwritten
        # (needed below -- comparing penalty >= penalty_best AFTER this
        # update would always be true right after an improvement, since
        # penalty_best was just set equal to penalty).
        improved = penalty < penalty_best
        if improved:
            penalty_best = penalty
            type_idx_best = step["type_idx"]
            if verbose:
                print(f"  [PHASE1] Q4C {q4c}: new best penalty={penalty:.4f} "
                      f"(E={E:.4f}, N_GAD={n_gad}, "
                      f"surrogate k_inf={surr_out['k_inf_max']:.5f}, "
                      f"PPF={surr_out['PPF']:.3f})", flush=True)

        c, alpha, rmsd = (qi.move_seed(c, type_idx_best, VAR_ALLOWED)
                           if type_idx_best is not None else (c, 0.0, 0.0))
        # CHANGE (2026-08-16, simplified further): reset is now a
        # deterministic rmsd threshold, no stuck-counting at all -- see
        # the 0-feasible branch's comment above for the full rationale
        # (confirmed structurally: q_best==x_best essentially never
        # holds in this QUBO regardless of sweeps/beta_range, and
        # extended plateaus coincided with rmsd sitting at its ceiling
        # regardless of the old stuck-based probability formula).
        if rmsd > 0.4995:
            c = qi.init_centroid(VAR_ALLOWED, N_TYPES)
            if verbose:
                print(f"  [PHASE1] Q4C {q4c}: RESET centroid to uniform "
                      f"(rmsd={rmsd:.4f}, site=main-loop)", flush=True)

    raise RuntimeError(
        f"Phase 1 exhausted max_qubo_calls={max_qubo_calls} without finding a "
        f"real-DRAGON5-feasible starting point ({n_dragon5_calls} real calls made, "
        f"{n_dragon5_rejected} rejected). Try a different seed, a larger "
        f"max_qubo_calls, or relax ppf_c."
    )


def run_phase1_only(seed=0, ppf_c=1.6, verbose=True,
                     dragon5_cfg=None, refit_ppf=False,
                     heartbeat_sec=60, max_rounds=20000, reset_window=20,
                     max_dragon5_calls=None,
                     phase0_path="phase0_calibration/phase0_points.json",
                     save_path="phase1_confirmed_feasible.json"):
    """(2026-08-16) Standalone entry point: runs ONLY Phase 1 (the new
    greedy-round ARR search, run_greedy_phase1_real_feasible), saves
    the confirmed real-feasible result to save_path, and returns --
    does NOT continue on into Phase 2. Phase 2's own design is being
    reconsidered separately following today's Phase 1 redesign, so
    nothing here should assume it runs next; use
    run_qubo_hybrid_phase2/run() (unchanged, still chains into Phase 2
    via the original QUBO-based path) if the old end-to-end behavior is
    wanted instead.

    Mirrors run_qubo_hybrid_phase2's own Phase 0 calibration seeding
    exactly (seed_from_phase0(), NOT the source paper's 3 solved
    patterns) -- see that function's docstring for why.
    """
    if dragon5_cfg is None:
        dragon5_cfg = dragon5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None)

    calib = hy.SurrogateCalibrationState(refit_ppf=refit_ppf)
    try:
        calib.seed_from_phase0(path=phase0_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Phase 0 calibration file not found at '{phase0_path}'. Run "
            f"run_phase0_calibration.py first (see its module docstring), "
            f"or pass phase0_path= to point at an existing one.")

    return run_greedy_phase1_real_feasible(
        calib, KINF_HI, ppf_c, seed=seed, max_rounds=max_rounds,
        dragon5_cfg=dragon5_cfg, verbose=verbose, heartbeat_sec=heartbeat_sec,
        max_dragon5_calls=max_dragon5_calls, reset_window=reset_window,
        save_path=save_path)


def run_qubo_hybrid_phase2(seed=0, ppf_c=1.6, verbose=True,
                            dragon5_cfg=None, refit_ppf=False,
                            heartbeat_sec=60, max_qubo_calls_phase1=2000,
                            max_qubo_calls=2000, num_reads=200, reset_window=20,
                            max_dragon5_calls=None, target_cl=None,
                            phase0_path="phase0_calibration/phase0_points.json",
                            num_reads_phase1=400, num_sweeps_phase1=500,
                            surrogate_feasible_threshold=0.02):
    """Phase 1 (real-feasible start search) -> Phase 2 (CL maximization,
    seeded directly from Phase 1's confirmed point, no re-verification).
    Returns a result dict; see module docstring.

    Calibration seeding: uses Phase 0's from-scratch 7-point calibration
    (seed_from_phase0(), see arr_hybrid_dragon5.py) by default -- NOT the
    source paper's 3 solved patterns (seed_from_dragon5()). Confirmed via
    run_phase0_analysis.py to be meaningfully more accurate and unbiased
    on held-out data; see that method's docstring for the numbers. Run
    run_phase0_calibration.py first if phase0_path doesn't exist yet.

    num_reads_phase1/num_sweeps_phase1 (added 2026-08-16): SEPARATE from
    Phase 2's own `num_reads` (still 200, unchanged) -- Phase 1 defaults
    to doubled SA settings (400 reads / 2000 sweeps) since its QUBO
    round explores ALL distinct feasible samples per round (qubo_round_
    multi), not just one, so a bigger/more thorough SA batch per round
    is worth the extra cost there specifically.

    surrogate_feasible_threshold (added 2026-08-16): see
    run_qubo_phase1_real_feasible's docstring -- Phase 1 only, Phase 2's
    own score-based objective is unaffected.
    """
    if dragon5_cfg is None:
        dragon5_cfg = dragon5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None)

    calib = hy.SurrogateCalibrationState(refit_ppf=refit_ppf)
    try:
        calib.seed_from_phase0(path=phase0_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Phase 0 calibration file not found at '{phase0_path}'. Run "
            f"run_phase0_calibration.py first (see its module docstring), "
            f"or pass phase0_path= to point at an existing one.")

    t0 = time.perf_counter()
    stopped_early, stop_reason = False, None

    try:
        phase1 = run_greedy_phase1_real_feasible(
            calib, KINF_HI, ppf_c, seed=seed, max_rounds=max_qubo_calls_phase1,
            dragon5_cfg=dragon5_cfg, verbose=verbose,
            heartbeat_sec=heartbeat_sec, max_dragon5_calls=max_dragon5_calls,
            reset_window=reset_window)
    except hy.ARRStopSignal as e:
        elapsed = time.perf_counter() - t0
        stop_reason = type(e).__name__
        if verbose:
            print(f"\n  [PHASE1] Stopping early ({stop_reason}): {e}", flush=True)
        return {
            "x_best": None, "score_best": None, "pattern": None,
            "q4c": None, "dragon5_log": [], "calib": calib,
            "n_surrogate_calls": None, "n_dragon5_calls": None,
            "n_dragon5_confirmed": 0, "n_dragon5_rejected": None,
            "elapsed_sec": elapsed, "stopped_early": True, "stop_reason": stop_reason,
        }

    def pattern_from_type_idx(type_idx):
        return build_grid_heuristic(np.asarray(type_idx))

    objective = hy.HybridVerifiedObjective(
        pattern_from_type_idx, kinf_hi=KINF_HI, ppf_c=ppf_c, calib_state=calib,
        cfg=dragon5_cfg, verbose=verbose, heartbeat_sec=heartbeat_sec,
        phase_label="QUBO-PHASE2", max_dragon5_calls=max_dragon5_calls,
        target_cl=target_cl)

    # --- Seed Phase 2's incumbent directly from Phase 1's confirmed
    # real-feasible point -- NO redundant DRAGON5 re-verification call,
    # since Phase 1's own confirming call already returned CL_proxy too. ---
    objective.real_best_pattern = phase1["pattern"]
    objective.real_best_score = phase1["real_CL_proxy"]
    objective.n_dragon5_calls = phase1["n_dragon5_calls"]
    objective.n_dragon5_confirmed = 1
    objective.n_dragon5_rejected = phase1["n_dragon5_rejected"]
    objective.log = list(phase1["log"])
    # threshold re-based under the current (already-refit) coefficients,
    # same staleness-avoidance principle used everywhere else here.
    objective.surrogate_best_score = surrogate.casmo4_bwr10x10(phase1["pattern"])["CL_proxy"]

    if verbose:
        print(f"\n=== Phase 1 -> Phase 2 handoff (seed={seed}, PPF<{ppf_c}) ===", flush=True)
        print(f"Real-feasible start: E={phase1['E']:.4f}  "
              f"k_inf_max={phase1['real_k_inf_max']:.5f}  PPF={phase1['real_PPF']:.4f}  "
              f"CL_proxy={phase1['real_CL_proxy']:.3f}", flush=True)
        print(f"Phase 1 cost: {phase1['q4c']} QUBO calls, "
              f"{phase1['n_dragon5_calls']} real DRAGON5 call(s) "
              f"({phase1['n_dragon5_rejected']} rejected)\n", flush=True)

    rng = np.random.default_rng(seed + 100000)
    sampler = SimulatedAnnealingSampler()
    c = qi.init_centroid(VAR_ALLOWED, N_TYPES)
    type_idx_best, score_best = phase1["type_idx"], objective.surrogate_best_score
    stuck, q4c_used = 0, 0

    try:
        for q4c in range(1, max_qubo_calls + 1):
            q4c_used = q4c
            step = qi.qubo_round(c, VAR_ALLOWED, PIN_TYPES, ADJACENCY_PAIRS,
                                  MULTIPLICITIES, rng, sampler=sampler, num_reads=num_reads)
            if not step["feasible"]:
                if verbose:
                    print(f"  [QUBO-PHASE2] Q4C {q4c}: reject "
                          f"(energy={step['energy']:.4f}, {step['detail']})", flush=True)
                score = hy.REJECT_SCORE
            else:
                score = objective(step["type_idx"])

            if score > score_best:
                score_best = score
                type_idx_best = step["type_idx"]
                stuck = 0
            else:
                stuck += 1

            c, alpha, rmsd = qi.move_seed(c, type_idx_best, VAR_ALLOWED) \
                if type_idx_best is not None else (c, 0.0, 0.0)
            if stuck >= reset_window and rng.random() < min(1.0, stuck / reset_window - 1 + rmsd):
                c = qi.init_centroid(VAR_ALLOWED, N_TYPES)
                stuck = 0
    except hy.ARRStopSignal as e:
        stopped_early = True
        stop_reason = type(e).__name__
        if verbose:
            print(f"\n  [QUBO-PHASE2] Stopping early ({stop_reason}): {e}", flush=True)

    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"\n=== QUBO-HYBRID Phase 2 result (seed={seed}, PPF<{ppf_c}) ===", flush=True)
        print(f"REAL best CL_proxy = {objective.real_best_score:.3f}", flush=True)
        print(f"QUBO calls (Phase1={phase1['q4c']} + Phase2={q4c_used}) = "
              f"{phase1['q4c'] + q4c_used}  "
              f"real DRAGON5 calls = {objective.n_dragon5_calls}  "
              f"confirmed = {objective.n_dragon5_confirmed}  "
              f"rejected = {objective.n_dragon5_rejected}", flush=True)
        print(f"surrogate refits during run = {calib.n_refits}", flush=True)
        print(f"wall time = {elapsed:.1f}s", flush=True)
        if stopped_early:
            print(f"  (stopped early: {stop_reason})", flush=True)

    return {
        "x_best": type_idx_best,
        "score_best": objective.real_best_score,
        "pattern": objective.real_best_pattern,
        "q4c": phase1["q4c"] + q4c_used,
        "q4c_phase1": phase1["q4c"],
        "q4c_phase2": q4c_used,
        "dragon5_log": objective.log,
        "calib": calib,
        "n_surrogate_calls": objective.n_surrogate_calls,
        "n_dragon5_calls": objective.n_dragon5_calls,
        "n_dragon5_confirmed": objective.n_dragon5_confirmed,
        "n_dragon5_rejected": objective.n_dragon5_rejected,
        "elapsed_sec": elapsed,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }


def run_greedy_phase2(seed=0, ppf_c=1.4, verbose=True, dragon5_cfg=None,
                       refit_ppf=True, heartbeat_sec=60, max_rounds=20000,
                       reset_window=20, max_dragon5_calls=12, target_cl=None,
                       phase0_path="phase0_calibration/phase0_points.json",
                       phase1_result_path="phase1_confirmed_feasible.json",
                       phase1=None, max_wall_hours=None, gate_margin_days=0.0,
                       surrogate_ppf_gate=True, prior_evaluations_path=None,
                       early_abort=True, surrogate_gate=True):
    """(2026-08-16) Phase 2 redesigned to match Phase 1's greedy_round
    approach, resuming from an ALREADY-COMPLETED Phase 1 result (either
    passed directly via `phase1=`, or -- the default -- loaded from
    phase1_result_path, i.e. run_phase1_only's saved JSON) rather than
    running Phase 1 itself. This function's own max_dragon5_calls is an
    INDEPENDENT budget (starts at 0), separate from however many calls
    Phase 1 already used to find its confirmed starting point.

    Per-round mechanics: same core loop shape as the original QUBO-
    based Phase 2, with ONE change -- greedy_round(perturb(c, ...))
    replaces qi.qubo_round(...) for generating each round's candidate.
    Since greedy_round GUARANTEES one-hot and GAD-adjacency by
    construction (see its own docstring), there's no "0 feasible
    samples this round" case to handle the way the old QUBO/SA path
    needed -- every round produces exactly one structurally-valid
    candidate, unconditionally.

    Objective/acceptance logic is UNCHANGED from the original Phase 2 --
    reused directly via hy.HybridVerifiedObjective, not reimplemented,
    since it already does precisely what's wanted here: check E/N_GAD
    (constraint set 1) first and pass on failure; then check surrogate
    k_inf_max/PPF and pass on failure; for a surrogate-feasible
    candidate, compare its surrogate CL_proxy against the tracked
    surrogate_best_score and only pay for a real DRAGON5 call when this
    candidate's surrogate CL genuinely beats it; recalibrate off of
    every real call regardless of outcome; accept as the new real
    incumbent only when DRAGON5 confirms BOTH feasible AND a real CL
    improvement.

    stuck/reset: unchanged from the original Phase 2 (additive
    threshold: stuck>=reset_window gates a rng draw against
    min(1, stuck/reset_window-1+rmsd)) -- this wasn't part of what was
    asked to change here, and Phase 2's own version of this (score-
    improvement-based stuck, exactly what Phase 1 eventually converged
    on too) was already the version proven to work.

    target_cl (default None): if left None, AUTO-SELECTED from
    TARGET_CL_PPF_1_4/TARGET_CL_PPF_1_6 based on ppf_c (<=1.5 -> the
    PPF<1.4 target, >1.5 -> the PPF<1.6 target) -- see those constants'
    own comment for why these are the paper's "RL_PPF_1_6"/"RL_PPF_1_4"
    patterns respectively, not the other way around (DRAGON5 re-
    verification inverted the paper's own PPF labels). Once Phase 2's
    real incumbent reaches this value, hy.HybridVerifiedObjective
    raises TargetCLReached (a subclass of hy.ARRStopSignal, already
    caught below) and this function returns immediately -- pass
    target_cl=False explicitly to disable auto-selection and run
    purely on the max_dragon5_calls budget instead.

    max_wall_hours (2026-11-06, default None = off): wall-clock budget for this Phase 2 run. When it is used, set
    max_dragon5_calls to a large safety cap (or None). The check runs at the start of every objective call; a DRAGON5
    call already running is not interrupted, so the run may overrun by one oracle call (<= ~110 min).

    gate_margin_days (2026-11-06, default 0.0 = old behaviour): candidates whose surrogate CL is up to this many days
    BELOW the incumbent's surrogate CL are also sent to the oracle (see HybridVerifiedObjective).

    surrogate_ppf_gate (2026-11-06, default True = old behaviour): False -> the diffusion PPF surrogate does not reject
    candidates; PPF is decided by the oracle (early abort at the BOL step).

    ABLATION switches (2026-11-10; defaults = old behaviour; used only when dragon5_cfg is None):
    early_abort=False -> the oracle is never stopped early: every call runs to completion (no BOL-PPF / k_inf abort), so
        infeasible candidates cost a full run; feasibility is then decided from the complete result.
    surrogate_gate=False -> the surrogate does not reject anything (neither the k_inf nor the CL gate); every candidate that
        passes the exact checks and is not in the cache goes to the oracle.

    prior_evaluations_path (2026-11-08, default None = old behaviour): JSON {"entries": [...]} of patterns that were already
    evaluated by the oracle in earlier phases (Phase 0 and Phase 1). Every entry is put in the verified cache, so such a pattern is
    never sent to the oracle again (a cache hit reuses its stored result: infeasible if aborted / PPF>=c / k>kinf_hi). Entries that
    are complete runs (k, PPF and CL known) and are not Phase-0 points (those are already in the calibration) or the start pattern
    are also added to the surrogate calibration.
    """
    if target_cl is None:
        target_cl = TARGET_CL_PPF_1_4 if ppf_c <= 1.5 else TARGET_CL_PPF_1_6
    elif target_cl is False:
        target_cl = None

    if phase1 is None:
        with open(phase1_result_path) as f:
            phase1 = json.load(f)

    if dragon5_cfg is None:
        # early abort: stop DRAGON5 as soon as the candidate is proven infeasible (BOL PPF >= ppf_c, or any
        # burnup step with k_inf > KINF_HI) -- see Dragon5Config.abort_*; aborted calls return CL_proxy=None.
        if early_abort:
            dragon5_cfg = dragon5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None,
                                                abort_ppf_above=ppf_c, abort_k_above=KINF_HI)
        else:      # ablation: no early termination, every call runs to completion
            dragon5_cfg = dragon5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None)

    calib = hy.SurrogateCalibrationState(refit_ppf=refit_ppf)
    try:
        calib.seed_from_phase0(path=phase0_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Phase 0 calibration file not found at '{phase0_path}'. Run "
            f"run_phase0_calibration.py first (see its module docstring), "
            f"or pass phase0_path= to point at an existing one.")

    def pattern_from_type_idx(type_idx):
        return build_grid_heuristic(np.asarray(type_idx))

    objective = hy.HybridVerifiedObjective(
        pattern_from_type_idx, kinf_hi=KINF_HI, ppf_c=ppf_c, calib_state=calib,
        cfg=dragon5_cfg, verbose=verbose, heartbeat_sec=heartbeat_sec,
        phase_label="GREEDY-PHASE2", max_dragon5_calls=max_dragon5_calls,
        target_cl=target_cl,
        max_wall_sec=None if max_wall_hours is None else float(max_wall_hours) * 3600.0,
        gate_margin_days=gate_margin_days, surrogate_ppf_gate=surrogate_ppf_gate,
        use_surrogate_gate=surrogate_gate)

    # Seed Phase 2's incumbent from Phase 1's ALREADY-confirmed
    # real-feasible point -- no redundant re-verification (Phase 1's own
    # confirming call already returned CL_proxy). Rebuild the pattern
    # from type_idx rather than trusting a possibly-JSON-reloaded
    # `phase1["pattern"]` (tuples become lists through json.dump/load,
    # which could trip up code expecting tuple entries) -- type_idx
    # (plain ints) round-trips through JSON safely either way.
    phase1_type_idx = phase1["type_idx"]
    phase1_pattern = pattern_from_type_idx(phase1_type_idx)
    objective.real_best_pattern = phase1_pattern
    objective.real_best_score = phase1["real_CL_proxy"]
    objective.real_best_x = tuple(int(k) for k in phase1_type_idx)
    # FIX (2026-11-06): the real-verified start pattern must be in verified_cache. With gate_margin_days > 0 the start pattern
    # itself passes the surrogate gate (its own surrogate CL > best - margin) and, not being cached, was re-run by the
    # oracle (seeds 3 and 4 of the first PPF<1.4 run each lost one ~107-min call on it). Same result as the Phase-1 call.
    objective.verified_cache[objective.real_best_x] = {
        "E": phase1.get("E"), "k_inf_max": phase1["real_k_inf_max"], "PPF": phase1["real_PPF"],
        "CL_proxy": phase1["real_CL_proxy"], "aborted": None, "burnup_history": {}}

    # FIX (2026-08-20): Phase 1's own real-DRAGON5-verified point was
    # previously used only as a threshold/starting pattern here, never
    # actually folded into the surrogate calibration -- a fresh calib
    # object, seeded from Phase 0's 7 points, would start Phase 2 (and
    # its first PPF refit) with only 7+1=8 verified points instead of
    # the 7+1+1=9 that were genuinely available (Phase 0's 7 + Phase
    # 1's own confirming call), simply because Phase 1 and Phase 2 now
    # run as separate processes/scripts (this used to happen for free
    # when they shared one process and one calib object). Folding it in
    # here recovers that lost data point for both the linear (k_inf/CL)
    # and, if refit_ppf, the nonlinear PPF fit.
    calib.add_verified_point(phase1["E"], phase1["real_k_inf_max"],
                              phase1["real_PPF"], phase1["real_CL_proxy"])
    if calib.refit_ppf:
        calib.register_pattern_for_ppf_refit(phase1_pattern, phase1["real_PPF"])
        calib._refit_ppf()

    n_prior_cached = n_prior_calib = 0
    if prior_evaluations_path:
        with open(prior_evaluations_path) as _f:
            _prior = json.load(_f)
        for _e in _prior["entries"]:
            _key = tuple(int(k) for k in _e["type_idx"])
            if _key in objective.verified_cache:          # includes the start pattern itself
                continue
            objective.verified_cache[_key] = {"E": _e["E"], "k_inf_max": _e["k_inf_max"], "PPF": _e["PPF"],
                                              "CL_proxy": _e["CL_proxy"], "aborted": _e.get("aborted"),
                                              "burnup_history": {}}
            n_prior_cached += 1
            if (not _e.get("calibrated_elsewhere") and not _e.get("aborted") and _e["CL_proxy"] is not None
                    and _e["k_inf_max"] is not None and _e["PPF"] is not None):
                if calib.refit_ppf:
                    calib.register_pattern_for_ppf_refit(pattern_from_type_idx(_e["type_idx"]), _e["PPF"])
                calib.add_verified_point(_e["E"], _e["k_inf_max"], _e["PPF"], _e["CL_proxy"])
                n_prior_calib += 1
        if verbose:
            print(f"[GREEDY-PHASE2] prior evaluations loaded from {prior_evaluations_path}: {n_prior_cached} patterns put in "
                  f"the verified cache (never re-evaluated), {n_prior_calib} complete runs added to the surrogate calibration "
                  f"({len(calib.points)} verified points total).", flush=True)

    objective.surrogate_best_score = surrogate.casmo4_bwr10x10(phase1_pattern)["CL_proxy"]
    # Independent budget: Phase 2's own n_dragon5_calls starts at 0,
    # NOT inherited from however many real calls Phase 1 already spent.

    if verbose:
        print(f"\n=== Phase 1 -> Phase 2 handoff (seed={seed}, PPF<{ppf_c}) ===", flush=True)
        print(f"Real-feasible start: E={phase1['E']:.4f}  "
              f"k_inf_max={phase1['real_k_inf_max']:.5f}  PPF={phase1['real_PPF']:.4f}  "
              f"CL_proxy={phase1['real_CL_proxy']:.3f}", flush=True)
        print(f"Phase 1 cost: {phase1['q4c']} rounds, "
              f"{phase1['n_dragon5_calls']} real DRAGON5 call(s) "
              f"({phase1['n_dragon5_rejected']} rejected) -- Phase 2 starts its own "
              f"independent budget of {max_dragon5_calls} real call(s), "
              f"target_cl={target_cl}.\n", flush=True)

    t0 = time.perf_counter()
    stopped_early, stop_reason = False, None
    rng = np.random.default_rng(seed + 100000)
    c = qi.init_centroid(VAR_ALLOWED, N_TYPES)
    # (2026-11-03) The centroid target is the oracle-CONFIRMED incumbent only. Before, the driver compared the value
    # returned by objective() with its own score_best; after a refit the objective re-bases its threshold, and a
    # candidate the oracle had just REJECTED could return a re-based value above the (stale) score_best and become the
    # centroid target (seen: the same rejected pattern re-screened dozens of times via CACHE HIT). In this hybrid
    # design every surrogate-improving candidate is sent to the oracle at once, so "improvement" == "new confirmed
    # incumbent". stuck counts rounds without a new confirmed incumbent.
    type_idx_best = list(phase1_type_idx)
    stuck, q4c_used = 0, 0

    try:
        for q4c in range(1, max_rounds + 1):
            q4c_used = q4c
            ptb = qi.perturb(c, VAR_ALLOWED, rng, mode="mult")
            type_idx = qi.greedy_round(ptb, VAR_ALLOWED, ADJACENCY_PAIRS, qi.GAD_TYPE_IDS)
            objective(type_idx)

            if objective.real_best_x is not None and objective.real_best_x != tuple(int(k) for k in type_idx_best):
                type_idx_best = list(objective.real_best_x)   # new oracle-confirmed incumbent
                stuck = 0
            else:
                stuck += 1

            c, alpha, rmsd = qi.move_seed(c, type_idx_best, VAR_ALLOWED) \
                if type_idx_best is not None else (c, 0.0, 0.0)
            if stuck >= reset_window and rng.random() < min(1.0, stuck / reset_window - 1 + rmsd):
                c = qi.init_centroid(VAR_ALLOWED, N_TYPES)
                stuck = 0
    except hy.ARRStopSignal as e:
        stopped_early = True
        stop_reason = type(e).__name__
        if verbose:
            print(f"\n  [GREEDY-PHASE2] Stopping early ({stop_reason}): {e}", flush=True)

    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"\n=== GREEDY Phase 2 result (seed={seed}, PPF<{ppf_c}) ===", flush=True)
        print(f"REAL best CL_proxy = {objective.real_best_score:.3f}", flush=True)
        print(f"Phase 2 rounds = {q4c_used}  real DRAGON5 calls = {objective.n_dragon5_calls}  "
              f"confirmed = {objective.n_dragon5_confirmed}  "
              f"rejected = {objective.n_dragon5_rejected}", flush=True)
        print(f"surrogate refits during Phase 2 = {calib.n_refits}", flush=True)
        print(f"wall time = {elapsed:.1f}s", flush=True)
        if stopped_early:
            print(f"stopped early: {stop_reason}", flush=True)

    return {
        "type_idx_best": type_idx_best, "real_best_score": objective.real_best_score,
        "real_best_pattern": objective.real_best_pattern,
        "q4c_phase2": q4c_used, "dragon5_log": objective.log, "calib": calib,
        "n_surrogate_calls": objective.n_surrogate_calls,
        "n_dragon5_calls": objective.n_dragon5_calls,
        "n_dragon5_confirmed": objective.n_dragon5_confirmed,
        "n_dragon5_rejected": objective.n_dragon5_rejected,
        "elapsed_sec": elapsed, "stopped_early": stopped_early, "stop_reason": stop_reason,
    }


def run_greedy_phase2_multicandidate(seed=0, ppf_c=1.4, verbose=True, dragon5_cfg=None,
                                      refit_ppf=True, heartbeat_sec=60, max_rounds=20000,
                                      reset_window=20, max_dragon5_calls=12, target_cl=None,
                                      n_candidates=200,
                                      phase0_path="phase0_calibration/phase0_points.json",
                                      phase1_result_path="phase1_confirmed_feasible.json",
                                      phase1=None):
    """(2026-08-20) Phase 2 variant: per ARR round, draw n_candidates
    (default 200) INDEPENDENT perturbations of the SAME centroid c
    (same trial/seed state -- c doesn't change within a round, only
    the random draw inside perturb() differs each of the 200 times),
    greedy_round-decode each into an integer solution, and prescreen
    ALL of them via hy.HybridVerifiedObjective.surrogate_feasible_score
    (constraint set 1 + surrogate k_inf/PPF -- cheap, no real DRAGON5).
    Among the surrogate-FEASIBLE candidates, pick the one with the
    highest surrogate CL_proxy. That single winner (not all 200) is
    then handed to objective(...) -- the real __call__ -- exactly as
    in run_greedy_phase2, so a real DRAGON5 call can be triggered AT
    MOST ONCE per round, only for this round's best candidate, only if
    it surrogate-beats the tracked incumbent.

    This is run_greedy_phase2's single-candidate-per-round design
    (itself descended from the original Phase 2's one-QUBO-sample-per-
    round cadence) extended with Phase 1's original multi-candidate
    idea (up to 400 samples/round, keep the best) -- applied here to
    Phase 2's greedy_round + surrogate-CL-ranking instead of Phase 1's
    QUBO energy ranking.

    If NONE of the n_candidates are surrogate-feasible this round, the
    round contributes no candidate at all (score behaves like a
    rejection -- stuck increments, exactly as if a single infeasible
    candidate had been drawn).

    All other parameters and behavior (target_cl auto-selection,
    independent max_dragon5_calls budget from Phase 1's own, stuck/
    reset formula, save/handoff format) are identical to
    run_greedy_phase2 -- see its docstring for the parts not repeated
    here.
    """
    if target_cl is None:
        target_cl = TARGET_CL_PPF_1_4 if ppf_c <= 1.5 else TARGET_CL_PPF_1_6
    elif target_cl is False:
        target_cl = None

    if phase1 is None:
        with open(phase1_result_path) as f:
            phase1 = json.load(f)

    if dragon5_cfg is None:
        dragon5_cfg = dragon5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None)

    calib = hy.SurrogateCalibrationState(refit_ppf=refit_ppf)
    try:
        calib.seed_from_phase0(path=phase0_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Phase 0 calibration file not found at '{phase0_path}'. Run "
            f"run_phase0_calibration.py first (see its module docstring), "
            f"or pass phase0_path= to point at an existing one.")

    def pattern_from_type_idx(type_idx):
        return build_grid_heuristic(np.asarray(type_idx))

    objective = hy.HybridVerifiedObjective(
        pattern_from_type_idx, kinf_hi=KINF_HI, ppf_c=ppf_c, calib_state=calib,
        cfg=dragon5_cfg, verbose=verbose, heartbeat_sec=heartbeat_sec,
        phase_label="GREEDY-PHASE2-MC", max_dragon5_calls=max_dragon5_calls,
        target_cl=target_cl)

    phase1_type_idx = phase1["type_idx"]
    phase1_pattern = pattern_from_type_idx(phase1_type_idx)
    objective.real_best_pattern = phase1_pattern
    objective.real_best_score = phase1["real_CL_proxy"]

    # FIX (2026-08-20): fold Phase 1's own real-DRAGON5-verified point
    # into calibration too -- see the matching comment in
    # run_greedy_phase2 above for why this was missing and what it
    # recovers (7+1=8 -> 7+1+1=9 verified points from the very start).
    calib.add_verified_point(phase1["E"], phase1["real_k_inf_max"],
                              phase1["real_PPF"], phase1["real_CL_proxy"])
    if calib.refit_ppf:
        calib.register_pattern_for_ppf_refit(phase1_pattern, phase1["real_PPF"])
        calib._refit_ppf()

    objective.surrogate_best_score = surrogate.casmo4_bwr10x10(phase1_pattern)["CL_proxy"]

    if verbose:
        print(f"\n=== Phase 1 -> Phase 2 (multi-candidate, n={n_candidates}) handoff "
              f"(seed={seed}, PPF<{ppf_c}) ===", flush=True)
        print(f"Real-feasible start: E={phase1['E']:.4f}  "
              f"k_inf_max={phase1['real_k_inf_max']:.5f}  PPF={phase1['real_PPF']:.4f}  "
              f"CL_proxy={phase1['real_CL_proxy']:.3f}", flush=True)
        print(f"Phase 1 cost: {phase1['q4c']} rounds, "
              f"{phase1['n_dragon5_calls']} real DRAGON5 call(s) "
              f"({phase1['n_dragon5_rejected']} rejected) -- Phase 2 starts its own "
              f"independent budget of {max_dragon5_calls} real call(s), "
              f"target_cl={target_cl}.\n", flush=True)

    t0 = time.perf_counter()
    stopped_early, stop_reason = False, None
    rng = np.random.default_rng(seed + 100000)
    c = qi.init_centroid(VAR_ALLOWED, N_TYPES)
    type_idx_best, score_best = phase1_type_idx, objective.surrogate_best_score
    stuck, q4c_used = 0, 0

    try:
        for q4c in range(1, max_rounds + 1):
            q4c_used = q4c

            # Draw n_candidates independent perturbations of the SAME
            # centroid c (this round's trial) and keep the surrogate-
            # feasible one with the highest surrogate CL_proxy.
            winner_type_idx, winner_surr_cl = None, float("-inf")
            for _ in range(n_candidates):
                ptb = qi.perturb(c, VAR_ALLOWED, rng, mode="mult")
                type_idx_cand = qi.greedy_round(ptb, VAR_ALLOWED, ADJACENCY_PAIRS,
                                                 qi.GAD_TYPE_IDS)
                feasible, surr_cl = objective.surrogate_feasible_score(type_idx_cand)
                if feasible and surr_cl > winner_surr_cl:
                    winner_surr_cl = surr_cl
                    winner_type_idx = type_idx_cand

            if winner_type_idx is not None:
                score = objective(winner_type_idx)
            else:
                score = hy.REJECT_SCORE

            if score > score_best:
                score_best = score
                type_idx_best = winner_type_idx
                stuck = 0
            else:
                stuck += 1

            c, alpha, rmsd = qi.move_seed(c, type_idx_best, VAR_ALLOWED) \
                if type_idx_best is not None else (c, 0.0, 0.0)
            if stuck >= reset_window and rng.random() < min(1.0, stuck / reset_window - 1 + rmsd):
                c = qi.init_centroid(VAR_ALLOWED, N_TYPES)
                stuck = 0
    except hy.ARRStopSignal as e:
        stopped_early = True
        stop_reason = type(e).__name__
        if verbose:
            print(f"\n  [GREEDY-PHASE2-MC] Stopping early ({stop_reason}): {e}", flush=True)

    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"\n=== GREEDY Phase 2 (multi-candidate, n={n_candidates}) result "
              f"(seed={seed}, PPF<{ppf_c}) ===", flush=True)
        print(f"REAL best CL_proxy = {objective.real_best_score:.3f}", flush=True)
        print(f"Phase 2 rounds = {q4c_used}  real DRAGON5 calls = {objective.n_dragon5_calls}  "
              f"confirmed = {objective.n_dragon5_confirmed}  "
              f"rejected = {objective.n_dragon5_rejected}", flush=True)
        print(f"surrogate refits during Phase 2 = {calib.n_refits}", flush=True)
        print(f"wall time = {elapsed:.1f}s", flush=True)
        if stopped_early:
            print(f"stopped early: {stop_reason}", flush=True)

    return {
        "type_idx_best": type_idx_best, "real_best_score": objective.real_best_score,
        "real_best_pattern": objective.real_best_pattern,
        "q4c_phase2": q4c_used, "dragon5_log": objective.log, "calib": calib,
        "n_surrogate_calls": objective.n_surrogate_calls,
        "n_dragon5_calls": objective.n_dragon5_calls,
        "n_dragon5_confirmed": objective.n_dragon5_confirmed,
        "n_dragon5_rejected": objective.n_dragon5_rejected,
        "elapsed_sec": elapsed, "stopped_early": stopped_early, "stop_reason": stop_reason,
    }


def run(seed=0, ppf_c=1.6, verbose=True, dragon5_cfg=None, refit_ppf=False,
        heartbeat_sec=60, max_rounds=20000, reset_window=20,
        max_dragon5_calls=None, target_cl=None,
        phase0_path="phase0_calibration/phase0_points.json",
        save_path="phase1_confirmed_feasible.json"):
    """Top-level entry point. (2026-08-16): now runs Phase 1 ONLY (see
    run_phase1_only) -- saves the confirmed real-feasible result and
    stops, rather than chaining into Phase 2 as before. Phase 2's own
    design is being reconsidered separately following today's Phase 1
    redesign. Call run_qubo_hybrid_phase2(...) directly for the old
    end-to-end (Phase 1 QUBO-based -> Phase 2) behavior if needed.

    target_cl: accepted but IGNORED here -- it's a Phase-2-only concept
    (CL maximization target) kept in this signature purely so existing
    calling scripts written for the old Phase1->Phase2 chain don't
    break on this now-unused keyword argument."""
    return run_phase1_only(
        seed=seed, ppf_c=ppf_c, verbose=verbose, dragon5_cfg=dragon5_cfg,
        refit_ppf=refit_ppf, heartbeat_sec=heartbeat_sec, max_rounds=max_rounds,
        reset_window=reset_window, max_dragon5_calls=max_dragon5_calls,
        phase0_path=phase0_path, save_path=save_path)



# ======================================================================
# Phase 1, oracle-penalty version (2026-11-07)
# ======================================================================
def phase1_penalty(E, n_gad, k_inf_max, ppf, ppf_c, kinf_hi=KINF_HI):
    """Phase 1 penalty P = p_E + p_N + p_k + p_PPF (relative violations, 0 iff every term is satisfied).
    k_inf_max / ppf may be None (not observed because the oracle call was stopped earlier); an unobserved term counts
    as 0, so P is then a LOWER bound of the true penalty."""
    pe = (E_LO - E) / E_LO if E < E_LO else ((E - E_HI) / E_HI if E > E_HI else 0.0)
    pn = (NGAD_LO - n_gad) / NGAD_LO if n_gad < NGAD_LO else ((n_gad - NGAD_HI) / NGAD_HI if n_gad > NGAD_HI else 0.0)
    pk = max(0.0, (k_inf_max - kinf_hi) / kinf_hi) if k_inf_max is not None else 0.0
    pp = max(0.0, (ppf - ppf_c) / ppf_c) if ppf is not None else 0.0
    return pe + pn + pk + pp


def _type_idx_from_pattern(pattern):
    """Inverse of build_grid_heuristic: the 35 group choices of a 10x10 pattern (matched cell by cell against the
    grids produced for every allowed choice; the round trip is verified)."""
    norm = lambda x: json.loads(json.dumps(x))
    pat = norm(pattern)
    base = [int(VAR_ALLOWED[v][0]) for v in range(N_VARS)]
    ti = list(base)
    for v, cells in enumerate(VAR_CELLS):
        r, c0 = cells[0]
        hits = []
        for k in VAR_ALLOWED[v]:
            trial = list(base); trial[v] = int(k)
            if norm(build_grid_heuristic(np.asarray(trial)))[r][c0] == pat[r][c0]:
                hits.append(int(k))
        if len(hits) != 1:
            raise ValueError(f"group {v}: cell {(r, c0)} = {pat[r][c0]} matches {len(hits)} allowed choices")
        ti[v] = hits[0]
    if norm(build_grid_heuristic(np.asarray(ti))) != pat:
        raise ValueError("pattern is not representable by the 35-group heuristic structure")
    return ti


def run_greedy_phase1_oracle(seed=0, ppf_c=1.4, kinf_hi=KINF_HI, phase0_path="phase0_calibration_sym/phase0_points.json",
                             save_dir=".", max_wall_hours=12.0, max_dragon5_calls=500, max_rounds=10**8,
                             reset_window=20, dragon5_cfg=None, verbose=True, heartbeat_sec=60, oracle_fn=None, start_index=None):
    """Phase 1 as penalty minimisation against the staged oracle (no surrogate).

    objective   P(x) = p_E + p_N + p_k + p_PPF (phase1_penalty); search stops at P = 0, i.e. an oracle-certified
                feasible pattern (PPF < ppf_c at BOL and k_inf <= kinf_hi over the whole history).
    start       the Phase-0 pattern with the smallest penalty (oracle values from phase0_path) is the initial best
                solution of EVERY seed; all Phase-0 patterns are pre-loaded in the evaluation cache.
    candidates  ARR with greedy_round (one-hot and GAD adjacency by construction), as in Phase 2.  A candidate whose
                E or N_GAD is out of range is rejected without an oracle call.  Exact repeats are never re-evaluated.
    oracle      every other candidate is run as a stream: killed at BOL if PPF >= ppf_c (cost ~20 min), killed once the
                k_inf peak has passed if k_inf_max > kinf_hi (exact k_inf_max, ~70 % of a run); a run that is not killed
                is a complete run and, by construction, feasible -- its CL is recorded only because Phase 2 needs it
                as the start value; CL is never used in Phase 1 comparisons.
    comparison  candidate becomes the incumbent iff its (possibly lower-bound) P is strictly smaller.
    budget      max_wall_hours: no new oracle call is started after it; a running call is not interrupted.
    start_index if given, ONLY the Phase-0 point with this index is known to the seed (it is the start incumbent and the only
                cache entry), so seeds that start from different Phase-0 points are fully independent.  A feasible start ends the
                seed at once (stop_reason 'StartFeasible', 0 oracle calls, 0 h).  Default None = the published behaviour.
    Writes <save_dir>/phase1_calls/call_XXXX.json after each oracle call and <save_dir>/phase1_result.json at the end.
    """
    import os
    t_start = time.time()
    os.makedirs(os.path.join(save_dir, "phase1_calls"), exist_ok=True)
    if dragon5_cfg is None:
        dragon5_cfg = dragon5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None,
                                            abort_ppf_above=ppf_c, abort_k_above=kinf_hi, abort_k_mode="peak")
    if oracle_fn is None:
        oracle_fn = lambda pattern, call_no: _run_dragon5_with_heartbeat(pattern, dragon5_cfg, "PHASE1", call_no,
                                                                         heartbeat_sec, verbose)
    log = lambda *a: print(*a, flush=True) if verbose else None

    # ---- Phase-0 patterns: oracle values, penalties, start incumbent, cache seed ----
    with open(phase0_path) as f:
        p0 = json.load(f)
    if start_index is not None:
        p0 = [pt for pt in p0 if pt["index"] == start_index]
        if len(p0) != 1:
            raise ValueError(f"start_index {start_index} not found exactly once in {phase0_path}")
    cache, table = {}, []
    for pt in p0:
        ti = tuple(_type_idx_from_pattern(pt["pattern"]))
        P = phase1_penalty(pt["E"], pt["n_gad"], pt["real_k_inf_max"], pt["real_PPF"], ppf_c, kinf_hi)
        cache[ti] = {"P": P, "E": pt["E"], "n_gad": pt["n_gad"], "k_inf_max": pt["real_k_inf_max"],
                     "PPF": pt["real_PPF"], "CL_proxy": pt["real_CL_proxy"], "aborted": None, "source": f"phase0[{pt['index']}]"}
        table.append((pt["index"], pt["E"], pt["real_k_inf_max"], pt["real_PPF"], P))
    best_key = min(cache, key=lambda k: cache[k]["P"])
    P_best, type_idx_best = cache[best_key]["P"], list(best_key)
    log(f"[PHASE1] seed={seed} ppf_c={ppf_c} kinf_hi={kinf_hi}  Phase-0 penalties (idx, E, k_inf_max, PPF, P):")
    for row in table:
        log("    %d  E=%.4f  k=%.5f  PPF=%.4f  P=%.5f" % row)
    log(f"[PHASE1] start incumbent = {cache[best_key]['source']}  P={P_best:.5f}")
    if P_best <= 0.0 and start_index is None:
        raise ValueError("a Phase-0 pattern is already feasible for this ppf_c; Phase 1 is not needed")
    start_P = P_best

    rng = np.random.default_rng(seed)
    c = qi.init_centroid(VAR_ALLOWED, N_TYPES)
    stuck = n_calls = n_rounds_exact_ok = n_rej_exact = n_cache_hit = n_failed = 0
    calls_log = []
    stop_reason = None
    result = None

    def _save_summary(stop):
        rec = {"seed": seed, "ppf_c": ppf_c, "kinf_hi": kinf_hi, "stop_reason": stop,
               "elapsed_sec": time.time() - t_start, "n_dragon5_calls": n_calls,
               "n_aborted_ppf": sum(1 for r in calls_log if r["aborted"] and "BOL PPF" in r["aborted"]),
               "n_aborted_k": sum(1 for r in calls_log if r["aborted"] and "k_inf" in r["aborted"]),
               "n_failed": n_failed, "n_cache_hits": n_cache_hit, "n_rejected_exact": n_rej_exact,
               "oracle_sec_sum": sum(r["dragon_elapsed_s"] or 0.0 for r in calls_log),
               "best_P": P_best, "best_type_idx": [int(k) for k in type_idx_best],
               "best_source": cache[tuple(type_idx_best)]["source"], "calls": calls_log,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        if start_index is not None:
            rec["start_index"] = start_index; rec["start_P"] = start_P
        if result is not None:
            rec["feasible"] = {k: result[k] for k in ("type_idx", "pattern", "real_k_inf_max", "real_PPF",
                                                       "real_CL_proxy", "E")}
        with open(os.path.join(save_dir, "phase1_result.json"), "w") as f:
            json.dump(rec, f, indent=2, default=float)
        return rec

    if P_best <= 0.0:                                        # only reachable with start_index: the start is feasible
        c0 = cache[tuple(type_idx_best)]
        result = {"type_idx": [int(k) for k in type_idx_best], "pattern": json.loads(json.dumps(p0[0]["pattern"])),
                  "real_k_inf_max": c0["k_inf_max"], "real_PPF": c0["PPF"], "real_CL_proxy": c0["CL_proxy"], "E": c0["E"]}
        rec = _save_summary("StartFeasible")
        log(f"[PHASE1] seed={seed} start point {start_index} is already feasible: nothing to search (0 calls, 0 h)")
        return rec

    for round_i in range(1, max_rounds + 1):
        ptb = qi.perturb(c, VAR_ALLOWED, rng, mode="mult")
        type_idx = qi.greedy_round(ptb, VAR_ALLOWED, ADJACENCY_PAIRS, qi.GAD_TYPE_IDS)
        _, detail = qi.check_feasibility(type_idx, VAR_ALLOWED, PIN_TYPES, MULTIPLICITIES, ADJACENCY_PAIRS,
                                         check_e_range=False, check_ngad_range=False)
        E, n_gad = detail["E"], detail["N_GAD"]
        key = tuple(int(k) for k in type_idx)
        if not (E_LO <= E <= E_HI and NGAD_LO <= n_gad <= NGAD_HI):
            P = float("inf"); n_rej_exact += 1
        elif key in cache:
            P = cache[key]["P"]; n_cache_hit += 1
        else:
            if (time.time() - t_start) / 3600.0 >= max_wall_hours:
                stop_reason = "WallBudgetReached"; break
            if n_calls >= max_dragon5_calls:
                stop_reason = "CallCapReached"; break
            n_calls += 1
            pattern = build_grid_heuristic(np.asarray(type_idx))
            t_call = time.time()
            try:
                out = oracle_fn(pattern, n_calls)
            except Exception as e:                               # a crashed run: do not lose the seed
                n_failed += 1
                log(f"  [PHASE1] call #{n_calls} FAILED: {type(e).__name__}: {e}")
                cache[key] = {"P": float("inf"), "E": E, "n_gad": n_gad, "k_inf_max": None, "PPF": None,
                              "CL_proxy": None, "aborted": f"failed: {e}", "source": f"call{n_calls}"}
                if n_failed >= 5:
                    stop_reason = "TooManyOracleFailures"; break
                continue
            aborted = out.get("aborted")
            k_obs, ppf_obs, cl = out.get("k_inf_max"), out.get("PPF"), out.get("CL_proxy")
            if aborted and "BOL PPF" in str(aborted):
                k_obs = None                                     # only the BOL step was seen: k_inf_max not observed
            P = phase1_penalty(E, n_gad, k_obs, ppf_obs, ppf_c, kinf_hi)
            feasible = (not aborted and cl is not None and k_obs is not None and ppf_obs is not None
                        and k_obs <= kinf_hi and ppf_obs < ppf_c)
            if feasible:
                P = 0.0
            rec = {"call": n_calls, "round": round_i, "E": E, "n_gad": n_gad, "type_idx": list(key),
                   "k_inf_max": k_obs, "PPF": ppf_obs, "CL_proxy": cl, "aborted": aborted, "P": P,
                   "feasible": feasible, "dragon_elapsed_s": out.get("dragon_elapsed_s") or (time.time() - t_call),
                   "k_inf_steps": (out.get("burnup_history") or {}).get("k_inf"),
                   "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            calls_log.append({k: rec[k] for k in ("call", "round", "E", "n_gad", "k_inf_max", "PPF", "CL_proxy",
                                                  "aborted", "P", "feasible", "dragon_elapsed_s")})
            with open(os.path.join(save_dir, "phase1_calls", f"call_{n_calls:04d}.json"), "w") as f:
                json.dump(dict(rec, pin_power_map_bol=(out.get("burnup_history") or {}).get("pin_power_map_kmax")),
                          f, default=float)
            cache[key] = {"P": P, "E": E, "n_gad": n_gad, "k_inf_max": k_obs, "PPF": ppf_obs, "CL_proxy": cl,
                          "aborted": aborted, "source": f"call{n_calls}"}
            log(f"  [PHASE1] call #{n_calls} round {round_i}: E={E:.4f} N_GAD={n_gad} PPF={ppf_obs} k_inf_max={k_obs} "
                f"P={P:.5f} {'ABORTED: ' + str(aborted) if aborted else ('FEASIBLE' if feasible else '')} "
                f"[{(time.time()-t_start)/3600:.2f} h]")
            if feasible:
                result = {"type_idx": [int(k) for k in key], "pattern": json.loads(json.dumps(pattern)),
                          "real_k_inf_max": k_obs, "real_PPF": ppf_obs, "real_CL_proxy": cl, "E": E}
                P_best, type_idx_best = 0.0, list(key)
                stop_reason = "FeasibleFound"
                break

        if P < P_best:
            P_best, type_idx_best = P, list(key)
            log(f"  [PHASE1] round {round_i}: new best P={P:.5f} (E={E:.4f}, N_GAD={n_gad}, source={cache[key]['source']})")
        stuck = stuck + 1 if tuple(type_idx_best) == key else 0
        c, alpha, rmsd = qi.move_seed(c, type_idx_best, VAR_ALLOWED)
        if rng.random() < min(1.0, stuck / reset_window) * rmsd:
            c = qi.init_centroid(VAR_ALLOWED, N_TYPES); stuck = 0
            log(f"  [PHASE1] round {round_i}: RESET centroid (rmsd={rmsd:.4f})")
    else:
        stop_reason = "MaxRoundsReached"

    rec = _save_summary(stop_reason)
    log(f"[PHASE1] seed={seed} finished: {stop_reason}  best P={P_best:.5f}  calls={n_calls} "
        f"(aborted PPF {rec['n_aborted_ppf']}, k {rec['n_aborted_k']}, cache hits {n_cache_hit})  "
        f"{rec['elapsed_sec']/3600:.2f} h")
    return rec


if __name__ == "__main__":
    run(max_dragon5_calls=8)
