"""
arr_hybrid_dragon5.py
========================
Hybrid ARR objective: the surrogate (bwr10x10_surrogate) drives every
trial as usual (fast, ~ms), but real DRAGON5 (bwr10x10_dragon5) is
called ONLY at the moment a candidate's SURROGATE score would beat the
current surrogate-tracked incumbent -- i.e. roughly once per incumbent
update, not once per trial.

EVERY real DRAGON5 call's (E, k_inf_max, PPF, CL_proxy) result is
folded into the surrogate's calibration set and refit immediately
(closed-form linear least squares for k_inf_max(E)/CL_proxy(E) -- cheap
enough to redo on every single real call), REGARDLESS of whether that
call turned out feasible or actually beat the current real incumbent.
This matters most exactly when the surrogate said "feasible" but real
DRAGON5 said "infeasible" -- that disagreement is precisely where the
surrogate is most wrong, so it's especially valuable calibration data,
not something to throw away just because the candidate itself gets
rejected. Only the REAL INCUMBENT (and hence what a run ultimately
reports as its answer) is gated on being both feasible and better;
calibration improvement is not.

Cost model: from the ARR technical report's own Table 3, Phase 2 saw
2-8 incumbent updates per run (out of hundreds of trials) -- so this
design should cost on the order of a FEW TO A FEW DOZEN real DRAGON5
evaluations per run, not hundreds/thousands, while still guaranteeing
the FINAL reported incumbent is DRAGON5-real, not surrogate-optimistic.

Re-anchoring the surrogate to DRAGON5 (not CASMO4)
-----------------------------------------------------
bwr10x10_surrogate's original 3-point calibration (_K_A/_K_B, _CL_A/
_CL_B, PPF's diffusion params) was fit against the PAPER's reported
CASMO4 values for REFERENCE/RL_PPF_1_6/RL_PPF_1_4. But this module's
whole point is "a fast proxy for DRAGON5", not for CASMO4 -- so
`seed_calibration_from_dragon5()` below re-fits k_inf_max(E) and
CL_proxy(E) from DRAGON5's OWN computed values on those same 3 patterns
first (see DRAGON5_SEED_POINTS), before any ARR trials run. This means
even trial 1's surrogate is already anchored to DRAGON5-truth, and
online refits during the run only sharpen that further.

DRAGON5_SEED_POINTS values below are taken directly from this session's
full-schedule (900K, 16-step burnup, specific_power=29.436 MW/tU)
validation runs -- see bwr10x10_dragon5.py's module docstring PPF/CL
PROVENANCE notes for full provenance/uncertainty discussion. PPF is
NOT re-anchored here (see PPF note below).
"""
import json
import os
import threading
import time

import numpy as np

import bwr10x10_surrogate as surrogate
import bwr10x10_dragon5 as dragon5


class ARRStopSignal(RuntimeError):
    """Base class for internal signals that stop the ARR loop cleanly
    (as opposed to real errors). Caught by run_phase2_hybrid() (in
    run_phase2_hybrid_dragon5.py), which returns whatever real
    incumbent has been confirmed so far instead of crashing."""
    pass


class Dragon5CallBudgetReached(ARRStopSignal):
    """Raised when max_dragon5_calls real DRAGON5 verifications have
    already been spent and another would-be verification is attempted."""
    pass


class WallBudgetReached(Dragon5CallBudgetReached):
    """Raised at the start of an objective call once max_wall_sec of wall-clock time has elapsed since the objective
    was created. A DRAGON5 call already running is never interrupted, so the overrun is at most one oracle call."""
    pass


class TargetCLReached(ARRStopSignal):
    """Raised right after a real DRAGON5 call CONFIRMS a new incumbent
    whose real CL_proxy meets or exceeds target_cl. The confirmed
    incumbent (and its incumbent_latest.json save) already happened
    before this is raised -- nothing is lost."""
    pass


REJECT_SCORE = -1e9

# ----------------------------------------------------------------------
# DRAGON5-computed values for the 3 calibration patterns (this session,
# 900K/560K, full 16-step burnup schedule, specific_power=29.436 MW/tU,
# CL_proxy re-derived from the paper's own k_inf=0.95-crossing
# definition). Used to re-anchor the surrogate to DRAGON5 before any ARR
# trial runs -- see module docstring.
# ----------------------------------------------------------------------
DRAGON5_SEED_POINTS = {   # SYM oracle (2026-11-03): BOL PPF, 10-step schedule, CL=BU/29.436
    "REFERENCE":  {"E": 4.31087,  "k_inf_max": 1.091662, "PPF": 1.5644, "CL_proxy": 1469.8},
    "RL_PPF_1_6": {"E": 4.354891, "k_inf_max": 1.100310, "PPF": 1.5354, "CL_proxy": 1488.0},
    "RL_PPF_1_4": {"E": 4.35,     "k_inf_max": 1.110986, "PPF": 1.3567, "CL_proxy": 1475.7},
}
# LEGACY gap-less oracle values: REFERENCE 1.098102/1.5478/1457.7, RL_PPF_1_6 1.100753/1.3325/1465.2,
# RL_PPF_1_4 1.10314/1.5119/1466.9.


def _fit_linear(xs, ys):
    """Closed-form least-squares fit of y = a + b*x. Returns (a, b)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    A = np.vstack([np.ones_like(xs), xs]).T
    (a, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
    return float(a), float(b)


class SurrogateCalibrationState:
    """Tracks the growing set of DRAGON5-VERIFIED (E, k_inf_max, PPF,
    CL_proxy) points and refits bwr10x10_surrogate's module-level linear
    coefficients (_K_A/_K_B for k_inf_max(E), _CL_A/_CL_B for CL_proxy(E))
    in place whenever a new point is added. This directly mutates the
    imported `surrogate` module's globals, so every subsequent call to
    surrogate.k_inf_max(E)/surrogate.cl_proxy(E) (and therefore
    surrogate.casmo4_bwr10x10()) immediately reflects the refit --
    no need to re-import or restart the ARR run.

    PPF is intentionally NOT refit here by default (see `refit_ppf`):
    bwr10x10_surrogate's PPF model is a 4-parameter nonlinear diffusion
    fit (scipy.optimize.least_squares, multi-start) -- more fragile to
    refit online than the two linear models, and PPF's role in this
    hybrid design is only to help the surrogate DECIDE which candidates
    look promising enough to verify with DRAGON5 (an exploration
    heuristic) -- correctness of the FINAL incumbent is always
    guaranteed by the real DRAGON5 check regardless of PPF surrogate
    accuracy. Set refit_ppf=True to also refit it (slower, uses
    bwr10x10_surrogate_dev.fit_ppf-equivalent NLLS) if PPF surrogate
    accuracy turns out to matter for search efficiency in practice.
    """

    def __init__(self, refit_ppf=False):
        self.points = []  # list of dicts: E, k_inf_max, PPF, CL_proxy
        self.refit_ppf = refit_ppf
        self.n_refits = 0

    def seed_from_dragon5(self, seed_points=DRAGON5_SEED_POINTS):
        # NOTE (added after the PPF<1.4 QUBO-hybrid run repeatedly wasted
        # real DRAGON5 calls on candidates the surrogate wrongly thought
        # had PPF<1.4 -- actual real PPF came back ~1.52/1.54, nowhere
        # close): previously this only seeded the linear k_inf_max(E)/
        # CL_proxy(E) models with the 3 seed points' NUMBERS. The
        # nonlinear PPF diffusion model was left at whatever it started
        # as (paper-CASMO4-calibrated) until enough REAL DRAGON5 calls
        # accumulated during the run itself to refit it -- meaning early
        # in any run (and for the whole run if refit_ppf were ever
        # effectively idle), PPF predictions could be badly off. Now,
        # when refit_ppf=True, the 3 seed patterns' actual grids are ALSO
        # registered for PPF refit here, so _refit() below performs an
        # initial DRAGON5-anchored PPF fit immediately -- before any ARR
        # trial or real verification call happens.
        from bwr10x10_reference_patterns import REFERENCE, RL_PPF_1_6, RL_PPF_1_4
        _seed_patterns = {
            "REFERENCE": REFERENCE,
            "RL_PPF_1_6": RL_PPF_1_6,
            "RL_PPF_1_4": RL_PPF_1_4,
        }
        for name, p in seed_points.items():
            self.points.append(dict(p))
            if self.refit_ppf and name in _seed_patterns:
                self.register_pattern_for_ppf_refit(_seed_patterns[name], p["PPF"])
        self._refit()

    def seed_from_phase0(self, path="phase0_calibration/phase0_points.json"):
        # Seeds from Phase 0's from-scratch, deliberately-diversified
        # 7-point set (run_phase0_calibration.py) instead of the source
        # paper's 3 solved patterns (seed_from_dragon5() above, kept
        # intact for comparison/fallback -- not removed). Motivation:
        # calibrating on the paper's own already-optimized answers is a
        # methodological problem for anything claiming to search without
        # knowing the answer in advance, and it also turned out to be
        # measurably worse -- run_phase0_analysis.py confirmed the paper-
        # pattern calibration is systematically biased on real, held-out
        # data (this Phase 0 set was never used to fit it): k_inf_max RMS
        # 2.0x worse, PPF RMS 4.6x worse (bias -0.176, i.e. PPF
        # SYSTEMATICALLY under-predicted -- directly explains why the
        # PPF<1.4 QUBO-hybrid run kept sending doomed candidates to real
        # DRAGON5), and CL_proxy RMS 2.6x worse (bias +9.09 days, i.e.
        # CL_proxy systematically OVER-predicted). Phase 0's own 7 points
        # (4 PPF parameters, so 3 residual degrees of freedom) show no
        # such bias once fit to themselves.
        import json
        with open(path) as f:
            points = json.load(f)
        for p in points:
            self.points.append({"E": p["E"], "k_inf_max": p["real_k_inf_max"],
                                 "PPF": p["real_PPF"], "CL_proxy": p["real_CL_proxy"]})
            if self.refit_ppf:
                self.register_pattern_for_ppf_refit(p["pattern"], p["real_PPF"])
        self._refit()

    def add_verified_point(self, E, k_inf_max, PPF, CL_proxy):
        self.points.append({"E": E, "k_inf_max": k_inf_max, "PPF": PPF, "CL_proxy": CL_proxy})
        self._refit()

    def _refit(self):
        if len(self.points) < 2:
            return  # need at least 2 points for a line
        Es = [p["E"] for p in self.points]
        kinfs = [p["k_inf_max"] for p in self.points]
        cls = [p["CL_proxy"] for p in self.points]

        k_a, k_b = _fit_linear(Es, kinfs)
        cl_a, cl_b = _fit_linear(Es, cls)
        surrogate._K_A, surrogate._K_B = k_a, k_b
        surrogate._CL_A, surrogate._CL_B = cl_a, cl_b
        self.n_refits += 1

        if self.refit_ppf and len(self.points) >= 3:
            self._refit_ppf()

    def _refit_ppf(self):
        # Lightweight re-fit of the PPF diffusion model's scalar
        # parameters against the CURRENT calibration set's (pattern,
        # PPF) pairs. Requires the actual patterns, not just (E, PPF)
        # summaries, since the diffusion model needs the full enrich/
        # gad/water grids -- callers that want this must additionally
        # register patterns via `register_pattern_for_ppf_refit`.
        # If no patterns are registered yet, this silently no-ops.
        if not getattr(self, "_ppf_patterns", None):
            return
        from scipy.optimize import least_squares

        def residuals(params):
            a2, a_gad, D, de = params
            # BUG FIX (found 2026-08-15): params were unpacked but never
            # actually applied before calling surrogate.ppf() below --
            # that function reads surrogate._A2/_A_GAD/_D/_DE as MODULE
            # GLOBALS, not as arguments, so every trial `params`
            # least_squares tried produced an IDENTICAL residual (the
            # objective didn't depend on params at all), and the fit
            # always silently converged back to its own x0 starting
            # point -- confirmed directly: BEFORE/AFTER surrogate._A2
            # etc. were bit-for-bit identical after seed_from_dragon5()
            # with refit_ppf=True. This means _refit_ppf() had been a
            # complete no-op (never actually adjusting the PPF model)
            # for as long as this code existed, regardless of how many
            # real DRAGON5 points were folded in during a run. Fix: set
            # the module globals to the CURRENT trial params before
            # calling surrogate.ppf(), so the residual actually responds
            # to what least_squares is trying.
            surrogate._A2, surrogate._A_GAD, surrogate._D, surrogate._DE = a2, a_gad, D, de
            res = []
            for entry in self._ppf_patterns:
                enrich, gad, water = surrogate.pattern_to_grids(entry["pattern"])
                pred = surrogate.ppf(enrich, gad, water)
                res.append(pred - entry["PPF"])
            return np.array(res)

        x0 = [surrogate._A2, surrogate._A_GAD, surrogate._D, surrogate._DE]
        try:
            sol = least_squares(residuals, x0, method="trf",
                                 bounds=([1e-6, 0, 1e-3, 1e-3], [1.0, 1.0, 10.0, 10.0]),
                                 max_nfev=200)
            surrogate._A2, surrogate._A_GAD, surrogate._D, surrogate._DE = sol.x
        except Exception:
            pass  # keep previous PPF params if refit fails -- never crash the ARR run over this

    def register_pattern_for_ppf_refit(self, pattern, PPF):
        if not hasattr(self, "_ppf_patterns"):
            self._ppf_patterns = []
        self._ppf_patterns.append({"pattern": pattern, "PPF": PPF})


class HybridVerifiedObjective:
    """Wraps a surrogate-based Phase-2/3-style objective with DRAGON5
    verification at incumbent-beating moments. Use as the `objective`
    argument to arr_optimize_categorical.

    pattern_from_x : callable(x) -> pattern (10x10 nested list)
        Same conversion the surrogate-only driver uses (e.g.
        `lambda x: build_grid_heuristic(x_to_type_idx(x))` for Phase 2,
        or the unmasked 51-var equivalent for Phase 3).
    kinf_hi, ppf_c : real feasibility bounds (same constants used by the
        surrogate-only drivers -- KINF_HI, ppf_c).
    cfg : bwr10x10_dragon5.Dragon5Config to use for real verification
        calls. Defaults to a fresh Dragon5Config(compute_ppf=True,
        compute_cl=True) if not given -- BOTH are needed since the real
        feasibility check needs PPF and the real objective needs CL.
        This is the expensive part (~tens of minutes per call with the
        full 16-step schedule) -- see module docstring's cost model.
    heartbeat_sec : while a real DRAGON5 call is blocking (it's a
        synchronous subprocess call -- see bwr10x10_dragon5._run_dragon),
        a background thread prints a "still running, Ns elapsed" message
        every this many seconds, so a long silent gap doesn't look like
        a hang. Set to None to disable.
    phase_label : printed on every trial/verify message so a log clearly
        shows which phase is running (e.g. "PHASE 2") -- purely cosmetic.
    save_dir : directory (created if needed) where every real DRAGON5
        call's pattern + result gets written to disk as JSON, and where
        the current confirmed incumbent is kept up to date in
        `incumbent_latest.json` -- so if the process is killed/crashes,
        the actual PATTERN (not just summary numbers) of the last
        confirmed incumbent is always recoverable from disk, not lost
        with the process's memory. Set to None to disable saving
        entirely. Default "hybrid_patterns" (relative to cwd).
    max_dragon5_calls : int or None
        Hard cap on the number of REAL DRAGON5 calls this objective will
        ever make. Once self.n_dragon5_calls has reached this value, any
        further attempt to verify a candidate for real raises
        Dragon5CallBudgetReached instead of calling DRAGON5 again --
        intended to be caught by the caller (run_phase2_hybrid_dragon5.py
        does this) to stop the whole ARR run cleanly. Default None =
        unlimited (unchanged behavior).
    target_cl : float or None
        Hard stopping criterion on quality: as soon as a CONFIRMED real
        incumbent's real CL_proxy meets or exceeds this value, raises
        TargetCLReached (a subclass of ARRStopSignal, same as
        Dragon5CallBudgetReached) -- intended to be caught the same way
        by the caller. Default None = no CL-based stop.
    """

    def __init__(self, pattern_from_x, kinf_hi, ppf_c, calib_state,
                 cfg=None, verbose=True, log=None, heartbeat_sec=60,
                 phase_label="PHASE 2", save_dir="hybrid_patterns",
                 max_dragon5_calls=None, target_cl=None, tie_tol=1e-6,
                 max_wall_sec=None, gate_margin_days=0.0, surrogate_ppf_gate=True, use_surrogate_gate=True):
        self.pattern_from_x = pattern_from_x
        self.kinf_hi = kinf_hi
        self.ppf_c = ppf_c
        self.calib = calib_state
        self.cfg = cfg if cfg is not None else dragon5.Dragon5Config(
            compute_ppf=True, compute_cl=True, cache_path=None)
        self.verbose = verbose
        self.log = log if log is not None else []
        self.heartbeat_sec = heartbeat_sec
        self.phase_label = phase_label
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.surrogate_best_score = REJECT_SCORE
        self.verified_cache = {}  # (2026-08-22) tuple(type_idx) -> real_out dict,
        # for exact-pattern real-DRAGON5-result reuse -- see __call__.
        self.real_best_score = None
        self.real_best_pattern = None
        self.n_calls = 0  # every __call__, whether or not it triggers DRAGON5
        self.n_surrogate_calls = 0
        self.n_dragon5_calls = 0
        self.n_dragon5_confirmed = 0
        self.n_dragon5_rejected = 0
        self.max_dragon5_calls = max_dragon5_calls  # None = unlimited
        # (2026-11-06) wall-clock budget (seconds since construction; None = unlimited). With early abort an oracle call
        # costs ~19-105 min depending on how it fails, so a call-count budget no longer measures the cost.
        self.max_wall_sec = max_wall_sec
        # (2026-11-06) gate_margin_days: a candidate goes to the oracle when its surrogate CL > surrogate_best -
        # gate_margin_days + tie_tol. The surrogate CL is a function of E alone, so with margin 0 only candidates with a
        # STRICTLY higher E than the incumbent are ever verified, and the same-E position effect (~9-11 d of real CL at
        # fixed E) can never be tried. margin 0.0 = old behaviour. Exact repeats are served from verified_cache.
        self.gate_margin_days = float(gate_margin_days)
        # (2026-11-06) surrogate_ppf_gate: when False the 1-group-diffusion PPF surrogate no longer rejects candidates
        # (the oracle decides PPF; with early abort a PPF failure costs ~19 min). Measured: with the Phase-0(SYM)+#11
        # calibration the surrogate PPF is 1.45-1.64 (median 1.48) for ALL candidates, so at c=1.4 the gate rejects
        # everything and the oracle would never be called. True = old behaviour.
        self.surrogate_ppf_gate = bool(surrogate_ppf_gate)
        # (2026-11-10, ablation) use_surrogate_gate: False -> the surrogate ranks nothing: neither the k_inf nor the CL gate
        # rejects a candidate, so EVERY candidate that passes the exact checks and is not in the cache goes to the oracle.
        # The surrogate is still refit on completed oracle runs (its scores are only printed). True = old behaviour.
        self.use_surrogate_gate = bool(use_surrogate_gate)
        self._t_start = time.time()
        self.target_cl = target_cl  # None = no CL-based stop
        # (2026-11-03) The CL surrogate is a function of the average enrichment E ALONE, so distinct patterns with the
        # same E have the same surrogate CL up to floating-point round-off (~1e-12 d). A strict `>` then fires a real
        # oracle call on a 1-ulp tie (observed: 4 of the 12 calls of the best PPF<1.4 seed). A candidate must beat the
        # tracked surrogate best by MORE than tie_tol days (default 1e-6 d; the smallest genuine E step is ~0.15 d).
        # tie_tol=0.0 reproduces the old behaviour.
        self.tie_tol = tie_tol
        # type_idx tuple of the oracle-CONFIRMED incumbent (set at confirmation; the driver seeds it from Phase 1).
        self.real_best_x = None

    def _save_pattern(self, filename, pattern, extra):
        """Writes {filename} in self.save_dir as JSON: the actual 10x10
        pattern plus whatever result/metadata fields are in `extra`.
        No-op if save_dir is None. Never lets a save failure crash the
        real run -- prints a warning and continues instead."""
        if self.save_dir is None:
            return
        try:
            path = os.path.join(self.save_dir, filename)
            payload = dict(extra)
            payload["pattern"] = pattern
            payload["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(path, "w") as f:
                json.dump(payload, f, indent=1)
        except Exception as e:
            print(f"    WARNING: failed to save pattern to {filename}: {e}", flush=True)

    def _run_dragon5_with_heartbeat(self, pattern):
        """Runs dragon5.casmo4_bwr10x10 in the current thread (still
        blocking -- DRAGON5 itself is a synchronous subprocess call, see
        bwr10x10_dragon5._run_dragon) while a separate daemon thread
        prints periodic elapsed-time heartbeats so a long silent gap is
        visibly "still working", not indistinguishable from a hang."""
        result_box = {}
        stop_event = threading.Event()
        t_start = time.time()

        def _heartbeat():
            n = 0
            while not stop_event.wait(self.heartbeat_sec):
                n += 1
                elapsed = time.time() - t_start
                print(f"    [{self.phase_label}] ... still running real DRAGON5 "
                      f"(call #{self.n_dragon5_calls}), {elapsed:.0f}s elapsed "
                      f"({elapsed/60:.1f} min)", flush=True)

        def _worker():
            try:
                result_box["out"] = dragon5.casmo4_bwr10x10(pattern, self.cfg)
            except Exception as e:
                # (2026-08-20) Without this, an exception raised inside this
                # background thread (e.g. DRAGON5 killed/crashed mid-run,
                # RuntimeError from bwr10x10_dragon5._run_dragon) used to be
                # printed by Python's default threading exception hook and
                # then SWALLOWED -- result_box["out"] was simply never set,
                # so the caller's `return result_box["out"]` below crashed
                # with an opaque KeyError: 'out' that hid the real cause.
                # Stashing it here lets the caller re-raise the ORIGINAL
                # exception with its own message intact.
                result_box["error"] = e

        hb_thread = None
        if self.heartbeat_sec and self.verbose:
            hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            hb_thread.start()

        worker_thread = threading.Thread(target=_worker)
        worker_thread.start()
        worker_thread.join()
        stop_event.set()
        if hb_thread is not None:
            hb_thread.join(timeout=1)

        elapsed = time.time() - t_start
        if "error" in result_box:
            if self.verbose:
                print(f"    [{self.phase_label}] real DRAGON5 call #{self.n_dragon5_calls} "
                      f"FAILED after {elapsed:.0f}s ({elapsed/60:.1f} min): "
                      f"{type(result_box['error']).__name__}: {result_box['error']}",
                      flush=True)
            raise result_box["error"]
        if self.verbose:
            print(f"    [{self.phase_label}] real DRAGON5 call #{self.n_dragon5_calls} "
                  f"finished in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
        return result_box["out"]

    def _constraint_set1(self, pattern):
        enrich, gad, water = surrogate.pattern_to_grids(pattern)
        E = surrogate.average_enrichment(enrich, water)
        n_gad = int(np.sum((~water) & (gad > 0)))
        from run_phase1_heuristic_10x10 import E_LO, E_HI, NGAD_LO, NGAD_HI
        feasible = (E_LO <= E <= E_HI) and (NGAD_LO <= n_gad <= NGAD_HI)
        return E, n_gad, feasible

    def surrogate_feasible_score(self, x):
        """(2026-08-20) Cheap prescreen: constraint-set-1 (E, N_GAD) +
        surrogate k_inf_max/PPF check -- WITHOUT ever triggering a real
        DRAGON5 call or touching incumbent/threshold state. Returns
        (feasible, surr_score): surr_score is the surrogate CL_proxy if
        feasible, else None.

        Purpose: multi-candidate-per-round Phase 2 variants (e.g.
        run_greedy_phase2_multicandidate) that draw many perturbed
        candidates per round and want to rank them by surrogate
        quality BEFORE handing just the single best one to __call__
        (which is where a real DRAGON5 call may actually get
        triggered, at most once per round regardless of how many
        candidates were screened here).
        """
        self.n_calls += 1
        pattern = self.pattern_from_x(x)
        E, n_gad, cs1_feasible = self._constraint_set1(pattern)
        if not cs1_feasible:
            return False, None
        self.n_surrogate_calls += 1
        surr_out = surrogate.casmo4_bwr10x10(pattern)
        surr_feasible = surr_out["k_inf_max"] <= self.kinf_hi and (not self.surrogate_ppf_gate or surr_out["PPF"] < self.ppf_c)
        if not surr_feasible:
            return False, None
        return True, surr_out["CL_proxy"]

    def __call__(self, x):
        if self.max_wall_sec is not None and time.time() - self._t_start >= self.max_wall_sec:
            raise WallBudgetReached(
                f"max_wall_sec={self.max_wall_sec:.0f} reached ({(time.time() - self._t_start) / 3600:.2f} h elapsed, "
                f"{self.n_dragon5_calls} oracle calls) -- stopping.")
        self.n_calls += 1
        pattern = self.pattern_from_x(x)

        E, n_gad, cs1_feasible = self._constraint_set1(pattern)
        if not cs1_feasible:
            return REJECT_SCORE

        self.n_surrogate_calls += 1
        surr_out = surrogate.casmo4_bwr10x10(pattern)
        surr_feasible = surr_out["k_inf_max"] <= self.kinf_hi and (not self.surrogate_ppf_gate or surr_out["PPF"] < self.ppf_c)
        if self.use_surrogate_gate and not surr_feasible:
            return REJECT_SCORE
        surr_score = surr_out["CL_proxy"]

        if self.use_surrogate_gate and surr_score <= self.surrogate_best_score - self.gate_margin_days + self.tie_tol:
            # Surrogate itself doesn't think this beats the tracked
            # incumbent -- no need to spend a real DRAGON5 call.
            return surr_score

        # Surrogate says this WOULD be a new incumbent -- verify for real,
        # UNLESS this EXACT pattern was already real-DRAGON5-verified
        # before (2026-08-22 fix: without this, a centroid that keeps
        # regenerating the same type_idx -- observed burning through
        # several real-call budget slots re-confirming an identical
        # already-rejected pattern each time the surrogate got re-fit
        # and momentarily re-crossed the threshold for it -- would waste
        # real DRAGON5 calls re-verifying something whose outcome is
        # already known with certainty; real DRAGON5 is deterministic,
        # so re-running it can never change the answer). The cache key
        # is the exact type_idx tuple, not anything surrogate-derived.
        key = tuple(int(k) for k in x)
        is_cache_hit = key in self.verified_cache

        if is_cache_hit:
            real_out = self.verified_cache[key]
            if self.verbose:
                now = time.strftime("%H:%M:%S")
                print(f"  [{self.phase_label}] [trial-call #{self.n_calls}] "
                      f"[CACHE HIT] @ {now} -- surrogate says CL_proxy={surr_score:.2f} "
                      f"(E={E:.4f}) beats current surrogate-best="
                      f"{self.surrogate_best_score:.2f}, but this EXACT pattern was "
                      f"already real-DRAGON5-verified (real_CL_proxy="
                      f"{real_out['CL_proxy'] if real_out['CL_proxy'] is None else format(real_out['CL_proxy'], '.2f')}"
                      f", aborted={real_out.get('aborted')}) -- reusing that result instead of "
                      f"spending another real DRAGON5 call.", flush=True)
        else:
            if self.max_dragon5_calls is not None and self.n_dragon5_calls >= self.max_dragon5_calls:
                raise Dragon5CallBudgetReached(
                    f"max_dragon5_calls={self.max_dragon5_calls} reached -- "
                    f"stopping before a {self.n_dragon5_calls + 1}th real DRAGON5 call."
                )
            self.n_dragon5_calls += 1
            if self.verbose:
                now = time.strftime("%H:%M:%S")
                print(f"  [{self.phase_label}] [trial-call #{self.n_calls}] "
                      f"[DRAGON5 verify #{self.n_dragon5_calls}] @ {now} -- surrogate says "
                      f"CL_proxy={surr_score:.2f} (E={E:.4f}) beats current "
                      f"surrogate-best={self.surrogate_best_score:.2f} -- running real DRAGON5 "
                      f"(compute_ppf={self.cfg.compute_ppf}, compute_cl={self.cfg.compute_cl}, "
                      f"{len(self.cfg.burnup_steps_mwd_tu)}-step schedule) ...", flush=True)
            real_out = self._run_dragon5_with_heartbeat(pattern)
            self.verified_cache[key] = real_out
        real_feasible = (real_out["k_inf_max"] is not None and real_out["PPF"] is not None
                          and real_out["k_inf_max"] <= self.kinf_hi and real_out["PPF"] < self.ppf_c
                          and not real_out.get("aborted") and real_out.get("CL_proxy") is not None)
        real_cl = real_out["CL_proxy"]

        # Everything below this point (disk logging, calibration folding,
        # per-call counters) applies ONLY to a genuine new real call --
        # a cache hit re-derives no new information, so there is nothing
        # fresh to log or refit on, and it must NOT count against
        # max_dragon5_calls or the confirmed/rejected tallies (those
        # already happened, correctly, the first time this pattern was
        # seen).
        if not is_cache_hit:
            entry = {
                "n_dragon5_calls": self.n_dragon5_calls, "E": E, "n_gad": n_gad,
                "surrogate_CL": surr_score, "real_k_inf_max": real_out["k_inf_max"],
                "real_PPF": real_out["PPF"], "real_CL_proxy": real_cl,
                "real_feasible": real_feasible, "aborted": real_out.get("aborted"),
                "dragon_elapsed_s": real_out.get("dragon_elapsed_s"),
            }
            self.log.append(entry)

            # Save THIS call's actual pattern + result to disk right away --
            # before any accept/reject branching below -- so even if the
            # process is killed/crashes immediately after this point, the
            # pattern (not just summary numbers) is already recoverable from
            # disk. One file per real call, plus (below, only on confirmed
            # updates) a single always-current `incumbent_latest.json`.
            self._save_pattern(
                f"call_{self.n_dragon5_calls:04d}.json", pattern,
                {"n_dragon5_calls": self.n_dragon5_calls, "E": E, "n_gad": n_gad,
                 "surrogate_CL": surr_score, "real_k_inf_max": real_out["k_inf_max"],
                 "real_PPF": real_out["PPF"], "real_CL_proxy": real_cl,
                 "real_feasible": real_feasible, "aborted": real_out.get("aborted"),
                 "dragon_elapsed_s": real_out.get("dragon_elapsed_s"),
                 "k_inf_steps": real_out.get("burnup_history", {}).get("k_inf"),
                 "pin_power_map_bol": real_out.get("burnup_history", {}).get("pin_power_map_kmax")},
            )

            # Fold this real data point into the surrogate's calibration set
            # and refit IMMEDIATELY, regardless of whether it's feasible or
            # beats the current incumbent -- an infeasible point (surrogate
            # said OK, real DRAGON5 said no) is exactly where the surrogate
            # is most wrong, so it's especially valuable calibration data,
            # not something to discard. Requires all three real values to be
            # non-None (a hard DRAGON5 run failure, as opposed to a mere
            # infeasibility, would leave these None -- don't fit on that).
            refit_happened = False
            if (real_out["k_inf_max"] is not None and real_out["PPF"] is not None
                    and real_cl is not None):
                # NOTE: register BEFORE add_verified_point -- add_verified_point
                # triggers _refit() -> _refit_ppf() immediately, which reads
                # self._ppf_patterns as it stands AT THAT MOMENT. If registered
                # after, this call's own PPF data isn't seen until the NEXT
                # call's refit fires -- a one-call lag. (Found + fixed 2026-08-11.)
                if self.calib.refit_ppf:
                    self.calib.register_pattern_for_ppf_refit(pattern, real_out["PPF"])
                self.calib.add_verified_point(E, real_out["k_inf_max"], real_out["PPF"], real_cl)
                refit_happened = True
                if self.verbose:
                    print(f"    (folded into surrogate calibration -- refit #{self.calib.n_refits}, "
                          f"{len(self.calib.points)} verified points total)")

            # CRITICAL: surrogate_best_score is the THRESHOLD every future
            # candidate's surrogate score gets compared against to decide
            # "is this worth a real DRAGON5 call?" -- it must stay on the
            # SAME coefficient scale as those future evaluations. The refit
            # above just changed bwr10x10_surrogate's global k_inf_max(E)/
            # CL_proxy(E) coefficients, but `surr_score` (computed earlier in
            # this call, BEFORE the refit) is stale w.r.t. the NEW
            # coefficients. So: whenever a refit happens AND we already have
            # a real incumbent pattern, re-evaluate THAT pattern's surrogate
            # score under the fresh coefficients and use that as the
            # threshold going forward -- not the pre-refit surr_score, even
            # in the branches below where this particular candidate gets
            # rejected (the refit still happened; the threshold still needs
            # to be re-based to match it).
            if refit_happened and self.real_best_pattern is not None:
                self.surrogate_best_score = surrogate.casmo4_bwr10x10(
                    self.real_best_pattern)["CL_proxy"]
                if self.verbose:
                    print(f"    (surrogate_best_score re-based to "
                          f"{self.surrogate_best_score:.2f} under refit coefficients)")

        if not real_feasible:
            if not is_cache_hit:
                self.n_dragon5_rejected += 1
            if self.verbose:
                cache_note = " [cached]" if is_cache_hit else ""
                ab = f" [ABORTED early: {real_out['aborted']}]" if real_out.get("aborted") else ""
                print(f"    -> DRAGON5 says INFEASIBLE (k_inf_max={real_out['k_inf_max']}, "
                      f"PPF={real_out['PPF']}){ab} -- rejecting, NOT updating incumbent.{cache_note}")
            # Reject this candidate for the optimizer's bookkeeping too --
            # a surrogate-optimistic-but-really-infeasible point must not
            # become the new center of the seed's random walk.
            return REJECT_SCORE if self.surrogate_best_score == REJECT_SCORE else self.surrogate_best_score

        is_new_real_best = (self.real_best_score is None) or (real_cl > self.real_best_score)
        if not is_new_real_best:
            if not is_cache_hit:
                self.n_dragon5_rejected += 1
            if self.verbose:
                cache_note = " [cached]" if is_cache_hit else ""
                print(f"    -> DRAGON5 feasible but real CL_proxy={real_cl:.2f} does NOT beat "
                      f"real incumbent={self.real_best_score:.2f} -- NOT updating incumbent."
                      f"{cache_note}")
            return self.surrogate_best_score

        # Confirmed: genuinely better AND feasible per real DRAGON5.
        if not is_cache_hit:
            self.n_dragon5_confirmed += 1
        self.real_best_score = real_cl
        self.real_best_pattern = pattern
        self.real_best_x = key
        # Re-base the threshold to THIS (new) incumbent, under the
        # latest (already-refit) coefficients -- consistent with the
        # re-basing done above for the reject branches. Also RETURN this
        # re-based value (not the stale pre-refit surr_score) -- this
        # becomes the optimizer's own internal score_best, which must
        # stay on the same coefficient scale as every future "no
        # verification needed" fast-path return (always freshly computed
        # under whatever coefficients are live at call time), or the
        # optimizer's own incumbent-tracking suffers the identical
        # staleness problem this whole fix is for.
        self.surrogate_best_score = surrogate.casmo4_bwr10x10(pattern)["CL_proxy"]
        self._save_pattern(
            "incumbent_latest.json", pattern,
            {"n_dragon5_calls": self.n_dragon5_calls, "n_dragon5_confirmed": self.n_dragon5_confirmed,
             "E": E, "n_gad": n_gad, "real_k_inf_max": real_out["k_inf_max"],
             "real_PPF": real_out["PPF"], "real_CL_proxy": real_cl},
        )
        if self.verbose:
            print(f"    -> CONFIRMED real incumbent update: CL_proxy={real_cl:.2f} "
                  f"(surrogate refit #{self.calib.n_refits}, "
                  f"surrogate_best_score={self.surrogate_best_score:.2f}, "
                  f"saved to {self.save_dir}/incumbent_latest.json)")
        if self.target_cl is not None and real_cl >= self.target_cl:
            raise TargetCLReached(
                f"target_cl={self.target_cl} reached -- confirmed real_cl={real_cl:.2f} "
                f"(incumbent already saved to {self.save_dir}/incumbent_latest.json)."
            )
        return self.surrogate_best_score
