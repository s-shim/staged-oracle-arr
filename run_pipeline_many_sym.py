"""
Phase 1 -> Phase 2 pipeline at scale (PPF < 1.4): many independent seeds in parallel, one oracle call per core,
every seed runs for the SAME total wall budget (default 24 h) and is never stopped early.

Starting point = the result of an earlier many-seed Phase 1 run (run_phase1_many_sym.py --start-each ...):

  seed N ended Phase 1 with a feasible pattern  (FeasibleFound / StartFeasible)
        -> Phase 2 starts at once from that pattern.  Its feasibility and cycle length are taken from the stored Phase-1 oracle
           result: NO oracle call is spent on checking the start.
  seed N ended Phase 1 without one (WallBudgetReached)
        -> Phase 1 is continued from its own earlier calls (all stored calls are in the cache, the best one is the incumbent; a new
           random stream is used) until the first oracle-certified feasible pattern; Phase 2 then starts from it and uses the rest of
           the budget.  If no feasible pattern is found, the seed spends the whole budget in Phase 1.

The wall budget (--wall-hours) is counted from the start of THIS run and covers the continued Phase 1 plus Phase 2; the time Phase 1
used in the earlier run is not included (it is reported as prior_phase1_hours).  No new oracle call starts after the budget; a running
call is finished (overrun <= one call).  Phase 2 never stops at a target cycle length (target_cl=False): it always uses the whole
remaining budget, so the cycle length reached after a given time can be read off the call files.

Phase 2 settings are those of the published / 128-seed Phase 2 run (reproduce.py): gate margin 3 days, no PPF gate in the surrogate,
early abort of the oracle, surrogate refit.  Surrogate calibration = all points of --phase0 (the 128-point Phase 0); the oracle cache
of a seed = all Phase-0 points + all of its own Phase-1 oracle calls (never evaluated again).

  python run_pipeline_many_sym.py --seeds 0-127 --phase0 phase0_calibration_128/phase0_points.json \\
         --phase1 runs/phase1_from128 --wall-hours 24 --out runs/pipeline_from128

Output per seed: <out>/seed_<N>/pipeline_seed<N>.log, [phase1_calls/ (continued Phase 1 only)], hybrid_patterns/call_XXXX.json and
incumbent_latest.json (Phase 2, same format as reproduce.py phase2), pipeline_result.json and result_summary.json (written at the end).
"""
import argparse, glob, json, multiprocessing, os, sys, time

PPF_C = 1.4
KINF_HI = 1.12
GATE_MARGIN_DAYS = 3.0
SURROGATE_PPF_GATE = False
MAX_DRAGON5_CALLS = 500            # safety cap (the real limit is the wall budget)
MAX_ROUNDS = 100000
RNG_OFFSET_PHASE1_RESUME = 500000  # continued Phase 1: default_rng(seed + this); Phase 2 uses seed + 100000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEASIBLE_STOPS = ("FeasibleFound", "StartFeasible")


def parse_seeds(spec):
    seeds = []
    for part in spec:
        for tok in part.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                lo, hi = tok.split("-", 1)
                seeds.extend(range(int(lo), int(hi) + 1))
            else:
                seeds.append(int(tok))
    if len(set(seeds)) != len(seeds):
        raise SystemExit("duplicate seeds in --seeds")
    return seeds


def _fake_phase1_oracle(pattern, call_no):
    sys.path.insert(0, BASE_DIR)
    from run_phase1_many_sym import _fake_oracle
    return _fake_oracle(pattern, call_no)


def _fake_phase2_oracle(pattern, cfg=None):
    """Self-test only: replaces bwr10x10_dragon5.casmo4_bwr10x10 (same output fields)."""
    out = _fake_phase1_oracle(pattern, 0)
    return out


def _prior_entries(phase0_points, call_dirs):
    """Oracle results already known to a seed: all Phase-0 points (already in the surrogate calibration) and every oracle call of its own
    Phase 1 (aborted calls are infeasible; the complete one is the start pattern itself and is skipped by run_greedy_phase2)."""
    from run_qubo_hybrid_dragon5 import _type_idx_from_pattern
    entries, seen = [], set()
    for pt in phase0_points:
        ti = [int(k) for k in _type_idx_from_pattern(pt["pattern"])]
        if tuple(ti) in seen:
            continue
        seen.add(tuple(ti))
        entries.append({"type_idx": ti, "E": pt["E"], "k_inf_max": pt["real_k_inf_max"], "PPF": pt["real_PPF"],
                        "CL_proxy": pt["real_CL_proxy"], "aborted": None, "calibrated_elsewhere": True})
    for d in call_dirs:
        for cp in sorted(glob.glob(os.path.join(d, "phase1_calls", "call_*.json"))):
            r = json.load(open(cp))
            ti = [int(k) for k in r["type_idx"]]
            if tuple(ti) in seen:
                continue
            seen.add(tuple(ti))
            entries.append({"type_idx": ti, "E": r["E"], "k_inf_max": r.get("k_inf_max"), "PPF": r.get("PPF"),
                            "CL_proxy": r.get("CL_proxy"), "aborted": r.get("aborted"), "calibrated_elsewhere": False})
    return entries


def worker(seed, a):
    t_run = time.time()
    sys.path.insert(0, BASE_DIR)
    work_dir = os.path.join(a["out"], f"seed_{seed}")
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)                                   # Phase 2 writes hybrid_patterns/ relative to the cwd
    log_file = open(os.path.join(work_dir, f"pipeline_seed{seed}.log"), "a", buffering=1)
    sys.stdout = sys.stderr = log_file
    print(f"=== pipeline worker seed={seed} PID={os.getpid()} started @ {time.strftime('%Y-%m-%d %H:%M:%S')} "
          f"(budget {a['wall_hours']} h) ===", flush=True)
    fake = a["fake"]
    if fake:
        import bwr10x10_dragon5
        bwr10x10_dragon5.casmo4_bwr10x10 = _fake_phase2_oracle
    from run_qubo_hybrid_dragon5 import run_greedy_phase1_oracle, run_greedy_phase2
    summary = {"seed": seed, "wall_hours_budget": a["wall_hours"], "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        p1_dir = os.path.join(a["phase1"], f"seed_{seed}")
        p1 = json.load(open(os.path.join(p1_dir, "phase1_result.json")))
        if "error" in p1:
            raise RuntimeError(f"earlier Phase 1 of seed {seed} failed: {p1['error']}")
        phase0_points = json.load(open(a["phase0"]))
        summary.update(prior_phase1_stop=p1["stop_reason"], prior_phase1_hours=p1["elapsed_sec"] / 3600.0,
                       prior_phase1_calls=p1["n_dragon5_calls"], start_index=p1.get("start_index"))
        call_dirs = [p1_dir]
        feas = p1.get("feasible")
        n_calls_p1 = p1["n_dragon5_calls"]
        n_rej_p1 = p1["n_aborted_ppf"] + p1["n_aborted_k"]
        p1b = None
        if p1["stop_reason"] in FEASIBLE_STOPS:
            summary["route"] = "phase2_direct"
            print(f"[PIPELINE] seed {seed}: Phase 1 already feasible ({p1['stop_reason']}, {summary['prior_phase1_hours']:.2f} h, "
                  f"{n_calls_p1} calls) -> Phase 2 at once, no check of the start", flush=True)
        else:
            summary["route"] = "phase1_then_phase2"
            print(f"[PIPELINE] seed {seed}: Phase 1 ended infeasible ({p1['stop_reason']}, best P={p1['best_P']:.5f}, "
                  f"{n_calls_p1} calls) -> continue Phase 1 for up to {a['wall_hours']} h", flush=True)
            kw = {"oracle_fn": _fake_phase1_oracle} if fake else {}
            p1b = run_greedy_phase1_oracle(seed=seed, ppf_c=PPF_C, kinf_hi=KINF_HI, phase0_path=a["phase0"],
                                           save_dir=work_dir, max_wall_hours=a["wall_hours"],
                                           max_dragon5_calls=MAX_DRAGON5_CALLS, verbose=True, start_index=p1["start_index"],
                                           resume_dir=p1_dir, rng_offset=RNG_OFFSET_PHASE1_RESUME, **kw)
            call_dirs.append(work_dir)
            summary.update(phase1b_stop=p1b["stop_reason"], phase1b_hours=p1b["elapsed_sec"] / 3600.0,
                           phase1b_calls=p1b["n_dragon5_calls"], phase1b_aborted_ppf=p1b["n_aborted_ppf"],
                           phase1b_aborted_k=p1b["n_aborted_k"])
            n_calls_p1 += p1b["n_dragon5_calls"]
            n_rej_p1 += p1b["n_aborted_ppf"] + p1b["n_aborted_k"]
            feas = p1b.get("feasible") if p1b["stop_reason"] == "FeasibleFound" else None
        if feas is None:
            summary.update(stop_reason="NoFeasibleInBudget", phase2_started=False, elapsed_sec=time.time() - t_run,
                           real_best_score=None, n_dragon5_calls=0, n_dragon5_confirmed=0, n_dragon5_rejected=0, q4c_phase2=0)
            print(f"[PIPELINE] seed {seed}: no feasible pattern within the budget; Phase 2 not started", flush=True)
        else:
            remaining_h = a["wall_hours"] - (time.time() - t_run) / 3600.0
            summary.update(start_CL=feas["real_CL_proxy"], start_PPF=feas["real_PPF"], start_k_inf_max=feas["real_k_inf_max"],
                           start_E=feas["E"], start_type_idx=feas["type_idx"],
                           phase2_start_h=(time.time() - t_run) / 3600.0)
            if remaining_h <= 0:
                summary.update(stop_reason="NoBudgetLeftForPhase2", phase2_started=False, elapsed_sec=time.time() - t_run,
                               real_best_score=feas["real_CL_proxy"], n_dragon5_calls=0, n_dragon5_confirmed=0,
                               n_dragon5_rejected=0, q4c_phase2=0)
            else:
                prior_path = os.path.join(work_dir, "prior_evaluations.json")
                json.dump({"entries": _prior_entries(phase0_points, call_dirs)}, open(prior_path, "w"))
                phase1 = {"type_idx": feas["type_idx"], "E": feas["E"], "real_k_inf_max": feas["real_k_inf_max"],
                          "real_PPF": feas["real_PPF"], "real_CL_proxy": feas["real_CL_proxy"], "q4c": None,
                          "n_dragon5_calls": n_calls_p1, "n_dragon5_rejected": n_rej_p1}
                print(f"[PIPELINE] seed {seed}: Phase 2 from CL={feas['real_CL_proxy']:.2f} d, budget {remaining_h:.2f} h, "
                      f"target stop OFF", flush=True)
                res = run_greedy_phase2(
                    seed=seed, ppf_c=PPF_C, refit_ppf=True, verbose=True, max_dragon5_calls=MAX_DRAGON5_CALLS,
                    max_wall_hours=remaining_h, gate_margin_days=GATE_MARGIN_DAYS, surrogate_ppf_gate=SURROGATE_PPF_GATE,
                    max_rounds=MAX_ROUNDS, target_cl=False, phase0_path=a["phase0"], phase1=phase1,
                    prior_evaluations_path=prior_path)
                summary.update(phase2_started=True, real_best_score=res["real_best_score"],
                               n_dragon5_calls=res["n_dragon5_calls"], n_dragon5_confirmed=res["n_dragon5_confirmed"],
                               n_dragon5_rejected=res["n_dragon5_rejected"], q4c_phase2=res["q4c_phase2"],
                               phase2_hours=res["elapsed_sec"] / 3600.0, stop_reason=res["stop_reason"],
                               elapsed_sec=time.time() - t_run)
    except Exception as e:
        print(f"!!! pipeline worker seed={seed} FAILED: {type(e).__name__}: {e}", flush=True)
        summary.update(error=f"{type(e).__name__}: {e}", elapsed_sec=time.time() - t_run)
        json.dump(summary, open(os.path.join(work_dir, "pipeline_result.json"), "w"), indent=2, default=float)
        json.dump(summary, open(os.path.join(work_dir, "result_summary.json"), "w"), indent=2, default=float)
        raise
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(summary, open(os.path.join(work_dir, "pipeline_result.json"), "w"), indent=2, default=float)
    json.dump(summary, open(os.path.join(work_dir, "result_summary.json"), "w"), indent=2, default=float)
    print(f"=== pipeline worker seed={seed} finished: {summary['stop_reason']}  best CL={summary['real_best_score']} "
          f"total {summary['elapsed_sec'] / 3600:.2f} h ===", flush=True)
    log_file.close()
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", default=["0-127"])
    ap.add_argument("--phase0", default=os.path.join(BASE_DIR, "phase0_calibration_128", "phase0_points.json"))
    ap.add_argument("--phase1", required=True, help="output directory of the earlier many-seed Phase 1 run (seed_<N>/phase1_result.json)")
    ap.add_argument("--wall-hours", type=float, default=24.0,
                    help="per seed, counted from the start of this run: continued Phase 1 + Phase 2 (no new oracle call starts after it)")
    ap.add_argument("--out", default="runs/pipeline_from128")
    ap.add_argument("--allow-existing", action="store_true", help="skip seeds that already have pipeline_result.json")
    ap.add_argument("--fake-oracle", action="store_true", help="self-test without DRAGON5 (not for real runs)")
    a = ap.parse_args()

    seeds = parse_seeds(a.seeds)
    out_root, phase0, phase1 = os.path.abspath(a.out), os.path.abspath(a.phase0), os.path.abspath(a.phase1)
    if not a.fake_oracle and (not os.environ.get("DRAGON5_EXE") or not os.environ.get("DRAGON5_LIBRARY")):
        print("DRAGON5_EXE / DRAGON5_LIBRARY are not set -- run `python reproduce.py check`."); return 1
    for p in (phase0, phase1):
        if not os.path.exists(p):
            print(f"missing input: {p}"); return 1
    missing = [s for s in seeds if not os.path.exists(os.path.join(phase1, f"seed_{s}", "phase1_result.json"))]
    if missing:
        print(f"no phase1_result.json for seeds {missing[:10]}{' ...' if len(missing) > 10 else ''} in {phase1}"); return 1
    stops = {}
    for s in seeds:
        r = json.load(open(os.path.join(phase1, f"seed_{s}", "phase1_result.json")))
        stops[r.get("stop_reason", "ERROR")] = stops.get(r.get("stop_reason", "ERROR"), 0) + 1
    if "ERROR" in stops:
        print(f"some earlier Phase-1 results are errors: {stops}"); return 1
    if os.path.exists(out_root) and os.listdir(out_root) and not a.allow_existing:
        print(f"{out_root} exists and is not empty -- choose another --out, or pass --allow-existing to continue it."); return 1
    os.makedirs(out_root, exist_ok=True)
    todo = [s for s in seeds if not os.path.exists(os.path.join(out_root, f"seed_{s}", "pipeline_result.json"))]
    n_cpu = os.cpu_count() or 1
    print(f"Pipeline (PPF<{PPF_C}): {len(todo)} seeds to run, earlier Phase 1 ended: {stops}; budget {a.wall_hours} h per seed; "
          f"{n_cpu} logical CPUs; output {out_root}; started {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    if len(todo) > n_cpu:
        print(f"warning: {len(todo)} workers on {n_cpu} CPUs -- seeds share cores and every oracle call is slower.", flush=True)
    args = {"out": out_root, "phase0": phase0, "phase1": phase1, "wall_hours": a.wall_hours, "fake": a.fake_oracle}
    ctx = multiprocessing.get_context("spawn")
    procs = []
    for s in todo:
        p = ctx.Process(target=worker, args=(s, args), name=f"pipeline-seed{s}")
        p.start(); procs.append(p)
    print(f"  started {len(procs)} workers", flush=True)
    for p in procs:
        p.join()
    print(f"\n=== FINISHED @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print(f"{'seed':>5} {'route':>20} {'start CL':>9} {'best CL':>9} {'p2 calls':>9} {'p2 start h':>10} {'total h':>8}  stop", flush=True)
    for s in seeds:
        try:
            r = json.load(open(os.path.join(out_root, f"seed_{s}", "pipeline_result.json")))
            f = lambda v, fmt: (fmt % v) if v is not None else "-"
            print(f"{s:>5} {r.get('route', '?'):>20} {f(r.get('start_CL'), '%.2f'):>9} {f(r.get('real_best_score'), '%.2f'):>9} "
                  f"{r.get('n_dragon5_calls', 0):>9} {f(r.get('phase2_start_h'), '%.2f'):>10} {r['elapsed_sec'] / 3600:>8.2f}  "
                  f"{r.get('stop_reason', r.get('error'))}", flush=True)
        except Exception as e:
            print(f"{s:>5}  (no result: {e})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
