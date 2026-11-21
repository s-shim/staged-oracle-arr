#!/usr/bin/env python3
"""reproduce.py -- one entry point for the full method (staged oracle with early abort + surrogate gate).

The "full method" is the best-performing configuration of the paper: every oracle call is evaluated in stages and aborted
as soon as a stage shows the pattern is infeasible (BOL PPF, then peak k-inf), and a surrogate gate removes candidates
that cannot beat the incumbent before they reach the oracle (margin 3 days; the PPF surrogate gate is off).

  python reproduce.py check                  # environment, inputs, DRAGON5 -- run this first (seconds)
  python reproduce.py tables                 # regenerate the paper tables and figure from the shipped logs (no DRAGON5 needed)
  python reproduce.py oracle-check           # one full oracle run of the Phase-2 start pattern vs. the stored values (~1-1.5 h, 1 core)
  python reproduce.py phase2                 # the main experiment: Phase 2, PPF<1.4, seeds 11-14, 24 h, 4 cores
  python reproduce.py phase2 --seeds 11 --hours 6        # a shorter / single-core variant
  python reproduce.py summary runs/phase2_full_method    # table of a finished (or running) run

Phase 2 starts from the shipped Phase-1 result (phase1_sym_s8_feasible.json, CL = 1462.7 d) and the shipped cache of 177
already-evaluated patterns (prior_evaluations_sym_ppf14.json), exactly as in the paper, so Phase 0/1 need not be repeated.
DRAGON5 and DLIB are NOT included: set DRAGON5_EXE and DRAGON5_LIBRARY (see README).
"""
import argparse, glob, json, multiprocessing, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.join(ROOT, "phase0_calibration_sym", "phase0_points.json")
START = os.path.join(ROOT, "phase1_sym_s8_feasible.json")
PRIOR = os.path.join(ROOT, "prior_evaluations_sym_ppf14.json")
P2_START_WALLCLOCK = "2026-11-08 20:54:00"      # only used by the table/figure generators for the published run

# full-method settings (identical to run_greedy_phase2_parallel_ppf14_s8start_sym.py, which produced the published run)
PPF_C = 1.4
GATE_MARGIN_DAYS = 3.0
SURROGATE_PPF_GATE = False
MAX_DRAGON5_CALLS = 200
MAX_ROUNDS = 100000


# ----------------------------------------------------------------------------- check
def cmd_check(a):
    ok = True

    def line(good, msg):
        nonlocal ok
        ok &= good
        print(("  [ OK ] " if good else "  [FAIL] ") + msg)

    print("Python / packages")
    line(sys.version_info >= (3, 9), f"python {sys.version.split()[0]} (>= 3.9 needed)")
    for m in ("numpy", "scipy", "matplotlib"):
        try:
            __import__(m); line(True, m)
        except Exception as e:
            line(False, f"{m}: {e}  ->  pip install -r requirements.txt")
    print("Input files")
    for p in (PHASE0, START, PRIOR):
        line(os.path.exists(p), os.path.relpath(p, ROOT))
    print("DRAGON5")
    exe, lib = os.environ.get("DRAGON5_EXE", ""), os.environ.get("DRAGON5_LIBRARY", "")
    line(bool(exe) and os.path.isfile(exe) and os.access(exe, os.X_OK), f"DRAGON5_EXE = {exe or '(not set)'}")
    line(bool(lib) and os.path.exists(lib), f"DRAGON5_LIBRARY = {lib or '(not set)'}")
    print("Code")
    sys.path.insert(0, ROOT)
    try:
        import run_qubo_hybrid_dragon5  # noqa: F401
        line(True, "run_qubo_hybrid_dragon5 imports")
    except Exception as e:
        line(False, f"import failed: {type(e).__name__}: {e}")
    print("Machine")
    n = os.cpu_count() or 1
    print(f"  logical CPUs: {n}  (the paper used 4 physical cores, one oracle call per core)")
    print("\nOVERALL:", "READY" if ok else "NOT READY -- fix the [FAIL] lines above")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- tables
def cmd_tables(a):
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    S = os.path.join(ROOT, "scripts")
    runs = [
        ["make_tables.py", "--arr", ROOT, "--out", out, "--p2-start", P2_START_WALLCLOCK],
        ["make_table_ppf16.py", "--arr", ROOT, "--out", out],
        ["make_table_ablation.py", "--arr", ROOT, "--out", out],
        ["make_fig_progress.py", "--arr", ROOT, "--out", out, "--p2-start", P2_START_WALLCLOCK],
    ]
    for r in runs:
        print("->", r[0], flush=True)
        p = subprocess.run([sys.executable, "-B", os.path.join(S, r[0])] + r[1:], stdout=subprocess.DEVNULL)
        if p.returncode:
            print(f"   FAILED ({r[0]}, exit {p.returncode})"); return 1
    print(f"\nWrote to {out}:")
    for f in sorted(glob.glob(os.path.join(out, "*"))):
        print("  ", os.path.basename(f))
    return 0


# ----------------------------------------------------------------------------- oracle-check
def cmd_oracle_check(a):
    sys.path.insert(0, ROOT)
    import bwr10x10_dragon5 as d5
    ref = json.load(open(START))
    print(f"Evaluating the Phase-2 start pattern once with real DRAGON5 (no early abort; takes ~1-1.5 h on one core)...", flush=True)
    t0 = time.time()
    cfg = d5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None)
    # JSON stores pattern cells as lists; the deck builder needs hashable cells (tuples)
    pattern = [[tuple(c) for c in row] for row in ref["pattern"]]
    out = d5.casmo4_bwr10x10(pattern, cfg)
    mins = (time.time() - t0) / 60
    rows = [("k_inf_max", out["k_inf_max"], ref["real_k_inf_max"], 2e-4),
            ("PPF (BOL)", out["PPF"], ref["real_PPF"], 5e-3),
            ("CL [days]", out["CL_proxy"], ref["real_CL_proxy"], 1.0)]
    print(f"\nelapsed {mins:.1f} min\n{'quantity':<12}{'this run':>14}{'stored':>14}{'|diff|':>12}   result")
    allok = True
    for name, v, r, tol in rows:
        good = v is not None and abs(v - r) <= tol
        allok &= good
        print(f"{name:<12}{v:>14.6f}{r:>14.6f}{abs(v - r):>12.6f}   {'PASS' if good else 'FAIL (tol %g)' % tol}")
    print("\nOVERALL:", "PASS -- your DRAGON5/DLIB setup reproduces the stored oracle values"
          if allok else "FAIL -- check DRAGON5 version, DLIB (SHEM295) and the deck settings")
    return 0 if allok else 1


# ----------------------------------------------------------------------------- phase2
def _worker(seed, run_root, hours, quiet_target):
    sys.path.insert(0, ROOT)
    work_dir = os.path.join(run_root, f"seed_{seed}")
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    log_file = open(os.path.join(work_dir, f"phase2_seed{seed}.log"), "a", buffering=1)
    sys.stdout = sys.stderr = log_file
    print(f"=== worker seed={seed} PID={os.getpid()} started @ {time.strftime('%F %T')} ===", flush=True)
    from run_qubo_hybrid_dragon5 import run_greedy_phase2
    try:
        res = run_greedy_phase2(
            seed=seed, ppf_c=PPF_C, refit_ppf=True, verbose=True,
            max_dragon5_calls=MAX_DRAGON5_CALLS, max_wall_hours=hours,
            gate_margin_days=GATE_MARGIN_DAYS, surrogate_ppf_gate=SURROGATE_PPF_GATE, max_rounds=MAX_ROUNDS,
            phase0_path=PHASE0, phase1_result_path=START, prior_evaluations_path=PRIOR,
            # early_abort=True and surrogate_gate=True are the defaults = the full method
        )
    except Exception as e:
        print(f"!!! worker seed={seed} FAILED: {type(e).__name__}: {e}", flush=True)
        json.dump({"seed": seed, "error": str(e)}, open(os.path.join(work_dir, "result_summary.json"), "w"), indent=2)
        raise
    summary = {k: res[k] for k in ("real_best_score", "n_dragon5_calls", "n_dragon5_confirmed",
                                    "n_dragon5_rejected", "q4c_phase2", "elapsed_sec", "stop_reason")}
    summary["seed"] = seed
    json.dump(summary, open(os.path.join(work_dir, "result_summary.json"), "w"), indent=2)
    print(f"=== worker seed={seed} finished: CL={res['real_best_score']:.3f} ({res['stop_reason']}) ===", flush=True)
    log_file.close()
    return summary


def cmd_phase2(a):
    run_root = os.path.abspath(a.out)
    if os.path.exists(run_root) and os.listdir(run_root):
        print(f"{run_root} exists and is not empty -- choose another --out or remove it (runs are never mixed)."); return 1
    for p in (START, PRIOR, PHASE0):
        if not os.path.exists(p):
            print(f"missing input: {p}"); return 1
    if not os.environ.get("DRAGON5_EXE") or not os.environ.get("DRAGON5_LIBRARY"):
        print("DRAGON5_EXE / DRAGON5_LIBRARY are not set -- run `python reproduce.py check`."); return 1
    n = os.cpu_count() or 1
    if len(a.seeds) > max(1, n // 2):
        print(f"warning: {len(a.seeds)} workers on {n} logical CPUs; the paper used one oracle call per PHYSICAL core. "
              f"Oracle times (and therefore time-to-target) will be longer if cores are shared.")
    os.makedirs(run_root, exist_ok=True)
    print(f"Full method, PPF<{PPF_C}: seeds {a.seeds}, budget {a.hours} h per seed, output {run_root}")
    print(f"Target CL >= 1475.7 d (a seed stops when it reaches it). Start CL = 1462.7 d. Logs: {run_root}/seed_<N>/phase2_seed<N>.log")
    ctx = multiprocessing.get_context("spawn")
    procs = []
    for s in a.seeds:
        p = ctx.Process(target=_worker, args=(s, run_root, a.hours, None), name=f"phase2-seed{s}")
        p.start(); procs.append(p); print(f"  started seed {s} (PID {p.pid})", flush=True)
    for p in procs:
        p.join()
    print()
    return cmd_summary(argparse.Namespace(run=run_root))


# ----------------------------------------------------------------------------- summary
def cmd_summary(a):
    run = os.path.abspath(a.run)
    print(f"{'seed':>5} {'best CL [d]':>12} {'oracle calls':>13} {'confirmed':>10} {'hours':>7}  stop reason")
    best = None
    for d in sorted(glob.glob(os.path.join(run, "seed_*"))):
        seed = os.path.basename(d).split("_")[1]
        f = os.path.join(d, "result_summary.json")
        if not os.path.exists(f):
            inc = os.path.join(d, "hybrid_patterns", "incumbent_latest.json")
            cl = json.load(open(inc)).get("real_CL_proxy") if os.path.exists(inc) else None
            print(f"{seed:>5} {('%.2f' % cl) if cl else '-':>12} {'(running or unfinished)':>40}"); continue
        s = json.load(open(f))
        if "error" in s:
            print(f"{seed:>5}  ERROR: {s['error']}"); continue
        print(f"{seed:>5} {s['real_best_score']:>12.2f} {s['n_dragon5_calls']:>13} {s['n_dragon5_confirmed']:>10} "
              f"{s['elapsed_sec'] / 3600:>7.1f}  {s['stop_reason']}")
        if best is None or s["real_best_score"] > best[1]:
            best = (seed, s["real_best_score"])
    if best:
        print(f"\nbest: seed {best[0]}, CL = {best[1]:.2f} d   (target 1475.7 d; published main run: 3 of 4 seeds reached it within 24 h)")
        print(f"pattern: {run}/seed_{best[0]}/hybrid_patterns/incumbent_latest.json")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(f=cmd_check)
    p = sub.add_parser("tables"); p.add_argument("--out", default="reproduced_tables"); p.set_defaults(f=cmd_tables)
    sub.add_parser("oracle-check").set_defaults(f=cmd_oracle_check)
    p = sub.add_parser("phase2")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14])
    p.add_argument("--hours", type=float, default=24.0, help="wall-clock budget per seed (no new oracle call starts after it)")
    p.add_argument("--out", default="runs/phase2_full_method")
    p.set_defaults(f=cmd_phase2)
    p = sub.add_parser("summary"); p.add_argument("run"); p.set_defaults(f=cmd_summary)
    a = ap.parse_args()
    sys.exit(a.f(a))


if __name__ == "__main__":
    main()
