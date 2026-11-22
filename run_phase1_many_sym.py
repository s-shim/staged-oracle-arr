"""
Phase 1 at scale (PPF < 1.4): many independent seeds in parallel, one oracle call per core.

Same algorithm and settings as run_phase1_parallel_ppf14_sym.py (penalty minimisation against the staged oracle, every seed
starts from the best Phase-0 pattern, a seed stops at its first oracle-certified feasible pattern or at its wall budget);
only the seed list, the output directory and the number of workers are free.  Nothing in the published Phase 1 results is touched.

  python run_phase1_many_sym.py --seeds 1-128 --wall-hours 12 --out runs/phase1_128

Output, per seed:  <out>/seed_<N>/phase1_ppf14_seed<N>.log, phase1_calls/call_XXXX.json (one per oracle call, with the
pin-power map of every call that reached the BOL step), phase1_result.json (at the end of the seed; a feasible seed also has the
full feasible pattern and its k_inf_max, PPF and cycle length).
Afterwards run  python collect_phase1_feasible.py <out>  to gather all feasible patterns, all oracle calls, the Pareto
front and the statistics into <out>/summary/.

Seeds 1-8 reproduce the published Phase 1 (round 1: seeds 1-4, round 2: seeds 5-8); with the same code, data and DRAGON5/DLIB
they should give the same calls.  A seed whose phase1_result.json already exists is skipped when --allow-existing is given
(use it to restart after an interruption; a seed that was cut off is run again from the start).
"""
import argparse, json, multiprocessing, os, sys, time

PPF_C = 1.4
KINF_HI = 1.12
MAX_DRAGON5_CALLS = 500       # safety cap (the real limit is the wall budget)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE0_PATH = os.path.join(BASE_DIR, "phase0_calibration_sym", "phase0_points.json")


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


def _fake_oracle(pattern, call_no):
    """Self-test only (--fake-oracle): deterministic pseudo-oracle, no DRAGON5 needed."""
    import hashlib, random
    h = int(hashlib.md5(json.dumps(pattern, default=str).encode()).hexdigest()[:8], 16)
    rnd = random.Random(h)
    ppf = rnd.uniform(1.34, 1.62)
    k = rnd.uniform(1.100, 1.135)
    cl = rnd.uniform(1440.0, 1480.0)
    out = {"aborted": None, "k_inf_max": k, "PPF": ppf, "CL_proxy": cl, "dragon_elapsed_s": 1.0,
           "burnup_history": {"k_inf": [k] * 3, "pin_power_map_kmax": [[1.0] * 10 for _ in range(10)]}}
    if ppf >= PPF_C:
        out.update(aborted="BOL PPF %.4f >= %.2f" % (ppf, PPF_C), k_inf_max=None, CL_proxy=None)
    elif k > KINF_HI:
        out.update(aborted="k_inf peak %.4f > %.2f" % (k, KINF_HI), CL_proxy=None)
    return out


def worker(seed, out_root, wall_hours, fake, phase0_path, start_each=False):
    sys.path.insert(0, BASE_DIR)
    work_dir = os.path.join(out_root, f"seed_{seed}")
    os.makedirs(work_dir, exist_ok=True)
    log_file = open(os.path.join(work_dir, f"phase1_ppf14_seed{seed}.log"), "a", buffering=1)
    sys.stdout = log_file; sys.stderr = log_file
    print(f"=== phase1 worker seed={seed} PID={os.getpid()} started @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    from run_qubo_hybrid_dragon5 import run_greedy_phase1_oracle
    kw = {"oracle_fn": _fake_oracle} if fake else {}
    if start_each:
        kw["start_index"] = seed                      # seed N starts from Phase-0 point N and knows only that point
    try:
        rec = run_greedy_phase1_oracle(seed=seed, ppf_c=PPF_C, kinf_hi=KINF_HI, phase0_path=phase0_path,
                                       save_dir=work_dir, max_wall_hours=wall_hours,
                                       max_dragon5_calls=MAX_DRAGON5_CALLS, verbose=True, **kw)
    except Exception as e:
        print(f"!!! worker seed={seed} FAILED: {type(e).__name__}: {e}", flush=True)
        with open(os.path.join(work_dir, "phase1_result.json"), "w") as f:
            json.dump({"seed": seed, "error": f"{type(e).__name__}: {e}"}, f, indent=2)
        raise
    print(f"=== worker seed={seed} finished: {rec['stop_reason']} best P={rec['best_P']:.5f} ===", flush=True)
    log_file.close()
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", default=["1-128"], help="e.g. 1-128  or  1-8 21 30-33")
    ap.add_argument("--wall-hours", type=float, default=12.0,
                    help="per seed: no new oracle call starts after this (a running call is finished)")
    ap.add_argument("--out", default="runs/phase1_128")
    ap.add_argument("--phase0", default=PHASE0_PATH,
                    help="Phase-0 oracle points (default: the 8 published points; for the 128-point Phase 0 use "
                         "phase0_calibration_128/phase0_points.json)")
    ap.add_argument("--start-each", action="store_true",
                    help="seed N starts from Phase-0 point N (index N in --phase0) and knows only that point; use with --seeds 0-127 and the"
                         " 128-point file. A feasible start point ends its seed at once (StartFeasible, 0 calls).")
    ap.add_argument("--allow-existing", action="store_true",
                    help="use an existing output directory; seeds that already have phase1_result.json are skipped")
    ap.add_argument("--fake-oracle", action="store_true", help="self-test without DRAGON5 (not for real runs)")
    a = ap.parse_args()

    seeds = parse_seeds(a.seeds)
    out_root = os.path.abspath(a.out)
    if not a.fake_oracle and (not os.environ.get("DRAGON5_EXE") or not os.environ.get("DRAGON5_LIBRARY")):
        print("DRAGON5_EXE / DRAGON5_LIBRARY are not set -- run `python reproduce.py check`."); return 1
    phase0_path = os.path.abspath(a.phase0)
    if not os.path.exists(phase0_path):
        print(f"missing input: {phase0_path}"); return 1
    sys.path.insert(0, BASE_DIR)
    from run_qubo_hybrid_dragon5 import phase1_penalty
    p0 = json.load(open(phase0_path))
    feas0 = [p["index"] for p in p0 if phase1_penalty(p["E"], p["n_gad"], p["real_k_inf_max"], p["real_PPF"], PPF_C, KINF_HI) <= 0.0]
    if feas0 and not a.start_each:
        print(f"Phase 0 already contains feasible pattern(s) (index {feas0}); Phase 1 starts from a feasible pattern, "
              f"so there is nothing to search. They are the feasible set -- go on to Phase 2 from them."); return 0
    print(f"Phase-0 file: {phase0_path} ({len(p0)} points, feasible: {feas0 if feas0 else 'none'}), "
          f"{'each seed starts from its own Phase-0 point' if a.start_each else 'all seeds start from the best Phase-0 point'}", flush=True)
    if a.start_each:
        have = {p['index'] for p in p0}
        missing = [s for s in seeds if s not in have]
        if missing:
            print(f"--start-each: seeds {missing[:10]} are not Phase-0 point indices in {phase0_path}"); return 1
    if os.path.exists(out_root) and os.listdir(out_root) and not a.allow_existing:
        print(f"{out_root} exists and is not empty -- choose another --out, or pass --allow-existing to continue it."); return 1
    os.makedirs(out_root, exist_ok=True)
    todo = [s for s in seeds if not os.path.exists(os.path.join(out_root, f"seed_{s}", "phase1_result.json"))]
    n_cpu = os.cpu_count() or 1
    print(f"Phase 1 (PPF<{PPF_C}, k_inf<={KINF_HI}): {len(todo)} seeds to run ({len(seeds) - len(todo)} already done), "
          f"wall budget {a.wall_hours} h per seed, {n_cpu} logical CPUs, output {out_root}, "
          f"started {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    if len(todo) > n_cpu:
        print(f"warning: {len(todo)} workers on {n_cpu} CPUs -- seeds share cores and every oracle call is slower.", flush=True)
    ctx = multiprocessing.get_context("spawn")
    procs = []
    for s in todo:
        p = ctx.Process(target=worker, args=(s, out_root, a.wall_hours, a.fake_oracle, phase0_path, a.start_each), name=f"phase1-seed{s}")
        p.start(); procs.append(p)
    print(f"  started {len(procs)} workers", flush=True)
    for p in procs:
        p.join()
    print(f"\n=== FINISHED @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print(f"{'seed':>5} {'stop':>20} {'best_P':>9} {'calls':>6} {'abortPPF':>9} {'abortK':>7} {'hours':>7}", flush=True)
    for s in seeds:
        try:
            r = json.load(open(os.path.join(out_root, f"seed_{s}", "phase1_result.json")))
            print(f"{s:>5} {r['stop_reason']:>20} {r['best_P']:>9.5f} {r['n_dragon5_calls']:>6} "
                  f"{r['n_aborted_ppf']:>9} {r['n_aborted_k']:>7} {r['elapsed_sec']/3600:>7.2f}", flush=True)
        except Exception as e:
            print(f"{s:>5}  (no result: {e})", flush=True)
    print(f"\nNext: python collect_phase1_feasible.py {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
