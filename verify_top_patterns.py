"""
Independent re-check of the best patterns of a run: the stored oracle values are compared with a fresh COMPLETE oracle run (no early
abort, one core per pattern), and the structural constraints are re-derived from the pattern grid alone.

  python verify_top_patterns.py select runs/pipeline_from128/summary/feasible_all.json --top 5 --low-ppf 1 --out runs/verify_top
  python verify_top_patterns.py run    --out runs/verify_top --workers 6      (needs DRAGON5_EXE / DRAGON5_LIBRARY; ~70-100 min)
  python verify_top_patterns.py report --out runs/verify_top

select  picks the --top distinct patterns with the longest cycle length and the --low-ppf non-dominated (cycle length, PPF) patterns with the
        lowest PPF that still beat --min-cl; writes <out>/pat_XX/pattern.json (+ the stored values) and prints the structural check
        (one pattern per group, E and N_GAD window, no two adjacent gadolinium cells, grid round trip) for each.
run     one DRAGON5 run per pattern, in parallel; writes <out>/pat_XX/verify_result.json (values, differences, PASS/FAIL).
report  prints the table.   Tolerances: k_inf_max 2e-4, PPF 5e-3, CL 1.0 d (the same as `reproduce.py oracle-check`).
A pattern passes when every value is within tolerance AND the fresh run is feasible (PPF < 1.4, k_inf_max <= 1.12).
"""
import argparse, glob, json, multiprocessing, os, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
TOL = {"k_inf_max": 2e-4, "PPF": 5e-3, "CL_proxy": 1.0}


def structural(pattern):
    sys.path.insert(0, BASE)
    import numpy as np
    from run_qubo_hybrid_dragon5 import (_type_idx_from_pattern, build_grid_heuristic, VAR_ALLOWED, PIN_TYPES, MULTIPLICITIES,
                                         ADJACENCY_PAIRS, E_LO, E_HI, NGAD_LO, NGAD_HI)
    import arr_quantum_inspired as qi
    ti = _type_idx_from_pattern(pattern)      # raises if the grid is not representable (round trip verified inside)
    ok, detail = qi.check_feasibility(ti, VAR_ALLOWED, PIN_TYPES, MULTIPLICITIES, ADJACENCY_PAIRS,
                                      check_e_range=False, check_ngad_range=False)
    E, n_gad = detail["E"], detail["N_GAD"]
    window = (E_LO <= E <= E_HI) and (NGAD_LO <= n_gad <= NGAD_HI)
    same = json.loads(json.dumps(build_grid_heuristic(np.asarray(ti)))) == json.loads(json.dumps(pattern))
    return {"one_hot_and_adjacency_ok": bool(ok), "E": E, "N_GAD": n_gad, "window_ok": bool(window), "grid_roundtrip_ok": bool(same),
            "type_idx": [int(k) for k in ti]}


def cmd_select(a):
    feas = json.load(open(a.feasible))
    seen, pats = set(), []
    for r in sorted(feas, key=lambda r: -r["CL_proxy"]):
        key = json.dumps(r["pattern"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key); pats.append(r)
    chosen = pats[:a.top]
    ids = {json.dumps(r["pattern"], sort_keys=True) for r in chosen}
    front = [r for r in pats if r.get("pareto_2d") and r["CL_proxy"] >= a.min_cl and json.dumps(r["pattern"], sort_keys=True) not in ids]
    chosen += sorted(front, key=lambda r: r["PPF"])[:a.low_ppf]
    os.makedirs(a.out, exist_ok=True)
    print(f"{'#':>2} {'seed':>5} {'source':>16} {'CL [d]':>9} {'PPF':>8} {'k_inf_max':>10}  structure")
    for i, r in enumerate(chosen):
        d = os.path.join(a.out, f"pat_{i:02d}")
        os.makedirs(d, exist_ok=True)
        st = structural(r["pattern"])
        ok = st["one_hot_and_adjacency_ok"] and st["window_ok"] and st["grid_roundtrip_ok"]
        json.dump({"index": i, "seed": r["seed"], "source": r.get("source"), "call": r.get("call"),
                   "stored": {"E": r.get("E"), "k_inf_max": r["k_inf_max"], "PPF": r["PPF"], "CL_proxy": r["CL_proxy"]},
                   "structure": st, "pattern": r["pattern"]}, open(os.path.join(d, "pattern.json"), "w"))
        print(f"{i:>2} {r['seed']:>5} {str(r.get('source')):>16} {r['CL_proxy']:>9.2f} {r['PPF']:>8.4f} {r['k_inf_max']:>10.5f}  "
              f"{'OK' if ok else 'FAIL'} (E={st['E']:.4f} %, N_GAD={st['N_GAD']})")
    print(f"\n{len(chosen)} patterns written to {a.out}")
    return 0


def _worker(d, fake):
    sys.path.insert(0, BASE)
    os.chdir(d)
    log = open(os.path.join(d, "verify.log"), "a", buffering=1)
    sys.stdout = sys.stderr = log
    rec = json.load(open(os.path.join(d, "pattern.json")))
    pattern = [[tuple(c) for c in row] for row in rec["pattern"]]
    t0 = time.time()
    try:
        if fake:
            import random
            st = rec["stored"]; rnd = random.Random(rec["index"])
            out = {"k_inf_max": st["k_inf_max"] + rnd.uniform(-1e-4, 1e-4), "PPF": st["PPF"] + rnd.uniform(-1e-3, 1e-3),
                   "CL_proxy": st["CL_proxy"] + rnd.uniform(-0.5, 0.5)}
        else:
            import bwr10x10_dragon5 as d5
            out = d5.casmo4_bwr10x10(pattern, d5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None))
        res = {"minutes": (time.time() - t0) / 60}
        allok = True
        for k, tol in TOL.items():
            v, s = out[k], rec["stored"][k]
            good = v is not None and abs(v - s) <= tol
            allok &= good
            res[k] = {"fresh": v, "stored": s, "diff": None if v is None else v - s, "tol": tol, "ok": bool(good)}
        feas = out["PPF"] is not None and out["k_inf_max"] is not None and out["PPF"] < 1.4 and out["k_inf_max"] <= 1.12
        res["fresh_feasible"] = bool(feas)
        res["ppf_margin_to_1.4"] = None if out["PPF"] is None else 1.4 - out["PPF"]
        res["kinf_margin_to_1.12"] = None if out["k_inf_max"] is None else 1.12 - out["k_inf_max"]
        res["PASS"] = bool(allok and feas)
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {e}", "PASS": False}
    json.dump(res, open(os.path.join(d, "verify_result.json"), "w"), indent=2, default=float)
    print(json.dumps(res, default=float)); log.close()


def cmd_run(a):
    if not a.fake_oracle and (not os.environ.get("DRAGON5_EXE") or not os.environ.get("DRAGON5_LIBRARY")):
        print("DRAGON5_EXE / DRAGON5_LIBRARY are not set."); return 1
    dirs = sorted(glob.glob(os.path.join(os.path.abspath(a.out), "pat_*")))
    todo = [d for d in dirs if not os.path.exists(os.path.join(d, "verify_result.json"))]
    print(f"{len(todo)} of {len(dirs)} patterns to run (workers: {a.workers})", flush=True)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(min(a.workers, max(1, len(todo)))) as pool:
        pool.starmap(_worker, [(d, a.fake_oracle) for d in todo])
    return cmd_report(a)


def cmd_report(a):
    print(f"{'pat':>3} {'seed':>5} {'CL stored':>10} {'CL fresh':>10} {'dCL':>7} {'PPF stored':>11} {'PPF fresh':>10} {'k stored':>9} {'k fresh':>9} {'min':>6}  result")
    n_pass = n = 0
    for d in sorted(glob.glob(os.path.join(os.path.abspath(a.out), "pat_*"))):
        rec = json.load(open(os.path.join(d, "pattern.json")))
        rp = os.path.join(d, "verify_result.json")
        if not os.path.exists(rp):
            print(f"{rec['index']:>3} {rec['seed']:>5}  (not run yet)"); continue
        r = json.load(open(rp)); n += 1
        if "error" in r:
            print(f"{rec['index']:>3} {rec['seed']:>5}  ERROR {r['error']}"); continue
        n_pass += r["PASS"]
        print(f"{rec['index']:>3} {rec['seed']:>5} {r['CL_proxy']['stored']:>10.2f} {r['CL_proxy']['fresh']:>10.2f} {r['CL_proxy']['diff']:>7.2f} "
              f"{r['PPF']['stored']:>11.4f} {r['PPF']['fresh']:>10.4f} {r['k_inf_max']['stored']:>9.5f} {r['k_inf_max']['fresh']:>9.5f} "
              f"{r['minutes']:>6.0f}  {'PASS' if r['PASS'] else 'FAIL'}")
    print(f"\n{n_pass}/{n} patterns reproduced within tolerance (k_inf_max 2e-4, PPF 5e-3, CL 1.0 d) and feasible")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select"); p.add_argument("feasible"); p.add_argument("--top", type=int, default=5)
    p.add_argument("--low-ppf", type=int, default=1); p.add_argument("--min-cl", type=float, default=1475.7)
    p.add_argument("--out", default="runs/verify_top"); p.set_defaults(f=cmd_select)
    p = sub.add_parser("run"); p.add_argument("--out", default="runs/verify_top"); p.add_argument("--workers", type=int, default=6)
    p.add_argument("--fake-oracle", action="store_true"); p.set_defaults(f=cmd_run)
    p = sub.add_parser("report"); p.add_argument("--out", default="runs/verify_top"); p.set_defaults(f=cmd_report)
    a = ap.parse_args()
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
