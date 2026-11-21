"""
Collect the results of a many-seed Phase 1 run (see run_phase1_many_sym.py).

  python collect_phase1_feasible.py runs/phase1_128 [--out runs/phase1_128/summary]

Reads <run>/seed_*/phase1_result.json and <run>/seed_*/phase1_calls/call_*.json and writes, to <out>:
  feasible_all.json   every oracle-certified feasible pattern (one record per feasible oracle call, all seeds), with the
                      pattern grid, k_inf_max, PPF, cycle length CL, E, N_GAD, seed, call number, and Pareto flags
  feasible_all.csv    the same without the grids
  seeds.csv           per seed: stop reason, number of oracle calls, aborted calls, oracle core-hours
  calls_cdf.csv       fraction of seeds with a feasible pattern within k oracle calls (censoring-aware, Wilson 95 % interval)
  all_calls.csv       every oracle call (feasible, aborted at BOL PPF, aborted at the k_inf peak, failed)
  pareto.json         non-dominated feasible patterns: pareto_3d = (max CL, min PPF, min k_inf_max),
                      pareto_2d = (max CL, min PPF)
  stats.json          success rate with a 95 % Wilson interval, time to the first feasible pattern, call and abort counts
  phase1_pareto.png   CL against PPF of the feasible patterns, Pareto front marked (if matplotlib is available)
Statistical-only note: every number comes from the DRAGON5 oracle runs of this directory; nothing is estimated.
"""
import argparse, csv, glob, json, math, os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def quantiles(xs, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    xs = sorted(xs)
    if not xs:
        return {}
    out = {}
    for q in qs:
        pos = q * (len(xs) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        out[f"q{int(q * 100)}"] = xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
    return out


def dominates(a, b, keys):
    """a dominates b: no worse in every objective and strictly better in one. keys = [(name, +1 max / -1 min)]."""
    better = False
    for name, sgn in keys:
        va, vb = sgn * a[name], sgn * b[name]
        if va < vb:
            return False
        if va > vb:
            better = True
    return better


def pareto_flags(recs, keys):
    flags = []
    for i, r in enumerate(recs):
        flags.append(not any(dominates(o, r, keys) for j, o in enumerate(recs) if j != i))
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the --out directory of run_phase1_many_sym.py")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    out = os.path.abspath(a.out) if a.out else os.path.join(run, "summary")
    os.makedirs(out, exist_ok=True)
    try:
        sys.path.insert(0, BASE_DIR)
        from run_qubo_hybrid_dragon5 import build_grid_heuristic
        import numpy as np
    except Exception as e:                                   # patterns are then taken from phase1_result.json only
        build_grid_heuristic = None
        print(f"note: cannot rebuild pattern grids ({type(e).__name__}: {e})")

    seed_dirs = sorted(glob.glob(os.path.join(run, "seed_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    seeds, calls, feas = [], [], []
    for sd in seed_dirs:
        seed = int(sd.rsplit("_", 1)[1])
        res_path = os.path.join(sd, "phase1_result.json")
        res = json.load(open(res_path)) if os.path.exists(res_path) else None
        if res is not None and "error" in res and "stop_reason" not in res:
            seeds.append({"seed": seed, "stop_reason": "WorkerError", "error": res["error"], "elapsed_h": None, "n_calls": 0})
            continue
        seeds.append({"seed": seed, "stop_reason": (res or {}).get("stop_reason", "NoResult(interrupted?)"),
                      "elapsed_h": (res or {}).get("elapsed_sec", float("nan")) / 3600.0 if res else None,
                      "n_calls": (res or {}).get("n_dragon5_calls"),
                      "n_aborted_ppf": (res or {}).get("n_aborted_ppf"), "n_aborted_k": (res or {}).get("n_aborted_k"),
                      "oracle_hours": (res or {}).get("oracle_sec_sum", float("nan")) / 3600.0 if res else None})
        res_pattern = ((res or {}).get("feasible") or {}).get("pattern")
        for cp in sorted(glob.glob(os.path.join(sd, "phase1_calls", "call_*.json"))):
            c = json.load(open(cp))
            if c.get("feasible"):
                outcome = "feasible"
            elif c.get("aborted"):
                outcome = "aborted_ppf" if "BOL PPF" in str(c["aborted"]) else "aborted_k"
            else:
                outcome = "other"
            calls.append({"seed": seed, "call": c["call"], "outcome": outcome, "E": c["E"], "n_gad": c["n_gad"],
                          "PPF": c["PPF"], "k_inf_max": c["k_inf_max"], "CL_proxy": c["CL_proxy"], "P": c["P"],
                          "dragon_elapsed_s": c["dragon_elapsed_s"], "saved_at": c.get("saved_at")})
            if c.get("feasible"):
                pat = res_pattern if (res_pattern is not None and (res or {}).get("feasible", {}).get("type_idx") == c["type_idx"]) else None
                if pat is None and build_grid_heuristic is not None:
                    pat = json.loads(json.dumps(build_grid_heuristic(np.asarray(c["type_idx"]))))
                feas.append({"seed": seed, "call": c["call"], "E": c["E"], "n_gad": c["n_gad"],
                             "k_inf_max": c["k_inf_max"], "PPF": c["PPF"], "CL_proxy": c["CL_proxy"],
                             "dragon_elapsed_s": c["dragon_elapsed_s"],
                             "seed_hours_to_feasible": (res or {}).get("elapsed_sec", float("nan")) / 3600.0 if res else None,
                             "type_idx": c["type_idx"], "pattern": pat})

    # distinct patterns and Pareto flags
    seen = {}
    for r in feas:
        key = tuple(r["type_idx"])
        r["distinct"] = key not in seen
        seen.setdefault(key, r)
    distinct = [r for r in feas if r["distinct"]]
    k3 = [("CL_proxy", +1), ("PPF", -1), ("k_inf_max", -1)]
    k2 = [("CL_proxy", +1), ("PPF", -1)]
    f3 = dict(zip(map(id, distinct), pareto_flags(distinct, k3)))
    f2 = dict(zip(map(id, distinct), pareto_flags(distinct, k2)))
    for r in feas:
        base = seen[tuple(r["type_idx"])]
        r["pareto_3d"] = f3[id(base)]
        r["pareto_2d"] = f2[id(base)]

    with open(os.path.join(out, "feasible_all.json"), "w") as f:
        json.dump(feas, f, indent=1)
    cols = ["seed", "call", "E", "n_gad", "k_inf_max", "PPF", "CL_proxy", "seed_hours_to_feasible", "dragon_elapsed_s",
            "distinct", "pareto_3d", "pareto_2d"]
    with open(os.path.join(out, "feasible_all.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols + ["type_idx"])
        for r in feas:
            w.writerow([r[c] for c in cols] + [" ".join(map(str, r["type_idx"]))])
    ccols = ["seed", "call", "outcome", "E", "n_gad", "PPF", "k_inf_max", "CL_proxy", "P", "dragon_elapsed_s", "saved_at"]
    with open(os.path.join(out, "all_calls.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(ccols)
        for r in calls:
            w.writerow([r[c] for c in ccols])
    par = {"pareto_3d": [r for r in feas if r["pareto_3d"] and r["distinct"]],
           "pareto_2d": [r for r in feas if r["pareto_2d"] and r["distinct"]]}
    with open(os.path.join(out, "pareto.json"), "w") as f:
        json.dump(par, f, indent=1)

    # ---- oracle-call cost: calls (and oracle core-hours) per seed until its first feasible pattern ----
    with open(os.path.join(out, "seeds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "stop_reason", "n_oracle_calls", "n_aborted_ppf", "n_aborted_k", "oracle_core_hours", "elapsed_hours"])
        for s_ in seeds:
            w.writerow([s_["seed"], s_["stop_reason"], s_.get("n_calls"), s_.get("n_aborted_ppf"), s_.get("n_aborted_k"),
                        s_.get("oracle_hours"), s_.get("elapsed_h")])
    found_s = [s_ for s_ in seeds if s_["stop_reason"] == "FeasibleFound" and s_.get("n_calls") is not None]
    calls_found = [s_["n_calls"] for s_ in found_s]
    oh_found = [s_["oracle_hours"] for s_ in found_s if s_.get("oracle_hours") is not None]
    # cumulative incidence: P(first feasible within k oracle calls). A seed that stopped at its wall budget after n < k calls
    # is censored for k > n and is left out of the denominator for that k (so no bias from the time limit).
    max_k = max([s_["n_calls"] for s_ in seeds if s_.get("n_calls")] or [0])
    cdf_rows = []
    for k in range(1, max_k + 1):
        at_risk = sum(1 for s_ in seeds if s_.get("n_calls") is not None and
                      (s_["stop_reason"] == "FeasibleFound" or s_["n_calls"] >= k))
        hit = sum(1 for s_ in found_s if s_["n_calls"] <= k)
        if at_risk:
            lo_k, hi_k = wilson(hit, at_risk)
            cdf_rows.append((k, hit, at_risk, hit / at_risk, lo_k, hi_k))
    with open(os.path.join(out, "calls_cdf.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["k_calls", "n_found_within_k", "n_at_risk", "fraction", "wilson95_lo", "wilson95_hi"])
        w.writerows(cdf_rows)

    n_seeds = len(seeds)
    n_found = sum(1 for s in seeds if s["stop_reason"] == "FeasibleFound")
    lo, hi = wilson(n_found, n_seeds)
    th = [s["elapsed_h"] for s in seeds if s["stop_reason"] == "FeasibleFound" and s["elapsed_h"] is not None]
    reasons = {}
    for s in seeds:
        reasons[s["stop_reason"]] = reasons.get(s["stop_reason"], 0) + 1
    outcomes = {}
    for c in calls:
        outcomes[c["outcome"]] = outcomes.get(c["outcome"], 0) + 1
    stats = {"n_seeds": n_seeds, "stop_reasons": reasons, "n_seeds_feasible": n_found,
             "success_rate": n_found / n_seeds if n_seeds else None, "success_rate_wilson95": [lo, hi],
             "hours_to_first_feasible_quantiles": quantiles(th),
             "oracle_calls_to_first_feasible": {
                 "n": len(calls_found), "mean": (sum(calls_found) / len(calls_found)) if calls_found else None,
                 "min": min(calls_found) if calls_found else None, "max": max(calls_found) if calls_found else None,
                 "quantiles": quantiles(calls_found),
                 "oracle_core_hours_mean": (sum(oh_found) / len(oh_found)) if oh_found else None,
                 "note": "seeds that found a feasible pattern only; see calls_cdf.csv for the censoring-aware curve"},
            "oracle_calls_all_seeds": {"mean": (sum(s_["n_calls"] for s_ in seeds if s_.get("n_calls") is not None) /
                                                max(1, sum(1 for s_ in seeds if s_.get("n_calls") is not None)))},
             "n_oracle_calls": len(calls), "call_outcomes": outcomes,
             "oracle_hours_total": sum((c["dragon_elapsed_s"] or 0.0) for c in calls) / 3600.0,
             "n_feasible_records": len(feas), "n_distinct_feasible_patterns": len(distinct),
             "n_pareto_3d": len(par["pareto_3d"]), "n_pareto_2d": len(par["pareto_2d"]),
             "feasible_ranges": {k: [min(r[k] for r in distinct), max(r[k] for r in distinct)]
                                 for k in ("CL_proxy", "PPF", "k_inf_max")} if distinct else {}}
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    print(f"seeds: {n_seeds}   feasible found: {n_found} ({100 * n_found / max(n_seeds, 1):.1f} %, 95 % Wilson {100 * lo:.1f}-{100 * hi:.1f} %)")
    print(f"stop reasons: {reasons}")
    if calls_found:
        print(f"oracle calls to first feasible (seeds that found one, n={len(calls_found)}): mean {sum(calls_found) / len(calls_found):.1f}, "
              f"min {min(calls_found)}, quartiles " + ", ".join(f"{v:.1f}" for v in list(quantiles(calls_found).values())[1:4]) +
              f", max {max(calls_found)}   (calls_cdf.csv has the censoring-aware curve)")
    if th:
        print("hours to first feasible (q0,q25,q50,q75,q100): " + ", ".join(f"{v:.1f}" for v in quantiles(th).values()))
    print(f"oracle calls: {len(calls)}  outcomes: {outcomes}  oracle time {stats['oracle_hours_total']:.1f} core-h")
    print(f"feasible patterns: {len(feas)} records, {len(distinct)} distinct; Pareto 3-D {len(par['pareto_3d'])}, 2-D {len(par['pareto_2d'])}")
    if distinct:
        for k, (lo_, hi_) in stats["feasible_ranges"].items():
            print(f"  {k}: {lo_:.5f} .. {hi_:.5f}")
    print(f"written to {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if distinct:
            fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=140)
            rest = [r for r in distinct if not r["pareto_2d"]]
            front = sorted([r for r in distinct if r["pareto_2d"]], key=lambda r: r["PPF"])
            ax.scatter([r["PPF"] for r in rest], [r["CL_proxy"] for r in rest], s=22, c="#9aa5b1", label="feasible")
            ax.scatter([r["PPF"] for r in front], [r["CL_proxy"] for r in front], s=34, c="#d1495b", label="Pareto (CL up, PPF down)", zorder=3)
            ax.plot([r["PPF"] for r in front], [r["CL_proxy"] for r in front], c="#d1495b", lw=1, zorder=2)
            ax.set_xlabel("BOL PPF"); ax.set_ylabel("cycle length CL [d]")
            ax.set_title(f"Phase 1 feasible patterns ({len(distinct)} distinct, {n_seeds} seeds)")
            ax.legend(frameon=False); fig.tight_layout()
            fig.savefig(os.path.join(out, "phase1_pareto.png"))
    except Exception as e:
        print(f"note: figure not written ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
