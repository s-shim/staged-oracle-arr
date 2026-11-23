"""
Collect the results of a many-seed Phase 2 run (python reproduce.py phase2 --seeds ... --out runs/phase2_128).

  python collect_phase2_many.py runs/phase2_128 [--out runs/phase2_128/summary] [--published phase2_sym_ppf14_s8start]

Run times of complete (confirmed) runs are taken from the time between consecutive call files (elapsed_source = timestamps),
those of stopped runs from the oracle itself.
Reads <run>/seed_*/result_summary.json and <run>/seed_*/hybrid_patterns/call_*.json and writes, to <out>:
  seeds.csv            per seed: best CL, oracle calls, confirmed (complete) runs, rejected (stopped) runs, hours, stop reason, reached target
  calls.csv            every oracle call: outcome (confirmed / aborted at BOL PPF / aborted at the k_inf peak / other), E, N_GAD, PPF, k_inf_max, CL, elapsed
  feasible_all.json/csv  every oracle-certified feasible pattern found by any seed (complete oracle run, PPF and k_inf in range), with grid, CL, PPF, k_inf_max
  pareto.json          non-dominated distinct feasible patterns: pareto_3d = (max CL, min PPF, min k_inf_max), pareto_2d = (max CL, min PPF)
  calls_cdf.csv        fraction of seeds that reached the target within k oracle calls (censoring-aware: a seed that stopped at its wall budget
                       after n < k calls is left out of the denominator for k > n), Wilson 95 % interval
  stats.json           success rate (Wilson 95 %), calls / oracle core-hours / hours to the target, best-CL distribution, oracle run times
  phase2_cdf.png, phase2_pareto.png   (if matplotlib is available)
--published DIR adds a per-seed comparison with the published 4-core run for the seeds that both runs share.
Every number comes from the DRAGON5 oracle runs in the directory; nothing is estimated.
"""
import argparse, csv, glob, json, math, os

TARGET_CL = 1475.7


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def quantiles(xs, qs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    xs = sorted(xs)
    if not xs:
        return {}
    out = {}
    for q in qs:
        pos = q * (len(xs) - 1); lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        out[f"q{int(q * 100)}"] = xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def dominates(a, b, keys):
    better = False
    for name, sgn in keys:
        va, vb = sgn * a[name], sgn * b[name]
        if va < vb:
            return False
        if va > vb:
            better = True
    return better


def load_seeds(run):
    seeds, calls, feas = [], [], []
    for sd in sorted(glob.glob(os.path.join(run, "seed_*")), key=lambda p: int(p.rsplit("_", 1)[1])):
        seed = int(sd.rsplit("_", 1)[1])
        rp = os.path.join(sd, "result_summary.json")
        res = json.load(open(rp)) if os.path.exists(rp) else None
        crec = []
        for cp in sorted(glob.glob(os.path.join(sd, "hybrid_patterns", "call_*.json"))):
            c = json.load(open(cp))
            ab = c.get("aborted")
            if c.get("real_feasible") and c.get("real_CL_proxy") is not None:
                outcome = "confirmed"
            elif ab and "BOL PPF" in str(ab):
                outcome = "aborted_ppf"
            elif ab:
                outcome = "aborted_k"
            else:
                outcome = "other"
            r = {"seed": seed, "call": c["n_dragon5_calls"], "outcome": outcome, "E": c.get("E"), "n_gad": c.get("n_gad"),
                 "PPF": c.get("real_PPF"), "k_inf_max": c.get("real_k_inf_max"), "CL_proxy": c.get("real_CL_proxy"),
                 "dragon_elapsed_s": c.get("dragon_elapsed_s"), "saved_at": c.get("saved_at")}
            calls.append(r); crec.append(r)
            if outcome == "confirmed":
                feas.append(dict(r, pattern=c.get("pattern")))
        # a complete (confirmed) run does not record its own run time; calls of one seed are sequential, so the time between two
        # consecutive call files is the run time of the later call (plus a few seconds of search work). The first call has no predecessor.
        import datetime as _dt
        prev = None
        for r in crec:
            t = _dt.datetime.strptime(r["saved_at"], "%Y-%m-%d %H:%M:%S") if r.get("saved_at") else None
            r["elapsed_source"] = "oracle" if r["dragon_elapsed_s"] else None
            if not r["dragon_elapsed_s"] and t is not None and prev is not None:
                r["dragon_elapsed_s"] = (t - prev).total_seconds(); r["elapsed_source"] = "timestamps"
            prev = t if t is not None else prev
        inc_p = os.path.join(sd, "hybrid_patterns", "incumbent_latest.json")
        inc = json.load(open(inc_p)) if os.path.exists(inc_p) else None
        if res is None or "error" in res:
            seeds.append({"seed": seed, "stop_reason": "WorkerError" if res else "NoResult(running or interrupted)",
                          "best_CL": (inc or {}).get("real_CL_proxy"), "n_calls": len(crec), "n_confirmed": None,
                          "n_rejected": None, "hours": None, "reached": False, "oracle_hours": sum((r["dragon_elapsed_s"] or 0) for r in crec) / 3600})
            continue
        seeds.append({"seed": seed, "stop_reason": res["stop_reason"], "best_CL": res["real_best_score"],
                      "n_calls": res["n_dragon5_calls"], "n_confirmed": res["n_dragon5_confirmed"],
                      "n_rejected": res["n_dragon5_rejected"], "hours": res["elapsed_sec"] / 3600.0,
                      "reached": res["stop_reason"] == "TargetCLReached",
                      "oracle_hours": sum((r["dragon_elapsed_s"] or 0) for r in crec) / 3600})
    return seeds, calls, feas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--published", default=None, help="directory of the published 4-core run (e.g. phase2_sym_ppf14_s8start)")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    out = os.path.abspath(a.out) if a.out else os.path.join(run, "summary")
    os.makedirs(out, exist_ok=True)
    seeds, calls, feas = load_seeds(run)
    done = [s for s in seeds if s["stop_reason"] in ("TargetCLReached", "WallBudgetReached", "CallCapReached") or s["hours"] is not None]
    unfinished = [s for s in seeds if s["hours"] is None]

    seen = {}
    for r in feas:
        key = json.dumps(r["pattern"], sort_keys=True)
        r["distinct"] = key not in seen
        seen.setdefault(key, r)
    distinct = [r for r in feas if r["distinct"]]
    k3 = [("CL_proxy", +1), ("PPF", -1), ("k_inf_max", -1)]
    k2 = [("CL_proxy", +1), ("PPF", -1)]
    for r in distinct:
        r["pareto_3d"] = not any(dominates(o, r, k3) for o in distinct if o is not r)
        r["pareto_2d"] = not any(dominates(o, r, k2) for o in distinct if o is not r)
    for r in feas:
        b = seen[json.dumps(r["pattern"], sort_keys=True)]
        r["pareto_3d"], r["pareto_2d"] = b["pareto_3d"], b["pareto_2d"]

    with open(os.path.join(out, "seeds.csv"), "w", newline="") as f:
        w = csv.writer(f); cols = ["seed", "stop_reason", "reached", "best_CL", "n_calls", "n_confirmed", "n_rejected", "hours", "oracle_hours"]
        w.writerow(cols)
        for s in seeds:
            w.writerow([s.get(c) for c in cols])
    ccols = ["seed", "call", "outcome", "E", "n_gad", "PPF", "k_inf_max", "CL_proxy", "dragon_elapsed_s", "elapsed_source", "saved_at"]
    with open(os.path.join(out, "calls.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(ccols)
        for r in calls:
            w.writerow([r[c] for c in ccols])
    with open(os.path.join(out, "feasible_all.json"), "w") as f:
        json.dump(feas, f, indent=1)
    fcols = ["seed", "call", "E", "n_gad", "k_inf_max", "PPF", "CL_proxy", "distinct", "pareto_3d", "pareto_2d"]
    with open(os.path.join(out, "feasible_all.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(fcols)
        for r in feas:
            w.writerow([r[c] for c in fcols])
    par = {"pareto_3d": [r for r in distinct if r["pareto_3d"]], "pareto_2d": [r for r in distinct if r["pareto_2d"]]}
    with open(os.path.join(out, "pareto.json"), "w") as f:
        json.dump(par, f, indent=1)

    finished = [s for s in seeds if s["hours"] is not None]
    reached = [s for s in finished if s["reached"]]
    n_f = len(finished)
    lo, hi = wilson(len(reached), n_f)
    # censoring-aware cumulative incidence over oracle calls
    max_k = max([s["n_calls"] for s in finished if s["n_calls"]] or [0])
    rows = []
    for k in range(1, max_k + 1):
        at_risk = sum(1 for s in finished if s["reached"] or s["n_calls"] >= k)
        hit = sum(1 for s in reached if s["n_calls"] <= k)
        if at_risk:
            l, h = wilson(hit, at_risk)
            rows.append((k, hit, at_risk, hit / at_risk, l, h))
    with open(os.path.join(out, "calls_cdf.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["k_calls", "n_reached_within_k", "n_at_risk", "fraction", "wilson95_lo", "wilson95_hi"]); w.writerows(rows)

    full = [c["dragon_elapsed_s"] / 60 for c in calls if c["outcome"] == "confirmed" and c["dragon_elapsed_s"]]
    ab_p = [c["dragon_elapsed_s"] / 60 for c in calls if c["outcome"] == "aborted_ppf" and c["dragon_elapsed_s"]]
    ab_k = [c["dragon_elapsed_s"] / 60 for c in calls if c["outcome"] == "aborted_k" and c["dragon_elapsed_s"]]
    outcomes = {}
    for c in calls:
        outcomes[c["outcome"]] = outcomes.get(c["outcome"], 0) + 1
    stats = {
        "n_seeds": len(seeds), "n_finished": n_f, "n_unfinished": len(unfinished),
        "target_CL_days": TARGET_CL, "n_reached": len(reached), "success_rate": len(reached) / n_f if n_f else None,
        "success_rate_wilson95": [lo, hi],
        "stop_reasons": {r: sum(1 for s in seeds if s["stop_reason"] == r) for r in sorted({s["stop_reason"] for s in seeds})},
        "calls_to_target": {"n": len(reached), "mean": mean([s["n_calls"] for s in reached]),
                            "min": min([s["n_calls"] for s in reached], default=None),
                            "max": max([s["n_calls"] for s in reached], default=None),
                            "quantiles": quantiles([s["n_calls"] for s in reached])},
        "hours_to_target": quantiles([s["hours"] for s in reached]),
        "oracle_core_hours_to_target_mean": mean([s["oracle_hours"] for s in reached]),
        "calls_all_seeds_mean": mean([s["n_calls"] for s in finished]),
        "best_CL_per_seed_quantiles": quantiles([s["best_CL"] for s in finished if s["best_CL"] is not None]),
        "best_CL_overall": max([s["best_CL"] for s in finished if s["best_CL"] is not None], default=None),
        "n_oracle_calls": len(calls), "call_outcomes": outcomes,
        "oracle_minutes": {"confirmed_full_run": quantiles(full), "aborted_at_BOL_PPF": quantiles(ab_p),
                           "aborted_at_kinf_peak": quantiles(ab_k)},
        "oracle_core_hours_total": sum((c["dragon_elapsed_s"] or 0) for c in calls) / 3600.0,
        "n_feasible_records": len(feas), "n_distinct_feasible_patterns": len(distinct),
        "n_pareto_3d": len(par["pareto_3d"]), "n_pareto_2d": len(par["pareto_2d"]),
    }
    if a.published and os.path.isdir(a.published):
        pub_seeds, _, _ = load_seeds(os.path.abspath(a.published))
        mine = {s["seed"]: s for s in seeds}
        cmp_rows = []
        for p in pub_seeds:
            m = mine.get(p["seed"])
            if m:
                cmp_rows.append({"seed": p["seed"], "published": {k: p[k] for k in ("best_CL", "n_calls", "hours", "stop_reason")},
                                 "this_run": {k: m[k] for k in ("best_CL", "n_calls", "hours", "stop_reason")}})
        stats["comparison_with_published_seeds"] = cmp_rows
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1, default=float)

    print(f"seeds: {len(seeds)}  finished: {n_f}  still running/unfinished: {len(unfinished)}")
    print(f"reached CL >= {TARGET_CL}: {len(reached)}/{n_f} = {100 * len(reached) / max(n_f, 1):.1f} %  (95 % Wilson {100 * lo:.1f}-{100 * hi:.1f} %)")
    print(f"stop reasons: {stats['stop_reasons']}")
    if reached:
        c2t = [s["n_calls"] for s in reached]
        print(f"oracle calls to target (n={len(c2t)}): mean {mean(c2t):.1f}, min {min(c2t)}, quartiles "
              + ", ".join(f"{v:.1f}" for v in list(quantiles(c2t).values())[1:4]) + f", max {max(c2t)}")
        print("hours to target (min,q25,median,q75,max): " + ", ".join(f"{v:.1f}" for v in quantiles([s['hours'] for s in reached]).values()))
        print(f"oracle core-hours to target (mean per reaching seed): {stats['oracle_core_hours_to_target_mean']:.1f}")
    print(f"best CL overall: {stats['best_CL_overall']}")
    print("oracle run minutes, median (n): " + ", ".join(
        f"{k} {v.get('q50', float('nan')):.0f} ({n})" for k, v, n in
        (("full", stats["oracle_minutes"]["confirmed_full_run"], len(full)),
         ("abort@BOL", stats["oracle_minutes"]["aborted_at_BOL_PPF"], len(ab_p)),
         ("abort@kinf", stats["oracle_minutes"]["aborted_at_kinf_peak"], len(ab_k)))))
    print(f"calls: {len(calls)}  outcomes: {outcomes}  oracle time {stats['oracle_core_hours_total']:.0f} core-h")
    print(f"feasible patterns: {len(feas)} records, {len(distinct)} distinct; Pareto 3-D {len(par['pareto_3d'])}, 2-D {len(par['pareto_2d'])}")
    print(f"written to {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if rows:
            fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=140)
            ax.step([r[0] for r in rows], [r[3] for r in rows], where="post", c="#1f6feb")
            ax.fill_between([r[0] for r in rows], [r[4] for r in rows], [r[5] for r in rows], step="post", alpha=0.2, color="#1f6feb")
            ax.set_xlabel("oracle calls per seed"); ax.set_ylabel(f"fraction of seeds with CL >= {TARGET_CL} d")
            ax.set_ylim(0, 1.02); ax.set_title(f"Phase 2: calls to target ({n_f} seeds)")
            fig.tight_layout(); fig.savefig(os.path.join(out, "phase2_cdf.png"))
        if distinct:
            fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=140)
            rest = [r for r in distinct if not r["pareto_2d"]]
            front = sorted([r for r in distinct if r["pareto_2d"]], key=lambda r: r["PPF"])
            ax.scatter([r["PPF"] for r in rest], [r["CL_proxy"] for r in rest], s=18, c="#9aa5b1", label="feasible")
            ax.scatter([r["PPF"] for r in front], [r["CL_proxy"] for r in front], s=34, c="#d1495b", label="Pareto (CL up, PPF down)", zorder=3)
            ax.plot([r["PPF"] for r in front], [r["CL_proxy"] for r in front], c="#d1495b", lw=1)
            ax.axhline(TARGET_CL, c="k", lw=0.6, ls=":")
            ax.set_xlabel("BOL PPF"); ax.set_ylabel("cycle length CL [d]")
            ax.set_title(f"Phase 2 feasible patterns ({len(distinct)} distinct)"); ax.legend(frameon=False)
            fig.tight_layout(); fig.savefig(os.path.join(out, "phase2_pareto.png"))
    except Exception as e:
        print(f"note: figure not written ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
