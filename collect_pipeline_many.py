"""
Collect the results of a many-seed Phase 1 -> Phase 2 pipeline run (python run_pipeline_many_sym.py ... --out runs/pipeline_from128).

  python collect_pipeline_many.py runs/pipeline_from128 [--phase1 runs/phase1_from128] [--out DIR] [--checkpoints 1,2,4,8,12,16,20,24]

Works on a finished run and on a run that is still going (unfinished seeds are marked; times are measured up to the last call file).
Time axis = wall-clock hours since the start of THIS run (the first line of each seed's log); "cumulative" = this time plus the hours
Phase 1 used in the earlier run (prior_phase1_hours).  All numbers come from the oracle call files; nothing is estimated except the
run times of complete oracle runs, which are not stored and are taken from the time between consecutive call files.

Writes, to <out> (default <run>/summary):
  seeds.csv              per seed: route, earlier Phase-1 result, continued-Phase-1 hours/calls, hours until a feasible pattern exists,
                         start CL, best CL, gain, Phase-2 calls (confirmed / rejected), hours, whether and when 1475.7 d was reached
  trajectories.csv       every improvement of the best feasible CL: seed, hours since run start, CL, source (start / phase1 / phase2)
  checkpoints.csv        for each checkpoint hour: seeds with a feasible pattern, quartiles of best CL, seeds with CL >= target
  target_cdf.csv         fraction of all seeds with CL >= target within t hours (Wilson 95 %), overall and by route
  calls.csv              every oracle call (continued Phase 1 and Phase 2): phase, outcome, E, N_GAD, PPF, k_inf_max, CL, elapsed, time
  feasible_all.json/csv  every feasible pattern (starts and Phase-2 confirmed runs) with grid and values; distinct / Pareto flags
  pareto.json            non-dominated distinct feasible patterns: pareto_3d = (max CL, min PPF, min k_inf_max), pareto_2d = (max CL, min PPF)
  stats.json             all headline numbers
  pipeline_cl_time.png, pipeline_target_cdf.png, pipeline_pareto.png   (if matplotlib is available)
"""
import argparse, csv, datetime as dt, glob, json, math, os, re

TARGET_CL = 1475.7
FEASIBLE_STOPS = ("FeasibleFound", "StartFeasible")


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
    xs = list(xs)
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


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S") if s else None


def read_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def run_start(sd):
    """wall-clock start of the seed = first line of its log"""
    for p in glob.glob(os.path.join(sd, "pipeline_seed*.log")):
        with open(p) as f:
            m = re.search(r"started @ (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", f.readline())
        if m:
            return ts(m.group(1))
    return None


def classify(c):
    ab = c.get("aborted")
    if c.get("real_feasible") and c.get("real_CL_proxy") is not None:
        return "confirmed"
    if ab and "BOL PPF" in str(ab):
        return "aborted_ppf"
    if ab:
        return "aborted_k"
    return "other"


def fill_elapsed(recs):
    prev = None
    for r in recs:
        t = ts(r.get("saved_at"))
        r["elapsed_source"] = "oracle" if r["dragon_elapsed_s"] else None
        if not r["dragon_elapsed_s"] and t is not None and prev is not None:
            r["dragon_elapsed_s"] = (t - prev).total_seconds(); r["elapsed_source"] = "timestamps"
        prev = t if t is not None else prev


def load_seed(sd, p1root):
    seed = int(sd.rsplit("_", 1)[1])
    t0 = run_start(sd)
    res = read_json(os.path.join(sd, "pipeline_result.json"))
    p1_old = read_json(os.path.join(p1root, f"seed_{seed}", "phase1_result.json")) if p1root else None
    hrs = lambda t: (t - t0).total_seconds() / 3600.0 if (t is not None and t0 is not None) else None
    calls, feas = [], []
    # --- continued Phase 1 (only seeds that had to continue)
    c1 = []
    for cp in sorted(glob.glob(os.path.join(sd, "phase1_calls", "call_*.json"))):
        c = read_json(cp)
        if not c:
            continue
        outcome = "confirmed" if c.get("feasible") else ("aborted_ppf" if c.get("aborted") and "BOL PPF" in str(c["aborted"])
                                                         else ("aborted_k" if c.get("aborted") else "other"))
        r = {"seed": seed, "phase": "phase1_continued", "call": c["call"], "outcome": outcome, "E": c.get("E"), "n_gad": c.get("n_gad"),
             "PPF": c.get("PPF"), "k_inf_max": c.get("k_inf_max"), "CL_proxy": c.get("CL_proxy"),
             "dragon_elapsed_s": c.get("dragon_elapsed_s"), "saved_at": c.get("saved_at"), "t_h": hrs(ts(c.get("saved_at")))}
        c1.append(r)
    fill_elapsed(c1)
    calls += c1
    # --- Phase 2
    c2 = []
    for cp in sorted(glob.glob(os.path.join(sd, "hybrid_patterns", "call_*.json"))):
        c = read_json(cp)
        if not c:
            continue
        r = {"seed": seed, "phase": "phase2", "call": c["n_dragon5_calls"], "outcome": classify(c), "E": c.get("E"), "n_gad": c.get("n_gad"),
             "PPF": c.get("real_PPF"), "k_inf_max": c.get("real_k_inf_max"), "CL_proxy": c.get("real_CL_proxy"),
             "dragon_elapsed_s": c.get("dragon_elapsed_s"), "saved_at": c.get("saved_at"), "t_h": hrs(ts(c.get("saved_at")))}
        c2.append(r)
        if r["outcome"] == "confirmed":
            feas.append(dict(r, source="phase2", pattern=c.get("pattern")))
    fill_elapsed(c2)
    calls += c2
    # --- start of Phase 2 and the trajectory of the best feasible CL
    route = (res or {}).get("route")
    start_CL = (res or {}).get("start_CL")
    traj = []
    t_feasible = None
    if res and start_CL is not None:
        if route == "phase2_direct":
            t_feasible = 0.0
        else:
            ok = [r for r in c1 if r["outcome"] == "confirmed" and r["t_h"] is not None]
            t_feasible = ok[-1]["t_h"] if ok else (res.get("phase2_start_h"))
        traj.append({"seed": seed, "t_h": t_feasible, "CL": start_CL, "source": "start" if route == "phase2_direct" else "phase1"})
        # start pattern as a feasible record (from the Phase-1 result that produced it)
        src = read_json(os.path.join(sd, "phase1_result.json")) if route != "phase2_direct" else p1_old
        fp = (src or {}).get("feasible")
        if fp:
            feas.append({"seed": seed, "phase": "phase1", "call": 0, "outcome": "confirmed", "E": fp.get("E"), "n_gad": None,
                         "PPF": fp.get("real_PPF"), "k_inf_max": fp.get("real_k_inf_max"), "CL_proxy": fp.get("real_CL_proxy"),
                         "source": "phase1_start" if route == "phase2_direct" else "phase1_continued", "pattern": fp.get("pattern"),
                         "t_h": t_feasible})
    best = start_CL
    for r in c2:
        if r["outcome"] == "confirmed" and r["CL_proxy"] is not None and (best is None or r["CL_proxy"] > best) and r["t_h"] is not None:
            best = r["CL_proxy"]
            traj.append({"seed": seed, "t_h": r["t_h"], "CL": best, "source": "phase2"})
    t_target = next((x["t_h"] for x in traj if x["CL"] >= TARGET_CL), None)
    prior_h = (res or {}).get("prior_phase1_hours")
    n2 = [r for r in c2]
    seedrec = {
        "seed": seed, "finished": res is not None and "error" not in (res or {}), "route": route,
        "prior_phase1_stop": (res or {}).get("prior_phase1_stop") or (p1_old or {}).get("stop_reason"),
        "prior_phase1_hours": prior_h, "prior_phase1_calls": (res or {}).get("prior_phase1_calls"),
        "phase1b_hours": (res or {}).get("phase1b_hours"), "phase1b_calls": (res or {}).get("phase1b_calls", len(c1) if route else None),
        "t_feasible_h": t_feasible, "start_CL": start_CL, "best_CL": best,
        "gain_CL": (best - start_CL) if (best is not None and start_CL is not None) else None,
        "p2_calls": len(n2), "p2_confirmed": sum(1 for r in n2 if r["outcome"] == "confirmed"),
        "p2_rejected": sum(1 for r in n2 if r["outcome"] != "confirmed"),
        "p2_improvements": sum(1 for x in traj if x["source"] == "phase2"),
        "total_hours": (res or {}).get("elapsed_sec", 0) / 3600.0 if res else None,
        "last_call_h": max([r["t_h"] for r in calls if r["t_h"] is not None], default=None),
        "stop_reason": (res or {}).get("stop_reason") if res else None, "error": (res or {}).get("error"),
        "reached_target": t_target is not None, "t_target_h": t_target,
        "t_target_cum_h": (t_target + prior_h) if (t_target is not None and prior_h is not None) else None,
        "oracle_core_hours": sum((r["dragon_elapsed_s"] or 0) for r in calls) / 3600.0,
    }
    return seedrec, calls, feas, traj


def best_at(traj, t):
    v = [x["CL"] for x in traj if x["t_h"] is not None and x["t_h"] <= t]
    return max(v) if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--phase1", default=None, help="earlier Phase-1 run (default: <run>/../phase1_from128)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoints", default="1,2,4,8,12,16,20,24")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    out = os.path.abspath(a.out) if a.out else os.path.join(run, "summary")
    p1root = os.path.abspath(a.phase1) if a.phase1 else os.path.join(os.path.dirname(run), "phase1_from128")
    if not os.path.isdir(p1root):
        print(f"note: earlier Phase-1 directory {p1root} not found (start patterns of directly started seeds are then missing)")
        p1root = None
    os.makedirs(out, exist_ok=True)
    checkpoints = [float(x) for x in a.checkpoints.split(",")]

    seeds, calls, feas, trajs = [], [], [], []
    for sd in sorted(glob.glob(os.path.join(run, "seed_*")), key=lambda p: int(p.rsplit("_", 1)[1])):
        s, c, f, t = load_seed(sd, p1root)
        seeds.append(s); calls += c; feas += f; trajs += [dict(x) for x in t]
    traj_by = {s["seed"]: [x for x in trajs if x["seed"] == s["seed"]] for s in seeds}
    n = len(seeds)
    fin = [s for s in seeds if s["finished"]]
    budget = max([s for s in [r.get("wall_hours_budget") for r in
                              (read_json(p) for p in glob.glob(os.path.join(run, "seed_*", "pipeline_result.json"))) if r] if s] or [24.0])

    # --- distinct / Pareto
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

    def wcsv(name, cols, rows):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c) for c in cols])

    scols = ["seed", "finished", "route", "prior_phase1_stop", "prior_phase1_hours", "prior_phase1_calls", "phase1b_hours", "phase1b_calls",
             "t_feasible_h", "start_CL", "best_CL", "gain_CL", "p2_calls", "p2_confirmed", "p2_rejected", "p2_improvements", "total_hours",
             "last_call_h", "stop_reason", "reached_target", "t_target_h", "t_target_cum_h", "oracle_core_hours", "error"]
    wcsv("seeds.csv", scols, seeds)
    wcsv("trajectories.csv", ["seed", "t_h", "CL", "source"], trajs)
    wcsv("calls.csv", ["seed", "phase", "call", "outcome", "E", "n_gad", "PPF", "k_inf_max", "CL_proxy", "dragon_elapsed_s", "elapsed_source",
                       "saved_at", "t_h"], calls)
    with open(os.path.join(out, "feasible_all.json"), "w") as f:
        json.dump(feas, f, indent=1)
    wcsv("feasible_all.csv", ["seed", "source", "call", "E", "n_gad", "k_inf_max", "PPF", "CL_proxy", "t_h", "distinct", "pareto_3d", "pareto_2d"], feas)
    par = {"pareto_3d": [r for r in distinct if r["pareto_3d"]], "pareto_2d": [r for r in distinct if r["pareto_2d"]]}
    with open(os.path.join(out, "pareto.json"), "w") as f:
        json.dump(par, f, indent=1)

    # --- checkpoints
    ck_rows = []
    for t in checkpoints:
        vals = [best_at(traj_by[s["seed"]], t) for s in seeds]
        have = [v for v in vals if v is not None]
        ck_rows.append({"hours": t, "n_seeds": n, "n_with_feasible": len(have), "best_CL_min": min(have, default=None),
                        "best_CL_q25": quantiles(have).get("q25"), "best_CL_median": quantiles(have).get("q50"),
                        "best_CL_q75": quantiles(have).get("q75"), "best_CL_max": max(have, default=None),
                        "n_target": sum(1 for v in have if v >= TARGET_CL), "frac_target": sum(1 for v in have if v >= TARGET_CL) / n if n else None})
    wcsv("checkpoints.csv", ["hours", "n_seeds", "n_with_feasible", "best_CL_min", "best_CL_q25", "best_CL_median", "best_CL_q75",
                             "best_CL_max", "n_target", "frac_target"], ck_rows)

    # --- target CDF over wall time, overall and by route (denominator = all seeds of the group; every seed has the same budget)
    groups = {"all": seeds, "phase2_direct": [s for s in seeds if s["route"] == "phase2_direct"],
              "phase1_then_phase2": [s for s in seeds if s["route"] == "phase1_then_phase2"]}
    grid = [i * 0.5 for i in range(0, int(budget * 2) + 1)] if budget >= 1 else [budget * i / 20 for i in range(21)]
    cdf_rows = []
    for g, ss in groups.items():
        for t in grid:
            k = sum(1 for s in ss if s["t_target_h"] is not None and s["t_target_h"] <= t)
            lo, hi = wilson(k, len(ss))
            cdf_rows.append({"group": g, "hours": t, "n_reached": k, "n_seeds": len(ss),
                             "fraction": k / len(ss) if ss else None, "wilson95_lo": lo, "wilson95_hi": hi})
    wcsv("target_cdf.csv", ["group", "hours", "n_reached", "n_seeds", "fraction", "wilson95_lo", "wilson95_hi"], cdf_rows)

    # --- stats
    def grp_stats(ss):
        reach = [s for s in ss if s["reached_target"]]
        fe = [s for s in ss if s["best_CL"] is not None]
        lo, hi = wilson(len(reach), len(ss))
        return {"n_seeds": len(ss), "n_with_feasible": len(fe), "n_reached_target": len(reach),
                "fraction_reached": len(reach) / len(ss) if ss else None, "wilson95": [lo, hi],
                "hours_to_target_this_run": quantiles([s["t_target_h"] for s in reach]),
                "hours_to_target_cumulative": quantiles([s["t_target_cum_h"] for s in reach if s["t_target_cum_h"] is not None]),
                "start_CL": quantiles([s["start_CL"] for s in fe]), "best_CL": quantiles([s["best_CL"] for s in fe]),
                "gain_CL": quantiles([s["gain_CL"] for s in fe if s["gain_CL"] is not None]),
                "p2_calls_mean": mean(s["p2_calls"] for s in ss), "p2_improvements_mean": mean(s["p2_improvements"] for s in ss)}
    full = [c["dragon_elapsed_s"] / 60 for c in calls if c["outcome"] == "confirmed" and c["dragon_elapsed_s"]]
    ab_p = [c["dragon_elapsed_s"] / 60 for c in calls if c["outcome"] == "aborted_ppf" and c["dragon_elapsed_s"]]
    ab_k = [c["dragon_elapsed_s"] / 60 for c in calls if c["outcome"] == "aborted_k" and c["dragon_elapsed_s"]]
    outcomes = {}
    for c in calls:
        k = f"{c['phase']}:{c['outcome']}"; outcomes[k] = outcomes.get(k, 0) + 1
    stats = {
        "n_seeds": n, "n_finished": len(fin), "n_unfinished": n - len(fin), "target_CL_days": TARGET_CL, "budget_hours": budget,
        "routes": {g: sum(1 for s in seeds if s["route"] == g) for g in ("phase2_direct", "phase1_then_phase2")},
        "no_feasible_at_end": [s["seed"] for s in seeds if s["best_CL"] is None],
        "all": grp_stats(seeds), "phase2_direct": grp_stats(groups["phase2_direct"]), "phase1_then_phase2": grp_stats(groups["phase1_then_phase2"]),
        "phase1_continued": {"n_seeds": len(groups["phase1_then_phase2"]),
                             "n_found_feasible": sum(1 for s in groups["phase1_then_phase2"] if s["t_feasible_h"] is not None),
                             "hours_to_feasible": quantiles([s["t_feasible_h"] for s in groups["phase1_then_phase2"] if s["t_feasible_h"] is not None])},
        "best_CL_overall": max([s["best_CL"] for s in seeds if s["best_CL"] is not None], default=None),
        "checkpoints": ck_rows, "n_oracle_calls": len(calls), "call_outcomes": outcomes,
        "oracle_minutes": {"confirmed_full_run": quantiles(full), "aborted_at_BOL_PPF": quantiles(ab_p), "aborted_at_kinf_peak": quantiles(ab_k)},
        "oracle_core_hours_total": sum((c["dragon_elapsed_s"] or 0) for c in calls) / 3600.0,
        "n_feasible_records": len(feas), "n_distinct_feasible_patterns": len(distinct),
        "n_pareto_3d": len(par["pareto_3d"]), "n_pareto_2d": len(par["pareto_2d"]),
        "stop_reasons": {str(r): sum(1 for s in seeds if s["stop_reason"] == r) for r in {s["stop_reason"] for s in seeds}},
    }
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1, default=float)

    # --- console summary
    fm = lambda v, f="%.1f": (f % v) if v is not None else "-"
    print(f"seeds: {n}  finished: {len(fin)}  unfinished: {n - len(fin)}   routes: {stats['routes']}")
    for g in ("all", "phase2_direct", "phase1_then_phase2"):
        s = stats[g]
        print(f"[{g}] seeds {s['n_seeds']}, with a feasible pattern {s['n_with_feasible']}, reached CL >= {TARGET_CL}: {s['n_reached_target']} "
              f"= {100 * (s['fraction_reached'] or 0):.1f} % (Wilson {100 * s['wilson95'][0]:.1f}-{100 * s['wilson95'][1]:.1f} %)")
        if s["best_CL"]:
            print(f"      best CL quartiles: " + ", ".join(fm(v, "%.1f") for v in s["best_CL"].values())
                  + (f";  hours to target (this run) q25/median/q75: " + "/".join(fm(s['hours_to_target_this_run'].get(k)) for k in ('q25', 'q50', 'q75'))
                     if s["hours_to_target_this_run"] else ""))
    pc = stats["phase1_continued"]
    print(f"continued Phase 1: {pc['n_found_feasible']}/{pc['n_seeds']} found a feasible pattern; hours (min,q25,median,q75,max): "
          + ", ".join(fm(v) for v in pc["hours_to_feasible"].values()))
    print("checkpoints (h: seeds with feasible / median best CL / reached target):")
    for r in ck_rows:
        print(f"  {r['hours']:>5g} h: {r['n_with_feasible']:>4} / {fm(r['best_CL_median'], '%.1f'):>7} / {r['n_target']}")
    print(f"best CL overall: {stats['best_CL_overall']}")
    print(f"oracle run minutes, median: full {fm(stats['oracle_minutes']['confirmed_full_run'].get('q50'), '%.0f')} ({len(full)}), "
          f"abort@BOL {fm(stats['oracle_minutes']['aborted_at_BOL_PPF'].get('q50'), '%.0f')} ({len(ab_p)}), "
          f"abort@kinf {fm(stats['oracle_minutes']['aborted_at_kinf_peak'].get('q50'), '%.0f')} ({len(ab_k)})")
    print(f"calls: {len(calls)} {outcomes}; oracle time {stats['oracle_core_hours_total']:.0f} core-h")
    print(f"feasible patterns: {len(feas)} records, {len(distinct)} distinct; Pareto 3-D {len(par['pareto_3d'])}, 2-D {len(par['pareto_2d'])}")
    print(f"written to {out}")

    # --- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        BLUE, ORANGE, GREY, RED = "#1f6feb", "#d9730d", "#9aa5b1", "#d1495b"
        tmax = max(budget, max([r["t_h"] for r in calls if r["t_h"] is not None] or [budget]))
        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=140)
        for s in seeds:
            tr = sorted([x for x in traj_by[s["seed"]] if x["t_h"] is not None], key=lambda x: x["t_h"])
            if not tr:
                continue
            xs = [x["t_h"] for x in tr] + [tmax]
            ys = [x["CL"] for x in tr] + [tr[-1]["CL"]]
            ax.step(xs, ys, where="post", lw=0.7, alpha=0.55, c=BLUE if s["route"] == "phase2_direct" else ORANGE)
        med = [(t, quantiles([v for v in (best_at(traj_by[s["seed"]], t) for s in seeds) if v is not None]).get("q50")) for t in grid]
        med = [(t, v) for t, v in med if v is not None]
        if med:
            ax.plot([m[0] for m in med], [m[1] for m in med], c="k", lw=2.0, label="median over seeds with a feasible pattern")
        ax.axhline(TARGET_CL, c="k", lw=0.8, ls=":")
        ax.text(0.995, TARGET_CL - 0.6, f"reference level {TARGET_CL} d", transform=ax.get_yaxis_transform(), ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=5)
        ax.plot([], [], c=BLUE, label="Phase 1 already feasible"); ax.plot([], [], c=ORANGE, label="Phase 1 continued")
        ax.set_xlabel("hours since start of this run"); ax.set_ylabel("best feasible cycle length [d]")
        ax.set_title(f"Phase 2 from every Phase-1 result ({n} seeds)"); ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(out, "pipeline_cl_time.png")); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
        for g, col, lab in (("all", "k", "all seeds"), ("phase2_direct", BLUE, "Phase 1 already feasible"),
                            ("phase1_then_phase2", ORANGE, "Phase 1 continued")):
            rr = [r for r in cdf_rows if r["group"] == g]
            if not rr or rr[0]["n_seeds"] == 0:
                continue
            ax.step([r["hours"] for r in rr], [r["fraction"] for r in rr], where="post", c=col, lw=2 if g == "all" else 1.3, label=f"{lab} (n={rr[0]['n_seeds']})")
            if g == "all":
                ax.fill_between([r["hours"] for r in rr], [r["wilson95_lo"] for r in rr], [r["wilson95_hi"] for r in rr], step="post", alpha=0.15, color="k")
        ax.set_xlabel("hours since start of this run"); ax.set_ylabel(f"fraction of seeds with CL >= {TARGET_CL} d")
        ax.set_ylim(0, 1.02); ax.set_title("Time to the reference cycle length"); ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(os.path.join(out, "pipeline_target_cdf.png")); plt.close(fig)

        if distinct:
            fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=140)
            rest = [r for r in distinct if not r["pareto_2d"]]
            front = sorted([r for r in distinct if r["pareto_2d"]], key=lambda r: r["PPF"])
            ax.scatter([r["PPF"] for r in rest], [r["CL_proxy"] for r in rest], s=14, c=GREY, label="feasible")
            ax.scatter([r["PPF"] for r in front], [r["CL_proxy"] for r in front], s=34, c=RED, label="Pareto (CL up, PPF down)", zorder=3)
            ax.plot([r["PPF"] for r in front], [r["CL_proxy"] for r in front], c=RED, lw=1)
            ax.axhline(TARGET_CL, c="k", lw=0.6, ls=":")
            ax.set_xlabel("BOL PPF"); ax.set_ylabel("cycle length CL [d]")
            ax.set_title(f"Feasible patterns ({len(distinct)} distinct)"); ax.legend(frameon=False)
            fig.tight_layout(); fig.savefig(os.path.join(out, "pipeline_pareto.png")); plt.close(fig)
    except Exception as e:
        print(f"note: figure not written ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
