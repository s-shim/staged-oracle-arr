"""
Phase 0 at scale: N space-filling calibration patterns (default 128), one full oracle run per core.

The published Phase 0 has 8 patterns (phase0_calibration_sym/phase0_points.json); by default they are NOT used: N brand-new patterns are chosen on their own (--with-published keeps the 8 as anchors and in the merged file).
and every one gets a COMPLETE DRAGON5 run (no early abort: the surrogate needs k_inf_max, PPF and CL of each pattern).

Candidates   draws of the same ARR step the search uses (perturb + greedy_round on the 0.5 centroid, one-hot and GAD
             adjacency by construction), kept only if E and N_GAD are inside the search window (E 4.25-4.36 %, N_GAD 16-18)
             and not already among the 8 published patterns.
Selection    greedy farthest-point (maximin) sampling, starting from the 8 published patterns:
                 d(a,b) = (number of the 35 group choices that differ)/35 + |E_a - E_b|/(E_hi - E_lo)
             so the new points spread over the layout (enrichment and gadolinia placement) and over E.  Deterministic
             for a given --seed.  The selection is written to <out>/selected_patterns.json BEFORE any oracle call.
Run          one spawn-process per pattern (--workers, default = number of CPUs), results saved per point as
             <out>/points/point_XXX.json; the merged file <out>/phase0_points.json = the 8 published points
             (indices 0-7, copied unchanged) + the new points (indices 8...), same schema as the published file,
             is rewritten after every finished point.  Resumable: finished points are never repeated.

  python run_phase0_many_sym.py --n 128 --out phase0_calibration_128
  python run_phase0_many_sym.py --n 12 --select-only --out /tmp/p0test      # look at the selection, no oracle
  python run_phase0_many_sym.py --n 12 --fake-oracle --workers 4 --out /tmp/p0fake   # self-test without DRAGON5

The merged file can be given to Phase 1 / Phase 2 (run_phase1_many_sym.py --phase0 phase0_calibration_128/phase0_points.json).
"""
import argparse, json, os, sys, time
import multiprocessing as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_PATH = os.path.join(BASE_DIR, "phase0_calibration_sym", "phase0_points.json")


def _norm(x):
    return json.loads(json.dumps(x))


def select_patterns(n, seed, draws, with_published=False, log=print):
    sys.path.insert(0, BASE_DIR)
    import numpy as np
    import run_qubo_hybrid_dragon5 as R
    qi = R.qi
    old = json.load(open(OLD_PATH)) if with_published else []
    old_ti = [tuple(R._type_idx_from_pattern(p["pattern"])) for p in old]
    old_E = [p["E"] for p in old]

    rng = np.random.default_rng(seed)
    c = qi.init_centroid(R.VAR_ALLOWED, R.N_TYPES)
    seen, cand_ti, cand_E, cand_N = set(old_ti), [], [], []
    t0 = time.time()
    for _ in range(draws):
        ptb = qi.perturb(c, R.VAR_ALLOWED, rng, mode="mult")
        ti = qi.greedy_round(ptb, R.VAR_ALLOWED, R.ADJACENCY_PAIRS, qi.GAD_TYPE_IDS)
        key = tuple(int(k) for k in ti)
        if key in seen:
            continue
        ok, d = qi.check_feasibility(ti, R.VAR_ALLOWED, R.PIN_TYPES, R.MULTIPLICITIES, R.ADJACENCY_PAIRS,
                                     check_e_range=False, check_ngad_range=False)
        if not ok or not (R.E_LO <= d["E"] <= R.E_HI and R.NGAD_LO <= d["N_GAD"] <= R.NGAD_HI):
            continue
        seen.add(key); cand_ti.append(key); cand_E.append(d["E"]); cand_N.append(d["N_GAD"])
    log(f"candidates: {len(cand_ti)} distinct valid patterns from {draws} draws ({time.time() - t0:.0f} s)")
    if len(cand_ti) < n:
        raise SystemExit(f"only {len(cand_ti)} valid candidates for n={n}; raise --draws")

    T = np.asarray(cand_ti, dtype=np.int16)
    E = np.asarray(cand_E)
    erange = float(R.E_HI - R.E_LO)
    ngroups = float(T.shape[1])
    mind = np.full(len(T), np.inf)
    anchors = list(zip(old_ti, old_E))
    for ti, e in anchors:
        d = (T != np.asarray(ti, dtype=np.int16)).sum(axis=1) / ngroups + np.abs(E - e) / erange
        mind = np.minimum(mind, d)
    chosen = []
    for _ in range(n):
        j = int(np.argmax(mind))
        chosen.append(j)
        d = (T != T[j]).sum(axis=1) / ngroups + np.abs(E - E[j]) / erange
        mind = np.minimum(mind, d)
        mind[j] = -1.0
    out = []
    for rank, j in enumerate(chosen):
        pat = _norm(R.build_grid_heuristic(np.asarray(cand_ti[j])))
        out.append(dict(index=len(old) + rank, E=float(cand_E[j]), n_gad=int(cand_N[j]),
                        type_idx=[int(k) for k in cand_ti[j]], pattern=pat))
    ng = {g: sum(1 for o in out if o["n_gad"] == g) for g in sorted(set(cand_N))}
    log(f"selected {n}: E {min(o['E'] for o in out):.4f}..{max(o['E'] for o in out):.4f}, N_GAD counts {ng}")
    return old, out


def _fake(pattern, idx):
    import hashlib, random
    rnd = random.Random(int(hashlib.md5(json.dumps(pattern).encode()).hexdigest()[:8], 16))
    return dict(real_k_inf_max=rnd.uniform(1.10, 1.13), real_PPF=rnd.uniform(1.40, 1.70),
                real_CL_proxy=rnd.uniform(1440, 1495))


def _work(args):
    idx, pattern, E, n_gad, point_dir, fake = args
    sys.path.insert(0, BASE_DIR)
    t0 = time.time()
    try:
        if fake:
            r = _fake(pattern, idx)
        else:
            import bwr10x10_dragon5 as d5
            pat = [[tuple(cell) for cell in row] for row in pattern]
            cfg = d5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None)
            out = d5.casmo4_bwr10x10(pat, cfg)
            r = dict(real_k_inf_max=out["k_inf_max"], real_PPF=out["PPF"], real_CL_proxy=out["CL_proxy"])
        rec = dict(index=idx, E=E, n_gad=n_gad, pattern=pattern, legacy=None, elapsed_s=time.time() - t0, **r)
    except Exception as e:
        rec = dict(index=idx, error=f"{type(e).__name__}: {e}", elapsed_s=time.time() - t0)
    tmp = os.path.join(point_dir, f".point_{idx:03d}.tmp")
    json.dump(rec, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(point_dir, f"point_{idx:03d}.json"))
    return rec


def merge(old, point_dir, dst):
    pts = [dict(p) for p in old]
    for fn in sorted(os.listdir(point_dir)):
        if fn.startswith("point_") and fn.endswith(".json"):
            r = json.load(open(os.path.join(point_dir, fn)))
            if "error" in r:
                continue
            pts.append({k: r[k] for k in ("index", "E", "n_gad", "real_k_inf_max", "real_PPF", "real_CL_proxy",
                                           "pattern", "legacy")})
    tmp = dst + ".tmp"
    json.dump(pts, open(tmp, "w"), indent=1)
    os.replace(tmp, dst)
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=128, help="number of NEW patterns (the 8 published ones are kept)")
    ap.add_argument("--out", default="phase0_calibration_128")
    ap.add_argument("--seed", type=int, default=0, help="seed of the candidate draws (selection is deterministic)")
    ap.add_argument("--draws", type=int, default=400000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--with-published", action="store_true",
                    help="keep the 8 published points as anchors and in the merged file (default: N brand-new points only)")
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--fake-oracle", action="store_true", help="self-test without DRAGON5")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    point_dir = os.path.join(out, "points")
    os.makedirs(point_dir, exist_ok=True)
    sel_path = os.path.join(out, "selected_patterns.json")
    if os.path.exists(sel_path):
        sel = json.load(open(sel_path))
        old = json.load(open(OLD_PATH)) if a.with_published else []
        if len(sel) != a.n:
            print(f"{sel_path} has {len(sel)} patterns but --n {a.n}: remove it or use another --out"); return 1
        print(f"using the existing selection {sel_path} ({len(sel)} patterns)", flush=True)
    else:
        old, sel = select_patterns(a.n, a.seed, a.draws, a.with_published, log=lambda s: print(s, flush=True))
        json.dump(sel, open(sel_path, "w"), indent=1)
        print(f"wrote {sel_path}", flush=True)
    if a.select_only:
        return 0
    if not a.fake_oracle and (not os.environ.get("DRAGON5_EXE") or not os.environ.get("DRAGON5_LIBRARY")):
        print("DRAGON5_EXE / DRAGON5_LIBRARY are not set -- run `python reproduce.py check`."); return 1
    done = {int(f[6:9]) for f in os.listdir(point_dir)
            if f.startswith("point_") and f.endswith(".json") and "error" not in json.load(open(os.path.join(point_dir, f)))}
    todo = [(s["index"], s["pattern"], s["E"], s["n_gad"], point_dir, a.fake_oracle) for s in sel if s["index"] not in done]
    n_cpu = os.cpu_count() or 1
    print(f"Phase 0: {len(done)} done, {len(todo)} to run, {a.workers} workers on {n_cpu} CPUs, "
          f"started {time.strftime('%F %T')}", flush=True)
    if a.workers > n_cpu:
        print("warning: more workers than CPUs -- every oracle call is slower.", flush=True)
    dst = os.path.join(out, "phase0_points.json")
    ctx = mp.get_context("spawn")
    nfin = 0
    with ctx.Pool(min(a.workers, max(1, len(todo)))) as pool:
        for r in pool.imap_unordered(_work, todo):
            nfin += 1
            if "error" in r:
                print(f"!!! point {r['index']} FAILED after {r['elapsed_s']/60:.0f} min: {r['error']}", flush=True)
                continue
            merge(old, point_dir, dst)
            print(f"[{nfin}/{len(todo)}] point {r['index']} {r['elapsed_s']/60:.0f} min  E={r['E']:.3f} N_GAD={r['n_gad']} "
                  f"kinf={r['real_k_inf_max']:.5f} PPF={r['real_PPF']:.4f} CL={r['real_CL_proxy']:.1f}", flush=True)
    pts = merge(old, point_dir, dst)
    nfeas = [p["index"] for p in pts if p["real_PPF"] < 1.4 and p["real_k_inf_max"] <= 1.12]
    print(f"\n=== FINISHED @ {time.strftime('%F %T')}: {len(pts)} points in {dst} ===", flush=True)
    print(f"Phase-0 patterns that are already feasible for PPF<1.4, k_inf<=1.12: {nfeas if nfeas else 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
