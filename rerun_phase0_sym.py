"""
rerun_phase0_sym.py -- re-evaluate the 8 Phase-0 calibration patterns with the SYM oracle
(bwr10x10_dragon5.py: channel box + water gap + 40 % void, PPF at BOL, 10-step schedule).

The 7 patterns of phase0_calibration/phase0_points.json (indices 0-6, unchanged) PLUS one extra pattern
(index 7, phase0_extra_pattern8.json: E=4.3592, N_GAD=17, chosen from 200k greedy_round draws as the
candidate with E>=4.355 that has the lowest GAD-cell Jaccard overlap with the other 7 and an under-represented
N_GAD, to cover the high-E end where the search actually operates). The old file is NOT touched.
Output: phase0_calibration_sym/phase0_points.json (same schema; the extra point has legacy=None).
Runs N_WORKERS oracle calls in parallel (default 4 = physical cores; override with env N_WORKERS); resumable.

  nohup python rerun_phase0_sym.py > phase0_sym.log 2>&1 &
"""
import json, os, sys, time
import multiprocessing as mp

SRC = os.path.join("phase0_calibration", "phase0_points.json")
DST_DIR = "phase0_calibration_sym"
DST = os.path.join(DST_DIR, "phase0_points.json")
N_WORKERS = int(os.environ.get("N_WORKERS", "4"))
EXTRA = "phase0_extra_pattern8.json"


def _work(item):
    idx, pattern = item
    sys.path.insert(0, os.path.abspath("."))
    import bwr10x10_dragon5 as d5
    pat = [[tuple(c) for c in row] for row in pattern]
    cfg = d5.Dragon5Config(compute_ppf=True, compute_cl=True, cache_path=None)
    t0 = time.time()
    out = d5.casmo4_bwr10x10(pat, cfg)
    return idx, dict(E=out["E"], n_gad=out["N_GAD"], real_k_inf_max=out["k_inf_max"],
                     real_PPF=out["PPF"], real_CL_proxy=out["CL_proxy"], elapsed_s=time.time() - t0)


def main():
    old = json.load(open(SRC))
    if os.path.exists(EXTRA):
        ex = json.load(open(EXTRA))
        old.append(dict(index=ex["index"], E=ex["E"], n_gad=ex["n_gad"], pattern=ex["pattern"],
                        real_k_inf_max=None, real_PPF=None, real_CL_proxy=None))
    else:
        print(f"WARNING: {EXTRA} not found -> running the original 7 points only", flush=True)
    os.makedirs(DST_DIR, exist_ok=True)
    done = {}
    if os.path.exists(DST):
        done = {p["index"]: p for p in json.load(open(DST))}
    todo = [(p["index"], p["pattern"]) for p in old if p["index"] not in done]
    print(f"{len(done)} done, {len(todo)} to run with {N_WORKERS} workers @ {time.strftime('%F %T')}", flush=True)
    byidx = {p["index"]: p for p in old}
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS) as pool:
        for idx, r in pool.imap_unordered(_work, todo):
            rec = dict(index=idx, E=r["E"], n_gad=r["n_gad"], real_k_inf_max=r["real_k_inf_max"],
                       real_PPF=r["real_PPF"], real_CL_proxy=r["real_CL_proxy"],
                       pattern=byidx[idx]["pattern"],
                       legacy=(None if byidx[idx]["real_PPF"] is None else
                               dict(k_inf_max=byidx[idx]["real_k_inf_max"], PPF=byidx[idx]["real_PPF"],
                                    CL_proxy=byidx[idx]["real_CL_proxy"])))
            done[idx] = rec
            json.dump([done[i] for i in sorted(done)], open(DST, "w"), indent=1)
            lg = rec["legacy"]
            lgs = "(no legacy value: new point)" if lg is None else (f"(legacy kinf={lg['k_inf_max']:.5f} "
                  f"PPF={lg['PPF']:.4f} CL={lg['CL_proxy']:.1f})")
            print(f"=== point {idx} done {r['elapsed_s']/60:.0f} min  E={r['E']:.3f} kinf={r['real_k_inf_max']:.5f} "
                  f"PPF={r['real_PPF']:.4f} CL={r['real_CL_proxy']:.1f}   {lgs}", flush=True)
    print("ALL DONE", time.strftime("%F %T"), flush=True)


if __name__ == "__main__":
    main()
