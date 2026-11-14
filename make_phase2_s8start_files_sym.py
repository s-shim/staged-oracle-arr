"""
Phase 2 (PPF<1.4) 재시작용 파일 두 개를 만든다 (arr 폴더에서 실행; 기존 파일은 --force 없이는 덮어쓰지 않음):

  phase1_sym_s8_feasible.json          Phase 1 시드 8의 feasible 패턴 = Phase 2 시작점
                                        (phase1_sym_ppf14/round2/seed_8/phase1_result.json 에서)
  prior_evaluations_sym_ppf14.json     Phase 0의 8개 패턴 + Phase 1 전체 오라클 호출(170개)의 중복 제거 목록.
                                        Phase 2는 이것을 캐시에 넣어 같은 패턴을 다시 평가하지 않는다.

입력: phase0_calibration_sym/phase0_points.json, phase1_sym_ppf14_all_calls.json (Phase 1 호출 170개를 합친 파일),
      phase1_sym_ppf14/round2/seed_8/phase1_result.json
기존 phase1_sym_confirmed_feasible*.json 은 읽지도 쓰지도 않는다.
"""
import json, os, sys, time

FORCE = "--force" in sys.argv
KINF_HI, PPF_C = 1.12, 1.4
P0 = "phase0_calibration_sym/phase0_points.json"
CALLS = "phase1_sym_ppf14_all_calls.json"
RES8 = "phase1_sym_ppf14/round2/seed_8/phase1_result.json"
OUT_START = "phase1_sym_s8_feasible.json"
OUT_PRIOR = "prior_evaluations_sym_ppf14.json"


def info_rank(e):                      # more informative evaluation wins when the same pattern was seen twice
    if e["CL_proxy"] is not None and not e["aborted"]:
        return 3
    if e["k_inf_max"] is not None:
        return 2
    return 1


def main():
    for p in (OUT_START, OUT_PRIOR):
        if os.path.exists(p) and not FORCE:
            raise SystemExit(f"{p} exists -- not overwriting (use --force)")
    sys.path.insert(0, os.path.abspath("."))
    from run_qubo_hybrid_dragon5 import _type_idx_from_pattern

    entries, n_in = {}, 0
    def put(e):
        nonlocal n_in
        n_in += 1
        key = tuple(e["type_idx"])
        old = entries.get(key)
        if old is not None:                                   # deterministic oracle: values must agree where both exist
            for f in ("PPF", "k_inf_max", "CL_proxy"):
                if old[f] is not None and e[f] is not None and abs(old[f] - e[f]) > 1e-4 * max(1.0, abs(old[f])):
                    print(f"  WARNING {f} differs for a repeated pattern: {old[f]} vs {e[f]} ({old['source']} / {e['source']})")
            if info_rank(e) <= info_rank(old):
                return
        entries[key] = e

    for pt in json.load(open(P0)):
        put({"type_idx": _type_idx_from_pattern(pt["pattern"]), "E": pt["E"], "n_gad": pt["n_gad"],
             "k_inf_max": pt["real_k_inf_max"], "PPF": pt["real_PPF"], "CL_proxy": pt["real_CL_proxy"],
             "aborted": None, "source": f"phase0[{pt['index']}]", "calibrated_elsewhere": True})
    d = json.load(open(CALLS))
    for c in d["calls"]:
        put({"type_idx": [int(k) for k in c["type_idx"]], "E": c["E"], "n_gad": c["n_gad"], "k_inf_max": c["k_inf_max"],
             "PPF": c["PPF"], "CL_proxy": c["CL_proxy"], "aborted": c["aborted"],
             "source": f"phase1/{c['batch']}/seed{c['seed']}/call{c['call']}"})

    lst = list(entries.values())
    feas = [e for e in lst if not e["aborted"] and e["CL_proxy"] is not None and e["PPF"] < PPF_C and e["k_inf_max"] <= KINF_HI]
    summary = {"patterns_unique": len(lst), "entries_read": n_in,
               "complete_runs": sum(1 for e in lst if not e["aborted"] and e["CL_proxy"] is not None),
               "aborted_bol_ppf": sum(1 for e in lst if e["aborted"] and "BOL PPF" in e["aborted"]),
               "aborted_k": sum(1 for e in lst if e["aborted"] and "k_inf" in e["aborted"]),
               "feasible_complete": [(e["source"], round(e["CL_proxy"], 2)) for e in feas]}
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "ppf_c": PPF_C, "kinf_hi": KINF_HI, "summary": summary,
               "entries": lst}, open(OUT_PRIOR, "w"))
    print("prior evaluations:", json.dumps(summary, indent=1))

    r = json.load(open(RES8))
    f = r["feasible"]
    from bwr10x10_geometry_heuristic import build_grid_heuristic
    import numpy as np
    assert json.loads(json.dumps(build_grid_heuristic(np.asarray(f["type_idx"])))) == json.loads(json.dumps(f["pattern"])), \
        "type_idx does not rebuild the stored pattern"
    assert f["real_PPF"] < PPF_C and f["real_k_inf_max"] <= KINF_HI and f["real_CL_proxy"] is not None
    call = [c for c in d["calls"] if c["batch"] == "round2" and c["seed"] == 8 and c["feasible"]]
    assert len(call) == 1 and call[0]["type_idx"] == f["type_idx"]
    n_calls = r["n_dragon5_calls"]
    start = {"type_idx": [int(k) for k in f["type_idx"]], "pattern": f["pattern"],
             "real_k_inf_max": f["real_k_inf_max"], "real_PPF": f["real_PPF"], "real_CL_proxy": f["real_CL_proxy"],
             "E": f["E"], "q4c": call[0]["round"], "n_dragon5_calls": n_calls, "n_dragon5_rejected": n_calls - 1,
             "log": [], "seed": 8, "ppf_c": PPF_C, "kinf_hi": KINF_HI, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
             "provenance": "Phase 1 (oracle-penalty search), batch round2 seed 8, oracle call #%d at trial %d, %.2f h; "
                           "full SYM-oracle run (BOL PPF, k_inf peak and CL)." % (call[0]["call"], call[0]["round"],
                                                                                  r["elapsed_sec"] / 3600)}
    json.dump(start, open(OUT_START, "w"), indent=1)
    print(f"start file {OUT_START}: E={start['E']:.4f} k={start['real_k_inf_max']:.6f} PPF={start['real_PPF']:.4f} "
          f"CL={start['real_CL_proxy']:.3f}")

if __name__ == "__main__":
    main()
