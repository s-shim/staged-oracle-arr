"""
make_phase1_start_files_sym.py -- writes the two Phase-2 start files (reproducibly):
  phase1_sym_confirmed_feasible.json        PPF<1.4 start = candidate #11 (full SYM-oracle run, full_check_results/11.json)
  phase1_sym_confirmed_feasible_ppf16.json  PPF<1.6 start = Phase-0 point 5 (full SYM-oracle run in phase0_points.json;
                                            type_idx from ring_design3.json #13, whose pattern is checked to be identical)
Existing files are NOT overwritten unless --force is given.   Run in the arr folder:  python make_phase1_start_files_sym.py
(Do not run prepare_phase1_ppf16_from_ppf14_sym.py any more: it would copy the PPF<1.4 start (CL 1458) to the 1.6 problem.)
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
from bwr10x10_geometry_heuristic import build_grid_heuristic

force = "--force" in sys.argv
norm = lambda p: json.loads(json.dumps(p))
ts = time.strftime("%Y-%m-%d %H:%M:%S")

f11 = json.load(open("full_check_results/11.json"))
P0 = json.load(open("phase0_calibration_sym/phase0_points.json"))
rd = json.load(open("ring_design3.json"))[13]
pat5 = build_grid_heuristic(np.array(rd["type_idx"]))
assert norm(pat5) == P0[5]["pattern"], "ring_design3 #13 does not reproduce Phase-0 point 5"
assert abs(P0[5]["E"] - rd["E"]) < 1e-9

rec14 = {"type_idx": f11["type_idx"], "pattern": norm(build_grid_heuristic(np.array(f11["type_idx"]))),
         "real_k_inf_max": f11["k_inf_max"], "real_PPF": f11["PPF"], "real_CL_proxy": f11["CL_proxy"], "E": f11["E"],
         "q4c": 0, "n_dragon5_calls": 1, "n_dragon5_rejected": 0, "log": [], "seed": 0, "ppf_c": 1.4, "kinf_hi": 1.12,
         "timestamp": ts,
         "provenance": "Designed BOL-screen experiments (bol_ring2_results #11) + one full SYM-oracle run "
                       "(full_check_results/11.json; stream_probe re-run: k=1.115291 PPF=1.3943 CL=1458.1238). "
                       "Not produced by run_phase1_only."}
rec16 = {"type_idx": rd["type_idx"], "pattern": norm(pat5), "real_k_inf_max": P0[5]["real_k_inf_max"],
         "real_PPF": P0[5]["real_PPF"], "real_CL_proxy": P0[5]["real_CL_proxy"], "E": P0[5]["E"],
         "q4c": 0, "n_dragon5_calls": 1, "n_dragon5_rejected": 0, "log": [], "seed": 0, "ppf_c": 1.6, "kinf_hi": 1.12,
         "timestamp": ts,
         "provenance": "Phase 0 calibration point 5 (SYM oracle, full run: k=1.113055 PPF=1.5102 CL=1486.8957); type_idx from "
                       "ring_design3 #13 (BOL re-run PPF 1.5102 matches). Feasible for PPF<1.6, infeasible for PPF<1.4."}
for r in (rec14, rec16):
    assert r["real_k_inf_max"] <= 1.12 and 4.25 <= r["E"] <= 4.36
assert rec14["real_PPF"] < 1.4 and rec16["real_PPF"] < 1.6
for name, rec in (("phase1_sym_confirmed_feasible.json", rec14), ("phase1_sym_confirmed_feasible_ppf16.json", rec16)):
    if os.path.exists(name) and not force:
        print(f"{name} exists -- not overwritten (use --force)"); continue
    json.dump(rec, open(name, "w"), indent=2)
    print(f"wrote {name}: E={rec['E']:.4f} k={rec['real_k_inf_max']:.5f} PPF={rec['real_PPF']:.4f} CL={rec['real_CL_proxy']:.3f}")
