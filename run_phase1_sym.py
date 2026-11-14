"""
run_phase1_sym.py -- Phase 1 (find a REAL-feasible start for PPF<1.4) with the SYM oracle, using the
re-evaluated Phase-0 calibration. Writes phase1_sym_confirmed_feasible.json (old file untouched).
Run AFTER rerun_phase0_sym.py has finished.

  nohup python run_phase1_sym.py > phase1_sym.log 2>&1 &
"""
from run_qubo_hybrid_dragon5 import run_phase1_only

result = run_phase1_only(seed=1, ppf_c=1.4, refit_ppf=True, verbose=True,
                         phase0_path="phase0_calibration_sym/phase0_points.json",
                         save_path="phase1_sym_confirmed_feasible.json")
print("PHASE 1 RESULT:", {k: result[k] for k in ("E", "real_k_inf_max", "real_PPF", "real_CL_proxy") if k in result})
