"""
[s8start 버전] Phase 1 시드 8의 feasible 패턴에서 시작, Phase 0+1에서 평가한 패턴은 재평가하지 않음, 시드 11-14, 24 h.
PPF_1_4 문제 (ppf_c=1.4) 전용: Phase 2 ARR을 4개 seed로 4개 코어에서 독립 병렬 수행.

PPF_1_6용(run_greedy_phase2_parallel_ppf16.py)과 동시에 같은 머신에서 돌려도
절대 충돌하지 않도록, 폴더/로그/시작파일 이름이 전부 "_ppf14"로 명확히 분리되어 있음.

각 워커는:
  - 자기만의 작업 디렉토리(phase2_parallel_ppf14/seed_N/)에서 실행됨
    (incumbent_latest.json 저장, DRAGON5 임시파일 등이 서로 충돌하지 않도록)
  - phase0_path/phase1_result_path는 원래 위치(절대경로)에서 그대로 읽음
    (Phase 1 결과와 Phase 0 캘리브레이션은 4개 워커가 공유해도 무방 -- 읽기 전용)
  - 자기만의 로그 파일에 출력을 씀

사용법:
  export DRAGON5_EXE=...
  export DRAGON5_LIBRARY=...
  nohup python3 run_greedy_phase2_parallel_ppf14_sym.py > parallel_launcher_ppf14.log 2>&1 &
"""
import json
import multiprocessing
import os
import sys
import time

# ---- 설정: 여기만 바꾸면 됩니다 ----
SEEDS = [11, 12, 13, 14]
PPF_C = 1.4
MAX_WALL_HOURS = 24.0           # 워커 1개당 시간 예산(시간). 진행 중인 DRAGON5 호출은 끝까지 두므로 최대 1콜(~110분) 초과 가능
SURROGATE_PPF_GATE = False      # 확산 서로게이트 PPF는 후보를 거르지 않음 (오라클 조기 중단이 PPF 판정)
GATE_MARGIN_DAYS = 3.0         # 서로게이트 CL이 인큐번트보다 최대 3일 낮은 후보까지 오라클이 판정 (0 = 기존)
MAX_DRAGON5_CALLS = 200         # 안전 상한(시간 예산이 실제 기준)
REFIT_PPF = True
MAX_ROUNDS = 100000
BASE_DIR = os.path.abspath(".")  # 이 스크립트를 실행하는 디렉토리 (원래 작업 디렉토리)
PARALLEL_ROOT = os.path.join(BASE_DIR, "phase2_sym_ppf14_s8start")   # 새 폴더: 이전 phase2_sym_ppf14 는 건드리지 않음
PHASE0_PATH = os.path.join(BASE_DIR, "phase0_calibration_sym", "phase0_points.json")
PHASE1_RESULT_PATH = os.path.join(BASE_DIR, "phase1_sym_s8_feasible.json")          # Phase 1 시드 8의 feasible 패턴
PRIOR_PATH = os.path.join(BASE_DIR, "prior_evaluations_sym_ppf14.json")             # Phase 0 + Phase 1에서 이미 평가한 178개 패턴
# ------------------------------------


def worker(seed):
    """자식 프로세스 진입점. 반드시 최상위 함수여야 함 (multiprocessing pickling)."""
    sys.path.insert(0, BASE_DIR)

    work_dir = os.path.join(PARALLEL_ROOT, f"seed_{seed}")
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    log_path = os.path.join(work_dir, f"greedy_phase2_seed{seed}_ppf14_s8start.log")
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    print(f"=== worker seed={seed} PID={os.getpid()} started @ {time.strftime('%H:%M:%S')} ===",
          flush=True)
    print(f"work_dir={work_dir}", flush=True)

    from run_qubo_hybrid_dragon5 import run_greedy_phase2

    try:
        result = run_greedy_phase2(
            seed=seed, ppf_c=PPF_C, refit_ppf=REFIT_PPF, verbose=True,
            max_dragon5_calls=MAX_DRAGON5_CALLS, max_wall_hours=MAX_WALL_HOURS, gate_margin_days=GATE_MARGIN_DAYS, surrogate_ppf_gate=SURROGATE_PPF_GATE, max_rounds=MAX_ROUNDS,
            phase0_path=PHASE0_PATH, phase1_result_path=PHASE1_RESULT_PATH, prior_evaluations_path=PRIOR_PATH,
            # target_cl은 지정 안 하면 ppf_c=1.4 기준으로 TARGET_CL_PPF_1_4 자동 선택
        )
    except Exception as e:
        print(f"!!! worker seed={seed} FAILED: {type(e).__name__}: {e}", flush=True)
        with open(os.path.join(work_dir, "result_summary.json"), "w") as f:
            json.dump({"seed": seed, "error": str(e)}, f, indent=2)
        raise

    summary = {
        "seed": seed,
        "real_best_score": result["real_best_score"],
        "n_dragon5_calls": result["n_dragon5_calls"],
        "n_dragon5_confirmed": result["n_dragon5_confirmed"],
        "n_dragon5_rejected": result["n_dragon5_rejected"],
        "q4c_phase2": result["q4c_phase2"],
        "elapsed_sec": result["elapsed_sec"],
        "stop_reason": result["stop_reason"],
    }
    with open(os.path.join(work_dir, "result_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"=== worker seed={seed} finished: CL={result['real_best_score']:.3f} "
          f"({result['stop_reason']}) ===", flush=True)
    log_file.close()
    return summary


def main():
    if os.path.exists(PARALLEL_ROOT) and os.listdir(PARALLEL_ROOT):
        raise SystemExit(f"{PARALLEL_ROOT} already exists and is not empty -- refusing to mix runs (move or rename it first).")
    os.makedirs(PARALLEL_ROOT, exist_ok=True)
    for p in (PHASE1_RESULT_PATH, PRIOR_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found -- run make_phase2_s8start_files_sym.py first.")
    if not os.path.exists(PHASE1_RESULT_PATH):
        raise FileNotFoundError(
            f"Phase 1 result not found at {PHASE1_RESULT_PATH} -- run Phase 1 first.")

    print(f"Launching {len(SEEDS)} parallel Phase 2 workers (PPF_1_4, ppf_c={PPF_C}): "
          f"seeds={SEEDS}", flush=True)
    print(f"Each worker: max_wall_hours={MAX_WALL_HOURS}, safety cap max_dragon5_calls={MAX_DRAGON5_CALLS} (independent per worker)",
          flush=True)
    print(f"Logs: {PARALLEL_ROOT}/seed_<N>/greedy_phase2_seed<N>_ppf14.log\n", flush=True)

    ctx = multiprocessing.get_context("spawn")
    procs = []
    for seed in SEEDS:
        p = ctx.Process(target=worker, args=(seed,), name=f"phase2-ppf14-seed{seed}")
        p.start()
        procs.append(p)
        print(f"  started seed={seed}  PID={p.pid}", flush=True)

    for p in procs:
        p.join()

    print("\n=== ALL WORKERS FINISHED (PPF_1_4) ===", flush=True)
    print(f"{'seed':>6} {'CL_proxy':>12} {'calls':>6} {'confirmed':>10} "
          f"{'rejected':>9} {'stop_reason':>22}", flush=True)
    best_seed, best_cl = None, float("-inf")
    for seed in SEEDS:
        summary_path = os.path.join(PARALLEL_ROOT, f"seed_{seed}", "result_summary.json")
        try:
            with open(summary_path) as f:
                s = json.load(f)
            cl = s.get("real_best_score")
            print(f"{seed:>6} {cl:>12.3f} {s.get('n_dragon5_calls'):>6} "
                  f"{s.get('n_dragon5_confirmed'):>10} {s.get('n_dragon5_rejected'):>9} "
                  f"{str(s.get('stop_reason')):>22}", flush=True)
            if cl is not None and cl > best_cl:
                best_cl, best_seed = cl, seed
        except Exception as e:
            print(f"{seed:>6}  (no result: {e})", flush=True)

    if best_seed is not None:
        print(f"\nBest (PPF_1_4): seed={best_seed}  CL_proxy={best_cl:.3f}  "
              f"-> {PARALLEL_ROOT}/seed_{best_seed}/hybrid_patterns/incumbent_latest.json",
              flush=True)


if __name__ == "__main__":
    main()
