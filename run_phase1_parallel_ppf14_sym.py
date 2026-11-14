"""
PPF<1.4 문제의 Phase 1 (오라클 페널티 최소화): 4개 시드를 4개 코어에서 독립 병렬 수행.
모든 시드는 Phase 0의 8개 패턴 중 페널티가 가장 작은 패턴(패턴 0)을 초기 best로 시작한다.

  python run_phase1_parallel_ppf14_sym.py --round 1     # 시드 1-4
  python run_phase1_parallel_ppf14_sym.py --round 2     # 시드 5-8
(두 라운드를 이어서: run_phase1_two_rounds_sym.py)

각 시드는 자기 디렉토리  phase1_sym_ppf14/round<R>/seed_<N>/  에서 돌며
  phase1_ppf14_seed<N>.log, phase1_calls/call_XXXX.json (호출마다), phase1_result.json (종료 시)
를 남긴다. 기존 phase1_sym_confirmed_feasible*.json 은 건드리지 않는다.
시드마다 독립: 한 시드가 feasible(P=0)을 찾아도 다른 시드는 계속 돌고, 각자 P=0 또는 시간 예산(기본 12 h)에서 멈춘다.
시간 예산은 "새 오라클 호출을 시작하지 않는" 기준이며 진행 중인 호출은 끝까지 둔다(최대 약 1콜 초과).
"""
import argparse, json, multiprocessing, os, sys, time

PPF_C = 1.4
KINF_HI = 1.12
MAX_WALL_HOURS = 12.0
MAX_DRAGON5_CALLS = 500       # 안전 상한 (실제 기준은 시간 예산)
BASE_DIR = os.path.abspath(".")
ROOT = os.path.join(BASE_DIR, "phase1_sym_ppf14")
PHASE0_PATH = os.path.join(BASE_DIR, "phase0_calibration_sym", "phase0_points.json")


def worker(seed, rnd, wall_hours):
    sys.path.insert(0, BASE_DIR)
    work_dir = os.path.join(ROOT, f"round{rnd}", f"seed_{seed}")
    os.makedirs(work_dir, exist_ok=True)
    log_file = open(os.path.join(work_dir, f"phase1_ppf14_seed{seed}.log"), "a", buffering=1)
    sys.stdout = log_file; sys.stderr = log_file
    print(f"=== phase1 worker seed={seed} round={rnd} PID={os.getpid()} started @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===",
          flush=True)
    from run_qubo_hybrid_dragon5 import run_greedy_phase1_oracle
    try:
        rec = run_greedy_phase1_oracle(seed=seed, ppf_c=PPF_C, kinf_hi=KINF_HI, phase0_path=PHASE0_PATH,
                                       save_dir=work_dir, max_wall_hours=wall_hours,
                                       max_dragon5_calls=MAX_DRAGON5_CALLS, verbose=True)
    except Exception as e:
        print(f"!!! worker seed={seed} FAILED: {type(e).__name__}: {e}", flush=True)
        with open(os.path.join(work_dir, "phase1_result.json"), "w") as f:
            json.dump({"seed": seed, "error": str(e)}, f, indent=2)
        raise
    print(f"=== worker seed={seed} finished: {rec['stop_reason']} best P={rec['best_P']:.5f} ===", flush=True)
    log_file.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--wall-hours", type=float, default=MAX_WALL_HOURS)
    a = ap.parse_args()
    seeds = [4 * (a.round - 1) + i for i in (1, 2, 3, 4)]
    os.makedirs(ROOT, exist_ok=True)
    if not os.path.exists(PHASE0_PATH):
        raise FileNotFoundError(PHASE0_PATH)
    print(f"Phase 1 (PPF<{PPF_C}), round {a.round}: seeds={seeds}, wall budget {a.wall_hours} h per seed, "
          f"started {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    ctx = multiprocessing.get_context("spawn")
    procs = []
    for s in seeds:
        p = ctx.Process(target=worker, args=(s, a.round, a.wall_hours), name=f"phase1-seed{s}")
        p.start(); procs.append(p); print(f"  started seed={s} PID={p.pid}", flush=True)
    for p in procs:
        p.join()
    print(f"\n=== ROUND {a.round} FINISHED @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print(f"{'seed':>5} {'stop':>20} {'best_P':>9} {'calls':>6} {'abortPPF':>9} {'abortK':>7} {'hours':>7}", flush=True)
    for s in seeds:
        try:
            r = json.load(open(os.path.join(ROOT, f"round{a.round}", f"seed_{s}", "phase1_result.json")))
            print(f"{s:>5} {r['stop_reason']:>20} {r['best_P']:>9.5f} {r['n_dragon5_calls']:>6} "
                  f"{r['n_aborted_ppf']:>9} {r['n_aborted_k']:>7} {r['elapsed_sec']/3600:>7.2f}", flush=True)
        except Exception as e:
            print(f"{s:>5}  (no result: {e})", flush=True)


if __name__ == "__main__":
    main()
