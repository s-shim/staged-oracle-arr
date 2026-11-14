"""PPF<1.4 Phase 1을 12 h x 2 라운드(시드 1-4, 5-8)로 연속 실행. 항상 4코어만 사용.
  export DRAGON5_EXE=...; export DRAGON5_LIBRARY=...
  nohup python3 run_phase1_two_rounds_sym.py > phase1_two_rounds.log 2>&1 &
"""
import subprocess, sys, time

def main():
    t0 = time.perf_counter()
    for rnd in (1, 2):
        print(f"\n{'=' * 70}\n# Phase 1 round {rnd} (seeds {4*(rnd-1)+1}-{4*rnd}) started @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 70}\n", flush=True)
        r = subprocess.run([sys.executable, "run_phase1_parallel_ppf14_sym.py", "--round", str(rnd)])
        print(f"\n# round {rnd} finished (exit {r.returncode}) @ {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"\n# ALL ROUNDS COMPLETE -- total {(time.perf_counter()-t0)/3600:.1f} h", flush=True)

if __name__ == "__main__":
    main()
