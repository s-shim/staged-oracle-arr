"""
PPF_1_4의 confirmed real-feasible 시작점을 PPF_1_6용으로 재사용.

PPF<1.4를 만족하는 해는 자동으로 PPF<1.6도 만족한다 (E, N_GAD, k_inf 조건은
ppf_c와 무관하게 동일). 따라서 real DRAGON5를 다시 부를 필요 없이,
phase1_confirmed_feasible.json(PPF_1_4용)을 그대로 phase1_confirmed_feasible_ppf16.json
으로 복사하고 ppf_c 필드만 1.6으로 갱신한다.

사용법:
  python3 prepare_phase1_ppf16_from_ppf14.py
"""
import json
import time

SRC = "phase1_sym_confirmed_feasible.json"       # PPF_1_4의 Phase 1 결과
DST = "phase1_sym_confirmed_feasible_ppf16.json"  # PPF_1_6용으로 재사용할 파일

with open(SRC) as f:
    d = json.load(f)

assert d["real_PPF"] < 1.6, (
    f"real_PPF={d['real_PPF']} >= 1.6 -- PPF_1_4 결과가 PPF_1_6에 대해 "
    f"feasible하지 않습니다. 재사용 불가, Phase 1을 새로 돌리세요.")

d["ppf_c"] = 1.6
d["reused_from"] = SRC
d["reuse_note"] = (
    "PPF<1.4를 만족하는 해는 자동으로 PPF<1.6도 만족하므로, 이 파일은 "
    "새로 real DRAGON5를 부르지 않고 PPF_1_4 결과를 그대로 재사용한 것입니다. "
    "real_k_inf_max/real_PPF/real_CL_proxy 등은 원본 PPF_1_4 검증값 그대로입니다.")
d["reuse_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

with open(DST, "w") as f:
    json.dump(d, f, indent=2)

print(f"{SRC} -> {DST}")
print(f"  real_PPF={d['real_PPF']:.4f} (< 1.6, OK)")
print(f"  real_CL_proxy={d['real_CL_proxy']:.3f}")
print(f"  ppf_c={d['ppf_c']} 로 갱신, real DRAGON5 콜 0번 사용")
