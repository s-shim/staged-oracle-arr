# Staged oracle evaluation with early termination — BWR 10x10 / DRAGON5

Code, data and experiment records for the paper (in preparation)

> *Staged oracle evaluation with early termination for combinatorial search against an expensive PDE oracle*
> Kunok Chang, Ho Jin Park, Sangho Shim.

The oracle is a DRAGON5 ("SYM") lattice calculation of a BWR 10x10 GE14-type assembly. The problem follows the source study of
Radaideh et al., *Nucl. Eng. Des.* **372**, 110966 (2021): maximize cycle length (CL) over fuel enrichment and gadolinia placement,
subject to peak k-infinity <= 1.12, beginning-of-life (BOL) power peaking factor (PPF) < p_max, and no adjacent gadolinia rods
(263 binary variables, 35 one-hot groups, about 2.4e30 patterns).

**The full method** (the best-performing configuration, and the one this repository is built around) evaluates every oracle call in
stages and aborts it as soon as a stage shows the pattern is infeasible (BOL PPF, then peak k-infinity, then the full ~95 min
depletion run), and a cheap surrogate gate removes candidates that cannot beat the incumbent before they reach the oracle.
In the published run, 3 of 4 seeds reach the target CL >= 1475.7 d (PPF < 1.4) within 24 h on 4 cores; without early abort 1 of 4 does, and
without the gate 1 of 4 does (ablation results are included).

## Quick start

Requirements: Linux (or WSL), Python >= 3.9, DRAGON5 (Version5.0) and the DLIB cross-section library (SHEM295). **DRAGON5 and DLIB are not
included.** Everything except steps 4-5 below works without DRAGON5.

```bash
git clone <this repository> && cd <repo>
pip install -r requirements.txt

export DRAGON5_EXE=/path/to/Version5.0/Dragon/bin/Linux_x86_64/Dragon
export DRAGON5_LIBRARY=/path/to/DLIB_295

python reproduce.py check          # 1. environment, inputs, DRAGON5 paths (seconds)
python reproduce.py tables         # 2. regenerate all paper tables and the figure from the shipped logs (no DRAGON5)
python reproduce.py summary phase2_sym_ppf14_s8start      # 3. look at the published main run
python reproduce.py oracle-check   # 4. one oracle run of the start pattern vs. stored values (~1-1.5 h, 1 core)
python reproduce.py phase2         # 5. the main experiment: seeds 11-14, 24 h, 4 cores, output in runs/phase2_full_method/
```

Shorter variants of step 5: `python reproduce.py phase2 --seeds 12 --hours 14` (one seed, one core; seed 12 reached the target at 13.0 h in the
published run), or `--seeds 11 12 --hours 6`. Use `python reproduce.py summary runs/phase2_full_method` at any time to see progress.
A seed stops when it reaches the target; no new oracle call starts after the time budget. Output: `seed_<N>/phase2_seed<N>.log`,
`hybrid_patterns/call_*.json` (one per oracle call), `hybrid_patterns/incumbent_latest.json` (best pattern so far), `result_summary.json`.

Phase 2 starts from the shipped Phase-1 result (`phase1_sym_s8_feasible.json`, CL = 1462.7 d) and the shipped cache of 177 patterns that were
already evaluated in Phase 0/1 (`prior_evaluations_sym_ppf14.json`), exactly as in the paper, so Phase 0/1 do not have to be repeated.
The search is seeded; on the same DRAGON5/DLIB the first oracle call of seed 11 is the same candidate as in the published run. Wall-clock times, and
therefore time-to-target, depend on your machine: the paper used 4 physical cores, one oracle call per core (BOL-PPF abort ~20 min,
peak-k abort ~64 min, completed run ~95 min under that load).

## What is here

| Path | Content |
|---|---|
| `reproduce.py` | Entry point: `check`, `tables`, `oracle-check`, `phase2`, `summary` |
| `bwr10x10_*.py` | Problem definition: geometry, pin map, surrogate model, DRAGON5 input generation and output parsing, reference patterns |
| `arr_quantum_inspired.py`, `arr_hybrid_dragon5.py`, `run_qubo_hybrid_dragon5.py` | Adaptive randomized rounding (ARR) search; hybrid version with the staged early abort and the surrogate gate (`early_abort`, `surrogate_gate`, `gate_margin_days`, `surrogate_ppf_gate`) |
| `prior_evaluations_sym_ppf14.json`, `phase1_sym_s8_feasible.json`, `phase0_calibration_sym/` | Inputs of Phase 2 |
| `phase2_sym_ppf14_s8start/seed_*/` | **Published main run** (full method, seeds 11-14): log, per-call JSON, incumbent, summary |
| `phase2_sym_ppf14_abl_noet/`, `phase2_sym_ppf14_abl_nogate/` | Ablations: no early abort / no surrogate gate (same start, seeds, budget, cache). Driver: `run_phase2_ablation_ppf14_sym.py noet|nogate` |
| `phase2_sym_ppf16/` | Variant PPF < 1.6 (target CL >= 1488.0 d, 12 h). Drivers: `prepare_phase1_ppf16_from_ppf14_sym.py`, `run_greedy_phase2_parallel_ppf16_sym.py` |
| `phase1_sym_ppf14/round{1,2}/seed_*/`, `phase1_sym_ppf14_all_calls.json` | Phase 1 (oracle-penalty search for a feasible pattern, 8 seeds, 12 h each): per-call records and logs. Drivers: `run_phase1_two_rounds_sym.py`, `run_phase1_parallel_ppf14_sym.py` |
| `phase0_extra_pattern8.json`, `phase0_sym.log`, `rerun_phase0_sym.py` | Phase 0: 8 calibration patterns evaluated with the oracle |
| `run_greedy_phase2_parallel_ppf14_s8start_sym.py` | The original launcher that produced the published main run (`reproduce.py phase2` uses identical settings) |
| `phase2_sym_ppf14/` | An earlier Phase 2 run (seeds 1-4, different start pattern); kept for completeness, not used for the paper tables |
| `scripts/` | Table and figure generators (called by `reproduce.py tables`) |

Table generators read only the raw logs: `scripts/make_tables.py` (Phase 1/2 tables, cost accounting, best patterns), `make_table_ppf16.py`,
`make_table_ablation.py`, `make_fig_progress.py`. Call counts in the tables are real oracle calls only; cached re-uses are not counted.

## Re-running everything from scratch

```bash
python rerun_phase0_sym.py                 # Phase 0 (8 patterns)
python run_phase1_two_rounds_sym.py        # Phase 1, 2 x 12 h, 4 cores
python make_phase2_s8start_files_sym.py    # start pattern + cache from Phase 1 (existing files are not overwritten without --force)
python reproduce.py phase2
```
These drivers write into directories next to the code and refuse to start in a non-empty output directory, so move the shipped result
folders away first (or work in a fresh clone). Phase 1 is a stochastic search over many hours; a different Phase 1 outcome gives a different
start pattern, which is why the shipped start pattern is used for Phase 2.

## Notes

- Logs record absolute paths of the original machine (`/mnt/d/dragon/arr`); they are left unmodified.
- Not included yet: the scripts behind the diagnostic checks of the paper's validation section (BOL pin-map, ring and mid-fidelity checks).
- Some code comments are in Korean.

## License

MIT, see `LICENSE`.

## Citation

To be added when the paper is available.
