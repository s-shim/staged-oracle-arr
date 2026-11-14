#!/usr/bin/env python3
"""Build the cost / result tables of the manuscript directly from the raw
oracle-call traces.  Reads (read-only):
  <arr>/phase1_sym_ppf14/round{1,2}/seed_*/phase1_calls/call_*.json  (Phase 1)
  <arr>/phase2_sym_ppf14_s8start/seed_*/greedy_phase2_seed*_ppf14_s8start.log (Phase 2 calls)
  <arr>/phase2_sym_ppf14_s8start/seed_*/hybrid_patterns/incumbent_latest.json (Phase 2 final)
  <arr>/phase1_sym_s8_feasible.json (Phase 2 start pattern)
Writes: generated_tables.tex, cost_stats.json (in --out).
Usage: make_tables.py --arr /mnt/d/dragon/arr --out . --p2-start "2026-11-08 20:54:00"
"""
import argparse, glob, json, os, re, statistics as st
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument('--arr', required=True)
ap.add_argument('--out', default='.')
ap.add_argument('--p2-start', required=True, help='wall-clock start of Phase 2 (local time of the logs)')
a = ap.parse_args()
ARR = a.arr
CL_TARGET = 1475.7

def stage(aborted_text):
    if aborted_text and 'BOL' in aborted_text: return 'BOL'
    if aborted_text: return 'K'
    return 'FULL'

# ---------------- Phase 1 ----------------
P1 = {}
for sd in sorted(glob.glob(f'{ARR}/phase1_sym_ppf14/round*/seed_*')):
    seed = int(sd.rsplit('_', 1)[1])
    calls = []
    for f in sorted(glob.glob(sd + '/phase1_calls/call_*.json')):
        j = json.load(open(f))
        calls.append((stage(j.get('aborted')), float(j['dragon_elapsed_s'])))
    res = json.load(open(sd + '/phase1_result.json'))
    P1[seed] = dict(calls=calls, stop=res.get('stop_reason'))

# ---------------- Phase 2 ----------------
P2 = {}
t0 = datetime.strptime(a.p2_start, '%Y-%m-%d %H:%M:%S')
for sd in sorted(glob.glob(f'{ARR}/phase2_sym_ppf14_s8start/seed_*')):
    seed = int(sd.rsplit('_', 1)[1])
    log = glob.glob(sd + '/*.log')[0]
    lines = open(log, errors='replace').read().splitlines()
    calls = []
    for i, l in enumerate(lines):
        m = re.search(r'real DRAGON5 call #(\d+) finished in (\d+)s', l)
        if not m: continue
        res = ''
        for j in range(i + 1, min(i + 6, len(lines))):
            if 'DRAGON5 says' in lines[j] or 'CONFIRMED' in lines[j] or 'does NOT beat' in lines[j]:
                res = lines[j]; break
        assert res, f'no result line after call {m.group(1)} in {log}'
        calls.append((stage('ABORTED early' in res and res), float(m.group(2))))
    txt = '\n'.join(lines)
    stop = re.search(r'finished: CL=([\d.]+) \((\w+)\)', txt)
    inc = json.load(open(sd + '/hybrid_patterns/incumbent_latest.json'))
    wm = re.search(r'wall time = ([\d.]+)s', txt)   # stored wall time of the seed (same source as the ablation table)
    wall = float(wm.group(1)) / 3600 if wm else (datetime.strptime(inc['saved_at'], '%Y-%m-%d %H:%M:%S') - t0).total_seconds() / 3600
    P2[seed] = dict(calls=calls, stop=stop.group(2), cl_final=float(stop.group(1)), inc=inc, t_inc_h=wall)
start = json.load(open(f'{ARR}/phase1_sym_s8_feasible.json'))

# ---------------- statistics ----------------
def agg(calls_by_stage):
    n = sum(len(v) for v in calls_by_stage.values())
    sec = sum(sum(v) for v in calls_by_stage.values())
    return n, sec / 3600
def pool(P):
    d = {'BOL': [], 'K': [], 'FULL': []}
    for s in P.values():
        for c, t in s['calls']: d[c].append(t)
    return d
D1, D2 = pool(P1), pool(P2)
DP = {k: D1[k] + D2[k] for k in D1}
tau = st.mean(DP['FULL'])            # pooled mean of completed runs
stats = {'tau_s': tau}
for name, D in (('phase1', D1), ('phase2', D2), ('pooled', DP)):
    n, h = agg(D)
    stats[name] = dict(calls=n, core_h=h,
        q={k: len(v) / n for k, v in D.items()},
        mean_s={k: st.mean(v) for k, v in D.items() if v},
        cbar_over_tau=h * 3600 / n / tau,
        allfull_core_h=n * tau / 3600)
json.dump(stats, open(os.path.join(a.out, 'cost_stats.json'), 'w'), indent=1)

def cell(D, k):  # "calls / core-h"
    return f'{len(D[k])} / {sum(D[k])/3600:.2f}'

def per_seed_stage(P, s):
    d = {'BOL': [], 'K': [], 'FULL': []}
    for c, t in P[s]['calls']: d[c].append(t)
    return d

out = []
# ---- Phase 1 per-seed table
out.append(r'''\begin{table}[t]
\caption{Phase 1 (PPF$<$1.4, oracle-penalty search from the best Phase-0 pattern), eight independent seeds in two batches of four: oracle calls by termination stage, shown as calls / core-hours.}
\label{tab:p1cost}
\centering
\begin{tabular}{rrrrrrl}
\toprule
Seed & BOL abort & $\kinf$ abort & Full run & Calls & Core-h & Outcome \\
\midrule''')
for s in sorted(P1):
    d = per_seed_stage(P1, s)
    n, h = agg(d)
    oc = 'feasible' if P1[s]['stop'] == 'FeasibleFound' else 'budget'
    out.append(f"{s} & {cell(d,'BOL')} & {cell(d,'K')} & {cell(d,'FULL')} & {n} & {h:.2f} & {oc} \\\\")
n, h = stats['phase1']['calls'], stats['phase1']['core_h']
nf = sum(1 for s in P1 if P1[s]['stop'] == 'FeasibleFound')
out.append(r'\midrule')
out.append(f"Sum & {cell(D1,'BOL')} & {cell(D1,'K')} & {cell(D1,'FULL')} & {n} & {h:.1f} & {nf} of {len(P1)} feasible \\\\")
out.append(r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}' + '\n')

# ---- Phase 2 per-seed cost table
out.append(r'''\begin{table}[t]
\caption{Phase 2 (PPF$<$1.4, four seeds started from the Phase-1 seed-8 pattern): oracle calls by termination stage, shown as calls / core-hours.}
\label{tab:p2cost}
\centering
\begin{tabular}{rrrrrr}
\toprule
Seed & BOL abort & $\kinf$ abort & Full run & Calls & Core-h \\
\midrule''')
for s in sorted(P2):
    d = per_seed_stage(P2, s); n, h = agg(d)
    out.append(f"{s} & {cell(d,'BOL')} & {cell(d,'K')} & {cell(d,'FULL')} & {n} & {h:.2f} \\\\")
n, h = stats['phase2']['calls'], stats['phase2']['core_h']
out.append(r'\midrule')
out.append(f"Sum & {cell(D2,'BOL')} & {cell(D2,'K')} & {cell(D2,'FULL')} & {n} & {h:.1f} \\\\")
out.append(r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}' + '\n')

# ---- Phase 2 outcome table
out.append(r'''\begin{table}[t]
\caption{Phase 2 outcome per seed. Start pattern: $\CL=%.1f$ days. Target: $\CL\ge%.1f$ days. ``Time to target'' is the wall-clock time at which the target-reaching incumbent was saved.}
\label{tab:results}
\centering
\begin{tabular}{rrrrrl}
\toprule
Seed & Oracle calls & Aborted & Final $\CL$ (d) & Time to target (h) & Stopped by \\
\midrule''' % (start['real_CL_proxy'], CL_TARGET))
for s in sorted(P2):
    d = per_seed_stage(P2, s); n, h = agg(d)
    ab = len(d['BOL']) + len(d['K'])
    hit = P2[s]['stop'] == 'TargetCLReached'
    tt = f"{P2[s]['t_inc_h']:.1f}" if hit else '--'
    why = 'target' if hit else 'wall budget'
    out.append(f"{s} & {n} & {ab} & {P2[s]['cl_final']:.2f} & {tt} & {why} \\\\")
out.append(r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}' + '\n')

# ---- stage timing table (pooled)
def rng(v): return f"{st.mean(v)/60:.1f} ({min(v)/60:.1f}--{max(v)/60:.1f})"
out.append(r'''\begin{table}[t]
\caption{Measured oracle time per call by termination stage, pooled over all Phase-1 and Phase-2 calls of the PPF$<$1.4 campaign (four runs in parallel on four physical cores): mean (minimum--maximum) in minutes.}
\label{tab:stagetimes}
\centering
\begin{tabular}{lrrr}
\toprule
Termination & Calls & Minutes per call & Share of calls \\
\midrule''')
nP = stats['pooled']['calls']
for k, lab in (('BOL', 'BOL: $\\PPF\\ge p_{\\max}$'), ('K', 'peak: $\\kinf>1.12$'), ('FULL', 'completed ($\\CL$ known)')):
    out.append(f"{lab} & {len(DP[k])} & {rng(DP[k])} & {100*len(DP[k])/nP:.0f}\\% \\\\")
out.append(r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}' + '\n')

# ---- final patterns table
out.append(r'''\begin{table}[t]
\caption{Oracle outputs of the Phase-2 start pattern and of the final incumbent of each seed (all with $N_{\mathrm{GAD}}=%d$ rods and certified by a completed oracle run).}
\label{tab:best}
\centering
\begin{tabular}{lrrrr}
\toprule
Pattern & $E$ (wt.\%%) & $\kinf$ & $\PPF$ & $\CL$ (d) \\
\midrule''' % P2[sorted(P2)[0]]['inc']['n_gad'])
out.append(f"Start (Phase-1 seed 8) & {start['E']:.4f} & {start['real_k_inf_max']:.4f} & {start['real_PPF']:.4f} & {start['real_CL_proxy']:.2f} \\\\")
for s in sorted(P2):
    i = P2[s]['inc']
    assert i['n_gad'] == P2[sorted(P2)[0]]['inc']['n_gad']
    out.append(f"Seed {s} & {i['E']:.4f} & {i['real_k_inf_max']:.4f} & {i['real_PPF']:.4f} & {i['real_CL_proxy']:.2f} \\\\")
out.append(r'\bottomrule' + '\n' + r'\end{tabular}' + '\n' + r'\end{table}' + '\n')

open(os.path.join(a.out, 'generated_tables.tex'), 'w').write('\n'.join(out))
print(json.dumps(stats, indent=1))

# one file per table (gen_<label>.tex), \input from main.tex
_txt = open(os.path.join(a.out, 'generated_tables.tex')).read()
for _chunk in re.split(r'(?=\\begin\{table\})', _txt):
    _m = re.search(r'\\label\{tab:(\w+)\}', _chunk)
    if _m:
        open(os.path.join(a.out, f'gen_{_m.group(1)}.tex'), 'w').write(_chunk.strip() + '\n')
