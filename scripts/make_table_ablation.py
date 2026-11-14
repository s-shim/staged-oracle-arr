#!/usr/bin/env python3
"""Ablation table: main Phase-2 run vs (a) no early termination vs (b) no surrogate gate (PPF<1.4, seeds 11-14).
Reads the per-seed logs (read-only):
  <arr>/phase2_sym_ppf14_s8start/seed_*/greedy_phase2_seed*_ppf14_s8start.log   (main)
  <arr>/phase2_sym_ppf14_abl_noet/seed_*/...abl_noet.log                         (a)
  <arr>/phase2_sym_ppf14_abl_nogate/seed_*/...abl_nogate.log                     (b)
Counts only real oracle calls ('real DRAGON5 call #N finished in Ts' + the result line that follows); cached re-uses
are not counted.  Writes gen_ablation.tex and ablation_stats.json to --out.
Usage: make_table_ablation.py --arr /mnt/d/dragon/arr --out .
"""
import argparse, glob, json, os, re
ap = argparse.ArgumentParser(); ap.add_argument('--arr', required=True); ap.add_argument('--out', default='.')
a = ap.parse_args()
TARGET = 1475.7
RUNS = [('Main run', 'phase2_sym_ppf14_s8start'), ('(a) no early termination', 'phase2_sym_ppf14_abl_noet'),
        ('(b) no surrogate gate', 'phase2_sym_ppf14_abl_nogate')]

def parse(log):
    lines = open(log, errors='replace').read().splitlines()
    calls = []   # (seconds, kind) kind in {'abort', 'infeasible', 'feasible'}
    for i, l in enumerate(lines):
        m = re.search(r'real DRAGON5 call #(\d+) finished in (\d+)s', l)
        if not m: continue
        res = ''
        for j in range(i + 1, min(i + 6, len(lines))):
            if 'DRAGON5 says' in lines[j] or 'CONFIRMED' in lines[j] or 'does NOT beat' in lines[j]:
                res = lines[j]; break
        assert res, f'no result after call {m.group(1)} in {log}'
        if 'ABORTED early' in res: kind = 'abort'
        elif 'INFEASIBLE' in res: kind = 'infeasible'
        else: kind = 'feasible'
        calls.append((float(m.group(2)), kind))
    txt = '\n'.join(lines)
    fin = re.search(r'finished: CL=([\d.]+) \((\w+)\)', txt)
    wall = re.search(r'wall time = ([\d.]+)s', txt)
    return calls, float(fin.group(1)), fin.group(2), float(wall.group(1)) / 3600

stats, rows = {}, []
for label, d in RUNS:
    tot = dict(calls=0, sec=0.0, abort=0.0, notfeas=0.0, reached=0)
    rows.append(('group', label))
    for sd in sorted(glob.glob(f'{a.arr}/{d}/seed_*')):
        seed = int(sd.rsplit('_', 1)[1])
        calls, cl, stop, wall_h = parse(glob.glob(sd + '/*.log')[0])
        sec = sum(c[0] for c in calls)
        ab = sum(c[0] for c in calls if c[1] == 'abort')
        nf = sum(c[0] for c in calls if c[1] in ('abort', 'infeasible'))
        reached = stop == 'TargetCLReached'
        stats[f'{d}/{seed}'] = dict(calls=len(calls), core_h=sec / 3600, abort_share=ab / sec, notfeas_share=nf / sec,
                                    cl=cl, stop=stop, t_target_h=wall_h if reached else None)
        tot['calls'] += len(calls); tot['sec'] += sec; tot['abort'] += ab; tot['notfeas'] += nf; tot['reached'] += reached
        rows.append(('seed', seed, len(calls), sec / 3600, 100 * ab / sec, 100 * nf / sec,
                     f'{wall_h:.1f}' if reached else '--', cl))
    rows.append(('sum', tot['reached'], tot['calls'], tot['sec'] / 3600, 100 * tot['abort'] / tot['sec'],
                 100 * tot['notfeas'] / tot['sec']))
    stats[d] = dict(calls=tot['calls'], core_h=tot['sec'] / 3600, abort_share=tot['abort'] / tot['sec'],
                    notfeas_share=tot['notfeas'] / tot['sec'], reached=tot['reached'])
json.dump(stats, open(os.path.join(a.out, 'ablation_stats.json'), 'w'), indent=1)

o = [r'''\begin{table}[t]
\caption{Ablations of the Phase-2 search for PPF$<$1.4 (same start pattern, seeds 11--14, 24~h budget, target $\CL\ge1475.7$~d, same cache of earlier evaluations). Calls are real oracle calls. ``Early-stopped'' is the share of the oracle time spent on calls aborted at BOL or at the $\kinf$ peak; ``infeasible'' is the share spent on calls that returned an infeasible pattern (aborted or run to completion). Time to target is the stored wall time of the seed that reached it.}
\label{tab:ablation}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{rrrrrrr}
\toprule
Seed & Calls & Core-h & Early-stopped (\%) & Infeasible (\%) & To target (h) & Final $\CL$ (d) \\
\midrule''']
first = True
for r in rows:
    if r[0] == 'group':
        if not first: o.append(r'\midrule')
        o.append(r'\multicolumn{7}{l}{\emph{%s}} \\' % r[1]); first = False
    elif r[0] == 'seed':
        o.append(f'{r[1]} & {r[2]} & {r[3]:.1f} & {r[4]:.0f} & {r[5]:.0f} & {r[6]} & {r[7]:.2f} \\\\')
    else:
        o.append(f'Sum & {r[2]} & {r[3]:.1f} & {r[4]:.0f} & {r[5]:.0f} & {r[1]}/4 reach target & \\\\')
o += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
open(os.path.join(a.out, 'gen_ablation.tex'), 'w').write('\n'.join(o))
print('\n'.join(o))
