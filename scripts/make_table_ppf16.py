#!/usr/bin/env python3
"""PPF<1.6 variant: Phase 2 started from the Phase-0 point 5 (already feasible).
Reads (read-only): <arr>/phase2_sym_ppf16/seed_*/{*.log,result_summary.json,hybrid_patterns/incumbent_latest.json},
<arr>/phase1_sym_confirmed_feasible_ppf16.json, <arr>/phase0_calibration_sym/phase0_points.json.
Writes gen_p16.tex and p16_stats.json into --out."""
import argparse, glob, json, os, re
ap = argparse.ArgumentParser(); ap.add_argument('--arr', required=True); ap.add_argument('--out', default='.')
a = ap.parse_args(); ARR = a.arr
start = json.load(open(f'{ARR}/phase1_sym_confirmed_feasible_ppf16.json'))
p0 = json.load(open(f'{ARR}/phase0_calibration_sym/phase0_points.json'))
pt5 = [p for p in p0 if p['index'] == 5][0]
assert abs(pt5['real_CL_proxy'] - start['real_CL_proxy']) < 1e-6      # start file = Phase-0 point 5
feas16 = [p['index'] for p in p0 if p['real_k_inf_max'] <= 1.12 and p['real_PPF'] < 1.6 and 4.25 <= p['E'] <= 4.36 and 16 <= p['n_gad'] <= 18]
feas14 = [p['index'] for p in p0 if p['real_k_inf_max'] <= 1.12 and p['real_PPF'] < 1.4 and 4.25 <= p['E'] <= 4.36 and 16 <= p['n_gad'] <= 18]
rows = []; stats = {'phase0_feasible_ppf16': feas16, 'phase0_feasible_ppf14': feas14, 'seeds': {}}
for sd in sorted(glob.glob(f'{ARR}/phase2_sym_ppf16/seed_*')):
    seed = int(sd.rsplit('_', 1)[1])
    lines = open(glob.glob(sd + '/*.log')[0], errors='replace').read().splitlines()
    calls = []
    for i, l in enumerate(lines):
        m = re.search(r'real DRAGON5 call #(\d+) finished in (\d+)s', l)
        if not m: continue
        res = next((lines[j] for j in range(i + 1, min(i + 6, len(lines)))
                    if any(t in lines[j] for t in ('DRAGON5 says', 'CONFIRMED', 'does NOT beat'))), None)
        assert res, f'no result line after call {m.group(1)} in seed {seed}'
        calls.append((float(m.group(2)), 'ABORTED' in res))
    summ = json.load(open(sd + '/result_summary.json'))
    inc = json.load(open(sd + '/hybrid_patterns/incumbent_latest.json'))
    assert summ['n_dragon5_calls'] == len(calls)
    core_h = sum(t for t, _ in calls) / 3600
    stats['seeds'][seed] = dict(calls=len(calls), aborted=sum(ab for _, ab in calls), core_h=core_h,
                                wall_h=summ['elapsed_sec'] / 3600, stop=summ['stop_reason'], cl=inc['real_CL_proxy'])
    rows.append(f"{seed} & {len(calls)} & {core_h:.2f} & {summ['elapsed_sec']/3600:.1f} & {inc['E']:.4f} & {inc['n_gad']} & "
                f"{inc['real_k_inf_max']:.4f} & {inc['real_PPF']:.4f} & {inc['real_CL_proxy']:.2f} \\\\")
tot_calls = sum(v['calls'] for v in stats['seeds'].values()); tot_h = sum(v['core_h'] for v in stats['seeds'].values())
stats['total_calls'] = tot_calls; stats['total_core_h'] = tot_h
tex = r'''\begin{table}[t]
\caption{Variant PPF$<$1.6: Phase~2 from the Phase-0 pattern with the largest cycle length (four seeds, target $\CL\ge1488.0$ days, $12$~h budget). Oracle outputs of the start pattern and of the final incumbent of each seed. No oracle call was terminated early.}
\label{tab:p16}
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrrr}
\toprule
Pattern & Calls & Core-h & Time (h) & $E$ & $N_{\mathrm{GAD}}$ & $\kinf$ & $\PPF$ & $\CL$ (d) \\
\midrule
''' + f"Start (Phase-0 pt.~5) & -- & -- & -- & {start['E']:.4f} & {pt5['n_gad']} & {start['real_k_inf_max']:.4f} & {start['real_PPF']:.4f} & {start['real_CL_proxy']:.2f} \\\\\n"
tex += '\n'.join('Seed ' + r for r in rows) + '\n\\midrule\n'
tex += f"Sum & {tot_calls} & {tot_h:.1f} & & & & & & \\\\\n" + r'''\bottomrule
\end{tabular}
\end{table}
'''
open(os.path.join(a.out, 'gen_p16.tex'), 'w').write(tex)
json.dump(stats, open(os.path.join(a.out, 'p16_stats.json'), 'w'), indent=1)
print(json.dumps(stats, indent=1)); print(tex)
