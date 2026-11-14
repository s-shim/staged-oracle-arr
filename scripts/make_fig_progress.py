#!/usr/bin/env python3
"""Figure: Phase 2 (PPF<1.4) progress.  (a) best-so-far CL versus wall-clock time for each seed, with start value
and target; (b) every oracle call of every seed as a bar, shaded by how it ended.
Reads the Phase-2 seed logs and result files (read-only).  Writes fig_progress.pdf/.png and fig_progress_data.json."""
import argparse, glob, json, os, re
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ap = argparse.ArgumentParser()
ap.add_argument('--arr', required=True); ap.add_argument('--out', default='.')
ap.add_argument('--p2-start', required=True); ap.add_argument('--target', type=float, default=1475.7)
a = ap.parse_args()
t0 = datetime.strptime(a.p2_start, '%Y-%m-%d %H:%M:%S')
start = json.load(open(f'{a.arr}/phase1_sym_s8_feasible.json'))
CL0 = start['real_CL_proxy']

def absolute(tod, prev):
    """time of day -> datetime after prev (roll over midnight)"""
    t = datetime.combine(prev.date(), datetime.strptime(tod, '%H:%M:%S').time())
    return t if t >= prev else t + timedelta(days=1)

data = {}
for sd in sorted(glob.glob(f'{a.arr}/phase2_sym_ppf14_s8start/seed_*')):
    seed = int(sd.rsplit('_', 1)[1])
    lines = open(glob.glob(sd + '/*.log')[0], errors='replace').read().splitlines()
    prev = t0; calls = {}; 
    for i, l in enumerate(lines):
        m = re.search(r'\[DRAGON5 verify #(\d+)\] @ (\d\d:\d\d:\d\d)', l)
        if m:
            prev = absolute(m.group(2), prev); calls[int(m.group(1))] = dict(start=prev)
        m = re.search(r'real DRAGON5 call #(\d+) finished in (\d+)s', l)
        if m:
            k = int(m.group(1)); c = calls[k]; c['dur'] = float(m.group(2))
            res = next(lines[j] for j in range(i + 1, i + 6)
                       if any(t in lines[j] for t in ('DRAGON5 says', 'CONFIRMED', 'does NOT beat')))
            c['kind'] = 'BOL' if 'BOL PPF' in res else ('K' if 'ABORTED early' in res else 'FULL')
            mm = re.search(r'CONFIRMED real incumbent update: CL_proxy=([\d.]+)', res)
            c['new_cl'] = float(mm.group(1)) if mm else None
    inc = json.load(open(sd + '/hybrid_patterns/incumbent_latest.json'))
    summ = re.search(r'finished: CL=([\d.]+) \((\w+)\)', '\n'.join(lines))
    rows = []
    for k in sorted(calls):
        c = calls[k]
        s_h = (c['start'] - t0).total_seconds() / 3600
        rows.append(dict(call=k, t_start=s_h, t_end=s_h + c['dur'] / 3600, kind=c['kind'], new_cl=c['new_cl']))
    steps = [(0.0, CL0)] + [(r['t_end'], r['new_cl']) for r in rows if r['new_cl']]
    assert abs(steps[-1][1] - inc['real_CL_proxy']) < 0.01 and abs(steps[-1][1] - float(summ.group(1))) < 0.01, seed
    data[seed] = dict(rows=rows, steps=steps, stop=summ.group(2), t_stop=rows[-1]['t_end'], cl_final=steps[-1][1])
json.dump({str(k): v for k, v in data.items()}, open(os.path.join(a.out, 'fig_progress_data.json'), 'w'), indent=1)

# ------------------------------------------------------------------ drawing
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2de', '#ffffff'
SEED = {11: ('#2a78d6', 'o'), 12: ('#eb6834', 's'), 13: ('#1baf7a', '^'), 14: ('#eda100', 'D')}
KIND = {'BOL': '#b0b0b0', 'K': '#7a7a7a', 'FULL': '#262626'}
plt.rcParams.update({'font.size': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
fig, (ax, bx) = plt.subplots(2, 1, figsize=(4.8, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1], hspace=0.12))
XMAX = 26.0

for seed, d in data.items():
    col, mk = SEED[seed]
    xs = [t for t, _ in d['steps']] + [d['t_stop']]
    ys = [c for _, c in d['steps']] + [d['cl_final']]
    ax.step(xs, ys, where='post', color=col, lw=1.4, zorder=3)
    ax.plot([t for t, _ in d['steps'][1:]], [c for _, c in d['steps'][1:]], mk, color=col, ms=4.5,
            mec=SURF, mew=0.8, zorder=4, ls='none')
    if d['stop'] == 'TargetCLReached':
        ax.plot([d['t_stop']], [d['cl_final']], '*', color=col, ms=10, mec=INK, mew=0.5, zorder=5, ls='none')
ax.plot([0], [CL0], 'o', color=INK2, ms=4.5, mec=SURF, mew=0.8, zorder=5)
ax.axhline(a.target, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
ax.text(0.2, a.target + 0.25, f'target {a.target:.1f} d', fontsize=7, va='bottom', color=INK)
ax.text(0.35, CL0 - 0.3, f'start {CL0:.2f} d', fontsize=7, va='top', color=INK2)
pos = {11: (24.0, 1472.85, 'right', 'top'), 12: (13.45, 1478.16, 'left', 'center'),
       13: (20.3, 1477.35, 'right', 'bottom'), 14: (23.85, 1477.9, 'center', 'bottom')}
for seed in data:
    x, y, ha, va = pos[seed]
    ax.text(x, y, f"seed {seed}", fontsize=7, va=va, ha=ha, color=INK)
ax.set_ylabel('best confirmed CL (days)')
ax.set_ylim(1461.5, 1479.5)
ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.tick_params(length=2.5, width=0.5)
ax.text(0.0, 1.02, '(a)', transform=ax.transAxes, fontsize=8, fontweight='bold')
ax.legend(handles=[Line2D([], [], marker='*', color='none', mfc='#bbbbbb', mec=INK, mew=0.5, ms=10, label='target reached'),
                   Line2D([], [], marker='o', color='none', mfc='#bbbbbb', mec=SURF, ms=4.5, label='new incumbent')],
          loc='lower right', frameon=False, fontsize=7, handletextpad=0.3, borderaxespad=0.2)

order = [11, 12, 13, 14]
for yi, seed in enumerate(order):
    for r in data[seed]['rows']:
        bx.barh(yi, r['t_end'] - r['t_start'], left=r['t_start'], height=0.62, color=KIND[r['kind']],
                edgecolor=SURF, linewidth=0.8, zorder=3)
        if r['new_cl']:
            bx.plot([r['t_end']], [yi - 0.44], 'v', color=INK, ms=3.4, zorder=4, mew=0)
bx.set_yticks(range(len(order))); bx.set_yticklabels([f'seed {s}' for s in order])
bx.set_ylim(len(order) - 0.45, -0.7)
bx.set_xlim(0, XMAX); bx.set_xticks(range(0, 25, 4))
bx.set_xlabel('wall-clock time since start of Phase 2 (h)')
bx.grid(axis='x', color=GRID, lw=0.5); bx.set_axisbelow(True)
for s in ('top', 'right'): bx.spines[s].set_visible(False)
bx.tick_params(length=2.5, width=0.5)
bx.text(0.0, 1.02, '(b)', transform=bx.transAxes, fontsize=8, fontweight='bold')
bx.legend(handles=[Patch(fc=KIND['BOL'], ec=SURF, label='stopped at BOL ($\\mathrm{PPF}\\geq1.4$)'),
                   Patch(fc=KIND['K'], ec=SURF, label='stopped at peak ($k_{\\infty}>1.12$)'),
                   Patch(fc=KIND['FULL'], ec=SURF, label='completed'),
                   Line2D([], [], marker='v', color='none', mfc=INK, mec=INK, ms=3.4, label='new incumbent')],
          loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False, fontsize=7, columnspacing=1.2,
          handlelength=1.4, handletextpad=0.4)
fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.17)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(a.out, f'fig_progress.{ext}'), dpi=220)
for seed, d in data.items():
    print(seed, d['stop'], 'stop %.2f h' % d['t_stop'], 'final %.2f' % d['cl_final'], 'n_calls', len(d['rows']),
          'updates', [(round(t, 2), c) for t, c in d['steps'][1:]])
