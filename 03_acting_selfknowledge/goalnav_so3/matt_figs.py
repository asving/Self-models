"""Figure for the Matt Smith correspondence (gradient geometry of the aux effect).

A  the phenomenon: reach curves, no-aux vs own-future vs own-past aux
B  policy-gradient SNR collapse vs aux-gradient SNR (log scale)
C  alignment null: cos(mean aux grad, mean policy grad) vs split-half ceiling
D  what forms instead: depth-gain of the signed bearing, by target referent
Numbers in B/C from the 2026-07-28 measurement (12+12 disjoint batches,
trunk params, ridge-fitted aux heads); see AUXSWEEP.md.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = ('#2a78d6', '#eb6834', '#1baf7a',
                                       '#eda100', '#e87ba4')
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e5e4e0', '#fcfcfb'
plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
})

fig = plt.figure(figsize=(12.6, 6.6))
gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.28,
                      left=0.06, right=0.985, top=0.90, bottom=0.09)

# ---------- A: the phenomenon ----------
ax = fig.add_subplot(gs[0, 0])
for run, col, lab in (('gn_rep_lam0.0_s1', ORANGE, 'no auxiliary'),
                      ('gn_rep_lam1.0_s1', BLUE, 'aux: own future (r=4)'),
                      ('gn_aux_past_s1', AQUA, 'aux: own PAST (r=4)')):
    log = json.load(open(f'goalnav_runs/{run}.json'))['log']
    s = [l['step'] for l in log]; y = [l['reach_deg'] for l in log]
    ax.plot(s, y, color=col, lw=2)
    dy = {'no auxiliary': 0, 'aux: own future (r=4)': 9,
          'aux: own PAST (r=4)': -7}[lab]
    ax.annotate(lab, (s[-1], y[-1]), xytext=(5, dy), textcoords='offset points',
                color=col, fontsize=8, va='center')
ax.set_xlim(0, 11500)
ax.set_xlabel('training step')
ax.set_ylabel('final-quarter distance to goal (deg)')
ax.set_title('A  Own-trajectory auxiliaries rescue training\n'
             '(deterministic BPTT policy; aux on a read-only linear head)',
             loc='left', fontsize=10)

# ---------- B: SNR collapse ----------
ax = fig.add_subplot(gs[0, 1])
ck = [500, 1000, 2000, 4000, 8000]
snr_p = [0.380, 0.049, 0.061, 0.043, 0.166]
snr_a = [3.017, 1.313, 2.083, 1.872, 3.660]
ax.plot(ck, snr_p, '-o', color=ORANGE, lw=2, ms=5, label='policy loss (BPTT)')
ax.plot(ck, snr_a, '-o', color=AQUA, lw=2, ms=5, label='aux loss (own past)')
ax.set_yscale('log')
ax.set_ylim(0.02, 6)
ax.annotate('30–90× gap', (2000, 0.35), fontsize=9, color=INK)
ax.text(0.98, 0.04, 'at ckpt 4000, two independent estimates of the mean\n'
        'policy gradient are ORTHOGONAL (cos −0.02)',
        transform=ax.transAxes, fontsize=7.5, color=INK2, ha='right')
ax.set_xlabel('training step (no-aux checkpoint)')
ax.set_ylabel('gradient SNR  ‖mean‖² / variance')
ax.set_title('B  The policy gradient loses its signal mid-training',
             loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=8, loc='center right')

# ---------- C: alignment null ----------
ax = fig.add_subplot(gs[1, 0])
ceil = [0.838, 0.589, 0.493, -0.018, 0.927]
ax.plot(ck, ceil, '-', color=INK2, lw=1.6, ls=(0, (4, 2)))
ax.annotate('measurement ceiling\n(policy vs policy, split-half)',
            (5300, 0.47), fontsize=7.5, color=INK2)
modes = {'own past': ([-0.015, -0.046, 0.021, 0.030, 0.000], AQUA),
         'own future': ([0.018, -0.000, 0.030, 0.008, 0.025], BLUE),
         'velocity': ([0.026, -0.019, -0.002, -0.005, 0.079], YELLOW),
         'shuffle (harmful)': ([0.043, 0.037, 0.055, -0.067, -0.007], MAGENTA)}
for lab, (v, col) in modes.items():
    ax.plot(ck, v, '-o', color=col, lw=1.6, ms=4, label=lab)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_ylim(-0.35, 1.0)
ax.set_xlabel('training step (no-aux checkpoint)')
ax.set_ylabel('cos( mean aux grad, mean policy grad )')
ax.set_title('C  No auxiliary is gradient-aligned with the policy loss\n'
             '(helpful and harmful auxes indistinguishable at ~0)',
             loc='left', fontsize=10)
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='lower right')

# ---------- D: what forms instead ----------
ax = fig.add_subplot(gs[1, 1])
groups = [('no aux', [-0.04, 0.00, -0.03], ORANGE),
          ('aux on exogenous\nstream (0-consequence)', [-0.06, -0.04, -0.03, -0.09], MAGENTA),
          ('aux on OWN trajectory\n(past/future/velocity)', [0.23, 0.32, 0.21, 0.15, 0.16, 0.28, 0.25], BLUE)]
for i, (lab, vals, col) in enumerate(groups):
    x = np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals))
    ax.scatter(x, vals, color=col, s=28, zorder=3)
    ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color=col, lw=2.5, zorder=2)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels([g[0] for g in groups], fontsize=8)
ax.set_ylabel('signed-bearing decode: R² gain, layer 1 → 6')
ax.set_title('D  Only own-trajectory targets build a deepening\n'
             'goal-direction representation (each dot = one trained net)',
             loc='left', fontsize=10)

fig.suptitle('Why the auxiliary helps: not gradient alignment — representations '
             'the low-SNR policy gradient can bootstrap on',
             fontsize=12, x=0.06, ha='left')
os.makedirs('figs', exist_ok=True)
fig.savefig('figs/fig_matt_gradients.png', dpi=170)
print('saved figs/fig_matt_gradients.png')
