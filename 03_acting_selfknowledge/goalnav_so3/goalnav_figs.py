"""Figure set for the goalnav self-simulation story (fig_goalnav_story.png).

A  reach vs training step, 4 seeds/arm (self-sim vs no-aux)
B  lambda dose-response + shuffle (content-free) control
C  legibility heatmaps: goal-probe R^2, layer x training step, both arms
D  sim-head vs great-circle extrapolation error by position (the stop)
Palette: validated reference instance (dataviz skill); light surface.
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goalnav import GoalNavNet, rollout  # noqa: E402
from goalnav_timeline import trunk_layers, ridge_r2  # noqa: E402
from so3agent import so3_exp  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RD = 'goalnav_runs'

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e5e4e0', '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('seqblue', ['#f3f7fc', '#123f78'])

plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
})


def curve(f):
    log = json.load(open(f'{RD}/{f}.json'))['log']
    return (np.array([l['step'] for l in log]),
            np.array([l['reach_deg'] for l in log]))


def final(f, k=3):
    log = json.load(open(f'{RD}/{f}.json'))['log']
    return float(np.mean([l['reach_deg'] for l in log[-k:]]))


@torch.no_grad()
def heat_data(steps_dir, t_probe=4):
    """Egocentric goal coords at position t_probe, per checkpoint:
    tangent (bearing) R^2 per layer, radial (distance) R^2 best layer."""
    files = sorted(glob.glob(f'{RD}/{steps_dir}/step_*.pt'))
    steps, mat, radials = [], [], []
    for f in files:
        ck = torch.load(f, map_location=DEV)
        a = ck['args']
        net = GoalNavNet(a['d_model'], a['n_layer'], a['n_head'], a['L'],
                         a['r']).to(DEV)
        net.load_state_dict(ck['state'])
        net.eval()
        X, obs, g = rollout(net, 512, a['L'], DEV, np.random.default_rng(7),
                            a['delta'], a['cutoff'])
        hs, _ = trunk_layers(net, obs)
        G = g.cpu().numpy()
        x_t = X[:, t_probe].cpu().numpy()
        rad = (G * x_t).sum(-1, keepdims=True)
        tang = G - rad * x_t
        Hs = [h[:, t_probe].cpu().numpy() for h in hs]
        mat.append([ridge_r2(H, tang, 350) for H in Hs])
        radials.append(max(ridge_r2(H, rad, 350) for H in Hs))
        steps.append(ck['step'])
    return np.array(steps), np.array(mat).T, np.array(radials)


@torch.no_grad()
def stop_data(ckpt, N=4096):
    ck = torch.load(ckpt, map_location=DEV)
    a = ck['args']
    net = GoalNavNet(a['d_model'], a['n_layer'], a['n_head'], a['L'],
                     a['r']).to(DEV)
    net.load_state_dict(ck['state'])
    net.eval()
    X, obs, g = rollout(net, N, a['L'], DEV, np.random.default_rng(7),
                        a['delta'], a['cutoff'])
    _, hf = trunk_layers(net, obs)
    r = a['r']
    pred = net.sim_head(hf).view(N, a['L'], r, 3)
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ts, se, xe = [], [], []
    for t in range(1, a['L'] - r):
        tgt = X[:, t + r]
        p = pred[:, t, r - 1]
        se.append(torch.rad2deg(torch.arccos(
            (p * tgt).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))).mean().item())
        u, v = X[:, t - 1], X[:, t]
        ax = torch.cross(u, v, dim=-1)
        ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ang = torch.arccos((u * v).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
        xp = (so3_exp(ax * (r * ang).unsqueeze(-1)) @ v.unsqueeze(-1)
              ).squeeze(-1)
        xp = xp / xp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        xe.append(torch.rad2deg(torch.arccos(
            (xp * tgt).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))).mean().item())
        ts.append(t)
    return np.array(ts), np.array(se), np.array(xe)


def main():
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.34,
                          left=0.055, right=0.985, top=0.92, bottom=0.09)

    # ---------------- A: reach curves ----------------
    ax = fig.add_subplot(gs[0, 0])
    arms = [('1.0', 'gn_alwayson_6L', BLUE, 'self-sim  (λ=1)'),
            ('0.0', 'gn_nosim_6L', ORANGE, 'no aux  (λ=0)')]
    for lam, orig, col, lab in arms:
        runs = [orig] + [f'gn_rep_lam{lam}_s{s}' for s in (1, 2, 3)]
        cs = [curve(r) for r in runs]
        grid = cs[0][0]
        ys = np.stack([np.interp(grid, s, y) for s, y in cs])
        for y in ys:
            ax.plot(grid, y, color=col, lw=0.7, alpha=0.3, zorder=1)
        ax.plot(grid, ys.mean(0), color=col, lw=2.2, zorder=3, label=lab)
        ax.annotate(lab, (grid[-1], ys.mean(0)[-1]), xytext=(4, 0),
                    textcoords='offset points', color=INK, fontsize=8,
                    va='center')
    ax.set_xlim(0, 12800)
    ax.set_xlabel('training step')
    ax.set_ylabel('distance to goal, final quarter (deg)')
    ax.set_title('A  Self-simulation rescues navigation (4 seeds/arm)',
                 loc='left', fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc='upper right')

    # ---------------- B: dose-response + shuffle ----------------
    ax = fig.add_subplot(gs[0, 1])
    lams = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    names = ['gn_lam0.03_s1', 'gn_lam0.1_s1', 'gn_lam0.3_s1',
             'gn_rep_lam1.0_s1', 'gn_lam3.0_s1', 'gn_lam10.0_s1']
    xs = np.log10(lams)
    ys = [final(n) for n in names]
    x0 = np.log10(0.03) - 0.9                     # slot for lambda = 0
    y0 = final('gn_rep_lam0.0_s1')
    ax.axhline(y0, color=INK2, lw=1, ls=(0, (4, 3)), alpha=0.7)
    ax.plot([x0], [y0], 'o', color=ORANGE, ms=7, zorder=3)
    ax.plot(xs, ys, '-o', color=BLUE, lw=2, ms=6, zorder=3)
    ysh = final('gn_shuffle_s1')
    ax.set_ylim(11, 42)
    ax.plot([0], [ysh], 's', color=AQUA, ms=8, zorder=4)
    ax.annotate('content-free targets\n(shuffled episodes)', (0, ysh),
                xytext=(6, -22), textcoords='offset points', fontsize=8,
                color=INK)
    ax.annotate('no-aux level', (xs[-1], y0), xytext=(0, 4), ha='right',
                textcoords='offset points', fontsize=7.5, color=INK2)
    ax.set_xticks([x0] + list(xs))
    ax.set_xticklabels(['0', '.03', '.1', '.3', '1', '3', '10'])
    ax.set_xlabel('sim-loss weight λ')
    ax.set_ylabel('final distance to goal (deg)')
    ax.set_title('B  Dose–response + controls',
                 loc='left', fontsize=10, color=INK)

    # ---------------- D: stop prediction ----------------
    axd = fig.add_subplot(gs[0, 2])
    ck = sorted(glob.glob(f'{RD}/gn_rep_lam1.0_s1_steps/step_*.pt'))[-1]
    ts, se, xe = stop_data(ck)
    axd.plot(ts, xe, color=INK2, lw=1.8, ls=(0, (4, 2)),
             label='extrapolate current heading')
    axd.plot(ts, se, color=BLUE, lw=2.2, label='sim head (4-step)')
    axd.fill_between(ts, se, xe, where=xe > se, color=BLUE, alpha=0.10,
                     lw=0)
    axd.annotate('policy orbits the goal —\nextrapolation flies off tangent',
                 (ts[-6], xe[-6]), xytext=(-8, 14),
                 textcoords='offset points', fontsize=8, color=INK,
                 ha='right')
    axd.set_xlabel('position in episode')
    axd.set_ylabel('4-step prediction error (deg)')
    axd.set_title('D  The sim head is not extrapolation',
                  loc='left', fontsize=10, color=INK)
    axd.legend(frameon=False, fontsize=8, loc='upper left')

    # ---------------- C: bearing-legibility heatmaps ----------------
    # egocentric goal coords: g = (g·x)x + tang.  The bearing tang is the
    # computed quantity (chance = R² 0); the radial part is the input.
    s1, m1, rad1 = heat_data('gn_rep_lam1.0_s1_steps')
    s0, m0, rad0 = heat_data('gn_rep_lam0.0_s1_steps')
    vmax = max(m1.max(), m0.max())
    for j, (st, m, lab) in enumerate(
            ((s1, m1, 'self-sim  (λ=1)'), (s0, m0, 'no aux  (λ=0)'))):
        ax = fig.add_subplot(gs[1, j])
        pc = ax.pcolormesh(np.append(st - 250, st[-1] + 250),
                           np.arange(0.5, m.shape[0] + 1),
                           m, cmap=SEQ, vmin=0, vmax=vmax,
                           edgecolors=SURF, linewidth=0.6)
        ax.set_yticks(range(1, m.shape[0] + 1))
        ax.set_ylabel('layer' if j == 0 else '')
        ax.set_xlabel('training step')
        ax.set_title(('C  Bearing legibility: probe R²(tang), layer × training'
                      if j == 0 else ' ') + f'\n{lab}',
                     loc='left', fontsize=10 if j == 0 else 9, color=INK)
        ax.grid(False)
        if j == 1:
            cb = fig.colorbar(pc, ax=ax, fraction=0.05, pad=0.03)
            cb.set_label('R² (bearing tang = g−(g·x)x, position 4)',
                         fontsize=8)
            cb.outline.set_visible(False)

    # ---------------- E: egocentric coords over training ----------------
    ax = fig.add_subplot(gs[1, 2])
    for st, rad in ((s1, rad1), (s0, rad0)):
        ax.plot(st, rad, color=INK2, lw=1.2, ls=(0, (2, 2)), alpha=0.7)
    for st, m, col, lab in ((s1, m1, BLUE, 'self-sim'),
                            (s0, m0, ORANGE, 'no aux')):
        ax.plot(st, m.max(0), '-o', color=col, lw=2, ms=3.5,
                label=f'bearing, {lab}')
    ax.annotate('radial g·x (distance) ≈ 1:\nre-read from the input, both arms',
                (s1[len(s1)//2], 1.0), xytext=(0, -26),
                textcoords='offset points', fontsize=7.5, color=INK2,
                ha='center')
    ax.set_ylim(0, 1.08)
    ax.set_xlabel('training step')
    ax.set_ylabel('best-layer R² at position 4')
    ax.set_title('E  Egocentric goal coords over training',
                 loc='left', fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc='center right')

    fig.suptitle('Forward self-simulation makes the goal BEARING linearly '
                 'legible — and that is what rescues policy learning',
                 fontsize=12, color=INK, x=0.055, ha='left')
    os.makedirs('figs', exist_ok=True)
    out = 'figs/fig_goalnav_story.png'
    fig.savefig(out, dpi=170)
    print('saved', out)


if __name__ == '__main__':
    main()
