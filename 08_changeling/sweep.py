"""World-selection sweep. Usage: sweep.py [running|terminal]

v0.1 (terminal reward): forgiving at d<=2 but identity collapse lands at
t ~ 42-63/64 — the urgency-gating law (terminal reward => deviation, hence
identity evidence, concentrates at the deadline). Results kept in
results/sweep_v0.1_terminal.json.
v0.2 (running reward, exponential tilt rho, kappa=1): tilt active from round
0, so deliberation should move early/mid-episode. Primary metric: mean tol-1
ball occupancy 'occ'; terminal ball success kept as secondary.

Per cell: informed / agnostic / live agents (R episodes each); per world:
closed-form base value, embodied-zero-tilt run, delusion-gap curves. Writes
results/sweep_<tag>.json, winner trajectories to results/winner_<tag>.npz, figs/.
Run with cwd = 08_changeling.
"""
import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World, Q0_GRID, COUPLING_GRID, KAPPA_GRID, RHO_GRID, DIST_GRID
from oracle import run_episodes, run_base

R = 3000
SEED = 0


def make_world(mode, tilt, **kw):
    if mode == 'running':
        return World(kappa=1.0, mode='running', rho=tilt, **kw)
    return World(kappa=tilt, mode='terminal', **kw)


def cell_metrics(w, seed):
    out = {}
    runs = {}
    for i, agent in enumerate(('informed', 'agnostic', 'live')):
        r = run_episodes(w, agent, R, seed + i, collect=(agent == 'live'))
        runs[agent] = r
        out[f'S_{agent}'] = float(r['exact'].mean())
        out[f'S_{agent}_tol1'] = float(r['tol1'].mean())
        out[f'occ_{agent}'] = float(r['occ'].mean())
    out['G_tol1'] = out['S_informed_tol1'] - out['S_agnostic_tol1']
    out['G_occ'] = out['occ_informed'] - out['occ_agnostic']
    out['regret_occ'] = out['occ_informed'] - out['occ_live']
    lo = runs['live']['traj']['signed_logodds']          # (R, T)
    med = np.median(lo, axis=0)
    out['median_curve'] = [round(float(x), 4) for x in med]
    out['evidence_rate'] = [round(float(x), 5) for x in
                            runs['live']['traj']['signed_dlog'].mean(axis=0)]
    cross = np.nonzero(med > 2.0)[0]
    out['cross2'] = int(cross[0]) if len(cross) else -1
    k = out['cross2'] if out['cross2'] > 3 else len(med)
    seg = med[:max(k, 6)]
    d2 = np.diff(seg, 2)
    out['convex_frac'] = float((d2 > 0).mean()) if len(d2) else float('nan')
    out['final_correct'] = float((lo[:, -1] > 0).mean())
    wrong = lo.min(axis=1) < -1.0
    out['wrong_side_frac'] = float(wrong.mean())
    out['recovered_frac'] = float((lo[wrong, -1] > 0).mean()) if wrong.any() else float('nan')
    return out, lo


def eligible(m, mode):
    if m['c_self'] == 0:
        return False
    if mode == 'running':
        # comfort = high informed occupancy (terminal-ball rate is the wrong
        # comfort measure here: the running-reward tilt tapers at the deadline
        # by design, so nothing pins the agent to the ball at exactly T)
        return (m['occ_informed'] >= 0.5 and m['G_occ'] >= 0.10
                and 4 <= m['cross2'] <= 24 and m['final_correct'] >= 0.9)
    return m['S_informed_tol1'] >= 0.75 and 6 <= m['cross2'] <= 28


def main(mode):
    tag = {'running': 'v0.2_running', 'terminal': 'v0.1_terminal'}[mode]
    tilt_grid = RHO_GRID if mode == 'running' else KAPPA_GRID
    gscore = 'G_occ' if mode == 'running' else 'G_tol1'
    t0 = time.time()
    cells, world_info = [], {}
    best = None
    for (c_o, c_s) in COUPLING_GRID:
        for q0 in Q0_GRID:
            for d in DIST_GRID:
                wkey = f'co{c_o}_cs{c_s}_q{q0}_d{d}'
                kw = dict(q0=q0, c_other=c_o, c_self=c_s, d_goal=d)
                w0 = make_world(mode, 0.0, **kw)
                zero = run_episodes(w0, 'zero', R, SEED + 7, collect=True)
                world_info[wkey] = {
                    'base_value_theory': float(w0.h[0].mean()),
                    'occ_base': float(run_base(w0, R, SEED + 8)['occ'].mean()),
                    'occ_zero_embodied': float(zero['occ'].mean()),
                    'tv_delusion_median': [round(float(x), 4) for x in
                                           np.median(zero['traj']['tv_self'], axis=0)],
                }
                for tilt in tilt_grid:
                    w = make_world(mode, tilt, **kw)
                    m, lo = cell_metrics(w, SEED)
                    m.update(c_other=c_o, c_self=c_s, q0=q0, d=d, tilt=tilt,
                             mode=mode, world=wkey)
                    m['eligible'] = bool(eligible(m, mode))
                    cells.append(m)
                    if m['eligible'] and (best is None or m[gscore] > best[0][gscore]):
                        best = (m, lo, w)
                    print(f"{wkey} tilt{tilt}: occ_inf {m['occ_informed']:.3f} "
                          f"occ_agn {m['occ_agnostic']:.3f} occ_live {m['occ_live']:.3f} "
                          f"G_occ {m['G_occ']:.3f} G_tol1 {m['G_tol1']:.3f} "
                          f"cross2 {m['cross2']} cvx {m['convex_frac']:.2f} "
                          f"elig {m['eligible']}")

    res = {'R': R, 'seed': SEED, 'mode': mode, 'cells': cells,
           'worlds': world_info, 'elapsed_s': round(time.time() - t0, 1)}
    if best is not None:
        m, lo, w = best
        res['winner'] = dict(m)
        np.savez(f'results/winner_{tag}.npz', signed_logodds=lo,
                 params=np.array([w.q0, w.c_other, w.c_self, w.d_goal,
                                  w.kappa, w.rho]))
        make_figs(res, m, lo, world_info[m['world']], tilt_grid, tag)
    with open(f'results/sweep_{tag}.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done in {res['elapsed_s']}s; winner: "
          f"{res.get('winner', {}).get('world', 'NONE')} "
          f"tilt {res.get('winner', {}).get('tilt')}")


def make_figs(res, m, lo, winfo, tilt_grid, tag):
    cells = res['cells']
    fig, axes = plt.subplots(1, len(tilt_grid),
                             figsize=(4.2 * len(tilt_grid), 3.4), sharey=True)
    for ax, tilt in zip(np.atleast_1d(axes), tilt_grid):
        for (c_o, c_s) in COUPLING_GRID:
            for q0 in Q0_GRID:
                ys = [c['G_occ'] for c in cells if c['tilt'] == tilt
                      and c['c_other'] == c_o and c['c_self'] == c_s
                      and c['q0'] == q0]
                ax.plot(DIST_GRID, ys, marker='o',
                        label=f'c_o={c_o}, c_s={c_s}, q0={q0}')
        ax.set_title(f'identity premium G_occ, tilt={tilt}')
        ax.set_xlabel('goal distance d'); ax.axhline(0, color='k', lw=.5)
    np.atleast_1d(axes)[0].set_ylabel('G_occ = occ_informed - occ_agnostic')
    np.atleast_1d(axes)[-1].legend(fontsize=6, loc='upper right')
    fig.tight_layout(); fig.savefig(f'figs/premium_{tag}.png', dpi=160)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    t = np.arange(lo.shape[1])
    q25, q75 = np.percentile(lo, [25, 75], axis=0)
    axes[0].fill_between(t, q25, q75, alpha=.25, label='IQR')
    axes[0].plot(t, np.median(lo, axis=0), lw=2, label='median')
    for i in range(8):
        axes[0].plot(t, lo[i], lw=.6, alpha=.6)
    axes[0].axhline(0, color='k', lw=.5)
    axes[0].set_xlabel('round t')
    axes[0].set_ylabel('signed identity log-odds (nats)')
    axes[0].set_title(f"live collapse — {m['world']} tilt={m['tilt']}\n"
                      f"occ: inf {m['occ_informed']:.2f} agn {m['occ_agnostic']:.2f} "
                      f"live {m['occ_live']:.2f}  G_occ {m['G_occ']:.2f}")
    axes[0].legend(fontsize=7)
    axes[1].plot(m['evidence_rate'], lw=2)
    axes[1].set_xlabel('round t')
    axes[1].set_ylabel('mean identity evidence rate (nats/round)')
    axes[1].set_title('when selfhood is learnable (deviation schedule)')
    fig.tight_layout(); fig.savefig(f'figs/winner_collapse_{tag}.png', dpi=160)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(winfo['tv_delusion_median'], lw=2)
    ax.set_xlabel('round t')
    ax.set_ylabel('median TV(evidence belief, dead-reckoned)')
    ax.set_title('delusion gap under embodiment, zero tilt (record stays lawful)')
    fig.tight_layout(); fig.savefig(f'figs/delusion_gap_{tag}.png', dpi=160)
    print(f'figs written for {tag}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'running')
