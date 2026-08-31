"""Horizon selection on the chosen world: how much time is 'just enough'?

For T in the grid, run informed / live / agnostic on the selected world
(c_o=.6, c_s=.35, q0=.9, d=2, running rho=8) and record per-round in-ball
probability curves plus the usual identity metrics. The informed curve's
plateau time = herding time from a uniform start; the live curve's plateau
= identification + herding; T is 'just enough' when the live curve barely
plateaus before the deadline. Writes results/horizon_v0.json + figs/horizon_v0.png.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World
from oracle import run_episodes

R = 4000
T_GRID = (16, 24, 32, 40, 48, 64)
PARAMS = dict(q0=0.9, c_other=0.6, c_self=0.35, d_goal=2, kappa=1.0,
              mode='running', rho=8.0)

res = {'R': R, 'params': PARAMS, 'horizons': {}}
curves = {}
for T in T_GRID:
    w = World(T=T, **PARAMS)
    row = {}
    for i, agent in enumerate(('informed', 'live', 'agnostic')):
        r = run_episodes(w, agent, R, seed=100 + i, collect=True)
        ball = r['traj']['ball'].mean(axis=0)
        row[agent] = {'occ': float(r['occ'].mean()),
                      'ball_curve': [round(float(x), 4) for x in ball],
                      'late_ball': float(ball[-8:].mean())}
        if agent == 'live':
            lo = r['traj']['signed_logodds']
            med = np.median(lo, axis=0)
            cr = np.nonzero(med > 2.0)[0]
            row['cross2'] = int(cr[0]) if len(cr) else -1
            row['final_correct'] = float((lo[:, -1] > 0).mean())
    row['G_occ'] = row['informed']['occ'] - row['agnostic']['occ']
    row['G_late'] = row['informed']['late_ball'] - row['agnostic']['late_ball']
    row['live_capture'] = ((row['live']['occ'] - row['agnostic']['occ'])
                           / max(row['G_occ'], 1e-9))
    res['horizons'][T] = row
    curves[T] = row
    print(f"T={T}: occ inf {row['informed']['occ']:.3f} live {row['live']['occ']:.3f} "
          f"agn {row['agnostic']['occ']:.3f} | late-8 ball inf "
          f"{row['informed']['late_ball']:.3f} live {row['live']['late_ball']:.3f} "
          f"agn {row['agnostic']['late_ball']:.3f} | G_occ {row['G_occ']:.3f} "
          f"capture {row['live_capture']:.2f} cross2 {row['cross2']} "
          f"fc {row['final_correct']:.3f}")

with open('results/horizon_v0.json', 'w') as f:
    json.dump(res, f, indent=1, default=float)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
w64 = curves[64]
for agent, c in (('informed', 'C0'), ('live', 'C1'), ('agnostic', 'C2')):
    axes[0].plot(w64[agent]['ball_curve'], c, lw=2, label=agent)
axes[0].set_xlabel('round t'); axes[0].set_ylabel('P(both chains in ball)')
axes[0].set_title('per-round success, T=64 (plateau times)')
axes[0].legend(fontsize=8)
Ts = list(T_GRID)
axes[1].plot(Ts, [curves[T]['G_occ'] for T in Ts], 'o-', label='G_occ')
axes[1].plot(Ts, [curves[T]['G_late'] for T in Ts], 's-', label='G on last 8 rounds')
axes[1].plot(Ts, [curves[T]['live_capture'] for T in Ts], '^-',
             label='live capture of premium')
axes[1].plot(Ts, [curves[T]['final_correct'] for T in Ts], 'v-',
             label='final correct side')
axes[1].set_xlabel('horizon T'); axes[1].set_title('what shortening costs')
axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig('figs/horizon_v0.png', dpi=160)
print('wrote results/horizon_v0.json, figs/horizon_v0.png')
