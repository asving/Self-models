"""Training-time mechanism analysis for the goalnav self-simulation result.

Per checkpoint (sim arm vs nosim arm):
  reach       final-quarter mean distance to goal (deg)
  goalR2@t    ridge R^2 decoding the hidden goal g* from trunk residuals at
              position t (best layer of 6) -- does goal representation LEAD
              policy improvement, and does it form at all without sim loss?
  sim vs xtr  sim-head angular error at horizon r vs a great-circle
              EXTRAPOLATION baseline (continue current heading) -- if the sim
              head merely extrapolates, errors match; goal-aware simulation
              beats extrapolation exactly where the path must TURN (early
              positions, pre/mid-triangulation).
Usage: python goalnav_timeline.py goalnav_runs/gn_alwayson_6L_steps [...]
"""
from __future__ import annotations
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goalnav import GoalNavNet, rollout, rotate  # noqa: E402
from so3agent import so3_exp  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def trunk_layers(net, obs):
    T = obs.shape[1]
    x = net.in_proj(obs) + net.pos(torch.arange(T, device=obs.device))[None]
    mask = torch.triu(torch.ones(T, T, device=obs.device,
                                 dtype=torch.bool), 1)
    hs = []
    for blk in net.blocks:
        x = blk(x, mask)
        hs.append(x)
    return hs, net.lnf(x)


def ridge_r2(H, Y, ntr, l2=10.0):
    mh, my = H[:ntr].mean(0), Y[:ntr].mean(0)
    W = np.linalg.solve((H[:ntr]-mh).T @ (H[:ntr]-mh)
                        + l2*np.eye(H.shape[1]), (H[:ntr]-mh).T @ (Y[:ntr]-my))
    P = (H[ntr:]-mh) @ W + my
    return 1 - ((Y[ntr:]-P)**2).sum() / ((Y[ntr:]-Y[ntr:].mean(0))**2).sum()


@torch.no_grad()
def analyze_ckpt(path):
    ck = torch.load(path, map_location=DEV)
    a = ck['args']
    net = GoalNavNet(a['d_model'], a['n_layer'], a['n_head'], a['L'],
                     a['r']).to(DEV)
    net.load_state_dict(ck['state'])
    net.eval()
    X, obs, gstar = rollout(net, 512, a['L'], DEV,
                            np.random.default_rng(7), a['delta'], a['cutoff'])
    dist = torch.arccos((X[:, 1:] * gstar[:, None]).sum(-1)
                        .clamp(-1+1e-6, 1-1e-6))
    reach = torch.rad2deg(dist[:, 3*a['L']//4:].mean()).item()
    hs, hf = trunk_layers(net, obs)
    G = gstar.cpu().numpy()
    ntr = 350
    goal_r2 = {}
    for t in (2, 4, 12, 30):
        best = -9
        for h in hs:
            best = max(best, ridge_r2(h[:, t].cpu().numpy(), G, ntr))
        goal_r2[t] = best
    # sim head vs great-circle extrapolation, horizon r, by position bucket
    r = a['r']
    pred = net.sim_head(hf).view(512, a['L'], r, 3)
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    out = {}
    for name, ts in (('early', range(1, 7)), ('late', range(20, 36))):
        se, xe = [], []
        for t in ts:
            tgt = X[:, t + r]                       # true x_{t+r}
            p = pred[:, t, r - 1]
            se.append(torch.rad2deg(torch.arccos(
                (p*tgt).sum(-1).clamp(-1+1e-6, 1-1e-6))).mean().item())
            # extrapolate current heading: rotation x_{t-1}->x_t applied r x
            u, v = X[:, t - 1], X[:, t]
            axis = torch.cross(u, v, dim=-1)
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            ang = torch.arccos((u*v).sum(-1).clamp(-1+1e-6, 1-1e-6))
            w = axis * (r * ang).unsqueeze(-1)
            xp = (so3_exp(w) @ v.unsqueeze(-1)).squeeze(-1)
            xp = xp / xp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            xe.append(torch.rad2deg(torch.arccos(
                (xp*tgt).sum(-1).clamp(-1+1e-6, 1-1e-6))).mean().item())
        out[name] = (float(np.mean(se)), float(np.mean(xe)))
    return ck['step'], reach, goal_r2, out


def main():
    for d in sys.argv[1:]:
        print(f'==== {d}')
        print(f"{'step':>6} {'reach':>6} | goalR2 t2/t4/t12/t30 | "
              f"simERR/xtrERR early | late")
        for f in sorted(glob.glob(os.path.join(d, 'step_*.pt'))):
            step, reach, g, sx = analyze_ckpt(f)
            print(f"{step:6d} {reach:6.1f} | "
                  f"{g[2]:.2f}/{g[4]:.2f}/{g[12]:.2f}/{g[30]:.2f} | "
                  f"{sx['early'][0]:5.1f}/{sx['early'][1]:5.1f} | "
                  f"{sx['late'][0]:5.1f}/{sx['late'][1]:5.1f}")


if __name__ == '__main__':
    main()
