"""Circuit-decode toolkit for the goalnav self-simulation result.

Consolidates the 2026-07-26 decode (previously inline): manual per-head trunk,
closed-loop ablations, content decodes, and the caveat-patch experiments.
Claims + numbers in CIRCUIT.md.

Subcommands:
  seeds   replicate the circuit signature across all sim/no-aux seeds:
          per-head ablation sweep (top-2 critical heads), L1-MLP summand decode,
          estimate-vs-truth bearing decode at t=8, depth profile of bearing at t=30
  equiv   interventional bilinearity test: rotate the believed bearing about x by
          phi via the polar write (rad unchanged -> observation-consistent);
          cross-product stage predicts realized velocity rotates by exactly phi
  weight  pin the estimator weighting: decode match of exponentially-recency-
          weighted LS variants (gamma sweep) against the residual
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goalnav import GoalNavNet, rollout, rotate, rand_unit  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RD = 'goalnav_runs'


def load(path):
    c = torch.load(path, map_location=DEV)
    a = c['args']
    net = GoalNavNet(a['d_model'], a['n_layer'], a['n_head'], a['L'],
                     a['r']).to(DEV)
    net.load_state_dict(c['state'])
    net.eval()
    return net, a


def manual_trunk(net, obs, a, ablate=None, record=False):
    """Faithful re-implementation of net.trunk with per-head access.
    ablate: set of ('h', layer_idx, head_idx) and/or ('m', layer_idx)."""
    H = a['n_head']
    DH = a['d_model'] // H
    T = obs.shape[1]
    x = net.in_proj(obs) + net.pos(torch.arange(T, device=DEV))[None]
    mask = torch.triu(torch.ones(T, T, device=DEV, dtype=torch.bool), 1)
    rec = []
    for j, blk in enumerate(net.blocks):
        hn = blk.ln1(x)
        qkv = hn @ blk.attn.in_proj_weight.T + blk.attn.in_proj_bias
        q, k, v = qkv.chunk(3, -1)
        B = q.shape[0]
        q = q.view(B, T, H, DH).transpose(1, 2)
        k = k.view(B, T, H, DH).transpose(1, 2)
        v = v.view(B, T, H, DH).transpose(1, 2)
        al = ((q @ k.transpose(-1, -2)) / DH**0.5
              ).masked_fill(mask, float('-inf')).softmax(-1)
        ctx = al @ v
        Wo = blk.attn.out_proj.weight
        contribs = [ctx[:, hh] @ Wo[:, hh*DH:(hh+1)*DH].T for hh in range(H)]
        keep = [c for hh, c in enumerate(contribs)
                if not (ablate and ('h', j, hh) in ablate)]
        x = x + sum(keep) + blk.attn.out_proj.bias
        m = blk.mlp(blk.ln2(x))
        if ablate and ('m', j) in ablate:
            m = 0 * m
        x = x + m
        if record:
            rec.append(dict(al=al, contribs=contribs, mlp=m, resid=x))
    return net.lnf(x), rec


@torch.no_grad()
def reach_abl(net, a, ablate=None, B=256, seed=11, edit=None):
    """Closed-loop rollout with optional ablation / residual edit; returns
    final-quarter mean distance to goal (deg). edit(x_resid, layer, obs) hook
    unused here (kept simple: ablations only)."""
    rng = np.random.default_rng(seed)
    x = rand_unit(B, DEV, rng)
    gstar = rand_unit(B, DEV, rng)
    obs_list, xs = [], [x]
    for t in range(a['L']):
        d = torch.arccos((x * gstar).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
        dv = 1.0 if (a['cutoff'] == 0 or t < a['cutoff']) else 0.0
        obs_t = torch.cat([x, d.unsqueeze(-1) * dv,
                           torch.full((B, 1), dv, device=DEV)], -1)
        h, _ = manual_trunk(net, torch.stack(obs_list + [obs_t], 1), a, ablate)
        x = rotate(a['delta'] * torch.tanh(net.action_head(h[:, -1])), x)
        obs_list.append(obs_t)
        xs.append(x)
    X = torch.stack(xs, 1)
    dist = torch.rad2deg(torch.arccos(
        (X[:, 1:] * gstar[:, None]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)))
    return dist[:, 3 * a['L'] // 4:].mean().item()


def ridge_dec(F, Y, frac=0.85, l2=10.0):
    ntr = int(len(F) * frac)
    mf, my = F[:ntr].mean(0), Y[:ntr].mean(0)
    W = np.linalg.solve((F[:ntr]-mf).T @ (F[:ntr]-mf) + l2*np.eye(F.shape[1]),
                        (F[:ntr]-mf).T @ (Y[:ntr]-my))
    P = (F[ntr:]-mf) @ W + my
    return 1 - ((Y[ntr:]-P)**2).sum() / ((Y[ntr:]-Y[ntr:].mean(0))**2).sum()


def gls_estimate(X, obs, t, gamma=1.0, eps=0.05):
    """(Recency-weighted) least-squares goal estimate from history s<=t."""
    xt = X[:, :t+1]
    w = gamma ** torch.arange(t, -1, -1, device=DEV, dtype=torch.float32)
    M = (w[None, :, None, None] * xt.unsqueeze(-1) * xt.unsqueeze(-2)).sum(1) \
        + eps * torch.eye(3, device=DEV)
    v = (w[None, :, None] * torch.cos(obs[:, :t+1, 3]).unsqueeze(-1) * xt).sum(1)
    g = torch.linalg.solve(M, v.unsqueeze(-1)).squeeze(-1)
    return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def cmd_seeds(args):
    sims = ['gn_alwayson_6L', 'gn_rep_lam1.0_s1', 'gn_rep_lam1.0_s2',
            'gn_rep_lam1.0_s3']
    noas = ['gn_nosim_6L', 'gn_rep_lam0.0_s1', 'gn_rep_lam0.0_s2',
            'gn_rep_lam0.0_s3']
    for arm, runs in (('SIM', sims), ('NOAUX', noas)):
        print(f'== {arm} seeds: circuit signature ==')
        print(f"{'run':>18} {'reach':>6} | top-2 critical heads (delta) | "
              f"{'summand':>7} {'estBrg':>6} {'truBrg':>6} | brg t30 L1->L6")
        for r in runs:
            net, a = load(f'{RD}/{r}.pt')
            with torch.no_grad():
                X, obs, g = rollout(net, 2048, a['L'], DEV,
                                    np.random.default_rng(7), a['delta'],
                                    a['cutoff'])
                _, rec = manual_trunk(net, obs, a, record=True)
            base = reach_abl(net, a)
            deltas = {}
            for j in range(a['n_layer']):
                for hh in range(a['n_head']):
                    deltas[(j, hh)] = reach_abl(net, a, {('h', j, hh)}) - base
            top = sorted(deltas.items(), key=lambda kv: -kv[1])[:2]
            tops = '  '.join(f'L{j+1}h{hh} {d:+5.1f}' for (j, hh), d in top)
            ss = list(range(1, 13))
            x_s = X[:, ss].cpu().numpy().reshape(-1, 3)
            cdx = torch.cos(obs[:, ss, 3]).cpu().numpy().reshape(-1, 1) * x_s
            summand = ridge_dec(
                rec[0]['mlp'][:, ss].cpu().numpy().reshape(-1, a['d_model']),
                cdx)
            t = 8
            gLS = gls_estimate(X, obs, t)
            x_t = X[:, t]
            tangT = (g - (g*x_t).sum(-1, keepdim=True)*x_t).cpu().numpy()
            tangE = (gLS - (gLS*x_t).sum(-1, keepdim=True)*x_t).cpu().numpy()
            F4 = rec[3]['resid'][:, t].cpu().numpy()
            eb, tb = ridge_dec(F4, tangE), ridge_dec(F4, tangT)
            t2 = 30
            x_t2 = X[:, t2]
            tang30 = (g - (g*x_t2).sum(-1, keepdim=True)*x_t2).cpu().numpy()
            b1 = ridge_dec(rec[0]['resid'][:, t2].cpu().numpy(), tang30)
            b6 = ridge_dec(rec[5]['resid'][:, t2].cpu().numpy(), tang30)
            print(f'{r:>18} {base:6.1f} | {tops:>28} | {summand:7.2f} '
                  f'{eb:6.2f} {tb:6.2f} | {b1:.2f} -> {b6:.2f}')
        print()


def cmd_equiv(args):
    net, a = load(f'{RD}/{args.net}.pt')
    LAYERS = [2, 3, 4, 5]
    with torch.no_grad():
        X, obs, g = rollout(net, 4096, a['L'], DEV, np.random.default_rng(7),
                            a['delta'], a['cutoff'])
        _, rec = manual_trunk(net, obs, a, record=True)
    L = a['L']
    x_all = X[:, :L]
    rad = (g[:, None] * x_all).sum(-1, keepdim=True)
    tang = g[:, None] - rad * x_all
    PHI = torch.cat([rad, tang], -1).reshape(-1, 4).cpu().numpy()
    mphi = PHI.mean(0)
    Wp = {}
    for li in LAYERS:
        Hm = rec[li-1]['resid'].reshape(-1, a['d_model']).cpu().numpy()
        mh = Hm.mean(0)
        W = np.linalg.solve((PHI-mphi).T @ (PHI-mphi) + 1e-3*np.eye(4),
                            (PHI-mphi).T @ (Hm-mh))
        Wp[li] = torch.tensor(W, device=DEV, dtype=torch.float32)

    def edited_omega(phi_deg, t=20, B=1024):
        ob = obs[:B, :t+1]
        xs = X[:B, :t+1]
        gs = g[:B]
        r1 = (gs[:, None] * xs).sum(-1, keepdim=True)
        t1 = gs[:, None] - r1 * xs
        c, s = np.cos(np.deg2rad(phi_deg)), np.sin(np.deg2rad(phi_deg))
        t2 = c * t1 + s * torch.cross(xs, t1, dim=-1)   # rotate tang about x
        dphi = torch.cat([torch.zeros_like(r1), t2 - t1], -1)
        T = t + 1
        x = net.in_proj(ob) + net.pos(torch.arange(T, device=DEV))[None]
        mask = torch.triu(torch.ones(T, T, device=DEV, dtype=torch.bool), 1)
        for j, blk in enumerate(net.blocks):
            hn = blk.ln1(x)
            av, _ = blk.attn(hn, hn, hn, attn_mask=mask, need_weights=False)
            x = x + av
            x = x + blk.mlp(blk.ln2(x))
            if (j + 1) in Wp and phi_deg is not None:
                x = x + dphi @ Wp[j + 1]
        h = net.lnf(x)[:, -1]
        w = a['delta'] * torch.tanh(net.action_head(h))
        return torch.cross(w, X[:B, t], dim=-1), X[:B, t]  # velocity, position

    v0, xt = edited_omega(0.0)
    print(f'== equivariance: rotate believed bearing about x by phi '
          f'({args.net}, t=20, edit L{LAYERS}) ==')
    print(f"{'phi':>5} | {'measured rotation of velocity':>29} | "
          f"{'|v|/|v0|':>8}")
    for phi in (30, 60, 90, 120, 150, 180):
        v, _ = edited_omega(float(phi))
        cosang = (v * v0).sum(-1) / (v.norm(dim=-1) * v0.norm(dim=-1) + 1e-9)
        sinang = (torch.cross(v0, v, dim=-1) * xt).sum(-1) / (
            v.norm(dim=-1) * v0.norm(dim=-1) + 1e-9)
        ang = torch.rad2deg(torch.atan2(sinang, cosang))
        ratio = (v.norm(dim=-1) / v0.norm(dim=-1).clamp_min(1e-9))
        print(f'{phi:5d} | {ang.mean().item():14.1f} +- '
              f'{ang.std().item():4.1f} deg | {ratio.mean().item():8.2f}')


def cmd_weight(args):
    net, a = load(f'{RD}/{args.net}.pt')
    with torch.no_grad():
        X, obs, g = rollout(net, 4096, a['L'], DEV, np.random.default_rng(7),
                            a['delta'], a['cutoff'])
        _, rec = manual_trunk(net, obs, a, record=True)
    print(f'== estimator weighting: decode R2 of gamma-recency-LS bearing '
          f'from resid L4 ({args.net}) ==')
    print(f"{'gamma':>6} | {'t=8':>5} {'t=20':>5} {'t=36':>5}")
    for gam in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
        row = []
        for t in (8, 20, 36):
            gE = gls_estimate(X, obs, t, gamma=gam)
            x_t = X[:, t]
            tangE = (gE - (gE*x_t).sum(-1, keepdim=True)*x_t).cpu().numpy()
            row.append(ridge_dec(rec[3]['resid'][:, t].cpu().numpy(), tangE))
        print(f'{gam:6.2f} | ' + ' '.join(f'{r:5.2f}' for r in row))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['seeds', 'equiv', 'weight'])
    ap.add_argument('--net', default='gn_rep_lam1.0_s1')
    args = ap.parse_args()
    dict(seeds=cmd_seeds, equiv=cmd_equiv, weight=cmd_weight)[args.cmd](args)
