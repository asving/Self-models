"""Closed-loop attention ablations measuring REACH ACCURACY (reliable metric), to separate:
  H1 goal recomputed from sensing tokens each step; H2 inferred from trajectory; H3 stored+propagated."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
torch.set_num_threads(4)
import goalnav as G
from goalnav_dissect import load, ridge, ang
DEV = "cpu"


def make_mask(T, cut, mode):
    base = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)        # True = disallow (future)
    if mode == "clean" or cut == 0:
        return base
    q = torch.arange(T)[:, None]; k = torch.arange(T)[None]
    if mode == "no_sensing":                                        # later can't see [0,cut)
        return base | ((q >= cut) & (k < cut))
    if mode == "only_sensing":                                      # later see [0,cut)+self only
        return base | ((q >= cut) & (k >= cut) & (k != q))
    if mode == "freeze":                                            # later see [0,cut)+self+local(t-1)
        return base | ((q >= cut) & (k >= cut) & (k != q) & (k != q - 1))
    raise ValueError(mode)


def trunk_masked(net, obs, mask):
    T = obs.shape[1]
    x = net.in_proj(obs) + net.pos(torch.arange(T))[None]
    for blk in net.blocks:
        x = blk(x, mask)
    return net.lnf(x)


@torch.no_grad()
def rollout_ablated(net, a, mode, N=600):
    L = a["L"]; cut = a["cutoff"]; rng = np.random.default_rng(4)
    x = G.rand_unit(N, DEV, rng); gstar = G.rand_unit(N, DEV, rng)
    obs_list, xs = [], [x]
    for t in range(L):
        d = torch.arccos((x * gstar).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
        dvalid = 1.0 if (cut == 0 or t < cut) else 0.0
        obs_t = torch.cat([x, (d.unsqueeze(-1) * dvalid), torch.full((N, 1), dvalid)], -1)
        seq = torch.stack(obs_list + [obs_t], 1); T = seq.shape[1]
        h = trunk_masked(net, seq, make_mask(T, cut, mode))[:, -1]
        x = G.rotate(a["delta"] * torch.tanh(net.action_head(h)), x)
        obs_list.append(obs_t); xs.append(x)
    Xt = torch.stack(xs, 1)
    reach = G.torch.rad2deg(torch.arccos((Xt[:, -1] * gstar).sum(-1).clamp(-1, 1))).mean().item()
    # also goal decode at tail (probe trained on clean run separately would be ideal; quick proxy here)
    return reach


if __name__ == "__main__":
    for name in ["gn_cutoff5_6L", "gn_alwayson_6L"]:
        net, a = load(name)
        print(f"\n=== {name} (cutoff={a['cutoff']}) | REACH accuracy (deg) under attention ablation ===")
        for mode in ["clean", "no_sensing", "only_sensing", "freeze"]:
            r = rollout_ablated(net, a, mode)
            print(f"  {mode:13s}: reach = {r:5.1f} deg")
        print("  interpretation: no_sensing OK => trajectory/relay suffices (H2/H3); "
              "only_sensing OK => sensing-recompute suffices (H1); both OK => over-determined")
