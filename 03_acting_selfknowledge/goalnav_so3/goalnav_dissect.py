"""Dissection of goalnav nets: (1) goal decodable-in-advance plot, (2) sim-vs-nosim goal rep,
(3) cutoff plateau, (4) per-layer goal localization, (5) closed-loop goal RUBBER-HAND."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
torch.set_num_threads(4)
import goalnav as G

DEV = "cpu"


def load(name):
    ck = torch.load(os.path.expanduser(f"~/self-models/goalnav_runs/{name}.pt"), map_location=DEV)
    a = ck["args"]
    net = G.GoalNavNet(a["d_model"], a["n_layer"], a["n_head"], a["L"], a["r"])
    net.load_state_dict(ck["state"]); net.eval()
    return net, a


def trunk_layers(net, obs):                       # list of residuals [embed, blk0.., lnf] each (B,T,d)
    T = obs.shape[1]
    x = net.in_proj(obs) + net.pos(torch.arange(T))[None]
    hs = [x.detach().clone()]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    for blk in net.blocks:
        x = blk(x, mask); hs.append(x.detach().clone())
    hs.append(net.lnf(x).detach())
    return hs


def ridge(X, Y, lam=10.0):
    d = X.shape[1]
    W = torch.linalg.solve(X.T @ X + lam * torch.eye(d), X.T @ Y)
    return W


def ang(pred, tgt):
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.rad2deg(torch.arccos((pred * tgt).sum(-1).clamp(-1, 1)))


@torch.no_grad()
def decode_vs_step(net, a, N=2500):
    """PER-STEP probe: for each step t, train a ridge on that step's residual (train eps) -> x*, eval on
    held-out eps. The fair test of when the goal is linearly decodable. lnf layer."""
    X, obs, g = G.rollout(net, N, a["L"], DEV, np.random.default_rng(1), a["delta"], a["cutoff"])
    lnf = trunk_layers(net, obs)[-1]              # (N,L,d)
    L = a["L"]; ntr = int(0.7 * N)
    err = []
    for t in range(L):
        W = ridge(lnf[:ntr, t], g[:ntr])
        err.append(ang(lnf[ntr:, t] @ W, g[ntr:]).mean().item())
    dist = [G.torch.rad2deg(torch.arccos((X[ntr:, t] * g[ntr:]).sum(-1).clamp(-1, 1))).mean().item() for t in range(L)]
    return np.array(err), np.array(dist)


@torch.no_grad()
def sim_head_per_k(net, a, N=1024):
    X, obs, g = G.rollout(net, N, a["L"], DEV, np.random.default_rng(3), a["delta"], a["cutoff"])
    h = net.trunk(obs); pred = net.sim_head(h).view(N, a["L"], a["r"], 3)
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    out = []
    for k in range(1, a["r"] + 1):
        v = a["L"] - k
        e = ang(pred[:, :v, k - 1], X[:, k:k + v]).mean().item()
        out.append(e)
    return out


@torch.no_grad()
def rubber_hand(net, a, N=500, scale=1.0, randomize=False, inject_from=None):
    """closed-loop: from step `inject_from`, override the decoded goal at EVERY block output (current
    position) to a fixed false g'. Does the trajectory navigate to g' instead of the true goal?"""
    rng = np.random.default_rng(5); L = a["L"]
    if inject_from is None:
        inject_from = a["cutoff"] if a["cutoff"] > 0 else 0
    # per-block goal probes + min-norm inverses (from a normal rollout, tail positions)
    X0, obs0, g0 = G.rollout(net, 1500, L, DEV, np.random.default_rng(9), a["delta"], a["cutoff"])
    H0 = trunk_layers(net, obs0)
    Yt = g0[:, None].expand(-1, L - L // 2, -1).reshape(-1, 3)
    Ws, Winvs = [], []
    for h in H0[1:-1]:                            # block outputs only
        W = ridge(h[:, L // 2:].reshape(-1, h.shape[-1]), Yt)
        Ws.append(W); Winvs.append(W @ torch.linalg.inv(W.T @ W))

    x = G.rand_unit(N, DEV, rng); gstar = G.rand_unit(N, DEV, rng); gprime = G.rand_unit(N, DEV, rng)
    rdir = G.rand_unit(N, H0[1].shape[-1], rng) if False else None
    obs_list, xs = [], [x]
    for t in range(L):
        d = torch.arccos((x * gstar).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
        dvalid = 1.0 if (a["cutoff"] == 0 or t < a["cutoff"]) else 0.0
        obs_t = torch.cat([x, (d.unsqueeze(-1) * dvalid), torch.full((N, 1), dvalid)], -1)
        seq = torch.stack(obs_list + [obs_t], 1); T = seq.shape[1]
        xx = net.in_proj(seq) + net.pos(torch.arange(T))[None]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        for li, blk in enumerate(net.blocks):
            xx = blk(xx, mask)
            if t >= inject_from:
                r = xx[:, -1].clone(); W = Ws[li]; Winv = Winvs[li]
                if randomize:
                    rd = torch.randn(N, r.shape[-1]); rd = rd / rd.norm(dim=-1, keepdim=True)
                    delta = scale * (gprime - r @ W).norm(dim=-1, keepdim=True) * rd  # matched norm, random dir
                else:
                    delta = scale * ((gprime - r @ W) @ Winv.T)                       # set decoded goal -> g'
                xx = xx.clone(); xx[:, -1] = r + delta
        a_t = net.action_head(net.lnf(xx)[:, -1])
        x = G.rotate(a["delta"] * torch.tanh(a_t), x)
        obs_list.append(obs_t); xs.append(x)
    Xt = torch.stack(xs, 1)
    ft = G.torch.rad2deg(torch.arccos((Xt[:, -1] * gstar).sum(-1).clamp(-1, 1))).mean().item()
    ff = G.torch.rad2deg(torch.arccos((Xt[:, -1] * gprime).sum(-1).clamp(-1, 1))).mean().item()
    return ft, ff


if __name__ == "__main__":
    for name in ["gn_alwayson_6L", "gn_nosim_6L", "gn_cutoff5_6L", "gn_alwayson_2L"]:
        net, a = load(name)
        err, dist = decode_vs_step(net, a)
        onset = next((t for t in range(len(err)) if err[t] < 20), None)
        print(f"\n=== {name} (cutoff={a['cutoff']}, {a['n_layer']}L) | PER-STEP goal probe ===")
        print(" goal decode-err by step:", " ".join(f"{e:.0f}" for e in err[:24]))
        print(" dist-to-goal  by step:  ", " ".join(f"{d:.0f}" for d in dist[:24]))
        if a["cutoff"] > 0:
            print(f" decode-err AT cutoff step {a['cutoff']} = {err[a['cutoff']]:.0f} deg (dist there {dist[a['cutoff']]:.0f}); goal must be stored by here")
        print(f" goal decodable (<20deg) from step {onset}; dist there={dist[onset] if onset is not None else float('nan'):.0f}; final dist={dist[-1]:.0f}")
        if a["cutoff"] == 0 and a["n_layer"] == 6:
            print(" sim-head err per k=1..r:", " ".join(f"{e:.1f}" for e in sim_head_per_k(net, a)))

    for name in ["gn_cutoff5_6L", "gn_alwayson_6L"]:
        net, a = load(name)
        print(f"\n=== GOAL RUBBER-HAND ({name}, inject from step {a['cutoff'] if a['cutoff']>0 else 0}): navigate to FALSE goal? ===")
        for sc in [0.0, 1.0, 2.0]:
            ft, ff = rubber_hand(net, a, scale=sc)
            print(f" scale={sc}: final dist to TRUE={ft:.0f} deg, to FALSE={ff:.0f} deg  (redirect => FALSE drops, TRUE rises)")
        ftr, ffr = rubber_hand(net, a, scale=2.0, randomize=True)
        print(f" random-dir control (scale=2): TRUE={ftr:.0f}, FALSE={ffr:.0f}")
