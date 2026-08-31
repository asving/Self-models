"""(A) Is attention to the sensing window what carries the goal? (cutoff net) + per-layer ablation.
   (B) How is the forward-sim executed -- iterative unroll or closed-form geodesic extrapolation?"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
torch.set_num_threads(4)
import goalnav as G
from goalnav_dissect import load, ridge, ang
DEV = "cpu"


def fwd_collect(net, obs, ablate_layers=(), cutoff=0):
    """manual trunk; optionally block queries t>=cutoff from attending keys<cutoff in `ablate_layers`.
    returns lnf residual and per-layer averaged attention weights (B,T,T)."""
    B, T, _ = obs.shape
    x = net.in_proj(obs) + net.pos(torch.arange(T))[None]
    base = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    abl = base.clone()
    if cutoff > 0:
        q = torch.arange(T)[:, None]; k = torch.arange(T)[None]
        abl = abl | ((q >= cutoff) & (k < cutoff))                  # forbid later->sensing
    ws = []
    for li, blk in enumerate(net.blocks):
        h = blk.ln1(x)
        mask = abl if li in ablate_layers else base
        a, w = blk.attn(h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=True)
        x = x + a; x = x + blk.mlp(blk.ln2(x)); ws.append(w.detach())
    return net.lnf(x).detach(), ws


@torch.no_grad()
def attn_to_sensing(net, a, N=600):
    L = a["L"]; cut = a["cutoff"]
    X, obs, g = G.rollout(net, N, L, DEV, np.random.default_rng(1), a["delta"], cut)
    lnf, ws = fwd_collect(net, obs, cutoff=0)
    print(f"  per-layer mean attention from queries t>={cut} to sensing keys [0,{cut}):")
    for li, w in enumerate(ws):
        m = w[:, cut:, :cut].sum(-1).mean().item()                  # total mass on sensing window
        print(f"    blk{li}: {m:.2f}")


@torch.no_grad()
def ablation(net, a, N=1500):
    """train goal probe (clean), then ablate later->sensing attention in early/late/all layers; measure
    goal decode error AND navigation degradation."""
    L = a["L"]; cut = a["cutoff"]; nl = len(net.blocks)
    X, obs, g = G.rollout(net, N, L, DEV, np.random.default_rng(2), a["delta"], cut)
    ntr = int(0.7 * N)
    lnf, _ = fwd_collect(net, obs)
    W = ridge(lnf[:ntr, L // 2:].reshape(-1, lnf.shape[-1]), g[:ntr, None].expand(-1, L - L // 2, -1).reshape(-1, 3))
    def gerr(LF): return ang(LF[ntr:, L // 2:].reshape(-1, LF.shape[-1]) @ W, g[ntr:, None].expand(-1, L - L // 2, -1).reshape(-1, 3)).mean().item()
    print(f"  goal decode-err (tail): clean={gerr(lnf):.0f}deg")
    for label, layers in [("early half", tuple(range(nl // 2))), ("late half", tuple(range(nl // 2, nl))), ("ALL", tuple(range(nl)))]:
        lf, _ = fwd_collect(net, obs, ablate_layers=layers, cutoff=cut)
        print(f"    ablate later->sensing attn in {label:10s}: goal decode-err={gerr(lf):.0f}deg")


@torch.no_grad()
def sim_mechanism(net, a, N=1500):
    """Is the sim prediction a closed-form geodesic 'advance toward goal'? Test: do TRUE future and SIM
    prediction lie on the x_t->goal geodesic (in span{x_t, goal})? off-geodesic angle = out-of-plane."""
    L = a["L"]; X, obs, g = G.rollout(net, N, L, DEV, np.random.default_rng(3), a["delta"], a["cutoff"])
    pred = net.sim_head(net.trunk(obs)).view(N, L, a["r"], 3)
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def offgeo(p, t):                                               # angle of p out of span{x_t, g}
        xt = X[:, t]; gp = g
        e1 = xt; e2 = gp - (gp * xt).sum(-1, keepdim=True) * xt
        e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        inplane = (p * e1).sum(-1, keepdim=True) * e1 + (p * e2).sum(-1, keepdim=True) * e2
        return torch.rad2deg(torch.arccos(inplane.norm(dim=-1).clamp(-1, 1)))   # angle between p and its in-plane proj

    print("  k | sim-vs-true | TRUE-future off-geodesic | SIM-pred off-geodesic   (off-geo ~0 => moves ON the x_t->goal great circle)")
    for k in range(1, a["r"] + 1):
        v = L - k
        true_fut = X[:, k:k + v]
        sv = ang(pred[:, :v, k - 1], true_fut).mean().item()
        og_true = torch.stack([offgeo(true_fut[:, i], i) for i in range(0, v, 4)]).mean().item()
        og_sim = torch.stack([offgeo(pred[:, i, k - 1], i) for i in range(0, v, 4)]).mean().item()
        print(f"  {k} |   {sv:4.1f}     |        {og_true:4.1f}            |     {og_sim:4.1f}")


if __name__ == "__main__":
    net, a = load("gn_cutoff5_6L")
    print("=== (A) gn_cutoff5_6L: attention to sensing window [0,5) ===")
    attn_to_sensing(net, a)
    print("=== (A) ablation: block later->sensing attention ===")
    ablation(net, a)
    print("\n=== (B) gn_alwayson_6L: forward-sim mechanism (geodesic extrapolation?) ===")
    net2, a2 = load("gn_alwayson_6L")
    sim_mechanism(net2, a2)
