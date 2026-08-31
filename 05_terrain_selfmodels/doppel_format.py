"""Format-sharing probe: does the net's OPPONENT-SLOT representation (fit on persona
episodes) fill with its OWN policy's content on self-play streams -- and does it fill
EARLY (rounds 0-2, before observational decoding is possible)?

Probe: ridge, per layer, residuals at decision positions -> the opponent's TRUE
next-action distribution. Fit on net-vs-persona rollouts (private streams); freeze;
apply to net-vs-copy rollouts (target = the copy's actual distribution). Compare
round-resolved R^2 on self vs held-out personas; early excess on self = own-policy
content entering the opponent slot ahead of evidence.

Usage: CUDA_VISIBLE_DEVICES=<id> python doppel_format.py <ckpt-path>
"""
from __future__ import annotations
import sys
import numpy as np
import torch, torch.nn.functional as F

from ambush import World, S, TOK_X0, TOK_A0, TOK_C0, TOK_BOS, BASE, filt_obs
from doppel import Personas
from mirror_probe import load, hiddens, DEV

T = 24


@torch.no_grad()
def record_rollout(net, B, seed, opponent):
    """opponent: 'personas' or 'self'. Returns tokens, per-round TRUE opponent dist,
    per-round opponent key stats."""
    rng = np.random.default_rng(seed)
    w = World(B, rng)
    opp = Personas(B, rng)
    cnet = net
    tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
    tt2 = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
    pdist = np.zeros((B, T, S))
    for t in range(T):
        z = w.emit(); z2 = w.emit()
        tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(DEV)], 1)
        tt2 = torch.cat([tt2, torch.from_numpy(TOK_X0 + z2[:, None]).to(DEV)], 1)
        a = torch.multinomial(F.softmax(net(tt)[:, -1, TOK_A0:TOK_C0], -1), 1).squeeze(1)
        if opponent == "self":
            pc = F.softmax(cnet(tt2)[:, -1, TOK_A0:TOK_C0], -1).cpu().numpy()
            c = np.array([np.random.default_rng(seed * 1000 + t * B + b).choice(S, p=pc[b])
                          for b in range(B)])
        else:
            opp.eta = filt_obs(opp.eta, z2)
            pc = opp.dist(opp.eta, z2)
            cs = pc.cumsum(-1); u = rng.random((B, 1))
            c = (u < cs).argmax(-1)
            opp.last = c.copy()
            opp.drift()
        pdist[:, t] = pc
        c_t = torch.from_numpy(c).to(DEV)
        tt = torch.cat([tt, torch.stack([TOK_A0 + a, TOK_C0 + c_t], 1)], 1)
        tt2 = torch.cat([tt2, torch.stack([TOK_A0 + c_t, TOK_C0 + a], 1)], 1)
        w.step()
    return tt, pdist


def ridge(Htr, Ytr, lam=1.0):
    H1 = np.concatenate([Htr, np.ones((len(Htr), 1))], 1)
    return np.linalg.solve(H1.T @ H1 + lam * np.eye(H1.shape[1]), H1.T @ Ytr)


def r2_of(W, H, Y):
    P = np.concatenate([H, np.ones((len(H), 1))], 1) @ W
    sse = ((P - Y) ** 2).sum(0); sst = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1 - sse / sst))


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/doppel_runs/P/p2_ckpt_008000.pt"
    net = load(ckpt)
    torch.set_grad_enabled(False)
    pos = 1 + 3 * np.arange(T)
    tt_p, pd_p = record_rollout(net, 1024, 61, "personas")
    hs_p = hiddens(net, tt_p)
    ntr = 700
    best = (None, -1)
    for li in range(hs_p.shape[0]):
        H = hs_p[li][:, pos]
        Wm = ridge(H[:ntr].reshape(-1, H.shape[-1]), pd_p[:ntr].reshape(-1, S))
        r2 = r2_of(Wm, H[ntr:].reshape(-1, H.shape[-1]), pd_p[ntr:].reshape(-1, S))
        print(f"persona-slot probe L{li}: heldout R2 = {r2:.3f}")
        if r2 > best[1]:
            best = ((li, Wm), r2)
    (li, Wm), r2p = best
    print(f"best layer L{li} (R2={r2p:.3f}); freezing probe")

    tt_s, pd_s = record_rollout(net, 1024, 62, "self")
    hs_s = hiddens(net, tt_s)[li][:, pos]
    hs_h = hs_p[li][:, pos][ntr:]                        # held-out personas
    print("\nround-resolved frozen-probe R2 (opponent-slot -> true opponent dist):")
    print(f"{'rounds':>8} {'personas(heldout)':>18} {'self':>8}")
    for lo, hi, tag in ((0, 3, "0-2"), (3, 8, "3-7"), (8, 16, "8-15"), (16, 24, "16-23")):
        rp = r2_of(Wm, hs_h[:, lo:hi].reshape(-1, hs_h.shape[-1]),
                   pd_p[ntr:, lo:hi].reshape(-1, S))
        rs = r2_of(Wm, hs_s[:, lo:hi].reshape(-1, hs_s.shape[-1]),
                   pd_s[:, lo:hi].reshape(-1, S))
        print(f"{tag:>8} {rp:18.3f} {rs:8.3f}")
    # shuffle control
    sh = np.random.default_rng(0).permutation(len(pd_s.reshape(-1, S)))
    rsh = r2_of(Wm, hs_s.reshape(-1, hs_s.shape[-1]), pd_s.reshape(-1, S)[sh])
    print(f"shuffle control: {rsh:.3f}")


if __name__ == "__main__":
    main()
