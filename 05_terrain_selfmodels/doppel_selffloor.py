"""Self-play excess measurement: net-vs-doppelganger catch minus the clairvoyant-state
family decoder run on the SAME self-streams. Usage: python doppel_selffloor.py <ckpt>"""
from __future__ import annotations
import sys
import numpy as np
import torch, torch.nn.functional as F

from ambush import World, S, TOK_X0, TOK_A0, TOK_C0, TOK_BOS, BASE, filt_obs
from doppel import Personas
from mirror_probe import load, DEV

T, M, B = 24, 256, 512


@torch.no_grad()
def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/doppel_runs/P/p2_ckpt_008000.pt"
    net = load(ckpt)
    rng = np.random.default_rng(555)
    w = World(B, rng)
    P = Personas(B * M, np.random.default_rng(7))
    logw = np.zeros((B, M))
    tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
    tt2 = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
    hit_net = np.zeros((B, T)); hit_floor = np.zeros((B, T))
    for t in range(T):
        z = w.emit(); z2 = w.emit()
        zM = np.stack([w.emit() for _ in range(M)], 1).reshape(B * M)
        tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(DEV)], 1)
        tt2 = torch.cat([tt2, torch.from_numpy(TOK_X0 + z2[:, None]).to(DEV)], 1)
        a = torch.multinomial(F.softmax(net(tt)[:, -1, TOK_A0:TOK_C0], -1), 1).squeeze(1)
        c = torch.multinomial(F.softmax(net(tt2)[:, -1, TOK_A0:TOK_C0], -1), 1).squeeze(1)
        c_np = c.cpu().numpy()
        P.eta = filt_obs(P.eta, zM)
        pd = P.dist(P.eta, zM).reshape(B, M, S)
        wts = np.exp(logw - logw.max(1, keepdims=True)); wts /= wts.sum(1, keepdims=True)
        hit_floor[:, t] = (wts[..., None] * pd).sum(1).argmax(1) == c_np
        hit_net[:, t] = a.cpu().numpy() == c_np
        logw += np.log(pd[np.arange(B)[:, None], np.arange(M)[None, :], c_np[:, None]] + 1e-12)
        P.last = np.repeat(c_np, M, 0); P.drift()
        tt = torch.cat([tt, torch.stack([TOK_A0 + a, TOK_C0 + c], 1)], 1)
        tt2 = torch.cat([tt2, torch.stack([TOK_A0 + c, TOK_C0 + a], 1)], 1)
        w.step()
    print("round      : " + " ".join(f"{t:4d}" for t in range(0, T, 3)))
    print("net-vs-self: " + " ".join(f"{hit_net[:, t].mean():.2f}" for t in range(0, T, 3)))
    print("floor-self : " + " ".join(f"{hit_floor[:, t].mean():.2f}" for t in range(0, T, 3)))
    for lo, hi, tag in ((0, 3, "rounds 0-2"), (12, 24, "late")):
        e = hit_net[:, lo:hi].mean() - hit_floor[:, lo:hi].mean()
        print(f"{tag}: net {hit_net[:, lo:hi].mean():.3f}  floor {hit_floor[:, lo:hi].mean():.3f}"
              f"  EXCESS {e:+.3f}")


if __name__ == "__main__":
    main()
