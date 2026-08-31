"""Similarity-response curve: does the late excess-over-floor decline with the target's
behavioral distance from the CURRENT self? Targets: the agent's own earlier RL
checkpoints (graded near-selves), the independently-trained sibling, vs per-target
clairvoyant floors. Self-centered template => excess falls with distance-from-me;
kind-centered => flat across neural targets."""
from __future__ import annotations
import numpy as np
import torch, torch.nn.functional as F

from ambush import World, S, TOK_X0, TOK_A0, TOK_C0, TOK_BOS, BASE, filt_obs
from doppel import Personas, gen_phase1
from mirror_probe import load, DEV

T, M, B = 24, 256, 512
AGENT = f"{BASE}/doppel_runs/P2/p2_ckpt_008000.pt"
TARGETS = [("self(8000)", f"{BASE}/doppel_runs/P2/p2_ckpt_008000.pt"),
           ("self@2000", f"{BASE}/doppel_runs/P2/p2_ckpt_002000.pt"),
           ("self@400", f"{BASE}/doppel_runs/P2/p2_ckpt_000400.pt"),
           ("self@0(pre)", f"{BASE}/doppel_runs/P2/p2_ckpt_000000.pt"),
           ("sibling", f"{BASE}/doppel_runs/P/p2_ckpt_008000.pt")]


@torch.no_grad()
def pairing_excess(agent, target, seed=555):
    rng = np.random.default_rng(seed)
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
        a = torch.multinomial(F.softmax(agent(tt)[:, -1, TOK_A0:TOK_C0], -1), 1).squeeze(1)
        c = torch.multinomial(F.softmax(target(tt2)[:, -1, TOK_A0:TOK_C0], -1), 1).squeeze(1)
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
    late = slice(12, T)
    return (float(hit_net[:, late].mean()), float(hit_floor[:, late].mean()))


@torch.no_grad()
def tv_distance(agent, target, seed=99):
    """Context-matched policy distance: both nets teacher-forced on the SAME persona
    streams; mean TV of a-slot policies at decision positions."""
    rng = np.random.default_rng(seed)
    toks = torch.from_numpy(gen_phase1(512, T, rng, private=True)).to(DEV)
    pos = 1 + 3 * np.arange(T)
    pa = F.softmax(agent(toks[:, :-1])[:, pos, TOK_A0:TOK_C0], -1)
    pt = F.softmax(target(toks[:, :-1])[:, pos, TOK_A0:TOK_C0], -1)
    return float(0.5 * (pa - pt).abs().sum(-1).mean())


def main():
    torch.set_grad_enabled(False)
    agent = load(AGENT)
    print(f"{'target':>12} {'TV-dist':>8} {'catch':>7} {'floor':>7} {'EXCESS':>8}")
    for name, path in TARGETS:
        tgt = load(path)
        d = tv_distance(agent, tgt)
        c, f = pairing_excess(agent, tgt)
        print(f"{name:>12} {d:8.3f} {c:7.3f} {f:7.3f} {c - f:+8.3f}")


if __name__ == "__main__":
    main()
