"""Graded-target curve for the joint-token net (working decode->exploit at last).

For each target checkpoint (its own past selves; sibling when available): private-stream
pairing, per-round catch + clairvoyant family floor on the same c-streams, and the
context-matched TV distance between agent and target policies. With identical twins the
catch is definitionally twin-correlation; the modeling content lives in (i) the
within-episode CLIMB beyond the floor's climb and (ii) the gradient across non-identical
graded targets. Usage: python doppel2_curve.py [sibling_ckpt]
"""
from __future__ import annotations
import sys
import numpy as np
import torch, torch.nn.functional as F

from ambush import World, S, filt_obs, BASE
from doppel import Personas
from doppel2 import Net2, TOK_X0, TOK_J0, TOK_BOS, jtok, joint_slice, gen_phase1

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T, M, B = 24, 256, 512
RUN = f"{BASE}/doppel2_runs"


def load2(path):
    sd = torch.load(path, map_location=DEV)
    net = Net2(sd["cfg"]["d"], sd["cfg"]["nl"], sd["cfg"]["nh"]).to(DEV)
    net.load_state_dict(sd["model"]); net.eval()
    return net


def amarg(net, tt):
    jl = joint_slice(net(tt)[:, -1])
    return F.softmax(jl.reshape(len(tt), 9), -1).view(-1, 3, 3).sum(-1)


@torch.no_grad()
def pairing(agent, target, seed=555):
    rng = np.random.default_rng(seed)
    w = World(B, rng)
    P = Personas(B * M, np.random.default_rng(7))
    logw = np.zeros((B, M))
    tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
    tt2 = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
    hit = np.zeros((B, T)); flr = np.zeros((B, T))
    for t in range(T):
        z = w.emit(); z2 = w.emit()
        zM = np.stack([w.emit() for _ in range(M)], 1).reshape(B * M)
        tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(DEV)], 1)
        tt2 = torch.cat([tt2, torch.from_numpy(TOK_X0 + z2[:, None]).to(DEV)], 1)
        a = torch.multinomial(amarg(agent, tt), 1).squeeze(1)
        c = torch.multinomial(amarg(target, tt2), 1).squeeze(1)
        c_np = c.cpu().numpy()
        P.eta = filt_obs(P.eta, zM)
        pd = P.dist(P.eta, zM).reshape(B, M, S)
        wts = np.exp(logw - logw.max(1, keepdims=True)); wts /= wts.sum(1, keepdims=True)
        flr[:, t] = (wts[..., None] * pd).sum(1).argmax(1) == c_np
        hit[:, t] = a.cpu().numpy() == c_np
        logw += np.log(pd[np.arange(B)[:, None], np.arange(M)[None, :], c_np[:, None]] + 1e-12)
        P.last = np.repeat(c_np, M, 0); P.drift()
        tt = torch.cat([tt, (TOK_J0 + 3 * a + c)[:, None]], 1)
        tt2 = torch.cat([tt2, (TOK_J0 + 3 * c + a)[:, None]], 1)
        w.step()
    return hit, flr


@torch.no_grad()
def tvdist(agent, target, seed=99):
    rng = np.random.default_rng(seed)
    toks = torch.from_numpy(gen_phase1(512, T, rng)).to(DEV)
    pos = 1 + 2 * np.arange(T)
    lg_a = agent(toks[:, :-1]); lg_t = target(toks[:, :-1])
    def pol(lg):
        jl = lg[:, pos, TOK_J0:TOK_J0 + 9].view(-1, T, 3, 3)
        return F.softmax(jl.reshape(-1, T, 9), -1).view(-1, T, 3, 3).sum(-1)
    return float(0.5 * (pol(lg_a) - pol(lg_t)).abs().sum(-1).mean())


def main():
    torch.set_grad_enabled(False)
    agent = load2(f"{RUN}/A/p2_ckpt_008000.pt")
    targets = [("self(8000)", f"{RUN}/A/p2_ckpt_008000.pt"),
               ("self@2000", f"{RUN}/A/p2_ckpt_002000.pt"),
               ("self@1000", f"{RUN}/A/p2_ckpt_001000.pt"),
               ("self@0(pre)", f"{RUN}/A/p2_ckpt_000000.pt")]
    if len(sys.argv) > 1:
        targets.append(("sibling", sys.argv[1]))
    print(f"{'target':>12} {'TV':>6} {'catch e/l':>12} {'floor e/l':>12} {'EXCESS e/l':>14} {'climb(net-flr)':>15}")
    for name, path in targets:
        tgt = load2(path)
        d = tvdist(agent, tgt)
        hit, flr = pairing(agent, tgt)
        he, hl = hit[:, :3].mean(), hit[:, 12:].mean()
        fe, fl = flr[:, :3].mean(), flr[:, 12:].mean()
        climb = (hl - he) - (fl - fe)
        print(f"{name:>12} {d:6.3f} {he:5.3f}/{hl:5.3f} {fe:5.3f}/{fl:5.3f} "
              f"{he-fe:+6.3f}/{hl-fl:+6.3f} {climb:+15.3f}")


if __name__ == "__main__":
    main()
