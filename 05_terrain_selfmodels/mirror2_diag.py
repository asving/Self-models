"""Diagnostic: where does the trained mirror2 net's camp-CE sit relative to the
record-decoder ceilings COMPUTED FOR ITS OWN POLICY? If observed CE ~ marginal ceiling,
the net acquired no per-context self-knowledge; the gap to the contextual ceiling is the
unclaimed premium (which its own high entropy may have shrunk -- report that too)."""
from __future__ import annotations
import numpy as np
import torch, torch.nn.functional as F

from ambush import S, GAMMA, TOK_X0, TOK_A0, TOK_C0, TOK_BOS, BASE, THETA
from ambush import World, sample_rows, filt_obs, filt_step
from mirror import Mirror, PSEUDO
from mirror2 import corrupt
from mirror_probe import load, DEV
from mirror2_ceiling import posterior

T, RHO, K_MC = 24, 0.3, 96


@torch.no_grad()
def rollout_net(net, B, seed):
    rng = np.random.default_rng(seed)
    w = World(B, rng)
    is_mirror = rng.random(B) < 0.5
    qcamp = rng.dirichlet(np.full(S, 0.5), B)
    mir = Mirror(B)
    eta = np.full((B, S), 1 / S)
    pri = np.zeros((B, T, S)); truths = np.zeros((B, T), int)
    recs = np.zeros((B, T), int); keys = np.zeros((B, T), int); camps = np.zeros((B, T), int)
    tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
    for t in range(T):
        z = w.emit(); eta = filt_obs(eta, z); keys[:, t] = z
        tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(DEV)], 1)
        p = F.softmax(net(tt)[:, -1, TOK_A0:TOK_C0], -1).cpu().numpy()
        pri[:, t] = p
        a = sample_rows(p, rng); truths[:, t] = a
        pc, _ = mir.camp_dist(z, is_mirror, qcamp)
        c = sample_rows(pc, rng); camps[:, t] = c
        rec = corrupt(a, RHO, rng); recs[:, t] = rec
        mir.update(z, a)
        tt = torch.cat([tt, torch.from_numpy(
            np.stack([TOK_A0 + rec, TOK_C0 + c], 1)).to(DEV)], 1)
        w.step(); eta = filt_step(eta)
    # net's own observed camp CE on these rollouts (teacher forced), mirror episodes
    lsm = F.log_softmax(net(tt[:, :-1]), -1)
    camp_pos = 2 + 3 * np.arange(T)
    tgt = tt[:, 1:]
    ce = -(lsm.gather(-1, tgt[..., None]).squeeze(-1)[:, camp_pos]).cpu().numpy()
    return pri, truths, recs, keys, camps, is_mirror, float(ce[is_mirror][:, 3:].mean())


def decoder_ce(pri, truths, recs, keys, camps, is_mirror, mode):
    B = len(pri)
    rng = np.random.default_rng(7)
    if mode == "contextual":
        prior = pri
    elif mode == "marginal":
        marg = np.bincount(truths.ravel(), minlength=S) / truths.size
        prior = np.broadcast_to(marg, pri.shape)
    else:
        prior = np.full_like(pri, 1 / S)
    post = posterior(recs, prior, RHO)
    samples = (rng.random((K_MC, B, T, 1)) < post.cumsum(-1)[None]).argmax(-1)
    terms = []
    for t in range(3, T):
        sk = keys[:, :t] == keys[:, t][:, None]
        cnt = np.zeros((K_MC, B, S))
        for a in range(S):
            cnt[..., a] = ((samples[:, :, :t] == a) & sk[None]).sum(-1)
        pe = (cnt + PSEUDO / S) / (cnt.sum(-1, keepdims=True) + PSEUDO)
        ex = np.exp(GAMMA * pe)
        cd = (ex / ex.sum(-1, keepdims=True)).mean(0)
        terms.append(-np.log(cd[np.arange(B), camps[:, t]] + 1e-12)[is_mirror])
    return float(np.mean(np.concatenate(terms)))


def main():
    torch.set_grad_enabled(False)
    import sys
    for arm in (sys.argv[1:] or ("A", "B")):
        net = load(f"{BASE}/mirror2_runs/{arm}/p2_ckpt_008000.pt")
        pri, truths, recs, keys, camps, is_m, obs = rollout_net(net, 1500, seed=17)
        H = float(-(pri * np.log(pri + 1e-12)).sum(-1).mean())
        print(f"== arm {arm} ==  policy entropy H={H:.2f}   observed camp-CE (mirror) = {obs:.3f}")
        for mode in ("uniform", "marginal", "contextual"):
            ce = decoder_ce(pri, truths, recs, keys, camps, is_m, mode)
            print(f"   {mode:>10}-prior decoder ceiling = {ce:.3f}")


if __name__ == "__main__":
    main()
