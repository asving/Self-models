"""The camps-aware referee: does camps-as-evidence explain ALL of the annealed net's
premium, or is there a residual self-prior?

Sequential Monte Carlo over TRUE-action histories, given the corrupted record AND the
observed camps (the channel the original decoder ladder omitted). Two variants differing
only in the proposal/prior for each round's action:
    uniform prior + camps   -> the no-self-knowledge, camps-aware ceiling
    contextual prior + camps -> the everything ceiling (own policy AND camps)
If the net's observed camp-CE ~ uniform+camps: looking-glass fully explains the premium.
If it sits between: residual self-prior after all.
"""
from __future__ import annotations
import numpy as np
import torch

from ambush import S, GAMMA, BASE
from mirror import PSEUDO
from mirror2_diag import rollout_net
from mirror_probe import load

M = 256   # particles per episode


def smc_camp_ce(pri, truths, recs, keys, camps, is_m, rho, prior_mode, seed=3):
    """Camp-prediction CE of the SMC decoder on mirror episodes, rounds >= 3."""
    rng = np.random.default_rng(seed)
    B, T = recs.shape
    counts = np.zeros((B, M, S, S))                      # (episode, particle, key, action)
    logw = np.zeros((B, M))
    terms = []
    for t in range(T):
        k = keys[:, t]
        # ---- predict this round's camp BEFORE seeing it (mirror episodes)
        if t >= 3:
            cnt = counts[np.arange(B)[:, None], np.arange(M)[None, :], k[:, None]]
            pe = (cnt + PSEUDO / S) / (cnt.sum(-1, keepdims=True) + PSEUDO)
            ex = np.exp(GAMMA * pe)
            cd = ex / ex.sum(-1, keepdims=True)
            wts = np.exp(logw - logw.max(1, keepdims=True))
            wts = wts / wts.sum(1, keepdims=True)
            pred = (wts[..., None] * cd).sum(1)                       # (B,S)
            terms.append(-np.log(pred[np.arange(B), camps[:, t]] + 1e-12)[is_m])
        # ---- camp likelihood update (mirror episodes only; type known to the referee)
        if t >= 1:
            cnt = counts[np.arange(B)[:, None], np.arange(M)[None, :], k[:, None]]
            pe = (cnt + PSEUDO / S) / (cnt.sum(-1, keepdims=True) + PSEUDO)
            ex = np.exp(GAMMA * pe); cd = ex / ex.sum(-1, keepdims=True)
            lik = cd[np.arange(B)[:, None], np.arange(M)[None, :], camps[:, t][:, None]]
            logw += np.where(is_m[:, None], np.log(lik + 1e-12), 0.0)
        # ---- propose this round's true action from prior x record-likelihood
        prior = pri[:, t] if prior_mode == "contextual" else np.full((B, S), 1 / S)
        lik_rec = rho * np.eye(S)[recs[:, t]] + (1 - rho) / 3
        post = prior * lik_rec
        post = post / post.sum(-1, keepdims=True)
        u = rng.random((B, M, 1))
        aprop = (u < post[:, None].cumsum(-1)).argmax(-1)             # (B,M)
        counts[np.arange(B)[:, None], np.arange(M)[None, :], k[:, None], aprop] += 1
        # ---- resample if degenerate
        wts = np.exp(logw - logw.max(1, keepdims=True))
        wts = wts / wts.sum(1, keepdims=True)
        ess = 1.0 / (wts ** 2).sum(1)
        bad = ess < M / 2
        if bad.any():
            for b in np.where(bad)[0]:
                idx = rng.choice(M, M, p=wts[b])
                counts[b] = counts[b, idx]
                logw[b] = 0.0
    return float(np.mean(np.concatenate(terms)))


def main():
    torch.set_grad_enabled(False)
    net = load(f"{BASE}/mirror2_runs/A_anneal/p2_ckpt_008000.pt")
    pri, truths, recs, keys, camps, is_m, obs = rollout_net(net, 1500, seed=17)
    print(f"annealed net observed camp-CE (mirror, t>=3): {obs:.3f}")
    for mode in ("uniform", "contextual"):
        ce = smc_camp_ce(pri, truths, recs, keys, camps, is_m, 0.3, mode)
        print(f"  camps-aware SMC decoder, {mode:>10} prior: CE = {ce:.3f}")


if __name__ == "__main__":
    main()
