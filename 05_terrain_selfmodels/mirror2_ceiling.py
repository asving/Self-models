"""Step 0 for the corrupted-record mirror (option C): certify the introspection premium.

Mirror episodes with a scripted persona. The stream shows a CORRUPTED action record
(true w.p. rho, else uniform symbol); camps track the TRUE p_emp. We compute the
camp-prediction CE and the dodge-relevant accuracy (argmax of estimated p_emp) for a
ladder of record-decoders that differ ONLY in the per-round PRIOR over the true action:

  naive    : trust the record (delta on the shown token)
  uniform  : corruption-aware, uninformative prior
  marginal : corruption-aware, persona's context-free action marginal
  contextual (SELF-KNOWLEDGE): corruption-aware, the persona's true per-context policy
             -- this is what introspective access to one's own policy buys.

Premium = marginal - contextual, in camp-CE nats and in dodge accuracy. Sweep rho.
Exact per-round posteriors (independent given the record); p_emp posterior via MC over
action histories. CPU only.
"""
from __future__ import annotations
import numpy as np

from ambush import S, GAMMA, E_MAT, onehot, sample_rows, filt_obs, filt_step, M_DRIFT
from ambush import World
from mirror import Mirror, PSEUDO

T, K_MC = 24, 96


def persona_policy(kind, eta, rng):
    if kind == "greedy":
        return onehot(eta.argmax(1)) * 0.95 + 0.05 / S
    if kind == "soft":
        p = eta ** (1 / 0.3)
        return p / p.sum(-1, keepdims=True)
    raise ValueError(kind)


def run(kind, rho, B, seed):
    rng = np.random.default_rng(seed)
    w = World(B, rng)
    mir = Mirror(B)
    eta = np.full((B, S), 1 / S)
    priors_ctx = np.zeros((B, T, S)); truths = np.zeros((B, T), dtype=int)
    records = np.zeros((B, T), dtype=int); keys = np.zeros((B, T), dtype=int)
    camps = np.zeros((B, T), dtype=int); pemps = np.zeros((B, T, S))
    for t in range(T):
        z = w.emit(); eta = filt_obs(eta, z)
        keys[:, t] = z
        pc, p_emp = mir.camp_dist(z, np.ones(B, bool), np.zeros((B, S)))
        pemps[:, t] = p_emp
        camps[:, t] = sample_rows(pc, rng)
        p = persona_policy(kind, eta, rng)
        priors_ctx[:, t] = p
        a = sample_rows(p, rng)
        truths[:, t] = a
        corrupt = rng.random(B) >= rho
        records[:, t] = np.where(corrupt, rng.integers(0, S, B), a)
        mir.update(z, a)
        w.step(); eta = filt_step(eta)
    marg = np.bincount(truths.ravel(), minlength=S) / truths.size
    return priors_ctx, truths, records, keys, camps, pemps, marg


def posterior(record, prior, rho):
    """P(a | shown record, prior). record: (...,) ints; prior: (...,S)."""
    lik = rho * np.eye(S)[record] + (1 - rho) / 3
    post = prior * lik
    return post / post.sum(-1, keepdims=True)


def evaluate(kind, rho, B=1500, seed=11):
    priors_ctx, truths, records, keys, camps, pemps, marg = run(kind, rho, B, seed)
    rng = np.random.default_rng(99)
    out = {}
    for name in ("naive", "uniform", "marginal", "contextual"):
        if name == "naive":
            post = np.eye(S)[records]
        else:
            prior = (np.full((B, T, S), 1 / S) if name == "uniform" else
                     np.broadcast_to(marg, (B, T, S)) if name == "marginal" else
                     priors_ctx)
            post = posterior(records, prior, rho)
        ce_terms, acc_terms, n_acc = [], [], 0
        # MC over action histories -> camp predictive at each round with >=1 same-key visit
        samples = (rng.random((K_MC, B, T, 1)) <
                   post.cumsum(-1)[None]).argmax(-1)      # (K,B,T)
        for t in range(3, T):
            sk = keys[:, :t] == keys[:, t][:, None]       # (B,t)
            nsk = sk.sum(1)
            cnt = np.zeros((K_MC, B, S))
            for a in range(S):
                cnt[..., a] = ((samples[:, :, :t] == a) & sk[None]).sum(-1)
            pe = (cnt + PSEUDO / S) / (cnt.sum(-1, keepdims=True) + PSEUDO)
            ex = np.exp(GAMMA * pe)
            cd = (ex / ex.sum(-1, keepdims=True)).mean(0)  # (B,S) camp predictive
            ce_terms.append(-np.log(cd[np.arange(B), camps[:, t]] + 1e-12))
            sel = nsk >= 2
            if sel.any():
                est = pe.mean(0)                          # (B,S) E[p_emp]
                acc_terms.append((est.argmax(-1) == pemps[:, t].argmax(-1))[sel])
                n_acc += sel.sum()
        out[name] = (float(np.mean(ce_terms)),
                     float(np.concatenate(acc_terms).mean()))
    return out


def main():
    for kind in ("soft", "greedy"):
        print(f"\n===== persona: {kind} =====")
        print(f"{'rho':>5} | camp-CE: {'naive':>6} {'unif':>6} {'marg':>6} {'ctx':>6} "
              f"{'PREM':>6} | dodge-acc: {'marg':>6} {'ctx':>6} {'PREM':>6}")
        for rho in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
            o = evaluate(kind, rho)
            prem_ce = o["marginal"][0] - o["contextual"][0]
            prem_ac = o["contextual"][1] - o["marginal"][1]
            print(f"{rho:5.1f} | {o['naive'][0]:6.3f} {o['uniform'][0]:6.3f} "
                  f"{o['marginal'][0]:6.3f} {o['contextual'][0]:6.3f} {prem_ce:6.3f} | "
                  f"{o['marginal'][1]:10.3f} {o['contextual'][1]:6.3f} {prem_ac:6.3f}")


if __name__ == "__main__":
    main()
