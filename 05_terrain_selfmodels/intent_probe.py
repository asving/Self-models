"""Prospective self-use probe: at the DECISION position (before the action token exists),
does the net's representation of the action-consequence already condition on its imminent
action? Three hypotheses, decoded and compared on positions where they diverge:
  T_null : no self-conditioning (consequence quantity left un-updated)
  T_marg : pre-updated on the policy DISTRIBUTION (Bayes-correct prospective use)
  T_cond : pre-committed to the argmax intention
  T_samp : updated on the REALIZED sample (unknowable at decision time -- honesty control:
           must NOT beat T_cond/T_marg; if it does, the probe is leaking).

Mirror family (consequence = own precedent -> future same-key camps): target = the
post-action p_emp of the current key. Models: mirror-A final (v3, true records), its
pretrained phase-1 net (watching personas: 'intention' = predicted persona action),
mirror2 annealed (corrupted records).
"""
from __future__ import annotations
import numpy as np
import torch, torch.nn.functional as F

from ambush import S, TOK_X0, TOK_A0, TOK_C0, TOK_BOS, BASE
from mirror import Mirror, PSEUDO, gen_phase1 as mirror_gen
from mirror_probe import load, hiddens, DEV, T
from mirror2_binding import rollout as m2_rollout

NP = PSEUDO


def targets_from(counts_t, p_t, a_samp):
    """counts_t: (N,3) true same-key counts BEFORE the action; p_t: (N,3) policy/prediction;
    a_samp: (N,) realized. Returns dict of four (N,3) targets."""
    n = counts_t.sum(-1, keepdims=True)
    t_null = (counts_t + NP / S) / (n + NP)
    t_marg = (counts_t + p_t + NP / S) / (n + 1 + NP)
    t_cond = (counts_t + np.eye(S)[p_t.argmax(-1)] + NP / S) / (n + 1 + NP)
    t_samp = (counts_t + np.eye(S)[a_samp] + NP / S) / (n + 1 + NP)
    return dict(null=t_null, marg=t_marg, cond=t_cond, samp=t_samp)


def ridge_fit(H, Y, lam=1.0):
    H1 = np.concatenate([H, np.ones((len(H), 1))], 1)
    return np.linalg.solve(H1.T @ H1 + lam * np.eye(H1.shape[1]), H1.T @ Y)


def r2_on(W, H, Y):
    P = np.concatenate([H, np.ones((len(H), 1))], 1) @ W
    sse = ((P - Y) ** 2).sum(0); sst = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1 - sse / sst))


def counts_series(keys, acts):
    """True same-key count vectors BEFORE each round's action. keys, acts: (B,T)."""
    B = len(keys)
    cnt = np.zeros((B, S, S))
    out = np.zeros((B, T, S))
    for t in range(T):
        out[:, t] = cnt[np.arange(B), keys[:, t]]
        cnt[np.arange(B), keys[:, t], acts[:, t]] += 1
    return out


def evaluate(name, net, toks, keys, acts, p_t, layer_sweep=(3, 4, 5, 6)):
    """toks: (B,L) token tensor on DEV; decision positions = 1+3t."""
    B = len(toks)
    pos = 1 + 3 * np.arange(T)
    cnts = counts_series(keys, acts)
    tg = {k: v.reshape(B * T, S) for k, v in
          targets_from(cnts.reshape(-1, S), p_t.reshape(-1, S), acts.reshape(-1)).items()}
    hs = hiddens(net, toks)
    tv = 0.5 * np.abs(tg["null"] - tg["cond"]).sum(-1)
    div = tv > 0.10
    ntr = int(0.7 * B) * T
    idx_te_div = np.where(div)[0]; idx_te_div = idx_te_div[idx_te_div >= ntr]
    best = {}
    for li in layer_sweep:
        H = hs[li][:, pos].reshape(B * T, -1)
        for tgt in ("null", "marg", "cond", "samp"):
            W = ridge_fit(H[:ntr], tg[tgt][:ntr])
            r2 = r2_on(W, H[idx_te_div], tg[tgt][idx_te_div])
            if tgt not in best or r2 > best[tgt][0]:
                best[tgt] = (r2, li)
    print(f"{name:>22} | n_div(test)={len(idx_te_div):5d} | " + "  ".join(
        f"{t}: R2={best[t][0]:.3f}(L{best[t][1]})" for t in ("null", "marg", "cond", "samp")))
    return best


@torch.no_grad()
def main():
    torch.set_grad_enabled(False)
    print("targets on DIVERGENT decision positions (TV(null,cond)>0.10), held-out episodes\n")

    # ---- mirror v3, final RL net (acts itself; true actions in tokens)
    net = load(f"{BASE}/mirror_runs/A/p2_ckpt_008000.pt")
    tt, is_m, p_emps, lastc, seen = __import__("mirror_probe").rollout(net, 1024, seed=91)
    toks = tt.cpu().numpy()
    keys = toks[:, 1 + 3 * np.arange(T)] - TOK_X0
    acts = toks[:, 2 + 3 * np.arange(T)] - TOK_A0
    lsm = F.softmax(net(tt[:, :-1])[:, 1 + 3 * np.arange(T), TOK_A0:TOK_C0], -1).cpu().numpy()
    evaluate("mirror-A (RL, self)", net, tt, keys, acts, lsm)

    # ---- mirror v3, PRETRAINED net watching scripted personas (intention = its prediction)
    net_p1 = load(f"{BASE}/mirror_runs/A/p1_ckpt_020000.pt")
    rng = np.random.default_rng(17)
    toks_np, types, is_m2 = mirror_gen(1024, T, 0.5, rng)
    tt2 = torch.from_numpy(toks_np).to(DEV)
    keys2 = toks_np[:, 1 + 3 * np.arange(T)] - TOK_X0
    acts2 = toks_np[:, 2 + 3 * np.arange(T)] - TOK_A0
    p2 = F.softmax(net_p1(tt2[:, :-1])[:, 1 + 3 * np.arange(T), TOK_A0:TOK_C0], -1).cpu().numpy()
    evaluate("mirror-p1 (watching)", net_p1, tt2, keys2, acts2, p2)

    # ---- mirror2 annealed (corrupted records; true actions NOT in tokens)
    net_a = load(f"{BASE}/mirror2_runs/A_anneal/p2_ckpt_008000.pt")
    tta, pri, truths, recs, keysa, campsa, is_ma, _ = m2_rollout(net_a, 1024, seed=93,
                                                                 puppet=False)
    evaluate("mirror2-anneal (self)", net_a, tta, keysa, truths, pri)

    # ---- mirror2 annealed, PUPPETED (intentions exist, actions forced -- dissociation)
    ttp, prip, truthsp, recsp, keysp, campsp, is_mp, _ = m2_rollout(net_a, 1024, seed=94,
                                                                    puppet=True)
    evaluate("mirror2-anneal PUPPET", net_a, ttp, keysp, truthsp, prip)


if __name__ == "__main__":
    main()
