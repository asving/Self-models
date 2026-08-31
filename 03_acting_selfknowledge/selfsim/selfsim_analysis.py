"""Analyze the self-sim net: HOW does it predict its own K-step-ahead output?
 [1] does it represent its own POLICY (the mode m) and does that sharpen with context?
 [2] self-simulation signature: at the query position, are the INTERMEDIATE iterates π_m^j(x_L)
     (j=1..K) present in the residual, ideally as a staircase across depth? (=> iterating the policy)
 [3] mechanism (a) iterate-policy vs (b) index-into-observed-orbit: behavioral test — does it still
     predict K-ahead when the context is too SHORT to have observed the answer in the orbit?
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import whitebox, probes
import selfsim as S

BASE = os.path.dirname(os.path.abspath(__file__))
LAYERS = ["embed"] + [f"L{i}.resid_post" for i in range(4)]
rng = np.random.default_rng(7)


def probe_acc(X, lab, nclass, tr=0.7):
    n = len(X); cut = int(n * tr); idx = rng.permutation(n)
    Xtr, Xte = X[idx[:cut]], X[idx[cut:]]; ltr, lte = lab[idx[:cut]], lab[idx[cut:]]
    W, b, _ = probes.ridge_fit(Xtr, np.eye(nclass)[ltr])
    return (((Xte @ W + b).argmax(1)) == lte).mean()


def main():
    W = whitebox.load_weights(BASE + "/runs/selfsim.pt")
    L = 12
    seq, tgt, m, K, _, pw = S.gen(4000, rng, L=L)           # pw[:,j] = π_m^j(x_L)
    acts, _ = whitebox.forward(W, seq, n_layer=4, n_head=4)
    sel = K == S.KMAX                                        # full-rollout queries (need all iterates)
    qpos = L                                                 # the Q_K token position

    print("=" * 68)
    print("[1] POLICY (mode m) decodable from the query position? (8-way; chance=0.125)")
    for ln in LAYERS:
        print(f"    {ln:14s}: acc={probe_acc(acts[ln][:, qpos, :], m, S.M):.3f}")
    print("    -- does mode-certainty rise with context length? (probe at last orbit pos, L1)")
    for Lc in [3, 5, 8, 12, 16]:
        sq, tg, mm, kk, _, _ = S.gen(3000, rng, L=Lc)
        ac, _ = whitebox.forward(W, sq, n_layer=4, n_head=4)
        print(f"      L={Lc:2d}: mode acc={probe_acc(ac['L1.resid_post'][:, Lc - 1, :], mm, S.M):.3f}")

    print("=" * 68)
    print("[2] SELF-SIMULATION: intermediate iterate π_m^j(x_L) decodable at query pos, per layer")
    print("    (6-way, chance=0.167). Staircase [π^j appears ~layer j] => rolling the policy forward.")
    print("    layer        | π^1   π^2   π^3   π^4   π^5(=answer)")
    Xq = {ln: acts[ln][sel, qpos, :] for ln in LAYERS}
    for ln in LAYERS:
        accs = [probe_acc(Xq[ln], pw[sel, j], S.P) for j in range(1, S.KMAX + 1)]
        print(f"    {ln:12s} | " + "  ".join(f"{a:.2f}" for a in accs))

    print("=" * 68)
    print("[3] MECHANISM — accuracy vs context length (short context can't have observed the answer)")
    print("    if it iterates the policy, K-ahead works even when L < orbit period observed.")
    for Lc in [2, 3, 4, 6, 10]:
        sq, tg, mm, kk, _, _ = S.gen(4000, rng, L=Lc)
        pred = whitebox.forward(W, sq, n_layer=4, n_head=4)[0]["logits"][:, Lc, :].argmax(-1)
        acc_by_k = [(pred[kk == k] == tg[kk == k]).mean() for k in range(1, S.KMAX + 1)]
        print(f"    L={Lc:2d}: " + " ".join(f"K{k}={acc_by_k[k-1]:.2f}" for k in range(1, S.KMAX + 1)))


if __name__ == "__main__":
    main()
