"""Mechanistic decomposition on top of the verified white-box (whitebox.py).
Validation-first: every claim is checked by reconstructing activations/logits and measuring error.

Analyses:
  1 embedding factoring   : does tok.weight = mu + U[z0] + W[z1] (additive over sub-tokens)?
  2 logit read-off        : are logits additive over (z0,z1) (factored product)? do marginals = belief-implied next-token?
  3 belief-update geometry: per layer/component, R2 to CONSTRAINED (1-layer forward-prop) vs FULL Bayesian belief, per factor.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import whitebox, probes  # noqa: E402
from factors import mess3_factor, asym3_factor, make_world, decode_subtokens, belief_filter  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
LAYERS = (["embed"] +
          sum([[f"L{i}.attn_out", f"L{i}.resid_mid", f"L{i}.mlp_out", f"L{i}.resid_post"] for i in range(4)], [])
          + ["final_ln"])


def anova2(M):
    """M (N,9) over tokens 3*z0+z1. Fraction of variance explained by the additive (factored)
    model f0[z0]+f1[z1]+c, per row. Returns mean fraction explained."""
    V = M.reshape(-1, 3, 3)
    mu = V.mean((1, 2), keepdims=True)
    U = V.mean(2, keepdims=True) - mu
    Wb = V.mean(1, keepdims=True) - mu
    resid = V - (mu + U + Wb)
    tot = ((V - mu) ** 2).sum((1, 2))
    fe = 1 - (resid ** 2).sum((1, 2)) / np.clip(tot, 1e-30, None)
    return float(np.mean(fe))


def constrained_belief(factor, sub):
    """Single-layer parallel forward-propagated belief (Piotrowski et al. Eq 27), per factor.
    r_t = pi + sum_{s<=t} (pi @ Tcond[z_s] @ M^{t-s} - pi)."""
    B, L = sub.shape; I = 3
    pi, M = factor.pi, factor.M
    rowsum = factor.T.sum(2)                                   # (z,i)=P(z|i)
    Tcond = factor.T / np.clip(rowsum[:, :, None], 1e-30, None)  # (z,i,j) row-normalized
    Mpow = [np.eye(I)]
    for _ in range(L):
        Mpow.append(Mpow[-1] @ M)
    out = np.empty((B, L, I))
    piT = np.einsum("i,zij->zj", pi, Tcond)                    # (z,j) = pi @ Tcond[z]
    for t in range(L):
        r = np.repeat(pi[None], B, 0).astype(float)
        for s in range(t + 1):
            out_s = piT[sub[:, s]] @ Mpow[t - s]               # (B,I)
            r = r + out_s - pi
        out[:, t] = r
    return out


def main():
    facs = [mess3_factor(0.6, 0.15), asym3_factor()]
    W = whitebox.load_weights(BASE + "/runs/uni_mess3_asym3.pt")
    rng = np.random.default_rng(0)
    world = make_world(facs, eps=0.0)
    toks, _ = world.sample(256, 64, rng)
    acts, attns = whitebox.forward(W, toks)

    print("=" * 70)
    print("[1] EMBEDDING FACTORING  (tok.weight = mu + U[z0] + W[z1]?)")
    emb = W["tok.weight"]
    fe_emb = anova2(emb.T)            # treat each of 128 dims as a 'row' over the 9 tokens
    s = np.linalg.svd(emb - emb.mean(0), compute_uv=False)
    print(f"    additive frac-var explained across 128 dims = {fe_emb:.4f}")
    print(f"    centered-embedding singular values: {np.round(s[:7], 2)}")

    print("=" * 70)
    print("[2] LOGIT READ-OFF")
    logits = acts["logits"].reshape(-1, 9)
    print(f"    logits additive over (z0,z1): frac explained = {anova2(logits):.4f}  (1.0 = factored product)")
    # marginals vs belief-implied next-token
    P = np.exp(logits - logits.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    Pz0 = P.reshape(-1, 3, 3).sum(2); Pz1 = P.reshape(-1, 3, 3).sum(1)
    sub = decode_subtokens(toks, 2)
    full = [belief_filter(f.T, sub[..., n], f.pi) for n, f in enumerate(facs)]
    belnext = [(full[n].reshape(-1, 3) @ facs[n].E) for n in range(2)]   # belief @ E = P(next subtoken)
    for n, (Pm, bn, nm) in enumerate([(Pz0, belnext[0], facs[0].name), (Pz1, belnext[1], facs[1].name)]):
        rmse = np.sqrt(((Pm - bn) ** 2).mean())
        print(f"    factor {n} ({nm}): model next-subtoken marginal vs belief-implied  RMSE={rmse:.4f}")

    print("=" * 70)
    print("[3] BELIEF-UPDATE GEOMETRY: R2 to [constrained | full] per layer/component, per factor")
    constr = [constrained_belief(f, sub[..., n]) for n, f in enumerate(facs)]
    for n in range(2):
        print(f"  --- factor {n} ({facs[n].name}) ---     [constrained | full]")
        for lname in LAYERS:
            A = acts[lname].reshape(-1, 128)
            _, _, r2c = probes.ridge_fit(A, constr[n].reshape(-1, 3))
            _, _, r2f = probes.ridge_fit(A, full[n].reshape(-1, 3))
            tag = "  <-- attn builds constrained" if (r2c > r2f and ".attn_out" in lname) else ""
            print(f"    {lname:16s}  {r2c:6.3f} | {r2f:6.3f}{tag}")

    print("=" * 70)
    print("[3b] ATTENTION STRUCTURE (mean attention by distance Δ) + FACTOR ORTHOGONALITY")
    L = toks.shape[1]
    for i in range(4):
        A = attns[i]                                   # (B,nh,L,L)
        for h in range(A.shape[1]):
            decay = [float(np.mean([A[:, h, t, t - D] for t in range(D, L)])) for D in range(8)]
            print(f"    L{i}h{h}  Δ=0..7: {np.round(decay, 3)}")
    A3 = acts["L3.resid_post"].reshape(-1, 128)
    W0, _, _ = probes.ridge_fit(A3, full[0].reshape(-1, 3))
    W1, _, _ = probes.ridge_fit(A3, full[1].reshape(-1, 3))
    onb = lambda Wm: np.linalg.svd(Wm, full_matrices=False)[0][:, :2]
    pa = np.linalg.svd(onb(W0).T @ onb(W1), compute_uv=False)
    print(f"    factor0/factor1 belief-subspace overlap (cos principal angles) = {np.round(pa, 3)}  (0=orthogonal)")

    print("=" * 70)
    print("[3c] REDUCED SIMULATOR: does a factored exact-Bayes filter reproduce the model's output?")
    ml = acts["logits"].reshape(-1, 9)
    model_p = np.exp(ml - ml.max(1, keepdims=True)); model_p /= model_p.sum(1, keepdims=True)
    pz0 = full[0].reshape(-1, 3) @ facs[0].E
    pz1 = full[1].reshape(-1, 3) @ facs[1].E
    filt_p = (pz0[:, :, None] * pz1[:, None, :]).reshape(-1, 9)        # independent -> product
    kl = (model_p * np.log(np.clip(model_p, 1e-30, None) / np.clip(filt_p, 1e-30, None))).sum(1)
    print(f"    KL(model || factored-Bayes filter) = {kl.mean():.4f} nats  (0 = model IS the factored filter)")


if __name__ == "__main__":
    main()
