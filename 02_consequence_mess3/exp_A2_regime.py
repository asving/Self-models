"""Q2/Q3 (two-trajectory): does the model hold a graded BELIEF over the regime (corrupted vs clean),
and does that belief GATE the action->asym3 routing?
Exact Bayes regime posterior: two hypotheses for the asym3 sub-stream x¹ —
  clean:   x¹ is asym3 emissions directly;     corrupt: decoded z¹=(x¹−aₜ₋₁) is asym3 emissions.
Running log-likelihood-ratio -> posterior p(corrupt | obs_{≤t}).
 [Q2] probe the residual for that CONTINUOUS posterior (not just the binary label), per position.
 [Q3] GATING: the Bayes-optimal predictive is  post·(with-action) + (1−post)·(naive).  Estimate the
      empirical action-routing GAIN g per (seq,pos) by projecting the net's asym3 prediction onto the
      (naive→with-action) axis; test g ≈ post  =>  the regime belief soft-gates the routing.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import whitebox, probes
from factors import mess3_factor, asym3_factor, belief_filter
from exp_A2 import generate_A2, action_from_belief
from exp_A2_circuit import reduced_sim, asym_marg

BASE = os.path.dirname(os.path.abspath(__file__))
f0, f1 = mess3_factor(0.6, 0.15), asym3_factor()
rng = np.random.default_rng(31)


def pred_loglik(f, em):
    """log P(em_t | em_{<t}) under HMM f, per position. (B,L)."""
    T, E, pi = f.T, f.E, f.pi
    B, L = em.shape
    b = np.repeat(pi[None], B, 0).astype(float)
    out = np.empty((B, L))
    for t in range(L):
        pred = b @ E
        out[:, t] = np.log(np.clip(pred[np.arange(B), em[:, t]], 1e-30, None))
        bn = np.einsum("bi,bij->bj", b, T[em[:, t]]); bn /= np.clip(bn.sum(1, keepdims=True), 1e-300, None); b = bn
    return out


def main():
    W = whitebox.load_weights(BASE + "/runs/expA2_two.pt")
    toks, a, z0, z1, corrupt = generate_A2(f0, f1, 3000, 64, rng, 0.5)
    B = len(toks); x1 = toks % 3
    a_prev = np.concatenate([np.zeros((B, 1), int), a[:, :-1]], 1)
    # exact Bayes regime posterior p(corrupt | obs_{<=t})
    llr = np.cumsum(pred_loglik(f1, (x1 - a_prev) % 3) - pred_loglik(f1, x1), 1)
    post = 1.0 / (1.0 + np.exp(-llr))
    acts, _ = whitebox.forward(W, toks)
    print(f"two-traj A2 | Bayes regime posterior at pos48: corrupt-seqs={post[corrupt,48].mean():.2f} "
          f"clean-seqs={post[~corrupt,48].mean():.2f}")

    print("=" * 70)
    print("[Q2] does the residual encode the GRADED regime posterior? R²(L3 → ·) per position")
    A3 = acts["L3.resid_post"]
    print("    pos |  R²→Bayes-posterior(graded) |  R²→binary-label")
    for p in [2, 4, 8, 16, 32, 48]:
        r2p = probes.ridge_fit(A3[:, p, :], post[:, p, None])[2]
        r2b = probes.ridge_fit(A3[:, p, :], corrupt.astype(float)[:, None].repeat(64, 1)[:, p, None])[2]
        print(f"    {p:2d}  |          {r2p:.3f}             |     {r2b:.3f}")

    print("=" * 70)
    print("[Q3] GATING — action-routing gain g vs Bayes regime posterior")
    lg = acts["logits"]; netp = np.exp(lg - lg.max(-1, keepdims=True)); netp /= netp.sum(-1, keepdims=True)
    model_m = asym_marg(netp, B)
    naive = asym_marg(reduced_sim(toks, np.zeros(toks.shape, int)), B)
    wa = asym_marg(reduced_sim(toks), B)
    d = wa - naive
    g = ((model_m - naive) * d).sum(-1) / np.clip((d * d).sum(-1), 1e-9, None)  # routing gain per (seq,pos)
    mask = (d * d).sum(-1) > 0.05                                  # only where action actually shifts the pred
    gf, pf = g[mask], post[mask]
    print(f"    corr(g, posterior) = {np.corrcoef(gf, pf)[0,1]:.3f}   (Bayes-optimal soft gating predicts g≈post)")
    sl = np.polyfit(pf, gf, 1)
    print(f"    g ≈ {sl[0]:.2f}·post + {sl[1]:.2f}")
    print("    posterior bin |  mean gain g  |  n")
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pf >= lo) & (pf < hi)
        if m.sum(): print(f"     [{lo:.1f},{hi:.1f})     |    {gf[m].mean():.3f}     | {m.sum()}")


if __name__ == "__main__":
    main()
