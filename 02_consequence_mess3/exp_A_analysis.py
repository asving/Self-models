"""Design A routing analysis. The action aₜ corrupts asym3 (x¹=z¹⊕aₜ). To track asym3 the model
must decode z¹=(x¹−aₜ)%3 — i.e. ROUTE its own action into the asym3 belief update. Signature:
the residual should decode the TRUE asym3 belief (filter on decoded z¹, needs aₜ) far better than
the NAIVE belief (filter on raw corrupted x¹, ignores aₜ). For two-trajectory, also: is the
'is-corrupted' trajectory type linearly decodable (in-context self-consequentiality inference),
and is the true belief recovered specifically in corrupted sequences?"""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import whitebox, probes  # noqa
from factors import mess3_factor, asym3_factor, belief_filter  # noqa
import exp_A  # generate_A  # noqa

BASE = os.path.dirname(os.path.abspath(__file__))
f0, f1 = mess3_factor(0.6, 0.15), asym3_factor()
rng = np.random.default_rng(9)
LAYERS = ["embed"] + [f"L{i}.resid_post" for i in range(4)] + ["final_ln"]


def analyze(tag, ckpt, corrupt_frac):
    W = whitebox.load_weights(ckpt)
    toks, a, z1, corrupt = exp_A.generate_A(f0, f1, 1500, 64, rng, corrupt_frac)
    acts, _ = whitebox.forward(W, toks)
    x1 = toks % 3
    true_bel = belief_filter(f1.T, z1, f1.pi)                  # filter on DECODED z1 (needs aₜ)
    naive_bel = belief_filter(f1.T, x1, f1.pi)                 # filter on raw corrupted x1 (ignores aₜ)
    print(f"\n=== {tag} (corrupt_frac={corrupt_frac}) ===  R²(resid → asym3 belief) [TRUE | naive]")
    sel = corrupt if corrupt_frac < 1.0 else np.ones(len(toks), bool)
    for ln in LAYERS:
        A = acts[ln][sel].reshape(-1, 128)
        _, _, r2t = probes.ridge_fit(A, true_bel[sel].reshape(-1, 3))
        _, _, r2n = probes.ridge_fit(A, naive_bel[sel].reshape(-1, 3))
        tag2 = "  <-- routed (true≫naive)" if (r2t - r2n > 0.2 and ".resid_post" in ln) else ""
        print(f"  {ln:14s}  {r2t:6.3f} | {r2n:6.3f}{tag2}")
    if corrupt_frac < 1.0:
        # in-context inference: is the 'corrupted?' trajectory type linearly decodable per position?
        A3 = acts["L3.resid_post"]                              # (B,L,128)
        y = corrupt.astype(float)[:, None].repeat(64, 1)        # (B,L) constant per sequence
        for p in [4, 16, 48]:
            _, _, r2c = probes.ridge_fit(A3[:, p, :], y[:, p][:, None])
            print(f"  is-corrupted decodable at pos {p:2d}: R²={r2c:.3f}")
        # belief recovery split by type
        for nm, s in [("corrupted seqs", corrupt), ("clean seqs", ~corrupt)]:
            A = acts["L3.resid_post"][s].reshape(-1, 128)
            _, _, rt = probes.ridge_fit(A, true_bel[s].reshape(-1, 3))
            print(f"  true-belief R² [{nm}] = {rt:.3f}")


for tag, ck, cf in [("PLAIN A", BASE + "/runs/expA_plain.pt", 1.0),
                    ("TWO-TRAJECTORY A", BASE + "/runs/expA_two.pt", 0.5)]:
    if os.path.exists(ck):
        analyze(tag, ck, cf)
    else:
        print(f"{tag}: {ck} not found yet")
