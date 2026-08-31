"""Two-trajectory A2: half the sequences are corrupted (action consequential, x¹=z¹+aₜ), half are
clean (action inert, x¹=z¹), type hidden. The model must INFER in-context whether its own action is
consequential and apply the efference copy CONDITIONALLY. Tests:
 [1] in-context self-consequentiality inference: is 'corrupted?' linearly decodable, rising with position.
 [2] conditional routing: in corrupted seqs the net matches the WITH-ACTION simulator (decode z¹=x¹−aₜ);
     in clean seqs it matches the NAIVE simulator (decode z¹=x¹, action ignored). Cross-matches fail.
 [3] conditional rubber-hand control: swap the Mess3 context (→ different action) holding x¹ fixed —
     in CLEAN seqs the percept must NOT move (the model correctly judges its action inert).
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import whitebox, probes
from factors import mess3_factor, asym3_factor
from exp_A2 import generate_A2, action_from_belief, sample_factor
from exp_A2_circuit import reduced_sim, asym_marg

BASE = os.path.dirname(os.path.abspath(__file__))
f0, f1 = mess3_factor(0.6, 0.15), asym3_factor()
rng = np.random.default_rng(21)


def rmse(a, b, sel, tail=slice(16, None)):
    return np.sqrt(((a[sel][:, tail] - b[sel][:, tail]) ** 2).mean())


def main():
    W = whitebox.load_weights(BASE + "/runs/expA2_two.pt")
    toks, a, z0, z1, corrupt = generate_A2(f0, f1, 2500, 64, rng, 0.5)
    B = len(toks); x1 = toks % 3
    acts, _ = whitebox.forward(W, toks)
    lg = acts["logits"]; net_p = np.exp(lg - lg.max(-1, keepdims=True)); net_p /= net_p.sum(-1, keepdims=True)
    model_m = asym_marg(net_p, B)
    with_action = asym_marg(reduced_sim(toks), B)                       # decode z¹=x¹−aₜ (corrupted mode)
    naive = asym_marg(reduced_sim(toks, np.zeros(toks.shape, int)), B)  # decode z¹=x¹    (clean mode)
    print(f"two-trajectory A2 | {corrupt.mean():.0%} corrupted")

    print("=" * 66)
    print("[1] IN-CONTEXT self-consequentiality inference — R²(L3 resid → is-corrupted)")
    A3 = acts["L3.resid_post"]; y = corrupt.astype(float)[:, None]
    for p in [2, 4, 8, 16, 32, 48]:
        print(f"  pos {p:2d}: R²={probes.ridge_fit(A3[:, p, :], y)[2]:.3f}")

    print("=" * 66)
    print("[2] CONDITIONAL ROUTING — RMSE(net asym3 , simulator), by sequence type")
    print("                          WITH-ACTION sim | NAIVE sim   (lower = the mode the net uses)")
    for nm, sel in [("CORRUPTED (action live)", corrupt), ("CLEAN (action inert)  ", ~corrupt)]:
        ra, rn = rmse(model_m, with_action, sel), rmse(model_m, naive, sel)
        pick = "WITH-ACTION" if ra < rn else "NAIVE"
        print(f"  {nm}:   {ra:.3f}        |  {rn:.3f}     -> net uses {pick}")

    print("=" * 66)
    print("[3] CONDITIONAL RUBBER-HAND — swap Mess3 context (→ new action), hold x¹ fixed")
    z0_cf = sample_factor(f0.T, f0.pi, B, 64, rng)
    a_cf = action_from_belief(__import__("factors").belief_filter(f0.T, z0_cf, f0.pi))
    toks_cf = z0_cf * 3 + x1
    ac, _ = whitebox.forward(W, toks_cf)
    lg = ac["logits"]; Pcf = np.exp(lg - lg.max(-1, keepdims=True)); Pcf /= Pcf.sum(-1, keepdims=True)
    cf_m = asym_marg(Pcf, B)
    moved = np.sqrt(((cf_m - model_m) ** 2)[:, 16:].mean(-1).mean(-1))   # per-seq percept shift from swap
    print(f"  asym3 percept shift from the context swap (mean RMSE):")
    print(f"    CLEAN seqs (action inert)      : {moved[~corrupt].mean():.3f}   <- should be ~0 (action ignored)")
    print(f"    CORRUPTED seqs (action live)   : {moved[corrupt].mean():.3f}   <- larger (action routed)")
    print(f"  action differs after swap at {np.mean(a_cf != a):.0%} of positions")


if __name__ == "__main__":
    main()
