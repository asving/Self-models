"""Exhaustive circuit decode of design A (plain). Three parts:
 [1] COMPONENTS: linear-probe R2 per layer for (mess3 belief, asym3 belief, ACTION aₜ) + orthogonality.
 [2] CAUSAL DESYNC (rubber-hand): patch the action subspace aₜ->s and verify the asym3 prediction
     shifts to (asym3 emission ⊕ s) — the model decodes/predicts using its OWN (patched) action.
 [3] SYNTHETIC PROGRAM: a pure-numpy reduced simulator (two belief filters + action argmax + the
     efference-copy decode/shift) that reproduces the net's next-token distribution (KL).
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import whitebox, probes
from factors import mess3_factor, asym3_factor, belief_filter
import exp_A

BASE = os.path.dirname(os.path.abspath(__file__))
f0, f1 = mess3_factor(0.6, 0.15), asym3_factor()
E0, E1 = f0.E, f1.E
rng = np.random.default_rng(13)
LAYERS = ["embed"] + [f"L{i}.resid_post" for i in range(4)] + ["final_ln"]
CK = BASE + "/runs/expA_plain.pt"


# ---------- the synthetic program (reduced mechanistic simulator) ----------
def reduced_sim(toks):
    """Pure-numpy reconstruction of the A net's algorithm. Returns next-token dist (B,L,9)."""
    B, L = toks.shape
    z0 = toks // 3; x1 = toks % 3
    mb = belief_filter(f0.T, z0, f0.pi)                 # mess3 belief (state emitting t+1)
    a = (mb @ E0).argmax(-1)                              # ACTION aₜ = argmax predicted next mess3 token
    a_prev = np.concatenate([np.zeros((B, 1), int), a[:, :-1]], 1)
    z1_dec = (x1 - a_prev) % 3                            # EFFERENCE-COPY decode: z¹=(x¹−aₜ₋₁)%3
    ab = belief_filter(f1.T, z1_dec, f1.pi)              # asym3 belief from the DECODED emissions
    p_z0 = mb @ E0                                        # P(next mess3 token)
    p_z1 = ab @ E1                                        # P(next asym3 emission)
    p_x1 = np.stack([np.take_along_axis(p_z1, ((c - a) % 3)[..., None], -1)[..., 0] for c in range(3)], -1)
    return (p_z0[..., :, None] * p_x1[..., None, :]).reshape(B, L, 9)   # factored product


def main():
    W = whitebox.load_weights(CK)
    toks, a, z1, _ = exp_A.generate_A(f0, f1, 1500, 64, rng, 1.0)
    acts, _ = whitebox.forward(W, toks)
    z0 = toks // 3; x1 = toks % 3
    mb = belief_filter(f0.T, z0, f0.pi); ab = belief_filter(f1.T, z1, f1.pi)
    a_oh = np.eye(3)[a]

    print("=" * 64)
    print("[1] COMPONENTS — R2(resid → ·) per layer:  mess3-bel | asym3-bel | ACTION")
    fits_act = None
    for ln in LAYERS:
        A = acts[ln].reshape(-1, 128)
        r2m = probes.ridge_fit(A, mb.reshape(-1, 3))[2]
        r2a = probes.ridge_fit(A, ab.reshape(-1, 3))[2]
        Wa, ba, r2act = probes.ridge_fit(A, a_oh.reshape(-1, 3))
        print(f"  {ln:14s}  {r2m:6.3f} | {r2a:6.3f} | {r2act:6.3f}")
        if ln == "L2.resid_post":
            fits_act = (Wa, ba)
    onb = lambda M: np.linalg.svd(M, full_matrices=False)[0][:, :2]
    A2 = acts["L2.resid_post"].reshape(-1, 128)
    Wm = probes.ridge_fit(A2, mb.reshape(-1, 3))[0]
    Wab = probes.ridge_fit(A2, ab.reshape(-1, 3))[0]
    Wa = fits_act[0]
    print(f"  subspace overlap (cos princ. angles) @L2:  action·asym3-belief={np.round(np.linalg.svd(onb(Wa).T@onb(Wab),compute_uv=False),2)}  "
          f"action·mess3-belief={np.round(np.linalg.svd(onb(Wa).T@onb(Wm),compute_uv=False),2)}")

    print("=" * 64)
    print("[2] CAUSAL DESYNC — patch action subspace aₜ→s; does the asym3 prediction follow (emit⊕s)?")
    Wa, ba = fits_act
    pinv = np.linalg.solve(Wa.T @ Wa, Wa.T)
    def patched_pred(s):
        tgt = np.eye(3)[s]
        def fn(nm, x):
            if nm != "L2.resid_post": return x
            x = x.copy(); cur = x @ Wa + ba
            x += ((tgt[None, None] - cur) @ pinv)
            return x
        ac, _ = whitebox.forward(W, toks, edit_fn=fn)
        lg = ac["logits"]; P = np.exp(lg - lg.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
        return P.reshape(toks.shape[0], 64, 3, 3).sum(2)        # predicted asym3 (x1) marginal
    base_emit = (belief_filter(f1.T, (x1 - np.concatenate([np.zeros((toks.shape[0],1),int), a[:,:-1]],1)) % 3, f1.pi) @ E1)
    for s in range(3):
        px1 = patched_pred(s)
        # expected if model uses patched action s: predicted x1 = asym3-emit cyclically shifted by s
        exp_shift = np.stack([base_emit[..., (c - s) % 3] for c in range(3)], -1)
        tail = slice(32, None)
        rmse = np.sqrt(((px1[:, tail] - exp_shift[:, tail]) ** 2).mean())
        print(f"  patch action→{s}: RMSE(predicted asym3 , emit⊕{s}) = {rmse:.3f}   "
              f"mean pred={np.round(px1[:,tail].reshape(-1,3).mean(0),2)}")

    print("=" * 64)
    print("[3] SYNTHETIC PROGRAM — reduced simulator vs net")
    pred = reduced_sim(toks)
    lg = acts["logits"]; net_p = np.exp(lg - lg.max(-1, keepdims=True)); net_p /= net_p.sum(-1, keepdims=True)
    kl = (net_p * np.log(np.clip(net_p, 1e-30, None) / np.clip(pred, 1e-30, None))).sum(-1)
    print(f"  KL(net || synthetic program) = {kl.mean():.4f} nats   (next-token match)")
    print(f"  net asym3 NLL vs synthetic asym3 NLL on observed x1 (tail): ", end="")
    def asym3_nll(p):
        Pz1 = p.reshape(-1, 64, 3, 3).sum(2)
        z = (toks % 3)
        return float(-np.log(np.clip(Pz1[:, :-1], 1e-30, None)[np.arange(len(toks))[:, None], np.arange(63), z[:, 1:]])[:, 31:].mean())
    print(f"{asym3_nll(net_p):.3f} | {asym3_nll(pred):.3f}")


if __name__ == "__main__":
    main()
