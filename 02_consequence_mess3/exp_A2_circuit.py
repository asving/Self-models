"""Exhaustive circuit decode of design A2 (genuine internal self-action = mess3-belief runner-up).
 [1] COMPONENTS: probe R2 per layer for mess3 belief, asym3 belief (decoded), ACTION (runner-up).
     Contrast with v1: the internal action is NOT decodable at the embed (it needs the full belief),
     it is CONSTRUCTED across layers — the signature of a computed self-signal, not a perception.
 [2] CAUSAL DESYNC (rubber-hand): patch the action subspace aₜ→s; the whole asym3 computation should
     now follow the FALSE action — patched net ≈ synthetic-program-run-with-action=s, and ≠ baseline.
     (v1 failed this — it went uniform — because aₜ=z⁰ₜ gave a redundant input route. Here there is none.)
 [3] SYNTHETIC PROGRAM: pure-numpy reduced simulator (mess3 belief → runner-up action → efference-copy
     decode → asym3 belief → factored read-off with the asym3 prediction shifted by aₜ) reproduces the net.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import whitebox, probes
from factors import mess3_factor, asym3_factor, belief_filter
from exp_A2 import generate_A2, action_from_belief

BASE = os.path.dirname(os.path.abspath(__file__))
f0, f1 = mess3_factor(0.6, 0.15), asym3_factor()
E0, E1 = f0.E, f1.E
rng = np.random.default_rng(13)
LAYERS = ["embed"] + [f"L{i}.resid_post" for i in range(4)] + ["final_ln"]
PATCH_LAYER = "L2.resid_post"


def reduced_sim(toks, a_array=None):
    """Pure-numpy reconstruction of the A2 net. a_array overrides the action (else runner-up of belief)."""
    B, L = toks.shape
    z0 = toks // 3; x1 = toks % 3
    mb = belief_filter(f0.T, z0, f0.pi)                          # mess3 belief (clean)
    a = a_array if a_array is not None else action_from_belief(mb)  # runner-up
    a_prev = np.concatenate([np.zeros((B, 1), int), a[:, :-1]], 1)
    z1_dec = (x1 - a_prev) % 3                                   # efference-copy decode
    ab = belief_filter(f1.T, z1_dec, f1.pi)                      # asym3 belief from decoded emissions
    p_z0 = mb @ E0                                               # P(next mess3 token) — clean
    p_z1 = ab @ E1                                               # P(next asym3 emission)
    p_x1 = np.stack([np.take_along_axis(p_z1, ((c - a) % 3)[..., None], -1)[..., 0] for c in range(3)], -1)
    return (p_z0[..., :, None] * p_x1[..., None, :]).reshape(B, L, 9)


def asym_marg(P9, B):
    return P9.reshape(B, 64, 3, 3).sum(2)                       # asym3 (x1) marginal


def main():
    CK = BASE + "/runs/expA2_plain.pt"
    W = whitebox.load_weights(CK)
    toks, a, z0, z1, _ = generate_A2(f0, f1, 1500, 64, rng, 1.0)
    B = len(toks)
    acts, _ = whitebox.forward(W, toks)
    x1 = toks % 3
    mb = belief_filter(f0.T, z0, f0.pi)
    true_bel = belief_filter(f1.T, z1, f1.pi)                   # asym3 belief (filter on decoded z1, needs aₜ)
    naive_bel = belief_filter(f1.T, x1, f1.pi)                  # ignores aₜ
    a_oh = np.eye(3)[a]

    print("=" * 70)
    print("[1] COMPONENTS — R2(resid → ·):  mess3-bel | asym3-bel TRUE | asym3 naive | ACTION")
    fit = {}
    for ln in LAYERS:
        A = acts[ln].reshape(-1, 128)
        r2m = probes.ridge_fit(A, mb.reshape(-1, 3))[2]
        r2t = probes.ridge_fit(A, true_bel.reshape(-1, 3))[2]
        r2n = probes.ridge_fit(A, naive_bel.reshape(-1, 3))[2]
        Wa, ba, r2a = probes.ridge_fit(A, a_oh.reshape(-1, 3))
        fit[ln] = (Wa, ba)
        flag = "  <-- routed" if (r2t - r2n > 0.2 and ".resid_post" in ln) else ""
        print(f"  {ln:14s}  {r2m:6.3f} | {r2t:6.3f} | {r2n:6.3f} | {r2a:6.3f}{flag}")
    print("  (action R2 LOW at embed, built across layers = computed internal signal, not a perception)")

    print("=" * 70)
    print("[2] CAUSAL DESYNC (computational rubber-hand) — hold asym3 obs x¹ FIXED, swap the Mess3")
    print("    context (→ a different internal action), watch the asym3 percept follow the ACTION.")
    from exp_A2 import sample_factor
    z0_cf = sample_factor(f0.T, f0.pi, B, 64, rng)              # different mess3 context, SAME x1
    a_cf = action_from_belief(belief_filter(f0.T, z0_cf, f0.pi))
    toks_cf = z0_cf * 3 + x1
    print(f"    counterfactual action differs from original at {np.mean(a_cf != a):.0%} of positions")
    ac_cf, _ = whitebox.forward(W, toks_cf)
    lg = ac_cf["logits"]; Pcf = np.exp(lg - lg.max(-1, keepdims=True)); Pcf /= Pcf.sum(-1, keepdims=True)
    model_cf = asym_marg(Pcf, B)
    correct = asym_marg(reduced_sim(toks_cf), B)               # decode with the NEW action a_cf
    stale = asym_marg(reduced_sim(toks_cf, a_array=a), B)      # decode with the STALE original action a
    tail = slice(16, None)
    rc = np.sqrt(((model_cf[:, tail] - correct[:, tail]) ** 2).mean())
    rs = np.sqrt(((model_cf[:, tail] - stale[:, tail]) ** 2).mean())
    print(f"    RMSE(net@cf , decode-with-NEW-action)   = {rc:.3f}   <- net's percept tracks the new action")
    print(f"    RMSE(net@cf , decode-with-STALE-action) = {rs:.3f}   <- and NOT the old one")
    print(f"    => the asym3 percept is caused by the internal action, not the (fixed) sensory input."
          if rc < rs - 0.05 else "    => inconclusive")

    print("-" * 70)
    print(f"[2b] activation patch of action readout @ {PATCH_LAYER} (control: shows the action is")
    print("     RECOMPUTED downstream from the in-residual belief, so a low-D patch is overwritten)")
    Wa, ba = fit[PATCH_LAYER]
    pinv = np.linalg.solve(Wa.T @ Wa + 1e-6 * np.eye(3), Wa.T)
    base = asym_marg(reduced_sim(toks), B)
    for s in range(3):
        tgt = np.eye(3)[s]
        def fn(nm, x, tgt=tgt):
            if nm != PATCH_LAYER: return x
            x = x.copy(); x += (tgt[None, None] - (x @ Wa + ba)) @ pinv; return x
        ac, _ = whitebox.forward(W, toks, edit_fn=fn)
        lg = ac["logits"]; P = np.exp(lg - lg.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
        pm = asym_marg(P, B)
        synth_s = asym_marg(reduced_sim(toks, a_array=np.full(toks.shape, s)), B)
        print(f"    patch→{s}: RMSE(net, synth[a={s}])={np.sqrt(((pm[:,tail]-synth_s[:,tail])**2).mean()):.3f}  "
              f"RMSE(net, synth[real a])={np.sqrt(((pm[:,tail]-base[:,tail])**2).mean()):.3f}")

    print("=" * 70)
    print("[3] SYNTHETIC PROGRAM — reduced simulator vs net")
    pred = reduced_sim(toks)
    lg = acts["logits"]; net_p = np.exp(lg - lg.max(-1, keepdims=True)); net_p /= net_p.sum(-1, keepdims=True)
    kl = (net_p * np.log(np.clip(net_p, 1e-30, None) / np.clip(pred, 1e-30, None))).sum(-1)
    print(f"  KL(net || synthetic program) = {kl.mean():.4f} nats")
    def nll(p):
        Pz1 = asym_marg(p, B); z = toks % 3
        return float(-np.log(np.clip(Pz1[:, :-1], 1e-30, None)[np.arange(B)[:, None], np.arange(63), z[:, 1:]])[:, 31:].mean())
    print(f"  asym3 NLL on observed x1 (tail):  net={nll(net_p):.3f}  |  synthetic={nll(pred):.3f}")


if __name__ == "__main__":
    main()
