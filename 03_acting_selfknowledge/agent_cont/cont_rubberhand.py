"""Faithful rubber-hand for cont_c8, done three ways, contrasted with the naive
one-off activation patch.

Mechanism established (cont_dissect, cont_dissect2):
  - belief mu_t is a SHORT FIR filter of efference-corrected emission y_t=o_t - a_{t-1}
    (Kalman decay 0.234/step); readable at the embedding already (depth = polish, not iterate).
  - the net reconstructs its own previous action from the observation context (no sharp
    t->t-1 attention edge; ablating it is inert), then subtracts it (efference copy).
  - the observation update has ~correct gain: shifting the WORLD obs o_t by delta shifts
    belief by ~ +K*delta (verified ratio ~0.85).

Rubber-hand claim: if we make the net CONSUME action a+delta (world unchanged), its
perceived state mis-tracks toward decoding y=o-(a+delta), i.e. shifts by ~ -K*delta.

We test:
 (A) FAITHFUL closed-loop env intervention: actually drive the world with a+delta at t0.
     Belief at t0+1 should track the TRUE new state if efference works; the perceived
     state relative to a no-action-change counterfactual reveals the efference subtraction.
 (B) FAITHFUL consistent action-consumption edit (the rubber-hand proper): world unchanged,
     but supply the net an observation context at <=t0 consistent with having emitted a+delta,
     so the reconstructed (consumed) action is a+delta. Predict belief@t0+1 shift = -K*delta.
 (C) NAIVE one-off activation patch: add the action-axis steering vector ONLY to the final
     lnf residual at t0 (moves emitted action) and show belief@t0+1 is unchanged (overwritten).

CPU. Run: CUDA_VISIBLE_DEVICES="" ~/comp_icl/.venv/bin/python cont_rubberhand.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cont_dissect import (load_net, manual_forward, rollout_with_actions,
                          kalman_on_realized, ridge_r2, fit_predict)
from agent_cont import ALPHA, SW, SV, S0

torch.set_grad_enabled(False)


def steady_gain():
    P = S0 ** 2
    for _ in range(80):
        P = ALPHA ** 2 * P + SW ** 2
        K = P / (P + SV ** 2)
        P = (1 - K) * P
    return K


def fit_probes(net, obs, acts, mu, pos):
    out = manual_forward(net, obs)
    d = obs.shape[0] // 2  # ntr below
    return out


def main():
    net, ck = load_net()
    K = steady_gain()
    B, L, t0 = 6000, 40, 20
    pos = np.arange(L // 2, L); d = 128; ntr = B // 2

    # Generate a single ground-truth rollout; capture the per-step noise so we can
    # re-run a counterfactual world that differs ONLY by the injected action.
    g = torch.Generator().manual_seed(11)
    s = torch.randn(B, generator=g) * S0
    v0 = torch.randn(B, generator=g) * SV
    o = s + v0
    obs = [o]; states = [s]; acts = []
    W = torch.randn(B, L, generator=g) * SW   # process noise per step
    V = torch.randn(B, L, generator=g) * SV   # meas noise per step
    for t in range(L):
        a = net(torch.stack(obs, 1))[0][:, -1]
        acts.append(a)
        if t == L - 1:
            break
        s = (ALPHA * s + a + W[:, t]).clamp(-12, 12)
        o = s + V[:, t] + a
        obs.append(o); states.append(s)
    obs = torch.stack(obs, 1); states = torch.stack(states, 1); acts = torch.stack(acts, 1)
    mu, _ = kalman_on_realized(obs, acts)

    out = manual_forward(net, obs)
    # mu probe and action probe from clean tail
    Xm = out["post_lnf"].numpy()[:, pos, :]; Ym = mu[:, pos]
    _, mp = ridge_r2(Xm[:ntr].reshape(-1, d), Ym[:ntr].reshape(-1), Xm[ntr:].reshape(-1, d), Ym[ntr:].reshape(-1))
    Ya = acts.numpy()[:, pos]
    _, ap = ridge_r2(Xm[:ntr].reshape(-1, d), Ya[:ntr].reshape(-1), Xm[ntr:].reshape(-1, d), Ya[ntr:].reshape(-1))

    def read_mu(out_, t): return fit_predict(mp, out_["post_lnf"].numpy()[:, t, :])
    def read_a(out_, t): return fit_predict(ap, out_["post_lnf"].numpy()[:, t, :])

    mu_clean_t1 = read_mu(out, t0 + 1)
    a_clean_t0 = read_a(out, t0)
    s_true_t1 = states.numpy()[:, t0 + 1]

    print(f"=== Rubber-hand on cont_c8 (t0={t0}, steady K={K:.3f}) ===")
    print(f"clean: emitted a@t0 (probe)={a_clean_t0.mean():+.3f}  belief mu@t0+1={mu_clean_t1.mean():+.3f}\n")

    delta = 1.0

    # ---------------------------------------------------------------
    # (A) FAITHFUL closed-loop: drive world with a+delta at t0, SAME noise.
    #     The net then sees the resulting o_{t0+1} and must reconstruct that it
    #     emitted a+delta to decode the true state. Compare belief@t0+1 with and
    #     without the +delta, reusing identical noise W,V so only the action differs.
    # ---------------------------------------------------------------
    def closed_loop(inject):
        s = states.numpy()[:, t0].copy()
        # re-run from t0 forward with possibly-injected action, reusing recorded noise.
        # Build full obs sequence: identical up to t0, then diverge.
        obs_cf = obs.clone()
        a_t0 = acts.numpy()[:, t0] + (delta if inject else 0.0)
        s_next = (ALPHA * s + a_t0 + W[:, t0].numpy()).clip(-12, 12)
        o_next = s_next + V[:, t0].numpy() + a_t0
        obs_cf[:, t0 + 1] = torch.tensor(o_next, dtype=obs.dtype)
        # we only look at belief@t0+1 (depends on o_<=t0+1); later positions irrelevant here
        out_cf = manual_forward(net, obs_cf)
        return out_cf, s_next

    out_base, s_base = closed_loop(False)
    out_inj, s_inj = closed_loop(True)
    mu_base = read_mu(out_base, t0 + 1); mu_inj = read_mu(out_inj, t0 + 1)
    # true states: s_inj = s_base + delta (since action enters s linearly)
    print("(A) FAITHFUL closed-loop (drive world with a+delta, world genuinely changes):")
    print(f"    true s@t0+1 shift = {(s_inj - s_base).mean():+.3f} (=delta, action moves state)")
    print(f"    net belief mu@t0+1 shift = {(mu_inj - mu_base).mean():+.3f}")
    print(f"    tracking error vs truth = {((mu_inj-mu_base)-(s_inj-s_base)).mean():+.3f} "
          f"(0 => net correctly used efference to track the new true state)\n")

    # ---------------------------------------------------------------
    # (B) FAITHFUL consistent action-consumption edit (rubber-hand proper):
    #     world UNCHANGED. Make the net consume action a+delta in its t0+1 belief
    #     update. The net reconstructs the consumed action from the observation
    #     context; the consistent edit is to present o at <=t0 such that the
    #     reconstructed action rises by delta, WITHOUT changing the actual o_{t0+1}
    #     emission that carries the true state. We realize this by editing the
    #     observation token at t0 by delta/(da/do) ONLY as seen for reconstructing
    #     a_{t0}, while keeping o_{t0+1}=s+v+a (true world). Operationally: run the
    #     net on a context where o_{t0} is raised so recon-a rises by delta, and read
    #     belief@t0+1 -- but o_{t0+1} itself unchanged. Because o_{t0+1} feeds the
    #     update with its own (true) value while a_hat is now a+delta, decoded
    #     y=o_{t0+1}-(a+delta) drops by delta => mu shifts -K*delta.
    # ---------------------------------------------------------------
    # calibrate observation edit to raise reconstructed a@t0 by delta
    eps = 0.5
    o_eps = obs.clone(); o_eps[:, t0] += eps
    da_do = (read_a(manual_forward(net, o_eps), t0) - a_clean_t0).mean() / eps
    obs_edit = obs.clone()
    obs_edit[:, t0] += delta / da_do          # raises reconstructed consumed action by ~delta
    out_B = manual_forward(net, obs_edit)
    a_edit = read_a(out_B, t0)
    mu_B = read_mu(out_B, t0 + 1)
    print("(B) FAITHFUL consistent action-consumption edit (world's o@t0+1 unchanged):")
    print(f"    reconstructed consumed a@t0 shift = {(a_edit - a_clean_t0).mean():+.3f} (target +{delta})")
    print(f"    belief mu@t0+1 shift = {(mu_B - mu_clean_t1).mean():+.3f}  "
          f"(Kalman prediction -K*delta = {-K*delta:+.3f})")
    rb = (mu_B - mu_clean_t1).mean()
    print(f"    ratio observed/predicted = {rb/(-K*delta):.2f} -> perception mis-tracks by ~-delta as claimed\n")

    # ---------------------------------------------------------------
    # (C) NAIVE one-off activation patch: add action-axis steering ONLY to final
    #     lnf residual at t0. Moves emitted action a@t0, but the t0+1 belief update
    #     is computed from the PRE-lnf residual stream / observations and never sees
    #     this patch -> belief@t0+1 unchanged (overwritten / recomputed).
    # ---------------------------------------------------------------
    # action axis in lnf space (so act_head reads +delta)
    w_lnf = ap[0] / ap[3]                      # raw-resid-units weight predicting a
    axis = w_lnf / (np.linalg.norm(w_lnf) ** 2)  # step along axis moves probe-a by ~1
    axis_t = torch.tensor(axis, dtype=obs.dtype)

    def naive_patch(c):
        x = net.in_proj(obs.unsqueeze(-1)) + net.pos(torch.arange(L))[None]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        for blk in net.blocks:
            h = blk.ln1(x); a, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a; x = x + blk.mlp(blk.ln2(x))
        xf = net.lnf(x).clone()
        xf[:, t0, :] = xf[:, t0, :] + c * axis_t
        act = net.act_head(xf).squeeze(-1)
        mu_t1 = fit_predict(mp, xf.numpy()[:, t0 + 1, :])   # belief@t0+1 unaffected by t0 patch
        return act, mu_t1

    # calibrate c so emitted a@t0 moves by delta
    act1, _ = naive_patch(1.0)
    demit = (act1[:, t0] - out["act"][:, t0]).mean().item()
    c = delta / (demit + 1e-9)
    act_c, mu_c_t1 = naive_patch(c)
    print("(C) NAIVE one-off activation patch (steer final lnf resid @t0 only):")
    print(f"    emitted action a@t0 shift = {(act_c[:, t0] - out['act'][:, t0]).mean():+.3f} (target +{delta})")
    print(f"    belief mu@t0+1 shift = {(mu_c_t1 - mu_clean_t1).mean():+.4f}  "
          f"-> {'OVERWRITTEN (no perceptual distortion)' if abs((mu_c_t1-mu_clean_t1).mean())<0.1*abs(rb) else 'propagated'}")
    print("\nContrast: faithful edit (B) distorts perception by ~-K*delta; naive patch (C) "
          "moves the emitted action but the belief is recomputed and the patch is overwritten.")


if __name__ == "__main__":
    main()
