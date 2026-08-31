"""Part 2 + 3: the SYNC across depth, attention routing, ablations, and faithful rubber-hand.

Key reframe from part 1: with ALPHA=-0.5 the Kalman belief mu_t is a SHORT FIR filter
(decay 0.234/step) of the efference-corrected emission y_t = o_t - a_{t-1}. So mu_t is
linearly readable from a 3-tap obs window already (R2~0.9998) and even from a random net
(0.95). The non-trivial computation is therefore NOT "accumulate belief over depth" but:
  (i) reconstruct the net's own previous action a_{t-1} (which lives at position t-1),
  (ii) subtract it from o_t (efference copy) to get the corrected emission,
  (iii) read out a_t = mu_t.
This module tests the t->t-1 routing (attention), depth profile, ablation, and the
faithful rubber-hand (consistent action edit) vs naive one-off activation patch.

CPU. Run: CUDA_VISIBLE_DEVICES="" ~/comp_icl/.venv/bin/python cont_dissect2.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_cont
from agent_cont import ContAgent, ALPHA, SW, SV, S0
from cont_dissect import (load_net, manual_forward, rollout_with_actions,
                          kalman_on_realized, ridge_r2)

torch.set_grad_enabled(False)
DEV = "cpu"


# ---------------------------------------------------------------------------
# Attention-routing: how strongly does each head at each block attend t -> t-1?
# o_t needs a_{t-1} (at pos t-1) for efference correction.
# ---------------------------------------------------------------------------
def attention_profile(net, obs):
    out = manual_forward(net, obs, want_attn=True)
    B, L = obs.shape
    nl = len(out["attn"]); nh = out["attn"][0].shape[1]
    # average over batch and over query positions in synced tail
    pos = np.arange(L // 2, L)
    print("\n[attention] mean weight query t -> key (t-1) and (self t), synced tail, per block/head")
    diag_prev = np.zeros((nl, nh)); diag_self = np.zeros((nl, nh))
    for b in range(nl):
        W = out["attn"][b].numpy()   # (B,nh,L,L)
        for h in range(nh):
            # for each query t in tail, weight on key t-1 and key t
            wp = np.mean([W[:, h, t, t - 1] for t in pos])
            ws = np.mean([W[:, h, t, t] for t in pos])
            diag_prev[b, h] = wp; diag_self[b, h] = ws
    for b in range(nl):
        s = "  block %d:" % b
        for h in range(nh):
            s += f"  h{h}: t-1={diag_prev[b,h]:.2f} self={diag_self[b,h]:.2f}"
        print(s)
    return diag_prev, diag_self


# ---------------------------------------------------------------------------
# Depth profile (recap from part1 numbers but focused): probe mu, s, a, a_prev
# per block. Already have it; here we also probe the EFFERENCE-CORRECTED
# emission y_t = o_t - a_{t-1} and a_{t-1} specifically, to see where the
# net reconstructs its previous action.
# ---------------------------------------------------------------------------
def depth_profile(net, obs, states, acts):
    B, L = obs.shape
    mu, Pv = kalman_on_realized(obs, acts)
    a_np = acts.numpy()
    aprev = np.concatenate([np.zeros((B, 1)), a_np[:, :-1]], axis=1)
    y = obs.numpy() - aprev   # efference-corrected emission = s_t + v_t
    out = manual_forward(net, obs)
    resid = out["resid"]; nl = len(net.blocks)
    pos = np.arange(L // 2, L); ntr = B // 2
    def gx(stream): return stream.numpy()[:, pos, :]
    def gy(arr): return arr[:, pos]
    targets = {"a_t": a_np, "a_prev": aprev, "y_eff": y, "mu": mu}
    print("\n[depth] held-out R^2 per block for action-routing variables (synced tail)")
    print("  block |  a_t    a_prev   y_eff    mu")
    prof = {k: [] for k in targets}
    for i in range(nl + 1):
        X = gx(resid[i]); d = X.shape[-1]
        Xtr = X[:ntr].reshape(-1, d); Xte = X[ntr:].reshape(-1, d)
        row = {}
        for tn, tv in targets.items():
            Y = gy(tv); r2, _ = ridge_r2(Xtr, Y[:ntr].reshape(-1), Xte, Y[ntr:].reshape(-1))
            row[tn] = max(0, r2); prof[tn].append(max(0, r2))
        print(f"  {('emb' if i==0 else 'b%d'%(i-1)):>5} | {row['a_t']:.4f} {row['a_prev']:.4f} {row['y_eff']:.4f} {row['mu']:.4f}")
    return prof


# ---------------------------------------------------------------------------
# ABLATION: zero the t->t-1 attention edge at a given block, measure downstream
# damage to belief (mu R2 at lnf) and to behavior (action MSE).
# We patch by modifying attention weights: set query-t weight on key t-1 to 0
# and renormalize. Implemented via a custom forward.
# ---------------------------------------------------------------------------
def forward_ablate_prev(net, obs, ablate_block):
    """Run forward, but at ablate_block zero every query's attention to key (t-1)
    (renormalizing remaining weights). Returns act, and lnf residual."""
    B, L = obs.shape
    x = net.in_proj(obs.unsqueeze(-1)) + net.pos(torch.arange(L))[None]
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    for bi, blk in enumerate(net.blocks):
        h = blk.ln1(x)
        if bi == ablate_block:
            # recompute attention manually to edit weights
            attn = blk.attn
            nh = attn.num_heads; d = h.shape[-1]; hd = d // nh
            qkv_w = attn.in_proj_weight; qkv_b = attn.in_proj_bias
            q = h @ qkv_w[:d].T + qkv_b[:d]
            k = h @ qkv_w[d:2*d].T + qkv_b[d:2*d]
            v = h @ qkv_w[2*d:].T + qkv_b[2*d:]
            q = q.view(B, L, nh, hd).transpose(1, 2)
            k = k.view(B, L, nh, hd).transpose(1, 2)
            v = v.view(B, L, nh, hd).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
            scores = scores.masked_fill(mask, float("-inf"))
            w = torch.softmax(scores, dim=-1)   # (B,nh,L,L)
            # zero key t-1 for each query t>=1, renormalize
            for t in range(1, L):
                w[:, :, t, t - 1] = 0.0
            w = w / (w.sum(-1, keepdim=True) + 1e-9)
            o_attn = (w @ v).transpose(1, 2).reshape(B, L, d)
            a = o_attn @ attn.out_proj.weight.T + attn.out_proj.bias
        else:
            a, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + blk.mlp(blk.ln2(x))
    xf = net.lnf(x)
    return net.act_head(xf).squeeze(-1), xf


def ablation_study(net, obs, states, acts):
    B, L = obs.shape
    mu, _ = kalman_on_realized(obs, acts)
    pos = np.arange(L // 2, L); ntr = B // 2
    tail = slice(L // 2, None)
    # baseline mu-probe at lnf (fit once on clean)
    out = manual_forward(net, obs)
    Xc = out["post_lnf"].numpy()[:, pos, :]; d = Xc.shape[-1]
    Ymu = mu[:, pos]
    r2_clean, wt = ridge_r2(Xc[:ntr].reshape(-1, d), Ymu[:ntr].reshape(-1),
                            Xc[ntr:].reshape(-1, d), Ymu[ntr:].reshape(-1))
    a_clean = out["act"]
    base_mse = ((a_clean[:, tail].numpy() - states.numpy()[:, tail]) ** 2).mean()
    print(f"\n[ablation] clean: lnf->mu R2={r2_clean:.4f}  action MSE(tail)={base_mse:.4f}")
    print("  ablate t->t-1 edge at block | action MSE(tail) | mu-recon R2 (reuse clean probe)")
    res = []
    for blk in range(len(net.blocks)):
        a_ab, xf_ab = forward_ablate_prev(net, obs, blk)
        mse_ab = ((a_ab[:, tail].numpy() - states.numpy()[:, tail]) ** 2).mean()
        # how well does the clean mu-probe still reconstruct mu from ablated lnf
        from cont_dissect import fit_predict
        Xab = xf_ab.numpy()[:, pos, :].reshape(-1, d)
        pred = fit_predict(wt, Xab)
        yte = Ymu.reshape(-1)
        r2_ab = 1 - ((yte - pred) ** 2).sum() / (((yte - yte.mean()) ** 2).sum() + 1e-12)
        res.append((blk, mse_ab, r2_ab))
        print(f"    block {blk}: MSE={mse_ab:.4f} (Δ{mse_ab-base_mse:+.4f})   mu-R2={r2_ab:.4f}")
    return res


# ---------------------------------------------------------------------------
# RUBBER-HAND. Make the net behave as if its action at time t0 were a_t0 + delta.
# (A) FAITHFUL: re-run the env consistently. We do an INTERVENTION ROLLOUT where
#     at step t0 we override the action actually fed to the world by a+delta, and
#     ALSO ensure the net's belief update at t0+1 uses that same a+delta. But the
#     net reconstructs a_{t-1} internally from position t-1; the world o_{t0+1}
#     was produced with a+delta. The clean test of the CLAIM ("perceived state
#     shifts by -delta") is: feed the net the observation sequence as if action
#     a+delta had been taken, and read its belief mu at t0+1.
#
#     We compare two worlds at t0+1:
#       world_true: o_{t0+1} generated with action a (unchanged world)
#       The net, to decode, computes y = o - a_hat where a_hat is its OWN
#       reconstruction of the action it took. If we make the net think it took
#       a+delta (faithful, consistent edit at the action's point of CONSUMPTION),
#       its decoded emission shifts: y' = o - (a+delta) = (s+v) - delta, so the
#       perceived state mu shifts by ~ -delta*gain relative to truth.
#
#  We implement the faithful edit by WEIGHT/ROUTE editing: add delta to the
#  action-head OUTPUT consistently at position t0 AND make every later position
#  that reads position t0's action see a+delta. Cleanest realization: edit the
#  residual at position t0 along the action direction by the amount that the
#  act_head maps to +delta, applied at EVERY block (so it is present whenever a
#  downstream position attends back to t0) -- i.e. a consistent steering vector.
#
#  (B) NAIVE: one-off activation patch -- add the steering vector only to the
#  final lnf residual at t0 (changes the emitted action a_t0 but is NOT seen by
#  the t0+1 belief update because that update reconstructs the action from
#  position t0's PRE-lnf residual / earlier blocks). Show it gets overwritten:
#  belief at t0+1 does NOT shift.
# ---------------------------------------------------------------------------
def action_direction(net):
    """Direction in residual (pre-lnf) space that the act_head reads as +action.
    act = act_head(lnf(x)). We want a vector v s.t. adding c*v at pre-lnf moves
    the read-out action by ~c (locally). Use the act_head weight backprojected
    through lnf's linear part (ignore the mean-subtraction; use diag(gamma)/std).
    Simpler & robust: fit the empirical 'action axis' = ridge direction predicting
    a_t from pre-lnf residual, normalized so unit step moves action by 1."""
    return None


def rubber_hand(net, seed=7, t0=20, delta=1.5):
    """Run a teacher-forced rollout, then at t0 do faithful vs naive action edit,
    measure the belief (mu read-out) at t0+1 and whether it shifts by ~ -delta*K."""
    B, L = 4000, 40
    obs, states, acts = rollout_with_actions(net, B, L, seed=seed)
    mu, Pv = kalman_on_realized(obs, acts)
    # gain at steady state
    K = SV  # placeholder; compute properly
    P = S0 ** 2
    for _ in range(60):
        P = ALPHA ** 2 * P + SW ** 2; K = P / (P + SV ** 2); P = (1 - K) * P
    pos_q = t0 + 1

    # Build a probe that READS mu_{t} from the lnf residual at position t (fit on clean tail)
    out = manual_forward(net, obs)
    pos = np.arange(L // 2, L); d = 128; ntr = B // 2
    Xc = out["post_lnf"].numpy()[:, pos, :]; Ymu = mu[:, pos]
    _, mu_probe = ridge_r2(Xc[:ntr].reshape(-1, d), Ymu[:ntr].reshape(-1),
                           Xc[ntr:].reshape(-1, d), Ymu[ntr:].reshape(-1))
    from cont_dissect import fit_predict

    # --- Find the residual "action axis" at pre-lnf level via ridge a_t ~ resid ---
    # use block outputs averaged? We'll steer at EVERY block's residual stream at pos t0.
    # Fit axis from resid AFTER block (use the embedding-level? we need pre-block resid).
    # Practical: steer the in_proj-level residual along act direction. Compute axis from
    # lnf residual predicting a_t, then map to a consistent additive vector.
    Xa = out["post_lnf"].numpy()[:, pos, :]; Ya = acts.numpy()[:, pos]
    _, a_probe = ridge_r2(Xa[:ntr].reshape(-1, d), Ya[:ntr].reshape(-1),
                          Xa[ntr:].reshape(-1, d), Ya[ntr:].reshape(-1))
    w_a = a_probe[0] / a_probe[3]   # un-standardize weight back to raw-resid units
    # unit vector along action axis; step that moves act_head readout by +1:
    # act_head reads lnf(x). Approx steering vector in pre-lnf space ~ w_a / ||w_a||^2 scaled.
    axis = w_a / (np.linalg.norm(w_a) + 1e-9)

    # === Custom forward with a steering hook on the residual at position t0 ===
    def forward_steer(obs, c, mode):
        Bb, Ll = obs.shape
        x = net.in_proj(obs.unsqueeze(-1)) + net.pos(torch.arange(Ll))[None]
        vec = torch.tensor(axis, dtype=x.dtype)
        if mode == "faithful":
            x[:, t0, :] = x[:, t0, :] + c * vec   # add at input; persists through all blocks
        mask = torch.triu(torch.ones(Ll, Ll, dtype=torch.bool), 1)
        for blk in net.blocks:
            h = blk.ln1(x); a, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a; x = x + blk.mlp(blk.ln2(x))
        xf = net.lnf(x)
        if mode == "naive":
            xf = xf.clone(); xf[:, t0, :] = xf[:, t0, :] + c * vec  # one-off at the very end
        act = net.act_head(xf).squeeze(-1)
        return act, xf

    # calibrate c so that the EMITTED action at t0 shifts by ~delta
    def emitted_shift(c, mode):
        act, _ = forward_steer(obs, c, mode)
        clean_act = out["act"]
        return (act[:, t0] - clean_act[:, t0]).mean().item()

    # find c for faithful to move emitted action by delta
    c_f = delta / (emitted_shift(1.0, "faithful") + 1e-9)
    c_n = delta / (emitted_shift(1.0, "naive") + 1e-9)
    print(f"\n[rubber-hand] t0={t0} delta(target emitted action shift)={delta} steady-K={K:.3f}")
    print(f"  calibrated c: faithful={c_f:.3f} naive={c_n:.3f}")

    results = {}
    for mode, c in [("faithful", c_f), ("naive", c_n)]:
        act, xf = forward_steer(obs, c, mode)
        emit = (act[:, t0] - out["act"][:, t0]).mean().item()
        # belief read-out at t0+1
        Xq = xf.numpy()[:, pos_q, :]
        mu_hat = fit_predict(mu_probe, Xq)
        mu_clean = fit_predict(mu_probe, out["post_lnf"].numpy()[:, pos_q, :])
        d_mu = (mu_hat - mu_clean).mean()
        # PREDICTED shift if perception mis-tracks: the net decodes y=o-(a+delta_emit)
        # one Kalman update => mu shifts by -K * delta_emit (relative to truth)
        pred = -K * emit
        results[mode] = (emit, d_mu, pred)
        print(f"  [{mode}] emitted Δaction@t0={emit:+.3f} | belief Δμ@t0+1={d_mu:+.4f} "
              f"| predicted -K·Δa={pred:+.4f}")

    print("\n  INTERPRETATION:")
    ef = results["faithful"]; en = results["naive"]
    print(f"   faithful: belief shifts {ef[1]:+.4f} vs predicted {ef[2]:+.4f} "
          f"(ratio {ef[1]/ef[2] if abs(ef[2])>1e-6 else 0:.2f}) -> perception distorted")
    print(f"   naive   : belief shifts {en[1]:+.4f} (emitted action moved {en[0]:+.3f}) "
          f"-> {'overwritten (no downstream distortion)' if abs(en[1])<0.3*abs(ef[1]) else 'unexpectedly propagated'}")
    return results


def main():
    net, ck = load_net()
    B, L = 3000, 40
    obs, states, acts = rollout_with_actions(net, B, L, seed=1)
    attention_profile(net, obs)
    depth_profile(net, obs, states, acts)
    ablation_study(net, obs, states, acts)
    rubber_hand(net)


if __name__ == "__main__":
    main()
