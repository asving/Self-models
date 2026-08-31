"""White-box dissection of trained SO(3) path-integration self-model nets.

Analyses (select with argv[1]):
  pose      : Analysis 1 -- per-layer pose build-up across depth (probe nets int_so3_*L)
  eff       : Analysis 2 -- efference copy decode + effective-window (closed-loop so3_8x64)
  rubber    : Analysis 3 -- rubber-hand causal injection (closed-loop)
  route     : Analysis 4 -- attention routing structure (closed-loop + probe)

Run: CUDA_VISIBLE_DEVICES="" ~/comp_icl/.venv/bin/python so3_dissect.py <which>
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from so3agent import so3_exp, AgentSO3, rollout, DELTA, skew
import so3_integrate as SI
from probes import ridge_fit

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "so3_runs")
dev = "cpu"
torch.manual_seed(0)


# ---------------- model loading ----------------
def load(name):
    ck = torch.load(os.path.join(RUNS, name + ".pt"), map_location=dev)
    a = ck["args"]
    in_dim = 3 if name.startswith("int_") else 6
    act_dim = a.get("act_dim", 3) if name.startswith("int_") else (1 if a["group"] == "so2" else 3)
    net = AgentSO3(in_dim, act_dim, a["d_model"], a["n_layer"], a["n_head"], a["L"])
    net.load_state_dict(ck["state"]); net.eval()
    return net, a, ck


# ---------------- per-layer hidden extractor ----------------
@torch.no_grad()
def hiddens(net, obs):
    """Returns list of residual streams: [embed, after block0, ..., after blockN-1, lnf].
    Each (B,T,d). All but the final are RAW residual; the last is post-lnf (what the head sees)."""
    B, T, _ = obs.shape
    x = net.in_proj(obs) + net.pos(torch.arange(T))[None]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    hs = [x.clone()]
    for blk in net.blocks:
        x = blk(x, mask)
        hs.append(x.clone())
    hs.append(net.lnf(x))  # final post-LN
    return [h.numpy() for h in hs]


def ang_np(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(1e-9)
    return np.rad2deg(np.arccos((a * b).sum(-1).clip(-1, 1)))


def mlp_fit(X, Y, hid=256, iters=400, lr=1e-2, wd=1e-4):
    """small torch MLP regressor, returns prediction on same X (train-set decode; we use held-out split)."""
    Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    n = len(Xt); ntr = int(0.8 * n)
    perm = torch.randperm(n); tr, te = perm[:ntr], perm[ntr:]
    m = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hid), torch.nn.GELU(),
                            torch.nn.Linear(hid, hid), torch.nn.GELU(),
                            torch.nn.Linear(hid, Y.shape[1]))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    for _ in range(iters):
        opt.zero_grad(); p = m(Xt[tr]); loss = ((p - Yt[tr]) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = m(Xt[te]).numpy()
    return pred, te.numpy()


# ============================================================
# ANALYSIS 1: pose build-up across depth (probe nets)
# ============================================================
def analysis_pose():
    rng = np.random.default_rng(7)
    B, L, mag = 1024, 32, 1.0
    X, Y = SI.gen_batch(B, L, dev, "so3", mag, rng)          # X(B,L,3) increments, Y(B,L,3) body-fwd
    ab, m2 = SI.baselines(X)
    # also build R_t entries and body-up for richer targets
    Rfull = []
    R = torch.eye(3).expand(B, 3, 3).contiguous()
    for t in range(L):
        R = so3_exp(X[:, t]) @ R; Rfull.append(R)
    Rfull = torch.stack(Rfull, 1)                            # (B,L,3,3)
    Yfwd = Rfull[..., :, 0].numpy()                          # body forward = R x_hat
    Yup = Rfull[..., :, 2].numpy()                           # body up      = R z_hat
    R9 = Rfull.reshape(B, L, 9).numpy()
    tail = slice(L // 2, None)

    ab_err = ang_np(ab.numpy()[:, tail].reshape(-1, 3), Yfwd[:, tail].reshape(-1, 3)).mean()
    m2_err = ang_np(m2.numpy()[:, tail].reshape(-1, 3), Yfwd[:, tail].reshape(-1, 3)).mean()
    print(f"\n### ANALYSIS 1: POSE BUILD-UP ACROSS DEPTH (so3 probe nets) ###")
    print(f"reference floors (tail, body-forward angular err): abelian={ab_err:.1f}deg  Magnus2={m2_err:.1f}deg\n")

    for nm in ["int_so3_1L", "int_so3_2L", "int_so3_4L", "int_so3_8L"]:
        net, a, ck = load(nm)
        with torch.no_grad():
            netpred = net(X).numpy()                          # final body-forward prediction
        net_err = ang_np(netpred[:, tail].reshape(-1, 3), Yfwd[:, tail].reshape(-1, 3)).mean()
        H = hiddens(net, X)
        names = ["embed"] + [f"blk{i}" for i in range(a["n_layer"])] + ["lnf"]
        print(f"--- {nm} (d={a['d_model']}, {a['n_layer']}L) | net output tail err={net_err:.1f}deg ---")
        print(f"  {'layer':6s} | {'linR2(R9)':9s} | {'lin fwd-err':11s} | {'mlp fwd-err':11s} | {'lin up-err':10s}")
        d = a["d_model"]
        for nmL, h in zip(names, H):
            Xt = h[:, tail].reshape(-1, d)
            # linear decode of full R (9 entries) -> reconstruct body-fwd & up from decoded columns
            _, _, r2_9 = ridge_fit(Xt, R9[:, tail].reshape(-1, 9))
            # linear body-forward
            W, b, _ = ridge_fit(Xt, Yfwd[:, tail].reshape(-1, 3))
            lin_fwd = ang_np(Xt @ W + b, Yfwd[:, tail].reshape(-1, 3)).mean()
            Wu, bu, _ = ridge_fit(Xt, Yup[:, tail].reshape(-1, 3))
            lin_up = ang_np(Xt @ Wu + bu, Yup[:, tail].reshape(-1, 3)).mean()
            # nonlinear (MLP) body-forward, held-out
            predm, te = mlp_fit(Xt, Yfwd[:, tail].reshape(-1, 3))
            mlp_fwd = ang_np(predm, Yfwd[:, tail].reshape(-1, 3)[te]).mean()
            print(f"  {nmL:6s} | {r2_9:9.3f} | {lin_fwd:11.1f} | {mlp_fwd:11.1f} | {lin_up:10.1f}")
        print()


# ============================================================
# ANALYSIS 2: efference copy
# ============================================================
def analysis_eff():
    nm = "so3_8x64"
    net, a, ck = load(nm)
    L = a["L"]
    rng = np.random.default_rng(11)
    B = 1024
    with torch.no_grad():
        err, Rs, obs, e, g = rollout(net, B, L, dev, "so3", rng, ret_states=True)
        # net's action at each position: feed full obs seq, read act_head at each t
        a_all = net(obs).numpy()                              # (B,L,3) raw action a_t
        wa_all = (DELTA * torch.tanh(net(obs))).numpy()       # applied rotation vector
    H = hiddens(net, obs)                                     # list of (B,L,d)
    names = ["embed"] + [f"blk{i}" for i in range(a["n_layer"])] + ["lnf"]
    d = a["d_model"]
    print(f"\n### ANALYSIS 2: EFFERENCE COPY (so3_8x64, closed-loop) ###")
    print(f"final tail err = {err[:, L//2:].mean():.4f}\n")

    # (a) when is a_k computed? decode a_k (and applied wa_k) at position k, per layer
    print("(a) decode net's OWN action at the SAME position k, R2 (a_k raw / wa applied):")
    print(f"  {'layer':6s} | {'R2(a_k)':8s} | {'R2(wa_k)':8s}")
    for nmL, h in zip(names, H):
        X = h.reshape(-1, d); Ya = a_all.reshape(-1, 3); Yw = wa_all.reshape(-1, 3)
        _, _, r2a = ridge_fit(X, Ya); _, _, r2w = ridge_fit(X, Yw)
        print(f"  {nmL:6s} | {r2a:8.3f} | {r2w:8.3f}")

    # (b) is a_k still decodable at LATER positions t>k (stored/routed forward)?
    # At a fixed read position t, decode the action emitted at earlier position k from resid[:,t].
    # Use deepest pre-lnf layer (blk last) and lnf.
    print("\n(b) decode EARLIER action a_k from residual at a LATER read position t (lag = t-k):")
    print("    (high R2 at lag>0 => the past action is stored/routed forward at position t)")
    read_layers = {"blk_last": H[-2], "lnf": H[-1]}
    t = L - 1                                                  # read at the last position
    for rl, h in read_layers.items():
        Xt = h[:, t]                                          # (B,d) resid at read pos t
        row = []
        for lag in range(0, 8):
            k = t - lag
            if k < 0: break
            Yk = a_all[:, k]                                  # action emitted at position k
            _, _, r2 = ridge_fit(Xt, Yk)
            row.append(f"lag{lag}:{r2:.2f}")
        print(f"  read@{rl:8s} (t={t}): " + "  ".join(row))

    # (c) effective window: randomize prefix, keep last k obs, watch last-pos action & pose-output
    print("\n(c) effective window: randomize prefix obs, keep last k; RMSE(action@last vs full):")
    with torch.no_grad():
        _, _, obsB, _, _ = rollout(net, B, L, dev, "so3", np.random.default_rng(99), ret_states=True)
        a_full = net(obs)[:, -1].numpy()
        a_diff = net(obsB)[:, -1].numpy()
        base = np.sqrt(((a_full - a_diff) ** 2).mean())
        for k in [1, 2, 3, 4, 6, 8, 12, 16]:
            if k > L: continue
            mix = torch.cat([obsB[:, :L - k], obs[:, L - k:]], 1)
            a_mix = net(mix)[:, -1].numpy()
            rmse = np.sqrt(((a_mix - a_full) ** 2).mean())
            print(f"  keep last k={k:2d}: RMSE={rmse:.4f}  ({100*(1-rmse/base):.0f}% recovered)")


# ============================================================
# ANALYSIS 3: rubber-hand (causal injection)
# ============================================================
def true_pose_fwd(net, obs, e, g, L):
    """Recompute true body-forward Y_t = R_t x_hat by re-running the closed-loop kinematics
    with the net's actions on this exact obs sequence (teacher-forced on given obs)."""
    B = obs.shape[0]
    with torch.no_grad():
        a = net(obs)                                          # (B,L,3)
    R = torch.eye(3).expand(B, 3, 3).contiguous()
    Y = []
    for t in range(L):
        wa = DELTA * torch.tanh(a[:, t]); we = e[:, t]
        Y.append((R @ torch.tensor([1., 0., 0.]))[:, :])      # R x_hat BEFORE update (pose at t)
        R = so3_exp(wa) @ so3_exp(we) @ R
    return torch.stack(Y, 1)


@torch.no_grad()
def forward_inject(net, obs, L, inj_layer, read_pos, delta_resid):
    """Run forward, adding delta_resid (B,d torch) to residual at inj_layer output, position read_pos.
    Returns (action_at_read_pos (B,act), lnf_at_read_pos (B,d))."""
    x = net.in_proj(obs) + net.pos(torch.arange(L))[None]
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    for li, blk in enumerate(net.blocks):
        x = blk(x, mask)
        if li == inj_layer and delta_resid is not None:
            x = x.clone(); x[:, read_pos] = x[:, read_pos] + delta_resid
    lnf = net.lnf(x)
    return net.act_head(lnf)[:, read_pos], lnf[:, read_pos]


def analysis_rubber():
    nm = "so3_8x64"
    net, a, ck = load(nm)
    L = a["L"]; d = a["d_model"]; nl = a["n_layer"]
    rng = np.random.default_rng(21)
    B = 2048
    with torch.no_grad():
        err, Rs, obs, e, g = rollout(net, B, L, dev, "so3", rng, ret_states=True)
    Yfwd = Rs[..., :, 0]                                      # (B,L,3) true body-forward at t
    H = hiddens(net, obs)                                     # [embed, blk0..blkN-1, lnf]
    read_pos = L - 1
    inj_layer = nl - 3                                        # inject upstream of the last couple blocks

    print(f"\n### ANALYSIS 3: RUBBER-HAND (so3_8x64, causal injection at blk{inj_layer}, read@pos{read_pos}) ###")

    # high-quality pose probe at the injection layer (resid -> body-forward), tail positions
    Xt = H[inj_layer + 1][:, L // 2:].reshape(-1, d)         # H[0]=embed so blk i is H[i+1]
    Yt = Yfwd[:, L // 2:].reshape(-1, 3).numpy()
    Wp, bp, r2p = ridge_fit(Xt, Yt)
    Wpinv = np.linalg.pinv(Wp.T)                             # (d,3): resid delta to move decoded fwd
    # clean final-layer pose probe (to read net's BELIEVED pose downstream)
    Wl, bl, r2l = ridge_fit(H[-1][:, L // 2:].reshape(-1, d), Yt)
    print(f"  pose probe R2: at blk{inj_layer}={r2p:.3f}, at lnf={r2l:.3f}")

    fwd_clean = H[-1][:, read_pos] @ Wl + bl                 # clean believed fwd at read pos
    a_clean, _ = forward_inject(net, obs, L, inj_layer, read_pos, None)

    # --- 3a. inject so decoded body-forward rotates by Delta about z; measure downstream shift ---
    print("\n3a. rotate DECODED body-forward by Delta (about z) at blk{}; downstream belief shift + action change:".format(inj_layer))
    print(f"  {'Delta':6s} | {'belief-shift':12s} | {'(ideal=Delta)':13s} | {'fidelity':9s} | {'|d action|':10s}")
    axis = np.array([0., 0., 1.])
    for ang_inj in [5, 10, 20, 40]:
        Rdelta = so3_exp(torch.tensor(np.deg2rad(ang_inj) * axis, dtype=torch.float32)).numpy()
        fwd_dec0 = H[inj_layer + 1][:, read_pos] @ Wp + bp   # decoded fwd at inj layer/pos
        dv = (Rdelta @ fwd_dec0.T).T - fwd_dec0              # desired change
        delta_resid = torch.tensor(dv @ Wpinv.T, dtype=torch.float32)
        a_inj, lnf_inj = forward_inject(net, obs, L, inj_layer, read_pos, delta_resid)
        fwd_after = lnf_inj.numpy() @ Wl + bl
        shift = ang_np(fwd_after, fwd_clean).mean()
        dact = np.sqrt(((a_inj.numpy() - a_clean.numpy()) ** 2).mean())
        print(f"  {ang_inj:5d}d | {shift:11.1f}d | {ang_inj:12d}d | {shift/ang_inj:9.2f} | {dact:10.4f}")

    # --- 3b. corrupt the EFFERENCE-COPY subspace (decoded action) -> does belief pose shift? ---
    print("\n3b. corrupt the efference-copy (push decoded action a_k) at blk{}; does believed pose shift?".format(inj_layer))
    with torch.no_grad():
        a_all = net(obs).numpy()
    Wa, ba, r2a = ridge_fit(H[inj_layer + 1][:, L // 2:].reshape(-1, d), a_all[:, L // 2:].reshape(-1, 3))
    Wa_pinv = np.linalg.pinv(Wa.T)
    print(f"  action probe R2 at blk{inj_layer}={r2a:.3f}")
    print(f"  {'|da|':6s} | {'belief-shift(deg)':17s}")
    for da_mag in [0.3, 0.7, 1.5]:
        ddir = (np.array([0., 0., 1.]) * da_mag)             # push action along z
        delta_resid = torch.tensor(np.tile(ddir @ Wa_pinv.T, (B, 1)), dtype=torch.float32)
        _, lnf_inj = forward_inject(net, obs, L, inj_layer, read_pos, delta_resid)
        fwd_after = lnf_inj.numpy() @ Wl + bl
        shift = ang_np(fwd_after, fwd_clean).mean()
        print(f"  {da_mag:6.1f} | {shift:17.1f}")

    # --- 3c. control: inject random-direction residual of matched norm (specificity check) ---
    print("\n3c. CONTROL: random-direction residual of matched norm (should shift belief LESS than the pose-aligned inject):")
    Rdelta = so3_exp(torch.tensor(np.deg2rad(20.) * axis, dtype=torch.float32)).numpy()
    fwd_dec0 = H[inj_layer + 1][:, read_pos] @ Wp + bp
    dv = (Rdelta @ fwd_dec0.T).T - fwd_dec0
    pose_delta = torch.tensor(dv @ Wpinv.T, dtype=torch.float32)
    target_norm = pose_delta.norm(dim=-1, keepdim=True)
    rnd = torch.randn(B, d); rnd = rnd / rnd.norm(dim=-1, keepdim=True) * target_norm
    _, lnf_pose = forward_inject(net, obs, L, inj_layer, read_pos, pose_delta)
    _, lnf_rnd = forward_inject(net, obs, L, inj_layer, read_pos, rnd)
    sh_pose = ang_np(lnf_pose.numpy() @ Wl + bl, fwd_clean).mean()
    sh_rnd = ang_np(lnf_rnd.numpy() @ Wl + bl, fwd_clean).mean()
    print(f"  pose-aligned inject (Delta=20deg): belief shift = {sh_pose:.1f}deg")
    print(f"  random-direction  matched-norm   : belief shift = {sh_rnd:.1f}deg")


# ============================================================
# ANALYSIS 4: routing / attention
# ============================================================
def analysis_route():
    for nm in ["so3_8x64", "int_so3_8L"]:
        net, a, ck = load(nm)
        L = a["L"]; nh = a["n_head"]; nl = a["n_layer"]
        if nm.startswith("int_"):
            rng = np.random.default_rng(5)
            X, Y = SI.gen_batch(64, L, dev, "so3", 1.0, rng)
            obs = X
        else:
            with torch.no_grad():
                _, _, obs, _, _ = rollout(net, 64, L, dev, "so3", np.random.default_rng(5), ret_states=True)
        # capture attention weights from each block by monkeypatching need_weights
        print(f"\n### ANALYSIS 4: ATTENTION ROUTING -- {nm} ({nl}L,{nh}h,L={L}) ###")
        attns = []
        x = net.in_proj(obs) + net.pos(torch.arange(L))[None]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        with torch.no_grad():
            for blk in net.blocks:
                h = blk.ln1(x)
                aout, w = blk.attn(h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=True)
                attns.append(w.mean(0).numpy())               # (L,L) avg over batch
                x = x + aout
                x = x + blk.mlp(blk.ln2(x))
        # characterize each layer: diagonal-band (self/recent) vs uniform-prefix (scan) vs t->t-1
        for li, w in enumerate(attns):
            # for query positions in tail, measure: attn to self, to t-1, to "uniform prefix" mass
            q = L - 1
            row = w[q, :q + 1]
            self_w = row[q]; prev_w = row[q - 1] if q >= 1 else 0
            # uniformity of prefix: entropy / log(q+1)
            p = row / row.sum()
            ent = -(p * np.log(p + 1e-12)).sum() / np.log(q + 1)
            # mass on first half vs recent (last 3)
            recent = row[max(0, q - 2):].sum(); early = row[:max(1, q - 2)].sum()
            print(f"  blk{li}: q={q} self={self_w:.2f} prev(t-1)={prev_w:.2f} "
                  f"recent3={recent:.2f} early={early:.2f} normEntropy={ent:.2f}")
        # avg over tail queries: mean attention-distance (q - attended index)
        print("  mean attn lag (tail queries q>=L/2), per layer:")
        for li, w in enumerate(attns):
            lags = []
            for q in range(L // 2, L):
                row = w[q, :q + 1]; idx = np.arange(q + 1)
                lags.append(((q - idx) * row).sum() / row.sum())
            print(f"    blk{li}: mean lag={np.mean(lags):.2f}  (0=self, 1=t-1, large=looks back far)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "pose"
    if which == "pose":
        analysis_pose()
    elif which == "eff":
        analysis_eff()
    elif which == "rubber":
        analysis_rubber()
    elif which == "route":
        analysis_route()
