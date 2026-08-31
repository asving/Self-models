"""White-box dissection of ContAgent (runs/cont_c8.pt): coupled belief<->action recursion.

Methodology (whitebox skill): exact recompute harness first, verify vs net.forward,
then locate semantic variables (Kalman mu_t, true state s_t, action a_t, a_{t-1}) per
layer with ridge probes + baselines, then causal tests (attention ablation, faithful
rubber-hand vs naive activation patch).

CPU-only. Run: CUDA_VISIBLE_DEVICES="" ~/comp_icl/.venv/bin/python cont_dissect.py
"""
from __future__ import annotations
import os, sys, math
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_cont
from agent_cont import ContAgent, ALPHA, SW, SV, S0

torch.set_grad_enabled(False)
DEV = "cpu"
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs/cont_c8.pt")


def load_net():
    ck = torch.load(CKPT, map_location="cpu")
    a = ck["args"]
    net = ContAgent(a["d_model"], a["n_layer"], a["n_head"], a["L"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


# ---------------------------------------------------------------------------
# Exact recompute harness: manual forward capturing per-block residual stream.
# Returns dict with residual after each block (pre-lnf) and post-lnf, plus
# per-block per-head attention weights.
# ---------------------------------------------------------------------------
def manual_forward(net, o, want_attn=False):
    B, L = o.shape
    x = net.in_proj(o.unsqueeze(-1)) + net.pos(torch.arange(L, device=o.device))[None]
    mask = torch.triu(torch.ones(L, L, device=o.device, dtype=torch.bool), 1)
    resid = [x.clone()]            # resid[0] = embedding (input to block 0)
    attns = []
    for blk in net.blocks:
        h = blk.ln1(x)
        if want_attn:
            a, w = blk.attn(h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=False)
            attns.append(w)        # (B, nh, L, L)
        else:
            a, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + blk.mlp(blk.ln2(x))
        resid.append(x.clone())    # resid[i+1] = output of block i
    xf = net.lnf(x)
    act = net.act_head(xf).squeeze(-1)
    obs = net.obs_head(xf).squeeze(-1)
    out = dict(resid=resid, post_lnf=xf, act=act, obs=obs)
    if want_attn:
        out["attn"] = attns
    return out


def verify_harness(net):
    o = torch.randn(8, 40)
    a_ref, op_ref = net(o)
    out = manual_forward(net, o)
    da = (out["act"] - a_ref).abs().max().item()
    do = (out["obs"] - op_ref).abs().max().item()
    print(f"[harness] max|act diff|={da:.2e}  max|obs diff|={do:.2e}")
    assert da < 1e-4 and do < 1e-4, "manual forward does not match net.forward"
    return da, do


# ---------------------------------------------------------------------------
# Kalman reference conditioned on the NET'S realized actions.
# The env: o depends on the action actually taken. So to get the OPTIMAL
# per-step posterior mean mu_t for the net's realized rollout, we run the
# Kalman recursion feeding it the net's own actions a_t (not a=mu), but it
# only ever observes o (and knows a_t via efference, y=o-a). This is the
# belief the net SHOULD hold given the observations it actually saw.
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_with_actions(net, B, L, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    s = torch.randn(B, generator=g) * S0
    o = s + torch.randn(B, generator=g) * SV          # o_0 (a_{-1}=0)
    obs, states, acts = [o], [s], []
    for t in range(L):
        a = net(torch.stack(obs, 1))[0][:, -1]        # action at current last pos
        acts.append(a)
        if t == L - 1:
            break
        s = (ALPHA * s + a + torch.randn(B, generator=g) * SW).clamp(-12, 12)
        o = s + torch.randn(B, generator=g) * SV + a
        obs.append(o); states.append(s)
    obs = torch.stack(obs, 1)        # (B,L)
    states = torch.stack(states, 1)  # (B,L)
    acts = torch.stack(acts, 1)      # (B,L)  net action at each position
    return obs, states, acts


def kalman_on_realized(obs, acts):
    """Optimal Kalman posterior mean mu_t / var P_t given the actually-observed
    o sequence and the net's actually-taken actions a_t (known via efference).
    Returns mu (B,L) = E[s_t | o_0..o_t, a_0..a_{t-1}] and P (B,L)."""
    o = obs.numpy(); a = acts.numpy()
    B, L = o.shape
    mu = np.zeros((B, L)); Pv = np.zeros((B, L))
    # init: prior s_0 ~ N(0, S0^2); update on o_0 = s_0 + v
    m = np.zeros(B); P = np.full(B, S0 ** 2)
    K = P / (P + SV ** 2); m = m + K * (o[:, 0] - m); P = (1 - K) * P
    mu[:, 0] = m; Pv[:, 0] = P
    for t in range(1, L):
        a_prev = a[:, t - 1]
        # predict using known previous action
        m = ALPHA * m + a_prev; P = ALPHA ** 2 * P + SW ** 2
        # decode emission y = o_t - a_{t-1} = s_t + v
        y = o[:, t] - a_prev
        K = P / (P + SV ** 2); m = m + K * (y - m); P = (1 - K) * P
        mu[:, t] = m; Pv[:, t] = P
    return mu, Pv


# ---------------------------------------------------------------------------
# Ridge probe with held-out R^2 + baselines.
# ---------------------------------------------------------------------------
def ridge_r2(X, y, Xte, yte, lam=10.0):
    # standardize features on train
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xs = (X - mu) / sd; Xtes = (Xte - mu) / sd
    d = Xs.shape[1]
    A = Xs.T @ Xs + lam * np.eye(d)
    w = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    b = y.mean()
    pred = Xtes @ w + b
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-12
    return 1 - ss_res / ss_tot, (w, b, mu, sd)


def fit_predict(w_tuple, Xnew):
    w, b, mu, sd = w_tuple
    return ((Xnew - mu) / sd) @ w + b


def main():
    net, ck = load_net()
    floor = ck["floor"]
    print(f"loaded cont_c8: {ck['args']['n_layer']} layers d={ck['args']['d_model']} floor={floor:.4f}")
    verify_harness(net)

    B, L = 3000, 40
    obs, states, acts = rollout_with_actions(net, B, L, seed=1)
    mu, Pv = kalman_on_realized(obs, acts)

    # net's realized action MSE vs s (tail) to confirm near-floor
    tail = slice(L // 2, None)
    net_mse = ((acts[:, tail] - states[:, tail]) ** 2).mean().item()
    kal_mse = ((torch.tensor(mu)[:, tail] - states[:, tail]) ** 2).mean().item()
    print(f"[behavior] net action MSE(tail)={net_mse:.4f}  Kalman-on-realized MSE(tail)={kal_mse:.4f}  floor={floor:.4f}")

    # net action vs mu: correlation + MSE
    a_np = acts.numpy()
    corr = np.corrcoef(a_np[:, tail].ravel(), mu[:, tail].ravel())[0, 1]
    amu_mse = ((a_np[:, tail] - mu[:, tail]) ** 2).mean()
    print(f"[behavior] corr(net a, Kalman mu) tail={corr:.4f}  MSE(a,mu) tail={amu_mse:.4f}")

    # capture residuals (post-lnf is what heads read; also probe per-block raw resid)
    out = manual_forward(net, obs)
    resid = out["resid"]   # list len 9: resid[0]=embed, resid[i+1]=after block i
    nl = ck["args"]["n_layer"]

    # ---- Probes per layer. Target positions: synced tail (t>=L//2). ----
    # We pool over (B, tail positions). Train/test split on batch.
    pos = np.arange(L // 2, L)
    ntr = B // 2
    def gather(resid_t):  # resid_t (B,L,d) -> (N, d) over tail positions
        r = resid_t.numpy()[:, pos, :]   # (B, |pos|, d)
        return r
    def tgt(arr):  # (B,L) -> (B,|pos|)
        return arr[:, pos]

    s_np = states.numpy()
    aprev = np.concatenate([np.zeros((B, 1)), a_np[:, :-1]], axis=1)  # a_{t-1}, a_{-1}=0

    targets = {"mu": mu, "s": s_np, "a": a_np, "aprev": aprev}
    print("\n[probes] held-out R^2 per layer (rows: layer 0=embed .. 8=final post-block; probe on synced tail)")
    print("  layer |   mu      s      a    a_prev")
    results = {}
    # Apply final LN-like normalization? Probe raw residual (block output). Also probe post-lnf separately.
    layer_streams = [("L%d" % i, resid[i]) for i in range(nl + 1)]
    layer_streams.append(("lnf", out["post_lnf"]))
    for name, stream in layer_streams:
        Xall = gather(stream)            # (B,|pos|,d)
        d = Xall.shape[-1]
        Xtr = Xall[:ntr].reshape(-1, d); Xte = Xall[ntr:].reshape(-1, d)
        row = {}
        for tn, tv in targets.items():
            Y = tgt(tv)
            ytr = Y[:ntr].reshape(-1); yte = Y[ntr:].reshape(-1)
            r2, _ = ridge_r2(Xtr, ytr, Xte, yte)
            row[tn] = max(0.0, r2)
        results[name] = row
        print(f"  {name:>5} | {row['mu']:.4f} {row['s']:.4f} {row['a']:.4f} {row['aprev']:.4f}")

    # ---- Baselines ----
    print("\n[baselines]")
    # shuffle control on lnf for mu
    Xall = gather(out["post_lnf"]); d = Xall.shape[-1]
    Xtr = Xall[:ntr].reshape(-1, d); Xte = Xall[ntr:].reshape(-1, d)
    Ymu = tgt(mu); ytr = Ymu[:ntr].reshape(-1); yte = Ymu[ntr:].reshape(-1)
    perm = np.random.default_rng(0).permutation(len(ytr))
    r2_shuf, _ = ridge_r2(Xtr, ytr[perm], Xte, yte)
    print(f"  shuffle control (lnf->mu): R2={r2_shuf:.4f} (should ~0)")

    # random-init net baseline
    rnet = ContAgent(ck['args']['d_model'], nl, ck['args']['n_head'], L); rnet.eval()
    rout = manual_forward(rnet, obs)
    Xr = rout["post_lnf"].numpy()[:, pos, :]; Xrtr = Xr[:ntr].reshape(-1, d); Xrte = Xr[ntr:].reshape(-1, d)
    r2_rand, _ = ridge_r2(Xrtr, ytr, Xrte, yte)
    print(f"  random-init net (lnf->mu): R2={max(0,r2_rand):.4f}")

    # input-window baseline: regress mu on last W observations (and actions via efference? net only sees o)
    for W in (1, 3, 8):
        feats = []
        for k in range(W):
            sh = np.zeros((B, L)); sh[:, k:] = obs.numpy()[:, :L - k] if k > 0 else obs.numpy()
            feats.append(sh)
        Xw = np.stack(feats, -1)[:, pos, :]   # (B,|pos|,W)
        Xwtr = Xw[:ntr].reshape(-1, W); Xwte = Xw[ntr:].reshape(-1, W)
        r2_w, _ = ridge_r2(Xwtr, ytr, Xwte, yte, lam=1e-3)
        print(f"  obs-window W={W} -> mu: R2={max(0,r2_w):.4f}")

    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cont_dissect_probes.npz"),
             results={k: v for k, v in results.items()}, floor=floor,
             net_mse=net_mse, kal_mse=kal_mse, corr=corr, amu_mse=amu_mse,
             allow_pickle=True)
    print("\nsaved cont_dissect_probes.npz")


if __name__ == "__main__":
    main()
