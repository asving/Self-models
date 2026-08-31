"""Probe the pretrained (and post-RL) dinner nets for the world-model mechanism.

Targets come from the RATE-MARGINALIZED exact filter (the true Bayes posterior an observer
without rate knowledge can hold): per factor, a joint posterior W(rate_bin, state) updated by
  - obs z:            W[i,s] *= P(z|s)
  - free transition:  W[i,:] @ M(x_i)          (per rate bin)
  - completed set->v: W[i,:] = W[i,:].sum() * TEND_ROW[v]   (state resets, RATE POSTERIOR SURVIVES)
Completion is replicated from the visible action stream (prog/budget), so all targets are
functions of the tokens -- fair probe targets.

Probes (ridge, closed form, per layer, split by token type):
  (a) state beliefs  eta  (9d)   -- the world model
  (b) rate-posterior means (3d)  -- the in-context system identification
Do-test: swap ONE action token mid-sequence, rerun, and check whether probed downstream
beliefs track the COUNTERFACTUAL filter (operators causally wired to action tokens).

Run:  CUDA_VISIBLE_DEVICES=<id> ~/comp_icl/.venv/bin/python dinner_probe.py
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from dinner import (Env, Actors, Net, N, WAIT, K_COOK, SET_BUDGET, X_LO, X_HI, E_MAT,
                    TEND_ROW, free_M, TOK_ACT0, TOK_BOS, enc_goal, enc_ts, enc_obs, BASE)

XBINS = 12
XGRID = np.linspace(X_LO + 0.01, X_HI - 0.01, XBINS)
MGRID = free_M(XGRID)                                   # (XBINS,3,3)


# ------------------------------------------------------------ data generation
def gen_batch(B, T, rng):
    env = Env(B, rng)
    actors = Actors(B, rng, T)
    g = actors.goals()
    toks = np.zeros((B, 3 + 2 * T), dtype=np.int64)
    toks[:, 0] = TOK_BOS; toks[:, 1] = enc_goal(g); toks[:, 2] = enc_ts(T)
    zs, acts = np.zeros((B, T, N), dtype=int), np.zeros((B, T), dtype=int)
    z = env.emit(); toks[:, 3] = enc_obs(z)
    zs[:, 0] = z
    for t in range(T):
        a = actors.act(env.states, z, t)
        acts[:, t] = a
        toks[:, 4 + 2 * t] = TOK_ACT0 + a
        env.step(a)
        if t + 1 < T:
            z = env.emit()
            zs[:, t + 1] = z
            toks[:, 5 + 2 * t] = enc_obs(z)
    return toks, zs, acts, env.xrates


# ------------------------------------------------------------ token filter (rate-marginalized)
def completions_from_tokens(acts):
    """Replicate env prog/budget logic from the visible action stream."""
    B, T = acts.shape
    prog = np.zeros(B, dtype=int); last = np.full(B, -1); budget = np.full(B, SET_BUDGET)
    comp = np.zeros((B, T), dtype=bool)
    for t in range(T):
        a = acts[:, t]
        tend = a < WAIT
        prog = np.where(tend & (a == last), prog + 1, np.where(tend, 1, 0))
        c = tend & (prog >= K_COOK) & (budget > 0)
        budget -= c
        prog = np.where(c, 0, prog)
        last = np.where(c | ~tend, -1, a)
        comp[:, t] = c
    return comp

def token_filter(zs, acts):
    """Returns per-position targets aligned to token index:
       eta[p] (B,N,3) state beliefs, xm[p] (B,N) rate-posterior means, for p >= 3."""
    B, T = acts.shape
    comp = completions_from_tokens(acts)
    W = np.full((B, N, XBINS, 3), 1.0 / (XBINS * 3))
    out_eta, out_xm, out_pos = [], [], []
    def push(p):
        eta = W.sum(2); eta = eta / eta.sum(-1, keepdims=True)
        wr = W.sum(3); wr = wr / wr.sum(-1, keepdims=True)
        out_eta.append(eta.copy()); out_xm.append(wr @ XGRID); out_pos.append(p)
    def obs_update(z):
        nonlocal W
        W = W * E_MAT.T[z][:, :, None, :]               # (B,N,1,3) likelihood over states
        W = W / W.sum((2, 3), keepdims=True)
    obs_update(zs[:, 0]); push(3)
    for t in range(T):
        a, c = acts[:, t], comp[:, t]
        n, v = a // 3, a % 3
        W = np.einsum("bfxs,xst->bfxt", W, MGRID)       # free propagation, all factors
        idx = np.where(c)[0]
        if len(idx):
            tot = W[idx, n[idx]].sum(-1, keepdims=True)               # (k,XBINS,1)
            W[idx, n[idx]] = tot * TEND_ROW[v[idx]][:, None, :]       # reset state, keep rate
        W = W / W.sum((2, 3), keepdims=True)
        push(4 + 2 * t)
        if t + 1 < T:
            obs_update(zs[:, t + 1]); push(5 + 2 * t)
    return (np.stack(out_eta, 1), np.stack(out_xm, 1), np.array(out_pos))  # (B,P,N,3),(B,P,N)


# ------------------------------------------------------------ model hiddens + ridge
@torch.no_grad()
def hiddens(net, toks, dev, bs=256):
    outs = None
    for i in range(0, len(toks), bs):
        tt = torch.from_numpy(toks[i:i + bs]).to(dev)
        L = tt.shape[1]
        x = net.emb(tt) + net.pos(torch.arange(L, device=dev))[None]
        m = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1)
        hs = [x.cpu().numpy()]
        for b in net.blocks:
            x = b(x, m); hs.append(x.cpu().numpy())
        hs = np.stack(hs)                                # (nl+1, b, L, d)
        outs = hs if outs is None else np.concatenate([outs, hs], 1)
    return outs                                          # (nl+1, B, L, d)

def ridge_r2(Htr, Ytr, Hte, Yte, lam=1.0):
    Htr = np.concatenate([Htr, np.ones((len(Htr), 1))], 1)
    Hte = np.concatenate([Hte, np.ones((len(Hte), 1))], 1)
    Wm = np.linalg.solve(Htr.T @ Htr + lam * np.eye(Htr.shape[1]), Htr.T @ Ytr)
    P = Hte @ Wm
    sse = ((P - Yte) ** 2).sum(0)
    sst = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1 - sse / sst)), Wm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=768)
    ap.add_argument("--T", type=int, default=24)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(3)
    toks, zs, acts, xrates = gen_batch(args.B, args.T, rng)
    eta, xm, pos = token_filter(zs, acts)
    print(f"targets: eta {eta.shape}, rate-mean {xm.shape}, positions {len(pos)}")

    results = {}
    for tag, ck in [("pretrained", "p1_ckpt_030000.pt"), ("post-RL", "p2_ckpt_004000.pt")]:
        sd = torch.load(os.path.join(BASE, "dinner_runs/v1", ck), map_location=dev)
        cfg = sd["cfg"]
        net = Net(cfg["d"], cfg["nl"], cfg["nh"]).to(dev)
        net.load_state_dict(sd["model"]); net.eval()
        hs = hiddens(net, toks, dev)                     # (nl+1, B, L, d)
        ntr = int(args.B * 2 / 3)
        is_act_pos = (pos >= 4) & ((pos - 4) % 2 == 0)
        res = {}
        for name, sel in [("obs", ~is_act_pos), ("act", is_act_pos)]:
            psel = np.where(sel)[0]
            Y_eta = eta[:, psel].reshape(args.B, -1, N * 3)
            Y_xm = xm[:, psel]
            for li in range(hs.shape[0]):
                H = hs[li][:, pos[psel]]                 # (B,P,d)
                Htr = H[:ntr].reshape(-1, H.shape[-1]); Hte = H[ntr:].reshape(-1, H.shape[-1])
                r2b, Wb = ridge_r2(Htr, Y_eta[:ntr].reshape(len(Htr), -1),
                                   Hte, Y_eta[ntr:].reshape(len(Hte), -1))
                r2x, _ = ridge_r2(Htr, Y_xm[:ntr].reshape(len(Htr), -1),
                                  Hte, Y_xm[ntr:].reshape(len(Hte), -1))
                res.setdefault(name, {})[f"L{li}"] = (round(r2b, 3), round(r2x, 3))
                if name == "act" and li == hs.shape[0] - 1:
                    probeW = Wb                           # top-layer act-position belief probe
        results[tag] = res
        print(f"\n[{tag}] test R^2 (belief, rate-mean) by layer:")
        for name in ("obs", "act"):
            print(f"  {name}-positions: " + "  ".join(
                f"{k}:{v[0]:.3f}/{v[1]:.3f}" for k, v in res[name].items()))

        # ---------------- do-test v2: cancel a COMPLETING action (swap to WAIT) -- actions
        # are causally inert except via completions, so this is the swap that matters.
        from dinner import WAIT as _W
        comp = completions_from_tokens(acts)
        win = comp[:, 6:15]
        sel = win.any(1) & (np.arange(args.B) >= ntr)     # test rows with a completion in window
        tstar = np.where(win.any(1), 6 + win.argmax(1), 6)
        acts_cf = acts.copy(); toks_cf = toks.copy()
        rows = np.where(sel)[0]
        acts_cf[rows, tstar[rows]] = _W
        toks_cf[rows, 4 + 2 * tstar[rows]] = TOK_ACT0 + _W
        eta_cf, _, _ = token_filter(zs, acts_cf)
        hs_cf = hiddens(net, toks_cf, dev)
        top = hs.shape[0] - 1
        print(f"  do-test v2 (cancel completing set at t*; n={len(rows)} test episodes):")
        dists = []
        for dt in range(0, 6):
            tt_ = np.minimum(tstar[rows] + dt, args.T - 1)
            p = 4 + 2 * tt_
            H = hs_cf[top][rows, p]
            H = np.concatenate([H, np.ones((len(H), 1))], 1)
            pred = H @ probeW
            Yf = eta[rows, p - 3].reshape(len(H), -1)
            Yc = eta_cf[rows, p - 3].reshape(len(H), -1)
            gap = np.abs(Yf - Yc).mean()                  # how different the two oracles are
            d_fact = np.abs(pred - Yf).mean()
            d_cf = np.abs(pred - Yc).mean()
            dists.append((dt, round(float(d_cf), 4), round(float(d_fact), 4),
                          round(float(gap), 4)))
        for dt, dc, dfa, gp in dists:
            print(f"    t*+{dt}:  to-CF {dc:.4f}   to-factual {dfa:.4f}   (oracle gap {gp:.4f})")
        results[tag]["dotest"] = dists
    json.dump(results, open(os.path.join(BASE, "dinner_runs/v1/probe.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
