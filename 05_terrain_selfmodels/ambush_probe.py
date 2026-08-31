"""Probe battery for the ambush nets: is the dodge ROUTED through a represented
self-inclination, or COMPILED into a direct context->action map?

P1  Decodability: ridge probes (per layer, at decision positions = x_t token positions)
    for p_hat = softmax(reader(h)) -- the mindreader's prediction of the agent. Ground
    truth = literally running the reader net. Baseline: regression from oracle features
    [belief eta, t/T, running camp-on-argmax rate] -- the probe must beat what belief
    alone explains.
P2  STEERING (routing test): patch the residual at layer L, decision position, with the
    minimal-norm delta that swaps the top-2 entries of the decoded p_hat; measure the
    change in the net's own action log-probs. ROUTING predicts anti-alignment (net stops
    avoiding the old p_hat argmax, starts avoiding the new one), specifically on
    reader episodes (type-conditional). Controls: random delta of matched norm; bias episodes.
P3  Transition timeline (introspection vs self-image): for phase-2 ckpts across the
    step-~700 dodge transition, decode BOTH targets -- p_hat of the (approximate) concurrent
    reader ckpt, and pi_own of the probed net itself -- and compare R^2 on the subset of
    positions where the two disagree (TV > 0.2). reader mapping exact for steps<=200
    (reader = phase-1 net), approximate later (nearest saved ckpt).
P4  Degeneracy check: mean TV(pi_4000, pi_8000) on-context for the lag-4000 arm.

Run: CUDA_VISIBLE_DEVICES=<id> ~/comp_icl/.venv/bin/python ambush_probe.py
"""
from __future__ import annotations
import json, os
import numpy as np
import torch, torch.nn.functional as F

from ambush import (Net, World, camp_dist, sample_rows, filt_obs, filt_step, S,
                    TOK_X0, TOK_A0, TOK_C0, TOK_BOS, BASE)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUN = os.path.join(BASE, "ambush_runs")
T, BETA = 24, 0.5


def load(path):
    sd = torch.load(path, map_location=DEV)
    net = Net(sd["cfg"]["d"], sd["cfg"]["nl"], sd["cfg"]["nh"]).to(DEV)
    net.load_state_dict(sd["model"]); net.eval()
    return net


@torch.no_grad()
def rollout(net, reader, B, seed):
    rng = np.random.default_rng(seed)
    w = World(B, rng)
    is_reader = rng.random(B) < BETA
    qcamp = rng.dirichlet(np.full(S, 0.5), B)
    eta = np.full((B, S), 1 / S)
    etas = np.zeros((B, T, S)); campstat = np.zeros((B, T))
    hit_run = np.zeros(B)
    tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
    for t in range(T):
        z = w.emit(); eta = filt_obs(eta, z); etas[:, t] = eta
        tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(DEV)], 1)
        la = net(tt)[:, -1, TOK_A0:TOK_C0]
        a = torch.multinomial(F.softmax(la, -1), 1).squeeze(1).cpu().numpy()
        ph = F.softmax(reader(tt)[:, -1, TOK_A0:TOK_C0], -1).cpu().numpy()
        c = sample_rows(camp_dist(is_reader, ph, qcamp), rng)
        hit_run += (c == eta.argmax(1)); campstat[:, t] = hit_run / (t + 1)
        tt = torch.cat([tt, torch.from_numpy(
            np.stack([TOK_A0 + a, TOK_C0 + c], 1)).to(DEV)], 1)
        w.step(); eta = filt_step(eta)
    return tt, is_reader, etas, campstat


@torch.no_grad()
def hiddens(net, tt, bs=512):
    outs = None
    for i in range(0, len(tt), bs):
        x = net.emb(tt[i:i+bs]) + net.pos(torch.arange(tt.shape[1], device=DEV))[None]
        m = torch.triu(torch.ones(tt.shape[1], tt.shape[1], device=DEV, dtype=torch.bool), 1)
        hs = [x.cpu().numpy()]
        for b in net.blocks:
            x = b(x, m); hs.append(x.cpu().numpy())
        hs = np.stack(hs)
        outs = hs if outs is None else np.concatenate([outs, hs], 1)
    return outs                                          # (nl+1, B, L, d)


@torch.no_grad()
def policy_at_decisions(net, tt):
    lg = []
    for i in range(0, len(tt), 512):
        lg.append(net(tt[i:i+512])[:, :, TOK_A0:TOK_C0].cpu())
    lg = torch.cat(lg)
    pos = 1 + 3 * np.arange(T)
    return F.softmax(lg[:, pos], -1).numpy()             # (B,T,3)


def ridge(Htr, Ytr, Hte, Yte, lam=1.0):
    H1 = np.concatenate([Htr, np.ones((len(Htr), 1))], 1)
    Wm = np.linalg.solve(H1.T @ H1 + lam * np.eye(H1.shape[1]), H1.T @ Ytr)
    H2 = np.concatenate([Hte, np.ones((len(Hte), 1))], 1)
    P = H2 @ Wm
    sse = ((P - Yte) ** 2).sum(0); sst = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1 - sse / sst)), Wm


def r2_on(W, H, Y):
    H1 = np.concatenate([H, np.ones((len(H), 1))], 1)
    P = H1 @ W
    sse = ((P - Y) ** 2).sum(0); sst = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1 - sse / sst))


def main():
    results = {}
    A_final = load(f"{RUN}/A/p2_ckpt_008000.pt")

    # ---------------- P1: decodability of p_hat (== own inclination at equilibrium)
    tt, is_reader, etas, campstat = rollout(A_final, A_final, 2048, seed=5)
    hs = hiddens(A_final, tt)
    ph = policy_at_decisions(A_final, tt)                # reader == self at equilibrium
    pos = 1 + 3 * np.arange(T)
    ntr = 1400
    Y = ph.reshape(2048, -1)
    feats = np.concatenate([etas, campstat[..., None],
                            np.broadcast_to(np.arange(T)[None, :, None] / T,
                                            (2048, T, 1))], -1)
    base_r2, _ = ridge(feats[:ntr].reshape(-1, 5), ph[:ntr].reshape(-1, 3),
                       feats[ntr:].reshape(-1, 5), ph[ntr:].reshape(-1, 3))
    print(f"P1 baseline (eta,+stats -> p_hat): R2 = {base_r2:.3f}")
    probes = {}
    for li in range(hs.shape[0]):
        H = hs[li][:, pos]
        r2, Wm = ridge(H[:ntr].reshape(-1, H.shape[-1]), ph[:ntr].reshape(-1, 3),
                       H[ntr:].reshape(-1, H.shape[-1]), ph[ntr:].reshape(-1, 3))
        probes[li] = (r2, Wm)
        print(f"P1 layer {li}: p_hat decode R2 = {r2:.3f}")
    results["P1"] = {f"L{k}": round(v[0], 3) for k, v in probes.items()} | \
                    {"baseline": round(base_r2, 3)}

    # ---------------- P2: steering at decision position t*=16
    tstar = 16; p = 1 + 3 * tstar
    rows = np.arange(2048)
    out2 = {}
    for li in (3, 4, 5):
        Wm = probes[li][1][:-1]                          # (d,3), drop bias row
        Wpinv = np.linalg.pinv(Wm)                       # (3,d)
        h = hs[li][rows, p]
        y = np.concatenate([h, np.ones((len(h), 1))], 1) @ probes[li][1]
        y_t = y.copy()
        top = y.argmax(1); sec = np.argsort(y, 1)[:, -2]
        y_t[rows, top], y_t[rows, sec] = y[rows, sec], y[rows, top]
        delta = (y_t - y) @ Wpinv                        # minimal-norm patch
        rnd = np.random.default_rng(0).standard_normal(delta.shape)
        rnd *= (np.linalg.norm(delta, axis=1, keepdims=True) /
                (np.linalg.norm(rnd, axis=1, keepdims=True) + 1e-9))
        def patched_logits(dvec):
            outs = []
            for i in range(0, 2048, 512):
                sl = slice(i, i + 512)
                x = A_final.emb(tt[sl]) + A_final.pos(
                    torch.arange(tt.shape[1], device=DEV))[None]
                m = torch.triu(torch.ones(tt.shape[1], tt.shape[1], device=DEV,
                                          dtype=torch.bool), 1)
                with torch.no_grad():
                    for bi, b in enumerate(A_final.blocks):
                        if bi == li:                     # inject at input of block li
                            x[:, p] += torch.from_numpy(dvec[sl]).float().to(DEV)
                        x = b(x, m)
                    lg = A_final.head(A_final.lnf(x))[:, p, TOK_A0:TOK_C0]
                outs.append(lg.cpu())
            return F.log_softmax(torch.cat(outs), -1).numpy()
        lp0 = patched_logits(np.zeros_like(delta))
        lp1 = patched_logits(delta)
        lpr = patched_logits(rnd)
        d_old = lp1[rows, top] - lp0[rows, top]          # routing: should INCREASE
        d_new = lp1[rows, sec] - lp0[rows, sec]          # routing: should DECREASE
        r_old = lpr[rows, top] - lp0[rows, top]
        out2[li] = dict(
            steer_old_reader=float(d_old[is_reader].mean()),
            steer_new_reader=float(d_new[is_reader].mean()),
            steer_old_bias=float(d_old[~is_reader].mean()),
            steer_new_bias=float(d_new[~is_reader].mean()),
            rand_old_reader=float(r_old[is_reader].mean()),
            dnorm=float(np.linalg.norm(delta, axis=1).mean()))
        print(f"P2 layer {li}: dlogp(old p_hat argmax) reader={out2[li]['steer_old_reader']:+.3f} "
              f"bias={out2[li]['steer_old_bias']:+.3f} rand-ctl={out2[li]['rand_old_reader']:+.3f} | "
              f"dlogp(new) reader={out2[li]['steer_new_reader']:+.3f}")
    results["P2"] = out2

    # ---------------- P3: transition timeline, introspection vs self-image
    stages = [(50, f"{RUN}/A/p1_ckpt_020000.pt", "exact"),
              (100, f"{RUN}/A/p1_ckpt_020000.pt", "exact"),
              (200, f"{RUN}/A/p1_ckpt_020000.pt", "exact"),
              (400, f"{RUN}/A/p2_ckpt_000200.pt", "approx(true=250)"),
              (700, f"{RUN}/A/p2_ckpt_000400.pt", "approx(true=500)"),
              (1000, f"{RUN}/A/p2_ckpt_000700.pt", "approx(true=750)"),
              (2000, f"{RUN}/A/p2_ckpt_001500.pt", "approx(true=1750)")]
    print("\nP3: net-step | mean TV(reader,self) | frac-disagree | R2->reader | R2->self  (top layer)")
    out3 = []
    for step, rpath, tag in stages:
        net = load(f"{RUN}/A/p2_ckpt_{step:06d}.pt")
        reader = load(rpath)
        tt3, ir3, _, _ = rollout(net, reader, 1024, seed=11)
        hs3 = hiddens(net, tt3)[-1][:, pos]              # top layer
        y_reader = policy_at_decisions(reader, tt3)
        y_self = policy_at_decisions(net, tt3)
        tv = 0.5 * np.abs(y_reader - y_self).sum(-1)
        H = hs3.reshape(-1, hs3.shape[-1])
        Yr = y_reader.reshape(-1, 3); Ys = y_self.reshape(-1, 3)
        mask = tv.reshape(-1) > 0.2
        ntr3 = int(0.7 * len(H))
        _, Wr = ridge(H[:ntr3], Yr[:ntr3], H[ntr3:], Yr[ntr3:])
        _, Ws = ridge(H[:ntr3], Ys[:ntr3], H[ntr3:], Ys[ntr3:])
        te = mask.copy(); te[:ntr3] = False
        r2r = r2_on(Wr, H[te], Yr[te]) if te.sum() > 50 else float("nan")
        r2s = r2_on(Ws, H[te], Ys[te]) if te.sum() > 50 else float("nan")
        out3.append(dict(step=step, tag=tag, tv=float(tv.mean()),
                         frac=float(mask.mean()), r2_reader=r2r, r2_self=r2s))
        print(f"  {step:5d} [{tag:18s}] TV={tv.mean():.3f} frac={mask.mean():.3f} "
              f"R2r={r2r:.3f} R2s={r2s:.3f}")
    results["P3"] = out3

    # ---------------- P4: degeneracy documentation for lag4000
    n4a = load(f"{RUN}/A_lag4000/p2_ckpt_004000.pt")
    n4b = load(f"{RUN}/A_lag4000/p2_ckpt_008000.pt")
    tt4, _, _, _ = rollout(n4b, n4a, 512, seed=13)
    tv4 = 0.5 * np.abs(policy_at_decisions(n4a, tt4) -
                       policy_at_decisions(n4b, tt4)).sum(-1)
    print(f"\nP4: mean TV(pi_4000, pi_8000) on-context [lag4000] = {tv4.mean():.4f}")
    results["P4"] = float(tv4.mean())
    json.dump(results, open(f"{RUN}/probe_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
