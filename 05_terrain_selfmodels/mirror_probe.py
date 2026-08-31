"""Probes for the precedent-mirror net: does the equilibrium policy still ROUTE through
the in-context self-summary? (Headline prediction: yes -- unlike ambush, where the dodge
compiled past the represented inclination.)

P1  Decode two targets from the residual stream at decision positions:
      (a) p_emp -- the agent's own per-key empirical action distribution for the CURRENT
          key (the true self-summary the mirror uses), exact ground truth;
      (b) last-camp -- onehot of the most recent camp seen in this key (the reputation
          shortcut: track the mirror's reports instead of own behavior).
P2  Steering at the best mid layer: swap top-2 of the decoded p_emp; ROUTING predicts the
    action logits shift anti-aligned with the steered summary ON MIRROR EPISODES
    (note: the predicted conditionality is the OPPOSITE of ambush, where steering moved
    bias episodes only). Controls: random direction; bias episodes; steering the
    last-camp direction instead (dissociation: which representation drives behavior?).

Run: CUDA_VISIBLE_DEVICES=<id> ~/comp_icl/.venv/bin/python mirror_probe.py
"""
from __future__ import annotations
import json, os
import numpy as np
import torch, torch.nn.functional as F

from ambush import Net, World, S, onehot, sample_rows, filt_obs, filt_step, \
    TOK_X0, TOK_A0, TOK_C0, TOK_BOS, BASE
from mirror import Mirror

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUN = os.path.join(BASE, "mirror_runs")
T, BETA = 24, 0.5


def load(path):
    sd = torch.load(path, map_location=DEV)
    net = Net(sd["cfg"]["d"], sd["cfg"]["nl"], sd["cfg"]["nh"]).to(DEV)
    net.load_state_dict(sd["model"]); net.eval()
    return net


@torch.no_grad()
def rollout(net, B, seed):
    rng = np.random.default_rng(seed)
    w = World(B, rng)
    is_mirror = rng.random(B) < BETA
    qcamp = rng.dirichlet(np.full(S, 0.5), B)
    mir = Mirror(B)
    eta = np.full((B, S), 1 / S)
    p_emps = np.zeros((B, T, S)); lastc = np.zeros((B, T, S)); seen = np.zeros((B, T))
    lastcamp = np.full((B, S), -1)                       # last camp per key
    tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
    for t in range(T):
        z = w.emit(); eta = filt_obs(eta, z)
        tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(DEV)], 1)
        pc, p_emp = mir.camp_dist(z, is_mirror, qcamp)
        p_emps[:, t] = p_emp
        lc = lastcamp[np.arange(B), z]
        seen[:, t] = lc >= 0
        lastc[:, t] = np.where(lc[:, None] >= 0, onehot(np.maximum(lc, 0)), 1 / S)
        la = net(tt)[:, -1, TOK_A0:TOK_C0]
        a = torch.multinomial(F.softmax(la, -1), 1).squeeze(1).cpu().numpy()
        c = sample_rows(pc, rng)
        mir.update(z, a)
        lastcamp[np.arange(B), z] = c
        tt = torch.cat([tt, torch.from_numpy(
            np.stack([TOK_A0 + a, TOK_C0 + c], 1)).to(DEV)], 1)
        w.step(); eta = filt_step(eta)
    return tt, is_mirror, p_emps, lastc, seen


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
    return outs


def ridge(Htr, Ytr, Hte, Yte, lam=1.0):
    H1 = np.concatenate([Htr, np.ones((len(Htr), 1))], 1)
    Wm = np.linalg.solve(H1.T @ H1 + lam * np.eye(H1.shape[1]), H1.T @ Ytr)
    H2 = np.concatenate([Hte, np.ones((len(Hte), 1))], 1)
    P = H2 @ Wm
    sse = ((P - Yte) ** 2).sum(0); sst = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1 - sse / sst)), Wm


def main():
    results = {}
    for arm in ("A", "B"):
        net = load(f"{RUN}/{arm}/p2_ckpt_008000.pt")
        tt, is_mirror, p_emps, lastc, seen = rollout(net, 2048, seed=9)
        hs = hiddens(net, tt)
        pos = 1 + 3 * np.arange(T)
        ntr = 1400
        Yp = p_emps.reshape(2048, -1); Yc = lastc.reshape(2048, -1)
        print(f"\n===== arm {arm} =====")
        probes = {}
        for li in range(hs.shape[0]):
            H = hs[li][:, pos]
            r2p, Wp = ridge(H[:ntr].reshape(-1, H.shape[-1]), p_emps[:ntr].reshape(-1, S),
                            H[ntr:].reshape(-1, H.shape[-1]), p_emps[ntr:].reshape(-1, S))
            r2c, Wc = ridge(H[:ntr].reshape(-1, H.shape[-1]), lastc[:ntr].reshape(-1, S),
                            H[ntr:].reshape(-1, H.shape[-1]), lastc[ntr:].reshape(-1, S))
            probes[li] = (r2p, Wp, r2c, Wc)
            print(f"P1 layer {li}: self-summary p_emp R2 = {r2p:.3f}   last-camp R2 = {r2c:.3f}")
        results[f"{arm}_P1"] = {f"L{k}": (round(v[0], 3), round(v[2], 3))
                                for k, v in probes.items()}

        # ---- P2 steering (both directions), decision position t*=16
        tstar = 16; p = 1 + 3 * tstar
        rows = np.arange(2048)
        for li in (3, 4, 5):
            for tag, Wfull in (("self-summary", probes[li][1]), ("last-camp", probes[li][3])):
                Wm = Wfull[:-1]
                Wpinv = np.linalg.pinv(Wm)
                h = hs[li][rows, p]
                y = np.concatenate([h, np.ones((len(h), 1))], 1) @ Wfull
                y_t = y.copy()
                top = y.argmax(1); sec = np.argsort(y, 1)[:, -2]
                y_t[rows, top], y_t[rows, sec] = y[rows, sec], y[rows, top]
                delta = (y_t - y) @ Wpinv
                rnd = np.random.default_rng(0).standard_normal(delta.shape)
                rnd *= (np.linalg.norm(delta, axis=1, keepdims=True) /
                        (np.linalg.norm(rnd, axis=1, keepdims=True) + 1e-9))
                def plog(dvec):
                    outs = []
                    for i in range(0, 2048, 512):
                        sl = slice(i, i + 512)
                        x = net.emb(tt[sl]) + net.pos(
                            torch.arange(tt.shape[1], device=DEV))[None]
                        m = torch.triu(torch.ones(tt.shape[1], tt.shape[1], device=DEV,
                                                  dtype=torch.bool), 1)
                        with torch.no_grad():
                            for bi, b in enumerate(net.blocks):
                                if bi == li:
                                    x[:, p] += torch.from_numpy(dvec[sl]).float().to(DEV)
                                x = b(x, m)
                            lg = net.head(net.lnf(x))[:, p, TOK_A0:TOK_C0]
                        outs.append(lg.cpu())
                    return F.log_softmax(torch.cat(outs), -1).numpy()
                lp0, lp1, lpr = plog(np.zeros_like(delta)), plog(delta), plog(rnd)
                d_old = lp1[rows, top] - lp0[rows, top]
                d_rnd = lpr[rows, top] - lp0[rows, top]
                key = f"{arm}_L{li}_{tag}"
                results[key] = dict(
                    mirror=float(d_old[is_mirror].mean()),
                    bias=float(d_old[~is_mirror].mean()),
                    rand=float(d_rnd[is_mirror].mean()))
                print(f"P2 L{li} steer {tag:12s}: dlogp(old argmax) "
                      f"mirror={results[key]['mirror']:+.3f} bias={results[key]['bias']:+.3f} "
                      f"rand={results[key]['rand']:+.3f}")
    json.dump(results, open(f"{RUN}/probe_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
