"""Formation timeline of the three channels across arm-A phase-2 checkpoints.

Per checkpoint: (a) behavioral self-avoidance (repeat-precedent, mirror vs bias);
(b) the bookkeeping channel: p_emp decode R^2 at layer 3; (c) causal surgery at t*=16:
policy TV under INCOHERENT edit (rotate own actions only -- match-detector signature)
vs COHERENT rotation (actions+camps -- record-consultation signature) on mirror episodes.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch, torch.nn.functional as F

from ambush import TOK_X0, TOK_A0, TOK_C0, BASE
from mirror_probe import load, rollout, hiddens, ridge, DEV, T
from mirror_surgery import policy_at

RUN = os.path.join(BASE, "mirror_runs")
CKPTS = [0, 50, 100, 200, 400, 700, 1000, 2000, 4000, 8000]
TSTAR = 16


def main():
    out = []
    print(f"{'step':>6} {'repM':>6} {'repB':>6} {'dodgeM':>7} {'R2_pemp':>8} "
          f"{'TVincoh':>8} {'TVcoh':>7}")
    for step in CKPTS:
        net = load(f"{RUN}/A/p2_ckpt_{step:06d}.pt")
        tt, is_m, p_emps, lastc, seen = rollout(net, 1024, seed=33)
        toks = tt.cpu().numpy()
        acts = toks[:, 2 + 3 * np.arange(T)] - TOK_A0
        hasp = p_emps.max(-1) > 0.34
        prec = p_emps.argmax(-1)
        rep = acts == prec
        repM = float(rep[is_m][hasp[is_m]].mean())
        repB = float(rep[~is_m][hasp[~is_m]].mean())
        eta_arg = None
        hs = hiddens(net, tt)
        pos = 1 + 3 * np.arange(T)
        H = hs[3][:, pos]; ntr = 700
        r2, _ = ridge(H[:ntr].reshape(-1, H.shape[-1]), p_emps[:ntr].reshape(-1, 3),
                      H[ntr:].reshape(-1, H.shape[-1]), p_emps[ntr:].reshape(-1, 3))
        # surgery
        p = 1 + 3 * TSTAR
        key = toks[:, p] - TOK_X0
        zpos = 1 + 3 * np.arange(TSTAR)
        samekey = toks[:, zpos] - TOK_X0 == key[:, None]
        nsk = samekey.sum(1)
        cf_a, cf_b = toks.copy(), toks.copy()
        for b in range(len(toks)):
            idx = np.where(samekey[b])[0]
            ap, cp = 2 + 3 * idx, 3 + 3 * idx
            cf_a[b, ap] = TOK_A0 + ((cf_a[b, ap] - TOK_A0 + 1) % 3)
            cf_b[b, ap] = TOK_A0 + ((cf_b[b, ap] - TOK_A0 + 1) % 3)
            cf_b[b, cp] = TOK_C0 + ((cf_b[b, cp] - TOK_C0 + 1) % 3)
        lp0 = policy_at(net, tt, p)
        lpa = policy_at(net, torch.from_numpy(cf_a).to(DEV), p)
        lpb = policy_at(net, torch.from_numpy(cf_b).to(DEV), p)
        sel = (nsk >= 2) & is_m
        tvi = float((0.5 * np.abs(np.exp(lpa) - np.exp(lp0)).sum(1))[sel].mean())
        tvc = float((0.5 * np.abs(np.exp(lpb) - np.exp(lp0)).sum(1))[sel].mean())
        dodgeM = float((acts != prec)[is_m].mean())
        rec = dict(step=step, repM=repM, repB=repB, r2_pemp=float(r2),
                   tv_incoh=tvi, tv_coh=tvc)
        out.append(rec)
        print(f"{step:6d} {repM:6.3f} {repB:6.3f} {1-repM:7.3f} {r2:8.3f} "
              f"{tvi:8.3f} {tvc:7.3f}")
    json.dump(out, open(f"{RUN}/formation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
