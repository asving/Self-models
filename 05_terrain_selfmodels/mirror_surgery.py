"""Counterfactual token surgery: does the mirror-episode dodge track OWN PAST ACTIONS
(self-bookkeeping) or PAST CAMPS (reputation-monitoring)? Probe-free causal dissociation.

At decision round t*, in mirror episodes, edit the episode history in the CURRENT key only:
  CF-actions: rotate the agent's own past action tokens (+1 mod 3), keep camps;
  CF-camps:   rotate the past camp tokens (+1 mod 3), keep actions.
Self-bookkeeping   => policy shifts under CF-actions, not CF-camps.
Reputation-monitor => policy shifts under CF-camps, not CF-actions.
Metric: change in log p(a = argmax of the TRUE p_emp) (the action a dodger avoids) and
total-variation of the policy, per edit, mirror vs bias episodes.
"""
from __future__ import annotations
import numpy as np
import torch, torch.nn.functional as F
import os

from ambush import Net, TOK_X0, TOK_A0, TOK_C0, BASE
from mirror_probe import load, rollout, DEV, T

RUN = os.path.join(BASE, "mirror_runs")
TSTAR = 16


@torch.no_grad()
def policy_at(net, tt, p):
    outs = []
    for i in range(0, len(tt), 512):
        outs.append(net(tt[i:i+512])[:, p, TOK_A0:TOK_C0].cpu())
    return F.log_softmax(torch.cat(outs), -1).numpy()


def main():
    for arm in ("A", "B"):
        net = load(f"{RUN}/{arm}/p2_ckpt_008000.pt")
        tt, is_mirror, p_emps, lastc, seen = rollout(net, 2048, seed=21)
        toks = tt.cpu().numpy()
        p = 1 + 3 * TSTAR
        key = toks[:, p] - TOK_X0                        # current key = obs at t*
        # positions of earlier rounds with the same key
        zpos = 1 + 3 * np.arange(TSTAR)
        samekey = toks[:, zpos] - TOK_X0 == key[:, None]           # (B, TSTAR)
        nsk = samekey.sum(1)
        cf_a, cf_c, cf_b = toks.copy(), toks.copy(), toks.copy()
        for b in range(len(toks)):
            idx = np.where(samekey[b])[0]
            ap = 2 + 3 * idx; cp = 3 + 3 * idx
            cf_a[b, ap] = TOK_A0 + ((cf_a[b, ap] - TOK_A0 + 1) % 3)
            cf_c[b, cp] = TOK_C0 + ((cf_c[b, cp] - TOK_C0 + 1) % 3)
            cf_b[b, ap] = TOK_A0 + ((cf_b[b, ap] - TOK_A0 + 1) % 3)   # coherent rotation:
            cf_b[b, cp] = TOK_C0 + ((cf_b[b, cp] - TOK_C0 + 1) % 3)   # mirror signature kept
        tt_a = torch.from_numpy(cf_a).to(DEV); tt_c = torch.from_numpy(cf_c).to(DEV)
        tt_b = torch.from_numpy(cf_b).to(DEV)
        lp0, lpa, lpc = policy_at(net, tt, p), policy_at(net, tt_a, p), policy_at(net, tt_c, p)
        lpb = policy_at(net, tt_b, p)
        avoid = p_emps[:, TSTAR].argmax(1)               # true precedent argmax (dodge target)
        sel = (nsk >= 2)                                 # episodes with real precedent
        m, b_ = sel & is_mirror, sel & ~is_mirror
        tv_a = 0.5 * np.abs(np.exp(lpa) - np.exp(lp0)).sum(1)
        tv_c = 0.5 * np.abs(np.exp(lpc) - np.exp(lp0)).sum(1)
        print(f"== arm {arm} ==  (n mirror={m.sum()}, bias={b_.sum()})")
        print(f"  dlogp(avoid-target) CF-actions: mirror={np.mean((lpa-lp0)[m, avoid[m]]):+.3f} "
              f"bias={np.mean((lpa-lp0)[b_, avoid[b_]]):+.3f}")
        print(f"  dlogp(avoid-target) CF-camps  : mirror={np.mean((lpc-lp0)[m, avoid[m]]):+.3f} "
              f"bias={np.mean((lpc-lp0)[b_, avoid[b_]]):+.3f}")
        print(f"  policy TV under CF-actions: mirror={tv_a[m].mean():.3f} bias={tv_a[b_].mean():.3f}")
        print(f"  policy TV under CF-camps  : mirror={tv_c[m].mean():.3f} bias={tv_c[b_].mean():.3f}")
        # coherent rotation: apparent precedent argmax becomes avoid+1; dodge-following
        # predicts logp(old avoid-target) RISES and logp(new avoid-target = avoid+1) FALLS
        new_avoid = (avoid + 1) % 3
        print(f"  COHERENT rotation (mirror): dlogp(old avoid)={np.mean((lpb-lp0)[m, avoid[m]]):+.3f}"
              f"  dlogp(new avoid)={np.mean((lpb-lp0)[m, new_avoid[m]]):+.3f}"
              f"  TV={0.5*np.abs(np.exp(lpb)-np.exp(lp0)).sum(1)[m].mean():.3f}")
        print(f"  COHERENT rotation (bias)  : dlogp(old avoid)={np.mean((lpb-lp0)[b_, avoid[b_]]):+.3f}"
              f"  dlogp(new avoid)={np.mean((lpb-lp0)[b_, new_avoid[b_]]):+.3f}")
        tv_b = 0.5 * np.abs(np.exp(lpb) - np.exp(lp0)).sum(1)
        import json
        json.dump(dict(cfa_m=float(tv_a[m].mean()), cfa_b=float(tv_a[b_].mean()),
                       cfc_m=float(tv_c[m].mean()), cfc_b=float(tv_c[b_].mean()),
                       coh_m=float(tv_b[m].mean()), coh_b=float(tv_b[b_].mean())),
                  open(os.path.join(RUN, f"surgery_{arm}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
