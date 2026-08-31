"""Seed validation: for each run (A, A_s1, A_s2, B, B_s1, B_s2), report
  - transition step (first logged step with repB - repM > 0.3),
  - entropy re-opening (max entropy in steps 100..1500 vs value at step 100),
  - final R / repeat split,
  - probe suite on the final ckpt: p_emp decode R2 (L3), surgery TV incoherent vs
    coherent on mirror episodes (the three-channel signature).
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
from ambush import TOK_X0, TOK_A0, TOK_C0, BASE
from mirror_probe import load, rollout, hiddens, ridge, DEV, T
from mirror_surgery import policy_at

RUN = os.path.join(BASE, "mirror_runs")
TSTAR = 16

def probe_final(run):
    net = load(f"{RUN}/{run}/p2_ckpt_008000.pt")
    tt, is_m, p_emps, lastc, seen = rollout(net, 1024, seed=33)
    toks = tt.cpu().numpy()
    acts = toks[:, 2 + 3 * np.arange(T)] - TOK_A0
    hasp = p_emps.max(-1) > 0.34
    prec = p_emps.argmax(-1)
    rep = acts == prec
    repM = float(rep[is_m][hasp[is_m]].mean()); repB = float(rep[~is_m][hasp[~is_m]].mean())
    hs = hiddens(net, tt); pos = 1 + 3 * np.arange(T)
    H = hs[3][:, pos]; ntr = 700
    r2, _ = ridge(H[:ntr].reshape(-1, H.shape[-1]), p_emps[:ntr].reshape(-1, 3),
                  H[ntr:].reshape(-1, H.shape[-1]), p_emps[ntr:].reshape(-1, 3))
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
    return repM, repB, r2, tvi, tvc

def main():
    print(f"{'run':>6} {'trans':>6} {'entmax':>7} {'R_fin':>6} {'repM':>6} {'repB':>6} "
          f"{'R2pemp':>7} {'TVinc':>6} {'TVcoh':>6}")
    out = {}
    for run in ("A", "A_s1", "A_s2", "B", "B_s1", "B_s2"):
        rows = [json.loads(l) for l in open(f"{RUN}/{run}/train2.jsonl")]
        trans = next((r["step"] for r in rows
                      if r["repeat_prec_bias"] - r["repeat_prec_mirror"] > 0.3), None)
        ent100 = next(r["ent"] for r in rows if r["step"] >= 100)
        entmax = max(r["ent"] for r in rows if 100 <= r["step"] <= 1500)
        Rfin = np.mean([r["R"] for r in rows if r["step"] >= 7000])
        repM, repB, r2, tvi, tvc = probe_final(run)
        out[run] = dict(trans=trans, entmax=entmax, ent100=ent100, R=Rfin,
                        repM=repM, repB=repB, r2=r2, tvi=tvi, tvc=tvc)
        print(f"{run:>6} {str(trans):>6} {entmax:7.2f} {Rfin:6.3f} {repM:6.3f} {repB:6.3f} "
              f"{r2:7.3f} {tvi:6.3f} {tvc:6.3f}")
    json.dump(out, open(f"{RUN}/validate.json", "w"), indent=1)

if __name__ == "__main__":
    main()
