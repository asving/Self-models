"""Mirror-game analog of the doppel hop analysis: WHEN does the policy consume the
self-record circuitry, at component grain?

Per checkpoint of mirror_runs/A phase 2: jointly ablate the OWN-ACTION gatherer pair
(L1h0+L1h1 -- key-matched attention on the agent's own past actions; the record's input;
joint ablation closes the redundancy caveat) and, for contrast, the camp-memory head
(L3h0, previously shown prediction-only at the final net). Measure per ablation:
  TV_pol   : policy change at decision positions, mirror episodes (consumption by POLICY)
  d_rep    : change in P(argmax-policy == precedent-argmax) (does avoidance depend on it?)
  d_campCE : change in camp-prediction CE, mirror episodes (consumption by PREDICTION)
Plus the intention-probe per checkpoint: R^2 of null/marg/cond targets on divergent
decision positions (when does prospective pre-updating appear?).
"""
from __future__ import annotations
import numpy as np
import torch, torch.nn.functional as F

from ambush import S, TOK_X0, TOK_A0, TOK_C0, BASE
from mirror_probe import load, rollout, hiddens, DEV, T
from mirror_circuit import wb_forward
from intent_probe import targets_from, ridge_fit, r2_on, counts_series

RUN = f"{BASE}/mirror_runs/A"
CKPTS = [0, 100, 200, 400, 700, 1500, 8000]
PAIR = frozenset([("h", 1, 0), ("h", 1, 1)])
CAMPHEAD = frozenset([("h", 3, 0)])


@torch.no_grad()
def metrics(net, tt, is_m, p_emps, ablate=frozenset()):
    lg = wb_forward(net, tt[:, :-1], ablate=ablate)
    pos = 1 + 3 * np.arange(T)
    pol = F.softmax(lg[:, pos, TOK_A0:TOK_C0], -1)
    camp_pos = 2 + 3 * np.arange(T)
    tgt = tt[:, 1:]
    lsm = F.log_softmax(lg, -1)
    ce_c = -(lsm.gather(-1, tgt[..., None]).squeeze(-1)[:, camp_pos]).cpu().numpy()
    rep = (pol.argmax(-1).cpu().numpy() == p_emps.argmax(-1))
    hasp = p_emps.max(-1) > 0.34
    return (pol.cpu().numpy(),
            float(ce_c[is_m][:, 3:].mean()),
            float(rep[is_m][hasp[is_m]].mean()))


def main():
    torch.set_grad_enabled(False)
    print(f"{'step':>6} | L1-pair(own-action gatherers): {'TVpol':>6} {'d_rep':>7} {'d_CE':>6}"
          f" | L3h0(camp-mem): {'TVpol':>6} {'d_CE':>6} | intent R2 null/marg/cond")
    for step in CKPTS:
        net = load(f"{RUN}/p2_ckpt_{step:06d}.pt")
        tt, is_m, p_emps, lastc, seen = rollout(net, 768, seed=101)
        toks = tt.cpu().numpy()
        keys = toks[:, 1 + 3 * np.arange(T)] - TOK_X0
        acts = toks[:, 2 + 3 * np.arange(T)] - TOK_A0
        pol0, ce0, rep0 = metrics(net, tt, is_m, p_emps)
        out = {}
        for name, ab in (("pair", PAIR), ("camp", CAMPHEAD)):
            pol1, ce1, rep1 = metrics(net, tt, is_m, p_emps, ab)
            tv = float((0.5 * np.abs(pol1 - pol0).sum(-1))[is_m].mean())
            out[name] = (tv, rep1 - rep0, ce1 - ce0)
        # intention probe (compact): best layer over {3,4,5,6}, divergent positions
        B = len(toks)
        cnts = counts_series(keys, acts)
        tg = targets_from(cnts.reshape(-1, S), pol0.reshape(-1, S), acts.reshape(-1))
        hs = hiddens(net, tt)
        tv_d = 0.5 * np.abs(tg["null"] - tg["cond"]).sum(-1)
        div = tv_d > 0.10
        ntr = int(0.7 * B) * T
        te = np.where(div)[0]; te = te[te >= ntr]
        best = {}
        for li in (3, 4, 5, 6):
            H = hs[li][:, 1 + 3 * np.arange(T)].reshape(B * T, -1)
            for t_ in ("null", "marg", "cond"):
                W = ridge_fit(H[:ntr], tg[t_][:ntr])
                r2 = r2_on(W, H[te], tg[t_][te])
                if t_ not in best or r2 > best[t_]:
                    best[t_] = r2
        p, c = out["pair"], out["camp"]
        print(f"{step:6d} | {'':30s} {p[0]:6.3f} {p[1]:+7.3f} {p[2]:6.3f}"
              f" | {'':15s} {c[0]:6.3f} {c[2]:6.3f}"
              f" | {best['null']:.2f}/{best['marg']:.2f}/{best['cond']:.2f}")


if __name__ == "__main__":
    main()
