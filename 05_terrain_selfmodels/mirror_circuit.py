"""Circuit analysis of the precedent-mirror net (whitebox methodology).

0. Exact recompute harness from raw weights (per-head patterns + OV contributions),
   verified against model.forward to <1e-3 -- everything gates on that assert.
1. Ablation matrix: zero each head / MLP -> deltas on the channel metrics:
     camp-CE (mirror) = the record/prediction channel
     type-conditionality (dodge_M - dodge_B) = match-detector -> dodge gating
     terrain quality (bias episodes) = belief pathway
2. Attention anatomy of implicated heads at decision positions: mass on same-key
   own-action tokens vs same-key camp tokens vs other (bookkeeping = key-matched attention).
3. Targeted causal tests (channel dissociation at component grain), incl. across
   checkpoints 0 / 200 / 8000: does ablating the record components move the POLICY
   early (record-following) but only the PREDICTION late (introspective dodge)?

Run: CUDA_VISIBLE_DEVICES=<id> ~/comp_icl/.venv/bin/python mirror_circuit.py
"""
from __future__ import annotations
import json, os
import numpy as np
import torch, torch.nn.functional as F

from ambush import TOK_X0, TOK_A0, TOK_C0, BASE, E_MAT
from mirror import Mirror
from mirror_probe import load, rollout, DEV, T

RUN = os.path.join(BASE, "mirror_runs")
M_DRIFT = None
from ambush import M_DRIFT as _MD
M_DRIFT = _MD
NL, NH, DH = 6, 4, 16
TSTAR_MIN = 6            # skip early rounds (no precedent yet)


# ---------------------------------------------------------------- exact recompute
def wb_forward(net, toks, ablate=frozenset(), want_attn=False):
    """Hand-rolled forward from raw weights. ablate: set of ('h', l, head) / ('mlp', l)."""
    B, L = toks.shape
    x = net.emb(toks) + net.pos(torch.arange(L, device=DEV))[None]
    causal = torch.full((L, L), float("-inf"), device=DEV).triu(1)
    attns = []
    for li, blk in enumerate(net.blocks):
        h = F.layer_norm(x, (x.shape[-1],), blk.ln1.weight, blk.ln1.bias)
        W, bqkv = blk.attn.in_proj_weight, blk.attn.in_proj_bias
        q, k, v = (h @ W.T + bqkv).chunk(3, -1)
        q = q.view(B, L, NH, DH).transpose(1, 2)
        k = k.view(B, L, NH, DH).transpose(1, 2)
        v = v.view(B, L, NH, DH).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-1, -2) / DH ** 0.5 + causal, -1)
        if want_attn:
            attns.append(att.detach())
        z = att @ v                                       # (B,NH,L,DH)
        for hd in range(NH):
            if ("h", li, hd) in ablate:
                z[:, hd] = 0.0
        z = z.transpose(1, 2).reshape(B, L, -1)
        x = x + z @ blk.attn.out_proj.weight.T + blk.attn.out_proj.bias
        if ("mlp", li) not in ablate:
            h2 = F.layer_norm(x, (x.shape[-1],), blk.ln2.weight, blk.ln2.bias)
            x = x + blk.mlp(h2)
    x = F.layer_norm(x, (x.shape[-1],), net.lnf.weight, net.lnf.bias)
    lg = x @ net.head.weight.T + net.head.bias
    return (lg, attns) if want_attn else lg


# ---------------------------------------------------------------- metrics
def eta_from_tokens(toks_np):
    B = len(toks_np)
    eta = np.full((B, 3), 1 / 3)
    etas = np.zeros((B, T, 3))
    for t in range(T):
        z = toks_np[:, 1 + 3 * t] - TOK_X0
        eta = eta * E_MAT.T[z]; eta /= eta.sum(-1, keepdims=True)
        etas[:, t] = eta
        eta = eta @ M_DRIFT; eta /= eta.sum(-1, keepdims=True)
    return etas

def channel_metrics(lg, toks, is_m, etas, p_emps, tsel):
    """All teacher-forced. tsel: rounds to include."""
    lsm = F.log_softmax(lg, -1)
    pos_a = 1 + 3 * np.array(tsel)                        # predicts a_t
    pos_c = 2 + 3 * np.array(tsel)                        # predicts c_t
    tgt = toks[:, 1:]
    ce_c = -(lsm[:, pos_c].gather(-1, tgt[:, pos_c][..., None]).squeeze(-1))
    pol = lsm[:, pos_a, TOK_A0:TOK_C0].argmax(-1).cpu().numpy()
    ea = etas[:, tsel].argmax(-1)
    pe = p_emps[:, tsel].argmax(-1)
    m = np.asarray(is_m)
    dodge = pol != ea
    rep = pol == pe
    return dict(
        ce_camp_m=float(ce_c[torch.from_numpy(m).to(DEV)].mean()),
        ce_camp_b=float(ce_c[torch.from_numpy(~m).to(DEV)].mean()),
        typecond=float(dodge[m].mean() - dodge[~m].mean()),
        terrain_b=float((~dodge)[~m].mean()),
        rep_m=float(rep[m].mean()))


def main():
    torch.set_grad_enabled(False)
    net = load(f"{RUN}/A/p2_ckpt_008000.pt")
    tt, is_m, p_emps, lastc, seen = rollout(net, 1536, seed=45)
    toks_np = tt.cpu().numpy()
    etas = eta_from_tokens(toks_np)
    tsel = list(range(TSTAR_MIN, T))
    # step 0: verify harness
    lg_wb = wb_forward(net, tt[:64])
    lg_ref = net(tt[:64])
    diff = float((lg_wb - lg_ref).abs().max())
    print(f"harness max|diff| vs model.forward = {diff:.2e}")
    assert diff < 1e-3, "recompute harness mismatch"

    lg0 = wb_forward(net, tt)
    base = channel_metrics(lg0, tt, is_m, etas, p_emps, tsel)
    print("clean:", json.dumps({k: round(v, 3) for k, v in base.items()}))

    # ---------------- 1. ablation matrix
    comps = [("h", l, h) for l in range(NL) for h in range(NH)] + \
            [("mlp", l) for l in range(NL)]
    rows = []
    p0 = F.softmax(lg0[:, 1 + 3 * np.array(tsel), TOK_A0:TOK_C0], -1)
    for c in comps:
        lg = wb_forward(net, tt, ablate=frozenset([c]))
        mt = channel_metrics(lg, tt, is_m, etas, p_emps, tsel)
        p1 = F.softmax(lg[:, 1 + 3 * np.array(tsel), TOK_A0:TOK_C0], -1)
        tv = 0.5 * (p1 - p0).abs().sum(-1).mean(1).cpu().numpy()
        rows.append(dict(comp=str(c),
                         d_ce_m=round(mt["ce_camp_m"] - base["ce_camp_m"], 3),
                         d_ce_b=round(mt["ce_camp_b"] - base["ce_camp_b"], 3),
                         d_typecond=round(mt["typecond"] - base["typecond"], 3),
                         d_terrain=round(mt["terrain_b"] - base["terrain_b"], 3),
                         tv_m=round(float(tv[is_m].mean()), 3),
                         tv_b=round(float(tv[~is_m].mean()), 3)))
    rows.sort(key=lambda r: -abs(r["d_typecond"]))
    print("\nablation matrix (sorted by |d_typecond|; top 12):")
    print(f"{'comp':>16} {'dCEm':>6} {'dCEb':>6} {'dTC':>6} {'dTerr':>6} {'TVm':>6} {'TVb':>6}")
    for r in rows[:12]:
        print(f"{r['comp']:>16} {r['d_ce_m']:6.3f} {r['d_ce_b']:6.3f} {r['d_typecond']:6.3f} "
              f"{r['d_terrain']:6.3f} {r['tv_m']:6.3f} {r['tv_b']:6.3f}")
    json.dump(rows, open(f"{RUN}/circuit_ablation.json", "w"), indent=1)

    # ---------------- 2. attention anatomy at decision positions
    _, attns = wb_forward(net, tt[:512], want_attn=True)
    key = toks_np[:512, 1 + 3 * np.arange(T)] - TOK_X0     # (b, T)
    print("\nattention anatomy at decision positions (mean mass, rounds >= 6):")
    print(f"{'head':>8} {'sameK_a':>8} {'sameK_c':>8} {'sameK_x':>8} {'diffK':>7} {'cur_x':>6}")
    anat = {}
    for li in range(NL):
        att = attns[li].cpu().numpy()                     # (b, NH, L, L)
        for hd in range(NH):
            cat = np.zeros(5)
            n = 0
            for t in range(TSTAR_MIN, T):
                p = 1 + 3 * t
                A = att[:, hd, p, :]                      # (b, L)
                for tp in range(t):
                    sk = key[:, tp] == key[:, t]
                    cat[2] += (A[:, 1 + 3 * tp] * sk).sum()
                    cat[0] += (A[:, 2 + 3 * tp] * sk).sum()
                    cat[1] += (A[:, 3 + 3 * tp] * sk).sum()
                    cat[3] += (A[:, [1 + 3 * tp, 2 + 3 * tp, 3 + 3 * tp]].sum(1) * ~sk).sum()
                cat[4] += A[:, p].sum()
                n += len(A)
            cat /= n
            anat[f"L{li}h{hd}"] = cat.tolist()
            if cat[0] > 0.08 or cat[1] > 0.08:
                print(f"  L{li}h{hd} {cat[0]:8.3f} {cat[1]:8.3f} {cat[2]:8.3f} "
                      f"{cat[3]:7.3f} {cat[4]:6.3f}")
    json.dump(anat, open(f"{RUN}/circuit_attn.json", "w"), indent=1)

    # ---------------- 3. usage migration across checkpoints
    # pick record components: top-2 by d_ce_m with small |d_typecond|
    rec_sorted = sorted(rows, key=lambda r: -(r["d_ce_m"]))
    rec_comps = [eval(r["comp"]) for r in rec_sorted
                 if abs(r["d_typecond"]) < 0.1][:2]
    det_comps = [eval(r["comp"]) for r in rows[:2]]
    print(f"\nrecord comps: {rec_comps}   detector comps: {det_comps}")
    print(f"{'ckpt':>6} {'ablate':>10} {'dCEm':>7} {'TVm':>6} {'dTC':>7}")
    mig = []
    for step in (0, 200, 8000):
        net_s = load(f"{RUN}/A/p2_ckpt_{step:06d}.pt")
        tt_s, is_s, pe_s, _, _ = rollout(net_s, 1024, seed=46)
        et_s = eta_from_tokens(tt_s.cpu().numpy())
        lg_c = wb_forward(net_s, tt_s)
        b_s = channel_metrics(lg_c, tt_s, is_s, et_s, pe_s, tsel)
        pc = F.softmax(lg_c[:, 1 + 3 * np.array(tsel), TOK_A0:TOK_C0], -1)
        for name, cs in (("record", rec_comps), ("detector", det_comps)):
            lg_a = wb_forward(net_s, tt_s, ablate=frozenset(cs))
            m_a = channel_metrics(lg_a, tt_s, is_s, et_s, pe_s, tsel)
            pa = F.softmax(lg_a[:, 1 + 3 * np.array(tsel), TOK_A0:TOK_C0], -1)
            tvm = float((0.5 * (pa - pc).abs().sum(-1).mean(1).cpu().numpy())[is_s].mean())
            mig.append(dict(step=step, which=name,
                            d_ce_m=round(m_a["ce_camp_m"] - b_s["ce_camp_m"], 3),
                            tv_m=round(tvm, 3),
                            d_tc=round(m_a["typecond"] - b_s["typecond"], 3)))
            print(f"{step:6d} {name:>10} {mig[-1]['d_ce_m']:7.3f} {mig[-1]['tv_m']:6.3f} "
                  f"{mig[-1]['d_tc']:7.3f}")
    json.dump(mig, open(f"{RUN}/circuit_migration.json", "w"), indent=1)


if __name__ == "__main__":
    main()
