"""RECOMPUTE vs MESSAGE-PASS: how does the RPS-IM net access its own previous action?

The net sees only o_t=(a_t-b_t)%3. To decode b_t it needs a_t, whose policy p_t was computed at the
PREVIOUS position. Two routes for that info to reach position tau=t+1:
  (M) message-pass: attention at tau reads position t's residual, which carries (a partial
      computation of) p_t. In an nl-layer net only x_ell (block-ell INPUT, ell>=1) at position t is
      readable; the finished policy (post-lnf) is NEVER readable by attention.
  (R) recompute: tau's own circuit re-derives p_t from the raw tokens o_{<t} (layer-0 K/V are pure
      token embeddings), duplicating the policy circuit shifted one position ("partial attention").
  (H) hardcode: p_t not context-dependent (e.g. fixed opening); no cross-position flow needed.

Subcommands (python rps_route.py <cmd> [run_name]):
  mech   behavioral characterization: opening determinism, mode-switch frequency by round,
         sharpness, discriminating-event (m_t != m_{t+1}) frequency. Validates manual forward.
  probe  per-layer probes at tau for: prev mode m_t, realized a_t, decoded b_hat, true b_t, current
         mode m_{t+1}; baselines (round-only, degeneracy m_{t+1}); discriminating subset; feasibility
         probe of m_t at its OWN position t per layer; forced-random control (mode vs realized).
  train  probe m_t at tau (final resid) across step checkpoints -> signal vs training.
  attn   per-(layer,head) probes of the head OUTPUT at tau for m_t (+ attention-mass profile).
  patch  THE DECISIVE TEST: recompute tau's residual row with per-layer K/V sources chosen from
         {clean, corrupt} cached contexts. R predicts belief-about-own-action follows the layer-0
         (token) source; M predicts it follows the layer>=1 (computed) source; H predicts neither.
         Readouts: probe-decoded a_hat, cyclic b_hat=(a_hat-o)%3, behavioral policy shift.

Runs: rps_b0.0 (2L d64), pretrain_bias (2L d64), rpsbig_b0.0 (6L d256). All CPU.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
torch.set_num_threads(8)
from rps_im import RPSNet                      # noqa: E402
from wb_pretrain import gen_games, ridge_probe  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
DEV = "cpu"


def load(name):
    ck = torch.load(f"{BASE}/rps_runs/{name}.pt", map_location=DEV)
    a = ck["args"]; net = RPSNet(a["d_model"], a["n_layer"], a["n_head"], a["T"])
    net.load_state_dict(ck["state"]); net.eval(); return net, a


# ===================================================================== #
# manual forward exposing every stream + K/V-source control
# ===================================================================== #
@torch.no_grad()
def manual_forward(net, tok, collect_heads=False):
    """Exact reimplementation of RPSNet.forward. Returns dict with:
    x[ell]   : block-ell INPUT residual (B,L,d), ell=0..nl  (x[nl]=pre-lnf; what K/V at layer ell see)
    attnout  : list of per-layer attention outputs (B,L,d)
    headout  : (if collect_heads) list of per-layer per-head outputs (B,nh,L,d) AFTER out_proj slice
    aw       : list of per-layer attention weights (B,nh,L,L)
    lnf      : final residual (B,L,d);  logits: act_head(lnf)"""
    B, L = tok.shape
    x = net.emb(tok) + net.pos(torch.arange(L))[None]
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    xs, attns, heads, aws = [x.clone()], [], [], []
    for blk in net.blocks:
        h = blk.ln1(x)
        a_out, hout, aw = mha_manual(blk.attn, h, mask, collect_heads)
        x = x + a_out
        x = x + blk.mlp(blk.ln2(x))
        xs.append(x.clone()); attns.append(a_out); heads.append(hout); aws.append(aw)
    lnf = net.lnf(x)
    return dict(x=xs, attnout=attns, headout=heads, aw=aws, lnf=lnf,
                logits=net.act_head(lnf))


def mha_manual(attn, h, mask, collect_heads=False):
    """nn.MultiheadAttention (batch_first, packed in_proj) manual. h (B,L,d)."""
    B, L, d = h.shape; nh = attn.num_heads; dh = d // nh
    W = attn.in_proj_weight; bb = attn.in_proj_bias
    q = h @ W[:d].T + bb[:d]; k = h @ W[d:2*d].T + bb[d:2*d]; v = h @ W[2*d:].T + bb[2*d:]
    q = q.view(B, L, nh, dh).transpose(1, 2); k = k.view(B, L, nh, dh).transpose(1, 2)
    v = v.view(B, L, nh, dh).transpose(1, 2)
    sc = (q @ k.transpose(-1, -2)) / np.sqrt(dh)
    sc = sc.masked_fill(mask[None, None], float("-inf"))
    aw = F.softmax(sc, -1)
    o = aw @ v                                              # (B,nh,L,dh)
    ocat = o.transpose(1, 2).reshape(B, L, d)
    out = ocat @ attn.out_proj.weight.T + attn.out_proj.bias
    hout = None
    if collect_heads:  # per-head contribution through out_proj (bias excluded)
        Wo = attn.out_proj.weight                            # (d, d)
        hout = torch.stack([o[:, i] @ Wo[:, i*dh:(i+1)*dh].T for i in range(nh)], 1)  # (B,nh,L,d)
    return out, hout, aw


@torch.no_grad()
def row_forward(net, tok_tau, tau, kv_src, pos_mask=None):
    """Compute ONLY row tau's stream. tok_tau (B,) = token at position tau (kept clean).
    kv_src: list over layers of cached block-input residuals (B,tau,d) for positions 0..tau-1
            (choose clean/corrupt/mixed per layer OUTSIDE this fn).
    pos_mask: optional (tau,) bool -- restrict which past positions the patch... (mixing is done by
            caller; this fn just consumes whatever kv_src rows it is given).
    Returns y_lnf (B,d), logits (B,3), y per layer."""
    B = tok_tau.shape[0]
    y = net.emb(tok_tau) + net.pos(torch.tensor([tau]))[0][None]
    ys = [y.clone()]
    for ell, blk in enumerate(net.blocks):
        rows = torch.cat([kv_src[ell], y[:, None]], 1)       # (B,tau+1,d) K/V input rows
        h = blk.ln1(rows)
        d = h.shape[-1]; nh = blk.attn.num_heads; dh = d // nh
        W = blk.attn.in_proj_weight; bb = blk.attn.in_proj_bias
        qv = h[:, -1] @ W[:d].T + bb[:d]                     # query = row tau only
        k = (h @ W[d:2*d].T + bb[d:2*d]).view(B, tau+1, nh, dh).transpose(1, 2)
        v = (h @ W[2*d:].T + bb[2*d:]).view(B, tau+1, nh, dh).transpose(1, 2)
        qv = qv.view(B, nh, 1, dh)
        aw = F.softmax((qv @ k.transpose(-1, -2)) / np.sqrt(dh), -1)
        o = (aw @ v).transpose(1, 2).reshape(B, d)
        y = y + o @ blk.attn.out_proj.weight.T + blk.attn.out_proj.bias
        y = y + blk.mlp(blk.ln2(y))
        ys.append(y.clone())
    lnf = net.lnf(y)
    return lnf, net.act_head(lnf), ys


# ===================================================================== #
# helpers
# ===================================================================== #
def cat_probe(Xtr, ytr, Xte, yte, lam=10.0):
    """3-way categorical via ridge-to-onehot. Returns held-out acc and the (W,b-free) weights."""
    W, _, pred = ridge_probe(Xtr, F.one_hot(ytr, 3).float(), Xte, F.one_hot(yte, 3).float(), lam)
    return (pred.argmax(1) == yte).float().mean().item(), W


def buckets(T):
    return [("open t=1-3", 1, 4), ("early t=4-9", 4, 10), ("mid t=10-24", 10, 25),
            ("late t=25+", 25, T)]


LAYER_NAMES = None


def stream_list(mf, nl):
    """[(name, tensor(B,L,d))] of probe-able streams."""
    out = [("x0=emb", mf["x"][0])]
    for ell in range(nl):
        out.append((f"att{ell}out", mf["attnout"][ell]))
        out.append((f"x{ell+1}", mf["x"][ell + 1]))
    out.append(("lnf", mf["lnf"]))
    return out


# ===================================================================== #
def cmd_mech(name):
    net, a = load(name); T = a["T"]
    torch.manual_seed(0)
    NB = 4000 if a["d_model"] <= 64 else 2000
    seq, bias, A, Bm, O, P = gen_games(net, NB, T, np.random.default_rng(1))
    mf = manual_forward(net, seq)
    err = (mf["logits"] - net(seq)[0]).abs().max().item()
    print(f"=== {name}: {a['n_layer']}L d{a['d_model']} {a['n_head']}h T={T} | "
          f"manual-forward max err {err:.2e} ===")
    mode = P.argmax(-1)
    print(f"opening move dist: {torch.bincount(mode[:,0], minlength=3).tolist()} "
          f"(top frac {torch.bincount(mode[:,0]).max().item()/len(mode):.3f})")
    ent = -(P * (P + 1e-9).log()).sum(-1)
    maxp = P.max(-1).values
    sw = (mode[:, 1:] != mode[:, :-1]).float()               # mode switch at round t (vs t-1)
    print(f"E[max p] overall {maxp.mean():.3f} | entropy early(t<10) {ent[:,:10].mean():.3f} "
          f"late {ent[:,30:].mean():.3f}")
    print("round-resolved: frac mode-switch (m_t != m_(t-1)) and E[max p]:")
    for lab, lo, hi in buckets(T):
        print(f"  {lab:14s} switch={sw[:, lo-1:hi-1].mean():.3f}  E[max p]={maxp[:, lo:hi].mean():.3f}")
    br_true = (bias.argmax(1) + 1) % 3
    print(f"late mode == BR(true bias): {(mode[:, 25:] == br_true[:, None]).float().mean():.3f}")
    pay = torch.where(((A - Bm) % 3) == 1, 1., torch.where(((A - Bm) % 3) == 2, -1., 0.))
    print(f"payoff/round: {pay.mean():+.3f}")


# ===================================================================== #
def cmd_probe(name):
    net, a = load(name); T = a["T"]; nl = a["n_layer"]; d = a["d_model"]
    torch.manual_seed(0)
    NB = 4000 if d <= 64 else 1500
    tr = gen_games(net, NB, T, np.random.default_rng(1))
    te = gen_games(net, NB, T, np.random.default_rng(2))
    mtr, mte = manual_forward(net, tr[0]), manual_forward(net, te[0])
    print(f"=== {name}: probes for PREV action m_t at position tau=t+1 (held-out acc) ===")
    print("targets: m_t=argmax p_t (what the net can know) | a_t realized | bhat=(m_t-o_t)%3 | "
          "true b_t | m_(t+1) current")

    def series(g, mf):
        _, bias, A, Bm, O, P = g
        mode = P.argmax(-1)                                   # (B,T) m_t
        t_idx = torch.arange(1, T)                            # rounds t with a PREV round (tau=t in seq = round t-1?)
        # position tau = t+1 holds o_t; targets at tau for t=0..T-1; use t>=1 so m_{t} exists AND
        # tau<=T. tau range: 2..T  <-> t=1..T-1.  (t=0's prev action is the opening; skip.)
        return mode, A, Bm, O, P

    mode_tr, A_tr, B_tr, O_tr, P_tr = series(tr, mtr)
    mode_te, A_te, B_te, O_te, P_te = series(te, mte)

    streams_tr = stream_list(mtr, nl); streams_te = stream_list(mte, nl)

    def probe_at_tau(target_tr, target_te, lo, hi, subset=None):
        """probe residual at tau=t+1 for target[:,t], rounds t in [lo,hi). subset: ((B,T) bool
        for train, same for test) -- probe is FIT on the train subset (so it cannot inherit the
        current-policy direction) and evaluated on the test subset."""
        accs = {}
        for (nm, Str), (_, Ste) in zip(streams_tr, streams_te):
            Xtr = torch.cat([Str[:, t+1] for t in range(lo, hi)])
            Xte = torch.cat([Ste[:, t+1] for t in range(lo, hi)])
            ytr = torch.cat([target_tr[:, t] for t in range(lo, hi)])
            yte = torch.cat([target_te[:, t] for t in range(lo, hi)])
            if subset is not None:
                str_ = torch.cat([subset[0][:, t] for t in range(lo, hi)])
                ste_ = torch.cat([subset[1][:, t] for t in range(lo, hi)])
                acc, _ = cat_probe(Xtr[str_], ytr[str_], Xte[ste_], yte[ste_])
                accs[nm] = acc
            else:
                acc, _ = cat_probe(Xtr, ytr, Xte, yte)
                accs[nm] = acc
        return accs

    # ---- headline: per-layer acc for each target, per round bucket ----
    bhat_tr = (mode_tr - O_tr) % 3; bhat_te = (mode_te - O_te) % 3
    curr_tr = torch.cat([mode_tr[:, 1:], mode_tr[:, -1:]], 1)  # m_{t+1} aligned at t (last dup)
    curr_te = torch.cat([mode_te[:, 1:], mode_te[:, -1:]], 1)
    targets = [("m_t (prev mode)", mode_tr, mode_te), ("a_t (realized)", A_tr, A_te),
               ("bhat_t", bhat_tr, bhat_te), ("b_t (true)", B_tr, B_te),
               ("m_t+1 (curr)", curr_tr, curr_te)]
    for lab, lo, hi in buckets(T):
        hi = min(hi, T - 1)
        print(f"\n-- rounds {lab} --   " + "  ".join(f"{nm:>9s}" for nm, _ in streams_tr))
        for tname, ytr2, yte2 in targets:
            accs = probe_at_tau(ytr2, yte2, lo, hi)
            print(f"  {tname:16s} " + "  ".join(f"{accs[nm]:9.3f}" for nm, _ in streams_tr))
        # baselines for m_t in this bucket
        yb_tr = torch.cat([mode_tr[:, t] for t in range(lo, hi)])
        yb_te = torch.cat([mode_te[:, t] for t in range(lo, hi)])
        # round-only: best per-round constant
        acc_round = np.mean([torch.bincount(mode_te[:, t], minlength=3).max().item() / len(mode_te)
                             for t in range(lo, hi)])
        acc_degen = np.mean([(mode_te[:, t] == curr_te[:, t]).float().mean().item()
                             for t in range(lo, hi)])
        print(f"  [baselines m_t] round-only={acc_round:.3f}  m_t==m_t+1 degeneracy={acc_degen:.3f} "
              f" E[max p_t]={torch.cat([P_te[:,t].max(-1).values for t in range(lo,hi)]).mean():.3f}")

    # ---- discriminating subset: rounds where m_t != m_{t+1} (probe cannot cheat off current) ----
    print("\n-- DISCRIMINATING subset (m_t != m_(t+1)), rounds 1..T-2, m_t probe --")
    sub_tr = mode_tr != curr_tr; sub_te = mode_te != curr_te
    accs = probe_at_tau(mode_tr, mode_te, 1, T - 1, subset=(sub_tr, sub_te))
    n_ev = sub_te[:, 1:T-1].sum().item()
    print(f"  n={n_ev}   " + "  ".join(f"{nm}={accs[nm]:.3f}" for nm, _ in streams_tr))
    print(f"  [chance on subset: m_t never == m_t+1 here; round-only baseline recomputed:] "
          f"{np.mean([torch.bincount(mode_te[sub_te[:,t],t],minlength=3).max().item()/max(sub_te[:,t].sum().item(),1) for t in range(1,T-1) if sub_te[:,t].sum()>20]):.3f}")

    # ---- FEASIBILITY: where is m_t decodable at its OWN position t? ----
    print("\n-- FEASIBILITY: m_t probed at its OWN position t (what K/V could message-pass) --")
    print("   (x_ell at pos t is the ONLY content attention at tau can read; lnf is unreadable)")
    for lab, lo, hi in [("all t=1..T-1", 1, T - 1)]:
        accs = {}
        for (nm, Str), (_, Ste) in zip(streams_tr, streams_te):
            Xtr = torch.cat([Str[:, t] for t in range(lo, hi)])
            Xte = torch.cat([Ste[:, t] for t in range(lo, hi)])
            ytr = torch.cat([mode_tr[:, t] for t in range(lo, hi)])
            yte = torch.cat([mode_te[:, t] for t in range(lo, hi)])
            accs[nm] = cat_probe(Xtr, ytr, Xte, yte)[0]
        print(f"  {lab:14s} " + "  ".join(f"{nm}={accs[nm]:.3f}" for nm, _ in streams_tr))

    # ---- forced-random control: net plays uniform random; does tau encode MODE or REALIZED? ----
    print("\n-- FORCED-RANDOM control (actions uniform; net never told the draw) --")
    trf = gen_games(net, NB, T, np.random.default_rng(5), force_sharp=0.0)
    tef = gen_games(net, NB, T, np.random.default_rng(6), force_sharp=0.0)
    mtrf, mtef = manual_forward(net, trf[0]), manual_forward(net, tef[0])
    modef_tr = trf[5].argmax(-1); modef_te = tef[5].argmax(-1)
    Af_tr, Af_te = trf[2], tef[2]
    for tname, ytr2, yte2 in [("mode(policy)", modef_tr, modef_te), ("realized a_t", Af_tr, Af_te)]:
        Xtr = torch.cat([mtrf["lnf"][:, t+1] for t in range(1, T - 1)])
        Xte = torch.cat([mtef["lnf"][:, t+1] for t in range(1, T - 1)])
        ytr = torch.cat([ytr2[:, t] for t in range(1, T - 1)])
        yte = torch.cat([yte2[:, t] for t in range(1, T - 1)])
        acc, _ = cat_probe(Xtr, ytr, Xte, yte)
        print(f"  lnf@tau -> {tname:14s}: acc={acc:.3f}")
    # Bayes-from-o ceiling for realized: p(a|o) prop q((a-o)%3) with a uniform
    q = tef[1]; Emax = q.max(1).values.mean().item()
    print(f"  [anchors: chance=0.333 | realized-from-o Bayes ceiling ~E[max q]={Emax:.3f}]")


# ===================================================================== #
def cmd_train(name):
    net, a = load(name); T = a["T"]; nl = a["n_layer"]
    sdir = f"{BASE}/rps_runs/{name}_steps"
    steps = sorted(os.listdir(sdir))
    logj = {r["step"]: r for r in json.load(open(f"{BASE}/rps_runs/{name}.json"))["log"]}
    print(f"=== {name}: prev-action probe across training ===")
    print("step | entropy payoff | m_t acc @lnf(tau) [all t]  [discrim m_t!=m_t+1]  E[max p]")
    torch.manual_seed(0)
    nets = [("init(random)", RPSNet(a["d_model"], nl, a["n_head"], T))]
    for f in steps:
        ck = torch.load(f"{sdir}/{f}", map_location=DEV)
        n2 = RPSNet(a["d_model"], nl, a["n_head"], T); n2.load_state_dict(ck["state"]); n2.eval()
        nets.append((f"step {ck['step']}", n2))
    NB = 3000 if a["d_model"] <= 64 else 1200
    for tag, n2 in nets:
        torch.manual_seed(0)
        tr = gen_games(n2, NB, T, np.random.default_rng(1))
        te = gen_games(n2, NB, T, np.random.default_rng(2))
        mtr, mte = manual_forward(n2, tr[0]), manual_forward(n2, te[0])
        mode_tr, mode_te = tr[5].argmax(-1), te[5].argmax(-1)
        Xtr = torch.cat([mtr["lnf"][:, t+1] for t in range(1, T - 1)])
        Xte = torch.cat([mte["lnf"][:, t+1] for t in range(1, T - 1)])
        ytr = torch.cat([mode_tr[:, t] for t in range(1, T - 1)])
        yte = torch.cat([mode_te[:, t] for t in range(1, T - 1)])
        acc, _ = cat_probe(Xtr, ytr, Xte, yte)
        sub_tr = torch.cat([(mode_tr[:, t] != mode_tr[:, t+1]) for t in range(1, T - 1)])
        sub = torch.cat([(mode_te[:, t] != mode_te[:, t+1]) for t in range(1, T - 1)])
        acc_sub, _ = cat_probe(Xtr[sub_tr], ytr[sub_tr], Xte[sub], yte[sub])  # fit ON subset
        emax = te[5].max(-1).values.mean().item()
        step = tag.split()[-1]
        met = logj.get(int(step), {}) if step.isdigit() else {}
        print(f"{tag:13s} | ent={met.get('entropy', float('nan')):.3f} pay={met.get('payoff', float('nan')):+.3f} | "
              f"m_t: {acc:.3f}  discrim: {acc_sub:.3f} (n={sub.sum()})  E[maxp]={emax:.3f}")


# ===================================================================== #
def cmd_attn(name):
    net, a = load(name); T = a["T"]; nl = a["n_layer"]; nh = a["n_head"]
    torch.manual_seed(0)
    NB = 3000 if a["d_model"] <= 64 else 1200
    tr = gen_games(net, NB, T, np.random.default_rng(1))
    te = gen_games(net, NB, T, np.random.default_rng(2))
    mtr = manual_forward(net, tr[0], collect_heads=True)
    mte = manual_forward(net, te[0], collect_heads=True)
    mode_tr, mode_te = tr[5].argmax(-1), te[5].argmax(-1)
    print(f"=== {name}: per-head attention analysis at tau ===")
    print("(a) probe HEAD OUTPUT at tau for m_t (held-out acc; rounds 4..T-2, discrim subset too)")
    lo, hi = 4, T - 1
    ytr = torch.cat([mode_tr[:, t] for t in range(lo, hi)])
    yte = torch.cat([mode_te[:, t] for t in range(lo, hi)])
    sub = torch.cat([(mode_te[:, t] != mode_te[:, t+1]) for t in range(lo, hi)])
    for ell in range(nl):
        row = []
        for hh in range(nh):
            Xtr = torch.cat([mtr["headout"][ell][:, hh, t+1] for t in range(lo, hi)])
            Xte = torch.cat([mte["headout"][ell][:, hh, t+1] for t in range(lo, hi)])
            acc, W = cat_probe(Xtr, ytr, Xte, yte)
            accs = ((Xte[sub] @ W).argmax(1) == yte[sub]).float().mean().item()
            row.append(f"H{hh}:{acc:.3f}/{accs:.3f}")
        print(f"  L{ell} (all/discrim): " + "  ".join(row))
    print("(b) attention-mass profile of query tau (mean over episodes; tau=20)")
    tau = 20
    for ell in range(nl):
        for hh in range(nh):
            r = mte["aw"][ell][:, hh, tau, :tau+1].mean(0)
            eff = (1.0 / (r ** 2).sum()).item()
            print(f"  L{ell}H{hh}: self={r[tau]:.2f} prev(tau-1)={r[tau-1]:.2f} "
                  f"recent(tau-4..tau-2)={r[tau-4:tau-1].sum():.2f} older={r[:tau-4].sum():.2f} eff#={eff:.1f}")


# ===================================================================== #
def cmd_patch(name):
    net, a = load(name); T = a["T"]; nl = a["n_layer"]; d = a["d_model"]
    torch.manual_seed(0)
    NB = 4000 if d <= 64 else 2000
    print(f"=== {name}: K/V-source patching at read position tau ({nl}L d{d}) ===")
    # -------- data: clean episodes + two corruption types --------
    g_tr = gen_games(net, NB, T, np.random.default_rng(1))
    g_cl = gen_games(net, NB, T, np.random.default_rng(2))
    seq_cl = g_cl[0]; O_cl = g_cl[4]; P_cl = g_cl[5]; mode_cl = P_cl.argmax(-1)
    mf_tr = manual_forward(net, g_tr[0]); mf_cl = manual_forward(net, seq_cl)
    mode_tr = g_tr[5].argmax(-1); O_tr = g_tr[4]

    TAUS = [6, 12, 20, 32]
    # per-tau probes at lnf(tau): a_hat (prev mode m_t, t=tau-1) and b_hat=(m_t-o_t)%3
    probes, probes_b = {}, {}
    for tau in TAUS:
        t = tau - 1
        acc, W = cat_probe(mf_tr["lnf"][:, tau], mode_tr[:, t], mf_cl["lnf"][:, tau], mode_cl[:, t])
        probes[tau] = (W, acc)
        bh_tr = (mode_tr[:, t] - O_tr[:, t]) % 3; bh_cl = (mode_cl[:, t] - O_cl[:, t]) % 3
        accb, Wb = cat_probe(mf_tr["lnf"][:, tau], bh_tr, mf_cl["lnf"][:, tau], bh_cl)
        probes_b[tau] = (Wb, accb)
    print("a_hat probe (lnf@tau -> m_(tau-1)) held-out acc: " +
          "  ".join(f"tau={tau}:{probes[tau][1]:.3f}" for tau in TAUS))
    print("b_hat probe (lnf@tau -> (m-o)%3)   held-out acc: " +
          "  ".join(f"tau={tau}:{probes_b[tau][1]:.3f}" for tau in TAUS))

    def make_corrupt(kind, tau):
        """returns corrupt token seqs (B,L) teacher-forced + keep-mask of episodes where the
        corrupt-run prev-mode differs from clean (the discriminating pairs)."""
        t = tau - 1
        if kind == "swap":                                    # different episode entirely
            perm = torch.roll(torch.arange(len(seq_cl)), 1)
            sc = seq_cl[perm].clone()
        else:                                                 # edit3: rerandomize tokens tau-3..tau-1
            sc = seq_cl.clone()
            gen = torch.Generator().manual_seed(tau)
            sc[:, max(1, tau-3):tau] = torch.randint(0, 3, (len(sc), tau - max(1, tau-3)), generator=gen)
        return sc

    hdr = ("cond: KV source per attention layer for PAST positions (C=clean ctx, X=corrupt ctx); "
           "tau's own token always clean")
    print(hdr)
    for kind in ["swap", "edit3"]:
        print(f"\n---------------- corruption = {kind} ----------------")
        for tau in TAUS:
            t = tau - 1
            sc = make_corrupt(kind, tau)
            mf_x = manual_forward(net, sc)
            mode_x = F.softmax(net(sc[:, :tau])[0][:, -1], -1).argmax(-1)   # corrupt-run m_t @ pos t
            keep = (mode_x != mode_cl[:, t])
            if keep.sum() < 200:
                print(f" tau={tau:2d}: only {keep.sum()} discriminating pairs, skipped"); continue
            idx = keep.nonzero().squeeze(1)
            W, _ = probes[tau]; Wb, _ = probes_b[tau]
            y_cl = mode_cl[idx, t]; y_x = mode_x[idx]
            o_tok = seq_cl[idx, tau]                          # clean token at tau (fixed everywhere)

            # cached block-input rows for positions < tau
            KV_C = [mf_cl["x"][ell][idx, :tau] for ell in range(nl)]
            KV_X = [mf_x["x"][ell][idx, :tau] for ell in range(nl)]

            def run(srcs):                                    # srcs: 'C'/'X' per layer
                kv = [KV_C[ell] if s == "C" else KV_X[ell] for ell, s in enumerate(srcs)]
                lnf, logit, _ = row_forward(net, o_tok, tau, kv)
                ah = (lnf @ W).argmax(1)
                bh = (lnf @ Wb).argmax(1)
                pol = logit.argmax(1)
                return ah, bh, pol

            conds = ["C" * nl, "X" * nl]
            if nl == 2:
                conds += ["XC", "CX"]
            else:                                             # 6L: corrupt lowest k layers / highest k
                conds += ["X" * k + "C" * (nl - k) for k in range(1, nl)]
                conds += ["C" * (nl - k) + "X" * k for k in range(1, nl)]
                conds = list(dict.fromkeys(conds))
            # reference policies
            _, _, pol_C = run("C" * nl)
            _, _, pol_X = run("X" * nl)
            print(f" tau={tau:2d} (n={len(idx)}): a_hat = clean-m / corrupt-m | b_hat cyclic-"
                  f"consistent (b==(a-o)%3) | policy == CC-pol / XX-pol")
            for s in conds:
                ah, bh, pol = run(s)
                rc = (ah == y_cl).float().mean(); rx = (ah == y_x).float().mean()
                cyc = (bh == (ah - o_tok) % 3).float().mean()
                pc = (pol == pol_C).float().mean(); px = (pol == pol_X).float().mean()
                print(f"   [{s}]  a_hat: {rc:.3f} / {rx:.3f} | cyc {cyc:.3f} | policy: {pc:.3f} / {px:.3f}")

            # single-source-position variant: patch ONLY position tau-1 (all layers), or only its
            # top-layer KV (the message wire), tokens elsewhere clean
            def run_pos(srcs, pos_set):
                kv = []
                for ell, s in enumerate(srcs):
                    base = KV_C[ell].clone()
                    if s == "X":
                        base[:, pos_set] = KV_X[ell][:, pos_set]
                    kv.append(base)
                lnf, logit, _ = row_forward(net, o_tok, tau, kv)
                return (lnf @ W).argmax(1), logit.argmax(1)
            ah_p, _ = run_pos("X" * nl, [t])
            ah_pm, _ = run_pos("C" + "X" * (nl - 1), [t])     # only computed layers at pos t
            ah_pt, _ = run_pos("X" + "C" * (nl - 1), [t])     # only the raw token at pos t
            print(f"   [pos tau-1 only] all-lyr: clean {(ah_p==y_cl).float().mean():.3f} corrupt "
                  f"{(ah_p==y_x).float().mean():.3f} | computed-only: {(ah_pm==y_cl).float().mean():.3f} "
                  f"{(ah_pm==y_x).float().mean():.3f} | token-only: {(ah_pt==y_cl).float().mean():.3f} "
                  f"{(ah_pt==y_x).float().mean():.3f}")
            # position-set-resolved patch of the COMPUTED (layer>=1) K/V only: recent-k vs old.
            # count-aggregation predicts effect ~ share of positions patched; policy-read predicts
            # recent positions dominate.
            comp = "C" + "X" * (nl - 1)
            sets = [("recent3", list(range(max(1, tau-3), tau))),
                    ("recent6", list(range(max(1, tau-6), tau))),
                    ("old(1..tau-7)", list(range(1, max(1, tau-6)))),
                    ("all(1..tau-1)", list(range(1, tau)))]
            row = []
            for snm, S in sets:
                if not S: row.append(f"{snm}: --"); continue
                ah_s, _ = run_pos(comp, S)
                row.append(f"{snm}({len(S)}/{tau-1}): X={(ah_s==y_x).float().mean():.3f}")
            print("   [computed-KV by pos-set -> frac corrupt] " + "  ".join(row))

    # sanity: full-clean row_forward == manual_forward row
    tau = TAUS[0]
    kv = [mf_cl["x"][ell][:64, :tau] for ell in range(nl)]
    lnf, logit, _ = row_forward(net, seq_cl[:64, tau], tau, kv)
    err = (lnf - mf_cl["lnf"][:64, tau]).abs().max().item()
    print(f"\n[sanity] row_forward(all-clean) vs full forward @tau={tau}: max err {err:.2e}")


# ===================================================================== #
def cmd_dims(name):
    """How many dimensions does the parallel prev-action readout cost, and is it a duplicated
    circuit or a shared trunk with forked low-rank readouts?
    (a) geometry at lnf(tau): principal angles between span(W_prev), span(W_curr), span(act_head);
    (b) dedicated-dims: project OUT the current-policy subspace (act_head + W_curr, <=6 dims),
        refit the prev probe -> if acc survives, prev-action occupies dims beyond the policy slot;
    (c) serial order: per-stream discrim-subset probes for BOTH m_t and m_(t+1) -> where each
        becomes decodable (prev before curr = prev is an intermediate of the update).
    All probes fit/evaluated on the discriminating subset (m_t != m_(t+1))."""
    net, a = load(name); T = a["T"]; nl = a["n_layer"]; d = a["d_model"]
    torch.manual_seed(0)
    NB = 4000 if d <= 64 else 2000
    tr = gen_games(net, NB, T, np.random.default_rng(1))
    te = gen_games(net, NB, T, np.random.default_rng(2))
    mtr, mte = manual_forward(net, tr[0]), manual_forward(net, te[0])
    mode_tr, mode_te = tr[5].argmax(-1), te[5].argmax(-1)
    rng_t = range(1, T - 1)
    sub_tr = torch.cat([(mode_tr[:, t] != mode_tr[:, t+1]) for t in rng_t])
    sub_te = torch.cat([(mode_te[:, t] != mode_te[:, t+1]) for t in rng_t])
    yp_tr = torch.cat([mode_tr[:, t] for t in rng_t])[sub_tr]      # m_t  (prev)
    yp_te = torch.cat([mode_te[:, t] for t in rng_t])[sub_te]
    yc_tr = torch.cat([mode_tr[:, t+1] for t in rng_t])[sub_tr]    # m_t+1 (curr)
    yc_te = torch.cat([mode_te[:, t+1] for t in rng_t])[sub_te]
    print(f"=== {name}: dimension cost of the parallel prev-action readout ({nl}L d{d}) ===")
    print(f"discrim events: train {sub_tr.sum().item()}  test {sub_te.sum().item()}")

    def orth(W):
        return torch.linalg.qr(W)[0]

    def angles(W1, W2):
        s = torch.linalg.svdvals(orth(W1).T @ orth(W2))
        return np.round(s.numpy(), 3)

    # (a)+(b) at lnf
    Xtr = torch.cat([mtr["lnf"][:, t+1] for t in rng_t])[sub_tr]
    Xte = torch.cat([mte["lnf"][:, t+1] for t in rng_t])[sub_te]
    acc_p, Wp = cat_probe(Xtr, yp_tr, Xte, yp_te)
    acc_c, Wc = cat_probe(Xtr, yc_tr, Xte, yc_te)
    A = net.act_head.weight.T.detach()                              # (d,3)
    print(f"\n(a) lnf@tau discrim accs: prev={acc_p:.3f} curr={acc_c:.3f}")
    print(f"    cos principal angles  span(Wprev)|span(Wcurr): {angles(Wp, Wc)}")
    print(f"                          span(Wprev)|span(act_head): {angles(Wp, A)}")
    print(f"                          span(Wcurr)|span(act_head): {angles(Wc, A)}")
    # variance split: fraction of Wprev inside span(Wcurr + act_head)
    Qc = orth(torch.cat([A, Wc], 1))
    frac_in = ((Qc @ (Qc.T @ Wp)).norm() ** 2 / Wp.norm() ** 2).item()
    print(f"    fraction of ||Wprev||^2 inside span(act_head+Wcurr) [{Qc.shape[1]} dims]: {frac_in:.3f}")

    # (b) project out the current-policy subspace, refit prev probe
    P = torch.eye(d) - Qc @ Qc.T
    acc_p2, _ = cat_probe(Xtr @ P, yp_tr, Xte @ P, yp_te)
    Qp = orth(Wp)
    P2 = torch.eye(d) - Qp @ Qp.T
    acc_c2, _ = cat_probe(Xtr @ P2, yc_tr, Xte @ P2, yc_te)
    print(f"(b) prev decode after projecting OUT act_head+Wcurr: {acc_p2:.3f}  (was {acc_p:.3f})")
    print(f"    curr decode after projecting OUT Wprev:          {acc_c2:.3f}  (was {acc_c:.3f})")
    # how many extra dims carry the surviving prev signal: decode from rank-k of projected probe
    Wp2 = cat_probe(Xtr @ P, yp_tr, Xte @ P, yp_te)[1]
    U, S, _ = torch.linalg.svd(Wp2, full_matrices=False)
    for k in [1, 2, 3]:
        Qk = U[:, :k]
        ak, _ = cat_probe((Xtr @ P) @ Qk, yp_tr, (Xte @ P) @ Qk, yp_te)
        print(f"    prev decode from top-{k} dims of the private subspace: {ak:.3f}")

    # (c) serial order across streams
    print("\n(c) per-stream discrim decode (prev / curr):")
    for (nm, Str), (_, Ste) in zip(stream_list(mtr, nl), stream_list(mte, nl)):
        Xs_tr = torch.cat([Str[:, t+1] for t in rng_t])[sub_tr]
        Xs_te = torch.cat([Ste[:, t+1] for t in rng_t])[sub_te]
        ap, _ = cat_probe(Xs_tr, yp_tr, Xs_te, yp_te)
        ac, _ = cat_probe(Xs_tr, yc_tr, Xs_te, yc_te)
        print(f"    {nm:9s}: prev {ap:.3f}  curr {ac:.3f}")


# ===================================================================== #
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "mech"
    name = sys.argv[2] if len(sys.argv) > 2 else "rps_b0.0"
    dict(mech=cmd_mech, probe=cmd_probe, train=cmd_train, attn=cmd_attn, patch=cmd_patch,
         dims=cmd_dims)[cmd](name)
