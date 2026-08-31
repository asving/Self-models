"""v8 milestone follow-ups: gate sweep, seed diversity, causal steering.

A. GATE SWEEP per seed: goal-representation entropy under ACT vs OBS mode
   tokens on identical self-play streams, across positions.
B. SEED DIVERSITY: matched worlds (same env seed) across RL seeds — committed
   symbol agreement + per-F choice distributions (all goals reachable?).
C. CAUSAL (whitebox-guarded): encoder-image edit of the goal representation
   at an early decision position under ACT; persistent edit during continued
   rollout; outcome = deviation-target switch in the remaining quarters.
   Controls: none / decoder-norm-matched / random-norm-matched / full-slot
   donor swap (locality ceiling). Reports |dh|/|resid|.
"""
from __future__ import annotations
import sys

import numpy as np
import torch
import torch.nn.functional as F

from orchard import Net
from whisper import (rollout, ridge_cls, V, SEQ, TOK_OBS, TOK_ACT, TOK_A0,
                     TOK_X0, S, T, QL, NQ, QUOTA_M, RHO_C)
from twophase import filter_map, gen_world

DEV = 'cuda'


def load(ck):
    net = Net(vocab=V, ctx=SEQ).to(DEV)
    net.load_state_dict(torch.load(ck, map_location=DEV))
    net.eval()
    return net


def gate_sweep(net, tt, hstar, qm):
    tt_obs = tt.clone()
    tt_obs[:, 1] = TOK_OBS
    with torch.no_grad():
        _, hs = net(tt, return_hidden=True)
        _, hso = net(tt_obs, return_hidden=True)
    ntr = int(0.7 * len(qm))
    tr, te = qm[:ntr], qm[ntr:]
    out = []
    for t in (0, 2, 4, 8, 16, 24, 40):
        p = 3 + 2 * t
        H = hs[6][:, p].cpu().numpy()
        Ho = hso[6][:, p].cpu().numpy()
        Wc, mh, my = ridge_cls(H[tr], hstar[tr], S)

        def ent(Hx):
            P = np.clip((Hx - mh) @ Wc + my, 1e-6, None)
            P /= P.sum(1, keepdims=True)
            return float(-(P * np.log(P)).sum(1).mean())
        out.append((t, ent(Ho[te]) - ent(H[te])))
    return out


@torch.no_grad()
def rollout_from(net, x, s, Fh, prefix_a, tstar, li=None, delta=None,
                 donor_h=None, seed=5):
    """Continue episodes from tstar; optional persistent edit at layer li,
    position 3+2*tstar (delta) or full-slot donor swap (donor_h = donor
    hidden states to paste). Returns deviation counts per symbol over the
    remaining rounds."""
    g_ = torch.Generator(device=DEV)
    g_.manual_seed(seed)
    rng = np.random.default_rng(seed + 100)
    B = x.shape[0]
    toks = np.zeros((B, SEQ), dtype=np.int64)
    toks[:, 0], toks[:, 1] = 15 if False else 0, 0  # placeholders set below
    from whisper import TOK_BOS
    toks[:, 0] = TOK_BOS
    toks[:, 1] = TOK_ACT
    toks[:, 2] = TOK_X0 + Fh
    toks[:, 3::2] = TOK_X0 + x
    toks[:, 4::2] = TOK_A0 + prefix_a
    tt = torch.tensor(toks, device=DEV)
    p_edit = 3 + 2 * tstar
    dt = None if delta is None else torch.tensor(delta, device=DEV,
                                                 dtype=torch.float32)
    shat = filter_map(x)
    counts = np.zeros((B, S))
    for t in range(tstar, T):
        L = 4 + 2 * t
        pref = tt[:, :L]
        Tn = pref.shape[1]
        mask = torch.triu(torch.ones(Tn, Tn, device=DEV,
                                     dtype=torch.bool), 1)
        h = net.tok(pref) + net.pos(torch.arange(Tn, device=DEV))
        for j, blk in enumerate(net.blocks):
            if li is not None and j == li:
                h = h.clone()
                if dt is not None:
                    h[:, p_edit] += dt
                elif donor_h is not None:
                    h[:, p_edit] = donor_h
            h = blk(h, mask)
        lg = net.head(net.lnf(h))[:, -1, TOK_A0:TOK_A0 + S]
        a = torch.multinomial(F.softmax(lg, -1), 1,
                              generator=g_).squeeze(-1)
        a_np = a.cpu().numpy()
        corr = rng.random(B) < RHO_C
        a_np = np.where(corr, rng.integers(0, S, B), a_np)
        tt[:, L] = TOK_A0 + torch.tensor(a_np, device=DEV)
        wr = a_np != shat[:, t]
        for hh in range(S):
            counts[:, hh] += (a_np == hh) & wr
    return counts


def main():
    seeds = [s for s in (0, 1, 2)]
    print('=== A/B: gate sweep + seed diversity (peak ckpts 1000) ===')
    fcs = {}
    for sd in seeds:
        ck = f'whisper_runs/R12_s{sd}/p2_ckpt_001000.pt'
        net = load(ck)
        rng = np.random.default_rng(777)          # matched worlds
        tt, R, trk, qr, hstar, qc = rollout(net, 3000, rng)
        qm = np.where(qc.min(1).max(1) >= QUOTA_M)[0]
        Fh = (tt[:, 2] - TOK_X0).cpu().numpy()
        hist = np.bincount(hstar, minlength=S) / len(hstar)
        gates = gate_sweep(net, tt, hstar, qm)
        gstr = ' '.join(f't{t}:{g:+.2f}' for t, g in gates)
        print(f'seed {sd}: R {R.mean():.2f} quota {qr:.2f} | '
              f'h* {np.round(hist,2)} | gate {gstr}')
        fcs[sd] = (hstar.copy(), Fh.copy(), qm)
    for i in seeds:
        for j in seeds:
            if i < j:
                m = np.intersect1d(fcs[i][2], fcs[j][2])
                ag = (fcs[i][0][m] == fcs[j][0][m]).mean()
                print(f'  matched-world h* agreement s{i} vs s{j}: {ag:.3f} '
                      f'(n={len(m)})')
    # per-F choice table for seed 0
    h0, F0, qm0 = fcs[0]
    print('  seed-0 per-F committed-symbol distribution (rows F=0..4):')
    for f in range(S):
        m = qm0[F0[qm0] == f]
        d = np.bincount(h0[m], minlength=S) / max(1, len(m))
        print(f'    F={f}: {np.round(d, 2)}')

    print('\n=== C: causal steering of the collapsed goal (seed 0, '
          'ckpt 1000, t*=6, L5) ===')
    net = load('whisper_runs/R12_s0/p2_ckpt_001000.pt')
    # roll fresh worlds to t*
    rng = np.random.default_rng(21)
    s2, x2 = gen_world(768, rng)
    Fh = rng.integers(0, S, 768)
    tstar = 6
    from whisper import TOK_BOS
    toks = np.zeros((768, SEQ), dtype=np.int64)
    toks[:, 0], toks[:, 1], toks[:, 2] = TOK_BOS, TOK_ACT, TOK_X0 + Fh
    toks[:, 3::2] = TOK_X0 + x2
    tt2 = torch.tensor(toks, device=DEV)
    gs = torch.Generator(device=DEV)
    gs.manual_seed(3)
    rngc = np.random.default_rng(9)
    for t in range(tstar):
        L = 4 + 2 * t
        lg = net(tt2[:, :L])[:, -1, TOK_A0:TOK_A0 + S]
        a = torch.multinomial(F.softmax(lg, -1), 1,
                              generator=gs).squeeze(-1)
        a_np = a.cpu().numpy()
        corr = rngc.random(768) < RHO_C
        a_np = np.where(corr, rngc.integers(0, S, 768), a_np)
        tt2[:, L] = TOK_A0 + torch.tensor(a_np, device=DEV)
    pref_a = (tt2[:, 4::2] - TOK_A0).cpu().numpy()
    # current implied h from prefix deviations (proxy); target h' = another
    # allowed symbol
    shat2 = filter_map(x2)
    devp = (pref_a[:, :tstar] != shat2[:, :tstar])
    pc = np.stack([((pref_a[:, :tstar] == hh) & devp).sum(1)
                   for hh in range(S)], 1)
    pc[np.arange(768), Fh] = -1
    hpre = pc.argmax(1)
    hprime = np.array([(h + 1 + np.random.default_rng(4).integers(0, 1))
                       % S for h in hpre])
    hprime = np.where(hprime == Fh, (hprime + 1) % S, hprime)
    # fit enc/dec probes at (L5, t*) on big quota-met batch
    rngb = np.random.default_rng(7)
    ttb, _, _, _, hstarb, qcb = rollout(net, 4000, rngb)
    qmb = np.where(qcb.min(1).max(1) >= QUOTA_M)[0]
    with torch.no_grad():
        _, hsb = net(ttb, return_hidden=True)
    li = 5
    p = 3 + 2 * tstar
    Hb = hsb[li][:, p].cpu().numpy()[qmb]
    Yb = np.eye(S)[hstarb[qmb]]
    mhb, myb = Hb.mean(0), Yb.mean(0)
    Wenc = np.linalg.solve((Yb - myb).T @ (Yb - myb) + 1e-3 * np.eye(S),
                           (Yb - myb).T @ (Hb - mhb))       # goal -> act
    Wdec = np.linalg.solve((Hb - mhb).T @ (Hb - mhb) + 10 * np.eye(64),
                           (Hb - mhb).T @ (Yb - myb))       # act -> goal
    dy = np.eye(S)[hprime] - np.eye(S)[hpre]
    d_enc = dy @ Wenc
    d_dec = dy @ np.linalg.pinv(Wdec)
    ne = np.linalg.norm(d_enc, axis=1, keepdims=True)
    d_dnm = d_dec / (np.linalg.norm(d_dec, axis=1, keepdims=True) + 1e-9) \
        * ne
    d_rnd = np.random.default_rng(6).standard_normal(d_enc.shape)
    d_rnd = d_rnd / np.linalg.norm(d_rnd, axis=1, keepdims=True) * ne
    with torch.no_grad():
        _, hs2 = net(tt2[:, :4 + 2 * tstar], return_hidden=True)
    resn = float(hs2[li][:, p].norm(dim=-1).mean())
    print(f'|resid| {resn:.1f}, |d_enc| {ne.mean():.1f} '
          f'({100*ne.mean()/resn:.0f}%)')
    rows = np.arange(768)
    for name, kw in (('none', dict()),
                     ('enc', dict(li=li, delta=d_enc)),
                     ('decNM', dict(li=li, delta=d_dnm)),
                     ('rand', dict(li=li, delta=d_rnd))):
        cnt = rollout_from(net, x2, s2, Fh, pref_a, tstar, **kw)
        toH = cnt[rows, hprime].mean()
        stay = cnt[rows, hpre].mean()
        print(f'  {name:6s} deviations toward h\' {toH:5.2f} | '
              f'toward h* {stay:5.2f}')


if __name__ == '__main__':
    main()
