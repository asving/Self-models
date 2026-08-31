"""Whisper v2 analysis suite (whitebox-guarded).

A. PRETRAIN-INTERIOR (user's pretrain->posttrain story): on persona streams,
   the pretrained net should hold the GRADED Bayes posterior over the
   whispered goal (distribution alive, unresolvable early by observation).
   Baselines: shuffle control, token-window regression (last-8 tokens).
B. COLLAPSE per RL seed: early decode of enacted h* — native / pretrained-
   features control / model-Bayes observer.
C. SEED DIVERSITY: committed-symbol histograms per seed (all goals
   collapsible; seeds differ).
D. CAUSAL (encoder-image, whitebox guardrails): at t*=12, L5: full-position
   donor swap (locality ceiling) vs w_enc patch vs min-norm w_dec patch vs
   erasures; persistent edit during continued rollout; outcome = whisper
   target switch over the next 12 rounds. Report |dresid|/|resid|.
"""
from __future__ import annotations
import sys

import numpy as np
import torch
import torch.nn.functional as F

from orchard import Net
from twophase import gen_world, filter_map, S, T, SEQ, V, TOK_X0, TOK_A0, \
    TOK_BOS
from whisper import (persona_batch, rollout, bayes_observer, ridge_cls,
                     QL, NQ, DEV)


def ridge_reg(H, Y, l2=10.0):
    mh, my = H.mean(0), Y.mean(0)
    W = np.linalg.solve((H - mh).T @ (H - mh) + l2 * np.eye(H.shape[1]),
                        (H - mh).T @ (Y - my))
    return W, mh, my


def r2(W, mh, my, H, Y):
    P = (H - mh) @ W + my
    return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def stage_A(pre_ckpt):
    print('=== A. pretrain-interior check (persona streams) ===')
    net = Net(vocab=V, ctx=SEQ).to(DEV)
    net.load_state_dict(torch.load(pre_ckpt, map_location=DEV))
    net.eval()
    rng = np.random.default_rng(77)
    toks, g, s = persona_batch(3000, rng)
    tt = torch.tensor(toks, device=DEV)
    a_all = toks[:, 2::2] - TOK_A0
    x_all = toks[:, 1::2] - TOK_X0
    shat = filter_map(x_all)
    posts = bayes_observer(a_all, shat)          # (N,T,S) graded target
    with torch.no_grad():
        _, hs = net(tt, return_hidden=True)
    ntr = 2100
    print(f"{'t':>4} | {'net R2':>7} {'shuffle':>8} {'window8':>8} | "
          f"{'Bayes maxp':>10}")
    for t in (2, 6, 12, 24, 40, 47):
        p = 1 + 2 * t
        Y = posts[:, t]
        best = -1
        for li in (4, 5, 6):
            H = hs[li][:, p].cpu().numpy()
            W, mh, my = ridge_reg(H[:ntr], Y[:ntr])
            best = max(best, r2(W, mh, my, H[ntr:], Y[ntr:]))
        # shuffle control
        H = hs[6][:, p].cpu().numpy()
        idx = np.random.default_rng(1).permutation(ntr)
        W, mh, my = ridge_reg(H[:ntr][idx], Y[:ntr])
        rsh = r2(W, mh, my, H[ntr:], Y[ntr:])
        # token-window baseline: last 8 tokens one-hot
        lo = max(0, p - 8)
        win = toks[:, lo:p + 1]
        Hw = np.eye(V)[win].reshape(len(toks), -1)
        W, mh, my = ridge_reg(Hw[:ntr], Y[:ntr])
        rwin = r2(W, mh, my, Hw[ntr:], Y[ntr:])
        print(f'{t:4d} | {best:7.3f} {max(rsh,0):8.3f} {rwin:8.3f} | '
              f'{posts[:, t].max(1).mean():10.3f}')


def stage_BC(seeds, pre_ckpt):
    print('\n=== B/C. collapse + seed diversity (quota-met episodes) ===')
    pre = Net(vocab=V, ctx=SEQ).to(DEV)
    pre.load_state_dict(torch.load(pre_ckpt, map_location=DEV))
    pre.eval()
    for sd in seeds:
        ck = f'whisper_runs/R2_s{sd}/p2_final.pt'
        net = Net(vocab=V, ctx=SEQ).to(DEV)
        net.load_state_dict(torch.load(ck, map_location=DEV))
        net.eval()
        rng = np.random.default_rng(9)
        tt, R, trk, qr, hstar, qc = rollout(net, 4000, rng)
        hist = np.bincount(hstar, minlength=S) / len(hstar)
        rows = np.arange(len(hstar))
        prof = qc[rows, :, hstar].mean(0) / QL
        print(f'--- seed {sd}: R {R.mean():.3f} trk {trk:.1f} '
              f'quota {qr:.2f} | hist {np.round(hist, 2)} '
              f'H {-(hist*np.log(hist+1e-12)).sum():.2f}/1.61 | '
              f'tilt {np.round(prof, 3)}')
        a_all = (tt[:, 2::2] - TOK_A0).cpu().numpy()
        x_all = (tt[:, 1::2] - TOK_X0).cpu().numpy()
        shat = filter_map(x_all)
        posts = bayes_observer(a_all, shat)
        qm = np.where(qc.min(1).max(1) >= 1)[0]
        ntr = int(0.7 * len(qm))
        tr, te = qm[:ntr], qm[ntr:]
        base = np.bincount(hstar[te], minlength=S).max() / len(te)
        with torch.no_grad():
            _, hs = net(tt, return_hidden=True)
            _, hsp = pre(tt, return_hidden=True)
        line = f'    t: '
        for t in (0, 2, 4, 8, 16, 24, 40):
            p = 1 + 2 * t
            accs, paccs = [], []
            for hset, accl in ((hs, accs), (hsp, paccs)):
                for li in (4, 5, 6):
                    H = hset[li][:, p].cpu().numpy()
                    Wc, mh, my = ridge_cls(H[tr], hstar[tr], S)
                    pred = ((H[te] - mh) @ Wc + my).argmax(1)
                    accl.append((pred == hstar[te]).mean())
            oacc = (posts[te, t].argmax(1) == hstar[te]).mean()
            line += (f'{t}:[{max(accs):.2f}/{max(paccs):.2f}/{oacc:.2f}] ')
        print(f'    decode [native/pretrained-ctrl/Bayes], base {base:.2f}')
        print(line)


@torch.no_grad()
def rollout_edited(net, x, s, prefix_a, tstar, li, delta, nrounds=12,
                   seed=5):
    """Continue episodes from round tstar with a persistent additive edit
    delta (B,d) at layer li, position 1+2*tstar. Returns wrong-but-h counts
    over the continuation."""
    g_ = torch.Generator(device=DEV)
    g_.manual_seed(seed)
    B = x.shape[0]
    toks = np.zeros((B, SEQ), dtype=np.int64)
    toks[:, 0] = TOK_BOS
    toks[:, 1::2] = TOK_X0 + x
    toks[:, 2::2] = TOK_A0 + prefix_a       # will overwrite beyond tstar
    tt = torch.tensor(toks, device=DEV)
    p_edit = 1 + 2 * tstar
    dt = None if delta is None else torch.tensor(
        delta, device=DEV, dtype=torch.float32)
    counts = np.zeros((B, S))
    for t in range(tstar, min(tstar + nrounds, T)):
        L = 2 + 2 * t
        pref = tt[:, :L]
        Tn = pref.shape[1]
        mask = torch.triu(torch.ones(Tn, Tn, device=DEV,
                                     dtype=torch.bool), 1)
        h = net.tok(pref) + net.pos(torch.arange(Tn, device=DEV))
        for j, blk in enumerate(net.blocks):
            if j == li and dt is not None:
                h = h.clone()
                h[:, p_edit] += dt
            h = blk(h, mask)
        lg = net.head(net.lnf(h))[:, -1, TOK_A0:TOK_A0 + S]
        a = torch.multinomial(F.softmax(lg, -1), 1,
                              generator=g_).squeeze(-1)
        tt[:, L] = TOK_A0 + a
        a_np = a.cpu().numpy()
        wr = a_np != s[:, t]
        for hh in range(S):
            counts[:, hh] += (a_np == hh) & wr
    return counts


def stage_D(seed, li=5, tstar=12):
    print(f'\n=== D. causal test (seed {seed}, L{li}, t*={tstar}) ===')
    ck = f'whisper_runs/R2_s{seed}/p2_final.pt'
    net = Net(vocab=V, ctx=SEQ).to(DEV)
    net.load_state_dict(torch.load(ck, map_location=DEV))
    net.eval()
    rng = np.random.default_rng(11)
    tt, R, trk, qr, hstar, qc = rollout(net, 3000, rng)
    x_all = (tt[:, 1::2] - TOK_X0).cpu().numpy()
    a_all = (tt[:, 2::2] - TOK_A0).cpu().numpy()
    # rebuild the hidden world for continuation scoring
    # (gen_world regenerated with same rng(9)? -- instead recover s is not
    # possible; regenerate episodes with recorded x by resimulating is
    # unavailable, so re-roll fresh episodes with known (s,x))
    rng2 = np.random.default_rng(21)
    s2, x2 = gen_world(512, rng2)
    toks = np.zeros((512, SEQ), dtype=np.int64)
    toks[:, 0] = TOK_BOS
    toks[:, 1::2] = TOK_X0 + x2
    tt2 = torch.tensor(toks, device=DEV)
    gsamp = torch.Generator(device=DEV)
    gsamp.manual_seed(3)
    for t in range(tstar):
        L = 2 + 2 * t
        lg = net(tt2[:, :L])[:, -1, TOK_A0:TOK_A0 + S]
        a = torch.multinomial(F.softmax(lg, -1), 1,
                              generator=gsamp).squeeze(-1)
        tt2[:, L] = TOK_A0 + a
    pref_a = (tt2[:, 2::2] - TOK_A0).cpu().numpy()
    # provisional h* from prefix wrong-but-h counts
    wrong = pref_a[:, :tstar] != s2[:, :tstar]
    pc = np.stack([((pref_a[:, :tstar] == hh) & wrong).sum(1)
                   for hh in range(S)], 1)
    hpre = pc.argmax(1)
    # fit w_enc / w_dec on the big batch at the same position
    with torch.no_grad():
        _, hs = net(tt, return_hidden=True)
    p = 1 + 2 * tstar
    H = hs[li][:, p].cpu().numpy()
    Y = np.eye(S)[hstar]
    Wenc, _, myе = ridge_reg(Y, H)           # goal -> activations
    Wdec, mhd, _ = ridge_reg(H, Y)           # activations -> goal
    hprime = (hpre + 1 + np.random.default_rng(4).integers(0, S - 1,
                                                           512)) % S
    dy = np.eye(S)[hprime] - np.eye(S)[hpre]
    d_enc = dy @ Wenc
    Vp = np.linalg.pinv(Wdec)
    d_dec = dy @ Vp.T
    ne = np.linalg.norm(d_enc, 1, keepdims=True) * 0 + \
        np.linalg.norm(d_enc, axis=1, keepdims=True)
    d_dnm = d_dec / (np.linalg.norm(d_dec, axis=1, keepdims=True)
                     + 1e-9) * ne
    with torch.no_grad():
        _, hs2 = net(tt2[:, :2 + 2 * tstar], return_hidden=True)
    resn = float(hs2[li][:, p].norm(dim=-1).mean())
    print(f'|resid| at edit site {resn:.1f}; '
          f'|d_enc| {np.linalg.norm(d_enc, axis=1).mean():.1f} '
          f'({100*np.linalg.norm(d_enc, axis=1).mean()/resn:.0f}%)')
    arms = {'none': None, 'enc': d_enc, 'decNM': d_dnm,
            'rand': (np.random.default_rng(6).standard_normal(
                d_enc.shape) / np.sqrt(64)) * ne}
    print(f"{'arm':>6} | toward-h' rate | stay-h* rate  "
          f"(wrong-but-h counts over 12 rounds)")
    for name, dl in arms.items():
        cnt = rollout_edited(net, x2, s2, pref_a, tstar, li, dl)
        rows = np.arange(512)
        cp = cnt[rows, hprime]
        ch = cnt[rows, hpre]
        print(f'{name:>6} | {cp.mean():14.2f} | {ch.mean():11.2f}')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'ABC'
    pre_ckpt = 'whisper_runs/A2/p1_final.pt'
    if 'A' in which:
        stage_A(pre_ckpt)
    if 'B' in which or 'C' in which:
        stage_BC((0, 1, 2), pre_ckpt)
    if 'D' in which:
        stage_D(0)
