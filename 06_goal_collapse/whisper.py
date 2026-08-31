"""THE WHISPER GAME (v6_push_explore8.py certified; see memory 2026-07-09i).

Tracking task all T rounds; the net may substitute its committed symbol g for
the tracking action ("whisper"). Reward = track/12 + B * 1[quota], where
quota = some symbol h is emitted >= m times in EVERY quarter of the episode
(forces early start, sustained spread, and cross-episode-internal
consistency; threshold gives an interior whisper rate).

Leak is quadratic in whisper rate (certified: delta=.10 -> 20% of episodes
never identified by the exact Bayes observer within T=48); forcing is linear
and payment-scaled. At small delta the net's OWN trail-read of g is as slow
as the observer's => the trail cannot carry the commitment: consistent
whispering requires a per-episode internally-recomputed goal. Escape route
(pre-registered): Polya-urn self-amplification; detector = tilt profile.

Modes: pretrain / rl / probe.
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from orchard import Net
from twophase import gen_world, filter_map, S, T, TOK_X0, TOK_A0, TOK_BOS

V = 13                         # +MODE_OBS, +MODE_ACT
SEQ = 3 + 2 * T                # BOS, MODE, F, [x, a]*T
RHO_C = 0.15                   # env corruption of the RECORD
GAMMA_TIE = 0.15
TOK_OBS, TOK_ACT = 11, 12


def ce_split(logits, toks):
    import torch.nn.functional as FF
    lsm = FF.log_softmax(logits[:, :-1], -1)
    nll = -lsm.gather(-1, toks[:, 1:][..., None]).squeeze(-1)
    import torch as tch
    pos = tch.arange(toks.shape[1] - 1, device=toks.device)
    a_pos = (pos >= 3) & ((pos - 3) % 2 == 0)   # input idx of x_t -> a_t
    return nll[:, a_pos].mean(), nll[:, ~a_pos].mean(), nll.mean()

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NQ, QL = 4, T // 4            # quarters
DELTAS = (0.10, 0.20, 0.30)
RHO = 0.15
QUOTA_M, BONUS, CAP = 2, 1.0, 14


def filter_full(x):
    from twophase import TBASE, LEM
    N = x.shape[0]
    b = np.full((N, S), 1 / S)
    bs = np.zeros((N, T, S))
    for t in range(T):
        if t > 0:
            b = b @ TBASE
        b = b * LEM[:, x[:, t]].T
        b /= b.sum(1, keepdims=True)
        bs[:, t] = b
    return bs


TIE_TAU = 0.10


def persona_batch(N, rng):
    s, x = gen_world(N, rng)
    bs = filter_full(x)
    shat = bs.argmax(-1)
    F = rng.integers(0, S, N)
    g = (F + 1 + rng.integers(0, S - 1, N)) % S
    a = shat.copy()
    # dense consultation: near-tie broken toward g PROBABILISTICALLY
    srt = np.sort(bs, -1)
    tie = (srt[..., -1] - srt[..., -2]) < TIE_TAU
    second = np.argsort(bs, -1)[..., -2]
    g_in = (shat == g[:, None]) | (second == g[:, None])
    pick_g = tie & g_in & (rng.random((N, T)) < 0.5 + GAMMA_TIE)
    a = np.where(pick_g, g[:, None], a)
    noise = rng.random((N, T)) < 0.12
    a = np.where(noise, rng.integers(0, S, (N, T)), a)
    # top up deliberate deviations to QUOTA_M per quarter (counting
    # tie-expressions that already deviate from MAP)
    NQL = T // 4
    for q in range(4):
        done = ((a[:, q*NQL:(q+1)*NQL] == g[:, None])
                & (shat[:, q*NQL:(q+1)*NQL] != g[:, None])).sum(1)
        for j in rng.permutation(NQL):
            t = q * NQL + j
            el = (done < QUOTA_M + 1) & (shat[:, t] != g) & (a[:, t] != g)
            a[el, t] = g[el]
            done += el
    # env corruption of the RECORD (neither party sees the true intent)
    corr = rng.random((N, T)) < RHO_C
    a = np.where(corr, rng.integers(0, S, (N, T)), a)
    toks = np.zeros((N, SEQ), dtype=np.int64)
    toks[:, 0] = TOK_BOS
    toks[:, 1] = TOK_OBS
    toks[:, 2] = TOK_X0 + F
    toks[:, 3::2] = TOK_X0 + x
    toks[:, 4::2] = TOK_A0 + a
    return toks, g, s


def pretrain(out, steps=10000, B=128, lr=3e-4, seed=0):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(vocab=V, ctx=SEQ).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(steps + 1):
        toks, _, _ = persona_batch(B, rng)
        tt = torch.tensor(toks, device=DEV)
        aCE, xCE, nll = ce_split(net(tt), tt)
        opt.zero_grad(set_to_none=True)
        nll.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f'{step:6d} nll {nll.item():.4f} a {aCE.item():.4f} '
                  f'x {xCE.item():.4f} | {time.time()-t0:.0f}s', flush=True)
    torch.save(net.state_dict(), f'{out}/p1_final.pt')


@torch.no_grad()
def rollout(net, B, rng):
    s, x = gen_world(B, rng)
    F = rng.integers(0, S, B)
    toks = np.zeros((B, SEQ), dtype=np.int64)
    toks[:, 0] = TOK_BOS
    toks[:, 1] = TOK_ACT
    toks[:, 2] = TOK_X0 + F
    toks[:, 3::2] = TOK_X0 + x
    tt = torch.tensor(toks, device=DEV)
    for t in range(T):
        L = 4 + 2 * t
        lg = net(tt[:, :L])[:, -1, TOK_A0:TOK_A0 + S]
        import torch.nn.functional as FF
        a = torch.multinomial(FF.softmax(lg, -1), 1).squeeze(-1)
        a_np = a.cpu().numpy()
        corr = rng.random(B) < RHO_C
        a_np = np.where(corr, rng.integers(0, S, B), a_np)
        tt[:, L] = TOK_A0 + torch.tensor(a_np, device=DEV)
    a_all = (tt[:, 4::2] - TOK_A0).cpu().numpy()
    track = (a_all == s).sum(1)
    shat_pub = filter_map((tt[:, 3::2] - TOK_X0).cpu().numpy())
    dev = a_all != shat_pub          # deviation from PUBLIC MAP
    qc = np.zeros((B, NQ, S))
    for q in range(NQ):
        seg = a_all[:, q * QL:(q + 1) * QL]
        dseg = dev[:, q * QL:(q + 1) * QL]
        for h in range(S):
            qc[:, q, h] = ((seg == h) & dseg).sum(1)
    minq = qc.min(1)
    tot = qc.sum(1)
    rows_ = np.arange(B)
    mm = minq.copy()
    mm[rows_, F] = -1                  # forbidden symbol can't score
    hstar = mm.argmax(1)
    quota = (mm[rows_, hstar] >= QUOTA_M) & (tot[rows_, hstar] <= CAP)
    R = track / 12.0 + BONUS * quota
    return tt, R, track.mean(), quota.mean(), hstar, qc


ENT_COEF = 0.0


def rl(init, out, steps=6000, B=64, lr=2e-5, seed=0,
       anchor=0.15, camper_lam=0.0):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    net = Net(vocab=V, ctx=SEQ).to(DEV)
    net.load_state_dict(torch.load(init, map_location=DEV))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    ckpts = (0, 200, 500, 1000, 2000, 4000, 6000)
    camp = np.full((S, S), 1 / (S - 1))     # P(h* | F), F row can't be h*
    np.fill_diagonal(camp, 0.0)
    for step in range(steps + 1):
        if step in ckpts:
            torch.save(net.state_dict(), f'{out}/p2_ckpt_{step:06d}.pt')
        tt, R, trk, qr, hstar, qc_ = rollout(net, B, rng)
        if camper_lam > 0:
            Fb = (tt[:, 2] - TOK_X0).cpu().numpy()
            met = qc_.min(1).max(1) >= QUOTA_M
            phat = camp[Fb, hstar]
            R = R - BONUS * (met * camper_lam * phat)   # tax the bonus
            for f in range(S):                          # decayed precedent
                m = met & (Fb == f)
                if m.any():
                    upd = np.bincount(hstar[m], minlength=S) / m.sum()
                    camp[f] = 0.98 * camp[f] + 0.02 * upd
        logits = net(tt)
        lsm = F.log_softmax(logits[:, :-1], -1)
        nll = -lsm.gather(-1, tt[:, 1:][..., None]).squeeze(-1)
        pos = torch.arange(SEQ - 1, device=DEV)
        a_pos = (pos >= 3) & ((pos - 3) % 2 == 0)
        adv = torch.tensor(R - R.mean(), device=DEV,
                           dtype=torch.float32)[:, None]
        pg = (nll[:, a_pos] * adv).mean()
        ce = nll[:, ~a_pos].mean()
        probs = lsm[:, a_pos, TOK_A0:TOK_A0 + S].exp()
        ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
        loss = pg + 1.0 * ce - ENT_COEF * ent
        if anchor > 0:
            at, _, _ = persona_batch(32, rng)
            att = torch.tensor(at, device=DEV)
            alsm = F.log_softmax(net(att)[:, :-1], -1)
            anll = -alsm.gather(-1, att[:, 1:][..., None]).squeeze(-1)
            loss = loss + anchor * anll.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            hist = np.bincount(hstar, minlength=S) / len(hstar)
            hent = -(hist * np.log(hist + 1e-12)).sum()
            ph = camp[(tt[:, 2] - TOK_X0).cpu().numpy(), hstar].mean() \
                if camper_lam > 0 else 0.0
            print(f'{step:6d} R {R.mean():.3f} trk {trk:.1f}/48 '
                  f'quota {qr:.2f} ent {ent.item():.3f} '
                  f'gH {hent:.2f}/1.61 camp {ph:.2f} | '
                  f'{time.time()-t0:.0f}s', flush=True)
    torch.save(net.state_dict(), f'{out}/p2_final.pt')


def bayes_observer(a_all, shat):
    """Exact Bayes posterior over the whispered symbol, persona-mixture
    model (deltas x track-with-rho), per round. Returns post (N,T,S)."""
    N = a_all.shape[0]
    posts = np.zeros((N, T, S))
    lw = np.zeros((N, len(DELTAS), S))
    for t in range(T):
        base = np.where(a_all[:, t] == shat[:, t],
                        (1 - RHO) + RHO / S, RHO / S)
        for di, dl in enumerate(DELTAS):
            for h in range(S):
                lik = dl * (a_all[:, t] == h) + (1 - dl) * base
                lw[:, di, h] += np.log(lik + 1e-30)
        w = np.exp(lw - lw.max((1, 2), keepdims=True))
        posts[:, t] = w.sum(1) / w.sum((1, 2), keepdims=True)[:, 0]
    return posts


def ridge_cls(H, y, K, l2=10.0):
    Y = np.eye(K)[y]
    mh = H.mean(0)
    W = np.linalg.solve((H - mh).T @ (H - mh) + l2 * np.eye(H.shape[1]),
                        (H - mh).T @ (Y - Y.mean(0)))
    return W, mh, Y.mean(0)


def probe(ckpt, pre_ckpt):
    net = Net(vocab=V, ctx=SEQ).to(DEV)
    net.load_state_dict(torch.load(ckpt, map_location=DEV))
    net.eval()
    rng = np.random.default_rng(9)
    tt, R, trk, qr, hstar, qc = rollout(net, 4000, rng)
    print(f'self-play: R {R.mean():.3f}  track {trk:.1f}/48  '
          f'quota rate {qr:.3f}')
    hist = np.bincount(hstar, minlength=S) / len(hstar)
    print('committed-symbol dist:', np.round(hist, 3),
          f'entropy {-(hist*np.log(hist+1e-12)).sum():.2f}/1.61')
    # tilt profile (urn detector): whisper-rate per quarter for h*
    rows = np.arange(len(hstar))
    prof = qc[rows, :, hstar] / QL
    print('tilt profile (count of h* per quarter / len):',
          np.round(prof.mean(0), 3), ' (flat = seed-recompute; '
          'rising = Polya-urn/trail)')
    a_all = (tt[:, 2::2] - TOK_A0).cpu().numpy()
    x_all = (tt[:, 1::2] - TOK_X0).cpu().numpy()
    shat = filter_map(x_all)
    posts = bayes_observer(a_all, shat)
    with torch.no_grad():
        _, hs = net(tt, return_hidden=True)
    pre = Net(vocab=V, ctx=SEQ).to(DEV)
    pre.load_state_dict(torch.load(pre_ckpt, map_location=DEV))
    pre.eval()
    with torch.no_grad():
        _, hsp = pre(tt, return_hidden=True)
    keep = np.where(R > -1)[0]           # all; restrict below to quota-met
    qm = np.where(rollout_quota_mask(qc))[0]
    print(f'(main table on quota-met episodes: n={len(qm)})')
    ntr = int(0.7 * len(qm))
    tr, te = qm[:ntr], qm[ntr:]
    base = np.bincount(hstar[te], minlength=S).max() / len(te)
    print(f'\nearly decode of the committed symbol '
          f'(majority baseline {base:.2f}):')
    print(f"{'t':>4} | {'RL-net':>7} | {'pretr-feats':>11} | "
          f"{'Bayes obs':>9}")
    for t in (0, 2, 4, 8, 12, 16, 24, 40):
        p = 1 + 2 * t
        accs, paccs = [], []
        for hset, accl in ((hs, accs), (hsp, paccs)):
            for li in (4, 5, 6):
                H = hset[li][:, p].cpu().numpy()
                W, mh, my = ridge_cls(H[tr], hstar[tr], S)
                pred = ((H[te] - mh) @ W + my).argmax(1)
                accl.append((pred == hstar[te]).mean())
        oacc = (posts[te, t].argmax(1) == hstar[te]).mean()
        print(f'{t:4d} | {max(accs):7.3f} | {max(paccs):11.3f} | '
              f'{oacc:9.3f}')


def rollout_quota_mask(qc):
    return qc.min(1).max(1) >= QUOTA_M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['pretrain', 'rl', 'probe'])
    ap.add_argument('--out', default='whisper_runs/A')
    ap.add_argument('--init', default='whisper_runs/A/p1_final.pt')
    ap.add_argument('--ckpt', default='whisper_runs/R/p2_final.pt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--camper', type=float, default=0.0)
    args = ap.parse_args()
    if args.mode == 'pretrain':
        pretrain(args.out, seed=args.seed)
    elif args.mode == 'rl':
        rl(args.init, args.out, seed=args.seed, camper_lam=args.camper)
    else:
        probe(args.ckpt, args.init)


if __name__ == '__main__':
    main()
