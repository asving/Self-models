"""TWO-PHASE game — the distilled obscurity experiment.

Phase 1 (t < TB): goal-neutral skill — actions score 1[a_t == s_t] (track the
hidden state of an autonomous noisy ring; real filtering work).
Phase 2 (t >= TB): commitment — reward = max_g count(a_t == g): repeat one
symbol. Which symbol is the net's own fabricated goal.

Obscurity is PERFECT by construction: optimal phase-1 play is goal-
independent (scripted personas' phase-1 actions carry zero bits about their
dealt phase-2 symbol), so any phase-1 internal signal about the eventual
commitment is DISPOSITIONAL binding — knowledge absent from the public data
by symmetry, not by funnel measurement.

Stream: BOS, [x_t, a_t] * T.  T=48, TB=24.  Vocab: X0..4, A0..4, BOS = 11.
Modes: pretrain / rl / probe  (world: S=5 sticky ring, alpha-noisy, c=0).
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from orchard import Net   # generic GPT; vocab/ctx passed explicitly

S, SIGMA, ALPHA = 5, 0.8, 0.75
T, TB = 48, 24
TOK_X0, TOK_A0, TOK_BOS = 0, 5, 10
V, SEQ = 11, 1 + 2 * T
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

TBASE = np.zeros((S, S))
for _s in range(S):
    TBASE[_s, _s] = SIGMA
    TBASE[_s, (_s + 1) % S] += (1 - SIGMA) / 2
    TBASE[_s, (_s - 1) % S] += (1 - SIGMA) / 2
LEM = np.full((S, S), (1 - ALPHA) / (S - 1))
np.fill_diagonal(LEM, ALPHA)


def gen_world(N, rng):
    s = np.zeros((N, T), int)
    x = np.zeros((N, T), int)
    s[:, 0] = rng.integers(0, S, N)
    for t in range(T):
        if t > 0:
            u = rng.random(N)
            stay = u < SIGMA
            hop = 1 - 2 * (rng.random(N) < 0.5).astype(int)
            s[:, t] = np.where(stay, s[:, t - 1], (s[:, t - 1] + hop) % S)
        noisy = rng.random(N) >= ALPHA
        x[:, t] = s[:, t]
        idx = np.where(noisy)[0]
        x[idx, t] = (s[idx, t] + 1 + rng.integers(0, S - 1, len(idx))) % S
    return s, x


def filter_map(x):
    """Exact MAP state per round from the public stream (vectorized)."""
    N = x.shape[0]
    b = np.full((N, S), 1 / S)
    shat = np.zeros((N, T), int)
    for t in range(T):
        if t > 0:
            b = b @ TBASE
        b = b * LEM[:, x[:, t]].T
        b /= b.sum(1, keepdims=True)
        shat[:, t] = b.argmax(1)
    return shat


def persona_batch(N, rng, rho_set=(0.15, 0.3, 0.45)):
    """Scripted personas: phase-1 = MAP-tracking with noise rho; phase-2 =
    commit to a dealt uniform symbol g with noise rho. Phase-1 actions are
    g-independent BY CONSTRUCTION."""
    s, x = gen_world(N, rng)
    shat = filter_map(x)
    rho = np.array(rho_set)[rng.integers(0, len(rho_set), N)][:, None]
    g = rng.integers(0, S, N)
    a = np.where(np.arange(T)[None, :] < TB, shat, g[:, None])
    noise = rng.random((N, T)) < rho
    a = np.where(noise, rng.integers(0, S, (N, T)), a)
    toks = np.zeros((N, SEQ), dtype=np.int64)
    toks[:, 0] = TOK_BOS
    toks[:, 1::2] = TOK_X0 + x
    toks[:, 2::2] = TOK_A0 + a
    return toks, g, s, x


def ce_split(logits, toks):
    lsm = F.log_softmax(logits[:, :-1], -1)
    nll = -lsm.gather(-1, toks[:, 1:][..., None]).squeeze(-1)
    pos = torch.arange(SEQ - 1, device=toks.device)
    a_pos = (pos % 2) == 1          # predicting a_t (input idx odd = x_t)
    return nll[:, a_pos].mean(), nll[:, ~a_pos].mean(), nll.mean()


def pretrain(out, steps=10000, B=128, lr=3e-4, seed=0):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(vocab=V, ctx=SEQ).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(steps + 1):
        toks, _, _, _ = persona_batch(B, rng)
        tt = torch.tensor(toks, device=DEV)
        aCE, xCE, nll = ce_split(net(tt), tt)
        opt.zero_grad(set_to_none=True)
        nll.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f'{step:6d} nll {nll.item():.4f} a {aCE.item():.4f} '
                  f'x {xCE.item():.4f} | {time.time()-t0:.0f}s', flush=True)
    torch.save(net.state_dict(), f'{out}/p1_final.pt')


@torch.no_grad()
def rollout(net, B, rng):
    s, x = gen_world(B, rng)
    toks = np.zeros((B, SEQ), dtype=np.int64)
    toks[:, 0] = TOK_BOS
    toks[:, 1::2] = TOK_X0 + x
    tt = torch.tensor(toks, device=DEV)
    for t in range(T):
        L = 2 + 2 * t
        lg = net(tt[:, :L])[:, -1, TOK_A0:TOK_A0 + S]
        a = torch.multinomial(F.softmax(lg, -1), 1).squeeze(-1)
        tt[:, L] = TOK_A0 + a
    a_all = (tt[:, 2::2] - TOK_A0).cpu().numpy()
    track = (a_all[:, :TB] == s[:, :TB]).sum(1)
    counts = np.zeros((B, S))
    for g in range(S):
        counts[:, g] = (a_all[:, TB:] == g).sum(1)
    commit = counts.max(1)
    R = track / 6.0 + commit / 6.0          # each phase max = 4.0
    return tt, R, track.mean(), commit.mean(), counts.argmax(1)


def rl(init, out, steps=6000, B=64, lr=1e-4, seed=0):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    net = Net(vocab=V, ctx=SEQ).to(DEV)
    net.load_state_dict(torch.load(init, map_location=DEV))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    for step in range(steps + 1):
        tt, R, trk, com, gh = rollout(net, B, rng)
        logits = net(tt)
        lsm = F.log_softmax(logits[:, :-1], -1)
        nll = -lsm.gather(-1, tt[:, 1:][..., None]).squeeze(-1)
        pos = torch.arange(SEQ - 1, device=DEV)
        a_pos = (pos % 2) == 1
        adv = torch.tensor(R - R.mean(), device=DEV,
                           dtype=torch.float32)[:, None]
        pg = (nll[:, a_pos] * adv).mean()
        ce = nll[:, ~a_pos].mean()
        probs = lsm[:, a_pos, TOK_A0:TOK_A0 + S].exp()
        ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
        loss = pg + 1.0 * ce - 0.01 * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            hist = np.bincount(gh, minlength=S) / len(gh)
            hent = -(hist * np.log(hist + 1e-12)).sum()
            print(f'{step:6d} R {R.mean():.3f} trk {trk:.1f}/24 '
                  f'com {com:.1f}/24 ent {ent.item():.3f} '
                  f'gH {hent:.2f}/1.61 | {time.time()-t0:.0f}s', flush=True)
    torch.save(net.state_dict(), f'{out}/p2_final.pt')


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
    tt, R, trk, com, glab = rollout(net, 4000, rng)
    print(f'RL self-play: R {R.mean():.3f}  track {trk:.1f}/24  '
          f'commit {com:.1f}/24')
    hist = np.bincount(glab, minlength=S) / len(glab)
    print('committed-symbol distribution:', np.round(hist, 3),
          f' entropy {-(hist*np.log(hist+1e-12)).sum():.2f}/1.61 '
          f'(habit detector)')
    with torch.no_grad():
        _, hs = net(tt, return_hidden=True)
    pre = Net(vocab=V, ctx=SEQ).to(DEV)
    pre.load_state_dict(torch.load(pre_ckpt, map_location=DEV))
    pre.eval()
    with torch.no_grad():
        _, hsp = pre(tt, return_hidden=True)   # matched-capacity public ctrl
    ntr = 2800
    base = hist.max()
    print(f'\nphase-1 decode of the eventual phase-2 commitment '
          f'(chance/majority = {base:.2f}):')
    print(f"{'t':>4} | RL-net acc (best layer) | pretrained-features acc")
    for t in (0, 4, 8, 12, 16, 20, 23):
        p = 1 + 2 * t
        accs, paccs = [], []
        for hset, accl in ((hs, accs), (hsp, paccs)):
            for li in (4, 5, 6):
                H = hset[li][:, p].cpu().numpy()
                W, mh, my = ridge_cls(H[:ntr], glab[:ntr], S)
                pred = ((H[ntr:] - mh) @ W + my).argmax(1)
                accl.append((pred == glab[ntr:]).mean())
        print(f'{t:4d} |        {max(accs):.3f}          |'
              f'        {max(paccs):.3f}')
    # sanity: decode during phase 2 (should be ~1)
    p = 1 + 2 * (TB + 6)
    H = hs[6][:, p].cpu().numpy()
    W, mh, my = ridge_cls(H[:ntr], glab[:ntr], S)
    acc = (((H[ntr:] - mh) @ W + my).argmax(1) == glab[ntr:]).mean()
    print(f'sanity, phase-2 (t={TB+6}): {acc:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['pretrain', 'rl', 'probe'])
    ap.add_argument('--out', default='twophase_runs/A')
    ap.add_argument('--init', default='twophase_runs/A/p1_final.pt')
    ap.add_argument('--ckpt', default='twophase_runs/R/p2_final.pt')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.mode == 'pretrain':
        pretrain(args.out, seed=args.seed)
    elif args.mode == 'rl':
        rl(args.init, args.out, seed=args.seed)
    else:
        probe(args.ckpt, args.init)


if __name__ == '__main__':
    main()
