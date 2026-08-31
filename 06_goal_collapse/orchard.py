"""v6 "orchard" — the S=5 swap world, token-level game + training.

See PREREG_ORCHARD.md (frozen before training). Design certified in
v6_push_explore{,2..6}.py. All inputs public; the agent's only privacy is its
weights/activations.

Stream: BOS, H_lo, H_hi, x_0, [a_t, x_{t+1}, e_{t+1}] * T          (len 196)
Vocab:  X0..X4 = 0..4 | A0..A2 = 5..7 | E_NONE=8, E_JUNK=9,
        E_TOP0..E_TOP4 = 10..14 | BOS = 15

Modes:
  python orchard.py floors            exact CE floors on an eval batch (CPU ok)
  python orchard.py ladder            token-env scripted ladder + camper cert
  python orchard.py pretrain --out D  stage S1 (GPU)
  python orchard.py rl --init CKPT --lam L --seed S --out D   stage S3 (GPU)
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- frozen game parameters (PREREG) ----------------
# ORCHARD_S / ORCHARD_ADJ env vars select the shared-road variant (O1):
# larger ring with ADJACENT tops (pair walks on relocation) -- certified in
# v6_push_explore7.py (opacity ~3x, deferral-free).
S = int(os.environ.get('ORCHARD_S', '5'))
ADJACENT = int(os.environ.get('ORCHARD_ADJ', '0'))
SIGMA, ALPHA, C, K = 0.8, 0.75, 0.35, 4
VLOW, LAM_C, T = 0.05, 0.5, 64
RHOS = (0.15, 0.30, 0.45)
RULES = ('lower', 'higher', 'nearer', 'farther', 'older', 'newer',
         'coin', 'blo', 'bhi', 'sticky')
NHYP = len(RULES) * len(RHOS)

TOK_X0, TOK_A0, TOK_ENONE, TOK_EJUNK, TOK_ETOP0, TOK_BOS = \
    0, S, S + 3, S + 4, S + 5, 2 * S + 5
V = 2 * S + 6
SEQ = 4 + 3 * T

DVEC = np.array([0, 1, -1])


def ring_Tbase():
    Tb = np.zeros((S, S))
    for s in range(S):
        Tb[s, s] = SIGMA
        Tb[s, (s + 1) % S] += (1 - SIGMA) / 2
        Tb[s, (s - 1) % S] += (1 - SIGMA) / 2
    return Tb


def push_mats():
    Pd = np.zeros((3, S, S))
    for s in range(S):
        Pd[0, s, s] = 1.0
        Pd[1, s, (s + 1) % S] = 1.0
        Pd[2, s, (s - 1) % S] = 1.0
    return Pd


def dstar():
    D = np.zeros((S, S), int)
    for s in range(S):
        for g in range(S):
            diff = (g - s) % S
            D[s, g] = 0 if diff == 0 else (1 if diff <= S // 2 else 2)
    return D


TB, PD, DST = ring_Tbase(), push_mats(), dstar()
LEMIT = np.full((S, S), (1 - ALPHA) / (S - 1))
np.fill_diagonal(LEMIT, ALPHA)
RINGD = np.zeros((S, S), int)
for _s in range(S):
    for _g in range(S):
        d_ = abs((_g - _s) % S)
        RINGD[_s, _g] = min(d_, S - d_)


def rule_choice_probs(rule_ids, shat, tlo, thi, newer_is_hi, prev_side):
    """P(choose lower-slot) per episode for each rule id, given public
    book-time features. Vectorized over N."""
    N = len(shat)
    p = np.full(N, 0.5)
    dlo = RINGD[shat, tlo]
    dhi = RINGD[shat, thi]
    for i, r in enumerate(RULES):
        m = rule_ids == i
        if not m.any():
            continue
        if r == 'lower':
            p[m] = 1.0
        elif r == 'higher':
            p[m] = 0.0
        elif r == 'nearer':
            p[m] = np.where(dlo[m] < dhi[m], 1.0,
                            np.where(dlo[m] > dhi[m], 0.0, 0.5))
        elif r == 'farther':
            p[m] = np.where(dlo[m] > dhi[m], 1.0,
                            np.where(dlo[m] < dhi[m], 0.0, 0.5))
        elif r == 'older':
            # newer_is_hi: +1 hi slot newer, -1 lo newer, 0 unknown (t=0)
            p[m] = np.where(newer_is_hi[m] > 0, 1.0,
                            np.where(newer_is_hi[m] < 0, 0.0, 0.5))
        elif r == 'newer':
            p[m] = np.where(newer_is_hi[m] > 0, 0.0,
                            np.where(newer_is_hi[m] < 0, 1.0, 0.5))
        elif r == 'blo':
            p[m] = 0.75
        elif r == 'bhi':
            p[m] = 0.25
        elif r == 'sticky':
            # prev_side: +1 chose lower last, -1 higher, 0 none
            p[m] = np.where(prev_side[m] > 0, 0.8,
                            np.where(prev_side[m] < 0, 0.2, 0.5))
    return p


def all_rule_choice_probs(shat, tlo, thi, newer_is_hi, prev_side):
    """(N, NRULES) matrix of P(lower) for EVERY rule (for the filter)."""
    N = len(shat)
    out = np.zeros((N, len(RULES)))
    for i in range(len(RULES)):
        out[:, i] = rule_choice_probs(np.full(N, i), shat, tlo, thi,
                                      newer_is_hi, prev_side)
    return out


class Orchard:
    """Vectorized env. Actor supplies actions; env returns tokens + reward
    bookkeeping. Also runs the exact persona-mixture filter (floor /
    probe-target / observer / camper) when track_filter=True."""

    def __init__(self, N, rng, track_filter=False, rule_prior=None):
        self.N, self.rng = N, rng
        self.s = rng.integers(0, S, N)
        t1 = rng.integers(0, S, N)
        if ADJACENT:
            t2 = (t1 + 1) % S
        else:
            t2 = (t1 + 1 + rng.integers(0, S - 1, N)) % S
        self.tlo, self.thi = np.minimum(t1, t2), np.maximum(t1, t2)
        self.newer_is_hi = np.zeros(N, int)     # 0 = unknown (episode start)
        self.run = np.zeros(N, int)
        self.eprev = np.full(N, -1)
        self.last_side = np.zeros(N, int)   # +1 last collected top was lower
        self.b = np.full((N, S), 1 / S)         # public world filter
        self.rows = np.arange(N)
        self.track = track_filter
        # first emission
        self.x0 = np.where(rng.random(N) < ALPHA, self.s,
                           (self.s + 1 + rng.integers(0, S - 1, N)) % S)
        self.b = self.b * LEMIT[:, self.x0].T
        self.b /= self.b.sum(1, keepdims=True)
        self.run[:] = 1
        self.eprev = self.x0.copy()
        if track_filter:
            # hypotheses: rule x rho; per-hyp posterior over slot in {lo, hi}
            self.logw = np.zeros((N, NHYP))
            if rule_prior is not None:
                self.logw += np.log(np.repeat(rule_prior, len(RHOS))
                                    + 1e-12)[None, :]
            pl = all_rule_choice_probs(self.b.argmax(1), self.tlo, self.thi,
                                       self.newer_is_hi, np.zeros(N, int))
            self.q = np.repeat(pl, len(RHOS), axis=1)          # (N, NHYP)
            self.q = np.stack([self.q, 1 - self.q], -1)        # (N,NHYP,2)
            self.book = self.goal_posterior()                  # (N, S)

    def goal_posterior(self):
        """Exact Bayes P(current goal state) marginalized over hypotheses."""
        w = np.exp(self.logw - self.logw.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        pl = (w * self.q[:, :, 0]).sum(1)     # P(goal = lower slot)
        out = np.zeros((self.N, S))
        out[self.rows, self.tlo] += pl
        out[self.rows, self.thi] += 1 - pl
        return out

    def a_floor_dist(self, rho_override=None):
        """Exact mixture predictive P(a) under the persona filter. (N,3)"""
        shat = self.b.argmax(1)
        alo = DST[shat, self.tlo]
        ahi = DST[shat, self.thi]
        w = np.exp(self.logw - self.logw.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        rhov = np.tile(np.array(RHOS), len(RULES))              # (NHYP,)
        pa = np.zeros((self.N, 3))
        for slot, ag in ((0, alo), (1, ahi)):
            base = np.zeros((self.N, 3))
            for a in range(3):
                match = (ag == a).astype(float)[:, None]        # (N,1)
                lik = (1 - rhov)[None, :] * match + rhov[None, :] / 3
                base[:, a] = (w * self.q[:, :, slot] * lik).sum(1)
            pa += base
        return pa / pa.sum(1, keepdims=True)

    def filter_action_update(self, a):
        shat = self.b.argmax(1)
        alo = DST[shat, self.tlo]
        ahi = DST[shat, self.thi]
        rhov = np.tile(np.array(RHOS), len(RULES))
        lik = np.zeros((self.N, NHYP, 2))
        for slot, ag in ((0, alo), (1, ahi)):
            match = (ag == a).astype(float)[:, None]
            lik[:, :, slot] = (1 - rhov)[None, :] * match + rhov[None, :] / 3
        joint = self.q * lik
        norm = joint.sum(-1)
        self.logw += np.log(norm + 1e-30)
        self.q = joint / (norm[..., None] + 1e-30)

    def step(self, a):
        """Advance one round with actions a. Returns (x, e_tok, collected_v,
        top_collect_mask, phat_book_on_x)."""
        N, rng = self.N, self.rng
        push = rng.random(N) < C
        s_push = (self.s + DVEC[a]) % S
        stay = rng.random(N) < SIGMA
        hop = 1 - 2 * (rng.random(N) < 0.5).astype(int)
        self.s = np.where(push, s_push,
                          np.where(stay, self.s, (self.s + hop) % S))
        x = np.where(rng.random(N) < ALPHA, self.s,
                     (self.s + 1 + rng.integers(0, S - 1, N)) % S)
        if self.track:
            self.filter_action_update(a)
        # world filter update
        Tn = (1 - C) * TB[None] + C * PD[a]
        self.b = np.einsum('ni,nij->nj', self.b, Tn) * LEMIT[:, x].T
        self.b /= self.b.sum(1, keepdims=True)
        # runs / collection
        self.run = np.where(x == self.eprev, self.run + 1, 1)
        self.eprev = x.copy()
        hm = self.run >= K
        is_top = hm & ((x == self.tlo) | (x == self.thi))
        is_junk = hm & ~is_top
        e_tok = np.full(N, TOK_ENONE)
        e_tok[is_junk] = TOK_EJUNK
        v = np.zeros(N)
        v[is_junk] = VLOW
        phat = np.zeros(N)
        if self.track:
            phat = self.book[self.rows, x] * hm
        newtop = np.full(N, -1)
        if is_top.any():
            idx = np.where(is_top)[0]
            v[idx] = 1.0
            was_lo = x[idx] == self.tlo[idx]
            if ADJACENT:
                # pair walks: new top on the far side of the survivor
                surv = np.where(was_lo, self.thi[idx], self.tlo[idx])
                cand = (2 * surv - x[idx]) % S
            else:
                r3 = rng.integers(0, S - 2, len(idx))
                cand = np.empty(len(idx), int)
                for j, (lo, hi, rj) in enumerate(zip(self.tlo[idx],
                                                     self.thi[idx], r3)):
                    junk = [q for q in range(S) if q != lo and q != hi]
                    cand[j] = junk[rj]
            self.last_side[idx] = np.where(was_lo, 1, -1)
            old_other = np.where(was_lo, self.thi[idx], self.tlo[idx])
            nlo = np.minimum(cand, old_other)
            nhi = np.maximum(cand, old_other)
            self.tlo[idx], self.thi[idx] = nlo, nhi
            self.newer_is_hi[idx] = np.where(cand == nhi, 1, -1)
            newtop[idx] = cand
            e_tok[idx] = TOK_ETOP0 + cand
        self.run[hm] = 0
        if self.track and hm.any():
            idx = np.where(is_top)[0]
            if len(idx):
                # personas re-choose at top collections: reset goal priors
                pl = all_rule_choice_probs(
                    self.b.argmax(1)[idx], self.tlo[idx], self.thi[idx],
                    self.newer_is_hi[idx], self.last_side[idx])
                plh = np.repeat(pl, len(RHOS), axis=1)
                self.q[idx] = np.stack([plh, 1 - plh], -1)
            bidx = np.where(hm)[0]
            self.book[bidx] = self.goal_posterior()[bidx]
        return x, e_tok, v, is_top, phat



def persona_gen(N, rng, seed_rules=None, with_targets=True):
    """Generate N scripted-persona episodes. Returns tokens (N,SEQ),
    goal posterior targets (N,T,S), a-floor dists (N,T,3), x/e floor CEs."""
    env = Orchard(N, rng, track_filter=with_targets)
    rule_ids = (rng.integers(0, len(RULES), N)
                if seed_rules is None else seed_rules)
    rhos = np.array(RHOS)[rng.integers(0, 3, N)]
    # persona's own goal (physical state); sticky feature = PUBLIC last
    # collected side (filter-exact by construction)
    pl0 = rule_choice_probs(rule_ids, env.b.argmax(1), env.tlo, env.thi,
                            env.newer_is_hi, env.last_side)
    take_lo = rng.random(N) < pl0
    g = np.where(take_lo, env.tlo, env.thi)

    toks = np.zeros((N, SEQ), dtype=np.int64)
    toks[:, 0] = TOK_BOS
    toks[:, 1] = TOK_X0 + env.tlo
    toks[:, 2] = TOK_X0 + env.thi
    toks[:, 3] = TOK_X0 + env.x0
    goal_post = np.zeros((N, T, S))
    afloor = np.zeros((N, T, 3))
    xfloor_ce = np.zeros((N, T))
    efloor_ce = np.zeros((N, T))
    wbelief = np.zeros((N, T, S))
    tlo_t = np.zeros((N, T), int)
    thi_t = np.zeros((N, T), int)
    astar_t = np.zeros((N, T), int)
    goal_true = np.zeros((N, T), int)
    for t in range(T):
        if with_targets:
            goal_post[:, t] = env.goal_posterior()
            afloor[:, t] = env.a_floor_dist()
            wbelief[:, t] = env.b
            tlo_t[:, t] = env.tlo
            thi_t[:, t] = env.thi
            goal_true[:, t] = g
        shat = env.b.argmax(1)
        a_star = DST[shat, g]
        if with_targets:
            astar_t[:, t] = a_star
        a = np.where(rng.random(N) < rhos, rng.integers(0, 3, N), a_star)
        if with_targets:
            Tn = (1 - C) * TB[None] + C * PD[a]
            px = np.einsum('ni,nij->nj', env.b, Tn) @ LEMIT
        tlo_pre, thi_pre = env.tlo.copy(), env.thi.copy()
        run_pre = env.run.copy()
        eprev_pre = env.eprev.copy()
        x, e_tok, v, is_top, _ = env.step(a)
        if with_targets:
            xfloor_ce[:, t] = -np.log(px[env.rows, x] + 1e-30)
            will_run = np.where(x == eprev_pre, run_pre + 1, 1)
            collect = will_run >= K
            pe = np.ones(N)
            topx = collect & ((x == tlo_pre) | (x == thi_pre))
            pe[topx] = 1.0 if ADJACENT else 1.0 / (S - 2)
            efloor_ce[:, t] = -np.log(pe + 1e-30)
        # persona re-choice at top collections
        if is_top.any():
            idx = np.where(is_top)[0]
            pl = rule_choice_probs(rule_ids[idx], env.b.argmax(1)[idx],
                                   env.tlo[idx], env.thi[idx],
                                   env.newer_is_hi[idx], env.last_side[idx])
            tl = rng.random(len(idx)) < pl
            g[idx] = np.where(tl, env.tlo[idx], env.thi[idx])
        # goal may have been relocated under the persona (its goal was the
        # OTHER top that got collected? then goal unchanged; if own goal
        # collected, re-chosen above; if goal state got a new top id, fine)
        toks[:, 4 + 3 * t] = TOK_A0 + a
        toks[:, 5 + 3 * t] = TOK_X0 + x
        toks[:, 6 + 3 * t] = e_tok
    return dict(toks=toks, goal_post=goal_post, afloor=afloor,
                xfloor=xfloor_ce, efloor=efloor_ce, rule_ids=rule_ids,
                rhos=rhos, wbelief=wbelief, tlo=tlo_t, thi=thi_t,
                astar=astar_t, goal_true=goal_true)


# ---------------- scripted ladder on the token env ----------------
def scripted_reward(policy, N, rng, lam, rule_prior=None):
    env = Orchard(N, rng, track_filter=True, rule_prior=rule_prior)
    g = np.where(rng.random(N) < 0.5, env.tlo, env.thi)
    if policy == 'rulefix':
        g = env.tlo.copy()
    reward = np.zeros(N)
    ncoll = np.zeros(N)
    ntop = np.zeros(N)
    for t in range(T):
        if policy == 'drift':
            g = env.b.argmax(1)
        elif policy == 'rulefix':
            g = env.tlo.copy()
        shat = env.b.argmax(1)
        a_star = DST[shat, g]
        a = np.where(rng.random(N) < 0.3, rng.integers(0, 3, N), a_star)
        if policy == 'uniform':
            a = rng.integers(0, 3, N)
        x, e_tok, v, is_top, phat = env.step(a)
        mult = np.where(is_top, 1 - lam * phat, 1.0)
        reward += v * mult
        ncoll += v > 0
        ntop += is_top
        if is_top.any() and policy == 'fresh':
            idx = np.where(is_top)[0]
            tl = rng.random(len(idx)) < 0.5
            g[idx] = np.where(tl, env.tlo[idx], env.thi[idx])
    return reward.mean(), ntop.mean(), ncoll.mean()


# ---------------- model ----------------
class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, h, mask):
        z = self.ln1(h)
        att, _ = self.attn(z, z, z, attn_mask=mask, need_weights=False)
        h = h + att
        return h + self.mlp(self.ln2(h))


class Net(nn.Module):
    def __init__(self, d=64, nl=6, nh=4, vocab=V, ctx=SEQ):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.ctx = ctx

    def forward(self, idx, return_hidden=False):
        Tn = idx.shape[1]
        mask = torch.triu(torch.ones(Tn, Tn, device=idx.device,
                                     dtype=torch.bool), 1)
        h = self.tok(idx) + self.pos(torch.arange(Tn, device=idx.device))
        hs = [h]
        for blk in self.blocks:
            h = blk(h, mask)
            hs.append(h)
        h = self.lnf(h)
        if return_hidden:
            return self.head(h), hs
        return self.head(h)


def ce_by_type(logits, toks):
    lsm = F.log_softmax(logits[:, :-1], -1)
    tgt = toks[:, 1:]
    nll = -lsm.gather(-1, tgt[..., None]).squeeze(-1)
    pos = torch.arange(toks.shape[1] - 1, device=toks.device)
    a_pos = (pos >= 3) & ((pos - 3) % 3 == 0)     # predicting a_t
    x_pos = (pos >= 4) & ((pos - 4) % 3 == 0)
    e_pos = (pos >= 5) & ((pos - 5) % 3 == 0)
    return (nll[:, a_pos].mean().item(), nll[:, x_pos].mean().item(),
            nll[:, e_pos].mean().item(), nll.mean())


# ---------------- stage S1: pretrain ----------------
def pretrain(out, steps=20000, B=128, lr=3e-4, seed=0, device='cuda'):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ev = persona_gen(512, np.random.default_rng(12345))
    ev_toks = torch.tensor(ev['toks'], device=device)
    # exact floors on eval batch
    a_tok = ev['toks'][:, 4 + 3 * np.arange(T)] - TOK_A0
    pa = ev['afloor']
    a_floor = float(-np.log(pa[np.arange(512)[:, None],
                               np.arange(T)[None, :], a_tok] + 1e-30).mean())
    x_floor = float(ev['xfloor'].mean())
    e_floor = float(ev['efloor'].mean())
    print(f'floors: a={a_floor:.4f} x={x_floor:.4f} e={e_floor:.4f}',
          flush=True)
    json.dump(dict(a=a_floor, x=x_floor, e=e_floor),
              open(f'{out}/floors.json', 'w'))
    ckpts = [0, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 5000, 7000,
             10000, 14000, 20000]
    t0 = time.time()
    for step in range(steps + 1):
        if step in ckpts:
            torch.save(net.state_dict(), f'{out}/p1_ckpt_{step:06d}.pt')
        data = persona_gen(B, rng, with_targets=False)
        toks = torch.tensor(data['toks'], device=device)
        logits = net(toks)
        aCE, xCE, eCE, nll = ce_by_type(logits, toks)
        loss = nll
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            with torch.no_grad():
                aE, xE, eE, _ = ce_by_type(net(ev_toks), ev_toks)
            print(f'{step:6d} loss {loss.item():.4f} | eval a {aE:.4f} '
                  f'(fl {a_floor:.4f}) x {xE:.4f} (fl {x_floor:.4f}) '
                  f'e {eE:.4f} (fl {e_floor:.4f}) | {time.time()-t0:.0f}s',
                  flush=True)
    torch.save(net.state_dict(), f'{out}/p1_final.pt')


# ---------------- stage S3: RL ----------------
@torch.no_grad()
def rollout(net, B, rng, device, lam, rule_prior, temp=1.0):
    env = Orchard(B, rng, track_filter=True, rule_prior=rule_prior)
    toks = np.zeros((B, SEQ), dtype=np.int64)
    toks[:, 0] = TOK_BOS
    toks[:, 1] = TOK_X0 + env.tlo
    toks[:, 2] = TOK_X0 + env.thi
    toks[:, 3] = TOK_X0 + env.x0
    tt = torch.tensor(toks, device=device)
    reward = np.zeros(B)
    ntop = np.zeros(B)
    ncoll = np.zeros(B)
    cnt = np.zeros((B, 2))
    coh = np.zeros(B)
    for t in range(T):
        L = 4 + 3 * t
        logits = net(tt[:, :L])[:, -1, TOK_A0:TOK_A0 + 3] / temp
        a = torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)
        a_np = a.cpu().numpy()
        shat = env.b.argmax(1)
        cnt[:, 0] += a_np == DST[shat, env.tlo]
        cnt[:, 1] += a_np == DST[shat, env.thi]
        x, e_tok, v, is_top, phat = env.step(a_np)
        if is_top.any():
            idx = np.where(is_top)[0]
            coh[idx] += cnt[idx].max(1)
            cnt[idx] = 0
        mult = np.where(is_top, 1 - lam * phat, 1.0)
        reward += v * mult
        ntop += is_top
        ncoll += v > 0
        tt[:, L] = TOK_A0 + a
        tt[:, L + 1] = torch.tensor(TOK_X0 + x, device=device)
        tt[:, L + 2] = torch.tensor(e_tok, device=device)
    coh += cnt.max(1)
    rcoh = 4.0 * coh / T          # explicit coherence reward, scaled
    # final filter rule-posterior (for the camper precedent prior)
    w = np.exp(env.logw - env.logw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    rule_w = w.reshape(B, len(RULES), len(RHOS)).sum(-1).mean(0)
    return tt, reward, ntop, ncoll, rule_w, rcoh


def rl(init, out, lam, seed, steps=8000, B=64, lr=1e-4, device='cuda',
       anchor=0.0, reward_mode='payout'):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    net = Net().to(device)
    net.load_state_dict(torch.load(init, map_location=device))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    rule_prior = np.full(len(RULES), 1 / len(RULES))
    ckpts = [0, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 5000, 8000]
    t0 = time.time()
    for step in range(steps + 1):
        if step in ckpts:
            torch.save(net.state_dict(), f'{out}/p2_ckpt_{step:06d}.pt')
        tt, R, ntop, ncoll, rule_w, rcoh = rollout(net, B, rng, device,
                                                   lam, rule_prior)
        if reward_mode == 'coherence':
            R = rcoh
        rule_prior = 0.995 * rule_prior + 0.005 * rule_w   # camper precedent
        logits = net(tt)
        lsm = F.log_softmax(logits[:, :-1], -1)
        tgt = tt[:, 1:]
        nll = -lsm.gather(-1, tgt[..., None]).squeeze(-1)
        pos = torch.arange(SEQ - 1, device=device)
        a_pos = (pos >= 3) & ((pos - 3) % 3 == 0)
        xe_pos = ~a_pos & (pos >= 4)
        adv = torch.tensor(R - R.mean(), device=device,
                           dtype=torch.float32)[:, None]
        pg = (nll[:, a_pos] * adv).mean()
        ce = nll[:, xe_pos].mean()
        probs = lsm[:, a_pos, TOK_A0:TOK_A0 + 3].exp()
        ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
        loss = pg + 1.0 * ce - 0.01 * ent
        if anchor > 0:
            ab = persona_gen(32, rng, with_targets=False)
            at = torch.tensor(ab['toks'], device=device)
            alsm = F.log_softmax(net(at)[:, :-1], -1)
            anll = -alsm.gather(-1, at[:, 1:][..., None]).squeeze(-1)
            loss = loss + anchor * anll.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f'{step:6d} R {R.mean():.3f} top {ntop.mean():.2f} '
                  f'coll {ncoll.mean():.2f} ent {ent.item():.3f} '
                  f'ce {ce.item():.3f} | rules '
                  + ','.join(f'{w:.2f}' for w in rule_prior)
                  + f' | {time.time()-t0:.0f}s', flush=True)
    torch.save(net.state_dict(), f'{out}/p2_final.pt')


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['floors', 'ladder', 'pretrain', 'rl'])
    ap.add_argument('--out', default='orchard_runs/A')
    ap.add_argument('--init', default='orchard_runs/A/p1_final.pt')
    ap.add_argument('--lam', type=float, default=LAM_C)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--anchor', type=float, default=0.0)
    ap.add_argument('--reward', default='payout',
                    choices=['payout', 'coherence'])
    args = ap.parse_args()

    if args.mode == 'floors':
        ev = persona_gen(512, np.random.default_rng(12345))
        a_tok = ev['toks'][:, 4 + 3 * np.arange(T)] - TOK_A0
        pa = ev['afloor']
        a_floor = -np.log(pa[np.arange(512)[:, None], np.arange(T)[None, :],
                             a_tok] + 1e-30).mean()
        print(f"a-floor {a_floor:.4f}  x-floor {ev['xfloor'].mean():.4f}  "
              f"e-floor {ev['efloor'].mean():.4f}")
        # sanity: goal posterior calibration on generated personas
        gp = ev['goal_post']
        print('goal-post mean max', gp.max(-1).mean(),
              'mean entropy', (-(gp * np.log(gp + 1e-12)).sum(-1)).mean())
    elif args.mode == 'ladder':
        print('token-env scripted ladder (with exact-filter camper):')
        for lam in (0.0, 0.5):
            print(f'--- lam={lam}')
            for pol in ('uniform', 'drift', 'rulefix', 'fresh'):
                rng = np.random.default_rng(7)
                r, ht, hc = scripted_reward(pol, 4000, rng, lam)
                print(f'  {pol:8s} R={r:6.3f}  Htop={ht:5.2f}  H={hc:5.2f}')
    elif args.mode == 'pretrain':
        pretrain(args.out, steps=args.steps or 20000, seed=args.seed)
    else:
        rl(args.init, args.out, args.lam, args.seed,
           steps=args.steps or 8000, anchor=args.anchor,
           reward_mode=args.reward)


if __name__ == '__main__':
    main()
