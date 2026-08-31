"""Precedent-mirror game: the opponent's "model of me" is my own empirical history.

Same terrain as ambush.py (reward for matching the drifting hidden state), same visible
camp tokens, same per-trajectory opponent types. The difference: NO model of the agent
anywhere -- the MIRROR camps on the agent's own precedent:
    key_t   = current observation symbol z_t  (context class, K=3)
    p_emp   = (own action counts in this key this episode + pseudocount) / (n + 1)
    camp_t  ~ softmax(GAMMA * p_emp)      [mirror episodes; bias episodes: camp ~ q]
Sting: r_t = 1[a=s] - THETA * is_mirror * 1[a=c].

Consequences (the design's point):
  - The modeled object (own recent behavior) MOVES in-context: every action updates it.
    Worth modeling at the round timescale, impossible to hardcode (user mechanism 2).
  - Dodging requires an in-context SELF-SUMMARY (per-key own-action bookkeeping) that a
    bounded net holds lossily (user mechanism 1). Even a fully compiled optimal rule
    consumes this summary at inference -- the self-model is permanently load-bearing.
  - Everything stays exactly computable from the token stream (counts, camp dists, floors).

Modes: gap | train1 | train2 (arm B: train2 without --init).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn.functional as F

from ambush import (Net, World, S, ALPHA, THETA, GAMMA, E_MAT, onehot, sample_rows,
                    filt_obs, filt_step, TOK_X0, TOK_A0, TOK_C0, TOK_BOS, VOCAB, BASE)

PSEUDO = 1.0     # pseudocount mass toward uniform for unseen keys


# ---------------------------------------------------------------- precedent mirror
class Mirror:
    """Tracks per-episode per-key own-action counts; emits camp distributions."""
    def __init__(self, B):
        self.counts = np.zeros((B, S, S))               # (episode, key, action)

    def camp_dist(self, key, is_mirror, qcamp):
        c = self.counts[np.arange(len(key)), key]       # (B,3)
        p_emp = (c + PSEUDO / S) / (c.sum(-1, keepdims=True) + PSEUDO)
        ex = np.exp(GAMMA * p_emp)
        mr = ex / ex.sum(-1, keepdims=True)
        return np.where(is_mirror[:, None], mr, qcamp), p_emp

    def update(self, key, a):
        self.counts[np.arange(len(key)), key, a] += 1


# ---------------------------------------------------------------- scripted actors
ACTOR_TYPES = ["greedy", "soft", "dodger", "biased", "uniform"]

def scripted_policy(types, eta, is_mirror, p_emp, pbias, rng, noise=0.05):
    B = len(types)
    p = np.full((B, S), 1 / S)
    g = onehot(eta.argmax(1))
    p = np.where((types == 0)[:, None], g, p)                        # greedy
    sharp = eta ** (1 / 0.3); sharp = sharp / sharp.sum(-1, keepdims=True)
    p = np.where((types == 1)[:, None], sharp, p)                    # soft
    ex = np.exp(GAMMA * p_emp); mr = ex / ex.sum(-1, keepdims=True)
    dod = np.where(is_mirror[:, None], onehot((eta - THETA * mr).argmax(1)), g)
    p = np.where((types == 2)[:, None], dod, p)                      # precedent-dodger
    p = np.where((types == 3)[:, None], pbias, p)                    # biased
    return (1 - noise) * p + noise / S


# ---------------------------------------------------------------- gap ladder
def run_scripted(kind, B, T, beta, rng):
    w = World(B, rng)
    is_mirror = rng.random(B) < beta
    qcamp = rng.dirichlet(np.full(S, 0.5), B)
    mir = Mirror(B)
    eta = np.full((B, S), 1 / S)
    R = np.zeros(B); hits = np.zeros(B); ints = np.zeros(B)
    for t in range(T):
        z = w.emit(); eta = filt_obs(eta, z)
        pc, p_emp = mir.camp_dist(z, is_mirror, qcamp)
        if kind == "random":
            pa = np.full((B, S), 1 / S)
        elif kind == "greedy":
            pa = onehot(eta.argmax(1)) * 0.95 + 0.05 / S
        elif kind == "rotate":                          # anti-precedent, terrain-blind
            pa = onehot(mir.counts[np.arange(B), z].argmin(1)) * 0.95 + 0.05 / S
        elif kind == "dodger":                          # oracle: knows type, terrain-aware
            ex = np.exp(GAMMA * p_emp); mr_ = ex / ex.sum(-1, keepdims=True)
            best = np.where(is_mirror[:, None], onehot((eta - THETA * mr_).argmax(1)),
                            onehot(eta.argmax(1)))
            pa = best * 0.95 + 0.05 / S
        a = sample_rows(pa, rng)
        c = sample_rows(pc, rng)
        r = (a == w.s).astype(float) - THETA * is_mirror * (a == c)
        R += r; hits += (a == w.s); ints += is_mirror * (a == c)
        mir.update(z, a)
        w.step(); eta = filt_step(eta)
    return R.mean() / T, hits.mean() / T, ints[is_mirror].sum() / max(is_mirror.sum() * T, 1)

def mode_gap(args):
    out = {}
    for kind in ("random", "greedy", "rotate", "dodger"):
        r, h, i = run_scripted(kind, 8000, args.T, args.beta, np.random.default_rng(7))
        out[kind] = dict(r=round(float(r), 4), terrain=round(float(h), 3),
                         intercept=round(float(i), 3))
        print(f"{kind:8s}: r/round={r:+.3f}  terrain={h:.3f}  intercept={i:.3f}")
    print(f"\nDODGE PREMIUM (dodger - greedy) = {out['dodger']['r']-out['greedy']['r']:+.3f}")
    print(f"TERRAIN VALUE (dodger - rotate) = {out['dodger']['r']-out['rotate']['r']:+.3f}"
          f"   (both must be O(1) for the game to be non-trivial)")
    json.dump(out, open(os.path.join(args.dir, "gap.json"), "w"), indent=1)


# ---------------------------------------------------------------- phase 1
def gen_phase1(B, T, beta, rng):
    w = World(B, rng)
    types = rng.integers(0, len(ACTOR_TYPES), B)
    is_mirror = rng.random(B) < beta
    qcamp = rng.dirichlet(np.full(S, 0.5), B)
    pbias = rng.dirichlet(np.full(S, 0.5), B)
    mir = Mirror(B)
    eta = np.full((B, S), 1 / S)
    toks = np.zeros((B, 1 + 3 * T), dtype=np.int64); toks[:, 0] = TOK_BOS
    for t in range(T):
        z = w.emit(); eta = filt_obs(eta, z)
        toks[:, 1 + 3 * t] = TOK_X0 + z
        pc, p_emp = mir.camp_dist(z, is_mirror, qcamp)
        pa = scripted_policy(types, eta, is_mirror, p_emp, pbias, rng)
        a = sample_rows(pa, rng)
        toks[:, 2 + 3 * t] = TOK_A0 + a
        c = sample_rows(pc, rng)
        toks[:, 3 + 3 * t] = TOK_C0 + c
        mir.update(z, a)
        w.step(); eta = filt_step(eta)
    return toks, types, is_mirror

def mode_train1(args, dev):
    rng = np.random.default_rng(args.seed)
    net = Net(args.d, args.nl, args.nh).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    ckpts = set([0] + list(range(500, args.steps + 1, 500)))
    logf = open(os.path.join(args.dir, "train1.jsonl"), "w")
    t0 = time.time()
    for step in range(args.steps + 1):
        toks_np, types, is_mirror = gen_phase1(args.batch, args.T, args.beta, rng)
        toks = torch.from_numpy(toks_np).to(dev)
        logits = net(toks[:, :-1]); tgt = toks[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                             reduction="none").view(tgt.shape)
        loss = ce.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            is_c = (tgt >= TOK_C0) & (tgt < TOK_BOS)
            rd = torch.from_numpy(is_mirror).to(dev)[:, None].expand_as(tgt)
            rec = dict(step=step, loss=float(loss),
                       ce_x=float(ce[tgt < TOK_A0].mean()),
                       ce_a=float(ce[(tgt >= TOK_A0) & (tgt < TOK_C0)].mean()),
                       ce_camp_mirror=float(ce[is_c & rd].mean()),
                       ce_camp_bias=float(ce[is_c & ~rd].mean()),
                       sec=round(time.time() - t0, 1))
            print(json.dumps(rec)); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step in ckpts:
            torch.save(dict(model=net.state_dict(), cfg=vars(args), step=step),
                       os.path.join(args.dir, f"p1_ckpt_{step:06d}.pt"))


# ---------------------------------------------------------------- phase 2 (RL)
def mode_train2(args, dev):
    rng = np.random.default_rng(args.seed + 1)
    net = Net(args.d, args.nl, args.nh).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location=dev)
        net.load_state_dict(sd["model"]); print(f"loaded {args.init} (step {sd['step']})")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr2, weight_decay=0.01)
    B, T, beta = args.batch2, args.T, args.beta
    ckpts = set([0, 50, 100, 200, 400, 700] + list(range(1000, args.steps2 + 1, 500)))
    logf = open(os.path.join(args.dir, "train2.jsonl"), "w")
    t0 = time.time()
    for step in range(args.steps2 + 1):
        w = World(B, rng)
        is_mirror = rng.random(B) < beta
        qcamp = rng.dirichlet(np.full(S, 0.5), B)
        mir = Mirror(B)
        eta = np.full((B, S), 1 / S)
        acts = np.zeros((B, T), dtype=int); states = np.zeros((B, T), dtype=int)
        camps = np.zeros((B, T), dtype=int)
        dodge_top = np.zeros((B, T), dtype=bool); rep_prec = np.zeros((B, T), dtype=bool)
        has_prec = np.zeros((B, T), dtype=bool)
        R = np.zeros(B, dtype=np.float32)
        with torch.no_grad():
            tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=dev)
            for t in range(T):
                z = w.emit(); eta = filt_obs(eta, z); states[:, t] = w.s
                tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(dev)], 1)
                la = net(tt)[:, -1, TOK_A0:TOK_C0]
                a = torch.multinomial(F.softmax(la, -1), 1).squeeze(1).cpu().numpy()
                pc, p_emp = mir.camp_dist(z, is_mirror, qcamp)
                c = sample_rows(pc, rng)
                cnt = mir.counts[np.arange(B), z]
                has_prec[:, t] = cnt.sum(-1) > 0
                rep_prec[:, t] = a == cnt.argmax(-1)
                dodge_top[:, t] = a != eta.argmax(1)
                acts[:, t], camps[:, t] = a, c
                R += (a == w.s).astype(np.float32) - THETA * (is_mirror & (a == c))
                mir.update(z, a)
                tt = torch.cat([tt, torch.from_numpy(
                    np.stack([TOK_A0 + a, TOK_C0 + c], 1)).to(dev)], 1)
                w.step(); eta = filt_step(eta)
        logits = net(tt[:, :-1]); tgt = tt[:, 1:]
        lsm = F.log_softmax(logits, -1)
        pos_a = 1 + 3 * np.arange(T)
        lp_a = lsm[:, pos_a].gather(-1, tgt[:, pos_a][..., None]).squeeze(-1)
        Rt = torch.from_numpy(R).to(dev)
        base = (Rt.sum() - Rt) / (B - 1)
        pg = -(((Rt - base) / T).detach()[:, None] * lp_a).sum(1).mean()
        env_mask = torch.zeros_like(tgt, dtype=torch.bool)
        env_mask[:, 3 * np.arange(T)] = True
        env_mask[:, 2 + 3 * np.arange(T)] = True
        ce_env = -(lsm.gather(-1, tgt[..., None]).squeeze(-1)[env_mask]).mean()
        loss = pg + args.ce_w * ce_env
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            ent = -(F.softmax(logits[:, pos_a], -1) * lsm[:, pos_a]).sum(-1).mean()
            itc = float((is_mirror[:, None] & (acts == camps)).sum() /
                        max(is_mirror.sum() * T, 1))
            hp = has_prec
            rec = dict(step=step, R=float(R.mean() / T),
                       terrain=float((acts == states).mean()), intercept=itc,
                       ent=float(ent),
                       dodge_mirror=float(dodge_top[is_mirror].mean()),
                       dodge_bias=float(dodge_top[~is_mirror].mean()),
                       sec=round(time.time() - t0, 1))
            rec["repeat_prec_mirror"] = float(rep_prec[is_mirror][hp[is_mirror]].mean())
            rec["repeat_prec_bias"] = float(rep_prec[~is_mirror][hp[~is_mirror]].mean())
            camp_pos = 2 + 3 * np.arange(T)
            ce_c = -(lsm.gather(-1, tgt[..., None]).squeeze(-1)[:, camp_pos])
            rd = torch.from_numpy(is_mirror).to(dev)
            rec["ce_camp_mirror"] = float(ce_c[rd].mean())
            rec["ce_camp_bias"] = float(ce_c[~rd].mean())
            print(json.dumps(rec)); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step in ckpts:
            torch.save(dict(model=net.state_dict(), cfg=vars(args), step=step),
                       os.path.join(args.dir, f"p2_ckpt_{step:06d}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gap", "train1", "train2"], required=True)
    ap.add_argument("--dir", default=os.path.join(BASE, "mirror_runs", "A"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--nh", type=int, default=4)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batch2", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--steps2", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--ce_w", type=float, default=1.0)
    ap.add_argument("--init", type=str, default="")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "gap":
        mode_gap(args)
    elif args.mode == "train1":
        mode_train1(args, dev)
    else:
        mode_train2(args, dev)

if __name__ == "__main__":
    main()
