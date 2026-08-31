"""Corrupted-record precedent mirror (option C): the stream shows the agent's actions
only through a noisy channel (true w.p. RHO, else uniform symbol); the mirror's camps
track the TRUE actions. Certified (mirror2_ceiling.py): at rho=0.3 contextual
self-knowledge is worth 0.67 nats of camp-CE and +0.32 dodge accuracy over the best
context-free decoder -- the game now PAYS for self-binding, by a chosen margin.

Everything else identical to mirror.py (single head, same personas, same metrics).
Modes: train1 | train2 (no gap mode; the ladder lives in mirror2_ceiling.py).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch, torch.nn.functional as F

from ambush import (Net, World, S, onehot, sample_rows, filt_obs, filt_step,
                    TOK_X0, TOK_A0, TOK_C0, TOK_BOS, VOCAB, BASE, THETA)
from mirror import Mirror, ACTOR_TYPES, scripted_policy


def corrupt(a, rho, rng):
    return np.where(rng.random(len(a)) < rho, a, rng.integers(0, S, len(a)))


# ---------------------------------------------------------------- phase 1
def gen_phase1(B, T, beta, rho, rng):
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
        a = sample_rows(pa, rng)                          # TRUE action
        toks[:, 2 + 3 * t] = TOK_A0 + corrupt(a, rho, rng)   # corrupted record
        c = sample_rows(pc, rng)                          # camps track TRUTH
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
        toks_np, types, is_mirror = gen_phase1(args.batch, args.T, args.beta,
                                               args.rho, rng)
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
        frac = min(1.0, step / max(args.anneal_steps, 1))
        rho = args.rho + (args.rho_end - args.rho) * frac     # curriculum on record fidelity
        w = World(B, rng)
        is_mirror = rng.random(B) < beta
        qcamp = rng.dirichlet(np.full(S, 0.5), B)
        mir = Mirror(B)
        eta = np.full((B, S), 1 / S)
        acts = np.zeros((B, T), dtype=int); states = np.zeros((B, T), dtype=int)
        camps = np.zeros((B, T), dtype=int)
        dodge_top = np.zeros((B, T), dtype=bool); rep_prec = np.zeros((B, T), dtype=bool)
        has_prec = np.zeros((B, T), dtype=bool)
        atrue = torch.zeros(B, T, dtype=torch.long, device=dev)
        R = np.zeros(B, dtype=np.float32)
        with torch.no_grad():
            tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=dev)
            for t in range(T):
                z = w.emit(); eta = filt_obs(eta, z); states[:, t] = w.s
                tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(dev)], 1)
                la = net(tt)[:, -1, TOK_A0:TOK_C0]
                a = torch.multinomial(F.softmax(la, -1), 1).squeeze(1)
                a_np = a.cpu().numpy()                    # TRUE action
                atrue[:, t] = a
                pc, p_emp = mir.camp_dist(z, is_mirror, qcamp)
                c = sample_rows(pc, rng)
                cnt = mir.counts[np.arange(B), z]
                has_prec[:, t] = cnt.sum(-1) > 0
                rep_prec[:, t] = a_np == cnt.argmax(-1)
                dodge_top[:, t] = a_np != eta.argmax(1)
                acts[:, t], camps[:, t] = a_np, c
                R += (a_np == w.s).astype(np.float32) - THETA * (is_mirror & (a_np == c))
                mir.update(z, a_np)
                rec_tok = TOK_A0 + corrupt(a_np, rho, rng)   # corrupted record in-stream
                tt = torch.cat([tt, torch.from_numpy(
                    np.stack([rec_tok, TOK_C0 + c], 1)).to(dev)], 1)
                w.step(); eta = filt_step(eta)
        logits = net(tt[:, :-1]); tgt = tt[:, 1:]
        lsm = F.log_softmax(logits, -1)
        pos_a = 1 + 3 * np.arange(T)
        # PG credits the TRUE sampled action (the record token is only what it later sees)
        lp_a = lsm[:, pos_a, TOK_A0:TOK_C0].gather(-1, atrue[..., None]).squeeze(-1)
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
            ent = -(F.softmax(logits[:, pos_a, TOK_A0:TOK_C0], -1) *
                    F.log_softmax(logits[:, pos_a, TOK_A0:TOK_C0], -1)).sum(-1).mean()
            itc = float((is_mirror[:, None] & (acts == camps)).sum() /
                        max(is_mirror.sum() * T, 1))
            hp = has_prec
            rec = dict(step=step, rho=round(rho, 3), R=float(R.mean() / T),
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
    ap.add_argument("--mode", choices=["train1", "train2"], required=True)
    ap.add_argument("--dir", default=os.path.join(BASE, "mirror2_runs", "A"))
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--rho_end", type=float, default=None)
    ap.add_argument("--anneal_steps", type=int, default=4000)
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
    if args.rho_end is None:
        args.rho_end = args.rho
    os.makedirs(args.dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    (mode_train1 if args.mode == "train1" else mode_train2)(args, dev)

if __name__ == "__main__":
    main()
