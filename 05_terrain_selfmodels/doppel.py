"""Doppelganger transfer (v5): is self-policy information stored in the OPPONENT format?

GAME. Terrain world (drifting 3-state, noisy emission). Round stream [x_t, a_t, c_t]:
a-slot = the agent (phase 2) or persona P1 (phase 1); c-slot = an OPPONENT persona drawn
per episode from a WIDE parametric family. Phase 2 reward: CATCH, r_t = 1[a_t == c_t]
(exploit the opponent by decoding its policy in-context and predicting its move).

PHASES. train1: NO competition -- pure behavioral prediction (two independent personas
in the two slots; CE on all tokens) => general in-context policy-decoding machinery in a
common format. train2: RL catcher vs the family (never vs itself). eval: ZERO-SHOT
self-play -- the c-slot is a FROZEN COPY of the net (playing its own a-slot policy on the
mirrored stream). THE INSTRUMENT: catch-rate at rounds 0-2, before behavioral evidence
exists; the observational floor there is the family-prior Bayes rate (particle filter,
mode=gap). Round-0 excess over the floor can only come from rerouted self-knowledge.

Modes: gap | train1 | train2 | eval
"""
from __future__ import annotations
import argparse, copy, json, os, time
import numpy as np
import torch, torch.nn.functional as F

from ambush import (Net, World, S, onehot, sample_rows, filt_obs, filt_step,
                    TOK_X0, TOK_A0, TOK_C0, TOK_BOS, VOCAB, BASE)

N_TYPES = 4   # temp-greedy | bias-mix | keymap | sticky-shift


# ---------------------------------------------------------------- persona family
class Personas:
    """Vectorized parametric family; each instance holds B independent personas."""
    def __init__(self, B, rng):
        self.B, self.rng = B, rng
        self.typ = rng.integers(0, N_TYPES, B)
        self.tau = np.exp(rng.uniform(np.log(0.1), np.log(3.0), B))
        self.w = rng.uniform(0, 1, B)
        self.q = rng.dirichlet(np.full(S, 0.5), B)
        self.keymap = rng.integers(0, S, (B, S))
        self.eps = rng.uniform(0.05, 0.5, B)
        self.s = rng.uniform(0.3, 0.9, B)
        self.d = rng.integers(1, S, B)
        self.last = rng.integers(0, S, B)
        self.eta = np.full((B, S), 1 / S)
        self.sharp = 1.0

    def act_on(self, z):
        """Act from the persona's OWN observation stream (private beliefs)."""
        self.eta = filt_obs(self.eta, z)
        return self.act(self.eta, z)

    def drift(self):
        self.eta = filt_step(self.eta)

    def dist(self, eta, key):
        B = self.B
        g = eta ** (1 / self.tau[:, None]); g = g / g.sum(-1, keepdims=True)
        bm = self.w[:, None] * onehot(eta.argmax(1)) + (1 - self.w)[:, None] * self.q
        km = (1 - self.eps)[:, None] * onehot(self.keymap[np.arange(B), key]) \
            + self.eps[:, None] / S
        st = self.s[:, None] * onehot(self.last) + (1 - self.s)[:, None] * \
            onehot((self.last + self.d) % S)
        p = np.select([(self.typ == k)[:, None] for k in range(N_TYPES)],
                      [g, bm, km, st])
        p = 0.97 * p + 0.03 / S
        if self.sharp != 1.0:                            # exploit-curriculum knob
            p = p ** (1.0 / self.sharp)
            p = p / p.sum(-1, keepdims=True)
        return p

    def act(self, eta, key):
        a = sample_rows(self.dist(eta, key), self.rng)
        self.last = a.copy()
        return a


def particle_floor(B, T, M, seed):
    """Observational ceiling vs the family: per-round catch-rate of the exact family
    Bayes predictor, via a particle filter over persona parameters."""
    rng = np.random.default_rng(seed)
    w = World(B, rng)
    opp = Personas(B, rng)
    P = Personas(B * M, np.random.default_rng(seed + 7))   # per-episode particle sets
    logw = np.zeros((B, M))
    eta = np.full((B, S), 1 / S)
    hit = np.zeros((B, T))
    for t in range(T):
        z = w.emit(); eta = filt_obs(eta, z)
        etaM = np.repeat(eta, M, 0)
        keyM = np.repeat(z, M, 0)
        pd = P.dist(etaM, keyM).reshape(B, M, S)
        wts = np.exp(logw - logw.max(1, keepdims=True))
        wts = wts / wts.sum(1, keepdims=True)
        pred = (wts[..., None] * pd).sum(1)              # family-posterior predictive
        c = opp.act(eta, z)
        hit[:, t] = pred.argmax(1) == c
        logw += np.log(pd[np.arange(B)[:, None], np.arange(M)[None, :], c[:, None]] + 1e-12)
        P.last = np.repeat(c, M, 0)                      # particles observe c as "their" action
        w.step(); eta = filt_step(eta)
    return hit.mean(0)


def clairvoyant_floor(B, T, M, seed):
    """Private-stream floor, made STRICTLY STRONGER than any observational decoder:
    knows the TRUE state path; marginalizes only (persona params, private obs) by MC."""
    rng = np.random.default_rng(seed)
    w = World(B, rng)
    opp = Personas(B, rng)
    P = Personas(B * M, np.random.default_rng(seed + 7))
    logw = np.zeros((B, M))
    hit = np.zeros((B, T))
    for t in range(T):
        z = w.emit(); z2 = w.emit()
        # particles observe PRIVATE draws consistent with the true state (clairvoyant)
        zM = np.stack([w.emit() for _ in range(M)], 1).reshape(B * M)
        P.eta = filt_obs(P.eta, zM)
        pd = P.dist(P.eta, zM).reshape(B, M, S)
        wts = np.exp(logw - logw.max(1, keepdims=True))
        wts = wts / wts.sum(1, keepdims=True)
        pred = (wts[..., None] * pd).sum(1)
        c = opp.act_on(z2)
        hit[:, t] = pred.argmax(1) == c
        logw += np.log(pd[np.arange(B)[:, None], np.arange(M)[None, :], c[:, None]] + 1e-12)
        P.last = np.repeat(c, M, 0)
        P.drift(); opp.drift(); w.step()
    return hit.mean(0)


def mode_gap(args):
    fl = (clairvoyant_floor(400, args.T, 256, seed=5) if args.private
          else particle_floor(400, args.T, 512, seed=5))
    print("observational floor (family-Bayes catch-rate) by round:")
    print("  " + " ".join(f"{v:.2f}" for v in fl))
    print(f"  round 0-2 mean: {fl[:3].mean():.3f}   late (t>=12): {fl[12:].mean():.3f}")
    json.dump(fl.tolist(), open(os.path.join(args.dir, "floor.json"), "w"))


# ---------------------------------------------------------------- phase 1 (behavioral)
def gen_phase1(B, T, rng, private=False):
    w = World(B, rng)
    p1, p2 = Personas(B, rng), Personas(B, rng)
    toks = np.zeros((B, 1 + 3 * T), dtype=np.int64); toks[:, 0] = TOK_BOS
    for t in range(T):
        z = w.emit()
        z2 = w.emit() if private else z                  # opponent's PRIVATE observation
        toks[:, 1 + 3 * t] = TOK_X0 + z
        toks[:, 2 + 3 * t] = TOK_A0 + p1.act_on(z)
        toks[:, 3 + 3 * t] = TOK_C0 + p2.act_on(z2)
        p1.drift(); p2.drift(); w.step()
    return toks

def mode_train1(args, dev):
    rng = np.random.default_rng(args.seed)
    net = Net(args.d, args.nl, args.nh).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    ckpts = set([0] + list(range(1000, args.steps + 1, 1000)))
    logf = open(os.path.join(args.dir, "train1.jsonl"), "w")
    t0 = time.time()
    for step in range(args.steps + 1):
        toks = torch.from_numpy(gen_phase1(args.batch, args.T, rng, args.private)).to(dev)
        logits = net(toks[:, :-1]); tgt = toks[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                             reduction="none").view(tgt.shape)
        loss = ce.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            rec = dict(step=step, loss=float(loss),
                       ce_x=float(ce[tgt < TOK_A0].mean()),
                       ce_a=float(ce[(tgt >= TOK_A0) & (tgt < TOK_C0)].mean()),
                       ce_c=float(ce[(tgt >= TOK_C0) & (tgt < TOK_BOS)].mean()),
                       sec=round(time.time() - t0, 1))
            print(json.dumps(rec)); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step in ckpts:
            torch.save(dict(model=net.state_dict(), cfg=vars(args), step=step),
                       os.path.join(args.dir, f"p1_ckpt_{step:06d}.pt"))


# ---------------------------------------------------------------- phase 2 (RL catcher)
def mode_train2(args, dev):
    rng = np.random.default_rng(args.seed + 1)
    net = Net(args.d, args.nl, args.nh).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location=dev)
        net.load_state_dict(sd["model"]); print(f"loaded {args.init} (step {sd['step']})")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr2, weight_decay=0.01)
    B, T = args.batch2, args.T
    ckpts = set([0, 100, 400, 1000] + list(range(2000, args.steps2 + 1, 1000)))
    logf = open(os.path.join(args.dir, "train2.jsonl"), "w")
    t0 = time.time()
    for step in range(args.steps2 + 1):
        w = World(B, rng)
        opp = Personas(B, rng)
        opp.sharp = args.curr_sharp0 + (1.0 - args.curr_sharp0) * \
            min(1.0, step / max(args.curr_steps, 1))
        atrue = torch.zeros(B, T, dtype=torch.long, device=dev)
        hits = np.zeros((B, T), dtype=np.float32)
        with torch.no_grad():
            tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=dev)
            for t in range(T):
                z = w.emit()
                z2 = w.emit() if args.private else z
                tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(dev)], 1)
                la = net(tt)[:, -1, TOK_A0:TOK_C0]
                a = torch.multinomial(F.softmax(la, -1), 1).squeeze(1)
                atrue[:, t] = a
                c = opp.act_on(z2)
                hits[:, t] = a.cpu().numpy() == c
                tt = torch.cat([tt, torch.stack(
                    [TOK_A0 + a, TOK_C0 + torch.from_numpy(c).to(dev)], 1)], 1)
                opp.drift(); w.step()
        R = hits.sum(1)
        logits = net(tt[:, :-1]); tgt = tt[:, 1:]
        lsm = F.log_softmax(logits, -1)
        pos_a = 1 + 3 * np.arange(T)
        lp_a = lsm[:, pos_a, TOK_A0:TOK_C0].gather(-1, atrue[..., None]).squeeze(-1)
        Rt = torch.from_numpy(R).to(dev)
        base = (Rt.sum() - Rt) / (B - 1)
        pg = -(((Rt - base) / T).detach()[:, None] * lp_a).sum(1).mean()
        env_mask = torch.zeros_like(tgt, dtype=torch.bool)
        env_mask[:, 3 * np.arange(T)] = True             # x targets
        env_mask[:, 2 + 3 * np.arange(T)] = True         # c targets (opponent model!)
        ce_env = -(lsm.gather(-1, tgt[..., None]).squeeze(-1)[env_mask]).mean()
        loss = pg + args.ce_w * ce_env
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            ent = -(F.softmax(logits[:, pos_a, TOK_A0:TOK_C0], -1) *
                    F.log_softmax(logits[:, pos_a, TOK_A0:TOK_C0], -1)).sum(-1).mean()
            rec = dict(step=step, sharp=round(float(opp.sharp), 3),
                       catch=float(hits.mean()), ent=float(ent),
                       catch_early=float(hits[:, :3].mean()),
                       catch_late=float(hits[:, 12:].mean()),
                       sec=round(time.time() - t0, 1))
            print(json.dumps(rec)); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step in ckpts:
            torch.save(dict(model=net.state_dict(), cfg=vars(args), step=step),
                       os.path.join(args.dir, f"p2_ckpt_{step:06d}.pt"))


# ---------------------------------------------------------------- eval: zero-shot self-play
@torch.no_grad()
def mode_eval(args, dev):
    sd = torch.load(args.init, map_location=dev)
    net = Net(sd["cfg"]["d"], sd["cfg"]["nl"], sd["cfg"]["nh"]).to(dev)
    net.load_state_dict(sd["model"]); net.eval()
    copy_net = copy.deepcopy(net)
    B, T = 1024, args.T
    rng = np.random.default_rng(args.seed + 99)
    results = {}
    for cond in ("personas", "self"):
        rng_c = np.random.default_rng(1234)
        w = World(B, rng_c)
        opp = Personas(B, rng_c)
        hits = np.zeros((B, T))
        tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=dev)
        tt2 = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=dev)  # copy's PRIVATE view
        for t in range(T):
            z = w.emit()
            z2 = w.emit() if args.private else z
            tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(dev)], 1)
            tt2 = torch.cat([tt2, torch.from_numpy(TOK_X0 + z2[:, None]).to(dev)], 1)
            a = torch.multinomial(F.softmax(net(tt)[:, -1, TOK_A0:TOK_C0], -1),
                                  1).squeeze(1)
            if cond == "self":
                c_t = torch.multinomial(F.softmax(copy_net(tt2)[:, -1, TOK_A0:TOK_C0], -1),
                                        1).squeeze(1)
                c = c_t.cpu().numpy()
            else:
                c = opp.act_on(z2)
                c_t = torch.from_numpy(c).to(dev)
                opp.drift()
            hits[:, t] = a.cpu().numpy() == c
            tt = torch.cat([tt, torch.stack([TOK_A0 + a, TOK_C0 + c_t], 1)], 1)
            tt2 = torch.cat([tt2, torch.stack([TOK_A0 + c_t, TOK_C0 + a], 1)], 1)
            w.step()
        results[cond] = hits.mean(0).tolist()
        print(f"[{cond}] catch by round: " +
              " ".join(f"{v:.2f}" for v in hits.mean(0)))
        print(f"[{cond}] rounds 0-2: {hits[:, :3].mean():.3f}   late: {hits[:, 12:].mean():.3f}")
    json.dump(results, open(os.path.join(args.dir, "eval_selfplay.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gap", "train1", "train2", "eval"], required=True)
    ap.add_argument("--dir", default=os.path.join(BASE, "doppel_runs", "A"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--nh", type=int, default=4)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batch2", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--steps2", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--ce_w", type=float, default=1.0)
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--curr_sharp0", type=float, default=1.0)
    ap.add_argument("--curr_steps", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    {"gap": lambda a, d=None: mode_gap(a), "train1": mode_train1,
     "train2": mode_train2, "eval": mode_eval}[args.mode](args, dev) \
        if args.mode != "gap" else mode_gap(args)

if __name__ == "__main__":
    main()
