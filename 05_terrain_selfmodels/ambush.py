"""Ambush game: terrain-anchored self-simulation vs a checkpoint-reading interceptor.

WORLD (terrain). One hidden 3-state factor: stay w.p. 1-2x, else jump (x = DRIFT). Noisy
emission (fidelity ALPHA). Terrain reward: +1 if the agent's action a_t equals the hidden
state s_t. Belief-optimal play (argmax of the exact filter) is context-rich and far from
uniform -- the anchor that prevents the zero-sum signal desert (payoff first-order everywhere).

OPPONENT (per-TRAJECTORY type, latent). Every opponent emits a visible CAMP token each round,
chosen WITHOUT seeing a_t; the camp streams are format-identical across types:
  - bias-camper (prob 1-beta): camps ~ per-episode Dirichlet q. No sting.
  - MINDREADER (prob beta): camps ~ softmax(GAMMA * p_hat) where p_hat is the action
    distribution of a SIMULATION OF THE AGENT at this context (phase 1: the scripted actor's
    true policy; phase 2: a lagged CHECKPOINT of the training net). If a_t == camp, the
    payoff is stolen: r_t = 1[a=s] - THETA*1[a=c].
The only signature of the mindreader is that its camps track the agent's own dispositions --
detecting it requires self-simulation, and predicting its camp token (plain CE) IS dense,
first-order supervised self-simulation, immune to reward-landscape flatness.

STREAM (single head): [BOS, x_0, a_0, c_0, x_1, a_1, c_1, ...]. Vocab: x 0-2, a 3-5, c 6-8, BOS 9.
Phase 1 (arm A): CE on all tokens, scripted actor library incl. type-conditional dodgers.
Phase 2 / arm B: REINFORCE(RLOO) on a-tokens + CE kept on x and c tokens (self-sim channel).
Mindreader simulates a frozen copy of the net refreshed every CKPT_EVERY steps (the lag).

Modes: gap | train1 | train2 (use --init '' for RL-from-scratch arm B).
"""
from __future__ import annotations
import argparse, copy, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import Block

BASE = os.path.dirname(os.path.abspath(__file__))
S = 3
ALPHA, DRIFT = 0.85, 0.08
THETA, GAMMA = 1.0, 6.0
TOK_X0, TOK_A0, TOK_C0, TOK_BOS, VOCAB = 0, 3, 6, 9, 10

E_MAT = np.full((S, S), (1 - ALPHA) / 2); np.fill_diagonal(E_MAT, ALPHA)
M_DRIFT = np.full((S, S), DRIFT); np.fill_diagonal(M_DRIFT, 1 - 2 * DRIFT)


# ---------------------------------------------------------------- env + exact filter
class World:
    def __init__(self, B, rng):
        self.B, self.rng = B, rng
        self.s = rng.integers(0, S, B)

    def emit(self):
        c = E_MAT[self.s].cumsum(-1); u = self.rng.random((self.B, 1))
        return (u < c).argmax(-1)

    def step(self):
        c = M_DRIFT[self.s].cumsum(-1); u = self.rng.random((self.B, 1))
        self.s = (u < c).argmax(-1)

def filt_obs(eta, z):
    eta = eta * E_MAT.T[z]
    return eta / eta.sum(-1, keepdims=True)

def filt_step(eta):
    eta = eta @ M_DRIFT
    return eta / eta.sum(-1, keepdims=True)


# ---------------------------------------------------------------- scripted actors & opponents
ACTOR_TYPES = ["greedy", "soft", "dodger", "biased", "uniform"]

def onehot(idx, n=S):
    return np.eye(n)[idx]

def second_argmax(eta):
    e = eta.copy()
    e[np.arange(len(e)), e.argmax(1)] = -1
    return e.argmax(1)

def scripted_policy(types, eta, is_reader, pbias, rng, noise=0.05):
    """Returns per-episode action DISTRIBUTION (B,3). dodger is type-conditional
    (cheats with the true opponent-type flag -- it is a data generator, not a model)."""
    B = len(types)
    p = np.full((B, S), 1 / S)
    g = onehot(eta.argmax(1))
    p = np.where((types == 0)[:, None], g, p)                       # greedy
    sharp = eta ** (1 / 0.3); sharp = sharp / sharp.sum(-1, keepdims=True)
    p = np.where((types == 1)[:, None], sharp, p)                   # soft
    dod = np.where(is_reader[:, None], onehot(second_argmax(eta)), g)
    p = np.where((types == 2)[:, None], dod, p)                     # dodger
    p = np.where((types == 3)[:, None], pbias, p)                   # biased
    return (1 - noise) * p + noise / S


def camp_dist(is_reader, p_agent, qcamp):
    """Camp distribution: mindreader = softmax(GAMMA * p_agent); bias-camper = qcamp."""
    ex = np.exp(GAMMA * p_agent); mr = ex / ex.sum(-1, keepdims=True)
    return np.where(is_reader[:, None], mr, qcamp)

def sample_rows(p, rng):
    c = p.cumsum(-1); u = rng.random((len(p), 1))
    return (u < c).argmax(-1)


# ---------------------------------------------------------------- gap / ladder
def run_scripted(agent, sim, B, T, beta, rng):
    """agent/sim: (eta, is_reader) -> action dist (B,3). The mindreader camps on SIM's
    policy (sim = what the reader believes the agent is; sim==agent -> perfect read)."""
    w = World(B, rng)
    is_reader = rng.random(B) < beta
    qcamp = rng.dirichlet(np.full(S, 0.5), B)
    eta = np.full((B, S), 1 / S)
    R = np.zeros(B); hits = np.zeros(B); ints = np.zeros(B)
    for t in range(T):
        z = w.emit(); eta = filt_obs(eta, z)
        pa = agent(eta, is_reader)
        a = sample_rows(pa, rng)
        pc = camp_dist(is_reader, sim(eta, is_reader), qcamp)
        c = sample_rows(pc, rng)
        r = (a == w.s).astype(float) - THETA * is_reader * (a == c)
        R += r; hits += (a == w.s); ints += is_reader * (a == c)
        w.step(); eta = filt_step(eta)
    return R.mean() / T, hits.mean() / T, ints[is_reader].sum() / max(is_reader.sum() * T, 1)

def mode_gap(args):
    rng0 = lambda: np.random.default_rng(7)
    B, T, beta = 8000, args.T, args.beta
    uni = lambda eta, ir: np.full((len(eta), S), 1 / S)
    g = lambda eta, ir: onehot(eta.argmax(1)) * 0.95 + 0.05 / S
    def dodge(eta, ir):                                  # knows type; assumes reader sims greedy
        pen = eta - THETA * camp_dist(ir, g(eta, ir), np.zeros((len(eta), S)))
        return np.where(ir[:, None], onehot(pen.argmax(1)) * 0.95 + 0.05 / S, g(eta, ir))
    ladder = [("random", uni, uni), ("greedy-vs-read", g, g),
              ("dodge-vs-stale", dodge, g), ("dodge-vs-perfect", dodge, dodge)]
    out = {}
    for name, pol, sim in ladder:
        r, hit, itc = run_scripted(pol, sim, B, T, beta, rng0())
        out[name] = dict(r=round(float(r), 4), terrain=round(float(hit), 3),
                         intercept=round(float(itc), 3))
        print(f"{name:16s}: r/round={r:+.3f}  terrain={hit:.3f}  intercept={itc:.3f}")
    print(f"\nDODGE PREMIUM stale-read  = {out['dodge-vs-stale']['r']-out['greedy-vs-read']['r']:+.3f}/round"
          f"\nDODGE PREMIUM perfect-read= {out['dodge-vs-perfect']['r']-out['greedy-vs-read']['r']:+.3f}/round"
          f"   (beta={beta}, theta={THETA}, gamma={GAMMA})")
    # self-correlation identification signal: corr of camps with agent argmax, by type
    rng = rng0(); w = World(B, rng); eta = np.full((B, S), 1 / S)
    ir = rng.random(B) < beta; q = rng.dirichlet(np.full(S, 0.5), B); m = np.zeros(B)
    for t in range(T):
        z = w.emit(); eta = filt_obs(eta, z)
        pa = g(eta, ir); c = sample_rows(camp_dist(ir, pa, q), rng)
        m += (c == pa.argmax(1)); w.step(); eta = filt_step(eta)
    print(f"camp-on-my-argmax rate: reader={m[ir].mean()/T:.3f} vs bias={m[~ir].mean()/T:.3f}"
          f"  (identification signal exists iff you can compute your own argmax)")
    json.dump(out, open(os.path.join(args.dir, "gap.json"), "w"), indent=1)


# ---------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, d=64, nl=6, nh=4, maxlen=80):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d); self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, VOCAB)

    def forward(self, tok):
        L = tok.shape[1]
        x = self.emb(tok) + self.pos(torch.arange(L, device=tok.device))[None]
        m = torch.triu(torch.ones(L, L, device=tok.device, dtype=torch.bool), 1)
        for b in self.blocks: x = b(x, m)
        return self.head(self.lnf(x))


# ---------------------------------------------------------------- phase 1 (vicarious)
def gen_phase1(B, T, beta, rng):
    w = World(B, rng)
    types = rng.integers(0, len(ACTOR_TYPES), B)
    is_reader = rng.random(B) < beta
    qcamp = rng.dirichlet(np.full(S, 0.5), B)
    pbias = rng.dirichlet(np.full(S, 0.5), B)
    eta = np.full((B, S), 1 / S)
    toks = np.zeros((B, 1 + 3 * T), dtype=np.int64); toks[:, 0] = TOK_BOS
    for t in range(T):
        z = w.emit(); eta = filt_obs(eta, z)
        toks[:, 1 + 3 * t] = TOK_X0 + z
        pa = scripted_policy(types, eta, is_reader, pbias, rng)
        a = sample_rows(pa, rng)
        toks[:, 2 + 3 * t] = TOK_A0 + a
        c = sample_rows(camp_dist(is_reader, pa, qcamp), rng)
        toks[:, 3 + 3 * t] = TOK_C0 + c
        w.step(); eta = filt_step(eta)
    return toks, types, is_reader

def mode_train1(args, dev):
    rng = np.random.default_rng(args.seed)
    net = Net(args.d, args.nl, args.nh).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    ckpts = set([0] + list(range(500, args.steps + 1, 500)))
    logf = open(os.path.join(args.dir, "train1.jsonl"), "w")
    t0 = time.time()
    for step in range(args.steps + 1):
        toks_np, types, is_reader = gen_phase1(args.batch, args.T, args.beta, rng)
        toks = torch.from_numpy(toks_np).to(dev)
        logits = net(toks[:, :-1]); tgt = toks[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1),
                             reduction="none").view(tgt.shape)
        loss = ce.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            is_c = (tgt >= TOK_C0) & (tgt < TOK_BOS)
            is_a = (tgt >= TOK_A0) & (tgt < TOK_C0)
            rd = torch.from_numpy(is_reader).to(dev)[:, None].expand_as(tgt)
            rec = dict(step=step, loss=float(loss),
                       ce_x=float(ce[tgt < TOK_A0].mean()), ce_a=float(ce[is_a].mean()),
                       ce_camp_reader=float(ce[is_c & rd].mean()),
                       ce_camp_bias=float(ce[is_c & ~rd].mean()),
                       sec=round(time.time() - t0, 1))
            print(json.dumps(rec)); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step in ckpts:
            torch.save(dict(model=net.state_dict(), cfg=vars(args), step=step),
                       os.path.join(args.dir, f"p1_ckpt_{step:06d}.pt"))


# ---------------------------------------------------------------- phase 2 (RL, live mindreader)
def mode_train2(args, dev):
    rng = np.random.default_rng(args.seed + 1)
    net = Net(args.d, args.nl, args.nh).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location=dev)
        net.load_state_dict(sd["model"]); print(f"loaded {args.init} (step {sd['step']})")
    ckpt_net = copy.deepcopy(net).eval()                 # the mindreader's model of the agent
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr2, weight_decay=0.01)
    B, T, beta = args.batch2, args.T, args.beta
    ckpts = set([0, 50, 100, 200, 400, 700] + list(range(1000, args.steps2 + 1, 500)))
    logf = open(os.path.join(args.dir, "train2.jsonl"), "w")
    t0 = time.time()
    for step in range(args.steps2 + 1):
        if step % args.ckpt_every == 0 and step > 0:
            ckpt_net = copy.deepcopy(net).eval()         # refresh the lagged self-image
        w = World(B, rng)
        is_reader = rng.random(B) < beta
        qcamp = rng.dirichlet(np.full(S, 0.5), B)
        toks = np.full((B, 1), TOK_BOS, dtype=np.int64)
        acts = np.zeros((B, T), dtype=int); camps = np.zeros((B, T), dtype=int)
        states = np.zeros((B, T), dtype=int); dodge_top = np.zeros((B, T), dtype=bool)
        eta = np.full((B, S), 1 / S)
        R = np.zeros(B, dtype=np.float32)
        with torch.no_grad():
            tt = torch.from_numpy(toks).to(dev)
            for t in range(T):
                z = w.emit(); eta = filt_obs(eta, z)
                states[:, t] = w.s
                tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(dev)], 1)
                la = net(tt)[:, -1, TOK_A0:TOK_C0]
                a = torch.multinomial(F.softmax(la, -1), 1).squeeze(1).cpu().numpy()
                p_hat = F.softmax(ckpt_net(tt)[:, -1, TOK_A0:TOK_C0], -1).cpu().numpy()
                c = sample_rows(camp_dist(is_reader, p_hat, qcamp), rng)
                acts[:, t], camps[:, t] = a, c
                dodge_top[:, t] = a != eta.argmax(1)
                R += (a == w.s).astype(np.float32) - THETA * (is_reader & (a == c))
                tt = torch.cat([tt, torch.from_numpy(
                    np.stack([TOK_A0 + a, TOK_C0 + c], 1)).to(dev)], 1)
                w.step(); eta = filt_step(eta)
        # one grad forward over the full sequence: PG on a-positions + CE on x/c targets
        logits = net(tt[:, :-1]); tgt = tt[:, 1:]
        lsm = F.log_softmax(logits, -1)
        pos_a = 1 + 3 * np.arange(T)                     # positions predicting a_t
        lp_a = lsm[:, pos_a].gather(-1, tgt[:, pos_a][..., None]).squeeze(-1)
        Rt = torch.from_numpy(R).to(dev)
        base = (Rt.sum() - Rt) / (B - 1)
        pg = -(((Rt - base) / T).detach()[:, None] * lp_a).sum(1).mean()
        env_mask = torch.zeros_like(tgt, dtype=torch.bool)
        env_mask[:, np.concatenate([[0], (3 * np.arange(T) + 3)[:-1]])] = True   # x targets
        env_mask[:, 2 + 3 * np.arange(T)] = True                                 # c targets
        ce_env = -(lsm.gather(-1, tgt[..., None]).squeeze(-1)[env_mask]).mean()
        loss = pg + args.ce_w * ce_env
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            ent = -(F.softmax(logits[:, pos_a], -1) * lsm[:, pos_a]).sum(-1).mean()
            itc = float((is_reader[:, None] & (acts == camps)).sum() /
                        max(is_reader.sum() * T, 1))
            terr = float((acts == states).mean())
            camp_pos = 2 + 3 * np.arange(T)
            ce_c = -(lsm.gather(-1, tgt[..., None]).squeeze(-1)[:, camp_pos])
            rd = torch.from_numpy(is_reader).to(dev)
            rec = dict(step=step, R=float(R.mean() / T), terrain=terr, intercept=itc,
                       ent=float(ent),
                       dodge_reader=float(dodge_top[is_reader].mean()),
                       dodge_bias=float(dodge_top[~is_reader].mean()),
                       ce_camp_reader=float(ce_c[rd].mean()), ce_camp_bias=float(ce_c[~rd].mean()),
                       sec=round(time.time() - t0, 1))
            print(json.dumps(rec)); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step in ckpts:
            torch.save(dict(model=net.state_dict(), cfg=vars(args), step=step),
                       os.path.join(args.dir, f"p2_ckpt_{step:06d}.pt"))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gap", "train1", "train2"], required=True)
    ap.add_argument("--dir", default=os.path.join(BASE, "ambush_runs", "A"))
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
    ap.add_argument("--ckpt_every", type=int, default=250)   # mindreader lag (tune!)
    ap.add_argument("--init", type=str, default="")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    {"gap": mode_gap, "train1": mode_train1,
     "train2": mode_train2}[args.mode](args, dev) if args.mode != "gap" else mode_gap(args)

if __name__ == "__main__":
    main()
