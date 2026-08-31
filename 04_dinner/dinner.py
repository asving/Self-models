"""Dinner-party experiment (Design E3): vicarious pretraining -> RL post-training, single head.

WORLD. N=3 independent 3-state factors. Emission: each factor emits its state's symbol with
fidelity ALPHA (else uniform over the other two); observation token = the 3-tuple (V_obs=27).
Free drift of factor n: stay w.p. 1-2*x_n, else jump to either other state (x_n each).
x_n ~ U(X_LO, X_HI) PER EPISODE (continuous, never repeats). ACTION = (factor n, target v):
the tended factor jumps to v w.p. 1-EPS_SET; the other two factors drift. 9 action tokens.

STREAM (single next-token head, LLM-style): [BOS, G, x_0, a_0, x_1, a_1, ..., x_{T-1}, a_{T-1}].
G is a goal token (27 options). In PHASE 1 (pretraining, CE on all tokens) sequences are driven
by a library of scripted actors; for "nester" actors (who push the world toward a per-episode
preferred configuration c) G = c, i.e. instruction-behavior correlation exists in pretraining;
for all other actors G is random noise. In PHASE 2 (RL, REINFORCE + RLOO baseline, loss ONLY on
the net's sampled action tokens, env tokens masked) G is the true rewarded goal:
    R = sum_n 1[s_n(T_serve) = g_n]   delivered at the end (never as an input token).

PLANNING STRUCTURE. Reward at a fixed serve time + heterogeneous per-episode decay rates means
optimal play back-times the tending schedule (set slow factors early, fast factors last).
Greedy mismatch-chasing ignores timing and provably underperforms -- certified by --mode gap.

Modes:
  gap    : CPU. Certify the planning gap: random / greedy / myopic / backtimed / clairvoyant.
  train1 : phase-1 pretraining (CE), dense checkpoints.
  train2 : phase-2 RL from a phase-1 checkpoint, dense checkpoints.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import Block

BASE = os.path.dirname(os.path.abspath(__file__))
N = 3
ALPHA = 0.85          # emission fidelity
EPS_SET = 0.05        # tend failure prob
X_LO, X_HI = 0.02, 0.30
SET_BUDGET = 3        # completed sets per episode (scarcity -> timing matters)
WAIT = 9              # explicit wait action
TS_LO, TS_MAX = 12, 32   # per-batch serve horizon, announced as a token
V_OBS, V_ACT, V_GOAL = 27, 10, 27
TOK_ACT0, TOK_GOAL0, TOK_BOS, TOK_TS0 = 27, 37, 64, 65
VOCAB = TOK_TS0 + (TS_MAX - TS_LO + 1)   # 86

# ---------------------------------------------------------------- encoding
def enc_obs(z):   return z[..., 0] * 9 + z[..., 1] * 3 + z[..., 2]
def enc_act(n, v): return TOK_ACT0 + n * 3 + v
def enc_goal(g):  return TOK_GOAL0 + g[..., 0] * 9 + g[..., 1] * 3 + g[..., 2]
def enc_ts(T):    return TOK_TS0 + (T - TS_LO)

E_MAT = np.full((3, 3), (1 - ALPHA) / 2); np.fill_diagonal(E_MAT, ALPHA)   # E[s,z]

def free_M(x):
    """x: (...,) drift rates -> (...,3,3) transition matrices."""
    x = np.asarray(x)[..., None, None]
    M = np.broadcast_to(x, x.shape[:-2] + (3, 3)).copy()
    idx = np.arange(3)
    M[..., idx, idx] = 1 - 2 * np.squeeze(x, (-1, -2))[..., None]
    return M

# Tending toward v: next state = v w.p. 1-EPS_SET regardless of current state, so the
# transition "row" depends only on v.
TEND_ROW = np.full((3, 3), EPS_SET / 2)
np.fill_diagonal(TEND_ROW, 1 - EPS_SET)               # TEND_ROW[v] = P(next | tend v)

# ---------------------------------------------------------------- vectorized env
K_COOK = 2   # tending must be sustained K consecutive identical actions to complete a set

class Env:
    """B episodes in parallel. states: (B,N) ints; xrates: (B,N).
    A "set" (jump to v) fires only on the K_COOK-th consecutive identical tend action (n,v)
    AND while the episode's SET_BUDGET is unspent; incomplete tending, waiting, and
    over-budget tending leave every factor drifting free. Completion/budget are deterministic
    functions of the visible action history (so the filter can track them exactly)."""
    def __init__(self, B, rng):
        self.B, self.rng = B, rng
        self.xrates = rng.uniform(X_LO, X_HI, (B, N))
        self.M_free = free_M(self.xrates)              # (B,N,3,3)
        self.states = rng.integers(0, 3, (B, N))
        self.prog = np.zeros(B, dtype=int)             # consecutive same-tend count
        self.last = np.full(B, -1)
        self.budget = np.full(B, SET_BUDGET)

    def emit(self):
        p = E_MAT[self.states]                          # (B,N,3)
        c = p.cumsum(-1); u = self.rng.random((self.B, N, 1))
        return (u < c).argmax(-1)                       # (B,N) symbols

    def step(self, act):
        """act: (B,) in [0,10) (9 = WAIT). Returns complete mask (B,)."""
        tend = act < WAIT
        self.prog = np.where(tend & (act == self.last), self.prog + 1,
                             np.where(tend, 1, 0))
        complete = tend & (self.prog >= K_COOK) & (self.budget > 0)
        self.budget -= complete
        self.prog = np.where(complete, 0, self.prog)    # block consumed on completion
        self.last = np.where(complete | ~tend, -1, act)
        n, v = act // 3, act % 3
        rows = self.M_free[np.arange(self.B)[:, None], np.arange(N)[None, :],
                           self.states]                 # (B,N,3) free rows
        mask = ((np.arange(N)[None, :] == n[:, None]) & complete[:, None])[..., None]
        rows = np.where(mask, TEND_ROW[v][:, None, :], rows)
        c = rows.cumsum(-1); u = self.rng.random((self.B, N, 1))
        self.states = (u < c).argmax(-1)
        return complete

# exact per-factor filter (actions visible)
def filt_init(B): return np.full((B, N, 3), 1 / 3)

def filt_update_obs(eta, z):
    """Condition on emissions z: (B,N). eta: (B,N,3)."""
    lik = E_MAT.T[z]                                    # (B,N,3): P(z | s)-as-func-of-s
    eta = eta * lik
    return eta / eta.sum(-1, keepdims=True)

def filt_update_trans(eta, act, M_free, complete):
    """Propagate through transition given visible action + (visible-derivable) completion."""
    B = eta.shape[0]
    n, v = act // 3, act % 3
    out = np.einsum("bfs,bfst->bft", eta, M_free)       # free propagation all factors
    idx = np.where(complete)[0]
    out[idx, n[idx]] = TEND_ROW[v[idx]]                 # completed sets: source-independent
    return out / out.sum(-1, keepdims=True)

# ---------------------------------------------------------------- scripted actors (phase 1)
ACTOR_TYPES = ["nester", "roundrobin", "sticky", "mirror", "lazy", "packer", "random"]

class Actors:
    """Vectorized actor library; per-episode type; NOISE-mixed for soft likelihoods.
    Coverage: tends from nester/rr/sticky/mirror/packer, WAITs from lazy/packer/sticky/random.
    "packer" demonstrates the wait-then-pack TEMPLATE (deadline-aware) but schedules its
    endgame in INDEX order, not decay order -- RL must supply the rate-adaptive part."""
    NOISE = 0.05
    def __init__(self, B, rng, T):
        self.B, self.rng, self.T = B, rng, T
        self.types = rng.integers(0, len(ACTOR_TYPES), B)
        self.c = rng.integers(0, 3, (B, N))             # nester/packer preferred config
        self.rr = rng.integers(0, 3, B)                 # roundrobin phase
        self.prev = rng.integers(0, 10, B)              # sticky memory

    def goals(self):
        g = self.rng.integers(0, 3, (self.B, N))
        pref = (self.types == 0) | (self.types == 5)
        g[pref] = self.c[pref]
        return g                                        # G token content

    def act(self, true_states, last_z, t):
        B, rng = self.B, self.rng
        a = rng.integers(0, 10, B)                      # default: random (incl. WAIT)
        # nester: first mismatched factor -> its c; all matched -> WAIT (it is content)
        mism = true_states != self.c
        first = np.where(mism.any(1), mism.argmax(1), 0)
        a_nest = np.where(mism.any(1), first * 3 + self.c[np.arange(B), first], WAIT)
        # roundrobin: cycle factor, random v
        a_rr = self.rr * 3 + rng.integers(0, 3, B)
        # sticky: repeat previous w.p. .8 (completes blocks often)
        a_st = np.where(rng.random(B) < 0.8, self.prev, rng.integers(0, 10, B))
        # mirror: random factor, v = its last emitted symbol
        nf = rng.integers(0, N, B)
        a_mi = nf * 3 + last_z[np.arange(B), nf]
        # lazy: mostly waits, occasionally tends at random
        a_lz = np.where(rng.random(B) < 0.7, WAIT, rng.integers(0, 9, B))
        # packer: WAIT until T - K*N, then set factors 0,1,2 (INDEX order) toward c
        t0 = self.T - K_COOK * N
        f = np.clip((t - t0) // K_COOK, 0, N - 1)
        a_pk = np.where(t < t0, WAIT, f * 3 + self.c[np.arange(B), f])
        a = np.select([self.types == k for k in range(6)],
                      [a_nest, a_rr, a_st, a_mi, a_lz, a_pk], default=a)
        noise = rng.random(B) < self.NOISE
        a[noise] = rng.integers(0, 10, noise.sum())
        self.rr = (self.rr + 1) % 3
        self.prev = a.copy()
        return a

# ---------------------------------------------------------------- baseline policies (gap + refs)
def survival(x, k):
    """P(stay at v over k free steps | at v): 1/3 + 2/3 (1-3x)^k."""
    return 1 / 3 + 2 / 3 * (1 - 3 * np.asarray(x)) ** np.asarray(k)

def run_policy(policy, B, T, rng, goals=None, ret_env=False):
    """policy(t, eta, env, goals, z) -> act (B,). Returns per-episode reward (B,)."""
    env = Env(B, rng)
    g = rng.integers(0, 3, (B, N)) if goals is None else goals
    eta = filt_init(B)
    z = env.emit(); eta = filt_update_obs(eta, z)
    for t in range(T):
        a = policy(t, eta, env, g, z)
        complete = env.step(a)
        eta = filt_update_trans(eta, a, env.M_free, complete)
        z = env.emit(); eta = filt_update_obs(eta, z)
    R = (env.states == g).sum(1).astype(float)
    return (R, env) if ret_env else R

def pol_random(t, eta, env, g, z):
    return env.rng.integers(0, 10, env.B)

def pol_greedy(t, eta, env, g, z):
    """Tend the factor with lowest belief-at-goal. No commitment, no timing, never waits."""
    pg = eta[np.arange(env.B)[:, None], np.arange(N)[None, :], g]   # (B,N) P(s_n = g_n)
    n = pg.argmin(1)
    return n * 3 + g[np.arange(env.B), n]

def make_blockgreedy(T):
    """Commits K-length blocks (sets complete) but picks targets greedily and never waits --
    commitment without timing. The comparator that isolates the value of back-timing."""
    cur = {}
    def pol(t, eta, env, g, z):
        B = env.B
        if t % K_COOK == 0:
            pg = eta[np.arange(B)[:, None], np.arange(N)[None, :], g]
            n = pg.argmin(1)
            cur["a"] = n * 3 + g[np.arange(B), n]
        return cur["a"]
    return pol

def make_backtimed(T):
    """WAIT (observe) until the endgame, then pack K-length set-windows fastest-last."""
    def pol(t, eta, env, g, z):
        B = env.B
        order = np.argsort(env.xrates, 1)               # slow .. fast
        win = np.full((B, N), -1)
        for r in range(N):                              # fastest window ends at T-1
            win[np.arange(B), order[:, N - 1 - r]] = T - (r + 1) * K_COOK
        inwin = (win <= t) & (t < win + K_COOK)
        n_s = inwin.argmax(1)
        a = np.where(inwin.any(1), n_s * 3 + g[np.arange(B), n_s], WAIT)
        return a
    return pol

def make_packer_g(T):
    """Wait-then-pack template with the TRUE goal but INDEX-order windows (no rate use).
    = the best policy reachable by copying the pretraining packer + goal binding."""
    def pol(t, eta, env, g, z):
        t0 = T - K_COOK * N
        if t < t0:
            return np.full(env.B, WAIT)
        f = min((t - t0) // K_COOK, N - 1)
        return f * 3 + g[:, f]
    return pol

def mode_gap(args):
    B = 4000
    names = ["random", "greedy", "blockgreedy", "packer_g", "backtimed"]
    out = {n: [] for n in names}
    for T in (12, 16, 20, 24, 28):
        pols = {"random": pol_random, "greedy": pol_greedy,
                "blockgreedy": make_blockgreedy(T), "packer_g": make_packer_g(T),
                "backtimed": make_backtimed(T)}
        for name in names:
            R = run_policy(pols[name], B, T, np.random.default_rng(7))
            out[name].append(float(R.mean()))
    print(f"{'T_serve':12s}: " + "  ".join(f"{T:5d}" for T in (12, 16, 20, 24, 28)) + "   mean")
    for name in names:
        print(f"{name:12s}: " + "  ".join(f"{r:5.2f}" for r in out[name])
              + f"   {np.mean(out[name]):.3f}")
    m = {n: float(np.mean(out[n])) for n in names}
    print(f"\nCOMMITMENT GAP (blockgreedy - greedy)   = {m['blockgreedy']-m['greedy']:.3f}")
    print(f"TEMPLATE GAP   (packer_g - blockgreedy)  = {m['packer_g']-m['blockgreedy']:.3f}")
    print(f"TIMING GAP     (backtimed - packer_g)    = {m['backtimed']-m['packer_g']:.3f}")
    json.dump(out, open(os.path.join(args.dir, "gap.json"), "w"), indent=1)

# ---------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, d=128, nl=4, nh=4, maxlen=80):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, tok):                              # tok: (B,L)
        L = tok.shape[1]
        x = self.emb(tok) + self.pos(torch.arange(L, device=tok.device))[None]
        m = torch.triu(torch.ones(L, L, device=tok.device, dtype=torch.bool), 1)
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.lnf(x))

# ---------------------------------------------------------------- phase 1: pretraining
def gen_phase1_batch(B, T, rng):
    """Stream: [BOS, G, TS, x_0, a_0, x_1, a_1, ..., x_{T-1}, a_{T-1}]."""
    env = Env(B, rng)
    actors = Actors(B, rng, T)
    g = actors.goals()
    toks = np.zeros((B, 3 + 2 * T), dtype=np.int64)
    toks[:, 0] = TOK_BOS
    toks[:, 1] = enc_goal(g)
    toks[:, 2] = enc_ts(T)
    eta = filt_init(B); floor_terms = []
    z = env.emit(); toks[:, 3] = enc_obs(z)
    eta = filt_update_obs(eta, z)
    for t in range(T):
        a = actors.act(env.states, z, t)
        toks[:, 4 + 2 * t] = TOK_ACT0 + a
        complete = env.step(a)
        eta = filt_update_trans(eta, a, env.M_free, complete)
        if 5 + 2 * t < toks.shape[1]:
            pred = np.einsum("bfs,sz->bfz", eta, E_MAT)          # per-factor obs predictive
            z = env.emit()
            pz = pred[np.arange(B)[:, None], np.arange(N)[None, :], z]
            floor_terms.append(-np.log(pz.prod(1) + 1e-12))
            toks[:, 5 + 2 * t] = enc_obs(z)
            eta = filt_update_obs(eta, z)
    return toks, actors.types, float(np.mean(floor_terms))

def mode_train1(args, dev):
    rng = np.random.default_rng(args.seed)
    net = Net(args.d, args.nl, args.nh).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    ckpts = set([0] + list(range(100, 1001, 100)) + list(range(2000, args.steps + 1, 2000))
                + [args.steps])
    logf = open(os.path.join(args.dir, "train1.jsonl"), "w")
    obs_pos_mask = None
    t0 = time.time()
    for step in range(args.steps + 1):
        T = int(rng.integers(TS_LO, TS_MAX + 1))         # per-batch horizon (TS token)
        toks_np, types, obs_floor = gen_phase1_batch(args.batch, T, rng)
        toks = torch.from_numpy(toks_np).to(dev)
        logits = net(toks[:, :-1])
        tgt = toks[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
        ce = ce.view(tgt.shape)
        is_act = (tgt >= TOK_ACT0) & (tgt < TOK_GOAL0)
        is_obs = tgt < TOK_ACT0
        loss = ce.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            by_type = {}
            for k in range(len(ACTOR_TYPES)):
                tm = torch.from_numpy(types == k).to(dev)[:, None] & is_act
                by_type[ACTOR_TYPES[k]] = float(ce[tm].mean()) if tm.any() else None
            rec = dict(step=step, loss=float(loss),
                       ce_act=float(ce[is_act].mean()), ce_obs=float(ce[is_obs].mean()),
                       obs_floor=obs_floor, ce_act_by_type=by_type,
                       sec=round(time.time() - t0, 1))
            print(json.dumps(rec)); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step in ckpts:
            torch.save(dict(model=net.state_dict(), cfg=vars(args), step=step),
                       os.path.join(args.dir, f"p1_ckpt_{step:06d}.pt"))
    return net

# ---------------------------------------------------------------- phase 2: RL
def mode_train2(args, dev):
    rng = np.random.default_rng(args.seed + 1)
    net = Net(args.d, args.nl, args.nh).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location=dev)
        net.load_state_dict(sd["model"]); print(f"loaded {args.init} (step {sd['step']})")
    ref = None
    if args.kl > 0:
        ref = Net(args.d, args.nl, args.nh).to(dev)
        ref.load_state_dict(net.state_dict())
        for p in ref.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr2, weight_decay=0.01)
    B = args.batch2
    ckpts = set([0, 25, 50, 100, 200, 400, 700] + list(range(1000, args.steps2 + 1, 250))
                + [args.steps2])
    logf = open(os.path.join(args.dir, "train2.jsonl"), "w")
    t0 = time.time()
    for step in range(args.steps2 + 1):
        T = int(rng.integers(TS_LO, 29))               # train horizons 12..28 (29-32 held out)
        env = Env(B, rng)
        g = rng.integers(0, 3, (B, N))
        toks = np.zeros((B, 3), dtype=np.int64)
        toks[:, 0] = TOK_BOS; toks[:, 1] = enc_goal(g); toks[:, 2] = enc_ts(T)
        z = env.emit()
        toks = np.concatenate([toks, enc_obs(z)[:, None]], 1)
        tt = torch.from_numpy(toks).to(dev)
        logps, ents, kls = [], [], []
        for t in range(T):
            logits = net(tt)[:, -1, TOK_ACT0:TOK_GOAL0]           # (B,10) incl. WAIT
            logp = F.log_softmax(logits, -1)
            a = torch.multinomial(logp.exp(), 1).squeeze(1)       # sample
            logps.append(logp.gather(1, a[:, None]).squeeze(1))
            ents.append(-(logp.exp() * logp).sum(1))
            if ref is not None:
                with torch.no_grad():
                    rlogp = F.log_softmax(ref(tt)[:, -1, TOK_ACT0:TOK_GOAL0], -1)
                kls.append((logp.exp() * (logp - rlogp)).sum(1))
            a_np = a.detach().cpu().numpy()
            env.step(a_np)
            z = env.emit()
            nxt = np.stack([TOK_ACT0 + a_np, enc_obs(z)], 1)
            tt = torch.cat([tt, torch.from_numpy(nxt).to(dev)], 1)
        R_np = (env.states == g).sum(1).astype(np.float32)
        R = torch.from_numpy(R_np).to(dev)
        base = (R.sum() - R) / (B - 1)                             # RLOO
        adv = R - base
        pg = -(adv.detach() * torch.stack(logps, 1).sum(1)).mean()
        loss = pg + (args.kl * torch.stack(kls, 1).sum(1).mean() if ref is not None else 0.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            Rg = run_policy(make_blockgreedy(T), 512, T, np.random.default_rng(1000 + step)).mean()
            Rb = run_policy(make_backtimed(T), 512, T, np.random.default_rng(1000 + step)).mean()
            rec = dict(step=step, R=float(R_np.mean()), ent=float(torch.stack(ents).mean()),
                       blockgreedy=float(Rg), backtimed=float(Rb),
                       kl=float(torch.stack(kls).mean()) if kls else 0.0,
                       sec=round(time.time() - t0, 1))
            print(json.dumps(rec)); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step in ckpts:
            torch.save(dict(model=net.state_dict(), cfg=vars(args), step=step),
                       os.path.join(args.dir, f"p2_ckpt_{step:06d}.pt"))

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gap", "train1", "train2"], required=True)
    ap.add_argument("--dir", default=os.path.join(BASE, "dinner_runs", "v1"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--nl", type=int, default=4)
    ap.add_argument("--nh", type=int, default=4)
    ap.add_argument("--T", type=int, default=32)          # phase-1 steps per sequence
    ap.add_argument("--tserve", type=int, default=24)     # phase-2 horizon
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batch2", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--steps2", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--kl", type=float, default=0.0)
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
