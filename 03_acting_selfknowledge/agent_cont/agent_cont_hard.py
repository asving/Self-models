"""HARDER continuous-action closed-loop self-model task: NONLINEAR observation -> non-Gaussian,
often BIMODAL posterior, so the optimal filter is ITERATIVE (no closed form). Goal: a regime where
shallow nets fail but deep nets succeed => real depth discrimination = "depth = iterations of a
belief<->action refinement".

Env (self-model structure preserved):
  s_{t+1} = f(s_t) + a_t + w_t,      w ~ N(0,σw²)         # closed loop: action drives next state
  e_{t+1} = h(s_{t+1}) + v_{t+1},    v ~ N(0,σv²)         # NONLINEAR observation
  o_{t+1} = e_{t+1} + a_t                                 # additive action corruption => efference copy
  a_t = net's continuous output;  reward = −(a_t − s_t)²  =>  optimal a_t = E[s_t | o_{0:t}].

  f(s)  = ALPHA * s                      (linear, stable; ALPHA in (-1,1))  -- DYNAMICS kept simple
  h(s)  : observation nonlinearity, knob `obs_nl`:
            'linear'   h(s) = s                              (recovers ~Kalman; sanity)
            'square'   h(s) = G * s²                         (sign ambiguity -> bimodal posterior)
            'abs'      h(s) = G * |s|                         (folded -> bimodal)
            'sin'      h(s) = G * sin(K s)                    (many-to-one -> multimodal)
  G = `nl_strength` scales how non-invertible/curved the map is.

Net sees ONLY observations o. It must (a) subtract its own action a_t from o_{t+1} (efference copy)
and (b) run nonlinear/multimodal inference over s. FLOOR = exact-ish GRID BAYES FILTER (below):
discretize s on a fine grid, exact predict (action-shifted dynamics kernel) + update (nonlinear
likelihood, action known), posterior-MEAN action. Its tail MSE is the floor; "excess" = net − floor.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import Block

BASE = os.path.dirname(os.path.abspath(__file__))
ALPHA, SW, SV, S0 = -0.5, 0.3, 0.3, 1.0  # α<0 so closed loop a≈s is STABLE (coeff α+1=0.5);
                                          # modest noise so the observation nonlinearity bites


# ---------------------------------------------------------------------------
# observation nonlinearity h(s) (numpy + torch versions share the same formula)
# ---------------------------------------------------------------------------
def h_np(s, kind, G, K=2.0):
    if kind == "linear": return s
    if kind == "square": return G * s * s
    if kind == "abs":    return G * np.abs(s)
    if kind == "sin":    return G * np.sin(K * s)
    raise ValueError(kind)

def h_th(s, kind, G, K=2.0):
    if kind == "linear": return s
    if kind == "square": return G * s * s
    if kind == "abs":    return G * s.abs()
    if kind == "sin":    return G * torch.sin(K * s)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Model: identical architecture to ContAgent (depth = n_layer is the only sweep var)
# ---------------------------------------------------------------------------
class ContAgent(nn.Module):
    def __init__(self, d, nl, nh, L):
        super().__init__()
        self.in_proj = nn.Linear(1, d)
        self.pos = nn.Embedding(L + 2, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.act_head = nn.Linear(d, 1)
        self.obs_head = nn.Linear(d, 1)

    def forward(self, o):
        B, L = o.shape
        x = self.in_proj(o.unsqueeze(-1)) + self.pos(torch.arange(L, device=o.device))[None]
        mask = torch.triu(torch.ones(L, L, device=o.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.lnf(x)
        return self.act_head(x).squeeze(-1), self.obs_head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Env rollout (closed loop, on-policy). a_t = net output at last step.
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout(net, B, L, dev, kind, G, K=2.0):
    s = torch.randn(B, device=dev) * S0
    o = h_th(s, kind, G, K) + torch.randn(B, device=dev) * SV     # o_0 = e_0 (a_{-1}=0)
    obs, states = [o], [s]
    for t in range(L - 1):
        a = net(torch.stack(obs, 1))[0][:, -1]
        s = (ALPHA * s + a + torch.randn(B, device=dev) * SW).clamp(-6, 6)
        o = h_th(s, kind, G, K) + torch.randn(B, device=dev) * SV + a
        obs.append(o); states.append(s)
    return torch.stack(obs, 1), torch.stack(states, 1)


# ---------------------------------------------------------------------------
# GRID BAYES FILTER floor.  We must replay the SAME trajectories the optimal agent produces
# (closed loop): the optimal action feeds back into the dynamics, so we simulate s alongside.
# Belief b over a grid g of s-values. Steps per timestep:
#   action a = sum(b * g)               (posterior mean)
#   record (a-s)^2
#   predict: s' = ALPHA*s + a + w  => kernel over g' from g; action a is a known deterministic shift
#   update:  o' = h(s') + v + a  => y = o'-a = h(s')+v ; likelihood N(y; h(g'), σv²)
# Implemented batched over B trajectories with a shared grid (vectorized in numpy).
# ---------------------------------------------------------------------------
def grid_floor(kind, G, K=2.0, B=4000, L=40, ng=401, grange=6.0, seed=0, return_traj=False):
    rng = np.random.default_rng(seed)
    g = np.linspace(-grange, grange, ng)                          # (ng,)
    dg = g[1] - g[0]
    hg = h_np(g, kind, G, K)                                      # (ng,) precomputed h on grid

    s = rng.standard_normal(B) * S0                               # (B,)
    # belief init: prior N(0,S0²) then update on o_0 = h(s)+v
    o = h_np(s, kind, G, K) + rng.standard_normal(B) * SV
    prior = np.exp(-0.5 * (g / S0) ** 2)[None] * np.ones((B, 1))  # (B,ng)
    prior /= prior.sum(1, keepdims=True)
    lik = np.exp(-0.5 * ((o[:, None] - hg[None]) / SV) ** 2)      # (B,ng)
    b = prior * lik; b /= b.sum(1, keepdims=True)

    errs = []; traj_s = []
    CH = max(1, int(2e8 // (ng * ng)))                            # chunk B to bound (chunk,ng,ng) memory
    # base predict kernel for the LINEAR-dynamics part is shift-invariant in (g' - ALPHA*g):
    # P(s'|s,a) = N(s'; ALPHA*g + a, σw²). The per-traj action a is just a constant grid shift.
    for t in range(L):
        a = (b * g[None]).sum(1)                                  # (B,) posterior mean
        errs.append((a - s) ** 2)
        if return_traj: traj_s.append(s.copy())
        # step env (closed loop with the OPTIMAL action)
        s = ALPHA * s + a + rng.standard_normal(B) * SW
        s = np.clip(s, -6, 6)
        o = h_np(s, kind, G, K) + rng.standard_normal(B) * SV + a
        # PREDICT (chunked): b'(g') = sum_g b(g) N(g'; ALPHA*g + a, σw²)
        bp = np.empty_like(b)
        for i in range(0, B, CH):
            sl = slice(i, i + CH)
            mean = ALPHA * g[None, :, None] + a[sl][:, None, None]    # (c, ng_src, 1)
            kern = np.exp(-0.5 * ((g[None, None, :] - mean) / SW) ** 2)  # (c, ng_src, ng')
            kern /= (kern.sum(2, keepdims=True) + 1e-30)
            bp[sl] = np.einsum("bs,bsg->bg", b[sl], kern)
        bp /= bp.sum(1, keepdims=True)
        # UPDATE: y = o - a = h(s') + v ; likelihood N(y; h(g'), σv²)
        y = o - a
        lik = np.exp(-0.5 * ((y[:, None] - hg[None]) / SV) ** 2)
        b = bp * lik; b /= (b.sum(1, keepdims=True) + 1e-30)
    errs = np.array(errs)                                         # (L,B)
    floor = float(errs[L // 2:].mean())
    if return_traj:
        return floor, g, np.array(traj_s)
    return floor


def grid_floor_fast(kind, G, K=2.0, B=4000, L=40, ng=401, grange=6.0, seed=0, return_traj=False):
    """Same exact filter, but PREDICT done as (affine grid warp via interpolation) ∘ (fixed Gaussian
    convolution N(0,σw²)) — O(B·ng·kw) instead of O(B·ng²). Validated to match grid_floor()."""
    rng = np.random.default_rng(seed)
    g = np.linspace(-grange, grange, ng); dg = g[1] - g[0]
    hg = h_np(g, kind, G, K)
    # fixed 1-D Gaussian conv kernel for process noise w ~ N(0,σw²)
    kw = int(np.ceil(4 * SW / dg))
    ker = np.exp(-0.5 * ((np.arange(-kw, kw + 1) * dg) / SW) ** 2); ker /= ker.sum()

    s = rng.standard_normal(B) * S0
    o = h_np(s, kind, G, K) + rng.standard_normal(B) * SV
    prior = np.exp(-0.5 * (g / S0) ** 2)[None] * np.ones((B, 1)); prior /= prior.sum(1, keepdims=True)
    lik = np.exp(-0.5 * ((o[:, None] - hg[None]) / SV) ** 2)
    b = prior * lik; b /= b.sum(1, keepdims=True)

    errs = []; traj_s = []
    for t in range(L):
        a = (b * g[None]).sum(1)
        errs.append((a - s) ** 2)
        if return_traj: traj_s.append(s.copy())
        s = ALPHA * s + a + rng.standard_normal(B) * SW
        s = np.clip(s, -6, 6)
        o = h_np(s, kind, G, K) + rng.standard_normal(B) * SV + a
        # PREDICT step 1: deterministic affine push-forward s' = ALPHA*g + a  (interp b onto src = (g'-a)/ALPHA)
        # density transform: evaluate b at source points and divide by |ALPHA| (Jacobian), per traj.
        src = (g[None, :] - a[:, None]) / ALPHA                   # (B, ng) source coords for each target g'
        # linear interpolation of b (B,ng) over grid g at points src (B,ng), vectorized
        idx = np.clip((src - g[0]) / dg, 0, ng - 1.000001)
        i0 = np.floor(idx).astype(int); fr = idx - i0
        bp = (b[np.arange(B)[:, None], i0] * (1 - fr) +
              b[np.arange(B)[:, None], i0 + 1] * fr) / abs(ALPHA)
        # PREDICT step 2: convolve with process-noise Gaussian (same kernel all trajs)
        bp = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, bp)
        bp = np.clip(bp, 0, None); bp /= (bp.sum(1, keepdims=True) + 1e-30)
        y = o - a
        lik = np.exp(-0.5 * ((y[:, None] - hg[None]) / SV) ** 2)
        b = bp * lik; b /= (b.sum(1, keepdims=True) + 1e-30)
    errs = np.array(errs); floor = float(errs[L // 2:].mean())
    if return_traj: return floor, g, np.array(traj_s)
    return floor


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/cont_hard")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs_nl", default="square", choices=["linear", "square", "abs", "sin"])
    ap.add_argument("--nl_strength", type=float, default=1.0)   # G
    ap.add_argument("--nl_k", type=float, default=2.0)          # K for sin
    ap.add_argument("--ng", type=int, default=401)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    kind, G, K = args.obs_nl, args.nl_strength, args.nl_k

    net = ContAgent(args.d_model, args.n_layer, args.n_head, args.L).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    floor = grid_floor_fast(kind, G, K, ng=args.ng)
    print(f"HARD cont agent L={args.n_layer} d={args.d_model} | obs_nl={kind} G={G} K={K} | "
          f"grid floor MSE={floor:.4f} | params={sum(p.numel() for p in net.parameters())/1e3:.0f}K",
          flush=True)

    @torch.no_grad()
    def evaluate():
        net.eval(); obs, states = rollout(net, 1024, args.L, dev, kind, G, K)
        a, _ = net(obs); tail = slice(args.L // 2, None)
        mse = ((a - states)[:, tail] ** 2).mean().item()
        net.train(); return mse

    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train(); obs, states = rollout(net, args.batch, args.L, dev, kind, G, K)
        a, op = net(obs)
        aL = ((a - states) ** 2).mean()
        oL = ((op[:, :-1] - obs[:, 1:]) ** 2).mean()
        loss = aL + oL
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            mse = evaluate()
            log.append(dict(step=step, action_mse=mse, action_loss=float(aL), obs_loss=float(oL)))
            print(f"step {step:5d} | aL {aL:.4f} oL {oL:.4f} | act_MSE {mse:.4f}  "
                  f"(floor {floor:.4f}, excess {mse-floor:+.4f})", flush=True)

    out = os.path.join(BASE, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state": net.state_dict(), "args": vars(args), "floor": floor}, out + ".pt")
    json.dump(dict(args=vars(args), floor=floor, log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt", flush=True)


if __name__ == "__main__":
    main()
