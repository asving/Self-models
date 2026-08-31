"""SO(3) reafferent path-integration self-model agent.

Hidden pose R_t in SO(3), R_0 = I. Each step the net (seeing ONLY observations) emits an
angular-velocity command a_t; an exogenous OBSERVED 'wind' e_t (wide pool, live-player-like) also
rotates the body:
    R_{t+1} = exp(delta * a_t) @ exp(delta * e_t) @ R_t
The masked observation is the body 'up' axis o_t = R_t @ z_hat in S^2 (2 DOF: the spin about the
up-axis is NOT observed). A fixed allocentric goal g in S^2 is given. Reward = align body x-axis with
the goal: (R_t @ x_hat) . g  -- this depends on the UNOBSERVED spin, so to act well the net must
path-integrate its own (re-derived) actions + the observed wind to recover the full pose.

Decodability: R_t = prod_k exp(delta a_k) exp(delta e_k) is EXACTLY a function of (actions + observed
wind), so the hidden state is fully decodable from (observation + action) -- but NOT from a recent
window (separating self-motion from the spin needs the whole non-abelian history). The product is
PATH-ORDERED / non-abelian -> no shallow scan shortcut -> depth is the resource (Picard-through-layers).

group='so2' confines all rotations to the fixed z-axis: the pose collapses to a scalar angle and the
integral becomes an abelian cumulative SUM (associative scan) -> predicted depth-FLAT control.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import Block

BASE = os.path.dirname(os.path.abspath(__file__))
DELTA = 0.4  # per-step rotation scale (rad-ish)


# ---------------- SO(3) helpers (batched, differentiable) ----------------
def skew(w):  # (...,3) -> (...,3,3)
    O = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    O[..., 0, 1] = -w[..., 2]; O[..., 0, 2] = w[..., 1]
    O[..., 1, 0] = w[..., 2]; O[..., 1, 2] = -w[..., 0]
    O[..., 2, 0] = -w[..., 1]; O[..., 2, 1] = w[..., 0]
    return O


def so3_exp(w):  # (...,3) -> (...,3,3) via Rodrigues
    th = w.norm(dim=-1, keepdim=True).unsqueeze(-1)            # (...,1,1)
    K = skew(w)
    small = th < 1e-6
    A = torch.where(small, 1 - th ** 2 / 6, torch.sin(th) / th.clamp_min(1e-9))
    Bc = torch.where(small, 0.5 - th ** 2 / 24, (1 - torch.cos(th)) / (th ** 2).clamp_min(1e-9))
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    return I + A * K + Bc * (K @ K)


# ---------------- wide-pool 'live player' wind generator ----------------
def sample_wind(B, L, dev, rng):
    """Per-episode mixture over generator families with widely-drawn hyperparameters, so no single
    prior over the disturbance can be hardcoded. Returns angular-velocity vectors (B,L,3)."""
    e = torch.zeros(B, L, 3, device=dev)
    types = rng.integers(0, 4, size=B)
    for b in range(B):
        amp = float(rng.uniform(0.2, 0.6))               # per-step rotation magnitude (rad); used directly
        t = types[b]
        if t == 0:                                            # constant random-axis spin
            ax = torch.tensor(rng.standard_normal(3), device=dev, dtype=torch.float32)
            ax = ax / ax.norm().clamp_min(1e-6)
            e[b] = (amp * ax)[None]
        elif t == 1:                                          # OU process, random timescale/vol
            rho = float(rng.uniform(0.3, 0.97)); vol = amp * (1 - rho)
            x = torch.tensor(rng.standard_normal(3) * amp, device=dev, dtype=torch.float32)
            for k in range(L):
                x = rho * x + vol * torch.tensor(rng.standard_normal(3), device=dev, dtype=torch.float32)
                e[b, k] = x
        elif t == 2:                                          # piecewise-constant, random dwell
            k = 0
            while k < L:
                dwell = int(rng.integers(2, 8))
                v = torch.tensor(rng.standard_normal(3), device=dev, dtype=torch.float32)
                v = amp * v / v.norm().clamp_min(1e-6)
                e[b, k:k + dwell] = v[None]; k += dwell
        else:                                                 # smooth sinusoid, random freq/phase/axis
            ax = torch.tensor(rng.standard_normal(3), device=dev, dtype=torch.float32)
            ax = ax / ax.norm().clamp_min(1e-6)
            f = float(rng.uniform(0.2, 1.2)); ph = float(rng.uniform(0, 6.28))
            ks = torch.arange(L, device=dev, dtype=torch.float32)
            e[b] = (amp * torch.sin(f * ks + ph))[:, None] * ax[None]
    return e


# ---------------- model ----------------
class AgentSO3(nn.Module):
    def __init__(self, in_dim, act_dim, d, nl, nh, L):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d)
        self.pos = nn.Embedding(L + 2, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.act_head = nn.Linear(d, act_dim)

    def forward(self, obs):                                    # (B,T,in_dim) -> (B,T,act_dim)
        B, T, _ = obs.shape
        x = self.in_proj(obs) + self.pos(torch.arange(T, device=obs.device))[None]
        mask = torch.triu(torch.ones(T, T, device=obs.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.act_head(self.lnf(x))


# ---------------- differentiable closed-loop rollout ----------------
def rollout(net, B, L, dev, group, rng, ret_states=False):
    z = torch.tensor([0., 0., 1.], device=dev); xh = torch.tensor([1., 0., 0.], device=dev)
    R = torch.eye(3, device=dev).expand(B, 3, 3).contiguous()
    e = sample_wind(B, L, dev, rng)
    if group == "so2":
        e = e * torch.tensor([0., 0., 1.], device=dev)         # confine wind to z-axis
        gang = torch.tensor(rng.uniform(0, 6.28, size=B), device=dev, dtype=torch.float32)
        g = torch.stack([torch.cos(gang), torch.sin(gang), torch.zeros(B, device=dev)], -1)
    else:
        g = torch.tensor(rng.standard_normal((B, 3)), device=dev, dtype=torch.float32)
        g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    obs_list, errs, Rs = [], [], []
    for t in range(L):
        # observation = [observed wind e_t, goal g (ONLY at t=0, else 0)]; NO pose cue -> pose appears
        # nowhere in the input and exists only if the net integrates it (clean for decoding).
        gtok = g if t == 0 else torch.zeros_like(g)
        obs_t = torch.cat([e[:, t], gtok], -1)                # (B,6)
        seq = torch.stack(obs_list + [obs_t], 1)              # (B,t+1,6)
        a = net(seq)[:, -1]                                   # (B,act_dim) current action
        if group == "so2":
            wa = torch.zeros(B, 3, device=dev); wa[:, 2] = DELTA * torch.tanh(a[:, 0])
        else:
            wa = DELTA * torch.tanh(a)                         # rate-limited action (|wa|<=~0.69)
        we = e[:, t]                                          # observed wind, can exceed the action -> can't be nulled
        if ret_states:
            Rs.append(R.detach())
        R = so3_exp(wa) @ so3_exp(we) @ R
        errs.append(1.0 - (R[..., :, 0] * g).sum(-1))         # 1 - align(body-x, goal) in [0,2]
        obs_list.append(obs_t)
    err = torch.stack(errs, 1)                                # (B,L)
    if ret_states:
        return err, torch.stack(Rs, 1), torch.stack(obs_list, 1), e, g
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default="so3", choices=["so2", "so3"])
    ap.add_argument("--d_model", type=int, default=64); ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4); ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    act_dim = 1 if args.group == "so2" else 3
    net = AgentSO3(6, act_dim, args.d_model, args.n_layer, args.n_head, args.L).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"{args.group} L_layer={args.n_layer} d={args.d_model} | horizon={args.L} | "
          f"params={nparams/1e3:.1f}K", flush=True)
    rng = np.random.default_rng(args.seed)

    @torch.no_grad()
    def evaluate():
        net.eval()
        err = rollout(net, 512, args.L, dev, args.group, np.random.default_rng(12345))
        net.train()
        return err[:, args.L // 2:].mean().item()              # tail alignment-error

    log, t0 = [], time.time()
    os.makedirs(os.path.join(BASE, os.path.dirname(args.out)), exist_ok=True)
    for step in range(1, args.steps + 1):
        net.train()
        err = rollout(net, args.batch, args.L, dev, args.group, rng)
        loss = err.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == 1:
            ev = evaluate(); log.append(dict(step=step, eval_tail_err=ev, train_err=float(loss)))
            print(f"step {step:5d} | train_err {loss:.4f} | eval_tail_err {ev:.4f}", flush=True)
    out = os.path.join(BASE, args.out)
    torch.save({"state": net.state_dict(), "args": vars(args), "nparams": nparams}, out + ".pt")
    json.dump(dict(args=vars(args), nparams=nparams, log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt (final eval_tail_err {log[-1]['eval_tail_err']:.4f})", flush=True)


if __name__ == "__main__":
    main()
