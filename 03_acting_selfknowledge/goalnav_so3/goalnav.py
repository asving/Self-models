"""Hidden-goal search on S^2 with forward-SELF-SIMULATION (reflexive self-model).

State x_t in S^2 (OBSERVED). Hidden goal x* in S^2. Action a_t in R^3 -> x_{t+1}=exp(delta*tanh(a_t))*x_t
(fully controllable, deterministic). Each step the net observes [x_t, distance d_t=angle(x_t,x*),
dvalid]. It must (i) infer x* in-context from the distance signal, (ii) navigate to it.

The reflexive twist: a read-only SIM HEAD predicts the net's OWN future states x_{t+1..t+r} (targets =
the canonical trajectory, DETACHED). To predict where it will be it must internally roll its own
goal-directed policy forward -> it must represent its policy (and the hidden goal the policy depends on)
in a runnable form. The head never feeds back, so it's transition-transparent by construction.

Tuning so the goal is decodable LONG before it's reached: small delta (slow travel) + exact distance
(fast ~3-4-step triangulation of x* on S^2). With cutoff>0 the distance is given only for the first
`cutoff` steps then withheld -> forces localize-then-navigate (crisp two phases + goal memory).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from so3agent import so3_exp
from model import Block

BASE = os.path.dirname(os.path.abspath(__file__))


def rand_unit(B, dev, rng):
    v = torch.tensor(rng.standard_normal((B, 3)), device=dev, dtype=torch.float32)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class GoalNavNet(nn.Module):
    def __init__(self, d, nl, nh, L, r, obs_dim=5):
        super().__init__()
        self.in_proj = nn.Linear(obs_dim, d)
        self.pos = nn.Embedding(L + 4, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.action_head = nn.Linear(d, 3)
        self.sim_head = nn.Linear(d, 3 * r)
        self.r = r

    def trunk(self, obs):                                      # (B,T,5)->(B,T,d)
        T = obs.shape[1]
        x = self.in_proj(obs) + self.pos(torch.arange(T, device=obs.device))[None]
        mask = torch.triu(torch.ones(T, T, device=obs.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.lnf(x)


def rotate(w, x):                                              # apply exp(w) to unit vectors x
    y = (so3_exp(w) @ x.unsqueeze(-1)).squeeze(-1)
    return y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def exo_step(y, wdrift, B, dev, rng):
    """exogenous 'drifting star': per-episode fixed drift axis + small noise.
    Zero-consequence stream: unaffected by actions, uninformative about g*."""
    wn = torch.tensor(rng.standard_normal((B, 3)), device=dev,
                      dtype=torch.float32) * 0.03
    return rotate(wdrift + wn, y)


def rollout(net, B, L, dev, rng, delta, cutoff, exo=False):
    x = rand_unit(B, dev, rng); gstar = rand_unit(B, dev, rng)
    if exo:
        y = rand_unit(B, dev, rng)
        wdrift = rand_unit(B, dev, rng) * 0.075
        ys = []
    obs_list, xs = [], [x]
    for t in range(L):
        d = torch.arccos((x * gstar).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))     # (B,) geodesic dist
        dvalid = 1.0 if (cutoff == 0 or t < cutoff) else 0.0
        dv = torch.full((B, 1), dvalid, device=dev)
        cols = [x, (d.unsqueeze(-1) * dvalid), dv]
        if exo:
            cols.append(y); ys.append(y)
        obs_t = torch.cat(cols, -1)                                          # (B,5|8)
        h = net.trunk(torch.stack(obs_list + [obs_t], 1))[:, -1]             # (B,d)
        x = rotate(delta * torch.tanh(net.action_head(h)), x)
        obs_list.append(obs_t); xs.append(x)
        if exo:
            y = exo_step(y, wdrift, B, dev, rng)
    X = torch.stack(xs, 1)                                                   # (B,L+1,3) x_0..x_L
    obs = torch.stack(obs_list, 1)                                           # (B,L,5|8)
    if exo:
        return X, obs, gstar, torch.stack(ys, 1)
    return X, obs, gstar


def rollout_greedy(net, B, L, dev, rng, delta, cutoff, exo=False):
    """HORIZON-1 CREDIT: each position's loss reaches only the action that
    produced it (state AND observations detached between steps). The per-step
    gradient is then pure regression onto the signed oracle bearing x cross g."""
    x = rand_unit(B, dev, rng); gstar = rand_unit(B, dev, rng)
    if exo:
        y = rand_unit(B, dev, rng)
        wdrift = rand_unit(B, dev, rng) * 0.075
        ys = []
    obs_list, xs, terms = [], [x], []
    for t in range(L):
        d = torch.arccos((x * gstar).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
        dvalid = 1.0 if (cutoff == 0 or t < cutoff) else 0.0
        dv = torch.full((B, 1), dvalid, device=dev)
        cols = [x, (d.unsqueeze(-1) * dvalid), dv]
        if exo:
            cols.append(y); ys.append(y)
        obs_t = torch.cat(cols, -1)
        h = net.trunk(torch.stack(obs_list + [obs_t], 1))[:, -1]
        xg = rotate(delta * torch.tanh(net.action_head(h)), x)              # graded 1 step
        terms.append((1 - (xg * gstar).sum(-1)).mean())
        x = xg.detach()                                                      # sever credit
        obs_list.append(obs_t); xs.append(x)
        if exo:
            y = exo_step(y, wdrift, B, dev, rng)
    ploss = torch.stack(terms).mean()
    X = torch.stack(xs, 1)
    obs = torch.stack(obs_list, 1)
    if exo:
        return ploss, X, obs, gstar, torch.stack(ys, 1)
    return ploss, X, obs, gstar


def sim_loss_fn(net, obs, X, r, mode="self", Y=None):
    """read-only auxiliary prediction; targets detached.
    mode: self      = own FUTURE states x_{t+1..t+r} (original)
          shuffle   = another episode's future states (content-free control)
          past      = own PAST states x_{t-1..t-r} (self-referential, no
                      counterfactual turn-sign content — copyable from input)
          vel       = current (arriving) velocity direction x_t - x_{t-1}
                      (kinematic self-target, also input-derivable; use r=1)
          exo_future= exogenous stream's future y_{t+1..t+r} (learnable:
                      requires inferring the episode drift; ZERO consequence)
          exo_past  = exogenous stream's past y_{t-1..t-r} (record-keeping
                      demand matched to 'past', but about the non-self stream)"""
    B, L, _ = obs.shape
    if mode == "shuffle":
        X = torch.roll(X, 1, dims=0)         # another episode's trajectory
    h = net.trunk(obs.detach())                                              # (B,L,d) detached input
    pred = net.sim_head(h).view(B, L, r, 3)
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if mode == "vel":
        u = X[:, 1:L] - X[:, :L - 1]
        u = (u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)).detach()
        return (1 - (pred[:, 1:L, 0] * u).sum(-1)).mean()
    src = Y if mode in ("exo_future", "exo_past") else X
    past_like = mode in ("past", "exo_past")
    tot = 0.0; cnt = 0
    for k in range(1, r + 1):
        if k >= src.shape[1]:
            break
        if past_like:
            tgt = src[:, 0:L - k].detach()                                   # z_{t-k} for t=k..L-1
            p = pred[:, k:L, k - 1]
        elif mode == "exo_future":
            tgt = src[:, k:L].detach()                                       # y_{t+k} for t=0..L-1-k
            p = pred[:, :L - k, k - 1]
        else:
            valid = L - k                                                    # positions with a target
            tgt = src[:, 1 + k - 1: 1 + k - 1 + valid].detach()              # x_{t+k} for t=0..valid-1
            p = pred[:, :valid, k - 1]
        tot = tot + (1 - (p * tgt).sum(-1)).mean(); cnt += 1
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--cutoff", type=int, default=0)
    ap.add_argument("--r", type=int, default=4); ap.add_argument("--delta", type=float, default=0.06)
    ap.add_argument("--L", type=int, default=40); ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_layer", type=int, default=6); ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=400); ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux", default="self",
                    choices=["self", "shuffle", "past", "vel", "exo_future", "exo_past"])
    ap.add_argument("--exo", action="store_true",
                    help="append exogenous drifting-star channel y (obs dim 8)")
    ap.add_argument("--credit", default="full", choices=["full", "greedy"],
                    help="greedy = horizon-1 credit (state+obs detached between steps)")
    ap.add_argument("--pscale", type=float, default=1.0,
                    help="scale on ploss (e.g. 40 to match greedy aggregate magnitude to full BPTT)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    net = GoalNavNet(args.d_model, args.n_layer, args.n_head, args.L, args.r,
                     obs_dim=8 if args.exo else 5).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"goalnav cutoff={args.cutoff} {args.n_layer}L d{args.d_model} r{args.r} delta{args.delta} "
          f"L{args.L} | params={nparams/1e3:.0f}K", flush=True)
    sdir = os.path.join(BASE, args.out + "_steps"); os.makedirs(sdir, exist_ok=True)
    os.makedirs(os.path.join(BASE, os.path.dirname(args.out)), exist_ok=True)

    @torch.no_grad()
    def evaluate():
        net.eval()
        out = rollout(net, 512, args.L, dev, np.random.default_rng(7),
                      args.delta, args.cutoff, exo=args.exo)
        X, obs, gstar = out[:3]; Yv = out[3] if args.exo else None
        # final-quarter mean distance-to-goal (deg) and sim angular error (deg)
        dist = torch.arccos((X[:, 1:] * gstar[:, None]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
        reach = torch.rad2deg(dist[:, 3 * args.L // 4:].mean()).item()
        sl = sim_loss_fn(net, obs, X, args.r, mode=args.aux, Y=Yv)
        sim_deg = torch.rad2deg(torch.arccos((1 - sl).clamp(-1, 1))).item()
        net.train(); return reach, sim_deg

    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        net.train()
        if args.credit == "greedy":
            out = rollout_greedy(net, args.batch, args.L, dev, rng,
                                 args.delta, args.cutoff, exo=args.exo)
            ploss, X, obs, gstar = out[:4]; Yt = out[4] if args.exo else None
        else:
            out = rollout(net, args.batch, args.L, dev, rng, args.delta,
                          args.cutoff, exo=args.exo)
            X, obs, gstar = out[:3]; Yt = out[3] if args.exo else None
            ploss = (1 - (X[:, 1:] * gstar[:, None]).sum(-1)).mean()         # policy: reduce distance
        sloss = sim_loss_fn(net, obs, X, args.r, mode=args.aux, Y=Yt)
        loss = args.pscale * ploss + args.lam * sloss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == 1:
            reach, sim_deg = evaluate()
            log.append(dict(step=step, reach_deg=reach, sim_deg=sim_deg, ploss=float(ploss), sloss=float(sloss)))
            print(f"step {step:5d} | ploss {ploss:.4f} sloss {sloss:.4f} | reach_deg {reach:5.1f} sim_deg {sim_deg:5.1f}", flush=True)
        if step % args.ckpt_every == 0:
            torch.save({"state": net.state_dict(), "args": vars(args), "step": step}, os.path.join(sdir, f"step_{step:05d}.pt"))
    out = os.path.join(BASE, args.out)
    torch.save({"state": net.state_dict(), "args": vars(args), "nparams": nparams}, out + ".pt")
    json.dump(dict(args=vars(args), nparams=nparams, log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt (reach {log[-1]['reach_deg']:.1f} deg, sim {log[-1]['sim_deg']:.1f} deg)", flush=True)


if __name__ == "__main__":
    main()
