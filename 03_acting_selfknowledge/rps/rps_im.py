"""Repeated RPS (n=3) under IMPERFECT MONITORING: the net sees only the cyclic outcome
o_t=(a_t-b_t) mod 3 (0=tie,1=win,2=lose), NOT the opponent's move, and gets NO action feedback.
To model/exploit the opponent it must decode b_t=(a_t-o_t) mod 3 -- which needs its own action; but
it only has its action DISTRIBUTION p_t (it samples a_t and is never told the draw), so its belief
about b_t is p_t shifted by o_t, with uncertainty = its own action entropy. Hence: be random to dodge
exploitation vs. be self-legible to read the opponent.

Opponent: q_t = (1-beta)*hidden_bias + beta*BR(p_t). beta=0 -> static exploitable bias (be
deterministic & exploit); beta=1 -> pure best-responder (be uniform). Trained by REINFORCE, NO entropy
bonus, so the equilibrium entropy is emergent. Sweep beta, read entropy(beta).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import Block

BASE = os.path.dirname(os.path.abspath(__file__))
GAMMA_BR = 6.0          # opponent best-response sharpness


class RPSNet(nn.Module):
    def __init__(self, d, nl, nh, T):
        super().__init__()
        self.emb = nn.Embedding(4, d)            # outcomes 0/1/2 + start(3)
        self.pos = nn.Embedding(T + 2, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.act_head = nn.Linear(d, 3)
        self.val_head = nn.Linear(d, 1)

    def forward(self, tok):                       # tok (B,L) int
        L = tok.shape[1]
        x = self.emb(tok) + self.pos(torch.arange(L, device=tok.device))[None]
        mask = torch.triu(torch.ones(L, L, device=tok.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.lnf(x)
        return self.act_head(x), self.val_head(x).squeeze(-1)


def rollout(net, B, T, dev, beta, bias, per_traj=False, is_br=None, full_obs=False, noisy_obs=0.0):
    """sequential closed-loop rollout; returns per-step logp, value, payoff, entropy (all (B,T)).
    per_traj=False: opponent move each turn ~ (1-beta)*bias + beta*BR (per-TURN blend).
    per_traj=True : opponent commits PER TRAJECTORY to pure-BR (mask is_br) else pure-bias.
    full_obs=True : the net observes the opponent's MOVE b_t directly (perfect monitoring) instead of
                    the pooled outcome o_t -- it can then identify the opponent without lowering entropy."""
    seq = torch.full((B, 1), 3, device=dev, dtype=torch.long)     # start token
    logps, vals, pays, ents = [], [], [], []
    for t in range(T):
        al, vl = net(seq)
        logits = al[:, -1]; val = vl[:, -1]
        p = F.softmax(logits, -1)                                 # (B,3)
        logp_all = F.log_softmax(logits, -1)
        a = torch.multinomial(p, 1).squeeze(1)                    # sample action
        # opponent: best-response to p (detached) vs hidden bias
        winprob = p.detach()[:, [2, 0, 1]]                        # winprob(b) = p[(b-1)%3]
        br = F.softmax(GAMMA_BR * winprob, -1)
        if per_traj:
            q = torch.where(is_br[:, None], br, bias)             # coherent type for whole trajectory
        else:
            q = (1 - beta) * bias + beta * br
        b = torch.multinomial(q, 1).squeeze(1)
        o = (a - b) % 3
        pay = torch.where(o == 1, 1.0, torch.where(o == 2, -1.0, 0.0))
        logps.append(logp_all.gather(1, a[:, None]).squeeze(1))
        vals.append(val); pays.append(pay)
        ents.append(-(p * logp_all).sum(-1))
        if full_obs:
            nxt = b                                               # perfect monitoring: see b_t directly
        elif noisy_obs > 0:                                       # INDEPENDENT noise: see b_t w.p. noisy_obs else random
            keep = torch.rand(B, device=dev) < noisy_obs          # noise NOT coupled to the agent's entropy
            nxt = torch.where(keep, b, torch.randint(0, 3, (B,), device=dev))
        else:
            nxt = o                                               # imperfect monitoring: pooled outcome (coupled)
        seq = torch.cat([seq, nxt[:, None]], 1)
    return (torch.stack(logps, 1), torch.stack(vals, 1),
            torch.stack(pays, 1), torch.stack(ents, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--T", type=int, default=40); ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layer", type=int, default=2); ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per_traj", action="store_true", help="opponent commits to pure type per trajectory")
    ap.add_argument("--full_obs", action="store_true", help="net observes opponent move b_t directly")
    ap.add_argument("--noisy_obs", type=float, default=0.0, help="see b_t w.p. RHO else random (noise INDEPENDENT of action)")
    ap.add_argument("--init", default="", help="checkpoint .pt to warm-start weights from (curriculum)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    net = RPSNet(args.d_model, args.n_layer, args.n_head, args.T).to(dev)
    if args.init:
        net.load_state_dict(torch.load(os.path.expanduser(args.init), map_location=dev)["state"])
        print(f"warm-started from {args.init}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    print(f"RPS-IM beta={args.beta} {args.n_layer}L d{args.d_model} T={args.T} | "
          f"uniform-entropy={np.log(3):.3f}", flush=True)
    sdir = os.path.join(BASE, args.out + "_steps"); os.makedirs(sdir, exist_ok=True)
    os.makedirs(os.path.join(BASE, os.path.dirname(args.out)), exist_ok=True)

    def gen_bias(B):
        g = rng.gamma(0.5, 1.0, size=(B, 3))                      # Dirichlet(0.5): varied, often peaked
        return torch.tensor(g / g.sum(1, keepdims=True), device=dev, dtype=torch.float32)

    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train()
        bias = gen_bias(args.batch)
        is_br = (torch.rand(args.batch, device=dev) < args.beta) if args.per_traj else None
        logp, val, pay, ent = rollout(net, args.batch, args.T, dev, args.beta, bias, args.per_traj, is_br, args.full_obs, args.noisy_obs)
        ret = pay.flip(1).cumsum(1).flip(1)                       # future return R_t
        adv = (ret - val).detach()
        pg = -(logp * adv).mean()
        vloss = F.mse_loss(val, ret.detach())
        loss = pg + 0.5 * vloss                                   # NO entropy bonus
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == 1:
            rec = dict(step=step, entropy=ent.mean().item(), payoff=pay.mean().item(),
                       ent_early=ent[:, :args.T // 4].mean().item(),
                       ent_late=ent[:, 3 * args.T // 4:].mean().item())
            if args.per_traj:   # split entropy by opponent type (the in-context inference test)
                rec["ent_bias"] = ent[~is_br].mean().item() if (~is_br).any() else float("nan")
                rec["ent_br"] = ent[is_br].mean().item() if is_br.any() else float("nan")
            log.append(rec)
            extra = (f" | bias-traj {rec['ent_bias']:.2f} BR-traj {rec['ent_br']:.2f}" if args.per_traj else "")
            print(f"step {step:5d} | entropy {ent.mean():.3f} (early {rec['ent_early']:.2f} "
                  f"late {rec['ent_late']:.2f}) | payoff/round {pay.mean():+.3f}{extra}", flush=True)
        if step % args.ckpt_every == 0:
            torch.save({"state": net.state_dict(), "args": vars(args), "step": step,
                        "entropy": log[-1]["entropy"]}, os.path.join(sdir, f"step_{step:05d}.pt"))
    out = os.path.join(BASE, args.out)
    torch.save({"state": net.state_dict(), "args": vars(args)}, out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt | final entropy {log[-1]['entropy']:.3f} "
          f"payoff {log[-1]['payoff']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
