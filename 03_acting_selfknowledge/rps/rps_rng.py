"""RPS imperfect-monitoring WITH a private RNG: each round the net also sees a private uniform trit
r_t in {0,1,2} (NOT seen by the opponent). The net can now source randomness from r_t -- making its
action unpredictable to the opponent (random marginally) while keeping it PERFECTLY SELF-LEGIBLE (the
action is a deterministic function of the observed r_t, re-derivable later to decode the opponent).
HYPOTHESIS (self-legibility drives entropy collapse): given a legible randomness source, the net should
make its action DETERMINISTIC given r_t -> the CONDITIONAL output entropy H(p_t | r_t) collapses
EVERYWHERE, while the MARGINAL entropy H(E_r p_t) (what the opponent sees) stays as high as the game
needs. The opponent best-responds to the MARGINAL (it can't see r_t)."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/comp_icl")); from model import Block
BASE = os.path.dirname(os.path.abspath(__file__)); N = 3; GAMMA_BR = 6.0


class RPSNetRNG(nn.Module):
    def __init__(self, d, nl, nh, T):
        super().__init__()
        self.eo = nn.Embedding(4, d)        # outcomes 0/1/2 + start=3
        self.er = nn.Embedding(3, d)        # private trit
        self.pos = nn.Embedding(T + 2, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d); self.ah = nn.Linear(d, 3); self.vh = nn.Linear(d, 1)

    def forward(self, ot, rt):              # ot,rt: (B,L)
        L = ot.shape[1]
        x = self.eo(ot) + self.er(rt) + self.pos(torch.arange(L, device=ot.device))[None]
        m = torch.triu(torch.ones(L, L, device=ot.device, dtype=torch.bool), 1)
        for b in self.blocks: x = b(x, m)
        x = self.lnf(x); return self.ah(x), self.vh(x).squeeze(-1)


def rollout(net, B, T, dev, beta, bias):
    r_all = torch.randint(0, 3, (B, T), device=dev)         # private trits, known in advance
    out = [torch.full((B, 1), 3, device=dev, dtype=torch.long)]  # start token
    logps, Hcond, Hmarg, pays, vals = [], [], [], [], []
    for t in range(T):
        o_seq = torch.cat(out, 1)                            # (B,t+1) [start,o_0..o_{t-1}]
        r_seq = r_all[:, :t + 1]                             # (B,t+1) [r_0..r_t]
        # evaluate the policy for ALL 3 values of the CURRENT coin (history fixed) -> p_by_r
        o3 = o_seq.repeat(3, 1); r3 = r_seq.repeat(3, 1)
        r3[:B, -1] = 0; r3[B:2 * B, -1] = 1; r3[2 * B:, -1] = 2
        al, vl = net(o3, r3); p3 = F.softmax(al[:, -1], -1).view(3, B, 3); lp3 = F.log_softmax(al[:, -1], -1).view(3, B, 3)
        ar = r_all[:, t]                                     # actual coin
        idx = torch.arange(B, device=dev)
        p_t = p3[ar, idx]; lp_t = lp3[ar, idx]               # policy for the actual coin
        m_t = p3.mean(0)                                     # MARGINAL over the coin (opponent's view)
        val_t = vl[:, -1].view(3, B)[ar, idx]
        a = torch.multinomial(p_t, 1).squeeze(1)
        br = F.softmax(GAMMA_BR * m_t.detach()[:, [2, 0, 1]], -1)   # BR responds to the MARGINAL
        q = (1 - beta) * bias + beta * br; b = torch.multinomial(q, 1).squeeze(1)
        o = (a - b) % 3
        pays.append(torch.where(o == 1, 1.0, torch.where(o == 2, -1.0, 0.0)))
        logps.append(lp_t.gather(1, a[:, None]).squeeze(1)); vals.append(val_t)
        Hcond.append(-(p_t * lp_t).sum(-1)); Hmarg.append(-(m_t * (m_t + 1e-9).log()).sum(-1))
        out.append(o[:, None])
    st = lambda L: torch.stack(L, 1)
    return st(logps), st(vals), st(pays), st(Hcond), st(Hmarg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--T", type=int, default=30); ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layer", type=int, default=2); ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--init", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    net = RPSNetRNG(args.d_model, args.n_layer, args.n_head, args.T).to(dev)
    if args.init:
        net.load_state_dict(torch.load(os.path.expanduser(args.init), map_location=dev)["state"]); print(f"warm-start from {args.init}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    print(f"RPS-RNG beta={args.beta} | H_cond=output entropy GIVEN coin (predict COLLAPSE), "
          f"H_marg=opponent-visible entropy (uniform=1.099)", flush=True)
    def gen_bias(B):
        g = rng.gamma(0.5, 1.0, size=(B, 3))
        return torch.tensor(g / g.sum(1, keepdims=True), device=dev, dtype=torch.float32)
    log = []
    for step in range(1, args.steps + 1):
        net.train(); bias = gen_bias(args.batch)
        lp, val, pay, Hc, Hm = rollout(net, args.batch, args.T, dev, args.beta, bias)
        ret = pay.flip(1).cumsum(1).flip(1); adv = (ret - val).detach()
        loss = -(lp * adv).mean() + 0.5 * F.mse_loss(val, ret.detach())   # NO entropy bonus
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == 1:
            log.append(dict(step=step, H_cond=Hc.mean().item(), H_marg=Hm.mean().item(), payoff=pay.mean().item()))
            print(f"step {step:5d} | H_cond {Hc.mean():.3f} | H_marg {Hm.mean():.3f} | payoff/rd {pay.mean():+.3f}", flush=True)
    out = os.path.join(BASE, args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state": net.state_dict(), "args": vars(args)}, out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done -> {out}.pt | final H_cond {log[-1]['H_cond']:.3f} H_marg {log[-1]['H_marg']:.3f} payoff {log[-1]['payoff']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
