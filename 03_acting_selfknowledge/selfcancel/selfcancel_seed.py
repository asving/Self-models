"""Self-cancellation with a LEGIBLE RANDOMNESS SOURCE (continuous analog of the RPS trit/RNG test) --
NO efference copy (that would trivialize it). The net is given a stream of random bits r_t and its
action sources its randomness from them:  a_t = mu_t + sigma_t * r_t  (r_t ~ N(0,1)). The seed r_{t-1}
that went into the action contaminating x_t is fed back paired with x_t -- but NOT the action itself.

So to cancel its own contribution and read the latent, the net must  e_t = x_t - g*(mu_{t-1}+sigma_{t-1}*r_{t-1})
i.e. RECOMPUTE its own past output (mu_{t-1},sigma_{t-1}) from context and combine it with the given bit.
That recompute is the self-model. If the net learns it, it can keep an HONEST fully-spread sigma AND
perceive perfectly -> over-confidence DISAPPEARS (sigma->sigma_eta, ratio->1). If it can't (the RPS-RNG
failure mode), the bits go unused and it stays over-confident. The net never sees its action, only the bit.
Single output, differentiable, no REINFORCE. mu_t,sigma_t are committed BEFORE r_t is revealed (next step),
so the forecast marginal stays N(mu,sigma)."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.expanduser("~/comp_icl")); from model import Block
BASE = os.path.dirname(os.path.abspath(__file__))


class RWForecastNetSeed(nn.Module):
    def __init__(self, d, nl, nh, T):
        super().__init__()
        self.inp = nn.Linear(2, d)                       # [x_t, r_{t-1}]: observation + the bit baked into it
        self.start = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Embedding(T + 2, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d); self.mu_head = nn.Linear(d, 1); self.ls_head = nn.Linear(d, 1)

    def forward(self, obs):                              # obs: (B, t, 2)
        B = obs.shape[0]
        tok = self.inp(obs) if obs.shape[1] > 0 else obs.new_zeros(B, 0, self.start.shape[-1])
        x = torch.cat([self.start.expand(B, -1, -1), tok], 1); L = x.shape[1]
        x = x + self.pos(torch.arange(L, device=obs.device))[None]
        m = torch.triu(torch.ones(L, L, device=obs.device, dtype=torch.bool), 1)
        for b in self.blocks: x = b(x, m)
        x = self.lnf(x[:, -1])
        return self.mu_head(x).squeeze(-1), self.ls_head(x).squeeze(-1)


def rollout(net, B, T, dev, g, sigma_eta, sigma0=1.0):
    e = torch.randn(B, device=dev) * sigma0
    obs = torch.zeros(B, 0, 2, device=dev)
    scores, mus, sigmas, es = [], [], [], []
    for t in range(T):
        mu, ls = net(obs); sig = torch.exp(ls.clamp(-7, 3))         # committed BEFORE seeing r_t
        r = torch.randn(B, device=dev)                             # the random bit
        a = mu + sig * r                                           # action sources randomness from the bit
        e = e + torch.randn(B, device=dev) * sigma_eta
        x = e + g * a
        scores.append(-0.5 * (((e - mu) / sig) ** 2 + 2 * torch.log(sig) + np.log(2 * np.pi)))
        mus.append(mu); sigmas.append(sig); es.append(e)
        tok = torch.stack([x, r], -1)                             # feed back [x_{t+1}, r_t] -- the BIT, not the action
        obs = torch.cat([obs, tok[:, None]], 1)
    st = lambda L: torch.stack(L, 1)
    return st(scores), st(mus), st(sigmas), st(es)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--g", type=float, default=2.0)
    ap.add_argument("--sigma_eta", type=float, default=0.5); ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=4); ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_every", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(args.seed)
    net = RWForecastNetSeed(args.d_model, args.n_layer, args.n_head, args.T).to(dev)
    if args.init:
        net.load_state_dict(torch.load(os.path.expanduser(args.init), map_location=dev)["state"]); print(f"warm-start {args.init}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    print(f"SELF-CANCEL+RANDOM-BITS g={args.g} sigma_eta={args.sigma_eta} {args.n_layer}L/d{args.d_model} | "
          f"net must RECOMPUTE its own action to use the bit. SUCCESS = sigma->{args.sigma_eta}, ratio->1, RMSE->{args.sigma_eta}", flush=True)
    half = args.T // 2; log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train()
        sc, mu, sg, ee = rollout(net, args.batch, args.T, dev, args.g, args.sigma_eta)
        loss = -sc.mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == 1:
            net.eval()
            with torch.no_grad():
                sc, mu, sg, ee = rollout(net, 2048, args.T, dev, args.g, args.sigma_eta)
                sig_ss = float(sg[:, half:].mean()); rmse_ss = float(((mu - ee)[:, half:] ** 2).mean().sqrt())
                rec = dict(step=step, score=float(sc.mean()), sigma_ss=sig_ss, rmse_ss=rmse_ss,
                           ratio_vs_floor=sig_ss / args.sigma_eta, rmse_vs_floor=rmse_ss / args.sigma_eta)
            log.append(rec)
            print(f"step {step:5d} | score {rec['score']:+.3f} | sigma_ss {sig_ss:.3f} (floor {args.sigma_eta}) "
                  f"ratio {rec['ratio_vs_floor']:.3f} | RMSE {rmse_ss:.3f} "
                  f"({'USES BITS (honest+legible)' if rec['ratio_vs_floor']<1.15 and rec['rmse_vs_floor']<1.4 else 'bits unused'})", flush=True)
    out = os.path.join(BASE, args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state": net.state_dict(), "args": vars(args)}, out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt | sigma_ss {log[-1]['sigma_ss']:.3f} "
          f"ratio {log[-1]['ratio_vs_floor']:.3f} RMSE {log[-1]['rmse_ss']:.3f}", flush=True)


if __name__ == "__main__":
    main()
