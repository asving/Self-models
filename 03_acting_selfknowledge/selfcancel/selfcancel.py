"""Self-cancellation forecasting (continuous, single-output, fully differentiable).

A hidden latent does a random walk:  e_{t+1} = e_t + eta_t,  eta ~ N(0, sigma_eta^2).
The net reads the observation history and emits ONE Gaussian N(mu_t, sigma_t^2). That single
distribution is BOTH (i) its forecast of the NEXT latent e_{t+1} (scored by proper log-score) and
(ii) its action a_t ~ N(mu_t, sigma_t^2). The action pools into the next observation:
    x_{t+1} = e_{t+1} + g * a_t .
The net never sees e or the realized a_t -- only the x's. To read e_{t+1} from x_{t+1} it subtracts
its own MEAN g*mu_t, leaving e_{t+1} + g*sigma_t*eps_t: the sample-noise g*sigma_t is the part it
CANNOT cancel. So its own reported spread sigma_t is the measurement noise on its own future
perception. Honest forecasting of e_{t+1} needs sigma_t >= sigma_eta (can't predict the innovation),
but legibility wants sigma_t SMALLER -> persistent self-induced OVER-CONFIDENCE: sigma_t below the
Kalman predictive std. Tunable by the pooling gain g (g=0 -> honest control).

Everything is reparameterized (a_t = mu_t + sigma_t*eps) so the whole rollout is differentiable:
plain backprop-through-time on the negative log-score. No REINFORCE."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/comp_icl")); from model import Block
BASE = os.path.dirname(os.path.abspath(__file__))


class RWForecastNet(nn.Module):
    def __init__(self, d, nl, nh, T):
        super().__init__()
        self.inp = nn.Linear(1, d)                       # embed scalar observation
        self.start = nn.Parameter(torch.zeros(1, 1, d))  # learned start token
        self.pos = nn.Embedding(T + 2, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.mu_head = nn.Linear(d, 1)
        self.ls_head = nn.Linear(d, 1)                   # log sigma

    def forward(self, obs):                              # obs: (B, t) scalars (may be empty)
        B = obs.shape[0]
        tok = self.inp(obs[..., None]) if obs.shape[1] > 0 else obs.new_zeros(B, 0, self.start.shape[-1])
        x = torch.cat([self.start.expand(B, -1, -1), tok], 1)        # prepend start
        L = x.shape[1]
        x = x + self.pos(torch.arange(L, device=obs.device))[None]
        m = torch.triu(torch.ones(L, L, device=obs.device, dtype=torch.bool), 1)
        for b in self.blocks: x = b(x, m)
        x = self.lnf(x[:, -1])
        return self.mu_head(x).squeeze(-1), self.ls_head(x).squeeze(-1)   # mu_t, log sigma_t


def rollout(net, B, T, dev, g, sigma_eta, sigma0=1.0, ls_min=-7.0, ls_max=3.0):
    e = torch.randn(B, device=dev) * sigma0                          # e_0
    obs = torch.zeros(B, 0, device=dev)
    scores, mus, sigmas, es, meas_noise = [], [], [], [], []
    for t in range(T):
        mu, ls = net(obs)
        sig = torch.exp(ls.clamp(ls_min, ls_max))
        e = e + torch.randn(B, device=dev) * sigma_eta               # e_{t+1} = e_t + eta_t
        eps = torch.randn(B, device=dev)
        a = mu + sig * eps                                           # reparameterized action = sampled forecast
        x = e + g * a                                               # pooled observation of e_{t+1}
        # proper log-score of e_{t+1} under N(mu, sig^2)
        r = -0.5 * (((e - mu) / sig) ** 2 + 2 * torch.log(sig) + np.log(2 * np.pi))
        scores.append(r); mus.append(mu); sigmas.append(sig); es.append(e); meas_noise.append((g * sig).detach())
        obs = torch.cat([obs, x[:, None]], 1)
    st = lambda L: torch.stack(L, 1)
    return st(scores), st(mus), st(sigmas), st(es)


@torch.no_grad()
def kalman_honest(net, B, T, dev, g, sigma_eta, sigma0=1.0):
    """Bayes-honest predictive std for e_{t+1}, GIVEN the net's own realized measurement-noise (g*sigma_t).
    Random-walk Kalman with time-varying R_t=(g*sigma_t)^2, Q=sigma_eta^2. Returns mean honest sigma per round
    and the net's mean sigma per round -> overconfidence = net below honest."""
    e = torch.randn(B, device=dev) * sigma0
    obs = torch.zeros(B, 0, device=dev)
    P = torch.full((B,), sigma0 ** 2, device=dev)                    # posterior var of current e (start prior)
    net_sig, hon_sig = [], []
    for t in range(T):
        mu, ls = net(obs); sig = torch.exp(ls.clamp(-7, 3))
        # honest predictive std for e_{t+1} = sqrt(P_post + Q): can't beat this without cheating
        pred_var = P + sigma_eta ** 2
        hon_sig.append(pred_var.sqrt().mean().item()); net_sig.append(sig.mean().item())
        e = e + torch.randn(B, device=dev) * sigma_eta
        eps = torch.randn(B, device=dev); a = mu + sig * eps; x = e + g * a
        # Kalman measurement update of e_{t+1} from x (measurement noise R=(g*sig)^2; H=1, after subtracting g*mu)
        R = (g * sig) ** 2 + 1e-8
        z = x - g * mu                                              # = e_{t+1} + g*sig*eps
        K = pred_var / (pred_var + R)
        P = (1 - K) * pred_var
        obs = torch.cat([obs, x[:, None]], 1)
    return net_sig, hon_sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--sigma_eta", type=float, default=0.5); ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_every", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(args.seed)
    net = RWForecastNet(args.d_model, args.n_layer, args.n_head, args.T).to(dev)
    if args.init:
        net.load_state_dict(torch.load(os.path.expanduser(args.init), map_location=dev)["state"]); print(f"warm-start {args.init}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    print(f"SELF-CANCEL g={args.g} sigma_eta={args.sigma_eta} {args.n_layer}L/d{args.d_model} T={args.T} "
          f"| over-confidence = net sigma BELOW honest(Kalman) sigma", flush=True)
    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train()
        scores, mus, sigmas, es = rollout(net, args.batch, args.T, dev, args.g, args.sigma_eta)
        loss = -scores.mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == 1:
            net.eval()
            with torch.no_grad():
                sc, mu, sg, ee = rollout(net, 2048, args.T, dev, args.g, args.sigma_eta)
                ns, hs = kalman_honest(net, 2048, args.T, dev, args.g, args.sigma_eta)
                # mean over the back half (steady state)
                half = args.T // 2
                ns_ss = float(np.mean(ns[half:])); hs_ss = float(np.mean(hs[half:]))
                rec = dict(step=step, score=sc.mean().item(), sigma_net=sg.mean().item(),
                           sigma_net_ss=ns_ss, sigma_honest_ss=hs_ss, overconf_ratio=ns_ss / hs_ss,
                           sigma_by_round=[round(x, 4) for x in sg.mean(0).tolist()],
                           honest_by_round=[round(x, 4) for x in hs])
            log.append(rec)
            print(f"step {step:5d} | score {rec['score']:+.3f} | net_sigma_ss {ns_ss:.3f} "
                  f"honest_ss {hs_ss:.3f} | ratio {rec['overconf_ratio']:.3f} "
                  f"({'OVERCONFIDENT' if rec['overconf_ratio']<0.95 else 'honest-ish'})", flush=True)
    out = os.path.join(BASE, args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state": net.state_dict(), "args": vars(args)}, out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt | final score {log[-1]['score']:+.3f} "
          f"overconf ratio {log[-1]['overconf_ratio']:.3f}", flush=True)


if __name__ == "__main__":
    main()
