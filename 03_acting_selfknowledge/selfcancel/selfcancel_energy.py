"""Self-cancellation task scored by the ENERGY SCORE (CRPS) instead of the Gaussian log-score.
Same setup otherwise (random-walk latent, single Gaussian output (mu,sigma) sampled for the action,
action pools into the observation x=e+g*a, NO random bits). The energy score is a proper scoring rule
for SAMPLES: ES = 0.5(|a-e|+|a'-e|) - 0.5|a-a'|, estimated per step from two reparameterized draws.
Its minimizer is the true predictive distribution (same as log-score), but the spread is driven by the
repulsion term rather than a density penalty. Question (exploratory): does the net still over-collapse
sigma below the honest (Kalman) std for self-legibility, like under the log-score?"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from selfcancel import RWForecastNet, kalman_honest          # reuse net + Kalman-honest baseline
BASE = os.path.dirname(os.path.abspath(__file__))


def rollout(net, B, T, dev, g, sigma_eta, sigma0=1.0):
    e = torch.randn(B, device=dev) * sigma0
    obs = torch.zeros(B, 0, device=dev)
    scores, mus, sigmas, es = [], [], [], []
    for t in range(T):
        mu, ls = net(obs); sig = torch.exp(ls.clamp(-7, 3))
        e = e + torch.randn(B, device=dev) * sigma_eta
        a = mu + sig * torch.randn(B, device=dev)             # action (pools into the observation)
        ap = mu + sig * torch.randn(B, device=dev)            # second draw for the energy-score repulsion term
        x = e + g * a
        ES = 0.5 * ((a - e).abs() + (ap - e).abs()) - 0.5 * (a - ap).abs()
        scores.append(ES); mus.append(mu); sigmas.append(sig); es.append(e)
        obs = torch.cat([obs, x[:, None]], 1)
    st = lambda L: torch.stack(L, 1)
    return st(scores), st(mus), st(sigmas), st(es)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--sigma_eta", type=float, default=0.5); ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_every", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(args.seed)
    net = RWForecastNet(args.d_model, args.n_layer, args.n_head, args.T).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    print(f"SELF-CANCEL ENERGY-SCORE g={args.g} sigma_eta={args.sigma_eta} {args.n_layer}L/d{args.d_model} | "
          f"does it still over-collapse sigma vs honest?", flush=True)
    half = args.T // 2; log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train()
        sc, mu, sg, ee = rollout(net, args.batch, args.T, dev, args.g, args.sigma_eta)
        loss = sc.mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == 1:
            net.eval()
            with torch.no_grad():
                ns, hs = kalman_honest(net, 2048, args.T, dev, args.g, args.sigma_eta)
                ns_ss = float(np.mean(ns[half:])); hs_ss = float(np.mean(hs[half:]))
                rec = dict(step=step, score=float(sc.mean()), sigma_net_ss=ns_ss, sigma_honest_ss=hs_ss,
                           overconf_ratio=ns_ss / hs_ss, sigma_by_round=[round(x, 4) for x in sg.mean(0).tolist()],
                           honest_by_round=[round(x, 4) for x in hs])
            log.append(rec)
            print(f"step {step:5d} | ES {rec['score']:+.3f} | net_sigma {ns_ss:.3f} honest {hs_ss:.3f} "
                  f"ratio {rec['overconf_ratio']:.3f} "
                  f"({'OVERCONFIDENT' if rec['overconf_ratio']<0.95 else 'honest-ish'})", flush=True)
    out = os.path.join(BASE, args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state": net.state_dict(), "args": vars(args)}, out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt | ratio {log[-1]['overconf_ratio']:.3f}", flush=True)


if __name__ == "__main__":
    main()
