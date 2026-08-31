"""Forecasting game under imperfect monitoring -- the 'concave' control that decouples EPISTEMIC
sharpening from exploitation. The net outputs ONE distribution p_t that is BOTH its forecast and its
move: it samples a move a_t~p_t (which pools into the observation o_t=(a_t-b_t)%n, imperfect monitoring)
and is SCORED by the proper log-score r_t=log p_t(b_t). Optimal forecast = the opponent's distribution
q (drawn per episode, MIXED), entropy H(q). To LEARN q it must decode b from o, which needs a SHARP
(self-legible) p_t -- but a sharp p_t is a deliberately-wrong forecast. So the clean epistemic signature
is: entropy(p_t) DIPS BELOW H(q) while identifying (over-sharpening to gather info), then relaxes up to
H(q). Honest forecasting only ever approaches H(q) from ABOVE, so any dip below H(q) is unambiguously
epistemic (NOT exploitation). full_obs control: net sees b_t directly -> no dip below H(q).

Training: maximize sum_t log p_t(b_t). Pathwise gradient for the immediate forecast (p_t given history)
+ REINFORCE crediting each action a_t with the FUTURE forecast reward (its information value)."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from rps_im import RPSNet

BASE = os.path.dirname(os.path.abspath(__file__))
N = 3


def rollout(net, B, T, dev, q, full_obs=False, noisy_obs=0.0, force_sharp=False):
    seq = torch.full((B, 1), N, dtype=torch.long, device=dev)        # start token = N
    flog, alog, vals, ents = [], [], [], []
    for t in range(T):
        al, vl = net(seq); logp = F.log_softmax(al[:, -1], -1); p = logp.exp()
        # force_sharp (eval only): play argmax so observations are self-legible -> measures the
        # CAPABILITY to decode+forecast if sharp, independent of whether the net CHOOSES to sharpen.
        a = p.argmax(-1) if force_sharp else torch.multinomial(p, 1).squeeze(1)
        b = torch.multinomial(q, 1).squeeze(1)                       # opponent ~ fixed mixed bias q
        flog.append(logp.gather(1, b[:, None]).squeeze(1))           # proper score: log p_t(b_t)
        alog.append(logp.gather(1, a[:, None]).squeeze(1))           # action log-prob (for REINFORCE)
        vals.append(vl[:, -1]); ents.append(-(p * logp).sum(-1))
        if full_obs:
            nxt = b
        elif noisy_obs > 0:
            keep = torch.rand(B, device=dev) < noisy_obs
            nxt = torch.where(keep, b, torch.randint(0, N, (B,), device=dev))
        else:
            nxt = (a - b) % N                                        # imperfect monitoring (coupled)
        seq = torch.cat([seq, nxt[:, None]], 1)
    return (torch.stack(flog, 1), torch.stack(alog, 1),
            torch.stack(vals, 1), torch.stack(ents, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--full_obs", action="store_true"); ap.add_argument("--noisy_obs", type=float, default=0.0)
    ap.add_argument("--T", type=int, default=40); ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layer", type=int, default=2); ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    net = RPSNet(args.d_model, args.n_layer, args.n_head, args.T).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    print(f"FORECAST alpha={args.alpha} full_obs={args.full_obs} noisy={args.noisy_obs} {args.n_layer}L "
          f"T={args.T} | (epistemic signature = entropy dips BELOW H(q))", flush=True)
    sdir = os.path.join(BASE, args.out + "_steps"); os.makedirs(sdir, exist_ok=True)
    os.makedirs(os.path.join(BASE, os.path.dirname(args.out)), exist_ok=True)

    def gen_q(B):
        g = rng.gamma(args.alpha, 1.0, size=(B, N)); return torch.tensor(g / g.sum(1, keepdims=True), device=dev, dtype=torch.float32)

    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train(); q = gen_q(args.batch)
        flog, alog, val, ent = rollout(net, args.batch, args.T, dev, q, args.full_obs, args.noisy_obs)
        ret = flog.flip(1).cumsum(1).flip(1)                         # R_t = sum_{s>=t} log p_s(b_s)
        ret_fut = torch.cat([ret[:, 1:], torch.zeros(args.batch, 1, device=dev)], 1)   # future (a_t affects s>t)
        adv = (ret_fut - val).detach()
        forecast_loss = -flog.mean()                                 # pathwise: immediate forecast given history
        reinforce_loss = -(alog * adv).mean()                        # REINFORCE: action's information value
        vloss = F.mse_loss(val, ret_fut.detach())
        loss = forecast_loss + reinforce_loss + 0.5 * vloss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == 1:
            with torch.no_grad():
                qe = gen_q(1024); fl, _, _, en = rollout(net, 1024, args.T, dev, qe, args.full_obs, args.noisy_obs)
                # CAPABILITY probe: same games but FORCE sharp actions -> how well it forecasts if legible
                fl_sharp, _, _, _ = rollout(net, 1024, args.T, dev, qe, args.full_obs, args.noisy_obs, force_sharp=True)
                Hq = -(qe * (qe + 1e-9).log()).sum(-1)               # (1024,)
                ent_by_t = en.mean(0)                                # (T,) entropy trajectory
                dip = (ent_by_t - Hq.mean()).min().item()            # most-below-H(q) (negative = epistemic dip)
                rec = dict(step=step, Hq=Hq.mean().item(), score=fl.mean().item(),
                           score_sharp=fl_sharp.mean().item(),       # CAPABILITY (forced-sharp forecast score)
                           ent_traj=[round(x, 3) for x in ent_by_t.tolist()],
                           min_minus_Hq=dip, ent_mean=en.mean().item())
            log.append(rec)
            print(f"step {step:5d} | H(q)={rec['Hq']:.3f} | mean-ent={rec['ent_mean']:.3f} | "
                  f"score={rec['score']:.3f} (opt -H(q)={-rec['Hq']:.3f}) | min(ent_t - H(q))={dip:+.3f}", flush=True)
        if step % args.ckpt_every == 0:
            torch.save({"state": net.state_dict(), "args": vars(args), "step": step}, os.path.join(sdir, f"step_{step:05d}.pt"))
    out = os.path.join(BASE, args.out)
    torch.save({"state": net.state_dict(), "args": vars(args)}, out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt | final min(ent_t-H(q))={log[-1]['min_minus_Hq']:+.3f} "
          f"(negative => epistemic dip below H(q))", flush=True)


if __name__ == "__main__":
    main()
