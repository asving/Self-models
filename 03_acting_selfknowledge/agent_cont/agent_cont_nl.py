"""Nonlinear-READOUT continuous-action self-model (the missing 'smooth AND nonlinear' cell).
Same efference-copy structure (o_{t+1}=s_{t+1}+v+a_t, closed loop, observations-only), but the action
is rewarded to track a BOUNDED NONLINEAR function of the state: reward −(a_t − g(s_t))², g=tanh(γ·s).
⇒ optimal a_t = E[g(s_t)|belief] = a NONLINEAR functional of the belief. Because a_t enters the belief
recursion (de-corrupting o_{t+1} needs a_t=f(b_t)), the recursion b_{t+1}=G(b_t, f(b_t), o) is NONLINEAR
in the belief — no linear-scan shortcut — so computing b_t from observations is a deep nested
composition and the (position×layer) diagonal march should make DEPTH the resource (unlike the linear
posterior-mean readout, which made the recursion a parallelizable scan ⇒ flat depth).
Bounded action (tanh head) keeps the closed loop stable; high α + noisy obs give long recursion memory.
Belief stays Gaussian (action is a known input given the belief) ⇒ exact Kalman+g-expectation floor.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from agent_cont import ContAgent

BASE = os.path.dirname(os.path.abspath(__file__))
ALPHA, SW, SV, S0, GAIN = 0.95, 0.3, 1.0, 1.0, 2.0     # high persistence + noisy obs -> long memory (mem~0.77)
_hx, _hw = np.polynomial.hermite.hermgauss(20)          # Gauss-Hermite for E[g(s)|N(mu,P)]


def g_np(s): return np.tanh(GAIN * s)
def g_t(s): return torch.tanh(GAIN * s)


def Eg(mu, P):                                           # E[tanh(γ s)] for s~N(mu,P), vectorized
    s = mu[..., None] + np.sqrt(2 * np.clip(P, 1e-9, None))[..., None] * _hx
    return (np.tanh(GAIN * s) * _hw).sum(-1) / np.sqrt(np.pi)


@torch.no_grad()
def rollout(net, B, L, dev):
    s = torch.randn(B, device=dev) * S0
    o = s + torch.randn(B, device=dev) * SV
    obs, states = [o], [s]
    for t in range(L - 1):
        a = torch.tanh(net(torch.stack(obs, 1))[0][:, -1])      # bounded action
        s = ALPHA * s + torch.randn(B, device=dev) * SW          # OPEN-LOOP transition (state stays bounded)
        o = s + torch.randn(B, device=dev) * SV + a              # action corrupts obs (efference copy)
        obs.append(o); states.append(s)
    return torch.stack(obs, 1), torch.stack(states, 1)


def kalman_floor(B=4000, L=40, ret_mem=False):
    rng = np.random.default_rng(0)
    s = rng.standard_normal(B) * S0
    o = s + rng.standard_normal(B) * SV
    mu = np.zeros(B); P = np.full(B, S0 ** 2)
    K = P / (P + SV ** 2); mu = mu + K * (o - mu); P = (1 - K) * P
    errs = []
    for t in range(L):
        a = Eg(mu, P)
        errs.append((a - g_np(s)) ** 2)
        s = ALPHA * s + rng.standard_normal(B) * SW              # OPEN-LOOP transition
        o = s + rng.standard_normal(B) * SV + a
        mu = ALPHA * mu; P = ALPHA ** 2 * P + SW ** 2
        y = o - a
        K = P / (P + SV ** 2); mu = mu + K * (y - mu); P = (1 - K) * P
    floor = float(np.mean(np.array(errs)[L // 2:]))
    if ret_mem:
        Pss = P.mean(); Kss = Pss / (Pss + SV ** 2)
        return floor, (1 - Kss) * ALPHA, float(np.std(s))            # effective-memory factor, state std
    return floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/nl"); ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4); ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--L", type=int, default=40); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    net = ContAgent(args.d_model, args.n_layer, args.n_head, args.L).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    floor, mem, sstd = kalman_floor(ret_mem=True)
    print(f"NL-readout L={args.n_layer} d={args.d_model} | floor MSE={floor:.4f} | "
          f"mem-factor={mem:.2f} state-std={sstd:.2f} | params={sum(p.numel() for p in net.parameters())/1e3:.0f}K", flush=True)

    @torch.no_grad()
    def evaluate():
        net.eval(); obs, states = rollout(net, 1024, args.L, dev)
        a = torch.tanh(net(obs)[0]); tail = slice(args.L // 2, None)
        mse = ((a - g_t(states))[:, tail] ** 2).mean().item()
        net.train(); return mse

    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train(); obs, states = rollout(net, args.batch, args.L, dev)
        ra, op = net(obs); a = torch.tanh(ra)
        aL = ((a - g_t(states)) ** 2).mean()
        oL = ((op[:, :-1] - obs[:, 1:]) ** 2).mean()
        loss = aL + oL
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            mse = evaluate(); log.append(dict(step=step, action_mse=mse, action_loss=float(aL), obs_loss=float(oL)))
            print(f"step {step:5d} | aL {aL:.4f} oL {oL:.4f} | act_MSE {mse:.4f}  (floor {floor:.4f}, "
                  f"excess {mse-floor:+.4f})", flush=True)

    out = os.path.join(BASE, args.out)
    torch.save({"state": net.state_dict(), "args": vars(args), "floor": floor}, out + ".pt")
    json.dump(dict(args=vars(args), floor=floor, log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt", flush=True)


if __name__ == "__main__":
    main()
