"""Continuous-action closed-loop self-model agent (smooth readout test).
Linear-Gaussian env so the optimal action is the POSTERIOR MEAN (a smooth/linear readout of the
belief, no hard argmax — the thing we hypothesised was blocking learning & making depth hurt):
  s_{t+1} = α s_t + a_t + w_t,  w~N(0,σw²)      (α<0 so closed loop a≈s is stable: coeff α+1)
  e_{t+1} = s_{t+1} + v_{t+1},  v~N(0,σv²)
  o_{t+1} = e_{t+1} + a_t                        (additive corruption -> efference copy: decode o−a)
  a_t = net's continuous output;  reward = −(a_t − s_t)²  ⇒  optimal a_t = E[s_t].
Net sees ONLY observations o. Floor = Kalman filter (a=μ). Prediction under the smoothing hypothesis:
the net now learns near the floor AND depth HELPS (reverses the discrete-argmax anti-scaling).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import Block

BASE = os.path.dirname(os.path.abspath(__file__))
ALPHA, SW, SV, S0 = -0.5, 0.5, 0.5, 1.0


class ContAgent(nn.Module):
    def __init__(self, d, nl, nh, L):
        super().__init__()
        self.in_proj = nn.Linear(1, d)
        self.pos = nn.Embedding(L + 2, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.act_head = nn.Linear(d, 1)
        self.obs_head = nn.Linear(d, 1)

    def forward(self, o):                                       # o (B,L) continuous
        B, L = o.shape
        x = self.in_proj(o.unsqueeze(-1)) + self.pos(torch.arange(L, device=o.device))[None]
        mask = torch.triu(torch.ones(L, L, device=o.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.lnf(x)
        return self.act_head(x).squeeze(-1), self.obs_head(x).squeeze(-1)


@torch.no_grad()
def rollout(net, B, L, dev):
    s = torch.randn(B, device=dev) * S0
    o = s + torch.randn(B, device=dev) * SV                     # o_0 = e_0 (a_{-1}=0)
    obs, states = [o], [s]
    for t in range(L - 1):
        a = net(torch.stack(obs, 1))[0][:, -1]
        s = (ALPHA * s + a + torch.randn(B, device=dev) * SW).clamp(-12, 12)
        o = s + torch.randn(B, device=dev) * SV + a
        obs.append(o); states.append(s)
    return torch.stack(obs, 1), torch.stack(states, 1)


def kalman_floor(B=4000, L=40):
    rng = np.random.default_rng(0)
    s = rng.standard_normal(B) * S0
    o = s + rng.standard_normal(B) * SV
    mu = np.zeros(B); P = np.full(B, S0 ** 2)
    K = P / (P + SV ** 2); mu = mu + K * (o - mu); P = (1 - K) * P   # update on o_0
    errs = []
    for t in range(L):
        a = mu
        errs.append((a - s) ** 2)
        s = ALPHA * s + a + rng.standard_normal(B) * SW
        o = s + rng.standard_normal(B) * SV + a
        mu = ALPHA * mu + a; P = ALPHA ** 2 * P + SW ** 2          # predict (uses known a)
        y = o - a                                                  # decode emission = s + v
        K = P / (P + SV ** 2); mu = mu + K * (y - mu); P = (1 - K) * P
    return float(np.mean(np.array(errs)[L // 2:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/cont"); ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4); ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--L", type=int, default=40); ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=250); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    net = ContAgent(args.d_model, args.n_layer, args.n_head, args.L).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    floor = kalman_floor()
    print(f"cont agent L={args.n_layer} d={args.d_model} | Kalman floor MSE={floor:.4f} | "
          f"params={sum(p.numel() for p in net.parameters())/1e3:.0f}K", flush=True)

    @torch.no_grad()
    def evaluate():
        net.eval(); obs, states = rollout(net, 1024, args.L, dev)
        a, _ = net(obs); tail = slice(args.L // 2, None)
        mse = ((a - states)[:, tail] ** 2).mean().item()
        net.train(); return mse

    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train(); obs, states = rollout(net, args.batch, args.L, dev)
        a, op = net(obs)
        aL = ((a - states) ** 2).mean()
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
