"""Closed-loop controlled-POMDP self-model agent (observations-only).
Env (3-state ring): s_{t+1}=(j+a_t)%3 with j~T0(·|s_t); e_{t+1}~E(·|s_{t+1}); o_{t+1}=(e_{t+1}+a_t)%3.
The net sees ONLY observations o_0..o_t and outputs a soft action p_t (softmax) + a next-obs prediction.
Action a_t~p_t is sampled by the env; the net is NOT told the realized a_t, so to track s it must
keep an INTERNAL trace of its own past action distributions (efference copy) and decode with them.
Losses (on-policy rollout): action = -E[match] = -p_t(s_t) (0/1 match reward, drives sharpening to
MAP); next-obs = -log P_obs(o_{t+1}) (perception: a soft action convolves the obs toward uninformative,
so this also pushes sharpening). No hard MAP supervision -> sharpening is emergent. Observations-only,
deterministic policy (no sampling in the *policy*; the softmax CAN stay soft, and we measure if it does).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT

BASE = os.path.dirname(os.path.abspath(__file__))
Q = 3
PI = np.array([1 / 3, 1 / 3, 1 / 3])
T0 = EM = None


def set_env(emit=0.6, stay=0.6):
    global T0, EM
    oe, ot = (1 - emit) / 2, (1 - stay) / 2
    EM = np.array([[emit, oe, oe], [oe, emit, oe], [oe, oe, emit]])
    T0 = np.array([[stay, ot, ot], [ot, stay, ot], [ot, ot, stay]])


set_env()


class Agent(nn.Module):
    def __init__(self, d, nl, nh, L):
        super().__init__()
        self.backbone = GPT(Q, d, nl, nh, max_len=L + 2)
        self.act_head = nn.Linear(d, Q)
        self.obs_head = nn.Linear(d, Q)

    def forward(self, obs):
        _, hs = self.backbone(obs, return_hidden=True)
        h = self.backbone.lnf(hs[-1])
        return self.act_head(h), self.obs_head(h)


def cat(probs):
    return torch.multinomial(probs, 1).squeeze(-1)


@torch.no_grad()
def rollout(net, B, L, dev, T, E, pi, det=False, open_loop=False):
    s = cat(pi.expand(B, Q))
    o = cat(E[s])                                      # t=0: a_{-1}=0, no corruption
    obs, states = [o], [s]
    for t in range(L - 1):
        al, _ = net(torch.stack(obs, 1))
        p = F.softmax(al[:, -1], -1)
        a = p.argmax(-1) if det else cat(p)            # deterministic argmax (env knows it exactly) or sample
        j = cat(T[s]); s = (j if open_loop else (j + a)) % Q
        o = (cat(E[s]) + a) % Q
        obs.append(o); states.append(s)
    return torch.stack(obs, 1), torch.stack(states, 1)


def oracle_filter(obs, p, open_loop=False):             # obs (L,), p (L,Q) -> belief (L,Q)
    b = PI.copy(); bels = [b]
    for t in range(len(obs) - 1):
        Tb = T0.T @ b; bn = np.zeros(Q)
        for k in range(Q):
            for a in range(Q):
                trans = Tb[k] if open_loop else Tb[(k - a) % Q]
                bn[k] += p[t, a] * trans * EM[k, (obs[t + 1] - a) % Q]
        bn /= bn.sum(); bels.append(bn); b = bn
    return np.array(bels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/agent")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--L", type=int, default=40)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--emit", type=float, default=0.6)          # emission own-prob (higher = cleaner)
    ap.add_argument("--stay", type=float, default=0.6)          # transition self-loop prob
    ap.add_argument("--det_action", action="store_true")        # env uses argmax p_t (deterministic, no aperture)
    ap.add_argument("--ce_action", action="store_true")         # CE action loss -log p(s) (vs match -p(s))
    ap.add_argument("--open_loop", action="store_true")         # action does NOT steer the transition
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_env(args.emit, args.stay)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    T = torch.tensor(T0, dtype=torch.float32, device=dev)
    E = torch.tensor(EM, dtype=torch.float32, device=dev)
    pi = torch.tensor(PI, dtype=torch.float32, device=dev)
    net = Agent(args.d_model, args.n_layer, args.n_head, args.L).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"agent d={args.d_model} L={args.n_layer} h={args.n_head} | "
          f"params={sum(p.numel() for p in net.parameters())/1e3:.0f}K", flush=True)

    @torch.no_grad()
    def evaluate():
        net.eval()
        obs, states = rollout(net, 512, args.L, dev, T, E, pi, args.det_action, args.open_loop)
        al, ol = net(obs); p = F.softmax(al, -1)
        tail = slice(args.L // 2, None)
        acc = (p.argmax(-1) == states)[:, tail].float().mean().item()
        ent = (-(p * (p + 1e-9).log()).sum(-1))[:, tail].mean().item()
        # oracle belief on a subset (deterministic action -> filter conditions on the realized argmax)
        obs_n, p_n, st_n = obs[:64].cpu().numpy(), p[:64].cpu().numpy(), states[:64].cpu().numpy()
        if args.det_action:
            p_n = np.eye(Q)[p_n.argmax(-1)]
        bels = np.stack([oracle_filter(obs_n[i], p_n[i], args.open_loop) for i in range(64)])
        orc_acc = (bels.argmax(-1) == st_n)[:, args.L // 2:].mean()
        bel_ent = (-(bels * np.log(bels + 1e-9)).sum(-1))[:, args.L // 2:].mean()
        net.train()
        return acc, ent, orc_acc, bel_ent

    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train()
        obs, states = rollout(net, args.batch, args.L, dev, T, E, pi, args.det_action, args.open_loop)
        al, ol = net(obs)
        p = F.softmax(al, -1)
        pa = p.gather(-1, states.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)   # p_t(s_t)
        action_loss = -pa.log().mean() if args.ce_action else -pa.mean()
        obs_loss = -F.log_softmax(ol, -1)[:, :-1].gather(-1, obs[:, 1:, None]).squeeze(-1).mean()
        loss = action_loss + obs_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            acc, ent, orc_acc, bel_ent = evaluate()
            log.append(dict(step=step, action_loss=float(action_loss), obs_loss=float(obs_loss),
                            act_acc=acc, act_ent=ent, oracle_acc=float(orc_acc), belief_ent=float(bel_ent)))
            print(f"step {step:5d} | aL {action_loss:+.3f} oL {obs_loss:.3f} | "
                  f"act_acc {acc:.3f} (oracle {orc_acc:.3f}) | "
                  f"act_H {ent:.3f}  belief_H {bel_ent:.3f}  (sharpening: act_H<belief_H?)", flush=True)

    out = os.path.join(BASE, args.out)
    torch.save({"state": net.state_dict(), "args": vars(args)}, out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt", flush=True)


if __name__ == "__main__":
    main()
