"""How did nl_n4 resolve the belief↔action coupling? Test the two competing pictures:
 (A) deep recurrence / Picard across layers  vs  (B) shallow function of a recent observation window.
 - per-layer R²(hidden → net's own action): if high already at the embedding/L0 → computed shallow.
 - effective window: keep only the last k observations fixed (randomize the prefix), see when the
   last-position action stops changing → the recurrence's memory horizon. Short k ⇒ windowed/shallow.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import probes
import agent_cont_nl as M
from agent_cont import ContAgent

rng = np.random.default_rng(0); dev = "cpu"
ck = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs/nl_n4.pt"), map_location=dev); a = ck["args"]
net = ContAgent(a["d_model"], a["n_layer"], a["n_head"], a["L"]); net.load_state_dict(ck["state"]); net.eval()
L, d = a["L"], a["d_model"]


def hiddens(o):                                              # per-block, post-lnf
    x = net.in_proj(o.unsqueeze(-1)) + net.pos(torch.arange(o.shape[1]))[None]
    mask = torch.triu(torch.ones(o.shape[1], o.shape[1], dtype=torch.bool), 1)
    hs = [net.lnf(net.in_proj(o.unsqueeze(-1)) + net.pos(torch.arange(o.shape[1]))[None]).detach().numpy()]
    for blk in net.blocks:
        x = blk(x, mask); hs.append(net.lnf(x).detach().numpy())
    return hs


with torch.no_grad():
    obs, states = M.rollout(net, 1500, L, dev)
    act = torch.tanh(net(obs)[0]).numpy()                   # net's action a_t
    H = hiddens(obs)

print("=== nl_n4: per-layer R²(hidden → net's own action a_t) ===")
y = act.reshape(-1, 1)
names = ["embed"] + [f"L{i}" for i in range(len(H) - 1)]
for nm, h in zip(names, H):
    print(f"  {nm:6s}: R²={probes.ridge_fit(h.reshape(-1, d), y)[2]:.3f}")

print("=== effective window: randomize prefix, keep last k obs; Δ(last-pos action) vs k ===")
with torch.no_grad():
    obsB, _ = M.rollout(net, 1500, L, dev)                  # independent prefix source
    a_full = torch.tanh(net(obs)[0])[:, -1].numpy()
    base = np.sqrt(((a_full - torch.tanh(net(obsB)[0])[:, -1].numpy()) ** 2).mean())  # fully-different ref
    for k in [1, 2, 3, 5, 8, 15, 40]:
        mix = torch.cat([obsB[:, :L - k], obs[:, L - k:]], 1)
        a_mix = torch.tanh(net(mix)[0])[:, -1].numpy()
        rmse = np.sqrt(((a_mix - a_full) ** 2).mean())
        print(f"  keep last k={k:2d}: RMSE(action vs full-context)={rmse:.4f}  ({100*(1-rmse/base):.0f}% recovered)")
