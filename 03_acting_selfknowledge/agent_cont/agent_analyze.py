"""Deeper circuit analysis of the closed-loop agent.
 1) THEORETICAL FLOOR: simulate the exact-belief myopic-optimal (MAP) agent -> max achievable match
    accuracy. Compare every capacity to it.
 2) ACTION FORMATION IN DEPTH: at which layer is the net's final action already decided (decode the
    final committed action from each layer's residual)?
 3) HOW IS THE STATE STORED? linear vs NON-LINEAR (MLP) decodability of s_t and the oracle belief at
    each layer — is the belief there but non-linear, or genuinely crude?
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import probes
import agent as A

BASE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
T0, EM, PI = A.T0, A.EM, A.PI
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def samp(P):
    return (rng.random((len(P), 1)) < P.cumsum(1)).argmax(1)


def optimal_floor(B=2000, L=40, open_loop=False):
    """Exact-belief MAP agent (optimal myopic policy). Returns tail match-acc and tail belief entropy."""
    T0, EM, PI = A.T0, A.EM, A.PI                       # read CURRENT env (after set_env), not import-time copy
    s = rng.choice(3, B, p=PI)
    o = samp(EM[s])
    b = PI[None] * EM[:, o].T; b /= b.sum(1, keepdims=True)
    matches, bent = [], []
    for t in range(L):
        a = b.argmax(1)
        matches.append(a == s); bent.append(-(b * np.log(b + 1e-9)).sum(1))
        j = samp(T0[s]); s = (j if open_loop else (j + a)) % 3
        o = (samp(EM[s]) + a) % 3
        Tb = b @ T0
        bn = np.stack([Tb[np.arange(B), k if open_loop else (k - a) % 3] * EM[k, (o - a) % 3] for k in range(3)], 1)
        b = bn / bn.sum(1, keepdims=True)
    M = np.array(matches).T; H = np.array(bent).T
    return M[:, L // 2:].mean(), H[:, L // 2:].mean()


def mlp_acc(X, y, nc, hid=256, steps=500):
    n = len(X); idx = rng.permutation(n); c = int(.7 * n)
    Xt = torch.tensor(X[idx[:c]], dtype=torch.float32, device=DEV); yt = torch.tensor(y[idx[:c]], device=DEV)
    Xe = torch.tensor(X[idx[c:]], dtype=torch.float32, device=DEV); ye = y[idx[c:]]
    m = nn.Sequential(nn.Linear(X.shape[1], hid), nn.GELU(), nn.Linear(hid, nc)).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(m(Xt), yt).backward(); opt.step()
    with torch.no_grad():
        return (m(Xe).argmax(1).cpu().numpy() == ye).mean()


def lin_acc(X, y, nc):
    n = len(X); idx = rng.permutation(n); c = int(.7 * n)
    W, b, _ = probes.ridge_fit(X[idx[:c]], np.eye(nc)[y[idx[:c]]])
    return ((X[idx[c:]] @ W + b).argmax(1) == y[idx[c:]]).mean()


def analyze(tag):
    ck = torch.load(BASE + f"/runs/agent_{tag}.pt", map_location="cpu"); a = ck["args"]
    A.set_env(a.get("emit", 0.6), a.get("stay", 0.6))               # use THIS net's env
    det, ol = a.get("det_action", False), a.get("open_loop", False)
    net = A.Agent(a["d_model"], a["n_layer"], a["n_head"], a["L"]).to(DEV); net.load_state_dict(ck["state"]); net.eval()
    T = torch.tensor(A.T0, dtype=torch.float32, device=DEV); E = torch.tensor(A.EM, dtype=torch.float32, device=DEV)
    pi = torch.tensor(A.PI, dtype=torch.float32, device=DEV)
    d = a["d_model"]
    with torch.no_grad():
        obs, states = A.rollout(net, 600, a["L"], DEV, T, E, pi, det, ol)
        al, _ = net(obs); p = F.softmax(al, -1)
        _, hs = net.backbone(obs, return_hidden=True)
        H = [net.backbone.lnf(h).cpu().numpy().reshape(-1, d) for h in hs]
    floor, fH = optimal_floor(open_loop=ol)
    acc = (p.argmax(-1) == states)[:, a["L"] // 2:].float().mean().item()
    pn = p.cpu().numpy()
    if det: pn = np.eye(3)[pn.argmax(-1)]
    bel = np.stack([A.oracle_filter(obs.cpu().numpy()[i], pn[i], ol) for i in range(len(obs))])
    st = states.cpu().numpy().reshape(-1); belmap = bel.argmax(-1).reshape(-1)
    fa = al.argmax(-1).cpu().numpy().reshape(-1)
    print("=" * 66)
    print(f"=== {tag} ({a['n_layer']}x{d}, emit={a.get('emit',0.6)} stay={a.get('stay',0.6)}) ===")
    print(f"FLOOR={floor:.3f} (belief_H={fH:.3f}) | net act_acc={acc:.3f} | gap={floor-acc:+.3f}")
    print("[action formed in depth] " + " ".join(f"L{i}={lin_acc(H[i],fa,3):.2f}" for i in range(len(H))))
    print("[state storage] layer | lin(s) | MLP(s) | lin(belief-MAP) | MLP(belief-MAP)")
    for i in range(len(H)):
        print(f"    L{i} |  {lin_acc(H[i],st,3):.3f} |  {mlp_acc(H[i],st,3):.3f} |"
              f"     {lin_acc(H[i],belmap,3):.3f}      |    {mlp_acc(H[i],belmap,3):.3f}")


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["mid"]):
        analyze(t)
