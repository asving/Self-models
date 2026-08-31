"""Localize the tracking gap + look for the self-model circuit in a trained agent.
 - belief: is the oracle action-conditioned belief decodable from the residual? (does the net HAVE it)
 - state: is s_t decodable? how does net action accuracy compare to a probe of the belief's MAP?
 - efference copy: at position t, is the net's OWN previous action a_{t-1} decodable (carried internally,
   since it's never an input token)? — the internal self-trace it needs to decode o_t.
 - depth: where (which layer) does each become available — does the action lead the belief?
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import probes
import agent as A

BASE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def probe_acc(X, lab, nc, tr=0.7):
    n = len(X); c = int(n * tr); idx = rng.permutation(n)
    W, b, _ = probes.ridge_fit(X[idx[:c]], np.eye(nc)[lab[idx[:c]]])
    return ((X[idx[c:]] @ W + b).argmax(1) == lab[idx[c:]]).mean()


def main(tag="mid"):
    ck = torch.load(BASE + f"/runs/agent_{tag}.pt", map_location="cpu")
    a = ck["args"]; dev = "cpu"
    net = A.Agent(a["d_model"], a["n_layer"], a["n_head"], a["L"]).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    T = torch.tensor(A.T0, dtype=torch.float32); E = torch.tensor(A.EM, dtype=torch.float32); pi = torch.tensor(A.PI, dtype=torch.float32)
    with torch.no_grad():
        obs, states = A.rollout(net, 600, a["L"], dev, T, E, pi)
        al, _ = net(obs); p = F.softmax(al, -1)
        _, hs = net.backbone(obs, return_hidden=True)
        H = [net.backbone.lnf(h).numpy() for h in hs]            # per-layer (post-ln) residual
    obs_n, st_n, p_n = obs.numpy(), states.numpy(), p.numpy()
    bel = np.stack([A.oracle_filter(obs_n[i], p_n[i]) for i in range(len(obs_n))])  # oracle belief (B,L,3)
    L = a["L"]; d = a["d_model"]
    prev_a = p_n.argmax(-1)[:, :-1]                              # a_{t-1} (committed prev action)

    tail = slice(L // 2, None)
    net_acc = (p_n.argmax(-1) == st_n)[:, tail].mean()
    orc_acc = (bel.argmax(-1) == st_n)[:, tail].mean()
    print(f"=== agent_{tag} ({a['n_layer']}x{a['d_model']}) ===")
    print(f"net act_acc={net_acc:.3f}  oracle_acc={orc_acc:.3f}  gap={orc_acc-net_acc:.3f}")
    print(f"belief-MAP probe vs net action: does the net ACT on its belief's MAP, or under-use it?")
    print("layer        | belief-R² | state-acc | belief-MAP-acc | effcopy a_{t-1}")
    for i, h in enumerate(H):
        r2 = probes.ridge_fit(h.reshape(-1, d), bel.reshape(-1, 3))[2]
        sacc = probe_acc(h.reshape(-1, d), st_n.reshape(-1), 3)
        bmap = probe_acc(h.reshape(-1, d), bel.argmax(-1).reshape(-1), 3)   # can we read the belief's MAP?
        eff = probe_acc(h[:, 1:, :].reshape(-1, d), prev_a.reshape(-1), 3)  # internal prev-action trace
        print(f"  L{i}.resid    |   {r2:5.3f}   |   {sacc:.3f}   |     {bmap:.3f}      |     {eff:.3f}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "mid")
