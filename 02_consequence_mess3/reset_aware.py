"""Does the model use its own q1 reset-AWAREly? At a reset, the state was re-drawn from q1, so
the optimal prediction of the first post-reset token is q1@E (use q1 directly). A reset-UNAWARE
model treats it as a normal chain step: q1@M@E (propagate through the transition first). Compare
the model's actual prediction of the first post-reset asym3 sub-token to both (and to the
context-free prior pi@E). Closest-to-aware => the model exploits its own seed/q1."""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT  # noqa
import whitebox, probes, exp_B  # noqa
from factors import mess3_factor, asym3_factor  # noqa

BASE = os.path.dirname(os.path.abspath(__file__))
facs = [mess3_factor(0.6, 0.15), asym3_factor()]
meta = json.load(open(BASE + "/runs/uni_mess3_asym3.json")); Aarg = meta["args"]; V = meta["V"]
pd = {k: np.load(BASE + "/runs/uni_mess3_asym3_probes.npz")[k] for k in np.load(BASE + "/runs/uni_mess3_asym3_probes.npz").files}
E, M = facs[1].E, facs[1].M; pi = facs[1].pi
rng = np.random.default_rng(3)


def tmodel(p):
    m = GPT(V, Aarg["d_model"], Aarg["n_layer"], Aarg["n_head"], max_len=64)
    m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); return m


def rmse(a, b): return float(np.sqrt(((a - b) ** 2).mean()))


for name, path in [("B-free", "runs/expB.pt"), ("B+RL beta=3", "runs/expB_rl_b3.pt")]:
    toks, rpos = exp_B.generate_closedloop(tmodel(BASE + "/" + path), facs, pd, 5, 512, 64, rng, "cpu")
    W = whitebox.load_weights(BASE + "/" + path)
    acts, _ = whitebox.forward(W, toks)
    idx = [t - 1 for t in rpos]                                   # positions predicting the first post-reset token
    q1 = probes.readout(acts["L3.resid_post"][:, idx, :].reshape(-1, 128), pd["W1"], pd["b1"])  # (B*R,3) seed readout
    lg = acts["logits"][:, idx, :].reshape(-1, 9)
    P = np.exp(lg - lg.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    model_z1 = P.reshape(-1, 3, 3).sum(1)                          # marginal over asym3 sub-token (sum over z0)
    aware = q1 @ E
    unaware = q1 @ M @ E
    prior = np.broadcast_to(pi @ E, aware.shape)
    print(f"\n=== {name} ===  (RMSE of model's first-post-reset asym3 prediction to each)")
    print(f"   reset-AWARE  (q1@E)   : {rmse(model_z1, aware):.4f}")
    print(f"   reset-UNAWARE(q1@M@E) : {rmse(model_z1, unaware):.4f}")
    print(f"   context-free (pi@E)   : {rmse(model_z1, prior):.4f}")
    print(f"   [aware vs unaware differ by {rmse(aware, unaware):.4f}; mean q1={np.round(q1.mean(0),3)}]")
