"""Does seeding from an EARLY layer let the seed propagate into the belief simplex?
For each B-free model (seed read at layer 3 vs layer 1), patch its seeding channel at a reseed
position to a target state and measure whether the asym3 BELIEF simplex (fresh probe at layer 3)
moves toward it at the reseed position and the next two. Layer-1 seeding CAN propagate via
attention to later positions; layer-3 cannot. Also report each model's predictive entropy over
the post-reset rollout (lower => uses the seed better)."""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT  # noqa
import whitebox, probes, exp_B  # noqa
from factors import mess3_factor, asym3_factor, decode_subtokens, belief_filter  # noqa

BASE = os.path.dirname(os.path.abspath(__file__))
facs = [mess3_factor(0.6, 0.15), asym3_factor()]
meta = json.load(open(BASE + "/runs/uni_mess3_asym3.json")); A = meta["args"]; V = meta["V"]
rng = np.random.default_rng(7)
K = 5


def tmodel(p):
    m = GPT(V, A["d_model"], A["n_layer"], A["n_head"], max_len=64)
    m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); return m


def ent(p): return -(np.clip(p, 1e-30, None) * np.log(np.clip(p, 1e-30, None))).sum(-1)


def analyze(label, ckpt, probes_file):
    pd = {k: np.load(probes_file)[k] for k in np.load(probes_file).files}
    ell = int(pd["ell1"]); Wseed, bseed = pd["W1"], pd["b1"]
    pinv = np.linalg.solve(Wseed.T @ Wseed, Wseed.T)
    toks, rpos = exp_B.generate_closedloop(tmodel(ckpt), facs, pd, K, 512, 64, rng, "cpu")
    W = whitebox.load_weights(ckpt)
    acts0, _ = whitebox.forward(W, toks)
    sub = decode_subtokens(toks, 2)
    bel1 = belief_filter(facs[1].T, sub[..., 1], facs[1].pi).reshape(-1, 3)
    Wbel, bbel, r2 = probes.ridge_fit(acts0["L3.resid_post"].reshape(-1, 128), bel1)
    seed_layer = f"L{ell}.resid_post"
    ridx = [t - 1 for t in rpos]

    def read(edit):
        a, _ = whitebox.forward(W, toks, edit_fn=edit)
        q = probes.readout(a["L3.resid_post"].reshape(-1, 128), Wbel, bbel).reshape(toks.shape[0], 64, 3)
        lg = a["logits"]; P = np.exp(lg - lg.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
        pz1 = P.reshape(toks.shape[0], 64, 3, 3).sum(2)
        return q, pz1

    base_q, base_p = read(None)
    # predictive entropy over the post-reset rollout (positions t..t+K-1 after each reseed)
    roll = [min(t + o, 63) for t in rpos for o in range(K)]
    print(f"\n=== {label} (seed read at layer {ell}) ===  belief R2={r2:.3f}  "
          f"rollout pred-entropy(asym3)={ent(base_p[:, roll, :]).mean():.3f}")
    for s in range(3):
        tgt = np.eye(3)[s]
        def fn(nm, x, _t=tgt):
            if nm != seed_layer: return x
            x = x.copy(); cur = x[:, ridx, :] @ Wseed + bseed
            x[:, ridx, :] += (_t[None, None] - cur) @ pinv
            return x
        q, p = read(fn)
        bshift = [float((q[:, [min(t - 1 + o, 63) for t in rpos], s] - base_q[:, [min(t - 1 + o, 63) for t in rpos], s]).mean()) for o in range(3)]
        pshift = float((p[:, ridx, s] - base_p[:, ridx, s]).mean())
        print(f"  seed->state{s}: Δ BELIEF(state{s}) at [t-1,t,t+1]={np.round(bshift,3)}   Δ PREDICT@t-1={pshift:+.3f}")


analyze("layer-3 B-free", BASE + "/runs/expB.pt", BASE + "/runs/uni_mess3_asym3_probes.npz")
analyze("layer-1 B-free (early seed)", BASE + "/runs/expB_L1seed.pt", BASE + "/runs/uni_L1seed_probes.npz")
