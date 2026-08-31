"""Analysis 5 — causal: does the seeding channel feed the belief subspace?

At a reseed (position index t-1 sets the seed for token t), we PATCH the asym3-belief readout at
an early layer to a target state s*, propagate, and decode the asym3 belief (fresh layer-3 probe)
and the asym3 prediction at the post-reseed positions. If they move toward s*, the model routes its
own seed-output forward as a prior (efference copy). World data is held fixed (= desync: model
'thinks it seeded s*' while the world did the real seed).
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT  # noqa: E402
import whitebox, probes, exp_B  # noqa: E402
from factors import mess3_factor, asym3_factor, decode_subtokens, belief_filter  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
facs = [mess3_factor(0.6, 0.15), asym3_factor()]
meta = json.load(open(BASE + "/runs/uni_mess3_asym3.json")); A = meta["args"]; V = meta["V"]
probes_d = {k: np.load(BASE + "/runs/uni_mess3_asym3_probes.npz")[k]
            for k in np.load(BASE + "/runs/uni_mess3_asym3_probes.npz").files}
rng = np.random.default_rng(2)
PATCH_LAYER = "L1.resid_post"     # early enough to propagate to later positions via attn
READ_LAYER = "L3.resid_post"
K = 5


def tmodel(p):
    m = GPT(V, A["d_model"], A["n_layer"], A["n_head"], max_len=64)
    m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); return m


def run(name, path):
    Wnp = whitebox.load_weights(path)
    toks, rpos = exp_B.generate_closedloop(tmodel(path), facs, probes_d, K, 512, 64, rng, "cpu")
    sub = decode_subtokens(toks, 2)
    # fit fresh asym3-belief probes at the PATCH layer and the READ layer (this net, this data)
    acts0, _ = whitebox.forward(Wnp, toks)
    bel1 = belief_filter(facs[1].T, sub[..., 1], facs[1].pi).reshape(-1, 3)   # asym3 naive filtering belief
    Wp, bp, _ = probes.ridge_fit(acts0[PATCH_LAYER].reshape(-1, 128), bel1)   # set knob at patch layer
    Wr, br, _ = probes.ridge_fit(acts0[READ_LAYER].reshape(-1, 128), bel1)    # readout at read layer
    pinv = np.linalg.solve(Wp.T @ Wp, Wp.T)                                    # (3,128), min-norm edit
    reseed_idx = [t - 1 for t in rpos]                                          # positions that hold the seed

    def edit_to(target):
        def fn(nm, x):
            if nm != PATCH_LAYER:
                return x
            x = x.copy()
            cur = x[:, reseed_idx, :] @ Wp + bp                 # (B, R, 3) current readout
            delta = (target[None, None] - cur) @ pinv           # (B, R, 128)
            x[:, reseed_idx, :] += delta
            return x
        return fn

    # baseline + patched belief readouts at READ layer, at the first post-reseed positions (t-1, t, t+1)
    def read_belief(edit):
        a, _ = whitebox.forward(Wnp, toks, edit_fn=edit)
        r = a[READ_LAYER]                                       # (B,L,d)
        q = probes.readout(r.reshape(-1, 128), Wr, br).reshape(toks.shape[0], 64, 3)
        return q

    base = read_belief(None)
    print(f"\n=== {name} ===  asym3 belief-subspace mass on each state, at reseed-relative offset")
    for s in range(3):
        tgt = np.eye(3)[s]
        q = read_belief(edit_to(tgt))
        # measure at offsets 0,1,2 after the reseed point (index t-1 -> offsets at t-1,t,t+1)
        row = []
        for off in range(3):
            idx = [min(t - 1 + off, 63) for t in rpos]
            shift = (q[:, idx, s] - base[:, idx, s]).mean()      # change in mass on the patched state
            row.append(shift)
        print(f"  patch seed->state{s}: Δmass(state{s}) at offsets[t-1,t,t+1] = {np.round(row,3)}")


run("PRETRAINED", BASE + "/runs/uni_mess3_asym3.pt")
run("B-free", BASE + "/runs/expB.pt")
run("B+RL beta=3", BASE + "/runs/expB_rl_b3.pt")
