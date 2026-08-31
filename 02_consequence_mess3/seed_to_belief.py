"""Does the delta_0 SEEDING CHANNEL write into the asym3 BELIEF simplex?
Patch the frozen seeding channel (uni W1 = the direction the reseed is read from) to a target
state at a reseed position, and measure whether the asym3 BELIEF readout (fresh probe, built from
tokens) moves toward that target -- at the reseed position and the next two. Also report the
effect on the model's asym3 PREDICTION (the consequence). Patch at layer 1 (can propagate) and
layer 3 (where the channel is actually read)."""
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
meta = json.load(open(BASE + "/runs/uni_mess3_asym3.json")); Aarg = meta["args"]; V = meta["V"]
pd = {k: np.load(BASE + "/runs/uni_mess3_asym3_probes.npz")[k] for k in np.load(BASE + "/runs/uni_mess3_asym3_probes.npz").files}
Wseed, bseed = pd["W1"], pd["b1"]                     # the frozen seeding channel (delta_0 subspace)
pinv = np.linalg.solve(Wseed.T @ Wseed, Wseed.T)
rng = np.random.default_rng(5)
PATH = BASE + "/runs/expB_rl_b3.pt"


def tmodel(p):
    m = GPT(V, Aarg["d_model"], Aarg["n_layer"], Aarg["n_head"], max_len=64)
    m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); return m


toks, rpos = exp_B.generate_closedloop(tmodel(PATH), facs, pd, 5, 512, 64, rng, "cpu")
W = whitebox.load_weights(PATH)
sub = decode_subtokens(toks, 2)
acts0, _ = whitebox.forward(W, toks)
# fresh asym3 BELIEF probe (the "belief simplex") at layer 3, and seeding readout for reference
bel1 = belief_filter(facs[1].T, sub[..., 1], facs[1].pi).reshape(-1, 3)
Wbel, bbel, r2 = probes.ridge_fit(acts0["L3.resid_post"].reshape(-1, 128), bel1)
# orthogonality of seeding channel vs belief subspace
onb = lambda M: np.linalg.svd(M, full_matrices=False)[0][:, :2]
ovl = np.linalg.svd(onb(Wseed).T @ onb(Wbel), compute_uv=False)
print(f"fresh belief R2={r2:.3f}   seeding-vs-belief subspace overlap (cos) = {np.round(ovl,3)}")
ridx = [t - 1 for t in rpos]


def read_bel(edit):
    a, _ = whitebox.forward(W, toks, edit_fn=edit)
    q = probes.readout(a["L3.resid_post"].reshape(-1, 128), Wbel, bbel).reshape(toks.shape[0], 64, 3)
    lg = a["logits"]; P = np.exp(lg - lg.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    pz1 = P.reshape(toks.shape[0], 64, 3, 3).sum(2)               # asym3 prediction marginal
    return q, pz1


base_q, base_p = read_bel(None)
for player in ["L1.resid_post", "L3.resid_post"]:
    print(f"\n--- patch SEEDING channel at {player} (set seed-readout to state s) ---")
    for s in range(3):
        tgt = np.eye(3)[s]
        def fn(nm, x, _t=tgt):
            if nm != player: return x
            x = x.copy(); cur = x[:, ridx, :] @ Wseed + bseed
            x[:, ridx, :] += (_t[None, None] - cur) @ pinv
            return x
        q, p = read_bel(fn)
        bel_shift = [float((q[:, [min(t-1+o, 63) for t in rpos], s] - base_q[:, [min(t-1+o, 63) for t in rpos], s]).mean()) for o in range(3)]
        pred_shift = float((p[:, ridx, s] - base_p[:, ridx, s]).mean())   # prediction at reseed pos (offset 0)
        print(f"  seed->state{s}: Δ BELIEF(state{s}) at [t-1,t,t+1]={np.round(bel_shift,3)}   Δ PREDICT(z1={s}) at t-1={pred_shift:+.3f}")
