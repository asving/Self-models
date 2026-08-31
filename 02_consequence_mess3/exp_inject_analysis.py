"""Does the model READ and ROUTE the injected seed v?
Counterfactual injection: generate data with real seeds, but inject a FIXED state s into the
side-channel, and measure (a) the predicted asym3 sub-token at the inject position -> emit(s)?
(reads it), and (b) the asym3 BELIEF (fresh probe) at the next positions -> shifts toward s?
(routes it forward). Also report the value-of-v loss gap (inject real v vs inject nothing)."""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import whitebox, probes  # noqa
from factors import mess3_factor, asym3_factor, decode_subtokens, belief_filter  # noqa
import exp_inject  # generate_injected  # noqa

BASE = os.path.dirname(os.path.abspath(__file__))
facs = [mess3_factor(0.6, 0.15), asym3_factor()]
E = facs[1].E
rng = np.random.default_rng(11)
K = 5


def ent(p): return -(np.clip(p, 1e-30, None) * np.log(np.clip(p, 1e-30, None))).sum(-1)


def analyze(tag, ckpt):
    W = whitebox.load_weights(ckpt)
    ev = np.load(ckpt[:-3] + "_ev.npz")["ev"]                       # (3,d) seed embeddings
    toks, inj_pos, seeds = exp_inject.generate_injected(facs, K, 1024, 64, rng)
    inj_pos = np.array(inj_pos)

    def edit_inject(target):   # target: None=real seeds, int=fixed counterfactual, 'off'=none
        def fn(nm, x):
            if nm != "embed": return x
            x = x.copy()
            for i, p in enumerate(inj_pos):
                if target == "off": continue
                v = seeds[:, i] if target is None else np.full(seeds.shape[0], target)
                x[:, p, :] += ev[v]
            return x
        return fn

    # fresh asym3 belief probe (layer 3) on the real-injection run
    acts_real, _ = whitebox.forward(W, toks, edit_fn=edit_inject(None))
    sub = decode_subtokens(toks, 2)
    bel1 = belief_filter(facs[1].T, sub[..., 1], facs[1].pi).reshape(-1, 3)
    Wb, bb, r2 = probes.ridge_fit(acts_real["L3.resid_post"].reshape(-1, 128), bel1)

    def pred_and_belief(edit):
        a, _ = whitebox.forward(W, toks, edit_fn=edit)
        lg = a["logits"]; P = np.exp(lg - lg.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
        pz1 = P.reshape(toks.shape[0], 64, 3, 3).sum(2)            # asym3 prediction marginal
        q = probes.readout(a["L3.resid_post"].reshape(-1, 128), Wb, bb).reshape(toks.shape[0], 64, 3)
        return pz1, q

    print(f"\n=== {tag} ===  belief R2={r2:.3f}")
    # value of v: asym3 rollout NLL with real injection vs none
    roll = [min(p + 1 + o, 63) for p in inj_pos for o in range(K)]
    cols = [c - 1 for c in roll if 1 <= c <= 63]
    for lab, ed in [("real-inject", edit_inject(None)), ("no-inject", edit_inject("off"))]:
        pz1, _ = pred_and_belief(ed)
        # NLL of actual asym3 sub-token
        z1 = (toks % 3)
        nll = -np.log(np.clip(pz1[:, :-1], 1e-30, None)[np.arange(toks.shape[0])[:, None], np.arange(63)[None], z1[:, 1:]])
        print(f"  asym3 rollout NLL [{lab:11s}] = {nll[:, cols].mean():.3f}")
    # counterfactual: inject fixed state s, watch prediction (at inject pos) and belief (next positions)
    base_pz1, base_q = pred_and_belief(edit_inject(None))
    for s in range(3):
        pz1, q = pred_and_belief(edit_inject(s))
        pred_at_inj = pz1[:, inj_pos, :].reshape(-1, 3).mean(0)     # predicted asym3 dist at inject pos
        emit_s = E[s]
        bel_shift = [float((q[:, np.clip(inj_pos + o, 0, 63), s] - base_q[:, np.clip(inj_pos + o, 0, 63), s]).mean()) for o in (0, 1, 2)]
        print(f"  inject s={s}: pred@inj={np.round(pred_at_inj,2)} vs emit(s)={np.round(emit_s,2)} "
              f"| Δbelief(s) at offsets[inj,+1,+2]={np.round(bel_shift,3)}")


for tag, ck in [("scale=1", BASE + "/runs/expInject_s1.pt"), ("scale=3", BASE + "/runs/expInject_s3.pt")]:
    if os.path.exists(ck):
        analyze(tag, ck)
    else:
        print(f"{tag}: {ck} not found yet")
