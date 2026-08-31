"""Pre vs post-RL circuit diff. Key questions:
  - Does the model still represent each factor's belief? (fresh-probe R2)
  - Did asym3's belief move OFF the frozen-probe direction? (frozen-probe R2 + readout entropy
    collapsed, but fresh-probe R2 high  ==>  decoupling: info retained on a new direction)
  - Is the output still a factored Bayesian read-off?
Target = naive filtering belief (belief_filter on observed sub-tokens; a consistent proxy for
"how much asym3-state info is linearly present", robust for the frozen-vs-fresh comparison)."""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT  # noqa: E402
import whitebox, probes, exp_B  # noqa: E402
from factors import mess3_factor, asym3_factor, make_world, decode_subtokens, belief_filter  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
facs = [mess3_factor(0.6, 0.15), asym3_factor()]
meta = json.load(open(BASE + "/runs/uni_mess3_asym3.json")); A = meta["args"]; V = meta["V"]
npz = np.load(BASE + "/runs/uni_mess3_asym3_probes.npz"); probes_d = {k: npz[k] for k in npz.files}
rng = np.random.default_rng(1)


def ent(p): return -(np.clip(p, 1e-30, None) * np.log(np.clip(p, 1e-30, None))).sum(-1)


def anova2(M):
    Vv = M.reshape(-1, 3, 3); mu = Vv.mean((1, 2), keepdims=True)
    r = Vv - (mu + Vv.mean(2, keepdims=True) - mu + Vv.mean(1, keepdims=True) - mu)
    return float(np.mean(1 - (r ** 2).sum((1, 2)) / np.clip(((Vv - mu) ** 2).sum((1, 2)), 1e-30, None)))


def tmodel(p):
    m = GPT(V, A["d_model"], A["n_layer"], A["n_head"], max_len=64); m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); return m


def analyze(name, path, reseed):
    Wnp = whitebox.load_weights(path)
    if reseed:
        toks, _ = exp_B.generate_closedloop(tmodel(path), facs, probes_d, 5, 512, 64, rng, "cpu")
    else:
        toks, _ = make_world(facs, eps=0.0).sample(512, 64, rng)
    acts, _ = whitebox.forward(Wnp, toks)
    A3 = acts["L3.resid_post"].reshape(-1, 128)
    sub = decode_subtokens(toks, 2)
    print(f"\n=== {name} ===")
    for n in range(2):
        Y = belief_filter(facs[n].T, sub[..., n], facs[n].pi).reshape(-1, 3)
        Wf, bf = probes_d[f"W{n}"], probes_d[f"b{n}"]
        pred = A3 @ Wf + bf
        r2_frozen = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
        H_frozen = float(ent(probes.readout(A3, Wf, bf)).mean())
        _, _, r2_fresh = probes.ridge_fit(A3, Y)
        print(f"  factor{n} ({facs[n].name:14s}): frozen-probe R2={r2_frozen:6.2f}  frozen readout H={H_frozen:.3f}  "
              f"FRESH-probe R2={r2_fresh:6.3f}")
    print(f"  logits additive over (z0,z1): {anova2(acts['logits'].reshape(-1, 9)):.4f}")


analyze("PRETRAINED (natural data)", BASE + "/runs/uni_mess3_asym3.pt", reseed=False)
analyze("B-free (own closed-loop)", BASE + "/runs/expB.pt", reseed=True)
analyze("B+RL beta=3 (own closed-loop)", BASE + "/runs/expB_rl_b3.pt", reseed=True)
