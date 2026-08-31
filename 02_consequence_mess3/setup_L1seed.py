"""Stage an early-seeding (layer-1) B-free run: fit a layer-1 asym3 belief probe and write a
probes file that seeds asym3 from layer 1 (asym3 ell1=1) while keeping the mess3 metric probe at
layer 3. Copy the pretrained ckpt so exp_B.py can train from it with --ckpt runs/uni_L1seed."""
import os, sys, json, shutil
import numpy as np
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import whitebox, probes
from factors import mess3_factor, asym3_factor, make_world, decode_subtokens, belief_filter

BASE = os.path.dirname(os.path.abspath(__file__))
facs = [mess3_factor(0.6, 0.15), asym3_factor()]
W = whitebox.load_weights(BASE + "/runs/uni_mess3_asym3.pt")
rng = np.random.default_rng(0)
toks, _ = make_world(facs, 0.0).sample(4000, 64, rng)
acts, _ = whitebox.forward(W, toks)
sub = decode_subtokens(toks, 2)
bel1 = belief_filter(facs[1].T, sub[..., 1], facs[1].pi).reshape(-1, 3)
W1, b1, r21 = probes.ridge_fit(acts["L1.resid_post"].reshape(-1, 128), bel1)   # layer-1 asym3 readout
orig = np.load(BASE + "/runs/uni_mess3_asym3_probes.npz")
np.savez(BASE + "/runs/uni_L1seed_probes.npz",
         W0=orig["W0"], b0=orig["b0"], ell0=orig["ell0"], r20=orig["r20"],
         W1=W1, b1=b1, ell1=np.array(1), r21=np.array(r21))
shutil.copy(BASE + "/runs/uni_mess3_asym3.pt", BASE + "/runs/uni_L1seed.pt")
shutil.copy(BASE + "/runs/uni_mess3_asym3.json", BASE + "/runs/uni_L1seed.json")
print(f"layer-1 asym3 probe R2={r21:.3f}; seeds asym3 from layer 1. files: runs/uni_L1seed.*")
