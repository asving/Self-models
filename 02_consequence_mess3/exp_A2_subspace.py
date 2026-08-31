"""Q1: what happens to the belief subspaces from PRETRAIN -> POST (A2 fine-tune)?
Hypothesis: the belief-update machinery (subspace + filter) is PRESERVED; A2 only inserts an
action-correction UPSTREAM of the (unchanged) asym3 belief update. Tests:
 [A] principal angles between pretrain & post belief subspaces (mess3, asym3) — high = same subspace.
 [B] transfer: pretrain's asym3 probe (frozen) decodes the post net's TRUE belief — does the subspace carry it?
 [C] was the action already present in pretrain? (it's a fn of the mess3 belief pretrain already had)
 [D] the frozen pretrain net, fed corrupted input, decodes the NAIVE belief (ignores aₜ); A2 decodes TRUE.
     => what fine-tuning ADDED is the routing/correction, in the SAME subspace.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import whitebox, probes
from factors import mess3_factor, asym3_factor, belief_filter
from exp_A2 import generate_A2

BASE = os.path.dirname(os.path.abspath(__file__))
f0, f1 = mess3_factor(0.6, 0.15), asym3_factor()
rng = np.random.default_rng(5)
LAYERS = [f"L{i}.resid_post" for i in range(4)]


def cos_principal(A, B):
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def r2_apply(X, Y, W, b):
    pred = X @ W + b
    return 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def main():
    Wpre = whitebox.load_weights(BASE + "/runs/uni_mess3_asym3.pt")
    Wpost = whitebox.load_weights(BASE + "/runs/expA2_plain.pt")
    toks, a, z0, z1, _ = generate_A2(f0, f1, 1500, 64, rng, 1.0)
    ctoks = z0 * 3 + z1                                            # clean (uncorrupted) twin
    x1 = toks % 3
    mb = belief_filter(f0.T, z0, f0.pi)
    bel_true = belief_filter(f1.T, z1, f1.pi)                     # asym3 TRUE belief (needs aₜ)
    bel_naive = belief_filter(f1.T, x1, f1.pi)                    # asym3 belief ignoring aₜ
    a_oh = np.eye(3)[a]
    A_preC, _ = whitebox.forward(Wpre, ctoks)                     # pretrain on CLEAN (its native regime)
    A_preX, _ = whitebox.forward(Wpre, toks)                      # pretrain on CORRUPTED (frozen)
    A_post, _ = whitebox.forward(Wpost, toks)                     # A2 on CORRUPTED

    rngU = np.random.default_rng(0)
    def proj_r2(Xpost, U, Y):
        return probes.ridge_fit(Xpost @ U, Y)[2]                  # refit 3->3 (offset/scale-free), only DIRECTIONS fixed

    print("=" * 72)
    print("[A/B] Is the POST net's TRUE asym3 belief readable from the PRETRAIN belief DIRECTIONS?")
    print("      (project post resid onto pretrain's 3 belief dirs, refit). ≈ceiling=same subspace; ≈random=new")
    print("    layer | ceiling(full) | in-PRETRAIN-asym3-subspace | in-pretrain-MESS3-sub | random-3d")
    for ln in LAYERS:
        Xp = A_post[ln].reshape(-1, 128)
        U_a = np.linalg.qr(probes.ridge_fit(A_preC[ln].reshape(-1, 128), bel_true.reshape(-1, 3))[0])[0]
        U_m = np.linalg.qr(probes.ridge_fit(A_preC[ln].reshape(-1, 128), mb.reshape(-1, 3))[0])[0]
        U_r = np.linalg.qr(rngU.standard_normal((128, 3)))[0]
        Y = bel_true.reshape(-1, 3)
        print(f"    {ln} |    {probes.ridge_fit(Xp,Y)[2]:.3f}     |        {proj_r2(Xp,U_a,Y):.3f}              "
              f"|       {proj_r2(Xp,U_m,Y):.3f}         |  {proj_r2(Xp,U_r,Y):.3f}")

    print("=" * 72)
    print("[C] was the ACTION already represented in the PRETRAIN net? (aₜ=runner-up of mess3 belief)")
    for ln in LAYERS:
        r2_pre = probes.ridge_fit(A_preX[ln].reshape(-1, 128), a_oh.reshape(-1, 3))[2]
        r2_post = probes.ridge_fit(A_post[ln].reshape(-1, 128), a_oh.reshape(-1, 3))[2]
        print(f"    {ln}: action R² pretrain={r2_pre:.3f}  post={r2_post:.3f}")

    print("=" * 72)
    print("[D] frozen PRETRAIN net on CORRUPTED input decodes which belief? (TRUE needs routing, NAIVE doesn't)")
    print("              PRETRAIN-frozen            |        A2-POST")
    print("    layer | TRUE-bel | naive-bel        | TRUE-bel | naive-bel")
    for ln in LAYERS:
        pt = probes.ridge_fit(A_preX[ln].reshape(-1, 128), bel_true.reshape(-1, 3))[2]
        pn = probes.ridge_fit(A_preX[ln].reshape(-1, 128), bel_naive.reshape(-1, 3))[2]
        qt = probes.ridge_fit(A_post[ln].reshape(-1, 128), bel_true.reshape(-1, 3))[2]
        qn = probes.ridge_fit(A_post[ln].reshape(-1, 128), bel_naive.reshape(-1, 3))[2]
        print(f"    {ln} |  {pt:.3f}  |  {pn:.3f}          |  {qt:.3f}  |  {qn:.3f}")


if __name__ == "__main__":
    main()
