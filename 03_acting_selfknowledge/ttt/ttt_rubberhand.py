"""TASK 3 -- RUBBER-HAND for the TTT self-model nets.

Construct minimal pairs from the evalset: two games whose decision-round-t
situation has the SAME color-blind occupancy but DIFFERENT true ownership (hence
different optimal move), because the net's OWN past moves differed.

Then INTERNALLY patch the net's recalled own-move representation at the relevant
past round(s) toward the counterfactual move -- holding the occupancy INPUT fixed
-- and test whether the round-t chosen move shifts to the counterfactual
ownership's optimal move (perceived ownership flips => move follows).

We try:
  (A) single-location activation patch (one layer, the past own-move position);
  (B) consistent all-layers ("weight-equivalent") patch.

Patch operator = class-mean steering: for own-move readout at a layer, replace the
component of the past-position residual along the own-move code with the mean
residual of examples whose own move == the counterfactual move. (Difference-of-
means steering, robust and faithful.)

Contrast: in d2/cont_c8 the analogous internal patch was OVERWRITTEN (the net
re-derived its action from observations), so perceived state did NOT follow.

CPU.  CUDA_VISIBLE_DEVICES="" ~/comp_icl/.venv/bin/python ttt_rubberhand.py --tag wide_shallow
"""
import argparse
import collections
import numpy as np
import torch
import ttt
from ttt_causal import Manual, load, chosen_moves

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttt_runs")


def build_own_move_means(hids_layer, my, valid, lag):
    """Per layer residual: for the decision position r, collect residual vectors
    grouped by own move made at r-lag. Return dict move-> mean residual, and the
    per-row (b,r)->residual lookup is implicit. Means computed over all rows."""
    B, L = valid.shape
    by = collections.defaultdict(list)
    for b in range(B):
        for r in range(L):
            if valid[b, r] == 0: continue
            rp = r - lag
            if rp < 0 or valid[b, rp] == 0: continue
            by[my[b, rp]].append(hids_layer[b, r])
    means = {mv: np.mean(np.stack(v), 0) for mv, v in by.items() if len(v) >= 5}
    return means


def find_pairs(occ, true, valid, my, t):
    """Minimal pairs at decision round t: same occ[:,t], different true ownership
    with different optimal move. Record which PAST round's own-move differs.
    Returns list of (b_a, b_b, past_round, m_a_past, m_b_past, opt_a, opt_b)."""
    B, L = valid.shape
    by = collections.defaultdict(list)
    for b in range(B):
        if valid[b, t] == 0: continue
        by[tuple(occ[b, t].astype(int))].append(b)
    pairs = []
    for k, bs in by.items():
        bytrue = collections.defaultdict(list)
        for b in bs:
            bytrue[tuple(true[b, t])].append(b)
        ts = list(bytrue.keys())
        for i in range(len(ts)):
            for j in range(len(ts)):
                if i == j: continue
                o_a = set(ttt.optimal_moves(list(ts[i]), 1))
                o_b = set(ttt.optimal_moves(list(ts[j]), 1))
                if o_a == o_b: continue
                ba = bytrue[ts[i]][0]; bb = bytrue[ts[j]][0]
                # find a past round where own moves differ
                pr = None
                for rp in range(t):
                    if valid[ba, rp] and valid[bb, rp] and my[ba, rp] != my[bb, rp]:
                        pr = rp; break
                if pr is None: continue
                pairs.append((ba, bb, pr, my[ba, pr], my[bb, pr], o_a, o_b))
    return pairs


def fit_self_readout(hids_layer, my, valid):
    """Decode own move made AT this position from residual at this position (lag=0).
    Returns (W,b) for a quick argmax readout, to verify patches change the code."""
    import torch as _t
    B, L = valid.shape
    X = []; y = []
    for b in range(B):
        for r in range(L):
            if valid[b, r] == 0: continue
            X.append(hids_layer[b, r]); y.append(my[b, r])
    X = np.stack(X).astype(np.float64); y = np.array(y)
    mu = X.mean(0); Xc = X - mu
    Y = np.eye(9)[y]
    d = Xc.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + 1e-1 * np.eye(d), Xc.T @ Y)
    return W, mu


def rubber_hand(man, occ, true, valid, my, t=2, patch_layers="all", n_max=4000, scale=1.0):
    occ_t = torch.tensor(occ, dtype=torch.float32)
    _, hiddens = man.forward(occ_t)
    hids = [h.detach().numpy() for h in hiddens]
    clean_logits, _ = man.forward(occ_t)
    clean = chosen_moves(clean_logits, occ_t).numpy()

    nl = man.nl
    # own-move means at each layer for lag=1 and lag=2 (so we can patch any past round)
    # We patch the residual at the PAST position rp toward the counterfactual move's
    # signature. Build means keyed by (layer) of residual at a position AS A FUNCTION
    # of the move made AT THAT position (lag=0 self), since we edit residual at rp.
    means_self = []
    for li in range(nl):
        means_self.append(build_own_move_means(hids[li], my, valid, lag=0))

    pairs = find_pairs(occ, true, valid, my, t)
    rng = np.random.default_rng(0)
    rng.shuffle(pairs)
    pairs = pairs[:n_max]

    if patch_layers == "all":
        layers = list(range(nl))
    else:
        layers = [int(patch_layers)]

    # BATCHED: each pair patches a DISTINCT game index `ba` (we dedup), so we can
    # apply all patches in a single forward. Build per-(layer) additive delta tensor.
    # Dedup on ba (keep first occurrence) so patches don't collide.
    seen = set(); upairs = []
    for p in pairs:
        if p[0] in seen: continue
        seen.add(p[0]); upairs.append(p)
    pairs = upairs

    B, L, _ = occ_t.shape
    deltas = {li: torch.zeros(B, L, man.d) for li in layers}
    valid_pair = []
    for (ba, bb, pr, ma, mb, o_a, o_b) in pairs:
        ok = True
        for li in layers:
            ms = means_self[li]
            if ma not in ms or mb not in ms:
                ok = False; break
        if not ok: continue
        for li in layers:
            ms = means_self[li]
            deltas[li][ba, pr] = scale * torch.tensor(ms[mb] - ms[ma], dtype=torch.float32)
        valid_pair.append((ba, bb, pr, ma, mb, o_a, o_b))
    pairs = valid_pair

    def edit(li, x):
        if li not in layers: return x
        return x + deltas[li]
    lg, ehid = man.forward(occ_t, edit_resid=edit)
    pmv = chosen_moves(lg, occ_t).numpy()

    # POSITIVE CONTROL: does the patch actually change the decoded own-move at the
    # patched position? Read self-move at the LAST patched layer's residual at pr.
    last_layer = layers[-1]
    Wr, mur = fit_self_readout(hids[last_layer], my, valid)
    ehl = ehid[last_layer].detach().numpy()
    stick = 0; sttot = 0
    for (ba, bb, pr, ma, mb, o_a, o_b) in pairs:
        pred = ((ehl[ba, pr] - mur) @ Wr).argmax()
        sttot += 1
        if pred == mb:
            stick += 1
    if sttot:
        print(f"    [ctrl] patched-position decoded own-move == counterfactual mb: "
              f"{stick/sttot:.3f} (patch potency at L{last_layer})")

    base_in_a = 0; base_in_b = 0
    patch_in_a = 0; patch_in_b = 0
    flipped = 0; tot = 0
    moved_off_a = 0
    for (ba, bb, pr, ma, mb, o_a, o_b) in pairs:
        tot += 1
        base_move = clean[ba, t]
        base_in_a += base_move in o_a
        base_in_b += base_move in o_b
        pm = pmv[ba, t]
        patch_in_a += pm in o_a
        patch_in_b += pm in o_b
        if pm != base_move:
            moved_off_a += 1
        if (base_move in o_a) and (pm in o_b) and (pm not in o_a):
            flipped += 1
    if tot == 0:
        print(f"  patch_layers={patch_layers}: no valid pairs"); return {}
    print(f"  patch_layers={patch_layers}  pairs={tot}  (decision round t={t}, "
          f"patch at differing past round)")
    print(f"    base move in opt_A: {base_in_a/tot:.3f}   in opt_B(counterfactual): {base_in_b/tot:.3f}")
    print(f"    patched move in opt_A: {patch_in_a/tot:.3f}   in opt_B: {patch_in_b/tot:.3f}")
    print(f"    moved off base move: {moved_off_a/tot:.3f}")
    print(f"    CLEAN FLIP (was A-optimal -> became B-optimal & not A): {flipped/tot:.3f}  ({flipped}/{tot})")
    return dict(base_in_a=base_in_a/tot, patch_in_b=patch_in_b/tot, flip=flipped/tot, tot=tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wide_shallow")
    ap.add_argument("--t", type=int, default=2)
    args = ap.parse_args()
    torch.manual_seed(0)
    d = torch.load(f"{RUNS}/evalset.pt", weights_only=False)
    occ = d["occ"].numpy(); true = d["true"].numpy()
    valid = d["valid"].numpy(); my = d["my_move"].numpy()
    model, ck = load(args.tag)
    man = Manual(model)
    print(f"=== RUBBER-HAND: {args.tag} ({ck['config']['n_layer']}x{ck['config']['d_model']}, "
          f"move_acc={ck['move_acc']:.3f}), decision round t={args.t} ===")
    print("\n(A) single-location patch (per layer), scale=1:")
    for li in range(man.nl):
        rubber_hand(man, occ, true, valid, my, t=args.t, patch_layers=str(li))
    print("\n(B) all-layers consistent patch, scale sweep:")
    for sc in (1.0, 2.0, 4.0):
        print(f"  -- scale={sc} --")
        rubber_hand(man, occ, true, valid, my, t=args.t, patch_layers="all", scale=sc)
    print("\n(C) patch ONLY the earliest layer (L0), scale sweep (most upstream):")
    for sc in (1.0, 2.0, 4.0):
        print(f"  -- scale={sc} --")
        rubber_hand(man, occ, true, valid, my, t=args.t, patch_layers="0", scale=sc)


if __name__ == "__main__":
    main()
