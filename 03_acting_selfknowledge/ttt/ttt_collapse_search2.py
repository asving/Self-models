"""Follow-ups to ttt_collapse_search.py.

THREAD 1: characterize the RL choice at ties -- positional bias (center/corner/edge,
lowest-index canonical ordering), opening move, symmetry.

CONFOUND CHECK for thread 2: legibility correlates with what positional feature?
Since the FULLOBS net (which cannot care about self-legibility -- it sees the truth)
picks MORE-legible moves than the colorblind net, any legibility "preference" is
likely a downstream correlate of a positional/board feature both nets select on.
We decompose: is legibility just (a) center>corner>edge, (b) lowest cell index,
(c) occupancy-count? Then ask: controlling for the positional choice profile, does
either net show RESIDUAL legibility preference?

THREAD 4: does the net's chosen move at a tie predict downstream decodability of the
true board in the on-policy rollout? (correlate argmax-move legibility with whether
the realized future board is decodable from occupancy alone).

CPU only.
"""
import collections, json, math
import numpy as np
import torch, torch.nn.functional as F
import ttt
from model import TTTNet
import policy_eval as PE
from ttt_collapse_search import (
    enumerate_contexts, build_legibility, child_legibility_of_move,
    net_policy_at, load, RUNS)

CENTER = {4}
CORNERS = {0, 2, 6, 8}
EDGES = {1, 3, 5, 7}


def postype(cell):
    if cell in CENTER: return "center"
    if cell in CORNERS: return "corner"
    return "edge"


def main():
    contexts = enumerate_contexts(p_strong=0.5)
    key_H = build_legibility(contexts)
    ties = {c["true_seq"]: c for c in contexts if len(c["opt"]) > 1}
    uties = list(ties.values())
    for c in uties:
        c["move_leg"] = {nm: child_legibility_of_move(
            list(c["board"]), nm, c["occ_seq"], c["round"], key_H) for nm in c["opt"]}
        v = np.array(list(c["move_leg"].values()))
        c["leg_spread"] = float(v.max() - v.min())

    nets = {"rl": load(f"{RUNS}/rl.pt"),
            "rl_fullobs": load(f"{RUNS}/rl_fullobs.pt"),
            "onpolicy_teacher": load(f"{RUNS}/onpolicy_teacher.pt")}

    # ---- THREAD 1: positional choice profile at ALL ties (argmax move) ----
    print("="*70)
    print("THREAD 1: choice profile at tie positions (argmax over optimal set)")
    print("="*70)
    for name, (model, fullobs) in nets.items():
        cnt = collections.Counter()
        lowidx = 0
        by_round = collections.defaultdict(collections.Counter)
        n = 0
        for c in uties:
            p = net_policy_at(model, fullobs, c["true_seq"], list(c["board"]))
            opt = list(c["opt"])
            pp = np.array([p[m] for m in opt])
            mv = opt[int(pp.argmax())]
            cnt[postype(mv)] += 1
            lowidx += int(mv == min(opt))
            by_round[c["round"]][postype(mv)] += 1
            n += 1
        # what's AVAILABLE in optimal sets (baseline rate of each postype among optima)
        avail = collections.Counter()
        for c in uties:
            for m in c["opt"]:
                avail[postype(m)] += 1
        tot_avail = sum(avail.values())
        print(f"\n  {name}: n_tie={n}")
        print(f"    chosen postype:  " + "  ".join(
            f"{k}={cnt[k]/n:.3f}" for k in ["center", "corner", "edge"]))
        print(f"    available postype(among optima): " + "  ".join(
            f"{k}={avail[k]/tot_avail:.3f}" for k in ["center", "corner", "edge"]))
        print(f"    picks-lowest-index frac: {lowidx/n:.3f}  (chance≈{np.mean([1/len(c['opt']) for c in uties]):.3f})")

    # opening move (round 0, empty board): single context
    print("\n  --- opening move (empty board) argmax + full optimal-set policy ---")
    empt = [c for c in uties if c["round"] == 0]
    # empty board is a tie context only if optimal set>1 -- TTT: all 9 are optimal (draw)
    c0 = None
    for c in contexts:
        if c["round"] == 0:
            c0 = c; break
    for name, (model, fullobs) in nets.items():
        p = net_policy_at(model, fullobs, c0["true_seq"], list(c0["board"]))
        mv = int(p.argmax())
        print(f"    {name:18s} opening argmax={mv} ({postype(mv)})  p[center]={p[4]:.3f} "
              f"p[corner_mean]={np.mean([p[i] for i in CORNERS]):.3f} "
              f"p[edge_mean]={np.mean([p[i] for i in EDGES]):.3f}")

    # ---- CONFOUND: legibility vs postype ----
    print("\n" + "="*70)
    print("CONFOUND: is per-move self-legibility just a positional correlate?")
    print("="*70)
    # collect (legibility, postype) over all moves in discriminating ties
    disc = [c for c in uties if c["leg_spread"] > 1e-4]
    by_pt = collections.defaultdict(list)
    rank_by_pt = collections.defaultdict(list)
    for c in disc:
        opt = list(c["opt"]); legs = np.array([c["move_leg"][m] for m in opt])
        order = np.argsort(legs); ranks = np.empty(len(legs)); ranks[order] = np.linspace(0,1,len(legs))
        for j, m in enumerate(opt):
            by_pt[postype(m)].append(legs[j]); rank_by_pt[postype(m)].append(ranks[j])
    for pt in ["center", "corner", "edge"]:
        if by_pt[pt]:
            print(f"    {pt:7s}: mean legibility(-H)={np.mean(by_pt[pt]):.4f}  "
                  f"mean within-context legibility-rank={np.mean(rank_by_pt[pt]):.3f}  "
                  f"n={len(by_pt[pt])}")

    # Within-context: is the MOST-legible move usually center? corner?
    most_leg_pt = collections.Counter()
    for c in disc:
        opt = list(c["opt"]); legs = np.array([c["move_leg"][m] for m in opt])
        most_leg_pt[postype(opt[int(legs.argmax())])] += 1
    print("    most-legible move postype distribution: " + "  ".join(
        f"{k}={most_leg_pt[k]/len(disc):.3f}" for k in ["center","corner","edge"]))

    # ---- residual legibility preference controlling for postype ----
    # For each net: among discriminating ties, restrict to contexts where the
    # most-legible move and least-legible move have the SAME postype (so postype
    # can't explain the choice). Does net still prefer the more-legible one?
    print("\n  Residual legibility preference (ties where most/least-legible "
          "moves share postype, so postype is controlled):")
    for name, (model, fullobs) in nets.items():
        rank_acc = []
        nctl = 0
        for c in disc:
            opt = list(c["opt"]); legs = np.array([c["move_leg"][m] for m in opt])
            hi, lo = int(legs.argmax()), int(legs.argmin())
            if postype(opt[hi]) != postype(opt[lo]):
                continue
            # restrict to the two extreme moves with same postype
            p = net_policy_at(model, fullobs, c["true_seq"], list(c["board"]))
            phi, plo = p[opt[hi]], p[opt[lo]]
            if phi + plo <= 0: continue
            rank_acc.append(phi / (phi + plo))  # >0.5 = prefers more legible
            nctl += 1
        print(f"    {name:18s} P(mass on more-legible | extremes same postype) "
              f"= {np.mean(rank_acc):.3f}  (n={nctl}, 0.5=neutral)")

    print("\nDone.")


if __name__ == "__main__":
    main()
