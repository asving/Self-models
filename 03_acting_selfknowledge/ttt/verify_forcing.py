"""Verify the forcing.

(1) Occupancy-ambiguity: the SAME color-blind occupancy occurs with DIFFERENT
    true ownerships and DIFFERENT optimal-move sets.  If so, occupancy alone is
    insufficient -> the net must attribute marks -> must recall its own moves.

(2) Occupancy-only ceiling: the BEST a memoryless map occ->move can achieve.
    Computed two ways:
      (a) Analytic Bayes-optimal ceiling: for each observed occupancy, the best
          single move maximizes P(move in optimal set | occupancy) under the data
          distribution.  This is the theoretical ceiling for any occ-only model.
      (b) Trained OccOnlyBaseline MLP (sanity check it approaches the ceiling).
    Report both, and contrast with the full model later.
"""
import collections
import numpy as np
import torch
import torch.nn.functional as F

import ttt
from data import build_dataset
from model import OccOnlyBaseline


def occ_key(occ):
    return tuple(int(x) for x in occ)


def occupancy_ambiguity(d):
    """Scan dataset rounds; group by occupancy; report ambiguous occupancies."""
    occ = d["occ"].numpy()
    true = d["true"].numpy()
    valid = d["valid"].numpy()
    my = d["my_move"].numpy()
    by_occ = collections.defaultdict(lambda: {"trues": set(), "opts": set(),
                                              "moves": set(), "count": 0})
    # recompute optimal set from true board (exact)
    for b in range(occ.shape[0]):
        for r in range(occ.shape[1]):
            if valid[b, r] == 0:
                continue
            k = occ_key(occ[b, r])
            tb = tuple(int(x) for x in true[b, r])
            opt = tuple(sorted(ttt.optimal_moves(list(tb), 1)))
            rec = by_occ[k]
            rec["trues"].add(tb)
            rec["opts"].add(opt)
            rec["moves"].add(int(my[b, r]))
            rec["count"] += 1
    n_occ = len(by_occ)
    amb_true = [k for k, v in by_occ.items() if len(v["trues"]) > 1]
    amb_opt = [k for k, v in by_occ.items() if len(v["opts"]) > 1]
    print(f"distinct occupancies seen: {n_occ}")
    print(f"  with >1 true ownership : {len(amb_true)} "
          f"({100*len(amb_true)/n_occ:.1f}%)")
    print(f"  with >1 optimal-set    : {len(amb_opt)} "
          f"({100*len(amb_opt)/n_occ:.1f}%)")
    # show a couple of concrete examples (with >=2 marks so it's nontrivial)
    print("\nExample ambiguous occupancies (same marks, different truth & optimal):")
    shown = 0
    for k in amb_opt:
        v = by_occ[k]
        if sum(k) >= 2 and v["count"] >= 5:
            print(f"  occ={k} ->")
            for tb in list(v["trues"])[:4]:
                opt = sorted(ttt.optimal_moves(list(tb), 1))
                print(f"      true={tb}  optimal={opt}")
            shown += 1
        if shown >= 3:
            break
    return by_occ, amb_opt


def analytic_ceiling(by_occ, d):
    """Bayes-optimal occ-only move accuracy against the optimal SET.

    For each round, 'correct' = chosen move is in that round's optimal set.
    An occ-only predictor must pick ONE move per occupancy. The best fixed move
    for occupancy k maximizes the fraction of rounds (with that occ) whose
    optimal set contains it.  We weight by occurrence counts.
    """
    occ = d["occ"].numpy(); true = d["true"].numpy(); valid = d["valid"].numpy()
    # accumulate per-occ: count of rounds, and for each cell how often it's optimal
    stat = collections.defaultdict(lambda: [0, np.zeros(9)])
    for b in range(occ.shape[0]):
        for r in range(occ.shape[1]):
            if valid[b, r] == 0:
                continue
            k = occ_key(occ[b, r])
            tb = list(int(x) for x in true[b, r])
            opt = ttt.optimal_moves(tb, 1)
            stat[k][0] += 1
            for m in opt:
                stat[k][1][m] += 1
    total = 0; correct = 0
    for k, (cnt, opthits) in stat.items():
        # best fixed move = argmax opthits (restricted to legal i.e. occ==0)
        legal = np.array([1.0 if c == 0 else 0.0 for c in k])
        scores = opthits * legal + (-1e9) * (1 - legal)
        best = scores.max()
        total += cnt
        correct += best
    return correct / total


def train_baseline(d_train, d_test, steps=4000, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = OccOnlyBaseline().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    occ = d_train["occ"].to(device); tgt = d_train["target"].to(device)
    legal = d_train["legal"].to(device); valid = d_train["valid"].to(device)
    B, L, _ = occ.shape
    occ_f = occ.reshape(B * L, 9); tgt_f = tgt.reshape(B * L, 9)
    legal_f = legal.reshape(B * L, 9); valid_f = valid.reshape(B * L)
    idx = valid_f > 0
    occ_f, tgt_f, legal_f = occ_f[idx], tgt_f[idx], legal_f[idx]
    for s in range(steps):
        logits = model(occ_f)
        logits = logits.masked_fill(legal_f == 0, -1e9)
        logp = F.log_softmax(logits, -1)
        loss = -(tgt_f * logp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return eval_baseline(model, d_test, device)


@torch.no_grad()
def eval_baseline(model, d, device):
    occ = d["occ"].to(device); tgt = d["target"].to(device)
    legal = d["legal"].to(device); valid = d["valid"].to(device)
    B, L, _ = occ.shape
    occ_f = occ.reshape(B * L, 9); tgt_f = tgt.reshape(B * L, 9)
    legal_f = legal.reshape(B * L, 9); valid_f = valid.reshape(B * L)
    idx = valid_f > 0
    logits = model(occ_f[idx]).masked_fill(legal_f[idx] == 0, -1e9)
    pred = logits.argmax(1)
    in_opt = tgt_f[idx][torch.arange(idx.sum()), pred] > 0
    return in_opt.float().mean().item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Building data for forcing verification...")
    d = build_dataset(20000, seed=7, p_strong=0.5)
    d_test = build_dataset(4000, seed=99, p_strong=0.5)
    print("\n=== (1) OCCUPANCY AMBIGUITY ===")
    by_occ, amb = occupancy_ambiguity(d)
    print("\n=== (2a) ANALYTIC OCC-ONLY CEILING (Bayes-optimal memoryless) ===")
    ceil = analytic_ceiling(by_occ, d)
    print(f"  analytic occ-only ceiling move-acc = {ceil:.4f}")
    print("\n=== (2b) TRAINED OCC-ONLY BASELINE MLP ===")
    acc = train_baseline(d, d_test, steps=4000, device=device)
    print(f"  trained occ-only baseline move-acc = {acc:.4f}")
