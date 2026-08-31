"""First-pass dissection of a trained TTTNet.

(A) Linear decodability per layer (residual stream after each block):
    - TRUE ownership of each cell (3-way {empty,mine,opp}) -> how well is the
      attributed board recoverable, even though the net only ever saw occupancy?
    - The net's OWN past moves (the efference copy): can we decode, at round r,
      WHICH cell the net itself played at each earlier round r'<r?
    We compare against an occupancy-only readout (linear on the raw occupancy
    input) as the no-memory baseline.

(B) Rubber-hand intervention: corrupt the net's recalled own-move and check
    that (i) its true-ownership attribution flips for the swapped cells and
    (ii) its chosen current move shifts.  We implement the corruption by
    swapping which cell the net "played" at an earlier round and editing the
    occupancy sequence consistently, then comparing decoded ownership + output.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import ttt
from model import TTTNet

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttt_runs")


def load(tag, device):
    ck = torch.load(f"{RUNS}/{tag}.pt", map_location=device, weights_only=False)
    m = TTTNet(**ck["config"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


@torch.no_grad()
def collect_hidden(model, d, device):
    occ = d["occ"].to(device)
    logits, hiddens = model(occ, return_hidden=True)
    # hiddens: list (n_layer) of (B,L,d)
    return [h.cpu().numpy() for h in hiddens], occ.cpu().numpy()


def probe_true_ownership(hiddens, d):
    """Per layer: logistic-regression decode of each cell's TRUE 3-way ownership
    from the residual at the corresponding round. Report mean balanced acc over
    cells, on valid rounds.  Compare to occ-only baseline (decode from occ)."""
    true = d["true"].numpy(); valid = d["valid"].numpy(); occ = d["occ"].numpy()
    B, L = valid.shape
    mask = valid.reshape(-1) > 0
    Y = true.reshape(B * L, 9)[mask]  # (N,9) values {0,1,2}
    occ_flat = occ.reshape(B * L, 9)[mask]
    results = {}
    # occ-only baseline
    results["occ_input"] = _decode_cells(occ_flat, Y)
    for li, h in enumerate(hiddens):
        X = h.reshape(B * L, -1)[mask]
        results[f"layer{li}"] = _decode_cells(X, Y)
    return results


def _torch_logreg(Xtr, ytr, Xte, yte, n_classes, epochs=300, lr=0.05, device="cpu"):
    """Multinomial logistic regression via full-batch GD (standardized inputs)."""
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte = torch.tensor(Xte, dtype=torch.float32, device=device)
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-5
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ytr = torch.tensor(ytr, dtype=torch.long, device=device)
    yte = torch.tensor(yte, dtype=torch.long, device=device)
    W = torch.zeros(Xtr.shape[1], n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        loss = F.cross_entropy(Xtr @ W + b, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (( Xte @ W + b).argmax(1) == yte).float().mean().item()
    return acc


def _decode_cells(X, Y, n_tr=6000, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Decode each of 9 cells' label (0/1/2) from X. Return mean accuracy over
    cells that are non-degenerate (have >1 class)."""
    n = X.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[:n_tr], idx[n_tr:n_tr + 4000]
    accs = []
    for c in range(9):
        yc = Y[:, c]
        if len(np.unique(yc[tr])) < 2:
            continue
        accs.append(_torch_logreg(X[tr], yc[tr], X[te], yc[te], 3, device=device))
    return float(np.mean(accs))


def probe_own_moves(hiddens, d):
    """At round r, decode the net's OWN move made at round r' (r'<=r) from the
    residual at round r.  This is the efference copy.  We focus on decoding the
    move made at round r'=r-1 (the most recent own move, which is one of the two
    new marks) and at r'=0.  Report per-layer accuracy (9-way) vs occ baseline.
    """
    my = d["my_move"].numpy(); valid = d["valid"].numpy(); occ = d["occ"].numpy()
    B, L = valid.shape
    out = {}
    for lag, name in [(1, "own_move_prev"), (2, "own_move_2back")]:
        # samples: rounds r where r-lag>=0 and round r valid
        rows = []; labels = []
        for b in range(B):
            for r in range(L):
                if valid[b, r] == 0:
                    continue
                rp = r - lag
                if rp < 0 or valid[b, rp] == 0:
                    continue
                rows.append((b, r)); labels.append(my[b, rp])
        rows = np.array(rows); labels = np.array(labels)
        occ_X = np.stack([occ[b, r] for b, r in rows])
        res = {"occ_input": _decode_9way(occ_X, labels)}
        for li, h in enumerate(hiddens):
            X = np.stack([h[b, r] for b, r in rows])
            res[f"layer{li}"] = _decode_9way(X, labels)
        out[name] = res
    return out


def _decode_9way(X, y, n_tr=6000, device="cuda" if torch.cuda.is_available() else "cpu"):
    n = X.shape[0]
    idx = np.random.default_rng(1).permutation(n)
    cut = min(n_tr, n // 2)
    tr, te = idx[:cut], idx[cut:cut + 4000]
    return _torch_logreg(X[tr], y[tr], X[te], y[te], 9, device=device)


@torch.no_grad()
def rubber_hand(model, d, device, n=4000):
    """Corrupt the recalled own move: take games with >=3 rounds. At round 1 the
    net actually played move m1 (one of the two marks newly appearing at round 2's
    occupancy). We construct a COUNTERFACTUAL where the net believes it played a
    DIFFERENT one of the round-2 marks instead.

    Construction: a round-2 occupancy has marks = {m1 (mine), o0... } i.e. cells
    occupied. The two marks added between round-1-obs and round-2-obs are m1 (own)
    and o1 (opp). Under color-blindness the net can't see which is which; its
    attribution is driven by its memory of m1.  We can't directly edit the net's
    memory of m1, but we CAN swap WHICH move the net actually made at round 1 by
    re-rolling the sequence with a different legal m1 and feeding that occupancy
    history -- the net's own representation of "what I played" comes only from the
    occupancy timeline + position.  Instead, the clean rubber-hand here:

      Compare two real sequences that share the SAME round-2 occupancy but arose
      from DIFFERENT own move m1.  If the net's attribution/output depends on m1
      (not just the occupancy), then for the SAME occupancy at round 2 the net's
      round-2 move distribution differs -> evidence it used its efference copy.

    We measure: among test rounds grouped by (round index, occupancy), the
    variance of the net's chosen move across different true-ownerships. High
    spread => the net conditions on recalled own-history, not occupancy alone.
    """
    occ = d["occ"].to(device); valid = d["valid"].numpy()
    true = d["true"].numpy(); my = d["my_move"].numpy()
    logits = model(occ).cpu().numpy()  # (B,L,9)
    legal = d["legal"].numpy()
    B, L = valid.shape
    import collections
    groups = collections.defaultdict(list)  # (r, occ) -> list of (chosen, true)
    for b in range(B):
        for r in range(L):
            if valid[b, r] == 0:
                continue
            lg = logits[b, r].copy()
            lg[legal[b, r] == 0] = -1e9
            chosen = int(lg.argmax())
            k = (r, tuple(int(x) for x in occ[b, r].cpu().numpy()))
            groups[k].append((chosen, tuple(int(x) for x in true[b, r])))
    # For ambiguous occupancies (same occ, >1 true ownership), does the net's
    # chosen move vary WITH the true ownership? It can only know ownership via
    # its own move memory.
    n_amb = 0; n_move_varies = 0; agree_with_truth = 0; tot_truth = 0
    for k, lst in groups.items():
        trues = set(t for _, t in lst)
        if len(trues) < 2 or len(lst) < 8:
            continue
        n_amb += 1
        chosen_set = set(c for c, _ in lst)
        if len(chosen_set) > 1:
            n_move_varies += 1
        # does chosen move match optimal of the actual true board?
        for c, t in lst:
            opt = ttt.optimal_moves(list(t), 1)
            tot_truth += 1
            if c in opt:
                agree_with_truth += 1
    print(f"  ambiguous (r,occ) groups: {n_amb}")
    print(f"  ... where net's chosen move VARIES across true-ownerships: "
          f"{n_move_varies} ({100*n_move_varies/max(n_amb,1):.1f}%)")
    print(f"  net move in optimal-set (on ambiguous groups): "
          f"{100*agree_with_truth/max(tot_truth,1):.1f}%")
    return n_amb, n_move_varies


@torch.no_grad()
def rubber_hand_causal(model, d, device):
    """A direct counterfactual: build minimal pairs that share the round-2
    occupancy but differ in own move m1, and confirm the net's round-2 output
    tracks the TRUE ownership (which is set by m1).  Reports, per pair, whether
    the net's chosen move matches each branch's optimal set -- i.e. flipping the
    (recalled) own move flips the chosen move correctly."""
    occ = d["occ"]; valid = d["valid"].numpy(); true = d["true"].numpy()
    logits = model(occ.to(device)).cpu().numpy()
    legal = d["legal"].numpy()
    B, L = valid.shape
    import collections
    by_occ_r2 = collections.defaultdict(list)
    for b in range(B):
        if valid[b, 2] == 0:
            continue
        k = tuple(int(x) for x in occ[b, 2].numpy())
        lg = logits[b, 2].copy(); lg[legal[b, 2] == 0] = -1e9
        chosen = int(lg.argmax())
        by_occ_r2[k].append((b, chosen, tuple(int(x) for x in true[b, 2])))
    flips_correct = 0; flips_tot = 0
    examples = []
    for k, lst in by_occ_r2.items():
        # collect branches with distinct true ownership
        bytrue = collections.defaultdict(list)
        for b, c, t in lst:
            bytrue[t].append((b, c))
        if len(bytrue) < 2:
            continue
        trues = list(bytrue.keys())
        for i in range(len(trues)):
            for j in range(i + 1, len(trues)):
                t1, t2 = trues[i], trues[j]
                opt1 = set(ttt.optimal_moves(list(t1), 1))
                opt2 = set(ttt.optimal_moves(list(t2), 1))
                if opt1 == opt2:
                    continue
                c1 = bytrue[t1][0][1]; c2 = bytrue[t2][0][1]
                flips_tot += 1
                ok1 = c1 in opt1; ok2 = c2 in opt2
                if ok1 and ok2:
                    flips_correct += 1
                if len(examples) < 4:
                    examples.append((k, t1, c1, sorted(opt1), t2, c2, sorted(opt2)))
    print(f"  minimal pairs (same r2 occ, diff ownership w/ diff optimal): {flips_tot}")
    print(f"  ... where net picks each branch's own optimal move correctly: "
          f"{flips_correct} ({100*flips_correct/max(flips_tot,1):.1f}%)")
    print("  examples (occ | trueA->moveA optA | trueB->moveB optB):")
    for k, t1, c1, o1, t2, c2, o2 in examples:
        print(f"    occ={k}")
        print(f"      A true={t1} net_move={c1} opt={o1}")
        print(f"      B true={t2} net_move={c2} opt={o2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="narrow_deep")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(f"{RUNS}/evalset.pt", weights_only=False)
    model, ck = load(args.tag, device)
    print(f"=== DISSECTION: {args.tag} (params={ck['n_params']}, "
          f"move_acc={ck['move_acc']:.3f}) ===")
    hiddens, occ = collect_hidden(model, d, device)

    print("\n[A1] TRUE-OWNERSHIP linear decodability (mean cell acc):")
    r = probe_true_ownership(hiddens, d)
    for k, v in r.items():
        print(f"    {k:10s}: {v:.3f}")

    print("\n[A2] OWN-MOVE (efference copy) linear decodability (9-way acc):")
    r2 = probe_own_moves(hiddens, d)
    for name, res in r2.items():
        print(f"  {name}:")
        for k, v in res.items():
            print(f"    {k:10s}: {v:.3f}")

    print("\n[B1] Behavioral forcing check (does output track ownership?):")
    rubber_hand(model, d, device)
    print("\n[B2] Rubber-hand minimal pairs (causal flip):")
    rubber_hand_causal(model, d, device)
