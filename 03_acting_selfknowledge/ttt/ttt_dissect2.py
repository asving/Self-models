"""Second-pass dissection of the color-blind per-round tic-tac-toe self-model nets.

Central question: does the net carry a LOCALIZED, CAUSAL, CORRUPTIBLE stored
efference copy of its own past moves, or does it RE-DERIVE its own moves from the
occupancy history each step (deterministic-policy => stored copy epiphenomenal,
as in d2/cont_c8)?

Tasks:
 1. SUBSPACES per layer (linear probes): policy / own-move efference copy /
    ownership-attribution (ONLY on the ambiguous occupancy subset).
 2. STORE vs RE-DERIVE (causal): attention ablation to past own-move positions;
    own-move subspace projection-out (load-bearing test).
 3. RUBBER-HAND: minimal pairs (same occ, different own-move -> different optimal),
    internally patch the recalled own-move representation, see if chosen move flips.
 4. Depth: narrow_deep vs wide_shallow -- shallow re-derive vs deep build-up.

CPU-fine.  Run: CUDA_VISIBLE_DEVICES="" ~/comp_icl/.venv/bin/python ttt_dissect2.py --tag wide_shallow
"""
import argparse
import collections
import numpy as np
import torch
import torch.nn.functional as F
import ttt
from model import TTTNet

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttt_runs")
DEV = "cpu"


def load(tag):
    ck = torch.load(f"{RUNS}/{tag}.pt", map_location=DEV, weights_only=False)
    m = TTTNet(**ck["config"]).to(DEV)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


# ---------------------------------------------------------------------------
# Ridge / logistic helpers
# ---------------------------------------------------------------------------
def ridge_logreg(Xtr, ytr, Xte, yte, n_classes, epochs=400, lr=0.05, wd=1e-3):
    Xtr = torch.tensor(Xtr, dtype=torch.float32); Xte = torch.tensor(Xte, dtype=torch.float32)
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-5
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ytr = torch.tensor(ytr, dtype=torch.long); yte = torch.tensor(yte, dtype=torch.long)
    W = torch.zeros(Xtr.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        loss = F.cross_entropy(Xtr @ W + b, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((Xte @ W + b).argmax(1) == yte).float().mean().item()
    return acc


def split(n, frac=0.6, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * frac)
    return idx[:cut], idx[cut:]


@torch.no_grad()
def collect_hidden(model, occ):
    occ_t = torch.tensor(occ, dtype=torch.float32)
    logits, hiddens = model(occ_t, return_hidden=True)
    return [h.numpy() for h in hiddens], logits.numpy()


# ---------------------------------------------------------------------------
# TASK 1: subspaces
# ---------------------------------------------------------------------------
def gather_rows(valid):
    B, L = valid.shape
    rows = [(b, r) for b in range(B) for r in range(L) if valid[b, r] > 0]
    return np.array(rows)


def probe_policy(hiddens, occ, logits, d, rows):
    """Decode the net's OWN chosen move (argmax over legal logits) from residual.
    This is the policy subspace: how linearly is the net's decision present."""
    legal = (occ == 0).astype(np.float32)
    chosen = []
    for (b, r) in rows:
        lg = logits[b, r].copy(); lg[legal[b, r] == 0] = -1e9
        chosen.append(int(lg.argmax()))
    chosen = np.array(chosen)
    tr, te = split(len(rows), seed=2)
    res = {}
    occ_X = np.stack([occ[b, r] for b, r in rows])
    res["occ_input"] = ridge_logreg(occ_X[tr], chosen[tr], occ_X[te], chosen[te], 9)
    for li, h in enumerate(hiddens):
        X = np.stack([h[b, r] for b, r in rows])
        res[f"L{li}"] = ridge_logreg(X[tr], chosen[tr], X[te], chosen[te], 9)
    return res, chosen


def probe_own_move(hiddens, occ, my, valid, lag):
    """Decode own move made at round r-lag from residual at round r (efference copy)."""
    B, L = valid.shape
    rows = []; labels = []
    for b in range(B):
        for r in range(L):
            if valid[b, r] == 0: continue
            rp = r - lag
            if rp < 0 or valid[b, rp] == 0: continue
            rows.append((b, r)); labels.append(my[b, rp])
    rows = np.array(rows); labels = np.array(labels)
    tr, te = split(len(rows), seed=3)
    # majority baseline
    maj = np.bincount(labels[tr], minlength=9).argmax()
    res = {"majority": (labels[te] == maj).mean()}
    occ_X = np.stack([occ[b, r] for b, r in rows])
    res["occ_input"] = ridge_logreg(occ_X[tr], labels[tr], occ_X[te], labels[te], 9)
    for li, h in enumerate(hiddens):
        X = np.stack([h[b, r] for b, r in rows])
        res[f"L{li}"] = ridge_logreg(X[tr], labels[tr], X[te], labels[te], 9)
    return res


def ambiguous_mask(occ, true, valid):
    """Per (round, occupancy): mark rows whose occupancy occurs with >1 true ownership."""
    B, L = valid.shape
    groups = collections.defaultdict(set)
    for b in range(B):
        for r in range(L):
            if valid[b, r] == 0: continue
            groups[(r, tuple(occ[b, r].astype(int)))].add(tuple(true[b, r]))
    mask = np.zeros((B, L), bool)
    for b in range(B):
        for r in range(L):
            if valid[b, r] == 0: continue
            if len(groups[(r, tuple(occ[b, r].astype(int)))]) > 1:
                mask[b, r] = True
    return mask


def probe_ownership(hiddens, occ, true, valid, amb_only=True):
    """Decode each cell's TRUE 3-way ownership {0 empty,1 mine,2 opp} from residual.
    On the ambiguous subset only, the occupancy cannot reveal ownership by itself,
    so a high number = efference-derived attribution. Report mean over cells that
    are non-degenerate AND ambiguous (cell value not determined by occ)."""
    B, L = valid.shape
    amask = ambiguous_mask(occ, true, valid) if amb_only else (valid > 0)
    rows = [(b, r) for b in range(B) for r in range(L) if valid[b, r] > 0 and amask[b, r]]
    rows = np.array(rows)
    Y = np.stack([true[b, r] for b, r in rows])          # (N,9) in {0,1,2}
    occ_X = np.stack([occ[b, r] for b, r in rows])
    tr, te = split(len(rows), seed=4)

    def decode_cells(X):
        accs = []
        for c in range(9):
            yc = Y[:, c]
            if len(np.unique(yc[tr])) < 2:
                continue
            # restrict to cells that are genuinely ambiguous: cell is marked (occ==1)
            # but could be mine or opp. Empty cells (occ==0) are trivially "empty".
            # We decode the FULL 3-way but only count cells where, among marked rows,
            # both 1 and 2 appear (the attribution-hard cells).
            marked = occ_X[:, c] == 1
            if marked.sum() < 50:
                continue
            vals = np.unique(yc[marked])
            if not (1 in vals and 2 in vals):
                continue
            accs.append(ridge_logreg(X[tr], yc[tr], X[te], yc[te], 3))
        return float(np.mean(accs)) if accs else float("nan")

    res = {"occ_input": decode_cells(occ_X)}
    for li, h in enumerate(hiddens):
        X = np.stack([h[b, r] for b, r in rows])
        res[f"L{li}"] = decode_cells(X)
    return res, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wide_shallow")
    args = ap.parse_args()
    torch.manual_seed(0)
    d = torch.load(f"{RUNS}/evalset.pt", weights_only=False)
    occ = d["occ"].numpy(); true = d["true"].numpy(); valid = d["valid"].numpy()
    my = d["my_move"].numpy()
    model, ck = load(args.tag)
    dm = ck["config"]["d_model"]
    print(f"=== SUBSPACES: {args.tag} (params={ck.get('params','?')}, "
          f"layers={ck['config']['n_layer']}x{dm}, move_acc={ck['move_acc']:.3f}) ===")
    hiddens, logits = collect_hidden(model, occ)
    rows = gather_rows(valid)

    print("\n[1a] POLICY: decode net's own chosen move (9-way) from residual:")
    pol, chosen = probe_policy(hiddens, occ, logits, dm, rows)
    for k, v in pol.items():
        print(f"    {k:10s}: {v:.3f}")

    print("\n[1b] EFFERENCE COPY: decode own move m_(r-lag) (9-way):")
    for lag in (1, 2):
        r = probe_own_move(hiddens, occ, my, valid, lag)
        print(f"  lag={lag} (m_(r-{lag})):")
        for k, v in r.items():
            print(f"    {k:10s}: {v:.3f}")

    print("\n[1c] OWNERSHIP attribution (3-way per cell, AMBIGUOUS subset only):")
    own, n = probe_ownership(hiddens, occ, true, valid, amb_only=True)
    print(f"    (n ambiguous rows = {n})")
    for k, v in own.items():
        print(f"    {k:10s}: {v:.3f}")


if __name__ == "__main__":
    main()
