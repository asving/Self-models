"""Decode the TRUE tic-tac-toe board (9 cells x 3-way ownership {empty,mine,opp})
from the residual stream of the trained color-blind TTT nets.

Foundation for the circuit analysis: where (if anywhere) does each color-blind net
reconstruct the true X/O board from color-blind occupancy + memory of its own moves,
ABOVE what the raw occupancy already affords by X-first parity statistics?

Pipeline (per net):
 1. ON-POLICY eval set: roll the net out closed-loop (it SAMPLES its own moves;
    opponent = unpredictable 50% random / 50% minimax mix, as in ttt.py). Record per
    net decision: color-blind occupancy (MODEL INPUT), TRUE board {0,1,2}, legal mask,
    valid, own-move history. For rl_fullobs the model INPUT is the true board (control).
 2. Collect residual at each layer: embedding (inp+pos), each block output, and the
    post-final-LN stream. Fit probes residual -> true board, held-out split:
      LINEAR (multinomial ridge / softmax regression) and NON-LINEAR (small MLP).
    Report per-cell 3-way accuracy AND whole-board exact-match.
 3. Controls: (a) OCCUPANCY-ONLY baseline (decode true board straight from the raw
    color-blind occupancy the net sees); (b) AMBIGUOUS SUBSET (occupancies, keyed by
    round, that occur with >1 distinct true board) -- the clean efference-derived
    attribution measure. The residual's GAIN over the occupancy baseline, especially
    on the ambiguous subset, is what matters (not the raw number, which can be parity).

CPU-fine.  Run:
  CUDA_VISIBLE_DEVICES="" ~/comp_icl/.venv/bin/python ttt_decode_board.py
"""
import argparse
import collections
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ttt
import policy_eval as PE
from model import TTTNet

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttt_runs")
DEV = "cpu"
MAXR = 5  # X moves at most 5 times

# These are tiny matmuls; torch intra-op multithreading is pure overhead here and
# (on this shared box) spawns a thread storm. Keep it single-threaded.
torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load(tag):
    ck = torch.load(f"{RUNS}/{tag}.pt", map_location=DEV, weights_only=False)
    m = TTTNet(**ck["config"]).to(DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    fullobs = bool(ck.get("fullobs", False))
    return m, ck, fullobs


# ---------------------------------------------------------------------------
# Per-net ON-POLICY eval set
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout_evalset(model, fullobs, n_games, seed, p_strong=0.5, sample=True):
    """Roll the net out closed-loop, BATCHED in lockstep by round (all live games
    share round index r, so their occ-sequences (B,r+1,9) batch through one forward).
    For each net decision store:
       occ (9,) MODEL INPUT (color-blind occ, or true-ownership obs if fullobs),
       true (9,) {0:empty,1:mine,2:opp}, legal (9,), my_move (), valid.
    Sampling its own move (sample=True) gives coverage of distinct own-move histories;
    set sample=False to use argmax (collapsed nets give few distinct games either way).
    The net forward is batched; the (cached, cheap) minimax opponent steps per game."""
    rng = np.random.default_rng(seed)
    boards = [[0] * 9 for _ in range(n_games)]
    occ_seqs = [[] for _ in range(n_games)]       # per-game list of obs vectors
    alive = np.ones(n_games, bool)
    # per-game padded records
    OCC = np.zeros((n_games, MAXR, 9), np.float32)
    TRUE = np.zeros((n_games, MAXR, 9), np.int64)
    LEG = np.zeros((n_games, MAXR, 9), np.float32)
    MV = np.full((n_games, MAXR), -1, np.int64)
    VAL = np.zeros((n_games, MAXR), np.float32)

    for r in range(MAXR):
        # mark games that are over before this net decision
        for g in range(n_games):
            if not alive[g]:
                continue
            if ttt.winner(boards[g]) != 0 or not ttt.legal_moves(boards[g]):
                alive[g] = False
        live = np.where(alive)[0]
        if len(live) == 0:
            break
        # append current obs to each live game's sequence; record true/legal
        for g in live:
            obs = PE.obs_from_board(boards[g], fullobs)
            occ_seqs[g].append(obs)
            OCC[g, r] = obs
            TRUE[g, r] = np.array(boards[g], dtype=np.int64)
            LEG[g, r] = (np.array(boards[g]) == 0).astype(np.float32)
            VAL[g, r] = 1.0
        # batched forward: all live games have a length-(r+1) sequence
        inp = torch.tensor(np.stack([np.stack(occ_seqs[g]) for g in live]),
                           dtype=torch.float32)           # (Nlive, r+1, 9)
        logits = model(inp)[:, -1]                          # (Nlive, 9)
        legal = torch.tensor(np.stack([LEG[g, r] for g in live]))  # (Nlive,9)
        logits = logits.masked_fill(legal == 0, -1e9)
        probs = F.softmax(logits, -1).numpy()
        # choose + play net move, then opponent move, per game
        for k, g in enumerate(live):
            p = probs[k]
            mv = int(rng.choice(9, p=p)) if sample else int(p.argmax())
            MV[g, r] = mv
            boards[g][mv] = 1
            if ttt.winner(boards[g]) != 0 or not ttt.legal_moves(boards[g]):
                alive[g] = False
                continue
            omv = ttt.opponent_move(boards[g], rng, p_strong=p_strong)
            if omv is None:
                alive[g] = False
                continue
            boards[g][omv] = 2
    return {"occ": OCC, "true": TRUE, "legal": LEG, "my_move": MV, "valid": VAL}


# ---------------------------------------------------------------------------
# Residual collection: embedding, each block output, post-final-LN
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_residuals(model, occ):
    """Return ordered dict {layer_name: (B,L,d)} arrays + logits (unused here).
    Layers: 'emb' (inp+pos, pre-block), 'L0..L{n-1}' (block outputs), 'lnf' (post final LN).
    Re-implements the forward to expose the embedding, since model.forward only
    returns block outputs as hiddens."""
    occ_t = torch.tensor(occ, dtype=torch.float32)
    B, L, _ = occ_t.shape
    pos = torch.arange(L)
    x = model.inp(occ_t) + model.pos(pos)[None]
    out = collections.OrderedDict()
    out["emb"] = x.numpy().copy()
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    for li, blk in enumerate(model.blocks):
        x = blk(x, mask)
        out[f"L{li}"] = x.numpy().copy()
    out["lnf"] = model.lnf(x).numpy().copy()
    return out


# ---------------------------------------------------------------------------
# Probes: linear (multinomial softmax regression == ridge-style) and MLP.
# Fit 9 independent 3-way cell heads jointly (shared trunk for MLP; independent
# linear maps for the linear probe). Report per-cell 3-way acc and exact-match.
# ---------------------------------------------------------------------------
def _standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-5
    return (Xtr - mu) / sd, (Xte - mu) / sd


class MLPProbe(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, 27)

    def forward(self, x):
        return self.head(self.trunk(x)).reshape(-1, 9, 3)


# ---------------------------------------------------------------------------
# Row gathering / ambiguous subset
# ---------------------------------------------------------------------------
def gather_rows(valid):
    B, L = valid.shape
    return np.array([(b, r) for b in range(B) for r in range(L) if valid[b, r] > 0])


def ambiguous_mask(occ, true, valid):
    """Row is ambiguous if its occupancy (keyed by round) co-occurs with >1 distinct
    true board across the data. Keying by round respects X-first parity per round.
    NOTE: for fullobs nets the 'occ' IS the true board, so no row is ambiguous (the
    map occ->true is the identity) -- expected for the control."""
    B, L = valid.shape
    groups = collections.defaultdict(set)
    for b in range(B):
        for r in range(L):
            if valid[b, r] == 0:
                continue
            groups[(r, tuple(occ[b, r].astype(int)))].add(tuple(true[b, r].tolist()))
    mask = np.zeros((B, L), bool)
    for b in range(B):
        for r in range(L):
            if valid[b, r] == 0:
                continue
            if len(groups[(r, tuple(occ[b, r].astype(int)))]) > 1:
                mask[b, r] = True
    return mask


def split(n, frac=0.7, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(n * frac)
    return idx[:cut], idx[cut:]


# ---------------------------------------------------------------------------
# Decode driver
# ---------------------------------------------------------------------------
def decode_from_features(Xall, Yall, rowsel, do_mlp=True, seed=0):
    """Xall: (N,d) features for selected rows. Yall: (N,9). rowsel boolean over N
    (e.g. ambiguous). Returns dict of metrics restricted to rowsel."""
    Xall = torch.tensor(Xall, dtype=torch.float32)
    Yall = torch.tensor(Yall, dtype=torch.long)
    N = Xall.shape[0]
    tr, te = split(N, seed=seed)
    Xtr, Xte = _standardize(Xall[tr], Xall[te])
    Ytr, Yte = Yall[tr], Yall[te]
    # Train probe on the full (parity-rich) train split; report metrics BOTH on the
    # whole test split and restricted to the ambiguous test rows (clean attribution).
    te_sel = torch.tensor(rowsel[te])
    out = {"lin": _eval_subset(Xtr, Ytr, Xte, Yte, te_sel, kind="linear")}
    if do_mlp:
        out["mlp"] = _eval_subset(Xtr, Ytr, Xte, Yte, te_sel, kind="mlp")
    return out


def _eval_subset(Xtr, Ytr, Xte, Yte, te_sel, kind):
    if kind == "linear":
        d = Xtr.shape[1]
        W = torch.zeros(d, 27, requires_grad=True); b = torch.zeros(27, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=0.05, weight_decay=1e-3)
        for _ in range(300):
            logits = (Xtr @ W + b).reshape(-1, 9, 3)
            loss = F.cross_entropy(logits.reshape(-1, 3), Ytr.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = (Xte @ W + b).reshape(-1, 9, 3).argmax(-1)
    else:
        net = MLPProbe(Xtr.shape[1], 256)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        n = Xtr.shape[0]
        for _ in range(300):
            perm = torch.randperm(n)
            for i in range(0, n, 4096):
                idx = perm[i:i + 4096]
                logits = net(Xtr[idx])
                loss = F.cross_entropy(logits.reshape(-1, 3), Ytr[idx].reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(Xte).argmax(-1)
    correct = (pred == Yte)
    full = {"cell": correct.float().mean().item(), "exact": correct.all(1).float().mean().item()}
    if te_sel.any():
        cs = correct[te_sel]
        amb = {"cell": cs.float().mean().item(), "exact": cs.all(1).float().mean().item(),
               "n": int(te_sel.sum())}
    else:
        amb = {"cell": float("nan"), "exact": float("nan"), "n": 0}
    return {"full": full, "amb": amb}


def analyze_net(tag, n_games, seed, do_mlp=True):
    model, ck, fullobs = load(tag)
    cfg = ck["config"]
    print(f"\n{'='*78}\n=== {tag}  (layers={cfg['n_layer']}x{cfg['d_model']}, "
          f"fullobs={fullobs}, collapse_tie_H="
          f"{ck.get('collapse',{}).get('tie_entropy','?')}) ===\n{'='*78}", flush=True)
    ev = rollout_evalset(model, fullobs, n_games, seed)
    occ, true, valid = ev["occ"], ev["true"], ev["valid"]
    rows = gather_rows(valid)
    occ_rows = np.stack([occ[b, r] for b, r in rows])         # (N,9) model input
    true_rows = np.stack([true[b, r] for b, r in rows])       # (N,9) in {0,1,2}
    amask = ambiguous_mask(occ, true, valid)
    amb_rows = np.array([amask[b, r] for b, r in rows])
    n_amb = int(amb_rows.sum())
    # number of distinct (round,occ) keys & distinct games for context
    n_dist_games = len({tuple(map(tuple, occ[b][valid[b] > 0].astype(int)))
                        for b in range(occ.shape[0])})
    print(f"  rows={len(rows)}  ambiguous_rows={n_amb} ({100*n_amb/len(rows):.1f}%)  "
          f"distinct_games={n_dist_games}", flush=True)

    resid = collect_residuals(model, occ)
    feat_rows = collections.OrderedDict()
    feat_rows["occ_input"] = occ_rows
    for name, arr in resid.items():
        feat_rows[name] = np.stack([arr[b, r] for b, r in rows])

    results = collections.OrderedDict()
    for name, X in feat_rows.items():
        do_m = do_mlp
        r = decode_from_features(X, true_rows, amb_rows, do_mlp=do_m, seed=7)
        results[name] = r
        line = f"  {name:9s} | LIN full cell {r['lin']['full']['cell']:.3f} exact {r['lin']['full']['exact']:.3f}"
        line += f" | amb cell {r['lin']['amb']['cell']:.3f} exact {r['lin']['amb']['exact']:.3f}"
        if do_m:
            line += f"  ||  MLP full cell {r['mlp']['full']['cell']:.3f} exact {r['mlp']['full']['exact']:.3f}"
            line += f" | amb cell {r['mlp']['amb']['cell']:.3f} exact {r['mlp']['amb']['exact']:.3f}"
        print(line, flush=True)
    return {"tag": tag, "fullobs": fullobs, "cfg": cfg, "n_rows": len(rows),
            "n_amb": n_amb, "amb_frac": n_amb / len(rows), "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=["onpolicy_teacher", "rl", "rl_fullobs", "wide_shallow"])
    ap.add_argument("--n_games", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--out", default=f"{RUNS}/decode_board_results.json")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    allres = {}
    for tag in args.tags:
        allres[tag] = analyze_net(tag, args.n_games, args.seed, do_mlp=not args.no_mlp)
    # JSON-safe
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    json.dump(clean(allres), open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
