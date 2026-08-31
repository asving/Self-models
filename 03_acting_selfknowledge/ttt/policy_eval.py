"""Shared policy-entropy / collapse measurement for the closed-loop TTT runs.

PRIMARY MEASUREMENT (see task):
  - Policy entropy at TIE positions (optimal set > 1) and at the OPENING (round 0).
  - Max-prob at the opening and at ties.
  - Move accuracy (argmax-legal in optimal set).
  - Play strength vs random and vs optimal.

The net's policy at a state = softmax over LEGAL cells of its move logits, computed
from the color-blind occupancy SEQUENCE up to that decision (history-conditioned).
We evaluate on a fixed eval set of true boards reached by on-policy/data rollouts,
using the TRUE board to decide the optimal set / ties (color-blind board is the
NET INPUT, true board only labels which positions are ties).

`fullobs=True` lets the net observe the true ownership board (3-way one-hot per
cell -> still fed through the 9-dim input as {empty=0, mine=+1, opp=-1}) for the
fully-observed control. Color-blind nets get {0,1} occupancy.
"""
import numpy as np
import torch
import torch.nn.functional as F
import ttt


def obs_from_board(board, fullobs):
    """board: list/array of 9 in {0,1,2}. Returns 9-dim float observation.
    color-blind: {0:empty,1:marked}. fullobs: empty=0, mine(1)=+1, opp(2)=-1."""
    if fullobs:
        return np.array([0.0 if c == 0 else (1.0 if c == 1 else -1.0) for c in board],
                        dtype=np.float32)
    return np.array([0.0 if c == 0 else 1.0 for c in board], dtype=np.float32)


@torch.no_grad()
def policy_probs(model, occ_seq, board, device):
    """occ_seq: list of 9-dim obs up to & including current decision. board: true
    board (for legal mask). Returns full 9-dim legal-renormalized probability vec."""
    inp = torch.tensor(np.stack(occ_seq), dtype=torch.float32, device=device)[None]
    logits = model(inp)[0, -1]
    legal = torch.tensor([0.0 if c != 0 else 1.0 for c in board], device=device)
    logits = logits.masked_fill(legal == 0, -1e9)
    return F.softmax(logits, -1).cpu().numpy()


def entropy(p):
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


@torch.no_grad()
def collapse_metrics(model, device, fullobs=False, n_games=3000, seed=4242,
                     p_strong=0.5):
    """Roll out on-policy games (net SAMPLES from its own current policy, opponent =
    unpredictable mix), and at each net decision record the policy distribution,
    bucketed into OPENING (round 0) vs TIE (optimal set>1) vs FORCED (optimal set==1).
    Returns dict of mean entropy / mean max-prob / move-accuracy per bucket."""
    rng = np.random.default_rng(seed)
    op_ent, op_max = [], []
    tie_ent, tie_max, tie_acc = [], [], []
    forced_acc = []
    all_acc = []
    for _ in range(n_games):
        board = [0] * 9
        occ_seq = []
        round_idx = 0
        while True:
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            occ_seq.append(obs_from_board(board, fullobs))
            p = policy_probs(model, occ_seq, board, device)
            opt = ttt.optimal_moves(board, 1)
            mx = float(p.max())
            argmax_mv = int(p.argmax())
            in_opt = argmax_mv in opt
            all_acc.append(in_opt)
            if round_idx == 0:
                op_ent.append(entropy(p)); op_max.append(mx)
            if len(opt) > 1:
                tie_ent.append(entropy(p)); tie_max.append(mx); tie_acc.append(in_opt)
            else:
                forced_acc.append(in_opt)
            # SAMPLE the net's own move (closed-loop) and play it
            mv = int(rng.choice(9, p=p))
            board[mv] = 1
            round_idx += 1
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            omv = ttt.opponent_move(board, rng, p_strong=p_strong)
            if omv is None:
                break
            board[omv] = 2
    def m(x):
        return float(np.mean(x)) if len(x) else float("nan")
    return {
        "opening_entropy": m(op_ent), "opening_maxprob": m(op_max),
        "tie_entropy": m(tie_ent), "tie_maxprob": m(tie_max),
        "tie_acc": m(tie_acc), "forced_acc": m(forced_acc),
        "move_acc": m(all_acc),
        "n_tie": len(tie_ent), "n_opening": len(op_ent),
    }


@torch.no_grad()
def net_play(model, device, opponent="random", n_games=2000, seed=123,
             fullobs=False, sample=False):
    """Net plays full games; argmax (sample=False) or sample its move. Opponent
    random or optimal. Returns win/draw/loss fractions for net (X)."""
    rng = np.random.default_rng(seed)
    res = {"win": 0, "draw": 0, "loss": 0}
    for _ in range(n_games):
        board = [0] * 9
        occ_seq = []
        while True:
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            occ_seq.append(obs_from_board(board, fullobs))
            p = policy_probs(model, occ_seq, board, device)
            mv = int(rng.choice(9, p=p)) if sample else int(p.argmax())
            board[mv] = 1
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            if opponent == "random":
                omv = int(rng.choice(ttt.legal_moves(board)))
            else:
                opt = ttt.optimal_moves(board, 2)
                omv = int(rng.choice(opt)) if opt else int(rng.choice(ttt.legal_moves(board)))
            board[omv] = 2
        w = ttt.winner(board)
        if w == 1: res["win"] += 1
        elif w == 2: res["loss"] += 1
        else: res["draw"] += 1
    n = sum(res.values())
    return {k: v / n for k, v in res.items()}


def fmt(d):
    return (f"open_H={d['opening_entropy']:.3f} open_maxp={d['opening_maxprob']:.3f} | "
            f"tie_H={d['tie_entropy']:.3f} tie_maxp={d['tie_maxprob']:.3f} "
            f"tie_acc={d['tie_acc']:.3f} | move_acc={d['move_acc']:.3f} "
            f"forced_acc={d['forced_acc']:.3f}")
