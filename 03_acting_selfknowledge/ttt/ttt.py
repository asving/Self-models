"""Tic-tac-toe environment, exact minimax solver, and color-blind data generator.

Conventions
-----------
Board: 9 cells, index 0..8 row-major. Cell ownership in {0:empty, 1:mine(X), 2:opp(O)}.
The NET plays X and moves FIRST. Opponent plays O.

Color-blind observation: per cell {0:empty, 1:marked}.  The net sees occupancy on
its OWN turns only.  Between consecutive net observations TWO new marks appear (the
net's own move + the opponent reply), pooled indistinguishably.

Minimax: from the perspective of the player to move; returns the set of optimal moves.
"""
import functools
import numpy as np

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diags
]


def winner(board):
    """board: tuple/list of 9 in {0,1,2}. Returns 1 or 2 if a winner, 0 otherwise."""
    for a, b, c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def legal_moves(board):
    return [i for i in range(9) if board[i] == 0]


@functools.lru_cache(maxsize=None)
def _minimax(board, player):
    """Return (value, best_moves) for `player` to move on `board`.
    Value is +1 if player(to move at root call=1 means X) ... we track from the
    perspective of the player whose turn it is.  We use a convention:
    value = +1 win for the player TO MOVE at THIS node, -1 loss, 0 draw."""
    w = winner(board)
    if w != 0:
        # Someone already won. The player to move did NOT make the winning move
        # (the previous player did). So the player to move has lost.
        return -1, ()
    moves = legal_moves(board)
    if not moves:
        return 0, ()  # draw
    best_val = -2
    best_moves = []
    for m in moves:
        nb = list(board)
        nb[m] = player
        opp = 2 if player == 1 else 1
        child_val, _ = _minimax(tuple(nb), opp)
        val = -child_val  # opponent's value negated
        if val > best_val:
            best_val = val
            best_moves = [m]
        elif val == best_val:
            best_moves.append(m)
    return best_val, tuple(best_moves)


def optimal_moves(board, player):
    """Set of minimax-optimal moves for `player` to move on `board`."""
    _, moves = _minimax(tuple(board), player)
    return list(moves)


def colorblind(board):
    """Map true board {0,1,2} -> occupancy {0:empty,1:marked}, 9-dim float array."""
    return np.array([0.0 if c == 0 else 1.0 for c in board], dtype=np.float32)


# ---------------------------------------------------------------------------
# Opponent policy (UNPREDICTABLE): mostly random legal, sometimes strong.
# ---------------------------------------------------------------------------
def opponent_move(board, rng, p_strong=0.5):
    """Opponent (O = player 2) chooses a move.

    With prob p_strong, plays a 'strong' move (minimax-optimal for itself);
    otherwise plays a uniformly random legal move.  Mixing keeps it
    unpredictable so the net cannot recover its own move by predicting O and
    subtracting.  When several optimal/random moves tie, pick uniformly.
    """
    moves = legal_moves(board)
    if not moves:
        return None
    if rng.random() < p_strong:
        opt = optimal_moves(board, 2)
        return int(rng.choice(opt)) if opt else int(rng.choice(moves))
    return int(rng.choice(moves))


# ---------------------------------------------------------------------------
# Game rollout: net plays X on-policy minimax-optimal; opponent as above.
# Produces, per net decision round, the data we need.
# ---------------------------------------------------------------------------
def play_game(rng, p_strong=0.5, net_policy=None):
    """Roll out one game.

    net_policy: optional callable(occupancy_seq_so_far, round_idx) -> move.
                If None, the net plays a minimax-optimal move (on-policy data gen).
                We randomize WHICH optimal move (uniform over optimal set) to get
                coverage of distinct own-move histories.

    Returns a dict with per-round arrays (one entry per net decision):
        occ:      list of 9-dim colorblind occupancies the net observed (input)
        true:     list of true boards (9-dim {0,1,2}) at each net decision
        own_hist: list of own-move sets so far ... we store own moves list
        target:   list of optimal-move sets (list of legal optimal cells)
        my_move:  the move the net actually made that round
    """
    board = [0] * 9
    occ_seq, true_seq, target_seq, my_moves = [], [], [], []
    round_idx = 0
    while True:
        # Net's turn (X = player 1)
        if winner(board) != 0 or not legal_moves(board):
            break
        occ = colorblind(board)
        opt = optimal_moves(board, 1)
        occ_seq.append(occ.copy())
        true_seq.append(np.array(board, dtype=np.int64))
        target_seq.append(list(opt))
        # choose move
        if net_policy is None:
            mv = int(rng.choice(opt))
        else:
            mv = net_policy(occ_seq, round_idx)
        my_moves.append(mv)
        board[mv] = 1
        round_idx += 1
        if winner(board) != 0 or not legal_moves(board):
            break
        # Opponent's turn
        omv = opponent_move(board, rng, p_strong=p_strong)
        if omv is None:
            break
        board[omv] = 2
    return {
        "occ": occ_seq,
        "true": true_seq,
        "target": target_seq,
        "my_moves": my_moves,
    }


if __name__ == "__main__":
    # quick self-test of minimax: optimal first move set for empty board
    print("optimal first moves (X):", optimal_moves([0] * 9, 1))
    # X first move must be able to at least draw -> value 0
    print("root value:", _minimax(tuple([0] * 9), 1)[0])
    rng = np.random.default_rng(0)
    g = play_game(rng)
    print("game rounds:", len(g["occ"]))
    for i in range(len(g["occ"])):
        print("  occ", g["occ"][i].astype(int), "true", g["true"][i],
              "opt", g["target"][i], "mv", g["my_moves"][i])
