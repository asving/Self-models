"""SECONDARY (brief): stored-self vs re-derivable check for a COLLAPSED color-blind
policy.

Prediction (task): if the closed-loop color-blind policy collapses to (near-)
deterministic, its OWN moves become a deterministic function of the color-blind
history again => RE-DERIVABLE, not stored. So a rubber-hand patch of the internally
recalled own-move should be OVERWRITTEN (perceived state does NOT follow), same as
the open-loop net. We do two quick things:

  (1) DETERMINISM: over on-policy states, how peaked is the policy? Report mean
      max-prob and the fraction of decisions with max-prob > {0.9, 0.99}. A fully
      collapsed policy => max-prob ~ 1 everywhere => no residual stochastic/stored
      self.

  (2) RE-DERIVABILITY via rubber-hand: reuse ttt_rubberhand.rubber_hand at decision
      round t=2 (all-layers, scale sweep). If the CLEAN-FLIP rate stays ~0 (patch
      overwritten), the self is re-derived; a high flip rate would indicate a stored,
      corruptible self-action.

CPU.  CUDA_VISIBLE_DEVICES="" ~/comp_icl/.venv/bin/python storedself_check.py --tag rl
(color-blind tags only: onpolicy_teacher, rl)
"""
import argparse
import numpy as np
import torch
import ttt
import policy_eval as PE
from ttt_causal import load, Manual
from ttt_rubberhand import rubber_hand

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttt_runs")


@torch.no_grad()
def determinism(model, device, n_games=3000, seed=31):
    """On-policy states, full max-prob distribution split opening/tie/forced/all."""
    rng = np.random.default_rng(seed)
    buckets = {"all": [], "opening": [], "tie": [], "forced": []}
    for _ in range(n_games):
        board = [0] * 9
        occ_seq = []
        rr = 0
        while True:
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            occ_seq.append(PE.obs_from_board(board, False))
            p = PE.policy_probs(model, occ_seq, board, device)
            mx = float(p.max())
            opt = ttt.optimal_moves(board, 1)
            buckets["all"].append(mx)
            if rr == 0: buckets["opening"].append(mx)
            if len(opt) > 1: buckets["tie"].append(mx)
            else: buckets["forced"].append(mx)
            mv = int(rng.choice(9, p=p)); board[mv] = 1; rr += 1
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            omv = ttt.opponent_move(board, rng, 0.5)
            if omv is None: break
            board[omv] = 2
    print("  DETERMINISM (max-prob over on-policy states):")
    for k, v in buckets.items():
        v = np.array(v)
        if len(v) == 0:
            print(f"    {k:8s}: (none)"); continue
        print(f"    {k:8s}: mean_maxp={v.mean():.3f}  frac>0.9={np.mean(v>0.9):.3f}  "
              f"frac>0.99={np.mean(v>0.99):.3f}  n={len(v)}")
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rl")
    args = ap.parse_args()
    torch.manual_seed(0)
    model, ck = load(args.tag)
    model.eval()
    print(f"=== STORED-SELF CHECK: {args.tag} "
          f"({ck['config']['n_layer']}x{ck['config']['d_model']}) ===")
    determinism(model, "cpu")

    print("\n  RE-DERIVABILITY (rubber-hand patch of recalled own-move, t=2):")
    d = torch.load(f"{RUNS}/evalset.pt", weights_only=False)
    occ = d["occ"].numpy(); true = d["true"].numpy()
    valid = d["valid"].numpy(); my = d["my_move"].numpy()
    man = Manual(model)
    for sc in (1.0, 2.0, 4.0):
        print(f"  -- all-layers scale={sc} --")
        rubber_hand(man, occ, true, valid, my, t=2, patch_layers="all", scale=sc)


if __name__ == "__main__":
    main()
