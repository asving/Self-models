"""Build padded tensor datasets of color-blind tic-tac-toe games.

Each game -> a sequence of rounds (net decisions). Per round we have:
  occ      (9,)  colorblind occupancy   [model input]
  true     (9,)  true board {0,1,2}     [for analysis / verification]
  target   (9,)  multi-hot optimal set  [training target, mass split over set]
  my_move  ()    the move actually made [own-move history label]
  legal    (9,)  legal-cell mask        [== occ==0]
We pad to MAX_ROUNDS and provide a round-validity mask.
"""
import numpy as np
import torch
import ttt

MAX_ROUNDS = 5  # X moves at most 5 times in 3x3


def game_to_arrays(g):
    R = len(g["occ"])
    occ = np.zeros((MAX_ROUNDS, 9), np.float32)
    true = np.zeros((MAX_ROUNDS, 9), np.int64)
    target = np.zeros((MAX_ROUNDS, 9), np.float32)  # uniform mass over optimal set
    legal = np.zeros((MAX_ROUNDS, 9), np.float32)
    my_move = np.full((MAX_ROUNDS,), -1, np.int64)
    valid = np.zeros((MAX_ROUNDS,), np.float32)
    for r in range(R):
        occ[r] = g["occ"][r]
        true[r] = g["true"][r]
        legal[r] = (g["occ"][r] == 0).astype(np.float32)
        opt = g["target"][r]
        if opt:
            target[r, opt] = 1.0 / len(opt)
        my_move[r] = g["my_moves"][r]
        valid[r] = 1.0
    return occ, true, target, legal, my_move, valid


def build_dataset(n_games, seed, p_strong=0.5):
    rng = np.random.default_rng(seed)
    OCC, TRUE, TGT, LEG, MV, VAL = [], [], [], [], [], []
    for _ in range(n_games):
        g = ttt.play_game(rng, p_strong=p_strong)
        a = game_to_arrays(g)
        OCC.append(a[0]); TRUE.append(a[1]); TGT.append(a[2])
        LEG.append(a[3]); MV.append(a[4]); VAL.append(a[5])
    return {
        "occ": torch.tensor(np.stack(OCC)),
        "true": torch.tensor(np.stack(TRUE)),
        "target": torch.tensor(np.stack(TGT)),
        "legal": torch.tensor(np.stack(LEG)),
        "my_move": torch.tensor(np.stack(MV)),
        "valid": torch.tensor(np.stack(VAL)),
    }


if __name__ == "__main__":
    d = build_dataset(2000, seed=1)
    for k, v in d.items():
        print(k, tuple(v.shape), v.dtype)
    # rounds-per-game distribution
    rpg = d["valid"].sum(1)
    print("avg rounds/game:", rpg.mean().item())
