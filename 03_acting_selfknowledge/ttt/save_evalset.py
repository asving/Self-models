"""Save a held-out evaluation set for the follow-up analysis.

Per game and per round we store:
  occ      (R,9)  color-blind occupancy sequence  (MODEL INPUT)
  my_moves (R,)   the net's own move-history       (efference-copy target)
  true     (R,9)  TRUE board ownership {0,1,2}
  target   (R,9)  optimal-move multi-hot target
  legal    (R,9)  legal mask
  valid    (R,)   round-validity
Stored padded to MAX_ROUNDS with a valid mask (same layout as data.build_dataset).
"""
import numpy as np
import torch
from data import build_dataset

OUT = "/data/users/asvin/self-models/ttt_runs/evalset.pt"

if __name__ == "__main__":
    d = build_dataset(8000, seed=2025, p_strong=0.5)
    torch.save(d, OUT)
    print("saved eval set ->", OUT)
    for k, v in d.items():
        print(" ", k, tuple(v.shape), v.dtype)
