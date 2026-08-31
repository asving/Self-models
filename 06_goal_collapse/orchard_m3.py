"""M3 seed-diversity (PREREG): do different RL seeds collapse their goals
DIFFERENTLY? Matched initial layouts (same env seed across nets); first
pursuit segment's pursued-goal label per episode; pairwise agreement +
pooled marginal bias. Predictions: arm-C agreement <= 0.65 (coin-like,
seed-idiosyncratic), arm-0 >= 0.85 (compiled public chooser).
Usage: python orchard_m3.py C orchard_runs/C_s0 orchard_runs/C_s1 [...]"""
from __future__ import annotations
import sys
from itertools import combinations

import numpy as np
import torch

from orchard import Net, T
from orchard_collapse import selfplay, segment_labels, DEV


def first_choice(ckpt, lam, N=1500):
    net = Net().to(DEV)
    net.load_state_dict(torch.load(f'{ckpt}/p2_final.pt', map_location=DEV))
    net.eval()
    sp = selfplay(net, N, seed=777, lam=lam)
    _, yp, d = segment_labels(sp, sp['tt'])
    fc = yp[:, 0].copy()                      # first segment's pursued slot
    return fc


def main():
    arm = sys.argv[1]
    runs = sys.argv[2:]
    lam = 0.5 if arm.startswith('C') else 0.0
    fcs = {r: first_choice(r, lam) for r in runs}
    print(f'--- arm {arm}: first-choice stats (matched layouts, seed 777)')
    for r, fc in fcs.items():
        v = fc[fc >= 0]
        print(f'  {r}: labeled {len(v)}/{len(fc)}  P(lower) = '
              f'{(v == 0).mean():.3f}')
    for r1, r2 in combinations(runs, 2):
        m = (fcs[r1] >= 0) & (fcs[r2] >= 0)
        ag = (fcs[r1][m] == fcs[r2][m]).mean()
        print(f'  agreement {r1} vs {r2}: {ag:.3f}  (n={m.sum()})')
    pooled = np.concatenate([fc[fc >= 0] for fc in fcs.values()])
    print(f'  pooled |P(lower) - 0.5| = {abs((pooled == 0).mean() - .5):.3f}'
          f'  (PREREG: <= 0.10)')


if __name__ == '__main__':
    main()
