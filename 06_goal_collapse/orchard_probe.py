"""S2 interp on the pretrained orchard net — PREREG threshold (c) + layer map.

Probe: ridge from residual stream at decision positions (input idx 3+3t) to
the exact Bayes goal posterior over states (N,T,5) from the persona-mixture
filter. Report per-layer R², plus the GRADED check: R² restricted to
INTERIOR positions (Bayes max-prob < .85, i.e. the funnel rounds right after
choices) — a net storing only the MAP would fail there.
"""
from __future__ import annotations
import sys

import numpy as np
import torch

from orchard import Net, persona_gen, T, S


def ridge_fit(H, Y, l2=10.0):
    Hb = np.concatenate([H, np.ones((len(H), 1))], 1)
    A = Hb.T @ Hb + l2 * np.eye(Hb.shape[1])
    return np.linalg.solve(A, Hb.T @ Y)


def r2(W, H, Y):
    Hb = np.concatenate([H, np.ones((len(H), 1))], 1)
    P = Hb @ W
    ss = ((Y - P) ** 2).sum()
    st = ((Y - Y.mean(0)) ** 2).sum()
    return 1 - ss / st, P


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        'orchard_runs/A/p1_final.pt'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net().to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev))
    net.eval()
    ev = persona_gen(2000, np.random.default_rng(999))
    toks = torch.tensor(ev['toks'], device=dev)
    with torch.no_grad():
        _, hs = net(toks, return_hidden=True)
    dec_pos = 3 + 3 * np.arange(T)
    Y = ev['goal_post'].reshape(-1, S)
    mx = ev['goal_post'].max(-1).reshape(-1)
    ntr = 1400 * T
    interior = mx < 0.85
    print(f'interior positions: {interior.mean():.2%}')
    best = (-1, None, None)
    for li, h in enumerate(hs):
        H = h[:, dec_pos].reshape(-1, h.shape[-1]).cpu().numpy()
        W = ridge_fit(H[:ntr], Y[:ntr])
        r_all, P = r2(W, H[ntr:], Y[ntr:])
        te_int = interior[ntr:]
        r_int = 1 - ((Y[ntr:][te_int] - P[te_int]) ** 2).sum() / \
            ((Y[ntr:][te_int] - Y[ntr:][te_int].mean(0)) ** 2).sum()
        # calibration: pooled slope of predicted vs true posterior mass
        sl = np.polyfit(Y[ntr:].reshape(-1), P.reshape(-1), 1)[0]
        print(f'layer {li}: R2 all {r_all:.3f} | R2 interior {r_int:.3f} '
              f'| calib slope {sl:.2f}')
        if r_all > best[0]:
            best = (r_all, li, r_int)
    print(f'BEST layer {best[1]}: R2 {best[0]:.3f}, interior {best[2]:.3f} '
          f'(PREREG (c): R2 >= 0.7, graded structure present)')


if __name__ == '__main__':
    main()
