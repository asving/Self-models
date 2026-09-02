"""Read the net's implicit Q-hat off its logits (v3 doc, Iteration 13).

At any stationary point of the KL-anchored objective, log pi - log pbar =
rho*Q^pi + const, so Q-hat(u) := (1/rho)*(log head(u) - log pbar(u)) is the
net's own action-value estimate. Compare it (centered per round) against
the myopic one-step value (log score from the M-table) and the optimal
bootstrapped Q (DP), on claimed channels with saturated claims (iota-true,
t >= 8). cwd = 08_changeling.
"""
import json
import numpy as np
import torch
from worlds import World
from rnn import TorchWorld, GOAL_PAIRS
from eval_rnn import rollout_record, WORLD_KW, DEV
from probe import load
from oracle import replay_dists
from optimality import dp_tables, RHO

RES = {}
BATCH = 256


def main():
    rng = np.random.default_rng(41)
    post = load('post_6000')
    ys, xh, xq = [], [], []
    for b in range(6):
        pair = GOAL_PAIRS[rng.integers(12)]
        w = World(goal_pair=pair, **WORLD_KW)
        _, QA = dp_tables(w, 'A')
        rec = rollout_record(post, TorchWorld(w, DEV), pair, BATCH, 4600 + b)
        rep = replay_dists(w, rec['u'], rec['v'], return_beliefs=True)
        io = rec['iota']
        for t in range(8, w.T):
            scA = np.einsum('ra,rb,abu->ru', rep['drA'][:, t],
                            rep['etaB'][:, t], w.M[t])
            qA = np.einsum('ra,rb,abu->ru', rep['drA'][:, t],
                           rep['etaB'][:, t], QA[t])
            y = np.log(rec['pu'][:, t] + 1e-12) - np.log(rep['pbar_u'][:, t] + 1e-12)
            xh_ = np.log(scA + 1e-300)
            xq_ = RHO * qA
            for arr, store in ((y, ys), (xh_, xh), (xq_, xq)):
                c = arr - arr.mean(1, keepdims=True)
                store.append(c[io])
    y = np.concatenate(ys).ravel()
    x1 = np.concatenate(xh).ravel()      # myopic h-value tilt
    x2 = np.concatenate(xq).ravel()      # optimal-Q tilt (rho*Q)
    def r2(pred):
        return round(float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)
    b1 = float((x1 * y).sum() / (x1 * x1).sum())
    b2 = float((x2 * y).sum() / (x2 * x2).sum())
    X = np.stack([x1, x2], 1)
    bb = np.linalg.lstsq(X, y, rcond=None)[0]
    RES['R2_vs_myopic_h'] = r2(b1 * x1)
    RES['slope_myopic'] = round(b1, 3)
    RES['R2_vs_optimalQ'] = r2(b2 * x2)
    RES['slope_optimalQ'] = round(b2, 3)
    RES['R2_joint'] = r2(X @ bb)
    RES['joint_coefs_h_q'] = [round(float(v), 3) for v in bb]
    RES['corr_x1_x2'] = round(float(np.corrcoef(x1, x2)[0, 1]), 3)
    print(json.dumps(RES, indent=1), flush=True)
    with open('results/rnn_qhat.json', 'w') as fj:
        json.dump(RES, fj, indent=1)


if __name__ == '__main__':
    main()
