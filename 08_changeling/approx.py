"""What approximation is the net running off-manifold? (v3, Iteration 23.)
Model-comparison ladder on the net's raw tilt tau = centered(log P - log
pbar), per record set, all held-out (episode 50/50 split):
  F1 schedule: per-(t,u) constant template
  F2 reactive: per-round tables over onehot(u_{t-1}) x onehot(v_{t-1})
  F3 bilinear exact beliefs (baseline family)
  F4 bilinear SHARPENED beliefs, gamma in {2, 4, inf(mode one-hot)}
Also per set: KL(net||neutral), KL(net||beta-plan) per arm — is the output
a neutral / goal policy or a third thing. cwd = 08_changeling.
"""
import json
import numpy as np
import torch
from worlds import World
from rnn import TorchWorld
from eval_rnn import WORLD_KW, DEV
from probe import load
from oracle import run_base
from optimality import dp_tables
from offmanifold import build_rep, tf_heads
from spectator import both_tilted_records
from eval_rnn import rollout_record
from synth2 import mean_kl

RES = {}
R = 768
BATCH = 256
PAIR = (0, 2)
BETA = 3.868


def sharpen(b, g):
    if g == 'mode':
        out = np.zeros_like(b)
        idx = b.argmax(-1)
        np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
        return out
    p = b ** g
    return p / p.sum(-1, keepdims=True)


def fit_tables(tau, fA, fB, tr, te, alpha=1.0):
    """Per-round ridge over outer(fA,fB); heldout R2 (pooled, centered)."""
    Rn, T, n = tau.shape
    dA, dB = fA.shape[-1], fB.shape[-1]
    preds = np.zeros_like(tau)
    for t in range(T):
        X = (fA[:, t, :, None] * fB[:, t, None, :]).reshape(Rn, dA * dB)
        A = X[tr].T @ X[tr] + alpha * np.eye(dA * dB)
        M = np.linalg.solve(A, X[tr].T @ tau[tr, t])
        preds[:, t] = X @ M
    y = tau[te] - tau[te].mean(-1, keepdims=True)
    p = preds[te] - preds[te].mean(-1, keepdims=True)
    return round(float(1 - ((y - p) ** 2).sum() / ((y ** 2).sum() + 1e-12)), 3)


def main():
    rng = np.random.default_rng(131)
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    _, QA = dp_tables(w, 'A')
    _, QB = dp_tables(w, 'B')
    goals = np.tile(np.array(PAIR), (R, 1))

    sets = {}
    recs = [rollout_record(post, tw, PAIR, BATCH, 12100 + k) for k in range(R // BATCH)]
    sets['onpolicy'] = (np.concatenate([x['u'] for x in recs]).astype(np.int64),
                        np.concatenate([x['v'] for x in recs]).astype(np.int64))
    b = run_base(w, R, 12200, collect=True)
    sets['base_law'] = (b['u'].astype(np.int64), b['v'].astype(np.int64))
    sets['random'] = (rng.integers(0, 6, (R, w.T)), rng.integers(0, 6, (R, w.T)))
    sets['both_tilted'] = both_tilted_records(w, 12300)

    tr = np.arange(R // 2)
    te = np.arange(R // 2, R)
    for tag, (U, V) in sets.items():
        rep = build_rep(w, U, V, QA, QB)
        pu, pv = tf_heads(post, U, V, goals, flag=None)
        tau_u = np.log(pu + 1e-12) - np.log(rep['pbar_u'] + 1e-12)
        tau_u -= tau_u.mean(-1, keepdims=True)
        planA = rep['pbar_u'] * np.exp(BETA * (rep['qA'] - rep['qA'].max(-1, keepdims=True)))
        planA /= planA.sum(-1, keepdims=True)
        planB = rep['pbar_v'] * np.exp(BETA * (rep['qB'] - rep['qB'].max(-1, keepdims=True)))
        planB /= planB.sum(-1, keepdims=True)
        row = {'KL_net_neutral_u': round(mean_kl(pu, rep['pbar_u']), 3),
               'KL_net_plan_u': round(mean_kl(pu, planA), 3),
               'KL_net_neutral_v': round(mean_kl(pv, rep['pbar_v']), 3),
               'KL_net_plan_v': round(mean_kl(pv, planB), 3)}
        Rn, T = U.shape
        ones = np.ones((Rn, T, 1))
        row['F1_schedule'] = fit_tables(tau_u, ones, ones, tr, te)
        tokA = np.zeros((Rn, T, 6)); tokB = np.zeros((Rn, T, 6))
        r_i = np.arange(Rn)[:, None]; t_i = np.arange(1, T)[None, :]
        tokA[r_i, t_i, U[:, :-1]] = 1; tokB[r_i, t_i, V[:, :-1]] = 1
        row['F2_recent_tokens'] = fit_tables(tau_u, tokA, tokB, tr, te)
        row['F3_bilinear_beliefs'] = fit_tables(tau_u, rep['drA'], rep['etaB'],
                                                tr, te)
        for g in (2, 4, 'mode'):
            row[f'F4_sharpened_g{g}'] = fit_tables(
                tau_u, sharpen(rep['drA'], g), sharpen(rep['etaB'], g), tr, te)
        RES[tag] = row
        print(tag, row, flush=True)

    with open('results/rnn_approx.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print('wrote results/rnn_approx.json', flush=True)


if __name__ == '__main__':
    main()
