"""Spectator streams & the default-level decomposition (v3, Iteration 20;
Asvin's challenge). Sets: both-tilted (u~piA, v~piB), base-law, random.
Per set: net claims; does the net emit the GOAL POLICY on both arms
(KL(net||plan), tilt-R2 vs plan); whitebox-v2 no-refit KL; whitebox REFIT
KL + refit default-claims sigma(c) — how much of the off-manifold gap is
the ill-pinned rho=0 default vs tilt-direction error. cwd = 08_changeling.
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
from whitebox_lambda import Filt
from offmanifold import build_rep, synth_claims_and_rho, coef, tf_heads, kl, tiltr2
from synth2 import heads as synth_heads_fn, mean_kl

RES = {}
R = 768
PAIR = (0, 2)


def both_tilted_records(w, seed):
    rng = np.random.default_rng(seed)
    f = Filt(w, R)
    U = np.zeros((R, w.T), np.int64); V = np.zeros((R, w.T), np.int64)
    for t in range(w.T):
        _, _, piA, piB = f.dists(t)
        c = np.cumsum(piA, 1); U[:, t] = np.argmax(c > rng.random((R, 1)), 1)
        c = np.cumsum(piB, 1); V[:, t] = np.argmax(c > rng.random((R, 1)), 1)
        f.update(U[:, t], V[:, t])
    return U, V


def refit(rep, rec, rng, iters=1500):
    best, best_th = 1e9, None
    for _ in range(iters):
        th = (rng.uniform(0.5, 8), rng.uniform(0, 1.5), rng.uniform(0, 1.5),
              rng.uniform(0.05, 1), rng.uniform(-2, 6), rng.uniform(-2, 6),
              rng.uniform(2, 40))
        P_u, P_v = synth_heads_fn(rep, rec, th)
        v = 0.5 * (mean_kl(rec['pu'], P_u) + mean_kl(rec['pv'], P_v))
        if v < best:
            best, best_th = v, th
    for _ in range(400):
        th = tuple(np.array(best_th) * np.exp(rng.normal(0, 0.07, 7)))
        P_u, P_v = synth_heads_fn(rep, rec, th)
        v = 0.5 * (mean_kl(rec['pu'], P_u) + mean_kl(rec['pv'], P_v))
        if v < best:
            best, best_th = v, th
    return best, best_th


def main():
    rng = np.random.default_rng(97)
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    _, QA = dp_tables(w, 'A')
    _, QB = dp_tables(w, 'B')
    goals = np.tile(np.array(PAIR), (R, 1))
    TH = (3.868, 0.622, 0.473, 0.274, 0.389, 0.409, 29.704)

    sets = {}
    sets['both_tilted'] = both_tilted_records(w, 9100)
    b = run_base(w, R, 9200, collect=True)
    sets['base_law'] = (b['u'].astype(np.int64), b['v'].astype(np.int64))
    sets['random_tokens'] = (rng.integers(0, 6, (R, w.T)),
                             rng.integers(0, 6, (R, w.T)))

    for tag, (U, V) in sets.items():
        rep = build_rep(w, U, V, QA, QB)
        pu, pv = tf_heads(post, U, V, goals, flag=None)
        rec = {'u': U, 'v': V, 'pu': pu, 'pv': pv}
        cu = coef(pu, rep['piA'], rep['pbar_u'])
        cv = coef(pv, rep['piB'], rep['pbar_v'])
        # synth plan at fitted beta (the net's own plan primitive)
        planU = rep['pbar_u'] * np.exp(TH[0] * (rep['qA'] - rep['qA'].max(-1, keepdims=True)))
        planU /= planU.sum(-1, keepdims=True)
        planV = rep['pbar_v'] * np.exp(TH[0] * (rep['qB'] - rep['qB'].max(-1, keepdims=True)))
        planV /= planV.sum(-1, keepdims=True)
        P_u0, P_v0 = synth_heads_fn(rep, rec, TH)
        kl_norefit = 0.5 * (mean_kl(pu, P_u0) + mean_kl(pv, P_v0))
        kl_plan = 0.5 * (mean_kl(pu, planU) + mean_kl(pv, planV))
        klb, thb = refit(rep, rec, np.random.default_rng(5))
        RES[tag] = {
            'net_claims_mean_u_v': [round(float(np.nanmean(cu[:, 4:])), 3),
                                    round(float(np.nanmean(cv[:, 4:])), 3)],
            'KL_net_to_pureplan': round(kl_plan, 4),
            'tiltR2_vs_plan': 0.5 * (tiltr2(pu, planU, rep['pbar_u'])
                                     + tiltr2(pv, planV, rep['pbar_v'])),
            'KL_whitebox_norefit': round(kl_norefit, 4),
            'KL_whitebox_refit': round(klb, 4),
            'refit_default_claims': [round(float(1 / (1 + np.exp(-thb[4]))), 3),
                                     round(float(1 / (1 + np.exp(-thb[5]))), 3)],
            'refit_beta': round(float(thb[0]), 3)}
        print(tag, RES[tag], flush=True)

    with open('results/rnn_spectator.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print('wrote results/rnn_spectator.json', flush=True)


if __name__ == '__main__':
    main()
