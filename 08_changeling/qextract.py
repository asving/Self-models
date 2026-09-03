"""Extract the net's own value function from its outputs and close the loop
(v3 doc, Iteration 21; Asvin's design).

Per record set {onpolicy, base_law, random, both_tilted}:
 1. EXTRACT per-point Qhat: invert the mixture with the net's claim
    coefficient (m clipped), Qhat = (1/beta) centered log(plan_est/pbar).
 2. FIT the synth's own value ontology: Qhat(u) ~ bilinear(drA, etaB)
    with per-round tables Mhat_t (ridge); same for v-channel.
 3. CONSISTENCY: cross-set R2 (fit on X, predict extracted Qhat on Y) +
    correlation of extracted Qhat with the exact optimal Q per set.
 4. RECONSTRUCTION: whitebox heads with q-arrays replaced by the fitted
    Qhat-function (cross-set = non-circular) -> KL(net||whitebox) per set,
    vs the exact-Q whitebox. cwd = 08_changeling.
"""
import json
import numpy as np
import torch
from worlds import World
from rnn import TorchWorld
from eval_rnn import rollout_record, WORLD_KW, DEV
from probe import load
from oracle import run_base
from optimality import dp_tables
from offmanifold import build_rep, tf_heads, coef
from spectator import both_tilted_records
from synth2 import heads as synth_heads, mean_kl

RES = {}
R = 768
BATCH = 256
PAIR = (0, 2)
TH = (3.868, 0.622, 0.473, 0.274, 0.389, 0.409, 29.704)
BETA = TH[0]


def extract_q(P, pbar, plan_exact):
    m = coef(P, plan_exact, pbar)
    m = np.clip(np.nan_to_num(m, nan=0.6), 0.2, 0.995)[..., None]
    plan_est = np.maximum(P - (1 - m) * pbar, 1e-3 * pbar) / m
    plan_est /= plan_est.sum(-1, keepdims=True)
    q = (np.log(plan_est + 1e-12) - np.log(pbar + 1e-12)) / BETA
    return q - q.mean(-1, keepdims=True)


def fit_bilinear(qhat, bA, bB, alpha=1.0):
    """Per-round tables Mhat[t][a,b,u] via ridge on outer(bA,bB)."""
    Rn, T, n = qhat.shape
    M = np.zeros((T, 36, n))
    for t in range(T):
        X = (bA[:, t, :, None] * bB[:, t, None, :]).reshape(Rn, 36)
        A = X.T @ X + alpha * np.eye(36)
        M[t] = np.linalg.solve(A, X.T @ qhat[:, t])
    return M


def apply_bilinear(M, bA, bB):
    Rn, T = bA.shape[:2]
    q = np.einsum('rta,rtb,tku->rtu',
                  bA, bB, M.reshape(M.shape[0], 36, -1)
                  ) if False else np.stack(
        [(bA[:, t, :, None] * bB[:, t, None, :]).reshape(Rn, 36) @ M[t]
         for t in range(T)], axis=1)
    return q - q.mean(-1, keepdims=True)


def r2(pred, y):
    pred = pred - pred.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    b = (pred * y).sum() / ((pred ** 2).sum() + 1e-12)
    return round(float(1 - ((y - b * pred) ** 2).sum() / ((y ** 2).sum() + 1e-12)), 3)


def main():
    rng = np.random.default_rng(101)
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    _, QA = dp_tables(w, 'A')
    _, QB = dp_tables(w, 'B')
    goals = np.tile(np.array(PAIR), (R, 1))

    sets = {}
    recs = [rollout_record(post, tw, PAIR, BATCH, 10100 + k) for k in range(R // BATCH)]
    sets['onpolicy'] = (np.concatenate([x['u'] for x in recs]).astype(np.int64),
                        np.concatenate([x['v'] for x in recs]).astype(np.int64))
    b = run_base(w, R, 10200, collect=True)
    sets['base_law'] = (b['u'].astype(np.int64), b['v'].astype(np.int64))
    sets['random'] = (rng.integers(0, 6, (R, w.T)), rng.integers(0, 6, (R, w.T)))
    sets['both_tilted'] = both_tilted_records(w, 10300)

    D = {}
    for tag, (U, V) in sets.items():
        rep = build_rep(w, U, V, QA, QB)
        pu, pv = tf_heads(post, U, V, goals, flag=None)
        planA = rep['pbar_u'] * np.exp(BETA * (rep['qA'] - rep['qA'].max(-1, keepdims=True)))
        planA /= planA.sum(-1, keepdims=True)
        planB = rep['pbar_v'] * np.exp(BETA * (rep['qB'] - rep['qB'].max(-1, keepdims=True)))
        planB /= planB.sum(-1, keepdims=True)
        qhat_u = extract_q(pu, rep['pbar_u'], planA)
        qhat_v = extract_q(pv, rep['pbar_v'], planB)
        Mu = fit_bilinear(qhat_u, rep['drA'], rep['etaB'])
        Mv = fit_bilinear(qhat_v, rep['etaA'], rep['drB'])
        D[tag] = dict(rep=rep, rec={'u': U, 'v': V, 'pu': pu, 'pv': pv},
                      qhat_u=qhat_u, qhat_v=qhat_v, Mu=Mu, Mv=Mv)
        qex = rep['qA'] - rep['qA'].mean(-1, keepdims=True)
        RES[f'{tag}_corr_qhat_vs_exact'] = r2(qex, qhat_u)
        print(tag, 'qhat-vs-exactQ R2:', RES[f'{tag}_corr_qhat_vs_exact'],
              flush=True)

    # consistency: cross-set function transfer (u-channel)
    cross = {}
    for src in sets:
        for dst in sets:
            pred = apply_bilinear(D[src]['Mu'], D[dst]['rep']['drA'],
                                  D[dst]['rep']['etaB'])
            cross[f'{src}->{dst}'] = r2(pred, D[dst]['qhat_u'])
    RES['cross_set_qhat_r2'] = cross
    print('cross-set Qhat R2:', cross, flush=True)

    # reconstruction: whitebox with fitted-Qhat function, cross-set
    for src in ('onpolicy', 'base_law', 'both_tilted'):
        for dst in ('onpolicy', 'base_law', 'random', 'both_tilted'):
            rep2 = dict(D[dst]['rep'])
            rep2['qA'] = apply_bilinear(D[src]['Mu'], rep2['drA'], rep2['etaB'])
            rep2['qB'] = apply_bilinear(D[src]['Mv'], rep2['etaA'], rep2['drB'])
            P_u, P_v = synth_heads(rep2, D[dst]['rec'], TH)
            klv = 0.5 * (mean_kl(D[dst]['rec']['pu'], P_u)
                         + mean_kl(D[dst]['rec']['pv'], P_v))
            RES[f'recon_{src}->{dst}'] = round(klv, 4)
    print({k: v for k, v in RES.items() if k.startswith('recon')}, flush=True)

    with open('results/rnn_qextract.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print('wrote results/rnn_qextract.json', flush=True)


if __name__ == '__main__':
    main()
