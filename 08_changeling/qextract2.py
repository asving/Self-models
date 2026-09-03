"""Suspect 1 (v3 doc, Iteration 22): condition the whitebox on the POST
net's OWN decoded beliefs instead of the exact filter.

1. Fit ridge belief decoders (etaA, etaB, drA, drB) on the post net's
   ON-POLICY hiddens vs exact beliefs (calibrated regime); freeze.
2. Teacher-force on each record set, decode internal beliefs; report their
   drift from exact (TV) per set.
3. Re-extract Qhat with the net-internal pbar_hat = bhatA @ E; fit bilinear
   value in decoded beliefs per set; reconstruct with the fully
   internal-belief whitebox (pbar AND value features decoded). Compare the
   reconstruction grid against Iteration 21. cwd = 08_changeling.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from worlds import World
from rnn import TorchWorld, features
from eval_rnn import rollout_record, WORLD_KW, DEV
from probe import load
from oracle import run_base
from optimality import dp_tables
from offmanifold import build_rep, coef
from spectator import both_tilted_records
from qextract import fit_bilinear, apply_bilinear, r2
from synth2 import heads as synth_heads, mean_kl

RES = {}
R = 768
BATCH = 256
PAIR = (0, 2)
TH = (3.868, 0.622, 0.473, 0.274, 0.389, 0.409, 29.704)
BETA = TH[0]


@torch.no_grad()
def tf_full(post, U, V, goals):
    pus, pvs, hs = [], [], []
    for i in range(0, len(U), BATCH):
        X = torch.tensor(features(U[i:i + BATCH], V[i:i + BATCH],
                                  goals[i:i + BATCH]), device=DEV)
        lu, lv, h = post(X)
        T = U.shape[1]
        pus.append(F.softmax(lu[:, :T], -1).cpu().numpy())
        pvs.append(F.softmax(lv[:, :T], -1).cpu().numpy())
        hs.append(h[:, :T].cpu().numpy())
    return np.concatenate(pus), np.concatenate(pvs), np.concatenate(hs)


def ridge_multi(H, G, alpha=1.0):
    Z = H.reshape(-1, 256)
    y = G.reshape(-1, G.shape[-1])
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    Z = (Z - mu) / sd
    A = Z.T @ Z + alpha * np.eye(256)
    W = np.linalg.solve(A, Z.T @ (y - y.mean(0)))
    return (W, mu, sd, y.mean(0))


def decode(dec, H, shape):
    W, mu, sd, ym = dec
    p = ((H.reshape(-1, 256) - mu) / sd) @ W + ym
    p = np.maximum(p.reshape(shape), 1e-4)
    return p / p.sum(-1, keepdims=True)


def extract_q_hat(P, pbar_hat, plan_basis, pbar_basis):
    m = coef(P, plan_basis, pbar_basis)
    m = np.clip(np.nan_to_num(m, nan=0.6), 0.2, 0.995)[..., None]
    plan_est = np.maximum(P - (1 - m) * pbar_hat, 1e-3 * pbar_hat) / m
    plan_est /= plan_est.sum(-1, keepdims=True)
    q = (np.log(plan_est + 1e-12) - np.log(pbar_hat + 1e-12)) / BETA
    return q - q.mean(-1, keepdims=True)


def main():
    rng = np.random.default_rng(113)
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    _, QA = dp_tables(w, 'A')
    _, QB = dp_tables(w, 'B')
    goals = np.tile(np.array(PAIR), (R, 1))

    sets = {}
    recs = [rollout_record(post, tw, PAIR, BATCH, 11100 + k) for k in range(R // BATCH)]
    sets['onpolicy'] = (np.concatenate([x['u'] for x in recs]).astype(np.int64),
                        np.concatenate([x['v'] for x in recs]).astype(np.int64))
    b = run_base(w, R, 11200, collect=True)
    sets['base_law'] = (b['u'].astype(np.int64), b['v'].astype(np.int64))
    sets['random'] = (rng.integers(0, 6, (R, w.T)), rng.integers(0, 6, (R, w.T)))
    sets['both_tilted'] = both_tilted_records(w, 11300)

    # pass + exact reps
    D = {}
    for tag, (U, V) in sets.items():
        rep = build_rep(w, U, V, QA, QB)
        pu, pv, H = tf_full(post, U, V, goals)
        D[tag] = dict(rep=rep, U=U, V=V, pu=pu, pv=pv, H=H)

    # decoders fit on-policy (first half), quality on second half
    d0 = D['onpolicy']
    half = R // 2
    decs = {k: ridge_multi(d0['H'][:half], d0['rep'][k][:half])
            for k in ('etaA', 'etaB', 'drA', 'drB')}
    for tag in sets:
        d = D[tag]
        for k in decs:
            d[f'hat_{k}'] = decode(decs[k], d['H'], d['rep'][k].shape)
        tv = 0.5 * np.abs(d['hat_etaA'] - d['rep']['etaA']).sum(-1).mean()
        tvB = 0.5 * np.abs(d['hat_etaB'] - d['rep']['etaB']).sum(-1).mean()
        RES[f'{tag}_belief_drift_TV_A_B'] = [round(float(tv), 3),
                                             round(float(tvB), 3)]
        print(tag, 'belief drift TV:', RES[f'{tag}_belief_drift_TV_A_B'],
              flush=True)

    # extraction + bilinear fit in DECODED beliefs
    for tag in sets:
        d = D[tag]
        rep = d['rep']
        pbar_hat_u = d['hat_etaA'] @ w.EA
        pbar_hat_v = d['hat_etaB'] @ w.EB
        planA = rep['pbar_u'] * np.exp(BETA * (rep['qA'] - rep['qA'].max(-1, keepdims=True)))
        planA /= planA.sum(-1, keepdims=True)
        planB = rep['pbar_v'] * np.exp(BETA * (rep['qB'] - rep['qB'].max(-1, keepdims=True)))
        planB /= planB.sum(-1, keepdims=True)
        d['qhat_u'] = extract_q_hat(d['pu'], pbar_hat_u, planA, rep['pbar_u'])
        d['qhat_v'] = extract_q_hat(d['pv'], pbar_hat_v, planB, rep['pbar_v'])
        d['Mu'] = fit_bilinear(d['qhat_u'], d['hat_drA'], d['hat_etaB'])
        d['Mv'] = fit_bilinear(d['qhat_v'], d['hat_etaA'], d['hat_drB'])
        d['pbar_hat_u'], d['pbar_hat_v'] = pbar_hat_u, pbar_hat_v

    # within/cross-set Qhat function consistency (decoded-belief features)
    cross = {}
    for src in sets:
        for dst in sets:
            pred = apply_bilinear(D[src]['Mu'], D[dst]['hat_drA'],
                                  D[dst]['hat_etaB'])
            cross[f'{src}->{dst}'] = r2(pred, D[dst]['qhat_u'])
    RES['cross_set_qhat_r2_decoded'] = cross
    print('cross-set (decoded beliefs):', cross, flush=True)

    # reconstruction with the fully internal-belief whitebox
    for src in ('onpolicy', 'base_law', 'both_tilted'):
        for dst in sets:
            d = D[dst]
            rep2 = {'pbar_u': d['pbar_hat_u'], 'pbar_v': d['pbar_hat_v'],
                    'qA': apply_bilinear(D[src]['Mu'], d['hat_drA'], d['hat_etaB']),
                    'qB': apply_bilinear(D[src]['Mv'], d['hat_etaA'], d['hat_drB'])}
            rec2 = {'u': d['U'], 'v': d['V'], 'pu': d['pu'], 'pv': d['pv']}
            P_u, P_v = synth_heads(rep2, rec2, TH)
            klv = 0.5 * (mean_kl(d['pu'], P_u) + mean_kl(d['pv'], P_v))
            RES[f'recon2_{src}->{dst}'] = round(klv, 4)
    print({k: v for k, v in RES.items() if k.startswith('recon2')}, flush=True)

    with open('results/rnn_qextract2.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print('wrote results/rnn_qextract2.json', flush=True)


if __name__ == '__main__':
    main()
