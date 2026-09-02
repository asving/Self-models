"""The right fidelity measures (v3 doc, Iteration 16), per Asvin's critique:
(A) tilt-space fidelity on-policy: R2 of (log net - log pbar) vs the
    program's tilt; fraction of departure-from-neutrality captured.
(B) OFF-TRAJECTORY transfer (no refit): teacher-force net and program on
    base-law records, uniform-random-token records, and informedQ records.
(C) wrong-backbone controls (replacing the flawed shuffle): refit all 7
    params with (i) uniform beliefs (filter off), (ii) misspecified world
    kernels, (iii) rotated goal. cwd = 08_changeling.
"""
import json
import numpy as np
import torch
from worlds import World
from rnn import TorchWorld, features, GOAL_PAIRS
from eval_rnn import rollout_record, WORLD_KW, DEV
from probe import load
from oracle import replay_dists, run_base
from optimality import dp_tables
from whitebox_lambda import Filt
from synth2 import heads, mean_kl

RES = {}
BATCH = 256
N_B = 4
TH = (3.868, 0.622, 0.473, 0.274, 0.389, 0.409, 29.704)
PAIR = (0, 2)


@torch.no_grad()
def net_heads(post, U, V, goals):
    import torch.nn.functional as F
    pus, pvs = [], []
    for i in range(0, len(U), BATCH):
        X = torch.tensor(features(U[i:i + BATCH], V[i:i + BATCH],
                                  goals[i:i + BATCH]), device=DEV)
        lu, lv, _ = post(X)
        T = U.shape[1]
        pus.append(F.softmax(lu[:, :T], -1).cpu().numpy())
        pvs.append(F.softmax(lv[:, :T], -1).cpu().numpy())
    return np.concatenate(pus), np.concatenate(pvs)


def build_rep(w, U, V, QA, QB, mode='true', w_wrong=None):
    if mode == 'uniform':
        R, T = U.shape
        unif = np.full(6, 1 / 6)
        pbu = np.tile(unif @ w.EA, (R, T, 1))
        pbv = np.tile(unif @ w.EB, (R, T, 1))
        qA = np.tile(np.einsum('a,b,tabu->tu', unif, unif, QA), (R, 1, 1))
        qB = np.tile(np.einsum('a,b,tabv->tv', unif, unif, QB), (R, 1, 1))
        return {'pbar_u': pbu, 'pbar_v': pbv, 'qA': qA, 'qB': qB}
    wf = w_wrong if mode == 'wrongworld' else w
    rep = replay_dists(wf, U, V, return_beliefs=True)
    qA = np.einsum('rta,rtb,tabu->rtu', rep['drA'], rep['etaB'], QA)
    qB = np.einsum('rta,rtb,tabv->rtv', rep['etaA'], rep['drB'], QB)
    return {'pbar_u': rep['pbar_u'], 'pbar_v': rep['pbar_v'],
            'qA': qA, 'qB': qB}


def tilt_r2(P_net, P_syn, pbar):
    y = np.log(P_net + 1e-12) - np.log(pbar + 1e-12)
    x = np.log(P_syn + 1e-12) - np.log(pbar + 1e-12)
    y = y - y.mean(-1, keepdims=True)
    x = x - x.mean(-1, keepdims=True)
    y, x = y.ravel(), x.ravel()
    b = (x * y).sum() / ((x * x).sum() + 1e-12)
    return float(1 - ((y - b * x) ** 2).sum() / ((y ** 2).sum() + 1e-12))


def metrics(rec, rep, th, tag):
    P_u, P_v = heads(rep, rec, th)
    kl = 0.5 * (mean_kl(rec['pu'], P_u) + mean_kl(rec['pv'], P_v))
    kl0 = 0.5 * (mean_kl(rec['pu'], rep['pbar_u'])
                 + mean_kl(rec['pv'], rep['pbar_v']))
    r2 = 0.5 * (tilt_r2(rec['pu'], P_u, rep['pbar_u'])
                + tilt_r2(rec['pv'], P_v, rep['pbar_v']))
    RES[tag] = {'kl': round(kl, 4), 'kl_vs_neutral': round(kl0, 4),
                'frac_captured': round(1 - kl / kl0, 3),
                'tilt_r2': round(r2, 3)}
    print(tag, RES[tag], flush=True)


@torch.no_grad()
def informedq_records(w, tw, QA, QB, R, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    iota = rng.random(R) < 0.5
    sA = torch.randint(0, 6, (R,), device=DEV)
    sB = torch.randint(0, 6, (R,), device=DEV)
    f = Filt(w, R)
    U = np.zeros((R, w.T), np.int64); V = np.zeros((R, w.T), np.int64)
    for t in range(w.T):
        pbu, pbv, _, _ = f.dists(t)
        qU = np.einsum('ra,rb,abu->ru', f.drA, f.etaB, QA[t])
        qV = np.einsum('ra,rb,abv->rv', f.etaA, f.drB, QB[t])
        pu = pbu * np.exp(8.0 * (qU - qU.max(1, keepdims=True)))
        pu /= pu.sum(1, keepdims=True)
        pv = pbv * np.exp(8.0 * (qV - qV.max(1, keepdims=True)))
        pv /= pv.sum(1, keepdims=True)
        cum = np.cumsum(pu, 1); un_n = np.argmax(cum > rng.random((R, 1)), 1)
        cum = np.cumsum(pv, 1); vn_n = np.argmax(cum > rng.random((R, 1)), 1)
        ue, ve = tw.emit(sA, sB)
        un = np.where(iota, un_n, ue.cpu().numpy())
        vn = np.where(iota, ve.cpu().numpy(), vn_n)
        U[:, t], V[:, t] = un, vn
        f.update(un, vn)
        sA, sB = tw.trans(sA, sB, torch.tensor(un, device=DEV),
                          torch.tensor(vn, device=DEV))
    return U, V


def main():
    rng = np.random.default_rng(61)
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    _, QA = dp_tables(w, 'A')
    _, QB = dp_tables(w, 'B')
    R = BATCH * N_B
    goals = np.tile(np.array(PAIR), (R, 1))

    # (A) on-policy
    recs = [rollout_record(post, tw, PAIR, BATCH, 6100 + b) for b in range(N_B)]
    U = np.concatenate([r['u'] for r in recs]).astype(np.int64)
    V = np.concatenate([r['v'] for r in recs]).astype(np.int64)
    rec = {'u': U, 'v': V,
           'pu': np.concatenate([r['pu'] for r in recs]),
           'pv': np.concatenate([r['pv'] for r in recs])}
    rep = build_rep(w, U, V, QA, QB)
    metrics(rec, rep, TH, 'onpolicy')

    # (B) off-trajectory transfer, NO refit
    sets = {}
    b = run_base(w, R, 6200, collect=True)
    sets['base_law'] = (b['u'].astype(np.int64), b['v'].astype(np.int64))
    sets['random_tokens'] = (rng.integers(0, 6, (R, w.T)),
                             rng.integers(0, 6, (R, w.T)))
    sets['informedQ'] = informedq_records(w, tw, QA, QB, R, 6300)
    for tag, (Us, Vs) in sets.items():
        pu, pv = net_heads(post, Us, Vs, goals)
        rc = {'u': Us, 'v': Vs, 'pu': pu, 'pv': pv}
        rp = build_rep(w, Us, Vs, QA, QB)
        metrics(rc, rp, TH, f'offpolicy_{tag}')

    # (C) wrong-backbone controls, refit on-policy
    w_wrong = World(goal_pair=PAIR, **{**WORLD_KW, 'q0': 0.7,
                                       'c_other': 0.45, 'c_self': 0.25})
    w_goal = World(goal_pair=(3, 5), **WORLD_KW)
    _, QAg = dp_tables(w_goal, 'A')
    _, QBg = dp_tables(w_goal, 'B')
    backbones = {
        'ablate_uniform_beliefs': build_rep(w, U, V, QA, QB, 'uniform'),
        'ablate_wrong_world': build_rep(w, U, V, QA, QB, 'wrongworld',
                                        w_wrong=w_wrong),
        'ablate_wrong_goal': build_rep(w_goal, U, V, QAg, QBg)}
    rr = np.random.default_rng(9)
    for tag, rp in backbones.items():
        best, best_th = 1e9, None
        for _ in range(1500):
            th = (rr.uniform(0.5, 8), rr.uniform(0, 1.5), rr.uniform(0, 1.5),
                  rr.uniform(0.05, 1), rr.uniform(0, 4), rr.uniform(0, 4),
                  rr.uniform(2, 40))
            P_u, P_v = heads(rp, rec, th)
            v = 0.5 * (mean_kl(rec['pu'], P_u) + mean_kl(rec['pv'], P_v))
            if v < best:
                best, best_th = v, th
        for _ in range(400):
            th = tuple(np.maximum(1e-3, np.array(best_th)
                                  * np.exp(rr.normal(0, 0.07, 7))))
            P_u, P_v = heads(rp, rec, th)
            v = 0.5 * (mean_kl(rec['pu'], P_u) + mean_kl(rec['pv'], P_v))
            if v < best:
                best, best_th = v, th
        metrics(rec, rp, best_th, tag)

    with open('results/rnn_fidelity.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print('wrote results/rnn_fidelity.json', flush=True)


if __name__ == '__main__':
    main()
