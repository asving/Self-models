"""f_synth v2 (v3 doc, Iteration 14): plan primitive = optimal-Q tilt at a
LEARNED temperature beta — testing 'the net is the optimal-weighted pi_g at
reduced strength'. Fit (beta, w_u, w_v, a, c_u, c_v, clip) on train
episodes; report held-out KL vs the v1 program (myopic plan, KL .0735) and
the closed-loop occupancy of the refit program (prediction: ~ net's .683).
cwd = 08_changeling.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World
from rnn import TorchWorld, GOAL_PAIRS
from eval_rnn import rollout_record, WORLD_KW, DEV
from probe import load
from oracle import replay_dists
from optimality import dp_tables
from whitebox_lambda import Filt

RES = {}
N_EP = 1536
BATCH = 256


def collect_q(post, rng):
    recs, reps, qAs, qBs = [], [], [], []
    for b in range(N_EP // BATCH):
        pair = GOAL_PAIRS[rng.integers(12)]
        w = World(goal_pair=pair, **WORLD_KW)
        _, QA = dp_tables(w, 'A')
        _, QB = dp_tables(w, 'B')
        rec = rollout_record(post, TorchWorld(w, DEV), pair, BATCH, 5200 + b)
        rep = replay_dists(w, rec['u'], rec['v'], return_beliefs=True)
        qA = np.einsum('rta,rtb,tabu->rtu', rep['drA'], rep['etaB'], QA)
        qB = np.einsum('rta,rtb,tabv->rtv', rep['etaA'], rep['drB'], QB)
        recs.append(rec); reps.append(rep); qAs.append(qA); qBs.append(qB)
    cat = lambda k, xs: np.concatenate([x[k] for x in xs])
    rec = {k: cat(k, recs) for k in ('u', 'v', 'pu', 'pv', 'iota')}
    rep = {k: cat(k, reps) for k in ('pbar_u', 'pbar_v')}
    rep['qA'] = np.concatenate(qAs).astype(np.float64)
    rep['qB'] = np.concatenate(qBs).astype(np.float64)
    return rec, rep


def heads(rep, rec, th):
    beta, w_u, w_v, a, c_u, c_v, clip = th
    planU = rep['pbar_u'] * np.exp(beta * (rep['qA'] - rep['qA'].max(-1, keepdims=True)))
    planU /= planU.sum(-1, keepdims=True)
    planV = rep['pbar_v'] * np.exp(beta * (rep['qB'] - rep['qB'].max(-1, keepdims=True)))
    planV /= planV.sum(-1, keepdims=True)
    R, T = rec['u'].shape
    r = np.arange(R)[:, None]; t = np.arange(T)[None, :]
    g_u = np.log(planU[r, t, rec['u']] + 1e-12) - np.log(rep['pbar_u'][r, t, rec['u']] + 1e-12)
    g_v = np.log(planV[r, t, rec['v']] + 1e-12) - np.log(rep['pbar_v'][r, t, rec['v']] + 1e-12)
    rho = np.clip(np.cumsum(w_u * g_u - w_v * g_v, 1), -clip, clip)
    rho = np.concatenate([np.zeros((R, 1)), rho[:, :-1]], 1)
    m_u = 1 / (1 + np.exp(-(a * rho + c_u)))
    m_v = 1 / (1 + np.exp(-(-a * rho + c_v)))
    P_u = m_u[..., None] * planU + (1 - m_u)[..., None] * rep['pbar_u']
    P_v = m_v[..., None] * planV + (1 - m_v)[..., None] * rep['pbar_v']
    return P_u, P_v


def mean_kl(P, Q):
    return float(np.mean((P * (np.log(P + 1e-12) - np.log(Q + 1e-12))).sum(-1)))


def obj(rep, rec, th, sl):
    P_u, P_v = heads(rep, rec, th)
    return 0.5 * (mean_kl(rec['pu'][sl], P_u[sl]) + mean_kl(rec['pv'][sl], P_v[sl]))


@torch.no_grad()
def agent(th, R=3000, seed=888):
    beta, w_u, w_v, a, c_u, c_v, clip = th
    w = World(goal_pair=(0, 2), **WORLD_KW)
    tw = TorchWorld(w, DEV)
    _, QA = dp_tables(w, 'A')
    _, QB = dp_tables(w, 'B')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    iota = rng.random(R) < 0.5
    sA = torch.randint(0, 6, (R,), device=DEV)
    sB = torch.randint(0, 6, (R,), device=DEV)
    f = Filt(w, R)
    rho = np.zeros(R)
    ball = np.zeros((R, w.T), np.float32)
    for t in range(w.T):
        pbar_u, pbar_v, _, _ = f.dists(t)
        qU = np.einsum('ra,rb,abu->ru', f.drA, f.etaB, QA[t])
        qV = np.einsum('ra,rb,abv->rv', f.etaA, f.drB, QB[t])
        planU = pbar_u * np.exp(beta * (qU - qU.max(1, keepdims=True)))
        planU /= planU.sum(1, keepdims=True)
        planV = pbar_v * np.exp(beta * (qV - qV.max(1, keepdims=True)))
        planV /= planV.sum(1, keepdims=True)
        m_u = 1 / (1 + np.exp(-(a * rho + c_u)))
        m_v = 1 / (1 + np.exp(-(-a * rho + c_v)))
        P_u = m_u[:, None] * planU + (1 - m_u)[:, None] * pbar_u
        P_v = m_v[:, None] * planV + (1 - m_v)[:, None] * pbar_v
        cum = np.cumsum(P_u, 1)
        un_net = np.argmax(cum > rng.random((R, 1)), 1)
        cum = np.cumsum(P_v, 1)
        vn_net = np.argmax(cum > rng.random((R, 1)), 1)
        u_env, v_env = tw.emit(sA, sB)
        un = np.where(iota, un_net, u_env.cpu().numpy())
        vn = np.where(iota, v_env.cpu().numpy(), vn_net)
        ri = np.arange(R)
        g_u = np.log(planU[ri, un] + 1e-12) - np.log(pbar_u[ri, un] + 1e-12)
        g_v = np.log(planV[ri, vn] + 1e-12) - np.log(pbar_v[ri, vn] + 1e-12)
        rho = np.clip(rho + w_u * g_u - w_v * g_v, -clip, clip)
        f.update(un, vn)
        sA, sB = tw.trans(sA, sB, torch.tensor(un, device=DEV),
                          torch.tensor(vn, device=DEV))
        ball[:, t] = tw.ball(sA, sB).float().cpu().numpy()
    return float(ball.mean())


def main():
    rng = np.random.default_rng(31)
    post = load('post_6000')
    rec, rep = collect_q(post, rng)
    idx = rng.permutation(N_EP)
    tr, te = idx[:N_EP * 3 // 4], idx[N_EP * 3 // 4:]
    best, best_th = 1e9, None
    for i in range(3000):
        th = (rng.uniform(0.5, 8), rng.uniform(0, 1.5), rng.uniform(0, 1.5),
              rng.uniform(0.05, 1), rng.uniform(0, 4), rng.uniform(0, 4),
              rng.uniform(2, 40))
        v = obj(rep, rec, th, tr)
        if v < best:
            best, best_th = v, th
    for _ in range(800):
        th = tuple(np.maximum(1e-3, np.array(best_th)
                              * np.exp(rng.normal(0, 0.07, 7))))
        v = obj(rep, rec, th, tr)
        if v < best:
            best, best_th = v, th
    RES['theta'] = {k: round(float(x), 3) for k, x in
                    zip(('beta', 'w_u', 'w_v', 'a', 'c_u', 'c_v', 'clip'),
                        best_th)}
    RES['kl_train'] = round(best, 4)
    RES['kl_test'] = round(obj(rep, rec, best_th, te), 4)
    RES['kl_test_v1_myopic_plan'] = 0.0735
    print(json.dumps(RES, indent=1), flush=True)
    RES['closed_loop_occ'] = round(agent(best_th), 4)
    RES['net_occ'] = 0.683
    print('closed-loop occ:', RES['closed_loop_occ'], flush=True)
    with open('results/rnn_synth2.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)


if __name__ == '__main__':
    main()
