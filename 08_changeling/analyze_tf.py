"""Transformer replication battery (v4 arc): the three questions.
Q1 pretrain: belief geometry + off-manifold filter fidelity.
Q2 midtrain: flag-conditioned tilt (toggle matrix, tilt-direction slope)
   + off-manifold generalization of both jobs.
Q3 posttrain: occupancy vs floors; whitebox-v2 (beta) fit; identity-
   conditional claims curves; lambda decodability; spectator idle test.
Writes results/tf_analysis.json. cwd = 08_changeling.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from worlds import World
from rnn import TorchWorld, features, step_features, GOAL_PAIRS, N
from eval_rnn import WORLD_KW, DEV
from tfmodel import ChangelingTF
from oracle import run_base, run_episodes, replay_dists
from optimality import dp_tables
from offmanifold import build_rep, coef, kl, tiltr2
from spectator import both_tilted_records
from fidelity import informedq_records
from synth2 import heads as synth_heads, mean_kl

RES = {}
R = 768
BATCH = 256
PAIR = (0, 2)


def load_tf(name):
    m = ChangelingTF().to(DEV)
    m.load_state_dict(torch.load(f'ckpt/tf_{name}.pt'))
    m.eval()
    return m


@torch.no_grad()
def tf_pass(model, U, V, goals=None, flag=None):
    pus, pvs, hs = [], [], []
    for i in range(0, len(U), BATCH):
        io = None if flag is None else np.full(len(U[i:i + BATCH]), flag, bool)
        g = None if goals is None else goals[i:i + BATCH]
        X = torch.tensor(features(U[i:i + BATCH], V[i:i + BATCH], g, io),
                         device=DEV)
        lu, lv, h = model(X)
        T = U.shape[1]
        pus.append(F.softmax(lu[:, :T], -1).cpu().numpy())
        pvs.append(F.softmax(lv[:, :T], -1).cpu().numpy())
        hs.append(h[:, :T].cpu().numpy())
    return np.concatenate(pus), np.concatenate(pvs), np.concatenate(hs)


@torch.no_grad()
def tf_rollout(model, w, tw, R_, seed):
    torch.manual_seed(seed)
    goals_t = torch.tensor(PAIR, device=DEV).repeat(R_, 1)
    iota = torch.rand(R_, device=DEV) < 0.5
    sA = torch.randint(0, N, (R_,), device=DEV)
    sB = torch.randint(0, N, (R_,), device=DEV)
    buf = None; u = v = None
    U = np.zeros((R_, w.T), np.int64); V = np.zeros((R_, w.T), np.int64)
    ball = np.zeros((R_, w.T), np.float32)
    for t in range(w.T):
        x = step_features(u, v, goals_t, t, w.T, DEV)
        lu, lv, buf = model.step(x, buf)
        pu, pv = F.softmax(lu, -1), F.softmax(lv, -1)
        u_net = torch.multinomial(pu, 1).squeeze(1)
        v_net = torch.multinomial(pv, 1).squeeze(1)
        ue, ve = tw.emit(sA, sB)
        u = torch.where(iota, u_net, ue)
        v = torch.where(iota, ve, v_net)
        U[:, t] = u.cpu().numpy(); V[:, t] = v.cpu().numpy()
        sA, sB = tw.trans(sA, sB, u, v)
        ball[:, t] = tw.ball(sA, sB).float().cpu().numpy()
    return U, V, iota.cpu().numpy(), float(ball.mean())


def ridge6(H, G, tr):
    Z = H.reshape(-1, H.shape[-1]); y = G.reshape(-1, 6)
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
    Zs = (Z - mu) / sd
    A = Zs[tr].T @ Zs[tr] + 1.0 * np.eye(Z.shape[-1])
    W = np.linalg.solve(A, Zs[tr].T @ (y[tr] - y[tr].mean(0)))
    return (W, mu, sd, y[tr].mean(0))


def r2d(dec, H, G):
    W, mu, sd, ym = dec
    pred = ((H.reshape(-1, H.shape[-1]) - mu) / sd) @ W + ym
    y = G.reshape(-1, 6)
    return round(float(1 - ((y - pred) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()), 4)


def main():
    rng = np.random.default_rng(171)
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    _, QA = dp_tables(w, 'A'); _, QB = dp_tables(w, 'B')
    goals = np.tile(np.array(PAIR), (R, 1))

    # ---------- Q1: pretrained transformer ----------
    pre = load_tf('pre_final')
    b = run_base(w, R, 14100, collect=True)
    sets1 = {'base_law': (b['u'].astype(np.int64), b['v'].astype(np.int64)),
             'informedQ': informedq_records(w, tw, QA, QB, R, 14200),
             'random': (rng.integers(0, 6, (R, 32)), rng.integers(0, 6, (R, 32)))}
    decs = None
    for tag, (U, V) in sets1.items():
        rep = replay_dists(w, U, V, return_beliefs=True)
        pu, pv, H = tf_pass(pre, U, V)
        RES[f'Q1_{tag}_KL_exact_to_net'] = round(
            0.5 * (kl(rep['pbar_u'], pu) + kl(rep['pbar_v'], pv)), 4)
        if tag == 'base_law':
            half = np.arange(R // 2 * 32)
            decs = (ridge6(H[:R // 2], rep['etaA'][:R // 2], np.arange(R // 2 * 32)),
                    ridge6(H[:R // 2], rep['etaB'][:R // 2], np.arange(R // 2 * 32)))
            RES['Q1_belief_r2_base'] = 0.5 * (
                r2d(decs[0], H[R // 2:], rep['etaA'][R // 2:])
                + r2d(decs[1], H[R // 2:], rep['etaB'][R // 2:]))
        else:
            RES[f'Q1_belief_transfer_{tag}'] = 0.5 * (
                r2d(decs[0], H, rep['etaA']) + r2d(decs[1], H, rep['etaB']))
    print('Q1:', {k: v for k, v in RES.items() if k.startswith('Q1')}, flush=True)

    # ---------- Q2: midtrained transformer ----------
    mid = load_tf('mid_final')
    r_inf = run_episodes(w, 'informed', R, 14300, collect=True)
    Ui, Vi = r_inf['traj']['u'].astype(np.int64), r_inf['traj']['v'].astype(np.int64)
    repi = replay_dists(w, Ui, Vi)
    row = {}
    heads_by_flag = {}
    for f_, lab in ((True, 'A'), (False, 'B'), (None, '0')):
        pu, pv, _ = tf_pass(mid, Ui, Vi, goals, flag=f_)
        heads_by_flag[lab] = (pu, pv)
        row[f'coef_u_flag{lab}'] = round(float(np.nanmean(
            coef(pu, repi['piA'], repi['pbar_u'])[:, 4:])), 3)
        row[f'coef_v_flag{lab}'] = round(float(np.nanmean(
            coef(pv, repi['piB'], repi['pbar_v'])[:, 4:])), 3)
    tilt = np.log(repi['piA'] + 1e-12) - np.log(repi['pbar_u'] + 1e-12)
    dlog = (np.log(heads_by_flag['A'][0] + 1e-12)
            - np.log(heads_by_flag['B'][0] + 1e-12))
    tc = tilt - tilt.mean(-1, keepdims=True)
    dc = dlog - dlog.mean(-1, keepdims=True)
    sl = float((tc * dc).sum() / (tc * tc).sum())
    r2 = float(1 - ((dc - sl * tc) ** 2).sum() / ((dc - dc.mean()) ** 2).sum())
    row['E2_slope_r2'] = [round(sl, 3), round(r2, 3)]
    for tag, (U, V) in (('base_law', sets1['base_law']),
                        ('random', sets1['random'])):
        rep = replay_dists(w, U, V)
        pu, pv, _ = tf_pass(mid, U, V, goals, flag=True)
        row[f'offman_{tag}_KLplan_u__KLpbar_v'] = [
            round(kl(rep['piA'], pu), 4), round(kl(rep['pbar_v'], pv), 4)]
    RES['Q2_mid'] = row
    print('Q2:', row, flush=True)

    # ---------- Q3: post transformer ----------
    post = load_tf('post_6000')
    U, V, io, occ = tf_rollout(post, w, tw, 1536, 14400)
    RES['Q3_occ_onpolicy'] = round(occ, 4)
    pu, pv, H = tf_pass(post, U, V, np.tile(np.array(PAIR), (1536, 1)))
    rep = build_rep(w, U, V, QA, QB)
    rec = {'u': U, 'v': V, 'pu': pu, 'pv': pv}
    # identity-conditional claims curves
    cu = coef(pu, rep['piA'], rep['pbar_u'])
    cv = coef(pv, rep['piB'], rep['pbar_v'])
    cs = np.where(io[:, None], cu, cv); co = np.where(io[:, None], cv, cu)
    RES['Q3_claims_self_t0_8_16_31'] = [round(float(np.nanmean(cs[:, k])), 3)
                                        for k in (0, 8, 16, 31)]
    RES['Q3_claims_other_t0_8_16_31'] = [round(float(np.nanmean(co[:, k])), 3)
                                         for k in (0, 8, 16, 31)]
    # whitebox-v2 fit
    idx = rng.permutation(1536)
    tr_, te_ = idx[:1152], idx[1152:]
    best, best_th = 1e9, None
    rr = np.random.default_rng(7)
    for _ in range(3000):
        th = (rr.uniform(0.5, 8), rr.uniform(0, 1.5), rr.uniform(0, 1.5),
              rr.uniform(0.05, 1), rr.uniform(0, 4), rr.uniform(0, 4),
              rr.uniform(2, 40))
        P_u, P_v = synth_heads(rep, rec, th)
        v_ = 0.5 * (mean_kl(pu[tr_], P_u[tr_]) + mean_kl(pv[tr_], P_v[tr_]))
        if v_ < best:
            best, best_th = v_, th
    for _ in range(600):
        th = tuple(np.maximum(1e-3, np.array(best_th) * np.exp(rr.normal(0, 0.07, 7))))
        P_u, P_v = synth_heads(rep, rec, th)
        v_ = 0.5 * (mean_kl(pu[tr_], P_u[tr_]) + mean_kl(pv[tr_], P_v[tr_]))
        if v_ < best:
            best, best_th = v_, th
    P_u, P_v = synth_heads(rep, rec, best_th)
    RES['Q3_whitebox_kl_test'] = round(
        0.5 * (mean_kl(pu[te_], P_u[te_]) + mean_kl(pv[te_], P_v[te_])), 4)
    RES['Q3_whitebox_theta'] = {k: round(float(x), 3) for k, x in
                                zip(('beta', 'w_u', 'w_v', 'a', 'c_u', 'c_v',
                                     'clip'), best_th)}
    RES['Q3_kl_vs_neutral'] = round(
        0.5 * (mean_kl(pu[te_], rep['pbar_u'][te_])
               + mean_kl(pv[te_], rep['pbar_v'][te_])), 4)
    # lambda decodability
    r_i = np.arange(1536)[:, None]; t_i = np.arange(32)[None, :]
    dl = (np.log(pu[r_i, t_i, U]) - np.log(rep['pbar_u'][r_i, t_i, U] + 1e-12)
          + np.log(rep['pbar_v'][r_i, t_i, V] + 1e-12) - np.log(pv[r_i, t_i, V]))
    lam = np.concatenate([np.zeros((1536, 1)),
                          np.clip(np.cumsum(dl, 1), -20, 20)[:, :-1]], 1)
    Z = H.reshape(-1, post.d)
    y = lam.reshape(-1)
    m_tr = np.isin(np.repeat(np.arange(1536), 32), tr_)
    mu, sd = Z[m_tr].mean(0), Z[m_tr].std(0) + 1e-8
    Zs = (Z - mu) / sd
    A = Zs[m_tr].T @ Zs[m_tr] + 1.0 * np.eye(post.d)
    ym = y[m_tr].mean()
    wd = np.linalg.solve(A, Zs[m_tr].T @ (y[m_tr] - ym))
    pred = Zs @ wd + ym
    m_te = ~m_tr
    RES['Q3_lambda_decode_r2'] = round(float(
        1 - ((y[m_te] - pred[m_te]) ** 2).sum()
        / ((y[m_te] - y[m_te].mean()) ** 2).sum()), 3)
    # spectator idle test
    for tag, (Us, Vs) in (('base_law', sets1['base_law']),
                          ('both_tilted', (both_tilted_records(w, 14500))),
                          ('random', sets1['random'])):
        reps = build_rep(w, Us, Vs, QA, QB)
        pus, pvs, Hs = tf_pass(post, Us, Vs, goals)
        cus = coef(pus, reps['piA'], reps['pbar_u'])
        cvs = coef(pvs, reps['piB'], reps['pbar_v'])
        lam_dec = (((Hs.reshape(-1, post.d) - mu) / sd) @ wd + ym).reshape(R, 32)
        RES[f'Q3_spect_{tag}'] = {
            'claims_u_v_late': [round(float(np.nanmean(cus[:, 8:])), 3),
                                round(float(np.nanmean(cvs[:, 8:])), 3)],
            'med_abs_lam_dec_late': round(float(
                np.median(np.abs(lam_dec[:, 8:]))), 2)}
        print(f'Q3 spect {tag}:', RES[f'Q3_spect_{tag}'], flush=True)

    with open('results/tf_analysis.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print(json.dumps({k: v for k, v in RES.items()
                      if not k.startswith('Q3_spect')}, indent=1), flush=True)
    print('wrote results/tf_analysis.json', flush=True)


if __name__ == '__main__':
    main()
