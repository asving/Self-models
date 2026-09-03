"""Localizing the post-net's off-manifold fraying (v3 doc, Iteration 18).

Record ladder: informed-oracle records (mid net's manifold), base-law,
uniform-random tokens, post-net on-policy records.
(1) MID net (flag=A teacher-forced): job fidelity per set — KL(plan||u-head),
    KL(pbar||v-head), tilt-R2 on u. Flat across sets => the fraying is a
    post-training acquisition.
(2) POST net: realized per-round claims (coef vs exact {plan, pbar} basis)
    vs the whitebox-v2 PREDICTED claims on the same records; and the gate
    law refit per set: logit(claim) ~ a*rho_template + c (same law? same
    constants?). Also the court-state coverage: distribution of
    (g_u, g_v) evidence increments per set (the never-visited
    both-negative cone). cwd = 08_changeling.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from worlds import World
from rnn import TorchWorld, features
from eval_rnn import rollout_record, WORLD_KW, DEV
from probe import load
from oracle import replay_dists, run_base, run_episodes
from optimality import dp_tables
from fidelity import informedq_records

RES = {}
R = 768
BATCH = 256
PAIR = (0, 2)
TH = dict(beta=3.868, w_u=0.622, w_v=0.473, a=0.274, c_u=0.389, c_v=0.409,
          clip=29.704)


@torch.no_grad()
def tf_heads(model, U, V, goals=None, flag=None):
    pus, pvs = [], []
    for i in range(0, len(U), BATCH):
        io = None if flag is None else np.full(len(U[i:i + BATCH]), flag, bool)
        g = None if goals is None else goals[i:i + BATCH]
        X = torch.tensor(features(U[i:i + BATCH], V[i:i + BATCH], g, io),
                         device=DEV)
        lu, lv, _ = model(X)
        T = U.shape[1]
        pus.append(F.softmax(lu[:, :T], -1).cpu().numpy())
        pvs.append(F.softmax(lv[:, :T], -1).cpu().numpy())
    return np.concatenate(pus), np.concatenate(pvs)


def kl(P, Q):
    return float(np.mean((P * (np.log(P + 1e-12) - np.log(Q + 1e-12))).sum(-1)))


def tiltr2(P, Pref, pbar):
    y = np.log(P + 1e-12) - np.log(pbar + 1e-12)
    x = np.log(Pref + 1e-12) - np.log(pbar + 1e-12)
    y -= y.mean(-1, keepdims=True); x -= x.mean(-1, keepdims=True)
    y, x = y.ravel(), x.ravel()
    b = (x * y).sum() / ((x * x).sum() + 1e-12)
    return round(float(1 - ((y - b * x) ** 2).sum() / ((y ** 2).sum() + 1e-12)), 3)


def coef(P, plan, pbar):
    d = plan - pbar
    den = (d * d).sum(-1)
    c = ((P - pbar) * d).sum(-1) / np.maximum(den, 1e-12)
    return np.where(den > 1e-4, np.clip(c, 0, 1), np.nan)


def synth_claims_and_rho(rep, rec):
    """Whitebox-v2 predicted claims + template-court rho on given records."""
    beta = TH['beta']
    planU = rep['pbar_u'] * np.exp(beta * (rep['qA'] - rep['qA'].max(-1, keepdims=True)))
    planU /= planU.sum(-1, keepdims=True)
    planV = rep['pbar_v'] * np.exp(beta * (rep['qB'] - rep['qB'].max(-1, keepdims=True)))
    planV /= planV.sum(-1, keepdims=True)
    Rn, T = rec['u'].shape
    r = np.arange(Rn)[:, None]; t = np.arange(T)[None, :]
    g_u = np.log(planU[r, t, rec['u']] + 1e-12) - np.log(rep['pbar_u'][r, t, rec['u']] + 1e-12)
    g_v = np.log(planV[r, t, rec['v']] + 1e-12) - np.log(rep['pbar_v'][r, t, rec['v']] + 1e-12)
    rho = np.clip(np.cumsum(TH['w_u'] * g_u - TH['w_v'] * g_v, 1),
                  -TH['clip'], TH['clip'])
    rho = np.concatenate([np.zeros((Rn, 1)), rho[:, :-1]], 1)
    m_u = 1 / (1 + np.exp(-(TH['a'] * rho + TH['c_u'])))
    m_v = 1 / (1 + np.exp(-(-TH['a'] * rho + TH['c_v'])))
    return m_u, m_v, rho, g_u, g_v, planU, planV


def build_rep(w, U, V, QA, QB):
    rep = replay_dists(w, U, V, return_beliefs=True)
    rep['qA'] = np.einsum('rta,rtb,tabu->rtu', rep['drA'], rep['etaB'], QA)
    rep['qB'] = np.einsum('rta,rtb,tabv->rtv', rep['etaA'], rep['drB'], QB)
    return rep


def main():
    rng = np.random.default_rng(83)
    mid = load('mid_final')
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    _, QA = dp_tables(w, 'A')
    _, QB = dp_tables(w, 'B')
    goals = np.tile(np.array(PAIR), (R, 1))

    sets = {}
    r_inf = run_episodes(w, 'informed', R, 8100, collect=True)
    sets['informed_oracle(mid manifold)'] = (r_inf['traj']['u'].astype(np.int64),
                                             r_inf['traj']['v'].astype(np.int64))
    b = run_base(w, R, 8200, collect=True)
    sets['base_law'] = (b['u'].astype(np.int64), b['v'].astype(np.int64))
    sets['random_tokens'] = (rng.integers(0, 6, (R, w.T)),
                             rng.integers(0, 6, (R, w.T)))
    recs = [rollout_record(post, tw, PAIR, BATCH, 8300 + k) for k in range(R // BATCH)]
    sets['post_onpolicy'] = (np.concatenate([x['u'] for x in recs]).astype(np.int64),
                             np.concatenate([x['v'] for x in recs]).astype(np.int64))

    for tag, (U, V) in sets.items():
        rep = build_rep(w, U, V, QA, QB)
        # (1) MID net, flag=A: u-head must play plan (=piA), v-head forecast
        pu, pv = tf_heads(mid, U, V, goals, flag=True)
        RES[f'mid_{tag}'] = {
            'KL_plan_to_uhead': round(kl(rep['piA'], pu), 4),
            'KL_pbar_to_vhead': round(kl(rep['pbar_v'], pv), 4),
            'tilt_r2_u': tiltr2(pu, rep['piA'], rep['pbar_u'])}
        print(f'MID {tag}:', RES[f'mid_{tag}'], flush=True)
        # (2) POST net: realized vs whitebox-predicted claims + gate refit
        pu, pv = tf_heads(post, U, V, goals, flag=None)
        rec = {'u': U, 'v': V}
        m_u_hat, m_v_hat, rho, g_u, g_v, _, _ = synth_claims_and_rho(rep, rec)
        cu = coef(pu, rep['piA'], rep['pbar_u'])
        cv = coef(pv, rep['piB'], rep['pbar_v'])
        gap = 0.5 * (np.nanmean(np.abs(cu - m_u_hat))
                     + np.nanmean(np.abs(cv - m_v_hat)))
        ok = (~np.isnan(cu)) & (cu > 0.02) & (cu < 0.98)
        yl = np.log(cu[ok] / (1 - cu[ok]))
        A2 = np.stack([rho[ok], np.ones(ok.sum())], 1)
        (aa, cc), *_ = np.linalg.lstsq(A2, yl, rcond=None)[:1]
        r2g = 1 - ((yl - A2 @ np.array([aa, cc])) ** 2).sum() / ((yl - yl.mean()) ** 2).sum()
        RES[f'post_{tag}'] = {
            'claim_gap_mean_abs': round(float(gap), 3),
            'net_claims_u_t0_8_16_31': [round(float(np.nanmean(cu[:, k])), 3)
                                        for k in (0, 8, 16, 31)],
            'synth_claims_u_t0_8_16_31': [round(float(np.nanmean(m_u_hat[:, k])), 3)
                                          for k in (0, 8, 16, 31)],
            'gate_refit_a_c_r2': [round(float(aa), 3), round(float(cc), 3),
                                  round(float(r2g), 3)],
            'evidence_cone_frac_bothneg': round(float(
                np.mean((g_u < 0) & (g_v < 0))), 3)}
        print(f'POST {tag}:', RES[f'post_{tag}'], flush=True)

    with open('results/rnn_offmanifold.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print('wrote results/rnn_offmanifold.json', flush=True)


if __name__ == '__main__':
    main()
