"""Internal identity state off-policy (v3, Iteration 24; Asvin's probe ask).
Per record set: decoded lambda (on-policy-calibrated ridge readout, frozen),
the causal m-hat register coefficient, the record-ground-truth lambda, and
the GATE COHERENCE test: does one sigma curve map decoded lambda to the
measured claims across ALL sets? Plus belief-sufficiency: do recent tokens
add tilt-variance beyond the beliefs? cwd = 08_changeling.
"""
import json
import numpy as np
import torch
from worlds import World
from rnn import TorchWorld
from eval_rnn import WORLD_KW, DEV, rollout_record
from probe import load, split
from probe3 import collect_full
from oracle import run_base
from optimality import dp_tables
from offmanifold import build_rep, coef
from spectator import both_tilted_records
from qextract2 import tf_full
from format import match_pairs
from distill import distill
from whitebox_lambda import prefix
from approx import fit_tables

RES = {}
R = 768
BATCH = 256
PAIR = (0, 2)


def main():
    rng = np.random.default_rng(151)
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    _, QA = dp_tables(w, 'A')
    _, QB = dp_tables(w, 'B')
    goals = np.tile(np.array(PAIR), (R, 1))

    # lambda decoder fit on-policy (calibrated regime)
    H, gt = collect_full(post, 1024, rng)
    tr, te = split(1024)
    Zf = H.reshape(-1, 256)
    y = np.clip(gt['lam_logodds'], -20, 20).reshape(-1)
    ep = np.repeat(np.arange(1024), 32)
    m_tr = np.isin(ep, tr)
    mu, sd = Zf[m_tr].mean(0), Zf[m_tr].std(0) + 1e-8
    Z = (Zf - mu) / sd
    A = Z[m_tr].T @ Z[m_tr] + 1.0 * np.eye(256)
    ym = y[m_tr].mean()
    wd = np.linalg.solve(A, Z[m_tr].T @ (y[m_tr] - ym))

    # m-hat register direction (distillation, as in Iteration 8)
    st = prefix(post, w, tw, seed=77)
    donor = match_pairs(st)
    mvecs = distill(post, w, st, donor)
    mhat = mvecs / (np.linalg.norm(mvecs, axis=1, keepdims=True) + 1e-9)
    sgn = np.where(st['iota'].cpu().numpy(), 1.0, -1.0)[:, None]
    _, _, Vt = np.linalg.svd(sgn * mhat, full_matrices=False)
    mg = Vt[0]
    if np.corrcoef(st['h'][0].cpu().numpy() @ mg,
                   st['iota'].cpu().numpy().astype(float))[0, 1] < 0:
        mg = -mg

    sets = {}
    recs = [rollout_record(post, tw, PAIR, BATCH, 13100 + k) for k in range(R // BATCH)]
    sets['onpolicy'] = (np.concatenate([x['u'] for x in recs]).astype(np.int64),
                        np.concatenate([x['v'] for x in recs]).astype(np.int64))
    b = run_base(w, R, 13200, collect=True)
    sets['base_law'] = (b['u'].astype(np.int64), b['v'].astype(np.int64))
    sets['random'] = (rng.integers(0, 6, (R, 32)), rng.integers(0, 6, (R, 32)))
    sets['both_tilted'] = both_tilted_records(w, 13300)

    pool_lam, pool_claim_u, pool_claim_v, pool_tag = [], [], [], []
    for tag, (U, V) in sets.items():
        rep = build_rep(w, U, V, QA, QB)
        pu, pv, Hs = tf_full(post, U, V, goals)
        lam_dec = (((Hs.reshape(-1, 256) - mu) / sd) @ wd + ym).reshape(R, 32)
        reg = (Hs @ mg).reshape(R, 32)
        # record-truth lambda from net heads vs exact pbar
        r = np.arange(R)[:, None]; t = np.arange(32)[None, :]
        dl = (np.log(pu[r, t, U]) - np.log(rep['pbar_u'][r, t, U] + 1e-12)
              + np.log(rep['pbar_v'][r, t, V] + 1e-12) - np.log(pv[r, t, V]))
        lam_true = np.concatenate([np.zeros((R, 1)),
                                   np.clip(np.cumsum(dl, 1), -20, 20)[:, :-1]], 1)
        cu = coef(pu, rep['piA'], rep['pbar_u'])
        cv = coef(pv, rep['piB'], rep['pbar_v'])
        RES[tag] = {
            'med_abs_lam_dec_t8_16_31': [round(float(np.median(np.abs(lam_dec[:, k]))), 2)
                                         for k in (8, 16, 31)],
            'med_abs_lam_true_t8_16_31': [round(float(np.median(np.abs(lam_true[:, k]))), 2)
                                          for k in (8, 16, 31)],
            'corr_lamdec_lamtrue': round(float(np.corrcoef(
                lam_dec[:, 4:].ravel(), lam_true[:, 4:].ravel())[0, 1]), 3),
            'med_abs_register': round(float(np.median(np.abs(reg[:, 8:]))), 3),
            'claims_u_v_late': [round(float(np.nanmean(cu[:, 8:])), 3),
                                round(float(np.nanmean(cv[:, 8:])), 3)]}
        print(tag, RES[tag], flush=True)
        pool_lam.append(lam_dec[:, 4:].ravel())
        pool_claim_u.append(cu[:, 4:].ravel())
        pool_claim_v.append(cv[:, 4:].ravel())
        pool_tag += [tag] * lam_dec[:, 4:].size
        if tag == 'onpolicy':
            RES['register_onpolicy_signedspread'] = round(
                float(np.median(np.abs(reg[:, 8:]))), 3)

    # gate coherence: one sigma curve across all sets?
    lam = np.concatenate(pool_lam)
    for ch, pool in (('u', pool_claim_u), ('v', pool_claim_v)):
        m = np.concatenate(pool)
        s = 1.0 if ch == 'u' else -1.0
        ok = (~np.isnan(m)) & (m > 0.02) & (m < 0.98)
        yl = np.log(m[ok] / (1 - m[ok]))
        X = np.stack([s * lam[ok], np.ones(ok.sum())], 1)
        (aa, cc), *_ = np.linalg.lstsq(X, yl, rcond=None)[:1]
        r2 = 1 - ((yl - X @ [aa, cc]) ** 2).sum() / ((yl - yl.mean()) ** 2).sum()
        RES[f'gate_pooled_{ch}'] = [round(float(aa), 3), round(float(cc), 3),
                                    round(float(r2), 3), int(ok.sum())]
    print('gate pooled:', RES['gate_pooled_u'], RES['gate_pooled_v'], flush=True)

    # belief sufficiency: tokens beyond beliefs (base_law set)
    U, V = sets['base_law']
    rep = build_rep(w, U, V, QA, QB)
    pu, pv, _ = tf_full(post, U, V, goals)
    tau = np.log(pu + 1e-12) - np.log(rep['pbar_u'] + 1e-12)
    tau -= tau.mean(-1, keepdims=True)
    trh, teh = np.arange(R // 2), np.arange(R // 2, R)
    r2_bel = fit_tables(tau, rep['drA'], rep['etaB'], trh, teh)
    tokA = np.zeros((R, 32, 6)); tokB = np.zeros((R, 32, 6))
    ri = np.arange(R)[:, None]; ti = np.arange(1, 32)[None, :]
    tokA[ri, ti, U[:, :-1]] = 1; tokB[ri, ti, V[:, :-1]] = 1
    fA = np.concatenate([rep['drA'], tokA], -1)
    fB = np.concatenate([rep['etaB'], tokB], -1)
    r2_both = fit_tables(tau, fA, fB, trh, teh)
    RES['belief_sufficiency_base'] = {'beliefs_only': r2_bel,
                                      'beliefs_plus_tokens': r2_both}
    print('belief sufficiency:', RES['belief_sufficiency_base'], flush=True)

    with open('results/rnn_lamprobe_off.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print('wrote results/rnn_lamprobe_off.json', flush=True)


if __name__ == '__main__':
    main()
