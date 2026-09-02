"""Identifiability & coverage of the 7-parameter whitebox (v3, Iter. 15).

(1) Per-parameter profile curves around theta-hat: CI where held-out KL
    rises by delta = 2*SE(KL) (SE via per-episode bootstrap at theta-hat).
(2) Hessian on log-parameters -> eigen-spectrum: stiff vs sloppy directions
    (the known register-scale gauge should appear as a near-flat direction).
(3) Coverage controls: KL distribution over random theta draws; best
    achievable KL when the family is fit to an EPISODE-SHUFFLED target
    (marginals kept, history-dependence broken). cwd = 08_changeling.
"""
import json
import numpy as np
from probe import load
from synth2 import collect_q, heads, N_EP

RES = {}
NAMES = ('beta', 'w_u', 'w_v', 'a', 'c_u', 'c_v', 'clip')
TH = np.array([3.868, 0.622, 0.473, 0.274, 0.389, 0.409, 29.704])


def kl_terms(P, Q):
    return (P * (np.log(P + 1e-12) - np.log(Q + 1e-12))).sum(-1)


def obj_ep(rep, rec, th, sl):
    """Per-episode mean KL (for bootstrap), and its mean."""
    P_u, P_v = heads(rep, rec, tuple(th))
    per = 0.5 * (kl_terms(rec['pu'][sl], P_u[sl]).mean(-1)
                 + kl_terms(rec['pv'][sl], P_v[sl]).mean(-1))
    return per


def main():
    rng = np.random.default_rng(31)
    post = load('post_6000')
    rec, rep = collect_q(post, rng)
    idx = rng.permutation(N_EP)
    tr, te = idx[:N_EP * 3 // 4], idx[N_EP * 3 // 4:]

    per = obj_ep(rep, rec, TH, te)
    kl0 = float(per.mean())
    se = float(per.std() / np.sqrt(len(per)))
    delta = 2 * se
    RES['kl_test_at_thetahat'] = round(kl0, 5)
    RES['se'] = round(se, 5)
    print(f'kl {kl0:.5f} +- {se:.5f}', flush=True)

    # (1) profiles
    mults = np.exp(np.linspace(np.log(0.4), np.log(2.5), 15))
    profiles, cis = {}, {}
    for i, nm in enumerate(NAMES):
        kls = []
        for m in mults:
            th = TH.copy(); th[i] *= m
            kls.append(float(obj_ep(rep, rec, th, te).mean()))
        kls = np.array(kls)
        ok = mults[kls <= kl0 + delta]
        profiles[nm] = [round(float(k), 5) for k in kls]
        cis[nm] = [round(float(ok.min()), 3), round(float(ok.max()), 3)] \
            if len(ok) else None
    RES['ci_multipliers_2se'] = cis
    print('CIs (x range):', cis, flush=True)

    # (2) Hessian on log-params (train split for stability)
    eps = 0.05
    def f(lth):
        return float(obj_ep(rep, rec, np.exp(lth), tr).mean())
    l0 = np.log(TH)
    H = np.zeros((7, 7))
    f0 = f(l0)
    for i in range(7):
        for j in range(i, 7):
            li = l0.copy(); li[i] += eps; lj = l0.copy(); lj[j] += eps
            lij = l0.copy(); lij[i] += eps; lij[j] += eps
            H[i, j] = H[j, i] = (f(lij) - f(li) - f(lj) + f0) / eps ** 2
    ev, evec = np.linalg.eigh(H)
    RES['hessian_eigs'] = [round(float(x), 4) for x in ev]
    RES['sloppiest_dir'] = {n: round(float(v), 2)
                            for n, v in zip(NAMES, evec[:, 0])}
    RES['stiffest_dir'] = {n: round(float(v), 2)
                           for n, v in zip(NAMES, evec[:, -1])}
    print('eigs:', RES['hessian_eigs'], flush=True)
    print('sloppy:', RES['sloppiest_dir'], 'stiff:', RES['stiffest_dir'],
          flush=True)

    # (3a) random-theta coverage
    rr = np.random.default_rng(7)
    kls = []
    for _ in range(400):
        th = np.array([rr.uniform(0.5, 8), rr.uniform(0, 1.5),
                       rr.uniform(0, 1.5), rr.uniform(0.05, 1),
                       rr.uniform(0, 4), rr.uniform(0, 4),
                       rr.uniform(2, 40)])
        kls.append(float(obj_ep(rep, rec, th, te).mean()))
    RES['random_theta_kl_quartiles'] = [round(float(q), 4) for q in
                                        np.percentile(kls, [5, 25, 50, 75, 95])]
    print('random theta KL quantiles:', RES['random_theta_kl_quartiles'],
          flush=True)

    # (3b) shuffled-target fit: same records, heads from permuted episodes
    perm = rr.permutation(N_EP)
    rec_sh = dict(rec)
    rec_sh['pu'] = rec['pu'][perm]
    rec_sh['pv'] = rec['pv'][perm]
    best = 1e9
    for _ in range(1500):
        th = np.array([rr.uniform(0.5, 8), rr.uniform(0, 1.5),
                       rr.uniform(0, 1.5), rr.uniform(0.05, 1),
                       rr.uniform(0, 4), rr.uniform(0, 4),
                       rr.uniform(2, 40)])
        v = float(obj_ep(rep, rec_sh, th, tr).mean())
        if v < best:
            best, best_th = v, th
    for _ in range(400):
        th = np.maximum(1e-3, best_th * np.exp(rr.normal(0, 0.07, 7)))
        v = float(obj_ep(rep, rec_sh, th, tr).mean())
        if v < best:
            best, best_th = v, th
    RES['shuffled_target_best_kl'] = round(best, 4)
    print('shuffled-target best KL:', best, flush=True)

    with open('results/rnn_params.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print('wrote results/rnn_params.json', flush=True)


if __name__ == '__main__':
    main()
