"""Exact changeling oracle: four-belief filter bank, (†) mixture policy,
identity log-odds, vectorized over episodes.

Beliefs per episode (design doc notation):
  etaA, etaB : shared evidence filter (the pretrained predictor's beliefs) —
               emission update THEN token-driven transition, both chains.
  drA,  drB  : dead-reckoned posteriors (transition only) — the TRUE posterior
               of a chain under the hypothesis that it is the self chain.
  logodds    : log [ P(record | iota=A) / P(record | iota=B) ].

Policy ((†) in the design doc), with lam = P(iota=A):
  out_u = lam * piA + (1-lam) * pbar_u      piA ~ pbar_u * score_A^kappa
  out_v = lam * pbar_v + (1-lam) * piB
Per-token identity evidence:
  dlog = log out_u[u] - log pbar_u[u] + log pbar_v[v] - log out_v[v].

Agents: 'informed' (lam pinned at truth), 'agnostic' (lam = 1/2),
'live' (lam = sigmoid(logodds)), 'zero' (kappa forced to 0, embodied).
run_base() is the UN-embodied base-law rollout (both channels genuine);
its success equals h[0].mean() exactly (validation V2).
"""
import numpy as np

LOGODDS_CLIP = 40.0
SCORE_FLOOR = 1e-12


def _sample_rows(P, rng):
    """One categorical draw per row of P (rows sum to 1)."""
    c = np.cumsum(P, axis=1)
    r = rng.random((P.shape[0], 1))
    return np.argmax(c > r, axis=1)


def _rows(A, idx):
    return A[np.arange(A.shape[0]), idx]


def run_episodes(world, agent, R, seed, collect=False):
    w = world
    n, T = w.n, w.T
    kappa = 0.0 if agent == 'zero' else w.kappa
    rng = np.random.default_rng(seed)

    sA = rng.integers(0, n, R)
    sB = rng.integers(0, n, R)
    iota = rng.random(R) < 0.5          # True: network embodies A
    etaA = np.full((R, n), 1 / n); etaB = etaA.copy()
    drA = etaA.copy(); drB = etaA.copy()
    logodds = np.zeros(R)

    occ = np.zeros(R)
    diag = {'max_abs_dlog': 0.0, 'clip_hits': 0, 'max_abs_logodds': 0.0}
    if collect:
        traj = {'signed_logodds': np.zeros((R, T)),
                'signed_dlog': np.zeros((R, T), dtype=np.float32),
                'record_ll': np.zeros(R),
                'tv_self': np.zeros((R, T)),
                'p_true_evid': np.zeros((R, T)),
                'p_true_dr': np.zeros((R, T)),
                'u': np.zeros((R, T), dtype=np.int8),
                'v': np.zeros((R, T), dtype=np.int8),
                'ball': np.zeros((R, T), dtype=np.uint8)}

    for t in range(T):
        pbar_u = etaA @ w.EA
        pbar_v = etaB @ w.EB
        if kappa > 0:
            scA = np.einsum('ra,rb,abu->ru', drA, etaB, w.M[t])
            scB = np.einsum('ra,rb,abv->rv', etaA, drB, w.N[t])
            scA = np.maximum(scA, SCORE_FLOOR * scA.max(axis=1, keepdims=True) + 1e-300)
            scB = np.maximum(scB, SCORE_FLOOR * scB.max(axis=1, keepdims=True) + 1e-300)
            piA = pbar_u * (scA / scA.max(axis=1, keepdims=True)) ** kappa
            piB = pbar_v * (scB / scB.max(axis=1, keepdims=True)) ** kappa
            piA /= piA.sum(axis=1, keepdims=True)
            piB /= piB.sum(axis=1, keepdims=True)
        else:
            piA, piB = pbar_u, pbar_v

        if agent == 'informed':
            lam = iota.astype(float)
        elif agent in ('agnostic', 'zero'):
            lam = np.full(R, 0.5)
        elif agent == 'live':
            lam = 1.0 / (1.0 + np.exp(-logodds))
        else:
            raise ValueError(agent)

        out_u = lam[:, None] * piA + (1 - lam)[:, None] * pbar_u
        out_v = lam[:, None] * pbar_v + (1 - lam)[:, None] * piB

        u = np.where(iota, _sample_rows(out_u, rng), _sample_rows(w.EA[sA], rng))
        v = np.where(iota, _sample_rows(w.EB[sB], rng), _sample_rows(out_v, rng))

        dlog = (np.log(_rows(out_u, u)) - np.log(_rows(pbar_u, u))
                + np.log(_rows(pbar_v, v)) - np.log(_rows(out_v, v)))
        diag['max_abs_dlog'] = max(diag['max_abs_dlog'], float(np.abs(dlog).max()))
        if agent == 'live':
            raw = logodds + dlog
            diag['clip_hits'] += int((np.abs(raw) > LOGODDS_CLIP).sum())
            diag['max_abs_logodds'] = max(diag['max_abs_logodds'],
                                          float(np.abs(raw).max()))
            logodds = np.clip(raw, -LOGODDS_CLIP, LOGODDS_CLIP)

        TAg = w.TA[u, v]                 # (R, n, n)
        TBg = w.TB[u, v]
        etaA = etaA * w.EA[:, u].T
        etaA = np.einsum('rs,rst->rt', etaA, TAg)
        etaA /= etaA.sum(axis=1, keepdims=True)
        etaB = etaB * w.EB[:, v].T
        etaB = np.einsum('rs,rst->rt', etaB, TBg)
        etaB /= etaB.sum(axis=1, keepdims=True)
        drA = np.einsum('rs,rst->rt', drA, TAg)
        drB = np.einsum('rs,rst->rt', drB, TBg)

        sA = _sample_rows(TAg[np.arange(R), sA], rng)
        sB = _sample_rows(TBg[np.arange(R), sB], rng)
        occ += w.success(sA, sB, 1)

        if collect:
            traj['u'][:, t] = u
            traj['v'][:, t] = v
            traj['ball'][:, t] = w.success(sA, sB, 1)
            # whole-record statistic under the FIXED shared-filter evaluator
            # (identity-blind); used by V1's iota=A vs iota=B two-sample test
            traj['record_ll'] += (np.log(_rows(pbar_u, u))
                                  + np.log(_rows(pbar_v, v)))
            traj['signed_dlog'][:, t] = np.where(iota, dlog, -dlog)
            traj['signed_logodds'][:, t] = np.where(iota, logodds, -logodds)
            tvA = 0.5 * np.abs(etaA - drA).sum(axis=1)
            tvB = 0.5 * np.abs(etaB - drB).sum(axis=1)
            traj['tv_self'][:, t] = np.where(iota, tvA, tvB)
            traj['p_true_evid'][:, t] = np.where(iota, _rows(etaA, sA), _rows(etaB, sB))
            traj['p_true_dr'][:, t] = np.where(iota, _rows(drA, sA), _rows(drB, sB))

    out = {'exact': w.success(sA, sB, 0), 'tol1': w.success(sA, sB, 1),
           'occ': occ / T, 'iota': iota, 'diag': diag}
    if collect:
        out['traj'] = traj
    return out


def run_base(world, R, seed, collect=False):
    """Un-embodied base-law rollout: both channels genuine readouts."""
    w = world
    rng = np.random.default_rng(seed)
    sA = rng.integers(0, w.n, R)
    sB = rng.integers(0, w.n, R)
    occ = np.zeros(R)
    if collect:
        U = np.zeros((R, w.T), dtype=np.int8)
        V = np.zeros((R, w.T), dtype=np.int8)
    for t in range(w.T):
        u = _sample_rows(w.EA[sA], rng)
        v = _sample_rows(w.EB[sB], rng)
        if collect:
            U[:, t] = u
            V[:, t] = v
        sA = _sample_rows(w.TA[u, v][np.arange(R), sA], rng)
        sB = _sample_rows(w.TB[u, v][np.arange(R), sB], rng)
        occ += w.success(sA, sB, 1)
    out = {'exact': w.success(sA, sB, 0), 'tol1': w.success(sA, sB, 1),
           'occ': occ / w.T}
    if collect:
        out['u'], out['v'] = U, V
    return out
