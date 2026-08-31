"""V1-V4 from the design doc. Run with cwd = 08_changeling. All must pass.

V1  Lemma: at kappa=0, (a) per-token identity evidence is exactly zero
    (wiring self-consistency), (b) the RECORD's law under embodiment equals
    the pure base law — token unigrams (at t=0, 8, T-1) and within-channel
    bigrams match within 4.5 sigma, per channel — while (printed, not
    asserted) TRUE-state success may differ: the delusion's flip side.
V2  Un-embodied base success == h[0].mean() within 4 sigma (closed form).
V3  Filters vs independent path enumeration (T=6): evidence filter matches
    the emission-weighted path posterior; dead-reckoned filter matches the
    transition-only posterior (both 1e-9).
V4  Live agent at kappa=8: all trajectories finite, log-odds within clip.
"""
import numpy as np
from itertools import product
from worlds import World
from oracle import run_episodes, run_base

RESULTS = {}


def zmax_hist(x, y, nbins):
    """Max |z| across bins comparing two categorical samples."""
    cx = np.bincount(x, minlength=nbins).astype(float)
    cy = np.bincount(y, minlength=nbins).astype(float)
    px, py = cx / cx.sum(), cy / cy.sum()
    p = (cx + cy) / (cx.sum() + cy.sum())
    se = np.sqrt(p * (1 - p) * (1 / cx.sum() + 1 / cy.sum())) + 1e-300
    return float(np.abs(px - py).max() / se[np.abs(px - py).argmax()]), \
        float(np.max(np.abs(px - py) / se))


def v1(R=20000):
    w = World(q0=0.80, c_other=0.55, c_self=0.25, d_goal=2, kappa=0.0)
    emb = run_episodes(w, 'live', R, seed=11, collect=True)
    base = run_base(w, R, seed=12, collect=True)
    assert emb['diag']['max_abs_dlog'] < 1e-10, emb['diag']
    assert np.abs(emb['traj']['signed_logodds']).max() < 1e-10
    worst = 0.0
    for ch in ('u', 'v'):
        for t in (0, 8, w.T - 1):
            _, z = zmax_hist(emb['traj'][ch][:, t], base[ch][:, t], w.n)
            worst = max(worst, z)
        big_e = (emb['traj'][ch][:, :-1].astype(int) * w.n
                 + emb['traj'][ch][:, 1:]).ravel()
        big_b = (base[ch][:, :-1].astype(int) * w.n + base[ch][:, 1:]).ravel()
        _, z = zmax_hist(big_e, big_b, w.n * w.n)
        worst = max(worst, z)
    assert worst < 4.5, f"record law mismatch, max |z| = {worst:.2f}"
    # whole-record test (Codex review item 12): per-episode record
    # log-likelihood under the fixed shared-filter evaluator must be equal in
    # distribution across the two identities (both should equal the base law)
    ll = emb['traj']['record_ll']; io = emb['iota']
    a, b = ll[io], ll[~io]
    z_mean = abs(a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
    qs = np.linspace(5, 95, 19)
    qgap = np.abs(np.percentile(a, qs) - np.percentile(b, qs)).max()
    scale = ll.std() * np.sqrt(2 / min(len(a), len(b)))
    assert z_mean < 4.5 and qgap < 6 * scale, (z_mean, qgap, scale)
    se, sb = emb['exact'].mean(), base['exact'].mean()
    RESULTS['V1'] = dict(max_z_record=round(worst, 2),
                         z_record_ll=round(float(z_mean), 2),
                         succ_embodied=round(float(se), 4),
                         succ_base=round(float(sb), 4))
    print(f"V1 PASS  max|z| record stats {worst:.2f}, record-ll z {z_mean:.2f}; "
          f"true-state success embodied {se:.4f} vs base {sb:.4f} (may differ)")


def v2(R=40000):
    # h is built on the tolerance-1 ball, so the closed form is tol-1 success
    w = World(q0=0.55, c_other=0.40, c_self=0.40, d_goal=2, kappa=0.0)
    p_theory = float(w.h[0].mean())
    p_sim = float(run_base(w, R, seed=21)['tol1'].mean())
    sig = np.sqrt(p_theory * (1 - p_theory) / R)
    assert abs(p_sim - p_theory) < 4 * sig, (p_sim, p_theory, sig)
    RESULTS['V2'] = dict(p_theory=round(p_theory, 5), p_sim=round(p_sim, 5))
    print(f"V2 PASS  base success sim {p_sim:.5f} vs closed form {p_theory:.5f}")


def v3(n_ep=40, T=6):
    w = World(q0=0.80, c_other=0.55, c_self=0.25, d_goal=2, kappa=0.0)
    rng = np.random.default_rng(31)
    n = w.n
    worst_e = worst_d = 0.0
    for _ in range(n_ep):
        sA, sB = rng.integers(0, n), rng.integers(0, n)
        us, vs = [], []
        for t in range(T):
            u = rng.choice(n, p=w.EA[sA]); v = rng.choice(n, p=w.EB[sB])
            us.append(u); vs.append(v)
            sA = rng.choice(n, p=w.TA[u, v, sA])
            sB = rng.choice(n, p=w.TB[u, v, sB])
        # filter replay (same equations as oracle.py) for chain A
        eta = np.full(n, 1 / n); dr = np.full(n, 1 / n)
        for u, v in zip(us, vs):
            eta = (eta * w.EA[:, u]) @ w.TA[u, v]
            eta /= eta.sum()
            dr = dr @ w.TA[u, v]
        # independent enumeration over all state paths of chain A
        post_e = np.zeros(n); post_d = np.zeros(n)
        for path in product(range(n), repeat=T + 1):
            pe = pd = 1.0 / n
            for t in range(T):
                tr = w.TA[us[t], vs[t], path[t], path[t + 1]]
                pe *= tr * w.EA[path[t], us[t]]
                pd *= tr
                if pe == 0.0 and pd == 0.0:
                    break
            post_e[path[-1]] += pe
            post_d[path[-1]] += pd
        post_e /= post_e.sum(); post_d /= post_d.sum()
        worst_e = max(worst_e, np.abs(post_e - eta).max())
        worst_d = max(worst_d, np.abs(post_d - dr).max())
    assert worst_e < 1e-9 and worst_d < 1e-9, (worst_e, worst_d)
    RESULTS['V3'] = dict(max_err_evidence=worst_e, max_err_deadreckon=worst_d)
    print(f"V3 PASS  enumeration: evidence {worst_e:.2e}, dead-reckon {worst_d:.2e}")


def v4(R=2000):
    w = World(q0=0.80, c_other=0.55, c_self=0.25, d_goal=3, kappa=8.0)
    out = run_episodes(w, 'live', R, seed=41, collect=True)
    lo = out['traj']['signed_logodds']
    assert np.isfinite(lo).all() and np.abs(lo).max() <= 40.0 + 1e-9
    assert np.isfinite(out['traj']['tv_self']).all()
    RESULTS['V4'] = dict(max_abs_logodds=round(float(np.abs(lo).max()), 2),
                         final_median_signed=round(float(np.median(lo[:, -1])), 3))
    print(f"V4 PASS  live kappa=8 finite; median final signed log-odds "
          f"{np.median(lo[:, -1]):.2f}")


if __name__ == '__main__':
    import json, time
    t0 = time.time()
    v1(); v2(); v3(); v4()
    RESULTS['elapsed_s'] = round(time.time() - t0, 1)
    with open('results/validation_v0.json', 'w') as f:
        json.dump(RESULTS, f, indent=1, default=float)
    print(f"ALL VALIDATIONS PASS ({RESULTS['elapsed_s']}s) -> results/validation_v0.json")
