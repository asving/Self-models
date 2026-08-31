"""v6 push-mechanics exploration, round 2 (fixes from round 1).

Round-1 findings this responds to:
  (1) drift-follower (park wherever the world sits; no internal goal) BEAT
      committed play because values (1,.75,.5) were too close -- throughput
      dominated selection.  Fix tested here: near-worthless third state
      (values 1, 1-delta, floor).
  (2) privacy catastrophic (observer pins the goal in 1-3 rounds at rho=.2,
      alpha=.85).  Fix tested: sweep rho (action-noise floor) x alpha
      (emission fidelity), target t90 ~ rounds-per-harvest.
  (3) committed vs per-round recomputer premium exists only with fat ties /
      noisy evaluation.  Structural fix tested: NOISY VALUE SENSING (private
      scent channel) -- values are not given, only noisy per-round private
      observations.  Then per-round re-rounding of the running posterior
      flips early and pays cancellation costs; integrate-then-LOCK should win.
      This endogenizes the recompute noise: in the real net it is the net's
      own inference noise, not an imposed parameter.
"""
from __future__ import annotations
import json
import numpy as np

from v6_push_explore import (DVEC, DSTAR, make_Tbase, push_mats, simulate,
                             privacy)


def simulate_scent(policy, P, N=8000, T=64, seed=0, snoise=0.25, lock_gap=0.5):
    """Relative-push world; values NOT known -- each round the agent receives
    a private noisy reading y_t = v + snoise*N(0,1) (all 3 states).
    policy 'recomputer': target = argmax of running-mean posterior, every round.
    policy 'locker'    : same until the top-2 gap of the running mean exceeds
                         lock_gap*snoise/sqrt(t) (i.e. z-score lock_gap), then
                         LOCK; re-lock only when the locked target depletes.
    policy 'oracle'    : knows v exactly, committed (ceiling).
    """
    rng = np.random.default_rng(seed)
    sigma, alpha, c, k = P['sigma'], P['alpha'], P['c'], P['k']
    rho, delta, tau = P['rho'], P['delta'], P['tau']
    Tb = make_Tbase(sigma)
    L = np.full((3, 3), (1 - alpha) / 2)
    np.fill_diagonal(L, alpha)
    E, Pd = push_mats()
    base_vals = np.array([1.0, 1 - delta, P.get('floor', 1 - 2 * delta)])
    v = base_vals[np.argsort(rng.random((N, 3)), axis=1)]

    s = rng.integers(0, 3, N)
    b = np.full((N, 3), 1 / 3)
    dep_until = np.zeros((N, 3), int)
    run = np.zeros(N, int)
    eprev = np.full(N, -1)
    vsum = np.zeros((N, 3))
    locked = np.zeros(N, bool)
    g = rng.integers(0, 3, N)
    reward = np.zeros(N)
    harv = np.zeros(N, int)
    rows = np.arange(N)

    for t in range(T):
        # private scent reading
        vsum += v + snoise * rng.standard_normal((N, 3))
        vhat = vsum / (t + 1)
        active = t >= dep_until
        vh = np.where(active, vhat, -np.inf)
        # ---- target
        if policy == 'oracle':
            va = np.where(active, v, -np.inf)
            if t == 0:
                g = va.argmax(1)
            else:
                need = ~active[rows, g] & active.any(1)
                g = np.where(need, va.argmax(1), g)
        elif policy == 'recomputer':
            g = vh.argmax(1)
        else:  # locker
            srt = np.sort(vh, axis=1)
            gap = srt[:, 2] - srt[:, 1]
            se = snoise * np.sqrt(2.0 / (t + 1))
            newly = ~locked & (gap > lock_gap * se)
            locked |= newly
            dep_now = locked & ~active[rows, g]
            locked &= ~dep_now            # unlock on depletion of own target
            g = np.where(locked & ~newly, g, vh.argmax(1))
        # ---- action (relative push)
        shat = b.argmax(1)
        a_pol = DSTAR[shat, g]
        alt = rng.integers(0, 3, N)
        a = np.where(rng.random(N) < rho, alt, a_pol)
        # ---- world
        push = rng.random(N) < c
        s_push = (s + DVEC[a]) % 3
        stay = rng.random(N) < sigma
        off = 1 + (rng.random(N) < 0.5)
        s = np.where(push, s_push, np.where(stay, s, (s + off) % 3)).astype(int)
        off2 = 1 + (rng.random(N) < 0.5)
        x = np.where(rng.random(N) < alpha, s, (s + off2) % 3).astype(int)
        run = np.where(x == eprev, run + 1, 1)
        eprev = x
        hm = (run >= k) & (t >= dep_until[rows, x])
        if hm.any():
            idx = np.where(hm)[0]
            reward[idx] += v[idx, x[idx]]
            harv[idx] += 1
            dep_until[idx, x[idx]] = t + tau
            run[idx] = 0
        Tn = (1 - c) * Tb[None] + c * Pd[a]
        b = np.einsum('ni,nij->nj', b, Tn) * L[:, x].T
        b /= b.sum(1, keepdims=True)
    return dict(R=float(reward.mean()),
                sem=float(reward.std() / np.sqrt(N)),
                H=float(harv.mean()))


def main():
    N, T = 8000, 64
    results = {}
    P0 = dict(sigma=0.8, alpha=0.85, c=0.35, k=4, rho=0.2, delta=0.10,
              floor=0.05, tau=16, beta=0.2)

    # ---- stage A: ladders with a near-worthless third state ---------------
    print('=' * 100)
    print('STAGE A: ladders with values (1, 1-delta, floor=0.05), delta=0.10')
    for variant in ('absolute', 'relative'):
        print(f'--- {variant}')
        for pol, kw in (('uniform', {}), ('drift', {}), ('hedger', {}),
                        ('recomputer', dict(eta=0.15)),
                        ('committed', dict(est='filter')),
                        ('committed', dict(est='oracle'))):
            r = simulate(variant, pol, P0, N, T, 3, **kw)
            tag = pol + ('/' + kw['est'] if 'est' in kw else '')
            print(f"  {tag:18s} R={r['R']:6.3f}±{r['sem']:.3f}  "
                  f"H={r['H']:5.2f}  top={r['top']:.2f}")
            results[f'A_{variant}_{tag}'] = r

    # ---- stage B: privacy vs control sweep (relative) ----------------------
    print('=' * 100)
    print('STAGE B: privacy/control sweep, relative variant. t90 = median '
          'rounds for observer to hit 0.9 on the goal (pursuit only);')
    print('         r/harv = mean rounds per harvest for committed play '
          '(t90 >= r/harv means the goal completes before it is decoded)')
    print(f"{'rho':>5} {'alpha':>6} | {'R_com':>6} {'R_uni':>6} {'ratio':>6} "
          f"{'r/harv':>7} | {'t90':>5} {'never':>6} {'P@8':>5}")
    for rho in (0.2, 0.35, 0.5):
        for alpha in (0.6, 0.75, 0.85):
            P = dict(P0, rho=rho, alpha=alpha)
            rc = simulate('relative', 'committed', P, N // 2, T, 5)
            ru = simulate('relative', 'uniform', P, N // 2, T, 6)
            pr = privacy('relative', P, N=2000)
            rph = T / max(rc['H'], 1e-9)
            print(f"{rho:5.2f} {alpha:6.2f} | {rc['R']:6.3f} {ru['R']:6.3f} "
                  f"{rc['R'] / max(ru['R'], 1e-9):6.2f} {rph:7.1f} | "
                  f"{pr['median_t90']:5.0f} {pr['frac_never']:6.2f} "
                  f"{pr['post_at'].get(8, float('nan')):5.2f}")
            results[f'B_rho{rho}_a{alpha}'] = dict(
                R_com=rc['R'], R_uni=ru['R'], rph=rph, **pr)

    # ---- stage C: flip-cost sweep (k) for the maintained-goal premium ------
    print('=' * 100)
    print('STAGE C: committed minus recomputer premium vs k (run length '
          'needed to harvest), delta=.05, eta=.15, known values')
    for k in (3, 4, 5, 6):
        P = dict(P0, k=k, delta=0.05)
        rc = simulate('relative', 'committed', P, N, T, 8)['R']
        rr = simulate('relative', 'recomputer', P, N, T, 8, eta=0.15)['R']
        pct = 100 * (rc - rr) / max(rc, 1e-9)
        print(f"  k={k}:  commit {rc:6.3f}  recomp {rr:6.3f}  "
              f"premium {pct:5.1f}%")
        results[f'C_k{k}'] = dict(rc=rc, rr=rr, pct=pct)

    # ---- stage D: noisy value sensing (scent channel) ----------------------
    print('=' * 100)
    print('STAGE D: values sensed via private noisy channel (scent), '
          'relative variant, delta=0.10, floor=0.05.')
    print('         locker = integrate scent, LOCK when top-2 gap is a '
          'z>lock_gap read, hold until depletion.')
    print(f"{'snoise':>7} | {'recomputer':>10} {'locker(z=1)':>12} "
          f"{'locker(z=2)':>12} {'oracle':>7} | {'premium%':>9}")
    for snoise in (0.15, 0.3, 0.6):
        rr = simulate_scent('recomputer', P0, N, T, 9, snoise=snoise)
        l1 = simulate_scent('locker', P0, N, T, 9, snoise=snoise, lock_gap=1.)
        l2 = simulate_scent('locker', P0, N, T, 9, snoise=snoise, lock_gap=2.)
        orc = simulate_scent('oracle', P0, N, T, 9, snoise=snoise)
        best = max(l1['R'], l2['R'])
        pct = 100 * (best - rr['R']) / max(best, 1e-9)
        print(f"{snoise:7.2f} | {rr['R']:10.3f} {l1['R']:12.3f} "
              f"{l2['R']:12.3f} {orc['R']:7.3f} | {pct:8.1f}%")
        results[f'D_s{snoise}'] = dict(recomp=rr['R'], lock1=l1['R'],
                                       lock2=l2['R'], oracle=orc['R'])

    with open('v6_push_explore2.json', 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print('\nsaved v6_push_explore2.json')


if __name__ == '__main__':
    main()
