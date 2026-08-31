"""v6 push-mechanics exploration (CPU-only, scripted policies, no training).

World: 3-state sticky chain (Mess3-like), alpha-noisy emissions. Agent actions
bias transitions ("pushing"). HARVEST: when k consecutive emissions show the
same symbol x and state x is not depleted, collect v_x; x then depletes for tau
rounds. Values v are a per-episode random permutation of (1, 1-delta, 1-2delta).

Push variants:
  absolute : action a in {0,1,2} = target vote; w.p. c world jumps to a.
  relative : action d in {hold,+1,-1}; w.p. c world moves s -> s+d. Pushing
             toward a goal requires knowing s (filtering demand). "hold" pins.
  momentum : absolute votes, but control strength scales with recent action
             consistency (EMA) -- cancellation is physical. Desert suspect.

Scripted policies (all with action-noise floor rho, full support):
  uniform     : random actions (RL init proxy).
  drift       : push toward argmax of own world-belief (commitment stored in
                the world, no internal goal) [relative: always hold].
  hedger      : alternate pushes between the top-2 valued states.
  recomputer  : per-round goal = argmax(v + eta*noise) over non-depleted --
                the "re-round every step from a noisy value read" null.
  committed   : pick argmax-value non-depleted goal, HOLD it until it depletes
                (own harvest), then re-commit. est in {filter, last, oracle}.

Measurements:
  ladder      : mean episode reward per policy per variant.
  eps-curve   : reward of mix(eps*committed + (1-eps)*uniform) -- slope at 0 is
                the REINFORCE-visibility check (RPS desert = flat).
  margin grid : committed minus recomputer vs (delta, eta) -- how fat must ties
                be before a maintained goal strictly beats re-rounding.
  privacy     : Bayes observer (independent emission stream + public actions)
                posterior over the agent's goal; rounds to 0.9 confidence.
"""
from __future__ import annotations
import json
import numpy as np

DVEC = np.array([0, 1, 2])  # relative action -> shift (2 == -1 mod 3)


def make_Tbase(sigma):
    Tb = np.full((3, 3), (1 - sigma) / 2)
    np.fill_diagonal(Tb, sigma)
    return Tb


def push_mats():
    E = np.zeros((3, 3, 3))   # E[a][s,s'] = 1[s'==a]
    Pd = np.zeros((3, 3, 3))  # Pd[d][s,s'] = 1[s'==s+DVEC[d]]
    for a in range(3):
        E[a, :, a] = 1.0
        for s in range(3):
            Pd[a, s, (s + DVEC[a]) % 3] = 1.0
    return E, Pd


def dstar_map():
    d = np.zeros((3, 3), int)  # d[s,g]
    for s in range(3):
        for g in range(3):
            d[s, g] = 0 if s == g else (1 if (g - s) % 3 == 1 else 2)
    return d


DSTAR = dstar_map()


def simulate(variant, policy, P, N=8000, T=64, seed=0, eps=1.0, eta=0.15,
             est='filter'):
    rng = np.random.default_rng(seed)
    sigma, alpha, c, k = P['sigma'], P['alpha'], P['c'], P['k']
    rho, delta, tau, beta = P['rho'], P['delta'], P['tau'], P['beta']
    Tb = make_Tbase(sigma)
    L = np.full((3, 3), (1 - alpha) / 2)
    np.fill_diagonal(L, alpha)
    E, Pd = push_mats()
    base_vals = np.array([1.0, 1 - delta, P.get('floor', 1 - 2 * delta)])
    v = base_vals[np.argsort(rng.random((N, 3)), axis=1)]
    top2 = np.argsort(-v, axis=1)[:, :2]

    s = rng.integers(0, 3, N)
    b = np.full((N, 3), 1 / 3)
    dep_until = np.zeros((N, 3), int)
    run = np.zeros(N, int)
    eprev = np.full(N, -1)
    g = np.zeros(N, int)
    mom = np.full((N, 3), 1 / 3)
    reward = np.zeros(N)
    harv = np.zeros(N, int)
    tophit = 0
    rows = np.arange(N)

    for t in range(T):
        active = t >= dep_until
        va = np.where(active, v, -np.inf)
        anyact = active.any(1)
        # ---- target selection
        if policy == 'committed':
            if t == 0:
                g = va.argmax(1)
            else:
                need = ~active[rows, g] & anyact
                g = np.where(need, va.argmax(1), g)
        elif policy == 'recomputer':
            g = (va + eta * rng.standard_normal((N, 3))).argmax(1)
        elif policy == 'hedger':
            g = top2[rows, t % 2]
        elif policy == 'drift':
            bm = np.where(active, b, -np.inf)
            g = np.where(anyact, bm.argmax(1), b.argmax(1))
        # ---- action
        if variant == 'relative':
            if est == 'filter':
                shat = b.argmax(1)
            elif est == 'last':
                shat = np.where(eprev >= 0, eprev, b.argmax(1))
            else:
                shat = s.copy()
            a_pol = DSTAR[shat, g]
        else:
            a_pol = g.copy()
        alt1 = rng.integers(0, 3, N)
        alt2 = rng.integers(0, 3, N)
        um = rng.random(N)
        ur = rng.random(N)
        if policy == 'uniform':
            a = alt1
        else:
            a = np.where(um < eps, a_pol, alt1)
            a = np.where(ur < rho, alt2, a)
        # ---- control strength
        if variant == 'momentum':
            mom = (1 - beta) * mom + beta * np.eye(3)[a]
            ceff = np.clip((mom[rows, a] - 1 / 3) * 1.5, 0, 1) * c
        else:
            ceff = np.full(N, c)
        # ---- world transition
        push = rng.random(N) < ceff
        s_push = (s + DVEC[a]) % 3 if variant == 'relative' else a
        stay = rng.random(N) < sigma
        off = 1 + (rng.random(N) < 0.5)
        s = np.where(push, s_push, np.where(stay, s, (s + off) % 3)).astype(int)
        # ---- emission
        off2 = 1 + (rng.random(N) < 0.5)
        x = np.where(rng.random(N) < alpha, s, (s + off2) % 3).astype(int)
        # ---- run / harvest
        run = np.where(x == eprev, run + 1, 1)
        eprev = x
        hm = (run >= k) & (t >= dep_until[rows, x])
        if hm.any():
            idx = np.where(hm)[0]
            got = v[idx, x[idx]]
            reward[idx] += got
            harv[idx] += 1
            tophit += int((got >= v[idx].max(1) - 1e-9).sum())
            dep_until[idx, x[idx]] = t + tau
            run[idx] = 0
        # ---- agent belief update
        if variant == 'relative':
            Tn = (1 - c) * Tb[None] + c * Pd[a]
        elif variant == 'absolute':
            Tn = (1 - c) * Tb[None] + c * E[a]
        else:
            Tn = (1 - ceff[:, None, None]) * Tb[None] \
                + ceff[:, None, None] * E[a]
        b = np.einsum('ni,nij->nj', b, Tn) * L[:, x].T
        b /= b.sum(1, keepdims=True)
    return dict(R=float(reward.mean()),
                sem=float(reward.std() / np.sqrt(N)),
                H=float(harv.mean()),
                top=float(tophit / max(1, harv.sum())))


def privacy(variant, P, N=4000, T=48, seed=1):
    """Committed agent, harvest disabled (pure pursuit). Observer sees public
    actions + an INDEPENDENT emission stream; Bayes posterior over the goal.
    Returns median rounds to 0.9 posterior + posterior on truth at checkpoints.
    (Observer assumes the agent acts on the true state -- mild upper bound on
    the leak for the relative variant.)"""
    rng = np.random.default_rng(seed)
    sigma, alpha, c = P['sigma'], P['alpha'], P['c']
    rho, delta = P['rho'], P['delta']
    Tb = make_Tbase(sigma)
    L = np.full((3, 3), (1 - alpha) / 2)
    np.fill_diagonal(L, alpha)
    E, Pd = push_mats()
    base_vals = np.array([1.0, 1 - delta, P.get('floor', 1 - 2 * delta)])
    v = base_vals[np.argsort(rng.random((N, 3)), axis=1)]
    g = v.argmax(1)
    s = rng.integers(0, 3, N)
    b_a = np.full((N, 3), 1 / 3)   # agent's filter (own stream)
    b_o = np.full((N, 3), 1 / 3)   # observer's filter (own stream)
    post = np.full((N, 3), 1 / 3)
    t90 = np.full(N, -1)
    rows = np.arange(N)
    checks = {}
    for t in range(T):
        shat = b_a.argmax(1) if variant == 'relative' else None
        a_pol = DSTAR[shat, g] if variant == 'relative' else g.copy()
        alt = rng.integers(0, 3, N)
        a = np.where(rng.random(N) < rho, alt, a_pol)
        # observer's per-goal action likelihood
        for gh in range(3):
            if variant == 'relative':
                match = (a[:, None] == DSTAR[:, gh][None, :])
                pa = (b_o * ((1 - rho) * match + rho / 3)).sum(1)
            else:
                pa = (1 - rho) * (a == gh) + rho / 3
            post[:, gh] *= pa
        post /= post.sum(1, keepdims=True)
        hit = (post[rows, g] > 0.9) & (t90 < 0)
        t90[hit] = t
        # world + two emission streams
        push = rng.random(N) < c
        s_push = (s + DVEC[a]) % 3 if variant == 'relative' else a
        stay = rng.random(N) < sigma
        off = 1 + (rng.random(N) < 0.5)
        s = np.where(push, s_push, np.where(stay, s, (s + off) % 3)).astype(int)
        xs = []
        for _ in range(2):
            off2 = 1 + (rng.random(N) < 0.5)
            xs.append(np.where(rng.random(N) < alpha, s,
                               (s + off2) % 3).astype(int))
        x_a, x_o = xs
        Tn = (1 - c) * Tb[None] + c * (Pd[a] if variant == 'relative' else E[a])
        b_a = np.einsum('ni,nij->nj', b_a, Tn) * L[:, x_a].T
        b_a /= b_a.sum(1, keepdims=True)
        b_o = np.einsum('ni,nij->nj', b_o, Tn) * L[:, x_o].T
        b_o /= b_o.sum(1, keepdims=True)
        if t + 1 in (8, 24, 48):
            checks[t + 1] = float(post[rows, g].mean())
    solved = t90[t90 >= 0]
    return dict(median_t90=float(np.median(solved)) if len(solved) else np.inf,
                frac_never=float((t90 < 0).mean()), post_at=checks)


def ladder(variant, P, N, T, seed=0):
    out = {}
    out['uniform'] = simulate(variant, 'uniform', P, N, T, seed)
    out['drift'] = simulate(variant, 'drift', P, N, T, seed + 1)
    out['hedger'] = simulate(variant, 'hedger', P, N, T, seed + 2)
    out['recomp(.15)'] = simulate(variant, 'recomputer', P, N, T, seed + 3,
                                  eta=0.15)
    out['commit/filter'] = simulate(variant, 'committed', P, N, T, seed + 4,
                                    est='filter')
    if variant == 'relative':
        out['commit/last'] = simulate(variant, 'committed', P, N, T, seed + 5,
                                      est='last')
    out['commit/oracle'] = simulate(variant, 'committed', P, N, T, seed + 6,
                                    est='oracle')
    return out


def main():
    N, T = 8000, 64
    results = {}

    # ---- stage 1: parameter pre-sweep (uniform vs committed vs drift) -------
    print('=' * 100)
    print('STAGE 1: parameter pre-sweep (relative variant), N=%d T=%d' % (N, T))
    print(f"{'sigma':>6} {'c':>5} {'alpha':>6} {'k':>3} | "
          f"{'R_unif':>7} {'R_drift':>8} {'R_commit':>9} | "
          f"{'com/uni':>8} {'com/drf':>8}")
    sweep = []
    for sigma in (0.5, 0.65, 0.8):
        for c in (0.2, 0.35):
            for k in (3, 4):
                for alpha in (0.7, 0.85):
                    P = dict(sigma=sigma, alpha=alpha, c=c, k=k, rho=0.2,
                             delta=0.25, tau=16, beta=0.2)
                    ru = simulate('relative', 'uniform', P, N // 2, T, 0)['R']
                    rd = simulate('relative', 'drift', P, N // 2, T, 1)['R']
                    rc = simulate('relative', 'committed', P, N // 2, T, 2)['R']
                    sweep.append(dict(P=P, ru=ru, rd=rd, rc=rc))
                    print(f"{sigma:6.2f} {c:5.2f} {alpha:6.2f} {k:3d} | "
                          f"{ru:7.3f} {rd:8.3f} {rc:9.3f} | "
                          f"{rc / max(ru, 1e-9):8.2f} {rc / max(rd, 1e-9):8.2f}")
    results['sweep'] = [dict(sigma=s['P']['sigma'], c=s['P']['c'],
                             alpha=s['P']['alpha'], k=s['P']['k'],
                             ru=s['ru'], rd=s['rd'], rc=s['rc'])
                        for s in sweep]
    # pick: committed clearly above both, but uniform not starved (signal)
    def score(e):
        if e['ru'] < 0.05 * e['rc']:      # desert at init
            return -1
        return (e['rc'] / max(e['ru'], 1e-9)) * (e['rc'] / max(e['rd'], 1e-9))
    best = max(sweep, key=score)
    P = best['P']
    print(f"\nchosen params: {P}")
    results['chosen'] = P

    # ---- stage 2: full ladders per variant ----------------------------------
    print('=' * 100)
    print('STAGE 2: policy ladders per push variant  (R = mean episode reward,'
          ' H = harvests/episode, top = frac harvests at top-value state)')
    for variant in ('absolute', 'relative', 'momentum'):
        lad = ladder(variant, P, N, T)
        results[f'ladder_{variant}'] = lad
        print(f'--- {variant}')
        for name, r in lad.items():
            print(f"  {name:15s} R={r['R']:6.3f}±{r['sem']:.3f}  "
                  f"H={r['H']:5.2f}  top={r['top']:.2f}")

    # ---- stage 3: eps-curves (learnability from uniform) ---------------------
    print('=' * 100)
    print('STAGE 3: eps-curve, mix(eps*committed + (1-eps)*uniform); '
          'slope at 0 = REINFORCE visibility')
    EPS = (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
    for variant in ('absolute', 'relative', 'momentum'):
        curve = [simulate(variant, 'committed', P, 20000, T, 7, eps=e)['R']
                 for e in EPS]
        results[f'eps_{variant}'] = dict(zip(map(str, EPS), curve))
        slope0 = (curve[1] - curve[0]) / 0.1
        mid = (curve[3] - curve[2]) / 0.2
        print(f"  {variant:9s} " +
              ' '.join(f'{r:6.3f}' for r in curve) +
              f"   slope@0={slope0:6.3f}  slope@.3={mid:6.3f}")

    # ---- stage 4: committed vs recomputer margin vs (delta, eta) ------------
    print('=' * 100)
    print('STAGE 4: maintained-goal premium (relative variant): '
          'R_committed - R_recomputer  as pct of R_committed')
    hdr = 'delta/eta'
    print(f"{hdr:>10} " + ' '.join(f'{e:>8.2f}' for e in (0.05, 0.15, 0.3)))
    grid = {}
    for delta in (0.05, 0.15, 0.3):
        rowvals = []
        for eta in (0.05, 0.15, 0.3):
            Pg = dict(P, delta=delta)
            rc = simulate('relative', 'committed', Pg, N, T, 11)['R']
            rr = simulate('relative', 'recomputer', Pg, N, T, 11, eta=eta)['R']
            pct = 100 * (rc - rr) / max(rc, 1e-9)
            rowvals.append(pct)
            grid[f'd{delta}_e{eta}'] = dict(rc=rc, rr=rr, pct=pct)
        print(f"{delta:>10.2f} " + ' '.join(f'{p:7.1f}%' for p in rowvals))
    results['margin_grid'] = grid

    # ---- stage 5: privacy funnel ---------------------------------------------
    print('=' * 100)
    print('STAGE 5: observer funnel (Bayes posterior over goal; independent '
          'emission stream + public actions; pursuit only, no harvest)')
    for variant in ('absolute', 'relative'):
        pr = privacy(variant, P)
        results[f'privacy_{variant}'] = pr
        print(f"  {variant:9s} median rounds to 0.9 = {pr['median_t90']}, "
              f"never within 48 = {pr['frac_never']:.2f}, "
              f"P(truth) @8/24/48 = "
              + '/'.join(f"{pr['post_at'].get(tt, float('nan')):.2f}"
                         for tt in (8, 24, 48)))

    with open('v6_push_explore.json', 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print('\nsaved v6_push_explore.json')


if __name__ == '__main__':
    main()
