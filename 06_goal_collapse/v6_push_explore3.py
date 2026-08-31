"""v6 push-mechanics exploration, round 3: the TREASURE world.

Design responding to rounds 1-2:
  - ONE treasure state per episode (v=1), others pay floor=0.05. Kills the
    throughput-beats-selection failure (drift-follower can no longer win by
    parking anywhere).
  - Treasure RELOCATES (to a different state) whenever it is collected --
    kills camping AND the hardcodable harvest cycle; every collection forces
    a fresh, privately-informed re-choice (recurring commitment events).
  - Values sensed only via a PRIVATE noisy scent channel y_t = v + noise.
    Public/private stream split: harvest adjudicated on a PUBLIC emission
    stream x_pub (observer sees it); agent additionally gets a private
    world-evidence stream x_a. Observer sees x_pub + actions only.

Policy rungs (the plausible RL learning path, each should out-earn the last):
  uniform < drift(park anywhere) < fresh(this round's scent only, stateless)
          < integrator(EMA scent posterior, re-round each step)
          < locker(integrate then LOCK until treasure collected)  ~ oracle.
The fresh->integrator gap = the reward paid for MAINTAINED private internal
state; that gap is the game's pressure to build a persistent intention.
"""
from __future__ import annotations
import json
import numpy as np

from v6_push_explore import DVEC, DSTAR, make_Tbase, push_mats


def sim3(policy, P, N=8000, T=64, seed=0, eps=1.0):
    rng = np.random.default_rng(seed)
    sigma, c, k, rho = P['sigma'], P['c'], P['k'], P['rho']
    a_pub, a_agt = P['alpha_pub'], P['alpha_agent']
    floor, snoise, lam, zlock = P['floor'], P['snoise'], P['lam'], P['zlock']
    Tb = make_Tbase(sigma)
    Lp = np.full((3, 3), (1 - a_pub) / 2)
    np.fill_diagonal(Lp, a_pub)
    La = np.full((3, 3), (1 - a_agt) / 2)
    np.fill_diagonal(La, a_agt)
    E, Pd = push_mats()

    s = rng.integers(0, 3, N)
    tstar = rng.integers(0, 3, N)
    b = np.full((N, 3), 1 / 3)
    vhat = np.full((N, 3), floor + (1 - floor) / 3)
    locked = np.zeros(N, bool)
    g = rng.integers(0, 3, N)
    run = np.zeros(N, int)
    eprev = np.full(N, -1)
    reward = np.zeros(N)
    treas = np.zeros(N, int)
    rows = np.arange(N)
    se = snoise * np.sqrt(lam / (2 - lam))  # EMA standard error (stationary)

    for t in range(T):
        v = np.full((N, 3), floor)
        v[rows, tstar] = 1.0
        y = v + snoise * rng.standard_normal((N, 3))
        vhat = (1 - lam) * vhat + lam * y
        # ---- target
        if policy == 'oracle':
            g = tstar.copy()
        elif policy == 'fresh':
            g = y.argmax(1)
        elif policy == 'integrator':
            g = vhat.argmax(1)
        elif policy == 'locker':
            srt = np.sort(vhat, 1)
            gap = srt[:, 2] - srt[:, 1]
            newly = ~locked & (gap > zlock * se * np.sqrt(2))
            locked |= newly
            g = np.where(locked & ~newly, g, vhat.argmax(1))
        elif policy == 'drift':
            g = b.argmax(1)
        # ---- action (relative push from filtered state estimate)
        shat = b.argmax(1)
        a_pol = DSTAR[shat, g]
        alt1 = rng.integers(0, 3, N)
        alt2 = rng.integers(0, 3, N)
        if policy == 'uniform':
            a = alt1
        else:
            a = np.where(rng.random(N) < eps, a_pol, alt1)
            a = np.where(rng.random(N) < rho, alt2, a)
        # ---- world
        push = rng.random(N) < c
        s_push = (s + DVEC[a]) % 3
        stay = rng.random(N) < sigma
        off = 1 + (rng.random(N) < 0.5)
        s = np.where(push, s_push, np.where(stay, s, (s + off) % 3)).astype(int)
        off1 = 1 + (rng.random(N) < 0.5)
        x_pub = np.where(rng.random(N) < a_pub, s, (s + off1) % 3).astype(int)
        off2 = 1 + (rng.random(N) < 0.5)
        x_a = np.where(rng.random(N) < a_agt, s, (s + off2) % 3).astype(int)
        # ---- harvest on PUBLIC runs
        run = np.where(x_pub == eprev, run + 1, 1)
        eprev = x_pub
        hm = run >= k
        if hm.any():
            idx = np.where(hm)[0]
            got = v[idx, x_pub[idx]]
            reward[idx] += got
            hit = x_pub[idx] == tstar[idx]
            treas[idx] += hit
            # relocate collected treasure to a different state
            tid = idx[hit]
            tstar[tid] = (tstar[tid] + 1
                          + (rng.random(len(tid)) < 0.5).astype(int)) % 3
            locked[tid] = False
            run[idx] = 0
        # ---- agent belief: public + private evidence
        Tn = (1 - c) * Tb[None] + c * Pd[a]
        b = np.einsum('ni,nij->nj', b, Tn) * Lp[:, x_pub].T * La[:, x_a].T
        b /= b.sum(1, keepdims=True)
    return dict(R=float(reward.mean()),
                sem=float(reward.std() / np.sqrt(N)),
                H=float(treas.mean()),
                rpt=float(T / max(treas.mean(), 1e-9)))


def privacy3(P, N=3000, T=48, seed=1, policy='integrator'):
    """Observer: sees x_pub + actions, knows dynamics, posterior over CURRENT
    treasure location. Harvest disabled (fixed treasure) = worst-case funnel.
    Agent runs the scent policy with its private streams."""
    rng = np.random.default_rng(seed)
    sigma, c, rho = P['sigma'], P['c'], P['rho']
    a_pub, a_agt = P['alpha_pub'], P['alpha_agent']
    floor, snoise, lam = P['floor'], P['snoise'], P['lam']
    Tb = make_Tbase(sigma)
    Lp = np.full((3, 3), (1 - a_pub) / 2)
    np.fill_diagonal(Lp, a_pub)
    La = np.full((3, 3), (1 - a_agt) / 2)
    np.fill_diagonal(La, a_agt)
    E, Pd = push_mats()
    s = rng.integers(0, 3, N)
    tstar = rng.integers(0, 3, N)
    b_a = np.full((N, 3), 1 / 3)
    b_o = np.full((N, 3), 1 / 3)
    vhat = np.full((N, 3), floor + (1 - floor) / 3)
    post = np.full((N, 3), 1 / 3)
    t90 = np.full(N, -1)
    rows = np.arange(N)
    checks = {}
    for t in range(T):
        v = np.full((N, 3), floor)
        v[rows, tstar] = 1.0
        y = v + snoise * rng.standard_normal((N, 3))
        vhat = (1 - lam) * vhat + lam * y
        g = vhat.argmax(1) if policy == 'integrator' else tstar.copy()
        shat = b_a.argmax(1)
        a_pol = DSTAR[shat, g]
        alt = rng.integers(0, 3, N)
        a = np.where(rng.random(N) < rho, alt, a_pol)
        # observer action-likelihood per goal hypothesis
        for gh in range(3):
            match = (a[:, None] == DSTAR[:, gh][None, :])
            pa = (b_o * ((1 - rho) * match + rho / 3)).sum(1)
            post[:, gh] *= pa
        post /= post.sum(1, keepdims=True)
        hit = (post[rows, tstar] > 0.9) & (t90 < 0)
        t90[hit] = t
        push = rng.random(N) < c
        s_push = (s + DVEC[a]) % 3
        stay = rng.random(N) < sigma
        off = 1 + (rng.random(N) < 0.5)
        s = np.where(push, s_push, np.where(stay, s, (s + off) % 3)).astype(int)
        off1 = 1 + (rng.random(N) < 0.5)
        x_pub = np.where(rng.random(N) < a_pub, s, (s + off1) % 3).astype(int)
        off2 = 1 + (rng.random(N) < 0.5)
        x_a = np.where(rng.random(N) < a_agt, s, (s + off2) % 3).astype(int)
        Tn = (1 - c) * Tb[None] + c * Pd[a]
        b_a = np.einsum('ni,nij->nj', b_a, Tn) * Lp[:, x_pub].T * La[:, x_a].T
        b_a /= b_a.sum(1, keepdims=True)
        b_o = np.einsum('ni,nij->nj', b_o, Tn) * Lp[:, x_pub].T
        b_o /= b_o.sum(1, keepdims=True)
        if t + 1 in (8, 16, 32, 48):
            checks[t + 1] = float(post[rows, tstar].mean())
    solved = t90[t90 >= 0]
    return dict(median_t90=float(np.median(solved)) if len(solved) else np.inf,
                frac_never=float((t90 < 0).mean()), post_at=checks)


def main():
    N, T = 8000, 64
    results = {}
    P0 = dict(sigma=0.8, alpha_pub=0.7, alpha_agent=0.85, c=0.35, k=4,
              rho=0.25, floor=0.05, snoise=0.35, lam=0.25, zlock=2.0)

    print('=' * 100)
    print('STAGE E: treasure-world rung ladder (relative push). '
          'R = mean episode reward, H = treasures/episode, '
          'rpt = rounds per treasure')
    for pol in ('uniform', 'drift', 'fresh', 'integrator', 'locker',
                'oracle'):
        r = sim3(pol, P0, N, T, 3)
        results[f'E_{pol}'] = r
        print(f"  {pol:11s} R={r['R']:6.3f}±{r['sem']:.3f}  H={r['H']:5.2f}  "
              f"rpt={r['rpt']:5.1f}")

    print('=' * 100)
    print('STAGE F: eps-curve, mix(eps*integrator + (1-eps)*uniform) -- '
          'learnability from uniform play; also k sweep')
    EPS = (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
    for k in (3, 4, 5):
        P = dict(P0, k=k)
        curve = [sim3('integrator', P, 16000, T, 7, eps=e)['R'] for e in EPS]
        results[f'F_k{k}'] = dict(zip(map(str, EPS), curve))
        slope0 = (curve[1] - curve[0]) / 0.1
        print(f"  k={k}: " + ' '.join(f'{r:6.3f}' for r in curve)
              + f"   slope@0={slope0:6.3f}")

    print('=' * 100)
    print('STAGE G: stateless penalty & lock premium vs scent noise '
          '(the internal-state necessity curve)')
    print(f"{'snoise':>7} | {'fresh':>7} {'integr':>7} {'locker':>7} "
          f"{'oracle':>7} | {'state prem%':>11}")
    for snoise in (0.2, 0.35, 0.6, 1.0):
        P = dict(P0, snoise=snoise)
        rf = sim3('fresh', P, N, T, 9)['R']
        ri = sim3('integrator', P, N, T, 9)['R']
        rl = sim3('locker', P, N, T, 9)['R']
        ro = sim3('oracle', P, N, T, 9)['R']
        best = max(ri, rl)
        prem = 100 * (best - rf) / max(best, 1e-9)
        results[f'G_s{snoise}'] = dict(fresh=rf, integ=ri, lock=rl, oracle=ro)
        print(f"{snoise:7.2f} | {rf:7.3f} {ri:7.3f} {rl:7.3f} {ro:7.3f} | "
              f"{prem:10.1f}%")

    print('=' * 100)
    print('STAGE H: privacy funnel (observer = public stream + actions; '
          'worst case: fixed treasure, integrator agent)')
    print(f"{'rho':>5} {'a_pub':>6} {'c':>5} | {'t90':>5} {'never':>6} "
          f"{'P@8':>5} {'P@16':>5} | {'rpt':>5} (rounds/treasure at these "
          f"params)")
    for rho in (0.25, 0.4):
        for a_pub in (0.6, 0.7):
            for c in (0.35, 0.5):
                P = dict(P0, rho=rho, alpha_pub=a_pub, c=c)
                pr = privacy3(P)
                rr = sim3('integrator', P, N // 2, T, 11)
                results[f'H_r{rho}_a{a_pub}_c{c}'] = dict(**pr, rpt=rr['rpt'])
                print(f"{rho:5.2f} {a_pub:6.2f} {c:5.2f} | "
                      f"{pr['median_t90']:5.0f} {pr['frac_never']:6.2f} "
                      f"{pr['post_at'][8]:5.2f} {pr['post_at'][16]:5.2f} | "
                      f"{rr['rpt']:5.1f}")

    with open('v6_push_explore3.json', 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print('\nsaved v6_push_explore3.json')


if __name__ == '__main__':
    main()
