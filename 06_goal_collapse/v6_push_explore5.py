"""v6 push-mechanics exploration, round 5: the SWAP WORLD with a
CHOICE-TIME camper (repairing round 4's negative).

Round-4 lesson: taxing goal-predictability during PURSUIT punishes having a
goal at all (drift/uniform win; commitment slope negative) -- the mirror-blur
equilibrium at goal level. Repair: the camper's book closes at the PREVIOUS
COLLECTION (public event = the re-choice moment). Within-episode pursuit is
untaxed; only the predictability of WHICH target you choose next is taxed.

World: 3 states; TWO tops pay 1.0, one junk state pays 0.05. Collecting a top
SWAPS it with the junk state (public relocation) and triggers a fresh binary
choice between the two current tops. All inputs public (Fork B). Payout at
collection x = v[x] * (1 - lam * p_book(x)), where p_book = an equilibrium
camper's prediction of the next collection state, formed at the previous
collection from public info + knowledge of the agent's choice RULE (the
across-episode precedent limit, as in the mirror game):
  fresh (coin flip between tops, carried internally): p = .5/.5 on tops
  rulefix (always the lower-indexed top -- a compiled public tiebreak): p = 1
  drift  (park where the world sits, value-blind): p = .5*b_obs + .5*uniform
  trail  (re-derives own target from own public trail, reset each choice):
          p = .5/.5 (camper cannot beat the coin at book time either)
  uniform: p = 1/3 flat
The design pocket exists iff fresh tops this ladder with healthy margins and
a positive eps-slope.
"""
from __future__ import annotations
import json
import numpy as np

from v6_push_explore import DVEC, DSTAR, make_Tbase, push_mats


def sim5(policy, P, N=8000, T=64, seed=0, eps=1.0):
    rng = np.random.default_rng(seed)
    sigma, c, k, rho = P['sigma'], P['c'], P['k'], P['rho']
    a_pub, lam, vlow = P['alpha_pub'], P['lam'], P['vlow']
    Tb = make_Tbase(sigma)
    Lp = np.full((3, 3), (1 - a_pub) / 2)
    np.fill_diagonal(Lp, a_pub)
    E, Pd = push_mats()

    s = rng.integers(0, 3, N)
    b = np.full((N, 3), 1 / 3)       # shared public world filter
    low = rng.integers(0, 3, N)      # junk state; other two are tops
    run = np.zeros(N, int)
    eprev = np.full(N, -1)
    reward = np.zeros(N)
    coll = np.zeros(N, int)
    topcoll = np.zeros(N, int)
    rows = np.arange(N)
    b_book = np.full((N, 3), 1 / 3)  # camper world belief at book time

    def tops_of(lowv):
        t1 = (lowv + 1) % 3
        t2 = (lowv + 2) % 3
        return t1, t2

    # trail policy: posterior over CURRENT target among tops, from own actions
    tpost = np.full((N, 3), 1 / 3)

    def choose(idx):
        """fresh choice among current tops for episodes idx"""
        t1, t2 = tops_of(low[idx])
        pick = np.where(rng.random(len(idx)) < 0.5, t1, t2)
        return pick

    if policy == 'fresh':
        g = choose(rows)
    elif policy == 'rulefix':
        t1, t2 = tops_of(low)
        g = np.minimum(t1, t2)
    else:
        g = rng.integers(0, 3, N)

    for t in range(T):
        t1, t2 = tops_of(low)
        # ---- target selection
        if policy == 'drift':
            g = b.argmax(1)
        elif policy == 'trail':
            m1 = tpost[rows, t1] >= tpost[rows, t2]
            g = np.where(m1, t1, t2)
        elif policy == 'rulefix':
            g = np.minimum(t1, t2)
        # fresh: g persists (set at choice events)
        # ---- action
        shat = b.argmax(1)
        a_pol = DSTAR[shat, g]
        alt1 = rng.integers(0, 3, N)
        alt2 = rng.integers(0, 3, N)
        if policy == 'uniform':
            a = alt1
        else:
            a = np.where(rng.random(N) < eps, a_pol, alt1)
            a = np.where(rng.random(N) < rho, alt2, a)
        # ---- trail self-read update (uses public action + world belief)
        if policy == 'trail':
            for gh in range(3):
                match = (a[:, None] == DSTAR[:, gh][None, :])
                pa = (b * ((1 - rho) * match + rho / 3)).sum(1)
                tpost[:, gh] *= pa
            tpost /= tpost.sum(1, keepdims=True)
        # ---- world
        push = rng.random(N) < c
        s_push = (s + DVEC[a]) % 3
        stay = rng.random(N) < sigma
        off = 1 + (rng.random(N) < 0.5)
        s = np.where(push, s_push, np.where(stay, s, (s + off) % 3)).astype(int)
        off1 = 1 + (rng.random(N) < 0.5)
        x = np.where(rng.random(N) < a_pub, s, (s + off1) % 3).astype(int)
        run = np.where(x == eprev, run + 1, 1)
        eprev = x
        hm = run >= k
        if hm.any():
            idx = np.where(hm)[0]
            xi = x[idx]
            is_top = xi != low[idx]
            # camper book prediction of this collection
            if policy in ('fresh', 'trail'):
                phat = np.where(is_top, 0.5, 0.0)
            elif policy == 'rulefix':
                lo = np.minimum((low[idx] + 1) % 3, (low[idx] + 2) % 3)
                phat = np.where(xi == lo, 1.0, 0.0)
            elif policy == 'drift':
                phat = 0.5 * b_book[idx, xi] + 0.5 / 3
            else:
                phat = np.full(len(idx), 1 / 3)
            v = np.where(is_top, 1.0, vlow)
            reward[idx] += v * (1 - lam * phat)
            coll[idx] += 1
            topcoll[idx] += is_top
            # swap collected top with junk; fresh re-choice; books close
            sw = idx[is_top]
            low[sw] = x[sw]
            b_book[idx] = b[idx]
            if policy == 'fresh' and len(sw):
                g[sw] = choose(sw)
            if policy == 'trail':
                tpost[idx] = 1 / 3
            run[idx] = 0
        # ---- shared world filter
        Tn = (1 - c) * Tb[None] + c * Pd[a]
        b = np.einsum('ni,nij->nj', b, Tn) * Lp[:, x].T
        b /= b.sum(1, keepdims=True)
    return dict(R=float(reward.mean()),
                sem=float(reward.std() / np.sqrt(N)),
                H=float(coll.mean()), Htop=float(topcoll.mean()))


def main():
    N, T = 8000, 64
    results = {}
    base = dict(sigma=0.8, alpha_pub=0.75, c=0.35, k=4, rho=0.3, lam=1.0,
                vlow=0.05)

    print('=' * 100)
    print('STAGE L: swap-world ladder, choice-time camper. R = mean episode '
          'reward, H = collections, Htop = top-value collections')
    for lam in (0.5, 1.0):
        P = dict(base, lam=lam)
        print(f'--- lam={lam}')
        for pol in ('uniform', 'drift', 'rulefix', 'trail', 'fresh'):
            r = sim5(pol, P, N, T, 3)
            results[f'L_l{lam}_{pol}'] = r
            print(f"  {pol:8s} R={r['R']:6.3f}±{r['sem']:.3f}  "
                  f"H={r['H']:5.2f}  Htop={r['Htop']:5.2f}")

    print('=' * 100)
    print('STAGE M: eps-curve for fresh (learnability from uniform), lam=1')
    EPS = (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
    curve = [sim5('fresh', base, 16000, T, 7, eps=e)['R'] for e in EPS]
    results['M'] = dict(zip(map(str, EPS), curve))
    slope0 = (curve[1] - curve[0]) / 0.1
    print('  ' + ' '.join(f'{r:6.3f}' for r in curve)
          + f"   slope@0={slope0:6.3f}")

    print('=' * 100)
    print('STAGE N: k sweep at lam=1 (fresh vs trail vs drift)')
    for k in (3, 4, 5):
        P = dict(base, k=k)
        rd = sim5('drift', P, N, T, 9)['R']
        rt = sim5('trail', P, N, T, 9)['R']
        rf = sim5('fresh', P, N, T, 9)['R']
        results[f'N_k{k}'] = dict(drift=rd, trail=rt, fresh=rf)
        print(f"  k={k}: drift {rd:6.3f}  trail {rt:6.3f}  fresh {rf:6.3f}"
              f"   fresh/drift={rf / max(rd, 1e-9):5.2f}"
              f"   fresh/trail={rf / max(rt, 1e-9):5.2f}")

    with open('v6_push_explore5.json', 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print('\nsaved v6_push_explore5.json')


if __name__ == '__main__':
    main()
