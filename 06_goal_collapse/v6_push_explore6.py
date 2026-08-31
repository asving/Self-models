"""v6 push-mechanics exploration, round 6: the S-STATE CYCLE swap world.

Round-5 diagnosis: with S=3 and two top-value states, a value-blind parker
(drift) hits a top 2/3 of the time by chance -- the world is too small for
target CHOICE to matter. This round generalizes to an S-cycle:

  World: S states on a ring; base dynamics sticky-local (stay sigma, else
  hop to a NEIGHBOR); relative actions hold/+1/-1 push the ring w.p. c;
  emissions alpha-faithful over S symbols; k-run parking on emissions.
  TWO top states pay 1.0, the other S-2 pay vlow=0.05. Value layout PUBLIC.
  Collecting a top swaps it to a random junk state (public relocation) and
  triggers a fresh binary choice between the two current tops.
  Choice-time camper: payout = v * (1 - lam * p_book), p_book formed at the
  previous collection (equilibrium camper: knows the agent's choice rule):
    fresh   coin between tops, carried internally      -> p = 1/2 on tops
    rulefix lower-indexed top (compiled public tiebreak)-> p = 1
    drift   park wherever the world sits (value-blind)  -> p = .5*b_book+.5/S
    uniform random actions                              -> p = 1/S
All inputs public (Fork B). Pocket test: fresh must top the ladder at S>=5
with a positive eps-slope.
"""
from __future__ import annotations
import json
import numpy as np


def ring_mats(S, sigma, c):
    Tb = np.zeros((S, S))
    for s in range(S):
        Tb[s, s] = sigma
        Tb[s, (s + 1) % S] += (1 - sigma) / 2
        Tb[s, (s - 1) % S] += (1 - sigma) / 2
    Pd = np.zeros((3, S, S))            # action 0=hold, 1=+1, 2=-1
    for s in range(S):
        Pd[0, s, s] = 1.0
        Pd[1, s, (s + 1) % S] = 1.0
        Pd[2, s, (s - 1) % S] = 1.0
    return Tb, Pd


def dstar(S):
    D = np.zeros((S, S), int)           # D[s, g] = action toward g
    for s in range(S):
        for g in range(S):
            diff = (g - s) % S
            D[s, g] = 0 if diff == 0 else (1 if diff <= S // 2 else 2)
    return D


def sim6(policy, P, N=8000, T=64, seed=0, eps=1.0):
    rng = np.random.default_rng(seed)
    S, sigma, c, k = P['S'], P['sigma'], P['c'], P['k']
    rho, a_pub, lam, vlow = P['rho'], P['alpha_pub'], P['lam'], P['vlow']
    Tb, Pd = ring_mats(S, sigma, c)
    D = dstar(S)
    Lp = np.full((S, S), (1 - a_pub) / (S - 1))
    np.fill_diagonal(Lp, a_pub)
    DVECS = np.array([0, 1, -1])

    s = rng.integers(0, S, N)
    b = np.full((N, S), 1 / S)
    # two tops at random distinct states
    top1 = rng.integers(0, S, N)
    top2 = (top1 + 1 + rng.integers(0, S - 1, N)) % S
    run = np.zeros(N, int)
    eprev = np.full(N, -1)
    reward = np.zeros(N)
    coll = np.zeros(N, int)
    topc = np.zeros(N, int)
    rows = np.arange(N)
    b_book = np.full((N, S), 1 / S)

    if policy == 'fresh':
        g = np.where(rng.random(N) < 0.5, top1, top2)
    elif policy == 'rulefix':
        g = np.minimum(top1, top2)
    else:
        g = rng.integers(0, S, N)

    for t in range(T):
        if policy == 'drift':
            g = b.argmax(1)
        elif policy == 'rulefix':
            g = np.minimum(top1, top2)
        shat = b.argmax(1)
        a_pol = D[shat, g]
        alt1 = rng.integers(0, 3, N)
        alt2 = rng.integers(0, 3, N)
        if policy == 'uniform':
            a = alt1
        else:
            a = np.where(rng.random(N) < eps, a_pol, alt1)
            a = np.where(rng.random(N) < rho, alt2, a)
        push = rng.random(N) < c
        s_push = (s + DVECS[a]) % S
        stay = rng.random(N) < sigma
        hop = 1 - 2 * (rng.random(N) < 0.5).astype(int)
        s = np.where(push, s_push,
                     np.where(stay, s, (s + hop) % S)).astype(int)
        x = s.copy()
        noisy = rng.random(N) >= a_pub
        if noisy.any():
            idx = np.where(noisy)[0]
            x[idx] = (s[idx] + 1 + rng.integers(0, S - 1, len(idx))) % S
        run = np.where(x == eprev, run + 1, 1)
        eprev = x
        hm = run >= k
        if hm.any():
            idx = np.where(hm)[0]
            xi = x[idx]
            is_top = (xi == top1[idx]) | (xi == top2[idx])
            if policy == 'fresh':
                phat = np.where(is_top, 0.5, 0.0)
            elif policy == 'rulefix':
                lo = np.minimum(top1[idx], top2[idx])
                phat = np.where(xi == lo, 1.0, 0.0)
            elif policy == 'drift':
                phat = 0.5 * b_book[idx, xi] + 0.5 / S
            else:
                phat = np.full(len(idx), 1 / S)
            v = np.where(is_top, 1.0, vlow)
            reward[idx] += v * (1 - lam * phat)
            coll[idx] += 1
            topc[idx] += is_top
            # swap collected top to a random junk state; re-choice; book
            sw = idx[is_top]
            if len(sw):
                for _ in range(1):
                    newpos = rng.integers(0, S, len(sw))
                    for tries in range(8):   # resample onto junk
                        bad = (newpos == top1[sw]) | (newpos == top2[sw]) \
                            | (newpos == x[sw])
                        if not bad.any():
                            break
                        repl = rng.integers(0, S, int(bad.sum()))
                        newpos[bad] = repl
                was1 = x[sw] == top1[sw]
                top1[sw] = np.where(was1, newpos, top1[sw])
                top2[sw] = np.where(~was1, newpos, top2[sw])
                if policy == 'fresh':
                    g[sw] = np.where(rng.random(len(sw)) < 0.5,
                                     top1[sw], top2[sw])
            b_book[idx] = b[idx]
            run[idx] = 0
        Tn = (1 - c) * Tb[None] + c * Pd[a]
        b = np.einsum('ni,nij->nj', b, Tn) * Lp[:, x].T
        b /= b.sum(1, keepdims=True)
    return dict(R=float(reward.mean()),
                sem=float(reward.std() / np.sqrt(N)),
                H=float(coll.mean()), Htop=float(topc.mean()))


def main():
    N, T = 8000, 64
    results = {}
    print('=' * 100)
    print('STAGE O: S-cycle swap-world ladders (choice-time camper, lam=1). '
          'R = mean episode reward, H = collections, Htop = top collections')
    for S in (3, 4, 5, 6):
        P = dict(S=S, sigma=0.8, alpha_pub=0.75, c=0.35, k=4, rho=0.3,
                 lam=1.0, vlow=0.05)
        print(f'--- S={S}')
        for pol in ('uniform', 'drift', 'rulefix', 'fresh'):
            r = sim6(pol, P, N, T, 3)
            results[f'O_S{S}_{pol}'] = r
            print(f"  {pol:8s} R={r['R']:6.3f}±{r['sem']:.3f}  "
                  f"H={r['H']:5.2f}  Htop={r['Htop']:5.2f}")

    print('=' * 100)
    print('STAGE P: eps-curve for fresh at S=5 (learnability from uniform)')
    EPS = (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
    P5 = dict(S=5, sigma=0.8, alpha_pub=0.75, c=0.35, k=4, rho=0.3, lam=1.0,
              vlow=0.05)
    curve = [sim6('fresh', P5, 16000, T, 7, eps=e)['R'] for e in EPS]
    results['P_S5'] = dict(zip(map(str, EPS), curve))
    slope0 = (curve[1] - curve[0]) / 0.1
    print('  ' + ' '.join(f'{r:6.3f}' for r in curve)
          + f"   slope@0={slope0:6.3f}")

    with open('v6_push_explore6.json', 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print('\nsaved v6_push_explore6.json')




def lam_sweep():
    N, T = 8000, 64
    print('STAGE Q: lam sweep at S=5 (find the pocket: fresh on top with '
          'rulefix still crushed)')
    out = {}
    for lam in (0.25, 0.5, 0.75, 1.0):
        P = dict(S=5, sigma=0.8, alpha_pub=0.75, c=0.35, k=4, rho=0.3,
                 lam=lam, vlow=0.05)
        row = {}
        for pol in ('uniform', 'drift', 'rulefix', 'fresh'):
            row[pol] = sim6(pol, P, N, T, 3)['R']
        out[f'lam{lam}'] = row
        print(f"  lam={lam:4.2f}: uniform {row['uniform']:6.3f}  "
              f"drift {row['drift']:6.3f}  rulefix {row['rulefix']:6.3f}  "
              f"fresh {row['fresh']:6.3f}   "
              f"fresh/drift={row['fresh'] / row['drift']:5.2f}  "
              f"fresh/rulefix={row['fresh'] / row['rulefix']:5.2f}")
    EPS = (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
    P = dict(S=5, sigma=0.8, alpha_pub=0.75, c=0.35, k=4, rho=0.3,
             lam=0.5, vlow=0.05)
    curve = [sim6('fresh', P, 16000, T, 7, eps=e)['R'] for e in EPS]
    slope0 = (curve[1] - curve[0]) / 0.1
    print('  eps-curve (lam=.5): ' + ' '.join(f'{r:6.3f}' for r in curve)
          + f"   slope@0={slope0:6.3f}")
    out['eps_lam0.5'] = dict(zip(map(str, EPS), curve))
    with open('v6_push_explore6b.json', 'w') as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'lam':
        lam_sweep()
    else:
        main()
