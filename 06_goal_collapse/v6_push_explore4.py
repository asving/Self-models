"""v6 push-mechanics exploration, round 4: NO PRIVATE INPUTS (Fork B).

User's cut: private evidence (scent / private streams) regressed the design to
dealt-secret privilege -- the goal became world-caused (an inference about a
dealt fact). Fork B: ALL inputs public; the agent's only edge is knowing its
own commitment (weights + activations). The scent's four jobs reassigned to
one mechanism, the CAMPER (mirror mechanics at goal level):

  World: symmetric values (every state pays 1 on a k-run park), collected
  state depletes tau rounds. Camper = Bayes trail-reader over target
  hypotheses (public stream + actions); payout at collection scaled by
  (1 - lam * p_hat(x)) where p_hat is the camper posterior snapshotted m
  rounds before the final run began; habit policies additionally face a
  precedent prior (camper learned the across-episode habit).

Policies:
  uniform  : random actions (max unpredictability, min parking competence)
  drift    : hold wherever the world is (target = public world state)
  hedger   : alternate two targets (physics-killed control)
  habit    : fixed target rotation, camper has 0.8 precedent prior on it
  trail    : looking-glass agent -- re-derives own target each round from the
             SAME camper machinery run on its own public trail
  fresh    : random target among non-depleted at each commitment, CARRIED
             internally (the self-caused committer)
The design pocket exists iff fresh tops the ladder with real margins.
"""
from __future__ import annotations
import json
import numpy as np

from v6_push_explore import DVEC, DSTAR, make_Tbase, push_mats


def sim4(policy, P, N=8000, T=64, seed=0, eps=1.0):
    rng = np.random.default_rng(seed)
    sigma, c, k, rho = P['sigma'], P['c'], P['k'], P['rho']
    a_pub, tau = P['alpha_pub'], P['tau']
    lam, m = P['lam'], P['m']
    Tb = make_Tbase(sigma)
    Lp = np.full((3, 3), (1 - a_pub) / 2)
    np.fill_diagonal(Lp, a_pub)
    E, Pd = push_mats()

    s = rng.integers(0, 3, N)
    b = np.full((N, 3), 1 / 3)          # shared world filter (public stream)
    post = np.full((N, 3), 1 / 3)       # camper posterior over target
    HIST = 9
    ph = np.tile(post[:, None, :], (1, HIST, 1))  # ring buffer of posteriors
    snap = np.full((N, 3), 1 / 3)       # camper book at current run start
    prior = np.full((N, 3), 1 / 3)
    dep_until = np.zeros((N, 3), int)
    run = np.zeros(N, int)
    eprev = np.full(N, -1)
    g = rng.integers(0, 3, N)
    reward = np.zeros(N)
    coll = np.zeros(N, int)
    mults = []
    rows = np.arange(N)

    if policy == 'habit':
        g = np.zeros(N, int)            # everyone starts at target 0
        prior = np.full((N, 3), 0.2 / 3)
        prior[:, 0] += 0.8              # camper learned the habit
        post = prior.copy()
        ph = np.tile(post[:, None, :], (1, HIST, 1))

    for t in range(T):
        active = t >= dep_until
        anyact = active.any(1)
        # ---- target selection
        if policy in ('fresh', 'habit'):
            need = ~active[rows, g] & anyact
            if need.any():
                if policy == 'fresh':
                    rnd = rng.random((N, 3))
                    pick = np.where(active, rnd, -1).argmax(1)
                else:                    # habit: deterministic rotation
                    pick = g.copy()
                    for _ in range(2):
                        pick = np.where(active[rows, pick], pick,
                                        (pick + 1) % 3)
                g = np.where(need, pick, g)
        elif policy == 'trail':
            g = post.argmax(1)          # own target re-read from public trail
        elif policy == 'drift':
            g = b.argmax(1)
        elif policy == 'hedger':
            g = np.argsort(-b, 1)[:, 0] * 0 + (t % 2)  # alternate 0/1... use targets 0,1
            g = np.full(N, t % 2)
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
        # ---- camper update (before world moves: likelihood of this action)
        for gh in range(3):
            match = (a[:, None] == DSTAR[:, gh][None, :])
            pa = (b * ((1 - rho) * match + rho / 3)).sum(1)
            post[:, gh] *= pa
        post /= post.sum(1, keepdims=True)
        ph = np.concatenate([ph[:, 1:], post[:, None, :]], 1)
        # ---- world
        push = rng.random(N) < c
        s_push = (s + DVEC[a]) % 3
        stay = rng.random(N) < sigma
        off = 1 + (rng.random(N) < 0.5)
        s = np.where(push, s_push, np.where(stay, s, (s + off) % 3)).astype(int)
        off1 = 1 + (rng.random(N) < 0.5)
        x = np.where(rng.random(N) < a_pub, s, (s + off1) % 3).astype(int)
        # ---- runs; snapshot camper book m rounds before each run start
        newrun = x != eprev
        run = np.where(newrun, 1, run + 1)
        eprev = x
        if newrun.any():
            idx = np.where(newrun)[0]
            snap[idx] = ph[idx, max(0, HIST - 1 - m)]
        hm = (run >= k) & (t >= dep_until[rows, x])
        if hm.any():
            idx = np.where(hm)[0]
            mult = 1 - lam * snap[idx, x[idx]]
            reward[idx] += mult
            mults.append(mult)
            coll[idx] += 1
            dep_until[idx, x[idx]] = t + tau
            run[idx] = 0
            post[idx] = prior[idx]      # camper: target likely changed
            ph[idx] = post[idx, None, :]
        # ---- shared world filter
        Tn = (1 - c) * Tb[None] + c * Pd[a]
        b = np.einsum('ni,nij->nj', b, Tn) * Lp[:, x].T
        b /= b.sum(1, keepdims=True)
    am = float(np.concatenate(mults).mean()) if mults else 0.0
    return dict(R=float(reward.mean()),
                sem=float(reward.std() / np.sqrt(N)),
                H=float(coll.mean()), mult=am)


def main():
    N, T = 8000, 64
    results = {}
    base = dict(sigma=0.8, alpha_pub=0.75, c=0.35, k=4, rho=0.3, tau=16,
                lam=1.0, m=0)

    print('=' * 100)
    print('STAGE I: camper-world ladders. R = mean episode reward '
          '(payout = 1 - lam*p_camp at run start), H = collections/episode, '
          'mult = mean payout multiplier earned')
    for lam in (0.6, 1.0):
        for m in (0, 4):
            P = dict(base, lam=lam, m=m)
            print(f'--- lam={lam} (camper tax), m={m} (book closes m rounds '
                  f'before final run)')
            for pol in ('uniform', 'drift', 'hedger', 'habit', 'trail',
                        'fresh'):
                r = sim4(pol, P, N, T, 3)
                results[f'I_l{lam}_m{m}_{pol}'] = r
                print(f"  {pol:8s} R={r['R']:6.3f}±{r['sem']:.3f}  "
                      f"H={r['H']:5.2f}  mult={r['mult']:.2f}")

    print('=' * 100)
    print('STAGE J: eps-curve for fresh-committed at the best-looking '
          'config -- learnability from uniform')
    EPS = (0.0, 0.1, 0.2, 0.4, 0.7, 1.0)
    for lam, m in ((0.6, 0), (1.0, 4)):
        P = dict(base, lam=lam, m=m)
        curve = [sim4('fresh', P, 16000, T, 7, eps=e)['R'] for e in EPS]
        results[f'J_l{lam}_m{m}'] = dict(zip(map(str, EPS), curve))
        slope0 = (curve[1] - curve[0]) / 0.1
        print(f"  lam={lam} m={m}: " + ' '.join(f'{r:6.3f}' for r in curve)
              + f"   slope@0={slope0:6.3f}")

    print('=' * 100)
    print('STAGE K: k sweep (parking difficulty vs camper tax) at lam=1, m=0')
    for k in (3, 4, 5):
        P = dict(base, k=k)
        ru = sim4('uniform', P, N, T, 9)['R']
        rt = sim4('trail', P, N, T, 9)['R']
        rf = sim4('fresh', P, N, T, 9)['R']
        results[f'K_k{k}'] = dict(uniform=ru, trail=rt, fresh=rf)
        print(f"  k={k}: uniform {ru:6.3f}  trail {rt:6.3f}  fresh {rf:6.3f}"
              f"   fresh/trail={rf / max(rt, 1e-9):5.2f} "
              f"fresh/uniform={rf / max(ru, 1e-9):5.2f}")

    with open('v6_push_explore4.json', 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print('\nsaved v6_push_explore4.json')


if __name__ == '__main__':
    main()
