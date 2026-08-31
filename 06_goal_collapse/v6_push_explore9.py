"""QUIET-BAND whisper certification. Quota now counts WRONG-BUT-H events
(a_t == h != s_t) so tracking hits are excluded; loudness taxed via tracking
weight; B sized so quiet > abstain > loud. Checks: free-money rate (abstain
meeting quota), reward ladder, observer id-time at the quiet point."""
import numpy as np, sys
sys.path.insert(0, '/data/users/asvin/self-models')
from twophase import gen_world, filter_map, S, T
NQ, QL = 4, T // 4

def run(policy, m, N=6000, seed=0, rho=0.05, B=0.5, delta_loud=0.4):
    rng = np.random.default_rng(seed)
    s, x = gen_world(N, rng)
    shat = filter_map(x)
    g = rng.integers(0, S, N)
    a = shat.copy()
    noise = rng.random((N, T)) < rho
    a = np.where(noise, rng.integers(0, S, (N, T)), a)
    if policy == 'quiet':
        # m+1 whispers per quarter at rounds where MAP != g (wrong-but-g)
        for q in range(NQ):
            need = np.full(N, m + 1)
            for t in range(q * QL, (q + 1) * QL):
                do = (need > 0) & (shat[:, t] != g)
                a[do, t] = g[do]
                need -= do
    elif policy == 'loud':
        wh = rng.random((N, T)) < delta_loud
        a = np.where(wh, g[:, None], a)
    wrong = (a != s)
    qc = np.zeros((N, NQ, S))
    for q in range(NQ):
        for h in range(S):
            qc[:, q, h] = ((a[:, q*QL:(q+1)*QL] == h)
                           & wrong[:, q*QL:(q+1)*QL]).sum(1)
    quota = qc.min(1).max(1) >= m
    track = (a == s).sum(1)
    R = track / 12 + B * quota
    # observer id-time (mixture-delta Bayes as in explore8)
    post = np.full((N, S), 1 / S)
    t90 = np.full(N, -1)
    dmix = (0.05, 0.1, 0.15)
    lw = np.zeros((N, len(dmix), S))
    for t in range(T):
        base = np.where(a[:, t] == shat[:, t], (1-rho) + rho/S, rho/S)
        for di, dl in enumerate(dmix):
            for h in range(S):
                lik = dl * (a[:, t] == h) + (1 - dl) * base
                lw[:, di, h] += np.log(lik + 1e-30)
        w = np.exp(lw - lw.max((1, 2), keepdims=True))
        post = w.sum(1) / w.sum((1, 2), keepdims=True)[:, 0]
        hit = (post[np.arange(N), g] > 0.9) & (t90 < 0)
        t90[hit] = t
    solved = t90[t90 >= 0]
    return dict(R=R.mean(), quota=quota.mean(), track=track.mean(),
                t90=np.median(solved) if len(solved) else np.inf,
                never=(t90 < 0).mean())

for m in (1, 2):
    print(f'--- quota m={m} wrong-but-h per quarter (B=0.5)')
    for pol in ('abstain', 'quiet', 'loud'):
        r = run(pol, m)
        print(f"  {pol:8s} R={r['R']:.3f} quota={r['quota']:.3f} "
              f"track={r['track']:.1f}/48  obs t90={r['t90']:.0f} "
              f"never={r['never']:.2f}")
