"""THE WHISPER GAME certification (many-to-many goal->behavior, Asvin's cut).

On the two-phase world's tracking task, ALL rounds: action = MAP-tracking,
except with prob delta the net emits its committed symbol g instead.
Reward = track_score + beta * max_h count(a==h). One dial (beta) sets the
optimal whisper rate delta endogenously.

Certify the scaling asymmetry:
  leak/round ~ O(delta^2)  (KL between tilted and untilted action dists)
  reward-forcing of EARLY commitment ~ beta*delta per round (LINEAR)
=> forced early binding with arbitrarily slow identification.
And the killer: the net's OWN trail-read of g needs the same ~1/delta^2
rounds => the trail cannot carry the commitment; a genuine internal
(context-seeded) goal representation is forced.
"""
import numpy as np
import sys
sys.path.insert(0, '/data/users/asvin/self-models')
from twophase import gen_world, filter_map, S, T

def sim(delta, N=4000, rho=0.1, seed=0):
    rng = np.random.default_rng(seed)
    s, x = gen_world(N, rng)
    shat = filter_map(x)
    g = rng.integers(0, S, N)
    a = shat.copy()
    noise = rng.random((N, T)) < rho
    a = np.where(noise, rng.integers(0, S, (N, T)), a)
    whisper = rng.random((N, T)) < delta
    a = np.where(whisper, g[:, None], a)
    track = (a == s).mean(1) * T
    counts = np.stack([(a == h).sum(1) for h in range(S)], 1)
    # observer: Bayes over which symbol is tilted (knows delta, rho, filter)
    post = np.full((N, S), 1 / S)
    t90 = np.full(N, -1)
    for t in range(T):
        base = np.where(a[:, t] == shat[:, t], (1 - rho) + rho / S, rho / S)
        for h in range(S):
            lik = delta * (a[:, t] == h) + (1 - delta) * base
            post[:, h] *= lik
        post /= post.sum(1, keepdims=True)
        hit = (post[np.arange(N), g] > 0.9) & (t90 < 0)
        t90[hit] = t
    solved = t90[t90 >= 0]
    med = np.median(solved) if len(solved) else np.inf
    return dict(track=track.mean(), dcount=(counts[np.arange(N), g]
                - np.sort(counts, 1)[:, -2].clip(min=0)).mean(),
                t90=med, frac_never=(t90 < 0).mean(),
                post_end=post[np.arange(N), g].mean())

print(f"{'delta':>6} | {'track':>6} {'t90':>5} {'never%':>7} {'P(g) end':>9}"
      f" | {'late-start cost (beta*delta*T/2), beta=1':>20}")
for delta in (0.05, 0.10, 0.15, 0.25, 0.40):
    r = sim(delta)
    print(f"{delta:6.2f} | {r['track']:6.1f} {r['t90']:5.0f} "
          f"{100*r['frac_never']:6.1f}% {r['post_end']:9.2f} | "
          f"{delta*T/2:8.1f} reward units")
