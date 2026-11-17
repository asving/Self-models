"""Exact-oracle policies with p = delta_0 and one-hot effort, to price decision rules (C0 = .8, T = 32)."""
import sys, numpy as np
sys.path.insert(0, '.')
from world import Oracle, step
B, T, C0 = 40000, 32, 0.8; P = np.broadcast_to(np.array([1., 0, 0]), (B, 3, 3)).copy()
def run(rule, seed=0):
    rng = np.random.default_rng(seed); s = rng.integers(3, size=(B, 3)); k = rng.integers(3, size=B)
    c = np.zeros((B, 3)); c[np.arange(B), k] = C0; orc = Oracle(B, C0); R = 0.0
    for t in range(T):
        post = orc.post(); fA, fI = orc.forecast0()
        if rule == 'informed': ch = k
        elif rule == 'greedy': ch = (post * (1 - fA)).argmax(1)
        elif rule == 'proxy': ch = (post * (C0 + (1 - C0) * fA) + (1 - post) * fI).argmax(1)
        elif rule == 'MAP': ch = post.argmax(1)
        w = np.eye(3)[ch]; x, s = step(rng, s, c, w, P); orc.update(x, w, P); R += (x == 0).sum(1).mean()
    return R / T
for rule in ('informed', 'greedy', 'proxy', 'MAP'):
    v = [run(rule, sd) for sd in (0, 1)]; print(f'{rule:9s} reward/step {np.mean(v):.4f}  (seeds {v[0]:.4f}, {v[1]:.4f})', flush=True)
