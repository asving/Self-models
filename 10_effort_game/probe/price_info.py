"""Is information-seeking worth anything in this game? Price an EIG-driven rule (pure, and for the first n steps then MAP)."""
import sys, numpy as np
sys.path.insert(0, '.'); sys.argv = sys.argv[:1]
from world import Oracle, step
B, T, C0 = 40000, 32, 0.8; P = np.broadcast_to(np.array([1., 0, 0]), (B, 3, 3)).copy()
def H(p): p = np.clip(p, 1e-12, 1); return -(p * np.log(p)).sum(-1)
def eig(post, fa, fi):
    out = np.zeros_like(post)
    for f in range(3):
        ph_k = np.where(np.arange(3) == f, C0 + (1 - C0) * fa[:, f:f+1], fi[:, f:f+1]); ph = (post * ph_k).sum(-1, keepdims=True)
        out[:, f] = H(post) - (ph[:, 0] * H(post * ph_k / ph) + (1 - ph[:, 0]) * H(post * (1 - ph_k) / (1 - ph)))
    return out
def run(n_explore, seed):
    rng = np.random.default_rng(seed); s = rng.integers(3, size=(B, 3)); k = rng.integers(3, size=B)
    c = np.zeros((B, 3)); c[np.arange(B), k] = C0; orc = Oracle(B, C0); R = 0.0
    for t in range(T):
        post = orc.post(); fA, fI = orc.forecast0()
        ch = eig(post, fA, fI).argmax(1) if t < n_explore else post.argmax(1)
        w = np.eye(3)[ch]; x, s = step(rng, s, c, w, P); orc.update(x, w, P); R += (x == 0).sum(1).mean()
    return R / T
for n in (0, 4, 8, 32):
    v = [run(n, sd) for sd in (0, 1)]; print(f'EIG for first {n:2d} steps, then MAP: reward/step {np.mean(v):.4f}', flush=True)
