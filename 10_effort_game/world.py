"""The effort game: three independent Mess3 factors; per episode a hidden gain c_f per factor.
Each step the agent outputs, per factor, a token distribution p_f (3,) and an effort split w (3,), sum 1.
For factor f: with prob c_f*w_f the next token is drawn from p_f and the hidden state follows that token
(s' ~ T[x][s,.]/P(x|s)); otherwise the factor takes an ordinary Mess3 step. Reward = number of factors showing token 0.
"""
import numpy as np
def mess3(x=0.15, a=0.6):
    b = (1 - a) / 2; tr = np.full((3, 3), x); np.fill_diagonal(tr, 1 - 2 * x)
    em = np.full((3, 3), b); np.fill_diagonal(em, a)
    return np.einsum('ij,jz->zij', tr, em)          # T[z,i,j] = P(emit z, next j | i)
T3 = mess3(); EM = T3.sum(2).T                        # EM[i,z] = P(z | state i)

class Oracle:
    """Exact filters for one-hot gains (one active factor with gain c0). Tracks, per factor, the belief and
    log-likelihood under 'this factor is the active one' (A) and 'it is not' (I)."""
    def __init__(self, B, c0):
        self.c0 = c0; self.bA = np.full((B, 3, 3), 1 / 3); self.bI = np.full((B, 3, 3), 1 / 3)
        self.LA = np.zeros((B, 3)); self.LI = np.zeros((B, 3))
    def post(self):
        lo = self.LA - self.LI; p = np.exp(lo - lo.max(1, keepdims=True)); return p / p.sum(1, keepdims=True)
    def forecast0(self):   # P(next token = 0) for each factor if left alone, under each hypothesis' belief
        return (self.bA @ EM)[:, :, 0], (self.bI @ EM)[:, :, 0]
    def update(self, x, w, p):
        Tx = T3[x]                                              # (B,3,3,3): [b,f,i,j]
        nI = np.einsum('bfi,bfij->bfj', self.bI, Tx); zI = nI.sum(2); self.bI = nI / zI[..., None]; self.LI += np.log(zI)
        px = np.take_along_axis(p, x[..., None], 2)[..., 0]     # p_f(x_f)
        ex = np.take_along_axis(np.broadcast_to(EM.T[None, None], x.shape + (3, 3)), x[..., None, None], 2)[:, :, 0, :]  # P(x|i)
        rho = (self.c0 * w)[..., None, None]
        K = (1 - rho) * Tx + rho * px[..., None, None] * Tx / ex[..., :, None]
        nA = np.einsum('bfi,bfij->bfj', self.bA, K); zA = nA.sum(2); self.bA = nA / zA[..., None]; self.LA += np.log(zA)

def step(rng, s, c, w, p):
    B = s.shape[0]; x = np.zeros((B, 3), int); s2 = np.zeros((B, 3), int)
    push = rng.random((B, 3)) < c * w
    for f in range(3):
        joint = T3[:, s[:, f], :].transpose(1, 0, 2).reshape(B, 9)          # (B, z*3+j)
        k = (rng.random(B)[:, None] > joint.cumsum(1)).sum(1).clip(0, 8); xz, sj = k // 3, k % 3
        xp = (rng.random(B)[:, None] > p[:, f].cumsum(1)).sum(1).clip(0, 2)
        cond = T3[xp, s[:, f], :]; cond = cond / cond.sum(1, keepdims=True)
        sp = (rng.random(B)[:, None] > cond.cumsum(1)).sum(1).clip(0, 2)
        x[:, f] = np.where(push[:, f], xp, xz); s2[:, f] = np.where(push[:, f], sp, sj)
    return x, s2

if __name__ == '__main__':
    import sys
    B, P0 = 20000, np.array([.98, .01, .01])
    for c0 in (0.5, 0.8):
        for T in (32, 64):
            for pol in ('agnostic', 'informed', 'greedy', 'soft'):
                rng = np.random.default_rng(0); s = rng.integers(3, size=(B, 3)); k = rng.integers(3, size=B)
                c = np.zeros((B, 3)); c[np.arange(B), k] = c0; orc = Oracle(B, c0)
                R = 0.0; det = np.full(B, T); p = np.broadcast_to(P0, (B, 3, 3)).copy()
                for t in range(T):
                    post = orc.post(); hit = (post[np.arange(B), k] >= .9) & (det == T); det[hit] = t
                    fA, _ = orc.forecast0()
                    if pol == 'agnostic': w = np.full((B, 3), 1 / 3)
                    elif pol == 'informed': w = np.eye(3)[k]
                    elif pol == 'greedy': w = np.eye(3)[(post * (1 - fA)).argmax(1)]
                    else: w = post
                    x, s = step(rng, s, c, w, p); orc.update(x, w, p); R += (x == 0).sum(1).mean()
                print(f'c0={c0} T={T} {pol:8s} reward/step {R/T:.3f}  detect median {np.median(det):.0f}  never {np.mean(det==T):.3f}', flush=True)
