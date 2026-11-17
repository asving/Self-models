"""How does the trained net choose which stream to push? Replays the exact oracle over a saved rollout
(roll_TAG.npz from analyze.py) and compares the net's choice (argmax effort) with candidate rules.
Run from the effort-game folder: python probe/choice_rules.py c"""
import sys, numpy as np
sys.path.insert(0, '.')
from world import Oracle, EM
tag = sys.argv[1] if len(sys.argv) > 1 else 'c'; C0 = 0.8
r = np.load(f'roll_{tag}.npz'); W, X, k, P = r['W'], r['X'], r['k'], r['P']; B, T, _ = W.shape
orc = Oracle(B, C0); POST, FA, FI = [], [], []
for t in range(T):
    POST.append(orc.post()); a, i = orc.forecast0(); FA.append(a); FI.append(i)
    orc.update(X[:, t], W[:, t], P[:, t])
POST, FA, FI = map(lambda z: np.stack(z, 1), (POST, FA, FI))
assert np.abs(POST - r['POST']).max() < 1e-6
top = W.argmax(2); ar = np.arange(B)

def H(p): p = np.clip(p, 1e-12, 1); return -(p * np.log(p)).sum(-1)
def eig(post, fa, fi):
    """expected info gain about k from pushing stream f with all effort and p = delta_0 (hit/miss on f only)"""
    out = np.zeros_like(post)
    for f in range(3):
        ph_k = np.where(np.arange(3) == f, C0 + (1 - C0) * fa[..., f:f + 1], fi[..., f:f + 1])   # P(hit | k=g)
        ph_k = np.broadcast_to(ph_k, post.shape)
        ph = (post * ph_k).sum(-1, keepdims=True)
        ph_post = post * ph_k / ph; pm_post = post * (1 - ph_k) / (1 - ph)
        out[..., f] = H(post) - (ph[..., 0] * H(ph_post) + (1 - ph[..., 0]) * H(pm_post))
    return out

rules = {
    'MAP (argmax belief)': POST.argmax(2),
    'reward-greedy  post*(1-fA)': (POST * (1 - FA)).argmax(2),
    'info-greedy (EIG)': eig(POST, FA, FI).argmax(2),
    'highest natural P(0)': FI.argmax(2),
    'lowest natural P(0)': FI.argmin(2),
}
print(f'run {tag}: agreement of the net\'s pushed stream with each rule')
for name, ch in rules.items():
    print(f'  {name:28s} all t>=1 {np.mean(ch[:, 1:] == top[:, 1:]):.3f}   t=1..4 {np.mean(ch[:, 1:5] == top[:, 1:5]):.3f}   t>=16 {np.mean(ch[:, 16:] == top[:, 16:]):.3f}')

# The clean test: after the forced first probe on z misses, x and y are exactly tied in belief.
# Which of them does the net pick at t=1, and does it depend on their natural P(0)?
z = 2; miss0 = X[:, 0, z] != 0; sel = miss0 & (top[:, 1] != z)
fx, fy = FI[sel, 1, 0], FI[sel, 1, 1]; pick = top[sel, 1]
print(f'\nafter a miss on z at t=0 ({miss0.sum()} episodes): net leaves z in {sel.sum() / miss0.sum():.3f}')
print(f'  belief gap |b_x - b_y| at t=1: median {np.median(np.abs(POST[sel,1,0]-POST[sel,1,1])):.2e}')
print(f'  picks the stream with LOWER natural P(0): {np.mean((pick == 0) == (fx < fy)):.3f}   (reward- and info-greedy both say lower)')
print(f'  share picking x: {np.mean(pick == 0):.3f}')
d = fx - fy
for lo, hi in ((-1, -.3), (-.3, -.1), (-.1, 0), (0, .1), (.1, .3), (.3, 1)):
    m = (d >= lo) & (d < hi)
    if m.sum(): print(f'    P0_x - P0_y in [{lo:+.1f},{hi:+.1f}): P(pick x) = {np.mean(pick[m] == 0):.2f}  (n={m.sum()})')
# what does it see? the tokens at t=0 on x and y
print('  P(pick x) by tokens (x_x, x_y) seen at t=0:')
for a in range(3):
    print('   ', ' '.join(f'{np.mean(pick[(X[sel,0,0]==a)&(X[sel,0,1]==b)]==0):.2f}' for b in range(3)))

# Switch target in general: when the net leaves stream f after a miss, which of the other two does it go to?
sw = (top[:, 1:] != top[:, :-1]); t_idx, = np.nonzero(sw.any(0))
rows, cols = np.nonzero(sw); cols = cols + 1; prev = top[rows, cols - 1]; new = top[rows, cols]
others = np.array([[1, 2], [0, 2], [0, 1]])[prev]
o1, o2 = others[:, 0], others[:, 1]
bo1, bo2 = POST[rows, cols, o1], POST[rows, cols, o2]
higher_b = np.where(bo1 >= bo2, o1, o2)
print(f'\nall switches ({len(rows)}): goes to the higher-belief alternative {np.mean(new == higher_b):.3f}')
close = np.abs(bo1 - bo2) < .05
lowP0 = np.where(FI[rows, cols, o1] < FI[rows, cols, o2], o1, o2)
print(f'  when the two alternatives are within .05 in belief (n={close.sum()}): goes to lower natural P(0) {np.mean(new[close] == lowP0[close]):.3f}')
# stay-after-miss: does it ever stay after a miss, and is that when the belief still favours the stream?
miss = np.take_along_axis(X[:, :-1], top[:, :-1, None], 2)[..., 0] != 0
stay = ~sw
bstay = np.take_along_axis(POST[:, 1:], top[:, :-1, None], 2)[..., 0]
mapstay = POST[:, 1:].argmax(2) == top[:, :-1]
print(f'\nafter a miss: stays {stay[miss].mean():.3f}; when the oracle MAP is still the same stream it stays {stay[miss & mapstay].mean():.3f} '
      f'(n={(miss & mapstay).sum()}), otherwise {stay[miss & ~mapstay].mean():.3f} (n={(miss & ~mapstay).sum()})')

# Proxy rule: maximize P(the pushed stream itself shows 0), ignoring that it would show 0 anyway at rate fI.
proxy = (POST * (C0 + (1 - C0) * FA) + (1 - POST) * FI).argmax(2)
print(f'\nproxy rule argmax P(pushed stream shows 0): all t>=1 {np.mean(proxy[:, 1:] == top[:, 1:]):.3f}   t=1..4 {np.mean(proxy[:, 1:5] == top[:, 1:5]):.3f}   t>=16 {np.mean(proxy[:, 16:] == top[:, 16:]):.3f}')
dis = rules['MAP (argmax belief)'] != proxy
print(f'  where MAP and proxy disagree (n={dis[:,1:].sum()}): net = MAP {np.mean(top[:,1:][dis[:,1:]] == rules["MAP (argmax belief)"][:,1:][dis[:,1:]]):.3f}, net = proxy {np.mean(top[:,1:][dis[:,1:]] == proxy[:,1:][dis[:,1:]]):.3f}')
g = rules['reward-greedy  post*(1-fA)']; dis = g != proxy
print(f'  where reward-greedy and proxy disagree (n={dis[:,1:].sum()}): net = greedy {np.mean(top[:,1:][dis[:,1:]] == g[:,1:][dis[:,1:]]):.3f}, net = proxy {np.mean(top[:,1:][dis[:,1:]] == proxy[:,1:][dis[:,1:]]):.3f}')
