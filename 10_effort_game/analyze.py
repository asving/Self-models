import torch, torch.nn.functional as F, numpy as np, sys
from world import T3, EM, Oracle
from net import Net
import post
dev = 'mps'; T = 32; B = 4000; tag = sys.argv[1] if len(sys.argv) > 1 else 'a'
net = Net().to(dev); net.load_state_dict(torch.load(f'post_{tag}.pt')); net.eval()
r = post.rollout(net, B, np.random.default_rng(7), oracle=True); ar = np.arange(B); k = r['k']
W, POST, X, P = r['W'], r['POST'], r['X'], r['P']; top = W.argmax(2); otop = POST.argmax(2)
print('reward/step', round((X == 0).sum(2).mean(), 3), ' late (t>=16)', round((X[:, 16:] == 0).sum(2).mean(), 3))
print('mean largest effort share per step', np.round(W.max(2).mean(0)[[0, 1, 2, 4, 8, 16, 31]], 2))
print('first-step choice of factor (share of episodes x,y,z):', np.round(np.bincount(top[:, 0], minlength=3) / B, 2))
print('effort on the true factor, by step      ', np.round(W[ar, :, k].mean(0)[[0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 31]], 2))
print('oracle belief in the true factor, by step', np.round(POST[ar, :, k].mean(0)[[0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 31]], 2))
print('top-effort factor is the true one, by step', np.round((top == k[:, None]).mean(0)[[0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 31]], 2))
print('oracle top factor is the true one, by step', np.round((otop == k[:, None]).mean(0)[[0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 31]], 2))
sw = (top[:, 1:] != top[:, :-1]); print('switches of top factor per episode: mean', round(sw.sum(1).mean(), 2), ' in steps 1-8:', round(sw[:, :8].sum(1).mean(), 2), ' in steps 16-31:', round(sw[:, 15:].sum(1).mean(), 2))
# win-stay / lose-shift: after pushing factor f at step t, did token 0 appear on f at t+1? then does top change at t+1?
got0 = np.take_along_axis(X, top[..., None], 2)[..., 0] == 0            # X[:,t] is the token produced after the step-t outputs
for lo, hi, name in ((0, 8, 'steps 0-7'), (8, 31, 'steps 8-30')):
    g = got0[:, lo:hi].ravel(); s = sw[:, lo:hi].ravel()
    print(f'{name}: P(switch | pushed factor showed 0) = {s[g].mean():.3f}   P(switch | it did not) = {s[~g].mean():.3f}')
# effort vs oracle belief, binned
pb = POST.ravel(); wb = W.ravel()
print('mean effort on a factor, binned by the oracle belief that it is the true one:')
for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
    m = (pb >= lo) & (pb < hi); print(f'   belief {lo:.1f}-{hi:.1f}: effort {wb[m].mean():.2f}  (n={m.sum()})')
# does early effort go where token 0 is least expected?
F0 = r['F0']; lowest = F0.argmin(2)
print('top-effort factor = factor with lowest forecast of token 0: steps 1-3', round((top[:, 1:4] == lowest[:, 1:4]).mean(), 3), ' (chance 0.333)')
# head: P(token 0) on true vs other factors, late
act = np.eye(3, dtype=bool)[k][:, None, :].repeat(T, 1); p0 = P[..., 0]
print('head P(token 0), steps 16+: true factor', round(p0[:, 16:][act[:, 16:]].mean(), 3), ' other factors', round(p0[:, 16:][~act[:, 16:]].mean(), 3), ' untouched forecast', round(F0[:, 16:][~act[:, 16:]].mean(), 3))
# linear read-out of the oracle belief from the last-layer state
Xt = torch.tensor(X, device=dev); inp = torch.cat([torch.full((B, 1, 3), 3, device=dev), Xt[:, :-1]], 1)
with torch.no_grad(): h = net(inp, hidden=True)[2].cpu().numpy()
H = h[:, 2:].reshape(-1, h.shape[-1]); Y = POST[:, 2:].reshape(-1, 3); n = len(H) // 2
A = np.c_[H, np.ones(len(H))]; beta = np.linalg.solve(A[:n].T @ A[:n] + 1e-2 * np.eye(A.shape[1]), A[:n].T @ Y[:n]); pred = A[n:] @ beta
print('linear read-out of oracle belief from final-layer state, held-out R^2:', round(1 - ((pred - Y[n:]) ** 2).sum() / ((Y[n:] - Y[n:].mean(0)) ** 2).sum(), 3))
np.savez(f'roll_{tag}.npz', W=W, POST=POST, X=X, k=k, P=P, F0=F0)
