"""Linear belief probes: decode per-factor hidden-state belief from residual stream.
Used (a) to localize the belief-carrying layer after pretraining, and (b) frozen, as the
consequential readout that reseeds the world in design B."""
from __future__ import annotations
import numpy as np
import torch
from factors import decode_subtokens, belief_filter


@torch.no_grad()
def get_hiddens(model, tokens, device, bs=2048):
    """Per-layer block-output residuals (pre final-LN). Returns list[ (B,L,d) ] over layers."""
    model.eval()
    chunks = None
    for i in range(0, len(tokens), bs):
        x = torch.from_numpy(tokens[i:i + bs]).to(device)
        _, hid = model(x, return_hidden=True)
        hid = [h.float().cpu().numpy() for h in hid]
        if chunks is None:
            chunks = [[] for _ in hid]
        for li, h in enumerate(hid):
            chunks[li].append(h)
    return [np.concatenate(c, 0) for c in chunks]


def belief_targets(factors, tokens):
    """Ground-truth per-factor hidden-state belief (independent filtering). list[ (B,L,Q) ]."""
    sub = decode_subtokens(tokens, len(factors))
    return [belief_filter(f.T, sub[..., n], f.pi) for n, f in enumerate(factors)]


def ridge_fit(X, Y, lam=1.0):
    """Closed-form ridge (bias unregularized). X (n,d), Y (n,k) -> W (d,k), b (k,), R2."""
    n, d = X.shape
    Xb = np.concatenate([X, np.ones((n, 1))], 1)
    A = Xb.T @ Xb + lam * np.eye(d + 1)
    A[-1, -1] -= lam
    Wb = np.linalg.solve(A, Xb.T @ Y)
    pred = Xb @ Wb
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return Wb[:-1], Wb[-1], float(1 - ss_res / ss_tot)


def readout(resid, W, b):
    """Apply a frozen linear belief probe and project to a distribution. resid (...,d)->(...,Q)."""
    q = resid @ W + b
    q = np.clip(q, 1e-6, None)
    return q / q.sum(-1, keepdims=True)


def localize(model, factors, tokens, device, n_layer):
    """Fit ridge resid@layer -> belief for each (layer, factor). Returns R2 table (n_layer,N)
    and, per factor, the best layer's (W,b,ell,R2)."""
    hid = get_hiddens(model, tokens, device)              # list (B,L,d)
    bel = belief_targets(factors, tokens)                  # list (B,L,Q)
    N = len(factors)
    R2 = np.zeros((n_layer, N))
    fits = [None] * N
    for li in range(n_layer):
        X = hid[li].reshape(-1, hid[li].shape[-1])
        for n in range(N):
            Y = bel[n].reshape(-1, bel[n].shape[-1])
            W, b, r2 = ridge_fit(X, Y)
            R2[li, n] = r2
            if fits[n] is None or r2 > fits[n][3]:
                fits[n] = (W, b, li, r2)
    return R2, fits
