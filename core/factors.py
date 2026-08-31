"""Factor adapters for the self-models worlds.

`generator.Factor` is Mess3-only (built from alpha,x). `GenericFactor` exposes the same
interface (.T, .M, .E, .pi) for an arbitrary labelled-operator tensor T (Z,I,J), so non-Mess3
factors (e.g. asym3) slot straight into `CompositionMixture`. Q is fixed at 3 in the generator,
so factors here are 3-state / 3-emission.
"""
from __future__ import annotations
import sys, os
import numpy as np
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from generator import mess3_operators, stationary, CompositionMixture  # noqa: E402


class GenericFactor:
    """Factor with arbitrary labelled operators T[z,i,j] = P(emit z, next j | i)."""
    def __init__(self, T, alpha=None, x=None, name=""):
        self.T = np.asarray(T, float)
        self.M = self.T.sum(0)          # (i,j) marginal transition
        self.E = self.T.sum(2).T        # (i,z) = P(z|i)
        self.pi = stationary(self.M)
        self.alpha, self.x, self.name = alpha, x, name   # alpha/x kept for train.py compat
        assert np.allclose(self.T.sum((0, 2)), 1.0), "factor not row-stochastic"


def mess3_factor(alpha, x):
    return GenericFactor(mess3_operators(alpha, x), alpha=alpha, x=x, name=f"mess3({alpha},{x})")


def asym3_factor(p_det=0.97, p_unif=1/3, stay=0.85):
    """Asymmetric 3-state HMM: state 0 = near-deterministic emission + strong self-loop
    (LOW forward entropy -> the 'predictable' collapse target); states 1,2 = near-uniform
    emission, mix among themselves, leak to 0. fwd-entropy spread across states ~2.5."""
    E = np.array([[p_det, (1 - p_det) / 2, (1 - p_det) / 2],
                  [p_unif, p_unif, p_unif],
                  [p_unif, p_unif, p_unif]])
    M = np.array([[stay, (1 - stay) / 2, (1 - stay) / 2],
                  [0.15, 0.45, 0.40],
                  [0.15, 0.40, 0.45]])
    T = (M[:, None, :] * E[:, :, None]).transpose(1, 0, 2)   # T[z,i,j] = M[i,j] E[i,z]
    return GenericFactor(T, name="asym3")


def make_world(factors, eps=0.0):
    """Independent (eps=0) factored world over the full N-tuple. Returns CompositionMixture
    with a single full-support composition; observed token = sum_n z_n * Q^(N-1-n)."""
    N = len(factors)
    full_mask = tuple(range(N))
    return CompositionMixture(factors, [full_mask], obs_model="tuple_coupled", eps=eps)


def decode_subtokens(tokens, N, Q=3):
    """observed int token -> per-factor sub-tokens (factor 0 = most significant)."""
    sub = np.empty(tokens.shape + (N,), dtype=np.int64)
    rem = tokens.copy()
    for n in range(N - 1, -1, -1):
        sub[..., n] = rem % Q
        rem //= Q
    return sub   # (..., N)


def belief_filter(T, subtokens, pi):
    """Exact per-factor hidden-state belief P(S_t | z_{1:t}) for one factor.
    subtokens: (B,L) that factor's emissions. Returns (B,L,I)."""
    B, L = subtokens.shape
    I = T.shape[1]
    b = np.repeat(pi[None, :], B, axis=0).astype(float)
    out = np.empty((B, L, I))
    for t in range(L):
        z = subtokens[:, t]
        bn = np.einsum("bi,bij->bj", b, T[z])
        bn /= np.clip(bn.sum(1, keepdims=True), 1e-300, None)
        out[:, t] = bn
        b = bn
    return out


if __name__ == "__main__":
    # smoke test: unified {Mess3, asym3} world
    facs = [mess3_factor(0.6, 0.15), asym3_factor()]
    world = make_world(facs, eps=0.0)
    print("V =", world.V, "  N =", world.N, "  factor names:", [f.name for f in facs])
    rng = np.random.default_rng(0)
    toks, _ = world.sample(64, 32, rng)
    print("tokens", toks.shape, "range", toks.min(), toks.max())
    # exact oracle floor (should be finite, sane)
    out = world.forward(toks)
    print("oracle next-token entropy (tail mean):", out["ent"][:, 16:].mean())
    # per-factor ground-truth beliefs via independent filtering
    sub = decode_subtokens(toks, world.N)
    for n, f in enumerate(facs):
        bel = belief_filter(f.T, sub[..., n], f.pi)
        H = -(np.clip(bel, 1e-30, None) * np.log(np.clip(bel, 1e-30, None))).sum(-1)
        print(f"factor {n} ({f.name}): belief shape {bel.shape}, mean stationary H = {H[:,16:].mean():.3f}")
    print("OK")
