"""
HMM selection for the self-models experiments.

Goal: pick (G)HMM factors that make the consequence analysis sharpest.

What we need from a factor, by design:
  - Belief-state geometry that is rich and *recoverable* (low state count so it is
    visualizable; a non-trivial fractal so "the geometry" is a real object).
  - For design B (self-resampling -> entropy collapse): the factor's stationary
    HIDDEN-STATE belief should carry substantial residual entropy H(q) -- that is
    exactly the quantity the model can collapse once the belief seeds the world.
    And states should differ in their k-step forward predictability, so "collapse
    onto the most predictable state" is a directional, visible choice.
  - For design A (action conditioning): rich, clear belief geometry for the
    consequential factor; a second distinguishable factor as the zero-consequence
    control.

Metrics per factor:
  Hbar      mean stationary hidden-state belief entropy (nats; max ln Q). High => more to collapse.
  sync      how fast belief entropy drops with context (entropy at pos 1 vs plateau).
  fwd_k     per-start-state k-step token-sequence entropy; we report spread across states.
  cover     spread of the belief cloud in the simplex (mean pairwise / eff-dim).

Pure CPU numpy. Uses ~/comp_icl/generator.py for Mess3 ops.
"""
from __future__ import annotations
import sys, os, json, itertools
import numpy as np

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from generator import mess3_operators, stationary  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


# ----------------------------- single-factor primitives ----------------------------- #
def factor_pieces(T):
    """T: (Z,I,J) labelled ops. Return marginal M (I,J), emission E (I,Z), stationary pi (I)."""
    M = T.sum(0)
    E = T.sum(2).T            # (I,Z) = P(z|i)
    pi = stationary(M)
    return M, E, pi


def sample_factor(T, pi, B, L, rng):
    """Sample B sequences of length L from a single factor. Returns tokens (B,L)."""
    Z, I, _ = T.shape
    flat = T.transpose(1, 0, 2).reshape(I, Z * I)   # (i, z*I + j)
    states = rng.choice(I, size=B, p=pi)
    toks = np.empty((B, L), dtype=np.int64)
    for t in range(L):
        cdf = np.cumsum(flat[states], axis=1)
        idx = (rng.random(B)[:, None] < cdf).argmax(1)
        toks[:, t] = idx // I
        states = idx % I
    return toks


def belief_filter(T, tokens, pi):
    """Exact hidden-state belief b_t = P(S_t | z_{1:t}) for each (b,t). Returns (B,L,I)."""
    B, L = tokens.shape
    I = T.shape[1]
    b = np.repeat(pi[None, :], B, axis=0).astype(float)   # prior = stationary
    out = np.empty((B, L, I))
    for t in range(L):
        z = tokens[:, t]
        # b' ∝ b @ T[z]   (per-row token-specific operator)
        bn = np.einsum("bi,bij->bj", b, T[z])
        bn /= np.clip(bn.sum(1, keepdims=True), 1e-300, None)
        out[:, t] = bn
        b = bn
    return out


def ent(p, axis=-1):
    p = np.clip(p, 1e-300, None)
    return -(p * np.log(p)).sum(axis)


def forward_entropy_per_state(T, k):
    """H(Z_{1:k} | S_0 = s) for each start state s. Exact enumeration (Z^k seqs)."""
    Z, I, _ = T.shape
    Hs = np.zeros(I)
    for s in range(I):
        # accumulate prob over all length-k token sequences
        probs = []
        for seq in itertools.product(range(Z), repeat=k):
            v = np.zeros(I); v[s] = 1.0
            for z in seq:
                v = v @ T[z]
            probs.append(v.sum())
        p = np.array(probs)
        Hs[s] = ent(p)
    return Hs   # nats, per state


def belief_geometry(T, pi, depth):
    """Enumerate all Z^depth token sequences; return belief points (n, I) and weights."""
    Z, I, _ = T.shape
    pts, wts = [], []
    for seq in itertools.product(range(Z), repeat=depth):
        b = pi.astype(float).copy(); w = 1.0
        for z in seq:
            bn = b @ T[z]
            m = bn.sum()
            w *= m
            b = bn / max(m, 1e-300)
        pts.append(b); wts.append(w)
    return np.array(pts), np.array(wts)


# ----------------------------- diagnostics ----------------------------- #
def characterize(T, name, k=5, B=4000, L=64, depth=9):
    M, E, pi = factor_pieces(T)
    toks = sample_factor(T, pi, B, L, rng)
    bel = belief_filter(T, toks, pi)            # (B,L,I)
    He = ent(bel, axis=-1)                       # (B,L) belief entropy
    Hbar = He[:, L // 2:].mean()                 # stationary mean belief entropy
    sync = (He[:, 0].mean(), He[:, 3].mean(), He[:, L // 2:].mean())
    Hfwd = forward_entropy_per_state(T, k)       # (I,)
    pts, wts = belief_geometry(T, pi, depth)
    wts = wts / wts.sum()
    mean_pt = (pts * wts[:, None]).sum(0)
    cov = (wts[:, None, None] * (pts - mean_pt)[:, :, None] * (pts - mean_pt)[:, None, :]).sum(0)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eff_dim = (evals.sum() ** 2) / (evals ** 2).sum() if evals.sum() > 0 else 0.0
    spread = float(np.sqrt(evals.sum()))         # total std over the simplex
    return dict(
        name=name, alpha=None, x=None,
        Hbar=float(Hbar), maxH=float(np.log(T.shape[1])),
        sync_pos1=float(sync[0]), sync_pos4=float(sync[1]), sync_plateau=float(sync[2]),
        fwd_k=k, fwd_per_state=[float(v) for v in Hfwd],
        fwd_spread=float(Hfwd.max() - Hfwd.min()), fwd_mean=float(Hfwd.mean()),
        belief_spread=spread, eff_dim=float(eff_dim),
        _pts=pts, _wts=wts,
    )


# ----------------------------- candidate HMMs ----------------------------- #
def asym3(p_det=0.97, p_unif=1/3, stay=0.85):
    """Asymmetric 3-state HMM: state 0 = 'predictable' (near-deterministic emission,
    strong self-loop), states 1,2 = 'random' (near-uniform emission). Z=3.
    Built so per-state forward entropy differs a lot -> a clear collapse target."""
    Z = I = 3
    E = np.array([[p_det, (1-p_det)/2, (1-p_det)/2],     # state 0: emits 0 almost surely
                  [p_unif, p_unif, p_unif],               # state 1: uniform
                  [p_unif, p_unif, p_unif]])              # state 2: uniform
    # transitions: state 0 self-loops strongly; 1,2 mix among themselves and leak to 0
    M = np.array([[stay, (1-stay)/2, (1-stay)/2],
                  [0.15, 0.45, 0.40],
                  [0.15, 0.40, 0.45]])
    T = np.zeros((Z, I, I))
    for i in range(I):
        for j in range(I):
            for z in range(Z):
                T[z, i, j] = M[i, j] * E[i, z]
    return T


def switch2(p_emit=0.8, stay=0.9):
    """Simplest factor: 2-state noisy switch. Belief is a scalar in [0,1]."""
    Z = I = 2
    E = np.array([[p_emit, 1-p_emit], [1-p_emit, p_emit]])
    M = np.array([[stay, 1-stay], [1-stay, stay]])
    T = np.zeros((Z, I, I))
    for i in range(I):
        for j in range(I):
            for z in range(Z):
                T[z, i, j] = M[i, j] * E[i, z]
    return T


def main():
    rows = []
    grid = [(a, x) for a in (0.4, 0.6, 0.85) for x in (0.05, 0.15, 0.3)]
    geoms = {}
    for a, x in grid:
        T = mess3_operators(a, x)
        r = characterize(T, f"mess3 a={a} x={x}")
        r["alpha"], r["x"] = a, x
        geoms[r["name"]] = (r.pop("_pts"), r.pop("_wts"))
        rows.append(r)
    for name, T in [("asym3", asym3()), ("switch2", switch2())]:
        r = characterize(T, name)
        geoms[name] = (r.pop("_pts"), r.pop("_wts"))
        rows.append(r)

    # ---- table ----
    hdr = f"{'name':<18}{'Hbar':>7}{'maxH':>7}{'H@1':>7}{'H@4':>7}{'plat':>7}{'fwd_mean':>9}{'fwd_spr':>8}{'b_spr':>7}{'effd':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<18}{r['Hbar']:>7.3f}{r['maxH']:>7.3f}{r['sync_pos1']:>7.3f}"
              f"{r['sync_pos4']:>7.3f}{r['sync_plateau']:>7.3f}{r['fwd_mean']:>9.3f}"
              f"{r['fwd_spread']:>8.3f}{r['belief_spread']:>7.3f}{r['eff_dim']:>6.2f}")
    with open(os.path.join(OUT, "hmm_metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # ---- geometry plots (2-simplex barycentric for Q=3) ----
    def bary(pts):
        v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        return pts @ v
    plot_names = ["mess3 a=0.6 x=0.15", "mess3 a=0.4 x=0.05", "mess3 a=0.85 x=0.05",
                  "mess3 a=0.4 x=0.3", "asym3"]
    fig, axes = plt.subplots(1, len(plot_names), figsize=(4*len(plot_names), 4))
    for ax, nm in zip(axes, plot_names):
        pts, wts = geoms[nm]
        xy = bary(pts)
        ax.scatter(xy[:, 0], xy[:, 1], c=pts, s=4)   # RGB = belief
        ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=0.5)
        ax.set_title(nm, fontsize=9); ax.axis("equal"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "belief_geometries.png"), dpi=110)
    print(f"\nsaved {OUT}/belief_geometries.png and hmm_metrics.json")


if __name__ == "__main__":
    main()
