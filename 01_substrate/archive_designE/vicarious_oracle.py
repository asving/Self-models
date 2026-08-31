"""Design E (vicarious) -- STEP 0, CPU-only: exact belief filter + theoretical attractor fan.

Controlled Mess3: after the usual Mess3(alpha,x) labelled transition, the actor's action
a in {0,1,2} cyclically shifts the hidden state with gain g:
    T_a[z] = T[z] @ ((1-g) I + g P_a),   P_a = cyclic shift by a.
Actions are OBSERVED (interleaved token stream), so the env-belief filter just applies the
realized (a,o) operator; the actor-type posterior updates from action likelihoods. The joint
posterior factorizes exactly: P(s, k | stream) = eta(s) * w(k).

Renders:
  fig 1 (the fan): env-belief attractor on the 2-simplex under each pretraining actor + union
  fig 2 (identification): actor-type posterior P(true type | t) and entropy vs t.

Run:  ~/comp_icl/.venv/bin/python vicarious_oracle.py [--g 1.0] [--alpha 0.6] [--x 0.15]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from generator import mess3_operators, stationary

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- environment
# Action selects the WORLD REGIME: each action a picks a different Mess3 kernel
# (different mixing rate alpha / emission fidelity x), i.e. genuinely different
# contraction structure -- not a state relabeling (a cyclic shift only recolors
# the attractor, by Mess3's symmetry; verified). Semantics: 0=watch, 1=stir, 2=fog.
def controlled_ops(params) -> np.ndarray:
    """A[a,z,i,j] = P(emit z, next j | i, action a); params = [(alpha,x)]*3."""
    return np.stack([mess3_operators(al, xx) for (al, xx) in params], axis=0)


# ---------------------------------------------------------------- actor library
# Each actor: policy(a_prev, o_cur) -> distribution over actions (3,)
U = np.full(3, 1 / 3)

def make_actors(noise=0.1):
    def const(c):
        return lambda ap, o: (1 - noise) * np.eye(3)[c] + noise * U
    actors = {
        "const0":  const(0),
        "const1":  const(1),
        "uniform": lambda ap, o: U.copy(),
        "sticky":  lambda ap, o: (0.75 * np.eye(3)[ap] + 0.25 * U) if ap >= 0 else U.copy(),
        "follow":  lambda ap, o: (1 - noise) * np.eye(3)[o] + noise * U,
        "avoid":   lambda ap, o: (1 - noise) * np.eye(3)[(o + 1) % 3] + noise * U,
    }
    return actors


# ---------------------------------------------------------------- exact filter rollout
def rollout2(A, pi0, policy, T_steps, n_seq, rng, burn=30):
    """Step order per t: actor picks a_t from (a_{t-1}, z_{t-1}); env applies A[a_t] from s_t:
    emits z_t and moves to s_{t+1}. Filter: eta' ~ eta @ A[a_t, z_t]."""
    pts, cols, N = [], [], 0
    for _ in range(n_seq):
        s = rng.choice(3, p=pi0)
        eta = pi0.copy()
        a_prev, z_prev = -1, rng.choice(3)              # dummy first obs
        for t in range(T_steps):
            p_a = policy(a_prev, z_prev)
            a = rng.choice(3, p=p_a)
            probs = A[a].sum(axis=2)[:, s]              # P(z | s, a) over z
            z = rng.choice(3, p=probs / probs.sum())
            nxt = A[a, z, s]                            # P(next j, this z | s) -> next state
            s = rng.choice(3, p=nxt / nxt.sum())
            eta = eta @ A[a, z]
            eta = eta / eta.sum()
            if t >= burn:
                pts.append(eta.copy()); cols.append(z)
            a_prev, z_prev = a, z
    return np.array(pts), np.array(cols)


# ---------------------------------------------------------------- actor posterior
def actor_posterior_curves(A, pi0, actors, T_steps, n_seq, rng):
    """For each true actor: run games, track posterior over actor types from action likelihoods."""
    names = list(actors)
    K = len(names)
    curves = {n: np.zeros(T_steps) for n in names}      # P(true type | t)
    ents = {n: np.zeros(T_steps) for n in names}
    for ti, true_name in enumerate(names):
        for _ in range(n_seq):
            s = rng.choice(3, p=pi0)
            w = np.full(K, 1 / K)
            a_prev, z_prev = -1, rng.choice(3)
            for t in range(T_steps):
                p_true = actors[true_name](a_prev, z_prev)
                a = rng.choice(3, p=p_true)
                lik = np.array([actors[n](a_prev, z_prev)[a] for n in names])
                w = w * lik; w = w / w.sum()
                probs = A[a].sum(axis=2)[:, s]
                z = rng.choice(3, p=probs / probs.sum())
                nxt = A[a, z, s]
                s = rng.choice(3, p=nxt / nxt.sum())
                curves[true_name][t] += w[ti]
                ents[true_name][t] += -(w * np.log(w + 1e-12)).sum()
                a_prev, z_prev = a, z
        curves[true_name] /= n_seq; ents[true_name] /= n_seq
    return curves, ents


# ---------------------------------------------------------------- plotting
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])   # simplex corners
COL = np.array([[0.86, 0.20, 0.15], [0.10, 0.55, 0.25], [0.15, 0.35, 0.80]])

def tri(ax):
    ax.plot(*V[[0, 1, 2, 0]].T, color="k", lw=0.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.axis("off")

def scatter_beliefs(ax, pts, cols, title):
    xy = pts @ V
    ax.scatter(xy[:, 0], xy[:, 1], s=0.25, c=COL[cols], alpha=0.35, linewidths=0)
    tri(ax); ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", type=float, nargs=2, default=[0.6, 0.15])   # action 0: watch
    ap.add_argument("--p1", type=float, nargs=2, default=[0.9, 0.05])   # action 1: stir
    ap.add_argument("--p2", type=float, nargs=2, default=[0.15, 0.25])  # action 2: fog
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--nseq", type=int, default=250)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    params = [tuple(args.p0), tuple(args.p1), tuple(args.p2)]
    A = controlled_ops(params)
    pi0 = stationary(A.sum(axis=(0, 1)) / 3.0)          # stationary of uniform-action marginal
    actors = make_actors()
    rng = np.random.default_rng(1)

    names = list(actors)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    allpts, allcols = [], []
    for i, n in enumerate(names):
        pts, cols = rollout2(A, pi0, actors[n], args.steps, args.nseq, rng)
        scatter_beliefs(axes.flat[i], pts, cols, n)
        allpts.append(pts); allcols.append(cols)
        print(f"{n:8s}: {len(pts)} beliefs, mean H = "
              f"{np.mean([-(p*np.log(p+1e-12)).sum() for p in pts[::37]]):.3f}")
    scatter_beliefs(axes.flat[6], np.concatenate(allpts)[::3], np.concatenate(allcols)[::3],
                    "union (all actors)")
    # panel 8: const2 (the third pure kernel, for the full per-action set)
    pts, cols = rollout2(A, pi0, lambda ap_, o: 0.9 * np.eye(3)[2] + 0.1 * U, args.steps,
                         args.nseq, rng)
    scatter_beliefs(axes.flat[7], pts, cols, "const2")
    fig.suptitle(f"Theoretical env-belief attractors under different actors -- controlled Mess3, "
                 f"kernels {params}", fontsize=11)
    fig.tight_layout()
    out = os.path.join(BASE, "figs", f"vic_fan{args.tag}.png")
    fig.savefig(out, dpi=160); print("wrote", out)

    curves, ents = actor_posterior_curves(A, pi0, actors, 60, 120, rng)
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3.5))
    for n in names:
        ax2[0].plot(curves[n], label=n); ax2[1].plot(ents[n], label=n)
    ax2[0].set_title("P(true actor | t)"); ax2[1].set_title("posterior entropy over actor")
    ax2[0].set_xlabel("t"); ax2[1].set_xlabel("t"); ax2[0].legend(fontsize=7)
    fig2.tight_layout()
    out2 = os.path.join(BASE, "figs", f"vic_ident{args.tag}.png")
    fig2.savefig(out2, dpi=160); print("wrote", out2)


if __name__ == "__main__":
    main()
