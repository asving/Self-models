"""Design E2 -- STEP 0b, CPU-only: HIDDEN-action exact filter (joint over actor x a_prev x state).

Same regime-controlled Mess3 (action selects kernel), but the action is NOT observed: the stream
is x_1, x_2, ... only. Exact joint filter over (k = actor type, a = last action, s = env state):
    w'[k,a,s'] ~ sum_{ap,s} w[k,ap,s] * pol[k, ap, x_t, a] * A[a, x_{t+1}, s, s']
The env-belief eta(s) and actor posterior P(k) are marginals. This is the literal
operator-MIXTURE update -- the joint T^{(a,x)} machinery is forced, per the design revision.

Renders: hidden-action attractor fan (marginal eta on the simplex) + identification funnel.
Run:  ~/comp_icl/.venv/bin/python vicarious_oracle2.py
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from generator import mess3_operators, stationary
from vicarious_oracle import make_actors, tri, scatter_beliefs, V, COL

BASE = os.path.dirname(os.path.abspath(__file__))


def controlled_ops(params):
    return np.stack([mess3_operators(al, xx) for (al, xx) in params], axis=0)


def policy_tensor(actors):
    """pol[k, a_prev, o, a] = p_k(a | a_prev, o)."""
    names = list(actors)
    pol = np.zeros((len(names), 3, 3, 3))
    for ki, n in enumerate(names):
        for ap in range(3):
            for o in range(3):
                pol[ki, ap, o] = actors[n](ap, o)
    return names, pol


def run_condition(A, pol, pi0, k_true, T_steps, n_seq, rng, burn=30):
    """Simulate true (actor k_true, env) with hidden actions; run exact joint filter."""
    K = pol.shape[0]
    pts, cols, ptrue, ents = [], [], np.zeros(T_steps), np.zeros(T_steps)
    for _ in range(n_seq):
        s = rng.choice(3, p=pi0)
        a_prev = rng.choice(3)
        x_prev = rng.choice(3)                     # dummy first obs for reactive policies
        w = np.ones((K, 3, 3)) / (K * 3) * pi0[None, None, :]
        w /= w.sum()
        for t in range(T_steps):
            p_a = pol[k_true, a_prev, x_prev]
            a = rng.choice(3, p=p_a)
            pz = A[a].sum(axis=2)[:, s]
            x = rng.choice(3, p=pz / pz.sum())
            nxt = A[a, x, s]
            s_next = rng.choice(3, p=nxt / nxt.sum())
            # filter: uses only x_prev (known) and new x
            w = np.einsum("kps,kpa,asj->kaj", w, pol[:, :, x_prev, :], A[:, x, :, :])
            w /= w.sum()
            eta = w.sum(axis=(0, 1))
            if t >= burn:
                pts.append(eta.copy()); cols.append(x)
            pk = w.sum(axis=(1, 2))
            ptrue[t] += pk[k_true]
            ents[t] += -(pk * np.log(pk + 1e-12)).sum()
            a_prev, x_prev, s = a, x, s_next
    return np.array(pts), np.array(cols), ptrue / n_seq, ents / n_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", type=float, nargs=2, default=[0.6, 0.15])
    ap.add_argument("--p1", type=float, nargs=2, default=[0.9, 0.05])
    ap.add_argument("--p2", type=float, nargs=2, default=[0.15, 0.25])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--nseq", type=int, default=150)
    ap.add_argument("--tag", type=str, default="_hidden")
    args = ap.parse_args()

    A = controlled_ops([tuple(args.p0), tuple(args.p1), tuple(args.p2)])
    pi0 = stationary(A.sum(axis=(0, 1)) / 3.0)
    names, pol = policy_tensor(make_actors())
    rng = np.random.default_rng(2)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3.5))
    allpts, allcols = [], []
    for ki, n in enumerate(names):
        pts, cols, ptrue, ents = run_condition(A, pol, pi0, ki, args.steps, args.nseq, rng)
        scatter_beliefs(axes.flat[ki], pts, cols, n)
        allpts.append(pts); allcols.append(cols)
        ax2[0].plot(ptrue, label=n); ax2[1].plot(ents, label=n)
        print(f"{n:8s}: P(true|t=60)={ptrue[59]:.2f}  P(true|end)={ptrue[-1]:.2f}  "
              f"eta meanH={np.mean([-(p*np.log(p+1e-12)).sum() for p in pts[::29]]):.3f}")
    scatter_beliefs(axes.flat[6], np.concatenate(allpts)[::3], np.concatenate(allcols)[::3],
                    "union (all actors)")
    axes.flat[7].axis("off")
    fig.suptitle("HIDDEN-action env-belief attractors (marginal filter) -- forced operator-mixture",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(BASE, "figs", f"vic_fan{args.tag}.png")
    fig.savefig(out, dpi=160); print("wrote", out)

    ax2[0].set_title("P(true actor | t), actions hidden"); ax2[1].set_title("actor-posterior entropy")
    ax2[0].set_xlabel("t"); ax2[1].set_xlabel("t"); ax2[0].legend(fontsize=7)
    fig2.tight_layout()
    out2 = os.path.join(BASE, "figs", f"vic_ident{args.tag}.png")
    fig2.savefig(out2, dpi=160); print("wrote", out2)


if __name__ == "__main__":
    main()
