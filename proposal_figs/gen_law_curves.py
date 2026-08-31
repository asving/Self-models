"""In-context LAW perturbations, completing the battery: perturb the agent's
initial policy pi0 (style emission biases) and its goal-tilt strength beta,
mid-episode. Model reused from gen_zeta_curves.py (interior-zeta ring world).

Experiments (perturbation at t0; acting agent + R+ carry the new law, R- the
old; all filter the same realized stream):
  P: pi0-law perturbation -- style-A bias 0.58 -> 0.72 (the agent's habits
     change). Expected: identity court drifts forever (law gaps cannot heal by
     state merging); world court identically zero.
  B: tilt-law perturbation -- beta 0.35 -> 0.70 (the agent becomes more
     goal-driven). Expected: same court structure; the reward court notices
     (time-at-goal changes), reported in the printout.

Output: courts_laws.pdf/.png (two panels, court totals + samples).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = open("gen_zeta_curves.py").read()
exec(src.split("def run(")[0])       # model tables, backward_h_running, policies

PI0_NEW = {0: np.array([0.72, 0.28]), 1: np.array([0.42, 0.58])}
BETA_NEW = 0.70

def make_law(pi0, beta):
    """Return (pol_style, zeta_update-likelihood fn) closures for a law."""
    h = np.ones((T + 1, R, 2))
    gvec = np.exp(beta * (np.arange(R) == GOAL))
    for t in range(T - 1, -1, -1):
        for z in (0, 1):
            for n in range(R):
                h[t, n, z] = sum(pi0[z][ai] * (PN[ai, n] @ (gvec * h[t + 1, :, z]))
                                 for ai in (0, 1))
    def pol_style_l(t, eta, z):
        w = np.array([pi0[z][ai] * (eta @ (PN[ai] @ (gvec * h[t + 1, :, z])))
                      for ai in (0, 1)])
        return w / w.sum()
    return pol_style_l

LAW_OLD = make_law(PI0, BETA_RUN)
LAWS_NEW = {'P': make_law(PI0_NEW, BETA_RUN), 'B': make_law(PI0, BETA_NEW)}

def pol_mix_l(law, t, eta, zeta):
    return zeta[0] * law(t, eta, 0) + zeta[1] * law(t, eta, 1)

def zupd_l(law, t, eta, zeta, ai):
    lik = np.array([law(t, eta, 0)[ai], law(t, eta, 1)[ai]])
    zn = zeta * lik
    return zn / zn.sum()

def run_law(kind, n_runs=8000, seed=3):
    r = np.random.default_rng(seed)
    W = T - T0
    La = np.zeros((n_runs, W)); Lx = np.zeros((n_runs, W))
    goal_time = 0.0
    law_new = LAWS_NEW[kind]
    for run_i in range(n_runs):
        n = int(r.integers(R))
        etaA = np.full(R, 1 / R); etaP = etaA.copy(); etaM = etaA.copy()
        zetaA = np.array([0.5, 0.5]); zetaP = zetaA.copy(); zetaM = zetaA.copy()
        la = lx = 0.0
        lawA = LAW_OLD
        for t in range(T):
            if t == T0:
                lawA = law_new             # the do(): the agent's law changes
            pa = pol_mix_l(lawA, t, etaA, zetaA)
            ai = 0 if r.random() < pa[0] else 1
            if t >= T0:
                lp = pol_mix_l(law_new, t, etaP, zetaP)[ai]
                lm = pol_mix_l(LAW_OLD, t, etaM, zetaM)[ai]
                la += np.log(lp / lm)
            n = int(r.choice(R, p=PN[ai, n]))
            goal_time += (n == GOAL)
            x = int(r.choice(R, p=PX[n]))
            if t >= T0:
                xp = etaP @ PN[ai] @ PX[:, x]; xm = etaM @ PN[ai] @ PX[:, x]
                lx += np.log(xp / xm)
                La[run_i, t - T0] = la; Lx[run_i, t - T0] = lx
            zetaA = zupd_l(lawA, t, etaA, zetaA, ai)
            zetaP = zupd_l(law_new, t, etaP, zetaP, ai)
            zetaM = zupd_l(LAW_OLD, t, etaM, zetaM, ai)
            etaA = filt(etaA, ai, x); etaP = filt(etaP, ai, x)
            etaM = filt(etaM, ai, x)
    return La, Lx, goal_time / n_runs

titles = {
    'P': r"initial policy $do(\pi_0 \to \pi_0')$: habit change",
    'B': r"tilt strength $do(\beta \to 2\beta)$: urgency change",
}
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
tgrid = np.arange(T0, T)
for ax, kind in zip(axes, ('P', 'B')):
    La, Lx, gt = run_law(kind)
    for i in range(4):
        ax.plot(tgrid, La[i], color='C0', alpha=0.3, lw=0.7)
    ax.plot(tgrid, La.mean(0), color='C0', lw=2, label=r"$\Lambda^a$ (identity court)")
    ax.plot(tgrid, Lx.mean(0), color='C1', lw=2, label=r"$\Lambda^x$ (world court)")
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.axvline(T0, color='gray', lw=0.6, ls='--')
    ax.set_title(titles[kind], fontsize=10)
    ax.set_xlabel("round $t$")
    ax.legend(fontsize=8, loc="upper left")
    print(f"{kind}: La_end {La[:, -1].mean():+.3f}  Lx_end {Lx[:, -1].mean():+.3f}"
          f"  mean time-at-goal {gt:.2f}")
axes[0].set_ylabel("court total (nats)")
fig.tight_layout()
fig.savefig("courts_laws.pdf"); fig.savefig("courts_laws.png", dpi=110)
print("saved courts_laws")
# baseline time-at-goal (no perturbation): reuse P machinery with law_new = old
LAWS_NEW['P'] = LAW_OLD
La, Lx, gt0 = run_law('P', n_runs=4000, seed=9)
print(f"baseline mean time-at-goal (no perturbation): {gt0:.2f}")
