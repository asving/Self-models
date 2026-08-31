"""Visualization variants for the court curves, per Asvin's suggestion:
  Variant A: row 1 cumulative Lambda (as now); row 2 per-token mean rate
             (the KL-gap flow) per court -- tau is where the rate dies.
  Variant B: row 1 cumulative Lambda; row 2 the two readers' PER-TOKEN mean
             surprises plotted separately per court -- reader convergence (or
             permanent separation) is literally visible as lines merging.
Reuses the ring-world model from gen_curves.py (exec of its model section).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = open("gen_curves.py").read()
exec(src.split("def run_experiment")[0])          # model, tables, policy, filt

def run_experiment_full(kind, n_runs=4000, seed=1):
    r = np.random.default_rng(seed)
    h_g, h_gp = backward_h(GOAL), backward_h(GOALP)
    W = T - T0
    La = np.zeros((n_runs, W)); Lx = np.zeros((n_runs, W))
    saP = np.zeros(W); saM = np.zeros(W); sxP = np.zeros(W); sxM = np.zeros(W)
    for run in range(n_runs):
        n = int(r.integers(R)); z_true = int(r.random() < 0.5)
        eta_A = np.full(R, 1 / R); etaP = eta_A.copy(); etaM = eta_A.copy()
        zP = zM = z_true
        hA = hP = hM = h_g
        la = lx = 0.0
        for t in range(T):
            if t == T0:
                if kind == 'eta':
                    eta_A = np.roll(eta_A, ETA_SHIFT); etaP = eta_A.copy()
                elif kind == 'z':
                    z_true ^= 1; zP = z_true
                elif kind == 'g':
                    hA = h_gp; hP = h_gp
            pa = policy_g(hA, t, eta_A, z_true)
            ai = 0 if r.random() < pa[0] else 1
            if t >= T0:
                lp = policy_g(hP, t, etaP, zP)[ai]
                lm = policy_g(hM, t, etaM, zM)[ai]
                la += np.log(lp / lm)
                saP[t - T0] += -np.log(lp); saM[t - T0] += -np.log(lm)
            n = int(r.choice(R, p=PN[ai, n]))
            x = int(r.choice(R, p=PX[n]))
            if t >= T0:
                xp = etaP @ PN[ai] @ PX[:, x]; xm = etaM @ PN[ai] @ PX[:, x]
                lx += np.log(xp / xm)
                sxP[t - T0] += -np.log(xp); sxM[t - T0] += -np.log(xm)
                La[run, t - T0] = la; Lx[run, t - T0] = lx
            eta_A = filt(eta_A, ai, x); etaP = filt(etaP, ai, x)
            etaM = filt(etaM, ai, x)
    for v in (saP, saM, sxP, sxM):
        v /= n_runs
    return dict(La=La, Lx=Lx, saP=saP, saM=saM, sxP=sxP, sxM=sxM)

titles = {
    'eta': r"env-belief $do(\eta \to \mathrm{rot}_2\,\eta)$: corrected",
    'z':   r"style register $do(z \to \bar{z})$: ratified",
    'g':   r"goal register $do(g \to g\,')$: ratified",
}
tgrid = np.arange(T0, T)
results = {k: run_experiment_full(k) for k in ('eta', 'z', 'g')}

def row1(ax, res, kind):
    for i in range(3):
        ax.plot(tgrid, res['La'][i], color='C0', alpha=0.22, lw=0.7)
        ax.plot(tgrid, res['Lx'][i], color='C1', alpha=0.22, lw=0.7)
    ax.plot(tgrid, res['La'].mean(0), color='C0', lw=2,
            label=r"$\Lambda^a$ (identity court)")
    ax.plot(tgrid, res['Lx'].mean(0), color='C1', lw=2,
            label=r"$\Lambda^x$ (world court)")
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.axvline(T0, color='gray', lw=0.6, ls='--')
    ax.set_title(titles[kind], fontsize=10)

# ---- Variant A: cumulative + per-token rate ----
fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
for j, kind in enumerate(('eta', 'z', 'g')):
    res = results[kind]
    row1(axes[0, j], res, kind)
    ra = np.diff(np.concatenate([[0], res['La'].mean(0)]))
    rx = np.diff(np.concatenate([[0], res['Lx'].mean(0)]))
    axes[1, j].plot(tgrid, ra, 'o-', color='C0', ms=3, lw=1.5,
                    label=r"$\langle\Delta\Lambda^a_t\rangle$")
    axes[1, j].plot(tgrid, rx, 's-', color='C1', ms=3, lw=1.5,
                    label=r"$\langle\Delta\Lambda^x_t\rangle$")
    axes[1, j].axhline(0, color='gray', lw=0.6, ls=':')
    axes[1, j].set_xlabel("round $t$")
axes[0, 0].set_ylabel("court total (nats)")
axes[1, 0].set_ylabel("per-token rate (nats)")
axes[0, 0].legend(fontsize=8, loc="upper left")
axes[1, 0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("courts_variantA.pdf"); fig.savefig("courts_variantA.png", dpi=110)
print("saved variant A")

# ---- Variant B: cumulative + separate per-token reader surprises ----
fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
for j, kind in enumerate(('eta', 'z', 'g')):
    res = results[kind]
    row1(axes[0, j], res, kind)
    ax = axes[1, j]
    ax.plot(tgrid, res['saP'], '-', color='C0', lw=2,
            label=r"$R^+$ surprise, $a$-tokens")
    ax.plot(tgrid, res['saM'], '--', color='C0', lw=2,
            label=r"$R^-$ surprise, $a$-tokens")
    ax.plot(tgrid, res['sxP'], '-', color='C1', lw=2,
            label=r"$R^+$ surprise, $x$-tokens")
    ax.plot(tgrid, res['sxM'], '--', color='C1', lw=2,
            label=r"$R^-$ surprise, $x$-tokens")
    ax.set_xlabel("round $t$")
axes[0, 0].set_ylabel("court total (nats)")
axes[1, 0].set_ylabel("per-token surprise (nats)")
axes[0, 0].legend(fontsize=8, loc="upper left")
axes[1, 0].legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig("courts_variantB.pdf"); fig.savefig("courts_variantB.png", dpi=110)
print("saved variant B")
