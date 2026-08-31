"""Analytic court curves for proposal.tex: the kappa(v) field in a concrete
coupled agent/environment system.

Environment: a ring Z_R (R=6). Action a in {+1,-1} moves the state by a
  w.p. 1-slip, leaves it w.p. slip. Emission: x = correct position w.p. p_c,
  else one of the two neighbors (uniform). Position is persistent (sticky) so
  belief perturbations neither heal instantly nor persist forever, and the
  direction one should move depends on believed position at ALL times.
Agent: Z = {A, B}, two movement styles, static (identity transition):
  pi0(a=+1 | A) = 0.7, pi0(a=+1 | B) = 0.3. Both styles reach any goal on the
  ring under the tilt (from opposite sides): identity is goal-equivalent.
Post-trained policy pi_g: KL-tilt of pi0 toward terminal reward
  exp(beta * 1[n_T = g]), one-step lookahead on the belief (control as
  inference / soft conditioning on the final state).

Perturbations at t0 (do() on the acting agent; two frozen readers R+/- differ
only in the perturbed coordinate; both filter the same realized stream):
  eta: believed position rotated by +2 ring steps   -> corrected
  z:   style register swapped A <-> B               -> ratified (drift)
  g:   goal register swapped g -> g'                -> ratified (drift)

Outputs: courts_incontext.pdf, courts_sgd.pdf. CPU, exact filters.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- parameters ----
R = 6                    # ring size
SLIP = 0.25
PCORR = 0.55             # correct-reading probability (see PX construction)
BETA = 6.0               # tilt strength
T, T0 = 64, 56
GOAL, GOALP = 3, 0       # goal and swapped goal
N_RUNS = 4000
PI0 = {0: np.array([0.7, 0.3]), 1: np.array([0.3, 0.7])}   # z: [P(+1), P(-1)]
ETA_SHIFT = 2            # ring rotation of the believed position

# transition P(n'|n,a) and emission P(x|n') tables
PN = np.zeros((2, R, R))            # [a_idx (0:+1, 1:-1), n, n']
for ai, a in enumerate((+1, -1)):
    for n in range(R):
        PN[ai, n, (n + a) % R] += 1 - SLIP
        PN[ai, n, n] += SLIP
PX = np.full((R, R), 0.05)          # PX[n', x]: full support (floor 0.05)
for n in range(R):
    PX[n, n] += 0.50                # correct reading: 0.55 total
    PX[n, (n + 1) % R] += 0.10      # neighbors: 0.15 total each
    PX[n, (n - 1) % R] += 0.10
assert np.allclose(PX.sum(1), 1) and PX.min() > 0

def backward_h(g):
    """h_t(n, z) = E[exp(beta 1[n_T = g]) | n_t = n, z, follow pi0]."""
    h = np.zeros((T + 1, R, 2))
    h[T] = 1.0
    h[T, g, :] = np.exp(BETA)
    for t in range(T - 1, -1, -1):
        for z in (0, 1):
            for n in range(R):
                h[t, n, z] = sum(PI0[z][ai] * (PN[ai, n] @ h[t + 1, :, z])
                                 for ai in (0, 1))
    return h

def policy_g(h, t, eta, z):
    """pi_g(a | eta, z) prop pi0(a|z) * E_eta[ E[h_{t+1} | n, a] ]."""
    w = np.array([PI0[z][ai] * (eta @ (PN[ai] @ h[t + 1, :, z]))
                  for ai in (0, 1)])
    return w / w.sum()

def filt(eta, ai, x):
    jn = eta @ PN[ai]
    jn = jn * PX[:, x]
    return jn / jn.sum()

def run_experiment(kind, n_runs=N_RUNS, seed=1):
    r = np.random.default_rng(seed)
    h_g, h_gp = backward_h(GOAL), backward_h(GOALP)
    La = np.zeros((n_runs, T - T0)); Lx = np.zeros((n_runs, T - T0))
    succ_p = succ_m = 0.0
    for run in range(n_runs):
        n = int(r.integers(R))
        z_true = int(r.random() < 0.5)
        eta_A = np.full(R, 1 / R); etaP = eta_A.copy(); etaM = eta_A.copy()
        zP = zM = z_true          # style known to both readers pre-perturbation
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
                la += np.log(policy_g(hP, t, etaP, zP)[ai]
                             / policy_g(hM, t, etaM, zM)[ai])
            n = int(r.choice(R, p=PN[ai, n]))
            x = int(r.choice(R, p=PX[n]))
            if t >= T0:
                lx += np.log((etaP @ PN[ai] @ PX[:, x])
                             / (etaM @ PN[ai] @ PX[:, x]))
                La[run, t - T0] = la; Lx[run, t - T0] = lx
            eta_A = filt(eta_A, ai, x); etaP = filt(etaP, ai, x)
            etaM = filt(etaM, ai, x)
        if kind == 'g':
            succ_p += (n == GOALP); succ_m += (n == GOAL)
    if kind == 'g':
        print(f"  g-swap terminal: P(n_T = g') = {succ_p/n_runs:.3f}, "
              f"P(n_T = g) = {succ_m/n_runs:.3f}")
    return La, Lx

titles = {
    'eta': r"env-belief $do(\eta \to \mathrm{rot}_2\,\eta)$: corrected",
    'z':   r"style register $do(z \to \bar{z})$: ratified",
    'g':   r"goal register $do(g \to g\,')$: ratified",
}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
tgrid = np.arange(T0, T)
for ax, kind in zip(axes, ('eta', 'z', 'g')):
    La, Lx = run_experiment(kind)
    for i in range(3):
        ax.plot(tgrid, La[i], color='C0', alpha=0.25, lw=0.7)
        ax.plot(tgrid, Lx[i], color='C1', alpha=0.25, lw=0.7)
    ax.plot(tgrid, La.mean(0), color='C0', lw=2, label=r"$\Lambda^a$ (identity court)")
    ax.plot(tgrid, Lx.mean(0), color='C1', lw=2, label=r"$\Lambda^x$ (world court)")
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.axvline(T0, color='gray', lw=0.6, ls='--')
    ax.set_title(titles[kind], fontsize=10)
    ax.set_xlabel("round $t$")
    print(f"{kind}: La_end {La[:, -1].mean():+.3f}  Lx_end {Lx[:, -1].mean():+.3f}")
axes[0].set_ylabel("court total (nats)")
axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("courts_incontext.pdf")
print("saved courts_incontext.pdf")

# ---- figure 2: the SGD court (linearized gradient flow) ----
tt = np.linspace(0, 10, 200)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for lamb, lab in [(1.0, "environment-model direction"),
                  (0.35, "belief-machinery direction"),
                  (0.12, "value/tilt direction")]:
    ax.plot(tt, np.exp(-lamb * tt), lw=2, label=lab)
ax.plot(tt, np.ones_like(tt), lw=2, ls='--',
        label="policy-gauge direction (which optimal policy)")
ax.set(xlabel="training time (units of $1/\\eta$)",
       ylabel="residual perturbation $\\|P_i\\,\\delta w\\|$",
       ylim=(-0.05, 1.15))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("courts_sgd.pdf")
print("saved courts_sgd.pdf")
