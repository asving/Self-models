"""Fourth figure: interior persona posterior (zeta in the simplex, not an atom).

Same ring world as gen_curves.py, but (i) the agent carries an interior mixture
zeta over the two styles, updated by own-action Bayes (reflexive: likelihoods
evaluated at its OWN env-belief), and (ii) the tilt is the running-reward
variant (reward accrued each round at the goal; multiplicative h-recursion),
so the policy is belief-responsive at every round and style evidence stays
scarce -- zeta remains liquid deep into the episode.

Two experiments (perturbation at t0, readers as always):
  Z: interior zeta-perturbation, log-odds shifted by delta toward style B.
     Predicted: Lambda^a a BOUNDED plateau (<= delta: the prior-odds
     displacement at the landing vertex, an optional-stopping quantity);
     paths bifurcate by destination; Lambda^x identically zero.
  H: eta-perturbation (believed position rotated by 2) with interior zeta.
     Predicted: Lambda^x heals as before (fast plateau), while the readers'
     zeta lineages are permanently displaced by mis-attribution during the
     healing window (the SCAR) -- Lambda^a keeps accruing on a second, slower
     timescale after the world court has gone quiet.

Outputs: courts_zeta.pdf/.png + printed checks. CPU. cwd = this folder.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- ring world (as in gen_curves.py) ----
R, SLIP = 6, 0.25
T, T0 = 64, 6
GOAL = 3
N_RUNS = 20000
BETA_RUN = 0.35           # running-reward tilt strength (per round at goal)
PI0 = {0: np.array([0.58, 0.42]), 1: np.array([0.42, 0.58])}
DELTA_ZETA = 2.0         # zeta perturbation, log-odds toward style B
ETA_SHIFT = 3

PN = np.zeros((2, R, R))
for ai, a in enumerate((+1, -1)):
    for n in range(R):
        PN[ai, n, (n + a) % R] += 1 - SLIP
        PN[ai, n, n] += SLIP
PX = np.full((R, R), 0.05)
for n in range(R):
    PX[n, n] += 0.50
    PX[n, (n + 1) % R] += 0.10
    PX[n, (n - 1) % R] += 0.10

def backward_h_running(g):
    """h_t(n,z) = E[ exp(BETA_RUN * sum_{s>t} 1[n_s=g]) | n_t=n, z, pi0 ]."""
    h = np.ones((T + 1, R, 2))
    for t in range(T - 1, -1, -1):
        for z in (0, 1):
            for n in range(R):
                cont = sum(PI0[z][ai] * (PN[ai, n] @ (np.exp(
                    BETA_RUN * (np.arange(R) == g)) * h[t + 1, :, z]))
                    for ai in (0, 1))
                h[t, n, z] = cont
    return h

H = backward_h_running(GOAL)

def pol_style(t, eta, z):
    """pi_g(a | eta, style z): running tilt, one-step lookahead."""
    gvec = np.exp(BETA_RUN * (np.arange(R) == GOAL))
    w = np.array([PI0[z][ai] * (eta @ (PN[ai] @ (gvec * H[t + 1, :, z])))
                  for ai in (0, 1)])
    return w / w.sum()

def pol_mix(t, eta, zeta):
    """The acting/reading law: mixture over styles."""
    return zeta[0] * pol_style(t, eta, 0) + zeta[1] * pol_style(t, eta, 1)

def zeta_update(t, eta, zeta, ai):
    """Own-action Bayes, likelihoods at the holder's OWN eta."""
    lik = np.array([pol_style(t, eta, 0)[ai], pol_style(t, eta, 1)[ai]])
    zn = zeta * lik
    return zn / zn.sum()

def filt(eta, ai, x):
    jn = eta @ PN[ai]
    jn = jn * PX[:, x]
    return jn / jn.sum()

def lodds(z):
    return np.log(z[1] / z[0])

def run(kind, n_runs=N_RUNS, seed=2):
    r = np.random.default_rng(seed)
    W = T - T0
    La = np.zeros((n_runs, W)); Lx = np.zeros((n_runs, W))
    zeta_gap = np.zeros(W)               # |log-odds gap| between readers' zetas
    eta_gap = np.zeros(W)                # L1 gap between readers' env beliefs
    saP = np.zeros(W); saM = np.zeros(W)  # per-token reader surprises, a-court
    sxP = np.zeros(W); sxM = np.zeros(W)  # per-token reader surprises, x-court
    liq0 = liqT = 0.0                    # liquidity probe: |log-odds| of agent zeta
    for run_i in range(n_runs):
        n = int(r.integers(R))
        etaA = np.full(R, 1 / R); etaP = etaA.copy(); etaM = etaA.copy()
        zetaA = np.array([0.5, 0.5]); zetaP = zetaA.copy(); zetaM = zetaA.copy()
        la = lx = 0.0
        for t in range(T):
            if t == T0:
                if kind == 'Z':
                    lo = lodds(zetaA) + DELTA_ZETA
                    zetaA = np.array([1, np.exp(lo)]); zetaA /= zetaA.sum()
                    zetaP = zetaA.copy()
                elif kind == 'H':
                    etaA = np.roll(etaA, ETA_SHIFT); etaP = etaA.copy()
            pa = pol_mix(t, etaA, zetaA)
            ai = 0 if r.random() < pa[0] else 1
            if t >= T0:
                lp = pol_mix(t, etaP, zetaP)[ai]
                lm = pol_mix(t, etaM, zetaM)[ai]
                la += np.log(lp / lm)
                saP[t - T0] += -np.log(lp); saM[t - T0] += -np.log(lm)
            n = int(r.choice(R, p=PN[ai, n]))
            x = int(r.choice(R, p=PX[n]))
            if t >= T0:
                xp = etaP @ PN[ai] @ PX[:, x]; xm = etaM @ PN[ai] @ PX[:, x]
                lx += np.log(xp / xm)
                sxP[t - T0] += -np.log(xp); sxM[t - T0] += -np.log(xm)
                La[run_i, t - T0] = la; Lx[run_i, t - T0] = lx
                zeta_gap[t - T0] += abs(lodds(zetaP) - lodds(zetaM))
                eta_gap[t - T0] += np.abs(etaP - etaM).sum() / 2
            # reflexive updates: zeta at OWN eta, then eta on (a, x)
            zetaA = zeta_update(t, etaA, zetaA, ai)
            zetaP = zeta_update(t, etaP, zetaP, ai)
            zetaM = zeta_update(t, etaM, zetaM, ai)
            etaA = filt(etaA, ai, x); etaP = filt(etaP, ai, x)
            etaM = filt(etaM, ai, x)
            if t == T0:
                liq0 += abs(lodds(zetaA))
            if t == T - 1:
                liqT += abs(lodds(zetaA))
    print(f"  [liquidity] mean |log-odds zeta_agent|: t0 {liq0/n_runs:.2f}, "
          f"T {liqT/n_runs:.2f}")
    for v in (saP, saM, sxP, sxM):
        v /= n_runs
    return dict(La=La, Lx=Lx, zg=zeta_gap / n_runs, eg=eta_gap / n_runs,
                saP=saP, saM=saM, sxP=sxP, sxM=sxM)

tgrid = np.arange(T0, T)
resZ = run('Z'); resH = run('H')
for kind, res in (('Z', resZ), ('H', resH)):
    print(f"{kind}: La_end {res['La'][:, -1].mean():+.3f}  "
          f"Lx_end {res['Lx'][:, -1].mean():+.3f}  "
          f"zeta-gap end {res['zg'][-1]:.3f}  eta-gap end {res['eg'][-1]:.3f}")

fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharex=True)

# column 1: Z experiment
ax = axes[0, 0]
for i in range(4):
    ax.plot(tgrid, resZ['La'][i], color='C0', alpha=0.3, lw=0.7)
ax.plot(tgrid, resZ['La'].mean(0), color='C0', lw=2, label=r"$\Lambda^a$")
ax.plot(tgrid, resZ['Lx'].mean(0), color='C1', lw=2, label=r"$\Lambda^x$")
ax.axhline(DELTA_ZETA, color='C0', lw=0.8, ls=':', label=r"$\delta$")
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.set_title(r"$do(\zeta \to \zeta + \delta)$: court totals", fontsize=10)
ax.set_ylabel("court total (nats)")
ax.legend(fontsize=8, loc="upper left")
ax = axes[1, 0]
ax.plot(tgrid, resZ['saP'], '-', color='C0', lw=2, label=r"$R^+$, $a$-tokens")
ax.plot(tgrid, resZ['saM'], '--', color='C0', lw=2, label=r"$R^-$, $a$-tokens")
ax.set_title("per-token reader surprises, identity court", fontsize=9)
ax.set_ylabel("per-token surprise (nats)")
ax.set_xlabel("round $t$")
ax.legend(fontsize=8)

# column 2: H experiment
ax = axes[0, 1]
for i in range(4):
    ax.plot(tgrid, resH['La'][i], color='C0', alpha=0.3, lw=0.7)
ax.plot(tgrid, resH['La'].mean(0), color='C0', lw=2, label=r"$\Lambda^a$")
ax.plot(tgrid, resH['Lx'].mean(0), color='C1', lw=2, label=r"$\Lambda^x$")
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.set_title(r"$do(\eta \to \mathrm{rot}_3\,\eta)$, interior $\zeta$: court totals",
             fontsize=10)
ax.legend(fontsize=8, loc="center right")
ax = axes[1, 1]
ax.plot(tgrid, resH['sxP'], '-', color='C1', lw=2, label=r"$R^+$, $x$-tokens")
ax.plot(tgrid, resH['sxM'], '--', color='C1', lw=2, label=r"$R^-$, $x$-tokens")
ax.set_title("per-token reader surprises, world court", fontsize=9)
ax.set_xlabel("round $t$")
ax.legend(fontsize=8)

# column 3: state-space view (wound vs scar)
ax = axes[0, 2]
ax.plot(tgrid, resZ['zg'], color='C2', lw=2, label=r"$\zeta$ gap (log-odds)")
ax.plot(tgrid, resZ['eg'], color='C3', lw=2, label=r"$\eta$ gap (TV)")
ax.axhline(DELTA_ZETA, color='C2', lw=0.8, ls=':')
ax.set_title(r"$do(\zeta)$: reader-state gaps", fontsize=10)
ax.set_ylabel("state gap")
ax.legend(fontsize=8)
ax = axes[1, 2]
ax.plot(tgrid, resH['zg'], color='C2', lw=2, label=r"$\zeta$ gap (log-odds)")
ax.plot(tgrid, resH['eg'], color='C3', lw=2, label=r"$\eta$ gap (TV)")
ax.set_title(r"$do(\eta)$: the wound closes, the scar remains", fontsize=10)
ax.set_ylabel("state gap")
ax.set_xlabel("round $t$")
ax.legend(fontsize=8)

for row in axes:
    for ax in row:
        ax.axvline(T0, color='gray', lw=0.6, ls='--')
fig.tight_layout()
fig.savefig("courts_zeta.pdf"); fig.savefig("courts_zeta.png", dpi=110)
print("saved courts_zeta (2x3)")
