#!/usr/bin/env python3
"""Binary family: perturbation dynamics under the two readers (mathpad companion).

World: s' = a w.p. c, else flip s w.p. lam.  Obs: x = s' flipped w.p. q.
Roster: Class A habits P(a=1)=beta_theta; Class B belief-responsive sigma(k*logit(eta)).
Readers R+/R- = frozen law rolled from perturbed/unperturbed state on the same stream.
Courts: Lambda^a (action tokens), Lambda^x (observation tokens).

CPU only. Figures -> figs/*.png, embedded in ~/mathpad.md.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
BLUE, ORANGE, GRAY = "#2a78d6", "#d95926", "#8a8a85"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "legend.frameon": False,
})

def logit(p):    return np.log(p) - np.log1p(-p)
def sig(l):      return 1.0 / (1.0 + np.exp(-l))
def clip(p):     return np.clip(p, 1e-12, 1 - 1e-12)

def eta_pred(eta, a, lam, c):
    """P(s'=1 | belief eta, action a)."""
    return c * a + (1 - c) * ((1 - lam) * eta + lam * (1 - eta))

def px1(ep, q):
    """P(x=1) given one-step-ahead state prob ep."""
    return ep * (1 - q) + (1 - ep) * q

def eta_update(ep, x, q):
    num = np.where(x == 1, ep * (1 - q), ep * q)
    den = num + np.where(x == 1, (1 - ep) * q, (1 - ep) * (1 - q))
    return clip(num / den)

# ---------------------------------------------------------------- world sim --
def world_step(s, a, lam, c, q, R):
    take = R.random(s.shape) < c
    flip = R.random(s.shape) < lam
    s2 = np.where(take, a, np.where(flip, 1 - s, s))
    x = np.where(R.random(s.shape) < q, 1 - s2, s2)
    return s2, x

# ------------------------------------------------- corrected / disowned rows --
def run_eta_row(N, T, t0, lam, c, q, k, dl, mode="do", win=15, seed=1):
    """Class B agent (theta known). Perturb world log-odds by dl at t0.
    mode='do': agent's own belief displaced (agent == R+).
    mode='inject': agent belief intact (== R-); window plays the shadow."""
    R = np.random.default_rng(seed)
    s = (R.random(N) < 0.5).astype(int)
    e_p = np.full(N, 0.5)   # R+ belief
    e_m = np.full(N, 0.5)   # R- belief
    La = np.zeros((T, N)); Lx = np.zeros((T, N)); gap = np.zeros((T, N))
    la = np.zeros(N); lx = np.zeros(N)
    for t in range(T):
        if t == t0:
            e_p = clip(sig(logit(e_p) + dl))
        # --- action ---
        if mode == "do":
            src = e_p
        else:
            src = e_p if (t0 <= t < t0 + win) else e_m   # shadow, then agent(=R-)
        pa_p, pa_m = sig(k * logit(e_p)), sig(k * logit(e_m))
        a = (R.random(N) < sig(k * logit(src))).astype(int)
        # a-court: only unforced acts carry information about A -> starts at release.
        # x-court: the world answers during the window too -> starts at t0.
        active_a = (t >= t0 + win) if mode == "inject" else (t >= t0)
        active_x = (t >= t0)
        if active_a:
            la = la + np.where(a == 1, np.log(pa_p / pa_m),
                               np.log((1 - pa_p) / (1 - pa_m)))
        # --- world & observation ---
        ep_p = eta_pred(e_p, a, lam, c); ep_m = eta_pred(e_m, a, lam, c)
        s, x = world_step(s, a, lam, c, q, R)
        q_p, q_m = px1(ep_p, q), px1(ep_m, q)
        if active_x:
            lx = lx + np.where(x == 1, np.log(q_p / q_m),
                               np.log((1 - q_p) / (1 - q_m)))
        e_p = eta_update(ep_p, x, q); e_m = eta_update(ep_m, x, q)
        La[t], Lx[t], gap[t] = la, lx, np.abs(e_p - e_m)
    return La, Lx, gap

# ------------------------------------------------------ identity-court rows --
def run_m_row(N, T, t0, b1, b2, p0, dp, vertex=False, seed=2):
    """Class A. vertex=False: interior agent, own p displaced by dp at t0 (ratified).
    vertex=True: agent swapped theta1 -> theta2 at t0 (self)."""
    R = np.random.default_rng(seed)
    p_p = np.full(N, p0); p_m = np.full(N, p0)   # readers' P(theta2)
    La = np.zeros((T, N)); la = np.zeros(N)
    for t in range(T):
        if t == t0:
            p_p = clip(p_p + dp) if not vertex else np.ones(N)
            if vertex: p_m = np.zeros(N)
        pa = p_p * b2 + (1 - p_p) * b1 if not vertex else np.full(N, b2)
        a = (R.random(N) < pa).astype(int)
        if t >= t0:
            q_p = p_p * b2 + (1 - p_p) * b1
            q_m = p_m * b2 + (1 - p_m) * b1
            la = la + np.where(a == 1, np.log(q_p / q_m),
                               np.log((1 - q_p) / (1 - q_m)))
            if not vertex:   # Bayes on own actions (both readers)
                for arr in (p_p, p_m):
                    lik2 = np.where(a == 1, b2, 1 - b2)
                    lik1 = np.where(a == 1, b1, 1 - b1)
                    arr[:] = clip(arr * lik2 / (arr * lik2 + (1 - arr) * lik1))
        La[t] = la
    return La

# ---------------------------------------------------------------- figures ----
import pathlib
FIG = pathlib.Path(__file__).parent / "figs"; FIG.mkdir(exist_ok=True)
N, T, T0 = 2000, 200, 25
LAM, C, K, DL = 0.10, 0.0, 1.0, 2.0
B1, B2 = 0.3, 0.7

def style_ax(ax, title):
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.axvline(T0, color=GRAY, lw=0.8, ls=":")

# --- Fig 1: belief trajectories & gap decay, three worlds ---
cfgs = [("static world, informative channel\n(lam=0, q=0.35)", 0.0, 0.35),
        ("mixing world, NO evidence\n(lam=0.1, q=0.5)", 0.10, 0.50),
        ("mixing world, informative channel\n(lam=0.1, q=0.3)", 0.10, 0.30)]
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4), sharex=True)
for j, (name, lam_, q_) in enumerate(cfgs):
    La, Lx, gap = run_eta_row(N, T, T0, lam_, C, q_, K, DL, "do", seed=10 + j)
    ax = axes[0, j]; style_ax(ax, name)
    m = gap.mean(1)
    ax.plot(m, color=BLUE, lw=2, label="mean |eta+ - eta-|")
    for i in range(3):
        ax.plot(gap[:, i], color=BLUE, lw=0.7, alpha=0.25)
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1)
    if j == 0: ax.set_ylabel("belief gap  |η⁺ − η⁻|")
    # theory overlays
    tt = np.arange(T0, T)
    if lam_ > 0:
        rho = (1 - C) * (1 - 2 * lam_)
        ax.plot(tt, m[T0] * rho ** (tt - T0), color=GRAY, lw=1.4, ls="--",
                label="forgetting rate (1−c)(1−2λ)")
    if q_ < 0.5:
        r = (1 - 2 * q_) * np.log((1 - q_) / q_)
        ax.plot(tt, m[T0] * np.exp(-r * (tt - T0)), color=ORANGE, lw=1.4, ls="--",
                label="evidence rate (1−2q)·w_q")
    ax.legend(fontsize=7.5, loc="upper right")
    ax2 = axes[1, j]; style_ax(ax2, "")
    ax2.plot(Lx.mean(1), color=ORANGE, lw=2, label="Λˣ (world court)")
    ax2.plot(La.mean(1), color=BLUE, lw=2, label="Λᵃ (identity court)")
    ax2.axhline(0, color=GRAY, lw=0.8)
    if j == 0: ax2.set_ylabel("court totals (mean)")
    ax2.set_xlabel("t"); ax2.legend(fontsize=7.5)
fig.suptitle("do(ℓ += 2) at t=25: gap decay (top, log scale) and the two courts (bottom)", y=1.0)
fig.tight_layout(); fig.savefig(FIG / "fig1_eta_perturbation.png", bbox_inches="tight"); plt.close(fig)

# --- Fig 2: the (S) table as time series ---
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.1), sharex=True)
rows = []
La, Lx, _ = run_eta_row(N, T, T0, 0.20, C, 0.35, K, DL, "do", seed=21)
rows.append(("corrected: do(ℓ+=2), belief-responsive", La, Lx, None))
LaR = run_m_row(N, T, T0, B1, B2, p0=0.4, dp=0.25, vertex=False, seed=22)
rows.append(("ratified: do(p+=0.25), interior identity", LaR, np.zeros_like(LaR), None))
LaS = run_m_row(N, T, T0, B1, B2, p0=0.0, dp=0.0, vertex=True, seed=23)
rows.append(("self: vertex swap θ₁→θ₂", LaS, np.zeros_like(LaS), None))
LaI, LxI, _ = run_eta_row(N, T, T0, 0.20, C, 0.35, K, DL, "inject", win=5, seed=24)
rows.append(("disowned: inject ℓ-shadow (window 25–30)", LaI, LxI, (T0, T0 + 5)))
for ax, (name, La_, Lx_, win) in zip(axes, rows):
    style_ax(ax, name)
    if win: ax.axvspan(*win, color=GRAY, alpha=0.15, lw=0)
    ax.plot(La_.mean(1), color=BLUE, lw=2, label="Λᵃ")
    ax.plot(Lx_.mean(1), color=ORANGE, lw=2, label="Λˣ")
    for i in range(3):
        ax.plot(La_[:, i], color=BLUE, lw=0.6, alpha=0.2)
        ax.plot(Lx_[:, i], color=ORANGE, lw=0.6, alpha=0.2)
    ax.axhline(0, color=GRAY, lw=0.8); ax.set_xlabel("t")
axes[0].set_ylabel("court totals"); axes[0].legend(fontsize=8)
fig.suptitle("Four rows of the sign table (S), lived forward  (means over 2000 runs + 3 sample paths)", y=1.04)
fig.tight_layout(); fig.savefig(FIG / "fig2_courts.png", bbox_inches="tight"); plt.close(fig)

# --- Fig 3: rates vs theory ---
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
lams = np.linspace(0.02, 0.4, 12)
for c_, col in [(0.0, BLUE), (0.3, ORANGE)]:
    taus = []
    for lam_ in lams:
        _, _, gap = run_eta_row(400, 120, 10, lam_, c_, 0.5, K, DL, "do", seed=31)
        m = gap.mean(1); m0 = m[10]
        below = np.where(m[10:] < m0 / np.e)[0]
        taus.append(below[0] if len(below) else np.nan)
    axes[0].plot(lams, taus, "o", color=col, ms=4, label=f"measured, c={c_}")
    axes[0].plot(lams, -1 / np.log((1 - c_) * (1 - 2 * lams)), color=col, lw=1.4, ls="--")
axes[0].set_xlabel("λ"); axes[0].set_ylabel("τ_η (e-folding, steps)")
axes[0].set_title("τ_η vs theory −1/log[(1−c)(1−2λ)]  (q=½)", fontsize=9, loc="left")
axes[0].legend(fontsize=8)
qs = np.linspace(0.05, 0.5, 10)
lx_inf, tq = [], []
for q_ in qs:
    La, Lx, gap = run_eta_row(600, 150, 10, LAM, C, q_, K, DL, "do", seed=32)
    lx_inf.append(Lx.mean(1)[-1])
    m = gap.mean(1); m0 = m[10]
    below = np.where(m[10:] < m0 / np.e)[0]
    tq.append(below[0] if len(below) else np.nan)
axes[1].plot(qs, np.abs(lx_inf), "o-", color=ORANGE, lw=1.6, ms=4)
axes[1].set_xlabel("q"); axes[1].set_ylabel("|Λˣ_∞|")
axes[1].set_title("the sign is earned only by evidence", fontsize=9, loc="left")
axes[2].plot(qs, tq, "o-", color=BLUE, lw=1.6, ms=4, label="measured")
axes[2].axhline(-1 / np.log((1 - C) * (1 - 2 * LAM)), color=GRAY, ls="--", lw=1.4,
                label="forgetting bound (q=½)")
axes[2].set_xlabel("q"); axes[2].set_ylabel("τ_η")
axes[2].set_title("τ bounded by forgetting; evidence only speeds it", fontsize=9, loc="left")
axes[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig3_rates.png", bbox_inches="tight"); plt.close(fig)

# --- Fig 4: Doob landing law (ratified, exact check) ---
fig, ax = plt.subplots(figsize=(4.2, 3.0))
outs = []
for dp in [0.0, 0.25]:
    R = np.random.default_rng(41)
    p = np.full(20000, 0.4 + dp)
    for t in range(4000):
        a = (R.random(p.shape) < p * B2 + (1 - p) * B1).astype(int)
        l2 = np.where(a == 1, B2, 1 - B2); l1 = np.where(a == 1, B1, 1 - B1)
        p = clip(p * l2 / (p * l2 + (1 - p) * l1))
    outs.append((p > 0.5).mean())
ax.bar([0, 1], outs, width=0.5, color=[BLUE, ORANGE])
ax.set_xticks([0, 1]); ax.set_xticklabels(["p₀ = 0.40", "p₀ = 0.65 (perturbed)"])
for i, (v, th) in enumerate(zip(outs, [0.40, 0.65])):
    ax.annotate(f"{v:.3f}\n(theory {th:.2f})", (i, v), ha="center", va="bottom", fontsize=8.5)
ax.set_ylim(0, 0.9); ax.set_ylabel("P(land θ₂)")
ax.set_title("ratification, exactly: landing law = prior", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig4_doob.png", bbox_inches="tight"); plt.close(fig)

print("figures written to", FIG)
