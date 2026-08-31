#!/usr/bin/env python3
"""The shape zoo: every asymptotic regime of the two-court curves, one panel each.

Constructions from theory_fable.md / theory_sol.md / theory_synthesis.md.
CPU only. Outputs figs/zoo_grid.png, figs/zoo_signflip.png, figs/zoo_fisher.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib

BLUE, ORANGE, GRAY = "#2a78d6", "#d95926", "#8a8a85"
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 8.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "legend.frameon": False,
})
FIG = pathlib.Path(__file__).parent / "figs"; FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(7)
clip = lambda p: np.clip(p, 1e-12, 1 - 1e-12)

def bern_llr_paths(pplus, pminus, ptrue, T, N):
    """Independent tokens: per-step Bernoulli predictions p+(t), p-(t), truth p*(t).
    Arrays broadcastable to (T,). Returns Lambda paths (T,N)."""
    pp = np.broadcast_to(np.asarray(pplus, float), (T,))
    pm = np.broadcast_to(np.asarray(pminus, float), (T,))
    pt = np.broadcast_to(np.asarray(ptrue, float), (T,))
    y = rng.random((T, N)) < pt[:, None]
    inc = np.where(y, np.log(pp / pm)[:, None], np.log((1 - pp) / (1 - pm))[:, None])
    return np.cumsum(inc, axis=0)

T, N = 3000, 400
t = np.arange(1, T + 1)

panels = []

# P1 plateau > 0: ratified — interior identity belief shifted, truth = R+ mixture.
b1, b2, p0, dp = 0.3, 0.7, 0.4, 0.25
# simulate collapse dynamics with both readers Bayes-updating on the agent's acts
def ratified(T, N):
    pP = np.full(N, p0 + dp); pM = np.full(N, p0); L = np.zeros((T, N)); acc = 0
    for k in range(T):
        pa_true = pP * b2 + (1 - pP) * b1
        a = rng.random(N) < pa_true
        qp = pP * b2 + (1 - pP) * b1; qm = pM * b2 + (1 - pM) * b1
        acc = acc + np.where(a, np.log(qp / qm), np.log((1 - qp) / (1 - qm)))
        L[k] = acc
        for arr in (pP, pM):
            l2 = np.where(a, b2, 1 - b2); l1 = np.where(a, b1, 1 - b1)
            arr[:] = clip(arr * l2 / (arr * l2 + (1 - arr) * l1))
    return L
panels.append(("plateau (+): ratified interior shift", ratified(T, N), None, "lin"))

# P2 plateau < 0: static bit revealed slowly, unfavorable prior (Sol §1.4 example, noisy)
# readers: priors w+ = .8, w- = .2 on theta=1; truth theta=0; obs = theta flipped w.p. .3
def static_reveal(T, N, wp=.8, wm=.2, q=.3):
    lp = np.full(N, np.log(wp / (1 - wp))); lm = np.full(N, np.log(wm / (1 - wm)))
    wq = np.log((1 - q) / q); L = np.zeros((T, N)); acc = 0
    for k in range(T):
        x = rng.random(N) < q  # truth theta=0 -> P(x=1)=q
        qp = 1 / (1 + np.exp(-lp)) * (1 - 2 * q) + q
        qm = 1 / (1 + np.exp(-lm)) * (1 - 2 * q) + q
        acc = acc + np.where(x, np.log(qp / qm), np.log((1 - qp) / (1 - qm)))
        L[k] = acc
        d = np.where(x, wq, -wq); lp += d; lm += d
    return L
panels.append(("plateau (−): wrong-way prior on a static bit", static_reveal(T, N), None, "lin"))

# P3 exactly zero: belief-blind policy, world perturbation -> identity court silent
panels.append(("exactly zero: empty action-shadow", np.zeros((T, 3)), None, "lin"))

# P4 linear +: vertex swap, truth = R+
panels.append(("linear (+): vertex→vertex, penetrated",
               bern_llr_paths(b2, b1, b2, T, N), ("slope KL", t * (b2 * np.log(b2 / b1) + (1 - b2) * np.log((1 - b2) / (1 - b1)))), "lin"))

# P5 linear −: false 'took' against vertex readers (truth = R-)
panels.append(("linear (−): false took, truth = R⁻",
               bern_llr_paths(b2, b1, b1, T, N), ("−slope KL", -t * (b1 * np.log(b1 / b2) + (1 - b1) * np.log((1 - b1) / (1 - b2)))), "lin"))

# P6 log n in a STATIONARY finite model: Sol's Jordan block (deterministic stream y^n)
rho, bb = 0.5, 0.5
jordan = np.log(1 + t * bb / rho)
panels.append(("log t: stationary Jordan block (Sol §1.4)", jordan[:, None] * np.ones((1, 1)),
               ("log(1+bt/ρ)", np.log(1 + t * bb / rho)), "logx"))

# P7 sqrt(t): clocked policy, delta_n = a n^(-1/4), penetrated
a_ = 0.2; d7 = a_ * t ** (-0.25)
panels.append(("√t drift: clocked δₙ = a·n^(−1/4), penetrated",
               bern_llr_paths(0.5 + d7, 0.5, 0.5 + d7, T, N), ("2a²√t/…", 4 * a_**2 * np.sqrt(t)), "lin"))

# P8 log t: clocked delta_n = a/sqrt(n), penetrated
d8 = a_ / np.sqrt(t)
panels.append(("log t drift: clocked δₙ = a/√n, penetrated",
               bern_llr_paths(0.5 + d8, 0.5, 0.5 + d8, T, N), ("2a² log t", 2 * a_**2 * np.log(t)), "logx"))

# P9 LIL oscillation: truth exactly between the readers
L9 = bern_llr_paths(0.75, 0.25, 0.5, T, N)
env = np.log(3) * np.sqrt(2 * t * np.log(np.maximum(np.log(t), 1e-9).clip(min=1e-9) + (t < 3) * 1))
env = np.log(3) * np.sqrt(2 * t * np.clip(np.log(np.clip(np.log(t), 1, None)), 0, None) + 2)
panels.append(("knife-edge: LIL oscillation, sign changes forever", L9, ("±LIL envelope", env), "lin"))

# P10 zigzag expectation: block-alternating true bias (blocks 50,100,200,400,800)
pt10 = np.empty(1550); start, bias = 0, 0.9
for blk in (50, 100, 200, 400, 800):
    pt10[start:start + blk] = bias; start += blk; bias = 1.0 - bias
panels.append(("non-monotone E[Λ]: block-alternating truth", bern_llr_paths(0.75, 0.25, pt10, 1550, N), None, "lin"))

# P11 interior-prior boundedness: total LLR trapped between prior-odds bands
def interior_bound(T, N, wp=.7, wm=.3):
    # two latents u in {0,1}, emissions Ber(.8)/Ber(.2); truth u=1
    lp = np.full(N, np.log(wp / (1 - wp))); lm = np.full(N, np.log(wm / (1 - wm)))
    L = np.zeros((T, N)); acc = 0
    for k in range(T):
        x = rng.random(N) < 0.8
        qp = 1 / (1 + np.exp(-lp)) * 0.6 + 0.2; qm = 1 / (1 + np.exp(-lm)) * 0.6 + 0.2
        acc = acc + np.where(x, np.log(qp / qm), np.log((1 - qp) / (1 - qm)))
        L[k] = acc
        d = np.where(x, np.log(4), -np.log(4)); lp += d; lm += d
    return L
L11 = interior_bound(600, N)
band = (np.log(0.7 / 0.3), np.log(0.3 / 0.7))
panels.append(("interior priors: trapped by prior odds (Sol §3.2)", L11, ("bands", None), "lin"))

# P12 mean vs a.s.: Borel–Cantelli spikes
def bc_spikes(T, N):
    L = np.zeros((T, N)); acc = np.zeros(N)
    stage = np.arange(1, 12)
    pos = np.cumsum(200 * np.ones_like(stage))  # spike slots at t=200,400,...
    for k in range(T):
        n_here = np.searchsorted(pos, k)
        if k in pos:
            n = n_here + 1
            gate = rng.random(N) < 2.0 ** (-n)
            acc = acc + np.where(gate, 4.0 ** n * 0.05, 0.0)
        else:
            acc = acc * 0.999  # gentle decay of spikes back toward 0
        L[k] = acc
    return L
L12 = bc_spikes(2400, 2000)
panels.append(("mean lies: a.s.→0 but E[Λ]→∞ (spikes)", L12, None, "mean-med"))

fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.2))
for ax, (name, L, overlay, kind) in zip(axes.flat, panels):
    Tl = L.shape[0]; tl = np.arange(1, Tl + 1)
    if kind == "mean-med":
        ax.plot(tl, L.mean(1), color=ORANGE, lw=1.8, label="mean")
        ax.plot(tl, np.median(L, 1), color=BLUE, lw=1.8, label="median")
        ax.legend(fontsize=7)
    else:
        m = L.mean(1)
        ax.plot(tl, m, color=BLUE, lw=1.8)
        for i in range(min(3, L.shape[1])):
            ax.plot(tl, L[:, i], color=BLUE, lw=0.6, alpha=0.25)
    if overlay is not None and overlay[1] is not None:
        ov = overlay[1][:Tl]
        ax.plot(tl, ov, color=GRAY, ls="--", lw=1.3)
        if "LIL" in overlay[0] or "envelope" in overlay[0]:
            ax.plot(tl, -ov, color=GRAY, ls="--", lw=1.3)
    if name.startswith("interior"):
        ax.axhline(band[0], color=GRAY, ls="--", lw=1.3)
        ax.axhline(band[1], color=GRAY, ls="--", lw=1.3)
    if kind == "logx":
        ax.set_xscale("log")
    ax.set_title(name, fontsize=8.3, loc="left")
    ax.axhline(0, color=GRAY, lw=0.7)
for ax in axes[-1]: ax.set_xlabel("t")
for r in axes: r[0].set_ylabel("Λ")
fig.suptitle("The shape zoo: every asymptotic regime of a court curve, realized  (mean + sample paths; gray dashed = theory)", y=1.0)
fig.tight_layout(); fig.savefig(FIG / "zoo_grid.png", bbox_inches="tight"); plt.close(fig)

# --- Sign flip in q (Sol §1.5): same perturbation endorsed at sharp channel, refuted at blurry
u_p, u_m, u_s = 0.6, 0.1, 0.33
qs = np.linspace(0, 0.5, 200)
def F(q):
    p_p = q + (1 - 2 * q) * u_p; p_m = q + (1 - 2 * q) * u_m; p_s = q + (1 - 2 * q) * u_s
    return p_s * np.log(p_p / p_m) + (1 - p_s) * np.log((1 - p_p) / (1 - p_m))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.0))
ax1.plot(qs, F(qs), color=BLUE, lw=2)
ax1.axhline(0, color=GRAY, lw=0.8)
ax1.set_xlabel("q"); ax1.set_ylabel("expected contribution F(q)")
ax1.set_title("one perturbation, two verdicts: F changes sign in q", fontsize=9, loc="left")
for q_, col, lab in [(0.02, BLUE, "q=0.02: endorsed"), (0.42, ORANGE, "q=0.42: refuted")]:
    p_p = q_ + (1 - 2 * q_) * u_p; p_m = q_ + (1 - 2 * q_) * u_m; p_s = q_ + (1 - 2 * q_) * u_s
    Lq = bern_llr_paths(p_p, p_m, p_s, 300, 500)
    ax2.plot(Lq.mean(1), color=col, lw=2, label=lab)
ax2.axhline(0, color=GRAY, lw=0.8); ax2.legend(fontsize=8)
ax2.set_xlabel("t"); ax2.set_ylabel("Λˣ (mean)")
ax2.set_title("the same v, lived at two noise levels", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(FIG / "zoo_signflip.png", bbox_inches="tight"); plt.close(fig)

# --- Fisher parabola: E[Lambda_inf] vs epsilon, both laws
eps = np.linspace(-0.3, 0.3, 13)
T5, N5 = 4000, 3000
means_pen, means_unp = [], []
for e in eps:
    d = 0.5 + e * (1 / np.sqrt(np.arange(1, T5 + 1)))  # decaying so Lambda_inf finite
    Lp = bern_llr_paths(d, 0.5, d, T5, N5)[-1].mean()      # penetrated (truth=R+)
    Lu = bern_llr_paths(d, 0.5, 0.5, T5, N5)[-1].mean()    # unperturbed law (truth=R-)
    means_pen.append(Lp); means_unp.append(Lu)
I = 2 * np.sum(1 / np.arange(1, T5 + 1)) * 2  # sum 4*delta^2 per eps^2: KL≈2d^2 => coeff 2*H_T
coef = 2 * np.log(T5) + 2 * 0.5772 * 2  # approx, fit visually instead
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.plot(eps, means_pen, "o", color=BLUE, ms=4, label="truth = perturbed (E[Λ] = +ε²·I/2)")
ax.plot(eps, means_unp, "o", color=ORANGE, ms=4, label="truth = unperturbed (−ε²·I/2)")
cfit = np.polyfit(eps, means_pen, 2)[0]
ax.plot(eps, cfit * eps ** 2, color=GRAY, ls="--", lw=1.2)
ax.plot(eps, -cfit * eps ** 2, color=GRAY, ls="--", lw=1.2)
ax.set_xlabel("perturbation size ε"); ax.set_ylabel("E[Λ_∞]")
ax.set_title("the local invariant: channel Fisher information", fontsize=9, loc="left")
ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "zoo_fisher.png", bbox_inches="tight"); plt.close(fig)

print("zoo written")
