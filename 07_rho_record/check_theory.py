"""Validate the rho-record theory (mathpad 2026-08-13) by simulation, and
produce the oracle floor curves for the training experiment.

Checks (each asserts, and all results land in THEORY_CHECKS.json):
  1. substitution principle: closed-form observer likelihood == explicit
     marginalization over the true action (exact, 1e-12).
  2. mirror: r and 1-r observer likelihoods agree after the sign relabel (exact).
  3. corner (lam=1/2, r=1/2): observer posterior over c never moves;
     agent posterior concentrates.
  4. Fisher about c (lam=1/2): MC score^2 vs formula b^2/(1-c^2 b^2) for both
     readers; ratio ~ br^2.
  5. Pi-rate (lam=1/2, c known): MC loss gap vs h2(.5(1+c bq br)) - h2(.5(1+c bq)).
  6. experiment floors (LAM, Q, R, c~U, theta~pm1): per-position expected x-slot
     loss of observer vs agent filters (the two curves the transformer is
     compared against), plus the mean per-round Pi-gap.

Run: cwd = this folder;  ~/comp_icl/.venv/bin/python check_theory.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import world as W

rng = np.random.default_rng(7)
out = {}

# ---------- 1. substitution principle ----------
worst = 0.0
for x in (+1, -1):
    for at in (+1, -1):
        for th in (+1, -1):
            for c in np.linspace(0, 1, 21):
                d = abs(W.observer_likelihood_marginalized(x, at, th, c)
                        - W.observer_likelihood_closed(x, at, th, c))
                worst = max(worst, d)
assert worst < 1e-12, worst
out["substitution_principle_max_abs_err"] = worst
print(f"[1] substitution principle exact: max err {worst:.2e}")

# ---------- 2. mirror r <-> 1-r ----------
worst = 0.0
for x in (+1, -1):
    for at in (+1, -1):
        for th in (+1, -1):
            for c in np.linspace(0, 1, 21):
                d = abs(W.observer_likelihood_closed(x, at, th, c, r=0.25)
                        - W.observer_likelihood_closed(x, -at, th, c, r=0.75))
                worst = max(worst, d)
assert worst < 1e-12, worst
out["mirror_max_abs_err"] = worst
print(f"[2] mirror r<->1-r exact: max err {worst:.2e}")

# ---------- 3. the corner ----------
ep = W.gen_batch(512, rng, lam=0.5, r=0.5)
obs = W.observer_filter(ep, lam=0.5, r=0.5)
agt = W.agent_filter(ep, lam=0.5)
drift = np.abs(obs["kappa"] - 0.5).max()
assert drift < 1e-9, drift              # observer's kappa never leaves the prior
# agent's posterior sd of c shrinks: track via kappa spread across truth
agt_final_err = np.abs(agt["kappa"][:, -1] - ep["c"]).mean()
obs_final_err = np.abs(obs["kappa"][:, -1] - ep["c"]).mean()
prior_err = np.abs(0.5 - ep["c"]).mean()
out["corner"] = dict(obs_kappa_drift=float(drift),
                     agent_final_abs_err=float(agt_final_err),
                     observer_final_abs_err=float(obs_final_err),
                     prior_abs_err=float(prior_err))
print(f"[3] corner: observer kappa pinned at prior (drift {drift:.1e}); "
      f"agent |kappa-c| {agt_final_err:.3f} vs prior {prior_err:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
tgrid = np.arange(1, W.T + 1)
ax.plot(tgrid, np.abs(agt["kappa"] - ep["c"][:, None]).mean(0), label="agent |κ̂−c|")
ax.plot(tgrid, np.abs(obs["kappa"] - ep["c"][:, None]).mean(0), label="observer |κ̂−c|")
ax.axhline(prior_err, ls=":", c="gray", label="prior")
ax.set(xlabel="round", ylabel="mean |κ̂ − c|",
       title="the corner (λ=½, r=½): κ is first-person only")
ax.legend(); fig.tight_layout(); fig.savefig("figs/theory_corner.png", dpi=150)

# ---------- 4. Fisher about c (lam = 1/2) ----------
def fisher_mc(c0, contrast, n=400_000):
    """MC E[score^2] for a Bernoulli coincidence channel with contrast b:
    P(y=1)=(1+c b)/2, score = b y/(1+ c b y)."""
    y = np.where(rng.uniform(size=n) < 0.5 * (1 + c0 * contrast), 1.0, -1.0)
    sc = contrast * y / (1 + c0 * contrast * y)
    return (sc ** 2).mean()

bq, br = 1 - 2 * W.Q, 1 - 2 * W.R
c0 = 0.5
rows = []
for r_ in [0.0, 0.1, 0.25, 0.4, 0.5]:
    b_o = bq * (1 - 2 * r_)
    I_A_mc, I_O_mc = fisher_mc(c0, bq), fisher_mc(c0, b_o)
    I_A_th = bq ** 2 / (1 - c0 ** 2 * bq ** 2)
    I_O_th = b_o ** 2 / (1 - c0 ** 2 * b_o ** 2)
    rows.append(dict(r=r_, I_A_mc=I_A_mc, I_A_th=I_A_th, I_O_mc=I_O_mc, I_O_th=I_O_th))
    assert abs(I_A_mc - I_A_th) / I_A_th < 0.02
    if I_O_th > 0:
        assert abs(I_O_mc - I_O_th) / I_O_th < 0.03
out["fisher"] = rows
print("[4] Fisher MC matches formulas; ratio at r=0.25:",
      f"{rows[2]['I_O_mc']/rows[2]['I_A_mc']:.4f} (theory ~ {(bq*br)**2/bq**2 * (1-c0**2*bq**2)/(1-c0**2*(bq*br)**2):.4f})")

# ---------- 5. Pi-rate (lam=1/2, c known) ----------
def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

cgrid = np.linspace(0, 1, 21)
pi_mc, pi_th = [], []
for c0 in cgrid:
    ep = W.gen_batch(4000, rng, lam=0.5, c_fixed=c0)
    # known-c, lam=1/2 predictives
    p_obs = 0.5 * (1 + c0 * bq * br * ep["atil"])
    p_agt = 0.5 * (1 + c0 * bq * ep["a"])
    gap = (W.xslot_loss(p_obs, ep["x"]) - W.xslot_loss(p_agt, ep["x"])).mean()
    pi_mc.append(gap)
    pi_th.append(h2(0.5 * (1 + c0 * bq * br)) - h2(0.5 * (1 + c0 * bq)))
pi_mc, pi_th = np.array(pi_mc), np.array(pi_th)
assert np.abs(pi_mc - pi_th).max() < 0.005
out["pi_rate"] = dict(c=cgrid.tolist(), mc=pi_mc.tolist(), exact=pi_th.tolist(),
                      leading_order=(cgrid ** 2 * bq ** 2 / 2 * (1 - br ** 2)).tolist())
print(f"[5] Pi-rate MC matches exact entropy-gap (max err "
      f"{np.abs(pi_mc-pi_th).max():.4f}); at c=1: {pi_th[-1]:.4f} nats/round")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(cgrid, pi_mc, "o", ms=4, label="simulation")
ax.plot(cgrid, pi_th, "-", label="exact  h₂ gap")
ax.plot(cgrid, cgrid ** 2 * bq ** 2 / 2 * (1 - br ** 2), "--",
        label="leading order  c²β_q²(1−β_r²)/2")
ax.set(xlabel="consequence c", ylabel="nats/round",
       title="privileged-information rate: privilege requires consequence")
ax.legend(); fig.tight_layout(); fig.savefig("figs/theory_pi_rate.png", dpi=150)

# ---------- 6. experiment floors ----------
B = 8192
rng_f = np.random.default_rng(1234)          # same seed family as the eval set
ep = W.gen_batch(B, rng_f)
obs = W.observer_filter(ep)
agt = W.agent_filter(ep)
lo = W.xslot_loss(obs["p_pos"], ep["x"]).mean(0)
la = W.xslot_loss(agt["p_pos"], ep["x"]).mean(0)
out["floors"] = dict(observer=lo.tolist(), agent=la.tolist(),
                     mean_gap_all=float((lo - la).mean()),
                     mean_gap_late=float((lo - la)[8:].mean()),
                     obs_final=float(lo[-16:].mean()), agt_final=float(la[-16:].mean()))
print(f"[6] floors at experiment params: observer late {lo[-16:].mean():.4f}, "
      f"agent late {la[-16:].mean():.4f}, mean Pi-gap {(lo-la).mean():.4f} nats/round")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(tgrid, lo, label="observer floor (stream only)")
ax.plot(tgrid, la, label="agent floor (knows true a)")
ax.axhline(np.log(2), ls=":", c="gray", label="log 2 (know nothing)")
ax.fill_between(tgrid, la, lo, alpha=0.15, label="Π (privileged information)")
ax.set(xlabel="round t", ylabel="expected x-slot loss (nats)",
       title=f"oracle floors  (λ={W.LAM}, q={W.Q}, r={W.R}, c~U[0,1])")
ax.legend(); fig.tight_layout(); fig.savefig("figs/theory_floors.png", dpi=150)

with open("THEORY_CHECKS.json", "w") as f:
    json.dump(out, f, indent=2)
print("all checks passed -> THEORY_CHECKS.json, figs/theory_*.png")
