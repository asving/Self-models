"""The SGD court, computed for real (replaces the cartoon).

Task: ring Z_6, observable MDP. Actions +-1 move with slip 0.25. Start n0=0,
goal g=3 (antipodal). Objective: expected time-at-goal over horizon K, plus an
entropy anchor alpha * sum_n KL(pi(.|n) || uniform) keeping the optimum
interior. Policy: tabular softmax, logits w in R^{6x2}. Everything exact:
the objective is computed by differentiable forward DP over the 6-state
distribution; gradients and the 12x12 Hessian by autograd (float64).

Structure forced by mirror symmetry (n -> -n mod 6 swaps actions, fixes 0,3):
  - 6 softmax-shift directions: exact reparameterization kernel (lambda = 0).
  - 2 functional tie directions (action-preference logits at states 0 and 3):
    reward-flat EXACTLY (both actions lead to the same state-3 continuation
    with mirror-equal hitting laws), so their curvature comes only from the
    anchor: lambda proportional to alpha -- the anchor as a weak identity court.
  - 4 task directions (preference logits at strict states 1,2,4,5): O(1)
    curvature from the reward itself.

Outputs:
  courts_sgd.pdf/.png -- left: measured gradient-flow restoration of
  perturbations along measured Hessian eigendirections, with predicted
  exp(-eta lambda_i t) dashed; right: the same directions under stochastic
  (REINFORCE) gradients -- task mode pinned, identity mode wandering (weakly
  tethered by the anchor), shift mode frozen (its per-sample gradient is
  identically zero).
  Printed: spectrum table, lambda_soft vs alpha sweep (linearity check).
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

R, SLIP, G, N0, K = 6, 0.25, 3, 0, 24
ALPHA = 0.02
SEED = 0

# movement kernels P(n'|n,a), a in {+1,-1}
PN = torch.zeros(2, R, R)
for ai, a in enumerate((+1, -1)):
    for n in range(R):
        PN[ai, n, (n + a) % R] += 1 - SLIP
        PN[ai, n, n] += SLIP

def objective(w, alpha=ALPHA):
    """L(w) = -E[sum_{t=1..K} 1[n_t=G]] + alpha * sum_n KL(pi_n || uniform)."""
    pi = torch.softmax(w, dim=1)                    # (R, 2)
    M = pi[:, 0, None] * PN[0] + pi[:, 1, None] * PN[1]   # (R, R) row-stoch
    p = torch.zeros(R); p = p.clone(); p[N0] = 1.0
    J = torch.zeros(())
    for _ in range(K):
        p = p @ M
        J = J + p[G]
    kl = (pi * (torch.log(pi) - np.log(0.5))).sum()
    return -J + alpha * kl

def converge(alpha=ALPHA, steps=6000, lr=0.1):
    w = torch.zeros(R, 2, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for i in range(steps):
        opt.zero_grad(); L = objective(w, alpha); L.backward(); opt.step()
    for i in range(3000):                           # vanilla GD polish
        g = torch.autograd.grad(objective(w, alpha), w)[0]
        with torch.no_grad():
            w -= 0.5 * g
    gn = torch.autograd.grad(objective(w, alpha), w)[0].norm().item()
    return w.detach(), gn

wstar, gn = converge()
print(f"converged: |grad| = {gn:.2e};  pi* =")
print(torch.softmax(wstar, 1).numpy().round(3))

# ---- Hessian and spectrum ----
def flat_obj(wf, alpha=ALPHA):
    return objective(wf.reshape(R, 2), alpha)

Hs = torch.autograd.functional.hessian(flat_obj, wstar.flatten())
evals, evecs = torch.linalg.eigh(Hs)
shift_basis = torch.zeros(R, 12)
for n in range(R):
    shift_basis[n, 2 * n] = shift_basis[n, 2 * n + 1] = 1 / np.sqrt(2)
def classify(v):
    sh = (shift_basis @ v).norm() ** 2
    loc = v.reshape(R, 2)
    tie = loc[[0, 3]].norm() ** 2 / v.norm() ** 2
    return f"shift {sh:.2f} tie-loc {tie:.2f}"
print("\nHessian spectrum (12 modes):")
for i in range(12):
    print(f"  lambda = {evals[i]:+.5f}   {classify(evecs[:, i])}")

# pick representative modes: largest-lambda task mode; the two anchor (tie)
# modes = smallest nonzero; a shift mode = |lambda| ~ 0 with shift overlap
lam = evals.numpy()
task_i = int(np.argmax(lam))
nonzero = [i for i in range(12) if lam[i] > 1e-8]
soft_i = nonzero[0]
shift_i = int(np.argmin(np.abs(lam)))
picks = [(task_i, "task direction (strict state)", 'C0'),
         (soft_i, "identity tie (anchor-tethered)", 'C2'),
         (shift_i, "softmax shift (reparameterization)", 'C7')]

# ---- panel 1: deterministic gradient-flow restoration ----
ETA, TSTEPS, EPS = 0.05, 4000, 0.5
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
ax = axes[0]
tt = np.arange(TSTEPS + 1) * ETA
for idx, lab, col in picks:
    v = evecs[:, idx]
    w = (wstar.flatten() + EPS * v).clone().requires_grad_(True)
    traj = [EPS]
    for s in range(TSTEPS):
        g = torch.autograd.grad(flat_obj(w), w)[0]
        with torch.no_grad():
            w -= ETA * g
        traj.append(float((w.detach() - wstar.flatten()) @ v))
    ax.plot(tt, np.abs(traj) / EPS, color=col, lw=2,
            label=f"{lab}: $\\lambda$ = {lam[idx]:.4f}")
    ax.plot(tt, np.exp(-lam[idx] * tt), color=col, lw=1, ls='--')
ax.set(xlabel=r"training time $\eta \cdot$ steps", ylabel="surviving fraction",
       title="measured restoration vs predicted $e^{-\\lambda t}$ (dashed)",
       ylim=(-0.05, 1.1))
ax.legend(fontsize=8)

# ---- panel 2: stochastic gradients (REINFORCE) ----
def reinforce_grad(w, batch, rng, alpha=ALPHA):
    pi = torch.softmax(w, 1).detach().numpy()
    gacc = np.zeros((R, 2))
    for _ in range(batch):
        n, glogp, rew = N0, np.zeros((R, 2)), np.zeros(K + 1)
        states, acts = [], []
        for t in range(K):
            ai = 0 if rng.random() < pi[n, 0] else 1
            states.append(n); acts.append(ai)
            n = rng.choice(R, p=PN[ai, n].numpy())
            rew[t + 1] = (n == G)
        rtg = np.cumsum(rew[::-1])[::-1]
        for t, (s, ai) in enumerate(zip(states, acts)):
            gl = -pi[s]; gl[ai] += 1
            gacc[s] += gl * rtg[t + 1]
    gJ = gacc / batch
    wt = w.clone().requires_grad_(True)
    pit = torch.softmax(wt, 1)
    kl = (pit * (torch.log(pit) - np.log(0.5))).sum()
    gK = torch.autograd.grad(kl, wt)[0].numpy()
    return -gJ + alpha * gK        # gradient of L = -J + alpha KL

rng = np.random.default_rng(SEED)
SG_STEPS, SG_LR, BATCH = 4000, 0.02, 8
w = wstar.flatten().clone()
comps = {idx: [0.0] for idx, _, _ in picks}
for s in range(SG_STEPS):
    g = torch.from_numpy(reinforce_grad(w.reshape(R, 2), BATCH, rng)).flatten()
    w = w - SG_LR * g
    d = w - wstar.flatten()
    for idx, _, _ in picks:
        comps[idx].append(float(d @ evecs[:, idx]))
ax = axes[1]
for idx, lab, col in picks:
    ax.plot(np.arange(SG_STEPS + 1), comps[idx], color=col, lw=1.2, label=lab)
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.set(xlabel="SGD steps (REINFORCE, batch 8)", ylabel="component along mode",
       title="stochastic training: task pinned, identity wanders")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("courts_sgd.pdf"); fig.savefig("courts_sgd.png", dpi=110)
print("saved courts_sgd (computed)")

# ---- alpha scaling of the identity modes ----
print("\nanchor scaling (lambda of the two tie modes vs alpha):")
for alpha in (0.01, 0.02, 0.04, 0.08):
    ws, _ = converge(alpha)
    Hs2 = torch.autograd.functional.hessian(
        lambda wf: flat_obj(wf, alpha), ws.flatten())
    ev2 = torch.linalg.eigh(Hs2).eigenvalues.numpy()
    nz = sorted(v for v in ev2 if v > 1e-8)
    print(f"  alpha={alpha:.2f}: two smallest nonzero = {nz[0]:.5f}, {nz[1]:.5f}"
          f"   ratio to alpha: {nz[0]/alpha:.3f}, {nz[1]/alpha:.3f}")
