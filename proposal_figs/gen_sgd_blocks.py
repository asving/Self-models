"""SGD court, block-resolved: the four semantic variables as parameter blocks.

Same ring task as gen_sgd_court.py, but the policy is computed through
structured machinery with deliberate redundancy (as in a network):
    logits L[n,a] = w_base[n,a] + beta * Qhat[n,a],
    Qhat[n,a]     = sum_{n'} Phat_sliphat(n'|n,a) * V[n'],
so the parameter space carries four blocks:
    w_base  (6x2) -- initial/base policy        (Asvin's variable 1)
    beta, V (1+6) -- goal-biased policy machinery (variable 2)
    sliphat (1)   -- internal environment model   (variable 4's SGD analogue)
    tie directions of w_base at states 0,3 -- the style/identity analogue (3)
Objective: exact expected time-at-goal (differentiable DP) + alpha KL anchor.

Measurements after convergence:
  A. Perturb each block; integrate exact gradient flow; track FUNCTIONAL
     distance (policy space) and PARAMETER distance per block.
     Prediction: function heals wherever loss sees the perturbation, but
     healing may occur by OTHER blocks compensating -- a permanent parameter
     displacement (the weight-space scar); tie directions heal only at the
     anchor rate; shifts never.
  B. The scar panel: for the env-model (sliphat) perturbation, plot sliphat(t),
     the compensating drift of the other blocks, and the functional distance.

Output: courts_sgd_blocks.pdf/.png + printed table. CPU, float64.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
R, SLIP, G, N0, K = 6, 0.25, 3, 0, 24
ALPHA = 0.02

PN = torch.zeros(2, R, R)
for ai, a in enumerate((+1, -1)):
    for n in range(R):
        PN[ai, n, (n + a) % R] += 1 - SLIP
        PN[ai, n, n] += SLIP

IDX = np.arange(R)
def phat(sliphat):
    P = torch.zeros(2, R, R)
    for ai, a in enumerate((+1, -1)):
        for n in range(R):
            P[ai, n, (n + a) % R] += 1 - sliphat
            P[ai, n, n] += sliphat
    return P

def unpack(p):
    w_base = p[:12].reshape(R, 2)
    V = p[12:18]
    beta = p[18]
    sliphat = torch.sigmoid(p[19])
    return w_base, V, beta, sliphat

def policy(p):
    w_base, V, beta, sliphat = unpack(p)
    Ph = phat(sliphat)
    Q = torch.stack([Ph[0] @ V, Ph[1] @ V], dim=1)      # (R, 2)
    return torch.softmax(w_base + beta * Q, dim=1)

def objective(p, alpha=ALPHA):
    pi = policy(p)
    M = pi[:, 0, None] * PN[0] + pi[:, 1, None] * PN[1]
    st = torch.zeros(R); st = st.clone(); st[N0] = 1.0
    J = torch.zeros(())
    for _ in range(K):
        st = st @ M
        J = J + st[G]
    kl = (pi * (torch.log(pi) - np.log(0.5))).sum()
    return -J + alpha * kl

# ---- converge ----
p0 = torch.zeros(20)
p0[18] = 1.0                       # beta init
p0[19] = torch.logit(torch.tensor(0.25))   # sliphat init at truth
p = p0.clone().requires_grad_(True)
opt = torch.optim.Adam([p], lr=0.05)
for i in range(5000):
    opt.zero_grad(); L = objective(p); L.backward(); opt.step()
for i in range(2000):
    g = torch.autograd.grad(objective(p), p)[0]
    with torch.no_grad():
        p -= 0.3 * g
pstar = p.detach()
gn = torch.autograd.grad(objective(p), p)[0].norm().item()
pistar = policy(pstar)
print(f"converged: |grad| = {gn:.2e}")
print("pi*:", pistar.detach().numpy().round(3))
wb, V, beta, sh = unpack(pstar)
print(f"beta* = {float(beta):.3f}, sliphat* = {float(sh):.3f}")

BLOCKS = {
    "task (base logits, strict state)": (lambda: pert_base_strict(), 'C0'),
    "identity tie (base logits, state 0)": (lambda: pert_base_tie(), 'C2'),
    "goal value V": (lambda: pert_V(), 'C1'),
    "tilt beta": (lambda: pert_beta(), 'C4'),
    "env model sliphat": (lambda: pert_slip(), 'C3'),
}
def pert_base_strict():
    d = torch.zeros(20); d[2 * 1] = 1 / np.sqrt(2); d[2 * 1 + 1] = -1 / np.sqrt(2)
    return d          # action-preference logit at strict state 1
def pert_base_tie():
    d = torch.zeros(20); d[0] = 1 / np.sqrt(2); d[1] = -1 / np.sqrt(2)
    return d          # action-preference logit at tied state 0
def pert_V():
    d = torch.zeros(20); d[12 + G] = 1.0
    return d
def pert_beta():
    d = torch.zeros(20); d[18] = 1.0
    return d
def pert_slip():
    d = torch.zeros(20); d[19] = 1.0
    return d

def fdist(p_):
    return float((policy(p_) - pistar).abs().max())

EPS, ETA, STEPS, THIN = 1.0, 0.05, 6000, 20
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
ax = axes[0]
print("\nblock perturbations: functional healing + parameter scar")
scar_traces = None
for name, (mk, col) in BLOCKS.items():
    d = mk()
    w = (pstar + EPS * d).clone().requires_grad_(True)
    fs, ts = [fdist(w.detach())], [0.0]
    slips, others = [float(torch.sigmoid(w.detach()[19]))], [0.0]
    for s in range(STEPS):
        g = torch.autograd.grad(objective(w), w)[0]
        with torch.no_grad():
            w -= ETA * g
        if s % THIN == 0:
            fs.append(fdist(w.detach())); ts.append((s + 1) * ETA)
            if name.startswith("env model"):
                slips.append(float(torch.sigmoid(w.detach()[19])))
                others.append(float((w.detach()[:18] - pstar[:18]).norm()))
    dp = w.detach() - pstar
    print(f"  {name:38s} f-dist: {fs[0]:.4f} -> {fs[-1]:.5f}   "
          f"|param scar| = {float(dp.norm()):.4f}")
    ax.plot(ts, fs, color=col, lw=2, label=name)
    if name.startswith("env model"):
        scar_traces = (ts, slips, others, fs)
ax.set(xlabel=r"training time $\eta\cdot$steps", ylabel="functional distance "
       r"$\max_n \|\pi_t - \pi^*\|_1$", yscale="log",
       title="gradient flow heals the FUNCTION (rates differ by block)")
ax.legend(fontsize=7)

ax = axes[1]
ts, slips, others, fs = scar_traces
ax.plot(ts, np.abs(np.array(slips) - float(sh)), color='C3', lw=2,
        label=r"env-model error $|\widehat{\mathrm{slip}} - \widehat{\mathrm{slip}}^*|$")
ax.plot(ts, others, color='C5', lw=2,
        label=r"compensation $\|\Delta(w_{\mathrm{base}}, V)\|$")
ax.plot(ts, fs, color='k', lw=1.5, ls='--', label="functional distance")
ax.set(xlabel=r"training time $\eta\cdot$steps",
       title="the weight-space scar: function heals, parameters migrate")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("courts_sgd_blocks.pdf"); fig.savefig("courts_sgd_blocks.png", dpi=110)
print("saved courts_sgd_blocks")
