"""SGD court, final consolidated computation (replaces gen_sgd_court/blocks
outputs in the document). Anchor alpha = 0.10 keeps the optimum interior so
every block is functionally visible; directions are chosen EXPLICITLY (no
eigen-selection bugs). Produces one 2x2 figure:

 (a) functional healing per semantic block under exact gradient flow;
 (b) the weight-space scar: function vs parameter distances for two blocks;
 (c) stochastic training (REINFORCE): component wander along the explicit
     tie direction vs a strict task direction vs a softmax shift;
 (d) the anchor law: fitted functional healing rate of the tie block vs alpha
     (predicted proportional to alpha).

Model as in gen_sgd_blocks.py: logits = w_base + beta * Qhat(sliphat, V).
All objectives exact (differentiable DP over 6 states); float64; CPU.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
R, SLIP, G, N0, K = 6, 0.25, 3, 0, 24
ALPHA = 0.10

PN = torch.zeros(2, R, R)
for ai, a in enumerate((+1, -1)):
    for n in range(R):
        PN[ai, n, (n + a) % R] += 1 - SLIP
        PN[ai, n, n] += SLIP

def phat(sliphat):
    P = torch.zeros(2, R, R)
    for ai, a in enumerate((+1, -1)):
        for n in range(R):
            P[ai, n, (n + a) % R] += 1 - sliphat
            P[ai, n, n] += sliphat
    return P

def unpack(p):
    return p[:12].reshape(R, 2), p[12:18], p[18], torch.sigmoid(p[19])

def policy(p):
    w_base, V, beta, sliphat = unpack(p)
    Ph = phat(sliphat)
    Q = torch.stack([Ph[0] @ V, Ph[1] @ V], dim=1)
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

def converge(alpha, adam=4000, polish=1500):
    p0 = torch.zeros(20); p0[18] = 1.0
    p0[19] = torch.logit(torch.tensor(0.25))
    p = p0.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.05)
    for _ in range(adam):
        opt.zero_grad(); L = objective(p, alpha); L.backward(); opt.step()
    for _ in range(polish):
        g = torch.autograd.grad(objective(p, alpha), p)[0]
        with torch.no_grad():
            p -= 0.3 * g
    return p.detach()

pstar = converge(ALPHA)
pistar = policy(pstar)
print("pi* (alpha=0.1):"); print(pistar.numpy().round(3))
_, Vs, betas, shs = unpack(pstar)
print(f"beta* = {float(betas):.3f}, sliphat* = {float(shs):.3f} (true slip 0.25)")

def direction(name):
    d = torch.zeros(20)
    if name == 'task':      d[2] , d[3]  = 1/np.sqrt(2), -1/np.sqrt(2)   # state 1 pref
    elif name == 'tie':     d[0] , d[1]  = 1/np.sqrt(2), -1/np.sqrt(2)   # state 0 pref
    elif name == 'shift':   d[0] , d[1]  = 1/np.sqrt(2),  1/np.sqrt(2)   # state 0 shift
    elif name == 'value':   d[12 + 2] = 1.0                              # V[2]
    elif name == 'beta':    d[18] = 1.0
    elif name == 'slip':    d[19] = 1.0
    return d

def fdist(p_):
    return float((policy(p_) - pistar).abs().max())

def flow(name, eps=1.0, eta=0.05, steps=8000, thin=25, alpha=ALPHA, base=None):
    b = pstar if base is None else base
    ref = policy(b)
    d = direction(name)
    w = (b + eps * d).clone().requires_grad_(True)
    ts, fs, own, oth = [0.0], [float((policy(w.detach()) - ref).abs().max())], [eps], [0.0]
    for s in range(steps):
        g = torch.autograd.grad(objective(w, alpha), w)[0]
        with torch.no_grad():
            w -= eta * g
        if s % thin == 0:
            dp = w.detach() - b
            ts.append((s + 1) * eta)
            fs.append(float((policy(w.detach()) - ref).abs().max()))
            own.append(float(dp @ d))
            oth.append(float((dp - (dp @ d) * d).norm()))
    return np.array(ts), np.array(fs), np.array(own), np.array(oth)

fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))

# (a) functional healing per block
ax = axes[0, 0]
BLOCKS = [('task', "base policy, strict state", 'C0'),
          ('tie', "identity tie, state 0", 'C2'),
          ('value', "goal value $V[2]$", 'C1'),
          ('beta', r"tilt $\beta$", 'C4'),
          ('slip', r"env model $\widehat{\mathrm{slip}}$", 'C3')]
flows = {}
for name, lab, col in BLOCKS:
    ts, fs, own, oth = flow(name)
    flows[name] = (ts, fs, own, oth)
    ax.plot(ts, np.maximum(fs, 1e-7), color=col, lw=2,
            label=f"{lab}  (f-dist$_0$={fs[0]:.2f})")
    print(f"{name:6s} f0={fs[0]:.3f} fend={fs[-1]:.5f} own_end={own[-1]:+.3f} "
          f"oth_end={oth[-1]:.3f}")
ax.set(yscale='log', ylabel=r"functional distance $\max_n\|\pi_t-\pi^*\|_1$",
       title="(a) gradient flow heals the function, block by block")
ax.legend(fontsize=7)

# (b) scar panel: tie and value blocks, function vs parameters
ax = axes[0, 1]
for name, col, lab in (('tie', 'C2', 'identity tie'), ('value', 'C1', 'goal value')):
    ts, fs, own, oth = flows[name]
    ax.plot(ts, fs / max(fs[0], 1e-9), color=col, lw=2, label=f"{lab}: function")
    ax.plot(ts, np.abs(own), color=col, lw=1.4, ls='--',
            label=f"{lab}: perturbed param")
    ax.plot(ts, oth, color=col, lw=1.4, ls=':', label=f"{lab}: other params")
ax.set(title="(b) function heals; parameters migrate (the weight scar)",
       xlabel=r"training time $\eta\cdot$steps")
ax.legend(fontsize=7)

# (c) REINFORCE wander along explicit directions
def reinforce_grad(p, batch, rng, alpha=ALPHA):
    pi = policy(p).detach().numpy()
    pt = p.clone().requires_grad_(True)
    pit = policy(pt)
    gacc = np.zeros((R, 2))
    for _ in range(batch):
        n, srec, arec, rew = N0, [], [], np.zeros(K + 1)
        for t in range(K):
            ai = 0 if rng.random() < pi[n, 0] else 1
            srec.append(n); arec.append(ai)
            n = rng.choice(R, p=PN[ai, n].numpy())
            rew[t + 1] = (n == G)
        rtg = np.cumsum(rew[::-1])[::-1]
        for t, (s, ai) in enumerate(zip(srec, arec)):
            gl = -pi[s]; gl[ai] += 1
            gacc[s] += gl * rtg[t + 1]
    # chain rule: dJ/dp = sum_{n,a} gacc[n,a]/batch * d logits.. via autograd:
    logits_grad = torch.from_numpy(gacc / batch)
    w_base, V, beta, sliphat = unpack(pt)
    Ph = phat(sliphat)
    Q = torch.stack([Ph[0] @ V, Ph[1] @ V], dim=1)
    Lg = ((w_base + beta * Q) * logits_grad).sum()
    gJ = torch.autograd.grad(Lg, pt, retain_graph=True)[0]
    kl = (pit * (torch.log(pit) - np.log(0.5))).sum()
    gK = torch.autograd.grad(kl, pt)[0]
    return (-gJ + alpha * gK).detach()

rng = np.random.default_rng(0)
w = pstar.clone()
dirs = [('tie', 'C2'), ('task', 'C0'), ('shift', 'C7')]
comps = {n: [0.0] for n, _ in dirs}
SG_STEPS = 3000
for s in range(SG_STEPS):
    g = reinforce_grad(w, 8, rng)
    w = w - 0.02 * g
    dp = w - pstar
    for n, _ in dirs:
        comps[n].append(float(dp @ direction(n)))
ax = axes[1, 0]
for n, col in dirs:
    ax.plot(comps[n], color=col, lw=1.2, label=n)
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.set(xlabel="REINFORCE steps (batch 8)", ylabel="component along direction",
       title="(c) gradient noise lives on the interior (identity) directions")
ax.legend(fontsize=8)

# (d) anchor law, in the NON-REDUNDANT 12-param model (no compensation route):
# lambda_tie(alpha) = d^T H d measured directly, predicted = alpha/2 exactly.
def objective12(w, alpha):
    pi = torch.softmax(w.reshape(R, 2), dim=1)
    M = pi[:, 0, None] * PN[0] + pi[:, 1, None] * PN[1]
    st = torch.zeros(R); st = st.clone(); st[N0] = 1.0
    J = torch.zeros(())
    for _ in range(K):
        st = st @ M
        J = J + st[G]
    kl = (pi * (torch.log(pi) - np.log(0.5))).sum()
    return -J + alpha * kl

def lam_tie12(alpha):
    pass_iters = None
    w = torch.zeros(12, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(6000):
        opt.zero_grad(); L = objective12(w, alpha); L.backward(); opt.step()
    for _ in range(2500):
        g = torch.autograd.grad(objective12(w, alpha), w)[0]
        with torch.no_grad():
            w -= 0.3 * g
    d = torch.zeros(12); d[0], d[1] = 1/np.sqrt(2), -1/np.sqrt(2)
    Hd = torch.autograd.functional.hessian(lambda x: objective12(x, alpha),
                                           w.detach())
    return float(d @ (Hd @ d))

ax = axes[1, 1]
alphas = [0.05, 0.10, 0.20]
lams = []
for al in alphas:
    l = lam_tie12(al)
    lams.append(l)
    print(f"12-param model: alpha={al:.2f}  lambda_tie = {l:.5f}  (alpha/2 = {al/2:.5f})")
ax.plot(alphas, lams, 'o-', color='C2', lw=2, label=r"measured $\lambda_{\mathrm{tie}} = d^\top H d$")
ax.plot(alphas, [al / 2 for al in alphas], 'k--', lw=1, label=r"$\alpha/2$ (predicted)")
ax.set(xlabel=r"anchor coefficient $\alpha$", ylabel=r"identity-tie curvature",
       title=r"(d) without redundancy, the anchor restores identity at $\alpha/2$")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("courts_sgd.pdf"); fig.savefig("courts_sgd.png", dpi=110)
print("saved courts_sgd (final, computed)")
