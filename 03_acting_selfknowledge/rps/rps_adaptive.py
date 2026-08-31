"""Adaptive (per-game) heuristic for RPS imperfect-monitoring, + empirical check that the net does it.
PART A: a SENSE-THEN-DECIDE agent. It plays SHARP for m rounds to estimate the opponent's bias (paying
the best-responder cost while sharp), then DECIDES per game: if its estimated exploit advantage beats the
cost it keeps exploiting (sharp), else it folds to uniform (hide). Per-game decision => mean entropy is an
interior MIXTURE of exploit(sharp) and hide(uniform) games => should reproduce the SMOOTH band that the
single-fixed-sharpness heuristic (bang-bang) missed.
PART B: roll out a trained near-threshold net and test whether its per-game entropy DROPS on strong-bias
games (the signature of the same per-game adaptiveness)."""
import os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
torch.set_num_threads(4)
from rps_im import RPSNet, rollout as net_rollout
rng = np.random.default_rng(0)
GAMMA = 6.0; PRIOR = 0.5; T = 40
SH = 0.97; H_SH = -(SH*np.log(SH) + (1-SH)*np.log((1-SH)/2)); H_UN = np.log(3)  # sharp/uniform entropies


def payoff(a, b): d = (a - b) % 3; return np.where(d == 1, 1.0, np.where(d == 2, -1.0, 0.0))


def sim_adaptive(beta, m, tau, N=4000):
    bias = rng.dirichlet([0.5, 0.5, 0.5], size=N)
    counts = np.full((N, 3), PRIOR); exploit = np.zeros(N, bool); decided = False
    pay = 0.0; ent = 0.0; frac_hide = 0.0
    for t in range(T):
        qhat = counts / counts.sum(1, keepdims=True)
        fav = qhat.argmax(1); mv = (fav + 1) % 3                     # counter to estimated favored move
        adv = qhat[np.arange(N), fav] - qhat[np.arange(N), (fav + 2) % 3]
        if t == m:                                                  # DECIDE per game
            exploit = ((1 - beta) * adv > tau); decided = True; frac_hide = float((~exploit).mean())
        sharp = np.ones(N, bool) if t < m else exploit              # sense(sharp) -> exploit(sharp) | hide(unif)
        s = np.where(sharp, SH, 1 / 3)
        p = np.full((N, 3), 0.0); off = (1 - s) / 2
        for j in range(3): p[:, j] = np.where(mv == j, s, off)
        a = (p.cumsum(1) > rng.random((N, 1))).argmax(1)
        win = p[:, [2, 0, 1]]; br = np.exp(GAMMA * win); br /= br.sum(1, keepdims=True)
        q = (1 - beta) * bias + beta * br
        b = (q.cumsum(1) > rng.random((N, 1))).argmax(1)
        o = (a - b) % 3; pay += payoff(a, b).mean()
        ent += np.where(sharp, H_SH, H_UN).mean()
        bhat = (mv - o) % 3; counts[np.arange(N), bhat] += 1
    return pay / T, ent / T, frac_hide


@torch.no_grad()
def net_per_game(name, N=3000):
    ck = torch.load(os.path.expanduser(f"~/self-models/rps_runs/{name}.pt"), map_location="cpu")
    a = ck["args"]; net = RPSNet(a["d_model"], a["n_layer"], a["n_head"], a["T"]); net.load_state_dict(ck["state"]); net.eval()
    g = rng.gamma(0.5, 1.0, size=(N, 3)); bias = torch.tensor(g / g.sum(1, keepdims=True), dtype=torch.float32)
    _, _, _, ent = net_rollout(net, N, a["T"], "cpu", a["beta"], bias)   # per_traj=False (per-turn mix)
    ent_g = ent.mean(1).numpy()                                     # per-game mean entropy
    delta = (bias.max(1).values - bias.min(1).values).numpy()       # per-game bias strength
    print(f"  {name} (beta={a['beta']}): per-game entropy vs bias-strength delta")
    edges = np.quantile(delta, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        msk = (delta >= edges[i]) & (delta <= edges[i + 1])
        print(f"    delta in [{edges[i]:.2f},{edges[i+1]:.2f}]: mean entropy = {ent_g[msk].mean():.2f}  (n={msk.sum()})")
    # correlation
    print(f"    corr(entropy, delta) = {np.corrcoef(ent_g, delta)[0,1]:+.2f}  (negative => sharper on stronger biases = per-game adaptive)")


if __name__ == "__main__":
    print("=== PART A: adaptive sense-then-decide heuristic — optimal mean entropy vs beta ===")
    emp = {0.0: 0.07, 0.2: 0.15, 0.26: 0.31, 0.3: 0.47, 0.34: 1.10, 0.5: 1.10}
    print("beta | best(m,tau) | payoff | mean-entropy | frac-hide | [empirical ent]")
    for beta in [0.0, 0.2, 0.26, 0.3, 0.34, 0.5, 1.0]:
        best = (-9, None, None, None)
        for m in [0, 1, 2, 3, 5, 8, 12]:
            for tau in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
                pay, ent, fh = sim_adaptive(beta, m, tau)
                if pay > best[0]: best = (pay, ent, m, tau, fh)
        pay, ent, m, tau, fh = best
        print(f"{beta:.2f} |  m={m},t={tau:.2f}  | {pay:+.3f} |    {ent:.2f}     |   {fh:.2f}    |   {emp.get(beta,float('nan')):.2f}")
    print("\n=== PART B: does the trained net do per-game adaptiveness? ===")
    for name in ["rpsfine_b0.26", "rpsfine_b0.3", "rps_b0.2"]:
        if os.path.exists(os.path.expanduser(f"~/self-models/rps_runs/{name}.pt")): net_per_game(name)
