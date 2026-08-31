"""Does the trained net linearly represent the DP's belief state (p, q_hat, kappa)?
For each per-trajectory net (opponent commits to pure BR or pure bias per game -- matching the DP),
roll out closed-loop, capture the residual stream at each step, and compute the EXACT Bayes belief
trajectory the optimal agent would maintain:
  p_t     = P(opponent is a best-responder | history)        -- type posterior (Bayesian)
  q_hat_t = Dirichlet posterior mean over the opponent's bias -- 'which bias'
  kappa_t = legibility-weighted concentration sum_s |P1(p_s)|^2 -- 'how much have I actually learned'
            (gated by the agent's own self-legibility: a uniform action adds ~0).
Then ridge-probe (train/test split) the residual for each -> test R^2 per beta, per layer.
The analyst knows the sampled action a_t, so the realized opponent move b_t=(a_t-o_t)%3 is exact;
the belief is what an agent that recovers its own action could compute. Simplex-style test."""
import os, sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
torch.set_num_threads(8)
from rps_im import RPSNet, GAMMA_BR
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__)); DEV = "cpu"
W = np.exp(2j * np.pi / 3)                                  # cube root of unity for |P1|^2


def resid_forward(net, tok):
    """replicate RPSNet.forward but return per-layer residuals (after each block, and after lnf)."""
    L = tok.shape[1]
    x = net.emb(tok) + net.pos(torch.arange(L))[None]
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    hs = []
    for blk in net.blocks:
        x = blk(x, mask); hs.append(x[:, -1].clone())          # last-pos residual after this block
    xf = net.lnf(x); hs.append(xf[:, -1].clone())              # post-lnf (what heads read)
    return net.act_head(xf[:, -1]), hs


@torch.no_grad()
def rollout_capture(net, B, T, beta, rng):
    """closed-loop per-traj rollout; capture residuals + (a,o,p) + true type/bias per step."""
    is_br = torch.rand(B) < beta
    g = rng.gamma(0.5, 1.0, size=(B, 3)); bias = torch.tensor(g / g.sum(1, keepdims=True), dtype=torch.float32)
    seq = torch.full((B, 1), 3, dtype=torch.long)
    H = [[] for _ in range(net.n_layer + 1)] if hasattr(net, "n_layer") else None
    nlayers = len(net.blocks) + 1
    H = [[] for _ in range(nlayers)]
    A, O, P = [], [], []
    for t in range(T):
        logits, hs = resid_forward(net, seq)
        for i in range(nlayers): H[i].append(hs[i])
        p = F.softmax(logits, -1); P.append(p)
        a = torch.multinomial(p, 1).squeeze(1)
        winprob = p[:, [2, 0, 1]]; br = F.softmax(GAMMA_BR * winprob, -1)
        q = torch.where(is_br[:, None], br, bias)
        b = torch.multinomial(q, 1).squeeze(1)
        o = (a - b) % 3
        A.append(a); O.append(o)
        seq = torch.cat([seq, o[:, None]], 1)
    stack = lambda L: torch.stack(L, 1)                        # (B,T,...)
    H = [torch.stack(h, 1).numpy() for h in H]                 # list of (B,T,d)
    return dict(H=H, a=stack(A).numpy(), o=stack(O).numpy(), p=stack(P).numpy(),
                is_br=is_br.numpy(), bias=bias.numpy())


def belief_traj(roll, beta):
    """exact Bayes belief (p_t, q_hat_t, kappa_t) given history BEFORE step t (aligns with residual_t)."""
    B, T = roll["a"].shape
    a, o, p = roll["a"], roll["o"], roll["p"]
    b = (a - o) % 3                                            # realized opponent move (analyst-known)
    P_br = np.full(B, beta); P_bias = 1 - beta                 # type prior; log-odds accumulation
    logodds = np.log((beta + 1e-9) / (1 - beta + 1e-9)) * np.ones(B)
    counts = np.full((B, 3), 0.5)                              # Dirichlet over bias
    kappa = np.full(B, 0.0)
    pt, qh, kp = [], [], []
    for t in range(T):
        # record belief BEFORE incorporating round-t outcome (this is what residual_t conditions on)
        pt.append(1 / (1 + np.exp(-logodds)))
        qh.append(counts / counts.sum(1, keepdims=True))
        kp.append(kappa.copy())
        # now update with round t's observation b_t
        bt = b[:, t]; pr = p[:, t]
        winprob = pr[:, [2, 0, 1]]; br = np.exp(GAMMA_BR * winprob); br /= br.sum(1, keepdims=True)
        ell_br = br[np.arange(B), bt]                          # P(b_t | BR), uses agent's marginal
        ell_bias = counts[np.arange(B), bt] / counts.sum(1)    # posterior-predictive under bias
        logodds = logodds + np.log(ell_br + 1e-9) - np.log(ell_bias + 1e-9)
        w = np.abs(pr[:, 0] + pr[:, 1] * W + pr[:, 2] * W**2) ** 2   # |P1(p_t)|^2 legibility weight in [0,1]
        counts[np.arange(B), bt] += 1.0
        kappa += w
    return np.stack(pt, 1), np.stack(qh, 1), np.stack(kp, 1)   # (B,T), (B,T,3), (B,T)


def ridge_r2(X, y, lam=10.0, split=0.7):
    n = X.shape[0]; idx = np.arange(n); ntr = int(split * n)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    Xtr1 = np.concatenate([Xtr, np.ones((ntr, 1))], 1); Xte1 = np.concatenate([Xte, np.ones((n - ntr, 1))], 1)
    d = Xtr1.shape[1]; A = Xtr1.T @ Xtr1 + lam * np.eye(d); A[-1, -1] = 0
    Wt = np.linalg.solve(A, Xtr1.T @ ytr)
    pred = Xte1 @ Wt
    ss_res = ((yte - pred) ** 2).sum(0); ss_tot = ((yte - yte.mean(0)) ** 2).sum(0) + 1e-9
    return 1 - ss_res / ss_tot


def main():
    betas = [0.2, 0.3, 0.4, 0.5, 0.8]
    rng = np.random.default_rng(0)
    rows = {}
    for beta in betas:
        f = f"{BASE}/rps_runs/rpstraj_b{beta}.pt"
        if not os.path.exists(f): print("skip", beta); continue
        ck = torch.load(f, map_location=DEV); ar = ck["args"]
        net = RPSNet(ar["d_model"], ar["n_layer"], ar["n_head"], ar["T"]); net.load_state_dict(ck["state"]); net.eval()
        roll = rollout_capture(net, 2500, ar["T"], beta, rng)
        pt, qh, kp = belief_traj(roll, beta)
        nl = len(roll["H"])
        flat = lambda z: z.reshape(-1, *z.shape[2:]) if z.ndim > 2 else z.reshape(-1)
        tcol = np.broadcast_to(np.arange(ar["T"])[None], pt.shape).reshape(-1).astype(float)  # round-index control
        # shuffle once for split consistency
        N = pt.size; perm = rng.permutation(N)
        targs = {"p(BR)": flat(pt)[perm, None], "q_hat": flat(qh)[perm], "kappa": flat(kp)[perm, None],
                 "round_t(ctrl)": tcol[perm, None]}
        layer_r2 = {}
        for li in range(nl):
            X = flat(roll["H"][li])[perm]
            r2 = {k: float(np.mean(ridge_r2(X, v))) for k, v in targs.items()}
            layer_r2[f"L{li}"] = r2
        rows[beta] = layer_r2
        best = layer_r2[f"L{nl-1}"]
        print(f"beta={beta} | post-lnf R^2: p(BR)={best['p(BR)']:.3f}  q_hat={best['q_hat']:.3f}  "
              f"kappa={best['kappa']:.3f}  [round_t ctrl={best['round_t(ctrl)']:.3f}]", flush=True)
    json.dump(rows, open(f"{BASE}/figs/belief_probe.json", "w"), indent=2)

    # figure: R^2 vs beta for the post-lnf layer
    if rows:
        bs = sorted(rows); last = f"L{len(next(iter(rows.values())))-1}"
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        for k, c, mk in [("p(BR)", "#d62728", "o"), ("q_hat", "#2ca02c", "s"),
                         ("kappa", "#9467bd", "^"), ("round_t(ctrl)", "#999999", "x")]:
            ax.plot(bs, [rows[b][last][k] for b in bs], "-" + mk, color=c, lw=2, ms=7,
                    label=k, ls=("--" if "ctrl" in k else "-"))
        ax.set_xlabel(r"$\beta$ = P(opponent is best-responder)"); ax.set_ylabel("probe test $R^2$ (post-lnf residual)")
        ax.set_ylim(-0.05, 1.02); ax.grid(alpha=0.25); ax.legend(fontsize=10)
        ax.set_title("Linear decodability of the optimal belief state from the net", fontsize=12.5)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        fig.savefig(f"{BASE}/figs/belief_probe.png", dpi=130, bbox_inches="tight")
        print("wrote figs/belief_probe.{json,png}")


if __name__ == "__main__":
    main()
