"""Design B + RL — restore the policy-gradient term that free prediction omits.

Same closed loop as exp_B (asym3 reseeded every k steps from the model's belief readout q1),
but the objective adds a REINFORCE term on the seed choice:

    total = CE(next-token)               # term (1): keep the predictor calibrated
          + beta * policy_loss           # term (2): reward seeds with predictable continuations
    policy_loss = - mean[ (R - baseline) * log pi(s_hat) ]
    R = mean log P(self-seeded asym3 sub-tokens over the next k steps)   # -surprise, detached

pi = the SAME belief readout used to sample the seed (frozen-probe clip-normalize), so the
REINFORCE estimator is on-policy. By causal masking, the residual at position t-1 in a single
full-sequence forward equals the readout that seeded the reset at t -> one grad pass gives both
the CE loss and all log pi(s_hat).

Prediction vs exp_B (free): now H(q1) for asym3 should COLLAPSE toward 0 and mass -> predictable
state 0, while the Mess3 control stays faithful.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT  # noqa: E402
from factors import mess3_factor, asym3_factor  # noqa: E402
import exp_B  # reuse generate_closedloop / measure / consequence  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
Q = 3


def t_readout(r, W, b, eps=1e-6):
    """Torch version of the frozen-probe belief readout (clip-normalize) — matches numpy path."""
    q = torch.clamp(r @ W + b, min=eps)
    return q / q.sum(-1, keepdim=True)


def generate_rl(model, facs, probes, k, B, L, rng, device, reseed=1):
    """Closed-loop generation recording the sampled seeds. Returns toks (B,L), rpos, seeds (B,R)."""
    N = len(facs)
    flats = [f.T.transpose(1, 0, 2).reshape(Q, Q * Q) for f in facs]
    states = [rng.choice(Q, size=B, p=f.pi) for f in facs]
    toks = np.zeros((B, L), dtype=np.int64)
    powers = Q ** np.arange(N - 1, -1, -1)
    W, b, ell = probes[f"W{reseed}"], probes[f"b{reseed}"], int(probes[f"ell{reseed}"])
    rpos, seeds = [], []
    for t in range(L):
        if t > 0 and t % k == 0:
            q = exp_B.belief_readout(model, toks[:, :t], ell, W, b, device)   # (B,Q) numpy
            s_hat = (rng.random(B)[:, None] < np.cumsum(q, 1)).argmax(1)
            states[reseed] = s_hat
            rpos.append(t); seeds.append(s_hat)
        z = np.empty((B, N), dtype=np.int64); nxt = np.empty((B, N), dtype=np.int64)
        for n in range(N):
            cdf = np.cumsum(flats[n][states[n]], axis=1)
            idx = (rng.random(B)[:, None] < cdf).argmax(1)
            z[:, n] = idx // Q; nxt[:, n] = idx % Q
        toks[:, t] = (z * powers[None, :]).sum(1)
        states = [nxt[:, n] for n in range(N)]
    return toks, rpos, (np.stack(seeds, 1) if seeds else np.zeros((B, 0), int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/uni_mess3_asym3")
    ap.add_argument("--out", default="runs/expB_rl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=3.0)   # RL term weight
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    ck = os.path.join(BASE, args.ckpt); out = os.path.join(BASE, args.out)

    meta = json.load(open(ck + ".json")); a = meta["args"]; V = meta["V"]
    facs = [mess3_factor(0.6, 0.15), asym3_factor()]
    npz = np.load(ck + "_probes.npz"); probes = {kk: npz[kk] for kk in npz.files}
    ell = int(probes["ell1"])
    Wt = torch.tensor(probes["W1"], dtype=torch.float32, device=dev)
    bt = torch.tensor(probes["b1"], dtype=torch.float32, device=dev)
    model = GPT(V, a["d_model"], a["n_layer"], a["n_head"], max_len=args.L).to(dev)
    model.load_state_dict(torch.load(ck + ".pt", map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"loaded {ck}.pt  beta={args.beta} k={args.k}  asym3 probe layer {ell} R2={float(probes['r21']):.3f}", flush=True)

    traj = []
    m0, qs0 = exp_B.measure(model, facs, probes, args.k, args.L, rng, dev)
    m0["step"] = 0; traj.append(m0)
    print(f"step    0 | H1={m0['H1_tail']:.3f} H0={m0['H0_tail']:.3f} mass1_s0={m0['mass1_s0_tail']:.3f}", flush=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        toks, rpos, seeds = generate_rl(model, facs, probes, args.k, args.batch, args.L, rng, dev)
        x = torch.from_numpy(toks).to(dev)
        seeds_t = torch.from_numpy(seeds).to(dev)
        logits, hid = model(x, return_hidden=True)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        # reward = mean log P(actual asym3 sub-token) over the self-seeded continuation (detached)
        P = F.softmax(logits, -1).view(args.batch, args.L, Q, Q)         # [b,p,z0,z1]
        Pz1 = P.sum(2).clamp_min(1e-30)                                   # (B,L,Q) P(next z1)
        z1 = (x % Q)
        lp_next = torch.log(Pz1[:, :-1].gather(-1, z1[:, 1:, None]).squeeze(-1))  # (B,L-1): idx p -> token p+1
        rewards, logpis = [], []
        for r, t in enumerate(rpos):
            cols = [j - 1 for j in range(t, min(t + args.k, args.L)) if 1 <= j <= args.L - 1]
            rewards.append(lp_next[:, cols].mean(1))                      # (B,)
            pi = t_readout(hid[ell][:, t - 1, :], Wt, bt)                 # (B,Q) with grad
            logpis.append(torch.log(pi.gather(-1, seeds_t[:, r:r + 1]).squeeze(-1).clamp_min(1e-30)))
        R = torch.stack(rewards, 1).detach()                              # (B,Rn)
        logpi = torch.stack(logpis, 1)                                    # (B,Rn) grad
        A = R - R.mean(0, keepdim=True)                                   # batch baseline per reseed
        policy_loss = -(A * logpi).mean()
        loss = ce + args.beta * policy_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0:
            m, _ = exp_B.measure(model, facs, probes, args.k, args.L, rng, dev)
            m.update(step=step, ce=float(ce.item()), reward=float(R.mean().item()),
                     policy_loss=float(policy_loss.item()))
            traj.append(m)
            print(f"step {step:5d} | ce {ce.item():.3f} reward {R.mean().item():.3f} | "
                  f"H1={m['H1_tail']:.3f} H0={m['H0_tail']:.3f} mass1_s0={m['mass1_s0_tail']:.3f}", flush=True)

    mf, qsf = exp_B.measure(model, facs, probes, args.k, args.L, rng, dev)
    tv, _ = exp_B.consequence(model, facs, probes, args.k, args.L, rng, dev)
    print(f"\nfinal | H1={mf['H1_tail']:.3f} (from {m0['H1_tail']:.3f}) | "
          f"mass1_s0={mf['mass1_s0_tail']:.3f} (from {m0['mass1_s0_tail']:.3f}) | "
          f"H0={mf['H0_tail']:.3f} | TV={tv:.3f}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    steps = [t["step"] for t in traj]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(steps, [t["H1_tail"] for t in traj], "-o", label="H(q1) asym3 (consequential)")
    ax[0].plot(steps, [t["H0_tail"] for t in traj], "-s", label="H(q0) mess3 (control)")
    ax[0].axhline(np.log(3), ls=":", c="gray", lw=.8); ax[0].set_xlabel("step"); ax[0].set_ylabel("belief entropy (nats)")
    ax[0].legend(); ax[0].set_title(f"B+RL (beta={args.beta}): collapse?")
    ax[1].plot(steps, [t["mass1_s0_tail"] for t in traj], "-o", c="C2"); ax[1].axhline(1/3, ls=":", c="gray")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("mass q1 -> asym3 state0"); ax[1].set_title("collapse direction"); ax[1].set_ylim(0, 1.05)
    def bary(p): return p @ np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    for q, lab, mk in [(qs0[1], "pre", "."), (qsf[1], "post", "x")]:
        xy = bary(q[:, q.shape[1]//2:, :].reshape(-1, 3)[::7])
        ax[2].scatter(xy[:, 0], xy[:, 1], s=4, alpha=.4, marker=mk, label=lab)
    ax[2].plot([0, 1, .5, 0], [0, 0, np.sqrt(3)/2, 0], "k-", lw=.5); ax[2].set_title("asym3 belief cloud"); ax[2].axis("equal"); ax[2].axis("off"); ax[2].legend()
    fig.tight_layout(); fig.savefig(out + ".png", dpi=110)
    json.dump(dict(args=vars(args), traj=traj, final=mf, m0=m0, consequence_tv=tv), open(out + ".json", "w"), indent=2)
    torch.save(model.state_dict(), out + ".pt")
    print(f"done in {time.time()-t0:.0f}s -> {out}.png/.json", flush=True)


if __name__ == "__main__":
    main()
