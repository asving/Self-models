"""Pure non-abelian path-integration PROBE (no control, no policy, no efference).

Given a stream of per-step rotation increments X_t (axis-angle vectors), the net must REPORT the
integrated body-forward direction Y_t = (prod_{k<=t} exp(X_k)) @ x_hat in S^2. This isolates the
INTEGRATION question from the control-recursion + capacity confounds of the closed-loop agent:
single forward pass, increments given as input.

group='so2' confines X to the z-axis -> the product is an abelian cumulative SUM -> a transformer
computes it with ONE attention layer -> predicted depth-FLAT. group='so3' -> non-commuting product ->
no abelian shortcut; the abelian / Magnus-2 reconstructions are reported as reference floors, and the
prediction is a depth staircase (error falls past abelian, past Magnus-2, toward 0, saturating ~log2 L).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from so3agent import so3_exp, AgentSO3

BASE = os.path.dirname(os.path.abspath(__file__))


def gen_batch(B, L, dev, group, mag, rng):
    """X (B,L,3) per-step increments; norms ~ U(0.2,0.6)*mag, random axis. Returns X and targets Y."""
    norms = torch.tensor(rng.uniform(0.2, 0.6, size=(B, L)) * mag, device=dev, dtype=torch.float32)
    if group == "so2":
        X = torch.zeros(B, L, 3, device=dev); X[..., 2] = norms * torch.tensor(
            rng.choice([-1.0, 1.0], size=(B, L)), device=dev, dtype=torch.float32)
    else:
        ax = torch.tensor(rng.standard_normal((B, L, 3)), device=dev, dtype=torch.float32)
        ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        X = ax * norms[..., None]
    # exact targets: running product applied to x_hat
    R = torch.eye(3, device=dev).expand(B, 3, 3).contiguous(); Y = []
    for t in range(L):
        R = so3_exp(X[:, t]) @ R; Y.append(R[:, :, 0])
    return X, torch.stack(Y, 1)                                # X (B,L,3), Y (B,L,3) unit


def baselines(X):
    """abelian (exp of cumsum) and Magnus-2 reconstructions of body-forward, for reference."""
    S = torch.cumsum(X, 1)                                     # prefix sums
    ab = torch.stack([so3_exp(S[:, t])[:, :, 0] for t in range(X.shape[1])], 1)
    corr = torch.zeros_like(X[:, 0]); Sp = torch.zeros_like(X[:, 0]); m2 = []
    for t in range(X.shape[1]):
        corr = corr + 0.5 * torch.cross(X[:, t], Sp, dim=-1); Sp = Sp + X[:, t]
        m2.append(so3_exp(Sp + corr)[:, :, 0])
    return ab, torch.stack(m2, 1)


def ang(a, b):
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.rad2deg(torch.arccos((a * b).sum(-1).clamp(-1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--group", default="so3", choices=["so2", "so3"])
    ap.add_argument("--d_model", type=int, default=64); ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4); ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--mag", type=float, default=1.0); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    net = AgentSO3(3, 3, args.d_model, args.n_layer, args.n_head, args.L).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    nparams = sum(p.numel() for p in net.parameters())

    # fixed eval set + analytic baselines (tail = second half, where integration is hardest)
    Xe, Ye = gen_batch(2048, args.L, dev, args.group, args.mag, np.random.default_rng(999))
    with torch.no_grad():
        ab, m2 = baselines(Xe); tail = slice(args.L // 2, None)
        ab_err = ang(ab, Ye)[:, tail].mean().item(); m2_err = ang(m2, Ye)[:, tail].mean().item()
    print(f"{args.group} {args.n_layer}L d{args.d_model} mag{args.mag} L{args.L} | params={nparams/1e3:.0f}K | "
          f"BASELINE tail err: abelian={ab_err:.1f} deg, Magnus2={m2_err:.1f} deg", flush=True)

    @torch.no_grad()
    def evaluate():
        net.eval(); pred = net(Xe); e = ang(pred, Ye)[:, tail].mean().item(); net.train(); return e

    log, t0 = [], time.time()
    os.makedirs(os.path.join(BASE, os.path.dirname(args.out)), exist_ok=True)
    for step in range(1, args.steps + 1):
        net.train(); X, Y = gen_batch(args.batch, args.L, dev, args.group, args.mag, rng)
        pred = net(X)
        predn = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        loss = (1 - (predn * Y).sum(-1)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == 1:
            ev = evaluate(); log.append(dict(step=step, eval_deg=ev, train_loss=float(loss)))
            print(f"step {step:5d} | train_loss {loss:.4f} | eval_tail_deg {ev:5.1f}", flush=True)
    out = os.path.join(BASE, args.out)
    torch.save({"state": net.state_dict(), "args": vars(args), "nparams": nparams,
                "abelian_deg": ab_err, "magnus2_deg": m2_err}, out + ".pt")
    json.dump(dict(args=vars(args), nparams=nparams, abelian_deg=ab_err, magnus2_deg=m2_err, log=log),
              open(out + ".json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s -> {out}.pt (final {log[-1]['eval_deg']:.1f} deg; "
          f"abelian {ab_err:.1f}, Magnus2 {m2_err:.1f})", flush=True)


if __name__ == "__main__":
    main()
