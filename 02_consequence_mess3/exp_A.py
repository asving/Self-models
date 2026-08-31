"""Design A: the model's own action corrupts a future observation, forcing an efference copy.
Action aₜ = argmax(next-Mess3-token prediction) = the token the model would emit (non-collapsible:
pinned by the Mess3 loss; genuinely the model's output). It corrupts the asym3 observation:
x¹ₜ₊₁ = (z¹ₜ₊₁ + aₜ) mod 3. aₜ comes from Mess3 (clean) and decodes asym3 (corrupted) -> no
circularity. To track asym3 the model MUST route its own action forward: z¹ = (x¹ − aₜ) mod 3.
All three requirements hold (varying, accessible=self-recomputable, necessary), and the action is
self-generated -> a genuine self-model. --corrupt_frac 1.0 = plain A; 0.5 = two-trajectory
(half the sequences uncorrupted, type hidden -> the model must infer in-context whether its
action is consequential and conditionally apply the efference copy).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT  # noqa
from factors import mess3_factor, asym3_factor, belief_filter  # noqa

BASE = os.path.dirname(os.path.abspath(__file__))
Q = 3


def sample_factor(T, pi, B, L, rng):
    I = T.shape[1]; flat = T.transpose(1, 0, 2).reshape(I, Q * I)
    s = rng.choice(I, size=B, p=pi); z = np.empty((B, L), np.int64)
    for t in range(L):
        cdf = np.cumsum(flat[s], 1); idx = (rng.random(B)[:, None] < cdf).argmax(1)
        z[:, t] = idx // I; s = idx % I
    return z


def generate_A(f0, f1, B, L, rng, corrupt_frac=1.0):
    """f0=mess3, f1=asym3. aₜ=argmax P(next mess3 token | mess3 0..t). x¹ₜ₊₁=(z¹ₜ₊₁+aₜ)%3."""
    z0 = sample_factor(f0.T, f0.pi, B, L, rng)
    z1 = sample_factor(f1.T, f1.pi, B, L, rng)
    bel0 = belief_filter(f0.T, z0, f0.pi)                      # (B,L,3) state emitting token t+1
    a = (bel0 @ f0.E).argmax(-1)                                # (B,L) aₜ = predicted next mess3 token
    corrupt = (rng.random(B) < corrupt_frac)
    x1 = z1.copy()
    x1[:, 1:] = (z1[:, 1:] + a[:, :-1] * corrupt[:, None]) % Q   # aₜ corrupts asym3 at t+1
    toks = z0 * Q + x1
    return toks, a, z1, corrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/uni_mess3_asym3")
    ap.add_argument("--out", default="runs/expA")
    ap.add_argument("--corrupt_frac", type=float, default=1.0)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    ck = os.path.join(BASE, args.ckpt); out = os.path.join(BASE, args.out)
    meta = json.load(open(ck + ".json")); a = meta["args"]; V = meta["V"]
    f0, f1 = mess3_factor(0.6, 0.15), asym3_factor()
    model = GPT(V, a["d_model"], a["n_layer"], a["n_head"], max_len=args.L).to(dev)
    model.load_state_dict(torch.load(ck + ".pt", map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"loaded {ck}.pt | design A | corrupt_frac={args.corrupt_frac}", flush=True)
    a_ent = None

    def gen(B): return generate_A(f0, f1, B, args.L, rng, args.corrupt_frac)

    @torch.no_grad()
    def asym3_nll(toks, corrupt):
        x = torch.from_numpy(toks).to(dev); lg = model(x)
        P = F.softmax(lg, -1).view(x.size(0), args.L, Q, Q); Pz1 = P.sum(2).clamp_min(1e-30)
        x1 = (x % Q)
        nll = -torch.log(Pz1[:, :-1].gather(-1, x1[:, 1:, None]).squeeze(-1))   # NLL of observed asym3 sub-token
        tail = slice(args.L // 2, None)
        c = torch.from_numpy(corrupt).to(dev)
        nc = nll[c][:, tail].mean().item() if c.any() else float("nan")
        ncl = nll[~c][:, tail].mean().item() if (~c).any() else float("nan")
        return nc, ncl

    pool = {"d": None, "p": 0}
    def nb():
        if pool["d"] is None or pool["p"] + args.batch > pool["d"][0].shape[0]:
            pool["d"] = gen(args.batch * 8); pool["p"] = 0
        s = slice(pool["p"], pool["p"] + args.batch); pool["p"] += args.batch
        t, av, z1, c = pool["d"]; return t[s]

    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        x = torch.from_numpy(nb()).to(dev)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            et, av, z1, ec = gen(2048)
            if a_ent is None:
                p = np.bincount(av.ravel(), minlength=Q) / av.size
                a_ent = float(-(p * np.log(np.clip(p, 1e-30, None))).sum())
            nc, ncl = asym3_nll(et, ec)
            log.append({"step": step, "loss": float(loss.item()),
                        "asym3_nll_corrupt": nc, "asym3_nll_clean": ncl, "action_entropy": a_ent})
            msg = f"step {step:5d} | loss {loss.item():.3f} | asym3 NLL corrupt={nc:.3f}"
            if not np.isnan(ncl): msg += f"  clean={ncl:.3f}"
            print(msg + f"  (action H={a_ent:.2f}; oracle asym3 NLL≈0.88)", flush=True)

    torch.save(model.state_dict(), out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s -> {out}.pt", flush=True)


if __name__ == "__main__":
    main()
