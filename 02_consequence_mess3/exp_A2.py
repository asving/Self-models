"""Design A v2 — genuine INTERNAL self-action.
Action aₜ = RUNNER-UP of the Mess3 belief (argsort[-2], the model's 2nd-most-likely next Mess3
state). Unlike v1's aₜ=argmax(mess3 pred)=z⁰ₜ (an observed token), the runner-up: varies maximally
(H=1.10), NEVER equals the Mess3 token (a≠z0 by construction) and is at chance vs the asym3 token,
and crucially REQUIRES THE FULL BELIEF ORDERING — the MAP (=the token) alone doesn't determine it,
so there's no single-token shortcut. It corrupts ASYM3: x¹ₜ₊₁=(z¹ₜ₊₁+aₜ)%3. aₜ from clean Mess3
belief, decodes corrupted asym3 → no circularity, non-collapsible (Mess3 belief pinned by predicting
clean Mess3). Because aₜ is a genuine internal signal (not an input token), the desync is clean:
patch aₜ → asym3 mis-decodes. --corrupt_frac 1.0=plain, 0.5=two-trajectory.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT
from factors import mess3_factor, asym3_factor, belief_filter
from exp_A import sample_factor

BASE = os.path.dirname(os.path.abspath(__file__))
Q = 3


def action_from_belief(bel0):
    return np.argsort(bel0, -1)[..., -2]                       # aₜ = runner-up of the mess3 belief


def generate_A2(f0, f1, B, L, rng, corrupt_frac=1.0):
    z0 = sample_factor(f0.T, f0.pi, B, L, rng)                 # mess3 (clean, drives action)
    z1 = sample_factor(f1.T, f1.pi, B, L, rng)                 # asym3 (corrupted)
    a = action_from_belief(belief_filter(f0.T, z0, f0.pi))
    corrupt = (rng.random(B) < corrupt_frac)
    x1 = z1.copy()
    x1[:, 1:] = (z1[:, 1:] + a[:, :-1] * corrupt[:, None]) % Q  # aₜ corrupts asym3 at t+1
    toks = z0 * Q + x1                                          # mess3 clean, asym3 corrupted
    return toks, a, z0, z1, corrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/uni_mess3_asym3"); ap.add_argument("--out", default="runs/expA2")
    ap.add_argument("--corrupt_frac", type=float, default=1.0); ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--eval_every", type=int, default=300)
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
    dt, da, dz0, dz1, _ = generate_A2(f0, f1, 4000, 64, rng, 1.0)
    p = np.bincount(da.ravel(), minlength=Q) / da.size
    print(f"loaded {ck}.pt | A2 internal action (mess3-belief runner-up) | corrupt_frac={args.corrupt_frac}", flush=True)
    print(f"  DEGENERACY: action==z0 frac={(da==dz0).mean():.3f}  action==z1 frac={(da==dz1).mean():.3f}  "
          f"action entropy={-(p*np.log(p+1e-30)).sum():.3f}/{np.log(3):.2f}", flush=True)

    def gen(B): return generate_A2(f0, f1, B, args.L, rng, args.corrupt_frac)

    @torch.no_grad()
    def asym3_nll(toks, corrupt):
        x = torch.from_numpy(toks).to(dev); lg = model(x)
        P = F.softmax(lg, -1).view(x.size(0), args.L, Q, Q); Pz1 = P.sum(2).clamp_min(1e-30)
        x1 = (x % Q); nll = -torch.log(Pz1[:, :-1].gather(-1, x1[:, 1:, None]).squeeze(-1))
        tl = slice(args.L // 2, None); c = torch.from_numpy(corrupt).to(dev)
        return (nll[c][:, tl].mean().item() if c.any() else float("nan"),
                nll[~c][:, tl].mean().item() if (~c).any() else float("nan"))

    pool = {"d": None, "p": 0}
    def nb():
        if pool["d"] is None or pool["p"] + args.batch > pool["d"][0].shape[0]:
            pool["d"] = gen(args.batch * 8); pool["p"] = 0
        s = slice(pool["p"], pool["p"] + args.batch); pool["p"] += args.batch
        return pool["d"][0][s]

    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        x = torch.from_numpy(nb()).to(dev); logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            et, _, _, _, ec = gen(2048); nc, ncl = asym3_nll(et, ec)
            log.append({"step": step, "loss": float(loss.item()), "asym3_nll_corrupt": nc, "asym3_nll_clean": ncl})
            msg = f"step {step:5d} | loss {loss.item():.3f} | asym3 NLL corrupt={nc:.3f}"
            if not np.isnan(ncl): msg += f"  clean={ncl:.3f}"
            print(msg + "  (oracle≈0.88)", flush=True)

    torch.save(model.state_dict(), out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s -> {out}.pt", flush=True)


if __name__ == "__main__":
    main()
