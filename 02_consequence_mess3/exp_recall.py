"""Delayed-recall test of requirement (3) — NECESSITY.
Factor 1 is a recall process: after a reseed to random v at a block start, emit k-1 uniform-random
tokens (zero info about v), then emit v itself (the recall) at the block end. v is injected (side-
channel) at the block start. The intervening tokens are pure noise, so the ONLY way to predict the
recall is to hold+route v. Contrast with asym3 (emissions revealed the state -> routing redundant).
If recall NLL -> 0, the model routed the injected self-signal into its belief. Factor 0 = mess3 (control).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from factors import mess3_factor  # noqa
from exp_inject import GPTInject, build_seed_emb  # noqa

BASE = os.path.dirname(os.path.abspath(__file__))
Q = 3


def generate_recall(f0, k, B, L, rng):
    """factor0=mess3 (normal). factor1=recall: random fill + recall(v) at block end; v injected at block start."""
    flat0 = f0.T.transpose(1, 0, 2).reshape(Q, Q * Q)
    s0 = rng.choice(Q, size=B, p=f0.pi)
    blocks = [(bs, bs + k - 1) for bs in range(0, L, k) if bs + k - 1 < L]
    inj_pos = [bs for bs, _ in blocks]
    recall_pos = [rp for _, rp in blocks]
    seeds = np.zeros((B, len(blocks)), dtype=np.int64)
    z1 = rng.integers(0, Q, size=(B, L))                       # uniform-random fill
    for bi, (bs, rp) in enumerate(blocks):
        v = rng.integers(0, Q, size=B); seeds[:, bi] = v; z1[:, rp] = v   # recall = v
    toks = np.zeros((B, L), dtype=np.int64)
    for t in range(L):
        cdf = np.cumsum(flat0[s0], axis=1); idx = (rng.random(B)[:, None] < cdf).argmax(1)
        z0 = idx // Q; s0 = idx % Q
        toks[:, t] = z0 * Q + z1[:, t]
    return toks, inj_pos, seeds, recall_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/uni_mess3_asym3")
    ap.add_argument("--out", default="runs/expRecall")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    ck = os.path.join(BASE, args.ckpt); out = os.path.join(BASE, args.out)
    meta = json.load(open(ck + ".json")); a = meta["args"]; V = meta["V"]; d = a["d_model"]
    f0 = mess3_factor(0.6, 0.15)

    g = np.random.default_rng(123).standard_normal((Q, d))
    ev = torch.tensor(np.linalg.qr(g.T)[0].T[:Q] * args.scale, dtype=torch.float32, device=dev)
    model = GPTInject(V, d, a["n_layer"], a["n_head"], max_len=args.L).to(dev)
    model.load_state_dict(torch.load(ck + ".pt", map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"loaded {ck}.pt | delayed-recall | scale={args.scale} k={args.k} (chance recall NLL=ln3={np.log(3):.3f})", flush=True)

    def gen(B): return generate_recall(f0, args.k, B, args.L, rng)

    @torch.no_grad()
    def recall_nll(inject):
        toks, inj_pos, seeds, rec_pos = gen(1024)
        x = torch.from_numpy(toks).to(dev)
        se = build_seed_emb(seeds, inj_pos, ev, args.L, dev) if inject else None
        lg = model(x, seed_emb=se)
        P = F.softmax(lg, -1).view(x.size(0), args.L, Q, Q); Pz1 = P.sum(2).clamp_min(1e-30)  # recall sub-token = z1
        # recall token at rec_pos predicted at rec_pos-1; true value = seeds
        pred_pos = [rp - 1 for rp in rec_pos if rp >= 1]
        vals = seeds[:, [i for i, rp in enumerate(rec_pos) if rp >= 1]]
        lp = torch.log(Pz1[:, pred_pos, :])                    # (B, nblocks, Q)
        nll = -lp.gather(-1, torch.from_numpy(vals).to(dev)[:, :, None]).squeeze(-1)
        return float(nll.mean())

    pool = {"d": None, "p": 0}
    def nb():
        if pool["d"] is None or pool["p"] + args.batch > pool["d"][0].shape[0]:
            pool["d"] = gen(args.batch * 8); pool["p"] = 0
        s = slice(pool["p"], pool["p"] + args.batch); pool["p"] += args.batch
        t, ip, sd, _ = pool["d"]; return t[s], ip, sd[s]

    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        toks, inj_pos, seeds = nb()
        x = torch.from_numpy(toks).to(dev)
        se = build_seed_emb(seeds, inj_pos, ev, args.L, dev)
        logits = model(x, seed_emb=se)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            ri, rn = recall_nll(True), recall_nll(False)
            log.append({"step": step, "loss": float(loss.item()), "recall_nll_inject": ri, "recall_nll_noinject": rn})
            print(f"step {step:5d} | loss {loss.item():.3f} | recall NLL: injected={ri:.3f}  no-inject={rn:.3f}  "
                  f"(0=perfect route, {np.log(3):.2f}=chance)", flush=True)

    np.savez(out + "_ev.npz", ev=ev.cpu().numpy(), scale=args.scale, k=args.k)
    torch.save(model.state_dict(), out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s -> {out}.pt", flush=True)


if __name__ == "__main__":
    main()
