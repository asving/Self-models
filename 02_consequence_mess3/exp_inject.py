"""Injected-seed B variant. Break the catch-22 by making the seed an EXTERNAL, varying, accessible
signal (not a sample of the belief): every k steps reset asym3 to a random v, and inject a fixed
per-state embedding e_v into the residual (a readable 'token' side-channel) at the position that
predicts the first post-reset token. v varies, is useful (it IS the seed -> determines the post-
reset emissions), and is accessible+early (injected at the embedding -> can propagate). Question:
does the model learn to read v and ROUTE it forward into its belief/prediction?
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT  # noqa
from factors import mess3_factor, asym3_factor  # noqa

BASE = os.path.dirname(os.path.abspath(__file__))
Q = 3


class GPTInject(GPT):
    """GPT that adds an external seed_emb (B,L,d) to the input embedding (a side-channel token)."""
    def forward(self, idx, seed_emb=None, return_hidden=False):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        if seed_emb is not None:
            x = x + seed_emb
        mask = torch.triu(torch.ones(L, L, device=idx.device, dtype=torch.bool), 1)
        hid = []
        for blk in self.blocks:
            x = blk(x, mask)
            if return_hidden:
                hid.append(x)
        x = self.lnf(x)
        logits = self.head(x)
        return (logits, hid) if return_hidden else logits


def generate_injected(facs, k, B, L, rng):
    """asym3 reset to a random v every k steps. Returns toks (B,L), inject positions (list),
    and seeds (B, n_inject) — v injected at position (reseed_step-1)."""
    N = 2
    flats = [f.T.transpose(1, 0, 2).reshape(Q, Q * Q) for f in facs]
    states = [rng.choice(Q, size=B, p=f.pi) for f in facs]
    toks = np.zeros((B, L), dtype=np.int64)
    powers = Q ** np.arange(N - 1, -1, -1)
    reseed_steps = list(range(k, L, k))
    inj_pos = [t - 1 for t in reseed_steps]
    seeds = np.zeros((B, len(reseed_steps)), dtype=np.int64)
    for t in range(L):
        if t in reseed_steps:
            v = rng.integers(0, Q, size=B)
            states[1] = v                                   # reset asym3 to random v
            seeds[:, reseed_steps.index(t)] = v
        z = np.empty((B, N), dtype=np.int64); nxt = np.empty((B, N), dtype=np.int64)
        for n in range(N):
            cdf = np.cumsum(flats[n][states[n]], axis=1)
            idx = (rng.random(B)[:, None] < cdf).argmax(1)
            z[:, n] = idx // Q; nxt[:, n] = idx % Q
        toks[:, t] = (z * powers[None, :]).sum(1)
        states = [nxt[:, n] for n in range(N)]
    return toks, inj_pos, seeds


def build_seed_emb(seeds, inj_pos, ev, L, dev):
    """seed_emb (B,L,d): add ev[v] at each inject position; 0 elsewhere."""
    B = seeds.shape[0]; d = ev.shape[1]
    se = torch.zeros(B, L, d, device=dev)
    for i, p in enumerate(inj_pos):
        se[:, p, :] = ev[torch.from_numpy(seeds[:, i]).to(dev)]
    return se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/uni_mess3_asym3")
    ap.add_argument("--out", default="runs/expInject")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--scale", type=float, default=1.0)     # magnitude of the seed embedding
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    ck = os.path.join(BASE, args.ckpt); out = os.path.join(BASE, args.out)
    meta = json.load(open(ck + ".json")); a = meta["args"]; V = meta["V"]; d = a["d_model"]
    facs = [mess3_factor(0.6, 0.15), asym3_factor()]

    # fixed orthonormal per-state seed embeddings (the side-channel 'token')
    g = np.random.default_rng(123).standard_normal((Q, d))
    g = np.linalg.qr(g.T)[0].T[:Q]                          # orthonormal rows
    ev = torch.tensor(g * args.scale, dtype=torch.float32, device=dev)

    model = GPTInject(V, d, a["n_layer"], a["n_head"], max_len=args.L).to(dev)
    model.load_state_dict(torch.load(ck + ".pt", map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"loaded {ck}.pt | injected-seed B | scale={args.scale} k={args.k}", flush=True)

    def make_pool():
        toks, inj_pos, seeds = generate_injected(facs, args.k, args.batch * 8, args.L, rng)
        return toks, inj_pos, seeds
    pool = {"t": None, "p": 0}

    def next_batch():
        if pool["t"] is None or pool["p"] + args.batch > pool["t"][0].shape[0]:
            pool["t"] = make_pool(); pool["p"] = 0
        s = slice(pool["p"], pool["p"] + args.batch); pool["p"] += args.batch
        toks, inj_pos, seeds = pool["t"]
        return toks[s], inj_pos, seeds[s]

    @torch.no_grad()
    def eval_loss(inject=True):
        toks, inj_pos, seeds = generate_injected(facs, args.k, 1024, args.L, rng)
        x = torch.from_numpy(toks).to(dev)
        se = build_seed_emb(seeds, inj_pos, ev, args.L, dev) if inject else None
        lg = model(x, seed_emb=se)
        # asym3 sub-token loss on the post-reset rollout positions
        roll = [min(t + o, args.L - 1) for t in [p + 1 for p in inj_pos] for o in range(args.k)]
        roll = [p for p in roll if p >= 1]
        P = F.softmax(lg, -1).view(x.size(0), args.L, Q, Q); Pz1 = P.sum(2).clamp_min(1e-30)
        z1 = (x % Q)
        nll = -torch.log(Pz1[:, :-1].gather(-1, z1[:, 1:, None]).squeeze(-1))   # (B,L-1) idx p->tok p+1
        cols = [p - 1 for p in roll if 1 <= p <= args.L - 1]
        return float(nll[:, cols].mean())

    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        toks, inj_pos, seeds = next_batch()
        x = torch.from_numpy(toks).to(dev)
        se = build_seed_emb(seeds, inj_pos, ev, args.L, dev)
        logits = model(x, seed_emb=se)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            l_inj = eval_loss(inject=True); l_no = eval_loss(inject=False)
            log.append({"step": step, "loss": float(loss.item()),
                        "asym3_rollout_nll_injected": l_inj, "asym3_rollout_nll_noinject": l_no})
            print(f"step {step:5d} | loss {loss.item():.3f} | asym3 rollout NLL: "
                  f"injected={l_inj:.3f}  no-inject={l_no:.3f}  (gap={l_no-l_inj:+.3f} = value of v)", flush=True)

    np.savez(out + "_ev.npz", ev=ev.cpu().numpy(), scale=args.scale, k=args.k)
    torch.save(model.state_dict(), out + ".pt")
    json.dump(dict(args=vars(args), log=log), open(out + ".json", "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s -> {out}.pt", flush=True)


if __name__ == "__main__":
    main()
