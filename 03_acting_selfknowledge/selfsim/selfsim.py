"""Stage 0 — self-simulation (supervised, closed-loop, traceable).
Each sequence is a deterministic orbit x_{t+1}=π_m(x_t) under a hidden permutation mode m (the net's
'policy', inferred in-context). A query token Q_K asks the net to emit π_m^K(x_L) — its OWN output K
steps ahead (K variable). To answer it must infer which policy it is running and ITERATE it K times.
Tests whether an internal self-simulation engine forms (probe for intermediate iterates π_m^j(x_L)).
Vocab = p symbols + Kmax query tokens. Reuses the small GPT.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT

BASE = os.path.dirname(os.path.abspath(__file__))
P = 6          # symbols / states
M = 8          # number of policy modes (hidden permutations)
KMAX = 5       # max look-ahead horizon
LMIN, LMAX = 6, 16
V = P + KMAX   # tokens: 0..P-1 symbols, P..P+KMAX-1 = query Q_1..Q_KMAX
PERMS = np.stack([np.random.default_rng(100 + i).permutation(P) for i in range(M)])  # (M,P) fixed


def gen(B, rng, L=None):
    L = L or int(rng.integers(LMIN, LMAX + 1))
    m = rng.integers(0, M, B)
    orbit = np.empty((B, L), np.int64)
    orbit[:, 0] = rng.integers(0, P, B)
    for t in range(1, L):
        orbit[:, t] = PERMS[m, orbit[:, t - 1]]
    # K-step iterates of the final state
    pw = np.empty((B, KMAX + 1), np.int64); pw[:, 0] = orbit[:, -1]
    for k in range(1, KMAX + 1):
        pw[:, k] = PERMS[m, pw[:, k - 1]]
    K = rng.integers(1, KMAX + 1, B)
    tgt = pw[np.arange(B), K]
    qtok = P + (K - 1)
    seq = np.concatenate([orbit, qtok[:, None]], 1)          # (B,L+1)
    return seq, tgt, m, K, L, pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/selfsim")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    out = os.path.join(BASE, args.out)
    model = GPT(V, args.d_model, args.n_layer, args.n_head, max_len=LMAX + 2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"self-sim | P={P} M={M} KMAX={KMAX} V={V} | params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    def batch_loss(seq, tgt, L, train=True):
        x = torch.from_numpy(seq).to(dev); y = torch.from_numpy(tgt).to(dev)
        logits = model(x)
        orbit_logits = logits[:, 0:L - 1, :].reshape(-1, V)        # predict orbit[:,1:L]
        orbit_tgt = x[:, 1:L].reshape(-1)
        ol = F.cross_entropy(orbit_logits, orbit_tgt)
        ql = F.cross_entropy(logits[:, L, :], y)                   # query position predicts π^K(x_L)
        return ol, ql, logits

    @torch.no_grad()
    def evaluate():
        model.eval()
        accs = {k: [] for k in range(1, KMAX + 1)}
        for _ in range(8):
            seq, tgt, m, K, L, pw = gen(1024, rng)
            x = torch.from_numpy(seq).to(dev)
            pred = model(x)[:, L, :].argmax(-1).cpu().numpy()
            for k in range(1, KMAX + 1):
                sel = K == k
                if sel.any(): accs[k].append((pred[sel] == tgt[sel]).mean())
        return {k: float(np.mean(v)) for k, v in accs.items()}

    log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        seq, tgt, m, K, L, pw = gen(args.batch, rng)
        ol, ql, _ = batch_loss(seq, tgt, L)
        loss = ol + ql
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0 or step == 1:
            acc = evaluate()
            log.append({"step": step, "orbit_loss": float(ol), "query_loss": float(ql), "acc": acc})
            astr = " ".join(f"K{k}={acc[k]:.2f}" for k in range(1, KMAX + 1))
            print(f"step {step:5d} | orbit {ol:.3f} query {ql:.3f} | acc {astr}", flush=True)

    torch.save(model.state_dict(), out + ".pt")
    np.save(out + "_perms.npy", PERMS)
    json.dump(dict(args=vars(args), P=P, M=M, KMAX=KMAX, V=V, log=log), open(out + ".json", "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s -> {out}.pt", flush=True)


if __name__ == "__main__":
    main()
