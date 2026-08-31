"""Design B — self-resampling, run FREE.

World: {Mess3(0.6,0.15) = factor 0 = exogenous control, asym3 = factor 1 = consequential}.
Closed loop: every k steps we read the model's OWN belief over asym3's hidden state (frozen
linear probe on the belief-carrying layer), sample a state from it, and RESET asym3 to that
state before generating the next tokens. So the model's belief readout q1 is consequential:
it seeds asym3's near-future emissions, which the model must then predict.

Prediction (consequence frame): to minimise its self-seeded predictive entropy
H ~ H(q1) + E_q1[H(future | s)], the model collapses q1 onto a point mass on asym3's
PREDICTABLE state (state 0). Signature: H(q1) drops toward 0 and mass(q1, state0) -> 1 over
post-training, while H(q0) for the exogenous Mess3 control stays at the faithful posterior
(~0.9 nats). Mess3 is never reseeded -> zero consequence -> no collapse.

Run free: post-training data is purely the closed-loop (on-policy) stream; no exogenous mixing.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT  # noqa: E402
from factors import mess3_factor, asym3_factor, make_world  # noqa: E402
from probes import readout  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
Q = 3


def ent(p, axis=-1):
    p = np.clip(p, 1e-30, None)
    return -(p * np.log(p)).sum(axis)


@torch.no_grad()
def belief_readout(model, prefix, ell, W, b, device):
    """Model's belief over a factor's state at the last position of `prefix` (B,t)->(B,Q)."""
    model.eval()
    x = torch.from_numpy(prefix).to(device)
    _, hid = model(x, return_hidden=True)
    r = hid[ell][:, -1, :].float().cpu().numpy()
    return readout(r, W, b)


def generate_closedloop(model, facs, probes, k, B, L, rng, device, reseed=1, do_set=None):
    """Model-in-the-loop generation. asym3 (factor `reseed`) is reset every k steps to a state
    sampled from the model's belief readout (or forced to `do_set` for interventions)."""
    N = len(facs)
    flats = [f.T.transpose(1, 0, 2).reshape(Q, Q * Q) for f in facs]   # (i, z*Q+j)
    states = [rng.choice(Q, size=B, p=f.pi) for f in facs]
    toks = np.zeros((B, L), dtype=np.int64)
    powers = Q ** np.arange(N - 1, -1, -1)
    W, b, ell = probes[f"W{reseed}"], probes[f"b{reseed}"], int(probes[f"ell{reseed}"])
    reseed_pos = []
    for t in range(L):
        if t > 0 and t % k == 0:
            if do_set is None:
                q = belief_readout(model, toks[:, :t], ell, W, b, device)
                s_hat = (rng.random(B)[:, None] < np.cumsum(q, 1)).argmax(1)
            else:
                s_hat = np.full(B, do_set)
            states[reseed] = s_hat
            reseed_pos.append(t)
        z = np.empty((B, N), dtype=np.int64)
        nxt = np.empty((B, N), dtype=np.int64)
        for n in range(N):
            cdf = np.cumsum(flats[n][states[n]], axis=1)
            idx = (rng.random(B)[:, None] < cdf).argmax(1)
            z[:, n] = idx // Q; nxt[:, n] = idx % Q
        toks[:, t] = (z * powers[None, :]).sum(1)
        states = [nxt[:, n] for n in range(N)]   # keep states as a per-factor list
    return toks, reseed_pos


@torch.no_grad()
def read_qs(model, toks, probes, device, N=2):
    """Read belief readouts q_n for all factors at all positions on a token batch. list[(B,L,Q)]."""
    model.eval()
    x = torch.from_numpy(toks).to(device)
    _, hid = model(x, return_hidden=True)
    qs = []
    for n in range(N):
        r = hid[int(probes[f"ell{n}"])].float().cpu().numpy()      # (B,L,d)
        qs.append(readout(r, probes[f"W{n}"], probes[f"b{n}"]))     # (B,L,Q)
    return qs


@torch.no_grad()
def pred_entropy(model, toks, device):
    model.eval()
    x = torch.from_numpy(toks).to(device)
    logits = model(x)
    p = torch.softmax(logits, -1).float().cpu().numpy()
    return ent(p, axis=-1)   # (B,L)


def measure(model, facs, probes, k, L, rng, device, B=2048):
    toks, rpos = generate_closedloop(model, facs, probes, k, B, L, rng, device)
    qs = read_qs(model, toks, probes, device, N=len(facs))
    tail = slice(L // 2, None)
    post = [p for p in rpos if p >= L // 2]                       # reseed positions in the tail
    H1 = ent(qs[1]); H0 = ent(qs[0])
    pe = pred_entropy(model, toks, device)
    return dict(
        H1_tail=float(H1[:, tail].mean()), H0_tail=float(H0[:, tail].mean()),
        H1_atreseed=float(H1[:, post].mean()) if post else float("nan"),
        mass1_s0_tail=float(qs[1][:, tail, 0].mean()),
        pred_ent_tail=float(pe[:, tail].mean()),
        pred_ent_postreseed=float(np.mean([pe[:, p:p + 2].mean() for p in post])) if post else float("nan"),
    ), qs


def consequence(model, facs, probes, k, L, rng, device, B=2048):
    """Do-set asym3's seeded state to 0/1/2; measure spread of the realized next-token dist."""
    dists = []
    for s in range(Q):
        toks, _ = generate_closedloop(model, facs, probes, k, B, L, rng, device, do_set=s)
        d = np.bincount(toks[:, L // 2:].ravel(), minlength=facs[0].T.shape[1] ** len(facs))
        dists.append(d / d.sum())
    dists = np.array(dists)                                   # (Q, V)
    mean = dists.mean(0)
    # total-variation spread of the realized token dist across the 3 forced seeds
    tv = float(np.mean([0.5 * np.abs(dists[s] - mean).sum() for s in range(Q)]))
    return tv, dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/uni_mess3_asym3")
    ap.add_argument("--out", default="runs/expB")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--pool", type=int, default=4096)
    ap.add_argument("--refresh", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    ck = os.path.join(BASE, args.ckpt); out = os.path.join(BASE, args.out)

    meta = json.load(open(ck + ".json"))
    a = meta["args"]; V = meta["V"]
    facs = [mess3_factor(0.6, 0.15), asym3_factor()]
    probes = {k_: np.load(ck + "_probes.npz")[k_] for k_ in np.load(ck + "_probes.npz").files}
    model = GPT(V, a["d_model"], a["n_layer"], a["n_head"], max_len=args.L).to(dev)
    model.load_state_dict(torch.load(ck + ".pt", map_location=dev))
    print(f"loaded {ck}.pt  V={V}  probe layers: mess3={int(probes['ell0'])} "
          f"asym3={int(probes['ell1'])}  R2: {float(probes['r20']):.3f}/{float(probes['r21']):.3f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    traj = []
    m0, qs0 = measure(model, facs, probes, args.k, args.L, rng, dev)
    m0["step"] = 0; traj.append(m0)
    print(f"step    0 | H1(asym3)={m0['H1_tail']:.3f} H0(mess3)={m0['H0_tail']:.3f} "
          f"mass1_s0={m0['mass1_s0_tail']:.3f} pred_ent={m0['pred_ent_tail']:.3f}", flush=True)

    buf = {"x": None, "p": 0, "ref": -10**9}
    def next_batch(step):
        if (buf["x"] is None or buf["p"] + args.batch > args.pool
                or step - buf["ref"] >= args.refresh):   # regenerate pool on-policy every `refresh` steps
            pool, _ = generate_closedloop(model, facs, probes, args.k, args.pool, args.L, rng, dev)
            buf["x"] = pool[rng.permutation(args.pool)]; buf["p"] = 0; buf["ref"] = step
        b = buf["x"][buf["p"]:buf["p"] + args.batch]; buf["p"] += args.batch
        return b

    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        x = torch.from_numpy(next_batch(step)).to(dev)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.eval_every == 0:
            m, _ = measure(model, facs, probes, args.k, args.L, rng, dev)
            m["step"] = step; m["loss"] = float(loss.item()); traj.append(m)
            print(f"step {step:5d} | loss {loss.item():.3f} | H1(asym3)={m['H1_tail']:.3f} "
                  f"H0(mess3)={m['H0_tail']:.3f} mass1_s0={m['mass1_s0_tail']:.3f} "
                  f"pred_ent={m['pred_ent_tail']:.3f}", flush=True)

    mf, qsf = measure(model, facs, probes, args.k, args.L, rng, dev)
    tv, _ = consequence(model, facs, probes, args.k, args.L, rng, dev)
    print(f"\nfinal | H1={mf['H1_tail']:.3f} (from {m0['H1_tail']:.3f}) | "
          f"H0={mf['H0_tail']:.3f} (from {m0['H0_tail']:.3f}) | mass1_s0={mf['mass1_s0_tail']:.3f} | "
          f"consequence TV(asym3 do-set)={tv:.3f}", flush=True)

    # ---- plots ----
    steps = [t["step"] for t in traj]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(steps, [t["H1_tail"] for t in traj], "-o", label="H(q1) asym3 (consequential)")
    ax[0].plot(steps, [t["H0_tail"] for t in traj], "-s", label="H(q0) mess3 (control)")
    ax[0].axhline(np.log(3), ls=":", c="gray", lw=0.8); ax[0].set_xlabel("post-train step")
    ax[0].set_ylabel("belief readout entropy (nats)"); ax[0].legend(); ax[0].set_title("entropy collapse")
    ax[1].plot(steps, [t["mass1_s0_tail"] for t in traj], "-o", c="C2")
    ax[1].set_xlabel("post-train step"); ax[1].set_ylabel("mean q1 mass on asym3 state 0")
    ax[1].set_title("collapse direction (-> predictable state)"); ax[1].axhline(1/3, ls=":", c="gray")
    def bary(p):
        v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]]); return p @ v
    for q, lab, mk in [(qs0[1], "asym3 pre", "."), (qsf[1], "asym3 post", "x")]:
        xy = bary(q[:, q.shape[1]//2:, :].reshape(-1, 3)[::7])
        ax[2].scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.4, marker=mk, label=lab)
    ax[2].plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], "k-", lw=0.5)
    ax[2].set_title("asym3 belief cloud pre/post"); ax[2].axis("equal"); ax[2].axis("off"); ax[2].legend()
    fig.tight_layout(); fig.savefig(out + ".png", dpi=110)

    json.dump(dict(args=vars(args), traj=traj, final=mf, consequence_tv=tv,
                   m0=m0, probe_layers=[int(probes["ell0"]), int(probes["ell1"])]),
              open(out + ".json", "w"), indent=2)
    torch.save(model.state_dict(), out + ".pt")
    print(f"done in {time.time()-t0:.0f}s -> {out}.png/.json", flush=True)


if __name__ == "__main__":
    main()
