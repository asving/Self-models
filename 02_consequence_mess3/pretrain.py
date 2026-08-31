"""Pretrain a GPT on the unified independent factored world {Mess3(0.6,0.15), asym3}
(the consequence-zero baseline). Online next-token CE. After training, localize the
belief-carrying layer per factor via ridge probes and save frozen probes for design B.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import GPT  # noqa: E402
from factors import mess3_factor, asym3_factor, make_world  # noqa: E402
from probes import localize  # noqa: E402


def incontext_curve(model, tokens, device, bs=2048):
    model.eval()
    L = tokens.shape[1]
    tot = np.zeros(L - 1); n = 0
    with torch.no_grad():
        for i in range(0, len(tokens), bs):
            x = torch.from_numpy(tokens[i:i + bs]).to(device)
            logits = model(x)
            ls = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                 x[:, 1:].reshape(-1), reduction="none").reshape(x.size(0), L - 1)
            tot += ls.sum(0).cpu().numpy(); n += x.size(0)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/uni_mess3_asym3")
    ap.add_argument("--traj_dir", default="runs/uni_traj")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool", type=int, default=16384)
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base, args.out); traj = os.path.join(base, args.traj_dir)
    os.makedirs(os.path.dirname(out), exist_ok=True); os.makedirs(traj, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    facs = [mess3_factor(0.6, 0.15), asym3_factor()]   # factor 0 = Mess3 (control), 1 = asym3 (B-consequential)
    world = make_world(facs, eps=0.0)
    V = world.V
    print(f"device={dev} V={V} factors={[f.name for f in facs]}", flush=True)

    eval_toks, _ = world.sample(8000, args.L, rng)
    oracle = world.forward(eval_toks)["ent"].mean(0)     # next-token entropy floor

    model = GPT(V, args.d_model, args.n_layer, args.n_head, max_len=args.L).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    CHUNK = max(args.batch, args.pool)
    buf = {"x": None, "p": 0}
    def next_batch():
        if buf["x"] is None or buf["p"] + args.batch > CHUNK:
            b, _ = world.sample(CHUNK, args.L, rng)
            buf["x"] = b[rng.permutation(CHUNK)]; buf["p"] = 0
        b = buf["x"][buf["p"]:buf["p"] + args.batch]; buf["p"] += args.batch
        return b

    log = []; t0 = time.time(); tail = slice(args.L // 2, None)
    for step in range(1, args.steps + 1):
        model.train()
        x = torch.from_numpy(next_batch()).to(dev)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.ckpt_every == 0 or step == 1:
            torch.save(model.state_dict(), f"{traj}/step{step}.pt")
        if step % args.eval_every == 0 or step == 1:
            c = incontext_curve(model, eval_toks, dev)
            gap = float(np.mean(c[tail] - oracle[1:][tail]))
            log.append({"step": step, "loss": float(loss.item()),
                        "model_tail": float(c[tail].mean()),
                        "oracle_tail": float(oracle[1:][tail].mean()), "gap_tail": gap})
            print(f"step {step:5d} | loss {loss.item():.3f} | model {c[tail].mean():.3f} "
                  f"oracle {oracle[1:][tail].mean():.3f} gap {gap:+.3f}", flush=True)

    # ---- localize belief-carrying layer + freeze probes ----
    loc_toks, _ = world.sample(4000, args.L, rng)
    R2, fits = localize(model, facs, loc_toks, dev, args.n_layer)
    print("\nbelief-recovery R2 (rows=layer, cols=factor[mess3, asym3]):", flush=True)
    for li in range(args.n_layer):
        print(f"  layer{li}: " + "  ".join(f"{R2[li,n]:.3f}" for n in range(len(facs))), flush=True)
    probe_save = {"r2_table": R2}
    for n, (W, b, ell, r2) in enumerate(fits):
        probe_save[f"W{n}"] = W; probe_save[f"b{n}"] = b
        probe_save[f"ell{n}"] = np.array(ell); probe_save[f"r2{n}"] = np.array(r2)
        print(f"  factor{n} ({facs[n].name}): best layer {ell}  R2={r2:.3f}", flush=True)

    torch.save(model.state_dict(), out + ".pt")
    np.savez(out + "_probes.npz", **probe_save)
    with open(out + ".json", "w") as f:
        json.dump(dict(args=vars(args), V=V, factors=[f.name for f in facs],
                       oracle_tail=float(oracle[1:][tail].mean()),
                       log=log, r2_table=R2.tolist(),
                       best_layers=[int(fits[n][2]) for n in range(len(facs))],
                       best_r2=[float(fits[n][3]) for n in range(len(facs))]), f, indent=2)
    print(f"\ndone in {time.time()-t0:.0f}s -> {out}.pt (+_probes.npz, .json)", flush=True)


if __name__ == "__main__":
    main()
