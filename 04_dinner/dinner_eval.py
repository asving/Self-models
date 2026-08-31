"""Paired evaluation of the RL net vs baselines on the dinner-party task.

For each horizon T (train range + held-out 29-32): run the net (sampled, T=1) and the
scripted baselines on env batches built from the SAME seed (same xrates/init/goals; paths
diverge with actions). Also behavior stats from the net's rollouts:
  - WAIT fraction as a function of time-to-deadline,
  - completion times of the three sets,
  - whether the completion ORDER is decay-ranked (fastest factor set last): Kendall-style
    pairwise agreement with the ideal order, vs the index-order template's agreement.

Run:  CUDA_VISIBLE_DEVICES=<id> ~/comp_icl/.venv/bin/python dinner_eval.py --ckpt <path>
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch, torch.nn.functional as F

from dinner import (Env, Net, N, WAIT, TOK_ACT0, TOK_GOAL0, TOK_BOS, enc_goal, enc_ts,
                    enc_obs, filt_init, filt_update_obs, filt_update_trans, run_policy,
                    pol_greedy, make_blockgreedy, make_packer_g, make_backtimed, BASE)

@torch.no_grad()
def net_rollout(net, B, T, seed, dev):
    rng = np.random.default_rng(seed)
    env = Env(B, rng)
    g = rng.integers(0, 3, (B, N))
    toks = np.zeros((B, 3), dtype=np.int64)
    toks[:, 0] = TOK_BOS; toks[:, 1] = enc_goal(g); toks[:, 2] = enc_ts(T)
    z = env.emit()
    tt = torch.from_numpy(np.concatenate([toks, enc_obs(z)[:, None]], 1)).to(dev)
    acts, completes = [], []
    for t in range(T):
        logits = net(tt)[:, -1, TOK_ACT0:TOK_GOAL0]
        a = torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
        a_np = a.cpu().numpy()
        comp = env.step(a_np)
        acts.append(a_np); completes.append(comp)
        z = env.emit()
        nxt = np.stack([TOK_ACT0 + a_np, enc_obs(z)], 1)
        tt = torch.cat([tt, torch.from_numpy(nxt).to(dev)], 1)
    R = (env.states == g).sum(1).astype(float)
    return R, np.stack(acts, 1), np.stack(completes, 1), env.xrates

def order_agreement(acts, completes, xrates):
    """Among episodes with 3 completed sets on 3 distinct factors: fraction of factor pairs
    ordered fastest-last (ideal). Random order = 0.5; perfect back-timing = 1.0."""
    B, T = completes.shape
    agree, tot = 0, 0
    for b in range(B):
        ts = np.where(completes[b])[0]
        if len(ts) != 3:
            continue
        fs = acts[b, ts] // 3
        if len(set(fs.tolist())) != 3:
            continue
        for i in range(3):
            for j in range(i + 1, 3):
                # factor completed earlier should be SLOWER (smaller xrate)
                ok = (xrates[b, fs[i]] < xrates[b, fs[j]]) == (ts[i] < ts[j])
                agree += ok; tot += 1
    return agree / max(tot, 1), tot // 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "dinner_runs/v1/p2_ckpt_004000.pt"))
    ap.add_argument("--B", type=int, default=2048)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(args.ckpt, map_location=dev)
    cfg = sd["cfg"]
    net = Net(cfg["d"], cfg["nl"], cfg["nh"]).to(dev)
    net.load_state_dict(sd["model"]); net.eval()
    print(f"ckpt {args.ckpt} (phase-2 step {sd['step']})")

    horizons = [12, 16, 20, 24, 28, 30, 32]
    table = {}
    print(f"\n{'T':>4s} {'net':>6s} {'packer_g':>9s} {'backtimed':>10s} {'blockgrdy':>10s} "
          f"{'order-agree':>12s} {'n3sets':>7s}")
    for T in horizons:
        R, acts, comp, xr = net_rollout(net, args.B, T, 7, dev)
        base = {}
        for name, pol in [("packer_g", make_packer_g(T)), ("backtimed", make_backtimed(T)),
                          ("blockgreedy", make_blockgreedy(T))]:
            base[name] = float(run_policy(pol, args.B, T, np.random.default_rng(7)).mean())
        oa, n3 = order_agreement(acts, comp, xr)
        tag = " <-- HELD-OUT T" if T > 28 else ""
        print(f"{T:4d} {R.mean():6.3f} {base['packer_g']:9.3f} {base['backtimed']:10.3f} "
              f"{base['blockgreedy']:10.3f} {oa:12.3f} {n3:7d}{tag}")
        table[T] = dict(net=float(R.mean()), sem=float(R.std() / np.sqrt(args.B)),
                        **base, order_agree=oa, n_full=n3)

    # behavior: WAIT fraction vs time-to-deadline (pooled over train horizons)
    print("\nWAIT fraction by time-to-deadline (T=24):")
    R, acts, comp, xr = net_rollout(net, args.B, 24, 11, dev)
    wf = (acts == WAIT).mean(0)
    ttd = 24 - np.arange(24)
    print("  t-to-serve: " + " ".join(f"{d:4d}" for d in ttd[::2]))
    print("  wait frac : " + " ".join(f"{w:4.2f}" for w in wf[::2]))
    print(f"  sets completed per episode: {comp.sum(1).mean():.2f}")
    json.dump(table, open(os.path.join(os.path.dirname(args.ckpt), "eval.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
