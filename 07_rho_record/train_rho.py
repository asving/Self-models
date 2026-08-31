"""Train a small GPT on variant-A rho-record streams (online, fresh data each
step), saving log-spaced checkpoints for the developmental analysis.

The model sees ONLY the public stream (BOS, atil, x, atil, x, ...): it is the
stream-observer. Optimality target = the exact observer filter (world.py);
the agent floor is unreachable by construction (Pi-gap ~ 0.088 nats/round).

Run: cwd = this folder;
  CUDA_VISIBLE_DEVICES=<free> ~/comp_icl/.venv/bin/python train_rho.py
"""
import importlib.util
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import world as W

_spec = importlib.util.spec_from_file_location(
    "_comp_icl_model", "/data/users/asvin/comp_icl/model.py")
_cim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cim)
GPT = _cim.GPT

# ---- config ----
CFG = dict(vocab=W.VOCAB, d_model=128, n_layer=4, n_head=4, max_len=1 + 2 * W.T)
SEED = 0
SEED_EVAL = 1234
BATCH = 128
STEPS = 20000
LR = 3e-4
LR_MIN = 3e-5
WARMUP = 200
WD = 0.01
CKPT_STEPS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700, 1000, 1500,
              2000, 3000, 5000, 7000, 10000, 14000, 20000]
EVAL_B = 4096          # held-out episodes (oracle curves precomputed)
EVAL_SUB = 512         # quick-eval subset during training

XMASK_START = 2        # x-slots are target positions 2,4,...; a-slots 1,3,...


def make_eval_set():
    if os.path.exists("eval_set.npz"):
        return
    print("generating eval set + oracle curves ...", flush=True)
    rng = np.random.default_rng(SEED_EVAL)
    ep = W.gen_batch(EVAL_B, rng)
    obs = W.observer_filter(ep)
    agt = W.agent_filter(ep)
    np.savez_compressed(
        "eval_set.npz",
        tokens=W.tokens(ep), theta=ep["theta"], c=ep["c"], a=ep["a"],
        x=ep["x"], atil=ep["atil"],
        obs_p=obs["p_pos"], obs_fe=obs["f_echo"], obs_fn=obs["f_nat"],
        obs_kappa=obs["kappa"], obs_thpos=obs["thpos"], agt_p=agt["p_pos"])
    print("eval set done.", flush=True)


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    u = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * u))


def slot_losses(logits, toks):
    """(full CE, a-slot CE, x-slot CE) averaged over batch; targets 1..L-1."""
    tgt = toks[:, 1:]
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (B, L-1)
    a_nll = nll[:, 0::2]   # targets at odd positions = atil slots
    x_nll = nll[:, 1::2]   # even positions = x slots
    return nll.mean(), a_nll.mean(), x_nll.mean()


def main():
    torch.manual_seed(SEED)
    dev = "cuda"
    make_eval_set()
    ev = np.load("eval_set.npz")
    ev_toks = torch.from_numpy(ev["tokens"][:EVAL_SUB]).to(dev)
    ev_obs_floor = float(W.xslot_loss(ev["obs_p"][:EVAL_SUB], ev["x"][:EVAL_SUB]).mean())

    model = GPT(**CFG).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(SEED)
    hist = []
    print(f"model {n_par/1e3:.0f}k params; obs floor on eval subset "
          f"{ev_obs_floor:.4f} nats", flush=True)

    def save_ckpt(step):
        torch.save(dict(step=step, cfg=CFG, state=model.state_dict()),
                   f"ckpt/step_{step:06d}.pt")

    t0 = time.time()
    for step in range(STEPS + 1):
        if step in CKPT_STEPS:
            model.eval()
            with torch.no_grad():
                lg = model(ev_toks)
                _, ea, ex = slot_losses(lg, ev_toks)
            save_ckpt(step)
            print(f"[ckpt] step {step:6d}  eval x {ex.item():.4f} "
                  f"(obs floor {ev_obs_floor:.4f})  eval a {ea.item():.4f} "
                  f"(log2 {math.log(2):.4f})", flush=True)
            model.train()
        if step == STEPS:
            break
        ep = W.gen_batch(BATCH, rng)
        toks = torch.from_numpy(W.tokens(ep)).to(dev)
        for g in optim.param_groups:
            g["lr"] = lr_at(step)
        logits = model(toks)
        loss, la, lx = slot_losses(logits, toks)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % 100 == 0:
            hist.append(dict(step=step, loss=float(loss), a=float(la), x=float(lx)))
            print(f"step {step:6d}  ce {loss:.4f}  a {la:.4f}  x {lx:.4f}  "
                  f"({(time.time()-t0):.0f}s)", flush=True)

    with open("train_hist.json", "w") as f:
        json.dump(hist, f)
    with open("ckpt/DONE", "w") as f:
        f.write("done\n")
    print(f"training complete in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
