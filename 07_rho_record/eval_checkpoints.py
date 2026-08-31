"""Evaluate all checkpoints against the oracle floors: how close does the
transformer get to the optimal stream-observer, and HOW does it get there.

Per checkpoint:
  - per-position x-slot CE (full-vocab: the honest objective) vs observer/agent
    floors; type-restricted predictive P(x=+1) for the agreement analyses
  - excess loss over the observer floor (positions >= 8, "late")
  - channel-acquisition regression: model contrast (2p-1) ~ b_e*f_echo + b_n*f_nat
    against the ORACLE's two predictive components (both -> 1 at optimality)
  - agreement R^2 between model and oracle predictive contrasts

Run: cwd = this folder;
  CUDA_VISIBLE_DEVICES=<free> ~/comp_icl/.venv/bin/python eval_checkpoints.py
"""
import glob
import importlib.util
import json
import math
import re

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import world as W

_spec = importlib.util.spec_from_file_location(
    "_comp_icl_model", "/data/users/asvin/comp_icl/model.py")
_cim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cim)
GPT = _cim.GPT

LATE = 8          # positions >= LATE count as "late" (in-context inference warm)
CHUNK = 512


def model_eval(ckpt_path, toks_np, dev="cuda"):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = GPT(**ck["cfg"]).to(dev).eval()
    model.load_state_dict(ck["state"])
    B = toks_np.shape[0]
    x_nll, a_nll, p_model = [], [], []
    with torch.no_grad():
        for i in range(0, B, CHUNK):
            toks = torch.from_numpy(toks_np[i:i + CHUNK]).to(dev)
            lg = model(toks)
            tgt = toks[:, 1:]
            lp = F.log_softmax(lg[:, :-1], dim=-1)
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            x_nll.append(nll[:, 1::2].cpu().numpy())
            a_nll.append(nll[:, 0::2].cpu().numpy())
            # type-restricted predictive at x-slots: P(X_POS | {X_NEG, X_POS})
            lx = lg[:, :-1][:, 1::2]                     # logits before x targets
            p2 = torch.softmax(lx[..., [W.X_NEG, W.X_POS]], dim=-1)[..., 1]
            p_model.append(p2.cpu().numpy())
    return (np.concatenate(x_nll), np.concatenate(a_nll), np.concatenate(p_model))


def regress2(y, f1, f2):
    """OLS y ~ b1 f1 + b2 f2 (no intercept; all zero-mean-ish by symmetry)."""
    X = np.stack([f1.ravel(), f2.ravel()], 1)
    yv = y.ravel()
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ b
    r2 = 1 - resid.var() / max(yv.var(), 1e-12)
    return b[0], b[1], r2


def main():
    ev = np.load("eval_set.npz")
    toks, x = ev["tokens"], ev["x"]
    obs_floor = W.xslot_loss(ev["obs_p"], x)      # (B,T)
    agt_floor = W.xslot_loss(ev["agt_p"], x)
    obs_c = 2 * ev["obs_p"] - 1                   # oracle predictive contrast
    fe, fn = ev["obs_fe"], ev["obs_fn"]

    ckpts = sorted(glob.glob("ckpt/step_*.pt"))
    rows = []
    curves = {}
    for cp in ckpts:
        step = int(re.search(r"step_(\d+)", cp).group(1))
        x_nll, a_nll, p_m = model_eval(cp, toks)
        mc = 2 * p_m - 1
        b_e, b_n, _ = regress2(mc[:, LATE:], fe[:, LATE:], fn[:, LATE:])
        _, _, r2 = regress2(mc[:, LATE:], obs_c[:, LATE:], np.zeros_like(obs_c[:, LATE:]))
        row = dict(step=step,
                   x_late=float(x_nll[:, LATE:].mean()),
                   x_all=float(x_nll.mean()),
                   a_all=float(a_nll.mean()),
                   excess_obs=float((x_nll - obs_floor)[:, LATE:].mean()),
                   excess_agt=float((x_nll - agt_floor)[:, LATE:].mean()),
                   coef_echo=float(b_e), coef_nature=float(b_n),
                   r2_vs_oracle=float(r2))
        rows.append(row)
        curves[step] = x_nll.mean(0)
        print(f"step {step:6d}  x_late {row['x_late']:.4f}  "
              f"excess_obs {row['excess_obs']:+.4f}  "
              f"b_echo {b_e:+.3f}  b_nat {b_n:+.3f}  R2 {r2:.4f}", flush=True)

    with open("eval_results.json", "w") as f:
        json.dump(rows, f, indent=2)

    tgrid = np.arange(1, W.T + 1)
    steps = [r["step"] for r in rows]

    # fig 1: loss vs position, selected checkpoints, with floors
    fig, ax = plt.subplots(figsize=(8, 5))
    sel = [s for s in [0, 100, 300, 1000, 3000, 20000] if s in curves]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(sel)))
    for s, col in zip(sel, cmap):
        ax.plot(tgrid, curves[s], color=col, label=f"step {s}")
    ax.plot(tgrid, obs_floor.mean(0), "k-", lw=2, label="observer floor")
    ax.plot(tgrid, agt_floor.mean(0), "k--", lw=2, label="agent floor (unreachable)")
    ax.axhline(math.log(2), ls=":", c="gray")
    ax.set(xlabel="round t", ylabel="x-slot CE (nats)",
           title="transformer vs the two Bayes floors, across training")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig("figs/loss_vs_pos_by_ckpt.png", dpi=150)

    # fig 2: excess over observer floor vs step
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [max(s, 1) for s in steps]
    ax.plot(xs, [r["excess_obs"] for r in rows], "o-", label="vs observer floor")
    ax.plot(xs, [r["excess_agt"] for r in rows], "s--",
            label="vs agent floor (Π stays)")
    ax.axhline(0, c="gray", ls=":")
    ax.set(xscale="log", xlabel="training step", ylabel="excess x-slot CE (nats)",
           title="approach to the observer bound")
    ax.legend(); fig.tight_layout()
    fig.savefig("figs/excess_vs_step.png", dpi=150)

    # fig 3: channel acquisition
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, [r["coef_nature"] for r in rows], "o-", label="nature channel  b_nat")
    ax.plot(xs, [r["coef_echo"] for r in rows], "s-", label="echo channel  b_echo")
    ax.plot(xs, [r["r2_vs_oracle"] for r in rows], "^--", c="gray",
            label="R² vs oracle predictive")
    ax.axhline(1, c="gray", ls=":")
    ax.set(xscale="log", xlabel="training step", ylabel="coefficient / R²",
           title="which channel is learned when")
    ax.legend(); fig.tight_layout()
    fig.savefig("figs/channel_acquisition.png", dpi=150)

    # fig 4: final agreement scatter
    x_nll, _, p_m = model_eval(ckpts[-1], toks)
    fig, ax = plt.subplots(figsize=(5, 5))
    idx = np.random.default_rng(0).choice(p_m[:, LATE:].size, 4000, replace=False)
    ax.plot(obs_c[:, LATE:].ravel()[idx], (2 * p_m - 1)[:, LATE:].ravel()[idx],
            ".", ms=2, alpha=0.4)
    ax.plot([-1, 1], [-1, 1], "k-", lw=1)
    ax.set(xlabel="oracle observer contrast 2p−1", ylabel="model contrast",
           title=f"final checkpoint agreement (positions ≥ {LATE})")
    fig.tight_layout(); fig.savefig("figs/agreement_final.png", dpi=150)
    print("eval done -> eval_results.json, figs/")


if __name__ == "__main__":
    main()
