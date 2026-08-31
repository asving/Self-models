"""White-box numpy reimplementation of the trained GPT.

Goal (validation-first interpretability): a transparent function that takes a token sequence
and reproduces EVERY layer's activations and the final logits, exactly matching the torch model.
This is the substrate for mechanistic analysis — once it matches bit-for-bit, we can replace
individual components (e.g. an attention head) with a hypothesized mechanism and measure exactly
how much of the activation it explains.

Architecture (from comp_icl/model.py): pre-LN decoder.
  x = tok[idx] + pos[:L]
  per block:  x = x + MHA(LN1(x));  x = x + MLP(LN2(x))
  out = Head(LNf(x))           # Head has no bias
MHA = nn.MultiheadAttention(batch_first): in_proj_weight=[Wq;Wk;Wv] (3d,d), out_proj (d,d),
causal mask (triu,1). MLP = Linear(d,4d)-GELU(exact)-Linear(4d,d). LN eps=1e-5, biased var.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.special import erf


def load_weights(path, dtype=np.float64):
    sd = torch.load(path, map_location="cpu")
    return {k: v.numpy().astype(dtype) for k, v in sd.items()}


def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)            # population variance (ddof=0) — matches torch
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def softmax_lastdim(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def mha(h, W, pfx, n_head):
    """Self-attention. h (B,L,d). Returns (out (B,L,d), attn (B,n_head,L,L), per-head v)."""
    B, L, d = h.shape
    hd = d // n_head
    Wi, bi = W[pfx + "in_proj_weight"], W[pfx + "in_proj_bias"]
    qkv = h @ Wi.T + bi                         # (B,L,3d)
    q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
    sp = lambda t: t.reshape(B, L, n_head, hd).transpose(0, 2, 1, 3)   # (B,nh,L,hd)
    q, k, v = sp(q), sp(k), sp(v)
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)                  # (B,nh,L,L)
    mask = np.triu(np.ones((L, L), bool), 1)
    scores = np.where(mask[None, None], -np.inf, scores)
    attn = softmax_lastdim(scores)
    ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, d)            # (B,L,d)
    out = ctx @ W[pfx + "out_proj.weight"].T + W[pfx + "out_proj.bias"]
    return out, attn, v


def forward(W, tokens, n_layer=4, n_head=4, edit_fn=None):
    """Full white-box forward. Returns dict of activations (every layer) + logits + attn patterns.
    edit_fn(name, x)->x is applied to each residual-stream activation as it is produced (and the
    edited value continues to propagate) — for causal interventions/patching."""
    tokens = np.asarray(tokens)
    L = tokens.shape[-1]
    def ed(name, x):
        if edit_fn is not None:
            x = edit_fn(name, x)
        return x
    x = W["tok.weight"][tokens] + W["pos.weight"][np.arange(L)]        # (B,L,d)
    x = ed("embed", x); acts = {"embed": x.copy()}
    attns = []
    for i in range(n_layer):
        p = f"blocks.{i}."
        h1 = layernorm(x, W[p + "ln1.weight"], W[p + "ln1.bias"])
        a, A, v = mha(h1, W, p + "attn.", n_head)
        acts[f"L{i}.attn_out"] = a.copy(); attns.append(A)
        x = x + a
        x = ed(f"L{i}.resid_mid", x); acts[f"L{i}.resid_mid"] = x.copy()
        h2 = layernorm(x, W[p + "ln2.weight"], W[p + "ln2.bias"])
        m = gelu(h2 @ W[p + "mlp.0.weight"].T + W[p + "mlp.0.bias"]) @ W[p + "mlp.2.weight"].T + W[p + "mlp.2.bias"]
        acts[f"L{i}.mlp_out"] = m.copy()
        x = x + m
        x = ed(f"L{i}.resid_post", x); acts[f"L{i}.resid_post"] = x.copy()  # == torch return_hidden[i]
    xf = layernorm(x, W["lnf.weight"], W["lnf.bias"])
    acts["final_ln"] = xf.copy()
    acts["logits"] = xf @ W["head.weight"].T       # no bias
    return acts, attns


# --------------------------- verification --------------------------- #
def verify(ckpt, n_layer=4, n_head=4, B=16, L=64, V=9, d_model=128, seed=0):
    """Compare white-box numpy forward to the torch model (run in float64) on random tokens."""
    import sys, os
    sys.path.insert(0, os.path.expanduser("~/comp_icl"))
    from model import GPT
    rng = np.random.default_rng(seed)
    toks = rng.integers(0, V, size=(B, L))
    W = load_weights(ckpt, dtype=np.float64)
    acts, _ = forward(W, toks, n_layer, n_head)
    # torch reference in double precision
    m = GPT(V, d_model, n_layer, n_head, max_len=L).double()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    with torch.no_grad():
        logits_t, hid_t = m(torch.from_numpy(toks), return_hidden=True)
    out = {}
    for i in range(n_layer):
        out[f"L{i}.resid_post"] = float(np.abs(acts[f"L{i}.resid_post"] - hid_t[i].numpy()).max())
    out["logits"] = float(np.abs(acts["logits"] - logits_t.numpy()).max())
    return out


if __name__ == "__main__":
    import os
    BASE = os.path.dirname(os.path.abspath(__file__))
    for name in ["runs/uni_mess3_asym3", "runs/expB", "runs/expB_rl_b3", "runs/expB_rl_b1", "runs/base_2mess3"]:
        p = os.path.join(BASE, name + ".pt")
        if not os.path.exists(p):
            print(f"{name:28s}  (missing)"); continue
        d = verify(p)
        worst = max(d.values())
        print(f"{name:28s}  max|Δ| logits={d['logits']:.2e}  worst-layer={worst:.2e}  "
              + ("OK" if worst < 1e-6 else "CHECK"))
