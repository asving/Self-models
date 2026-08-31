"""TASK 2 (causal store-vs-re-derive) + TASK 3 (rubber-hand) for the TTT nets.

We need surgical control over the forward pass:
  - ablate attention from the decision (query) round to specific past (key) rounds;
  - project a learned subspace (the own-move / efference direction) out of the
    residual at a chosen layer;
  - patch a recalled-own-move representation to a counterfactual value.

nn.MultiheadAttention with a single packed in_proj is unpacked manually here so we
can reproduce attention exactly and intervene on the attention pattern.

CPU.  CUDA_VISIBLE_DEVICES="" ~/comp_icl/.venv/bin/python ttt_causal.py --tag wide_shallow
"""
import argparse
import collections
import numpy as np
import torch
import torch.nn.functional as F
import ttt
from model import TTTNet

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttt_runs")


def load(tag):
    ck = torch.load(f"{RUNS}/{tag}.pt", map_location="cpu", weights_only=False)
    m = TTTNet(**ck["config"]).to("cpu")
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


# ---------------------------------------------------------------------------
# Manual forward replicating TTTNet/Block, with intervention hooks.
# ---------------------------------------------------------------------------
class Manual:
    def __init__(self, model):
        self.m = model
        self.nl = model.cfg["n_layer"]
        self.nh = model.cfg["n_head"]
        self.d = model.cfg["d_model"]
        self.hd = self.d // self.nh

    def block_attn(self, blk, x, attn_mask, key_block=None):
        """Reproduce MultiheadAttention. attn_mask (L,L) bool, True=blocked.
        key_block: optional (B,Lq,Lk) extra additive mask in logit space, or a
        callable(scores)->scores to edit attention logits."""
        B, L, d = x.shape
        h = blk.ln1(x)
        W = blk.attn.in_proj_weight; b = blk.attn.in_proj_bias
        q, k, v = (h @ W.T + b).split(d, dim=-1)
        q = q.view(B, L, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, L, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, L, self.nh, self.hd).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.hd ** 0.5)
        scores = scores.masked_fill(attn_mask[None, None], float("-inf"))
        if key_block is not None:
            scores = key_block(scores)
        attn = scores.softmax(-1)
        o = attn @ v
        o = o.transpose(1, 2).reshape(B, L, d)
        o = blk.attn.out_proj(o)
        x = x + o
        x = x + blk.mlp(blk.ln2(x))
        return x

    def forward(self, occ, edit_resid=None, attn_edit=None):
        """occ (B,L,9) tensor. edit_resid: callable(layer_idx, x)->x applied AFTER
        each block. attn_edit: callable(layer_idx, scores)->scores."""
        B, L, _ = occ.shape
        pos = torch.arange(L)
        x = self.m.inp(occ) + self.m.pos(pos)[None]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        hiddens = []
        for li, blk in enumerate(self.m.blocks):
            kb = (lambda s, li=li: attn_edit(li, s)) if attn_edit else None
            x = self.block_attn(blk, x, mask, key_block=kb)
            if edit_resid is not None:
                x = edit_resid(li, x)
            hiddens.append(x)
        logits = self.m.head(self.m.lnf(x))
        return logits, hiddens


def chosen_moves(logits, occ):
    """argmax over legal cells. logits (B,L,9) tensor, occ (B,L,9)."""
    legal = (occ == 0)
    lg = logits.clone()
    lg[~legal] = -1e9
    return lg.argmax(-1)


# ---------------------------------------------------------------------------
# TASK 2a: attention ablation from decision round to past rounds.
# ---------------------------------------------------------------------------
def attn_ablation(man, occ, valid, my):
    """For decision round t, block attention (all layers, all heads) from query t
    to past key rounds. Compare chosen-move agreement with the clean run, by round.
    A re-deriving net still needs the past OCCUPANCY at t' to recompute m_t' -- so
    blocking attention to the past should hurt recall-dependent rounds REGARDLESS
    of store vs re-derive. The diagnostic value: which past positions matter."""
    occ_t = torch.tensor(occ, dtype=torch.float32)
    B, L, _ = occ_t.shape
    clean_logits, _ = man.forward(occ_t)
    clean = chosen_moves(clean_logits, occ_t)

    print("\n[2a] ATTENTION ABLATION (block query->past keys, all layers/heads):")
    # (i) block ALL past (keep only self) -> upper bound on recall dependence
    def block_all_past(li, s):
        L = s.shape[-1]
        m = torch.zeros(L, L, dtype=torch.bool)
        for q in range(L):
            for kk in range(q):
                m[q, kk] = True
        s = s.masked_fill(m[None, None], float("-inf"))
        return s
    lg, _ = man.forward(occ_t, attn_edit=block_all_past)
    ch = chosen_moves(lg, occ_t)
    print("  block ALL past keys (self-only attention):")
    per_round_agree(ch, clean, valid)

    # (ii) block attention to EACH single past round k, decision at round t
    print("  block attention to a SINGLE past round k (effect on each decision round t):")
    for kblock in range(L - 1):
        def blk_k(li, s, kb=kblock):
            s = s.clone(); s[:, :, :, kb] = float("-inf")
            # but a query at position kb attending to itself must stay valid; only
            # block when query > kb (causal future queries). Position kb itself keeps
            # at least self via other keys; guard against all -inf rows handled by softmax.
            return s
        lg, _ = man.forward(occ_t, attn_edit=blk_k)
        ch = chosen_moves(lg, occ_t)
        # report degradation at each round t>kblock
        line = f"    block key round {kblock}: "
        parts = []
        for t in range(kblock + 1, L):
            mask = valid[:, t] > 0
            if mask.sum() == 0: continue
            agr = (ch[:, t][mask] == clean[:, t][mask]).float().mean().item()
            parts.append(f"t{t}={agr:.3f}")
        print(line + " ".join(parts))


def per_round_agree(ch, clean, valid):
    L = valid.shape[1]
    parts = []
    for t in range(L):
        mask = valid[:, t] > 0
        if mask.sum() == 0: continue
        agr = (ch[:, t][mask] == clean[:, t][mask]).float().mean().item()
        parts.append(f"t{t}={agr:.3f}")
    print("      agree-with-clean: " + " ".join(parts))


# ---------------------------------------------------------------------------
# TASK 2b: own-move subspace projection-out (load-bearing test).
# ---------------------------------------------------------------------------
def fit_own_subspace(hiddens_layer, my, valid, lag, rank=None):
    """Ridge-fit a linear map residual->onehot(own move m_(r-lag)). Return the
    column space (the subspace the own-move code lives in) for projection-out."""
    B, L = valid.shape
    rows = []; labels = []
    for b in range(B):
        for r in range(L):
            if valid[b, r] == 0: continue
            rp = r - lag
            if rp < 0 or valid[b, rp] == 0: continue
            rows.append((b, r)); labels.append(my[b, rp])
    rows = np.array(rows); labels = np.array(labels)
    X = np.stack([hiddens_layer[b, r] for b, r in rows]).astype(np.float64)
    Y = np.eye(9)[labels]
    mu = X.mean(0); Xc = X - mu
    # ridge
    d = Xc.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + 1e-2 * np.eye(d), Xc.T @ (Y - Y.mean(0)))
    return W, mu  # W: (d,9), subspace = column space of W


def subspace_projout(man, model, occ, valid, my):
    """Project the own-move (m_(r-1) and m_(r-2)) subspace OUT of the residual at
    every layer and see if the chosen move degrades. If epiphenomenal (re-derived
    elsewhere / overwritten), projecting it out should NOT hurt the move much."""
    occ_t = torch.tensor(occ, dtype=torch.float32)
    _, hiddens = man.forward(occ_t)
    hids = [h.detach().numpy() for h in hiddens]
    clean_logits, _ = man.forward(occ_t)
    clean = chosen_moves(clean_logits, occ_t)

    print("\n[2b] OWN-MOVE SUBSPACE PROJECTION-OUT (load-bearing test):")
    # build projection matrices per layer from combined m_(r-1),m_(r-2) directions
    Ps = {}
    for li in range(man.nl):
        cols = []
        for lag in (1, 2):
            W, _ = fit_own_subspace(hids[li], my, valid, lag)
            cols.append(W)
        Wcat = np.concatenate(cols, 1)  # (d, 18)
        Q, _ = np.linalg.qr(Wcat)        # orthonormal basis (d, <=18)
        Q = torch.tensor(Q, dtype=torch.float32)
        Ps[li] = Q

    # (i) project out at a single layer
    for target in range(man.nl):
        def edit(li, x, tgt=target):
            if li != tgt: return x
            Q = Ps[tgt]
            proj = (x @ Q) @ Q.T
            return x - proj
        lg, _ = man.forward(occ_t, edit_resid=edit)
        ch = chosen_moves(lg, occ_t)
        parts = []
        for t in range(valid.shape[1]):
            mask = valid[:, t] > 0
            if mask.sum() == 0: continue
            agr = (ch[:, t][mask] == clean[:, t][mask]).float().mean().item()
            parts.append(f"t{t}={agr:.3f}")
        dim = Ps[target].shape[1]
        print(f"  proj-out @ L{target} (dim={dim}): " + " ".join(parts))

    # (ii) project out at ALL layers simultaneously
    def edit_all(li, x):
        Q = Ps[li]
        return x - (x @ Q) @ Q.T
    lg, _ = man.forward(occ_t, edit_resid=edit_all)
    ch = chosen_moves(lg, occ_t)
    parts = []
    for t in range(valid.shape[1]):
        mask = valid[:, t] > 0
        if mask.sum() == 0: continue
        agr = (ch[:, t][mask] == clean[:, t][mask]).float().mean().item()
        parts.append(f"t{t}={agr:.3f}")
    print(f"  proj-out @ ALL layers: " + " ".join(parts))

    # control: project out a RANDOM subspace of the same dim at all layers
    rng = np.random.default_rng(0)
    Rs = {}
    for li in range(man.nl):
        dimr = Ps[li].shape[1]
        A = rng.standard_normal((man.d, dimr))
        Q, _ = np.linalg.qr(A)
        Rs[li] = torch.tensor(Q, dtype=torch.float32)
    def edit_rand(li, x):
        Q = Rs[li]
        return x - (x @ Q) @ Q.T
    lg, _ = man.forward(occ_t, edit_resid=edit_rand)
    ch = chosen_moves(lg, occ_t)
    parts = []
    for t in range(valid.shape[1]):
        mask = valid[:, t] > 0
        if mask.sum() == 0: continue
        agr = (ch[:, t][mask] == clean[:, t][mask]).float().mean().item()
        parts.append(f"t{t}={agr:.3f}")
    print(f"  CONTROL random-subspace proj-out @ ALL: " + " ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wide_shallow")
    ap.add_argument("--task", default="all")
    args = ap.parse_args()
    torch.manual_seed(0)
    d = torch.load(f"{RUNS}/evalset.pt", weights_only=False)
    occ = d["occ"].numpy(); valid = d["valid"].numpy(); my = d["my_move"].numpy()
    model, ck = load(args.tag)
    man = Manual(model)
    print(f"=== CAUSAL: {args.tag} ({ck['config']['n_layer']}x{ck['config']['d_model']}, "
          f"move_acc={ck['move_acc']:.3f}) ===")

    # sanity: manual forward matches model.forward
    occ_t = torch.tensor(occ[:64], dtype=torch.float32)
    with torch.no_grad():
        ref = model(occ_t)
        man_lg, _ = man.forward(occ_t)
    print(f"manual-vs-model max abs logit diff: {(ref-man_lg).abs().max().item():.2e}")

    if args.task in ("all", "ablate"):
        attn_ablation(man, occ, valid, my)
    if args.task in ("all", "projout"):
        subspace_projout(man, model, occ, valid, my)


if __name__ == "__main__":
    main()
