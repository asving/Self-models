"""Distinguish (C) constant-commit vs (R) per-round-route vs (D) per-round-recompute for how
the 'own move' used in decode b=(own_move - o)%3 is obtained. Reuses wb_pretrain setup.

EXP1: (a) mode-trajectory stability; (b) constant-commit synthetic fidelity vs per-round-decode;
      (c) per-head retrieval attention at the decode position.
EXP2: activation patch on the intended-move direction (tokens FIXED), single-early vs all vs mid,
      across layers; effect-size vs matched-norm random control.
"""
import os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
torch.set_num_threads(8)
from wb_pretrain import load, gen_games, trunk_resid, ridge_probe, M
DEV = "cpu"


# ---- residual-stream capture with per-layer hooks (block input/output) ----
@torch.no_grad()
def run_with_resid(net, tok, edits=None):
    """Forward pass; optionally apply edits = list of (layer_idx, where, fn) applied to the
    running residual x. where in {'pre0','post0','post1'} = before block0, after block i.
    Returns logits (act head) and the list of post-block residuals."""
    L = tok.shape[1]
    x = net.emb(tok) + net.pos(torch.arange(L))[None]
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    if edits:
        for (li, where, fn) in edits:
            if where == 'pre0' and li == 0:
                x = fn(x)
    posts = []
    for i, blk in enumerate(net.blocks):
        h = blk.ln1(x); a, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a; x = x + blk.mlp(blk.ln2(x))
        if edits:
            for (li, where, fn) in edits:
                if where == 'post' and li == i:
                    x = fn(x)
        posts.append(x)
    xf = net.lnf(x)
    return net.act_head(xf), posts


def policy_from_logits(logits):
    return F.softmax(logits[:, -1] if logits.dim() == 3 else logits, -1)


if __name__ == "__main__":
    net, a = load(); T = a["T"]
    print(f"=== mechanism discrimination C/R/D on pretrain_bias ({a['n_layer']}L) ===\n")
    seq, bias, A, B, O, P = gen_games(net, 4000, T, np.random.default_rng(11))
    mode = P.argmax(-1)
    Bn = seq.shape[0]

    # =====================================================================
    # EXP 1a: mode-trajectory stability (C vs R/D)
    # =====================================================================
    print("--- EXP1a: within-game mode trajectory ---")
    # number of mode changes per game (over t=0..T-1)
    changes = (mode[:, 1:] != mode[:, :-1]).sum(1).float()
    print(f"mean mode-changes per game (T={T}): {changes.mean():.2f}  median {changes.median():.0f}")
    print(f"frac games with 0 changes after opening (t>=1 constant): "
          f"{(changes==0).float().mean():.3f}")
    # settle time: first t after which mode never changes
    settle = torch.full((Bn,), T - 1)
    for t in range(T - 1, 0, -1):
        same_after = (mode[:, t:] == mode[:, t:t+1]).all(1)
        settle = torch.where(same_after, torch.full((Bn,), t), settle)
    # settle relative to final move
    print(f"settle time (first t s.t. mode constant thereafter): mean {settle.float().mean():.1f}  "
          f"median {settle.median():.0f}")
    # does the committed (final) move adapt to the inferred bias per game? (if always same -> not adapting)
    fin = mode[:, -1]
    print(f"committed (final) move distribution: {torch.bincount(fin, minlength=3).tolist()} "
          f"(adapts per-game if spread, not a fixed move)")
    favb_true = bias.argmax(1)
    print(f"committed move == counter to true favored-b: {(fin==((favb_true+1)%3)).float().mean():.3f}")
    # fraction of games where the mode visits >1 distinct value over t>=1
    distinct = torch.stack([(mode[:, 1:] == k).any(1) for k in range(3)], 1).sum(1)
    print(f"distinct mode values visited after opening: mean {distinct.float().mean():.2f} "
          f"(1.0 => truly constant-commit; >1 => adapts/switches)")

    # =====================================================================
    # EXP 1b: constant-commit synthetic vs per-round-decode synthetic
    # =====================================================================
    print("\n--- EXP1b: constant-commit synthetic fidelity ---")
    @torch.no_grad()
    def kl_agree(Ps, Pn):
        kl = (Pn * ((Pn + 1e-9).log() - (Ps + 1e-9).log())).sum(-1).mean().item()
        ag = (Ps.argmax(-1) == Pn.argmax(-1)).float().mean().item()
        return kl, ag

    # constant-commit: histogram observed outcomes o over prefix, shift by the net's SINGLE
    # committed mode m (its settled/final argmax), decode b=(m - o)%3 for ALL past steps, BR.
    @torch.no_grad()
    def synth_constant(O_seq, m_commit, alpha0=1.0, gain=6.0):
        Bn, Tt = O_seq.shape; ps = []
        for t in range(Tt):
            pref = O_seq[:, :t]  # observed outcomes before step t
            if t == 0:
                cnt = torch.full((Bn, 3), alpha0)
            else:
                b = (m_commit[:, None] - pref) % 3  # SAME constant m applied to every past outcome
                cnt = torch.full((Bn, 3), alpha0).scatter_add_(
                    1, b, torch.ones_like(b, dtype=torch.float))
            qhat = cnt / cnt.sum(1, keepdim=True)
            ps.append(F.softmax(gain * qhat[:, [2, 0, 1]], -1))
        return torch.stack(ps, 1)

    # per-round-decode (teacher-forced with the net's actual per-step mode) -- from prior work
    @torch.no_grad()
    def synth_perround(O_seq, mode_seq, alpha0=1.0, gain=6.0):
        Bn, Tt = O_seq.shape; count = torch.full((Bn, 3), alpha0); ps = []
        for t in range(Tt):
            qhat = count / count.sum(1, keepdim=True)
            ps.append(F.softmax(gain * qhat[:, [2, 0, 1]], -1))
            b = (mode_seq[:, t] - O_seq[:, t]) % 3
            count = count + F.one_hot(b, 3).float()
        return torch.stack(ps, 1)

    Pc = synth_constant(O, fin)           # constant = final committed move
    Pr = synth_perround(O, mode)          # per-round actual modes
    klc, agc = kl_agree(Pc, P); klr, agr = kl_agree(Pr, P)
    print(f"constant-commit (single m=final mode):  KL={klc:.4f}  argmax-agree={agc:.3f}")
    print(f"per-round-decode (actual mode/step):    KL={klr:.4f}  argmax-agree={agr:.3f}")
    # also constant using the move at t=1 (early commit), to be fair to C
    Pc1 = synth_constant(O, mode[:, 1])
    kl1, ag1 = kl_agree(Pc1, P)
    print(f"constant-commit (single m=move@t=1):    KL={kl1:.4f}  argmax-agree={ag1:.3f}")
    print("  -> if constant ~ per-round, the per-round-decode story is UNNECESSARY (favors C)")

    # =====================================================================
    # EXP 1c: per-head retrieval attention at the decode
    # =====================================================================
    print("\n--- EXP1c: per-head attention (retrieval signature) ---")
    @torch.no_grad()
    def per_head_attn(net, tok):
        L = tok.shape[1]
        x = net.emb(tok) + net.pos(torch.arange(L))[None]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        out = []
        for blk in net.blocks:
            h = blk.ln1(x)
            _, aw = blk.attn(h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=False)
            out.append(aw)  # (B, nh, L, L)
            a, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a; x = x + blk.mlp(blk.ln2(x))
        return out
    AW = per_head_attn(net, seq[:512])
    for q in [10, 20, 39]:  # query positions
        print(f" query pos t={q}:")
        for li, aw in enumerate(AW):
            for hh in range(aw.shape[1]):
                row = aw[:, hh, q, :q+1].mean(0)  # over keys 0..q
                w_prev = row[q-1].item() if q >= 1 else 0.0   # previous position (route signature)
                w_self = row[q].item()
                w_open = row[1].item() if q >= 1 else 0.0      # early commit pos (t=1)
                w_pref = row[:q-1].sum().item() if q >= 2 else 0.0  # prefix 0..t-2 (recompute)
                eff = (1.0 / (row**2).sum()).item()
                print(f"   L{li}H{hh}: prev(t-1)={w_prev:.2f} self={w_self:.2f} "
                      f"open(t1)={w_open:.2f} prefix[:t-1]={w_pref:.2f} eff#pos={eff:.1f}")

    # =====================================================================
    # EXP 2: ACTIVATION PATCH on the intended-move direction (tokens FIXED)
    # =====================================================================
    print("\n--- EXP2: activation patch on own-move representation ---")
    # Build an intended-move probe (multinomial) at the POST-block-0 residual (where move is set up
    # for the decode in layer 1) and also post-block-1. We patch by moving along probe directions.
    seq_tr, _, _, _, O_tr, P_tr = gen_games(net, 4000, T, np.random.default_rng(21))
    _, posts_tr = run_with_resid(net, seq_tr)
    mode_tr = P_tr.argmax(-1)

    # probe own-move from each layer's post residual, all positions
    def fit_move_probe(post, mode_seq):
        X = post[:, :T].reshape(-1, post.shape[-1])
        Y = F.one_hot(mode_seq.reshape(-1), 3).float()
        W, r2, _ = ridge_probe(X, Y, X, Y)  # in-sample dirs ok for steering
        acc = ((X @ W).argmax(1) == mode_seq.reshape(-1)).float().mean().item()
        return W, acc
    probes = []
    for li in range(len(posts_tr)):
        W, acc = fit_move_probe(posts_tr[li], mode_tr)
        probes.append(W)
        print(f"own-move probe @post-L{li}: train acc={acc:.3f}")

    # Clean SET-intervention: at each patched position, replace the residual's component in the
    # 3-D move-subspace (span of probe columns) with a vector that the probe reads as one-hot(m'),
    # scaled to that position's typical move-signal magnitude. Everything orthogonal is preserved.
    @torch.no_grad()
    def move_subspace(W):
        Q, _ = torch.linalg.qr(W)            # (d,3) orthonormal basis of move subspace
        return Q
    @torch.no_grad()
    def set_move(x, p, Q, m_target, scale):
        """x (B,L,d): at position p, zero the move-subspace component and write +scale on basis m'."""
        comp = x[:, p] @ Q                   # (B,3) coords in subspace
        x[:, p] = x[:, p] - comp @ Q.T       # remove move component
        tgt = torch.zeros(3); tgt[m_target] = scale
        x[:, p] = x[:, p] + tgt @ Q.T
        return x

    @torch.no_grad()
    def patch_effect(seqs, layer, positions, m_target, scale, control=None):
        W = probes[layer]
        Q = move_subspace(W) if control is None else control
        readpos = seqs.shape[1] - 1          # the position the policy is read from
        def fn(x):
            x = x.clone()
            for p in positions:
                x = set_move(x, p, Q, m_target, scale)
            return x
        log_base, posts_b = run_with_resid(net, seqs)
        log_pat, posts_p = run_with_resid(net, seqs, [(layer, 'post', fn)])
        pb = F.softmax(log_base[:, -1], -1); pp = F.softmax(log_pat[:, -1], -1)
        kl = (pb * ((pb+1e-9).log() - (pp+1e-9).log())).sum(-1).mean().item()
        favb_b = (pb.argmax(-1) - 1) % 3; favb_p = (pp.argmax(-1) - 1) % 3
        flip = (favb_p != favb_b).float().mean().item()
        # sanity: did the patch actually change the probe-decoded move at a patched position?
        if positions:
            pp_pos = positions[0]
            mv_b = (posts_b[layer][:, pp_pos] @ W).argmax(-1)
            mv_p = (posts_p[layer][:, pp_pos] @ W).argmax(-1)
            took = (mv_p == m_target).float().mean().item()
        else:
            took = float('nan')
        return kl, flip, took

    seqE, biasE, AE, BE, OE, PE = gen_games(net, 1500, T, np.random.default_rng(31))
    modeE = PE.argmax(-1); fin_E = modeE[:, -1]
    L = seqE.shape[1]                       # = T+1 = 41; positions 0..T, policy read at index T
    m_tgt = int(((fin_E + 1) % 3).float().mode().values)  # a move != typical committed move
    _, postsE = run_with_resid(net, seqE)
    print("\npatch SET own-move -> m'={}; metrics: KL(final policy), frac favored-b flipped, "
          "probe-took@pos".format(m_tgt))
    for layer in range(len(probes)):
        # scale = typical magnitude of the move component in subspace at the read position
        Q = move_subspace(probes[layer])
        scale = (postsE[layer][:, L-1] @ Q).norm(dim=-1).mean().item()
        # (i) single EARLY commit position t=1
        kl_e, fl_e, tk_e = patch_effect(seqE, layer, [1], m_tgt, scale)
        # (ii) ALL positions 1..T (INCLUDING the read position T)
        kl_a, fl_a, tk_a = patch_effect(seqE, layer, list(range(1, L)), m_tgt, scale)
        # (iii) just the READ position T (the decode site itself)
        kl_r, fl_r, tk_r = patch_effect(seqE, layer, [L-1], m_tgt, scale)
        # (iv) one MID position
        kl_m, fl_m, tk_m = patch_effect(seqE, layer, [L//2], m_tgt, scale)
        # control: random orthonormal 3-D subspace, ALL positions, matched scale
        g = torch.linalg.qr(torch.randn(probes[layer].shape[0], 3))[0]
        kl_c, fl_c, _ = patch_effect(seqE, layer, list(range(1, L)), m_tgt, scale, control=g)
        print(f" L{layer} (move-scale~{scale:.2f}):")
        print(f"   single-early(t=1): KL={kl_e:.3f} flip={fl_e:.2f} took={tk_e:.2f}")
        print(f"   read-pos(t={L-1}):   KL={kl_r:.3f} flip={fl_r:.2f} took={tk_r:.2f}")
        print(f"   mid(t={L//2}):        KL={kl_m:.3f} flip={fl_m:.2f} took={tk_m:.2f}")
        print(f"   ALL pos:           KL={kl_a:.3f} flip={fl_a:.2f}")
        print(f"   RANDOM-ctrl ALL:   KL={kl_c:.3f} flip={fl_c:.2f}")
    print("\n interpretation:")
    print("  'took' confirms the patch actually set the move at that position (else null result is moot).")
    print("  single-early flips downstream everywhere -> CONSTANT-COMMIT (C)")
    print("  read-pos patch flips final policy & sticks -> the move is USED at decode (R: routed)")
    print("  patch took but NO downstream effect -> healed/recomputed from intact context (D)")

    # =====================================================================
    # EXP 2b: HEALING TEST -- patch L0-move at one early pos, read L1-move at SAME pos.
    #   recompute (D): L1 re-derives the move from tokens -> stays at the UNPATCHED value.
    #   route (R): the patched L0 value propagates -> moves to target.
    # =====================================================================
    print("\n--- EXP2b: healing test (patch L0-move @ early pos, read L1-move there) ---")
    W0, W1 = probes[0], probes[1]; Q0 = move_subspace(W0)
    base0 = (posts_tr[0][:, :T].reshape(-1, W0.shape[0]) @ Q0).norm(dim=-1).mean().item()
    p0 = 5
    for mult in [1, 3, 8]:
        sc = base0 * mult
        def fn(x):
            x = x.clone(); comp = x[:, p0] @ Q0; x[:, p0] = x[:, p0] - comp @ Q0.T
            tgt = torch.zeros(3); tgt[m_tgt] = sc; x[:, p0] = x[:, p0] + tgt @ Q0.T
            return x
        _, pb_ = run_with_resid(net, seqE)
        _, pp_ = run_with_resid(net, seqE, [(0, 'post', fn)])
        l1_base = (pb_[1][:, p0] @ W1).argmax(-1)
        l1_pat = (pp_[1][:, p0] @ W1).argmax(-1)
        to_tgt = (l1_pat == m_tgt).float().mean().item()
        unchanged = (l1_pat == l1_base).float().mean().item()
        tag = "(8x = OOD corruption)" if mult == 8 else "(in-distribution)" if mult <= 3 else ""
        print(f"  mult={mult} {tag}: L1-move@p0 -> target {to_tgt:.2f}, stays UNPATCHED {unchanged:.2f}")
    print("  stays-unpatched at realistic strength => L1 RECOMPUTES the move from context (D), not routed.")
