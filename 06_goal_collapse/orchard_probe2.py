"""S2 causal probing of the pretrained orchard net — DECODER vs ENCODER
subspaces (Asvin's reversed-probe technique) + the beliefs->action circuit.

Two probe directions per target y in {world belief b_t (5d), goal posterior
g_t (5d)}, per layer:
  decoder  y ~ V (h - mu_h)        subspace = row-space(V)
  encoder  h ~ mu_h + W (y - mu_y) subspace = col-space(W)   <- claimed causal

Causal tests at single decision positions (cropped prefixes, logits at the
edited position only):
  goal-swap counterfactual  g' = swap top-lo/top-hi masses
  world-rotate counterfactual b' = roll(b, +1)
  edits: encoder  dh = W (y'-y)
         decoder  dh = V^+ (y'-y)          (least-norm, achieves same decode)
         decoder norm-matched (scaled to ||encoder edit||)
         random direction (norm-matched)
  success = post-edit action argmax equals the counterfactual-predicted
  action D(shat', g'argmax), on divergent contexts only; plus dlogp(cf act).

Circuit map: per-layer R2/accuracy of b, g, and the COMPOSED direction
d* = D(argmax b, goal); layer-profile of edit effectiveness locates the
composition point (edits stop working after the goal is consumed into d*).

Ablation check: project decision-position activations off each subspace ->
aCE must rise, xCE must not (goal channel is action-specific).
"""
from __future__ import annotations
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from orchard import Net, persona_gen, DST, T, S, TOK_A0

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def fit_probes(H, Y, l2=10.0):
    """Returns decoder V (dy,d), encoder W (d,dy), means."""
    mh, my = H.mean(0), Y.mean(0)
    Hc, Yc = H - mh, Y - my
    d, dy = H.shape[1], Y.shape[1]
    V = np.linalg.solve(Hc.T @ Hc + l2 * np.eye(d), Hc.T @ Yc).T
    W = np.linalg.solve(Yc.T @ Yc + l2 * np.eye(dy), Yc.T @ Hc).T
    return V, W, mh, my


def r2_dec(V, mh, my, H, Y):
    P = (H - mh) @ V.T + my
    return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def subspace_angles(A, B):
    """Principal cosines between column spaces of A and B."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return sv


@torch.no_grad()
def hiddens_prefix(net, toks_pref):
    _, hs = net(toks_pref, return_hidden=True)
    return hs


@torch.no_grad()
def logits_with_edit(net, toks_pref, layer, delta):
    """Forward with delta (B,d) added to hs[layer] at the LAST position.
    hs[0]=embeddings, hs[k]=after block k-1... blocks[j]: hs[j]->hs[j+1]."""
    Tn = toks_pref.shape[1]
    mask = torch.triu(torch.ones(Tn, Tn, device=toks_pref.device,
                                 dtype=torch.bool), 1)
    h = net.tok(toks_pref) + net.pos(torch.arange(Tn,
                                                  device=toks_pref.device))
    for j, blk in enumerate(net.blocks):
        if j == layer:
            h = h.clone()
            h[:, -1] += delta
        h = blk(h, mask)
    if layer == len(net.blocks):
        h = h.clone()
        h[:, -1] += delta
    return net.head(net.lnf(h))[:, -1]


@torch.no_grad()
def full_ce_with_projection(net, toks, layer, P, mh, dec_pos):
    """Project decision-position activations off subspace P (orthonormal
    cols) at layer; return aCE, xCE over the whole batch."""
    Tn = toks.shape[1]
    mask = torch.triu(torch.ones(Tn, Tn, device=toks.device,
                                 dtype=torch.bool), 1)
    h = net.tok(toks) + net.pos(torch.arange(Tn, device=toks.device))
    Pt = torch.tensor(P, device=toks.device, dtype=h.dtype)
    mht = torch.tensor(mh, device=toks.device, dtype=h.dtype)

    def project(h):
        hd = h[:, dec_pos] - mht
        h = h.clone()
        h[:, dec_pos] = hd - (hd @ Pt) @ Pt.T + mht
        return h

    for j, blk in enumerate(net.blocks):
        if j == layer:
            h = project(h)
        h = blk(h, mask)
    if layer == len(net.blocks):
        h = project(h)
    logits = net.head(net.lnf(h))
    lsm = F.log_softmax(logits[:, :-1], -1)
    tgt = toks[:, 1:]
    nll = -lsm.gather(-1, tgt[..., None]).squeeze(-1)
    pos = torch.arange(toks.shape[1] - 1, device=toks.device)
    a_pos = (pos >= 3) & ((pos - 3) % 3 == 0)
    x_pos = (pos >= 4) & ((pos - 4) % 3 == 0)
    return nll[:, a_pos].mean().item(), nll[:, x_pos].mean().item()


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else 'orchard_runs/A/p1_final.pt'
    net = Net().to(DEV)
    net.load_state_dict(torch.load(ckpt, map_location=DEV))
    net.eval()
    ev = persona_gen(2000, np.random.default_rng(999))
    toks = torch.tensor(ev['toks'], device=DEV)
    with torch.no_grad():
        _, hs_full = net(toks, return_hidden=True)
    dec_pos = 3 + 3 * np.arange(T)
    NL = len(hs_full)                       # 7: emb + 6 blocks
    B = ev['wbelief'].reshape(-1, S)
    G = ev['goal_post'].reshape(-1, S)
    shat = B.argmax(1)
    gmax = G.argmax(1)
    dstar = DST[shat, gmax]                 # composed intended direction
    ntr = 1400 * T
    out = {}

    print('=== per-layer probes (test split): dec-R2 / enc-image dim-5 '
          'principal cosines vs dec rows / d* readout acc ===')
    probes = {}
    for li in range(NL):
        H = hs_full[li][:, dec_pos].reshape(-1, 64).cpu().numpy()
        row = {}
        for name, Y in (('world', B), ('goal', G)):
            V, W, mh, my = fit_probes(H[:ntr], Y[:ntr])
            r2 = r2_dec(V, mh, my, H[ntr:], Y[ntr:])
            cos = subspace_angles(V.T, W)   # dec rows vs enc cols
            probes[(li, name)] = (V, W, mh, my)
            row[name] = (r2, cos)
        # composed-direction readout
        Yd = np.eye(3)[dstar]
        Vd, Wd, mhd, myd = fit_probes(H[:ntr], Yd[:ntr])
        acc = ((H[ntr:] - mhd) @ Vd.T + myd).argmax(1) == dstar[ntr:]
        base = np.bincount(dstar[ntr:]).max() / len(dstar[ntr:])
        print(f"L{li}: world R2 {row['world'][0]:.3f} "
              f"(cos {row['world'][1].mean():.2f}) | "
              f"goal R2 {row['goal'][0]:.3f} "
              f"(cos {row['goal'][1].mean():.2f}) | "
              f"d* acc {acc.mean():.3f} (base {base:.2f})")
        out[f'L{li}'] = dict(world_r2=float(row['world'][0]),
                             goal_r2=float(row['goal'][0]),
                             dstar_acc=float(acc.mean()))

    # ---------------- causal edits at single positions ----------------
    print('\n=== causal edits (pooled t* in {4,12,24,40}; divergent '
          'contexts; flip = argmax -> counterfactual action) ===')
    rng = np.random.default_rng(5)
    for target in ('goal', 'world'):
        print(f'--- target: {target}')
        for li in range(NL):
            V, W, mh, my = probes[(li, target)]
            Vp = np.linalg.pinv(V)          # (d, dy) least-norm decoder edit
            flips = {k: [] for k in ('enc', 'dec', 'decNM', 'rand')}
            dlp = {k: [] for k in flips}
            norms = {k: [] for k in flips}
            base_flip = []
            for tstar in (4, 12, 24, 40):
                Lp = 3 + 3 * tstar + 1
                pref = toks[:512, :Lp]
                idx = np.arange(512)
                b = ev['wbelief'][:512, tstar]
                g = ev['goal_post'][:512, tstar]
                sh = b.argmax(1)
                if target == 'goal':
                    tlo = ev['tlo'][:512, tstar]
                    thi = ev['thi'][:512, tstar]
                    gp = g.copy()
                    gp[idx, tlo], gp[idx, thi] = g[idx, thi], g[idx, tlo]
                    cf_act = DST[sh, gp.argmax(1)]
                    fa_act = DST[sh, g.argmax(1)]
                    dy = gp - g
                else:
                    bp = np.roll(b, 1, axis=1)
                    cf_act = DST[bp.argmax(1), g.argmax(1)]
                    fa_act = DST[sh, g.argmax(1)]
                    dy = bp - b
                div = cf_act != fa_act
                if div.sum() < 20:
                    continue
                dy_t = dy[div]
                d_enc = dy_t @ W.T
                d_dec = dy_t @ Vp.T
                ne = np.linalg.norm(d_enc, axis=1, keepdims=True)
                nd = np.linalg.norm(d_dec, axis=1, keepdims=True) + 1e-9
                d_dnm = d_dec / nd * ne
                d_rnd = rng.standard_normal(d_enc.shape)
                d_rnd = d_rnd / np.linalg.norm(d_rnd, axis=1,
                                               keepdims=True) * ne
                pref_d = pref[div]
                with torch.no_grad():
                    lg0 = net(pref_d)[:, -1, TOK_A0:TOK_A0 + 3]
                p0 = F.softmax(lg0, -1).cpu().numpy()
                cfd = cf_act[div]
                base_flip.append((p0.argmax(1) == cfd).mean())
                for k, dvec in (('enc', d_enc), ('dec', d_dec),
                                ('decNM', d_dnm), ('rand', d_rnd)):
                    dt = torch.tensor(dvec, device=DEV, dtype=torch.float32)
                    lg = logits_with_edit(net, pref_d, li, dt)
                    lg = lg[:, TOK_A0:TOK_A0 + 3]
                    p1 = F.softmax(lg, -1).cpu().numpy()
                    flips[k].append((p1.argmax(1) == cfd).mean())
                    dlp[k].append(np.log(p1[np.arange(len(cfd)), cfd] + 1e-9)
                                  .mean()
                                  - np.log(p0[np.arange(len(cfd)), cfd]
                                           + 1e-9).mean())
                    norms[k].append(np.linalg.norm(dvec, axis=1).mean())
            if not base_flip:
                continue
            msg = f'  L{li}: base {np.mean(base_flip):.2f}'
            for k in ('enc', 'dec', 'decNM', 'rand'):
                msg += (f' | {k} flip {np.mean(flips[k]):.2f} '
                        f'dlp {np.mean(dlp[k]):+.2f} '
                        f'|d| {np.mean(norms[k]):.1f}')
            print(msg)
            out[f'edit_{target}_L{li}'] = {
                k: dict(flip=float(np.mean(flips[k])),
                        dlp=float(np.mean(dlp[k])),
                        norm=float(np.mean(norms[k]))) for k in flips}

    # ---------------- subspace ablations ----------------
    print('\n=== subspace ablation at decision positions (aCE/xCE deltas; '
          'baseline from clean forward) ===')
    with torch.no_grad():
        lsm = F.log_softmax(net(toks[:1000])[:, :-1], -1)
        tgt = toks[:1000, 1:]
        nll = -lsm.gather(-1, tgt[..., None]).squeeze(-1)
        pos = torch.arange(toks.shape[1] - 1, device=DEV)
        a_pos = (pos >= 3) & ((pos - 3) % 3 == 0)
        x_pos = (pos >= 4) & ((pos - 4) % 3 == 0)
        a0, x0 = nll[:, a_pos].mean().item(), nll[:, x_pos].mean().item()
    print(f'  clean: aCE {a0:.4f} xCE {x0:.4f}')
    for target in ('goal', 'world'):
        for li in (4, 5, 6):
            V, W, mh, my = probes[(li, target)]
            for name, M in (('enc', W), ('dec', V.T)):
                Q, _ = np.linalg.qr(M)
                aC, xC = full_ce_with_projection(net, toks[:1000], li, Q,
                                                 mh, dec_pos)
                print(f'  {target} L{li} {name}-span ablate: '
                      f'daCE {aC - a0:+.4f}  dxCE {xC - x0:+.4f}')
                out[f'abl_{target}_L{li}_{name}'] = dict(daCE=aC - a0,
                                                         dxCE=xC - x0)
    json.dump(out, open('orchard_probe2.json', 'w'), indent=1)
    print('\nsaved orchard_probe2.json')


if __name__ == '__main__':
    main()
