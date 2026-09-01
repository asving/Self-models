"""Storage FORMAT of the identity integrator (v3 doc, Iteration 8).

F1 per-round decoders: R2(t) + rotation of the code direction across rounds.
F2 donor swap in PER-ROUND-fit subspace vs pooled subspace (one-shot and
   clamped) — if the rotating-coordinates story is right, per-round works.
F3 perturbation survival: matched-norm deltas along {full-swap difference,
   pooled decoder dir, per-round decoder dir, random}, propagated through
   paired identical-token rollouts: ||dh_t|| decay + lambda-content.
F4 unit census: update-gate persistence z_i and participation ratio of the
   identity-difference vector (committee vs population).
Writes results/rnn_format.json, figs/format.png. cwd = 08_changeling.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World
from rnn import TorchWorld, step_features, N
from eval_rnn import WORLD_KW, DEV
from probe import load, split
from probe3 import collect_full, coef_np
from whitebox_lambda import prefix, Filt, continue_closed
from lambda_circuit import gru_step_manual

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
RES = {}
T_STAR = 16
R = 1024
PAIR = (0, 2)


def ridge_fit(Z, y, alpha=1.0):
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    ym = y.mean()
    return np.linalg.solve(A, Z.T @ (y - ym)), ym


def per_round_decoders(H, lam, tr, te):
    ws, r2s = [], []
    y = np.clip(lam, -20, 20)
    mu = H.reshape(-1, 256).mean(0)
    sd = H.reshape(-1, 256).std(0) + 1e-8
    for t in range(32):
        Z = (H[:, t] - mu) / sd
        w, ym = ridge_fit(Z[tr], y[tr, t])
        pred = Z[te] @ w + ym
        r2s.append(round(float(1 - ((y[te, t] - pred) ** 2).sum()
                               / ((y[te, t] - y[te, t].mean()) ** 2).sum() + 1e-12), 3))
        ws.append(w / (np.linalg.norm(w) + 1e-12))
    return np.stack(ws), r2s, mu, sd


@torch.no_grad()
def swap_subspace(model, w, tw, st, P, clamp=False):
    """Donor swap in span(P) at entry (and every round if clamp)."""
    donor = match_pairs(st)
    Pt = torch.tensor(P, dtype=torch.float32, device=DEV)
    h, u, v = st['h'].clone(), st['u'].clone(), st['v'].clone()
    sA, sB = st['sA'].clone(), st['sB'].clone()
    iota = st['iota']
    f = Filt.__new__(Filt); f.w = w
    for k in ('etaA', 'etaB', 'drA', 'drB'):
        setattr(f, k, getattr(st['f'], k).copy())
    out = {k: np.full((R, w.T - T_STAR), np.nan, np.float32)
           for k in ('m_u', 'm_v')}
    dn = torch.tensor(donor, device=DEV)
    for j, t in enumerate(range(T_STAR, w.T)):
        if j == 0 or clamp:
            proj = h[0] @ Pt.T
            h = (h[0] + (proj[dn] - proj) @ Pt).unsqueeze(0)
        x = step_features(u, v, st['goals'], t, w.T, DEV)
        lu, lv, h = model.step(x, h)
        pbar_u, pbar_v, piA, piB = f.dists(t)
        pu, pv = F.softmax(lu, -1), F.softmax(lv, -1)
        out['m_u'][:, j] = coef_np(pu.cpu().numpy(), piA, pbar_u)
        out['m_v'][:, j] = coef_np(pv.cpu().numpy(), piB, pbar_v)
        u_net = torch.multinomial(pu, 1).squeeze(1)
        v_net = torch.multinomial(pv, 1).squeeze(1)
        u_env, v_env = tw.emit(sA, sB)
        u = torch.where(iota, u_net, u_env)
        v = torch.where(iota, v_env, v_net)
        f.update(u.cpu().numpy(), v.cpu().numpy())
        sA, sB = tw.trans(sA, sB, u, v)
    io = iota.cpu().numpy()[:, None]
    return (np.nanmean(np.where(io, out['m_u'], out['m_v']), 0),
            np.nanmean(np.where(io, out['m_v'], out['m_u']), 0))


def match_pairs(st):
    io = st['iota'].cpu().numpy()
    eta = np.concatenate([st['f'].etaA, st['f'].etaB], 1)
    donor = np.arange(R)
    idxA, idxB = np.where(io)[0], np.where(~io)[0]
    for i in idxA:
        donor[i] = idxB[np.argmin(np.abs(eta[idxB] - eta[i]).sum(1))]
    for i in idxB:
        donor[i] = idxA[np.argmin(np.abs(eta[idxA] - eta[i]).sum(1))]
    return donor


@torch.no_grad()
def survival(model, w, st, deltas, n_rounds=10):
    """Propagate perturbations through paired identical-token rollouts.
    deltas: dict name -> (R, 256) initial dh. Neutral-token feed from base."""
    base_h = st['h'].clone()
    hs = {k: st['h'] + torch.tensor(d, dtype=torch.float32,
                                    device=DEV).unsqueeze(0)
          for k, d in deltas.items()}
    u, v = st['u'].clone(), st['v'].clone()
    f = Filt.__new__(Filt); f.w = w
    for k in ('etaA', 'etaB', 'drA', 'drB'):
        setattr(f, k, getattr(st['f'], k).copy())
    curves = {k: [float(np.linalg.norm(d, axis=1).mean())] for k, d in deltas.items()}
    h0 = base_h
    for j, t in enumerate(range(T_STAR, min(T_STAR + n_rounds, w.T))):
        x = step_features(u, v, st['goals'], t, w.T, DEV)
        lu, lv, h0 = model.step(x, h0)
        pu = F.softmax(lu, -1).cpu().numpy()
        pv = F.softmax(lv, -1).cpu().numpy()
        pbar_u, pbar_v, _, _ = f.dists(t)
        un = np.argmin(np.abs(np.log(pu + 1e-12) - np.log(pbar_u)), 1)
        vn = np.argmin(np.abs(np.log(pv + 1e-12) - np.log(pbar_v)), 1)
        for k in hs:
            _, _, hs[k] = model.step(x, hs[k])
            d = (hs[k] - h0)[0].cpu().numpy()
            curves[k].append(float(np.linalg.norm(d, axis=1).mean()))
        f.update(un, vn)
        u = torch.tensor(un, device=DEV); v = torch.tensor(vn, device=DEV)
    return curves


@torch.no_grad()
def unit_census(model, w, tw, st):
    """Mean update-gate per unit + identity-difference participation."""
    # z-gate census on one natural continuation round
    zs = []
    h, u, v = st['h'].clone(), st['u'].clone(), st['v'].clone()
    for t in range(T_STAR, min(T_STAR + 8, w.T)):
        x = step_features(u, v, st['goals'], t, w.T, DEV)
        e = torch.relu(model.inp(x))
        gi = e @ model.gru.weight_ih_l0.T + model.gru.bias_ih_l0
        gh = h[0] @ model.gru.weight_hh_l0.T + model.gru.bias_hh_l0
        i_r, i_z, i_n = gi.chunk(3, -1)
        h_r, h_z, h_n = gh.chunk(3, -1)
        z = torch.sigmoid(i_z + h_z)
        zs.append(z.cpu().numpy())
        _, _, h = model.step(x, h)
        # natural closed loop not needed; teacher-force same tokens
    zbar = np.concatenate(zs).mean(0)
    donor = match_pairs(st)
    d = (st['h'][0][torch.tensor(donor, device=DEV)] - st['h'][0]).cpu().numpy()
    d2 = (d ** 2).mean(0)
    pr = float((d2.sum() ** 2) / ((d2 ** 2).sum() * len(d2)) * len(d2))
    pr_eff = float((d2.sum() ** 2) / (d2 ** 2).sum())
    return zbar, d2, pr_eff


@torch.no_grad()
def main():
    rng = np.random.default_rng(13)
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)

    # F1: per-round decoders + rotation
    H, gt = collect_full(post, 2048, rng)
    tr, te = split(2048)
    Wt, r2t, mu, sd = per_round_decoders(H, gt['lam_logodds'], tr, te)
    RES['F1_r2_by_round'] = r2t
    ang = Wt @ Wt.T
    RES['F1_cos_w16_w20_w24_w31_vs_w16'] = [round(float(ang[16, k]), 3)
                                            for k in (16, 20, 24, 31)]
    RES['F1_cos_adjacent_mean'] = round(float(np.mean(
        [ang[t, t + 1] for t in range(8, 30)])), 3)
    RES['F1_cos_pooled_vs_w16'] = None
    print('F1:', {k: RES[k] for k in ('F1_r2_by_round',) if False} or
          {'r2_t16': r2t[16], 'cos_row': RES['F1_cos_w16_w20_w24_w31_vs_w16'],
           'adj': RES['F1_cos_adjacent_mean']}, flush=True)

    # pooled decoder for comparison
    Zf = (H.reshape(-1, 256) - mu) / sd
    yf = np.clip(gt['lam_logodds'], -20, 20).reshape(-1)
    ep = np.repeat(np.arange(2048), 32)
    wp, ymp = ridge_fit(Zf[np.isin(ep, tr)], yf[np.isin(ep, tr)])
    wp_u = wp / np.linalg.norm(wp)
    RES['F1_cos_pooled_vs_w16'] = round(float(np.abs(wp_u @ Wt[16])), 3)

    # F2: swaps — per-round-fit subspace vs pooled, one-shot and clamped
    st = prefix(post, w, tw, seed=77)
    sham_self, sham_oth = swap_subspace(model=post, w=w, tw=tw, st=st,
                                        P=np.zeros((1, 256)), clamp=False)
    RES['F2_sham'] = {'self_rest': round(float(np.nanmean(sham_self)), 3),
                      'oth_rest': round(float(np.nanmean(sham_oth)), 3)}
    # per-round subspace: top-k per-round dirs around t* (w_t for t=16..31,
    # applied per current round when clamped; one-shot uses w_16 block)
    for name, P, clamp in (
            ('perround_k1_oneshot', Wt[16:17] / sd, False),
            ('perround_k1_clamp', None, True),      # handled below
            ('pooled_k1_oneshot', wp_u[None] / sd, False)):
        if P is None:
            continue
        Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
        s, o = swap_subspace(post, w, tw, st, Pn, clamp=False)
        RES[f'F2_{name}'] = {'self_rest': round(float(np.nanmean(s)), 3),
                             'oth_rest': round(float(np.nanmean(o)), 3),
                             'self_t0': round(float(s[0]), 3),
                             'oth_t0': round(float(o[0]), 3)}
        print(f'F2 {name}:', RES[f'F2_{name}'], flush=True)
    # clamped per-round: swap along w_t each round t (rotating clamp)
    class RotP:
        pass
    # implement rotating clamp inline
    donor = match_pairs(st)
    dn = torch.tensor(donor, device=DEV)
    h, u, v = st['h'].clone(), st['u'].clone(), st['v'].clone()
    sA, sB = st['sA'].clone(), st['sB'].clone()
    iota = st['iota']
    f = Filt.__new__(Filt); f.w = w
    for k in ('etaA', 'etaB', 'drA', 'drB'):
        setattr(f, k, getattr(st['f'], k).copy())
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEV)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=DEV)
    out = {k: np.full((R, w.T - T_STAR), np.nan, np.float32)
           for k in ('m_u', 'm_v')}
    for j, t in enumerate(range(T_STAR, w.T)):
        wt = torch.tensor(Wt[t], dtype=torch.float32, device=DEV)
        z = (h[0] - mu_t) / sd_t
        proj = z @ wt
        z = z + (proj[dn] - proj)[:, None] * wt[None]
        h = (z * sd_t + mu_t).unsqueeze(0)
        x = step_features(u, v, st['goals'], t, w.T, DEV)
        lu, lv, h = post.step(x, h)
        pbar_u, pbar_v, piA, piB = f.dists(t)
        pu, pv = F.softmax(lu, -1), F.softmax(lv, -1)
        out['m_u'][:, j] = coef_np(pu.cpu().numpy(), piA, pbar_u)
        out['m_v'][:, j] = coef_np(pv.cpu().numpy(), piB, pbar_v)
        u_net = torch.multinomial(pu, 1).squeeze(1)
        v_net = torch.multinomial(pv, 1).squeeze(1)
        u_env, v_env = tw.emit(sA, sB)
        u = torch.where(iota, u_net, u_env)
        v = torch.where(iota, v_env, v_net)
        f.update(u.cpu().numpy(), v.cpu().numpy())
        sA, sB = tw.trans(sA, sB, u, v)
    io = iota.cpu().numpy()[:, None]
    RES['F2_perround_k1_rotclamp'] = {
        'self_rest': round(float(np.nanmean(
            np.where(io, out['m_u'], out['m_v']))), 3),
        'oth_rest': round(float(np.nanmean(
            np.where(io, out['m_v'], out['m_u']))), 3)}
    print('F2 rotclamp:', RES['F2_perround_k1_rotclamp'], flush=True)

    # F3: survival of matched-norm perturbations
    donor_np = match_pairs(st)
    d_full = (st['h'][0][torch.tensor(donor_np, device=DEV)]
              - st['h'][0]).cpu().numpy()
    scale = np.linalg.norm(d_full, axis=1, keepdims=True).mean()
    rnd = rng.standard_normal((R, 256))
    deltas = {
        'full_swap_diff': d_full,
        'pooled_dir': np.tile(wp_u / sd, (R, 1))
                      / np.linalg.norm(wp_u / sd) * scale
                      * np.sign(rng.standard_normal((R, 1))),
        'perround_dir': np.tile(Wt[16] / sd, (R, 1))
                        / np.linalg.norm(Wt[16] / sd) * scale
                        * np.sign(rng.standard_normal((R, 1))),
        'random': rnd / np.linalg.norm(rnd, axis=1, keepdims=True) * scale}
    curves = survival(post, w, st, deltas, n_rounds=10)
    RES['F3_survival'] = {k: [round(x / v[0], 3) for x in v]
                          for k, v in curves.items()}
    print('F3:', {k: v[:6] for k, v in RES['F3_survival'].items()}, flush=True)

    # F4: unit census
    zbar, d2, pr_eff = unit_census(post, w, tw, st)
    RES['F4_zbar_frac_gt_0.9'] = round(float((zbar > 0.9).mean()), 3)
    RES['F4_zbar_top10'] = [round(float(x), 3) for x in np.sort(zbar)[-10:]]
    RES['F4_identity_diff_participation_units'] = round(pr_eff, 1)
    RES['F4_lambda_decoder_participation_units'] = round(
        float((wp_u ** 2).sum() ** 2 / (wp_u ** 4).sum()), 1)
    print('F4:', {k: RES[k] for k in RES if k.startswith('F4')}, flush=True)

    with open('results/rnn_format.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    im = axes[0].imshow(np.abs(Wt @ Wt.T), cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('|cos| between per-round λ directions')
    axes[0].set_xlabel('round'); axes[0].set_ylabel('round')
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    for k, v in RES['F3_survival'].items():
        axes[1].plot(v, 'o-', label=k, lw=1.5)
    axes[1].set_yscale('log'); axes[1].set_xlabel('rounds after perturbation')
    axes[1].set_ylabel('relative ||Δh||'); axes[1].set_title('F3: what the dynamics preserve')
    axes[1].legend(fontsize=7)
    axes[2].hist(zbar, bins=40)
    axes[2].set_xlabel('mean update gate z̄ per unit')
    axes[2].set_title(f'F4: persistence census; identity-diff PR '
                      f'≈ {pr_eff:.0f} units')
    fig.tight_layout(); fig.savefig('figs/format.png', dpi=160)
    print('wrote results/rnn_format.json, figs/format.png', flush=True)


if __name__ == '__main__':
    main()
