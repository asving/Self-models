"""Probe iteration 3 — answering four questions (see v2 doc, Iteration 3):
(1) belief co-variation: world-level canonical correlation vs code-level
    block angles;
(2) healing rate: write-then-read decoded-lambda trajectories around a
    one-shot flip, vs the incoming evidence rate (Bayes-speed comparison);
(3) claim structure: does the net carry per-channel claims (m_u, m_v) with
    varying sum, on separate directions from lambda?
(4) two-lever body swap: clamp withdraw-self + raise-other together.
Writes results/rnn_probes3.json, figs/steering_efficacy.png.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World
from oracle import replay_dists
from rnn import ChangelingGRU, TorchWorld, step_features, GOAL_PAIRS, N
from eval_rnn import rollout_record, WORLD_KW, DEV
from probe import load, collect_post, split, slope_r2, N_EPS, BATCH, decoder_r2

T_STAR = 16
RES = {}


# ---------- data with claim coefficients ----------

def coef_from(rec, rep):
    out = {}
    for head, plan, pb, key in ((rec['pu'], rep['piA'], rep['pbar_u'], 'm_u'),
                                (rec['pv'], rep['piB'], rep['pbar_v'], 'm_v')):
        d = plan - pb
        den = (d * d).sum(-1)
        c = np.clip(((head - pb) * d).sum(-1) / np.maximum(den, 1e-12), 0, 1)
        out[key] = np.where(den > 1e-4, c, np.nan).astype(np.float32)
    return out


@torch.no_grad()
def collect_full(model, n_eps, rng):
    Hs, gts = [], []
    for b in range(n_eps // BATCH):
        pair = GOAL_PAIRS[rng.integers(12)]
        w = World(goal_pair=pair, **WORLD_KW)
        rec = rollout_record(model, TorchWorld(w, DEV), pair, BATCH, 7000 + b)
        rep = replay_dists(w, rec['u'], rec['v'], return_beliefs=True)
        r = np.arange(BATCH)[:, None]; t = np.arange(w.T)[None, :]
        dlog = (np.log(rec['pu'][r, t, rec['u']]) - np.log(rep['pbar_u'][r, t, rec['u']])
                + np.log(rep['pbar_v'][r, t, rec['v']]) - np.log(rec['pv'][r, t, rec['v']]))
        lo = np.cumsum(dlog, axis=1)
        rep['lam_logodds'] = np.concatenate([np.zeros((BATCH, 1)), lo[:, :-1]], 1)
        rep['dlog'] = dlog
        rep.update(coef_from(rec, rep))
        rep['iota'] = rec['iota']
        Hs.append(rec['hiddens'].astype(np.float32))
        gts.append(rep)
    H = np.concatenate(Hs)
    gt = {k: np.concatenate([g[k] for g in gts]) for k in gts[0]}
    return H, gt


# ---------- fits ----------

def fit_all(H, gt, tr, te):
    blocks = [('etaA', gt['etaA'].reshape(len(H), -1, 6)),
              ('etaB', gt['etaB'].reshape(len(H), -1, 6)),
              ('drA', gt['drA']), ('drB', gt['drB']),
              ('lam', np.clip(gt['lam_logodds'], -20, 20)[..., None]),
              ('m_u', np.nan_to_num(gt['m_u'], nan=0.5)[..., None]),
              ('m_v', np.nan_to_num(gt['m_v'], nan=0.5)[..., None])]
    G = np.concatenate([b[1].reshape(len(H), H.shape[1], -1) for b in blocks], -1)
    Hf = H.reshape(-1, H.shape[-1])
    Gf = G.reshape(-1, G.shape[-1])
    ep = np.repeat(np.arange(len(H)), H.shape[1])
    m_tr = np.isin(ep, tr)
    Gc = np.concatenate([Gf[m_tr], np.ones((m_tr.sum(), 1))], 1)
    W = np.linalg.lstsq(Gc, Hf[m_tr], rcond=None)[0][:-1].astype(np.float32)
    offs, o = {}, 0
    for k, b in blocks:
        d = b.shape[-1] if b.ndim == 3 else 1
        offs[k] = (o, o + d); o += d
    return W, offs


def block_angles(W, offs, k1, k2):
    a, b = W[slice(*offs[k1])], W[slice(*offs[k2])]
    q1 = np.linalg.qr(a.T)[0]; q2 = np.linalg.qr(b.T)[0]
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def ridge_decoder(H, y, tr, te, alpha=1.0):
    Hf = H.reshape(-1, H.shape[-1]); yf = y.reshape(-1)
    ep = np.repeat(np.arange(len(H)), H.shape[1])
    m_tr, m_te = np.isin(ep, tr), np.isin(ep, te)
    ok = ~np.isnan(yf)
    mu, sd = Hf[m_tr & ok].mean(0), Hf[m_tr & ok].std(0) + 1e-8
    Z = (Hf - mu) / sd
    A = Z[m_tr & ok].T @ Z[m_tr & ok] + alpha * np.eye(H.shape[-1])
    ym = yf[m_tr & ok].mean()
    w = np.linalg.solve(A, Z[m_tr & ok].T @ (yf[m_tr & ok] - ym))
    pred = Z @ w + ym
    m = m_te & ok
    r2 = 1 - ((yf[m] - pred[m]) ** 2).sum() / ((yf[m] - yf[m].mean()) ** 2).sum()
    return (w, mu, sd, ym), float(r2)


# ---------- interventions with per-round decoding ----------

@torch.no_grad()
def rollout_intervene3(model, pair, R, seed, mode, W=None, offs=None,
                       dec=None, belief_scatter=False):
    """mode: sham | flip1 (one-shot lambda flip) | clampL (lambda clamp) |
    withdraw (m_self -> 0 clamp) | raise (m_other -> 1 clamp) |
    swap2 (both levers clamp) | belief (one-shot etaA rotation)."""
    w = World(goal_pair=pair, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    torch.manual_seed(seed)
    T = w.T
    goals = torch.tensor(pair, device=DEV).repeat(R, 1)
    iota = torch.rand(R, device=DEV) < 0.5
    io = iota.cpu().numpy()
    sA = torch.randint(0, N, (R,), device=DEV)
    sB = torch.randint(0, N, (R,), device=DEV)
    h = None; u = v = None
    etaA = np.full((R, N), 1 / N); etaB = etaA.copy()
    drA = etaA.copy(); drB = etaA.copy()
    logodds = np.zeros(R)
    out = {k: np.full((R, T), np.nan, np.float32)
           for k in ('ball', 'm_u', 'm_v', 'lam_dec', 'dlog')}
    for t in range(T):
        x = step_features(u, v, goals, t, T, DEV)
        lu, lv, h = model.step(x, h)
        pbar_u = etaA @ w.EA; pbar_v = etaB @ w.EB
        scA = np.einsum('ra,rb,abu->ru', drA, etaB, w.M[t])
        scB = np.einsum('ra,rb,abv->rv', etaA, drB, w.N[t])
        piA = pbar_u * (scA / (scA.max(1, keepdims=True) + 1e-300)) ** w.kappa
        piB = pbar_v * (scB / (scB.max(1, keepdims=True) + 1e-300)) ** w.kappa
        piA /= piA.sum(1, keepdims=True); piB /= piB.sum(1, keepdims=True)
        one_shot = t == T_STAR and mode in ('flip1', 'belief')
        clamped = t >= T_STAR and mode in ('clampL', 'withdraw', 'raise', 'swap2')
        if one_shot or clamped:
            hn = h[0].cpu().numpy()
            dh = np.zeros((R, hn.shape[-1]), np.float32)
            if mode == 'belief':
                g = etaA.astype(np.float32); gp = np.roll(g, 3, 1)
                dh = (gp - g) @ W[slice(*offs['etaA'])]
                out['dpred_u'] = np.log(gp @ w.EA) - np.log(pbar_u)
                out['head_u_pre'] = F.log_softmax(lu, -1).cpu().numpy()
                out['head_v_pre'] = F.log_softmax(lv, -1).cpu().numpy()
            if mode in ('flip1', 'clampL'):
                dl = np.clip(-2.0 * logodds, -40, 40).astype(np.float32)
                dh = dl[:, None] * W[slice(*offs['lam'])]
            if mode in ('withdraw', 'swap2'):
                cur = cur_coef(lu, lv, io, piA, piB, pbar_u, pbar_v, 'self')
                blk = np.where(io[:, None], W[slice(*offs['m_u'])],
                               W[slice(*offs['m_v'])])
                dh = dh + (0.0 - cur)[:, None] * blk
            if mode in ('raise', 'swap2'):
                cur = cur_coef(lu, lv, io, piA, piB, pbar_u, pbar_v, 'oth')
                blk = np.where(io[:, None], W[slice(*offs['m_v'])],
                               W[slice(*offs['m_u'])])
                dh = dh + (1.0 - cur)[:, None] * blk
            h = h + torch.tensor(dh.astype(np.float32), device=DEV).unsqueeze(0)
            lu = model.head_u(h[0]); lv = model.head_v(h[0])
            if mode == 'belief':
                out['head_u_post'] = F.log_softmax(lu, -1).cpu().numpy()
                out['head_v_post'] = F.log_softmax(lv, -1).cpu().numpy()
        pu, pv = F.softmax(lu, -1), F.softmax(lv, -1)
        # per-round readouts
        if dec is not None:
            wd, mu_, sd_, ym = dec
            out['lam_dec'][:, t] = ((h[0].cpu().numpy() - mu_) / sd_) @ wd + ym
        cm = coef_np(pu.cpu().numpy(), piA, pbar_u)
        out['m_u'][:, t] = cm
        out['m_v'][:, t] = coef_np(pv.cpu().numpy(), piB, pbar_v)
        u_net = torch.multinomial(pu, 1).squeeze(1)
        v_net = torch.multinomial(pv, 1).squeeze(1)
        u_env, v_env = tw.emit(sA, sB)
        u = torch.where(iota, u_net, u_env)
        v = torch.where(iota, v_env, v_net)
        un, vn = u.cpu().numpy(), v.cpu().numpy()
        r = np.arange(R)
        d = (np.log(pu.cpu().numpy()[r, un]) - np.log(pbar_u[r, un])
             + np.log(pbar_v[r, vn]) - np.log(pv.cpu().numpy()[r, vn]))
        out['dlog'][:, t] = d
        logodds += d
        TAg = w.TA[un, vn]; TBg = w.TB[un, vn]
        etaA = etaA * w.EA[:, un].T
        etaA = np.einsum('rs,rst->rt', etaA, TAg); etaA /= etaA.sum(1, keepdims=True)
        etaB = etaB * w.EB[:, vn].T
        etaB = np.einsum('rs,rst->rt', etaB, TBg); etaB /= etaB.sum(1, keepdims=True)
        drA = np.einsum('rs,rst->rt', drA, TAg)
        drB = np.einsum('rs,rst->rt', drB, TBg)
        sA, sB = tw.trans(sA, sB, u, v)
        out['ball'][:, t] = tw.ball(sA, sB).float().cpu().numpy()
    out['iota'] = io
    out['logodds_true'] = logodds
    return out


def coef_np(head, plan, pb):
    d = plan - pb
    den = (d * d).sum(1)
    c = np.clip(((head - pb) * d).sum(1) / np.maximum(den, 1e-12), 0, 1)
    return np.where(den > 1e-4, c, np.nan)


def cur_coef(lu, lv, io, piA, piB, pbu, pbv, which):
    pu = F.softmax(lu, -1).cpu().numpy(); pv = F.softmax(lv, -1).cpu().numpy()
    cu = np.nan_to_num(coef_np(pu, piA, pbu), nan=0.5)
    cv = np.nan_to_num(coef_np(pv, piB, pbv), nan=0.5)
    if which == 'self':
        return np.where(io, cu, cv).astype(np.float32)
    return np.where(io, cv, cu).astype(np.float32)


def main():
    rng = np.random.default_rng(11)
    post = load('post_6000')
    H, gt = collect_full(post, N_EPS, rng)
    tr, te = split(N_EPS)

    # (1) world-level co-variation vs code-level separation
    A = gt['etaA'].reshape(-1, 6) - gt['etaA'].reshape(-1, 6).mean(0)
    B = gt['etaB'].reshape(-1, 6) - gt['etaB'].reshape(-1, 6).mean(0)
    qa = np.linalg.qr(A)[0][:, :5]; qb = np.linalg.qr(B)[0][:, :5]
    cc = np.linalg.svd(qa.T @ qb, compute_uv=False)
    RES['value_canoncorr_etaA_etaB'] = [round(float(x), 3) for x in cc[:3]]
    W, offs = fit_all(H, gt, tr, te)
    RES['code_angles_etaA_etaB'] = [round(float(x), 1)
                                    for x in block_angles(W, offs, 'etaA', 'etaB')]
    for pairk in (('lam', 'm_u'), ('lam', 'm_v'), ('m_u', 'm_v')):
        RES[f'angle_{pairk[0]}_{pairk[1]}'] = round(
            float(block_angles(W, offs, *pairk)[0]), 1)

    # (3) claim structure: decoders + sum variability
    for var in ('m_u', 'm_v'):
        _, r2 = ridge_decoder(H, gt[var], tr, te)
        RES[f'r2_dec_{var}'] = round(r2, 3)
    s = gt['m_u'] + gt['m_v']
    RES['claim_sum_mean_std'] = [round(float(np.nanmean(s)), 3),
                                 round(float(np.nanstd(s)), 3)]
    RES['claim_sum_frac_gt_1.3'] = round(float(np.nanmean(s > 1.3)), 3)
    dec_lam, r2l = ridge_decoder(H, np.clip(gt['lam_logodds'], -20, 20), tr, te)
    RES['r2_dec_lam'] = round(r2l, 3)
    print(json.dumps(RES, indent=1), flush=True)

    # (2) write-then-read healing + evidence rate
    conds = {}
    for mode in ('sham', 'flip1', 'clampL', 'withdraw', 'raise', 'swap2'):
        conds[mode] = rollout_intervene3(post, (0, 2), 1024, 99, mode,
                                         W, offs, dec_lam)
    sham = conds['sham']
    ev_rate = float(np.nanmean(np.abs(sham['dlog'][:, T_STAR:T_STAR + 4])))
    lam_true_t = np.nanmedian(np.abs(np.clip(
        np.cumsum(sham['dlog'], 1), -40, 40))[:, T_STAR - 1])
    RES['evidence_rate_nats_per_round'] = round(float(ev_rate), 3)
    RES['median_abs_logodds_at_tstar'] = round(float(lam_true_t), 2)
    RES['bayes_heal_rounds_expected'] = round(
        float(2 * lam_true_t / max(ev_rate, 1e-9)), 1)
    heal = {}
    for mode in ('sham', 'flip1'):
        sgn = np.where(conds[mode]['iota'], 1, -1)[:, None]
        heal[mode] = np.nanmedian(sgn * conds[mode]['lam_dec'], 0)
    d0 = heal['sham'][T_STAR] - heal['flip1'][T_STAR]
    rec_rounds = -1
    for k in range(T_STAR, sham['ball'].shape[1]):
        if abs(heal['sham'][k] - heal['flip1'][k]) < 0.2 * abs(d0):
            rec_rounds = k - T_STAR
            break
    RES['writeread_flip_depth_nats'] = round(float(d0), 2)
    RES['writeread_heal_rounds_observed'] = rec_rounds

    # (4) levers
    for mode in ('clampL', 'withdraw', 'raise', 'swap2'):
        o = conds[mode]
        cs = np.where(o['iota'][:, None], o['m_u'], o['m_v'])
        co = np.where(o['iota'][:, None], o['m_v'], o['m_u'])
        RES[f'lever_{mode}'] = {
            'coef_self_rest': round(float(np.nanmean(cs[:, T_STAR:])), 3),
            'coef_oth_rest': round(float(np.nanmean(co[:, T_STAR:])), 3),
            'occ_rest': round(float(np.nanmean(o['ball'][:, T_STAR:])), 3)}
    css = np.where(sham['iota'][:, None], sham['m_u'], sham['m_v'])
    cos_ = np.where(sham['iota'][:, None], sham['m_v'], sham['m_u'])
    RES['lever_sham'] = {
        'coef_self_rest': round(float(np.nanmean(css[:, T_STAR:])), 3),
        'coef_oth_rest': round(float(np.nanmean(cos_[:, T_STAR:])), 3),
        'occ_rest': round(float(np.nanmean(sham['ball'][:, T_STAR:])), 3)}
    print(json.dumps({k: v for k, v in RES.items()
                      if k.startswith(('lever', 'evidence', 'median', 'bayes',
                                       'writeread'))},
                     indent=1, default=float), flush=True)

    # belief steering scatter (for the efficacy figure)
    ob = rollout_intervene3(post, (0, 2), 512, 47, 'belief', W, offs, dec_lam)
    sl, r2 = slope_r2(ob['head_u_post'] - ob['head_u_pre'], ob['dpred_u'])
    RES['belief_steering'] = {'slope': round(sl, 3), 'r2': round(r2, 3)}

    with open('results/rnn_probes3.json', 'w') as f:
        json.dump(RES, f, indent=1, default=float)

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    x = ob['dpred_u'].ravel(); y = (ob['head_u_post'] - ob['head_u_pre']).ravel()
    ii = np.random.default_rng(0).choice(len(x), min(4000, len(x)),
                                         replace=False)
    axes[0].scatter(x[ii], y[ii], s=2, alpha=.15)
    xs = np.linspace(x.min(), x.max(), 2)
    axes[0].plot(xs, sl * xs, 'r-', lw=1.5, label=f'slope {sl:.2f}, R² {r2:.2f}')
    axes[0].plot(xs, xs, 'k--', lw=.8, label='perfect transfer')
    axes[0].set_xlabel('predicted Δlog p (exact filter)')
    axes[0].set_ylabel('observed Δlog p (network head)')
    axes[0].set_title('belief steering, post net'); axes[0].legend(fontsize=7)
    tt = np.arange(len(heal['sham']))
    axes[1].plot(tt, heal['sham'], 'gray', lw=1.8, label='sham')
    axes[1].plot(tt, heal['flip1'], 'C3', lw=1.8, label='one-shot λ flip')
    axes[1].axvline(T_STAR, ls=':', c='k')
    axes[1].set_title(f"write-then-read: decoded λ (heals in {rec_rounds} rounds;\n"
                      f"Bayes speed would need ~{RES['bayes_heal_rounds_expected']})")
    axes[1].set_xlabel('round t'); axes[1].set_ylabel('decoded λ log-odds (signed)')
    axes[1].legend(fontsize=7)
    for mode, c in (('sham', 'gray'), ('clampL', 'C0'), ('swap2', 'C3')):
        o = conds[mode]
        cs = np.where(o['iota'][:, None], o['m_u'], o['m_v'])
        co = np.where(o['iota'][:, None], o['m_v'], o['m_u'])
        axes[2].plot(np.nanmean(cs, 0), c, lw=1.8, label=f'{mode} self')
        axes[2].plot(np.nanmean(co, 0), c, lw=1.8, ls='--', label=f'{mode} other')
    axes[2].axvline(T_STAR, ls=':', c='k')
    axes[2].set_title('claims under clamped levers')
    axes[2].set_xlabel('round t'); axes[2].set_ylabel('plan coefficient')
    axes[2].legend(fontsize=6)
    modes = ['sham', 'clampL', 'withdraw', 'raise', 'swap2']
    occ = [RES[f'lever_{m}']['occ_rest'] for m in modes]
    axes[3].bar(modes, occ, color=['gray', 'C0', 'C2', 'C4', 'C3'])
    axes[3].set_ylabel('occupancy (rest of episode)')
    axes[3].set_title('behavioral cost of each lever')
    axes[3].tick_params(axis='x', labelsize=7)
    fig.tight_layout(); fig.savefig('figs/steering_efficacy.png', dpi=160)
    print('wrote results/rnn_probes3.json, figs/steering_efficacy.png', flush=True)


if __name__ == '__main__':
    main()
