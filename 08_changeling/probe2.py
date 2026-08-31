"""Probe iteration 2 (v2 doc, Q1/Q5 follow-up): JOINT encoder so each
variable's block is partialled against its correlates; belief-intervention
selectivity re-test; body-swap one-shot vs CLAMPED with time-resolved
coefficients (separates 'causally inert' from 'heals against incoming
evidence'). Writes results/rnn_probes2.json, figs/rnn_bodyswap.png.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World
from rnn import ChangelingGRU, TorchWorld, step_features, N
from eval_rnn import WORLD_KW, DEV
from probe import (load, collect_pre, collect_post, split, slope_r2, N_EPS)

T_STAR = 16


def joint_encoder(gt, H, tr, te, with_lam):
    blocks = [('etaA', 6), ('etaB', 6), ('drA', 6), ('drB', 6)]
    Gs = [gt[k].reshape(-1, 6) for k, _ in blocks]
    if with_lam:
        blocks.append(('lam', 1))
        Gs.append(np.clip(gt['lam_logodds'], -20, 20).reshape(-1, 1))
    G = np.concatenate(Gs, 1)
    Hf = H.reshape(-1, H.shape[-1])
    n_ep, T = H.shape[0], H.shape[1]
    ep = np.repeat(np.arange(n_ep), T)
    m_tr = np.isin(ep, tr)
    Gc = np.concatenate([G[m_tr], np.ones((m_tr.sum(), 1))], 1)
    Wb = np.linalg.lstsq(Gc, Hf[m_tr], rcond=None)[0]
    W = Wb[:-1]
    offs = {}
    o = 0
    for k, d in blocks:
        offs[k] = (o, o + d)
        o += d
    return W, offs


@torch.no_grad()
def interv_rollout2(model, pair, R, seed, mode, W=None, offs=None,
                    clamp=False):
    """mode: 'sham' | 'belief' | 'lamswap'. Belief: rotate etaA by 3 at
    T_STAR using the JOINT etaA block. Lamswap: flip current log-odds along
    the joint lam block at T_STAR (and every later round if clamp)."""
    w = World(goal_pair=pair, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    torch.manual_seed(seed)
    T = w.T
    goals = torch.tensor(pair, device=DEV).repeat(R, 1)
    iota = torch.rand(R, device=DEV) < 0.5
    sA = torch.randint(0, N, (R,), device=DEV)
    sB = torch.randint(0, N, (R,), device=DEV)
    h = None; u = v = None
    etaA = np.full((R, N), 1 / N); etaB = etaA.copy()
    drA = etaA.copy(); drB = etaA.copy()
    logodds = np.zeros(R)
    out = {'ball': np.zeros((R, T), np.float32),
           'coef_u': np.full((R, T), np.nan, np.float32),
           'coef_v': np.full((R, T), np.nan, np.float32)}
    for t in range(T):
        x = step_features(u, v, goals, t, T, DEV)
        lu, lv, h = model.step(x, h)
        pbar_u = etaA @ w.EA
        pbar_v = etaB @ w.EB
        scA = np.einsum('ra,rb,abu->ru', drA, etaB, w.M[t])
        scB = np.einsum('ra,rb,abv->rv', etaA, drB, w.N[t])
        piA = pbar_u * (scA / (scA.max(1, keepdims=True) + 1e-300)) ** w.kappa
        piB = pbar_v * (scB / (scB.max(1, keepdims=True) + 1e-300)) ** w.kappa
        piA /= piA.sum(1, keepdims=True); piB /= piB.sum(1, keepdims=True)
        hit = (t == T_STAR and mode != 'sham') or \
              (clamp and mode == 'lamswap' and t > T_STAR)
        if hit:
            if mode == 'belief':
                g = etaA.astype(np.float32)
                gp = np.roll(g, 3, axis=1)
                s, e = offs['etaA']
                dh = (gp - g) @ W[s:e]
                out['dpred_u'] = np.log(gp @ w.EA) - np.log(pbar_u)
            else:
                s, e = offs['lam']
                dl = np.clip(-2.0 * logodds, -40, 40).astype(np.float32)
                dh = dl[:, None] * W[s:e]
            if t == T_STAR:
                out['head_u_pre'] = F.log_softmax(lu, -1).cpu().numpy()
                out['head_v_pre'] = F.log_softmax(lv, -1).cpu().numpy()
            h = h + torch.tensor(dh, dtype=torch.float32, device=DEV).unsqueeze(0)
            lu = model.head_u(h[0]); lv = model.head_v(h[0])
            if t == T_STAR:
                out['head_u_post'] = F.log_softmax(lu, -1).cpu().numpy()
                out['head_v_post'] = F.log_softmax(lv, -1).cpu().numpy()
        pu, pv = F.softmax(lu, -1), F.softmax(lv, -1)
        u_net = torch.multinomial(pu, 1).squeeze(1)
        v_net = torch.multinomial(pv, 1).squeeze(1)
        u_env, v_env = tw.emit(sA, sB)
        u = torch.where(iota, u_net, u_env)
        v = torch.where(iota, v_env, v_net)
        for head, plan, pb, key in ((pu.cpu().numpy(), piA, pbar_u, 'coef_u'),
                                    (pv.cpu().numpy(), piB, pbar_v, 'coef_v')):
            d = plan - pb
            den = (d * d).sum(1)
            ok = den > 1e-4
            out[key][ok, t] = np.clip(((head - pb) * d).sum(1)[ok] / den[ok], 0, 1)
        un, vn = u.cpu().numpy(), v.cpu().numpy()
        r = np.arange(R)
        logodds += (np.log(pu.cpu().numpy()[r, un]) - np.log(pbar_u[r, un])
                    + np.log(pbar_v[r, vn]) - np.log(pv.cpu().numpy()[r, vn]))
        TAg = w.TA[un, vn]; TBg = w.TB[un, vn]
        etaA = etaA * w.EA[:, un].T
        etaA = np.einsum('rs,rst->rt', etaA, TAg)
        etaA /= etaA.sum(1, keepdims=True)
        etaB = etaB * w.EB[:, vn].T
        etaB = np.einsum('rs,rst->rt', etaB, TBg)
        etaB /= etaB.sum(1, keepdims=True)
        drA = np.einsum('rs,rst->rt', drA, TAg)
        drB = np.einsum('rs,rst->rt', drB, TBg)
        sA, sB = tw.trans(sA, sB, u, v)
        out['ball'][:, t] = tw.ball(sA, sB).float().cpu().numpy()
    out['iota'] = iota.cpu().numpy()
    return out


def main():
    rng = np.random.default_rng(7)
    res = {}
    tr, te = split(N_EPS)

    pre = load('pre_final')
    post = load('post_6000')
    H_pre, gt_pre = collect_pre(pre, N_EPS)
    H_post, gt_post, _ = collect_post(post, N_EPS, rng)
    W_pre, offs_pre = joint_encoder(gt_pre, H_pre, tr, te, with_lam=False)
    W_post, offs_post = joint_encoder(gt_post, H_post, tr, te, with_lam=True)

    # belief intervention, joint blocks
    for name, model, W, offs in (('pre_final', pre, W_pre, offs_pre),
                                 ('post_6000', post, W_post, offs_post)):
        o = interv_rollout2(model, (0, 2), 512, 47, 'belief', W, offs)
        sl, r2 = slope_r2(o['head_u_post'] - o['head_u_pre'], o['dpred_u'])
        slc, r2c = slope_r2(o['head_v_post'] - o['head_v_pre'], o['dpred_u'])
        res[f'belief_joint_{name}'] = {
            'slope': round(sl, 3), 'r2': round(r2, 3),
            'ctrl_slope': round(slc, 3), 'ctrl_r2': round(r2c, 3)}
        print(name, res[f'belief_joint_{name}'], flush=True)

    # body swap: sham / one-shot / clamped, time-resolved
    conds = {'sham': interv_rollout2(post, (0, 2), 1024, 99, 'sham'),
             'oneshot': interv_rollout2(post, (0, 2), 1024, 99, 'lamswap',
                                        W_post, offs_post),
             'clamp': interv_rollout2(post, (0, 2), 1024, 99, 'lamswap',
                                      W_post, offs_post, clamp=True)}
    curves = {}
    for k, o in conds.items():
        cs = np.where(o['iota'][:, None], o['coef_u'], o['coef_v'])
        co = np.where(o['iota'][:, None], o['coef_v'], o['coef_u'])
        curves[k] = {'coef_self': np.nanmean(cs, 0), 'coef_oth': np.nanmean(co, 0),
                     'ball': o['ball'].mean(0)}
        res[f'bodyswap_{k}'] = {
            'occ_rest': float(o['ball'][:, T_STAR:].mean()),
            'coef_self_rest': float(np.nanmean(cs[:, T_STAR:])),
            'coef_oth_rest': float(np.nanmean(co[:, T_STAR:])),
            'coef_self_t+1': float(np.nanmean(cs[:, T_STAR + 1])),
            'coef_oth_t+1': float(np.nanmean(co[:, T_STAR + 1]))}
        print(k, res[f'bodyswap_{k}'], flush=True)
    # immediate head-delta check for the one-shot swap
    o = conds['oneshot']
    du = np.abs(o['head_u_post'] - o['head_u_pre']).mean()
    dv = np.abs(o['head_v_post'] - o['head_v_pre']).mean()
    res['bodyswap_headdelta_t*'] = {'mean_abs_dlog_u': round(float(du), 4),
                                    'mean_abs_dlog_v': round(float(dv), 4)}
    print('head delta at t*:', res['bodyswap_headdelta_t*'], flush=True)

    with open('results/rnn_probes2.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for k, c in (('sham', 'gray'), ('oneshot', 'C0'), ('clamp', 'C3')):
        axes[0].plot(curves[k]['coef_self'], c, lw=1.8, label=k)
        axes[1].plot(curves[k]['coef_oth'], c, lw=1.8, label=k)
        axes[2].plot(curves[k]['ball'], c, lw=1.8, label=k)
    for ax, ttl in zip(axes, ('plan coef — TRUE self channel',
                              'plan coef — other channel',
                              'P(in ball)')):
        ax.axvline(T_STAR, ls=':', c='k', lw=1)
        ax.set_title(ttl); ax.set_xlabel('round t')
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig('figs/rnn_bodyswap.png', dpi=160)
    print('wrote results/rnn_probes2.json, figs/rnn_bodyswap.png', flush=True)


if __name__ == '__main__':
    main()
