"""The lambda circuit, both directions (v2 doc, Iteration 7; v1.2 post net).

WRITE side: counterfactual-token increment profiles (net vs exact Bayes),
cross-channel additivity, and the efference test (net's own current policy
vs a fixed plan-template as the comparator).
READ side: INLP-style iterative erasure to measure the dimension of the
readable lambda code; runtime mean-ablation of the full subspace (cut the
register); donor swaps in subspaces of growing dimension — healing time vs
k is the isolation criterion (complete carrier => healing at evidence rate).
Also: manual GRU recompute harness, verified against model.step.

Writes results/rnn_lambda_circuit.json, figs/lambda_circuit.png.
cwd = 08_changeling.
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
from probe import load, split
from probe3 import collect_full, coef_np

RES = {}
T_STAR = 16
K_LIST = (1, 4, 8, 16)
# exact-recompute discipline: cuDNN TF32 gives ~7e-4 GRU-vs-manual diffs
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


# ---------- 0. manual GRU harness ----------

@torch.no_grad()
def gru_step_manual(model, x, h):
    e = torch.relu(model.inp(x))
    gi = e @ model.gru.weight_ih_l0.T + model.gru.bias_ih_l0
    gh = h[0] @ model.gru.weight_hh_l0.T + model.gru.bias_hh_l0
    i_r, i_z, i_n = gi.chunk(3, -1)
    h_r, h_z, h_n = gh.chunk(3, -1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    hn = (1 - z) * n + z * h[0]
    return hn.unsqueeze(0)


@torch.no_grad()
def verify_harness(model):
    torch.manual_seed(0)
    h = torch.zeros(1, 64, 256, device=DEV)
    worst = 0.0
    for _ in range(8):
        x = torch.rand(64, 28, device=DEV)
        _, _, h_ref = model.step(x, h)
        h_man = gru_step_manual(model, x, h)
        worst = max(worst, float((h_ref - h_man).abs().max()))
        h = h_ref
    assert worst < 1e-4, worst
    return worst


# ---------- 2. INLP ----------

def inlp(H, lam, tr, te, kmax=24, floor=0.15):
    Hf = H.reshape(-1, 256)
    y = np.clip(lam, -20, 20).reshape(-1)
    ep = np.repeat(np.arange(H.shape[0]), H.shape[1])
    m_tr, m_te = np.isin(ep, tr), np.isin(ep, te)
    mu, sd = Hf[m_tr].mean(0), Hf[m_tr].std(0) + 1e-8
    Z = (Hf - mu) / sd
    dirs, r2s = [], []
    for k in range(kmax):
        A = Z[m_tr].T @ Z[m_tr] + 1.0 * np.eye(256)
        ym = y[m_tr].mean()
        w = np.linalg.solve(A, Z[m_tr].T @ (y[m_tr] - ym))
        pred = Z @ w + ym
        r2 = 1 - ((y[m_te] - pred[m_te]) ** 2).sum() / ((y[m_te] - y[m_te].mean()) ** 2).sum()
        r2s.append(round(float(r2), 3))
        if r2 < floor:
            break
        wh = w / np.linalg.norm(w)
        dirs.append(wh)
        Z = Z - np.outer(Z @ wh, wh)
    P = np.stack(dirs) if dirs else np.zeros((0, 256))
    return P, r2s, mu, sd


# ---------- flexible rollout with subspace hooks ----------

@torch.no_grad()
def rollout_sub(model, pair, R, seed, P, mu, sd, mean_proj=None, mode='sham',
                donor_k=None, collect_ctx_at=None, dec=None):
    """mode: 'sham' | 'erase' (every round, project span(P) comp to the
    per-round population mean) | 'swap' (donor swap in span(P[:donor_k]) at
    T_STAR). Returns behavior + decoded-lambda + per-round true logodds
    (+ optional stored context for counterfactual stepping)."""
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
    Pt = torch.tensor(P, dtype=torch.float32, device=DEV)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEV)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=DEV)
    out = {k: np.full((R, T), np.nan, np.float32)
           for k in ('ball', 'm_u', 'm_v', 'lam_dec', 'lo_true')}
    ctx = None
    for t in range(T):
        x = step_features(u, v, goals, t, T, DEV)
        lu, lv, h = model.step(x, h)
        z = (h[0] - mu_t) / sd_t
        if mode == 'erase' and len(P):
            proj = z @ Pt.T
            tgt = torch.tensor(mean_proj[t], dtype=torch.float32, device=DEV)
            z = z - (proj - tgt) @ Pt
            h = (z * sd_t + mu_t).unsqueeze(0)
            lu = model.head_u(h[0]); lv = model.head_v(h[0])
        if mode == 'swap' and t == T_STAR and len(P):
            Pk = Pt[:donor_k]
            lo_s = np.where(iota.cpu().numpy(), logodds, -logodds)
            order = np.argsort(lo_s)
            donor = np.empty(R, int)
            donor[order] = order[::-1]        # pair most-A-ish with most-B-ish
            proj = z @ Pk.T
            z = z + (proj[donor] - proj) @ Pk
            h = (z * sd_t + mu_t).unsqueeze(0)
            lu = model.head_u(h[0]); lv = model.head_v(h[0])
        pbar_u = etaA @ w.EA; pbar_v = etaB @ w.EB
        scA = np.einsum('ra,rb,abu->ru', drA, etaB, w.M[t])
        scB = np.einsum('ra,rb,abv->rv', etaA, drB, w.N[t])
        piA = pbar_u * (scA / (scA.max(1, keepdims=True) + 1e-300)) ** w.kappa
        piB = pbar_v * (scB / (scB.max(1, keepdims=True) + 1e-300)) ** w.kappa
        piA /= piA.sum(1, keepdims=True); piB /= piB.sum(1, keepdims=True)
        if dec is not None:
            wd, ym = dec
            out['lam_dec'][:, t] = (((h[0] - mu_t) / sd_t).cpu().numpy() @ wd + ym)
        pu, pv = F.softmax(lu, -1), F.softmax(lv, -1)
        out['m_u'][:, t] = coef_np(pu.cpu().numpy(), piA, pbar_u)
        out['m_v'][:, t] = coef_np(pv.cpu().numpy(), piB, pbar_v)
        if collect_ctx_at is not None and t == collect_ctx_at:
            ctx = {'h': h.clone(), 'etaA': etaA.copy(), 'etaB': etaB.copy(),
                   'drA': drA.copy(), 'drB': drB.copy(),
                   'piA': piA.copy(), 'piB': piB.copy(),
                   'pbar_u': pbar_u.copy(), 'pbar_v': pbar_v.copy(),
                   'logpu': F.log_softmax(lu, -1).cpu().numpy(),
                   'logpv': F.log_softmax(lv, -1).cpu().numpy(),
                   'goals': goals, 'iota': iota.cpu().numpy(), 't': t}
        u_net = torch.multinomial(pu, 1).squeeze(1)
        v_net = torch.multinomial(pv, 1).squeeze(1)
        u_env, v_env = tw.emit(sA, sB)
        u = torch.where(iota, u_net, u_env)
        v = torch.where(iota, v_env, v_net)
        un, vn = u.cpu().numpy(), v.cpu().numpy()
        r = np.arange(R)
        logodds += (np.log(pu.cpu().numpy()[r, un]) - np.log(pbar_u[r, un])
                    + np.log(pbar_v[r, vn]) - np.log(pv.cpu().numpy()[r, vn]))
        out['lo_true'][:, t] = logodds
        TAg = w.TA[un, vn]; TBg = w.TB[un, vn]
        etaA = etaA * w.EA[:, un].T
        etaA = np.einsum('rs,rst->rt', etaA, TAg); etaA /= etaA.sum(1, keepdims=True)
        etaB = etaB * w.EB[:, vn].T
        etaB = np.einsum('rs,rst->rt', etaB, TBg); etaB /= etaB.sum(1, keepdims=True)
        drA = np.einsum('rs,rst->rt', drA, TAg)
        drB = np.einsum('rs,rst->rt', drB, TBg)
        sA, sB = tw.trans(sA, sB, u, v)
        out['ball'][:, t] = tw.ball(sA, sB).float().cpu().numpy()
    out['iota'] = iota.cpu().numpy()
    return out, ctx


def signed_med(o):
    s = np.where(o['iota'][:, None], 1.0, -1.0)
    return np.nanmedian(s * o['lam_dec'], 0)


# ---------- counterfactual increment profiles ----------

@torch.no_grad()
def cf_profiles(model, ctx, dec, wobj, channel='u'):
    """For each context, sweep the round-t token on one channel (other at
    its modal value), step the manual GRU, decode lambda. Returns centered
    net profiles + centered Bayes/efference/template profiles (R, 6)."""
    wd, ym = dec
    h, t = ctx['h'], ctx['t']
    R = h.shape[1]
    goals = ctx['goals']
    T = wobj.T
    other_mode = (np.argmax(ctx['pbar_v'], 1) if channel == 'u'
                  else np.argmax(ctx['pbar_u'], 1))
    prof = np.zeros((R, N), np.float32)
    for tok in range(N):
        uu = torch.full((R,), tok, dtype=torch.long, device=DEV)
        vv = torch.tensor(other_mode, dtype=torch.long, device=DEV)
        if channel == 'u':
            x = step_features(uu, vv, goals, t + 1, T, DEV)
        else:
            x = step_features(vv, uu, goals, t + 1, T, DEV)
        hn = gru_step_manual(model, x, h)
        zn = (hn[0].cpu().numpy() - MU) / SD
        prof[:, tok] = zn @ wd + ym
    prof -= prof.mean(1, keepdims=True)
    if channel == 'u':
        eff = ctx['logpu'] - np.log(ctx['pbar_u'])
        tem = np.log(ctx['piA'] + 1e-12) - np.log(ctx['pbar_u'])
        sign = 1.0
    else:
        eff = -(ctx['logpv'] - np.log(ctx['pbar_v']))
        tem = -(np.log(ctx['piB'] + 1e-12) - np.log(ctx['pbar_v']))
        sign = 1.0
    eff = sign * (eff - eff.mean(1, keepdims=True))
    tem = sign * (tem - tem.mean(1, keepdims=True))
    return prof, eff, tem


def slope_r2(y, x):
    x, y = x.ravel(), y.ravel()
    sl = float((x * y).sum() / ((x * x).sum() + 1e-12))
    ss = ((y - sl * x) ** 2).sum()
    return sl, float(1 - ss / (((y - y.mean()) ** 2).sum() + 1e-12))


def main():
    global MU, SD
    rng = np.random.default_rng(5)
    post = load('post_6000')
    RES['harness_maxdiff'] = verify_harness(post)
    print('harness OK', RES['harness_maxdiff'], flush=True)

    H, gt = collect_full(post, 2048, rng)
    tr, te = split(2048)
    P, r2s, MU, SD = inlp(H, gt['lam_logodds'], tr, te)
    RES['inlp_r2_sequence'] = r2s
    RES['d_lambda_readable'] = int(len(P))
    print('INLP:', r2s, '-> d =', len(P), flush=True)
    # runtime decoder = first INLP direction fit (use full-rank ridge for dec)
    Zf = (H.reshape(-1, 256) - MU) / SD
    y = np.clip(gt['lam_logodds'], -20, 20).reshape(-1)
    ep = np.repeat(np.arange(2048), 32)
    m_tr = np.isin(ep, tr)
    A = Zf[m_tr].T @ Zf[m_tr] + 1.0 * np.eye(256)
    ym = y[m_tr].mean()
    wd = np.linalg.solve(A, Zf[m_tr].T @ (y[m_tr] - ym))
    dec = (wd, ym)

    # per-round mean projection for erasure
    sham, ctx16 = rollout_sub(post, (0, 2), 1024, 99, P, MU, SD, mode='sham',
                              collect_ctx_at=T_STAR, dec=dec)
    Hs, gts = collect_full(post, 512, rng)
    Zs = ((Hs.reshape(-1, 256) - MU) / SD) @ P.T
    mean_proj = Zs.reshape(512, 32, -1).mean(0)

    era, _ = rollout_sub(post, (0, 2), 1024, 99, P, MU, SD,
                         mean_proj=mean_proj, mode='erase', dec=dec)
    for name, o in (('sham', sham), ('erase_full', era)):
        io = o['iota'][:, None]
        cs = np.where(io, o['m_u'], o['m_v'])
        co = np.where(io, o['m_v'], o['m_u'])
        RES[f'reg_{name}'] = {
            'occ': round(float(np.nanmean(o['ball'])), 3),
            'coef_self': round(float(np.nanmean(cs[:, 8:])), 3),
            'coef_oth': round(float(np.nanmean(co[:, 8:])), 3),
            'final_correct': round(float(
                (np.where(o['iota'], 1, -1) * o['lo_true'][:, -1] > 0).mean()), 3)}
    print('erasure:', RES['reg_sham'], RES['reg_erase_full'], flush=True)

    # donor swaps: healing time vs k
    heal = {}
    ev_rate = float(np.nanmean(np.abs(np.diff(sham['lo_true'], axis=1)
                                      [:, T_STAR:T_STAR + 4])))
    sm = signed_med(sham)
    for k in [kk for kk in K_LIST if kk <= len(P)]:
        o, _ = rollout_sub(post, (0, 2), 1024, 99, P, MU, SD, mode='swap',
                           donor_k=k, dec=dec)
        m = signed_med(o)
        d0 = sm[T_STAR] - m[T_STAR]
        rec = -1
        for j in range(T_STAR, 32):
            if abs(sm[j] - m[j]) < 0.2 * abs(d0):
                rec = j - T_STAR
                break
        heal[k] = {'flip_depth': round(float(d0), 2), 'heal_rounds': rec,
                   'curve': [round(float(x), 2) for x in m]}
        io = o['iota'][:, None]
        co = np.where(io, o['m_v'], o['m_u'])
        heal[k]['coef_oth_rest'] = round(float(np.nanmean(co[:, T_STAR:])), 3)
        heal[k]['occ_rest'] = round(float(np.nanmean(o['ball'][:, T_STAR:])), 3)
        print(f'swap k={k}:', {kk: vv for kk, vv in heal[k].items()
                               if kk != 'curve'}, flush=True)
    RES['donor_swap_by_k'] = heal
    RES['evidence_rate'] = round(ev_rate, 3)
    RES['sham_curve'] = [round(float(x), 2) for x in sm]

    # WRITE side: counterfactual profiles at t*=16 (both channels), pooled
    wobj = World(goal_pair=(0, 2), **WORLD_KW)
    for ch in ('u', 'v'):
        prof, eff, tem = cf_profiles(post, ctx16, dec, wobj, ch)
        io = ctx16['iota'] if ch == 'u' else ~ctx16['iota']
        sl_e, r2_e = slope_r2(prof, eff)
        RES[f'cf_{ch}_slope_r2_vs_bayes'] = [round(sl_e, 3), round(r2_e, 3)]
        # efference test on the group where this channel is NOT mine
        pn, pe, pt = prof[~io], eff[~io], tem[~io]
        se, re_ = slope_r2(pn, pe)
        st, rt_ = slope_r2(pn, pt)
        RES[f'cf_{ch}_notmine_efference_slope_r2'] = [round(se, 3), round(re_, 3)]
        RES[f'cf_{ch}_notmine_template_slope_r2'] = [round(st, 3), round(rt_, 3)]
        RES[f'cf_{ch}_notmine_profile_sd_ratio'] = round(
            float(pn.std() / (prof[io].std() + 1e-9)), 3)
    print('cf:', {k: v for k, v in RES.items() if k.startswith('cf_')}, flush=True)

    with open('results/rnn_lambda_circuit.json', 'w') as f:
        json.dump(RES, f, indent=1, default=float)

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    axes[0].plot(range(1, len(r2s) + 1), r2s, 'o-')
    axes[0].set_xlabel('directions removed'); axes[0].set_ylabel('λ decoder R²')
    axes[0].set_title(f'INLP: readable λ code has d ≈ {len(P)}')
    tt = np.arange(32)
    axes[1].plot(tt, sm, 'gray', lw=2, label='sham')
    for k in heal:
        axes[1].plot(tt, heal[k]['curve'], lw=1.6, label=f'swap k={k}')
    axes[1].axvline(T_STAR, ls=':', c='k')
    axes[1].set_xlabel('round'); axes[1].set_ylabel('decoded λ (signed median)')
    axes[1].set_title('healing vs swapped-subspace dimension')
    axes[1].legend(fontsize=7)
    prof, eff, tem = cf_profiles(post, ctx16, dec, wobj, 'u')
    ii = np.random.default_rng(0).choice(prof.size, min(4000, prof.size), False)
    axes[2].scatter(eff.ravel()[ii], prof.ravel()[ii], s=2, alpha=.15)
    sl_e, r2_e = slope_r2(prof, eff)
    xs = np.array([eff.min(), eff.max()])
    axes[2].plot(xs, sl_e * xs, 'r-', label=f'slope {sl_e:.2f}, R² {r2_e:.2f}')
    axes[2].set_xlabel('Bayes increment  log π(u\') − log p̄(u\')')
    axes[2].set_ylabel('net Δλ_dec (counterfactual token)')
    axes[2].set_title('the write rule (u-channel, all contexts)')
    axes[2].legend(fontsize=7)
    labs, vals = [], []
    for ch in ('u', 'v'):
        labs += [f'{ch}: efference', f'{ch}: template']
        vals += [RES[f'cf_{ch}_notmine_efference_slope_r2'][1],
                 RES[f'cf_{ch}_notmine_template_slope_r2'][1]]
    axes[3].bar(labs, vals, color=['C0', 'C3'] * 2)
    axes[3].set_ylabel('R² of counterfactual profile')
    axes[3].set_title('comparator on NOT-mine channels:\nown policy vs plan template')
    axes[3].tick_params(axis='x', labelsize=7)
    fig.tight_layout(); fig.savefig('figs/lambda_circuit.png', dpi=160)
    print('wrote results/rnn_lambda_circuit.json, figs/lambda_circuit.png',
          flush=True)


if __name__ == '__main__':
    main()
