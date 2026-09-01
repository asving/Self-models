"""Distill the memory direction by letting the dynamics clean it (v3 doc,
Iteration 8 closure). Propagate matched-twin state differences through 4
paired identical-token rounds; the surviving vector m is the candidate
store. Tests: PCA dimensionality of {m}; composition (overlap with belief
blocks / lambda decoder); CAUSAL: transplant along per-pair m only vs full
swap vs sham, closed-loop persistence. cwd = 08_changeling."""
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
from probe import load
from probe3 import coef_np
from whitebox_lambda import prefix, Filt
from format import match_pairs, per_round_decoders

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
RES = {}
T_STAR = 16
R = 1024
PAIR = (0, 2)
N_CLEAN = 4


@torch.no_grad()
def distill(model, w, st, donor, n_rounds=N_CLEAN):
    """Paired identical-neutral-token propagation of the twin difference."""
    dn = torch.tensor(donor, device=DEV)
    h0 = st['h'].clone()
    h1 = st['h'][0][dn].unsqueeze(0).clone()
    u, v = st['u'].clone(), st['v'].clone()
    f = Filt.__new__(Filt); f.w = w
    for k in ('etaA', 'etaB', 'drA', 'drB'):
        setattr(f, k, getattr(st['f'], k).copy())
    for t in range(T_STAR, T_STAR + n_rounds):
        x = step_features(u, v, st['goals'], t, w.T, DEV)
        lu, lv, h0 = model.step(x, h0)
        _, _, h1 = model.step(x, h1)
        pu = F.softmax(lu, -1).cpu().numpy()
        pv = F.softmax(lv, -1).cpu().numpy()
        pbar_u, pbar_v, _, _ = f.dists(t)
        un = np.argmin(np.abs(np.log(pu + 1e-12) - np.log(pbar_u)), 1)
        vn = np.argmin(np.abs(np.log(pv + 1e-12) - np.log(pbar_v)), 1)
        f.update(un, vn)
        u = torch.tensor(un, device=DEV); v = torch.tensor(vn, device=DEV)
    return (h1 - h0)[0].cpu().numpy()


@torch.no_grad()
def swap_run(model, w, tw, st, mode, donor, mdirs=None, Pk=None):
    """Closed-loop continuation with transplant at entry.
    mode: sham|full|distilled (per-pair 1-dim along m̂)|pca (span Pk)."""
    dn = torch.tensor(donor, device=DEV)
    h, u, v = st['h'].clone(), st['u'].clone(), st['v'].clone()
    sA, sB = st['sA'].clone(), st['sB'].clone()
    iota = st['iota']
    f = Filt.__new__(Filt); f.w = w
    for k in ('etaA', 'etaB', 'drA', 'drB'):
        setattr(f, k, getattr(st['f'], k).copy())
    if mode == 'full':
        h = h[0][dn].unsqueeze(0)
    elif mode == 'distilled':
        md = torch.tensor(mdirs, dtype=torch.float32, device=DEV)
        diff = h[0][dn] - h[0]
        coef = (diff * md).sum(1, keepdim=True)
        h = (h[0] + coef * md).unsqueeze(0)
    elif mode == 'pca':
        Pt = torch.tensor(Pk, dtype=torch.float32, device=DEV)
        proj = h[0] @ Pt.T
        h = (h[0] + (proj[dn] - proj) @ Pt).unsqueeze(0)
    out = {k: np.full((R, w.T - T_STAR), np.nan, np.float32)
           for k in ('m_u', 'm_v', 'ball')}
    for j, t in enumerate(range(T_STAR, w.T)):
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
        out['ball'][:, j] = tw.ball(sA, sB).float().cpu().numpy()
    io = iota.cpu().numpy()[:, None]
    return (np.nanmean(np.where(io, out['m_u'], out['m_v']), 0),
            np.nanmean(np.where(io, out['m_v'], out['m_u']), 0),
            float(np.nanmean(out['ball'])))


@torch.no_grad()
def main():
    rng = np.random.default_rng(17)
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    st = prefix(post, w, tw, seed=77)
    donor = match_pairs(st)

    m = distill(post, w, st, donor)
    nm = np.linalg.norm(m, axis=1, keepdims=True)
    mhat = m / (nm + 1e-9)
    d0 = (st['h'][0][torch.tensor(donor, device=DEV)]
          - st['h'][0]).cpu().numpy()
    RES['distilled_norm_frac'] = round(float(
        (nm[:, 0] / (np.linalg.norm(d0, axis=1) + 1e-9)).mean()), 3)
    # composition: PCA across episodes of the signed, normalized m
    sgn = np.where(st['iota'].cpu().numpy(), 1.0, -1.0)[:, None]
    U_, S_, Vt = np.linalg.svd(sgn * mhat, full_matrices=False)
    ev = S_ ** 2 / (S_ ** 2).sum()
    RES['pca_var_top5'] = [round(float(x), 3) for x in ev[:5]]
    RES['pca_participation'] = round(float((ev.sum() ** 2) / (ev ** 2).sum()), 1)
    # overlap with per-round lambda decoder at t*+N_CLEAN
    from probe3 import collect_full
    from probe import split
    H, gt = collect_full(post, 1024, rng)
    tr, te = split(1024)
    Wt, _, mu, sd = per_round_decoders(H, gt['lam_logodds'], tr, te)
    wt = Wt[T_STAR + N_CLEAN] / sd
    wt /= np.linalg.norm(wt)
    RES['cos_mhat_lambdadir'] = round(float(
        np.abs((sgn[:, 0] * (mhat @ wt))).mean()), 3)
    RES['cos_top_pca_lambdadir'] = round(float(np.abs(Vt[0] @ wt)), 3)
    print('distill:', RES, flush=True)

    # causal: transplant tests
    conds = {'sham': ('sham', None, None),
             'full': ('full', None, None),
             'distilled_1d': ('distilled', mhat, None),
             'pca_k2': ('pca', None, Vt[:2]),
             'pca_k8': ('pca', None, Vt[:8])}
    curves = {}
    for name, (mode, md, Pk) in conds.items():
        s, o, occ = swap_run(post, w, tw, st, mode, donor, md, Pk)
        curves[name] = (s, o)
        RES[f'swap_{name}'] = {
            'self_t0_t4_t8_t15': [round(float(s[k]), 3) for k in (0, 4, 8, 15)],
            'oth_t0_t4_t8_t15': [round(float(o[k]), 3) for k in (0, 4, 8, 15)],
            'occ_rest': round(occ, 3)}
        print(f'swap {name}:', RES[f'swap_{name}'], flush=True)

    with open('results/rnn_distill.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    tt = np.arange(T_STAR, 32)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, c in (('sham', 'gray'), ('full', 'C3'), ('distilled_1d', 'C0'),
                    ('pca_k8', 'C2')):
        ax.plot(tt, curves[name][0], c, lw=1.8, label=f'{name} self')
        ax.plot(tt, curves[name][1], c, lw=1.8, ls='--')
    ax.axvline(T_STAR, ls=':', c='k')
    ax.set_xlabel('round'); ax.set_ylabel('plan coefficient')
    ax.set_title('transplants along the DISTILLED memory direction\n'
                 '(solid self, dashed other)')
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig('figs/distill_swap.png', dpi=160)
    print('wrote results/rnn_distill.json, figs/distill_swap.png', flush=True)


if __name__ == '__main__':
    main()
