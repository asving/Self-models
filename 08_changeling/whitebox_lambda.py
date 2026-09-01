"""Whitebox discriminating tests E1-E4 (DESIGN_changeling_v3_whitebox.md).
cwd = 08_changeling. Writes results/rnn_whitebox_lambda.json + 2 figures.
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
from probe import load
from probe3 import coef_np

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
RES = {}
T_STAR = 16
R = 1024
PAIR = (0, 2)


class Filt:
    """Vectorized exact filter bank + plan machinery on a token stream."""

    def __init__(self, w, R):
        self.w = w
        self.etaA = np.full((R, N), 1 / N); self.etaB = self.etaA.copy()
        self.drA = self.etaA.copy(); self.drB = self.etaA.copy()

    def dists(self, t):
        w = self.w
        pbar_u = self.etaA @ w.EA; pbar_v = self.etaB @ w.EB
        scA = np.einsum('ra,rb,abu->ru', self.drA, self.etaB, w.M[t])
        scB = np.einsum('ra,rb,abv->rv', self.etaA, self.drB, w.N[t])
        piA = pbar_u * (scA / (scA.max(1, keepdims=True) + 1e-300)) ** w.kappa
        piB = pbar_v * (scB / (scB.max(1, keepdims=True) + 1e-300)) ** w.kappa
        piA /= piA.sum(1, keepdims=True); piB /= piB.sum(1, keepdims=True)
        return pbar_u, pbar_v, piA, piB

    def update(self, un, vn):
        w = self.w
        TAg = w.TA[un, vn]; TBg = w.TB[un, vn]
        self.etaA = self.etaA * w.EA[:, un].T
        self.etaA = np.einsum('rs,rst->rt', self.etaA, TAg)
        self.etaA /= self.etaA.sum(1, keepdims=True)
        self.etaB = self.etaB * w.EB[:, vn].T
        self.etaB = np.einsum('rs,rst->rt', self.etaB, TBg)
        self.etaB /= self.etaB.sum(1, keepdims=True)
        self.drA = np.einsum('rs,rst->rt', self.drA, TAg)
        self.drB = np.einsum('rs,rst->rt', self.drB, TBg)


@torch.no_grad()
def prefix(model, w, tw, seed):
    """Natural closed-loop rounds 0..T_STAR-1. Returns state bundle."""
    torch.manual_seed(seed)
    goals = torch.tensor(PAIR, device=DEV).repeat(R, 1)
    iota = torch.rand(R, device=DEV) < 0.5
    sA = torch.randint(0, N, (R,), device=DEV)
    sB = torch.randint(0, N, (R,), device=DEV)
    h = None; u = v = None
    f = Filt(w, R)
    ev_hist = []
    for t in range(T_STAR):
        x = step_features(u, v, goals, t, w.T, DEV)
        lu, lv, h = model.step(x, h)
        pbar_u, pbar_v, piA, piB = f.dists(t)
        pu, pv = F.softmax(lu, -1), F.softmax(lv, -1)
        u_net = torch.multinomial(pu, 1).squeeze(1)
        v_net = torch.multinomial(pv, 1).squeeze(1)
        u_env, v_env = tw.emit(sA, sB)
        u = torch.where(iota, u_net, u_env)
        v = torch.where(iota, v_env, v_net)
        un, vn = u.cpu().numpy(), v.cpu().numpy()
        r = np.arange(R)
        ev_hist.append(np.log(pu.cpu().numpy()[r, un]) - np.log(pbar_u[r, un])
                       + np.log(pbar_v[r, vn]) - np.log(pv.cpu().numpy()[r, vn]))
        f.update(un, vn)
        sA, sB = tw.trans(sA, sB, u, v)
    return dict(h=h, u=u, v=v, f=f, iota=iota, goals=goals, sA=sA, sB=sB,
                ev=np.array(ev_hist).sum(0))


@torch.no_grad()
def continue_diet(model, w, st, mode, rng):
    """Teacher-forced continuation with a chosen token diet. Claims per round."""
    h, u, v = st['h'].clone(), st['u'].clone(), st['v'].clone()
    f = Filt.__new__(Filt); f.w = w
    for k in ('etaA', 'etaB', 'drA', 'drB'):
        setattr(f, k, getattr(st['f'], k).copy())
    out = {k: np.full((R, w.T - T_STAR), np.nan, np.float32)
           for k in ('m_u', 'm_v')}
    for j, t in enumerate(range(T_STAR, w.T)):
        x = step_features(u, v, st['goals'], t, w.T, DEV)
        lu, lv, h = model.step(x, h)
        pbar_u, pbar_v, piA, piB = f.dists(t)
        pu = F.softmax(lu, -1).cpu().numpy()
        pv = F.softmax(lv, -1).cpu().numpy()
        out['m_u'][:, j] = coef_np(pu, piA, pbar_u)
        out['m_v'][:, j] = coef_np(pv, piB, pbar_v)
        if mode == 'neutral':
            un = np.argmin(np.abs(np.log(pu + 1e-12) - np.log(pbar_u)), 1)
            vn = np.argmin(np.abs(np.log(pv + 1e-12) - np.log(pbar_v)), 1)
        elif mode == 'pbar':
            un = vec_sample(pbar_u, rng); vn = vec_sample(pbar_v, rng)
        elif mode == 'pi':
            un = vec_sample(pu, rng); vn = vec_sample(pv, rng)
        f.update(un, vn)
        u = torch.tensor(un, device=DEV); v = torch.tensor(vn, device=DEV)
    return out


def vec_sample(P, rng):
    c = np.cumsum(P, 1)
    return np.argmax(c > rng.random((len(P), 1)), 1)


@torch.no_grad()
def continue_closed(model, w, tw, st, swap=None, Q=None):
    """Closed-loop continuation; optional transplant at entry.
    swap: (donor_idx, 'full'|'row'|'comp')."""
    h, u, v = st['h'].clone(), st['u'].clone(), st['v'].clone()
    sA, sB = st['sA'].clone(), st['sB'].clone()
    iota = st['iota']
    f = Filt.__new__(Filt); f.w = w
    for k in ('etaA', 'etaB', 'drA', 'drB'):
        setattr(f, k, getattr(st['f'], k).copy())
    if swap is not None:
        donor, kind = swap
        hd = h[0][donor]
        if kind == 'full':
            h = hd.unsqueeze(0)
        else:
            comp_own = h[0] @ Q.T @ Q
            comp_don = hd @ Q.T @ Q
            if kind == 'row':
                h = (h[0] - comp_own + comp_don).unsqueeze(0)
            else:
                h = (hd - comp_don + comp_own).unsqueeze(0)
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
        un, vn = u.cpu().numpy(), v.cpu().numpy()
        f.update(un, vn)
        sA, sB = tw.trans(sA, sB, u, v)
        out['ball'][:, j] = tw.ball(sA, sB).float().cpu().numpy()
    return out


def cs_co(out, iota_np):
    io = iota_np[:, None]
    return (np.nanmean(np.where(io, out['m_u'], out['m_v']), 0),
            np.nanmean(np.where(io, out['m_v'], out['m_u']), 0))


def main():
    rng = np.random.default_rng(3)
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    st = prefix(post, w, tw, seed=77)
    io = st['iota'].cpu().numpy()

    # ---- E1: evidence diets ----
    diets = {}
    for mode in ('neutral', 'pbar', 'pi'):
        o = continue_diet(post, w, st, mode, rng)
        diets[mode] = cs_co(o, io)
        RES[f'E1_{mode}'] = {
            'self_t0_t4_t8_t15': [round(float(diets[mode][0][k]), 3)
                                  for k in (0, 4, 8, 15)],
            'oth_t0_t4_t8_t15': [round(float(diets[mode][1][k]), 3)
                                 for k in (0, 4, 8, 15)]}
        print(f'E1 {mode}:', RES[f'E1_{mode}'], flush=True)

    # ---- E2: transplant bisect ----
    Wu = post.head_u.weight.detach()
    Wv = post.head_v.weight.detach()
    Q, _ = torch.linalg.qr(torch.cat([Wu, Wv]).T)
    Q = Q.T.contiguous()                      # (12, 256)
    # matched pairs: opposite identity, nearest beliefs (L1 on etaA+etaB)
    eta = np.concatenate([st['f'].etaA, st['f'].etaB], 1)
    donor = np.arange(R)
    idxA = np.where(io)[0]; idxB = np.where(~io)[0]
    m = min(len(idxA), len(idxB))
    for i in idxA[:m]:
        d = np.abs(eta[idxB] - eta[i]).sum(1)
        donor[i] = idxB[np.argmin(d)]
    for i in idxB[:m]:
        d = np.abs(eta[idxA] - eta[i]).sum(1)
        donor[i] = idxA[np.argmin(d)]
    donor_t = torch.tensor(donor, device=DEV)
    conds = {'sham': None, 'full': (donor_t, 'full'),
             'row': (donor_t, 'row'), 'comp': (donor_t, 'comp')}
    curves = {}
    for name, swap in conds.items():
        o = continue_closed(post, w, tw, st, swap=swap, Q=Q)
        curves[name] = cs_co(o, io)
        RES[f'E2_{name}'] = {
            'self_t0_t1_t4_t8_t15': [round(float(curves[name][0][k]), 3)
                                     for k in (0, 1, 4, 8, 15)],
            'oth_t0_t1_t4_t8_t15': [round(float(curves[name][1][k]), 3)
                                    for k in (0, 1, 4, 8, 15)],
            'occ_rest': round(float(np.nanmean(o['ball'])), 3)}
        print(f'E2 {name}:', RES[f'E2_{name}'], flush=True)

    # ---- E3: window sufficiency for claims (natural rollouts) ----
    from probe3 import collect_full
    H, gt = collect_full(post, 1024, rng)
    ev = (gt['e_u'] + gt['e_v'])
    sgn = np.where(gt['iota'], 1.0, -1.0)[:, None]
    y = np.where(gt['iota'][:, None], gt['m_v'], gt['m_u'])  # other-claim
    ok = ~np.isnan(y)
    r2w = {}
    for Wn in (1, 2, 4, 8, 16):
        feats = [np.roll(ev, k, axis=1) for k in range(Wn)]
        for k in range(Wn):
            feats[k][:, :k] = 0
        Xw = np.stack(feats, -1) * sgn[..., None]
        Xf = np.concatenate([Xw, np.ones((*Xw.shape[:2], 1))], -1)
        A = Xf[ok]; yy = y[ok]
        beta = np.linalg.lstsq(A, yy, rcond=None)[0]
        pred = A @ beta
        r2w[Wn] = round(float(1 - ((yy - pred) ** 2).sum()
                              / ((yy - yy.mean()) ** 2).sum()), 3)
    lam = (np.clip(gt['lam_logodds'], -20, 20) * sgn)
    A = np.stack([lam, np.ones_like(lam)], -1)[ok]
    beta = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    r2w['cum_lambda'] = round(float(1 - ((y[ok] - A @ beta) ** 2).sum()
                                    / ((y[ok] - y[ok].mean()) ** 2).sum()), 3)
    RES['E3_r2_by_window'] = r2w
    print('E3:', r2w, flush=True)

    # ---- E4: shadow decomposition ----
    tfeat = np.tile(np.arange(32), (1024, 1))
    asym = np.nan_to_num(gt['m_u'] - gt['m_v'], nan=0.0)
    lam_t = np.clip(gt['lam_logodds'], -20, 20)
    Xs = np.stack([tfeat, asym, tfeat * asym, np.ones_like(asym)], -1)
    Xf = Xs.reshape(-1, 4); yf = lam_t.reshape(-1)
    beta = np.linalg.lstsq(Xf, yf, rcond=None)[0]
    RES['E4_lambda_from_time_x_expression_r2'] = round(
        float(1 - ((yf - Xf @ beta) ** 2).sum()
              / ((yf - yf.mean()) ** 2).sum()), 3)
    print('E4:', RES['E4_lambda_from_time_x_expression_r2'], flush=True)

    with open('results/rnn_whitebox_lambda.json', 'w') as fjs:
        json.dump(RES, fjs, indent=1, default=float)

    tt = np.arange(T_STAR, 32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for mode, c in (('neutral', 'C0'), ('pbar', 'C3'), ('pi', 'C2')):
        axes[0].plot(tt, diets[mode][0], c, lw=1.8, label=f'{mode} self')
        axes[0].plot(tt, diets[mode][1], c, lw=1.8, ls='--', label=f'{mode} other')
    axes[0].axhline(1 / (1 + np.exp(-1.3)), ls=':', c='gray')
    axes[0].text(31, 0.8, 'default σ(1.3)', fontsize=6, ha='right')
    axes[0].set_title('E1: claims under controlled evidence diets')
    axes[0].set_xlabel('round'); axes[0].set_ylabel('plan coefficient')
    axes[0].legend(fontsize=6)
    for name, c in (('sham', 'gray'), ('full', 'C3'), ('row', 'C0'),
                    ('comp', 'C2')):
        axes[1].plot(tt, curves[name][0], c, lw=1.8, label=f'{name} self')
        axes[1].plot(tt, curves[name][1], c, lw=1.8, ls='--')
    axes[1].set_title('E2: transplants (solid self, dashed other)')
    axes[1].set_xlabel('round'); axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig('figs/whitebox_diet.png', dpi=160)
    print('wrote results/rnn_whitebox_lambda.json, figs/whitebox_diet.png',
          flush=True)


if __name__ == '__main__':
    main()
