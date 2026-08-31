"""Encoder probes + causal interventions on the changeling GRU
(DESIGN_changeling_v2_probes.md). Writes results/rnn_probes.json,
figs/rnn_probes.png, figs/rnn_bodyswap.png. cwd = 08_changeling.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World
from oracle import replay_dists, run_base
from rnn import ChangelingGRU, TorchWorld, features, step_features, GOAL_PAIRS, N
from eval_rnn import rollout_record, WORLD_KW, DEV

N_EPS = 2048
BATCH = 256
SEED = 4242
D = 256


def load(name):
    m = ChangelingGRU().to(DEV)
    m.load_state_dict(torch.load(f'ckpt/{name}.pt'))
    m.eval()
    return m


# ---------- data collection ----------

@torch.no_grad()
def collect_pre(model, n_eps):
    """Pre-net hiddens on un-embodied base streams (native regime)."""
    w = World(goal_pair=(0, 2), **WORLD_KW)
    Hs, GT = [], []
    for b in range(n_eps // BATCH):
        base = run_base(w, BATCH, SEED + b, collect=True)
        X = torch.tensor(features(base['u'], base['v']), device=DEV)
        _, _, hs = model(X)
        Hs.append(hs[:, :w.T].cpu().numpy())
        GT.append(replay_dists(w, base['u'].astype(np.int64),
                               base['v'].astype(np.int64), return_beliefs=True))
    H = np.concatenate(Hs)
    gt = {k: np.concatenate([g[k] for g in GT]) for k in GT[0]}
    return H, gt


@torch.no_grad()
def collect_post(model, n_eps, rng):
    """Closed-loop rollouts of an embodied net + exact ground truths + lambda."""
    Hs, recs, gts, lams = [], [], [], []
    for b in range(n_eps // BATCH):
        pair = GOAL_PAIRS[rng.integers(12)]
        w = World(goal_pair=pair, **WORLD_KW)
        rec = rollout_record(model, TorchWorld(w, DEV), pair, BATCH, SEED + 100 + b)
        rep = replay_dists(w, rec['u'], rec['v'], return_beliefs=True)
        r = np.arange(BATCH)[:, None]; t = np.arange(w.T)[None, :]
        dlog = (np.log(rec['pu'][r, t, rec['u']]) - np.log(rep['pbar_u'][r, t, rec['u']])
                + np.log(rep['pbar_v'][r, t, rec['v']]) - np.log(rec['pv'][r, t, rec['v']]))
        lo = np.cumsum(dlog, axis=1)
        lo_shift = np.concatenate([np.zeros((BATCH, 1)), lo[:, :-1]], axis=1)
        Hs.append(rec['hiddens'].astype(np.float32))
        recs.append(rec); gts.append(rep); lams.append(lo_shift)
    H = np.concatenate(Hs)
    gt = {k: np.concatenate([g[k] for g in gts]) for k in gts[0]}
    gt['lam_logodds'] = np.concatenate(lams)
    gt['iota'] = np.concatenate([r['iota'] for r in recs])
    tokens = {k: np.concatenate([r[k] for r in recs]) for k in ('u', 'v')}
    return H, gt, tokens


@torch.no_grad()
def teacher_force_hiddens(model, U, V):
    """Pre-regime hiddens (goals zeroed) on given token streams."""
    Hs = []
    for i in range(0, U.shape[0], BATCH):
        X = torch.tensor(features(U[i:i + BATCH], V[i:i + BATCH]), device=DEV)
        _, _, hs = model(X)
        Hs.append(hs[:, :U.shape[1]].cpu().numpy())
    return np.concatenate(Hs)


# ---------- fits ----------

def flat(A):
    return A.reshape(-1, A.shape[-1]) if A.ndim == 3 else A.reshape(-1, 1)


def split(n_eps, frac=0.8, seed=0):
    idx = np.random.default_rng(seed).permutation(n_eps)
    k = int(frac * n_eps)
    return idx[:k], idx[k:]


def encoder_fit(G, H, tr, te):
    """OLS g -> h. Returns W (k, D), R2_enc on test."""
    Gtr, Htr = flat(G[tr]), flat(H[tr])
    Gc = np.concatenate([Gtr, np.ones((len(Gtr), 1))], 1)
    Wb = np.linalg.lstsq(Gc, Htr, rcond=None)[0]
    W, b = Wb[:-1], Wb[-1]
    Gte, Hte = flat(G[te]), flat(H[te])
    pred = Gte @ W + b
    r2 = 1 - ((Hte - pred) ** 2).sum() / ((Hte - Hte.mean(0)) ** 2).sum()
    return W, b, float(r2)


def decoder_r2(H, G, tr, te, alpha=1.0):
    Htr, Gtr = flat(H[tr]), flat(G[tr])
    Hte, Gte = flat(H[te]), flat(G[te])
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-8
    Htr = (Htr - mu) / sd; Hte = (Hte - mu) / sd
    A = Htr.T @ Htr + alpha * np.eye(H.shape[-1])
    W = np.linalg.solve(A, Htr.T @ (Gtr - Gtr.mean(0)))
    pred = Hte @ W + Gtr.mean(0)
    r2 = 1 - ((Gte - pred) ** 2).sum(0) / ((Gte - Gte.mean(0)) ** 2).sum(0)
    return float(np.mean(r2))


def partial_decoder_r2(H, G, C, tr, te):
    """Decoder R2 of G from H after regressing the confounder C out of both."""
    def resid(Y):
        Ctr = np.concatenate([flat(C[tr]), np.ones((len(flat(C[tr])), 1))], 1)
        B = np.linalg.lstsq(Ctr, flat(Y[tr]), rcond=None)[0]
        out = np.zeros_like(flat(Y)).reshape(Y.shape[0], Y.shape[1], -1)
        Call = np.concatenate([flat(C), np.ones((len(flat(C)), 1))], 1)
        return (flat(Y) - Call @ B).reshape(Y.shape[0], Y.shape[1], -1)
    return decoder_r2(resid(H), resid(G), tr, te)


def principal_angles(W1, W2):
    q1 = np.linalg.qr(W1.T)[0]
    q2 = np.linalg.qr(W2.T)[0]
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


# ---------- interventions ----------

@torch.no_grad()
def interv_rollout(model, pair, R, seed, t_star, mode, Wenc=None, benc=None,
                   dose=1.0):
    """Closed-loop rollout with an intervention at t_star.
    mode: 'sham' | 'belief' (rotate etaA by 3) | 'lamswap' (log-odds -> -dose*logodds... flips sign, scaled).
    Returns single-step head changes + per-round remaining stats."""
    w = World(goal_pair=pair, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    torch.manual_seed(seed)
    T = w.T
    goals = torch.tensor(pair, device=DEV).repeat(R, 1)
    iota = torch.rand(R, device=DEV) < 0.5
    sA = torch.randint(0, N, (R,), device=DEV)
    sB = torch.randint(0, N, (R,), device=DEV)
    h = None; u = v = None
    # incremental exact filter (numpy)
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
        if t == t_star and mode != 'sham':
            if mode == 'belief':
                g = etaA.astype(np.float32)
                gp = np.roll(g, 3, axis=1)
                dh = torch.tensor((gp - g) @ Wenc, dtype=torch.float32,
                                  device=DEV)
                out['dpred_u'] = np.log(gp @ w.EA) - np.log(pbar_u)
            else:  # lamswap: logodds -> -dose * logodds
                dl = (-dose * logodds - logodds).astype(np.float32)
                dh = torch.tensor(dl[:, None] * Wenc, dtype=torch.float32,
                                  device=DEV)
            out['head_u_pre'] = F.log_softmax(lu, -1).cpu().numpy()
            out['head_v_pre'] = F.log_softmax(lv, -1).cpu().numpy()
            h = h + dh.unsqueeze(0)
            lu = model.head_u(h[0]); lv = model.head_v(h[0])
            out['head_u_post'] = F.log_softmax(lu, -1).cpu().numpy()
            out['head_v_post'] = F.log_softmax(lv, -1).cpu().numpy()
        pu, pv = F.softmax(lu, -1), F.softmax(lv, -1)
        u_net = torch.multinomial(pu, 1).squeeze(1)
        v_net = torch.multinomial(pv, 1).squeeze(1)
        u_env, v_env = tw.emit(sA, sB)
        u = torch.where(iota, u_net, u_env)
        v = torch.where(iota, v_env, v_net)
        # plan coefficients of the heads against the exact basis
        for head, plan, pb, key in ((pu.cpu().numpy(), piA, pbar_u, 'coef_u'),
                                    (pv.cpu().numpy(), piB, pbar_v, 'coef_v')):
            d = plan - pb
            den = (d * d).sum(1)
            ok = den > 1e-4
            out[key][ok, t] = np.clip(((head - pb) * d).sum(1)[ok] / den[ok], 0, 1)
        un, vn = u.cpu().numpy(), v.cpu().numpy()
        r = np.arange(R)
        # lambda update under the net's actual heads
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


def slope_r2(dobs, dpred):
    x, y = dpred.ravel(), dobs.ravel()
    sl = float((x * y).sum() / (x * x).sum())
    r2 = float(1 - ((y - sl * x) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return sl, r2


def main():
    res = {}
    rng = np.random.default_rng(7)

    # ---------- fits ----------
    pre = load('pre_final')
    H_pre, gt_pre = collect_pre(pre, N_EPS)
    tr, te = split(N_EPS)
    fits = {}
    enc_store = {}
    for netname, H, gt in [('pre_final', H_pre, gt_pre)]:
        pass
    nets = {'pre_final': (H_pre, gt_pre)}
    post_models = {}
    for name in ('post_0', 'post_6000'):
        m = load(name)
        post_models[name] = m
        Hp, gtp, toks = collect_post(m, N_EPS, rng)
        nets[name] = (Hp, gtp)
        if name == 'post_6000':
            # Q4 control: pre-net hiddens on the SAME embodied streams
            Hctrl = teacher_force_hiddens(pre, toks['u'], toks['v'])
            nets['pre_on_embodied'] = (Hctrl, gtp)
    for name, (H, gt) in nets.items():
        f = {}
        for var in ('etaA', 'etaB', 'drA', 'drB'):
            if var not in gt:
                continue
            W, b, r2e = encoder_fit(gt[var], H, tr, te)
            f[var] = {'r2_enc': r2e, 'r2_dec': decoder_r2(H, gt[var], tr, te)}
            enc_store[(name, var)] = (W, b)
        for var in ('drA', 'drB'):
            if var in gt:
                C = np.concatenate([gt['etaA'], gt['etaB']], -1)
                f[var]['r2_dec_partial'] = partial_decoder_r2(H, gt[var], C, tr, te)
        if 'lam_logodds' in gt:
            L = np.clip(gt['lam_logodds'], -20, 20)[..., None]
            W, b, r2e = encoder_fit(L, H, tr, te)
            f['lam_logodds'] = {'r2_enc': r2e, 'r2_dec': decoder_r2(H, L, tr, te)}
            enc_store[(name, 'lam')] = (W, b)
        fits[name] = f
        print(name, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()}
                                for k, v in f.items()}), flush=True)
    for var in ('etaA', 'etaB'):
        ang = principal_angles(enc_store[('pre_final', var)][0],
                               enc_store[('post_6000', var)][0])
        fits[f'angles_{var}_pre_vs_post6000'] = [round(float(a), 1) for a in ang]
    res['fits'] = fits

    # ---------- belief interventions ----------
    interv = {}
    for name in ('pre_final', 'post_6000'):
        model = pre if name == 'pre_final' else post_models['post_6000']
        Wsrc = 'pre_final' if name == 'pre_final' else 'post_6000'
        W, b = enc_store[(Wsrc, 'etaA')]
        sl_t, ctl_t = [], []
        for t_star in (8, 16, 24):
            o = interv_rollout(model, (0, 2), 512, 31 + t_star, t_star,
                               'belief', Wenc=W)
            d_obs_u = o['head_u_post'] - o['head_u_pre']
            d_obs_v = o['head_v_post'] - o['head_v_pre']
            sl, r2 = slope_r2(d_obs_u, o['dpred_u'])
            slc, _ = slope_r2(d_obs_v, o['dpred_u'])
            sl_t.append((t_star, sl, r2, slc))
        interv[name] = {'belief_slope_r2_ctrl_by_t':
                        [[t, round(s, 3), round(r, 3), round(c, 3)]
                         for t, s, r, c in sl_t]}
        print(name, interv[name], flush=True)

    # ---------- body swap ----------
    W, b = enc_store[('post_6000', 'lam')]
    model = post_models['post_6000']
    t_star = 16
    sham = interv_rollout(model, (0, 2), 1024, 99, t_star, 'sham')
    swaps = {}
    for dose in (0.5, 1.0, 2.0):
        swaps[dose] = interv_rollout(model, (0, 2), 1024, 99, t_star,
                                     'lamswap', Wenc=W[0], dose=dose)
    bs = {}
    def coef_self(o):
        return np.where(o['iota'][:, None], o['coef_u'], o['coef_v'])
    def coef_oth(o):
        return np.where(o['iota'][:, None], o['coef_v'], o['coef_u'])
    rest = slice(t_star, None)
    bs['sham'] = {'occ_rest': float(np.nanmean(sham['ball'][:, rest])),
                  'coef_self_rest': float(np.nanmean(coef_self(sham)[:, rest])),
                  'coef_oth_rest': float(np.nanmean(coef_oth(sham)[:, rest]))}
    for dose, o in swaps.items():
        bs[f'dose_{dose}'] = {
            'occ_rest': float(np.nanmean(o['ball'][:, rest])),
            'coef_self_rest': float(np.nanmean(coef_self(o)[:, rest])),
            'coef_oth_rest': float(np.nanmean(coef_oth(o)[:, rest]))}
    res['bodyswap'] = bs
    print('bodyswap:', json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()}
                                   for k, v in bs.items()}), flush=True)

    with open('results/rnn_probes.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    names = ['pre_final', 'post_0', 'post_6000']
    vars_ = ['etaA', 'etaB', 'drA', 'drB']
    xs = np.arange(len(vars_))
    for i, nm in enumerate(names):
        ys = [fits[nm].get(v, {}).get('r2_dec', np.nan) for v in vars_]
        axes[0].bar(xs + 0.25 * i, ys, 0.23, label=nm)
    axes[0].set_xticks(xs + 0.25); axes[0].set_xticklabels(vars_)
    axes[0].set_ylabel('decoder R² (held-out)')
    axes[0].set_title('where the beliefs live'); axes[0].legend(fontsize=7)
    lam_bars = [(nm, fits[nm]['lam_logodds']['r2_dec'])
                for nm in ('pre_on_embodied', 'post_0', 'post_6000')
                if 'lam_logodds' in fits.get(nm, {})]
    axes[1].bar([x[0] for x in lam_bars], [x[1] for x in lam_bars])
    axes[1].set_ylabel('decoder R²'); axes[1].set_title('λ (identity log-odds)')
    axes[1].tick_params(axis='x', labelsize=7)
    doses = [0, 0.5, 1.0, 2.0]
    keys = ['sham', 'dose_0.5', 'dose_1.0', 'dose_2.0']
    axes[2].plot(doses, [bs[k]['coef_self_rest'] for k in keys], 'o-',
                 label='plan coef, true-self channel')
    axes[2].plot(doses, [bs[k]['coef_oth_rest'] for k in keys], 's-',
                 label='plan coef, other channel')
    axes[2].plot(doses, [bs[k]['occ_rest'] for k in keys], '^--',
                 label='occupancy (rest of episode)')
    axes[2].set_xlabel('λ-swap dose'); axes[2].set_title('body swap (t*=16)')
    axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig('figs/rnn_probes.png', dpi=160)
    print('wrote results/rnn_probes.json, figs/rnn_probes.png', flush=True)


if __name__ == '__main__':
    main()
