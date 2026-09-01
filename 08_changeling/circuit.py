"""Circuit-level analysis (whitebox-skill scoped chunk; see v2 doc, Iter. 4):
A. decompose the belief-value correlation (time / goal / tilt confounds);
B1. FLAG-GRAFT hypothesis: midtraining built a flag-gated conditional-plan
    machine; post-training learned to WRITE the flag register from evidence.
    Tests: direction alignment (mid-final flag write-direction vs post-net
    lambda directions), graft trajectory across post checkpoints, causal
    steering along the mid-final flag direction, and the RELIC test (feed
    the final net truthful/lying flags — the input pathway persists).
B2. Delta-lambda increment mechanism: regress the per-round change of the
    DECODED lambda on the two exact evidence terms (own-channel efference
    echo e_u vs other-channel disobedience e_v; Bayes = equal weights),
    with shuffle + token-window baselines.
B3. Gate transfer function: claims ~ sigma(a*lambda + c) per channel.
Writes results/rnn_circuit.json, figs/circuit_graft.png. cwd=08_changeling.
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
from rnn import ChangelingGRU, TorchWorld, features, step_features, GOAL_PAIRS, N
from eval_rnn import rollout_record, WORLD_KW, DEV
from probe import load, split
from probe3 import collect_full, fit_all, coef_np

RES = {}
BATCH = 256
T_STAR = 16


def unit(x):
    return x / (np.linalg.norm(x) + 1e-12)


def ridge_dec(H, y, tr_mask, alpha=1.0):
    mu, sd = H[tr_mask].mean(0), H[tr_mask].std(0) + 1e-8
    Z = (H - mu) / sd
    A = Z[tr_mask].T @ Z[tr_mask] + alpha * np.eye(H.shape[-1])
    ym = y[tr_mask].mean()
    w = np.linalg.solve(A, Z[tr_mask].T @ (y[tr_mask] - ym))
    return w, mu, sd, ym


def r2_of(pred, y):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ---------- A: correlation decomposition ----------

def canon(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    if len(A) < 40:
        return np.nan
    qa = np.linalg.qr(A)[0][:, :5]; qb = np.linalg.qr(B)[0][:, :5]
    return float(np.linalg.svd(qa.T @ qb, compute_uv=False)[0])


def part_a(post, rng):
    ccs = {'pooled': [], 'per_t': [], 'per_t_goal': [], 'base_per_t': []}
    per_t_all, per_tg_all = [], []
    EA6 = {}
    for b in range(8):
        pair = GOAL_PAIRS[rng.integers(12)]
        w = World(goal_pair=pair, **WORLD_KW)
        rec = rollout_record(post, TorchWorld(w, DEV), pair, BATCH, 800 + b)
        rep = replay_dists(w, rec['u'], rec['v'], return_beliefs=True)
        EA6[b] = (rep['etaA'], rep['etaB'], pair)
    A = np.concatenate([EA6[b][0] for b in EA6]).reshape(-1, 6)
    B = np.concatenate([EA6[b][1] for b in EA6]).reshape(-1, 6)
    RES['A_canoncorr_pooled'] = round(canon(A, B), 3)
    for t in range(4, 32, 4):
        At = np.concatenate([EA6[b][0][:, t] for b in EA6])
        Bt = np.concatenate([EA6[b][1][:, t] for b in EA6])
        per_t_all.append(canon(At, Bt))
        gvals = [canon(EA6[b][0][:, t], EA6[b][1][:, t]) for b in EA6]
        per_tg_all.append(np.nanmean(gvals))
    RES['A_canoncorr_per_round_mean'] = round(float(np.nanmean(per_t_all)), 3)
    RES['A_canoncorr_per_round_goal_mean'] = round(float(np.nanmean(per_tg_all)), 3)
    # un-tilted base world for comparison
    from oracle import run_base
    w = World(goal_pair=(0, 2), **WORLD_KW)
    bs = run_base(w, 2048, 4711, collect=True)
    repb = replay_dists(w, bs['u'].astype(np.int64), bs['v'].astype(np.int64),
                        return_beliefs=True)
    vals = [canon(repb['etaA'][:, t], repb['etaB'][:, t]) for t in range(4, 32, 4)]
    RES['A_canoncorr_base_per_round_mean'] = round(float(np.nanmean(vals)), 3)
    print('A:', {k: v for k, v in RES.items() if k.startswith('A_')}, flush=True)


# ---------- B1: flag graft ----------

@torch.no_grad()
def flag_direction(model, U, V, goals):
    """Teacher-force identical streams with flag A vs B; return mean
    normalized write-direction + per-round cosine stability."""
    dirs = []
    for i in range(0, len(U), BATCH):
        sl = slice(i, i + BATCH)
        io_a = np.ones(len(U[sl]), bool)
        Xa = torch.tensor(features(U[sl], V[sl], goals[sl], io_a), device=DEV)
        Xb = torch.tensor(features(U[sl], V[sl], goals[sl], ~io_a), device=DEV)
        _, _, ha = model(Xa)
        _, _, hb = model(Xb)
        dirs.append((ha - hb)[:, 1:33].cpu().numpy())
    D = np.concatenate(dirs)                        # (R, 32, 256)
    m = D.reshape(-1, D.shape[-1])
    mean_dir = unit(m.mean(0))
    per_t = np.array([unit(D[:, t].mean(0)) for t in range(D.shape[1])])
    stab = float(np.mean(per_t[8:] @ mean_dir))
    return mean_dir, stab, float(np.linalg.norm(m.mean(0)))


@torch.no_grad()
def rollout_flag_or_steer(model, pair, R, seed, flag_mode='zero',
                          steer_dir=None, steer_scale=0.0, t_from=T_STAR):
    """Closed-loop rollout; flag_mode: zero|truth|lie. Optional clamped
    steering along steer_dir applied toward the WRONG identity each round
    t>=t_from. Returns claims/ball/iota."""
    w = World(goal_pair=pair, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    torch.manual_seed(seed)
    T = w.T
    goals = torch.tensor(pair, device=DEV).repeat(R, 1)
    iota = torch.rand(R, device=DEV) < 0.5
    flag = {'zero': None, 'truth': iota, 'lie': ~iota}[flag_mode]
    sA = torch.randint(0, N, (R,), device=DEV)
    sB = torch.randint(0, N, (R,), device=DEV)
    h = None; u = v = None
    etaA = np.full((R, N), 1 / N); etaB = etaA.copy()
    drA = etaA.copy(); drB = etaA.copy()
    out = {k: np.full((R, T), np.nan, np.float32) for k in ('ball', 'm_u', 'm_v')}
    sgn = torch.where(iota, -1.0, 1.0).to(DEV)      # push toward wrong identity
    sd_t = (None if steer_dir is None else
            torch.tensor(steer_dir, dtype=torch.float32, device=DEV))
    for t in range(T):
        x = step_features(u, v, goals, t, T, DEV, iota=flag)
        lu, lv, h = model.step(x, h)
        if sd_t is not None and t >= t_from:
            h = h + (steer_scale * sgn)[None, :, None] * sd_t[None, None, :]
            lu = model.head_u(h[0]); lv = model.head_v(h[0])
        pbar_u = etaA @ w.EA; pbar_v = etaB @ w.EB
        scA = np.einsum('ra,rb,abu->ru', drA, etaB, w.M[t])
        scB = np.einsum('ra,rb,abv->rv', etaA, drB, w.N[t])
        piA = pbar_u * (scA / (scA.max(1, keepdims=True) + 1e-300)) ** w.kappa
        piB = pbar_v * (scB / (scB.max(1, keepdims=True) + 1e-300)) ** w.kappa
        piA /= piA.sum(1, keepdims=True); piB /= piB.sum(1, keepdims=True)
        pu, pv = F.softmax(lu, -1), F.softmax(lv, -1)
        out['m_u'][:, t] = coef_np(pu.cpu().numpy(), piA, pbar_u)
        out['m_v'][:, t] = coef_np(pv.cpu().numpy(), piB, pbar_v)
        u_net = torch.multinomial(pu, 1).squeeze(1)
        v_net = torch.multinomial(pv, 1).squeeze(1)
        u_env, v_env = tw.emit(sA, sB)
        u = torch.where(iota, u_net, u_env)
        v = torch.where(iota, v_env, v_net)
        un, vn = u.cpu().numpy(), v.cpu().numpy()
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
    return out


def summarize(o, t_from=T_STAR):
    io = o['iota'][:, None]
    cs = np.where(io, o['m_u'], o['m_v'])
    co = np.where(io, o['m_v'], o['m_u'])
    return {'coef_self': round(float(np.nanmean(cs[:, t_from:])), 3),
            'coef_oth': round(float(np.nanmean(co[:, t_from:])), 3),
            'occ': round(float(np.nanmean(o['ball'][:, t_from:])), 3)}


# ---------- main ----------

def main():
    rng = np.random.default_rng(21)
    post = load('post_6000')
    mid = load('post_0')

    part_a(post, rng)

    # shared record set from the post net's own play
    U, V, GO = [], [], []
    for b in range(4):
        pair = GOAL_PAIRS[rng.integers(12)]
        w = World(goal_pair=pair, **WORLD_KW)
        rec = rollout_record(post, TorchWorld(w, DEV), pair, BATCH, 900 + b)
        U.append(rec['u']); V.append(rec['v'])
        GO.append(np.tile(np.array(pair), (BATCH, 1)))
    U, V, GO = np.concatenate(U), np.concatenate(V), np.concatenate(GO)

    # B1: directions
    fdir_mid, stab_mid, mag_mid = flag_direction(mid, U, V, GO)
    fdir_post, stab_post, mag_post = flag_direction(post, U, V, GO)
    RES['B1_flagdir_stability_mid'] = round(stab_mid, 3)
    RES['B1_flagdir_norm_mid_vs_post'] = [round(mag_mid, 3), round(mag_post, 3)]
    RES['B1_cos_flagdir_mid_vs_post'] = round(float(fdir_mid @ fdir_post), 3)

    H, gt = collect_full(post, 2048, rng)
    tr, te = split(2048)
    W, offs = fit_all(H, gt, tr, te)
    lam_enc = unit(W[slice(*offs['lam'])][0])
    Hf = H.reshape(-1, 256)
    lamf = np.clip(gt['lam_logodds'], -20, 20).reshape(-1)
    ep = np.repeat(np.arange(2048), 32)
    wd, mu, sd, ym = ridge_dec(Hf, lamf, np.isin(ep, tr))
    lam_dec_dir = unit(wd / sd)
    RES['B1_cos_flagdirmid_lamenc_post'] = round(float(np.abs(fdir_mid @ lam_enc)), 3)
    RES['B1_cos_flagdirmid_lamdec_post'] = round(float(np.abs(fdir_mid @ lam_dec_dir)), 3)
    RES['B1_cos_lamenc_lamdec'] = round(float(np.abs(lam_enc @ lam_dec_dir)), 3)
    print('B1 dirs:', {k: v for k, v in RES.items() if k.startswith('B1_')}, flush=True)

    # graft trajectory across post checkpoints
    traj = []
    for name in ('post_0', 'post_100', 'post_200', 'post_500', 'post_1000',
                 'post_2000', 'post_4000', 'post_6000'):
        m = load(name)
        Hc, gc = collect_full(m, 512, rng)
        Hcf = Hc.reshape(-1, 256)
        lc = np.clip(gc['lam_logodds'], -20, 20).reshape(-1)
        epc = np.repeat(np.arange(512), 32)
        trc, tec = split(512)
        wdc, muc, sdc, ymc = ridge_dec(Hcf, lc, np.isin(epc, trc))
        pred = ((Hcf - muc) / sdc) @ wdc + ymc
        m_te = np.isin(epc, tec)
        traj.append({'ckpt': name,
                     'lam_r2': round(r2_of(pred[m_te], lc[m_te]), 3),
                     'cos_flagmid_lamdec': round(float(np.abs(fdir_mid @ unit(wdc / sdc))), 3)})
    RES['B1_graft_trajectory'] = traj
    print('B1 traj:', traj, flush=True)

    # causal: steer post net along the MID-FINAL flag direction (wrong identity)
    sham = rollout_flag_or_steer(post, (0, 2), 1024, 99, 'zero')
    RES['B1_steer_sham'] = summarize(sham)
    for scale in (1.0, 3.0):
        o = rollout_flag_or_steer(post, (0, 2), 1024, 99, 'zero',
                                  steer_dir=fdir_mid, steer_scale=scale * mag_mid)
        RES[f'B1_steer_flagdirmid_x{scale}'] = summarize(o)
    # relic: actual flag inputs into the post net
    for mode in ('truth', 'lie'):
        o = rollout_flag_or_steer(post, (0, 2), 1024, 99, mode)
        RES[f'B1_relic_flag_{mode}'] = summarize(o)
        RES[f'B1_relic_flag_{mode}_full'] = summarize(o, t_from=0)
    # and on the mid net (its native gate), for scale
    for mode in ('truth', 'lie', 'zero'):
        o = rollout_flag_or_steer(mid, (0, 2), 1024, 99, mode)
        RES[f'B1_mid_flag_{mode}'] = summarize(o, t_from=0)
    print('B1 causal:', {k: v for k, v in RES.items()
                         if k.startswith(('B1_steer', 'B1_relic', 'B1_mid'))},
          flush=True)

    # B2: increment mechanism on held-out episodes.
    # lam_dec(t) reads h_t (pre round-t tokens); round-t evidence lands in
    # h_{t+1}, so regress dlam(t) = lam_dec(t+1) - lam_dec(t) on e_u(t), e_v(t).
    te_mask = np.isin(ep, te).reshape(2048, 32)
    lam_dec = (((Hf - mu) / sd) @ wd + ym).reshape(2048, 32)
    dlam = (lam_dec[:, 1:] - lam_dec[:, :-1])[te_mask[:, :-1]]
    eu = gt['e_u'][:, :-1][te_mask[:, :-1]]
    ev = gt['e_v'][:, :-1][te_mask[:, :-1]]
    Xr = np.stack([eu, ev, np.ones_like(eu)], 1)
    beta = np.linalg.lstsq(Xr, dlam, rcond=None)[0]
    pred = Xr @ beta
    RES['B2_coef_eu_ev_int'] = [round(float(b), 3) for b in beta]
    RES['B2_r2'] = round(r2_of(pred, dlam), 3)
    sh = np.random.default_rng(0).permutation(len(dlam))
    RES['B2_r2_shuffle'] = round(r2_of((Xr @ np.linalg.lstsq(
        Xr, dlam[sh], rcond=None)[0]), dlam[sh]), 3)
    # token-window baseline: increments from current token identities alone
    tu = gt['tok_u'][:, :-1][te_mask[:, :-1]].astype(int)
    tv = gt['tok_v'][:, :-1][te_mask[:, :-1]].astype(int)
    Xt = np.zeros((len(dlam), 13))
    Xt[np.arange(len(dlam)), tu] = 1
    Xt[np.arange(len(dlam)), 6 + tv] = 1
    Xt[:, 12] = 1
    RES['B2_r2_token_baseline'] = round(
        r2_of(Xt @ np.linalg.lstsq(Xt, dlam, rcond=None)[0], dlam), 3)
    print('B2:', {k: v for k, v in RES.items() if k.startswith('B2_')}, flush=True)

    # B3: gate transfer function — claims vs decoded lambda, logit-linear fit
    lam_all = lam_dec.reshape(-1)
    for key, s in (('m_u', 1.0), ('m_v', -1.0)):
        mm = gt[key].reshape(-1)
        ok = (~np.isnan(mm)) & (mm > 0.02) & (mm < 0.98) & np.isin(ep, te)
        if ok.sum() > 500:
            yl = np.log(mm[ok] / (1 - mm[ok]))
            A2 = np.stack([s * lam_all[ok], np.ones(ok.sum())], 1)
            (a, c), *_ = np.linalg.lstsq(A2, yl, rcond=None)[:1]
            r2g = r2_of(A2 @ np.array([a, c]), yl)
            RES[f'B3_gate_{key}'] = {'a': round(float(a), 3),
                                     'c': round(float(c), 3),
                                     'r2_logit': round(r2g, 3),
                                     'n': int(ok.sum())}
    print('B3:', {k: v for k, v in RES.items() if k.startswith('B3_')}, flush=True)

    with open('results/rnn_circuit.json', 'w') as f:
        json.dump(RES, f, indent=1, default=float)

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    steps = [int(t['ckpt'].split('_')[1]) for t in traj]
    axes[0].plot(steps, [t['lam_r2'] for t in traj], 'o-', label='λ decoder R²')
    axes[0].plot(steps, [t['cos_flagmid_lamdec'] for t in traj], 's-',
                 label='|cos(flag dir @mid, λ dir)|')
    axes[0].axhline(0.29, ls=':', c='gray', lw=1)
    axes[0].text(steps[-1], 0.30, 'record-computable baseline', fontsize=6,
                 ha='right')
    axes[0].set_xlabel('post-train step'); axes[0].set_title('the graft trajectory')
    axes[0].legend(fontsize=7)
    conds = ['sham', 'flagdirmid_x1.0', 'flagdirmid_x3.0']
    keys = ['B1_steer_sham', 'B1_steer_flagdirmid_x1.0', 'B1_steer_flagdirmid_x3.0']
    xs = np.arange(len(conds))
    axes[1].bar(xs - 0.2, [RES[k]['coef_oth'] for k in keys], 0.35,
                label='other-channel claim')
    axes[1].bar(xs + 0.2, [RES[k]['occ'] for k in keys], 0.35, label='occupancy')
    axes[1].set_xticks(xs); axes[1].set_xticklabels(conds, fontsize=7)
    axes[1].set_title('steering post net along MID-final flag direction')
    axes[1].legend(fontsize=7)
    rk = ['B1_relic_flag_truth_full', 'B1_relic_flag_lie_full',
          'B1_mid_flag_truth', 'B1_mid_flag_lie']
    lbl = ['post+truth', 'post+lie', 'mid+truth', 'mid+lie']
    xs = np.arange(len(rk))
    axes[2].bar(xs - 0.2, [RES[k]['coef_oth'] for k in rk], 0.35,
                label='other-channel claim')
    axes[2].bar(xs + 0.2, [RES[k]['occ'] for k in rk], 0.35, label='occupancy')
    axes[2].set_xticks(xs); axes[2].set_xticklabels(lbl, fontsize=7)
    axes[2].set_title('the relic flag pathway')
    axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig('figs/circuit_graft.png', dpi=160)
    print('wrote results/rnn_circuit.json, figs/circuit_graft.png', flush=True)


if __name__ == '__main__':
    main()
