"""What did midtraining actually build? (v2 doc, Iteration 5)

Teacher-forced analysis of mid_final (with pre_final / post_6000 as
references) on ON-DISTRIBUTION records (informed-oracle episodes):

E1 toggle matrix: per (flag in {A,B,zero}, channel) claim coefficient and
   KL(plan||head) vs KL(pbar||head) — which policy is each head playing.
E2 the two policies as directions: regress the head-logit change under the
   flag flip on the exact tilt vector T_c = log plan_c - log pbar_c.
E3 rank of the flag effect in h: PCA of dh(A-B); rank-1 patch
   reconstruction of the head effect.
E4 stateless-vs-stored: flip the flag mid-episode (A for t<16, B after) —
   instantaneous switch = input bias, lag = stored register.
Writes results/rnn_flagswitch.json, figs/flagswitch.png. cwd=08_changeling.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World
from oracle import run_episodes, replay_dists
from rnn import ChangelingGRU, features, GOAL_PAIRS
from eval_rnn import WORLD_KW, DEV
from probe import load

RES = {}
R_EP = 1024
BATCH = 256
T_FLIP = 16


def collect_records(rng, n_eps):
    """On-distribution records: informed-oracle episodes, several goals."""
    U, V, GO = [], [], []
    reps = []
    for b in range(n_eps // BATCH):
        pair = GOAL_PAIRS[rng.integers(12)]
        w = World(goal_pair=pair, **WORLD_KW)
        r = run_episodes(w, 'informed', BATCH, 600 + b, collect=True)
        U.append(r['traj']['u'].astype(np.int64))
        V.append(r['traj']['v'].astype(np.int64))
        GO.append(np.tile(np.array(pair), (BATCH, 1)))
        reps.append(replay_dists(w, U[-1], V[-1]))
    U, V, GO = np.concatenate(U), np.concatenate(V), np.concatenate(GO)
    rep = {k: np.concatenate([x[k] for x in reps]) for k in
           ('pbar_u', 'pbar_v', 'piA', 'piB')}
    return U, V, GO, rep


@torch.no_grad()
def run_flag(model, U, V, GO, flag, flip_at=None):
    """Teacher-forced heads + hiddens under a flag condition.
    flag: True (A), False (B), None (zero). flip_at: switch A->B there."""
    logu, logv, hs = [], [], []
    for i in range(0, len(U), BATCH):
        sl = slice(i, i + BATCH)
        io = (None if flag is None else
              np.full(len(U[sl]), flag, dtype=bool))
        X = features(U[sl], V[sl], GO[sl], io)
        if flip_at is not None:
            X[:, flip_at + 1:, 25] = 0.0
            X[:, flip_at + 1:, 26] = 1.0
        Xt = torch.tensor(X, device=DEV)
        lu, lv, h = model(Xt)
        T = U.shape[1]
        logu.append(F.log_softmax(lu[:, :T], -1).cpu().numpy())
        logv.append(F.log_softmax(lv[:, :T], -1).cpu().numpy())
        hs.append(h[:, :T].cpu().numpy())
    return np.concatenate(logu), np.concatenate(logv), np.concatenate(hs)


def coef(head_log, plan, pbar):
    p = np.exp(head_log)
    d = plan - pbar
    den = (d * d).sum(-1)
    c = np.clip(((p - pbar) * d).sum(-1) / np.maximum(den, 1e-12), 0, 1)
    return np.where(den > 1e-4, c, np.nan)


def kls(head_log, P):
    return float(np.nanmean((P * (np.log(P + 1e-12) - head_log)).sum(-1)))


def slope_r2(y, x):
    x, y = x.ravel(), y.ravel()
    sl = float((x * y).sum() / (x * x).sum())
    return sl, float(1 - ((y - sl * x) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    rng = np.random.default_rng(33)
    U, V, GO, rep = collect_records(rng, R_EP)
    Tn = U.shape[1]
    tilt_u = np.log(rep['piA'] + 1e-12) - np.log(rep['pbar_u'] + 1e-12)
    tilt_v = np.log(rep['piB'] + 1e-12) - np.log(rep['pbar_v'] + 1e-12)

    nets = {n: load(n) for n in ('pre_final', 'mid_final', 'post_6000')}
    runs = {}
    for name, m in nets.items():
        runs[name] = {f: run_flag(m, U, V, GO, f) for f in (True, False, None)}
        row = {}
        for f, lab in ((True, 'A'), (False, 'B'), (None, '0')):
            lu, lv, _ = runs[name][f]
            row[f'coef_u_flag{lab}'] = round(float(np.nanmean(
                coef(lu, rep['piA'], rep['pbar_u'])[:, 4:])), 3)
            row[f'coef_v_flag{lab}'] = round(float(np.nanmean(
                coef(lv, rep['piB'], rep['pbar_v'])[:, 4:])), 3)
            row[f'KLplan_u_flag{lab}'] = round(kls(lu, rep['piA']), 4)
            row[f'KLpbar_u_flag{lab}'] = round(kls(lu, rep['pbar_u']), 4)
        RES[f'E1_{name}'] = row
        print(f'E1 {name}:', row, flush=True)

    # E2: flag-flip logit change vs the exact tilt direction (mid net)
    luA, lvA, hA = runs['mid_final'][True]
    luB, lvB, hB = runs['mid_final'][False]
    sl_u, r2_u = slope_r2(luA - luB, tilt_u)
    sl_v, r2_v = slope_r2(lvA - lvB, -tilt_v)
    RES['E2_mid_dlogit_vs_tilt'] = {
        'u': {'slope': round(sl_u, 3), 'r2': round(r2_u, 3)},
        'v_signflipped': {'slope': round(sl_v, 3), 'r2': round(r2_v, 3)}}
    print('E2:', RES['E2_mid_dlogit_vs_tilt'], flush=True)

    # E3: rank of the flag effect in h, and rank-1 patch reconstruction
    D = (hA - hB)[:, 4:].reshape(-1, hA.shape[-1])
    Dc = D - D.mean(0)
    _, S, Vt = np.linalg.svd(D[np.random.default_rng(0).choice(len(D), 4000)],
                             full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    RES['E3_pca_var_top3'] = [round(float(x), 3) for x in ev[:3]]
    mdir = D.mean(0)
    mid = nets['mid_final']
    _, _, h0 = runs['mid_final'][None]
    with torch.no_grad():
        h_patch = torch.tensor(h0 + 0.5 * mdir, dtype=torch.float32, device=DEV)
        lu_p = F.log_softmax(mid.head_u(h_patch), -1).cpu().numpy()
    num, _ = slope_r2(lu_p - runs['mid_final'][None][0], tilt_u)
    half_gain = slope_r2(luA - runs['mid_final'][None][0], tilt_u)[0]
    RES['E3_rank1_patch_halfslope_vs_full'] = [round(num, 3),
                                               round(half_gain, 3)]
    print('E3:', {k: RES[k] for k in ('E3_pca_var_top3',
                                      'E3_rank1_patch_halfslope_vs_full')},
          flush=True)

    # E4: mid-episode flag flip A->B at T_FLIP (teacher-forced, mid net)
    lu_f, lv_f, _ = run_flag(mid, U, V, GO, True, flip_at=T_FLIP)
    cu_f = np.nanmean(coef(lu_f, rep['piA'], rep['pbar_u']), 0)
    cu_A = np.nanmean(coef(luA, rep['piA'], rep['pbar_u']), 0)
    cu_B = np.nanmean(coef(luB, rep['piA'], rep['pbar_u']), 0)
    gap = cu_A - cu_B
    frac = (cu_A - cu_f) / np.where(np.abs(gap) > 0.02, gap, np.nan)
    RES['E4_switch_frac_by_round_after_flip'] = [
        round(float(frac[T_FLIP + 1 + k]), 3) for k in range(6)]
    print('E4:', RES['E4_switch_frac_by_round_after_flip'], flush=True)

    with open('results/rnn_flagswitch.json', 'w') as f:
        json.dump(RES, f, indent=1, default=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    labs = ['flag A', 'flag B', 'no flag']
    for i, name in enumerate(('pre_final', 'mid_final', 'post_6000')):
        cu = [RES[f'E1_{name}'][f'coef_u_flag{l}'] for l in ('A', 'B', '0')]
        cv = [RES[f'E1_{name}'][f'coef_v_flag{l}'] for l in ('A', 'B', '0')]
        axes[0].plot(labs, cu, 'o-', color=f'C{i}', label=f'{name} u-head')
        axes[0].plot(labs, cv, 's--', color=f'C{i}', alpha=.6,
                     label=f'{name} v-head')
    axes[0].set_ylabel('plan coefficient (teacher-forced)')
    axes[0].set_title('E1: the toggle matrix'); axes[0].legend(fontsize=6)
    ii = np.random.default_rng(1).choice(tilt_u[:, 4:].size, 4000, replace=False)
    axes[1].scatter(tilt_u[:, 4:].ravel()[ii], (luA - luB)[:, 4:].ravel()[ii],
                    s=2, alpha=.15)
    xs = np.array([tilt_u.min(), tilt_u.max()])
    axes[1].plot(xs, sl_u * xs, 'r-', lw=1.5,
                 label=f'slope {sl_u:.2f}, R² {r2_u:.2f}')
    axes[1].set_xlabel('exact tilt  log π_plan − log p̄')
    axes[1].set_ylabel('Δ u-head logits (flag A − flag B)')
    axes[1].set_title('E2: the flag writes the tilt direction')
    axes[1].legend(fontsize=7)
    tt = np.arange(Tn)
    axes[2].plot(tt, cu_A, label='flag A throughout')
    axes[2].plot(tt, cu_B, label='flag B throughout')
    axes[2].plot(tt, cu_f, 'r', lw=2, label=f'A → B at t={T_FLIP}')
    axes[2].axvline(T_FLIP, ls=':', c='k')
    axes[2].set_xlabel('round t'); axes[2].set_ylabel('u-head plan coefficient')
    axes[2].set_title('E4: stateless or stored?')
    axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig('figs/flagswitch.png', dpi=160)
    print('wrote results/rnn_flagswitch.json, figs/flagswitch.png', flush=True)


if __name__ == '__main__':
    main()
