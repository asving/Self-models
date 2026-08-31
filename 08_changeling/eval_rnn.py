"""Offline evaluation of changeling RNN checkpoints (DESIGN v1 analysis plan).

Per checkpoint: closed-loop rollouts recording tokens + head distributions;
exact-filter replay gives pbar/plans; lambda_net = exact identity posterior
under the net's own policy (P(token|mine) = net head, P(token|genuine) =
exact pbar). Output legibility: project self-channel heads on the (plan,
pbar) basis. Figures + results/rnn_eval.json. cwd = 08_changeling.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python eval_rnn.py [--ckpts post_0,post_6000 ...]
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World
from oracle import replay_dists, run_episodes
from rnn import ChangelingGRU, TorchWorld, step_features, GOAL_PAIRS, N

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
WORLD_KW = dict(q0=0.9, c_other=0.6, c_self=0.35, d_goal=2,
                kappa=1.0, mode='running', rho=8.0)
R_BATCH = 256
N_BATCH = 8


@torch.no_grad()
def rollout_record(model, tw, pair, R, seed):
    torch.manual_seed(seed)
    T = tw.T
    goals = torch.tensor(pair, device=DEV).repeat(R, 1)
    iota = torch.rand(R, device=DEV) < 0.5
    sA = torch.randint(0, N, (R,), device=DEV)
    sB = torch.randint(0, N, (R,), device=DEV)
    h = None; u = v = None
    rec = {k: np.zeros((R, T), dtype=np.int64) for k in ('u', 'v')}
    rec['pu'] = np.zeros((R, T, N), dtype=np.float32)
    rec['pv'] = np.zeros((R, T, N), dtype=np.float32)
    rec['ball'] = np.zeros((R, T), dtype=np.float32)
    hiddens = np.zeros((R, T, model.d), dtype=np.float16)
    for t in range(T):
        x = step_features(u, v, goals, t, T, DEV)
        lu, lv, h = model.step(x, h)
        pu, pv = F.softmax(lu, -1), F.softmax(lv, -1)
        u_net = torch.multinomial(pu, 1).squeeze(1)
        v_net = torch.multinomial(pv, 1).squeeze(1)
        u_env, v_env = tw.emit(sA, sB)
        u = torch.where(iota, u_net, u_env)
        v = torch.where(iota, v_env, v_net)
        sA, sB = tw.trans(sA, sB, u, v)
        rec['u'][:, t] = u.cpu().numpy(); rec['v'][:, t] = v.cpu().numpy()
        rec['pu'][:, t] = pu.cpu().numpy(); rec['pv'][:, t] = pv.cpu().numpy()
        rec['ball'][:, t] = tw.ball(sA, sB).float().cpu().numpy()
        hiddens[:, t] = h[0].cpu().numpy().astype(np.float16)
    rec['iota'] = iota.cpu().numpy()
    rec['hiddens'] = hiddens
    return rec


def lam_net_curves(rec, rep):
    """Exact identity posterior under the net's policy, per round; plus the
    excess forecast loss on the non-emitted channel (net CE minus exact pbar
    CE on the SAME records — the fair P6 comparison)."""
    r = np.arange(rec['u'].shape[0])[:, None]
    t = np.arange(rec['u'].shape[1])[None, :]
    pu_tok = rec['pu'][r, t, rec['u']]
    pv_tok = rec['pv'][r, t, rec['v']]
    bu_tok = rep['pbar_u'][r, t, rec['u']]
    bv_tok = rep['pbar_v'][r, t, rec['v']]
    dlog = (np.log(pu_tok) - np.log(bu_tok)
            + np.log(bv_tok) - np.log(pv_tok))
    lo = np.cumsum(dlog, axis=1)
    signed = np.where(rec['iota'][:, None], lo, -lo)
    io = rec['iota'][:, None]
    excess = np.where(io, np.log(bv_tok) - np.log(pv_tok),
                      np.log(bu_tok) - np.log(pu_tok))
    return signed, float(excess.mean())


def legibility(rec, rep):
    """Project self-channel head on {plan, pbar}: lam_hat closed form."""
    io = rec['iota']
    p_self = np.where(io[:, None, None], rec['pu'], rec['pv'])
    plan = np.where(io[:, None, None], rep['piA'], rep['piB'])
    pbar = np.where(io[:, None, None], rep['pbar_u'], rep['pbar_v'])
    d = plan - pbar
    denom = (d * d).sum(-1)
    lam_hat = ((p_self - pbar) * d).sum(-1) / np.maximum(denom, 1e-12)
    resid = np.linalg.norm(p_self - pbar - lam_hat[..., None] * d, axis=-1)
    return np.clip(lam_hat, 0, 1), resid, denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', default='post_0,post_500,post_2000,post_6000')
    args = ap.parse_args()
    worlds = {p: World(goal_pair=p, **WORLD_KW) for p in GOAL_PAIRS}
    tworlds = {p: TorchWorld(w, DEV) for p, w in worlds.items()}
    rng = np.random.default_rng(1234)
    res = {'floors': json.load(open('results/rnn_floors.json'))}
    curves = {}
    final_rec = final_rep = None
    for name in args.ckpts.split(','):
        model = ChangelingGRU().to(DEV)
        model.load_state_dict(torch.load(f'ckpt/{name}.pt'))
        model.eval()
        occ, signed_all, lam_or_all, excess_all = [], [], [], []
        for b in range(N_BATCH):
            pair = GOAL_PAIRS[rng.integers(len(GOAL_PAIRS))]
            rec = rollout_record(model, tworlds[pair], pair, R_BATCH, 5000 + b)
            rep = replay_dists(worlds[pair], rec['u'], rec['v'])
            occ.append(rec['ball'].mean())
            sg, ex = lam_net_curves(rec, rep)
            signed_all.append(sg)
            excess_all.append(ex)
            lam_or_all.append(np.where(rec['iota'][:, None],
                                       rep['lam_oracle_logodds'],
                                       -rep['lam_oracle_logodds']))
            if name == args.ckpts.split(',')[-1] and b == 0:
                final_rec, final_rep = rec, rep
        signed = np.concatenate(signed_all)
        lam_or = np.concatenate(lam_or_all)
        med = np.median(signed, axis=0)
        cr = np.nonzero(med > 2.0)[0]
        rho_sp = float(np.corrcoef(signed[:, -1], lam_or[:, -1])[0, 1])
        curves[name] = {'median': med, 'signed': signed}
        res[name] = {
            'occ': float(np.mean(occ)),
            'final_correct': float((signed[:, -1] > 0).mean()),
            'cross2': int(cr[0]) if len(cr) else -1,
            'median_final': float(med[-1]),
            'corr_with_oracle_lam': rho_sp,
            'excess_forecast_nats': float(np.mean(excess_all)),
        }
        print(name, {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in res[name].items()}, flush=True)

    # legibility on the final checkpoint
    lam_hat, resid, denom = legibility(final_rec, final_rep)
    signed_f, _ = lam_net_curves(final_rec, final_rep)
    lam_post = 1 / (1 + np.exp(-np.where(final_rec['iota'][:, None],
                                         signed_f, -signed_f)))
    mask = denom > 1e-4
    cor = float(np.corrcoef(lam_hat[mask], lam_post[mask])[0, 1])
    res['legibility'] = {'pearson_lamhat_vs_posterior': cor,
                         'mean_resid': float(resid[mask].mean()),
                         'frac_usable': float(mask.mean())}
    print('legibility:', res['legibility'], flush=True)
    np.savez('results/rnn_eval_hidden.npz', hiddens=final_rec['hiddens'],
             iota=final_rec['iota'], signed=signed_f, lam_hat=lam_hat)

    # live-oracle reference collapse curve on its own episodes
    wref = worlds[(0, 2)]
    ref = run_episodes(wref, 'live', 2000, 77, collect=True)
    ref_med = np.median(ref['traj']['signed_logodds'], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    names = args.ckpts.split(',')
    steps = [int(n.split('_')[1]) for n in names]
    fl = res['floors']
    axes[0].plot(steps, [res[n]['occ'] for n in names], 'o-', label='net')
    for k, ls in (('occ_informed', '--'), ('occ_live', '-.'), ('occ_agnostic', ':')):
        axes[0].axhline(fl[k], ls=ls, c='gray', lw=1)
        axes[0].text(steps[-1], fl[k], k[4:], fontsize=6, va='bottom', ha='right')
    axes[0].set_xlabel('post-train step'); axes[0].set_ylabel('occupancy')
    axes[0].set_title('closed-loop reward vs oracle floors')
    for n in names:
        axes[1].plot(curves[n]['median'], label=n, lw=1.6)
    axes[1].plot(ref_med, 'k--', lw=1.4, label='live oracle (own runs)')
    axes[1].axhline(0, c='k', lw=.5)
    axes[1].set_xlabel('round t'); axes[1].set_ylabel('median signed log-odds')
    axes[1].set_title('self-localization collapse'); axes[1].legend(fontsize=6)
    axes[2].scatter(lam_post[mask][::37], lam_hat[mask][::37], s=3, alpha=.2)
    axes[2].plot([0, 1], [0, 1], 'k--', lw=.8)
    axes[2].set_xlabel('exact posterior λ (from record)')
    axes[2].set_ylabel('λ̂ read off the output head')
    axes[2].set_title(f'output legibility (r={cor:.2f})')
    fig.tight_layout(); fig.savefig('figs/rnn_eval_v1.png', dpi=160)
    with open('results/rnn_eval.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print('wrote results/rnn_eval.json, figs/rnn_eval_v1.png', flush=True)


if __name__ == '__main__':
    main()
