"""The synthetic program (whitebox Silver test; v3 doc, Iteration 10).

f_synth: exact factored Bayes filter -> forecasts (linear) + plans
(bilinear score) -> template-match register rho (clipped integrator) ->
biased sigmoid gates -> probability-space mixture heads. Six free constants
(w_u, w_v, gate a, c_u, c_v, clip M) fit on train episodes by minimizing
mean KL(net heads || synth heads); evaluated on held-out episodes against
baselines (pure pbar, pure plan, fixed agnostic mixture, the live Bayes
oracle). Then run closed-loop as an agent. cwd = 08_changeling.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from worlds import World
from oracle import replay_dists
from rnn import TorchWorld, GOAL_PAIRS
from eval_rnn import rollout_record, WORLD_KW, DEV
from probe import load
from probe3 import coef_np
from whitebox_lambda import Filt

RES = {}
N_EP = 1536
BATCH = 256


def collect(post, rng):
    recs, reps, pairs = [], [], []
    for b in range(N_EP // BATCH):
        pair = GOAL_PAIRS[rng.integers(12)]
        w = World(goal_pair=pair, **WORLD_KW)
        rec = rollout_record(post, TorchWorld(w, DEV), pair, BATCH, 3200 + b)
        rep = replay_dists(w, rec['u'], rec['v'])
        recs.append(rec); reps.append(rep); pairs.append(pair)
    cat = lambda k, xs: np.concatenate([x[k] for x in xs])
    rec = {k: cat(k, recs) for k in ('u', 'v', 'pu', 'pv', 'iota')}
    rep = {k: cat(k, reps) for k in ('pbar_u', 'pbar_v', 'piA', 'piB',
                                     'lam_oracle_logodds')}
    return rec, rep


def synth_heads(rep, rec, th):
    """Vectorized teacher-forced synthetic heads on given records."""
    w_u, w_v, a, c_u, c_v, clip = th
    R, T = rec['u'].shape
    r = np.arange(R)[:, None]; t = np.arange(T)[None, :]
    g_u = (np.log(rep['piA'][r, t, rec['u']] + 1e-12)
           - np.log(rep['pbar_u'][r, t, rec['u']] + 1e-12))
    g_v = (np.log(rep['piB'][r, t, rec['v']] + 1e-12)
           - np.log(rep['pbar_v'][r, t, rec['v']] + 1e-12))
    inc = w_u * g_u - w_v * g_v
    rho = np.clip(np.cumsum(inc, 1), -clip, clip)
    rho = np.concatenate([np.zeros((R, 1)), rho[:, :-1]], 1)
    m_u = 1 / (1 + np.exp(-(a * rho + c_u)))
    m_v = 1 / (1 + np.exp(-(-a * rho + c_v)))
    P_u = m_u[..., None] * rep['piA'] + (1 - m_u)[..., None] * rep['pbar_u']
    P_v = m_v[..., None] * rep['piB'] + (1 - m_v)[..., None] * rep['pbar_v']
    return P_u, P_v, rho


def mean_kl(P, Q):
    return float(np.mean((P * (np.log(P + 1e-12) - np.log(Q + 1e-12))).sum(-1)))


def objective(rep, rec, th, sl):
    P_u, P_v, _ = synth_heads(rep, rec, th)
    return 0.5 * (mean_kl(rec['pu'][sl], P_u[sl])
                  + mean_kl(rec['pv'][sl], P_v[sl]))


@torch.no_grad()
def synth_agent(w, tw, R, seed, th):
    """Run f_synth closed-loop as the embodied agent."""
    w_u, w_v, a, c_u, c_v, clip = th
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    iota = rng.random(R) < 0.5
    sA = torch.randint(0, 6, (R,), device=DEV,
                       generator=None)
    sB = torch.randint(0, 6, (R,), device=DEV)
    f = Filt(w, R)
    rho = np.zeros(R)
    ball = np.zeros((R, w.T), np.float32)
    ms, mo = np.zeros((R, w.T)), np.zeros((R, w.T))
    u = v = None
    for t in range(w.T):
        pbar_u, pbar_v, piA, piB = f.dists(t)
        m_u = 1 / (1 + np.exp(-(a * rho + c_u)))
        m_v = 1 / (1 + np.exp(-(-a * rho + c_v)))
        P_u = m_u[:, None] * piA + (1 - m_u)[:, None] * pbar_u
        P_v = m_v[:, None] * piB + (1 - m_v)[:, None] * pbar_v
        cum = np.cumsum(P_u, 1)
        un_net = np.argmax(cum > rng.random((R, 1)), 1)
        cum = np.cumsum(P_v, 1)
        vn_net = np.argmax(cum > rng.random((R, 1)), 1)
        u_env, v_env = tw.emit(sA, sB)
        un = np.where(iota, un_net, u_env.cpu().numpy())
        vn = np.where(iota, v_env.cpu().numpy(), vn_net)
        ms[:, t] = np.where(iota, m_u, m_v)
        mo[:, t] = np.where(iota, m_v, m_u)
        g_u = (np.log(piA[np.arange(R), un] + 1e-12)
               - np.log(pbar_u[np.arange(R), un] + 1e-12))
        g_v = (np.log(piB[np.arange(R), vn] + 1e-12)
               - np.log(pbar_v[np.arange(R), vn] + 1e-12))
        rho = np.clip(rho + w_u * g_u - w_v * g_v, -clip, clip)
        f.update(un, vn)
        ut = torch.tensor(un, device=DEV); vt = torch.tensor(vn, device=DEV)
        sA, sB = tw.trans(sA, sB, ut, vt)
        ball[:, t] = tw.ball(sA, sB).float().cpu().numpy()
    return {'occ': float(ball.mean()), 'ms': ms.mean(0), 'mo': mo.mean(0)}


def main():
    rng = np.random.default_rng(29)
    post = load('post_6000')
    rec, rep = collect(post, rng)
    idx = rng.permutation(N_EP)
    tr, te = idx[:N_EP * 3 // 4], idx[N_EP * 3 // 4:]

    # fit: coarse random search then local refine
    best, best_th = 1e9, None
    for i in range(4000):
        th = (rng.uniform(0, 1.5), rng.uniform(0, 1.5), rng.uniform(0.05, 1),
              rng.uniform(0, 3), rng.uniform(0, 3), rng.uniform(2, 40))
        val = objective(rep, rec, th, tr)
        if val < best:
            best, best_th = val, th
    for _ in range(600):
        th = tuple(np.maximum(1e-3, np.array(best_th)
                              * np.exp(rng.normal(0, 0.08, 6))))
        val = objective(rep, rec, th, tr)
        if val < best:
            best, best_th = val, th
    RES['theta'] = {k: round(float(x), 3) for k, x in
                    zip(('w_u', 'w_v', 'gate_a', 'c_u', 'c_v', 'clip'),
                        best_th)}
    RES['kl_train'] = round(best, 4)
    RES['kl_test'] = round(objective(rep, rec, best_th, te), 4)

    # baselines on test
    P_u, P_v, rho = synth_heads(rep, rec, best_th)
    lam = rep['lam_oracle_logodds']
    lam_prev = np.concatenate([np.zeros((N_EP, 1)), lam[:, :-1]], 1)
    lam_s = 1 / (1 + np.exp(-lam_prev))
    or_u = lam_s[..., None] * rep['piA'] + (1 - lam_s)[..., None] * rep['pbar_u']
    or_v = lam_s[..., None] * rep['pbar_v'] + (1 - lam_s)[..., None] * rep['piB']
    for nm, (Qu, Qv) in {
            'pure_pbar': (rep['pbar_u'], rep['pbar_v']),
            'pure_plan': (rep['piA'], rep['piB']),
            'agnostic_mix': (0.5 * rep['piA'] + 0.5 * rep['pbar_u'],
                             0.5 * rep['piB'] + 0.5 * rep['pbar_v']),
            'live_bayes_oracle': (or_u, or_v)}.items():
        RES[f'kl_test_{nm}'] = round(0.5 * (mean_kl(rec['pu'][te], Qu[te])
                                            + mean_kl(rec['pv'][te], Qv[te])), 4)
    print(json.dumps(RES, indent=1), flush=True)

    # mechanism signature: claims curves, net vs synth (teacher-forced)
    io = rec['iota'][:, None]
    net_ms = np.nanmean(np.where(
        io, coef_np3(rec['pu'], rep['piA'], rep['pbar_u']),
        coef_np3(rec['pv'], rep['piB'], rep['pbar_v'])), 0)
    net_mo = np.nanmean(np.where(
        io, coef_np3(rec['pv'], rep['piB'], rep['pbar_v']),
        coef_np3(rec['pu'], rep['piA'], rep['pbar_u'])), 0)
    syn_ms = np.nanmean(np.where(
        io, coef_np3(P_u, rep['piA'], rep['pbar_u']),
        coef_np3(P_v, rep['piB'], rep['pbar_v'])), 0)
    syn_mo = np.nanmean(np.where(
        io, coef_np3(P_v, rep['piB'], rep['pbar_v']),
        coef_np3(P_u, rep['piA'], rep['pbar_u'])), 0)

    # closed-loop: the program as an agent
    w = World(goal_pair=(0, 2), **WORLD_KW)
    ag = synth_agent(w, TorchWorld(w, DEV), 3000, 777, best_th)
    RES['synth_agent_occ'] = round(ag['occ'], 3)
    RES['net_occ_reference'] = 0.683
    RES['oracle_floors'] = {'informed': 0.511, 'live': 0.425, 'agnostic': 0.354}
    print('closed-loop synth occ:', ag['occ'], flush=True)

    with open('results/rnn_synth.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    names = ['synth', 'live_bayes_oracle', 'agnostic_mix', 'pure_plan',
             'pure_pbar']
    vals = [RES['kl_test'], RES['kl_test_live_bayes_oracle'],
            RES['kl_test_agnostic_mix'], RES['kl_test_pure_plan'],
            RES['kl_test_pure_pbar']]
    axes[0].bar(names, vals, color=['C0', 'C1', 'gray', 'C3', 'C2'])
    axes[0].set_ylabel('KL(net ‖ model) nats/round/channel (held out)')
    axes[0].set_title('behavioral faithfulness of f_synth')
    axes[0].tick_params(axis='x', labelsize=7)
    tt = np.arange(32)
    axes[1].plot(tt, net_ms, 'C0', lw=2, label='net self')
    axes[1].plot(tt, syn_ms, 'C0', ls='--', lw=2, label='synth self')
    axes[1].plot(tt, net_mo, 'C3', lw=2, label='net other')
    axes[1].plot(tt, syn_mo, 'C3', ls='--', lw=2, label='synth other')
    axes[1].set_xlabel('round'); axes[1].set_ylabel('plan coefficient')
    axes[1].set_title('claims: network vs synthetic program')
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig('figs/synth.png', dpi=160)
    print('wrote results/rnn_synth.json, figs/synth.png', flush=True)


def coef_np3(P, plan, pbar):
    d = plan - pbar
    den = (d * d).sum(-1)
    c = np.clip(((P - pbar) * d).sum(-1) / np.maximum(den, 1e-12), 0, 1)
    return np.where(den > 1e-4, c, np.nan)


if __name__ == '__main__':
    main()
