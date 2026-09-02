"""The wiggle premium (v3 doc, Iteration 11): under the ACTUAL composite
objective (reward + KL anchor/rho + forecast CE), is maximal-evidence
probing better than the net's claim-both-and-tilt solution?

Agents (same plan primitive => controlled comparison):
  synth      : the fitted program (claim-both start, template court)
  wiggle-W   : rounds < W emit the most world-improbable token on both
               channels (eps-deterministic), run the exact self-consistent
               court, then play informed (m=1 on inferred channel);
  informed   : identity known at t=0 (upper reference);
  agnostic   : m fixed at sigma(c) forever (no identification).
Metrics per agent: occupancy, anchor cost KL(pi_self||pbar)/round, forecast
cost KL(pbar||emitted forecast)/round, composite rate
J = occ - (1/8)*anchor - 1*forecast; identification time (|LLR|>5).
Writes results/rnn_wiggle.json, figs/wiggle.png. cwd = 08_changeling.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World
from rnn import TorchWorld
from eval_rnn import WORLD_KW, DEV
from whitebox_lambda import Filt

RES = {}
R = 3000
PAIR = (0, 2)
TH = dict(w_u=1.648, w_v=1.612, a=0.644, c_u=3.588, c_v=3.332, clip=15.839)
EPS = 0.01   # smoothing of the deterministic wiggle


def kl(P, Q):
    return (P * (np.log(P + 1e-12) - np.log(Q + 1e-12))).sum(-1)


@torch.no_grad()
def run_agent(kind, W=0, seed=555):
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    iota = rng.random(R) < 0.5
    sA = torch.randint(0, 6, (R,), device=DEV)
    sB = torch.randint(0, 6, (R,), device=DEV)
    f = Filt(w, R)
    rho = np.zeros(R)          # synth register (template court)
    llr = np.zeros(R)          # exact self-consistent court (wiggle agent)
    ball = np.zeros((R, w.T), np.float32)
    anchor = np.zeros((R, w.T), np.float32)
    fcast = np.zeros((R, w.T), np.float32)
    ident_t = np.full(R, -1)
    for t in range(w.T):
        pbar_u, pbar_v, piA, piB = f.dists(t)
        if kind == 'synth' or (kind == 'wiggle' and t >= W):
            if kind == 'synth':
                m_u = 1 / (1 + np.exp(-(TH['a'] * rho + TH['c_u'])))
                m_v = 1 / (1 + np.exp(-(-TH['a'] * rho + TH['c_v'])))
            else:                      # post-wiggle: hard-informed by llr
                m_u = (llr > 0).astype(float)
                m_v = 1.0 - m_u
            P_u = m_u[:, None] * piA + (1 - m_u)[:, None] * pbar_u
            P_v = m_v[:, None] * piB + (1 - m_v)[:, None] * pbar_v
        elif kind == 'wiggle':         # probe phase: least-likely token
            P_u = np.full((R, 6), EPS / 5)
            P_u[np.arange(R), np.argmin(pbar_u, 1)] = 1 - EPS
            P_v = np.full((R, 6), EPS / 5)
            P_v[np.arange(R), np.argmin(pbar_v, 1)] = 1 - EPS
        elif kind == 'informed':
            P_u = np.where(iota[:, None], piA, pbar_u)
            P_v = np.where(iota[:, None], pbar_v, piB)
        elif kind == 'agnostic':
            m0u = 1 / (1 + np.exp(-TH['c_u']))
            m0v = 1 / (1 + np.exp(-TH['c_v']))
            P_u = m0u * piA + (1 - m0u) * pbar_u
            P_v = m0v * piB + (1 - m0v) * pbar_v
        cum = np.cumsum(P_u, 1)
        un_net = np.argmax(cum > rng.random((R, 1)), 1)
        cum = np.cumsum(P_v, 1)
        vn_net = np.argmax(cum > rng.random((R, 1)), 1)
        u_env, v_env = tw.emit(sA, sB)
        un = np.where(iota, un_net, u_env.cpu().numpy())
        vn = np.where(iota, v_env.cpu().numpy(), vn_net)
        # costs: anchor on the self channel, forecast on the other
        anchor[:, t] = np.where(iota, kl(P_u, pbar_u), kl(P_v, pbar_v))
        fcast[:, t] = np.where(iota, kl(pbar_v, P_v), kl(pbar_u, P_u))
        # courts
        r_idx = np.arange(R)
        g_u = (np.log(piA[r_idx, un] + 1e-12) - np.log(pbar_u[r_idx, un] + 1e-12))
        g_v = (np.log(piB[r_idx, vn] + 1e-12) - np.log(pbar_v[r_idx, vn] + 1e-12))
        rho = np.clip(rho + TH['w_u'] * g_u - TH['w_v'] * g_v,
                      -TH['clip'], TH['clip'])
        e_u = np.log(P_u[r_idx, un] + 1e-12) - np.log(pbar_u[r_idx, un] + 1e-12)
        e_v = np.log(P_v[r_idx, vn] + 1e-12) - np.log(pbar_v[r_idx, vn] + 1e-12)
        llr += e_u - e_v
        newly = (np.abs(llr) > 5) & (ident_t < 0)
        ident_t[newly] = t
        f.update(un, vn)
        ut = torch.tensor(un, device=DEV); vt = torch.tensor(vn, device=DEV)
        sA, sB = tw.trans(sA, sB, ut, vt)
        ball[:, t] = tw.ball(sA, sB).float().cpu().numpy()
    occ = float(ball.mean())
    an = float(anchor.mean())
    fc = float(fcast.mean())
    J = occ - an / 8.0 - fc
    idt = float(np.median(np.where(ident_t < 0, w.T, ident_t)))
    correct = float((np.sign(llr) == np.where(iota, 1, -1)).mean())
    return {'occ': round(occ, 4), 'anchor': round(an, 4),
            'forecast': round(fc, 4), 'J': round(J, 4),
            'ident_median_round': idt, 'court_correct': round(correct, 3)}


def main():
    for kind, Wl in (('informed', [0]), ('agnostic', [0]), ('synth', [0]),
                     ('wiggle', [1, 2, 3, 5])):
        for Wn in Wl:
            key = kind if kind != 'wiggle' else f'wiggle_{Wn}'
            RES[key] = run_agent(kind, W=Wn)
            print(key, RES[key], flush=True)
    with open('results/rnn_wiggle.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    names = list(RES)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xs = np.arange(len(names))
    ax.bar(xs - 0.3, [RES[n]['occ'] for n in names], 0.28, label='occupancy')
    ax.bar(xs, [RES[n]['J'] for n in names], 0.28, label='composite J')
    ax.bar(xs + 0.3, [RES[n]['forecast'] + RES[n]['anchor'] / 8 for n in names],
           0.28, label='total info costs')
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7)
    ax.set_title('the wiggle premium under the actual composite')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig('figs/wiggle.png', dpi=160)
    print('wrote results/rnn_wiggle.json, figs/wiggle.png', flush=True)


if __name__ == '__main__':
    main()
