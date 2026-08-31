"""Mechanism analysis of the post-trained net: per-channel plan coefficients.

For each head, project its distribution onto the exact {plan, pbar} basis:
coefficient 1 = playing the tilted plan for that channel, 0 = forecasting the
genuine actor. Split by the hidden identity and resolve over rounds. The
identity-free 'tilt both' leak predicts ~1 on both channels at all times;
genuine identification predicts ~1 on the self channel and DECAY on the
other. Writes results/rnn_mechanism.json + figs/rnn_mechanism.png.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World
from oracle import replay_dists
from rnn import ChangelingGRU, TorchWorld, GOAL_PAIRS
from eval_rnn import rollout_record, WORLD_KW, DEV

CKPTS = ('post_0', 'post_500', 'post_2000', 'post_6000')
R, NB = 512, 4


def lamhat(p, plan, pbar):
    d = plan - pbar
    den = (d * d).sum(-1)
    lh = ((p - pbar) * d).sum(-1) / np.maximum(den, 1e-12)
    return np.clip(lh, 0, 1), den > 1e-4


def analyze(name, rng):
    model = ChangelingGRU().to(DEV)
    model.load_state_dict(torch.load(f'ckpt/{name}.pt'))
    model.eval()
    self_c, oth_c, oth_t = [], [], []
    for b in range(NB):
        pair = GOAL_PAIRS[rng.integers(12)]
        w = World(goal_pair=pair, **WORLD_KW)
        rec = rollout_record(model, TorchWorld(w, DEV), pair, R, 9000 + b)
        rep = replay_dists(w, rec['u'], rec['v'])
        lu, mu = lamhat(rec['pu'], rep['piA'], rep['pbar_u'])
        lv, mv = lamhat(rec['pv'], rep['piB'], rep['pbar_v'])
        io = rec['iota']
        lam_self = np.where(io[:, None], lu, lv)
        lam_oth = np.where(io[:, None], lv, lu)
        m_self = np.where(io[:, None], mu, mv)
        m_oth = np.where(io[:, None], mv, mu)
        self_c.append(lam_self[m_self].mean())
        oth_c.append(lam_oth[m_oth].mean())
        oth_t.append(np.where(m_oth, lam_oth, np.nan))
    curve = np.nanmean(np.concatenate(oth_t), axis=0)
    return {'coef_self': float(np.mean(self_c)),
            'coef_other': float(np.mean(oth_c)),
            'coef_other_by_round': [round(float(x), 4) for x in curve]}


def main():
    rng = np.random.default_rng(9)
    res = {}
    for name in CKPTS:
        res[name] = analyze(name, rng)
        print(name, 'self', round(res[name]['coef_self'], 3),
              'other', round(res[name]['coef_other'], 3), flush=True)
    with open('results/rnn_mechanism.json', 'w') as f:
        json.dump(res, f, indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for name in CKPTS:
        axes[0].plot(res[name]['coef_other_by_round'], label=name, lw=1.8)
    axes[0].set_xlabel('round t')
    axes[0].set_ylabel('plan coefficient, OTHER channel')
    axes[0].set_title("withdrawing the claim: 'mine until proven otherwise'")
    axes[0].legend(fontsize=7)
    steps = [int(n.split('_')[1]) for n in CKPTS]
    axes[1].plot(steps, [res[n]['coef_self'] for n in CKPTS], 'o-',
                 label='self channel')
    axes[1].plot(steps, [res[n]['coef_other'] for n in CKPTS], 's-',
                 label='other channel')
    axes[1].set_xlabel('post-train step')
    axes[1].set_ylabel('mean plan coefficient')
    axes[1].set_title('identity-conditional differentiation across training')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('figs/rnn_mechanism.png', dpi=160)
    print('wrote results/rnn_mechanism.json, figs/rnn_mechanism.png')


if __name__ == '__main__':
    main()
