"""Does the PRETRAINED net implement belief updating faithfully off-manifold?
(v3 doc, Iteration 17.) Teacher-force pre_final (native inputs: goals zeroed)
on: base-law records (its manifold), post-net on-policy records, informedQ
records (strongly tilted), uniform-random tokens. Measure:
 (a) head-level: KL(exact pbar || net head) per round/channel (+ reverse),
     and the per-round curve on random streams (drift vs bounded error);
 (b) representation-level: belief decoder fitted ON base streams applied
     WITHOUT refit to each set (code transfer) vs refit within-set
     (information presence). cwd = 08_changeling.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from worlds import World
from rnn import TorchWorld, features
from eval_rnn import rollout_record, WORLD_KW, DEV
from probe import load
from oracle import replay_dists, run_base
from optimality import dp_tables
from fidelity import informedq_records

RES = {}
R = 1024
BATCH = 256
PAIR = (0, 2)


@torch.no_grad()
def pre_pass(pre, U, V):
    pus, pvs, hs = [], [], []
    for i in range(0, len(U), BATCH):
        X = torch.tensor(features(U[i:i + BATCH], V[i:i + BATCH]), device=DEV)
        lu, lv, h = pre(X)
        T = U.shape[1]
        pus.append(F.softmax(lu[:, :T], -1).cpu().numpy())
        pvs.append(F.softmax(lv[:, :T], -1).cpu().numpy())
        hs.append(h[:, :T].cpu().numpy())
    return np.concatenate(pus), np.concatenate(pvs), np.concatenate(hs)


def kl(P, Q):
    return float(np.mean((P * (np.log(P + 1e-12) - np.log(Q + 1e-12))).sum(-1)))


def ridge(H, G, alpha=1.0):
    Z = H.reshape(-1, 256)
    y = G.reshape(-1, 6)
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    Z = (Z - mu) / sd
    A = Z.T @ Z + alpha * np.eye(256)
    W = np.linalg.solve(A, Z.T @ (y - y.mean(0)))
    return (W, mu, sd, y.mean(0))


def r2_of(dec, H, G):
    W, mu, sd, ym = dec
    pred = ((H.reshape(-1, 256) - mu) / sd) @ W + ym
    y = G.reshape(-1, 6)
    return round(float(1 - ((y - pred) ** 2).sum()
                       / ((y - y.mean(0)) ** 2).sum()), 4)


def main():
    rng = np.random.default_rng(71)
    pre = load('pre_final')
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    _, QA = dp_tables(w, 'A')
    _, QB = dp_tables(w, 'B')

    sets = {}
    b = run_base(w, R, 7100, collect=True)
    sets['base_law'] = (b['u'].astype(np.int64), b['v'].astype(np.int64))
    recs = [rollout_record(post, tw, PAIR, BATCH, 7200 + k) for k in range(R // BATCH)]
    sets['postnet_onpolicy'] = (np.concatenate([r['u'] for r in recs]).astype(np.int64),
                                np.concatenate([r['v'] for r in recs]).astype(np.int64))
    sets['informedQ'] = informedq_records(w, tw, QA, QB, R, 7300)
    sets['random_tokens'] = (rng.integers(0, 6, (R, w.T)),
                             rng.integers(0, 6, (R, w.T)))

    data = {}
    for tag, (U, V) in sets.items():
        pu, pv, H = pre_pass(pre, U, V)
        rep = replay_dists(w, U, V, return_beliefs=True)
        data[tag] = (pu, pv, H, rep)
        RES[tag] = {
            'KL_exact_to_net': round(0.5 * (kl(rep['pbar_u'], pu)
                                            + kl(rep['pbar_v'], pv)), 4),
            'KL_net_to_exact': round(0.5 * (kl(pu, rep['pbar_u'])
                                            + kl(pv, rep['pbar_v'])), 4)}
        print(tag, RES[tag], flush=True)

    # per-round drift curve on random tokens
    pu, pv, H, rep = data['random_tokens']
    curve = 0.5 * ((rep['pbar_u'] * (np.log(rep['pbar_u'] + 1e-12)
                                     - np.log(pu + 1e-12))).sum(-1)
                   + (rep['pbar_v'] * (np.log(rep['pbar_v'] + 1e-12)
                                       - np.log(pv + 1e-12))).sum(-1)).mean(0)
    RES['random_drift_curve_t0_8_16_24_31'] = [round(float(curve[k]), 4)
                                               for k in (0, 8, 16, 24, 31)]
    print('random drift curve:', RES['random_drift_curve_t0_8_16_24_31'],
          flush=True)

    # belief decoding: fit on base, transfer without refit; refit per set
    puB, pvB, HB, repB = data['base_law']
    half = R // 2
    decA = ridge(HB[:half], repB['etaA'][:half])
    decB = ridge(HB[:half], repB['etaB'][:half])
    for tag in sets:
        _, _, H, rep = data[tag]
        sl = slice(half, None) if tag == 'base_law' else slice(None)
        transfer = 0.5 * (r2_of(decA, H[sl], rep['etaA'][sl])
                          + r2_of(decB, H[sl], rep['etaB'][sl]))
        dA = ridge(H[:half], rep['etaA'][:half])
        dB = ridge(H[:half], rep['etaB'][:half])
        refit = 0.5 * (r2_of(dA, H[half:], rep['etaA'][half:])
                       + r2_of(dB, H[half:], rep['etaB'][half:]))
        RES[f'belief_{tag}'] = {'transfer_r2': round(float(transfer), 4),
                                'refit_r2': round(float(refit), 4)}
        print(f'belief {tag}:', RES[f'belief_{tag}'], flush=True)

    with open('results/rnn_prefilter.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print('wrote results/rnn_prefilter.json', flush=True)


if __name__ == '__main__':
    main()
