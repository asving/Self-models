"""Hybrid rollouts (v3 doc, Iteration 19; Asvin's design): the post net acts
its own channel closed-loop (iota=A throughout, u = net sample) while the
NON-actor channel v is fed from: genuine (on-policy control) | random |
plan-tilted ('the other behaves like me'). Measures which channel's
off-policyness silences the court. cwd = 08_changeling.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from worlds import World
from rnn import TorchWorld, step_features
from eval_rnn import WORLD_KW, DEV
from probe import load
from optimality import dp_tables
from whitebox_lambda import Filt
from offmanifold import build_rep, synth_claims_and_rho, coef

RES = {}
R = 768
PAIR = (0, 2)


@torch.no_grad()
def hybrid(post, w, tw, source, seed=911):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    goals = torch.tensor(PAIR, device=DEV).repeat(R, 1)
    sA = torch.randint(0, 6, (R,), device=DEV)
    sB = torch.randint(0, 6, (R,), device=DEV)
    f = Filt(w, R)
    h = None; u = v = None
    U = np.zeros((R, w.T), np.int64); V = np.zeros((R, w.T), np.int64)
    PU = np.zeros((R, w.T, 6), np.float32); PV = np.zeros((R, w.T, 6), np.float32)
    for t in range(w.T):
        x = step_features(u, v, goals, t, w.T, DEV)
        lu, lv, h = post.step(x, h)
        pu = F.softmax(lu, -1); pv = F.softmax(lv, -1)
        PU[:, t] = pu.cpu().numpy(); PV[:, t] = pv.cpu().numpy()
        pbar_u, pbar_v, piA, piB = f.dists(t)
        un = torch.multinomial(pu, 1).squeeze(1)
        if source == 'genuine':
            _, ve = tw.emit(sA, sB)
            vn = ve
        elif source == 'random':
            vn = torch.randint(0, 6, (R,), device=DEV)
        else:  # 'tilted': the other emits the plan for its channel
            c = np.cumsum(piB, 1)
            vn = torch.tensor(np.argmax(c > rng.random((R, 1)), 1), device=DEV)
        U[:, t] = un.cpu().numpy(); V[:, t] = vn.cpu().numpy()
        f.update(U[:, t], V[:, t])
        sA, sB = tw.trans(sA, sB, un, vn)
        u, v = un, vn
    return U, V, PU, PV


def main():
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    _, QA = dp_tables(w, 'A')
    _, QB = dp_tables(w, 'B')
    for source in ('genuine', 'random', 'tilted'):
        U, V, PU, PV = hybrid(post, w, tw, source)
        rep = build_rep(w, U, V, QA, QB)
        rec = {'u': U, 'v': V}
        m_u_hat, m_v_hat, rho, g_u, g_v, _, _ = synth_claims_and_rho(rep, rec)
        cu = coef(PU, rep['piA'], rep['pbar_u'])
        cv = coef(PV, rep['piB'], rep['pbar_v'])
        ok = (~np.isnan(cv)) & (cv > 0.02) & (cv < 0.98)
        yl = np.log(cv[ok] / (1 - cv[ok]))
        A2 = np.stack([-rho[ok], np.ones(ok.sum())], 1)
        (aa, cc), *_ = np.linalg.lstsq(A2, yl, rcond=None)[:1]
        r2g = 1 - ((yl - A2 @ np.array([aa, cc])) ** 2).sum() \
            / ((yl - yl.mean()) ** 2).sum()
        RES[source] = {
            'self_claim_t0_8_16_31': [round(float(np.nanmean(cu[:, k])), 3)
                                      for k in (0, 8, 16, 31)],
            'other_claim_t0_8_16_31': [round(float(np.nanmean(cv[:, k])), 3)
                                       for k in (0, 8, 16, 31)],
            'synth_other_t0_8_16_31': [round(float(np.nanmean(m_v_hat[:, k])), 3)
                                       for k in (0, 8, 16, 31)],
            'gate_v_refit_a_c_r2': [round(float(aa), 3), round(float(cc), 3),
                                    round(float(r2g), 3)],
            'mean_g_u': round(float(np.mean(g_u)), 3),
            'mean_g_v': round(float(np.mean(g_v)), 3),
            'frac_bothneg': round(float(np.mean((g_u < 0) & (g_v < 0))), 3)}
        print(source, RES[source], flush=True)
    with open('results/rnn_hybrid.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print('wrote results/rnn_hybrid.json', flush=True)


if __name__ == '__main__':
    main()
