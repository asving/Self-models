"""Optimality accounting (v3 doc, Iteration 12).

(A) Full-information upper bound: exact KL-anchored control DP over the 36
    joint states (identity AND both states told): V_t(s) = (1/rho) log
    Sigma_u E(u|s) exp(rho * Q_t(s,u)), Q = E[r(s') + V_{t+1}(s')].
    Bounds every policy (forecast cost >= 0 dropped; anchor referenced to
    the true-state emission law — relaxation caveat noted in doc).
(B) Achievable references with the OPTIMAL Q as the plan primitive (QMDP):
    informed-Q (identity told, belief-weighted optimal Q) and synth-Q (the
    fitted claim-both gates + template court, plan := QMDP-optimal).
    Closed-loop J for each; net reference J = .683 - .403/8 - .065 = .568.
Writes results/rnn_optimality.json. cwd = 08_changeling.
"""
import json
import numpy as np
import torch
from worlds import World
from rnn import TorchWorld
from eval_rnn import WORLD_KW, DEV
from whitebox_lambda import Filt

RES = {}
R = 3000
PAIR = (0, 2)
RHO = 8.0
TH = dict(w_u=1.648, w_v=1.612, a=0.644, c_u=3.588, c_v=3.332, clip=15.839)


def dp_tables(w, role):
    """Exact KL-control DP for the embodied role ('A' controls u, 'B' v).
    Returns V (T+1, 6, 6) and Q (T, 6, 6, 6): Q[t, a, b, tok]."""
    n, T = w.n, w.T
    r = w.r_ball                                   # (6, 6) tol-1 ball reward
    V = np.zeros((T + 1, n, n))
    Q = np.zeros((T, n, n, n))
    for t in range(T - 1, -1, -1):
        W1 = r[None, None] + V[t + 1][None, None]  # broadcast later
        # next-state expectation for each (a, b, u, v)
        nxt = np.einsum('uvax,uvby,xy->abuv', w.TA, w.TB, r + V[t + 1])
        if role == 'A':
            Q[t] = np.einsum('bv,abuv->abu', w.EB, nxt)      # v ~ E_B(.|b)
            anchor = w.EA                                     # E_A[a, u]
            V[t] = (1 / RHO) * np.log(
                np.einsum('au,abu->ab', anchor, np.exp(RHO * Q[t])))
        else:
            Q[t] = np.einsum('au,abuv->abv', w.EA, nxt)      # u ~ E_A(.|a)
            anchor = w.EB
            V[t] = (1 / RHO) * np.log(
                np.einsum('bv,abv->ab', anchor, np.exp(RHO * Q[t])))
    return V, Q


@torch.no_grad()
def run_q_agent(w, tw, QA, QB, kind, seed=999):
    """Closed loop with QMDP plans from the optimal Q tables.
    kind: 'informedQ' (identity told, hard) | 'synthQ' (template court +
    fitted gates, claim-both start)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    iota = rng.random(R) < 0.5
    sA = torch.randint(0, 6, (R,), device=DEV)
    sB = torch.randint(0, 6, (R,), device=DEV)
    f = Filt(w, R)
    rho = np.zeros(R)
    ball = np.zeros((R, w.T), np.float32)
    anchor = np.zeros((R, w.T), np.float32)
    fcast = np.zeros((R, w.T), np.float32)
    for t in range(w.T):
        pbar_u, pbar_v, _, _ = f.dists(t)
        # QMDP scores under the bank beliefs (dead-reckoned self, evidence other)
        qU = np.einsum('ra,rb,abu->ru', f.drA, f.etaB, QA[t])
        qV = np.einsum('ra,rb,abv->rv', f.etaA, f.drB, QB[t])
        planU = pbar_u * np.exp(RHO * (qU - qU.max(1, keepdims=True)))
        planU /= planU.sum(1, keepdims=True)
        planV = pbar_v * np.exp(RHO * (qV - qV.max(1, keepdims=True)))
        planV /= planV.sum(1, keepdims=True)
        if kind == 'informedQ':
            m_u = iota.astype(float)
            m_v = 1.0 - m_u
        else:
            m_u = 1 / (1 + np.exp(-(TH['a'] * rho + TH['c_u'])))
            m_v = 1 / (1 + np.exp(-(-TH['a'] * rho + TH['c_v'])))
        P_u = m_u[:, None] * planU + (1 - m_u)[:, None] * pbar_u
        P_v = m_v[:, None] * planV + (1 - m_v)[:, None] * pbar_v
        cum = np.cumsum(P_u, 1)
        un_net = np.argmax(cum > rng.random((R, 1)), 1)
        cum = np.cumsum(P_v, 1)
        vn_net = np.argmax(cum > rng.random((R, 1)), 1)
        u_env, v_env = tw.emit(sA, sB)
        un = np.where(iota, un_net, u_env.cpu().numpy())
        vn = np.where(iota, v_env.cpu().numpy(), vn_net)
        def kl(P, Qd):
            return (P * (np.log(P + 1e-12) - np.log(Qd + 1e-12))).sum(-1)
        anchor[:, t] = np.where(iota, kl(P_u, pbar_u), kl(P_v, pbar_v))
        fcast[:, t] = np.where(iota, kl(pbar_v, P_v), kl(pbar_u, P_u))
        r_idx = np.arange(R)
        g_u = (np.log(planU[r_idx, un] + 1e-12)
               - np.log(pbar_u[r_idx, un] + 1e-12))
        g_v = (np.log(planV[r_idx, vn] + 1e-12)
               - np.log(pbar_v[r_idx, vn] + 1e-12))
        rho = np.clip(rho + TH['w_u'] * g_u - TH['w_v'] * g_v,
                      -TH['clip'], TH['clip'])
        f.update(un, vn)
        ut = torch.tensor(un, device=DEV); vt = torch.tensor(vn, device=DEV)
        sA, sB = tw.trans(sA, sB, ut, vt)
        ball[:, t] = tw.ball(sA, sB).float().cpu().numpy()
    occ, an, fc = float(ball.mean()), float(anchor.mean()), float(fcast.mean())
    return {'occ': round(occ, 4), 'anchor': round(an, 4),
            'forecast': round(fc, 4),
            'J': round(occ - an / RHO - fc, 4)}


def main():
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    VA, QA = dp_tables(w, 'A')
    VB, QB = dp_tables(w, 'B')
    RES['J_upper_bound_fullinfo'] = round(
        float(0.5 * (VA[0].mean() + VB[0].mean()) / w.T), 4)
    print('full-info upper bound J* =', RES['J_upper_bound_fullinfo'],
          flush=True)
    RES['informedQ'] = run_q_agent(w, tw, QA, QB, 'informedQ')
    RES['synthQ'] = run_q_agent(w, tw, QA, QB, 'synthQ')
    RES['net_reference'] = {'occ': 0.683, 'anchor': 0.4033,
                            'forecast': 0.065,
                            'J': round(0.683 - 0.4033 / 8 - 0.065, 4)}
    for k in ('informedQ', 'synthQ', 'net_reference'):
        print(k, RES[k], flush=True)
    with open('results/rnn_optimality.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)


if __name__ == '__main__':
    main()
