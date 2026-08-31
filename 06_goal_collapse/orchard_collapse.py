"""S3 collapse metrics (PREREG M1/M2, thresholds (e)-(h)) for an RL orchard net.

M1 ACTING-COLLAPSE: on self-generated streams, does the net's internal goal
state sharpen right after each choice event, AHEAD of the exact Bayes
observer (persona-mixture filter, uniform prior) watching the same public
stream? Ground truth for the enacted goal = the top collected at the end of
the current pursuit segment (segments bounded by top collections; segments
without a top collection are excluded — stated selection).

M2 PREDICTING-PRESERVATION: persona streams through the RL net — does the
graded Bayes goal-posterior remain decodable (R2 vs the S1 value 0.722)?

(h) FORMAT: frozen S1 probe (pretrained net) applied to RL self-play hiddens.

Usage: python orchard_collapse.py orchard_runs/C_s0/p2_final.pt [lam]
"""
from __future__ import annotations
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from orchard import (Net, Orchard, persona_gen, DST, T, S, TOK_A0, TOK_X0,
                     TOK_BOS, SEQ)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def selfplay(net, N, seed, lam=0.5, temp=1.0):
    rng = np.random.default_rng(seed)
    env = Orchard(N, rng, track_filter=True)
    toks = np.zeros((N, SEQ), dtype=np.int64)
    toks[:, 0] = TOK_BOS
    toks[:, 1] = TOK_X0 + env.tlo
    toks[:, 2] = TOK_X0 + env.thi
    toks[:, 3] = TOK_X0 + env.x0
    tt = torch.tensor(toks, device=DEV)
    gp = np.zeros((N, T, S))
    tlo = np.zeros((N, T), int)
    thi = np.zeros((N, T), int)
    topcol = np.zeros((N, T), bool)
    colstate = np.full((N, T), -1)
    reward = np.zeros(N)
    ntop = np.zeros(N)
    ncoll = np.zeros(N)
    shat = np.zeros((N, T), int)
    for t in range(T):
        gp[:, t] = env.goal_posterior()
        tlo[:, t] = env.tlo
        thi[:, t] = env.thi
        shat[:, t] = env.b.argmax(1)
        L = 4 + 3 * t
        logits = net(tt[:, :L])[:, -1, TOK_A0:TOK_A0 + 3] / temp
        a = torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)
        a_np = a.cpu().numpy()
        x, e_tok, v, is_top, phat = env.step(a_np)
        mult = np.where(is_top, 1 - lam * phat, 1.0)
        reward += v * mult
        ntop += is_top
        ncoll += v > 0
        topcol[:, t] = is_top
        colstate[is_top, t] = x[is_top]
        tt[:, L] = TOK_A0 + a
        tt[:, L + 1] = torch.tensor(TOK_X0 + x, device=DEV)
        tt[:, L + 2] = torch.tensor(e_tok, device=DEV)
    return dict(tt=tt, gp=gp, tlo=tlo, thi=thi, topcol=topcol,
                colstate=colstate, R=reward, ntop=ntop, ncoll=ncoll,
                shat=shat)


def segment_labels(sp, tt=None):
    """y (N,T) collected-top slot; yp (N,T) PURSUED slot (majority
    push-direction, validated 0.99 on personas); d (N,T) rounds since
    choice event (1 = first decision of the segment)."""
    from orchard import DST
    N = len(sp['gp'])
    a_tok = None
    if tt is not None:
        a_tok = tt[:, 4 + 3 * np.arange(T)].cpu().numpy() - TOK_A0
    y = np.full((N, T), -1)
    yp = np.full((N, T), -1)
    d = np.zeros((N, T), int)
    for n in range(N):
        start = 0
        dd = 1
        for t in range(T):
            d[n, t] = dd
            dd += 1
            if sp['topcol'][n, t]:
                cs = sp['colstate'][n, t]
                tl, th = sp['tlo'][n, start], sp['thi'][n, start]
                if cs in (tl, th):
                    y[n, start:t + 1] = 0 if cs == tl else 1
                if a_tok is not None:
                    sh = sp['shat'][n, start:t + 1]
                    aa = a_tok[n, start:t + 1]
                    sc_l = (aa == DST[sh, tl]).sum()
                    sc_h = (aa == DST[sh, th]).sum()
                    yp[n, start:t + 1] = 0 if sc_l >= sc_h else 1
                start = t + 1
                dd = 1
    return y, yp, d


def ridge2(H, y, l2=10.0):
    Y = np.eye(2)[y]
    mh = H.mean(0)
    Hc = H - mh
    W = np.linalg.solve(Hc.T @ Hc + l2 * np.eye(H.shape[1]), Hc.T @ Y)
    return W, mh


def platt(scores, y):
    from scipy.optimize import minimize
    s = scores

    def nll(ab):
        p = 1 / (1 + np.exp(-(ab[0] * s + ab[1])))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    r = minimize(nll, np.array([1.0, 0.0]), method='Nelder-Mead')
    return r.x


def main():
    ckpt = sys.argv[1]
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    net = Net().to(DEV)
    net.load_state_dict(torch.load(ckpt, map_location=DEV))
    net.eval()
    out = {}

    # ---------------- self-play + reward thresholds ----------------
    sp = selfplay(net, 2000, seed=42, lam=lam)
    r, ht, hc = sp['R'].mean(), sp['ntop'].mean(), sp['ncoll'].mean()
    print(f'self-play: R={r:.3f}  Htop={ht:.2f}  H={hc:.2f}  '
          f'Htop/H={ht / max(hc, 1e-9):.2f}')
    print(f'  PREREG (e): R >= 0.85*fresh-floor; Htop/H >= 0.55 '
          f'(drift sits at 0.32)')
    out['reward'] = dict(R=float(r), Htop=float(ht), H=float(hc))

    # ---------------- M1: collapse vs observer ----------------
    ycol, ypur, d = segment_labels(sp, sp['tt'])
    with torch.no_grad():
        _, hs = net(sp['tt'], return_hidden=True)
    dec_pos = 3 + 3 * np.arange(T)
    p_obs_lo = np.take_along_axis(sp['gp'], sp['tlo'][..., None],
                                  2).squeeze(-1)
    p_obs_hi = np.take_along_axis(sp['gp'], sp['thi'][..., None],
                                  2).squeeze(-1)
    p_obs = p_obs_lo / (p_obs_lo + p_obs_hi + 1e-12)   # P(goal = lower)
    y = ypur                                # PRIMARY label: pursued goal
    yf, df = y.reshape(-1), d.reshape(-1)
    agree = float((ycol[ycol >= 0] == ypur[ycol >= 0]).mean())
    print(f'\nlabel check: collected-top vs pursued agree {agree:.3f}')
    obsf = p_obs.reshape(-1)
    ntr = 1400 * T
    valid = yf >= 0
    print('M1 collapse table (label = PURSUED goal; native probe, '
          'held-out episodes):')
    best = None
    for li in (4, 5, 6):
        H = hs[li][:, dec_pos].reshape(-1, 64).cpu().numpy()
        tr = valid.copy()
        tr[ntr:] = False
        te = valid.copy()
        te[:ntr] = False
        W, mh = ridge2(H[tr], yf[tr])
        sc_tr = ((H[tr] - mh) @ W)[:, 1] - ((H[tr] - mh) @ W)[:, 0]
        ab = platt(sc_tr, yf[tr])
        sc_te = ((H[te] - mh) @ W)[:, 1] - ((H[te] - mh) @ W)[:, 0]
        p_hi = 1 / (1 + np.exp(-(ab[0] * sc_te + ab[1])))
        acc = ((p_hi > .5).astype(int) == yf[te]).mean()
        if best is None or acc > best[0]:
            best = (acc, li, p_hi, te)
    acc, li, p_hi, te = best
    print(f'  best layer L{li} overall acc {acc:.3f}')
    maxp_net = np.maximum(p_hi, 1 - p_hi)
    corr_net = (p_hi > .5).astype(int) == yf[te]
    p_o = obsf[te]
    maxp_obs = np.maximum(p_o, 1 - p_o)
    corr_obs = (p_o <= .5).astype(int) == yf[te]
    dte = df[te]
    print(f"  {'d':>3} {'n':>6} | net acc / maxp | obs acc / maxp | lead")
    tab = {}
    for dd in range(1, 9):
        m = dte == dd
        if m.sum() < 50:
            continue
        na, nm = corr_net[m].mean(), maxp_net[m].mean()
        oa, om = corr_obs[m].mean(), maxp_obs[m].mean()
        print(f'  {dd:3d} {m.sum():6d} |  {na:.3f} / {nm:.3f} |'
              f'  {oa:.3f} / {om:.3f} | {na - oa:+.3f}')
        tab[dd] = dict(n=int(m.sum()), net_acc=float(na), net_maxp=float(nm),
                       obs_acc=float(oa), obs_maxp=float(om))
    out['M1'] = dict(layer=li, table=tab)
    e12 = [tab[k] for k in (1, 2) if k in tab]
    if e12:
        nm = np.mean([r['net_maxp'] for r in e12])
        om = np.mean([r['obs_maxp'] for r in e12])
        print(f'  PREREG (f): net maxp d<=2 = {nm:.3f} (>=0.85?) while '
              f'obs maxp = {om:.3f} (<=0.65?)')

    # ---------------- M2: preservation on persona streams ----------------
    ev = persona_gen(1500, np.random.default_rng(999))
    ptoks = torch.tensor(ev['toks'], device=DEV)
    with torch.no_grad():
        _, hsp = net(ptoks, return_hidden=True)
    G = ev['goal_post'].reshape(-1, S)
    ntr2 = 1000 * T
    r2best = -1
    for li2 in (4, 5, 6):
        Hp = hsp[li2][:, dec_pos].reshape(-1, 64).cpu().numpy()
        mh, mg = Hp[:ntr2].mean(0), G[:ntr2].mean(0)
        Vv = np.linalg.solve(
            (Hp[:ntr2] - mh).T @ (Hp[:ntr2] - mh) + 10 * np.eye(64),
            (Hp[:ntr2] - mh).T @ (G[:ntr2] - mg))
        P = (Hp[ntr2:] - mh) @ Vv + mg
        r2 = 1 - ((G[ntr2:] - P) ** 2).sum() / \
            ((G[ntr2:] - G[ntr2:].mean(0)) ** 2).sum()
        r2best = max(r2best, r2)
    print(f'\nM2 preservation: goal-posterior R2 on persona streams = '
          f'{r2best:.3f} (S1 value 0.722; threshold >= 0.578)')
    out['M2'] = float(r2best)

    # ---------------- (h) frozen-probe format transfer ----------------
    pre = Net().to(DEV)
    pre.load_state_dict(torch.load('orchard_runs/A/p1_final.pt',
                                   map_location=DEV))
    pre.eval()
    with torch.no_grad():
        _, hpre = pre(ptoks, return_hidden=True)
    Hq = hpre[6][:, dec_pos].reshape(-1, 64).cpu().numpy()
    mh, mg = Hq[:ntr2].mean(0), G[:ntr2].mean(0)
    Vfroz = np.linalg.solve(
        (Hq[:ntr2] - mh).T @ (Hq[:ntr2] - mh) + 10 * np.eye(64),
        (Hq[:ntr2] - mh).T @ (G[:ntr2] - mg))
    Hrl = hs[6][:, dec_pos].reshape(-1, 64).cpu().numpy()
    Pg = (Hrl[te] - mh) @ Vfroz + mg
    lo_m = np.take_along_axis(Pg, sp['tlo'].reshape(-1)[te][:, None],
                              1).squeeze(-1)
    hi_m = np.take_along_axis(Pg, sp['thi'].reshape(-1)[te][:, None],
                              1).squeeze(-1)
    acc_froz = ((hi_m > lo_m).astype(int) == yf[te]).mean()
    print(f'(h) frozen S1 probe on RL self-play: acc {acc_froz:.3f} vs '
          f'native {acc:.3f} -> ratio {acc_froz / acc:.2f} (>= 0.7?)')
    out['h_frozen'] = dict(acc=float(acc_froz), native=float(acc))
    tag = ckpt.replace('/', '_').replace('.pt', '')
    json.dump(out, open(f'collapse_{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
