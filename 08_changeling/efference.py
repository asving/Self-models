"""The write mechanism, read through the register (v3 doc, Iteration 9).

Now that the store m-hat is known, redo the counterfactual-token increment
test on the REGISTER readout rho = h·m-hat (the shadow-readout version was
noise). Questions:
 (1) register content: decode lambda from rho alone (sign vs magnitude);
 (2) write rule: per-token increment profiles d-rho(u') vs three comparator
     hypotheses — EFFERENCE (net's own current intention log pi(u'|h) −
     log pbar(u')), TEMPLATE (log plan(u') − log pbar(u')), WORLD-SURPRISE
     (−log pbar(u')); discriminated on channels the net has WITHDRAWN from
     (pi ≈ pbar there: efference predicts flat, template predicts tilted);
 (3) comparator locus: recompute counterfactual steps with the token's
     contribution frozen (mean over u') in (a) the gate inputs i_r,i_z or
     (b) the candidate input i_n — which freeze kills the profile.
Writes results/rnn_efference.json, figs/efference.png. cwd = 08_changeling.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worlds import World
from rnn import TorchWorld, step_features, N
from eval_rnn import WORLD_KW, DEV
from probe import load, split
from probe3 import collect_full
from whitebox_lambda import prefix, Filt
from format import match_pairs
from distill import distill

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
RES = {}
T_STAR = 16
PAIR = (0, 2)


@torch.no_grad()
def gru_step_parts(model, x, h, freeze=None, ref_parts=None):
    """Manual step; freeze in {'gates','cand'} replaces the corresponding
    input contributions with ref_parts (mean over counterfactual tokens)."""
    e = torch.relu(model.inp(x))
    gi = e @ model.gru.weight_ih_l0.T + model.gru.bias_ih_l0
    i_r, i_z, i_n = gi.chunk(3, -1)
    if freeze == 'gates':
        i_r, i_z = ref_parts[0], ref_parts[1]
    if freeze == 'cand':
        i_n = ref_parts[2]
    gh = h[0] @ model.gru.weight_hh_l0.T + model.gru.bias_hh_l0
    h_r, h_z, h_n = gh.chunk(3, -1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return ((1 - z) * n + z * h[0]), (i_r, i_z, i_n)


def slope_r2(y, x):
    x, y = x.ravel(), y.ravel()
    sl = float((x * y).sum() / ((x * x).sum() + 1e-12))
    return sl, float(1 - ((y - sl * x) ** 2).sum()
                     / (((y - y.mean()) ** 2).sum() + 1e-12))


@torch.no_grad()
def main():
    rng = np.random.default_rng(23)
    post = load('post_6000')
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)

    # the register direction (sign-calibrated toward identity A)
    st = prefix(post, w, tw, seed=77)
    donor = match_pairs(st)
    m = distill(post, w, st, donor)
    mhat = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    sgn = np.where(st['iota'].cpu().numpy(), 1.0, -1.0)[:, None]
    _, _, Vt = np.linalg.svd(sgn * mhat, full_matrices=False)
    mg = Vt[0]
    rho = st['h'][0].cpu().numpy() @ mg
    if np.corrcoef(rho, st['iota'].cpu().numpy().astype(float))[0, 1] < 0:
        mg = -mg
    mg_t = torch.tensor(mg, dtype=torch.float32, device=DEV)

    # (1) register content: decode lambda from rho alone
    H, gt = collect_full(post, 1024, rng)
    tr, te = split(1024)
    rho_all = H.reshape(-1, 256) @ mg
    lam = np.clip(gt['lam_logodds'], -20, 20).reshape(-1)
    ep = np.repeat(np.arange(1024), 32)
    m_te = np.isin(ep, te)
    a, b = np.polyfit(rho_all[~m_te], lam[~m_te], 1)
    pred = a * rho_all[m_te] + b
    RES['reg_lambda_r2'] = round(float(
        1 - ((lam[m_te] - pred) ** 2).sum()
        / ((lam[m_te] - lam[m_te].mean()) ** 2).sum()), 3)
    late = (np.tile(np.arange(32), 1024) >= 8) & m_te & (np.abs(lam) > 1)
    RES['reg_sign_acc_late'] = round(float(
        (np.sign(rho_all[late] - np.median(rho_all[m_te]))
         == np.sign(lam[late])).mean()), 3)
    print('register content:', {k: RES[k] for k in
                                ('reg_lambda_r2', 'reg_sign_acc_late')},
          flush=True)

    # (2)+(3) counterfactual profiles at t*, both channels
    h = st['h']
    f = st['f']
    pbar_u, pbar_v, piA, piB = f.dists(T_STAR)
    lu = post.head_u(h[0]); lv = post.head_v(h[0])
    logpu = F.log_softmax(lu, -1).cpu().numpy()
    logpv = F.log_softmax(lv, -1).cpu().numpy()
    io = st['iota'].cpu().numpy()
    R = h.shape[1]
    goals = st['goals']
    for ch in ('u', 'v'):
        other_mode = np.argmax(pbar_v if ch == 'u' else pbar_u, 1)
        # gather counterfactual parts first (for freezes)
        parts = []
        hs_by_tok = {}
        for tok in range(N):
            tt = torch.full((R,), tok, dtype=torch.long, device=DEV)
            om = torch.tensor(other_mode, dtype=torch.long, device=DEV)
            x = (step_features(tt, om, goals, T_STAR + 1, w.T, DEV) if ch == 'u'
                 else step_features(om, tt, goals, T_STAR + 1, w.T, DEV))
            hn, p = gru_step_parts(post, x, h)
            hs_by_tok[tok] = hn
            parts.append(p)
        ref = tuple(torch.stack([p[i] for p in parts]).mean(0) for i in range(3))
        prof = {}
        for mode in ('full', 'gates', 'cand'):
            pr = np.zeros((R, N), np.float32)
            for tok in range(N):
                if mode == 'full':
                    hn = hs_by_tok[tok]
                else:
                    tt = torch.full((R,), tok, dtype=torch.long, device=DEV)
                    om = torch.tensor(other_mode, dtype=torch.long, device=DEV)
                    x = (step_features(tt, om, goals, T_STAR + 1, w.T, DEV)
                         if ch == 'u' else
                         step_features(om, tt, goals, T_STAR + 1, w.T, DEV))
                    hn, _ = gru_step_parts(post, x, h, freeze=mode,
                                           ref_parts=ref)
                pr[:, tok] = (hn @ mg_t).cpu().numpy()
            prof[mode] = pr - pr.mean(1, keepdims=True)
        # identity-signed profiles (increment toward TRUE identity)
        s = np.where(io if ch == 'u' else io, 1.0, -1.0)[:, None]
        sgn_ch = (np.where(io, 1.0, -1.0) if ch == 'u'
                  else np.where(io, -1.0, 1.0))[:, None] * 0 + 1  # keep raw A-frame
        # hypotheses in the A-frame (rho grows toward A)
        if ch == 'u':
            eff = logpu - np.log(pbar_u)
            tem = np.log(piA + 1e-12) - np.log(pbar_u)
            wsr = -np.log(pbar_u)
        else:
            eff = -(logpv - np.log(pbar_v))
            tem = -(np.log(piB + 1e-12) - np.log(pbar_v))
            wsr = -np.log(pbar_v)
        for nm, hyp in (('eff', eff), ('tem', tem), ('wsr', wsr)):
            hc = hyp - hyp.mean(1, keepdims=True)
            sl, r2 = slope_r2(prof['full'], hc)
            RES[f'{ch}_full_vs_{nm}'] = [round(sl, 4), round(r2, 3)]
        # the discriminating split: channel NOT mine (withdrawn head)
        notmine = ~io if ch == 'u' else io
        for nm, hyp in (('eff', eff), ('tem', tem)):
            hc = (hyp - hyp.mean(1, keepdims=True))[notmine]
            sl, r2 = slope_r2(prof['full'][notmine], hc)
            RES[f'{ch}_notmine_vs_{nm}'] = [round(sl, 4), round(r2, 3)]
        # locus: slope retention under freezes (vs efference hypothesis)
        hc = eff - eff.mean(1, keepdims=True)
        base_sl, _ = slope_r2(prof['full'], hc)
        for mode in ('gates', 'cand'):
            sl, _ = slope_r2(prof[mode], hc)
            RES[f'{ch}_slope_retention_{mode}'] = round(sl / (base_sl + 1e-12), 3)
        print(f'channel {ch}:', {k: v for k, v in RES.items()
                                 if k.startswith(ch + '_')}, flush=True)

    with open('results/rnn_efference.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    hc = (eff - eff.mean(1, keepdims=True))
    ii = np.random.default_rng(0).choice(hc.size, min(4000, hc.size), False)
    axes[0].scatter(hc.ravel()[ii], prof['full'].ravel()[ii], s=2, alpha=.15)
    sl, r2 = slope_r2(prof['full'], hc)
    xs = np.array([hc.min(), hc.max()])
    axes[0].plot(xs, sl * xs, 'r-', label=f'slope {sl:.3f}, R² {r2:.2f}')
    axes[0].set_xlabel('efference hypothesis (centered)')
    axes[0].set_ylabel('register increment Δρ')
    axes[0].set_title('v-channel write rule (register readout)')
    axes[0].legend(fontsize=7)
    labs = ['u eff', 'u tem', 'v eff', 'v tem']
    vals = [RES['u_notmine_vs_eff'][1], RES['u_notmine_vs_tem'][1],
            RES['v_notmine_vs_eff'][1], RES['v_notmine_vs_tem'][1]]
    axes[1].bar(labs, vals, color=['C0', 'C3', 'C0', 'C3'])
    axes[1].set_ylabel('R² (not-mine channels)')
    axes[1].set_title('comparator: own intention vs plan template')
    labs2 = ['u gates', 'u cand', 'v gates', 'v cand']
    vals2 = [RES['u_slope_retention_gates'], RES['u_slope_retention_cand'],
             RES['v_slope_retention_gates'], RES['v_slope_retention_cand']]
    axes[2].bar(labs2, vals2, color=['C4', 'C2'] * 2)
    axes[2].axhline(1.0, ls=':', c='k')
    axes[2].set_ylabel('write-slope retention under freeze')
    axes[2].set_title('comparator locus: gate inputs vs candidate input')
    fig.tight_layout(); fig.savefig('figs/efference.png', dpi=160)
    print('wrote results/rnn_efference.json, figs/efference.png', flush=True)


if __name__ == '__main__':
    main()
