"""Analysis for the minimal on-policy experiment (DESIGN_onpolicy_v1).
Per run {gru,tf} x {probe,coupled}: identity accuracy by round; external
Bayes-court record identifiability (net head vs exact pbar); deviation
budget KL(pi_self || exact pbar) + entropy gap per round; CE_other excess
vs the 1.0458 floor; watermark structure (per-token mean deviation,
context dependence via belief-conditioned profile). cwd = 08_changeling.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from worlds import World
from rnn import TorchWorld, step_features, N
from eval_rnn import WORLD_KW, DEV
from whitebox_lambda import Filt
from post_predict import Wrapped, PAIR

torch.set_num_threads(8)
RES = {}
R = 2048
BATCH = 256
FLOOR = 1.0458


@torch.no_grad()
def eval_run(tag):
    arch, variant = tag.split('_')
    model = Wrapped(arch).to(DEV)
    model.load_state_dict(torch.load(f'ckpt/onpolicy_{tag}_final.pt'))
    model.eval()
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    acc = np.zeros((0, w.T))
    llr = np.zeros((0, w.T))
    kls, ents, ceo = [], [], []
    dev_profile = np.zeros(N)
    for b in range(R // BATCH):
        torch.manual_seed(20_000 + b)
        iota = torch.rand(BATCH, device=DEV) < 0.5
        io = iota.cpu().numpy()
        sA = torch.randint(0, N, (BATCH,), device=DEV)
        sB = torch.randint(0, N, (BATCH,), device=DEV)
        goals = torch.zeros(BATCH, 2, dtype=torch.long, device=DEV)
        f = Filt(w, BATCH)
        st = None; u = v = None
        a_ep = np.zeros((BATCH, w.T)); l_ep = np.zeros((BATCH, w.T))
        lo = np.zeros(BATCH)
        for t in range(w.T):
            x = step_features(u, v, goals, t, w.T, DEV, with_goals=False)
            lu, lv, hid, st = model.step(x, st)
            pu = F.softmax(lu, -1).cpu().numpy()
            pv = F.softmax(lv, -1).cpu().numpy()
            pbar_u, pbar_v, _, _ = f.dists(t)
            # deviation of the SELF head from the exact predictive law
            ps = np.where(io[:, None], pu, pv)
            pb = np.where(io[:, None], pbar_u, pbar_v)
            kls.append((ps * (np.log(ps + 1e-12) - np.log(pb + 1e-12))).sum(-1).mean())
            ents.append((-(ps * np.log(ps + 1e-12)).sum(-1)
                         + (pb * np.log(pb + 1e-12)).sum(-1)).mean())
            dev_profile += (ps - pb).mean(0)
            a_ep[:, t] = (model.head_id(hid).argmax(-1)
                          == iota.long()).float().cpu().numpy()
            u_net = torch.multinomial(torch.tensor(pu, device=DEV), 1).squeeze(1)
            v_net = torch.multinomial(torch.tensor(pv, device=DEV), 1).squeeze(1)
            ue, ve = tw.emit(sA, sB)
            u = torch.where(iota, u_net, ue)
            v = torch.where(iota, ve, v_net)
            un, vn = u.cpu().numpy(), v.cpu().numpy()
            ri = np.arange(BATCH)
            ceo.append(np.where(io, -np.log(pv[ri, vn] + 1e-12),
                                -np.log(pu[ri, un] + 1e-12)).mean())
            # external Bayes court: P(tok|mine)=net head, P(tok|genuine)=pbar
            dl = (np.log(pu[ri, un] + 1e-12) - np.log(pbar_u[ri, un] + 1e-12)
                  + np.log(pbar_v[ri, vn] + 1e-12) - np.log(pv[ri, vn] + 1e-12))
            lo = np.clip(lo + dl, -30, 30)
            l_ep[:, t] = np.where(io, lo, -lo)
            f.update(un, vn)
            sA, sB = tw.trans(sA, sB, u, v)
        acc = np.concatenate([acc, a_ep])
        llr = np.concatenate([llr, l_ep])
    out = {
        'id_acc_t0_7': round(float(acc[:, :8].mean()), 3),
        'id_acc_t8_15': round(float(acc[:, 8:16].mean()), 3),
        'id_acc_t16_31': round(float(acc[:, 16:].mean()), 3),
        'id_acc_final_round': round(float(acc[:, -1].mean()), 3),
        'court_correct_final': round(float((llr[:, -1] > 0).mean()), 3),
        'court_median_absLLR_final': round(float(
            np.median(np.abs(llr[:, -1]))), 2),
        'KL_self_vs_exact_pbar': round(float(np.mean(kls)), 4),
        'entropy_gap_self': round(float(np.mean(ents)), 4),
        'ce_other_excess_vs_floor': round(float(np.mean(ceo) - FLOOR), 4),
        'dev_profile_per_token': [round(float(x) / (R // BATCH) / 32, 4)
                                  for x in dev_profile]}
    return out


def main():
    for tag in ('gru_probe', 'gru_coupled', 'tf_probe', 'tf_coupled'):
        RES[tag] = eval_run(tag)
        print(tag, json.dumps(RES[tag]), flush=True)
    with open('results/onpolicy_analysis.json', 'w') as fj:
        json.dump(RES, fj, indent=1, default=float)
    print('wrote results/onpolicy_analysis.json', flush=True)


if __name__ == '__main__':
    main()
