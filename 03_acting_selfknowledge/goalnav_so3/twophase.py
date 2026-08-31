"""Two-phase interventional test of the trajectory-level auxiliary mechanism.

From a mid-training no-aux checkpoint: run N AUX-ONLY steps (no policy loss),
then N POLICY-ONLY steps; compare the policy phase against a policy-only
control at matched policy budget. If aux-only pretreatment (which contributes
zero first-order policy descent -- measured cos of mean gradients ~ 0)
accelerates the subsequent policy phase, the auxiliary's benefit is
second-order (landscape/representation), not gradient alignment.
Arms: policy-only | past-aux -> policy | shuffle-aux -> policy (control).
Also logs reach and policy-gradient SNR at phase boundaries.
"""
from __future__ import annotations
import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goalnav import GoalNavNet, rollout, sim_loss_fn  # noqa: E402

DEV = 'cuda'
CKPT = 'goalnav_runs/gn_rep_lam0.0_s1_steps/step_01000.pt'
N = 1000
B = 96


def load():
    c = torch.load(CKPT, map_location=DEV)
    a = c['args']
    net = GoalNavNet(a['d_model'], a['n_layer'], a['n_head'], a['L'],
                     a['r']).to(DEV)
    net.load_state_dict(c['state'])
    net.train()
    return net, a


@torch.no_grad()
def reach(net, a):
    net.eval()
    X, obs, g = rollout(net, 512, a['L'], DEV, np.random.default_rng(7),
                        a['delta'], a['cutoff'])
    dist = torch.arccos((X[:, 1:] * g[:, None]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
    net.train()
    return torch.rad2deg(dist[:, 3 * a['L'] // 4:].mean()).item()


def snr_p(net, a, nb=8):
    ps = [p for p in list(net.in_proj.parameters()) + list(net.pos.parameters())
          + list(net.blocks.parameters()) + list(net.lnf.parameters())]
    gs = []
    for i in range(nb):
        X, obs, g = rollout(net, B, a['L'], DEV, np.random.default_rng(9000 + i),
                            a['delta'], a['cutoff'])
        loss = (1 - (X[:, 1:] * g[:, None]).sum(-1)).mean()
        gr = torch.autograd.grad(loss, ps, allow_unused=True)
        gs.append(torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1)
                             for x, p in zip(gr, ps)]))
    G = torch.stack(gs)
    return (G.mean(0).norm()**2 / ((G - G.mean(0)).norm(dim=1)**2).mean()).item()


def aux_phase(net, a, mode, rng):
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.0)
    for step in range(1, N + 1):
        with torch.no_grad():
            X, obs, g = rollout(net, B, a['L'], DEV, rng, a['delta'], a['cutoff'])
        sloss = sim_loss_fn(net, obs, X, a['r'], mode=mode)
        opt.zero_grad(); sloss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            print(f'    aux[{mode}] step {step} sloss {float(sloss):.4f} '
                  f'reach {reach(net, a):.1f}', flush=True)


def policy_phase(net, a, rng):
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.0)
    marks = {}
    for step in range(1, N + 1):
        X, obs, g = rollout(net, B, a['L'], DEV, rng, a['delta'], a['cutoff'])
        ploss = (1 - (X[:, 1:] * g[:, None]).sum(-1)).mean()
        opt.zero_grad(); ploss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 200 == 0:
            marks[step] = reach(net, a)
            print(f'    policy step {step} reach {marks[step]:.1f}', flush=True)
    return marks


results = {}
for arm, mode in (('policy-only', None), ('past->policy', 'past'),
                  ('shuffle->policy', 'shuffle')):
    print(f'== ARM {arm} ==', flush=True)
    net, a = load()
    rng = np.random.default_rng(42)
    print(f'  start: reach {reach(net, a):.1f} | SNR_p {snr_p(net, a):.3f}', flush=True)
    if mode is not None:
        aux_phase(net, a, mode, rng)
        print(f'  after aux phase: reach {reach(net, a):.1f} | '
              f'SNR_p {snr_p(net, a):.3f}', flush=True)
    results[arm] = policy_phase(net, a, rng)

print('\n=== SUMMARY: reach at matched POLICY-step budget (start = noaux ckpt1000) ===')
print(f"{'arm':>16} | " + ' '.join(f'p{k:>4}' for k in (200, 400, 600, 800, 1000)))
for arm, m in results.items():
    print(f'{arm:>16} | ' + ' '.join(f'{m[k]:5.1f}' for k in (200, 400, 600, 800, 1000)))
