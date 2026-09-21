"""On-policy detection, minimal version (DESIGN_onpolicy_v1_predictid.md).

Embodied prediction post-training: hidden coin iota; the net predicts both
channels; the embodied channel's token is sampled from its own head; the
other channel runs the genuine HMM. Losses: CE on both channels vs
realized + (1/8) KL(self head || frozen pretrained head) + lambda_id * CE
of a linear identity head on the hidden state.

Usage: CUDA_VISIBLE_DEVICES=<g> python post_predict.py --arch {gru,tf}
       --variant {probe,coupled} [--steps 6000] [--smoke]
cwd = 08_changeling. No goals, no reward, no midtraining.
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from worlds import World
from rnn import ChangelingGRU, TorchWorld, step_features, N
from eval_rnn import WORLD_KW, DEV

torch.set_num_threads(8)
PAIR = (0, 2)   # goals unused (no reward); kernels only


class Wrapped(nn.Module):
    """Uniform stepping + identity head over both architectures."""

    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        if arch == 'gru':
            self.body = ChangelingGRU()
            d = 256
        else:
            from tfmodel import ChangelingTF
            self.body = ChangelingTF()
            d = self.body.d
        self.head_id = nn.Linear(d, 2)

    def load_pre(self):
        pfx = '' if self.arch == 'gru' else 'tf_'
        self.body.load_state_dict(torch.load(f'ckpt/{pfx}pre_final.pt'))

    def step(self, x, state):
        """-> (logits_u, logits_v, hidden (R,d), new_state)"""
        if self.arch == 'gru':
            lu, lv, h = self.body.step(x, state)
            return lu, lv, h[0], h
        buf = x.unsqueeze(1) if state is None else torch.cat(
            [state, x.unsqueeze(1)], 1)
        lu, lv, hs = self.body(buf)
        return lu[:, -1], lv[:, -1], hs[:, -1], buf


def run_batch(model, frozen, tw, R, seed, variant, train=True):
    torch.manual_seed(seed)
    iota = torch.rand(R, device=DEV) < 0.5
    sA = torch.randint(0, N, (R,), device=DEV)
    sB = torch.randint(0, N, (R,), device=DEV)
    goals = torch.zeros(R, 2, dtype=torch.long, device=DEV)  # unused (zeroed)
    st = stf = None
    u = v = None
    T = tw.T
    parts = {'ce_other': 0., 'ce_self': 0., 'kl_anchor': 0., 'ce_id': 0.}
    accs = []
    for t in range(T):
        x = step_features(u, v, goals, t, T, DEV, with_goals=False)
        lu, lv, hid, st = model.step(x, st)
        with torch.no_grad():
            flu, flv, _, stf = frozen.step(x, stf)
        logpu, logpv = F.log_softmax(lu, -1), F.log_softmax(lv, -1)
        with torch.no_grad():
            u_net = torch.multinomial(logpu.exp(), 1).squeeze(1)
            v_net = torch.multinomial(logpv.exp(), 1).squeeze(1)
            u_env, v_env = tw.emit(sA, sB)
            u = torch.where(iota, u_net, u_env)
            v = torch.where(iota, v_env, v_net)
        gu = logpu.gather(1, u[:, None]).squeeze(1)
        gv = logpv.gather(1, v[:, None]).squeeze(1)
        parts['ce_self'] = parts['ce_self'] + torch.where(iota, -gu, -gv).mean() / T
        parts['ce_other'] = parts['ce_other'] + torch.where(iota, -gv, -gu).mean() / T
        flogu, flogv = F.log_softmax(flu, -1), F.log_softmax(flv, -1)
        klu = (logpu.exp() * (logpu - flogu)).sum(-1)
        klv = (logpv.exp() * (logpv - flogv)).sum(-1)
        parts['kl_anchor'] = parts['kl_anchor'] + torch.where(iota, klu, klv).mean() / T
        hin = hid if variant == 'coupled' else hid.detach()
        lid = model.head_id(hin)
        parts['ce_id'] = parts['ce_id'] + F.cross_entropy(
            lid, iota.long()) / T
        accs.append((lid.argmax(-1) == iota.long()).float().mean().item())
        with torch.no_grad():
            sA, sB = tw.trans(sA, sB, u, v)
    loss = (parts['ce_other'] + parts['ce_self']
            + parts['kl_anchor'] / 8.0 + parts['ce_id'])
    stats = {k: float(val.item()) for k, val in parts.items()}
    stats['acc_early'] = float(np.mean(accs[:8]))
    stats['acc_late'] = float(np.mean(accs[-8:]))
    return loss, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=('gru', 'tf'), required=True)
    ap.add_argument('--variant', choices=('probe', 'coupled'), required=True)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    steps, batch = (60, 64) if args.smoke else (args.steps, 256)
    tag = f'{args.arch}_{args.variant}'

    torch.manual_seed(args.seed)
    w = World(goal_pair=PAIR, **WORLD_KW)
    tw = TorchWorld(w, DEV)
    model = Wrapped(args.arch).to(DEV)
    model.load_pre()
    frozen = Wrapped(args.arch).to(DEV)
    frozen.load_pre()
    frozen.eval()
    [p.requires_grad_(False) for p in frozen.parameters()]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    log = {'arch': args.arch, 'variant': args.variant, 'steps': steps,
           'seed': args.seed, 'trace': []}
    t0 = time.time()
    for step in range(steps):
        loss, stats = run_batch(model, frozen, tw, batch, 10_000 + step,
                                args.variant)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == steps - 1:
            stats['step'] = step
            log['trace'].append(stats)
            print(f"{tag} {step}: ce_oth {stats['ce_other']:.4f} "
                  f"ce_self {stats['ce_self']:.4f} kl {stats['kl_anchor']:.4f} "
                  f"ce_id {stats['ce_id']:.4f} acc_early {stats['acc_early']:.3f} "
                  f"acc_late {stats['acc_late']:.3f}", flush=True)
        if step == steps // 2:
            torch.save(model.state_dict(), f'ckpt/onpolicy_{tag}_mid.pt')
    torch.save(model.state_dict(), f'ckpt/onpolicy_{tag}_final.pt')
    log['elapsed_s'] = round(time.time() - t0, 1)
    with open(f'results/onpolicy_train_{tag}.json', 'w') as f:
        json.dump(log, f, indent=1, default=float)
    print(f'{tag} done in {log["elapsed_s"]}s', flush=True)


if __name__ == '__main__':
    main()
