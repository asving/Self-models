"""RIVAL RING — symmetric-choice ring game with referent-factored aux heads.

See RIVALRING.md for the preregistration. Streams: self (authored+consequential),
rival B (consequential only), weather W (neither), flourish F (authored only;
no policy gradient, masked from later attention). REINFORCE + baseline.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF

sys.path.insert(0, os.path.expanduser('~/comp_icl'))
from model import Block  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
S, K, T, WIN = 12, 4, 36, 9
GOALS = [0, 3, 6, 9]
EPS, WAKE_MAX = 0.3, 8
MESS_X, MESS_A = 0.05, 0.85
# vocab layout
BOS, MODE_ACT, MODE_OBS = 0, 1, 2
W0, B0, A0, M0, F0 = 3, 6, 18, 30, 33
NV = 38
SEQ = 2 + 5 * T
KIND_F = np.zeros(SEQ, bool)
POS_A = [2 + 5 * t + 2 for t in range(T)]      # A-token positions (move logits)
POS_M = [2 + 5 * t + 3 for t in range(T)]      # M-token positions (flourish + aux)
for t in range(T):
    KIND_F[2 + 5 * t + 4] = True               # F-token positions


def build_mask(dev):
    """causal + flourish keys invisible to all later queries."""
    i = torch.arange(SEQ, device=dev)
    causal = i[None, :] > i[:, None]                       # True = blocked
    fkey = torch.tensor(KIND_F, device=dev)[None, :] & (i[None, :] < i[:, None])
    return causal | fkey


class RingNet(nn.Module):
    def __init__(self, d=128, nl=6, nh=4):
        super().__init__()
        self.tok = nn.Embedding(NV, d)
        self.pos = nn.Embedding(SEQ, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, NV, bias=False)           # policy/LM head
        self.aux_head = nn.Linear(d, NV * 4, bias=False)   # aux, horizons 1..4

    def trunk(self, idx, mask):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x, mask[:idx.shape[1], :idx.shape[1]])
        return self.lnf(x)


def mess3_step(state, rng, B):
    sym = np.where(rng.random(B) < MESS_A, state,
                   (state + rng.integers(1, 3, B)) % 3)
    mv = rng.random(B)
    nxt = np.where(mv < MESS_X, (state + 1) % 3,
                   np.where(mv < 2 * MESS_X, (state + 2) % 3, state))
    return sym, nxt


def ring_dirs(a, b):
    """optimal step set toward b; list per element."""
    d = (b - a) % S
    out = np.zeros_like(a)
    tie = (d == S - d)
    out = np.where((d > 0) & (d < S - d), 1, np.where(d > S - d, -1, 0))
    return out, tie, d


@torch.no_grad()
def rollout(net, Bn, dev, rng, mode=MODE_ACT, greedy=False):
    mask = build_mask(dev)
    apos = rng.integers(S, size=Bn)
    bpos = rng.integers(S, size=Bn)
    gB = np.array(GOALS)[rng.integers(K, size=Bn)]
    wake = rng.integers(WAKE_MAX + 1, size=Bn)
    wstate = rng.integers(3, size=Bn)
    seq = torch.full((Bn, SEQ), BOS, dtype=torch.long, device=dev)
    seq[:, 1] = mode
    rec = dict(apos=[], bpos=[], w=[], mv=[], fl=[])
    for t in range(T):
        base = 2 + 5 * t
        wsym, wstate = mess3_step(wstate, rng, Bn)
        seq[:, base] = W0 + torch.tensor(wsym, device=dev)
        seq[:, base + 1] = B0 + torch.tensor(bpos, device=dev)
        seq[:, base + 2] = A0 + torch.tensor(apos, device=dev)
        h = net.trunk(seq[:, :base + 3], mask)
        mlog = net.head(h[:, -1])[:, M0:M0 + 3]
        mdist = torch.distributions.Categorical(logits=mlog)
        mv = mlog.argmax(-1) if greedy else mdist.sample()
        seq[:, base + 3] = M0 + mv
        h2 = net.trunk(seq[:, :base + 4], mask)
        flog = net.head(h2[:, -1])[:, F0:F0 + 5]
        fl = torch.distributions.Categorical(logits=flog).sample()
        seq[:, base + 4] = F0 + fl
        rec['w'].append(wsym); rec['bpos'].append(bpos.copy())
        rec['apos'].append(apos.copy())
        rec['mv'].append(mv.cpu().numpy()); rec['fl'].append(fl.cpu().numpy())
        # env transitions
        delta = mv.cpu().numpy() - 1                        # 0,1,2 -> -1,0,+1? no:
        delta = np.array([-1, 1, 0])[mv.cpu().numpy()]      # L, R, stay
        apos = (apos + delta) % S
        opt, tie, dd = ring_dirs(bpos, gB)
        step = np.where(tie, np.where(rng.random(Bn) < .5, 1, -1), opt)
        noise = rng.integers(-1, 2, Bn)
        use_noise = rng.random(Bn) < EPS
        bstep = np.where(t < wake, 0, np.where(use_noise, noise, step))
        bpos = (bpos + bstep) % S
    ap = np.stack(rec['apos'] + [apos], 0)                  # (T+1, B)
    occ = ((np.isin(ap[T - WIN + 1:], GOALS)) & (ap[T - WIN + 1:] != gB[None]))
    R = occ.sum(0).astype(np.float32)
    return seq, R, dict(gB=gB, wake=wake, **{k: np.stack(v) for k, v in rec.items()})


def scripted_episode(Bn, rng):
    """defer(P>.9) -> random unblocked demo, for OBS anchor batches (token seq only)."""
    apos = rng.integers(S, size=Bn); bpos = rng.integers(S, size=Bn)
    gB = np.array(GOALS)[rng.integers(K, size=Bn)]
    wake = rng.integers(WAKE_MAX + 1, size=Bn)
    wstate = rng.integers(3, size=Bn)
    post = np.ones((Bn, K)) / K
    com = np.full(Bn, -1)
    seq = np.full((Bn, SEQ), BOS, dtype=np.int64)
    seq[:, 1] = MODE_OBS
    for t in range(T):
        base = 2 + 5 * t
        wsym, wstate = mess3_step(wstate, rng, Bn)
        seq[:, base] = W0 + wsym
        seq[:, base + 1] = B0 + bpos
        seq[:, base + 2] = A0 + apos
        # commit rule
        sharp = post.max(1) > 0.9
        for i in np.where(sharp & (com < 0))[0]:
            cand = [g for g in GOALS if g != GOALS[post[i].argmax()]]
            com[i] = cand[rng.integers(3)]
        mv = np.zeros(Bn, int) + 2
        for i in range(Bn):
            if com[i] >= 0 and apos[i] != com[i]:
                d = (com[i] - apos[i]) % S
                mv[i] = 1 if (0 < d <= S - d) else 0
        seq[:, base + 3] = M0 + mv
        seq[:, base + 4] = F0 + rng.integers(5, size=Bn)
        delta = np.array([-1, 1, 0])[mv]
        apos = (apos + delta) % S
        # B step + exact-ish posterior update (move likelihood, marginal over wake simplified)
        opt, tie, dd = ring_dirs(bpos, gB)
        step = np.where(tie, np.where(rng.random(Bn) < .5, 1, -1), opt)
        noise = rng.integers(-1, 2, Bn)
        bstep = np.where(t < wake, 0,
                         np.where(rng.random(Bn) < EPS, noise, step))
        for i in range(Bn):
            if t >= wake[i]:
                for gi, g in enumerate(GOALS):
                    o, ti, _ = ring_dirs(bpos[i:i+1], np.array([g]))
                    p = EPS / 3
                    if ti[0]:
                        p += (1 - EPS) / 2 * (abs(bstep[i]) == 1)
                    elif bstep[i] == o[0]:
                        p += (1 - EPS)
                    post[i, gi] *= max(p, 1e-9)
                post[i] /= post[i].sum()
        bpos = (bpos + bstep) % S
    return torch.tensor(seq)


AUX_ARMS = ('none', 'self_fut', 'self_past', 'b_fut', 'b_past',
            'w_fut', 'w_past', 'fl_fut', 'shuffle')


def aux_targets(seq, arm):
    """target token at horizon k (1..4) for each step-block, read at M positions."""
    Bn = seq.shape[0]
    tg = torch.full((Bn, T, 4), -100, dtype=torch.long, device=seq.device)
    src = {'self': 2, 'b': 1, 'w': 0}.get(arm.split('_')[0], None)
    for t in range(T):
        for k in range(1, 5):
            if arm == 'fl_fut':
                tt = t + k
                if tt < T: tg[:, t, k - 1] = seq[:, 2 + 5 * tt + 4]
            elif arm.endswith('fut') or arm == 'shuffle':
                tt = t + k
                if tt < T: tg[:, t, k - 1] = seq[:, 2 + 5 * tt + src if arm != 'shuffle' else 2 + 5 * tt + 2]
            else:
                tt = t - k
                if tt >= 0: tg[:, t, k - 1] = seq[:, 2 + 5 * tt + src]
    if arm == 'shuffle':
        tg = torch.roll(tg, 1, dims=0)
    return tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--aux', default='none', choices=AUX_ARMS)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--eval_every', type=int, default=400)
    ap.add_argument('--ckpt_every', type=int, default=500)
    args = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = RingNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    mask = build_mask(dev)
    posA = torch.tensor(POS_A, device=dev)
    posM = torch.tensor(POS_M, device=dev)
    sdir = os.path.join(BASE, args.out + '_steps'); os.makedirs(sdir, exist_ok=True)
    log = []
    print(f'rivalring aux={args.aux} lam={args.lam} seed={args.seed} '
          f'params={sum(p.numel() for p in net.parameters())/1e3:.0f}K', flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train()
        if step % 10 == 0:                                  # OBS anchor batch
            # CE on OBSERVATION tokens only (W,B,A): keeps OBS-mode world
            # conditionals alive for the gate WITHOUT cloning the demo policy
            # (moves and flourishes are excluded from the anchor loss).
            seq = scripted_episode(args.batch, rng).to(dev)
            h = net.trunk(seq, mask)
            logits = net.head(h[:, :-1])
            tgt = seq[:, 1:].clone()
            for t in range(T):
                tgt[:, 2 + 5 * t + 2] = -100                # M token position-1
                tgt[:, 2 + 5 * t + 3] = -100                # F token position-1
            ce = TF.cross_entropy(logits.reshape(-1, NV), tgt.reshape(-1),
                                  ignore_index=-100)
            loss = 0.1 * ce
        else:
            seq, R, info = rollout(net, args.batch, dev, rng)
            Rt = torch.tensor(R, device=dev)
            adv = Rt - Rt.mean()
            h = net.trunk(seq, mask)
            mlog = net.head(h[:, posA])[:, :, M0:M0 + 3]
            mlp = TF.log_softmax(mlog, -1)
            mv = (seq[:, posM] - M0)
            lp = mlp.gather(-1, mv.unsqueeze(-1)).squeeze(-1).sum(1)
            ent = torch.distributions.Categorical(logits=mlog).entropy().mean()
            loss = -(adv * lp).mean() - args.ent * ent
            if args.aux != 'none' and args.lam > 0:
                tg = aux_targets(seq, args.aux)
                alog = net.aux_head(h[:, posM]).view(args.batch, T, 4, NV)
                aloss = TF.cross_entropy(alog.reshape(-1, NV), tg.reshape(-1),
                                         ignore_index=-100)
                loss = loss + args.lam * aloss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % args.eval_every == 0 or step == 1:
            net.eval()
            _, Re, _ = rollout(net, 256, dev, np.random.default_rng(7))
            log.append(dict(step=step, R=float(Re.mean())))
            print(f'step {step:6d} | R {Re.mean():5.2f} (max 9) | '
                  f'{time.time()-t0:5.0f}s', flush=True)
        if step % args.ckpt_every == 0:
            torch.save(dict(state=net.state_dict(), args=vars(args), step=step),
                       f'{sdir}/step_{step:05d}.pt')
    torch.save(dict(state=net.state_dict(), args=vars(args)),
               os.path.join(BASE, args.out + '.pt'))
    json.dump(dict(args=vars(args), log=log),
              open(os.path.join(BASE, args.out + '.json'), 'w'), indent=2)
    print(f'done {time.time()-t0:.0f}s reward {log[-1]["R"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
