"""Three-stage changeling RNN training (DESIGN_changeling_v1_rnn.md).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python train_rnn.py [--smoke] [--seed 0]
Writes ckpt/{pre,mid}_final.pt, ckpt/post_<step>.pt, results/rnn_floors.json,
results/rnn_train_log.json. Run with cwd = 08_changeling.
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from worlds import World
from oracle import run_episodes, run_base
from rnn import ChangelingGRU, TorchWorld, features, step_features, GOAL_PAIRS, N

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
WORLD_KW = dict(q0=0.9, c_other=0.6, c_self=0.35, d_goal=2,
                kappa=1.0, mode='running', rho=8.0)


def make_worlds():
    return {p: World(goal_pair=p, **WORLD_KW) for p in GOAL_PAIRS}


def floors(worlds, out_path, R=10000):
    w = worlds[(0, 2)]
    out = {}
    for agent in ('informed', 'live', 'agnostic'):
        out[f'occ_{agent}'] = float(run_episodes(w, agent, R, 900)['occ'].mean())
    out['occ_informed_rot35'] = float(
        run_episodes(worlds[(3, 5)], 'informed', 4000, 901)['occ'].mean())
    w0 = World(goal_pair=(0, 2), **{**WORLD_KW, 'rho': 0.0})
    z = run_episodes(w0, 'zero', 8192, 902, collect=True)
    out['pretrain_ce_floor'] = float(-z['traj']['record_ll'].mean() / (2 * w.T))
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print('floors:', {k: round(v, 4) for k, v in out.items()})
    return out


def pretrain_batch(w, R, seed):
    b = run_base(w, R, seed, collect=True)
    X = torch.tensor(features(b['u'], b['v']), device=DEV)
    U = torch.tensor(b['u'], dtype=torch.long, device=DEV)
    V = torch.tensor(b['v'], dtype=torch.long, device=DEV)
    return X, U, V


def ce_loss(model, X, U, V, T):
    lu, lv, _ = model(X)
    cu = F.cross_entropy(lu[:, :T].reshape(-1, N), U.reshape(-1))
    cv = F.cross_entropy(lv[:, :T].reshape(-1, N), V.reshape(-1))
    return 0.5 * (cu + cv)


def run_pretrain(model, worlds, cfg, log):
    w = worlds[(0, 2)]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['pre_lr'], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg['pre_steps'], eta_min=cfg['pre_lr'] / 10)
    for step in range(cfg['pre_steps']):
        X, U, V = pretrain_batch(w, cfg['batch'], 10_000_000 + step)
        loss = ce_loss(model, X, U, V, w.T)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == cfg['pre_steps'] - 1:
            log['pre'].append({'step': step, 'ce': float(loss.item())})
            print(f"pre {step}: ce/token {loss.item():.4f}", flush=True)


@torch.no_grad()
def record_rollout_flagged(model, tw, pair, R):
    """Net closed-loop rollout WITH truthful identity flag; returns the record
    (tokens + iota) for DAgger relabeling by the exact oracle."""
    T = tw.T
    goals = torch.tensor(pair, device=DEV).repeat(R, 1)
    iota = torch.rand(R, device=DEV) < 0.5
    sA = torch.randint(0, N, (R,), device=DEV)
    sB = torch.randint(0, N, (R,), device=DEV)
    h = None; u = v = None
    U = np.zeros((R, T), dtype=np.int64); Vv = np.zeros((R, T), dtype=np.int64)
    for t in range(T):
        x = step_features(u, v, goals, t, T, DEV, iota=iota)
        lu, lv, h = model.step(x, h)
        u_net = torch.multinomial(F.softmax(lu, -1), 1).squeeze(1)
        v_net = torch.multinomial(F.softmax(lv, -1), 1).squeeze(1)
        u_env, v_env = tw.emit(sA, sB)
        u = torch.where(iota, u_net, u_env)
        v = torch.where(iota, v_env, v_net)
        sA, sB = tw.trans(sA, sB, u, v)
        U[:, t] = u.cpu().numpy(); Vv[:, t] = v.cpu().numpy()
    return U, Vv, iota.cpu().numpy()


def distill_step(model, w, pair, U, V, iota, tu, tv):
    goals = np.tile(np.array(pair), (U.shape[0], 1))
    X = torch.tensor(features(U, V, goals, iota), device=DEV)
    lu, lv, _ = model(X)
    return 0.5 * (-(tu * F.log_softmax(lu[:, :w.T], -1)).sum(-1).mean()
                  - (tv * F.log_softmax(lv[:, :w.T], -1)).sum(-1).mean())


def distill_batch(model, worlds, tworlds, cfg, rng, seed, dagger):
    pair = GOAL_PAIRS[rng.integers(len(GOAL_PAIRS))]
    w = worlds[pair]
    if dagger:
        from oracle import replay_dists
        U, V, iota = record_rollout_flagged(model, tworlds[pair], pair,
                                            cfg['batch'])
        rep = replay_dists(w, U, V)
        io = iota[:, None, None]
        tu = torch.tensor(np.where(io, rep['piA'], rep['pbar_u']), device=DEV)
        tv = torch.tensor(np.where(io, rep['pbar_v'], rep['piB']), device=DEV)
    else:
        r = run_episodes(w, 'informed', cfg['batch'], seed, collect=True)
        tr = r['traj']
        U, V, iota = tr['u'].astype(np.int64), tr['v'].astype(np.int64), r['iota']
        tu = torch.tensor(tr['dist_u'], device=DEV)
        tv = torch.tensor(tr['dist_v'], device=DEV)
    return distill_step(model, w, pair, U, V, iota, tu, tv)


def run_midtrain(model, worlds, tworlds, cfg, log, rng):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['mid_lr'], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg['mid_steps'], eta_min=cfg['mid_lr'] / 10)
    for step in range(cfg['mid_steps']):
        x = rng.random()
        if x < cfg['mid_premix']:
            X, U, V = pretrain_batch(worlds[(0, 2)], cfg['batch'], 20_000_000 + step)
            loss = ce_loss(model, X, U, V, worlds[(0, 2)].T)
        else:
            # DAgger relabeling (net rollout, oracle targets) fixes the
            # compounding error seen in v1.0 (flag-given occ .367 vs floor .508)
            loss = distill_batch(model, worlds, tworlds, cfg, rng,
                                 30_000_000 + step, dagger=(x > 0.7))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == cfg['mid_steps'] - 1:
            log['mid'].append({'step': step, 'loss': float(loss.item())})
            print(f"mid {step}: soft-ce {loss.item():.4f}", flush=True)


def rollout(model, frozen, tworlds, cfg, rng, n_eps, train=True, flag_truth=False):
    """One on-policy pass, single goal pair. Returns loss parts + stats."""
    pair = GOAL_PAIRS[rng.integers(len(GOAL_PAIRS))]
    tw = tworlds[pair]
    R, T = n_eps, tw.T
    goals = torch.tensor(pair, device=DEV).repeat(R, 1)
    iota = torch.rand(R, device=DEV) < 0.5
    sA = torch.randint(0, N, (R,), device=DEV)
    sB = torch.randint(0, N, (R,), device=DEV)
    h = hf = None
    u = v = None
    LP, KL, CE, RW, DLAM = [], [], [], [], []
    for t in range(T):
        x = step_features(u, v, goals, t, T, DEV,
                          iota=(iota if flag_truth else None))
        lu, lv, h = model.step(x, h)
        with torch.no_grad():
            xf = step_features(u, v, goals, t, T, DEV, with_goals=False)
            flu, flv, hf = frozen.step(xf, hf)
        logpu, logpv = F.log_softmax(lu, -1), F.log_softmax(lv, -1)
        flogu, flogv = F.log_softmax(flu, -1), F.log_softmax(flv, -1)
        with torch.no_grad():
            u_net = torch.multinomial(logpu.exp(), 1).squeeze(1)
            v_net = torch.multinomial(logpv.exp(), 1).squeeze(1)
            u_env, v_env = tw.emit(sA, sB)
            u = torch.where(iota, u_net, u_env)
            v = torch.where(iota, v_env, v_net)
        gu = logpu.gather(1, u[:, None]).squeeze(1)
        gv = logpv.gather(1, v[:, None]).squeeze(1)
        fgu = flogu.gather(1, u[:, None]).squeeze(1)
        fgv = flogv.gather(1, v[:, None]).squeeze(1)
        LP.append(torch.where(iota, gu, gv))
        klu = (logpu.exp() * (logpu - flogu)).sum(-1)
        klv = (logpv.exp() * (logpv - flogv)).sum(-1)
        KL.append(torch.where(iota, klu, klv))
        CE.append(torch.where(iota, -gv, -gu))
        toward_A = (gu - fgu) + (fgv - gv)
        DLAM.append(torch.where(iota, toward_A, -toward_A).detach())
        with torch.no_grad():
            sA, sB = tw.trans(sA, sB, u, v)
            RW.append(tw.ball(sA, sB).float())
    LP, KL, CE = torch.stack(LP, 1), torch.stack(KL, 1), torch.stack(CE, 1)
    RW, DLAM = torch.stack(RW, 1), torch.stack(DLAM, 1)
    G = torch.flip(torch.cumsum(torch.flip(RW, [1]), 1), [1])
    # natural reward units (v1.0's std-normalization shrank the effective
    # rho from 8 to ~3 and let the anchor erode the tilt)
    A = G - G.mean(0, keepdim=True)
    parts = {'pg': -(A.detach() * LP).mean(), 'kl': KL.mean(), 'ce': CE.mean()}
    lam = DLAM.sum(1)
    # 'ident' = record-identifiability of the emitter (a Bayes-observer
    # quantity), NOT the net's knowledge — that is measured by the occupancy
    # premium and by output legibility in eval_rnn.py
    stats = {'occ': float(RW.mean().item()),
             'ce_other': float(CE.mean().item()),
             'kl_self': float(KL.mean().item()),
             'ident_correct': float((lam > 0).float().mean().item()),
             'ident_median': float(lam.median().item())}
    return parts, stats


def evaluate(model, frozen, tworlds, cfg, seed, flag_truth=False):
    rng = np.random.default_rng(seed)
    agg = []
    model.eval()
    with torch.no_grad():
        for _ in range(max(cfg['eval_eps'] // cfg['batch'], 1)):
            _, s = rollout(model, frozen, tworlds, cfg, rng, cfg['batch'],
                           train=False, flag_truth=flag_truth)
            agg.append(s)
    model.train()
    return {k: float(np.mean([a[k] for a in agg])) for k in agg[0]}


def run_posttrain(model, frozen, worlds, tworlds, cfg, log, rng):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['post_lr'], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg['post_steps'], eta_min=cfg['post_lr'] / 3)
    for step in range(cfg['post_steps'] + 1):
        if step in cfg['ckpt_post']:
            torch.save(model.state_dict(), f"ckpt/post_{step}.pt")
        if step % cfg['eval_every'] == 0 or step == cfg['post_steps']:
            ev = evaluate(model, frozen, tworlds, cfg, 777 + step)
            ev['step'] = step
            log['post_eval'].append(ev)
            print(f"post {step}: occ {ev['occ']:.4f} ce_other {ev['ce_other']:.4f} "
                  f"kl {ev['kl_self']:.4f} ident_correct {ev['ident_correct']:.3f} "
                  f"ident_med {ev['ident_median']:.2f}", flush=True)
        if step == cfg['post_steps']:
            break
        if rng.random() < cfg['post_rehearsal']:
            # flag-given distillation rehearsal: keeps the planning machinery
            # constant so occupancy gains attribute to identification
            loss = distill_batch(model, worlds, tworlds, cfg, rng,
                                 40_000_000 + step, dagger=False)
        else:
            parts, _ = rollout(model, frozen, tworlds, cfg, rng, cfg['batch'])
            loss = parts['pg'] + (1.0 / cfg['rho']) * parts['kl'] + parts['ce']
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    cfg = dict(d=256, batch=256, pre_steps=4000, pre_lr=3e-4,
               mid_steps=8000, mid_lr=1e-4, mid_premix=0.2,
               post_steps=6000, post_lr=3e-5, rho=8.0, post_rehearsal=0.1,
               eval_every=500, eval_eps=2048,
               ckpt_post=(0, 100, 200, 500, 1000, 2000, 4000, 6000))
    if args.smoke:
        cfg.update(pre_steps=40, mid_steps=40, post_steps=60, batch=64,
                   eval_every=30, eval_eps=128, ckpt_post=(0, 60))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    worlds = make_worlds()
    fl = floors(worlds, 'results/rnn_floors.json', R=1000 if args.smoke else 10000)
    tworlds = {p: TorchWorld(w, DEV) for p, w in worlds.items()}
    model = ChangelingGRU(cfg['d']).to(DEV)
    log = {'cfg': cfg, 'seed': args.seed, 'floors': fl,
           'pre': [], 'mid': [], 'post_eval': [], 'mid_eval': {}}

    run_pretrain(model, worlds, cfg, log)
    torch.save(model.state_dict(), 'ckpt/pre_final.pt')
    frozen = ChangelingGRU(cfg['d']).to(DEV)
    frozen.load_state_dict(torch.load('ckpt/pre_final.pt'))
    frozen.eval()
    [p.requires_grad_(False) for p in frozen.parameters()]

    run_midtrain(model, worlds, tworlds, cfg, log, rng)
    torch.save(model.state_dict(), 'ckpt/mid_final.pt')
    log['mid_eval']['flag_truth'] = evaluate(model, frozen, tworlds, cfg, 555,
                                             flag_truth=True)
    log['mid_eval']['flag_unknown'] = evaluate(model, frozen, tworlds, cfg, 556)
    print('mid closed-loop (flag=truth):', log['mid_eval']['flag_truth'], flush=True)
    print('mid closed-loop (flag=unknown):', log['mid_eval']['flag_unknown'], flush=True)

    run_posttrain(model, frozen, worlds, tworlds, cfg, log, rng)
    log['elapsed_s'] = round(time.time() - t0, 1)
    with open('results/rnn_train_log.json', 'w') as f:
        json.dump(log, f, indent=1, default=float)
    print(f"done in {log['elapsed_s']}s", flush=True)


if __name__ == '__main__':
    main()
