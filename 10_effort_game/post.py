"""Post-training in the effort game. Loss = -(score-function reward term) + LAM * prediction CE.
Score function uses the realized token's exact law log[(1-c w)P(x|s) + c w p(x)], differentiable in the net's w and p."""
import torch, torch.nn.functional as F, numpy as np, time, sys, json
from world import T3, EM, step, Oracle
from net import Net
dev = 'mps'; T = 32
C0 = 0.8; LAM = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; GAMMA = 0.9; TAG = sys.argv[3] if len(sys.argv) > 3 else 'a'; ENT = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
EMt = torch.tensor(EM, dtype=torch.float32, device=dev)

def rollout(net, B, rng, graded=False, oracle=False):
    s = rng.integers(3, size=(B, 3)); k = rng.integers(3, size=B)
    c = np.zeros((B, 3)); c[np.arange(B), k] = C0
    inp = torch.full((B, 1, 3), 3, device=dev); S, X, W, P, POST, F0 = [], [], [], [], [], []
    orc = Oracle(B, C0) if oracle else None
    with torch.no_grad():
        for t in range(T):
            lg, ef = net(inp); p = F.softmax(lg[:, -1], -1).cpu().numpy().astype(float); w = F.softmax(ef[:, -1], -1).cpu().numpy().astype(float)
            p /= p.sum(-1, keepdims=True)
            if oracle: POST.append(orc.post()); F0.append(orc.forecast0()[1])
            x, s2 = step(rng, s, c, w, p); S.append(s); X.append(x); W.append(w); P.append(p)
            if oracle: orc.update(x, w, p)
            s = s2; inp = torch.cat([inp, torch.tensor(x, device=dev)[:, None]], 1)
    out = dict(S=np.stack(S, 1), X=np.stack(X, 1), W=np.stack(W, 1), P=np.stack(P, 1), k=k, c=c)
    if oracle: out.update(POST=np.stack(POST, 1), F0=np.stack(F0, 1))
    return out

def evaluate(net, rng, B=2000):
    r = rollout(net, B, rng, oracle=True); ar = np.arange(B); k = r['k']
    rew = (r['X'] == 0).sum(2).mean(0)                                   # per step
    w_true = r['W'][ar, :, k].mean(0); post_true = r['POST'][ar, :, k].mean(0)
    p0 = r['P'][..., 0]; tilt = p0 - r['F0']                             # head's P(token 0) minus exact untouched forecast
    act = np.eye(3, dtype=bool)[k][:, None, :].repeat(T, 1)
    return dict(reward=float(rew.mean()), reward_late=float(rew[16:].mean()),
                w_true=[round(float(v), 2) for v in w_true[[0, 1, 2, 4, 6, 8, 12, 16, 24, 31]]],
                oracle_post_true=[round(float(v), 2) for v in post_true[[0, 1, 2, 4, 6, 8, 12, 16, 24, 31]]],
                corr_w_post=float(np.corrcoef(r['W'][:, 4:].ravel(), r['POST'][:, 4:].ravel())[0, 1]),
                tilt_active_late=float(tilt[:, 16:][act[:, 16:]].mean()), tilt_inactive_late=float(tilt[:, 16:][~act[:, 16:]].mean()),
                wmax_mean=float(r['W'].max(2).mean()))

if __name__ == '__main__':
    STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; B = 256
    torch.manual_seed(0); rng = np.random.default_rng(1)
    net = Net().to(dev); net.load_state_dict(torch.load('pre.pt')); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.0)
    t0 = time.time(); log = []
    for it in range(STEPS + 1):
        if it % 250 == 0:
            ev = evaluate(net, np.random.default_rng(123)); ev['step'] = it; log.append(ev)
            print(json.dumps(ev), f'{time.time()-t0:.0f}s', flush=True)
            torch.save(net.state_dict(), f'post_{TAG}.pt'); json.dump(log, open(f'post_{TAG}.json', 'w'))
        if it == STEPS: break
        r = rollout(net, B, rng)
        X = torch.tensor(r['X'], device=dev); S = torch.tensor(r['S'], device=dev); c = torch.tensor(r['c'], dtype=torch.float32, device=dev)
        inp = torch.cat([torch.full((B, 1, 3), 3, device=dev), X[:, :-1]], 1)
        lg, ef = net(inp); p = F.softmax(lg, -1); w = F.softmax(ef, -1)
        px = p.gather(-1, X[..., None])[..., 0]; hx = EMt[S, X]                 # (B,T,3)
        rho = c[:, None, :] * w
        ll = torch.log((1 - rho) * hx + rho * px + 1e-12).sum(2)                # (B,T)
        rew = (X == 0).sum(2).float(); G = torch.zeros_like(rew); g = torch.zeros(B, device=dev)
        for t in range(T - 1, -1, -1): g = rew[:, t] + GAMMA * g; G[:, t] = g
        adv = G - G.mean(0, keepdim=True)
        loss_rl = -(adv.detach() * ll).mean(); loss_pred = F.cross_entropy(lg.reshape(-1, 3), X.reshape(-1))
        ent = -(w * torch.log(w + 1e-9)).sum(-1).mean(); loss = loss_rl + LAM * loss_pred - ENT * ent
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
