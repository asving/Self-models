import torch, torch.nn.functional as F, numpy as np, time, sys
from world import T3, EM, step, Oracle
from net import Net
dev = 'mps'; torch.manual_seed(0); rng = np.random.default_rng(0); T, B = 32, 512
def sample(B):
    s = rng.integers(3, size=(B, 3)); xs = []
    z = np.zeros((B, 3)); p = np.full((B, 3, 3), 1 / 3)
    for t in range(T): x, s = step(rng, s, z, z, p); xs.append(x)
    return np.stack(xs, 1)
net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.0)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=STEPS, pct_start=0.05)
t0 = time.time()
for it in range(STEPS + 1):
    x = torch.tensor(sample(B), device=dev); inp = torch.cat([torch.full((B, 1, 3), 3, device=dev), x[:, :-1]], 1)
    lg, _ = net(inp); loss = F.cross_entropy(lg.reshape(-1, 3), x.reshape(-1))
    if it < STEPS: opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 500 == 0:
        xn = x.cpu().numpy(); orc = Oracle(B, 0.0); fl = 0.0; z = np.zeros((B, 3)); p = np.full((B, 3, 3), 1 / 3); L0 = orc.LI.copy()
        for t in range(T): orc.update(xn[:, t], z, p)
        print(f'step {it} CE/factor-token {loss.item():.4f}  exact-filter floor {-(orc.LI).mean()/T:.4f}  {time.time()-t0:.0f}s', flush=True)
torch.save(net.state_dict(), 'pre.pt')
