"""Which variable does the policy actually consume? Encoder-edits in three
codes: absolute goal posterior (5d), goal RELATIVE to world argmax
((g - shat) mod 5, 5d one-hot), and the composed direction d* (3d one-hot)."""
import numpy as np
import torch
import torch.nn.functional as F
from orchard import Net, persona_gen, DST, T, S, TOK_A0
from orchard_probe2 import fit_probes, logits_with_edit, DEV

net = Net().to(DEV)
net.load_state_dict(torch.load('orchard_runs/A/p1_final.pt',
                               map_location=DEV))
net.eval()
ev = persona_gen(2000, np.random.default_rng(999))
toks = torch.tensor(ev['toks'], device=DEV)
with torch.no_grad():
    _, hs_full = net(toks, return_hidden=True)
dec_pos = 3 + 3 * np.arange(T)
Bm = ev['wbelief'].reshape(-1, S)
Gm = ev['goal_post'].reshape(-1, S)
sh_all = Bm.argmax(1)
g_all = Gm.argmax(1)
rel_all = np.eye(S)[(g_all - sh_all) % S]
d_all = np.eye(3)[DST[sh_all, g_all]]
ntr = 1400 * T

for li in (4, 5):
    H = hs_full[li][:, dec_pos].reshape(-1, 64).cpu().numpy()
    pr = {}
    for name, Y in (('abs', Gm), ('rel', rel_all), ('dir', d_all)):
        pr[name] = fit_probes(H[:ntr], Y[:ntr])
    for name in ('abs', 'rel', 'dir'):
        _, W, _, _ = pr[name]
        fl, dl = [], []
        for tstar in (4, 12, 24, 40):
            Lp = 3 + 3 * tstar + 1
            idx = np.arange(512)
            b = ev['wbelief'][:512, tstar]
            g = ev['goal_post'][:512, tstar]
            sh = b.argmax(1)
            tlo = ev['tlo'][:512, tstar]
            thi = ev['thi'][:512, tstar]
            gp = g.copy()
            gp[idx, tlo], gp[idx, thi] = g[idx, thi], g[idx, tlo]
            cf = DST[sh, gp.argmax(1)]
            fa = DST[sh, g.argmax(1)]
            div = cf != fa
            if div.sum() < 20:
                continue
            if name == 'abs':
                dy = (gp - g)[div]
            elif name == 'rel':
                dy = (np.eye(S)[(gp.argmax(1) - sh) % S]
                      - np.eye(S)[(g.argmax(1) - sh) % S])[div]
            else:
                dy = (np.eye(3)[cf] - np.eye(3)[fa])[div]
            d_enc = dy @ W.T
            dt = torch.tensor(d_enc, device=DEV, dtype=torch.float32)
            with torch.no_grad():
                lg0 = net(toks[:512, :Lp][div])[:, -1, TOK_A0:TOK_A0+3]
            p0 = F.softmax(lg0, -1).cpu().numpy()
            lg = logits_with_edit(net, toks[:512, :Lp][div], li, dt)
            p1 = F.softmax(lg[:, TOK_A0:TOK_A0+3], -1).cpu().numpy()
            cfd = cf[div]
            fl.append((p1.argmax(1) == cfd).mean())
            dl.append(np.log(p1[np.arange(len(cfd)), cfd]+1e-9).mean()
                      - np.log(p0[np.arange(len(cfd)), cfd]+1e-9).mean())
        print(f'L{li} {name}-code edit: flip {np.mean(fl):.2f} '
              f'dlogp {np.mean(dl):+.2f}')
