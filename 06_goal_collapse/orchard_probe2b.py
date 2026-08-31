"""Follow-up: is the goal stored ACROSS positions (attention-served)?
Apply the encoder goal-edit at the last w positions (same delta, coherent
since the goal is constant within a pursuit); also joint goal+world edit."""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from orchard import Net, persona_gen, DST, T, S, TOK_A0
from orchard_probe2 import fit_probes, DEV

@torch.no_grad()
def logits_edit_window(net, pref, layer, delta, w):
    Tn = pref.shape[1]
    mask = torch.triu(torch.ones(Tn, Tn, device=pref.device,
                                 dtype=torch.bool), 1)
    h = net.tok(pref) + net.pos(torch.arange(Tn, device=pref.device))
    lo = max(0, Tn - w)
    for j, blk in enumerate(net.blocks):
        if j == layer:
            h = h.clone(); h[:, lo:] += delta[:, None, :]
        h = blk(h, mask)
    return net.head(net.lnf(h))[:, -1]

net = Net().to(DEV)
net.load_state_dict(torch.load('orchard_runs/A/p1_final.pt',
                               map_location=DEV))
net.eval()
ev = persona_gen(2000, np.random.default_rng(999))
toks = torch.tensor(ev['toks'], device=DEV)
with torch.no_grad():
    _, hs_full = net(toks, return_hidden=True)
dec_pos = 3 + 3 * np.arange(T)
Bm = ev['wbelief'].reshape(-1, S); Gm = ev['goal_post'].reshape(-1, S)
ntr = 1400 * T
probes = {}
for li in (3, 4, 5):
    H = hs_full[li][:, dec_pos].reshape(-1, 64).cpu().numpy()
    for name, Y in (('world', Bm), ('goal', Gm)):
        probes[(li, name)] = fit_probes(H[:ntr], Y[:ntr])

rng = np.random.default_rng(5)
print('goal-swap flips vs edit window w (positions), encoder edits:')
for li in (4, 5):
    _, Wg, _, _ = probes[(li, 'goal')]
    for w in (1, 4, 12, 36, 999):
        fl, n = [], []
        for tstar in (4, 12, 24, 40):
            Lp = 3 + 3 * tstar + 1
            pref = toks[:512, :Lp]
            idx = np.arange(512)
            b = ev['wbelief'][:512, tstar]; g = ev['goal_post'][:512, tstar]
            sh = b.argmax(1)
            tlo = ev['tlo'][:512, tstar]; thi = ev['thi'][:512, tstar]
            gp = g.copy()
            gp[idx, tlo], gp[idx, thi] = g[idx, thi], g[idx, tlo]
            cf = DST[sh, gp.argmax(1)]; fa = DST[sh, g.argmax(1)]
            div = cf != fa
            if div.sum() < 20: continue
            d_enc = (gp - g)[div] @ Wg.T
            dt = torch.tensor(d_enc, device=DEV, dtype=torch.float32)
            lg = logits_edit_window(net, pref[div], li, dt, w)
            p1 = F.softmax(lg[:, TOK_A0:TOK_A0+3], -1).cpu().numpy()
            fl.append((p1.argmax(1) == cf[div]).mean())
        print(f'  L{li} w={w:4d}: flip {np.mean(fl):.2f}')

print('joint goal-swap + world-rotate edit (full counterfactual), L4, w=12:')
_, Wg, _, _ = probes[(4, 'goal')]
_, Ww, _, _ = probes[(4, 'world')]
fl = []
for tstar in (4, 12, 24, 40):
    Lp = 3 + 3 * tstar + 1
    pref = toks[:512, :Lp]
    idx = np.arange(512)
    b = ev['wbelief'][:512, tstar]; g = ev['goal_post'][:512, tstar]
    tlo = ev['tlo'][:512, tstar]; thi = ev['thi'][:512, tstar]
    gp = g.copy(); gp[idx, tlo], gp[idx, thi] = g[idx, thi], g[idx, tlo]
    bp = np.roll(b, 1, axis=1)
    cf = DST[bp.argmax(1), gp.argmax(1)]
    fa = DST[b.argmax(1), g.argmax(1)]
    div = cf != fa
    if div.sum() < 20: continue
    d = (gp - g)[div] @ Wg.T + (bp - b)[div] @ Ww.T
    dt = torch.tensor(d, device=DEV, dtype=torch.float32)
    lg = logits_edit_window(net, pref[div], 4, dt, 12)
    p1 = F.softmax(lg[:, TOK_A0:TOK_A0+3], -1).cpu().numpy()
    fl.append((p1.argmax(1) == cf[div]).mean())
print(f'  joint flip {np.mean(fl):.2f}')
