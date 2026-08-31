"""Binding tests on the corrupted-record nets: is the self-knowledge (a) used as a prior
at all (PUPPET test), and (b) computed by the live policy circuitry vs a separate copy
(STEERING test)? Run on the annealed net (has self-knowledge) and cold arm A (control).

PUPPET: roll out with FORCED uniform actions (net's intentions ignored; mirror + record
follow the forced truth). A tokens-only camp predictor is unharmed; a net that fuses an
own-policy prior mispredicts toward its INTENTIONS. Compare the net's camp-CE against two
reference decoders on the same rollouts: prior = intended policy (wrong now) vs prior =
uniform (correct now). Position between them = the intention leak.

STEERING: on self-rollouts, inject a gradient-computed direction at (layer 3, decision
position t') that boosts a target action a* the net did NOT take (tokens unchanged).
If the fused record-posterior is computed by the live policy machinery, the net's camp
predictions at LATER same-key rounds shift toward a*. Random-direction control.
"""
from __future__ import annotations
import numpy as np
import torch, torch.nn.functional as F

from ambush import S, TOK_X0, TOK_A0, TOK_C0, TOK_BOS, BASE, THETA
from ambush import World, sample_rows, filt_obs, filt_step
from mirror import Mirror
from mirror2 import corrupt
from mirror_probe import load, DEV
from mirror2_diag import decoder_ce

T, RHO = 24, 0.3


@torch.no_grad()
def rollout(net, B, seed, puppet=False):
    rng = np.random.default_rng(seed)
    w = World(B, rng)
    is_m = rng.random(B) < 0.5
    qcamp = rng.dirichlet(np.full(S, 0.5), B)
    mir = Mirror(B)
    pri = np.zeros((B, T, S)); truths = np.zeros((B, T), int)
    recs = np.zeros((B, T), int); keys = np.zeros((B, T), int); camps = np.zeros((B, T), int)
    eta = np.full((B, S), 1 / S)
    tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
    for t in range(T):
        z = w.emit(); eta = filt_obs(eta, z); keys[:, t] = z
        tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(DEV)], 1)
        p = F.softmax(net(tt)[:, -1, TOK_A0:TOK_C0], -1).cpu().numpy()
        pri[:, t] = p                                    # intended policy (recorded always)
        a = rng.integers(0, S, B) if puppet else sample_rows(p, rng)
        truths[:, t] = a
        pc, _ = mir.camp_dist(z, is_m, qcamp)
        c = sample_rows(pc, rng); camps[:, t] = c
        rec = corrupt(a, RHO, rng); recs[:, t] = rec
        mir.update(z, a)
        tt = torch.cat([tt, torch.from_numpy(
            np.stack([TOK_A0 + rec, TOK_C0 + c], 1)).to(DEV)], 1)
        w.step(); eta = filt_step(eta)
    lsm = F.log_softmax(net(tt[:, :-1]), -1)
    camp_pos = 2 + 3 * np.arange(T)
    tgt = tt[:, 1:]
    ce = -(lsm.gather(-1, tgt[..., None]).squeeze(-1)[:, camp_pos]).cpu().numpy()
    obs = float(ce[is_m][:, 3:].mean())
    return tt, pri, truths, recs, keys, camps, is_m, obs


def puppet_test(net, tag):
    _, priS, trS, reS, keS, caS, imS, obsS = rollout(net, 1500, seed=31, puppet=False)
    _, priP, trP, reP, keP, caP, imP, obsP = rollout(net, 1500, seed=31, puppet=True)
    ce_int = decoder_ce(priP, trP, reP, keP, caP, imP, "contextual")   # intended prior (WRONG)
    ce_cor = decoder_ce(priP, trP, reP, keP, caP, imP, "uniform")      # correct prior
    leak = (obsP - ce_cor) / max(ce_int - ce_cor, 1e-6)
    print(f"[{tag}] PUPPET: camp-CE self={obsS:.3f} puppet={obsP:.3f} | reference decoders on "
          f"puppet: intended-prior={ce_int:.3f} correct-prior={ce_cor:.3f} | "
          f"leak fraction={leak:+.2f}")
    return dict(obsS=obsS, obsP=obsP, ce_int=ce_int, ce_cor=ce_cor, leak=leak)


def forward_inject(net, tt, layer, pos, delta):
    B, L = tt.shape
    x = net.emb(tt) + net.pos(torch.arange(L, device=DEV))[None]
    m = torch.triu(torch.ones(L, L, device=DEV, dtype=torch.bool), 1)
    for i, b in enumerate(net.blocks):
        if i == layer and delta is not None:
            x = x.clone(); x[:, pos] = x[:, pos] + delta
        x = b(x, m)
    return net.head(net.lnf(x))


def steer_test(net, tag, B=768, tp=8, layer=3):
    tt, pri, truths, recs, keys, camps, is_m, _ = rollout(net, B, seed=37, puppet=False)
    p = 1 + 3 * tp
    astar = (truths[:, tp] + 1) % 3
    at = torch.from_numpy(astar).to(DEV)
    # gradient direction for boosting a* at t'
    x = net.emb(tt) + net.pos(torch.arange(tt.shape[1], device=DEV))[None]
    m = torch.triu(torch.ones(tt.shape[1], tt.shape[1], device=DEV, dtype=torch.bool), 1)
    with torch.no_grad():
        for i in range(layer):
            x = net.blocks[i](x, m)
    x = x.detach().requires_grad_(True)
    y = x
    for i in range(layer, len(net.blocks)):
        y = net.blocks[i](y, m)
    lg = net.head(net.lnf(y))
    loss = lg[torch.arange(B), p, TOK_A0 + at].sum()
    loss.backward()
    g = x.grad[:, p]
    g = g / g.norm(dim=-1, keepdim=True)
    torch.set_grad_enabled(False)
    lp0 = F.log_softmax(forward_inject(net, tt, layer, p, None), -1)
    out = {}
    for lam in (4.0, 8.0):
        dp = []
        for dvec, name in ((lam * g, "steer"),
                           (lam * torch.randn_like(g) / g.shape[-1] ** 0.5 * g.norm(dim=-1,
                            keepdim=True) * 0 + lam * F.normalize(torch.randn_like(g), dim=-1),
                            "rand")):
            lp1 = F.log_softmax(forward_inject(net, tt, layer, p, dvec), -1)
            dpol = float((lp1[torch.arange(B), p, TOK_A0 + at] -
                          lp0[torch.arange(B), p, TOK_A0 + at]).mean())
            # later same-key camp predictions: does p(camp = a*) rise?
            terms = []
            for t2 in range(tp + 1, T):
                cp = 2 + 3 * t2
                sel = (keys[:, t2] == keys[:, tp]) & is_m
                if sel.sum() < 10:
                    continue
                d = (lp1[:, cp, TOK_C0 + at] - lp0[:, cp, TOK_C0 + at]).cpu().numpy()
                terms.append(d[sel])
            dcamp = float(np.concatenate(terms).mean())
            dp.append((name, dpol, dcamp))
        out[lam] = dp
        print(f"[{tag}] STEER lam={lam}: " + "  ".join(
            f"{n}: dlogp(a*@t')={a:+.2f}, dlogp(camp=a* later)={c:+.4f}" for n, a, c in dp))
    torch.set_grad_enabled(True)
    return out


def main():
    for tag, path in (("annealed", "mirror2_runs/A_anneal/p2_ckpt_008000.pt"),
                      ("cold-A", "mirror2_runs/A/p2_ckpt_008000.pt")):
        net = load(f"{BASE}/{path}")
        puppet_test(net, tag)
        steer_test(net, tag)


if __name__ == "__main__":
    main()
