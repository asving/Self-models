"""The generalization hop: how does RL turn persona-PREDICTION into persona-BEATING?

A. Representation preservation: fit an opponent-policy probe (residuals at decision
   positions -> the c-persona's true next-action distribution) on the PRETRAINED net;
   apply it FROZEN to the RL net (and vice versa). High transfer + small subspace angles
   = RL kept the pretraining persona-decoder and only re-routed its readout.
B. Ablation overlap: per head/MLP, measure (i) damage to c-prediction CE in the
   pretrained net (teacher-forced persona streams) and (ii) damage to AIM QUALITY in the
   RL net (P(action = argmax of the opponent's true dist), teacher-forced on its own
   rollout streams). Overlap of the critical sets = the same circuitry, re-consumed.
C. OOD personas: anti-greedy (avoids the belief argmax), cycler (terrain-independent
   period-3), copycat (repeats the AGENT's last action -- reactive, unlike any training
   persona). Catch curves vs each; chance = 1/3; in-family reference = .53 early /.63 late.
"""
from __future__ import annotations
import numpy as np
import torch, torch.nn.functional as F

from ambush import World, S, filt_obs, filt_step, onehot, sample_rows, BASE
from doppel import Personas
from doppel2 import Net2, TOK_X0, TOK_J0, TOK_BOS, jtok, joint_slice
from doppel2_curve import load2, amarg
from mirror_probe import hiddens
from mirror_circuit import wb_forward

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T = 24
RUN = f"{BASE}/doppel2_runs/A"


# ---------------------------------------------------------------- data with true c-dists
@torch.no_grad()
def passive_gen(B, seed):
    """Two personas play (pretraining distribution); record c-persona's TRUE dist."""
    rng = np.random.default_rng(seed)
    w = World(B, rng)
    p1, p2 = Personas(B, rng), Personas(B, rng)
    toks = np.zeros((B, 1 + 2 * T), dtype=np.int64); toks[:, 0] = TOK_BOS
    cd = np.zeros((B, T, S))
    for t in range(T):
        z = w.emit(); z2 = w.emit()
        toks[:, 1 + 2 * t] = TOK_X0 + z
        a = p1.act_on(z)
        p2.eta = filt_obs(p2.eta, z2)
        d = p2.dist(p2.eta, z2); cd[:, t] = d
        c = sample_rows(d, rng); p2.last = c.copy()
        toks[:, 2 + 2 * t] = jtok(a, c)
        p1.drift(); p2.eta = filt_step(p2.eta); w.step()
    return torch.from_numpy(toks).to(DEV), cd


@torch.no_grad()
def acting_gen(net, B, seed):
    """RL net acts; personas oppose; record c-dists and the net's actions."""
    rng = np.random.default_rng(seed)
    w = World(B, rng)
    opp = Personas(B, rng)
    tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
    cd = np.zeros((B, T, S)); acts = np.zeros((B, T), dtype=int)
    for t in range(T):
        z = w.emit(); z2 = w.emit()
        tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(DEV)], 1)
        a = torch.multinomial(amarg(net, tt), 1).squeeze(1)
        acts[:, t] = a.cpu().numpy()
        opp.eta = filt_obs(opp.eta, z2)
        d = opp.dist(opp.eta, z2); cd[:, t] = d
        c = sample_rows(d, rng); opp.last = c.copy()
        tt = torch.cat([tt, (TOK_J0 + 3 * a + torch.from_numpy(c).to(DEV))[:, None]], 1)
        opp.eta = filt_step(opp.eta); w.step()
    return tt, cd, acts


def ridge_fit(H, Y, lam=1.0):
    H1 = np.concatenate([H, np.ones((len(H), 1))], 1)
    return np.linalg.solve(H1.T @ H1 + lam * np.eye(H1.shape[1]), H1.T @ Y)

def r2_of(W, H, Y):
    P = np.concatenate([H, np.ones((len(H), 1))], 1) @ W
    sse = ((P - Y) ** 2).sum(0); sst = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1 - sse / sst))


def part_A(p1, p2):
    print("== A. representation preservation (opponent-policy probe transfer) ==")
    tt, cd = passive_gen(1024, 41)
    pos = 1 + 2 * np.arange(T)
    ntr = 700
    best = {}
    for tag, net in (("pretrained", p1), ("RL", p2)):
        hs = hiddens(net, tt)
        for li in (3, 4, 5, 6):
            H = hs[li][:, pos].reshape(-1, 64)
            W = ridge_fit(H[:ntr * T], cd.reshape(-1, S)[:ntr * T])
            r2 = r2_of(W, H[ntr * T:], cd.reshape(-1, S)[ntr * T:])
            if tag not in best or r2 > best[tag][0]:
                best[tag] = (r2, li, W, hs)
    (r1, l1, W1, hs1), (r2_, l2, W2, hs2) = best["pretrained"], best["RL"]
    H2 = hs2[l1][:, pos].reshape(-1, 64)
    H1 = hs1[l2][:, pos].reshape(-1, 64)
    Yte = cd.reshape(-1, S)
    t12 = r2_of(W1, H2[ntr * T:], Yte[ntr * T:])
    t21 = r2_of(W2, H1[ntr * T:], Yte[ntr * T:])
    def colspace(W):
        Q, _ = np.linalg.qr(W[:-1]); return Q[:, :S]
    cos = np.linalg.svd(colspace(W1).T @ colspace(W2), compute_uv=False)
    print(f"native: pretrained R2={r1:.3f}(L{l1})  RL R2={r2_:.3f}(L{l2})")
    print(f"frozen transfer: p1-probe on RL net = {t12:.3f}   RL-probe on p1 net = {t21:.3f}")
    print(f"probe-subspace principal cosines: {np.round(cos, 3)}")


def part_B(p1, p2):
    print("\n== B. ablation overlap: predict-critical (pretrained) vs aim-critical (RL) ==")
    tt_p, cd_p = passive_gen(768, 43)
    tt_a, cd_a, acts_a = acting_gen(p2, 768, 44)
    pos = 1 + 2 * np.arange(T)
    pos_t = torch.from_numpy(pos).to(DEV)
    comps = [("h", l, h) for l in range(6) for h in range(4)] + [("mlp", l) for l in range(6)]
    def ce_c(net, tt, cd, ablate=frozenset()):
        lg = wb_forward(net, tt[:, :-1], ablate=ablate)
        jl = lg[:, pos_t][..., TOK_J0:TOK_J0 + 9].view(len(tt), T, 3, 3)
        lc = torch.logsumexp(F.log_softmax(jl.reshape(len(tt), T, 9), -1).view(
            len(tt), T, 3, 3), dim=2)                      # log c-marginal
        return float(-(torch.from_numpy(cd).to(DEV) * lc).sum(-1).mean())
    def aim(net, tt, cd, ablate=frozenset()):
        lg = wb_forward(net, tt[:, :-1], ablate=ablate)
        jl = lg[:, pos_t][..., TOK_J0:TOK_J0 + 9].view(len(tt), T, 3, 3)
        pa = F.softmax(jl.reshape(len(tt), T, 9), -1).view(len(tt), T, 3, 3).sum(-1)
        return float((pa.argmax(-1).cpu().numpy() == cd.argmax(-1)).mean())
    base_p, base_a = ce_c(p1, tt_p, cd_p), aim(p2, tt_a, cd_a)
    rows = []
    for c in comps:
        d_pred = ce_c(p1, tt_p, cd_p, frozenset([c])) - base_p
        d_aim = base_a - aim(p2, tt_a, cd_a, frozenset([c]))
        rows.append((str(c), d_pred, d_aim))
    rows.sort(key=lambda r: -r[2])
    print(f"baseline: c-pred CE (p1) = {base_p:.3f}; aim quality (RL) = {base_a:.3f}")
    print(f"{'comp':>12} {'d_predCE(p1)':>13} {'d_aim(RL)':>10}")
    for r in rows[:8]:
        print(f"{r[0]:>12} {r[1]:13.3f} {r[2]:10.3f}")
    pred_rank = {r[0]: i for i, r in enumerate(sorted(rows, key=lambda r: -r[1]))}
    top_aim = [r[0] for r in rows[:5]]
    top_pred = [r[0] for r in sorted(rows, key=lambda r: -r[1])[:5]]
    print(f"top-5 overlap (aim-critical ∩ predict-critical): "
          f"{len(set(top_aim) & set(top_pred))}/5   {sorted(set(top_aim) & set(top_pred))}")


class OODPersonas:
    """anti-greedy | cycler | copycat (reactive to the agent -- novel structure)."""
    def __init__(self, B, kind, rng):
        self.B, self.kind, self.rng = B, kind, rng
        self.eta = np.full((B, S), 1 / S)
        self.phase = rng.integers(0, 3, B)
        self.agent_last = rng.integers(0, 3, B)
    def dist(self, z, t):
        self.eta = filt_obs(self.eta, z)
        if self.kind == "antigreedy":
            e = self.eta.copy(); e[np.arange(self.B), e.argmax(1)] = -1
            p = onehot(e.argmax(1))
        elif self.kind == "cycler":
            p = onehot((self.phase + t) % 3)
        else:
            p = onehot(self.agent_last)
        return 0.95 * p + 0.05 / S
    def post(self):
        self.eta = filt_step(self.eta)


@torch.no_grad()
def part_C(p2):
    print("\n== C. OOD personas (chance .33; in-family reference .53 early / .63 late) ==")
    for kind in ("antigreedy", "cycler", "copycat"):
        rng = np.random.default_rng(51)
        B = 512
        w = World(B, rng)
        opp = OODPersonas(B, kind, rng)
        tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=DEV)
        hits = np.zeros((B, T))
        for t in range(T):
            z = w.emit(); z2 = w.emit()
            tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(DEV)], 1)
            a = torch.multinomial(amarg(p2, tt), 1).squeeze(1)
            a_np = a.cpu().numpy()
            d = opp.dist(z2, t)
            c = sample_rows(d, rng)
            opp.agent_last = a_np.copy()
            hits[:, t] = a_np == c
            tt = torch.cat([tt, (TOK_J0 + 3 * a + torch.from_numpy(c).to(DEV))[:, None]], 1)
            opp.post(); w.step()
        print(f"  {kind:>10}: early {hits[:, :3].mean():.3f}  late {hits[:, 12:].mean():.3f}"
              f"  curve: " + " ".join(f"{hits[:, t].mean():.2f}" for t in range(0, T, 4)))


def main():
    torch.set_grad_enabled(False)
    p1 = load2(f"{RUN}/p1_ckpt_020000.pt")
    p2 = load2(f"{RUN}/p2_ckpt_008000.pt")
    part_A(p1, p2)
    part_B(p1, p2)
    part_C(p2)


if __name__ == "__main__":
    main()
