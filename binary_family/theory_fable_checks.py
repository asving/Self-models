#!/usr/bin/env python3
"""Numerical checks for theory_fable.md (two-court renormalization curves).

Five checks, all CPU / numpy, seconds each:
  1. FACTORIZATION / E6: exact joint filter mu in Delta({0,1}x{th1,th2});
     identity-only perturbation at c>0 => Lambda^x == 0, mu stays factored.
     Plus: the hidden-action reader reproduces E6's Delta-eta* formula.
  2. q-MONOTONICITY: class A (k=0) eta-do => |E Lambda^x_inf| monotone in q
     (data-processing theorem); class B (k=1) may be non-monotone (fig3b).
  3. MIDPOINT P*: (a) vertex readers: LIL-scale oscillation, sign changes ~ sqrt(T);
     (b) interior readers: bounded non-convergent oscillation between odds values.
  4. STATIC WORLD + FAIR-COIN STREAM (misspecified channel): exotic sublinear
     scale of Lambda^x (predicted ~ s^{1/4} from local-time differences).
  5. LAN CHECK: small eta-do, class A (global faithfulness to R-):
     Var(Lambda_inf) / |E Lambda_inf| ~ 2.
"""
import numpy as np

rng_global = np.random.default_rng(0)

def logit(p): return np.log(p) - np.log1p(-p)
def sig(l):   return 1.0 / (1.0 + np.exp(-l))
def clip(p):  return np.clip(p, 1e-15, 1 - 1e-15)

# ---------------------------------------------------------------- check 1 ----
def check1_factorization(T=400, N=400, lam=0.1, c=0.4, q=0.3,
                         b=(0.3, 0.7), classB=False, kB=(1.0, 1.5), bB=(0.0, 0.5),
                         seed=1):
    """Joint exact filter over (s, theta). Identity-only perturbation:
    R+ prior p+=(0.2,0.8), R- prior p-=(0.5,0.5); same eta0=0.5. Agent = theta2.
    World consequence c>0. Track Lambda^x, factorization error, conditional split."""
    R = np.random.default_rng(seed)
    # mu[n, s, th]
    def init_mu(p_th):
        mu = np.zeros((N, 2, 2))
        for s in range(2):
            for th in range(2):
                mu[:, s, th] = 0.5 * p_th[th]
        return mu
    mu_p = init_mu([0.2, 0.8])   # R+ (perturbed identity prior)
    mu_m = init_mu([0.5, 0.5])   # R-
    s = (R.random(N) < 0.5).astype(int)
    eta_ag = np.full(N, 0.5)     # agent's own world belief (public filter)
    Lx = np.zeros(N)
    max_Lx, max_facterr, max_condsplit = 0.0, 0.0, 0.0

    def pol(th_idx, ell):
        if classB:
            return sig(kB[th_idx] * ell + bB[th_idx])
        return np.full_like(ell, b[th_idx])

    for t in range(T):
        ell_pub = logit(clip(eta_ag))
        # agent acts (true theta = theta2)
        pa_true = pol(1, ell_pub)
        a = (R.random(N) < pa_true).astype(int)
        # --- readers' a-step: reweight theta slices ---
        for mu in (mu_p, mu_m):
            for th in range(2):
                pa = pol(th, ell_pub)
                lik = np.where(a == 1, pa, 1 - pa)
                mu[:, :, th] *= lik[:, None]
            mu /= mu.sum(axis=(1, 2), keepdims=True)
        # --- world moves, emits x ---
        take = R.random(N) < c
        flip = R.random(N) < lam
        s2 = np.where(take, a, np.where(flip, 1 - s, s))
        x = np.where(R.random(N) < q, 1 - s2, s2)
        # --- readers' x-step ---
        # per current state s: P(s'=1|s,a)
        preds = []
        for mu in (mu_p, mu_m):
            eta1 = mu[:, 1, :].sum(axis=1)          # P(s=1)
            ep = c * a + (1 - c) * (lam + (1 - 2 * lam) * eta1)  # P(s'=1)
            px = ep * (1 - q) + (1 - ep) * q         # P(x=1)
            preds.append(clip(px))
        pxp, pxm = preds
        Lx += np.where(x == 1, np.log(pxp / pxm), np.log((1 - pxp) / (1 - pxm)))
        max_Lx = max(max_Lx, np.abs(Lx).max())
        for mu in (mu_p, mu_m):
            new = np.zeros_like(mu)
            for sold in range(2):
                ep_s = c * a + (1 - c) * (lam + (1 - 2 * lam) * sold)
                pxs1 = ep_s * (1 - q) + (1 - ep_s) * q     # P(x=1 | s'=... ) marginal over s'
                # T^{a,x}[sold, s'] = P(s'|sold,a) P(x|s')
                for snew in range(2):
                    ps_new = ep_s if snew == 1 else 1 - ep_s
                    pxg = (1 - q) if snew == 1 else q      # P(x=1|s'=snew)
                    px_tok = np.where(x == 1, pxg, 1 - pxg)
                    new[:, snew, :] += mu[:, sold, :] * (ps_new * px_tok)[:, None]
            new /= new.sum(axis=(1, 2), keepdims=True)
            mu[...] = new
            # factorization error and conditional split
            eta_marg = mu[:, 1, :].sum(axis=1)
            p_marg = mu[:, :, 1].sum(axis=1)
            outer = np.stack([np.stack([(1-eta_marg)*(1-p_marg), (1-eta_marg)*p_marg], -1),
                              np.stack([eta_marg*(1-p_marg), eta_marg*p_marg], -1)], 1)
            max_facterr = max(max_facterr, np.abs(mu - outer).max())
            pth = mu.sum(axis=1)                       # (N, th)
            cond1 = mu[:, 1, :] / clip(pth)            # P(s=1 | th)
            max_condsplit = max(max_condsplit, np.abs(cond1[:, 0] - cond1[:, 1]).max())
        # agent public filter
        ep_ag = c * a + (1 - c) * (lam + (1 - 2 * lam) * eta_ag)
        num = np.where(x == 1, ep_ag * (1 - q), ep_ag * q)
        den = num + np.where(x == 1, (1 - ep_ag) * q, (1 - ep_ag) * (1 - q))
        eta_ag = clip(num / den)
        s = s2
    return max_Lx, max_facterr, max_condsplit


def check1_hidden_action(T=200, lam=0.1, c=0.4, q=0.3, b=(0.3, 0.7), seed=2):
    """Reader that does NOT see actions (marginalizes over pi_theta):
    conditional world beliefs split; stationary gap should approach
    c (b2-b1) / (1 - (1-c)(1-2 lam))  [E6's formula]."""
    R = np.random.default_rng(seed)
    N = 2000
    mu = np.zeros((N, 2, 2)); mu[:, 0, :], mu[:, 1, :] = 0.25, 0.25
    s = (R.random(N) < 0.5).astype(int)
    gaps = []
    for t in range(T):
        a = (R.random(N) < b[1]).astype(int)   # agent = theta2, class A
        take = R.random(N) < c
        flip = R.random(N) < lam
        s2 = np.where(take, a, np.where(flip, 1 - s, s))
        x = np.where(R.random(N) < q, 1 - s2, s2)
        new = np.zeros_like(mu)
        for th in range(2):
            for sold in range(2):
                # marginalize hidden action under pi_th
                for aa in range(2):
                    pa = b[th] if aa == 1 else 1 - b[th]
                    ep_s = c * aa + (1 - c) * (lam + (1 - 2 * lam) * sold)
                    for snew in range(2):
                        ps_new = ep_s if snew == 1 else 1 - ep_s
                        pxg = (1 - q) if snew == 1 else q
                        px_tok = np.where(x == 1, pxg, 1 - pxg)
                        new[:, snew, th] += mu[:, sold, th] * pa * ps_new * px_tok
        new /= new.sum(axis=(1, 2), keepdims=True)
        mu = new
        pth = mu.sum(axis=1)
        cond1 = mu[:, 1, :] / clip(pth)
        gaps.append(np.abs(cond1[:, 1] - cond1[:, 0]).mean())
        s = s2
    pred = c * (b[1] - b[0]) / (1 - (1 - c) * (1 - 2 * lam))
    return gaps[-1], pred

# ---------------------------------------------------------------- check 2 ----
def run_eta_row(N, T, t0, lam, c, q, k, dl, seed=1):
    """do-mode eta row (agent == R+), class set by k (k=0: class A)."""
    R = np.random.default_rng(seed)
    s = (R.random(N) < 0.5).astype(int)
    e_p = np.full(N, 0.5); e_m = np.full(N, 0.5)
    la = np.zeros(N); lx = np.zeros(N)
    for t in range(T):
        if t == t0:
            e_p = clip(sig(logit(e_p) + dl))
        pa_p, pa_m = sig(k * logit(e_p)), sig(k * logit(e_m))
        a = (R.random(N) < pa_p).astype(int)
        if t >= t0:
            la += np.where(a == 1, np.log(clip(pa_p) / clip(pa_m)),
                           np.log(clip(1 - pa_p) / clip(1 - pa_m)))
        ep_p = c * a + (1 - c) * (lam + (1 - 2 * lam) * e_p)
        ep_m = c * a + (1 - c) * (lam + (1 - 2 * lam) * e_m)
        take = R.random(N) < c
        flip = R.random(N) < lam
        s = np.where(take, a, np.where(flip, 1 - s, s))
        x = np.where(R.random(N) < q, 1 - s, s)
        q_p = clip(ep_p * (1 - q) + (1 - ep_p) * q)
        q_m = clip(ep_m * (1 - q) + (1 - ep_m) * q)
        if t >= t0:
            lx += np.where(x == 1, np.log(q_p / q_m), np.log((1 - q_p) / (1 - q_m)))
        num_p = np.where(x == 1, ep_p * (1 - q), ep_p * q)
        e_p = clip(num_p / (num_p + np.where(x == 1, (1 - ep_p) * q, (1 - ep_p) * (1 - q))))
        num_m = np.where(x == 1, ep_m * (1 - q), ep_m * q)
        e_m = clip(num_m / (num_m + np.where(x == 1, (1 - ep_m) * q, (1 - ep_m) * (1 - q))))
    return la, lx


def check2_qsweep():
    qs = np.linspace(0.05, 0.45, 9)
    out = {}
    for k in (0.0, 1.0):
        vals = []
        for i, q_ in enumerate(qs):
            la, lx = run_eta_row(4000, 400, 10, 0.1, 0.0, q_, k, 2.0, seed=100 + i)
            vals.append(abs(lx.mean()))
        out[k] = np.array(vals)
    return qs, out

# ---------------------------------------------------------------- check 3 ----
def check3_midpoint(T=20000, N=512, b1=0.3, b2=0.7, seed=5):
    R = np.random.default_rng(seed)
    A = (R.random((T, N)) < 0.5).astype(int)      # P*: fair-coin conduct
    inc = np.where(A == 1, np.log(b2 / b1), np.log((1 - b2) / (1 - b1)))
    W = np.cumsum(inc, axis=0)                    # vertex-vertex Lambda^a
    signchg_half = ((np.sign(W[T // 2:-1]) * np.sign(W[T // 2 + 1:])) < 0).sum(0).mean()
    scaled = W[-1] / np.sqrt(T)
    # interior readers p+=0.8, p-=0.5 on theta2: closed form via W
    Rr = np.exp(W)
    Lint = np.log((0.2 + 0.8 * Rr) / (0.5 + 0.5 * Rr))
    lo, hi = np.log(0.4), np.log(1.6)
    inside = (Lint[-1] >= lo - 1e-9).all() and (Lint[-1] <= hi + 1e-9).all()
    late_range = (Lint[T // 2:].max(0) - Lint[T // 2:].min(0)).mean()
    late_cross = ((np.sign(Lint[T // 2:-1]) * np.sign(Lint[T // 2 + 1:])) < 0).sum(0).mean()
    return dict(vertex_signchg_late=signchg_half, vertex_std_scaled=scaled.std(),
                interior_bounds_ok=bool(inside), interior_late_range=late_range,
                interior_late_crossings=late_cross)

# ---------------------------------------------------------------- check 4 ----
def check4_static_faircoin(T=100000, N=256, q_r=0.3, seed=7, l0=-1.0, dl=2.0):
    """Static world (lam=0,c=0) readers with channel q_r; true stream: iid Ber(1/2).
    Readers' log-odds gap stays exactly dl; Lambda^x predicted ~ s^{1/4} scale."""
    R = np.random.default_rng(seed)
    w = np.log((1 - q_r) / q_r)
    lp = np.full(N, l0 + dl); lm = np.full(N, l0)
    Lx = np.zeros(N)
    snaps, times = [], []
    X = (R.random((T, N)) < 0.5).astype(int)
    for t in range(T):
        x = X[t]
        pp = clip(sig(lp) * (1 - q_r) + (1 - sig(lp)) * q_r)
        pm = clip(sig(lm) * (1 - q_r) + (1 - sig(lm)) * q_r)
        Lx += np.where(x == 1, np.log(pp / pm), np.log((1 - pp) / (1 - pm)))
        step = np.where(x == 1, w, -w)
        lp += step; lm += step
        if t + 1 in (1000, 3162, 10000, 31623, 100000):
            snaps.append(np.median(np.abs(Lx))); times.append(t + 1)
    gap_const = np.abs((lp - lm) - dl).max()
    tt, ss = np.log(np.array(times)), np.log(np.array(snaps))
    slope = np.polyfit(tt, ss, 1)[0]
    return times, snaps, slope, gap_const

# ---------------------------------------------------------------- check 5 ----
def check5_lan(N=20000, T=200, lam=0.1, q=0.3, dl=0.3, seed=9):
    la, lx = run_eta_row(N, T, 10, lam, 0.0, q, 0.0, dl, seed=seed)
    tot = la + lx        # la == 0 for class A
    return tot.mean(), tot.var(), tot.var() / abs(tot.mean())


if __name__ == "__main__":
    print("=== Check 1: factorization / E6 (public-action reader, c=0.4) ===")
    for cb in (False, True):
        mx, fe, cs = check1_factorization(classB=cb)
        print(f"  classB={cb}:  max|Lambda^x|={mx:.3e}   max factorization err={fe:.3e}"
              f"   max conditional split |l1-l2| (eta units)={cs:.3e}")
    g, pred = check1_hidden_action()
    print(f"  hidden-action reader: measured stationary split {g:.4f}  vs  E6 formula {pred:.4f}")

    print("=== Check 2: |E Lambda^x_inf| vs q, class A (k=0) vs class B (k=1) ===")
    qs, out = check2_qsweep()
    for k, v in out.items():
        mono = np.all(np.diff(v) <= 1e-3)
        print(f"  k={k}: values={np.array2string(v, precision=3)}  monotone-decreasing={mono}")

    print("=== Check 3: midpoint P* (fair-coin conduct), beta=(0.3,0.7) ===")
    d = check3_midpoint()
    for kk, vv in d.items():
        print(f"  {kk}: {vv}")

    print("=== Check 4: static world, misspecified channel (true stream fair coins) ===")
    times, snaps, slope, gapc = check4_static_faircoin()
    print(f"  median|Lambda^x| at s={times}: {np.array2string(np.array(snaps), precision=3)}")
    print(f"  log-log slope = {slope:.3f}   (s^1/4 prediction: 0.25);  gap deviation {gapc:.2e}")

    print("=== Check 5: LAN Var ~= 2|E| (class A, small dl=0.3, P*=P-) ===")
    m, v, ratio = check5_lan()
    print(f"  E={m:.5f}  Var={v:.5f}  Var/|E|={ratio:.3f}  (prediction ~2)")
