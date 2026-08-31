"""The rho-record echo world: generative process + exact Bayes filters.

Variant A ("variant 1"): the TRUE actions never enter the stream; the record
shows a_tilde = a * eps, P(eps=-1)=r. The world hears the true a.

Model (memoryless-nature version, the "pure kappa cell" generalized):
  per episode: theta ~ Unif{+1,-1}, c ~ Unif[0,1]
  per round t=1..T:
    a_t ~ Unif{+1,-1}                      (true action; PRIVATE)
    s_t = a_t            w.p. c            (echo)
        = nu_t           w.p. 1-c,  P(nu_t = theta) = 1-lam   (nature)
    x_t = s_t flipped w.p. q               (observation channel)
    atil_t = a_t flipped w.p. r            (the corrupted record)
  public stream: BOS, atil_1, x_1, atil_2, x_2, ...

Exact filters (derivation: ~/mathpad.md "The rho-record echo world", 2026-08-13):
  agent    per-round likelihood  L = 1/2 [1 + bq( c * x*a     + (1-c) bl x*theta )]
  observer per-round likelihood  L = 1/2 [1 + bq( c*br * x*at + (1-c) bl x*theta )]
  (substitution principle: observer = agent with (xa, bq) -> (x atil, bq br)
   in the echo term only), bq=1-2q, bl=1-2lam, br=1-2r.

All filters are exact Bayes over theta in {+1,-1} x c on a midpoint grid.
CPU numpy, vectorized over episodes. float64.
"""
import numpy as np

# ---- experiment constants (the single source of truth) ----
LAM = 0.20     # nature flip rate       (bl = 0.6)
Q   = 0.10     # observation noise      (bq = 0.8)
R   = 0.25     # record corruption      (br = 0.5)
T   = 128      # rounds per episode  -> 257 tokens with BOS
NC  = 200     # c-grid size (midpoints)

# token ids
BOS, A_NEG, A_POS, X_NEG, X_POS = 0, 1, 2, 3, 4
VOCAB = 5


def gen_batch(B, rng, T=T, lam=LAM, q=Q, r=R, c_fixed=None, theta_fixed=None):
    """Sample B episodes. Returns dict of (B,) latents and (B,T) +/-1 arrays."""
    theta = (theta_fixed * np.ones(B) if theta_fixed is not None
             else rng.choice([-1.0, 1.0], size=B))
    c = (c_fixed * np.ones(B) if c_fixed is not None else rng.uniform(0, 1, size=B))
    a = rng.choice([-1.0, 1.0], size=(B, T))
    echo = rng.uniform(size=(B, T)) < c[:, None]
    nu = theta[:, None] * np.where(rng.uniform(size=(B, T)) < 1 - lam, 1.0, -1.0)
    s = np.where(echo, a, nu)
    x = s * np.where(rng.uniform(size=(B, T)) < 1 - q, 1.0, -1.0)
    atil = a * np.where(rng.uniform(size=(B, T)) < 1 - r, 1.0, -1.0)
    return dict(theta=theta, c=c, a=a, s=s, x=x, atil=atil)


def tokens(ep):
    """(B, 1+2T) int64 token stream: BOS, atil_1, x_1, ..., atil_T, x_T."""
    B, T_ = ep["a"].shape
    out = np.empty((B, 1 + 2 * T_), dtype=np.int64)
    out[:, 0] = BOS
    out[:, 1::2] = np.where(ep["atil"] > 0, A_POS, A_NEG)
    out[:, 2::2] = np.where(ep["x"] > 0, X_POS, X_NEG)
    return out


def _filter(act, x, echo_contrast, lam, q, Nc):
    """Shared exact-Bayes filter. act = the action stream the reader conditions
    on (true a for the agent, atil for the observer); echo_contrast = bq for
    the agent, bq*br for the observer (the substitution principle).

    Returns per-round PRE-update quantities (i.e. the predictive for x_t given
    rounds 1..t-1 and act_t):
      p_pos (B,T)  P(x_t = +1 | past, act_t)
      f_echo (B,T) echo component of the predictive:  echo_contrast * E[c] * act_t
      f_nat  (B,T) nature component:                  bq*bl * E[(1-c)theta]
      kappa  (B,T) posterior mean of c BEFORE round t
      thpos  (B,T) P(theta=+1) BEFORE round t
    """
    B, T_ = act.shape
    bq, bl = 1 - 2 * q, 1 - 2 * lam
    cg = (np.arange(Nc) + 0.5) / Nc
    w = np.full((B, 2, Nc), 1.0 / (2 * Nc))          # posterior over theta x c
    th = np.array([1.0, -1.0])[None, :, None]        # theta axis
    c = cg[None, None, :]
    p_pos = np.empty((B, T_)); f_echo = np.empty((B, T_)); f_nat = np.empty((B, T_))
    kappa = np.empty((B, T_)); thpos = np.empty((B, T_))
    for t in range(T_):
        Ec = np.einsum('bt,t->b', w.sum(axis=1), cg)
        E1cth = (w * (1 - c) * th).sum(axis=(1, 2))
        fe = echo_contrast * Ec * act[:, t]
        fn = bq * bl * E1cth
        p_pos[:, t] = 0.5 * (1 + fe + fn)
        f_echo[:, t] = fe; f_nat[:, t] = fn
        kappa[:, t] = Ec; thpos[:, t] = w[:, 0, :].sum(axis=1)
        at = act[:, t][:, None, None]; xt = x[:, t][:, None, None]
        L = 0.5 * (1 + echo_contrast * c * xt * at + bq * bl * (1 - c) * xt * th)
        w = w * L
        w /= w.sum(axis=(1, 2), keepdims=True)
    return dict(p_pos=p_pos, f_echo=f_echo, f_nat=f_nat, kappa=kappa, thpos=thpos)


def observer_filter(ep, lam=LAM, q=Q, r=R, Nc=NC):
    """Optimal stream-observer: conditions on (atil, x) only."""
    return _filter(ep["atil"], ep["x"], (1 - 2 * q) * (1 - 2 * r), lam, q, Nc)


def agent_filter(ep, lam=LAM, q=Q, Nc=NC):
    """Optimal agent: conditions on (true a, x). atil adds nothing given a."""
    return _filter(ep["a"], ep["x"], (1 - 2 * q), lam, q, Nc)


def xslot_loss(p_pos, x):
    """Per-round log-loss of a predictive p_pos = P(x_t=+1|...): (B,T) nats."""
    p_real = np.where(x > 0, p_pos, 1 - p_pos)
    return -np.log(np.clip(p_real, 1e-12, None))


def observer_likelihood_marginalized(x, atil, theta, c, lam=LAM, q=Q, r=R):
    """P(x, atil | theta, c) by EXPLICIT marginalization over the true action —
    used only to validate the closed form (substitution principle)."""
    bq, bl = 1 - 2 * q, 1 - 2 * lam
    p_nat = 0.5 * (1 + bq * bl * x * theta)
    tot = 0.0
    for a in (+1.0, -1.0):
        p_a = 0.5
        p_at = (1 - r) if a == atil else r
        p_x = c * 0.5 * (1 + bq * x * a) + (1 - c) * p_nat
        tot += p_a * p_at * p_x
    return tot


def observer_likelihood_closed(x, atil, theta, c, lam=LAM, q=Q, r=R):
    bq, bl, br = 1 - 2 * q, 1 - 2 * lam, 1 - 2 * r
    return 0.25 * (1 + bq * (c * br * x * atil + (1 - c) * bl * x * theta))
