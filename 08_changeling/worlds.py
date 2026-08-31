"""Changeling world v0: coupled ring actors — exact kernels + planning tables.

Single source of truth for parameters (design doc: DESIGN_changeling_v0_worldselect.md).

Conventions (all float64 numpy):
  states/tokens live in Z_n.
  E[s, u]            = P(token u | state s)                  (honest readout)
  K[o, w, s, s']     = P(s' | s, other-token o, own-token w) (pull kernel)
  Chain A: own token u, other token v  ->  TA[u, v] = K[v, u]
  Chain B: own token v, other token u  ->  TB[u, v] = K[u, v]
  h[t][sA, sB]       = P_base(success at T | states at t)
  M[t][sA, sB, u]    = E over v ~ E_B(.|sB) of h[t+1] at next states given (u, v)
  N[t][sA, sB, v]    = E over u ~ E_A(.|sA) of h[t+1] at next states given (u, v)

The two actors are deliberately DIFFERENT (B's readout is blunter by DQ):
the central lemma says identity stays unidentifiable at zero tilt anyway,
and validate.py V1 tests exactly that with the asymmetric kernels.
"""
import numpy as np
from dataclasses import dataclass, field

# v0.1 amendment (2026-08-31): the v0.0 grid (T=32, eps=.10, kappa<=8, exact-
# match reward) failed the forgivingness falsifier — S_informed <= .50 even at
# d=0 and identity collapse arrived at t~28-31/32. Changes: T 32->64, slip
# .10->.05, reward = tolerance-1 ball (the natural "comfortably reach"),
# kappa {8,16,32}. v0.0 numbers preserved in results/sweep_v0.json.
N_STATES = 6
T_HORIZON = 64
EPS_SLIP = 0.05
G_FLOOR = 0.05
DQ_ASYM = 0.15          # E_B fidelity = q0 - DQ_ASYM (actors differ)
REWARD_TOL = 1          # success = both chains within this ring distance
# v0.2 selection grid. Documented and dropped: pure-crossed coupling (0.7, 0)
# — no self-control channel (P7, sweeps v0.0/v0.1); d=3 antipodal goals —
# infeasible for every agent (occ_inf ~ occ_base, both prior sweeps); rho<1 —
# evidence too weak to cross 2 nats in T=64; rho saturates by ~4-8 (anchor floor).
# Final probe round: sluggish kernels (stay mass >= .15) cap informed
# occupancy at ~.44 at d=2; responsive kernels (c_o+c_s+eps = 1) + sharper
# readout q0=.9 lift it to ~.54 with collapse at t~14. Grid re-centered there.
Q0_GRID = (0.80, 0.90)
COUPLING_GRID = ((0.70, 0.25), (0.60, 0.35))   # (c_other, c_self)
KAPPA_GRID = (8.0, 16.0, 32.0)                 # terminal mode (v0.1, kept)
DIST_GRID = (0, 2)      # goal = (0, d)


def emission(n, q0, g=G_FLOOR):
    E = np.zeros((n, n))
    for s in range(n):
        E[s, s] += q0
        E[s, (s + 1) % n] += (1 - q0) / 2
        E[s, (s - 1) % n] += (1 - q0) / 2
    return (1 - g) * E + g / n


def step_toward(n):
    """D[target, s, s'] = one-step-toward-target distribution (ties split)."""
    D = np.zeros((n, n, n))
    for tgt in range(n):
        for s in range(n):
            d = (tgt - s) % n
            if d == 0:
                D[tgt, s, s] = 1.0
            elif d < n - d:
                D[tgt, s, (s + 1) % n] = 1.0
            elif d > n - d:
                D[tgt, s, (s - 1) % n] = 1.0
            else:  # antipodal tie
                D[tgt, s, (s + 1) % n] = 0.5
                D[tgt, s, (s - 1) % n] = 0.5
    return D


def pull_kernel(n, c_other, c_self, eps=EPS_SLIP):
    """K[o, w, s, s']: pulled toward other-token o w.p. c_other, own token w
    w.p. c_self, +-1 slip w.p. eps, stay otherwise."""
    D = step_toward(n)
    stay = np.eye(n)
    slip = 0.5 * (np.roll(np.eye(n), 1, axis=1) + np.roll(np.eye(n), -1, axis=1))
    rest = 1.0 - c_other - c_self - eps
    assert rest >= -1e-12
    rest = max(rest, 0.0)
    K = (c_other * D[:, None, :, :]           # o indexes first axis
         + c_self * D[None, :, :, :]          # w indexes second axis
         + eps * slip[None, None, :, :]
         + rest * stay[None, None, :, :])
    return np.broadcast_to(K, (n, n, n, n)).copy()


RHO_GRID = (2.0, 4.0, 8.0)      # running-reward tilt strengths (kappa=1 there)


@dataclass
class World:
    q0: float
    c_other: float
    c_self: float
    d_goal: int
    kappa: float
    mode: str = 'terminal'       # 'terminal': reward at T; 'running': per-round
    rho: float = 0.0             # running mode: exponential tilt e^{rho * r_t}
    n: int = N_STATES
    T: int = T_HORIZON
    EA: np.ndarray = field(init=False)
    EB: np.ndarray = field(init=False)
    TA: np.ndarray = field(init=False)   # [u, v, s, s']
    TB: np.ndarray = field(init=False)   # [u, v, s, s']
    h: np.ndarray = field(init=False)    # [t, sA, sB], t = 0..T
    M: np.ndarray = field(init=False)    # [t, sA, sB, u], t = 0..T-1
    N: np.ndarray = field(init=False)    # [t, sA, sB, v]

    def __post_init__(self):
        n, T = self.n, self.T
        self.goal = (0, self.d_goal % n)
        self.EA = emission(n, self.q0)
        self.EB = emission(n, max(self.q0 - DQ_ASYM, 0.40))
        K = pull_kernel(n, self.c_other, self.c_self)
        self.TA = np.transpose(K, (1, 0, 2, 3)).copy()  # TA[u, v] = K[v, u]
        self.TB = K                                      # TB[u, v] = K[u, v]
        gA = np.arange(n); gB = np.arange(n)
        dA = np.minimum((gA - self.goal[0]) % n, (self.goal[0] - gA) % n)
        dB = np.minimum((gB - self.goal[1]) % n, (self.goal[1] - gB) % n)
        self.r_ball = ((dA[:, None] <= REWARD_TOL)
                       & (dB[None, :] <= REWARD_TOL)).astype(float)
        h = np.zeros((T + 1, n, n))
        # terminal: h_t = P_base(final state in ball | s_t).  running: exact
        # exponential tilt, h_t = E_base[exp(rho * sum_{tau>t} r(s_tau)) | s_t]
        # (reward counted on arrival); plan uses kappa=1 there by convention.
        h[T] = self.r_ball if self.mode == 'terminal' else np.ones((n, n))
        M = np.zeros((T, n, n, n))
        Nt = np.zeros((T, n, n, n))
        for t in range(T - 1, -1, -1):
            h_eff = h[t + 1] if self.mode == 'terminal' \
                else np.exp(self.rho * self.r_ball) * h[t + 1]
            # joint[u, v, sA, sB] = E over next states of h_eff
            joint = np.einsum('uvax,uvby,xy->uvab', self.TA, self.TB, h_eff)
            M[t] = np.einsum('bv,uvab->abu', self.EB, joint)
            Nt[t] = np.einsum('au,uvab->abv', self.EA, joint)
            h[t] = np.einsum('au,abu->ab', self.EA, M[t])
        self.h, self.M, self.N = h, M, Nt

    def success(self, sA, sB, tol=0):
        n = self.n
        dA = np.minimum((sA - self.goal[0]) % n, (self.goal[0] - sA) % n)
        dB = np.minimum((sB - self.goal[1]) % n, (self.goal[1] - sB) % n)
        return (dA <= tol) & (dB <= tol)
