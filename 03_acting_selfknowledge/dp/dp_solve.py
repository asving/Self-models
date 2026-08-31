"""Exact finite-horizon POMDP solve for blindfolded RPS under imperfect monitoring.

GAME (n=3, cyclic; move a beats a-1, loses to a+1). Horizon T rounds. Per game an
opponent TYPE is drawn and FIXED:
  - prob beta : BEST-RESPONDER. sees agent's action distribution p_t, plays
                b ~ softmax(gamma*win(p_t)),  win(b)=p_t[(b-1)%3], gamma=6.
  - prob 1-beta : EXPLOITABLE FIXED BIAS. fixed q ~ Dirichlet(0.5,0.5,0.5), b ~ q iid.
Agent observes ONLY o_t=(a_t-b_t) mod 3 in {0 tie,1 win,2 loss}; no b_t, no action feedback.
reward = +1 win / 0 tie / -1 loss.

BELIEF (hierarchical):
  pi_t = P(opponent exploitable | history),  and conditional-on-exploitable a Dirichlet
  posterior over q (pseudo-counts alpha in R^3_{>0}).

OUTCOME LIKELIHOOD (self-legibility coupling): the agent only sees o, and to update it
must use its OWN action distribution p_t. For a given opponent b-distribution d(.),
  P(o | d, p) = sum_b d(b) * p[(o+b) mod 3]
i.e. the outcome distribution = (p circularly-correlated with d). Under BIAS d=q; under
BEST-RESPONDER d=BR(p). Sharper p  =>  outcome more diagnostic of b (and of the type).

DP (Bellman, finite horizon):
  state = belief; action = agent distribution p_t (compact: target move m in {0,1,2},
  sharpness s on a grid => p = (1-s)/3 *uniform-ish... see below). ~24 actions.
  V_t(b) = max_p [ r(b,p) + sum_o P(o|b,p) V_{t+1}(b'(o)) ],  V_T = 0.

We solve by PARTICLE / fitted value iteration is unnecessary for the q-posterior because we
can keep the EXACT continuous-belief update and discretize the *sufficient statistics* of the
belief for the value-function table. Concretely we exploit:
  - ROTATIONAL SYMMETRY over the 3 moves: WLOG store the q-Dirichlet posterior in a canonical
    orientation. The value only depends on the SHAPE of the belief (how concentrated / how the
    pseudo-count mass is distributed), not on which physical move is favored, because the agent
    can rotate its action to match. So the belief sufficient statistic is
       (pi, sorted-normalized-Dirichlet-mean m_(1)>=m_(2)>=m_(3), total concentration c=sum alpha).
  We discretize (pi, the simplex point sort(mhat), log c) on a grid and do grid value iteration
  with the EXACT one-step belief update + a nearest-grid lookup for V_{t+1}.

This is an APPROXIMATION (grid interpolation of V over belief sufficient stats) but the belief
DYNAMICS and rewards are computed EXACTLY per action; only the V_{t+1} read-back is interpolated.
We validate against the known limits (beta=0,1) and monotonicity.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import itertools, json, time
from math import lgamma

rng = np.random.default_rng(0)

N = 3
GAMMA_BR = 6.0
EPS = 1e-12

# ---------------------------------------------------------------------------
# core game primitives
# ---------------------------------------------------------------------------

def br_dist(p):
    """opponent best-response move distribution given agent action dist p (len-3).
    win(b) = p[(b-1)%3]; b ~ softmax(gamma*win)."""
    win = np.array([p[(b - 1) % N] for b in range(N)])
    z = GAMMA_BR * win
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def outcome_dist(p, d):
    """P(o | agent dist p, opponent move dist d).  o=(a-b)%3.
    P(o) = sum_{a,b} p[a] d[b] 1[(a-b)%3==o] = sum_b d[b] p[(o+b)%3]."""
    out = np.zeros(N)
    for o in range(N):
        s = 0.0
        for b in range(N):
            s += d[b] * p[(o + b) % N]
        out[o] = s
    return out


def reward_of(p, d):
    """expected per-round reward: +1 win(o=1), -1 loss(o=2), 0 tie(o=0)."""
    od = outcome_dist(p, d)
    return od[1] - od[2]


# ---------------------------------------------------------------------------
# action parametrization: target move m in {0,1,2}, sharpness s in grid.
#   p = s on the target move, (1-s)/2 on each other move.  s in [1/3, ~0.98].
# 8 sharpness levels x 3 target moves = 24 actions.
# ---------------------------------------------------------------------------
SHARP = np.array([1/3, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.975])

def action_dist(m, s):
    p = np.full(N, (1 - s) / 2)
    p[m] = s
    return p

def entropy(p):
    p = np.clip(p, EPS, 1)
    return -(p * np.log(p)).sum()

# ---------------------------------------------------------------------------
# Belief representation.
#   pi  : P(exploitable)
#   alpha : Dirichlet pseudo-counts over q (the bias), shape (3,)
# Conditional on best-responder there is nothing to estimate (BR is a deterministic
# function of the agent's own p), so the only continuous unknown is q under the bias branch.
#
# Belief update on observing o after playing p:
#   Let d_BR = BR(p).  predictive outcome under BR branch:  L_BR(o) = outcome_dist(p, d_BR)[o]
#   Under bias branch, q ~ Dir(alpha); the marginal predictive over o (integrating q) is
#       L_bias(o) = sum_b qbar[b] p[(o+b)%3],  with qbar = alpha/sum(alpha)  (since linear in q)
#   pi'  proportional to  pi * L_bias(o)   ;  (1-pi)' prop (1-pi)*L_BR(o)
#   Dirichlet update under the bias branch: the posterior over q after seeing o is a MIXTURE
#   over which opponent move b produced o, with responsibility
#       r(b) = qbar[b]*p[(o+b)%3] / L_bias(o)
#   We do the standard moment-matched / expected-count Dirichlet update (assumed-density
#   filtering): alpha[b] += r(b).  This is the exact expected sufficient statistic increment
#   and is the natural conjugate-style update when b is latent (a soft count). The sharpness of
#   p enters through r(b): uniform p => r(b)=qbar(b) (o uninformative => no info gain), sharp p
#   => r concentrates => fast q-resolution. This realizes the self-legibility coupling.
# ---------------------------------------------------------------------------

def br_predictive(p):
    return outcome_dist(p, br_dist(p))

def bias_predictive(p, alpha):
    qbar = alpha / alpha.sum()
    return outcome_dist(p, qbar)

def belief_update(pi, alpha, p, o):
    L_br = br_predictive(p)[o]
    qbar = alpha / alpha.sum()
    # responsibilities over b under bias branch
    rb = np.array([qbar[b] * p[(o + b) % N] for b in range(N)])
    L_bias = rb.sum()
    rb_norm = rb / (L_bias + EPS)
    # type posterior
    num = pi * L_bias
    den = num + (1 - pi) * L_br
    pi2 = num / (den + EPS)
    # Dirichlet expected-count update
    alpha2 = alpha + rb_norm
    Ltot = den  # P(o | belief, p)
    return pi2, alpha2, Ltot


def expected_reward(pi, alpha, p):
    """E[reward | belief, p] = pi*r_bias + (1-pi)*r_br."""
    qbar = alpha / alpha.sum()
    r_bias = reward_of(p, qbar)        # linear in q so qbar suffices
    r_br = reward_of(p, br_dist(p))
    return pi * r_bias + (1 - pi) * r_br


# ---------------------------------------------------------------------------
# Symmetry-reduced belief sufficient statistics for the value table.
#   We summarize (pi, alpha) by:
#     pi
#     conc = sum(alpha)              (concentration; how much evidence about q)
#     shape = sort(alpha/sum, desc)  (canonical orientation by rotation symmetry)
#   shape lives on the 2-simplex restricted to descending order. We grid it.
# Value lookup: nearest neighbor on the grid in (pi, log conc, shape) space.
#
# The KEY symmetry claim: V is invariant to a simultaneous cyclic relabeling of moves
# (rotating q's orientation AND the agent's available target moves rotates with it). So V
# depends only on the SORTED qbar shape + conc + pi. The optimal action's target move is then
# read relative to the canonical orientation and rotated back.
# ---------------------------------------------------------------------------

def canon_rotation(alpha):
    """Cyclic-rotation canonicalization. The move space is CYCLIC (m beats m-1), so the only
    symmetry that preserves the 'beats' structure is the 3 cyclic rotations -- NOT arbitrary
    permutations (a reflection would swap 'the move I beat' with 'the move that beats me').
    Canonical frame: rotate so the opponent's MOST-LIKELY move sits at index 0. Returns the
    rotation r (real index of the max) and the rotated alpha (max-first, other two in cyclic order)."""
    r = int(np.argmax(alpha))
    rot = np.array([alpha[(r + k) % N] for k in range(N)])
    return r, rot


def belief_feats(pi, alpha):
    conc = alpha.sum()
    _, rot = canon_rotation(alpha)
    shape = rot / conc          # max-first, remaining two in fixed cyclic order
    return pi, conc, shape


# ---------- grid value iteration over the reduced belief ----------
class DPGrid:
    def __init__(self, T, beta,
                 n_pi=17, conc_grid=None, n_shape=None, verbose=False):
        self.T = T
        self.beta = beta
        self.verbose = verbose
        # pi grid
        self.pis = np.linspace(0, 1, n_pi)
        # concentration grid (log spaced); prior conc=1.5 (Dir(.5,.5,.5)); grows by <=1/round
        if conc_grid is None:
            conc_grid = np.array([1.5, 3, 5, 8, 12, 18, 26, 36])
            conc_grid = conc_grid[conc_grid <= 1.5 + T]
            conc_grid = np.append(conc_grid, 1.5 + T)
        self.concs = np.unique(conc_grid)
        # shape grid: MAX-FIRST points on 2-simplex (cyclic-canonical). Component 0 is the
        # opponent's most-likely move; components 1,2 are the next two in fixed cyclic order and
        # may be in EITHER magnitude order (reflections are distinct -- beats is directional).
        # Enumerate (i,j,k)/G with i>=j and i>=k (i is the max), NOT requiring j>=k.
        G = 12 if n_shape is None else n_shape
        shapes = []
        for i in range(G + 1):
            for j in range(G + 1):
                k = G - i - j
                if k < 0:
                    continue
                if i >= j and i >= k:           # component 0 is the (weak) max
                    shapes.append(np.array([i, j, k]) / G)
        self.shapes = np.array(shapes)              # (Ns,3) max-first, sum=1
        # build flat grid of belief-feature nodes
        self.nodes = []
        for pi in self.pis:
            for c in self.concs:
                for sh in self.shapes:
                    self.nodes.append((pi, c, sh))
        self.npi = len(self.pis); self.nc = len(self.concs); self.ns = len(self.shapes)
        # index helpers
        self._shape_arr = self.shapes
        self._logc = np.log(self.concs)

    # nearest-node lookup -> returns value from a V table indexed [ipi,ic,ish]
    def _nn_value(self, Vtab, pi, conc, shape):
        ipi = np.argmin(np.abs(self.pis - pi))
        ic = np.argmin(np.abs(self._logc - np.log(max(conc, EPS))))
        ish = np.argmin(((self._shape_arr - shape) ** 2).sum(1))
        return Vtab[ipi, ic, ish]

    def _alpha_from(self, conc, shape):
        return shape * conc  # canonical orientation: descending

    def solve(self):
        """backward induction. V[t] table over (pi,conc,shape). returns list of tables and
        the greedy policy info at each node/time."""
        T = self.T
        Vtabs = [None] * (T + 1)
        Vtabs[T] = np.zeros((self.npi, self.nc, self.ns))
        # precompute, for each (pi,alpha) node and each action, the reward and the 3 successor
        # feats + outcome probs. Reward & successors do NOT depend on t, only V_{t+1} read does.
        # We store per-node per-action: r, [(o_prob, succ_pi, succ_conc, succ_shape)].
        if self.verbose:
            print(f"  building transition cache (beta={self.beta}) ...", flush=True)
        # action set
        self.actions = [(m, s) for s in range(len(SHARP)) for m in range(N)]
        # but by symmetry the target move m should be chosen relative to canonical orientation.
        # canonical alpha is descending (move0 most likely under bias). The natural exploit is
        # m=0. We still allow all m to let the DP discover sense-vs-exploit, but rewards/updates
        # are computed in canonical frame.
        cache = {}
        for ipi, pi in enumerate(self.pis):
            for ic, c in enumerate(self.concs):
                for ish, sh in enumerate(self.shapes):
                    alpha = self._alpha_from(c, sh)
                    peract = []
                    for (m, si) in self.actions:
                        p = action_dist(m, SHARP[si])
                        r = expected_reward(pi, alpha, p)
                        succ = []
                        for o in range(N):
                            pi2, alpha2, Lo = belief_update(pi, alpha, p, o)
                            if Lo < 1e-9:
                                continue
                            f_pi, f_c, f_sh = belief_feats(pi2, alpha2)
                            succ.append((o, Lo, f_pi, f_c, f_sh))
                        peract.append((r, succ))
                    cache[(ipi, ic, ish)] = peract
        self.cache = cache
        # backward induction
        self.policy = [None] * T
        for t in range(T - 1, -1, -1):
            Vt = np.zeros((self.npi, self.nc, self.ns))
            pol = np.zeros((self.npi, self.nc, self.ns), dtype=int)
            Vnext = Vtabs[t + 1]
            for ipi in range(self.npi):
                for ic in range(self.nc):
                    for ish in range(self.ns):
                        peract = cache[(ipi, ic, ish)]
                        best = -1e18; besta = 0
                        for ai, (r, succ) in enumerate(peract):
                            ev = r
                            for (o, Lo, f_pi, f_c, f_sh) in succ:
                                ev += Lo * self._nn_value(Vnext, f_pi, f_c, f_sh)
                            if ev > best:
                                best = ev; besta = ai
                        Vt[ipi, ic, ish] = best
                        pol[ipi, ic, ish] = besta
            Vtabs[t] = Vt
            self.policy[t] = pol
            if self.verbose and (t % 5 == 0 or t == T - 1):
                print(f"  t={t} done, V0_prior≈{self._prior_value(Vtabs[t]):.4f}", flush=True)
        self.Vtabs = Vtabs
        return Vtabs

    def _prior_node(self):
        """index of the prior belief: pi=1-beta?? NO -- pi is P(exploitable). prior pi=1-beta.
        alpha = Dir(.5,.5,.5) prior -> conc=1.5, shape uniform=(1/3,1/3,1/3)."""
        prior_pi = 1 - self.beta
        ipi = np.argmin(np.abs(self.pis - prior_pi))
        ic = np.argmin(np.abs(self.concs - 1.5))
        unif = np.array([1/3, 1/3, 1/3])
        ish = np.argmin(((self.shapes - unif) ** 2).sum(1))
        return ipi, ic, ish

    def _prior_value(self, Vtab):
        ipi, ic, ish = self._prior_node()
        return Vtab[ipi, ic, ish]

    # ---- greedy action at an arbitrary (continuous) belief at time t ----
    def greedy_action(self, t, pi, alpha):
        """returns (m_relative, sharpness_index, p_in_canonical_frame). Uses cache-free eval."""
        Vnext = self.Vtabs[t + 1]
        best = -1e18; besta = (0, 0)
        for (m, si) in self.actions:
            p = action_dist(m, SHARP[si])
            r = expected_reward(pi, alpha, p)
            ev = r
            for o in range(N):
                pi2, alpha2, Lo = belief_update(pi, alpha, p, o)
                if Lo < 1e-9:
                    continue
                f_pi, f_c, f_sh = belief_feats(pi2, alpha2)
                ev += Lo * self._nn_value(Vnext, f_pi, f_c, f_sh)
            if ev > best:
                best = ev; besta = (m, si)
        return besta, best

    def myopic_action(self, pi, alpha):
        """maximize immediate expected reward only (no lookahead)."""
        best = -1e18; besta = (0, 0)
        for (m, si) in self.actions:
            p = action_dist(m, SHARP[si])
            r = expected_reward(pi, alpha, p)
            if r > best:
                best = r; besta = (m, si)
        return besta, best


# ---------------------------------------------------------------------------
# SIMULATION under a solved policy (uses the EXACT belief; canonical-frame action rotated
# to track the true argmax of the agent's q-posterior).
#   In canonical frame the agent's belief is rotated so its q-mean is descending. The action's
#   target move m is in that canonical frame; we map back to the real move by the inverse of the
#   permutation that sorted qbar. For exploitation what matters is that the agent targets the move
#   that BEATS the opponent's most likely move. We track the real belief (pi, alpha) un-rotated
#   and, to query the policy, rotate to canonical, get m_canon, then rotate the action back.
# ---------------------------------------------------------------------------

def to_canonical(alpha):
    """cyclic rotation r mapping canonical index -> real index: real = (r + canon) % N,
    where r = argmax(alpha). So canonical alpha = [alpha[r], alpha[r+1], alpha[r+2]]."""
    r = int(np.argmax(alpha))
    return r

def simulate(dp, beta, n_games=4000, mode="optimal", seed=1):
    """Monte-Carlo evaluate a policy. Returns per-round arrays of entropy and reward, plus
    per-game type/bias info."""
    r = np.random.default_rng(seed)
    T = dp.T
    ent = np.zeros((n_games, T)); rew = np.zeros((n_games, T))
    is_bias = np.zeros(n_games, dtype=bool)
    bias_maxq = np.zeros(n_games)            # strength of bias (max q component)
    realized_pay = np.zeros(n_games)
    for g in range(n_games):
        exploitable = (r.random() > beta)    # prob 1-beta exploitable bias
        if exploitable:
            q = r.dirichlet([0.5, 0.5, 0.5])
        is_bias[g] = exploitable
        bias_maxq[g] = q.max() if exploitable else np.nan
        # belief
        pi = 1 - beta
        alpha = np.array([0.5, 0.5, 0.5])
        for t in range(T):
            rot = to_canonical(alpha)
            alpha_canon = np.array([alpha[(rot + k) % N] for k in range(N)])
            if mode == "optimal":
                (m_c, si), _ = dp.greedy_action(t, pi, alpha_canon)
            elif mode == "myopic":
                (m_c, si), _ = dp.myopic_action(pi, alpha_canon)
            elif mode == "uniform":
                m_c, si = 0, 0
            # map canonical target move back to real frame by the SAME cyclic rotation
            m_real = (rot + m_c) % N
            p = action_dist(m_real, SHARP[si])
            ent[g, t] = entropy(p)
            # sample agent action
            a = r.choice(N, p=p)
            # opponent move
            if exploitable:
                d = q
            else:
                d = br_dist(p)               # BR sees p
            b = r.choice(N, p=d)
            o = (a - b) % N
            rr = (1 if o == 1 else (-1 if o == 2 else 0))
            rew[g, t] = rr
            # belief update with REAL p and observed o
            pi, alpha, _ = belief_update(pi, alpha, p, o)
        realized_pay[g] = rew[g].mean()
    return dict(ent=ent, rew=rew, is_bias=is_bias, bias_maxq=bias_maxq,
                realized_pay=realized_pay)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=25)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dp_results.json"))
    args = ap.parse_args()
    print("(self-test placeholder; see dp_run.py for the full sweep)")
