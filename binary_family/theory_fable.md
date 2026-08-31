# Two-court renormalization curves: shapes, limits, conditions

*Fable-side note. Provenance: written by the main session (window 4) after four
subagent attempts died on API-overload; composed WITHOUT reading theory_sol.md
beyond its notation preamble. Claims labeled **proved / sketched / conjectured**.*

Notation: per-token increment $\Delta\Lambda_\tau = \log R^+(y_\tau|h_\tau) - \log R^-(y_\tau|h_\tau)$,
channel-restricted sums $\Lambda^c_s$. All expectations under the true law $P^*$.

## 0. The master decomposition (proved)

Conditionally on history,
$$\mathbb E^*[\Delta\Lambda_\tau \mid h_\tau] \;=\; \mathrm{KL}(P^*_\tau \,\|\, R^-_\tau) - \mathrm{KL}(P^*_\tau \,\|\, R^+_\tau) \;=:\; g_\tau,$$
the **KL gap** — how much closer the took-account sits to the truth than the
not-account, at that token. Hence the Doob decomposition
$$\Lambda^c_s = \underbrace{\textstyle\sum_{\tau \le s,\, \tau \in c} g_\tau}_{A^c_s \ \text{(predictable)}} + \ M^c_s, \qquad M^c \ \text{a } P^*\text{-martingale},$$
with $\mathrm{Var}(\Delta M_\tau \mid h_\tau) = \mathrm{Var}^*_\tau(\Delta\Lambda_\tau)$. **Every question about
shapes reduces to the trajectory of the gap sequence $(g_\tau)$ plus a martingale
fluctuation of scale $\sqrt{\sum \mathrm{Var}}$.** Note $g_\tau$ is a *difference* of KLs: it has
no sign in general — it is nonnegative for all $\tau$ iff the truth is (weakly)
on the took side pointwise.

## 1. One-sided barriers when $P^*$ coincides with a reader (proved)

If $P^* = R^+$ (penetrated $do$): $L_s = e^{-\Lambda_s}$ (total, both channels) is a
nonnegative $P^*$-martingale with mean 1. By Ville's inequality,
$$P^*\Big(\inf_s \Lambda_s \le -b\Big) \le e^{-b} \qquad (b > 0),$$
and $\Lambda_s$ converges a.s. in $(-\infty, +\infty]$: **the false account can never win;
the running total can never drift to $-\infty$, and its worst excursion against the
truth has an exponential tail.** Symmetrically, if $P^* = R^-$ (a non-penetrating
injection), $e^{+\Lambda_s}$ is the martingale: $P^*(\sup_s \Lambda_s \ge b) \le e^{-b}$ — *a false
"it took" can never accumulate unbounded evidence*. These are the anytime-valid
guarantees of the instrument, and they are completely model-free.

When $P^*$ equals neither reader (the generic injection: the $a$-channel truth
follows the adopted state, the $x$-channel truth the untouched belief), no global
martingale exists, but each channel separately obeys §0, and the barriers apply
per channel whenever $P^*$'s conditionals agree with one reader's on that channel.
This is why opposite-signed courts are possible and generic (proved by the
binary family's corrected row: $\Lambda^a \uparrow$, $\Lambda^x \downarrow$; the unsplit total can cancel).

## 2. The shape classification

**Theorem A (dichotomy under merging/singularity; proved for $P^* = R^+$, sketched
in general).** With $P^* = R^+$, define the predictable Hellinger-type sum
$H_s = \sum_{\tau\le s} h^2(R^+_\tau, R^-_\tau)$ (squared conditional Hellinger distances). On
$\{H_\infty < \infty\}$, $\Lambda_s$ converges a.s. to a finite limit; on $\{H_\infty = \infty\}$,
$\Lambda_s \to +\infty$ a.s. This is the adapted-sequence version of Kakutani's dichotomy
(via predictable Hellinger processes, Jacod–Shiryaev-style; for independent
tokens it is literally Kakutani — that case proved, the adapted general case
sketched with citation).

**Theorem B (all sublinear shapes are realizable; proved by construction).**
Independent tokens; $R^+$ predicts $\mathrm{Ber}(\tfrac12 + \varepsilon_\tau)$, $R^-$ predicts $\mathrm{Ber}(\tfrac12)$,
truth $= R^+$, with $\varepsilon_\tau = \tau^{-\alpha}$:
$$g_\tau \asymp \varepsilon_\tau^2 \;\Longrightarrow\; \mathbb E[\Lambda_s] \asymp \begin{cases} s^{1-2\alpha} & 0 < \alpha < \tfrac12 \\ \log s & \alpha = \tfrac12 \\ \text{finite} & \alpha > \tfrac12,\end{cases}$$
and the martingale part is $O(\sqrt{A_s})$, subdominant. So **every power $s^\gamma$,
$\gamma \in (0,1)$, and $\log s$ occur** — but only through *decaying, non-summable*
disagreement (a reader-merging rate that is polynomial rather than exponential).

**Theorem C (plateau-or-linear completeness in the mixing regime; sketched — SEE
CORRECTION).** *Correction (post-comparison, 2026-08-10): the gloss "nothing
sublinear survives stationarity" is wrong as stated — theory_sol.md §1.4 exhibits
a stationary finite-state reducible model (Jordan block, boundary-initialized
readers) with $\Lambda^x \sim \log n$. The exclusion requires irreducibility + uniform
positivity, not stationarity. Details in theory_synthesis.md.*
If the closed loop is a uniformly ergodic finite-state process, the channel noise
is fixed and nondegenerate, and the readers are exact Bayes copies, then reader
disagreement decays exponentially whenever the initializations are mutually
absolutely continuous (world side: filter stability / Dobrushin contraction of
the joint chain; identity side: Blackwell–Dubins merging at the rate of the
posterior's exponential concentration), and is bounded below in Cesàro mean when
they are mutually singular with distinct stationary conduct. Hence exactly two
asymptotic shapes exist there: **finite plateau** (exponentially approached) or
**linear drift**. Nothing sublinear survives stationarity — Theorem B's shapes
require nonstationary or polynomially-mixing structure (null-recurrent worlds,
shrinking discrimination schedules). *Gap to close: a uniform filter-stability
rate for the controlled/closed-loop chain; standard for primitive kernels with
positive observation noise, cited not re-proved.*

**Sign changes and non-monotone expectations (proved by construction).** $g_\tau$
is a KL difference and can change sign along a trajectory: e.g. a perturbation
that happens to sit nearer the world's current state is *endorsed* ($g^x > 0$)
until the world moves, then *objected to* ($g^x < 0$). Expected trajectories can
be non-monotone in $s$, and the running total can cross zero; only the §1
barriers constrain excursions. In expectation, under $P^* = R^+$, $\mathbb E[\Lambda_s]$ is
nondecreasing in $s$ (each $g_\tau \ge 0$ pointwise since the KL gap with $P^*=R^+$
equals $\mathrm{KL}(R^+\|R^-) \ge 0$) — so under *penetration* the total court is
monotone in expectation; non-monotone means non-penetration or mixed truth.

## 3. Limits, slopes, fluctuations (Q2)

- **Plateau value.** When finite, $\mathbb E[\Lambda^c_\infty] = \sum_\tau \mathbb E[g_\tau]$: the *integrated
  channel information* about took-vs-not — everything channel $c$ ever learns
  about the perturbation. For $P^* = R^+$ this is $\sum \mathbb E\,\mathrm{KL}(R^+_\tau\|R^-_\tau)$, an
  integral of (instantaneous discrimination) along the (merging trajectory):
  with per-token rate $r$ and merging time $T$, plateau $\approx r \cdot T$. **The observed
  non-monotonicity in channel informativeness is generic**: any parameter that
  raises $r$ while shortening $T$ faster produces an interior maximum. (Proved as
  a mechanism; the location of the peak is model-specific.)
- **Slope under drift.** $\lim \Lambda^c_s / s = \bar g^c$, the ergodic mean KL gap — for
  vertex-vertex identity perturbations, the KL *divergence rate* between the two
  readers' stationary predictive conduct on channel $c$ (binary family:
  $\mathrm{KL}(\mathrm{Ber}(\beta_2)\|\mathrm{Ber}(\beta_1))$). A.s., not just in mean, by the martingale SLLN
  ($\sum \mathrm{Var}/s^2 < \infty$). (Proved under bounded log-likelihood ratios.)
- **Fluctuations.** $\mathrm{Var}(\Lambda^c_s) = \mathbb E\sum \mathrm{Var}^*_\tau(\Delta\Lambda_\tau) + \mathrm{Var}(A^c_s)$; with bounded
  LRs the per-token variance is comparable to the per-token KL (Bernstein
  regime), so plateau fluctuation $\asymp \sqrt{\mathbb E[\Lambda_\infty]}$ and drift obeys a CLT with
  variance rate $= $ the stationary variance of $\Delta\Lambda$. Wald's identities give the
  expected crossing time of a level $m_0$: $\mathbb E[\tau_{m_0}] \approx m_0 / \bar g$ — **the $(\tau^*)$
  depth-over-discrimination law is Wald's SPRT sample-size approximation**, which
  also supplies its correction terms (overshoot, fluctuation band $O(\sqrt{\tau^*})$).

## 4. Conditions separating the regimes (Q3)

| Regime | Sharp(est) condition |
|---|---|
| finite plateau | $\sum_\tau h^2(R^+_\tau, R^-_\tau) < \infty$ under $P^*$ (Thm A); sufficient: mutually a.c. initializations + exponentially stable filter (mixing world OR informative channel — *either* route; both is faster) |
| linear drift | initializations mutually singular on the tail (vertex–vertex) + distinct stationary conduct; slope $=\bar g$ |
| sublinear $s^\gamma$, $\log s$ | vanishing non-summable gaps: polynomial reader-merging (excluded by uniform ergodicity; realizable via decaying schedules / null-recurrent structure) |
| identically zero | channel-degenerate predictions (belief-blind policy for $\Lambda^a$; $q=\tfrac12$ for $\Lambda^x$) or empty action-shadow |

Interior-vs-vertex is exactly the a.c.-vs-singular line; channel informativeness
moves *rates and plateau values*, never the regime (the regime is decided by
absolute continuity and ergodicity alone). (Proved/sketched as marked above.)

## 5. Universal vs family-specific (Q4)

**Universal** (any model, two exact-Bayes readers, shared stream): §0's
decomposition; §1's Ville barriers; Thm A's dichotomy; plateau = integrated
channel information; Wald form of crossing times; per-channel sign independence.
**Generic but not universal**: plateau-or-linear completeness (needs uniform
ergodicity); non-monotone plateau in informativeness (needs rate×duration
tradeoff, which fixed-gap families all have). **Family-specific**: every explicit
rate (the $(1-c_w)(1-2\lambda)$ contraction, $(\tau^*)$'s constants, the leak $\propto c_w^2$).

## 6. Machinery used (Q5)

Doob decomposition & martingale convergence (proved uses); Ville's inequality
(proved); Kakutani dichotomy, adapted version via predictable Hellinger
processes — Jacod & Shiryaev, *Limit Theorems for Stochastic Processes*, ch. IV–V
(sketched); Blackwell–Dubins merging (cited for identity-side a.c. merging);
filter stability / Dobrushin contraction (cited for the world side; the closed-
loop uniform-rate statement is the one honest gap); Wald identities / SPRT
theory (proved uses, standard); martingale SLLN and CLT (standard).

**Open lemmas:** (L1) uniform exponential filter stability for the closed-loop
controlled chain under mutual a.c. initialization — needed to make Theorem C
fully proved; (L2) the adapted Kakutani dichotomy in the exact generality of §2
Thm A; (L3) sharp characterization of when the *unsplit* total $\Lambda$ can cross
zero infinitely often under $P^*$ equal to neither reader.
