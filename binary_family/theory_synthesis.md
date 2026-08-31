# Synthesis: theory_fable.md vs theory_sol.md (2026-08-10, window 4)

Two decorrelated derivations of the same charge. Provenance: theory_sol.md =
gpt-5.6-sol, high reasoning effort; theory_fable.md = the Fable main session
(subagent died 4x on API overload), written before reading theory_sol.md.

## Consensus skeleton (independently derived by both — treat as solid)

1. **Master decomposition** (identical in both): conditional expected increment
   $= \mathrm{KL}(P^*\|R^-) - \mathrm{KL}(P^*\|R^+)$; curve = integrated KL-gap + martingale.
2. **Court-correct barriers**: if $P^*$ equals a reader, the other side's
   exponential is a nonnegative martingale ⇒ the false account never wins:
   convergence to a finite limit or divergence toward the true side only;
   expected trajectory monotone. (Fable adds the quantitative Ville tail
   $P(\text{wrong-way excursion} \ge b) \le e^{-b}$.)
3. **The regime dial is cumulative channel information** $\sum_k D_k$:
   finite ⇒ plateau; $\asymp \log n \Rightarrow \pm\log n$; $\asymp n^\alpha \Rightarrow \pm n^\alpha$; $\asymp n$ ⇒ linear.
   Both notes constructed the SAME clocked-Bernoulli family ($\delta_n \sim n^{-(1-\alpha)/2}$)
   independently — decorrelated convergence on the counterexample.
4. **Slope** = ergodic mean KL gap (Birkhoff / martingale SLLN); CLT around
   drift; exponential plateau-approach under filter contraction; *no* rate from
   existence of a limit alone.
5. **Same honest gap flagged by both**: uniform filter stability for the
   *controlled/closed-loop* chain — positivity can be destroyed by reducibility,
   deterministic observations, or the policy avoiding mixing actions.

## Corrections across the notes

- **Sol corrects Fable (accepted).** Fable's Theorem C glossed "nothing sublinear
  survives stationarity." Sol's Jordan-block example (§1.4): a *stationary,
  finite-state, fixed-parameter* world (reducible, boundary-initialized readers,
  defective transition matrix) with $\Lambda^x = \log(1 + nb/\rho) \sim \log n$. Arithmetic
  checked; correct. So the exclusion needs *irreducibility + uniform positivity*,
  not stationarity — and even then the closed loop can dodge mixing (point 5).
  Fable's note now carries a correction pointer.
- **Fable adds what Sol omits.** (i) Wald/SPRT identification: the
  depth-over-discrimination law $(\tau^*)$ is Wald's expected-sample-size formula,
  inheriting its fluctuation band and overshoot corrections. (ii) The explicit
  rate×duration mechanism for the non-monotone plateau (fig 3b). Sol instead
  proves the *sharper* phenomenon: the expected contribution $F(q)$ can change
  SIGN in $q$ (checked: $F(0) \approx +0.048$, $F \approx -0.04t^2 < 0$ near $q = \tfrac12$) —
  a perturbation can be endorsed by a sharp channel and refuted by a blurry one.

## Sol's unique results worth importing into the program

1. **Interior-prior boundedness theorem** (§3.2): if both readers are mixtures
   over the SAME finite latent set with everywhere-positive priors, the total
   likelihood ratio is bounded for all time between the min and max prior odds
   ⇒ BOTH channel curves converge a.s. to finite limits, with plateau magnitude
   bounded by log prior odds. The rigorous general form of "interior ⇒ plateau."
   With the crucial caveat: this concerns *Bayesian uncertainty over a fixed
   latent identity* — an interior point of a policy simplex that is a physical
   randomized controller is NOT covered. (Formalizes the E5 mixture-as-belief
   vs mixture-as-randomization distinction.)
2. **The local invariant** (§2.2): for small perturbations $\epsilon v$,
   $\mathbb E[\Lambda^c_\infty] = \pm\tfrac{\epsilon^2}{2}\mathcal I_c(v) + o(\epsilon^2)$ — a channel-restricted
   **Fisher information** of the path law in direction $v$, with the derivative
   given by the score propagated through the filter cocycle $J_k$. This is the
   natural local geometry for $\kappa$: the courts measure a per-channel Fisher
   quadratic form. Should eventually enter proposal §4.
3. **LIL boundary case**: when $P^*$ sits exactly between the readers, the curve
   is a $\sqrt{2n\log\log n}$-envelope random walk with infinitely many sign
   changes — the zero-drift knife-edge between the regimes.
4. **A.s.-vs-expectation divergence** (§1.6): $\Lambda \to 0$ a.s. while $\mathbb E\Lambda \to \infty$
   (Borel–Cantelli spikes) — the extreme form of fig 1's "means lie"; read
   medians/paths, never bare ensemble means.
5. **Zero-support explosions**: a single token outside a reader's support is an
   instant infinite verdict — measurement protocols must clip or smooth supports.

## Verdict

The two classifications agree wherever they overlap; each contributed one
correction or sharpening the other missed; no unresolved disagreement remains.
The general theory is: **shapes are exactly the shapes of cumulative channel
information; court-correct laws add one-sided barriers; interiors over shared
latents bound plateaus by prior odds; small perturbations are measured by
channel Fisher information.** Next consumers: proposal §4 (import 1, 2), the
measurement protocol (4, 5), and the $(\tau^*)$ discussion (Wald form).
