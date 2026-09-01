# DESIGN — changeling v3: whitebox of the identity mechanism (hypothesis space + discriminating tests)

*2026-09-01. Target: the v1.2 post net (single-layer GRU d=256; verified
manual recompute harness, max diff 1.5e-6). Facts to explain (Iterations
3-7): λ linearly readable (R² .91) but high-dim redundant (24 INLP dirs →
R² .31) and causally inert (erasure/24-nat swaps change nothing; heal in
1-2 rounds); claims gated σ(±.28λ+1.3); only the state-interactive flag
INPUT ever swapped identity.*

## Hypothesis space (preregistered before testing)

- **H1 low-dim integrator register + gate readout** — REFUTED (Iter. 7).
- **H2/H5 gain coding (bilinear):** no λ variable; identity stored as the
  per-channel AMPLITUDE of the context-dependent plan-tilt pattern in h
  (m_c · tilt_pattern_c(context)); patterns rotate with context so linear
  probes see the code only diffusely (the shadow), while the heads read it
  exactly. Memory = the gains persist through the recurrence. H5: the two
  channel gains are independent coordinates.
- **H3 saturation/attractor code** — disfavored a priori (a latched sign
  pattern is linearly decodable; INLP would have caught it; erasure-to-mean
  did nothing, not even transiently).
- **H4 windowed re-derivation, no persistent identity state:** the gate
  saturates by λ~8 nats, so behavior needs only a few rounds of "were my
  intentions echoed?"; identity is recomputed each round from recent
  evidence. The memory is the token stream itself — which no intervention
  touched. Predicts all Iter.-7 results (erasure-proof, 1-2-round healing =
  window refill, shadow = round × sign byproduct).
- **H6 expression funnel:** whichever carrier, its expression converges on
  the gate machinery the flag input drives.

## Discriminating experiments

- **E1 evidence diet** (teacher-forced continuations from t=16 after a
  natural closed-loop prefix): (a) NEUTRAL tokens (per round, per channel,
  argmin |log π − log p̄| — zero identity evidence): integrator (H2) HOLDS
  claims ≥ 12 rounds; window re-deriver (H4) RELAXES to the claim-both
  default σ(1.3)≈.79 within ~its window. (b) p̄-fed (both channels look
  genuine): counter-evidence on the self channel — measures the un-learning
  rate, behaviorally. (c) π-fed (both look mine): other-claim should rise.
  (b)+(c) read the update rule out behaviorally — what the shadow-based
  counterfactual test could not.
- **E2 memory-vs-expression transplant bisect** at t=16 between matched
  pairs (same goal, opposite identity, nearest beliefs): swap (a) full h —
  positive control, should persist and erode only at the evidence rate
  (~.6 nats/round against the transplanted deficit); (b) the 12-dim HEAD
  ROWSPACE component only (behavior factors through W_u⊕W_v, so this must
  swap the CURRENT round's behavior; persistence ⇔ rowspace is also the
  memory); (c) the complement only (inert this round; later re-expression
  ⇔ memory lives outside the rowspace).
- **E3 window sufficiency:** regress per-round claims on the last-W exact
  evidence terms, W ∈ {1,2,4,8,16}, vs on cumulative λ. H4: small W matches
  the full history. H2: cumulative wins.
- **E4 shadow decomposition:** R² of λ from (round, claims, round×claim
  asymmetry) — how much of the .91 decode is time × expressed sign.

Falsifier discipline: if E1-neutral holds AND E2(b) persists, the memory is
the expressed gains (H2, memory=expression); if E1-neutral relaxes AND E3
saturates at small W, H4 wins; full-h swap eroding at ~evidence rate vs
~window rate separates integrator-somewhere from H4 globally. Mixed
outcomes localize the split (e.g., gains persistent but window-topped-up).

## Files

whitebox_lambda.py → results/rnn_whitebox_lambda.json,
figs/whitebox_diet.png, figs/whitebox_transplant.png.

---

## Measured outcomes (2026-09-01, same session)

**E1 (diets).** NEUTRAL: self-claim holds .993→.988 over 16 evidence-free
rounds (other-claim drifts .21→.34) — a persistent integrator exists; H4
(window re-derivation) is dead. P̄-DIET (counter-evidence): other-claim
rises .21→.81 while self-claim falls only .99→.92 — exactly the shared-λ
sigmoid gates at different operating points (self-gate deep in saturation,
other-gate at its linear range); implied internal λ falls ~10 nats/16
rounds ≈ the Bayes evidence rate — the write rule confirmed behaviorally.
π-DIET: flat (the withdrawn v-head ≈ p̄, so the diet is uninformative
there) — consistent.
**E2 (transplants; note: measured one recurrent step after the swap).**
FULL-state swap between matched opposite-identity pairs: claims jump toward
the donor identity (self .99→.71, other .21→.88) and erode back over ~15+
rounds (evidence-rate scale); occupancy cost real (.761→.652). ROWSPACE-only
(the 12 dims the heads read) and COMPLEMENT-only swaps both produce only a
transient other-claim bump (healed by ~8 rounds) and never move the
self-claim — EITHER half regenerates the whole from the other. Holographic
storage signature: no subspace carries the memory exclusively; only the
full state does. (Retroactively explains Iter.-7: 24-dim erasure left 232
dims of redundant copy.)
**E3.** Other-claim R² from last-W evidence: .001/.046/.13/.31/.57 for
W=1/2/4/8/16 vs cumulative-λ .59 — behavior tracks the long integral, not
a short window. H4 dead twice.
**E4.** The λ-shadow is fully explained by round × expressed claim
asymmetry: R² .914 ≈ the .91 ridge decode. The readable λ is a computed
display of (time, expression), not the store.

**Hypothesis scorecard:** H1 dead (Iter. 7). H3 dead. H4 dead (E1+E3).
H5 (independent per-channel gains) down-weighted — the p̄-diet coupling
(other-claim rising as self-evidence falls) is exactly shared-λ gating.
Verdict: **a single persistent identity integrator, updated at ~the Bayes
rate, stored holographically/redundantly across the whole state (no
carrier subspace), expressed each round through biased sigmoid gates
σ(±.28λ + 1.25), with the linear 'λ code' a mere shadow (time ×
expression).** Remaining open: the micro-format of the distributed store
(unit products / gate configurations) — DAS + gate-freezing surgery with
the verified harness; and H6 (does the flag input feed the same funnel).
Figures: figs/whitebox_diet.png (diet curves + transplant curves).

## Iteration 8 (2026-09-01; format.py + distill.py) — THE FORMAT, FOUND

"Holographic" was an artifact of intervening only along regression-fitted
directions. The specific storage format:

- **F1/F2**: per-round λ decoders reach R² .93 and their directions ROTATE
  (adjacent-round cos .64; 16 rounds apart .41), but even per-round-fitted
  and rotating-clamped swaps stay inert — rotation is not the answer.
- **F3 (survival)**: matched-norm perturbations through paired
  identical-token rollouts: the full twin-difference decays ×.88/round
  (memory-like) while pooled, per-round, AND RANDOM directions all collapse
  identically (×.37 after one round) — the dynamics treat every decoder
  direction exactly as noise; 'healing' was always just contraction.
- **F4**: no latched-unit committee (2/256 units with z̄>.9); the identity
  difference is spread over ~62 effective units.
- **DISTILLATION (the closure)**: propagate the matched-twin difference 4
  rounds under paired identical tokens — the dynamics discard the noise
  components (67.8% of norm survives) and what remains is, after sign
  alignment, ONE GLOBAL DIRECTION: PCA top-1 = 83.1% of variance,
  participation ratio 1.4. **Transplanting the 1-dim coefficient along
  this direction reproduces the full 256-dim state swap almost exactly**
  (self .79 vs .69, other .89 vs .93 at t+0; identical slow evidence-rate
  erosion and occupancy cost .648 vs .636; PCA k=2/k=8 add nothing).
- The direction is **near-orthogonal to every readout**: cos .06 with the
  matched-round λ decoder — the store and its per-round emission (what all
  probes read; = time × expression per E4) are separate, almost
  perpendicular objects. Ridge provably picked the emission (it carries
  the λ-covariance); the store carries the causal power.
- Flag pathway (H6): cos(memory dir, flag write-direction) = .14 — the
  flag does not write the register additively; it moves it through the
  recurrent dynamics (matching the measured 4-5-round flag-flip
  conversion).

**The verified format:** identity is a 1-DIMENSIONAL REGISTER after all —
the coefficient along a single, global, context-independent direction m̂,
spread over ~60 units, (i) protected by dynamics that contract everything
off-axis ×.37/round, (ii) plausibly re-polarized under partial writes
(explains half-swap healing; bistability test pending), (iii) invisible to
regression readouts, which latch onto the near-orthogonal per-round
emission, and (iv) findable only by letting the network's own dynamics
filter a genuine memory difference. Loose ends: decode λ from h·m̂ alone
(magnitude vs sign content); bistability/basin test along m̂; then the
synthetic program + verified-circuit diagram (all components now in hand:
evidence terms → m̂-register → σ-gates → heads).

## Iteration 9 (2026-09-01; efference.py) — the write mechanism, read on the register

Architecture note for the record: single-layer GRU — ONE state h_t; heads
read h_t directly; h_{t+1} = GRU(input_t+1, h_t). The efference copy is
structurally free (the state that generated the intention IS the next
recurrent input; the intended DISTRIBUTION, not the sample, is what
persists — the sample returns as input).

- **Register content:** rho = h·m̂ decodes λ at only R² .376 (late sign
  accuracy .72). The register is coarse/sign-like — the causal store, not
  the graded readout; the fine-grained readable λ (R² .93) is assembled
  downstream (time × gate expression), matching E4 and the bistability
  picture.
- **The comparator is a PLAN TEMPLATE, not an efference copy.** On
  channels the net has withdrawn from (its own head ≈ p̄ there, so a true
  efference comparator would generate NO authorship evidence — flat
  counterfactual profiles), the measured register-increment profiles are
  tilt-shaped, matching the template hypothesis (u: R² .39, slope .18;
  v: R² .15) while the efference regression carries no signal (its
  hypothesis has ~no variance there, and the observed profile is not
  flat). Authorship is detected TELEOLOGICALLY — "is this channel trying
  to do what I would try to do?" — not by self-consistency with the
  current policy. Design rationale (why training found this): a template
  comparator preserves a recovery route (a withdrawn channel that starts
  obeying my goals re-accrues mine-evidence — the measured 94%
  wrong-commitment recovery needs exactly this), whereas an efference
  comparator makes withdrawal self-fulfilling and unrecoverable.
- **Comparator locus:** freezing the token's contribution to the CANDIDATE
  input i_n kills the write (slope retention .16/.14 on u/v) while
  freezing its gate inputs i_r, i_z barely matters (.84/.88). The
  token-vs-template comparison happens in tanh(i_n + r∘(U_n h)) — the
  token's one-hot slice meeting the state-carried template inside the
  candidate nonlinearity — not via gate modulation.

Updated verified circuit: evidence = TEMPLATE-match of each incoming token
(computed in the candidate path) → ±increments on the coarse m̂ register
(protected axis, re-polarizing) → per-round emission into rotating readout
coordinates (the λ shadow) → σ(±.28λ+1.3) gates → plan-vs-forecast heads.
Remaining for Silver/Gold: the synthetic numpy program + stitching test +
the verified-circuit diagram.
