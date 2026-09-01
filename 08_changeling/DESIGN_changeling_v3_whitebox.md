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
