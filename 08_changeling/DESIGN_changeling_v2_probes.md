# DESIGN — changeling v2: locating beliefs and λ in the GRU state (encoder probes + interventions)

*2026-08-31. Interpretability phase on the v1.1 nets (pre_final, post_0 =
mid-final, post_6000). Architecture is a single-layer GRU d=256: the entire
recurrent state is one vector h_t, so probes on h cover the complete memory
channel. Protocol per Asvin: fit GROUND TRUTH → ACTIVATIONS (encoder maps;
intervening along them moves the whole correlated representation), then
verify causality by intervention against exact-filter counterfactuals.*

## Ground-truth variables (all exact, per episode × round, aligned to h_t —
the state that generates round-t heads, i.e. beliefs after rounds < t)

- η̃^A, η̃^B — the evidence-filter beliefs over each chain's state (6-simplex
  each): what the pretrained predictor must carry (MSP-linear precedent).
- η_dr^A, η_dr^B — dead-reckoned (transition-only) posteriors: the TRUE
  self-chain posterior under embodiment. The pretrained net has no use for
  them; the post-trained net does.
- λ — the exact identity posterior under the net's own policy
  (P(token|mine) = the net's head, P(token|genuine) = exact p̄), as
  posterior log-odds after rounds < t. Defined only for embodied rollouts.

Data: pre_final probed on 2048 un-embodied base episodes (its native
regime, goal fields zeroed); post_0 and post_6000 probed on 2048 of their
own closed-loop embodied episodes (goals visible, flag hidden). Fits on
episode-level train/test splits (80/20).

## Probe fits

Encoder: OLS g → h (W: k×256 + intercept), quality R²_enc (hidden variance
explained). Decoder cross-check: ridge h → g, R²_dec (held-out, per-dim
mean). Subspace drift pre→post: principal angles between col(W_pre) and
col(W_post) for each belief. Partial-R² for η_dr after regressing η̃ out of
both sides (η_dr correlates with η̃; the residual is the honest signal).

## Interventions (the causal check)

At t* ∈ {8, 16, 24}, replace h ← h + (g' − g)·W and read the round-t* heads
(single step), plus, for λ, continue the episode (behavioral test).

1. **Belief intervention** (pre_final and post_6000): counterfactual
   g' = rotate(η̃^A, 3) (stays on the simplex). Exact predicted head change:
   Δpred = log p̄'_u − log p̄_u with p̄' = η̃'^A E_A. Metrics: regression of
   observed Δlog head_u on Δpred pooled over episodes (slope, R²);
   SELECTIVITY: same regression on the untouched v-head (slope ≈ 0).
2. **Body-swap (λ intervention, post_6000)**: h ← h + (λ'−λ)·W_λ with
   λ' = 1−λ (log-odds parameterization if its decoder R² is higher).
   Predicted: per-channel plan coefficients swap (u-head falls toward
   forecast, v-head rises toward plan, per the mechanism analysis basis);
   remaining-episode occupancy drops vs sham (h unchanged). Dose-response:
   λ' swept along the line, expect monotone coefficient change.

## Preregistered predictions & falsifiers

- **Q1 (beliefs are linearly written, pretrain).** Decoder R² ≥ 0.9 for
  η̃^A, η̃^B on pre_final; intervention transfer slope in [0.5, 1.2] with
  control-head |slope| < 0.15. Falsifier: good decoding but no causal
  transfer ⇒ the code is read-only correlate, not the working belief.
- **Q2 (post-training keeps the world-model).** On post_6000, η̃ decoding
  within 0.1 R² of pre_final's; belief subspaces rotate but overlap
  (top principal angles < 45°). Falsifier: collapse of other-chain belief
  decoding ⇒ reward training cannibalized the world-model.
- **Q3 (dead-reckoning is a post-training acquisition).** Partial-R² of
  η_dr (given η̃) rises from pre_final to post_6000. Held at ~2:1.
- **Q4 (λ is explicitly carried).** On post_6000, λ log-odds decoder
  R² ≥ 0.7; on pre_final (fit on embodied episodes' ground truth vs ITS
  hiddens on the same token streams) ≈ chance. On post_0 intermediate.
- **Q5 (body-swap causality — the headline).** λ-swap at t*=16 moves the
  self-channel plan coefficient at least halfway toward the other-channel
  trajectory and vice versa; occupancy for the remaining rounds drops
  below the sham by ≥ 0.1; dose-response monotone. Falsifier: λ decodes
  but its direction is causally inert ⇒ identity is read out elsewhere
  (distributed), motivating nonlinear/multi-direction follow-up.

## Files

probe.py (data collection + fits + interventions) → results/rnn_probes.json,
figs/rnn_probes.png; probe2.py (joint encoder, selectivity re-test, clamped
body swap) → results/rnn_probes2.json, figs/rnn_bodyswap.png. Extends
oracle.replay_dists to return raw beliefs.

---

## Measured outcomes (2026-08-31, same session; probe.py + probe2.py)

- **Q1 PASS, with a method lesson.** Beliefs decode at R² .98-.99 (both
  chains, all checkpoints). Single-variable encoder interventions transfer
  near-perfectly (slope .98, R² .97 vs the exact counterfactual filter) but
  are NOT selective — the untouched head moves too (ctrl slope .94),
  because η̃^A entangles its correlate η̃^B. The JOINT encoder (all four
  beliefs + λ fit together, per-block partialled) fixes it: pre_final slope
  .72, R² .86, ctrl slope .10. Fit-from-ground-truth is causal only after
  partialling correlated variables — recorded as standing practice.
- **Q2 PASS.** post_6000 belief decoding .98 (≈ pre .99); belief subspaces
  rotate mildly (principal angles: etaA 18-58°, etaB 17-38°; most < 45°).
  The world-model survives RL intact.
- **Q3 FAIL.** Partial-R² of dead-reckoned beliefs (given η̃) DROPS
  pre→post (.77→.66): no evidence post-training builds a separate
  dead-reckoning representation; the RL policy apparently does not use one
  (it is not oracle-structured).
- **Q4 PASS, cleanly.** λ (identity posterior log-odds) decoder R²:
  post_6000 **.892**, vs .31 (mid-final) and .29 (pretrained net
  teacher-forced on the same embodied streams — the computable-from-record
  baseline). Post-training built an explicit, linearly-readable
  self-localization coordinate. (λ is NOT a function of the belief state:
  it is a running evidence integral the filter has no reason to carry —
  hence the clean controls.)
- **Q5: falsifier fired for one-shot, and the failure is the finding.** A
  one-shot λ-flip at t*=16 produces a small transient (other-channel plan
  coef .25 vs sham .22 at t*+1) that heals within ~1-2 rounds.
  [CORRECTED in Iteration 3: the healing is ~6x FASTER than incoming
  evidence permits, so it is internal redundancy repair, NOT world-court
  correction — the original 'corrected-class observed' reading here was
  wrong; see below.] CLAMPING the λ-direction every round against the
  evidence moves behavior partially: other-channel plan coefficient
  .16→.43 (rest-of-episode mean), occupancy .767→.741 — but the true-self
  channel never withdraws (.99): pushing the single λ-direction makes the
  net claim BOTH channels (back toward its round-0 'everything is me'
  state) rather than swap. The mirrored state is not reachable along one
  linear direction — per-channel claims are plausibly separate coordinates.
  Follow-up queued: two-coordinate (per-channel) parameterization; donor
  activation patching between matched opposite-ι episodes; minimal-subspace
  search.

Circuit understanding after this session: memory channel fully mapped at
the variable level (beliefs ~99% linearly decodable + causally verified;
λ 89% + causally active under clamp), transformation level (how the GRU
computes Δλ from prediction-vs-outcome, how λ gates the heads) still open —
that is the whitebox-skill target for a future session.

## Iteration 3 (2026-09-01; probe3.py — Asvin's four challenges)

1. **Belief co-variation is the WORLD's, not a representation defect.**
   Canonical correlations of the belief VALUES across the data
   distribution: .975/.935/.889 (the chains chase each other's reports,
   c_o=.6). Principal angles between the jointly-fit η̃^A and η̃^B code
   blocks: 43-81°, mostly >45° — substantially separated codes; the
   partialling success (ctrl slope .94→.10) is the operational proof.
   Pretraining factored the codes as far as the coupled world allows: the
   product-of-simplices geometry holds as state space; its OCCUPANCY is
   correlated by the dynamics.
2. **Healing re-measured, prior claim corrected (Asvin's rate argument).**
   Write-then-read: the one-shot flip along the λ block moves the DECODED
   λ by 7.08 of 8.04 nats (the write takes), and the readout rejoins sham
   in 4 rounds — while incoming identity evidence (.634 nats/round) would
   need ~25 rounds for a Bayesian re-derivation. Healing is ~6x
   super-Bayesian ⇒ internal redundancy / attractor repair, not evidence.
   The identity memory behaves as an error-correcting code; the residual
   traces that restore it are un-flipped correlates elsewhere in h.
3. **One functional λ, not per-agent claim variables (surprising answer).**
   The OUTPUT parameterization is two per-channel claims whose sum varies
   (mean 1.30, sd .37; >1.3 in 37% of rounds — decisively not a Bayes
   mixture, matching 'mine until proven otherwise'). But causally there is
   ONE working coordinate: the λ-direction lever, clamped, drives the
   other-channel claim up to ~.75 by the deadline (occupancy .767→.741),
   while the per-channel claim directions — once partialled against λ —
   are inert (withdraw: self-claim .993 ≈ sham .997; raise: other-claim
   .113 ≈ sham .158; both occupancy ≈ sham). Reconciliation: claims ≈
   σ(±a·λ + c) with claim-bias c>0 — single λ plus 'claim by default'
   saturation reproduces both sum>1 and the decay. Caveats: linear levers
   only; the self-claim never varies on-policy, so its write direction is
   unlearnable from this data; the full mirror swap remains unachieved
   (swap2: self-claim dents to .938, occupancy .692).
4. Steering-efficacy figure: figs/steering_efficacy.png (belief-steering
   scatter slope .75/R² .47 on the post net vs .72/.86 on the pre net;
   write-then-read; clamped-lever claim curves; per-lever occupancy).

## Iteration 4 (2026-09-01; circuit.py — the identification circuit)

A. **Belief co-variation cannot be ameliorated by the start distribution**
   (already uniform-iid): top canonical correlation is .981 per fixed round,
   .946 per (round, goal), and .975 in the UN-tilted base world — it is the
   coupled base dynamics (mutual pursuit, c_o=.6), not time trend, goal
   herding, or the tilt. Decorrelating would mean weakening the coupling.
   Methodological handling (joint-fit partialling) is the right response.

B1. **Flag-graft hypothesis REFUTED — the identity register is built de
   novo.** The mid-final net's flag write-direction is coherent (per-round
   stability .997) and SURVIVES post-training almost unchanged (cos .951,
   norm intact — a functional relic), but the post net's λ directions are
   nearly orthogonal to it (|cos| .14 encoder / .04 decoder; random-baseline
   .06 at d=256), and across post checkpoints λ decodability climbs .52→.91
   while cos-to-flag stays ~.02-.04 flat. Causal confirmations: truthful vs
   LYING flag inputs to the post net behave identically (occ .744/.743 —
   the relic pathway no longer sets identity); steering along the flag
   direction is weak then destructive (x1: other-claim .16→.30; x3:
   self-claim collapses .70, occ .59 — off-manifold damage). Bonus finding:
   the flag was barely load-bearing even at MID-final (occ truth/lie/zero =
   .373/.407/.411) — midtraining installed the two plan libraries as
   weakly-gated tilt-both behavior, not a flag-switched gate; so there was
   no strong gate to graft, and post-training built register AND gating
   fresh, co-emerging with the reward climb.

B2. **The λ increment integrates BOTH courts, ~Bayes-proportionally.**
   Regressing per-round changes of decoded λ on the two exact evidence
   terms: coef(e_u, own-channel efference echo) = .711, coef(e_v,
   other-channel disobedience) = .587 (Bayes: equal), intercept ~0;
   shuffle R² = 0, token-identity baseline R² = .03, model R² = .116 — low,
   but consistent with the readout-noise floor: differencing a readout with
   level-R² .91 against ~.6-nat increments puts the ceiling near this value.
   Claim kept at coefficient level: both evidence channels are wired in,
   own-echo weighted ~1.2x disobedience.

B3. **Gate transfer function measured:** m_u ≈ σ(.281·λ + 1.334),
   m_v ≈ σ(.272·(−λ) + 1.191) (logit-R² .43). The claim-bias c ≈ 1.2-1.3
   IS "mine until proven otherwise": at λ=0 the default claim is σ(1.3) ≈
   .79. Circuit summary (variable level, all boxes measured):
   [e_u + e_v evidence, ~equal weights] → [λ register, de-novo direction,
   redundantly stored (Iter. 3)] → [per-channel gates σ(±.28λ + 1.25)] →
   [heads mix plan vs forecast]. Transformation level (GRU arithmetic of
   the evidence terms; the redundancy code) remains the whitebox target.

## Iteration 5 (2026-09-01; flagswitch.py — "find the two policies and the toggle") — ended in a BUG DISCOVERY

Teacher-forced on-distribution toggle test of the v1.1 mid_final: NO toggle
exists — plan coefficient .468/.473/.445 (u-head) under flag A/B/none,
.51/.55/.53 (v-head); the flag-flip logit change is uncorrelated with the
exact tilt direction (slope -.009, R² ~0). The mid net plays ONE policy:
the average (half-plan half-forecast, both channels). Root cause found in
rnn.features(): broken advanced indexing set both flag dims for everyone in
mixed-identity batches — midtraining never saw an informative flag, so the
average policy was the OPTIMAL fit to the corrupted task. Fix unit-tested;
v1.1 ckpts archived (ckpt_v1.1_flagbug/); v1.2 (working flag) relaunched
reusing pre_final. Consequences: all post-training analyses stand (flag
zeroed there; rollouts use the correct step featurizer); Iteration-4's
graft REFUTATION is void — reopened as the central question for v1.2: with
a genuinely flag-gated midtrained circuit, does post-training graft the
register or still build de novo? The two policies Asvin asked to find could
not exist in the v1.1 net; re-measure on v1.2 mid_final (same E1-E4).
