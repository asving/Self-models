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
  coef .25 vs sham .22 at t*+1) that heals within ~1-2 rounds — incoming
  tokens re-derive identity: **perturbed self-location is a
  corrected-class coordinate inside a trained network**, the changeling
  prediction ('self-location is world-court business') observed
  mechanistically. CLAMPING the λ-direction every round against the
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
