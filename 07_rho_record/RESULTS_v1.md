# RESULTS — ρ-record v1 (variant A), run 2026-08-13

Design: DESIGN_rho_record_v1.md. Model 800k-param GPT, 20k online steps,
5.3 min on one H100. Eval: 4096 held-out episodes vs exact Bayes floors.

## Headline

The transformer converges to the exact stream-observer bound **to within
4·10⁻⁵ nats** (late positions), with pointwise predictive agreement
R² = 0.9995, while remaining exactly Π above the agent floor: measured final
excess over agent floor **0.08809** nats/round vs theoretical Π-gap **0.0882**.
It never dips below the observer floor at any checkpoint (min excess +3.8e-5
— no leakage; the harness is clean). a-slot loss = 0.69315 = log 2 to five
decimals throughout (the record's action slots are pure noise, and the model
knows it).

## Prereg scorecard

- **P1 (bound approach): CONFIRMED.** Excess vs observer floor → +0.0000;
  gap to agent floor = Π to three decimals.
- **P2 (a-slots pinned at log 2): CONFIRMED** at every checkpoint.
- **P3 (nature before echo): PARTIAL / strict form falsified.** Onset order
  as predicted: nature coefficient moves first (b_nat 0.29 vs b_echo 0.07 at
  step 20). But the echo channel then rises *steeper* and overtakes mid-
  transition (step 50: b_echo 0.66 > b_nat 0.49); both complete together by
  step ~100–200. One main acquisition event grabs both channels; nature only
  leads in the pre-transition trickle.
- **P4 (pointwise optimality): CONFIRMED.** R² 0.9995 final. Notable: loss is
  within 0.002 nats of the floor by step ~1500, but pointwise R² keeps
  improving 0.966 → 0.9995 over steps 1500 → 20000 — a long calibration-
  polishing tail invisible in the loss.
- **P5 (mirror r↔1−r): deferred**, not run.

## Timeline (steps; 1 step = 128 fresh episodes)

0–10: grammar (token-type alternation) + weak nature statistic. 10–100: the
main transition — both inference channels form (nature onset first, echo
steeper). 100–1500: loss closes to within 0.002 nats of the floor.
1500–20000: pointwise calibration polishing (R² 0.966 → 0.9995).

## Files

eval_results.json (per-checkpoint metrics), figs/loss_vs_pos_by_ckpt.png,
figs/excess_vs_step.png, figs/channel_acquisition.png,
figs/agreement_final.png, figs/theory_*.png, train_hist.json,
ckpt/step_*.pt (21 checkpoints), logs/.

## Next (queued)

Residual-stream probes for κ̂ and θ̂ trajectories (core/probes kit) — is the
observer's κ̂ a findable direction, and when does it form (predict: during the
10–100 transition). Variant B (efference side channel) → beat-the-ceiling
test. r-sweep incl. mirror check. Dual-control (agent chooses probes).
