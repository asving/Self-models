# Self-models project — update for Naja (from Asvin, 2026-07-03)

Summary of the last two experiment cycles on the self-models project. All code, runs,
checkpoints and logs live in **`/data/users/asvin/self-models/`** (paths below are relative
to that). Conceptual background docs there too: `working_notes.md`, `SELF_LEGIBILITY.md`,
`DEPTH_AND_RECURRENCE.md`, `CIRCUIT_FINDINGS.md`, `scoping/design_E_vicarious.md`.

## Framing (very compressed)

- **Retrospective self-model = a belief factor whose prior you control.** Three regimes for
  the own-action latent: *eliminate* (entropy collapse), *record* (efference copy / legible
  seed), *infer* (filter your own past action like any HMM factor). Formally, the self-model
  is the mixing distribution over which belief-update operator `T^{(a,x)}` the world model
  applies.
- **Generator vs predictor = the belief/action split**: consequence forces the emitted
  channel away from the calibrated posterior; managing that divergence is the self-model.
- **Anti-fusion principle**: any self-latent the network can *see* gets absorbed into context
  and compiled away ("visibility ⇒ absorbability"); a persistent self-model needs an
  architecturally guaranteed self-prediction gap (capacity, or a genuinely lagged self).

## Experiment 1 — "Dinner party" (pretrain → RL, LLM-microcosm). `dinner.py`

**Setup.** 3 drifting Mess3-style factors (per-episode continuous decay rates), goal +
deadline announced as tokens, budget of 3 "sets" each requiring 2 sustained identical
actions, explicit WAIT. Single next-token head, interleaved obs/action tokens (deliberately
LLM-shaped). Phase 1: CE pretraining on 7 scripted actors — including a **packer** that
demonstrates the wait-then-pack *template* with deliberately wrong (index-order) scheduling.
Phase 2: REINFORCE/RLOO, terminal reward = factors at goal at deadline. Certified ladder
before training: greedy ≈ 1.10 ≪ packer-template 1.92 ≪ rate-adaptive backtimed 2.03.

**Results** (runs + 49 dense ckpts: `dinner_runs/v1/`, logs `logs/dinner_p*_20260703_*.log`):
- Pretraining reaches the exact-Bayes floor on observations; who-machinery forms.
- **RL = pure template retrieval**: R 1.18→1.90 in ~150 steps with entropy 1.32→0.02, then
  flat for 3850 steps. Final policy matches the packer baseline **to 3 decimals at every
  horizon including held-out deadlines** (so the deadline arithmetic generalizes OOD), but
  order-agreement with decay-ranking = 0.494 ≈ chance: **zero** of the +0.10 planning rung.
  "Pretraining proposes, RL selects" — and only selects. Exploration trap: once entropy
  collapsed, no set-order variation remains for REINFORCE to credit.
- **Mechanism probes** (`dinner_probe.py`, targets = rate-marginalized filter):
  world model is **parallel re-derivation, not stored-belief+operator recurrence** (belief
  R² at action positions: L0 .02 → L2 .87 → L4 .97 — assembled by attention per position);
  a completion-cancelling do-test shows decoded beliefs track the **counterfactual** filter
  (operators causally wired to action tokens); and — the big one — **drift-rate estimates are
  linearly decodable at R² ≈ .83 in the same residual stream that drives the rate-blind
  policy**, surviving RL almost unchanged. Knowledge present, unconsulted: the planning
  failure is a *wiring* failure, not a knowledge failure.

## Experiment 2 — "Ambush game" (self-simulation vs a mindreading opponent). `ambush.py`

**Why.** To make the net *use* its machinery at inference we wanted a task where the optimum
moves and simulation of one's own policy is load-bearing. Naive choice (RPS vs a
best-responder to your own checkpoint) fails: zero-sum interior equilibria are **signal
deserts** — payoff = ε_pᵀAε_d is bilinear in deviations, so near uniform everything is
O(ε²) and the adversarial pressure *regenerates* the desert. Two fixes:
1. **Terrain**: base reward for matching a drifting hidden state (belief-anchored play,
   first-order gradients everywhere, policies forced off-uniform and context-rich);
2. **Prediction channel**: opponents emit visible *camp* tokens; the mindreader (per-episode
   latent type, format-identical to bias-campers) camps on `softmax(γ·p̂)` where `p̂` = a
   **lagged checkpoint of the agent itself** evaluated at this context. Predicting camp
   tokens by plain CE *is* dense supervised self-simulation, immune to payoff flatness —
   and even *detecting* the mindreader requires computing what you yourself would do
   (camps sit on your would-be argmax 99% vs 33% for bias types).

Certified ladder: greedy +.349 (intercepted 96%), dodge-vs-stale-reader +.471, dodge-vs-
perfect-reader +.005 (the arms-race bracket). Nets are deliberately narrow-deep (d=64, 6L)
so building machinery parallel to the pretrained one is expensive.

**Results** (runs: `ambush_runs/{A,B,A_lag1000,A_lag4000}`, dense ckpts around transitions):
- **This time RL left the template.** Arm A (pretrained): greedy plateau for ~400 steps —
  terrain .83, standing in 89% of ambushes *while predicting the camps at CE 0.17* (it knows
  exactly where it will be hit) — then at **step ~700 type-conditional dodging emerges**
  (dodge rate .63 on reader episodes vs .16 on bias episodes; interception .89→.36). The
  interception cost made the template's flaw first-order, and the CE-channel had pre-built
  the self-simulation; RL wired it in. Contrast with dinner, where the flaw was O(0.1) and
  the trap closed first.
- **Pretraining leverage**: arm B (RL from scratch) reaches similar behavior ~6× slower.
- **Self-image dissociation, visible in raw logs**: during the plateau, camp-CE on reader
  episodes is 0.17 (checkpoint = current self, introspection suffices); at the transition it
  **spikes to 1.7** (policy moved; self-image broke) and settles at 1.1 — now *worse* than
  bias-camp CE (0.84), the reverse of pretraining. Tracking a moving self is harder than
  tracking an external bias.
- **Lag sweep (250/1000/4000)**: weak sawtooth in the predicted direction at lag 4000
  (within-cycle R .371→.382; self-image CE improves as the checkpoint ages). Root cause of
  weakness: the policy *converges* (~step 2000, entropy .05), and a lagged reader of a
  stationary policy is a perfect reader. **Lagged self-play's moving target is transient**:
  it stalls at the equilibrium where best-response-to-the-reader-of-me = me.

## Where this leaves us (the open fork)

At the converged equilibrium the dodge is still context-dependent through `p̂(h)`. The live
question: is it **routed** through inference-time self-simulation (compute "what would I
play here", avoid it) or has convergence **compiled** it into a direct context→dodge map
(fusion, one level up)? Decidable with the probe battery: decode `p̂` from the residual
stream (ground truth = literally run the checkpoint), steer the decoded direction and test
whether the dodge follows; run it across the dense checkpoints spanning the step-700
transition (possible "scaffolding-then-compilation" story). Alternative next move: redesign
for perpetual motion (per-episode θ/γ, level-2 readers, population play).

## Quick file map

```
/data/users/asvin/self-models/
  working_notes.md, SELF_LEGIBILITY.md, DEPTH_AND_RECURRENCE.md   # conceptual docs
  scoping/design_E_vicarious.md        # design doc (E-series rationale)
  vicarious_oracle{,2}.py, figs/vic_*  # controlled-Mess3 belief-attractor feasibility figs
  dinner.py / dinner_eval.py / dinner_probe.py     # experiment 1 + eval + mechanism probes
  dinner_runs/v1/                       # 49 ckpts (dense early), logs, gap.json, probe.json
  ambush.py                             # experiment 2 (gap | train1 | train2 modes)
  ambush_runs/{A,B,A_lag1000,A_lag4000} # both arms + lag sweep, dense ckpts
  logs/dinner_*, logs/ambush_*          # tee'd training logs
```

Everything runs with `~/comp_icl/.venv/bin/python`; small models, single GPU, minutes per
run. Happy to walk through any of it — and the probe battery for the ambush equilibrium is
the natural next thing to build if you want to grab it.
