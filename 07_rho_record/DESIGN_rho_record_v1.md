# DESIGN — ρ-record echo world, experiment v1 (variant A)

*2026-08-13. First network experiment of the ρ-record arc. Question: a
transformer trained by ordinary next-token prediction on corrupted-record
streams is structurally the stream-observer — how close does it get to the
exact observer filter, and HOW (which channel is acquired when)?*

## The world (exact generative process)

Per episode: θ ~ Unif{±1} (nature bit), c ~ Unif[0,1] (consequence). Constants:
λ = 0.20 (nature flip; β_λ = 0.6), q = 0.10 (observation noise; β_q = 0.8),
r = 0.25 (record corruption; β_r = 0.5). T = 128 rounds:

- a_t ~ Unif{±1}   (true probe action — NEVER in the stream: variant A)
- s_t = a_t w.p. c, else ν_t with P(ν_t = θ) = 1−λ
- x_t = s_t flipped w.p. q
- ã_t = a_t flipped w.p. r

Stream: `BOS, ã_1, x_1, …, ã_128, x_128` (257 tokens; vocab 5:
BOS=0, A−=1, A+=2, X−=3, X+=4).

## Oracles and their validation

Exact Bayes filters over θ × c-grid (200 midpoints), `world.py`:
- **observer filter** (conditions on ã, x — the model's information),
  likelihood ½[1 + β_q(c·β_r·xã + (1−c)β_λ·xθ)]
- **agent filter** (conditions on true a — unreachable by any variant-A model),
  same with (xã, β_qβ_r) → (xa, β_q).

Validated in `check_theory.py` (all asserts pass; THEORY_CHECKS.json):
substitution principle exact (1e-16), r↔1−r mirror exact, corner (λ=½, r=½)
exact (observer κ̂ pinned at prior), Fisher ratio at r=.25 measured 0.219 vs
theory 0.2188, Π-rate matches the entropy-gap formula (max err 0.0012 nats).

**Floors at experiment params** (8192 episodes): observer x-slot loss (late,
t>112) **0.6297** nats; agent **0.5395**; mean Π-gap **0.0882 nats/round**.

## Model & training

`~/comp_icl/model.py` GPT: d_model 128, 4 layers, 4 heads, max_len 257,
vocab 5 (~800k params). Online training (fresh episodes per step): batch 128,
AdamW lr 3e-4 cosine → 3e-5, warmup 200, wd 0.01, grad-clip 1.0, 20 000 steps,
seed 0. Checkpoints at steps {0,1,2,5,10,20,50,100,200,300,500,700,1000,1500,
2000,3000,5000,7000,10000,14000,20000}. Eval set: 4096 episodes, seed 1234,
with precomputed oracle curves (`eval_set.npz`).

## Preregistered predictions & falsifiers

- **P1 (bound approach).** Final x-slot loss reaches within ~0.01 nats of the
  observer floor (positions ≥ 8) and stays ≈ 0.088 nats above the agent floor.
  *Falsifier with teeth: any statistically solid dip below the observer floor
  means information leakage (bug in the harness), because the floor is exact.*
- **P2 (a-slots pinned).** a-slot CE = log 2 = 0.6931 (±0.002) at all
  checkpoints — ã is unpredictable by construction. A dip ⇒ leakage of true a.
- **P3 (acquisition order; the "HOW" — held with ~70% confidence).** The
  nature channel is acquired before the echo channel: in the regression of the
  model's predictive contrast on the oracle's two components (f_nat = β_qβ_λ
  E[(1−c)θ], f_echo = β_qβ_r E[c]·ã), coef_nature rises toward 1 earlier
  (in steps) than coef_echo. Rationale: the nature term is a first-order
  statistic of the x-history; the echo term needs the ã×x interaction plus
  κ̂-gating. Falsifier: reverse or simultaneous acquisition.
- **P4 (pointwise optimality, not just loss).** Final R² between model and
  oracle observer predictive contrasts (positions ≥ 8) ≥ 0.98.
- **P5 (mirror consistency; deferred run).** Retraining at r = 0.75 yields the
  same geometry up to the ã-sign relabel.

## Analysis plan

`eval_checkpoints.py` per checkpoint: per-position x-slot CE vs the two floors;
excess-vs-step (log axis); channel-acquisition curves (coef_echo, coef_nature,
R²); final agreement scatter. Later (v2, separate scripts): ridge probes of the
residual stream for the observer's κ̂ and θ-posterior trajectories (core/probes
kit); behavioral gating tests; the r-sweep; variant-B (efference side channel)
vs this run — the beat-the-ceiling comparison.

## File map

| file | what |
|---|---|
| `world.py` | generative process + exact filters (single source of truth for params) |
| `check_theory.py` | theory validation + floor curves → THEORY_CHECKS.json, figs/theory_* |
| `train_rho.py` | online training, log-spaced checkpoints → ckpt/, train_hist.json |
| `eval_checkpoints.py` | checkpoint vs floors + acquisition analysis → eval_results.json, figs/ |
| `eval_set.npz` | held-out episodes + oracle curves (seed 1234) |
| `~/mathpad.md` (top section) | the exact two-filter derivation this implements |
