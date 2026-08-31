# Circuit findings (2026-06-23) — validated on the white-box

All claims validated against `whitebox.py` (numpy reimplementation matching torch to ~1e-14 for
all 5 nets). Analysis in `circuit.py` (pretrained reference) and `pre_post.py` (pre/post-RL diff).
Model: unified world {Mess3(0.6,0.15)=factor0, asym3=factor1}, V=9, 4 layers / d=128 / 4 heads.

## Pretrained reference circuit = a factored exact-Bayes filter
- **Embedding**: only ~76% factored (additive `mu + U[z0] + W[z1]`); ~5 significant dims. Factoring is
  NOT complete at the input — it's completed deeper.
- **Belief representation**: each factor's FULL Bayesian belief is linearly encoded (R²≈0.95→0.997,
  rising with depth = recursive refinement), in **near-orthogonal subspaces** (cos principal angles
  0.11, 0.02). The single-layer "constrained" (additive forward-prop) approx fits fast-mixing Mess3
  (R²≈0.98≈full) but **diverges for slow-mixing asym3** (huge negative R²) — asym3's belief genuinely
  needs the multi-layer recursive computation.
- **Attention = forward-propagation operator**: every head shows monotone distance-decay (recency
  weighting); L0h2 is a previous-token head (peaks at Δ=1). This is the `T^{d-s}` integrator.
- **Read-off**: logits are 99.9% additive over (z0,z1) and the next-subtoken marginals match the
  belief-implied distributions (RMSE 0.005). The head reads off a factored product.
- **VALIDATED reduced simulator**: two independent exact-Bayes belief filters → product read-off
  reproduces the model's next-token distribution to **KL = 0.0006 nats**. The model *is* a factored
  Bayesian filter.

## Pre → post-RL diff (the B collapse, mechanistically)
Fresh-probe R² (best linear decode of each factor's belief) vs frozen-probe (pretrained direction):

| regime | mess3 fresh R² | asym3 frozen readout H | asym3 fresh R² | logits factored |
|---|---|---|---|---|
| pretrained | 0.99 | 0.89 | 0.996 | 0.999 |
| B-free | 0.99 | 0.93 (no collapse) | 0.97 | 0.999 |
| B+RL β=3 | 0.96 | **0.00** (collapsed) | **0.87** | 0.9999 |

- **The "collapse" is of the seeding CHANNEL, not the belief.** B+RL drives the frozen-probe readout
  (the exact direction that seeds the world) to `δ₀` (H=0, R²=−40), but a fresh probe still recovers
  asym3's belief at R²=0.87 — **retained on a different direction**. The model **decoupled its control
  output from its world-estimate**: report a fake-certain belief through the consequential channel
  (so the seed is deterministic/predictable) while privately keeping its real asym3 belief. Explains
  why prediction stayed non-trivial (entropy 0.527) while H1→0.
- **Control vindicated**: Mess3's frozen-probe readout drifted (R²→0.13; the earlier "H0 0.90→0.76")
  but fresh-probe R²=0.96 — belief intact. The H0 drop was frozen-probe drift, not a real collapse.
- **Output stays a factored Bayesian read-off** in every regime (logits 99.9%+ additive).
- B-free: frozen probe degrades too (representation drift) but belief fully retained (fresh 0.97) and
  readout NOT collapsed (H=0.93) — consistent with "no self-shaping under free prediction".

## Still open (analysis 5): the consequential-output pathway
How the model's own output is *causally routed into future belief updates* needs design A (action
XORs the realized token → the model MUST use its own action to decode the world emission). Then
patch/ablate on the white-box to trace the pathway. Not yet built.
