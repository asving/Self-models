# HMM selection — what makes the consequence analysis sharpest

Diagnostics: `characterize.py` → `hmm_metrics.json`, `belief_geometries.png`.
Metrics: **Hbar** = mean stationary hidden-state belief entropy (nats, max ln Q) = how much
uncertainty design B can collapse; **fwd_spr** = spread across start-states of k=5-step token
entropy = whether "collapse onto the *most predictable* state" has a directional target;
**b_spr / effd** = how spread / high-dim the belief cloud is (geometric richness).

| factor            | Hbar | maxH | fwd_mean | fwd_spr | b_spr | effd |
|-------------------|------|------|----------|---------|-------|------|
| mess3 a=0.4 x=*   | ~1.08| 1.099| ~5.49    | 0.000   | ~0.10 | 2.0  |  ← too ambiguous: belief stuck ≈ uniform, no geometry
| mess3 a=0.6 x=0.15| 0.90 | 1.099| 5.40     | 0.000   | 0.37  | 2.0  |  ← **sweet spot**: rich 2-D fractal + collapsible entropy
| mess3 a=0.6 x=0.05| 0.74 | 1.099| 5.14     | 0.000   | 0.49  | 2.0  |
| mess3 a=0.85 x=*  | 0.30–0.52|1.099| 3.8–5.5| 0.000   | ~0.66 | 2.0  |  ← too synchronized: little residual entropy to collapse
| **asym3**         | 0.89 | 1.099| 4.28     | **2.532**| 0.32 | 1.0  |  ← **directional**: one predictable state, two random
| switch2 (2-state) | 0.51 | 0.693| 2.93     | 0.000   | 0.41  | 1.0  |  ← simplest, but symmetric & low-entropy

## Key learnings (from actually running it)
1. **Mess3 is symmetric ⇒ `fwd_spr = 0` for every (α,x).** All three states are equally
   (un)predictable forward, so design B's collapse in a Mess3 factor can only kill the
   *which-state* uncertainty H(q) (collapse to *some* vertex) — it cannot test the sharper,
   directional claim "collapse onto the *most predictable* state." There is no preferred target.
2. **α controls a clean tradeoff** between collapsible entropy (Hbar) and geometric richness
   (b_spr). α≈0.4 → belief never leaves ≈uniform (high Hbar but no structure to collapse against,
   and the world is ≈i.i.d. so collapse barely helps prediction). α≈0.85 → belief synchronizes
   to near-certainty on its own (rich fractal but little residual entropy left to collapse).
   **α≈0.6 is the sweet spot:** Hbar 0.74–0.90 *and* a rich 2-D fractal (see `belief_geometries.png`).
3. **`asym3` gives B a directional, falsifiable target.** A 3-state HMM with one near-deterministic,
   self-looping "predictable" state (state 0) and two near-uniform states: `fwd_spr = 2.53`. Under
   B the model should collapse its belief readout specifically onto state 0 — and then its
   *self-seeded* continuation becomes a trivially predictable stream of 0s. That is the sharpest
   possible "the agent makes its own future anticipable / is unsurprised by itself" signature, and
   it is *directional* (measure: does the readout drift to δ₀?), not just "entropy dropped." Its
   geometry is 1-D (effd 1.0) — easy to visualize the collapse as motion along a line.

## Recommended worlds (independent factors, eps=0 ⇒ orthogonal subspaces, factored-rep result)
- **Pipeline / simplest-first (harness unchanged, already pretraining):**
  `{Mess3(α=0.6,x=0.15), Mess3(α=0.85,x=0.10)}` — two visibly distinct fractals; vocab 9.
- **Target unified world (needs a ~10-line generic-Factor adapter for asym3):**
  `{Mess3(α=0.6,x=0.15), asym3}`. For **A**: make Mess3 consequential (rich 2-D geometry to see
  the action-conditioned belief update), asym3 = control. For **B**: make asym3 consequential
  (directional collapse onto its predictable state), Mess3 = control. Each design's consequential
  factor is the other's zero-consequence control — one world serves both, with a built-in control.

Keep N=2 (vocab 9, exact oracle trivial, beliefs visualizable). Mess3 stays the workhorse; asym3
is the targeted addition that makes B's prediction directional.
