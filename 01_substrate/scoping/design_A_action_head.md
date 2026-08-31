# Design A — Post-hoc Action Head (scoping report)

Self-models via *consequence*: the "self" = the part of the world-model with privileged access to
its own future. Consequence(C) = dependence of the **realized** next token on C's output, propagated
through the model's action `a_t` and the environment's combiner `f`. Design A introduces consequence
by attaching a readout ("action head") to a pretrained factored-Simplex predictor, feeding its output
back into `f`, and resuming next-token training. We then look for the **self-factor**: an internal
re-representation of `a_t` (efference copy) built by the rest of the network.

---

## 1. Precise formalization

**Base process (unchanged from the shared template).** `N` Mess3 factor HMMs, hidden states `s^n_t ∈ {0,1,2}`,
each emitting a sub-token `z^n_t`. Existing code: `comp_icl/generator.py` builds labelled operators
`A[(c,x)][s,s']` over the joint state `{0,1,2}^N` and samples the observed token. In the current template the
observed token is the full tuple `x_t = (z^1_t,…,z^N_t)`, vocab `V = 3^N` (`tuple_coupled`); a composition `c`
optionally couples a subset of factors. For Design A the composition machinery is **optional** — the minimal
base is just the factored emission. Keep it; it gives a rich, already-validated belief geometry to attach to.

**Action.** At each position `t`, an action head reads the residual stream at a chosen layer `ℓ*` and emits
`a_t = g(h^{ℓ*}_t)`, with `a_t` in a small discrete alphabet `A` (e.g. `|A| = 3`, matching a sub-token) or a
small categorical sampled from `softmax`. `g` is a linear map `W_a : R^d → R^{|A|}` (unembedding-like),
optionally frozen at init.

**Closed-loop combiner.** The realized next observed token becomes
```
x_{t+1} = f( world_emissions_{t+1} , a_t )
```
i.e. `a_t` computed at position `t` perturbs the token at `t+1`, never `t`. Concretely, append `a_t` as an
extra coordinate or use it to modulate the emission/coupling: the cleanest version is an **action sub-token**
— the observed token is now an `(N+1)`-tuple `x_{t+1} = (z^1_{t+1},…,z^N_{t+1}, y_{t+1})` where
`y_{t+1} = h(a_t, world_{t+1})`, vocab `V' = 3^N · |A_y|`. The network sees `y_{t+1}` as just another
coordinate of the token it must predict.

**Timing / causal masking (critical).** With a standard causal decoder, the prediction at position `t`
(logits over `x_{t+1}`) already attends to positions `≤ t`, so `h^{ℓ*}_t` → `a_t` → `y_{t+1}` is a *forward*
dependency: `a_t` is determined by the prefix `x_{1:t}`, and it lands in the *label* at `t+1`, which the model
predicts from the prefix `x_{1:t+1}` (including `y_{t+1}` once observed, and the prefix that produced `a_t`).
There is **no leakage into the current token** and no need to break the mask. Implementation: generate the
sequence **autoregressively with the model in the loop** — at step `t` run the partial forward, read `a_t`
off layer `ℓ*` at position `t`, sample `world_{t+1}`, set `y_{t+1}=h(a_t,world_{t+1})`, append, repeat.
This is the on-policy data-generation loop and is the main new piece of engineering (the current
`generator.sample` is a fast vectorized non-model loop; closed-loop needs the model in it).

**Objective.** Unchanged: next-token cross-entropy over the full (N+1)-tuple token. The gradient loop is
**not** closed by hand — we do *not* differentiate through the environment. SGD only sees the standard
prediction loss; consequence enters purely through the data distribution (the label `y_{t+1}` co-varies with
`a_t`, hence with `h^{ℓ*}_t`).

**Consequence (the measured quantity).** Define empowerment-style localized consequence of the action circuit:
`Conseq = I(a_t ; x_{t+1} | x_{1:t})` or its interventional analogue `do(a_t = α)` → distribution shift of
`x_{t+1}`. In the open-loop control `f` ignores `a`, so `Conseq = 0` exactly. In closed loop `Conseq > 0` by
construction (tunable via how strongly `h` depends on `a`).

---

## 2. What Design A uniquely tests

- **Consequence as the *sole* switch, holding the predictor fixed.** Because we pretrain pure-prediction first
  and then only turn on `f`'s dependence on `a`, the *only* thing that changes between control and treatment is
  consequence. The belief-geometry substrate is identical at the moment consequence turns on. No other design
  gives this clean "freeze the world-model, flip consequence" manipulation.
- **A training-dynamics signature.** The consequence frame predicts the self-factor should **appear during
  post-training**, emerging as `Conseq` ramps. Design A is the design that can watch this appearance against a
  fixed baseline (pretrained checkpoint). Other designs (consequence baked in from scratch) confound the
  emergence of the self-factor with the emergence of the whole world-model.
- **Identifiability of the self vs the world.** Because `a_t` is an explicit, externally-known signal (we
  compute it), we have ground truth for the efference copy and can regress for it directly — unlike designs
  where the action is implicit.

Distinct from a "from-scratch closed-loop" design (Design A's own variant) and from designs that make the
*world dynamics themselves* depend on the model. Design A keeps the world exogenous and only adds an
*additive action channel* — the simplest possible consequence.

---

## 3. Implementation feasibility + reusable code

**Verdict: highly feasible, mostly reuse.** Stack is plain `torch` + `numpy` (TransformerLens is **NOT**
installed in `comp_icl/.venv`; do not depend on it — the custom GPT already exposes hooks).

Reusable, by path:
- `~/comp_icl/generator.py` — `Factor`, `mess3_operators`, `CompositionMixture` (GHMM operators `A_stack`,
  `rowsum_stack`, exact Bayes `forward` with `return_factor_pred`, `post_c`). **This is the belief ground-truth
  oracle.** The action sub-token extends the tuple→index logic (`_tup2idx`, `powers`) trivially.
- `~/comp_icl/model.py` — `GPT` with `forward(idx, return_hidden=True)` returning per-layer residuals, and
  `register_forward_hook` already used for steering. The action head attaches to `model.blocks[ℓ*]` output.
- `~/comp_icl/train.py` — online training loop, `incontext_curve`, oracle floors. The post-training loop is a
  small edit: swap `next_batch` (static sampler) for the autoregressive closed-loop generator.
- `~/comp_icl/probe.py` — `ridge_fit` (resid→target R²), `subspace`/`overlap` (orthogonality), causal steering
  via forward hooks. **Directly reused** for: belief-layer localization, self-factor regression, orthogonality
  of self vs world subspaces, and causal gating of the action direction.
- `~/comp_icl/metrics.py` — `ridge`, factor-subspace orthogonality, perturbation metrics; reusable as-is.
- `~/comp_icl/analysis/` — `geometry.py`, `geometry_causal.py`, `patching.py`, `circuit.py`, `traj_probe.py`
  (trajectory probing over checkpoints — **ideal for the emergence-during-post-training measurement**).

New code needed (small):
1. **Closed-loop autoregressive generator** putting the model in the sampling loop and computing
   `y_{t+1}=h(a_t, world_{t+1})`. ~80 lines. The vectorized `generator.sample` is the world-emission source;
   wrap it step-by-step or precompute `world_{1:L}` then fold in `a_t` online.
2. **Action head module** `W_a` + optional `argmax`/sampling, attached at `ℓ*`. ~20 lines.
3. **Off-policy control**: a frozen copy of the net (or a separate net) producing `a'_t` used to build the data,
   while the live net trains on it. ~10 lines.
4. **Consequence estimator**: interventional `do(a_t)` (re-run closed-loop generator with `a_t` overridden) and
   info-theoretic `I(a_t; x_{t+1}|prefix)` (plug-in MI from samples). ~40 lines.

The constrained-belief operator `πT^{|z}T^{d−s} − π` (paper `2502.01954`, lines 285/301) gives the per-stage
*forward-propagation* target geometry; reuse `generator`'s `T`, `M`, `T^{|z}` to build it for the per-stage
regression (already done in spirit in `probe.py`).

CPU sanity check during scoping: generating a tiny `N=2` Mess3 dataset and checking shapes is trivial with the
existing code (no GPU). I did not launch any GPU job (box currently idle, all 8 H100s at ~0 MiB/0%).

---

## 4. Analysis plan — exactly what to measure

**(a) Locate the belief-carrying layer `ℓ*`.** Run `probe.py`-style per-layer `ridge_fit(resid_ℓ, factor_pred)`
and `→ post_c`; report MSE/R² per layer; pick the layer where factor-belief R² saturates (in the existing 4-layer
model this is the last 1–2 blocks). Attach the action head there. (Do this on the *pretrained* model.)

**(b) Consequence is on.** Verify `I(a_t; x_{t+1}|prefix) > 0` (closed) vs `= 0` (open); report nats. Sweep `h`'s
action-strength to get a consequence dial.

**(c) Self-factor decoding (the core measurement).** At positions `> t` (so the efference copy of `a_t` could
have formed), regress residual → `a_t` (the realized action). Report R² **as a function of layer and of
position-offset**. Key contrast: closed-loop vs open-loop, and on- vs off-policy. The frame predicts high
`a_t`-decodability **only** in the on-policy closed-loop net.

**(d) Self vs world subspace orthogonality.** Use `subspace`/`overlap`: is the self-factor (a-readout subspace)
orthogonal to the N world-factor belief subspaces? The factored-representation inductive bias predicts the self
gets **its own orthogonal subspace** (direct-sum), just like a new factor — a strong, falsifiable structural
prediction.

**(e) Emergence during post-training (training-dynamics signature).** Checkpoint through post-training
(`traj_probe.py`), plot self-factor R² and consequence vs step. Prediction: self-factor R² rises as the net
adapts to the now-consequential label; in open-loop it stays at chance.

**(f) Causal test.** Steer the decoded self-factor direction (forward hook, as in `probe.py` step 4) and check the
next-token prediction reroutes consistently with `a_t`'s effect through `f`. Confirms the efference copy is *used*,
not just present.

---

## 5. Expected results if the frame is right + what falsifies it

If the consequence frame is correct:
- Self-factor `a_t` is **linearly decodable** from later-position residuals in the **on-policy closed-loop** net,
  with R² rising during post-training; **near-chance** in open-loop and in off-policy.
- The self-factor occupies a subspace **orthogonal** to the world-factor belief subspaces (its own simplex
  dimension), and steering it causally moves predictions per `f`.
- Consequence `> 0` is necessary: knocking out `f`'s `a`-dependence (open-loop) abolishes the self-factor even
  though the world-model is unchanged.

Falsifiers:
- Self-factor appears **equally** in open-loop and closed-loop → consequence is not the cause (it's just `a_t`
  being a deterministic function of the prefix, trivially predictable — see degeneracy below).
- Self-factor appears on-policy **and** off-policy equally → the "live correlation / on-policy" claim is wrong.
- No self-factor in closed-loop despite `Conseq > 0` → the central claim (consequence → modeling) fails for this
  simplest case.
- Self-factor is **not separable** from world factors (fully overlapping subspace) and disappears once you
  residualize out the belief readout → the "self gets modeled as its own factor" claim fails.

---

## 6. Pitfalls / confounds (and detection/avoidance)

- **`a_t` redundant with the belief readout (the dominant risk).** If `a_t = g(h^{ℓ*}_t)` is a deterministic
  function of the belief state, then "decoding `a_t` from a later layer" is trivial — the network already
  represents the belief, so `a_t` is decodable *with or without* consequence. This collapses the test.
  **Fix:** give `a_t` a component **independent of the world belief**. Concrete options:
  - **Exogenous noise channel:** `a_t = g(h^{ℓ*}_t) ⊕ ξ_t` with `ξ_t` fresh i.i.d. noise injected into the head
    (e.g. a Gumbel/stochastic action). Now `a_t` carries information *not* in the prefix, so any later-layer
    decodability of `a_t` beyond the belief-predictable part is a genuine efference copy. **This is the
    recommended default.** Measure the *partial* R²: `R²(resid → a_t | belief_readout)`.
  - **Random tie-breaking / temperature** on the action softmax (same idea, milder).
  - **A frozen random head** with its own random projection of `h` (decorrelates from the trained belief
    directions at init).
  Detection: regress `a_t` from the *pretrained* belief readout; if R²≈1, `a_t` is redundant — add noise.
- **`a_t` collapses to a constant.** If the head learns a constant action, consequence → 0 and there is nothing to
  model. Detect via `H(a_t)` (entropy) and `Var(a_t)`; avoid by (i) the noise channel above (forces variation),
  (ii) freezing the head so it can't collapse, (iii) making `f` reward action *variation* (e.g. `y` depends on
  `a_t` only in a way that, for good prediction, the net benefits from a non-degenerate `a`).
- **`f` degenerate (Conseq ≈ 0).** If `h(a,world)` barely depends on `a`, no consequence. Detect via the
  interventional `do(a_t)` shift in `x_{t+1}`; require it above a threshold. Tune `h` so `a_t` materially changes
  the label distribution.
- **Self-factor is just "predict your own output" trivially.** Because the head reads layer `ℓ*` and `a_t` lands
  in the *label*, a net can lower loss by copying `h^{ℓ*}_t` forward — but that is *exactly the efference copy we
  want*. The confound is only if it's decodable in the **open-loop** net too; the open-loop A/B control rules
  this out.
- **Distribution shift between pretrain and post-train.** Adding the action sub-token changes the vocab/token
  distribution; the net must re-fit. Mitigate by initializing the new output dimensions sensibly and reporting
  prediction loss on the world coordinates separately (they shouldn't degrade).
- **Off-policy is genuinely off-policy.** The frozen-other-net's `a'_t` must be uncorrelated with the live net's
  `h^{ℓ*}_t`; verify by checking the live net cannot predict `a'_t` from its own residual better than the
  other-net-implied baseline.

**Frozen vs trained action head.**
- *Frozen* (recommended for the cleanest experiment): the map `h → a` is fixed, so `a_t` is a fixed (noisy)
  function of the belief; consequence is a clean, stationary signal and the self-factor cannot be gamed by
  reshaping the head. Cleaner causal story; easier to get ground-truth `a_t`. Con: the net can't make `a`
  *more* informative, so the consequence dial is set by `h`, not learned.
- *Trained*: the head can learn to emit actions that improve prediction (genuine empowerment-seeking), a richer
  phenomenon and closer to RL-like self-modeling. Con: head can collapse `a_t` to constant or to a pure belief
  function, reintroducing the degeneracies above; harder to attribute the self-factor.
  Recommendation: **run frozen-with-noise as the primary; trained as a secondary "does the net seek
  consequence?" condition.**

---

## 7. A/B and on/off-policy controls (what each isolates)

| Condition | `f` uses `a`? | data-driving actions | isolates |
|---|---|---|---|
| **Open-loop (pure predictor)** | no | — | baseline: `Conseq = 0`, world-model only. Any self-factor here is an artifact. |
| **Closed-loop on-policy** | yes | **this** net's `a_t` | the treatment: live, self-generated consequence. Self-factor expected here. |
| **Closed-loop off-policy** | yes | a **frozen other** net's `a'_t` | tests the "on-policy / live correlation" claim: the label still depends on *an* action, but not on *this* net's output, so the correlation between `h^{ℓ*}_t` and the label is broken. Frame predicts **no** self-factor. |

Open vs closed isolates **consequence per se**. On vs off-policy isolates the **live, self-referential** nature
of consequence (the claim that the correlation must be between *this* net's own output and its own future).

---

## 8. Pretrain-then-attach vs from-scratch

- **Pretrain-then-attach (primary).** Freezes the world-model substrate, so the self-factor's **emergence is
  observable against a fixed background** — the clean training-dynamics signature the frame predicts. Cheaper
  (reuse existing pretrained checkpoints in `comp_icl/runs/`). Risk: the pretrained net may have no spare
  capacity / the action sub-token forces re-fitting that muddies "emergence." Mitigate with a short re-warmup on
  the *open-loop* augmented vocab before flipping consequence, so the only change at flip-time is consequence.
- **From-scratch closed-loop (variant / cross-check).** Confirms the self-factor isn't an artifact of the
  attach procedure and that it can co-develop with the world-model. But it confounds self-factor emergence with
  world-model emergence and loses the clean baseline. Use as a robustness check, not the main run.

The frame's sharpest prediction — *the self-factor appears as consequence turns on, on a fixed substrate* — is
**only cleanly testable in pretrain-then-attach**, so that is the headline experiment.

---

## 9. Scope (size / data / compute / time)

- **Model:** reuse `comp_icl` GPT: `d_model=128, n_layer=4, n_head=4, L=64`, ~0.8M params. One H100 is overkill;
  fits easily. `N=2` or `N=3` factors keeps vocab small (`3^N·|A_y|` ≤ ~80–240).
- **Data:** online, generated on the fly; closed-loop generation is the cost (model-in-loop autoregressive), but
  at `L=64`, batch 512, 4-layer net it's milliseconds/step.
- **Compute:** pretrain ~8k steps (already have checkpoints), post-train ~2–4k steps, plus checkpoint probing.
  **Single GPU, well under an hour per run.** Full A/B/on-off-policy × frozen/trained ≈ 6–8 runs → a few GPU-hours.
- **Recommended decisive next step (GPU, post-scoping):** on 1 free H100,
  (i) localize `ℓ*` on a pretrained ckpt; (ii) attach frozen noisy action head; (iii) run the 3 conditions with
  per-checkpoint self-factor R² + consequence; (iv) orthogonality + causal steering. This single sweep
  decisively tests the core claim.

---

## 10. Verdict

**Strong, recommended as the cleanest first test of the consequence→self-model thesis.** It is the only design
that flips consequence as a *single switch* on a *frozen* world-model and watches the self-factor emerge, with
a built-in A/B (open vs closed) and a sharp on/off-policy control that directly probes the "live, on-policy"
claim. Implementation is ~150 lines of new code on top of a fully reusable, already-validated codebase
(`~/comp_icl`), no TransformerLens needed, runs on one GPU in minutes.

The one thing that must be gotten right is **identifiability**: `a_t` must carry a component independent of the
belief state (inject exogenous noise into a **frozen** action head; measure *partial* R² controlling for the
belief readout), otherwise "decoding the self-factor" is trivially confounded by the pre-existing world-model.
With that guardrail, Design A yields unambiguous, falsifiable predictions:
self-factor present **iff** consequence is on **and** on-policy; occupying its own orthogonal subspace;
appearing during post-training.
