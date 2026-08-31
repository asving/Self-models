# Design B — Self-conditioned resampling of the world

Scoping report. Verdict up front: **as literally specified, this design is a near no-op (consequence ≈ 0) at convergence.** It can be made consequential, but the minimal modification that does so turns it into a *biased/quantized* resampling, and the cleanest version of that is observationally close to an action-head design with extra steps. Below I formalize it, prove the no-op, give the minimal fix, and lay out what it uniquely tests if you pursue it.

---

## 1. Precise formalization

### Substrate (shared template)
Factored Simplex with N=3 (or 4) Mess3 factors, each Q=3 hidden states, ternary emission. Observed token is the full N-tuple, V = 3^N (the `tuple_coupled` obs model in `~/comp_icl/generator.py`; for Design B set `eps=0` so factors are independent — see §6). Joint hidden state s ∈ {0,1,2}^N, S = Q^N. The generator already tracks per-factor hidden states `states[:,n]` and an exact joint belief `b` of shape (B, |H|, S).

The network (`~/comp_icl/model.py`, GPT, d_model=128, n_layer=4) predicts next token by SGD. By the Simplex result it linearly represents the factored belief: factor n's posterior over its current hidden state lives in an orthogonal subspace of the residual stream, decodable by ridge regression (this is exactly what `~/comp_icl/probe.py` already fits: `resid → factor_pred`, the per-factor next-subtoken distribution).

### The loop (literal reading of the spec)
Pretrain to convergence on the stationary factored process. Then run a **resampling rollout** to generate post-training data:

For each sequence, maintain true hidden states {s_1,...,s_N} as in `sample()`. Generate tokens autoregressively. Every k≈5 tokens, pick one factor m (round-robin or random) and:

1. **Read a distribution.** Take the model's represented posterior over factor m's *current hidden state*, `q_m ∈ Δ^{Q-1}`. Two sub-choices:
   - (a) read the **true** filtering posterior `b` marginalized to factor m (you have it from `forward()`), or
   - (b) read the **model's** posterior via a probe `P_m: resid_ℓ → Δ^{Q-1}` fit once after pretraining (frozen), evaluated on the live residual at the current position.
2. **Sample** a hidden state `ŝ_m ~ q_m`.
3. **Reset** factor m: `s_m ← ŝ_m`.
4. **Roll out** the next k tokens from all factors (m re-seeded, others continuing).
5. Repeat: pick a factor, read, sample, reset, roll.

Loss: next-token cross-entropy on the tokens produced by this resampled process. Question posed in spec — is another objective needed. (Answer: yes, or rather, the loop must change; see §3/§6.)

### Which distribution, which layer
- The belief over factor m's **current latent** is what you reset, so read it from the layer where the per-factor belief R² peaks (in comp_icl this is typically the last residual layer; `probe.py` reports per-layer R²). Important subtlety: the residual encodes the **next-emission** distribution / filtering posterior over the *current* state given the prefix — i.e. `factor_pred` = E ⋅ belief. You want belief over the state, not over the emission. The probe target must therefore be the marginal hidden-state posterior `b.sum over other factors`, NOT `factor_pred`. comp_icl computes `factor_pred = einsum(belief, E)`; you need the pre-emission belief, which `forward()` has as `b` (just add a marginalization). This is a one-line addition.
- Use a **frozen probe (option b)** for the causal loop, not the true posterior. Reason: the entire thesis is that the *self-factor* is the network's own internal readout. Re-seeding from the true posterior (option a) tests nothing about the model — it's an exogenous intervention. Re-seeding from the model's readout is what makes "the model's own belief causally re-seed the world." Keep option (a) only as a control / ceiling.

### What is consequential, what is the self-factor
- **Consequential internal quantity:** factor m's hidden-state posterior readout `q_m` at resampling steps. Its sampled value `ŝ_m` is written back into the world, so the *realized* next k tokens depend on it.
- **Self-factor (the modeled part):** the rest of the network should learn that, immediately after a reset, factor m's state is *exactly* `ŝ_m` (a delta), not the stationary prior — i.e. it should carry an "I just re-seeded m to ŝ_m" variable forward and use it to predict the next k tokens. That carried variable is the efference copy. Consequence flows through reseeding the world (no action head, no added channel into f).

---

## 2. The critical question: is this consequential or a self-consistent no-op?

**Claim: at convergence, with an unbiased readout and a faithful sampler, reseeding-from-own-belief leaves the data distribution unchanged ⇒ consequence ≈ 0.**

Proof sketch. Mess3 / the factored process is a stationary HMM. At time t the filtering posterior over factor m's state given the prefix is `q_m = P(s_m,t | x_{<t})`. The *actual* latent `s_m,t` is, by construction of filtering, a sample from this very posterior (the prefix was generated by the true chain; conditioning on it, the true state is distributed exactly as `q_m`). Therefore:

- **continue the chain** = keep `s_m,t`, which is one draw from `q_m`;
- **reset to a resampled state** = draw a fresh `ŝ_m ~ q_m`.

Both `s_m,t` and `ŝ_m` are i.i.d. draws from the *same* distribution `q_m`. Marginally over the randomness, the joint law of (state, future tokens) is **identical** whether you continue or resample. The resampled "world" is in-distribution with the original process. The label distribution the model must predict is unchanged. SGD has no new signal: the next-token target after a reset has the same conditional law as without one. **Consequence = 0.** The readout co-varies with the realized future only as much as the true latent already did in pure prediction — i.e. it's still pure prediction of an exogenous process. The loop is self-consistent.

This is the same degeneracy as the central-claim caveat: *if the posterior equals the true posterior, resampling from it draws from the same process.* Filtering is the fixed point that makes resampling a no-op.

### Exactly when it is NOT a no-op
The no-op breaks precisely when the resampled state is drawn from a distribution **different from the true conditional law of the latent given the prefix**. Concretely:

1. **Imperfect / biased readout (option b, realistic).** A learned probe `P_m` is never exactly `q_m`; it has finite-sample bias, is read at a layer that encodes the *emission* posterior not the *state* posterior, and is distorted by the linear-probe approximation. Reseeding from `P_m(resid)` ≠ reseeding from true `q_m`. The induced process differs from the original ⇒ nonzero consequence. **But** this is consequence-from-probe-error: its magnitude is the probe's KL from truth, which shrinks as the probe improves, and as the model's belief improves the probe target itself converges to truth. So consequence here is bounded by representational error and **vanishes in the well-trained limit** — fragile, and confounded with "probe is bad," which is the opposite of what you want to demonstrate.

2. **Sampling collapses uncertainty (the real lever).** If instead of *sampling* `ŝ_m ~ q_m` you take the **argmax** (`ŝ_m = argmax q_m`), you replace a distribution with a point mass. Now the reset state is NOT a draw from `q_m`; it is a deterministic function of it. The future is reseeded from a *sharpened* world. This genuinely changes the data distribution: high-entropy posteriors get forced to their mode, so the realized tokens after a reset depend on the readout in a way the original process never produced. **Consequence > 0**, and it does not vanish at convergence (a perfect belief still has nonzero entropy, so argmax still collapses it). This is the minimal modification (see §3).

3. **Reset semantics ≠ continuation.** "Reset factor m to state ŝ_m" applies the *prior/initial* transition convention, whereas "continue" applies the learned transition from the actual previous state. If your reset re-initializes m to stationary-then-step vs. step-from-ŝ_m, you inject a one-step distributional seam. This is a weak, transient effect (one token) and largely an implementation artifact, not a principled consequence source. Don't rely on it.

4. **Filtering-vs-latent mismatch under coupling (eps>0).** If factors are coupled (`eps>0`), the per-factor marginal posterior is NOT a sufficient statistic for that factor's latent (the latent is correlated across factors through the coupling). Resampling factor m from its *marginal* posterior ignores the cross-factor correlation, so the reset state is drawn from the wrong (marginalized) law ⇒ the joint process changes ⇒ nonzero consequence. This is a genuine, non-vanishing source. But it conflates "consequence" with "you used the wrong factorization of the posterior," which muddies interpretation.

**Bottom line:** the only *clean, non-vanishing, principled* consequence source is **#2 (argmax / temperature < 1 collapse)**. Faithful sampling from an unbiased readout is provably a no-op. State this plainly in any writeup.

---

## 3. Minimal modification that makes consequence nonzero (and meaningful)

Replace faithful resampling with a **deterministic or sharpened readout-to-state map**, and make the rollout *long enough that the seam is predictable*:

> At each reset, set `ŝ_m = argmax_s P_m(resid_ℓ)[s]` (or sample at temperature τ<1). Reset factor m to `ŝ_m`. Roll k tokens.

Why this is the right minimal change:
- It makes the realized future a **deterministic function of the model's readout** — exactly the "C's output co-varies with the realized next token" condition. The argmax readout *is* circuit C's output; the next k tokens are wired to it.
- It does **not** vanish at zero loss: even the Bayes-optimal posterior has entropy, so argmax always moves probability mass. Consequence is bounded below by the expected posterior entropy at reset points.
- The "self-factor" prediction becomes well-posed and learnable on-policy: after a reset, the optimal predictor must read `argmax q_m` (its own readout) and predict a *delta* at `ŝ_m`, then propagate Mess3 dynamics for k steps. Off-policy (feed sequences where resets used a *different* rule), this wiring mispredicts ⇒ consequence vanishes off-policy, matching the robustness claim.

An even cleaner variant for measuring consequence: make the reset **inject information that wasn't in the past** — e.g. `ŝ_m = argmax` but then *also emit a marker token* announcing a reset occurred (so the model can localize seams). This sharpens the efference-copy signal without changing the principle. Optional.

Note this still uses **only next-token CE loss** — no auxiliary objective is required once the readout map is non-faithful. The spec's "analyze whether another objective is needed" resolves to: *no extra objective; instead make the readout→state map lossy.*

---

## 4. How it differs from an action-head design

| | Action-head (e.g. Design A-style) | Design B (resampling) |
|---|---|---|
| Channel to the world | Explicit readout head → fed into combiner f | No head; world re-seeded by reading an *internal belief subspace* |
| What is "C's output" | The head's emitted action/symbol | The (sharpened) belief readout `q_m` |
| Consequence at convergence | Nonzero by construction (head value enters f) | **Zero unless readout map is lossy** (§2) |
| Gradient loop | Open (no backprop through env), label co-varies via f | Open, label co-varies via reseed — but only if §3 fix applied |
| Cleanliness | Consequence magnitude is a design knob (how strongly head feeds f) | Consequence magnitude = posterior entropy × readout lossiness; harder to control, partly confounded with probe error |
| What it uniquely tests | Whether SGD models an *explicitly emitted* action | Whether SGD models a *latent it had to infer and then re-injected* — closer to "self as privileged-access part of the world-model" |

The conceptual appeal of B over an action head: the consequential quantity is *not an arbitrary added output* — it is a belief the model already had to compute for prediction. So B tests the thesis in its purest form ("the self is the part of the world-model with privileged access to its own future") **iff** you can get nonzero consequence without it being an artifact. That "iff" is the whole risk.

---

## 5. What it uniquely tests vs other designs

- Whether a network, by ordinary next-token SGD, will build an **efference copy of its own inferred belief** when that belief is fed back into the world — *without* any explicit action channel. If yes, the self-factor is a genuine emergent consequence of on-policy co-variation, not of a hand-wired output.
- The **on-policy/off-policy robustness** prediction in the sharpest form: the self-wiring should be present only when evaluated on data generated by its *own* readout rule and absent off-policy.
- The **localization** prediction: the modeled self-variable should appear *after* reset points (positions t_reset+1 … t_reset+k) and decode the injected `ŝ_m`, distinguishable from the ordinary filtering belief.

---

## 6. Feasibility + reusable code

All of comp_icl is directly reusable; this is the strongest argument for B being cheap to try.

- **Generator / Mess3 / factored process:** `~/comp_icl/generator.py`
  - `mess3_operators(alpha,x)`, `Factor`, `CompositionMixture`. Set `eps=0` and use the per-factor states already tracked in `sample()` (`states[:,n]`, `nxt`). The resampling loop is a ~30-line modification of `sample()`: after every k steps, overwrite `states[:,m]` with `ŝ_m`.
  - **Belief / ground truth:** `forward()` maintains exact joint belief `b` (B,|H|,S) and returns `factor_pred` (emission posterior). You need the **state** posterior: add `b_state[:,t,n] = marginalize(b.sum(axis=1), factor n)` (the code already builds `self.G` and the factor decode table — reuse `decode` to marginalize). This is the read target / ceiling control.
- **Model:** `~/comp_icl/model.py` GPT with `return_hidden=True` — gives per-layer residuals for the probe.
- **Training loop / online data + oracle baseline:** `~/comp_icl/train.py` (`incontext_curve`, online `pool` sampling). Repurpose: the data source becomes the resampling generator instead of `mix.sample`.
- **Analysis (the core toolkit, already implemented):** `~/comp_icl/probe.py` — `ridge_fit` (resid→target R²), `subspace`, `overlap` (orthogonality), per-layer sweep, and a **causal-gating** routine (steer a belief direction, watch logits move) that is essentially the consequence-measurement primitive. Reuse `ridge_fit` for the frozen probe `P_m`.
- **Papers:** `~/papers/factored-representations_2602.02385.txt` (orthogonal factor subspaces, vary-one analysis, effective-dim — §H), `belief-state-geometry_2405.15943.txt`, `constrained-belief-updates_2502.01954.txt`.

What must be built new (small):
1. State-posterior marginalization in `forward()` (~5 lines).
2. Frozen probe `P_m` (one `ridge_fit` call post-pretrain, save W).
3. Resampling rollout: wrap `sample()` to (i) run the model online to get residuals, (ii) apply `P_m`, (iii) argmax/sample, (iv) reset `states[:,m]`, (v) continue. This requires running the model *inside* data generation (GPU forward passes interleaved with sampling) — the one genuinely new engineering piece. Batch it: generate k tokens for the whole batch, forward the batch, read residuals at the reset position, reset, repeat.
4. A consequence estimator (§4 below).

Compute: model is tiny (d=128, 4 layers, V=27). Pretrain ~8k steps as in train.py (minutes on one H100). The resampling rollout is the cost driver because it interleaves model forwards with sampling, but at L=64, k=5, batch 512 it's still small. **Single H100 is plenty.** Box check: all 8 GPUs free (1 MiB, 0%), /data has 14T free. Set `HF_HOME=$PWD/.hf_home` is moot (no HF downloads). No GPU job launched (scoping pass).

---

## 7. Analysis plan

1. **Consequence measurement (primary).** Define consequence of the readout circuit C_m via intervention: at a reset point, do/operator on the readout (force `ŝ_m` to a different admissible value), regenerate the next k tokens, measure the shift in the realized-token distribution (KL or total-variation), averaged. In the no-op regime this is ≈0; under the argmax fix it is bounded below by posterior entropy. **This single number is the experiment.** The `probe.py` causal-gating code is the template.
2. **Self-factor decoding.** Post-reset, probe whether residuals at positions t_reset+1..+k linearly decode the *injected* `ŝ_m` (a delta) better than they decode the ordinary filtering belief. Use `ridge_fit`; compare R² of (resid → onehot(ŝ_m)) vs (resid → stationary prior). Emergence of high R² for the former = efference copy learned.
3. **On/off-policy ablation.** Evaluate the trained model on (a) its own resampled data, (b) data resampled with a *different* readout rule, (c) the original stationary process. Self-factor decoding R² and consequence should be high on (a), collapse on (b),(c).
4. **Orthogonality / geometry.** Re-run the factored-subspace overlap analysis (`overlap`) to confirm the re-seeded factor's belief still occupies its own subspace, and locate the self-variable subspace relative to it.
5. **Belief-fidelity ceiling control.** Repeat with option (a) (reset from *true* posterior, faithful sample): predict consequence ≈ 0 and no self-factor. This is the negative control that proves the effect is about the readout, not about resetting per se.

---

## 8. Expected results + falsifiers

**If the frame is right (with the §3 argmax fix):**
- Consequence(C_m) > 0 and stable across training (does not decay to 0 as loss falls).
- Post-reset residuals decode the injected `ŝ_m` (delta) with high R²; this *increases* over training — the model learns the efference copy.
- On/off-policy: self-decoding present on own-policy data, absent off-policy and on the stationary process.
- Faithful-resample control (option a): consequence ≈ 0, no self-factor.

**Falsifiers / kill criteria:**
- **Faithful-sampling version shows consequence ≈ 0 and no self-factor** — this is *expected* (§2) and is the falsifier of the naive design, not of the thesis. If even the argmax version shows ≈0 consequence and no post-reset delta-decoding, the thesis (SGD models self-factors on-policy) is challenged for this mechanism.
- If post-reset the model predicts from the *stationary prior* rather than `ŝ_m` (ignores the reset), no efference copy formed → falsifier.
- If consequence magnitude tracks probe error (shrinks as probe R² → 1) rather than posterior entropy, the "consequence" is an artifact (case §2.1), not the real effect → falsifier.

---

## 9. Pitfalls / confounds (esp. no-op risk)

1. **The no-op (dominant risk).** Faithful resampling from an accurate belief = drawing from the same process = zero consequence (§2). Anyone running the literal spec will see nothing and may wrongly conclude the thesis is false. **Mitigation: lead with the argmax/temperature fix; keep faithful-resample as the explicit null control.**
2. **Consequence-as-probe-error confound.** With a learned probe, any nonzero consequence may just be probe inaccuracy, which *vanishes* as representations improve — backwards from the thesis. Mitigation: use argmax (lossy by construction, not by error), and report consequence vs posterior-entropy correlation, not vs probe-error.
3. **Read target mismatch.** The residual encodes the *emission/filtering* posterior; you must marginalize to the *state* posterior to reset correctly. Resetting from the emission posterior introduces a wrong-distribution seam that looks like consequence but is a bug.
4. **Train/eval distribution shift.** The resampling process is non-stationary at seams; the in-context-loss baseline from train.py assumes stationarity. Recompute oracle baselines on the resampled process or comparisons are meaningless.
5. **Coupling confound (eps>0).** Don't enable coupling thinking it adds consequence "for free" — it adds consequence via filtering-vs-latent mismatch (§2.4), which conflates the effect with using the wrong posterior factorization. Keep eps=0 for the clean test.
6. **Interleaved model-forward cost / nondeterminism.** Running the model inside data gen makes data depend on current weights → moving target, possible instability. Mitigate by generating data with a *frozen* snapshot of the model for a window of steps (on-policy but lagged), refreshing periodically.

---

## 10. Verdict

**Conditional go, with eyes open.** The design as written (faithful resample from own belief) is **provably a self-consistent no-op** — its single most important property, and a result worth stating cleanly as the null. The design becomes a genuine test of the self-model thesis only under the minimal fix of §3: **a lossy readout→state map (argmax / τ<1), eps=0, reset the *state* posterior, frozen-snapshot data generation.** Under that fix it uniquely tests whether SGD builds an efference copy of an *inferred latent it re-injected*, with no explicit action channel — the purest operationalization of "the self is the privileged-access part of the world-model."

But note the honest comparison to an action head (§4): once you sharpen the readout into a deterministic value fed back to the world, Design B is mechanistically close to "an action head whose action happens to be the argmax of an existing belief subspace." Its distinctive value over an action-head design is *conceptual* (the consequential quantity is endogenous to prediction, not added), and that value survives only if the analysis cleanly separates real consequence (entropy-bounded, on-policy, persistent) from artifacts (probe error, distribution-shift seams). Recommend running it **as the second design**, after an action-head design establishes the positive control, and primarily to demonstrate that the self-factor emerges even when consequence is routed through an *endogenous* belief rather than a bolted-on output. Cheap (single H100, minutes-to-hours, all code reusable), so worth doing — but budget the analysis effort, not the compute, since the entire result lives in disentangling consequence from confounds.

### Concrete next GPU step (recommended, not run)
On one free GPU: (1) pretrain factored Simplex (eps=0, N=3) via `train.py` (~8k steps); (2) fit frozen state-posterior probe with `ridge_fit`; (3) implement argmax-reset rollout wrapping `sample()`; (4) fine-tune/continue-train on resampled data; (5) measure consequence (intervention KL) and post-reset `ŝ_m` decoding vs the faithful-resample null control.
