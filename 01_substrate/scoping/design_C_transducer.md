# Design C — Alternating Action/Observation Transducer (agent in an environment)

Scoping report. Self-models-via-consequence project. Verdict up front: **this is the
strongest "true agency" design of the family, and it is the one design where consequence
is structurally guaranteed > 0 rather than engineered in. Build it — but the headline
risk (efference cancellation / policy collapse) is not a bug to avoid, it is the central
phenomenon to instrument. Recommend running it second, after a simpler design has
validated the analysis pipeline, because the degeneracy makes it the hardest to read.**

---

## 1. Precise formalization

### 1.1 The transducer (input-driven HMM / Mealy machine)

Standard Simplex generators are **autonomous** edge-emitting HMMs: labelled operators
`T[x][s,s']  = P(emit x, next s' | s)`, and the only "input" is the previous emission fed
back as the next position's token. A **transducer** adds an exogenous input symbol per
step. Formally a Mealy HMM / input-driven HMM with:

- hidden states `s ∈ S`, `|S| = m`
- input (action) alphabet `a ∈ A`, `|A| = k`
- output (observation) alphabet `o ∈ O`, `|O| = v`
- **input-conditioned labelled operators**

  `T^{(o|a)}[s,s'] = P(emit o, next state s' | current state s, input a)`

  one (m×m) matrix per (a,o) pair; for each (s,a), `Σ_{o,s'} T^{(o|a)}[s,s'] = 1`.

This is exactly the "edge-emitting / Mealy HMM" of Piotrowski et al. (constrained belief
updates, 2502.01954, line 160: "edge-emitting hidden Markov model (Mealy HMM)") and the
POMDP framing they cite (Kaelbling 1998; their line 138). The autonomous Mess3 generator
in `~/comp_icl/generator.py` is the **special case** `A = O` with the input ignored except
as the fed-back previous token — i.e. our existing code is already a degenerate transducer.

**Belief over transducer hidden states.** The optimal predictor of `o_t` is Bayesian
over `s`. Given an input/output history, the belief updates as

  `b_t(s')  ∝  Σ_s b_{t-1}(s) · T^{(o_t | a_t)}[s, s']`

i.e. the **input selects which operator** is applied, then the observed output relabels
it — identical structure to `generator.forward()` (lines 224–255), where today the token
both selects and labels. Here the action `a_t` selects the operator (`T^{(·|a)}`) and the
observation `o_t` indexes the labelled output. The mixed-state presentation (set of
reachable beliefs) is now a function of the **policy** that drives the inputs — a
*controlled* MSP. This is the key new object: belief geometry conditional on the agent's
own action distribution.

Relation to GHMM-with-input / POMDP: this is precisely a POMDP's observation+transition
kernel `O(o|s',a) · P(s'|s,a)`, fused into one labelled operator per (a,o). With a reward
it is a full POMDP; without reward (pure prediction) it is an "input-driven HMM" /
transducer, and the predictor's belief is the POMDP belief-state.

### 1.2 Concrete small example — "noisy controllable switch" (recommended seed)

Pick the smallest generator that makes consequence non-trivial and belief-tracking
non-degenerate:

- **States** `S = {L, R}` (m=2): a latent two-mode environment.
- **Actions** `A = {hold, flip}` (k=2): emitted by the network.
- **Observations** `O = {0, 1}` (v=2).

Dynamics (one tunable noise `η ∈ (0,½)` for transition, one `ε` for emission):

- `flip`: deterministic-ish toggle of mode, `P(s'=¬s|flip) = 1-η`, `P(s'=s|flip)=η`.
- `hold`: persist, `P(s'=s|hold) = 1-η`, `P(s'=¬s|hold)=η`.
- emission tied to mode: `P(o = [s==R] | s) = 1-ε`, else `ε`.

So `o_t` depends on `s_t` which the agent steers through `a_t` — **token-level
consequence is direct: the realized `o_t` co-varies with the just-emitted `a_t`.** This
is the cleanest possible instantiation of the consequence quantity: the next *realized*
token literally has the action in its causal cone.

Scale knobs once the pipeline works: m=3 (Mess3-style ternary mode), k=3 actions, v=3
observations → still tiny (operators 3×(3×3)=27 entries), exact oracle trivial.

**Factored variant (matches the shared template).** N independent transducers, each
`(S_n, A_n, O_n)`, the network emits an action *tuple* `a_t = (a_t^1,…,a_t^N)` and observes
the observation tuple `o_t`. Fixed combiner `f` over sub-observations (reuse
`tuple_coupled` / the kron construction in `_build_operators`). Each transducer's belief
should live in an orthogonal residual subspace exactly as in factored-representations
(2602.02385). This is the version that plugs straight into the existing probe suite.

### 1.3 Sequence / loss

Interleaved stream `a_1, o_1, a_2, o_2, …, a_T, o_T`. Vocabulary is the **union**
`A ⊔ O` (disjoint id ranges) so the model can emit actions and predict observations in one
head; positions alternate by parity. Two training regimes (Sec. 2).

---

## 2. What this design uniquely tests

### 2.1 Supervised-on-observations vs RL

**Supervised on observations (recommended primary).** Loss = cross-entropy on
**observation positions only** (mask out action positions, or weight them 0). The network
predicts `o_t` from the prefix that *includes its own emitted `a_t`*. Action tokens are
**sampled** from the model's own distribution at action positions (the generation loop,
Sec. 4) and are **discrete / non-differentiable** — no gradient flows back through the
sampling op into "how the action was chosen." This is the crux:

> The map the network learns under this loss is the **forward** map `a_t, history → o_t`.
> It is *not* learning to choose actions to make observations better (that would need a
> gradient through the action choice, i.e. closing the loop). It is learning to *read its
> own emitted action as a cause of the upcoming label*. That is exactly the project's
> "consequence makes C's output co-vary with the realized next token, so SGD wires the
> prediction to read it — without closing the gradient loop" claim, made literal at the
> token level.

So supervised-on-observations is the **purest test of the central mechanistic claim**: an
efference copy of the action emerges purely to lower forward prediction loss, with the
action-generation path held off-policy w.r.t. the gradient.

Caveat that *strengthens* the test: with masked-out action positions, the action head is
untrained by the prediction loss → actions are whatever the (frozen/random/exploratory)
policy emits. You must specify the action policy explicitly (Sec. 2.3, Sec. 4).

**RL.** Adds a reward and trains the action head (policy gradient / REINFORCE, since
actions are discrete). Sensible rewards:
- (a) **Predictability / negative surprise**: `r_t = log p_model(o_t | ·)` — reward the
  agent for making its own observations predictable. This is the natural "minimize
  self-induced surprise" objective and directly induces the degeneracy of Sec. 5.
- (b) **Task reward**: e.g. "keep the latent mode at R" → `r_t = [s_t == R]`, or "reach a
  target observation." External, not predictability — avoids the trivial collapse and
  gives the agent a *reason* to act that is independent of its predictor.

**Which better tests the consequence frame:** *supervised-on-observations* tests the
**representation** claim (efference copy emerges from prediction alone) most cleanly and
cheaply, and is on-method (next-token SGD, no loop closed). *RL with a task reward* tests
the **stronger** claim that a self-model is functional for control — but it confounds
"self-model from consequence" with "self-model because the RL objective demands it,"
which is a different (and weaker-for-our-thesis) reason to model the self. **Primary =
supervised-on-observations. RL(task) = an optional second arm to show the self-factor is
not an artifact of the prediction loss.** RL with predictability reward = the *probe* for
the cancellation prediction, not a main result.

### 2.2 The self-factor here

To predict `o_t` the network must compute `P(o_t | history, a_t)` = it must (i) maintain
the transducer belief `b_{t-1}`, (ii) **read its own just-emitted action `a_t`**, (iii)
apply the input-conditioned operator. Step (ii) is the **efference copy**: an internal
representation of "what I just did," used to condition the world-model update. The
self-factor = the action variable re-represented at/after the action position and consumed
at the observation position.

Where it lives / how to detect:
- It must be present in the residual stream **at the observation position** (the position
  whose loss depends on `a_t`). Decode `a_t` linearly from residual activations at
  observation positions (later positions in particular — does it persist / get bound to
  belief?). High decode R²/accuracy = efference copy present.
- Mechanistically: an **attention head at the observation position attending back to the
  immediately preceding action position** is the predicted carrier. Per constrained-belief
  (2502.01954), one attention layer does the forward-propagation `T^{d−s}`; here that
  operator should be **input-conditioned** — i.e. the head's effective OV/QK should gate on
  the action token. Detect via attention pattern (obs→prev-action edge) + activation
  patching the action token (Sec. 4 control).
- The decisive signal: **the belief update at obs positions is conditioned on the action**.
  Regress residual → oracle belief `b_t`; show the regression is only accurate when the
  action is in scope (the off-policy / patched-action controls break it).

### 2.3 The degeneracy / "regularize the impact" prediction — feature, not bug

If actions are **free** and loss is observation-**predictability** (RL reward (a), or even
supervised if the action head is trained jointly), the policy can collapse to a
**fixed/constant action** that drives the environment into its most predictable regime
(e.g. always `hold` until the mode is pinned, ε→ as low as the noise floor). The
self-induced surprise is minimized by *removing the self's influence's unpredictability* —
**efference cancellation**: the optimizer cancels the part of the future it controls.

This is a **predicted feature** of the consequence frame (the model minimizing
self-induced surprise = exactly what a predictor that has internalized its own consequence
would do), but it is **experimentally fatal if uncontrolled** — a constant policy makes
`a_t` carry no information, consequence becomes unmeasurable, the efference copy has nothing
to represent. So:

How to tell collapse-as-prediction from collapse-as-degenerate-artifact:
- Measure **action entropy** over training. Collapse = entropy → 0.
- Measure **realized observation entropy / surprise** vs. action entropy: the prediction
  says surprise drops *because* the agent steered into a predictable regime, not because the
  predictor got better at a fixed-difficulty stream. Compare to the **oracle floor for the
  realized policy** (recompute the controlled-MSP myopic entropy for the actual policy) — if
  the model tracks the *policy-conditioned* floor, the drop is genuine steering, not slack.

How to keep the experiment informative (pick one or combine):
1. **Inject action stochasticity / fixed exogenous policy** (primary, supervised arm):
   actions sampled from a *fixed, high-entropy* policy (uniform or a fixed random HMM),
   NOT the trained head. Guarantees `a_t` carries information → consequence is measurable →
   efference copy must be learned. This is also the natural **off-policy control** (Sec. 4).
2. **Constrain the policy**: entropy bonus / KL-to-uniform regularizer on the action head
   (RL arm) so it cannot fully collapse.
3. **Reward against collapse**: use task reward (b) instead of predictability, so acting
   matters for a reason orthogonal to predictability.
4. **Make the controllable part not the only part**: add an exogenous (uncontrolled)
   sub-process to the observation (a factor the agent can't touch). Then "predictability"
   has an irreducible floor and the agent can only cancel the part it owns — which is
   precisely the consequence we want to localize. This is the cleanest design choice.

Recommendation: **primary supervised arm uses a FIXED exogenous high-entropy policy**
(option 1) — this both guarantees measurability and *is* the on/off-policy control axis.
The trained-policy / collapse experiment is a deliberate, separately-run probe of the
cancellation prediction (option 2 or 3), not the main measurement.

---

## 3. Feasibility + reusable code

Everything needed exists in `~/comp_icl` and adapts with modest edits. The generator is
**already a near-transducer**; the model/train/probe stack is directly reusable.

| Need | Reuse | Path | Edit required |
|---|---|---|---|
| Labelled operators `T^{(o|a)}`, exact Bayes belief | `mess3_operators`, `CompositionMixture._build_operators`, `forward` | `~/comp_icl/generator.py:47,132,224` | Generalize operator key from `(comp,x)` to `(a,o)`; belief update selects operator by action, labels by obs. ~½ day. |
| Factored kron construction / orthogonal subspaces | `_build_operators` tuple_coupled branch, `G` readout | `generator.py:139,182` | Reuse as-is for factored variant. |
| GPT decoder (TransformerLens-free, has `return_hidden`) | `GPT` | `~/comp_icl/model.py` | Vocab = `|A|+|O|`. Add observation-position loss mask. Trivial. |
| Online next-token training loop, oracle floor compare | `train.py` (`next_batch`, `incontext_curve`, oracle floors) | `~/comp_icl/train.py:84,30,95` | Add (a) generation/sampling loop for action positions (autoregressive rollout), (b) loss mask on obs positions only. The autoregressive rollout is the one genuinely new piece — see Sec. 4. ~1 day. |
| Linear belief regression (MSE/R² per layer), subspace orthogonality, causal steering | `ridge_fit`, `subspace`, `overlap`, gating hook | `~/comp_icl/probe.py:18,27,36,116` | Targets become transducer belief + **decode-the-action** probe. Reuse wholesale. |
| Constrained-vs-full belief / per-stage regression | `M_structural`, factor-belief readout | `~/comp_icl/metrics.py:46` | Reuse; "constrained" = attention-only forward-propagation belief (cf. 2502.01954). |
| OOD / retention probe | `ood_probe.py` | `~/comp_icl` | Reuse for on-policy vs off-policy retention. |
| Exact myopic-entropy / loss floor for the *controlled* MSP | `myopic_entropy` | `generator.py:257` | Must be recomputed **per policy** (policy-conditioned floor) — the key new oracle. |

External references on disk (context only, no code): `~/papers/external/disentanglement-insufficient`,
`explainable-composition-circuit`; circuit-discovery PDFs in `~/papers/circuits-discovery`
(ACDC, EAP, sparse feature circuits) if mechanistic localization of the efference head is
wanted. Papers `belief-state-geometry_2405.15943`, `constrained-belief-updates_2502.01954`
(Mealy/POMDP framing, lines 138/160), `factored-representations_2602.02385` all on disk as
.txt.

**New code to write** (small): (1) transducer generator subclass; (2) policy-conditioned
oracle (recompute beliefs/floors under a given action distribution — needed because the
MSP depends on the policy); (3) autoregressive rollout that interleaves model-sampled (or
fixed-policy) actions with transducer-sampled observations; (4) action-decode probe (a
2-line addition to `probe.py`). Env: existing `~/comp_icl/.venv` (uv). No TransformerLens
needed unless doing head-level circuit work.

GPU: all GPUs currently free (verified read-only `nvidia-smi`: 0–7 all ~1 MiB / 0%). Per
box rules, **no training launched in this scoping pass.** Models are tiny (d=128, ≤6
layers, ≈0.5–2M params, vocab ≤ ~10–30); a single H100 with `CUDA_VISIBLE_DEVICES=<one
free idx>` is ample. Each run is minutes-to-low-hours.

---

## 4. Generation loop & on/off-policy control

The new structural piece is that the data is **partly model-generated** (action positions)
and partly **environment-generated** (observation positions). Two ways to realize it:

**A. True autoregressive rollout (on-policy).** At each action position, run the model
forward over the prefix, sample `a_t` from its action-head distribution (restricted to the
`A` id range), append; the transducer (numpy) consumes `a_t`, samples `(o_t, s_{t+1})`,
append `o_t`; repeat. Sequences are generated *on the model's current policy*. Loss =
CE on obs positions of the rolled-out sequence. This is the on-policy regime where
consequence is real and (per thesis) the self-factor is robust. Cost: sequential
generation (T steps × forward pass) — batch it; T≤64, tiny model → cheap. Reuse nothing
from `next_batch`; write a `rollout()` that mixes torch-sampling with the numpy transducer
step (the transducer's vectorized `sample` loop, `generator.py:186`, is the template).

**B. Fixed exogenous policy (off-policy control + the recommended primary supervised
setup).** Actions sampled from a **fixed** distribution (uniform, or a fixed independent
HMM/another network), independent of the model being trained. Data can be **pre-generated
in bulk** (no rollout, reuse `next_batch` directly) → much cheaper and stabler. This is
both: (i) the way to guarantee action information (Sec. 2.3 option 1), and (ii) the
**off-policy control**: train/evaluate the *same* prediction objective on a stream whose
actions come from a fixed other policy.

**The on-policy vs off-policy contrast is the project's central robustness prediction:**
- On-policy (rollout from this net, or fixed policy *matched* to test): efference copy
  present, belief update conditioned on action, consequence > 0.
- Off-policy stress test: train on fixed-policy-A data, **evaluate / probe on
  fixed-policy-B** (different action distribution). Thesis predicts the efference-copy
  mechanism (read `a_t`, apply `T^{(·|a)}`) transfers (it's the *forward* map, policy-
  agnostic), whereas any *anticipatory* "what will I do next" representation should NOT
  transfer. Decoding "next action `a_{t+1}`" from pre-action positions and showing it
  collapses off-policy while "current action `a_t`" decode at obs positions survives = the
  clean separation of self-model-of-consequence (robust) from policy-prediction (fragile).
  This operationalizes "robust because on-policy; off-policy it vanishes."

Recommended sequencing: do **B (fixed policy)** first — it isolates the representation
claim with the cheapest, most stable setup and gives the off-policy control for free. Add
**A (rollout)** to test that the on-policy self-factor matches, and to run the collapse
probe.

---

## 5. Analysis plan, expected results, falsifiers

### Analysis (all reuse the `probe.py`/`metrics.py` toolkit)
1. **Belief recovery (primary).** Linear regression residual → oracle transducer belief
   `b_t`, MSE/R² per layer, at observation positions. Compare to the **policy-conditioned**
   oracle (Sec. 3 new code). Expect R²→~1, sharpening with depth.
2. **Efference-copy decode.** Linear decode of `a_t` from residual at observation
   positions (and later positions). Expect high accuracy at obs positions; this is the
   self-factor.
3. **Action-conditioning of the belief update (the consequence signature).** Show the
   obs-position belief regression is accurate *only with the action in scope*: activation-
   patch the action token (swap `a_t` for `a'_t`) and show the read-out belief and the
   prediction move to `T^{(·|a')}`'s prediction. This is consequence made causal — the
   exact analogue of `probe.py`'s gating hook (line 116), retargeted to the action.
4. **Subspace structure (factored variant).** Per-transducer belief subspace orthogonality
   (`overlap`, `subspace` in probe.py); efference copy of each sub-action should align with
   its transducer's subspace.
5. **Constrained vs full belief** (cf. 2502.01954): attention-only forward-propagation
   approximation vs MLP-completed belief, per stage — does the action-conditioned operator
   live in attention?
6. **On- vs off-policy** (Sec. 4): repeat 1–3 under policy shift; decode `a_t` (current,
   should survive) vs `a_{t+1}` (next, should collapse off-policy).
7. **Collapse instrumentation** (rollout/RL arm): action entropy, realized-obs surprise vs
   policy-conditioned floor, over training.

### Expected results (if thesis holds)
- Efference copy (decode `a_t` at obs positions) emerges as loss drops; obs-loss reaches
  the policy-conditioned oracle floor.
- Belief update at obs positions is causally action-conditioned (patching action reroutes
  prediction to the swapped operator's distribution, high alignment).
- Off-policy: `a_t`-decode + action-conditioned update **transfer**; any next-action /
  anticipatory representation does **not**.
- Factored: orthogonal per-transducer belief subspaces; per-sub-action efference copy in
  the matching subspace.
- Trained-policy arm: action entropy collapses toward the predictable regime (efference
  cancellation) and the model tracks the policy-conditioned floor — confirming the
  cancellation prediction.

### Falsifiers (what would refute the design's claim)
- `a_t` is **not** decodable / **not** causally used at obs positions even though obs-loss
  is near-optimal (would mean the model predicts `o_t` without an efference copy — e.g. by
  inferring `s_t` from observation history alone, ignoring the action). **This is the
  sharpest confound** (Sec. 6) and must be designed out by making the action *necessary*
  for prediction (short observation history insufficient to pin the mode; high transition
  noise so only the action disambiguates).
- Patching the action does **not** move the prediction toward `T^{(·|a')}` → no consequence
  in the computation.
- On-policy and off-policy are **indistinguishable** in the anticipatory probe → the
  "robust on-policy / vanishes off-policy" claim fails (the design then doesn't separate
  self-of-consequence from generic world-modeling).
- Collapse happens but model loss does **not** track the policy-conditioned floor (drop is
  optimizer slack, not steering) → cancellation reading is wrong.

---

## 6. Pitfalls / confounds

1. **The action may be redundant with observation history.** If the latent mode is
   inferable from past observations alone, the model can predict `o_t` *without* reading
   `a_t`, and no efference copy is needed — destroying the test. **Mitigation: design the
   transducer so the action is the dominant disambiguator** — high transition noise on the
   uncontrolled dynamics, action strongly determines the transition, short effective
   observation memory. Verify with an oracle ablation: compare oracle belief with vs
   without conditioning on actions; require a large myopic-entropy gap.
2. **Policy collapse making `a_t` uninformative** (Sec. 2.3) — handle via fixed exogenous
   high-entropy policy in the primary arm.
3. **Action-position loss leakage.** If action positions are not masked out (or the action
   head is trained when it shouldn't be), the "no loop closed" claim is violated. Be
   explicit: supervised arm masks action-position loss → gradient never shapes action
   choice.
4. **Position parity as a shortcut.** The model could use absolute position parity rather
   than token type to know "this is an obs position." Harmless for the main claim but
   confounds attention-pattern reads; control by jittering / using type embeddings.
5. **Policy-conditioned oracle is mandatory.** Using the *uniform*-policy floor to judge a
   *collapsed*-policy run will mis-attribute the loss drop. The floor must be recomputed for
   the realized action distribution.
6. **REINFORCE variance** (RL arm) — high; needs baseline/entropy bonus, more compute, and
   is the confounded arm. Keep it secondary.
7. **"True agency" overclaim.** This design has genuine token-level consequence (the
   realized next token depends on the emitted action) — the strongest of the family. But
   *agency in the decision-theoretic sense* (acting toward a goal) only exists in the RL
   arm; the supervised arm has consequence without purpose. Be precise in claims:
   supervised = "the predictor models its own causal influence"; RL(task) = "the agent
   acts and models its influence." Both are stronger than the autonomous-HMM designs where
   consequence is zero by construction.

**Assessment of the "cleanest true-agency setup" framing:** Largely correct, with one
sharpening. It is the cleanest setup for **consequence** (the realized future provably
depends on the model's output — consequence > 0 by construction, vs. designs that must
engineer covariance). It is *not* automatically the cleanest for an emergent self-model,
because the very directness of the action→observation link means the model can exploit it
*or* route around it (confound #1), and the predictability objective invites collapse
(#2). So: cleanest *consequence*, but the analysis is the **most delicate** of the family
precisely because consequence is so strong it can hide (cancellation) or be bypassed
(redundancy). The other designs trade weaker/engineered consequence for easier reads. That
trade-off is why I recommend running this **second**.

---

## 7. Scope (model / data / compute / time)

- **Model:** GPT from `model.py`, d=128, n_layer ∈ {2,3,4,6}, n_head=4, max_len=128
  (T≤64 interleaved → 128 positions). ≈0.5–2M params.
- **Data:** seed transducer m=2,k=2,v=2 (Sec. 1.2); scale to m=3,k=3,v=3; factored variant
  N=2–4 transducers. All exact-oracle-tractable (joint state ≤ a few hundred).
- **Compute:** single H100, `CUDA_VISIBLE_DEVICES=<one free idx>`. Supervised/fixed-policy
  runs: minutes each (cf. existing `train.py` 8k steps online). Rollout (on-policy) runs:
  sequential generation overhead, still well under an hour batched. RL arm: a few× more,
  higher variance. Run in `tmux`, `… 2>&1 | tee logs/run_$(date ...).log`. Set
  `HF_HOME=$PWD/.hf_home` (no external model downloads expected; irrelevant here).
- **Time to results:** generator + policy-conditioned oracle ≈ 1 day; train/loss-mask +
  fixed-policy supervised arm ≈ 1 day; probes (reuse) ≈ 0.5 day → **first belief +
  efference-copy + off-policy results in ~2.5–3 days.** Rollout + collapse probe +
  factored variant + optional RL arm ≈ +3–4 days.
- **Out of scope (this design):** circuit-level localization of the efference head (could
  follow with ACDC/EAP from `~/papers/circuits-discovery`); large vocab; learned
  combiner `f`; multi-agent / two networks acting on each other.

---

## 8. Verdict

**Build it — as the deepest test of the consequence thesis, run second.**

- It is the **only** design where consequence is *structurally guaranteed* (the realized
  next token literally contains the just-emitted action in its causal cone), rather than
  engineered via correlation. That makes it the definitive test of "high-consequence
  circuits get modeled" — here the high-consequence circuit is the action emission, and the
  predicted self-factor is the efference copy.
- The **supervised-on-observations + fixed-exogenous-policy** arm is the recommended
  primary: it is exactly on-method (next-token SGD, action path off-policy w.r.t. the
  gradient → "no loop closed"), it guarantees action information, and it hands you the
  off-policy control for free. This is the cleanest realization of the central claim in the
  whole family.
- The **efference-cancellation / policy-collapse** prediction is a genuine, falsifiable,
  *novel* consequence of the frame — treat the trained-policy/RL-predictability arm as a
  dedicated probe of it (with action entropy + policy-conditioned floor instrumentation),
  not as a main training regime.
- Highest scientific payoff, highest analysis risk. Two confounds must be designed out
  before any GPU time: (1) action redundancy with observation history (make the action the
  dominant disambiguator; verify via oracle action-ablation gap), and (2) policy collapse
  (fixed high-entropy policy in the primary arm). Both are addressable with generator-design
  choices, no new ML.
- ~90% of the stack is reusable from `~/comp_icl` (generator/oracle, GPT, train loop, full
  probe/orthogonality/gating suite). New code is a transducer generator subclass, a
  policy-conditioned oracle, an autoregressive rollout, and a one-line action-decode probe.

Recommended order across the family: validate the analysis pipeline on a simpler
(stronger-read, weaker-consequence) design first, then run Design C as the capstone "true
consequence" experiment.
