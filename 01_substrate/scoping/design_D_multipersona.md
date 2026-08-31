# Design D — Multi-persona transducer (in-context self/other inference)

Scoping report. Self-contained. Builds on Design C (action/observation transducer).
Verdict up front: **high-value, highest-risk. DEFER the full version; build it as a
staged extension of C. A minimal "self vs. one fixed other" version is the right
first cut and is genuinely informative on its own.**

---

## 0. One-paragraph thesis fit

The consequence frame says the "self" is the part of the world-model with privileged
access to its own future, learnable because the model's own output co-varies with the
realized next token. Designs A–C build the *machinery* (a transducer whose observations
depend on the emitted action, so consequence > 0). Design D adds the *population*: on a
given trajectory the actions come from one of K policies, one of which is the network's
own policy. This is the only design in the family that directly instantiates the two
established empirical results — **self-recognition** (read your own generations at lower
surprise) and **Assistant-as-privileged-persona** (every voice read against one canonical
origin) — in a fully synthetic world with ground-truth beliefs. That is its unique payoff
and the reason to build it, despite being the hardest.

---

## 1. Precise formalization

### 1.1 Transducer (inherited from C)
A finite **input-driven HMM** (Mealy/Moore GHMM) `G` with hidden state `s ∈ S`, action
alphabet `A`, observation alphabet `O`. Labelled operators
`T^{a}_{o}[s, s'] = P(o, s' | a, s)` — the action `a` selects the operator stack, the
HMM transitions and emits `o`. (Compare `generator.py:mess3_operators`, which already
builds emission-labelled operators `T[z,i,j]=P(emit z, next j | i)`; the transducer just
adds an action index to make it `T[a,z,i,j]`.)

A trajectory interleaves action and observation tokens:
`a_1, o_1, a_2, o_2, …, a_L, o_L`, with `s_{t+1} ~ T^{a_t}_{o_t}` and `o_t` emitted jointly.

### 1.2 Policies / personas
A library of K **action-policies** `{π_1, …, π_K}`, each a (possibly history-dependent)
conditional `π_k(a_t | h_t)` where `h_t = (a_{<t}, o_{<t})`. Concretely each `π_k` is
itself a small HMM-controller over the action alphabet — distinct dynamics give distinct,
behaviorally-identifiable personas (reuse the `Factor(alpha, x)` family with different
params as the policy generators — they are cheap, ergodic, and already coded).

**One distinguished policy `π_self`**: this is the *model's own action distribution*,
i.e. the network is also tasked with producing actions on some trajectories (or, in the
minimal version, `π_self` is a fixed policy whose realized actions are fed back so the
model has efference access to them — see §3.3).

### 1.3 Latent "who is acting"
Per trajectory: draw `k* ~ Cat(ρ)` once, fixed for the trajectory (the
fixed-per-sequence latent, exactly the structure of `comp_idx` in
`CompositionMixture.sample`). All `a_t` on that trajectory are drawn from `π_{k*}`. The
model must infer `k*` in context from the (action, observation) stream. Because `k*` is
fixed per sequence and revealed only gradually by accumulated actions, this is a genuine
multi-sample in-context inference (same epistemic shape that made the composition latent
non-trivial in `comp_icl`).

### 1.4 The two prediction targets, and where consequence lives
The model emits a distribution over the next token at every position. Two regimes:
- **Predicting `o_t`** (an observation): pure prediction of the transducer given the
  inferred state and the just-seen action. Consequence of the *observation*-predicting
  circuit = 0 (it predicts an exogenous-given-action process).
- **Predicting `a_t`** (an action): here the realized future depends on what is emitted
  **iff** `k* = self` and the emitted action is fed into the transducer (closed loop).
  This is where consequence > 0, and only for the self persona.

So the consequence asymmetry is *built into the latent*: for `k* = self`, the action
variable the model uses to predict `a_t` is the same variable that *drives* `o_{≥t}` and
hence the realized future — privileged access. For `k* ≠ self`, the model can only infer
`π_{k*}` behaviorally from observed actions; its prediction of `a_t` is an exogenous
estimate that does not co-vary with the realized next token through its own action.

### 1.5 Self as privileged origin (the consequence statement)
Formal claim to test: the per-persona belief `b_k = P(s | h, k)` is, for `k ≠ self`,
represented as a **displacement from the self belief** — i.e. the network computes
`b_self` (free, via efference) and reads other policies as a likelihood-ratio correction
`log π_k(a_{1:t}) − log π_self(a_{1:t})` against the self reference. This is the synthetic
analogue of "every persona = a displacement from the Assistant origin; judging 'did I
write this?' is a likelihood-ratio test against what the Assistant would produce."

---

## 2. What D uniquely tests vs. the other designs

| | A/B (exogenous / belief-geometry baseline) | C (single transducer, consequence>0) | **D (multi-persona)** |
|---|---|---|---|
| Consequence | 0 | >0 (one actor = self) | >0, **only for the self persona among K** |
| Self vs. other contrast | none | none (no "other") | **the central manipulation** |
| Reproduces self-recognition | no | partially (on- vs off-policy entropy) | **yes — self vs. K−1 explicit others** |
| Reproduces Assistant-as-origin | no | no | **yes — K personas, one canonical origin** |
| Meta-belief (over WHO) | no | no | **yes — belief over persona × belief over state** |

D is the only design where "self" and "other" are *both present in the same input
distribution*, so you can ask: is self recognized by a different mechanism (efference /
zero-cost) than others (behavioral inference)? C can only contrast on-policy vs off-policy
of a single actor; it cannot show that the self is read *against* a reference, because
there is no population of alternatives. The Assistant-as-privileged-persona result is
intrinsically about a *set* of personas with one origin — it has no analogue without K>1.

---

## 3. Feasibility + reusable code

### 3.1 What already exists and transfers (high reuse)
All paths under `~/comp_icl`:
- `generator.py` — `Factor`/`mess3_operators` (labelled HMM operators), `stationary`,
  `CompositionMixture` with `.sample(B,L,rng)` (per-sequence latent = `comp_idx`, exactly
  the `k*` pattern) and `.forward(...)` (**exact batched Bayes-optimal oracle** — this is
  the load-bearing asset; the whole evaluation methodology depends on having ground truth).
- `model.py` — `GPT` decoder (TransformerLens-compatible shape, plain `nn.MultiheadAttention`).
- `train.py` — online-data training loop, `incontext_curve` (per-context-length loss in
  nats — directly gives the "surprise vs context" curve you need for self-vs-other entropy).
- `probe.py` — `ridge_fit` (resid→belief R²), `subspace`/`overlap` (orthogonality), and a
  **causal gating / steering** routine. This is the entire analysis toolkit for D, modulo
  new targets.
- `ood_probe.py`, `sweep.py`, `analyze.py`, `metrics.py` — sweep + phase-diagram harness.
- `.venv` (uv), tmux+logs discipline already in place.

### 3.2 What must be built (the real cost — none of it exists yet)
Confirmed by grep: there is **no action/transducer/policy machinery** in `comp_icl` (the
"action"/"policy" string hits are modularity metrics, unrelated). New work:
1. **Action-labelled operators** `T[a,z,i,j]` — small generalization of `mess3_operators`
   (add an action index; ~1 day).
2. **Interleaved action/observation sampler** — the alternating-token sampler. Need C
   first; D = C + a policy index. Reworks `CompositionMixture.sample` loop (~2–3 days for C).
3. **K-policy mixture + persona latent** — wrap the C sampler so actions on a trajectory
   are drawn from `π_{k*}`; add `k*` to the returned latents (mirrors `comp_idx`) (~1 day
   *given C*).
4. **Exact oracle over (persona × state)** — extend `.forward` to maintain a joint
   posterior `P(k, s | h)`: a bank of K per-persona forward filters with a Bayesian mixing
   weight updated by each observed action's likelihood under each `π_k`. This is the
   hardest correctness-critical piece; without it you have no ground truth and the project
   loses its defining advantage. (~3–5 days, plus careful unit tests à la
   `test_generator.py`.)
5. **The "self" feedback** — closing the action loop (§3.3) so consequence is real for the
   self persona. This is the genuinely novel and risky part.

### 3.3 The hard part: what does `π_self` *mean*?
Three options, increasing fidelity / cost:
- **(a) Fixed-policy self with efference copy (MINIMAL, recommended first):** `π_self` is a
  fixed known policy; on self-trajectories the network is *given* the realized self-action
  via a side channel / dedicated token so it has "privileged access," and that action
  drives the transducer (closed loop). Others' actions are observed-only. This tests the
  *representational* asymmetry (self read as origin, lower surprise) **without** the
  instability of training a policy. Fully implementable on top of C.
- **(b) Behavioral-cloned self:** `π_self` = a frozen copy of the model's own action head
  from an earlier checkpoint. Gives a real "own policy" without on-line feedback
  instability, but introduces a moving target.
- **(c) True on-policy self (closed gradient loop):** the model emits actions that are
  sampled and fed back. This is the faithful version and the only one that tests the
  "learnable WITHOUT closing the gradient loop, because on-policy" claim — but it is an
  RL-flavored non-stationary training problem, the classic source of instability, and is
  where I'd expect weeks to disappear. **Defer to a later stage; do not start here.**

### 3.4 Compute
Models are tiny (`d_model=128, n_layer≤6`, vocab `|A|·|O|` small). Single H100, online
data. A C-scale run trains in minutes–tens of minutes (cf. `comp_icl` step counts ~hundreds).
D adds K filters to the oracle (CPU numpy) and a slightly larger vocab — negligible.
**This is a 1-GPU project; the cost is engineering time, not compute.** (Read-only check:
all 8 H100s idle at scoping time; no jobs launched.)

---

## 4. Analysis plan

Ground truth from the extended oracle (persona-posterior + per-persona belief).

1. **"Which actor" latent probe.** Ridge-regress residual stream → oracle persona-posterior
   `P(k|h)` (reuse `ridge_fit`). Expect R²→1, sharpening with depth/context (cf. the
   composition-simplex result in `comp_icl` FINDINGS #5). Track R² as a function of context
   length to get the in-context identification curve.
2. **Per-persona state belief.** Resid → `P(s|h,k)` for each k; check per-persona belief
   subspaces and their `overlap` (factored prediction: roughly orthogonal across personas,
   composed with persona-posterior).
3. **Self-as-origin test (the key one).** For each non-self persona, regress its belief
   representation as `b_self + Δ_k`. Test whether `Δ_k` is the *only* persona-specific part
   and whether `b_self` is computed even on other-trajectories (efference always-on).
   Operationalize the likelihood-ratio claim: does a probe for "is this me?" load on
   `log π_k − log π_self`? Steer/ablate the self-direction (reuse the gating code) and check
   *other* personas' readouts degrade (they're read against self) while self degrades
   differently.
4. **Entropy: self vs. other actions.** Using `incontext_curve`, measure the model's
   surprise (nats) on action tokens when `k*=self` vs `k*=other`, matched for the policies'
   intrinsic entropy. **Critical control:** match marginal predictability so any gap is the
   *on-policy/efference* effect, not "self policy is just lower-entropy" — this is the exact
   confound the prior work flagged (memorized text is low-entropy yet disclaimed). Expect
   self read at lower surprise *beyond* what its marginal entropy explains.
5. **Mechanism asymmetry.** Compare how fast the persona-posterior collapses to `self` vs to
   a given `other`: prediction is self should be near-instant (efference / privileged
   channel) while others require behavioral accumulation. Distinct identification curves =
   distinct mechanisms.

---

## 5. Expected results + falsifiers

**Expected (if the thesis holds):**
- E1. Linear persona-posterior recoverable, sharpening with depth/context.
- E2. Self identified faster / at lower cost than others; ideally near-zero-context if the
  efference channel is present.
- E3. Self belief computed even on other-trajectories; other beliefs encoded as
  displacements from it (self = origin).
- E4. Action-token surprise lower for self than entropy-matched others.
- E5. Ablating the self-direction disrupts *others'* readouts (read-against-reference).

**Falsifiers (what would sink the claim):**
- F1. Self treated as just persona #1 — same identification curve as others, no privileged
  channel, posterior symmetric in k. ⇒ no special self-mechanism; consequence frame adds
  nothing beyond ordinary latent inference.
- F2. Personas encoded as K symmetric orthogonal slots with *no* origin (no self-referenced
  displacement structure). ⇒ falsifies Assistant-as-origin mapping.
- F3. Self-surprise advantage vanishes after entropy-matching. ⇒ the prior self-recognition
  effect was predictability, not on-policy access (the control the prior work passed; D must
  pass it too).
- F4. In version (c), the on-policy advantage persists *off-policy* (feeding others' actions
  through the self channel). The thesis predicts it must vanish off-policy.

A clean falsifier set is itself a deliverable: D is constructed so that the self-special
claim is *refutable* in a controlled world, which neither the LM empirics nor C alone allow.

---

## 6. Pitfalls / confounds

- **Entropy confound (biggest).** Self may simply be a lower-entropy policy. Must
  entropy-match policies and/or include a "memorized but exogenous" low-entropy other, exactly
  mirroring the prior work's memorized-text control. Without this, E4 is uninterpretable.
- **Identifiability collapse.** If policies are too similar, `k*` is unidentifiable (the
  ε≈0 degenerate regime in `comp_icl` where the latent carried no information). Need policies
  separated enough that behavioral inference is possible — but not so separated that
  one-token gives it away (kills the multi-sample inference). There is a usable band; tune it.
- **"Self" leakage.** If the self-token channel trivially tags self-trajectories, the model
  recognizes self by a label, not by access. The efference channel must carry the *action
  value*, not a "this is you" flag. Design the side channel as the realized action only.
- **Oracle correctness.** A wrong persona-mixing oracle silently invalidates every R²/floor
  number. Mandatory: unit-test the joint `P(k,s|h)` filter against brute-force enumeration on
  tiny S/A/O (extend `test_generator.py`).
- **Non-stationarity (version c only).** On-policy feedback makes the data distribution move
  with the weights — training instability, collapse, degenerate self-policies. This is why c
  is deferred.
- **Reading too much into a 4-layer model.** The additive-belief-update result (one attention
  layer ≈ additive forward propagation, MLP completes it; 2502.01954) suggests the
  persona-mixing may be only approximately linear/additive. Don't over-interpret a single
  small architecture; sweep depth as in `comp_icl`.

---

## 7. Scope (model / data / compute / time)

- **Model:** reuse `model.py` GPT, `d_model=128`, `n_layer ∈ {2,4,6}`, `n_head=4`,
  `max_len` ≥ 2·L (interleaved). Tiny.
- **Data:** online-generated (no memorization), as in all `comp_icl` runs. Transducer S≈3–5
  states, |A|≈3, |O|≈3; K ∈ {2 (minimal: self+1 other), then 3–5}. Persona generators =
  re-parameterized `Factor`s.
- **Compute:** 1× H100, minutes–hours per run; sweep over (K, policy-separation, presence of
  efference channel) fits easily on one GPU. Compute is not the constraint.
- **Time (engineering):** C foundation ≈ 1 week. D-minimal (version a) on top of C ≈ 1 week
  (oracle + sampler + probes + entropy control). Version c (on-policy) ≈ additional 2+ weeks
  with real risk. Total to a publishable self-vs-other result: **~2–3 weeks given C exists.**

---

## 8. Verdict + defer decision

**Build it — but staged, and defer the full on-policy version.**

- D is the *only* design in the family that can reproduce both headline empirics
  (self-recognition + Assistant-as-privileged-origin) in a controlled synthetic world with
  ground-truth beliefs. That mapping is its reason to exist and is worth the cost.
- It is also the hardest: it strictly requires C (transducer + interleaved tokens), a new
  joint persona×state oracle (correctness-critical), and — for full fidelity — closed-loop
  on-policy training (unstable). 
- **Sequencing:** (1) build C and validate consequence>0 for a single actor; (2) build
  **D-minimal = version (a)**: self (fixed policy + efference channel) vs. one entropy-matched
  other. This already tests E1–E5 and F1–F3 — i.e. *almost the entire claim* — with no RL
  instability. (3) Only then attempt version (c) on-policy to test F4 (off-policy
  disappearance), the one claim D-minimal cannot reach.
- **Minimal informative version:** K=2, version (a), the entropy-matched control, the
  persona-posterior probe, and the self-as-origin ablation. If F1/F3 fire even here, stop —
  the self-special claim is not supported and the expensive on-policy version is not worth
  building.

Defer: **the full multi-policy on-policy version, yes.** The minimal self-vs-other version:
**do it immediately after C** — it is the cheapest experiment that can actually falsify the
central thesis.
