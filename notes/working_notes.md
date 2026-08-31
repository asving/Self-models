# Working notes — action entropy collapse & self-models

*Running conceptual notes for the self-models project. The thesis throughout: an action is special
because it influences your future sensory input, and that creates a pressure to make your own actions
**legible to yourself** — which shows up as **entropy collapse**, and, taken further, as a **self-model**.*

**Companion artifacts (referenced inline):**
- **Fellows paper** — *"From Simulation to Enaction: Post-trained Language Models Recognize and React to
  their own Generations,"* Asvin G. & Jack Lindsey, **arXiv:2605.25459** (local copy
  `fellows_paper_2605.25459.pdf`). Key results we lean on: post-trained models recognize their *on-policy*
  generations; on-policy output entropy is **3–4× lower** than off-policy across families/sizes; an internal
  **input-surprise** representation (how unlikely the most recent token was under the model's own prior)
  **causally modulates** output entropy; and post-trained (not pretrained) models **collapse uncertainty over
  the topic of the upcoming response before the first output token**, with violating that cached intention
  being costly.
- **`SELF_LEGIBILITY.md`** — the entropy-collapse / self-legibility experiments (blindfolded
  rock–paper–scissors under imperfect monitoring): the controlled small-scale version of the same effect.
- **`DEPTH_AND_RECURRENCE.md`** — the depth / Picard-through-layers analysis (how the action↔belief loop is
  realized in a feedforward transformer).

---

## 1. Action entropy collapse

*Why an information-seeking agent is pushed toward near-deterministic action distributions.*

- **Actions are not predictions — they influence your future input.** Assume the agent processes inputs in
  time order: it consumes `x_1, …, x_t` and emits an action `a_t` (a **distribution** that gets sampled), and
  the next input is `x_{t+1} = f(a ~ a_t, e_t)`, where `e_t` is the environment's emissions. A **pure
  predictor** (e.g. LLM **pretraining**) has `f` independent of `a_t`; in general `f` depends on both `a` and
  `e` to varying degrees.

- **To extract information, you must subtract off your own action.** If the agent is using its inputs to
  figure out the best strategy, it needs to **decode `f` and subtract the effect of `a`** to recover `e_t` —
  the thing it actually cares about. But if `a_t` is **highly entropic** and `f` depends strongly on its
  action argument, the **sampling noise `a ~ a_t` drowns out** the information about `e_t`.

- **⇒ strong pressure for low-entropy (near-deterministic) `a_t`.** This is a plausible account of the
  post-trained-LLM effect in the **Fellows paper (arXiv:2605.25459)** — the 3–4× on-policy entropy drop and
  the causal input-surprise → output-entropy link — and perhaps **why RL tends to collapse entropy.** It is
  also exactly what the controlled experiments show (**`SELF_LEGIBILITY.md`**): in RPS against an opponent
  that switches between a weak/exploitable and a strong/best-responding strategy, trained nets **move between
  low entropy** (to decode the opponent) **and high entropy** (to play optimally / avoid exploitation),
  tracking how the opponent switches. We have since derived this analytically — the information an outcome
  carries about the opponent is gated by `|P₁(a_t)|²` (the deconvolution SNR), which vanishes as
  `a_t → uniform`.

- **The argument still bites when `f` is just "sample `a_t`" (e.g. LLM text generation), at a different level
  of abstraction.** The net still needs to decode information about its own **future trajectory**, and under
  computational constraints it has limited information about where it is heading. To **plan or execute on
  goals**, it wants to minimize uncertainty about its future trajectory from *extraneous* (sampling) noise —
  so it is incentivized toward **deterministic `a_t`**. This is precisely the Fellows-paper finding that
  post-trained models **collapse topic-uncertainty before the first output token** and pay a cost when that
  cached intention is violated.

## 2. Self-models as models of internal generative processes that impact the future

*What "self-model" should mean, why it implies actions, and why it is recursive (hence subtle).*

- **Definition.** Treat the environment as a collection of **generative processes (including the agent)** that
  all contribute to the inputs `x_t`. A good agent needs an internal model of **each process** that might
  contribute to `x_{t+1}` — and in particular has a strong incentive to model **the pieces of *itself* that
  contribute to `x_{t+1}`** (e.g. to cancel its own action's contribution out of the next input, per §1). **That
  is what should be called a "self-model."**

- **Tied to actions ⇒ absent in pure predictors, present after post-training.** I am tying self-models
  directly to **actions** — the parts of you that *directly affect the future inputs you receive*. The
  correlation must be learned (so it could be spurious), but in the limit you learn a **faithful model of
  yourself that pertains to the future.** Consequently **pure predictors (base models) should have no
  self-model in this sense, while post-trained models can develop one** — which is what the **Fellows paper
  (arXiv:2605.25459)** found signs of: recognition of one's own on-policy generation *is* a self-model in
  this sense, present after post-training and absent/weaker in the pretrained predictor.

- **The recursive catch — and how a transformer pulls it off.** This self-modelling is hard to pin down
  because of a **loop**: your self-model is a model of *the cause of your actions*, yet to be useful it must
  *impact your actions*. That is not straightforward on a feedforward transformer, but in the small-scale
  experiments the nets manage it inventively (see **`DEPTH_AND_RECURRENCE.md`** and the RPS circuit
  dissections):
  - **Recompute-by-attention** — re-derive your previous-turn action by effectively *zeroing attention to the
    current token*, reconstructing "what I would have done" from the context-minus-the-last-input. Costs
    **fewer layers but more residual-stream space.** *(In the RPS circuit this is what we found: the move used
    to subtract off your own contribution is re-derived per round from context, not routed forward as a stored
    value — established by an activation-patch **healing** test, not by output perturbation, which can't
    distinguish the two.)*
  - **Picard iteration** — solve the fixed point `y = f(x), x = g(y)` by iterating `y_t = f(x_{t-1}),
    x_t = g(y_{t-1})`, unrolling the action↔belief loop **through depth** (the Picard-through-layers picture).
    Costs **more layers but less space.**

  The recursion is the crux of why a genuine self-model is subtle: the quantity you want to represent (your
  own action / its cause) is both an *input to* and an *output of* the same computation.
