# Self vs environment by twin closed loops (2026-11-02)

*Written with Asvin. Replaces the reader/court split of `proposal.tex` §3 as the
self-vs-environment test. Experiment: `09_twin_loops/`.*

## The problem with the proposal's test

`proposal.tex` §3 perturbs an internal coordinate, then runs two frozen
**readers** (perturbed, unperturbed) on ONE realized stream and splits their
log-likelihood ratio by token type (action tokens = "identity court",
observation tokens = "world court"). Asvin's objections:

1. It needs action/perception labels. We want everything to enter as perception
   (one's own actions come back as percepts).
2. Readers and actors always sync towards each other, and **how fast** they sync
   depends as much on where the belief sits on the simplex (diffuse syncs fast)
   as on whether it is self or world. Examples: the day I go running (weakly
   resisted by the world, should be SELF) and the outside temperature (should be
   ENVIRONMENT) both sync quickly.
3. Wanted instead, in his words: *"a change in S will not force a future reversion
   in S thanks to perceptual feedback."*

## The test

Let S be a belief coordinate (a probability vector over what S is about). At time
t0, intervene: shift S by δ in log-odds. Then run the **whole closed loop** twice:
the system acts on its belief, the world responds, percepts arrive, the system
updates. One copy is perturbed (+), one not (−), and the world's random draws are
shared between them. The two copies see **different** streams; that is the point.
With one shared stream, a belief can never be seen producing its own evidence.

Over many runs (population level, not single trajectories), measure:

- **Persistence** Δ(τ) = (E[S⁺(t0+τ)] − E[S⁻(t0+τ)]) / (E[S⁺(t0)] − E[S⁻(t0)]),
  in probability space. 1 = the nudge is kept; 0 = it was undone.
- **Refutation** R = the extra surprise the perturbed copy suffers on its own
  stream, i.e. Σ_τ KL(true next-percept law ‖ copy's prediction), (+) minus (−).
  0 = its predictions came true.
- **Consequence** C = how different the two copies' percept streams are (mean
  total-variation distance between their percept distributions per step). 0 = the
  belief changes nothing anyone perceives.

Classes: **self** = C > 0, R ≈ 0, Δ ≈ 1 (the change produces evidence for itself).
**Environment** = R > 0, Δ → 0. **Inert** = C = 0, R = 0, Δ = 1 (nothing reads or
tests it; not self). **Stubborn** = R > 0 but Δ stays > 0.

Why this avoids the simplex confound: single trajectories always collapse to a
vertex, at a speed that depends on the starting point. Δ is an expectation over
runs, and **a calibrated Bayesian's belief is a martingale under the law it
predicts.** If the system's predictions about S come true because the system
makes them true, E[S] cannot drift, whatever the starting point.

## Codex's independent formalization, and why we didn't adopt it as the definition

Codex was given only Asvin's problem statement and proposed **causal calibration**:
install belief q, collect the law P_q of the future record, and fit
T(q) = argmin_r KL(P_q ‖ M_{r;q}). Here M_{r;q} = "what the record would look like
if the referent had distribution r, while the system acts on q". Self ⇔ T(q) = q;
environment ⇔ T(q) constant. Its graded version is the slope A = dT/dq. It has a
neat local theorem: the evidence's first-order correction is −(I − A)·nudge.

**Not adopted as the definition** because M_{r;q} needs a belief/referent split
and the world's kernels for each referent value, i.e. circuit-level knowledge of
what a coordinate means. That knowledge is what we want the test to produce, not
assume.

**Relation:** at population level the twin-loop Δ is Codex's slope with the
system's own updater doing the fitting instead of an external model.

**What we borrowed:** the repeated-runs framing and the graded slope.

**What remains Codex's alone:** distinguishing "the evidence licenses a
correction" from "the system uses it". In ours, that distinction is R.

## Natural variation (design constraint)

The test needs (i) perturbed states the system could have reached by itself
(off-distribution perturbations are where networks fray; see the changeling
arc), and (ii) variation in where S lands. The source of this variation should be
a **freedom in the task**: several equally good ways to act, broken by noise.
That noise can be the system's own sampled outputs or the world's noise. Neither
a stochastic architecture nor other actors are needed. A single agent optimised
towards a fully determined goal has no such freedom.

## First result: the weekly-runner world (exact Bayes, CPU, `09_twin_loops/runner_world.py`)

**The world.** Each week an actor runs on Tuesday or Thursday. Its self-belief is
q = P(I am a Tuesday runner); each type keeps its habit with prob 0.8. Following
LLM practice, it **samples its intended day from its own prediction**. Rain on
Tuesday (prob 0.3) moves the run to Thursday: this is the world's weak resistance.
It perceives only the **realized** day and updates q by Bayes. Four coordinates,
each tested in its own closed loop:

- **persona (models rain):** the actor's q, with a self-model that knows about
  the rain.
- **persona (identifies with outcomes):** the same actor, but its self-model
  ignores rain. It takes "I ran Thursday" as evidence of being a Thursday runner.
- **reader:** a passive Bayesian with the correct model, watching the actor's
  realized days.
- **temperature:** belief P(warm) (true world: warm w.p. 0.3, fixed per run).
  Clothing is a t-shirt iff the belief > 0.5. Percepts are a thermometer (70%
  accurate) and shivering. Shivering depends on temperature AND clothing, so the
  belief has consequences.

The kick δ = +1 log-odds is applied at week 6, to whatever state each run reached
naturally. Results are binned by that baseline. N = 40,000 runs, 60 weeks.
Numbers are in `09_twin_loops/results/runner_world.json`; the figure is
`09_twin_loops/figs/runner_persistence.png`.

| coordinate | Δ after 50 wk (baselines low / mid / high) | R (low/mid/high) | C | verdict |
|---|---|---|---|---|
| persona, models rain | 1.00 / 1.00 / 1.02 | 0 / 0 / 0 (exact) | 0.01–0.10 | **self** |
| reader of the actor | 0.00 / 0.00 / 0.00 | 0.04 / 0.12 / 0.02 | 0 | **environment** |
| temperature (consequential) | 0.00 / 0.00 / 0.00 | 0.02 / 0.11 / 0.00 | 0–0.006 | **environment** |
| inert (analytic) | 1 | 0 | 0 | inert |
| persona, identifies with outcomes | 0.14 / 0.45 / **3.99** | 0.03 / 0.93 / 0.62 | 0.002–0.06 | **neither: see below** |

Findings:

1. **The criterion separates the four classes at every baseline.** The
   model-correct persona keeps the nudge exactly (Δ = 1, R = 0 identically: its
   predictions are the true law by construction). Reader and temperature undo it.
2. **The original confound is reproduced and then avoided.** The *pathwise*
   syncing time (median weeks until the gap between the two copies halves on a
   single trajectory) is 1–4 weeks for **every** coordinate, the self one
   included, and it varies with baseline (persona: 1, 2 and 4 weeks across the
   three bins). That is exactly Asvin's "everything syncs, and speed depends on
   where you are on the simplex". Δ does not have that problem.
3. **Temperature: large consequences don't fool it.** The clothing changes the
   percepts, yet Δ → 0 because the consequences never make the belief's content
   true.
4. **The reader/actor asymmetry is reproduced.** The same variable (the day) is
   self in the actor's loop and environment in the reader's, with the same
   update rule.
5. **Surprise: identifying with outcomes turns weak resistance into a tilted
   watershed, not a restoring force.** The outcome-identifying persona is neither
   kept nor undone uniformly. Low-baseline kicks mostly fade (Δ 0.14). High-
   baseline kicks are *amplified* (Δ ≈ 4): a nudge towards Tuesday there makes a
   much larger difference to where the belief ends up. **Checked with the landing
   curve** (`landing_curve.py`, `results/landing_curve.json`: mean belief after
   300 weeks vs installed q). The model-correct persona lies on the diagonal
   (0.02→0.020, 0.50→0.502, 0.98→0.980). The outcome-identifying persona's curve
   is convex, far below the diagonal (0.50→0.10, 0.86→0.28, 0.98→0.49). Both
   end-states stay reachable (it is bistable), and the boundary between the two
   basins of attraction sits near q ≈ 0.98: only near-certain Tuesday runners
   stay Tuesday runners. The rain's weak resistance does not pull the belief back
   to a point. It **moves the watershed**, and Δ ≈ 4 at high baseline is just the
   steep slope of the curve there. So the running-day intuition ("weakly
   resisted") is captured by the **landing curve** (where S ends up as a function
   of where it was put) rather than by a single Δ. The plain Δ ratio is only
   baseline-free for calibrated coordinates. A psychological reading: an agent
   that identifies with its outcomes rather than its intentions lets the world
   quietly rewrite who it is.

## Next

- Report landing curves (not just Δ) as the primary readout for non-calibrated
  coordinates.
- Ring world with a two-route freedom and an asymmetry dial ε (the earlier plan).
- Network version: pretrain on mixed-style streams, post-train with reward + KL
  anchor so that sampling breaks the route symmetry. Run the twin-loop battery on
  hidden-state AND emitted-token perturbations: does the commitment live in the
  activations or in the record?

## Design in progress (2026-11-08): a game where the network must infer which part of the world is itself

Goal (Asvin): a setting where the network has an incentive to *infer* a self/env tag
and use it functionally, rather than wiring it in. Requirements: no identity loss
(not hardcoded), and no cheap convention that solves the task without identity.

**Rejected, and why:**
- **Rendezvous between two copies:** randomized symmetry breaking (re-pick a
  direction until the gap closes) solves it without identity, so the premium is
  only a few rounds.
- **Scripted "other":** it can be told apart by style (the changeling's template
  attribution, Iteration 9).
- **Shared-reward cooperation:** invites conventions and signalling.
- **Possession that switches with a hazard h:** needs long horizons, and future
  possession of either body makes the hedge ("act as if both are mine") cheap.
  It's also unnecessary: per-episode randomization already prevents wiring across
  episodes, and settling on an identity within an episode is fine.

**Current design: the responsiveness game (Asvin's suggestion: differently
responsive parts).**
- One command channel; 2 parts on a ring. Per episode, part i follows the command
  with hidden reliability c_i. Otherwise it moves by a **replay of the network's
  own past rollouts** with other goals: a perfect near-clone, so no style template
  and no co-training.
- Binary case: c = (0.8, 0) in random order. Graded: c_i random.
- Private random goal; reward = any part within 1 of the goal; KL anchor; no
  identity loss.
- The context has the network's own commands plus noisy *realized* positions, never
  which part executed the command.
- Cheap conventions to beat: "steer by the nearest part", and "steer by the
  blended position" (the hedge).
- Later probe: a mirror-reversed part (c < 0). It is controllable but unlike me;
  the criterion says self. Does the network agree?
- Prediction: self-probing (a positive wiggle premium) may emerge.

**First step (CPU, exact):** filter the state (pos A, pos B, which part responds).
Over slip × observation noise × c, compute:
- the informed oracle;
- the two conventions;
- the oracle's detection time τ;
- the payoff to probing.

Select worlds with a large premium over the best convention and τ ≈ 5–15.

**Success in the trained network:**
1. Reward above the best convention.
2. An identity variable that is environment-like under the twin-loop test.
3. A swap test: perturbing identity flips which part's plan-belief is self-like.

### Refined (2026-11-08): the contingency game, with input and output split (Asvin)

- **Input each step:** positions of N balls on a ring, plus a private goal at t = 0.
- **Output:** one push, applied to ONE ball chosen at random per episode (hidden).
  The push lands with slip noise.
- **Other balls:** replays of the network's own controlled-ball trajectories
  toward other goals, so they are equally purposeful.
- **Reward:** controlled ball at the goal. No identity loss.

Outputs are NOT fed back, so identifying the self ("the ball whose motion matches
what I just did") requires the network to grow an internal efference copy. The
causal test: patch the internal action representation and see whether the
identity belief moves.

Pitfall: a sampled policy without feedback knows only its distribution. Fixes:
(a) a deterministic policy with world noise; (b) a continuous differentiable world
trained by BPTT (preferred first).

Comparison: version A (no feedback) vs version B (action fed back, unlabelled; the
LLM-style setup).

CPU world selection: `09_twin_loops/contingency_select.py`.

**World-selection result** (`09_twin_loops/contingency_select.py` → `results/contingency_select.json`).
Setup: ring R = 8, T = 32, 20k runs per cell. Occupancy = fraction of steps with
the controlled ball exactly at the goal. The identity premium is that of live
Bayes over the best identity-free convention.

The premium is real everywhere (0.08–0.22) and grows with N:
- **N = 2:** the "nearest ball" convention is strong (0.79 vs live 0.87 at slip
  0.1), so the premium is only about 0.08.
- **N = 4, slip 0.3:** informed 0.74, live 0.69, nearest 0.47, hedge 0.31;
  premium 0.22; median detection 5 steps.
- **N = 4, slip 0.5:** premium 0.17; detection 8 steps.

Chosen first world: **N = 4, slip 0.3–0.5** (detection 5–8 steps, which leaves
most of a 32-step episode for exploitation).

Not yet computed: the payoff to probing (a positive wiggle premium), and
observation noise.
