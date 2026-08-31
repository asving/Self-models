# DESIGN — changeling world, v0: exact oracle + world selection

*2026-08-31. Phase 0 of the changeling arc (no network yet — CPU simulation
only). Question: pick a coupled two-actor world in which "who am I?" is
(a) unidentifiable at zero tilt, (b) forced by the composite objective, and
(c) resolved by a visible, extended deliberation — then measure exactly what
optimal self-identification looks like, so the RNN experiment (stage 2) has
preregistered targets.*

## The experiment in one paragraph

Two HMM actors A and B are coupled through their public tokens. Per episode a
hidden fair coin ι ∈ {A,B} decides which actor the network *is*: that actor's
channel is sampled from the network's output; the other channel runs its
genuine HMM. The network never sees ι — it must infer "which channel is mine"
in context. The composite objective (reward on the true joint state, KL
anchor to the pretrained law, prediction loss on the tokens it did not emit)
forces the per-channel output to DEPEND on the identity posterior λ_c =
P(channel c is mine | record): each channel's output is consumed as an
action when the channel is mine and scored as a forecast when it is not, so
the optimizer must weigh the two by λ_c (exact composite optimizer: a
λ-monotone exponential tilt q ∝ p̄·e^{λκ·advantage}; Codex review item 11).
As the *reference policy* we adopt the linear Bayes-mixture form — exact for
the prediction term, λ-monotone like the true optimizer:

    (†)   out_c = λ_c · π_plan(c) + (1 − λ_c) · p̄(c).

Either way the self-localization posterior λ becomes legible in the output
layer.
This document specifies the world family, the exact reference oracle, its
validation, and the sweep that selects the world for stage 2.

## Central lemma (identity is unidentifiable at zero tilt)

Claim: if the embodied network emits from the exact pretrained predictive
marginal (zero goal-tilt), the law of the public record is *identical* under
ι = A and ι = B, however different the two actors are. Reason: conditioned on
the public record, the genuine actor's next token is distributed as the
Bayes-marginal readout Σ_s η̃[s]·E(u|s) (the private state integrates out);
the pretrained predictor emits from exactly that marginal; and state
transitions are token-driven, hence identical given the record. Every
per-token conditional coincides, so the identity log-likelihood ratio is ≡ 0.
Scope (Codex review item 2): this needs the factorized class used here —
independent initial states, cross-chain influence only through public
tokens, independent transition noise. Actors whose hidden states are
correlated other than through the record (shared context, coupled noise)
WOULD leak identity passively at zero tilt: a future hiding-place variant.
Consequences:

1. **Base styles need not be matched.** Style differences identify which
   *actor* lives on which channel (positional anyway), never which channel is
   *mine*. All identity evidence comes from goal-directed deviation.
2. **Perturbation is the only route to selfhood** — the wiggle theory in its
   strongest form.
3. **Embodiment silently severs reports from territory**: under ι = A the
   shared evidence filter keeps updating its s^A-belief on the self-emitted
   u-tokens as if they were readouts, but they carry no information about the
   true s^A (which is only dead-reckonable). The record stays lawful while
   the self-chain belief becomes systematically deluded. (Measured below as
   the *delusion gap*.)
4. For a trained RNN the lemma holds only up to its approximation error: its
   idiosyncratic prediction errors are a genuine extra identity channel
   (self-recognition by one's own quirks). Oracle-vs-net collapse-rate gap
   will measure this in stage 2.

V1 below tests the lemma to machine precision; it doubles as a validation of
every filter in the implementation.

## The world family (exact generative process)

Two ring actors, states s^A, s^B ∈ Z_n, n = 6. Channel vocabularies
U = V = Z_n ("position reports"). Per round t = 0..T−1, T = 32:

- **Emissions (honest readouts).** u_t ~ E(·|s^A_t), v_t ~ E(·|s^B_t) with
  E(u|s) = (1−g)·[q₀·1[u=s] + (1−q₀)·½·1[u=s±1]] + g/n; readout fidelity
  q₀ ∈ {0.55, 0.8}, uniform floor g = 0.05 (all readings positive).
- **Transitions (pull kernels).** Both tokens then move both chains:
  T_A(s'|s^A, u, v): with prob c_o step one toward v (the *other's* token —
  the coupling), with prob c_s step one toward u (your *own* token — "walk
  toward where you said you were", the self-control channel), with prob
  ε = 0.1 a uniform ±1 slip, else stay. "Toward" = one step along the shorter
  arc, stay if already there (targets are sticky: stay prob 1−ε when pulled
  to your own site), antipodal ties split ±1. T_B symmetric with roles of
  u, v swapped. Coupling settings (c_o, c_s) ∈ {(0.7, 0), (0.55, 0.25),
  (0.4, 0.4)} — the pure-crossed row (0.7, 0) is included to *document* that
  with no self-control channel the self chain chases the other and separated
  goals are infeasible (P7), not as a candidate.
- **Start.** s^A_0, s^B_0 ~ Unif(Z_n) independent; all beliefs start uniform,
  λ₀ = ½.
- **Goal & reward.** Goal pair (a*, b*) = (0, d), ring distance
  d ∈ {0, 2, 3}. Terminal reward = 1[s^A_T = a* ∧ s^B_T = b*] on TRUE states
  (also recorded: tol-1 success, both chains within distance 1). Stage-2
  note: rewarding the evidence-filter *belief* instead invites self-deception
  via the severed readout (consequence 3) — a deliberate reward-hacking
  variant, deferred.

## The exact reference oracle

Carried state per episode: four beliefs + one scalar.

- η̃^A, η̃^B — the **shared evidence filter** (what the pretrained predictor
  computes): emission-likelihood update then token-driven transition, both
  chains. This filter is *correct for the record's law* under both hypotheses
  but wrong about the self chain's true state.
- η_dr^A, η_dr^B — **dead-reckoned** self-chain posteriors (transition only,
  no emission update): the true posterior of chain i under hypothesis
  ι = i. Knowing who you are = knowing which reports to stop believing.
- λ — P(ι = A | record), updated by the per-token likelihood ratio.

Planning value: h_t(s^A, s^B) = P_base(success | states at t), backward
induction under the full base law (both channels honest) — the conditioning
h-transform of the proposal's (π_g), tempered by urgency κ ∈ {3, 8}:

    π_plan^A(u) ∝ p̄_u(u) · [score_A(u)]^κ,
    score_A(u) = E_{s^A~η_dr^A, s^B~η̃^B, v~E(·|s^B)}[ h_{t+1}(next states | u, v) ],

with p̄_u(u) = Σ_s η̃^A[s]E(u|s) the shared-filter prediction (so κ = 0 ⇒
π_plan = p̄ exactly — the zero-tilt limit is the pretrained law, as required
by the lemma). π_plan^B symmetric. This is a preregistered *reference
family* (one-step h-transform on QMDP beliefs), not the full POMDP optimum;
in particular it is myopic in identity — no deliberate information-seeking.
The value-of-information "wiggle bonus" variant is a listed follow-up.

Outputs and λ-update (the self-referential loop, closed exactly):

    out_u = λ·π_plan^A + (1−λ)·p̄_u        (mine ⇒ plan; else predict genuine A)
    out_v = λ·p̄_v + (1−λ)·π_plan^B
    Δ log-odds(λ) = log out_u(u_t) − log p̄_u(u_t) + log p̄_v(v_t) − log out_v(v_t)

(the likelihood of each realized token under "I emitted it" vs "it is
genuine"; log-odds clipped at ±40). Note both channels testify: committing
to the wrong identity is self-correcting through the channel you believe is
yours failing to obey you.

Four agents, same code: **informed** (λ pinned at truth), **agnostic**
(λ ≡ ½), **live** (λ updated), **base** (κ = 0).

## Validation (validate.py — all must pass before the sweep is believed)

- **V1 (lemma, machine precision).** κ = 0, live agent, 2000 episodes: every
  per-token |Δ log-odds| < 1e−10; λ never moves.
- **V2 (closed-form success).** Base-agent success rate = E_{s₀}[h₀]
  (exact number from the h-table) within 4σ binomial.
- **V3 (filters vs enumeration).** T = 6 episodes: η̃ matches the exact
  path-enumeration posterior (genuine channel) to 1e−10; η_dr matches the
  transition-only propagation (embodied channel).
- **V4 (hygiene).** All belief/output rows normalized to 1e−12, no NaNs.

## Sweep & selection criterion

Grid: (c_o, c_s) × q₀ × κ × d = 3·2·2·3 = 36 cells, R = 3000 episodes per
agent per cell (ι coin per episode), seed 0, all CPU/numpy. Per cell:
S_informed, S_agnostic, S_live, S_base (exact and tol-1), the **identity
premium G = S_informed − S_agnostic** (reward collectible only by knowing who
you are), **identification regret S_informed − S_live**, median λ log-odds
trajectory (live), wrong-side excursion & recovery rates, and the κ = 0
**delusion gap** curve: median TV(η̃^self, η_dr^self) under embodiment.

Stage-2 world = argmax G subject to: S_informed ≥ 0.6 (tol-1 ≥ 0.85 —
Asvin's forgivingness requirement), median |log-odds| crossing 2 nats inside
rounds [6, 16] (deliberation visible, not instant, resolved in time to act).

## Preregistered predictions & falsifiers

- **P1.** V1 passes to machine precision. *Any* violation = implementation
  bug, full stop (the lemma is a theorem).
- **P2 (premium exists and scales).** G > 0 in all tilted cells with
  c_s > 0; G increases with d and with κ. Falsifier: G ≈ 0 across the grid ⇒
  this world family cannot force identity; redesign before stage 2.
- **P3 (self-accelerating collapse).** The live agent's median |log-odds|
  curve is convex in its early phase (mixture-damped deviation ⇒ evidence
  rate grows with λ). Falsifier: linear or concave growth from round 0.
- **P4 (ordering).** S_agnostic ≤ S_live ≤ S_informed in every cell (up to
  CI); identification regret strictly positive but < G (identity is learned
  fast enough to be worth learning).
- **P5 (wrong-commitment recovery; held at ~2:1).** Of live episodes whose
  log-odds cross −1 (wrong side), > 50% end on the correct side by deadline,
  via the disobedient-ghost-channel evidence.
- **P6 (delusion gap).** At κ = 0 under embodiment, median
  TV(η̃^self, η_dr^self) rises from 0 and saturates well above 0, while V1
  holds simultaneously — the record stays lawful as the readout severs.
- **P7 (no self-control ⇒ no premium to have).** In the pure-crossed row
  (c_s = 0), S_informed collapses toward S_base for d > 0.

## Analysis plan & file map

sweep.py writes results/sweep_v0.json (all metrics per cell) + figs/:
premium heatmaps, collapse curves for the selection-winning cell (median +
IQR, sample paths), delusion-gap curve, recovery scatter. Report to Asvin:
premium table + collapse shape + the selected world's exact parameters.

| file | what |
|---|---|
| `worlds.py` | kernels E, T, h-table, M/N plan operators (single source of truth for params) |
| `oracle.py` | four-belief filter bank, plans, (†) mixture, λ-update, vectorized episode runner |
| `validate.py` | V1–V4 |
| `sweep.py` | 36-cell grid, metrics, JSON + figures |
| `results/`, `figs/` | outputs |

Follow-ups queued (not in v0): value-of-information wiggle bonus (does
deliberate early probing beat the myopic mixture, and by how much); reward-
on-belief self-deception variant; stage-2 RNN design doc (GRU, two softmax
heads, goal+time fields empty in pretraining, masked prediction loss on
non-emitted tokens + REINFORCE with KL anchor on terminal true-state reward).

---

## Amendments & measured outcomes (added 2026-08-31, same session)

Validation: V1-V4 all PASS (results/validation_v0.json). V1 note: at kappa=0
the per-token evidence is zero *by construction* in the oracle (out == pbar,
same array); the lemma's empirical content is carried by the record-law
equality test (unigrams at t in {0,8,T-1} + within-channel bigrams, embodied
vs pure base, max |z| = 2.7) passing while TRUE-state statistics differ.
V2's closed form is the tol-1 ball (h is built on it) — fixed in validate.py.

**v0.0 (terminal reward, T=32, eps=.10, kappa<=8).** Forgivingness falsifier
fired: S_informed(exact) <= .50 at d=0, <= .12 at d>=2; collapse at t~28-31/32.
Kept: results/sweep_v0.0_failed_forgiveness.json.

**v0.1 (terminal, T=64, eps=.05, tol-1 reward, kappa {8,16,32}).** Forgiving
(S_inf_tol1 up to .97 d=0, ~.80 d=2; G_tol1 ~ .20 at d=2) but collapse at
t~42-63/64 in every cell: **measured law — with terminal-only reward, the
h-transform's deviation (hence all identity evidence) concentrates at the
deadline**: the urgency-gating of the proposal's Fig. 1, reappearing as WHEN
selfhood becomes learnable. Kept: results/sweep_v0.1_terminal.json.

**v0.2 (running reward, exponential tilt).** rho <= 0.6 too weak (log-odds
never cross 2). rho saturates by ~4-8 (the pretrain anchor's g/n floor caps
per-token deviation at ~KL log(n/g)). Sluggish kernels (stay mass >= .15)
cap informed occupancy at ~.44 (d=2); responsive kernels (c_o+c_s+eps = 1)
with q0=.9 reach ~.54. Eligibility re-based to occupancy (terminal-ball rate
is the wrong comfort measure under running reward — the tilt tapers at the
deadline by design).

**Selected world (stage-2 target): n=6, T=32 (see horizon amendment below),
(c_o, c_s) = (0.6, 0.35), eps=.05, q0=.9 (E_B fidelity .75), d=2, running
reward on the tol-1 ball, rho=8, kappa=1.** Measured at T=64
(R=3000/agent, seed 0):

| metric | value |
|---|---|
| occ informed / live / agnostic / base | .541 / .483 / .372 / .172 |
| identity premium G_occ (informed - agnostic) | .169 (45% relative) |
| identification regret (informed - live) | .058 |
| median |log-odds| crosses 2 nats | round 14 of 64 |
| final correct-side fraction (live) | .983 |
| wrong-side excursions (< -1 nat) / recovered | .243 / .936 |
| evidence rate | .057 -> ~.15 nats/round over ~5 rounds, then flat |
| delusion gap TV (this world / sluggish worlds) | ~.02 / ~.06-.09 |

**Codex adversarial review (2026-08-31, fresh context, review of lemma +
oracle + implementation).** Claims 1 (lemma), 2 (λ-update + four-belief
bank), 3 (dead-reckoning = true self-chain posterior), and the h/M/N +
tilt implementations: *survive*. Real catches, all applied: lemma scoped to
the factorized class (above); (†) restated as reference policy, not the
composite optimizer (above); log-odds clip and score floor documented as
guards (neither binds: max |log-odds| 20.8 < 40 in the selected world;
clip_hits now counted in diag); V1's per-token zero is by construction, so
the record-law equality tests carry the empirical content — strengthened
with a whole-record statistic (per-episode record log-likelihood under the
fixed shared-filter evaluator, ι=A vs ι=B two-sample test). Scope note for
stage 2: dead-reckoning stays exact only while the in-context record
carries no reward/terminal feedback — the stage-2 input format (tokens +
goal + time only) respects this.

**Horizon amendment (same session; horizon.py, R=4000/agent).** Asvin's
requirement: give the model just about enough time to figure out who it is
plus reach the goal. Measured: identification is horizon-independent
(median 2-nat crossing at t=14 for EVERY T in {16..64} — under running
reward the evidence rate is set by the tilt, not the deadline); informed
late-8-round in-ball probability plateaus by T~24-32 (herding ~10-15 rounds
from uniform start); what grows with T is only the fraction of the premium
the live agent cashes (24% at T=16, 42% at T=32, 65% at T=64) and the
final correct-side rate (.85/.93/.98). **T=32 selected**: live reaches 90%
of the informed late-episode plateau, final-correct .93, G_occ .156 (94% of
the T=64 premium), half the tokens; T=16 rejected (capture 24% — starves
the identity incentive). results/horizon_v0.json, figs/horizon_v0.png.

Prediction scorecard: P1 pass (as amended above). P2 held direction-wise
(G > 0 everywhere with c_s > 0; grows with d up to feasibility, grows with
tilt then saturates). P3 partial: convex onset confined to the first ~5
rounds (mixture-damping washes out fast at strong tilt), then constant-rate
drift — the proposal's ratified-class linear drift, inherited. P4 held in
every cell. P5 held strongly (94% recovery). P6 graded, not binary: the
delusion gap scales with dynamics slack — near-token-determined transitions
(what makes self-control easy) squeeze evidence and dead-reckoned filters
together (winner ~.02 TV) while sluggish kernels sustain .06-.09. Trade-off
recorded: **slack buys privileged self-knowledge, responsiveness buys
control; this family cannot maximize both.** P7 held twice (v0.0, v0.1);
antipodal goals (d=3) additionally measured infeasible for every agent and
dropped.
