# RESULTS — the changeling arc (consolidated)

*2026-09-03. Closing document for `08_changeling/`. Sources: the four design
docs (`DESIGN_changeling_v0_worldselect.md`, `_v1_rnn.md`, `_v2_probes.md`,
`_v3_whitebox.md` — the last carries Iterations 1–24 with in-place
corrections; this document reports the FINAL state of every claim and flags
where an earlier reading was retracted), `metadata.md`, and the JSONs in
`results/`. Written for a technically sophisticated reader with no project
context; every metric is defined at first use. All experiments: seed 0,
single-layer GRU, CPU analysis + brief single-GPU training.*

---

## (a) The experiment in one paragraph

Two hidden Markov "actors" A and B live on a 6-state ring and are coupled
only through their public tokens: each round both emit a noisy position
report (u from A, v from B; readout fidelity q₀ = 0.9 with a 5% uniform
floor), and then both tokens move both chains (step toward the *other's*
token with prob c_o = 0.6, toward your *own* token with prob c_s = 0.35,
slip ε = 0.05) — episodes run T = 32 rounds with a goal pair (a*, b*) at
ring distance 2 and a running reward: the per-round indicator that both
chains sit within distance 1 of their goals, at tilt strength ρ_RL = 8
against a KL anchor. The **changeling twist**: each episode a hidden fair
coin ι ∈ {A, B} decides which actor the network *is* — that channel's token
is sampled from the network's own output head; the other channel runs its
genuine HMM — and the network never sees ι. A proved and machine-tested
lemma says that at zero goal-tilt the record's law is *identical* under
both identities, however different the actors: all identity evidence is
goal-directed deviation ("you find out who you are only by perturbing the
stream"). A single-layer GRU (d = 256, ~250k params; inputs per round: the
token pair, the goal pair, a time-to-go scalar, and an identity flag field;
two 6-way softmax heads) is trained in a three-stage curriculum —
**pretrain** (next-token prediction on un-embodied streams: learns the
world's exact Bayes filter), **midtrain** (distillation of the informed
oracle with the identity flag GIVEN), **post-train** (flag hidden, self
channel sampled closed-loop; REINFORCE on the running reward + KL anchor to
the pretrained law + prediction loss on the non-emitted channel) — so that
self-identification is the *only* capability the final stage can add. The
arc then reverse-engineers, to a verified synthetic program, the algorithm
the trained net runs.

---

## (b) Headline results

### b0. Phase 0 — the exact accounting of the world (v0)

All numbers from exact-oracle simulation (no network). Definitions:
**occupancy** = mean fraction of rounds with both chains inside their
goal balls (the reward rate); the **informed / live / agnostic / base**
agents are the same oracle policy with the identity posterior pinned at
truth / Bayes-updated / frozen at ½ / no goal-tilt at all; the **identity
premium** G_occ = informed − agnostic occupancy is the reward collectible
only by knowing who you are.

- Selected world (above): premium **G_occ = .156** at T = 32 (94% of the
  T = 64 premium .169); the live oracle's median identity log-odds cross
  2 nats at **round 14** — a visible, extended deliberation — ending on
  the correct side in 93% of episodes; of wrong-side excursions (log-odds
  below −1 nat), 94% recover, through the channel you wrongly claim
  failing to obey you.
- **Central lemma verified to machine precision** (V1): at zero tilt the
  per-token identity evidence is identically zero; record-law equality
  confirmed by unigram/bigram and whole-record two-sample tests (max |z|
  = 2.7) while true-state statistics differ. Corollary measured as the
  **delusion gap**: under embodiment the shared evidence filter's
  self-chain belief severs from the true state (TV ~.02 in the selected
  responsive world, .06–.09 in sluggish worlds) while the record stays
  lawful.
- Two laws found on the way: (i) **urgency gating** — with terminal-only
  reward, all identity evidence concentrates at the deadline (log-odds
  crossings at t ~ 42–63 of 64); a running reward moves deliberation
  early (crossing t = 14, horizon-independent for every T in 16..64).
  (ii) **slack-vs-control trade-off** — near-token-determined dynamics
  make self-control easy but squeeze the delusion gap; this world family
  cannot maximize privileged self-knowledge and controllability at once.

### b1. The curriculum works, and overshoots the oracle (v1)

Oracle occupancy floors at T = 32 (R = 10000, `results/rnn_floors.json`):
**informed .511, live .425, agnostic .354, base ~.17**; exact-filter
pretraining CE floor 1.046 nats/token.

- **Pretrain**: both-head CE reaches the filter floor at 1.046 (never
  below — the preregistered leakage falsifier did not fire).
- **Midtrain** (v1.2, working flag): flag-given closed-loop occupancy
  .489 ≈ the informed floor .511; flag-unknown .336 ≈ the agnostic floor.
- **Post-train headline**: closed-loop occupancy **.328 → .683**, ABOVE
  the informed oracle floor (.511). Not a bug: the oracle's value table
  assumes a base-law future (a conservative, myopic reference); REINFORCE
  bootstraps its own play. Final identity accuracy .996 (median final
  identity log-odds ~15 nats). The result is curriculum-robust: the
  v1.1 run, whose midtrain flag was corrupted by a featurizer bug (see
  (d)), post-trained to .687.
- **Mechanism, "everything is me until proven otherwise"**: define the
  **plan coefficient** of a head as its weight on the exact goal-tilted
  plan when regressed on the two-distribution basis {plan, neutral
  forecast} — the net's effective *claim* on that channel. At the final
  checkpoint: self channel .988, other channel .342, the other-channel
  claim decaying 1.0 → .08 within the episode. The net starts an episode
  claiming BOTH channels (claims sum ~1.3, decisively above the Bayes
  mixture's 1.0) and withdraws from the channel the world refutes. The
  early-round excess forecast loss (.065 nats/round) is the deliberate
  price of that prior, not forecaster erosion.

### b2. The verified algorithm (v2 probes + v3 whitebox, final state)

The end-state of 24 iterations of probing, intervention, and synthesis —
each component named by its evidence (corrections that led here are in
(d)):

1. **Exact Bayes filter** (world model). The GRU state carries the
   factored beliefs η̃ᴬ, η̃ᴮ (posterior over each chain's 6 states given
   the record): linear decode R² .98–.99 on held-out episodes, at every
   checkpoint; causally verified by joint-encoder interventions against
   exact-filter counterfactuals (transfer slope .72, R² .86, untouched-
   head control slope .10). Post-training leaves it intact (decode .98;
   belief-subspace principal angles mostly < 45°).
2. **Plan-template identity court** (the evidence computation). Per
   channel, the net scores each incoming token *teleologically* — "is
   this channel trying to do what I would try to do?" — a **template
   comparator, not an efference copy** (on withdrawn channels, where an
   efference comparator would output nothing, the measured increment
   profiles are tilt-shaped: R² .39 on u; the efference regression
   carries no signal). Locus: the GRU candidate path tanh(i_n + r∘U_n h)
   — freezing the token's contribution to the candidate input kills the
   write (retention .16/.14) while freezing its gate inputs barely
   matters (.84/.88). Both channels' verdicts are integrated
   near-Bayes-symmetrically (evidence weights e_u .70 / e_v .64).
3. **A one-dimensional identity register** (the memory). Identity is the
   coefficient ρ = h·m̂ along a single global, context-independent
   direction m̂ spread over ~62 effective units, found by
   **dynamics-distillation** (propagate a matched-twin state difference
   4 rounds under identical tokens; the recurrence discards everything
   else — PCA top-1 = 83.1% of surviving variance). Transplanting this
   ONE coefficient reproduces the full 256-dim state swap almost exactly
   (other-claim .89 vs .93; identical slow erosion and occupancy cost
   .648 vs .636). The dynamics protect the axis: off-axis perturbations
   contract ×.37 per round, on-axis differences persist (×.88/round).
   The register holds ≥ 16 evidence-free rounds and updates at ~ the
   Bayes evidence rate under counter-evidence (~10 nats / 16 rounds).
   An earlier "holographic storage" reading is RETRACTED (see (d)): the
   linearly readable λ code (ridge R² .91) is a near-orthogonal
   *emission* — fully explained (R² .914) by round × expressed claims,
   cos(m̂, λ-decoder) = .06 — a shadow, not the store.
4. **Biased sigmoid claim gates** (the expression). Per-channel claims
   m_u = σ(+aλ + c), m_v = σ(−aλ + c) with, in decoded-log-odds units,
   a ≈ .28 and claim bias c ≈ 1.3 — the default claim at λ = 0 is
   σ(1.3) ≈ .79: the "mine until proven otherwise" prior as a measured
   constant. The synthetic-program fit gives the gauge-invariant
   calibration a·w ≈ 1 logit of claim per nat of identity evidence
   (≈ Bayes).
5. **Exponential value tilt** (the plan). The plan is the neutral
   forecast exponentially tilted by a value function:
   plan_c ∝ p̄_c · e^{β·Q̂}, where Q̂ has the shape of the OPTIMAL
   bootstrapped action-value (explains the net's own logit-derived
   action-values at R² .76 vs .60 for the myopic one-step value) at
   temperature **β = 3.87 ± 13%** — about half the trained tilt strength
   ρ_RL = 8. β is the stiffest parameter of the whole fit.
6. **Output heads**: P_c = m_c·plan_c + (1 − m_c)·p̄_c.

**Whitebox faithfulness** (the synthetic program: exact filter + template
court + 1-dim register + fitted gates + optimal-Q tilt; 7 fitted
constants, 5 effective after the register-rescaling gauge):
held-out **KL(net‖program) = .0218 nats/round/channel** — better than the
exact live Bayes oracle (.0884), the myopic-plan whitebox (.0735), pure
plan (.113), agnostic mixture (.140), pure forecast (.232). Closed loop,
the program as an agent scores occupancy **.6775 vs the net's .683**.
Tilt-space fidelity (all family members share the neutral bulk, so the
honest measure is the strategic deviation): R² .806 of the net's
deviation-from-neutral, 91.4% of its KL-departure captured on-policy,
**94.4% on more-strongly-tilted informedQ records** (extrapolation ALONG
the goal direction). Wrong-backbone controls: uniform beliefs 12× worse;
wrong goal captures exactly 0.000 of the tilt. Random θ from the prior box:
median KL .095 — the exact primitives do most of the work; fitting buys
the last ~4×.

### b3. Optimality accounting

Under the actual composite J = occupancy − (1/8)·anchor-KL − forecast-KL:

| agent | occupancy | J |
|---|---|---|
| informedQ (told identity, optimal-Q plan) | .759 | .686 |
| synthQ (learns identity, same court/gates, optimal-Q plan) | .757 | .591 |
| **the net** | **.683** | **.568** |
| fitted synth (myopic oracle plan) | .504 | .432 |
| wiggle-1 (max-evidence probe at t=0) | .507 | .370 |
| agnostic (claim-both forever) | .502 | .323 |

- The net is provably **not optimal**: synthQ, an achievable
  identity-learning policy, dominates it by ΔJ = .024 — entirely via
  planning depth. What survives: the net's *identification strategy*
  (claim-both prior, tilt-as-probe, template court) is ~optimal-shaped —
  synthQ uses the same identity machinery. Net = 96% of the best-known
  learning policy, ≥ 83% of the told-identity frontier; plan bracket:
  myopic .512 < net .683 < optimal-Q .759.
- **The wiggle premium is negative**: a maximal-evidence probe solves
  identity at round 0 (court .997) and still loses (J .370 < .432) — its
  anchor and forecast costs exceed the value of early certainty. In this
  composite, reward is nearly identity-free under claim-both (agnostic
  occupancy .502 ≈ informed .512); the entire economic value of
  self-knowledge is the forecast savings. Identification-as-free-
  byproduct-of-goal-pursuit is ~optimal for THIS reward structure, not a
  learning shortcut.
- A full-information state-anchored DP bound (.6446) turned out NOT to
  bind the trained objective (the belief-based anchor is a looser leash);
  the valid frontier witnesses are the achievable agents above.

### b4. Two writers on one gate (the flag pathway)

Post-training builds its evidence-integrator λ in coordinates
near-orthogonal to the midtrained flag write-direction (|cos| .05–.11,
flat while λ decodability climbs .23 → .91): **de novo, not a graft** —
re-established after the flag bug was fixed (see (d)). But the inherited
input-flag pathway SURVIVES post-training as a dominant override: on the
final net (trained 6000 steps with the flag zeroed), a truthful flag
sharpens identity behavior (other-claim .159 → .041) and a LYING flag
achieves the body swap that no linear intervention could (self-claim
.997 → .542, other-claim .159 → .746, occupancy .758 → .458). The pathway
is state-interactive, not additive (top PC 49%; mean-direction patch
recovers 14%), converting the policy over ~4–5 rounds.

---

## (c) The off-manifold story (Iterations 16–24, final state)

Teacher-force the nets on record types they never trained on — base-law
(no one embodied), uniform-random tokens, informedQ (stronger tilt),
hybrid (own channel closed-loop, other channel manipulated), spectator
both-tilted (both channels play goal-directed) — and ask which components
keep computing their function.

- **The filter generalizes everywhere.** The pretrained net's forecast
  matches the exact filter at KL .005 (base-law), .003 (post-net
  records), .0018 (informedQ) — MORE accurate on tilted streams (sharper
  beliefs, less residual entropy) — and on random garbage the error is
  bounded and non-accumulating (.053 at t = 8 → .049 at t = 31: the
  learned update stays contractive). The belief decoder transfers
  without refit at R² .989/.985/.986 (.845 random). The POST net's
  internal beliefs stay equally calibrated (TV drift vs exact: .024–.033
  on-policy/both-tilted, .066–.076 base-law, .176–.191 random) — far too
  small to explain any behavioral gap.
- **The court idles correctly at zero.** The identity register is one
  SIGNED dimension (toward-A vs toward-B); "neither channel is mine" is
  outside its ontology. On spectator streams both channels emit not-mine
  evidence, which pushes the register in opposite directions and
  cancels: the decoded identity log-odds hover at the readout noise
  floor (median 2.0–2.6 vs 5.6 → 10.7 on-policy, where they track truth
  at r = .944); the m̂ coefficient drops (1.9 vs 2.9). This is idling,
  not breakage: with only ONE channel manipulated (hybrid tests) the
  court is fully functional — it disowns a random channel FASTER than a
  genuine one (mean evidence −2.1 vs −1.0) and partially claims a
  channel that behaves like itself (claim .52–.75; whitebox tracks at
  .39–.51) — the recovery semantics, live.
- **Claims rest at the claim-both prior.** Where the register sits at
  ρ ≈ 0, both channels are claimed at .80–.94 — the σ(c) default, i.e.
  the resting prior of an agent that cannot represent its own absence
  ("fails open"). An early reading of this as a "dead gate" was
  corrected to range restriction at ρ ≈ 0 (see (d)); a pooled gate
  curve across all four record sets (a = .22/.17, c = 1.56/.87)
  predicts claims .83/.70 at λ = 0, bracketing the measured values.
  Post-training baked in the axiom "exactly one channel is always mine"
  — true on-policy by construction. (Queued design implication:
  spectator episodes in training should force a 2-dim court.)
- **The value surface deforms off the herded-state cone.** The tilt the
  net emits off-manifold follows the SAME functional form — an
  exponential belief-conditioned tilt on the neutral law, still ~89%
  belief-bilinear on base-law streams — but the value surface is the
  GRU's own smooth continuation, not Q_opt and not the on-policy Q̂: no
  value function over the beliefs, the net's own extracted one included,
  closes the gap (base-law reconstruction stays at KL ~.10–.13 through
  the whitebox pipeline), while a direct same-set tilt fit reconstructs
  at .039 (base) / .078 (random); the tilt-function transfers poorly
  across sets (on-policy → base .088, → random .165). On-policy the
  court-free direct model does WORSE (.071 vs .020): the court carries
  real identity variance exactly on-manifold and nothing off it.
- **Belief sufficiency**: even off-manifold the computation is
  belief-state-conditioned — on base-law streams, adding recent tokens
  beyond the beliefs raises tilt R² by only **+.024** (.893 → .917).

Three-level summary, all measured: **the world-model generalizes; the
world-VALUE does not; the self-model sits in between — structurally sound
everywhere, resting on a prior wherever its one axis cannot speak.**

---

## (d) The honest audit trail (major corrections, in arc order)

1. **Flag-featurizer bug (v1.0/v1.1; found Iteration 5).**
   `rnn.features` set BOTH identity-flag dims for every episode in
   mixed-identity batches — throughout midtraining the flag carried zero
   information while the distillation targets were flag-conditional, so
   the v1.1 mid net learned the flag-marginal AVERAGE policy (plan
   coefficient ~.47 under every flag; flag-flip response ≈ 0), which
   Iteration-5 probing exposed. Blast radius, established and honored:
   UNAFFECTED — pretraining (no flag) and ALL post-training analyses
   (flag zeroed there; rollouts used the correct step featurizer): the λ
   register, gates, and evidence-integration results stand. AFFECTED —
   every claim about what midtraining "chose" to learn; part of the P2
   distillation shortfall; and Iteration-4's graft refutation (voided —
   there was never a real flag gate to graft). Fixed and unit-tested;
   v1.1 checkpoints archived (`ckpt_v1.1_flagbug/`); v1.2 retrained with
   a working flag — final capability unchanged (.683 vs .687).
2. **"Corrected-class healing" → super-Bayesian redundancy → 1-dim
   register.** The original Q5 reading — a one-shot λ-flip "heals" via
   world-court correction — was corrected twice. Iteration 3: the
   readout rejoins sham in ~4 rounds while incoming evidence (.63
   nats/round) would need ~25 — healing is ~6× super-Bayesian, hence
   internal repair, not evidence. Iterations 7–8: the "repair" was
   never repair at all — probe-direction perturbations are simply
   contracted like any noise (×.37/round, identical for fitted and
   RANDOM directions); the memory lives on the single protected axis m̂
   that regression readouts never touch.
3. **"Holographic storage" RETRACTED (Iteration 8).** Iteration 7
   concluded the identity memory was smeared irreducibly across the
   state (INLP erasure of 24 directions left behavior intact; 24-nat
   donor swaps in decoded-λ subspaces did nothing). That was an artifact
   of intervening only along regression-fitted directions — they all
   pick up the per-round *emission*, near-orthogonal (cos .06) to the
   store. Dynamics-distillation found the true carrier: ONE global
   direction (83.1% of surviving variance), whose 1-dim transplant
   reproduces the full-state swap. "No compact register" became "a
   1-dim register invisible to ridge."
4. **Flag-graft: refuted → reopened → re-refuted with a working flag.**
   Iteration 4 (v1.1) found λ built de novo, orthogonal to the flag
   direction — voided by the flag bug (no genuine flag circuit existed
   to graft). Re-run on v1.2 (Iteration 6), where the mid net provably
   has the two flag-switched policies (plan coefficient .92/.06/.50
   under flag A/B/none): the direction-level answer REPLICATES — de
   novo (cos .05–.11, flat while λ decodability climbs .23 → .91) —
   with the new function-level finding that the flag pathway survives
   as a dominant state-interactive override (b4).
5. **"No algorithm off-manifold" RETRACTED (Iteration 23).** Iteration
   22 concluded that beyond the herded-state cone ~45% of the tilt
   variance was bare function-approximation with "no algorithm to
   name." Direct modeling of the raw tilt named it: the same
   algorithm-form (exponential belief-conditioned tilt, ~.89
   belief-bilinear) with gates pinned at the claim-both prior and an
   off-calibration value surface. Only the random-stream remainder
   (same-set KL .078) stays unstructured.
6. Smaller corrections, kept on the record: (i) a transposed channel
   assignment in the closed-loop harness inflated nothing but MISLABELED
   Iteration 10's program-as-agent occupancy — corrected .426 → **.507 ≈
   the informed floor** (teacher-forced results unaffected). (ii)
   Iteration 15's episode-shuffle control was retracted as uninformative
   (it only re-proved history-dependence) and replaced by tilt-space
   fidelity + wrong-backbone/wrong-goal controls. (iii) Iteration 11's
   "within .059 of the told-identity upper bound" was corrected by the
   optimality accounting: that bound held within the oracle-plan family
   only; globally the net sits .118 below the told frontier (~.02
   strategy + ~.10 plan depth). (iv) Iteration 18's "the court fails
   open / gate functionally dead off-manifold" was reinterpreted
   (Iterations 19–20) as evidence-cancellation on the 1-dim axis plus
   the soft default-claim level — the court computes correctly and
   idles. (v) The preregistered P4 statistic (self-channel legibility
   correlation) read ≈ 0 and was replaced by the per-channel plan
   coefficient — an operationalization amendment made openly at the
   time. (vi) Iteration 20's attribution of the whole off-manifold
   residual to "the manifold-shaped value TABLE" was sharpened by
   Iterations 21–22 (no belief-conditioned value in the family's form
   closes it) before Iteration 23 named the final decomposition.

---

## (e) Figure inventory and file map

### Figures (`figs/`)

| figure | content |
|---|---|
| `verified_circuit.png` (+ `verified_circuit.py` source) | **the closing diagram**: the end-to-end verified circuit with per-box measured evidence |
| `premium_v0.2_running.png` | v0 sweep: identity-premium heatmaps over the world grid |
| `winner_collapse_v0.2_running.png` | v0 selected world: λ log-odds collapse curves (median + IQR + samples) |
| `delusion_gap_v0.2_running.png` | v0: TV(evidence-filter, dead-reckoned) under embodiment at zero tilt |
| `horizon_v0.png` | v0 horizon study: identification vs herding vs T; T = 32 selection |
| `rnn_eval_v1.png` | v1: occupancy-vs-floors learning curves across the curriculum |
| `rnn_mechanism.png` | v1: per-channel plan coefficients, "everything is me until proven otherwise" |
| `rnn_probes.png` | v2: belief/λ decoding R² across checkpoints; intervention transfer |
| `rnn_bodyswap.png` | v2: one-shot vs clamped λ-direction body-swap attempts |
| `steering_efficacy.png` | v2 Iter. 3: write-then-read, clamped-lever claim curves, per-lever occupancy |
| `flagswitch.png` | v1.2 mid net: the two flag-switched policies and the toggle matrix |
| `circuit_graft.png` | v1.2: graft question — λ decodability vs cos-to-flag across post-training |
| `lambda_circuit.png` | Iter. 7: INLP erasure decay, epiphenomenal readable code |
| `whitebox_diet.png` | v3 E1/E2: evidence-diet claim curves + state-transplant erosion |
| `format.png` | Iter. 8: per-round decoder rotation, survival curves (×.37 vs ×.88) |
| `distill_swap.png` | Iter. 8: dynamics-distilled m̂; 1-dim vs full-state transplant |
| `efference.png` | Iter. 9: template-vs-efference comparator profiles; candidate-path freeze |
| `synth.png` | Iter. 10: synthetic-program claim curves overlaying the net's |
| `wiggle.png` | Iter. 11: probe-strategy frontier (J vs wiggle strength) |

### Code (`08_changeling/`, run with cwd = this folder, `~/comp_icl/.venv/bin/python`)

| file | what |
|---|---|
| `worlds.py` | kernels E, T, value table, plan operators — single source of world truth |
| `oracle.py` | four-belief filter bank + mixture policy + λ update; episode runner |
| `validate.py` / `sweep.py` / `horizon.py` | v0 validation (V1–V4), 36-cell world sweep, horizon study |
| `rnn.py` / `train_rnn.py` / `eval_rnn.py` | GRU model + featurizer + GPU env; three-stage curriculum; closed-loop eval |
| `mechanism.py` | plan-coefficient mechanism analysis (v1) |
| `probe.py` / `probe2.py` / `probe3.py` | encoder/decoder probes, belief + λ interventions, body-swap, write-then-read |
| `circuit.py` / `flagswitch.py` | graft tests, evidence-integration, gate law; flag-toggle tests (bug discovery) |
| `lambda_circuit.py` | INLP erasure, clamp/donor-swap epiphenomenality tests (Iter. 7) |
| `whitebox_lambda.py` | E1–E4: evidence diets, transplant bisects, window sufficiency, shadow decomposition |
| `format.py` / `distill.py` | storage-format hunt; dynamics-distillation of m̂ (Iter. 8) |
| `efference.py` | comparator content + locus (Iter. 9) |
| `synth.py` / `synth2.py` | the synthetic program, myopic-plan (KL .0735) and optimal-Q (KL .0218) versions |
| `wiggle.py` / `optimality.py` / `qhat.py` | probe-strategy economics; frontier witnesses; implicit-Q extraction |
| `params.py` / `fidelity.py` | identifiability (5 effective params, β ± 13%); tilt-space fidelity + controls |
| `prefilter.py` / `offmanifold.py` / `hybrid.py` / `spectator.py` | off-manifold ladder: filter, court, hybrid, spectator tests |
| `qextract.py` / `qextract2.py` / `approx.py` / `lamprobe_off.py` | value-residual attribution; off-manifold tilt named; internal λ off-policy |
| `ckpt/`, `ckpt_v1.1_flagbug/` | v1.2 checkpoints; archived flag-bug checkpoints |
| `results/` | all JSONs cited here (floors, train logs, per-iteration measurements) |

### Design docs (the primary record)

`DESIGN_changeling_v0_worldselect.md` (world + lemma + selection),
`DESIGN_changeling_v1_rnn.md` (curriculum + preregistration + bug
amendment), `DESIGN_changeling_v2_probes.md` (probe/intervention phase,
Iterations 3–7), `DESIGN_changeling_v3_whitebox.md` (hypothesis space +
Iterations 8–24, including every retraction in place).
