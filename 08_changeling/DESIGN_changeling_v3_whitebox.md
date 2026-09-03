# DESIGN — changeling v3: whitebox of the identity mechanism (hypothesis space + discriminating tests)

*2026-09-01. Target: the v1.2 post net (single-layer GRU d=256; verified
manual recompute harness, max diff 1.5e-6). Facts to explain (Iterations
3-7): λ linearly readable (R² .91) but high-dim redundant (24 INLP dirs →
R² .31) and causally inert (erasure/24-nat swaps change nothing; heal in
1-2 rounds); claims gated σ(±.28λ+1.3); only the state-interactive flag
INPUT ever swapped identity.*

## Hypothesis space (preregistered before testing)

- **H1 low-dim integrator register + gate readout** — REFUTED (Iter. 7).
- **H2/H5 gain coding (bilinear):** no λ variable; identity stored as the
  per-channel AMPLITUDE of the context-dependent plan-tilt pattern in h
  (m_c · tilt_pattern_c(context)); patterns rotate with context so linear
  probes see the code only diffusely (the shadow), while the heads read it
  exactly. Memory = the gains persist through the recurrence. H5: the two
  channel gains are independent coordinates.
- **H3 saturation/attractor code** — disfavored a priori (a latched sign
  pattern is linearly decodable; INLP would have caught it; erasure-to-mean
  did nothing, not even transiently).
- **H4 windowed re-derivation, no persistent identity state:** the gate
  saturates by λ~8 nats, so behavior needs only a few rounds of "were my
  intentions echoed?"; identity is recomputed each round from recent
  evidence. The memory is the token stream itself — which no intervention
  touched. Predicts all Iter.-7 results (erasure-proof, 1-2-round healing =
  window refill, shadow = round × sign byproduct).
- **H6 expression funnel:** whichever carrier, its expression converges on
  the gate machinery the flag input drives.

## Discriminating experiments

- **E1 evidence diet** (teacher-forced continuations from t=16 after a
  natural closed-loop prefix): (a) NEUTRAL tokens (per round, per channel,
  argmin |log π − log p̄| — zero identity evidence): integrator (H2) HOLDS
  claims ≥ 12 rounds; window re-deriver (H4) RELAXES to the claim-both
  default σ(1.3)≈.79 within ~its window. (b) p̄-fed (both channels look
  genuine): counter-evidence on the self channel — measures the un-learning
  rate, behaviorally. (c) π-fed (both look mine): other-claim should rise.
  (b)+(c) read the update rule out behaviorally — what the shadow-based
  counterfactual test could not.
- **E2 memory-vs-expression transplant bisect** at t=16 between matched
  pairs (same goal, opposite identity, nearest beliefs): swap (a) full h —
  positive control, should persist and erode only at the evidence rate
  (~.6 nats/round against the transplanted deficit); (b) the 12-dim HEAD
  ROWSPACE component only (behavior factors through W_u⊕W_v, so this must
  swap the CURRENT round's behavior; persistence ⇔ rowspace is also the
  memory); (c) the complement only (inert this round; later re-expression
  ⇔ memory lives outside the rowspace).
- **E3 window sufficiency:** regress per-round claims on the last-W exact
  evidence terms, W ∈ {1,2,4,8,16}, vs on cumulative λ. H4: small W matches
  the full history. H2: cumulative wins.
- **E4 shadow decomposition:** R² of λ from (round, claims, round×claim
  asymmetry) — how much of the .91 decode is time × expressed sign.

Falsifier discipline: if E1-neutral holds AND E2(b) persists, the memory is
the expressed gains (H2, memory=expression); if E1-neutral relaxes AND E3
saturates at small W, H4 wins; full-h swap eroding at ~evidence rate vs
~window rate separates integrator-somewhere from H4 globally. Mixed
outcomes localize the split (e.g., gains persistent but window-topped-up).

## Files

whitebox_lambda.py → results/rnn_whitebox_lambda.json,
figs/whitebox_diet.png, figs/whitebox_transplant.png.

---

## Measured outcomes (2026-09-01, same session)

**E1 (diets).** NEUTRAL: self-claim holds .993→.988 over 16 evidence-free
rounds (other-claim drifts .21→.34) — a persistent integrator exists; H4
(window re-derivation) is dead. P̄-DIET (counter-evidence): other-claim
rises .21→.81 while self-claim falls only .99→.92 — exactly the shared-λ
sigmoid gates at different operating points (self-gate deep in saturation,
other-gate at its linear range); implied internal λ falls ~10 nats/16
rounds ≈ the Bayes evidence rate — the write rule confirmed behaviorally.
π-DIET: flat (the withdrawn v-head ≈ p̄, so the diet is uninformative
there) — consistent.
**E2 (transplants; note: measured one recurrent step after the swap).**
FULL-state swap between matched opposite-identity pairs: claims jump toward
the donor identity (self .99→.71, other .21→.88) and erode back over ~15+
rounds (evidence-rate scale); occupancy cost real (.761→.652). ROWSPACE-only
(the 12 dims the heads read) and COMPLEMENT-only swaps both produce only a
transient other-claim bump (healed by ~8 rounds) and never move the
self-claim — EITHER half regenerates the whole from the other. Holographic
storage signature: no subspace carries the memory exclusively; only the
full state does. (Retroactively explains Iter.-7: 24-dim erasure left 232
dims of redundant copy.)
**E3.** Other-claim R² from last-W evidence: .001/.046/.13/.31/.57 for
W=1/2/4/8/16 vs cumulative-λ .59 — behavior tracks the long integral, not
a short window. H4 dead twice.
**E4.** The λ-shadow is fully explained by round × expressed claim
asymmetry: R² .914 ≈ the .91 ridge decode. The readable λ is a computed
display of (time, expression), not the store.

**Hypothesis scorecard:** H1 dead (Iter. 7). H3 dead. H4 dead (E1+E3).
H5 (independent per-channel gains) down-weighted — the p̄-diet coupling
(other-claim rising as self-evidence falls) is exactly shared-λ gating.
Verdict: **a single persistent identity integrator, updated at ~the Bayes
rate, stored holographically/redundantly across the whole state (no
carrier subspace), expressed each round through biased sigmoid gates
σ(±.28λ + 1.25), with the linear 'λ code' a mere shadow (time ×
expression).** Remaining open: the micro-format of the distributed store
(unit products / gate configurations) — DAS + gate-freezing surgery with
the verified harness; and H6 (does the flag input feed the same funnel).
Figures: figs/whitebox_diet.png (diet curves + transplant curves).

## Iteration 8 (2026-09-01; format.py + distill.py) — THE FORMAT, FOUND

"Holographic" was an artifact of intervening only along regression-fitted
directions. The specific storage format:

- **F1/F2**: per-round λ decoders reach R² .93 and their directions ROTATE
  (adjacent-round cos .64; 16 rounds apart .41), but even per-round-fitted
  and rotating-clamped swaps stay inert — rotation is not the answer.
- **F3 (survival)**: matched-norm perturbations through paired
  identical-token rollouts: the full twin-difference decays ×.88/round
  (memory-like) while pooled, per-round, AND RANDOM directions all collapse
  identically (×.37 after one round) — the dynamics treat every decoder
  direction exactly as noise; 'healing' was always just contraction.
- **F4**: no latched-unit committee (2/256 units with z̄>.9); the identity
  difference is spread over ~62 effective units.
- **DISTILLATION (the closure)**: propagate the matched-twin difference 4
  rounds under paired identical tokens — the dynamics discard the noise
  components (67.8% of norm survives) and what remains is, after sign
  alignment, ONE GLOBAL DIRECTION: PCA top-1 = 83.1% of variance,
  participation ratio 1.4. **Transplanting the 1-dim coefficient along
  this direction reproduces the full 256-dim state swap almost exactly**
  (self .79 vs .69, other .89 vs .93 at t+0; identical slow evidence-rate
  erosion and occupancy cost .648 vs .636; PCA k=2/k=8 add nothing).
- The direction is **near-orthogonal to every readout**: cos .06 with the
  matched-round λ decoder — the store and its per-round emission (what all
  probes read; = time × expression per E4) are separate, almost
  perpendicular objects. Ridge provably picked the emission (it carries
  the λ-covariance); the store carries the causal power.
- Flag pathway (H6): cos(memory dir, flag write-direction) = .14 — the
  flag does not write the register additively; it moves it through the
  recurrent dynamics (matching the measured 4-5-round flag-flip
  conversion).

**The verified format:** identity is a 1-DIMENSIONAL REGISTER after all —
the coefficient along a single, global, context-independent direction m̂,
spread over ~60 units, (i) protected by dynamics that contract everything
off-axis ×.37/round, (ii) plausibly re-polarized under partial writes
(explains half-swap healing; bistability test pending), (iii) invisible to
regression readouts, which latch onto the near-orthogonal per-round
emission, and (iv) findable only by letting the network's own dynamics
filter a genuine memory difference. Loose ends: decode λ from h·m̂ alone
(magnitude vs sign content); bistability/basin test along m̂; then the
synthetic program + verified-circuit diagram (all components now in hand:
evidence terms → m̂-register → σ-gates → heads).

## Iteration 9 (2026-09-01; efference.py) — the write mechanism, read on the register

Architecture note for the record: single-layer GRU — ONE state h_t; heads
read h_t directly; h_{t+1} = GRU(input_t+1, h_t). The efference copy is
structurally free (the state that generated the intention IS the next
recurrent input; the intended DISTRIBUTION, not the sample, is what
persists — the sample returns as input).

- **Register content:** rho = h·m̂ decodes λ at only R² .376 (late sign
  accuracy .72). The register is coarse/sign-like — the causal store, not
  the graded readout; the fine-grained readable λ (R² .93) is assembled
  downstream (time × gate expression), matching E4 and the bistability
  picture.
- **The comparator is a PLAN TEMPLATE, not an efference copy.** On
  channels the net has withdrawn from (its own head ≈ p̄ there, so a true
  efference comparator would generate NO authorship evidence — flat
  counterfactual profiles), the measured register-increment profiles are
  tilt-shaped, matching the template hypothesis (u: R² .39, slope .18;
  v: R² .15) while the efference regression carries no signal (its
  hypothesis has ~no variance there, and the observed profile is not
  flat). Authorship is detected TELEOLOGICALLY — "is this channel trying
  to do what I would try to do?" — not by self-consistency with the
  current policy. Design rationale (why training found this): a template
  comparator preserves a recovery route (a withdrawn channel that starts
  obeying my goals re-accrues mine-evidence — the measured 94%
  wrong-commitment recovery needs exactly this), whereas an efference
  comparator makes withdrawal self-fulfilling and unrecoverable.
- **Comparator locus:** freezing the token's contribution to the CANDIDATE
  input i_n kills the write (slope retention .16/.14 on u/v) while
  freezing its gate inputs i_r, i_z barely matters (.84/.88). The
  token-vs-template comparison happens in tanh(i_n + r∘(U_n h)) — the
  token's one-hot slice meeting the state-carried template inside the
  candidate nonlinearity — not via gate modulation.

Updated verified circuit: evidence = TEMPLATE-match of each incoming token
(computed in the candidate path) → ±increments on the coarse m̂ register
(protected axis, re-polarizing) → per-round emission into rotating readout
coordinates (the λ shadow) → σ(±.28λ+1.3) gates → plan-vs-forecast heads.
Remaining for Silver/Gold: the synthetic numpy program + stitching test +
the verified-circuit diagram.

## Iteration 10 (2026-09-01; synth.py) — THE SYNTHETIC PROGRAM (behavioral Silver)

f_synth (all primitives exact, six fitted constants):
  1. beliefs: factored Bayes filter, input-indexed linear + normalize
  2. forecasts p̄ = ηE (linear); plans ∝ p̄ · (ηᴬᵀM_t[goal]ηᴮ) (bilinear)
  3. register: ρ += w_u·tmpl(u) − w_v·tmpl(v), clip ±M  (template-match)
  4. gates m_u = σ(aρ + c_u), m_v = σ(−aρ + c_v)
  5. heads = m·plan + (1−m)·forecast.
Fitted: w_u 1.65 ≈ w_v 1.61 (symmetric evidence weighting), a .64,
c_u 3.59 / c_v 3.33 (default claim σ(3.5) ≈ .97 — matches the measured
round-0 claim-both), clip 15.8 (register saturation).
**Held-out faithfulness: KL(net‖synth) = .0735 nats/round/channel — better
than every reference: live Bayes oracle .0884, pure plan .113, agnostic
mixture .140, pure forecast .232.** The claims curves overlay the net's
almost exactly (figs/synth.png). Closed loop, f_synth as the agent scores
occupancy .426 = the live-oracle floor (.425), vs the net's .683: the
identity machinery is fully captured; the residual is PLAN quality — the
net's RL-improved plan beats the oracle h-table primitive the program
borrows. That plan module is the one remaining black box.
Status on the skill ladder: behavioral Silver for the identity circuit
(distributional match + closed-loop dynamics), with the causal component
verifications already banked (register transplant ≈ full swap; belief
interventions slope .72). Queued for Gold: activation-level stitching with
per-round embed maps + the verified-circuit diagram (example.png standard).

## Iteration 11 (2026-09-02; wiggle.py) — the wiggle premium is NEGATIVE

BUG NOTE first: the closed-loop agent harnesses in synth.py/wiggle.py had a
transposed channel assignment (vn took the net sample when iota=A);
teacher-forced results unaffected; Iteration 10's "f_synth closed-loop occ
.426 = live-oracle grade" is CORRECTED to **.507 ≈ the informed floor
(.512)** — the program as an agent nearly matches known-identity play (the
net's .683 gap over all oracle-plan agents remains pure plan quality).

Measured under the actual composite J = occupancy − (1/8)·anchor-KL −
forecast-KL (all closed-loop, R=3000, same plan primitive so the strategy
comparison is controlled):

| agent | occ | anchor | forecast | J | identity solved |
|---|---|---|---|---|---|
| informed (told) | .512 | .166 | 0 | **.491** | — |
| fitted synth (tilt-as-probe) | .504 | .163 | .052 | **.432** | ~round 17 (5-nat) |
| agnostic (claim-both forever) | .502 | .159 | .159 | .323 | never |
| wiggle-1 (max-evidence probe) | .507 | .214 | .111 | .370 | round 0, court .997 |
| wiggle-2/3/5 | .48/.47/.45 | ↑ | ↑ | .18/.01/−.34 | round 0 |

Readings: (1) **reward is nearly identity-free under claim-both** —
agnostic occupancy .502 ≈ informed .512: the entire value of
self-knowledge in this composite is the forecast savings (.159→.052 ≈
.11). (2) **Maximal-evidence probing works epistemically (identity at
round 0, court .997 — confirming the ~4.8 nats/round bound) and loses
economically**: its own costs (anchor spike + the probe round's terrible
forecasts) exceed the value of earlier certainty. J(wiggle-1)=.370 <
J(synth)=.432. (3) The net's learned strategy — identification as a free
byproduct of goal-directed tilt — is within .059 of the told-identity
upper bound, and that residual is mostly the irreducible cost of having
to learn identity at all. Conclusion: the no-wiggle, claim-both,
tilt-as-probe solution is ~optimal FOR THIS REWARD STRUCTURE, not a
learning shortcut. To make probing optimal, the composite must gate
REWARD on identity (single action budget, wrong-tilt penalties,
conflicting goals) or raise the forecast weight — the selfhood-pressure
dials from the v1 design notes.

## Iteration 12 (2026-09-02; optimality.py) — optimality accounting, and the plan module identified

(A) Full-information KL-control DP (36 joint states, exact): J*_fullinfo =
.6446 — but anchored to the TRUE-STATE emission law, while the training
objective anchors to the belief-based p̄; the achievable belief-anchored
informedQ agent scores J = .6858 > .6446, so the state-anchored DP is NOT
a valid bound for the trained objective (the belief anchor is a looser
leash: deviating from a diffuse p̄ costs less than from the sharp E(·|s)).
Valid structure instead: J_net ≤ J*_learning ≤ J*_told, with achievable
witnesses below.
(B) QMDP agents with the DP-optimal Q as plan primitive:
  informedQ (told identity):  occ .759, J .686   [told-frontier witness]
  synthQ (claim-both + template court + fitted gates, Q-plan):
                              occ .757, J .591   [learning witness]
  net:                        occ .683, J .568
**Consequences.** (1) Q1 ANSWERED IN THE NEGATIVE: the net is provably NOT
optimal among all policies — synthQ, an achievable identity-learning
policy, dominates it by ΔJ = .024 (and by .074 occupancy), entirely via a
better plan. The refined claim that survives: the net's IDENTIFICATION
strategy (claim-both, tilt-as-probe, template court) is ~optimal-shaped —
synthQ uses the same identity machinery; the shortfall is planning depth.
Net = 96% of the best-known learning policy, ≥83% of the told frontier.
(2) The 'mystery plan module' is BRACKETED: one-step base-law plan (occ
.512) < net (.683) < optimal-Q plan (.759). The net's planner is a
partially bootstrapped value — REINFORCE moved it partway from the myopic
h-transform toward the self-consistent optimum. Next: refit f_synth with a
planning-depth interpolation parameter and test whether teacher-forced
faithfulness beats KL .0735.
(3) Iteration-11's "within .059 of the told-identity upper bound" was
within the ORACLE-PLAN family only; globally the told frontier is ≥ .686
and the net sits .118 below it (~.02 strategy + ~.10 plan depth + identity
cost).

## Iteration 13 (2026-09-02; qhat.py) — the net's implicit Q, read off its logits

At any stationary point of the KL-anchored objective, log π − log p̄ =
ρ·Q^π + const(state), so Q̂ := (1/ρ)(log head − log p̄) is the net's own
action-value, extractable with no probes. Measured on claimed channels
(t ≥ 8, centered per round): Q̂ is explained by the OPTIMAL bootstrapped Q
at R² .76 (myopic one-step value: .596; joint adds nothing, refs correlate
.855) with slope .34 on ρQ_opt — i.e. the net's tilt has the optimal Q's
SHAPE at about one-third the optimal STRENGTH (effective ρ ≈ 2.7 vs the
trained 8), over-tilted 2.1x relative to the myopic value. This quantifies
the plan bracket (.512/.683/.759) in the value domain: right value
function, under-committed exponent — presumably where 6k more REINFORCE
steps would go.

## Iteration 14 (2026-09-02; synth2.py) — the net IS the optimal-weighted π_g, at temperature β ≈ ρ/2

Refit f_synth with plan ∝ p̄·e^{β·Q_opt} (optimal bootstrapped Q, learned
temperature β; 7 constants total). Held-out KL(net‖synth-v2) = **0.0218**
nats/round/channel — 3.4x better than the myopic-plan whitebox (.0735) and
4x better than the exact live Bayes oracle (.0884). Fitted β = 3.87 ≈ half
the trained ρ = 8 (supersedes the regression-slope estimate .34·ρ ≈ 2.7,
which was attenuation-biased). Closed loop, the refit program scores
occupancy **.6775 vs the net's .683** — the whitebox now matches the
network's performance as well as its distributions. Verdict on Asvin's
question: to the extent the whitebox describes the net (now to .022
nats/round), the net has learned the optimal-weighted policy — the
exponential tilt of the neutral law by the OPTIMAL value function — at
about half the objective's nominal temperature; the remaining gap to the
told-identity frontier is that temperature deficit plus the identity cost,
not value-function error. (Gate/register constants rescale with the new
register units; only invariant combinations are comparable across fits.)

## Iteration 15 (2026-09-02; params.py) — identifiability & coverage of the 7-parameter fit

Held-out KL at theta-hat: .02184 ± .00113 (SE over 384 episodes).
- **Effective parameter count: 5.** Hessian on log-params has TWO zero
  eigenvalues; the sloppiest eigenvector is (w_u +.51, w_v +.51, a −.51,
  clip +.47) — exactly the predicted register-rescaling gauge. Eigenvalue
  spectrum spans 0, 0, .0013, .0018, .016, .027, .217.
- **Pinnedness hierarchy** (2·SE profile intervals, as multipliers):
  β ∈ [.88, 1.14]x — the temperature is the stiffest quantity (the top
  Hessian direction is 95% pure β): β = 3.87 ± ~13%. Evidence/gate
  combinations intermediate (individually loose BECAUSE of the gauge).
  Claim-biases c and the clip are soft ([.4, 2.5]x) — c only matters in
  the opening rounds before evidence dominates; the clip rarely binds.
- **Coverage:** random θ from the prior box lands at KL .039–.234
  (5–95%), median .095 — the exact primitives (filter, Q, gate form) do
  most of the work; fitting buys the last ~4x. The family CANNOT reach
  unrelated targets: best fit to an episode-shuffled target (marginals
  kept, history-dependence broken) is KL **2.00** — ~100x worse than the
  true-target fit. The family covers a thin structured manifold of policy
  space; the net lies inside it, and the data pin the scientifically
  loaded parameter (the temperature) tightly.

## Iteration 16 (2026-09-02; fidelity.py) — proper fidelity measures (Asvin's critique; shuffle control RETRACTED)

The episode-shuffle control of Iter. 15 is retracted as uninformative: the
target heads are history-dependent, so even the net itself fails on
mismatched records — it only re-proved history-dependence. Raw KL is also
bulk-dominated (all family members share the neutral mass; the entire
dynamic range is the strategic deviation, total budget KL(net‖neutral) =
.232/round/channel). Replacements, per Asvin:

- **Tilt-space fidelity (on-policy):** R² of the net's
  deviation-from-neutral against the program's = **.806**; fraction of the
  KL-departure captured = **.914**.
- **Off-trajectory transfer, NO refit** (the algorithm-decoding test: the
  program is beliefs-from-record → policy, so it must match on any
  HMM-legal record): informedQ records (stronger tilt): frac captured
  **.944**, tilt-R² .843 — extrapolates ALONG the goal-directed direction
  better than on-policy. Base-law records: frac .68, tilt-R² .60.
  Uniform-random tokens: frac .65, tilt-R² .58. Honest boundary: the net
  implements the decoded algorithm faithfully on its manifold and beyond
  it in the tilt direction, but its behavior on neutral/never-visited
  streams deviates measurably from the exact algorithm (no training
  pressure there) — the decoding is manifold+extrapolation, not
  everywhere-exact.
- **Wrong-backbone controls (7 params refit each, on-policy):** uniform
  beliefs (filter off): KL .237 (12x worse — the filter is load-bearing).
  Misspecified kernels (q0 .9→.7, c_o .6→.45): KL .040 (2x). WRONG GOAL
  (rotated): KL .232 = the neutral baseline exactly, fraction captured
  0.000 — with the wrong goal the family can do no better than giving up
  the tilt entirely. The goal content is fully load-bearing: the proper
  answer to 'would the family fit anything history-dependent' is no.

## Iteration 17 (2026-09-02; prefilter.py) — the pretrained filter off-manifold

Teacher-forcing pre_final (native zero-goal inputs) across record types,
against the exact filter: KL(exact p̄ ‖ net head) per round/channel:
base-law (its manifold) .0051; post-net on-policy records .0029; informedQ
(strongly tilted) .0018; uniform-random tokens .0465. The filter is MORE
accurate on tilted streams than on its own manifold (informative tokens
sharpen beliefs → less residual entropy to model), and on random garbage
the error is 9x larger but BOUNDED and non-accumulating (drift curve flat:
.053 at t=8 → .049 at t=31 — the learned update stays contractive; it
tracks with constant lag rather than diverging). Belief-code stability:
the base-fitted decoder transfers WITHOUT refit at R² .989/.985/.986
(base/post/informedQ) and .845 on random tokens (refit .890).
**Attribution:** the post-net whitebox's off-manifold gap (68% captured on
base-law streams) is NOT the substrate's fault — the filter is near-exact
there (.005) — it lives in the post-trained decision machinery (court +
gates + tilt) running off-label on neutral streams it never met in
training. World-model: faithful far beyond the manifold. Self-model:
manifold-shaped.

## Iteration 18 (2026-09-03; offmanifold.py) — the fraying localized: the court fails OPEN off-manifold

MID net (flag=A, its two jobs, across the record ladder): KL(plan‖u-head) /
KL(p̄‖v-head) = .011/.008 on its manifold, **.024/.008 on base-law**
(tilt-R² .89 — fine), .072/.044 on random tokens (≈ the substrate filter's
own degradation), .005/.005 on post-net records. The flag-gated dual
policy GENERALIZES — the fraying is a post-training acquisition. Confirmed.

POST net, realized claims vs whitebox-predicted claims per record set:
- own records: familiar evidence geometry (both-negative evidence cone
  occupies only 6% of rounds); gate law present (refit slope .15).
- base-law records: the both-negative cone is 51% of rounds (both courts
  say 'not yours') — a region on-policy life NEVER produces — and there
  the net's claims DO NOT respond: pinned at ~.89-.91 all episode (synth
  predicts ~.56); gate-law refit slope .038, R² .03 — the gate is
  functionally DEAD. Same on random tokens (claims ~.92 flat, slope .03).
**The post net's court is wired only for the one-sided evidence geometry
its own on-policy behavior produces; faced with the never-visited
both-negative pattern it goes silent and the net FAILS OPEN — claims
everything at its 'mine until proven otherwise' prior.** The whitebox's
σ-gate extrapolates Bayes-wards where the net saturates — which is exactly
the 68%/65% fidelity boundary of Iter. 16, now explained. Asvin's
'on-policy self-calibration' hypothesis holds in a plain form: not subtle
efference calibration, but evidence-cone coverage.

Three-level summary, all measured: the world-model generalizes (filter:
Iter. 17); the instructed self generalizes (mid net: above); the INFERRED
self is manifold-shaped and fails open. Follow-up queued: register-vs-gate
localization of the silence (does the m̂ register move on neutral streams
and the gates ignore it, or does the write itself not fire?).

## Iteration 19 (2026-09-03; hybrid.py — Asvin's half-off-policy design) — REINTERPRETATION of Iter. 18

Post net acts its own channel closed-loop; only the non-actor channel is
fed off-policy:
- other=genuine (control): self-claim 1.0; other-claim decays →.10. Normal.
- other=RANDOM: court fully functional — the net disowns the garbage
  channel FASTER (other-claim →.11 by t=16; mean g_v = −2.1 vs −1.0
  genuine), self-claim untouched at 1.0.
- other=TILTED ('the other behaves like me'): the net partially CLAIMS it
  (other-claim .52-.75; whitebox tracks at .39-.51; gate law alive, refit
  R² .45) — the recovery semantics, live.
So off-policy tokens per se do NOT break the court. Combined with Iter. 18:
the anomaly occurs only on SPECTATOR records (no channel carries the
net's actions) — and the mechanism is STRUCTURAL, not a training gap:
**the identity register is one signed dimension (toward-A vs toward-B);
'neither channel is mine' is outside its ontology.** On base-law records
both channels emit not-mine evidence, which pushes the register in
OPPOSITE directions and cancels (g_u<0 → toward B, g_v<0 → toward A);
ρ hovers at ~0, and both net and whitebox rest at their ρ=0 default
claims. The Iter.-18 'dead gate' was range restriction at ρ≈0; the
net-vs-whitebox gap there is mostly the ill-pinned default-claim level
(exactly the c-parameter sloppiness measured in Iter. 15 — on-policy data
barely visits ρ≈0). The same cancellation explains hybrid-tilted
(both-positive evidence also cancels → claim-both, appropriately).
**Refined conclusion: post-training baked in the axiom 'exactly one
channel is always mine' — embodiment is guaranteed on-policy, so the
court's 1-dim design never needed a 'neither' state. The fail-open on
neutral streams is the resting prior of an agent that cannot represent
its own absence.** Design implication (queued): spectator episodes
(ι ∈ {A, B, neither}) in post-training should force a 2-dim identity
court — a sharp, falsifiable prediction.

## Iteration 20 (2026-09-03; spectator.py — Asvin's both-arms-π_g test and the final decomposition)

- **Both-tilted spectator streams** (u~piA, v~piB, net purely spectating):
  net claims high on both (.87/.83) as Asvin predicted, and the whitebox
  matches at ON-POLICY GRADE WITHOUT REFIT: KL .0231 (on-policy: .020).
  The cancellation→default behavior is real and SHARED by net and program.
- **Base-law / random spectator streams**: net claims also high (.91/.80,
  .93/.90 — 'emit the goal policy' approximately holds at the claims
  level), but whitebox KL stays high even after full on-set REFIT
  (.126→.101, .154→.122): the default-level (c) miscalibration accounts
  for only ~.02-.03 of the gap. The rest is the net's emitted TILT
  DIRECTION deviating (tilt-R² vs exact plan ~.58-.60).
- **The three-layer decomposition, final:** (1) the identity court &
  register generalize (hybrid + both-tilted: whitebox-grade). (2) the
  ρ≈0 default level is a minor, fixable calibration. (3) the true
  off-manifold residual is the VALUE FUNCTION: Q̂ is manifold-shaped in
  STATE space — trained only on herded state distributions. Both-tilted
  streams herd the chains (states stay on-manifold → whitebox holds);
  base/random streams wander into un-herded belief regions where Q̂
  deviates from the exact Q in ways no family parameter can absorb.
  Filter (Iter. 17): generalizes. Court (Iter. 19-20): generalizes,
  with the conflation-of-cancellation-states ontology and a claim-both
  prior. Value: manifold-shaped. **The world-model generalizes; the
  world-VALUE does not; the self-model sits in between — structurally
  sound everywhere, resting on a prior wherever its one axis cannot
  speak.**

## Iteration 21 (2026-09-03; qextract.py — Asvin's closure test) — the value-fraying attribution is INSUFFICIENT

Extracted the net's own Qhat per point (mixture-inverted with its claim
coefficient, beta fixed), fit the synth's value ontology (per-round
bilinear-in-beliefs tables) per record set, tested cross-set consistency
and whitebox reconstruction with the extracted value.
- Consistency: the Qhat function transfers across sets about as well as it
  fits within sets (cross .51-.77 vs within .62-.81) — no evidence of
  set-specific value functions — but it is only MODERATELY bilinear in the
  exact beliefs anywhere (within-set ceiling .62-.81).
- **Reconstruction: on-policy and both-tilted reconstruct at whitebox
  grade (.020-.037), but base-law and random streams stay at KL .11-.18
  EVEN WITH the net's own Qhat fitted on the very same set (base->base
  .107 vs exact-Q .126).** The un-herded-stream residual is therefore NOT
  a value-table error: no value function over the exact beliefs, the
  net's own included, closes it. Iteration 20's attribution ('the
  residual is the manifold-shaped value') is corrected: on un-herded
  states the net's emitted tilt is not representable in the whitebox's
  functional form (gates x exponential tilt of a belief-bilinear value)
  at all.
- Remaining suspects, next probes: (a) the POST net's own internal belief
  estimates on weird streams (pre-net beliefs transfer at .99/.85; post
  unmeasured — extract Qhat as a function of the net's DECODED beliefs
  instead of exact beliefs); (b) genuinely non-bilinear state dependence
  of its tilt off-manifold.

## Iteration 22 (2026-09-03; qextract2.py) — suspect 1 refuted; the algorithm has a support boundary

Decoded the post net's internal beliefs (readout fit on-policy, frozen,
applied everywhere) and ran the fully internal-belief whitebox (pbar and
value features both from decoded beliefs).
- **Belief drift (TV, net-internal vs exact):** onpolicy .024/.030 (decoder
  floor), both_tilted .027/.033, base_law .066/.076, random .176/.191. The
  POST net's world-model stays calibrated off-manifold (matching the
  pre-net result) — drift is far too small to explain the gap.
- **Reconstruction:** base->base .102 (vs .107 exact-belief, .126 exact-Q);
  random->random .118. NO improvement. Suspect 1 dead.
**By elimination, suspect 2 stands: on un-herded streams the net's emitted
tilt is not any belief-conditioned value tilt through the whitebox's form —
it is bare function-approximation extrapolation of the GRU outside its
training support. The decoded algorithm has a DOMAIN: the herded-state
cone plus goal-directed extrapolations (whitebox grade .02-.04); outside
it, ~45% of the tilt variance follows no algorithm we can name (within-set
bilinear ceiling .62-.75 there).** Final ledger of the off-manifold thread:
filter — algorithmic everywhere; court — algorithmic everywhere (1-dim
ontology, claim-both prior); value/tilt — algorithmic on the visited cone
only; beyond it, no algorithm to decode. Thread closed; queued next:
spectator-episode training variant (2-dim court + broadened state support,
both instruments ready), activation stitch, verified-circuit diagram, seeds.

## Iteration 23 (2026-09-03; approx.py + rnn_approx2) — the off-manifold approximation NAMED; Iter-22's 'no algorithm' retracted

Direct question first: off-manifold the net's per-arm output is a THIRD
thing — roughly equidistant from neutral and from the exact plan (base-law:
KL .42/.34 to neutral, .34/.51 to plan per arm).
Model ladder on the raw tilt (held-out): schedule .58-.76; recent-tokens
.74-.88; belief-bilinear .71-.89 (sharpening gamma>1: no gain — H-sharpen
rejected). The tilt is ~89% belief-bilinear on base-law streams.
**Direct reconstruction (P ∝ p̄·e^{fitted tilt}, NO gate layer, same-set
fit, held-out episodes): base_law .039, random .078, both_tilted .038 —
vs .10-.15 through the whitebox pipeline. And on-policy the direct
(court-free) model does WORSE than the whitebox (.071 vs .020): the court
carries real identity-dependent variance exactly on-manifold and
contributes nothing off it.** Cross-set transfer of the tilt-function is
poor (onpolicy→base .088, →random .165): the value surface deforms
off-support.
**The approximation, named:** off the herded-state cone the net runs the
SAME functional form — exponential belief-conditioned tilt on the neutral
law — with (a) gates resting at the claim-both prior (the 1-dim court
cancellation; effectively constant high gain), and (b) a value surface
that is the GRU's own smooth continuation: still bilinear-in-beliefs to
~.89 R², but a deformed extension of the on-policy value, not Q_opt and
not the on-policy Qhat. Iteration 22's 'bare extrapolation / no algorithm'
is retracted in favor of: same algorithm-form, prior-pinned gates,
off-calibration value surface. Residual truly unstructured piece: only
the random-stream remainder (same-set .078).

## Iteration 24 (2026-09-03; lamprobe_off.py) — internal identity state off-policy: the court is idle, as decomposed

- **Internal λ (frozen on-policy readout):** on-policy |λ_dec| median
  5.6→10.7 across rounds, tracking record-truth at r=.944. On spectator
  sets it HOVERS NEAR ZERO: 2.0-2.6 (base/random; ≈ the readout noise
  floor), 2.9-4.9 (both-tilted); the m̂-register coefficient likewise
  drops (1.9 vs 2.9 on-policy). The court is idle off-policy, exactly per
  the cancellation account — and the arms are both CLAIMED AT THE PRIOR
  (measured .80-.94), not unclaimed.
- **Gate coherence:** one pooled σ(aλ+c) curve across all four sets gives
  a=.22/.17, c=1.56/.87 — at λ≈0 it predicts claims .83/.70, bracketing
  the measured off-policy claims; pooled per-point R² is only ~.3 (readout
  + coefficient noise dominate at small λ), so coherence holds at the
  set-median level, not per-point.
- **Belief sufficiency:** on base-law streams, recent tokens add only
  +.024 R² beyond the beliefs (.893 → .917): the off-policy tilt depends
  on the trajectory essentially only THROUGH the belief state.
Asvin's decomposition confirmed end-to-end: λ/court computation behaves
correctly (idly) off-policy; the value/tilt is the sole weirdly-
generalizing module; the whole computation remains belief-state-
conditioned everywhere.

## Iteration 25 (2026-09-03; tfmodel.py + analyze_tf.py) — TRANSFORMER REPLICATION: same algorithm, hotter temperature

Capacity-matched causal transformer (2 layers, d=128, 4 heads, 406,284
params vs GRU 405k), identical featurizer/heads/protocol/step counts.
- **Q1 pretrain:** belief decode R² .981 (GRU .989); KL to exact filter:
  base .0038 / informedQ .0012 / random .0567 (GRU .0051/.0018/.0465) —
  same better-on-tilted phenomenon, same graceful bounded degradation;
  decoder no-refit transfer .965/.771 (GRU .985/.845).
- **Q2 midtrain:** SHARPER toggle: coef_u = .971/.041/.565 under flag
  A/B/none (GRU .919/.060/.495); flag-flip writes 92% of the exact tilt
  (slope .919, R² .935 vs GRU .79/.86); informed closed-loop occ .506 ≈
  floor .511; off-manifold jobs fine (base .012/.005).
- **Q3 posttrain:** occupancy .713 (GRU .683; optimal-Q agent .759). The
  SAME seven-constant whitebox fits at held-out KL **.0195** (GRU .0218),
  with β = 5.61 (~70% of trained ρ; GRU 3.87 ≈ 48%) — the transformer
  commits a hotter tilt, explaining its higher occupancy: the architecture
  difference expresses as TEMPERATURE/calibration, not algorithm. Claims:
  same 'everything is me until proven otherwise' (self 1.0→1.0; other
  1.0→.28→.18, faster withdrawal than GRU). λ decode R² .941 (GRU .91).
  Spectator idle test: claims rest at the claim-both prior (.73-.91 late)
  with decoded λ hovering small (2.6-3.5) on base/random/both-tilted —
  the SAME one-dimensional identity ontology and fail-open prior.
**Conclusion: the five-component algorithm (filter → template court →
1-dim register → biased gates → exponential value tilt) is determined by
the task and composite objective, not by the inductive bias — the
transformer converges to it with better calibration on every axis.**
