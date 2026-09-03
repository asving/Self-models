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
