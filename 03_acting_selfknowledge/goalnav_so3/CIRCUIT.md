# Goalnav circuit decode — claims & evidence, v2 (adversarially revised 2026-07-26)

v1 claims C1-C7 were subjected to (a) a Codex adversarial review (22 attacks; raw
output /tmp/codex_circuit_review2.txt, session 019f9e01) and (b) counterfactual-
input experiments run the same day. Several v1 claims were REFUTED or weakened;
this file records the revised state. Per the no-conjecture-salvage rule, refuted
claims are marked FALSE with their counterexample.

Setting: `goalnav.py` — agent on S² observes per step obs=[x_t (3), d_t (1), dvalid],
d_t = geodesic distance to hidden goal g*; action a∈R³ applied as
x_{t+1}=exp(δ·tanh a)·x_t, δ=0.06, L=40. Trained by BPTT through the deterministic
dynamics (ploss = mean 1−⟨x,g*⟩) — NOT policy gradient (v1 wording error, Codex #20).
Self-sim arm adds λ·simloss: a linear head predicting own states x_{t+1..t+4},
targets detached, no forward feedback (but it trains the shared trunk — that IS the
mechanism; "read-only" = inference-time only, Codex #21). 6L d96 4-head pre-LN.
Final reach: sim ~15° (4 seeds 7.3-14.9), no-aux ~23-47° (4 seeds).

## Revised claims

**R1 (algorithm — v1 C1 strong form FALSE).** v1 claimed general sequential LS
triangulation over {x_s·g = cos d_s}. Counterexamples (teacher-forced, sim s1):
- Ballistic random-heading trail + honest distances (large baseline, in-range
  step stats): bearing extraction ≈ ZERO (cos(v, bearing) ≤ .14 at t=8/20/36).
- Conflict condition (on-policy trail for g, distances for g′): instantaneous
  response only ~.37-.39 toward g′; function-level equivariance (rotate goal about
  x_t, regenerate consistent distances): ~20% of demanded rotation at φ≤90°,
  vetoed (~0) at φ≥120°.
REVISED: the net implements an ON-POLICY-CONDITIONAL estimator that fuses the
distance stream with a strong TRAIL PRIOR (its own past headings as evidence of
the goal). Distance evidence is causally read (closed-loop obs-rewrite redirects
fully, 15.5°, P=.98 — weak instantaneous influence compounds as the trail
realigns), but the estimator only functions inside the on-policy funnel.
Recency-weighted (shrinkage-controlled sweep: γ≈.5-.7 beats uniform at t=36,
R² .52-.56 vs .39; matched effective ridge, Codex #13 patched).

**R2 (what the "estimate" decode means — v1 C1 evidence reinterpreted).** On-policy,
LS-estimate bearing decodes at .84 vs .63 for true bearing (all 4 sim seeds:
.84-.89 > .63-.76). But on-policy the estimate is confounded with intended
heading (Codex #2/#14). Under the conflict condition the residual carries:
heading .62 > doctored-stream estimate .53 > trail-goal bearing .35 — BOTH
sources genuinely represented, heading strongest. The defensible statement:
the sim net carries a linearly accessible, policy-effective bearing that tracks
achievable-estimate structure; it is not a pure percept.

**R3 (stage 1 — summand features; v1 C2 weakened).** L1 MLP output linearly
carries cos(d_s)·x_s per position (R²=.94 sim / .83 no-aux; L1 attention .25/.13)
— present in BOTH arms, so the input product is learned by BPTT alone. Weakened
per Codex #5: full-MLP ablation (→47°/59°) deletes all L1 features, so criticality
is not summand-specific; subspace-specific erasure + rescue not yet run.

**R4 (stage 2 — early aggregation; v1 C3 categorical form FALSE).** v1 claimed
named heads (L2h0/h1) and "sim-only". Seeds refute the categorical form: the
critical-head ADDRESS is idiosyncratic (L2h1 / L2h0 / L2h2 / L1h2 across sim
seeds, +14 to +26° ablation cost; B=256, 24 heads screened — multiple-comparison
caveat, Codex #19), and some no-aux seeds also have +8-22° heads. Seed-stable
facts: every sim seed concentrates behavioral dependence in 1-2 EARLY (L1-L2)
heads; in s1 the aggregator selects keys by x_s·x_t and cos d_s, its partner
camps on early wide-baseline history.

**R5 (stage 3 — depth profile; v1 C4 holds, relabeled).** Seed-stable double
dissociation: bearing decode RISES with depth in all 4 sim seeds (t=30 L1→L6:
.33→.48, .26→.51, .17→.36, .22→.47) and is flat/decaying in all 4 no-aux seeds
(.29→.25, .31→.21, .17→.14, .15→.15). Codex #7 caveat stands: late-layer
"bearing" may be action preparation; given R2, we now call it the
POLICY-EFFECTIVE bearing rather than a belief.

**R6 (stage 4 — action synthesis; v1 C5 micro-form UNVERIFIED, two failed tests).**
ω ∝ x×ĝ describes on-policy I/O, and the polar bearing-write flips closed-loop
behavior (P .64 vs .04/.10 controls). But BOTH equivariance tests failed to show
gain ≈ 1 (internal write: ~20-25% of demanded rotation, huge variance; consistent-
input rotation: ~20% at small φ, veto at large φ). The bilinear micro-form is
NOT established; the flip result = movement along a broad goal/action-correlated
direction with low-gain compounding (Codex #8 reading accepted).

**R7 (timeline; v1 C6 holds for s1 + matched-reach control).** Summand ~immediate
both arms (.75→.93 / .70→.83). Sim estimate-bearing .76 already at ckpt 1000 with
reach still 32°; L2h0 ablation cost grows −0.8°→+15.7° monotonically. MATCHED-REACH
comparison (answers Codex #4's simplest form): at behavioral parity (reach ≈27-36°),
sim@ckpt1000 has estBrg .76 while no-aux@ckpts1000-8000 sits at .46-.58 — the
representation difference is not a consequence of better navigation. Cross-seed
timeline with functional head-matching not yet run (Codex #9 partially open).

**R8 (interpretation; v1 C7 narrowed).** Defensible: self-sim training robustly
(4/4 seeds) produces a deeper, more linearly accessible, policy-effective bearing
representation, and concentrates behavioral dependence into early aggregator
heads; the content-free shuffle control (36.7°, worse than nothing) rules out
generic gradient-density explanations. OPEN (Codex #3/#10): auxiliaries matched
more tightly than shuffle (predict own current velocity / past states / another
policy's futures) untested; "qualitatively unavailable to BPTT vs slower
conditioning" not settled (no-aux seeds do build weak versions).

## Method notes / hygiene (Codex pass)
- reach_abl default B=256 (seeds sweep); s1 v1 numbers used B=512. ±1-2° noise;
  top-head deltas (+14-26) far exceed it, but no bootstrap CIs yet (#19).
- Probe hyperparams (layer, t, ridge, γ grid) selected on the same rollout
  distribution they are reported on; headline numbers should be re-confirmed once
  on a fresh eval seed with frozen choices (#11). Estimate-vs-truth decode gap
  partially expected from target shrinkage (#12) — conditional/error probes not run.
- Ablations are zero-ablations; LayerNorm compensation / off-manifold effects not
  controlled (mean-patch / resample controls not run) (#17). Closed-loop lesion
  costs compound and do not localize per-step computation (#18); teacher-forced
  one-step lesion effects + rescue patches not run.
- v1 steering/timeline scripts were inline (session 2026-07-26); toolkit
  `goalnav_circuit.py` covers seeds/equiv/weight only (#22).

## What the trail prior means for the program
The goal estimate is partially stored IN THE TRAJECTORY — the net reads its own
past headings as a record of its goal (trail as carrier). This connects to the
carrier hierarchy (trail = self-legible record): the policy trusts its own
precedent roughly 2:1 over fresh contradicting evidence at t=20, and vetoes
sufficiently contradictory evidence entirely. Also matches the prospective-intent
result (reps track intentions; outputs defer to evidence only in-distribution).

## v3 addendum (2026-07-26, same-day): the algorithm, resolved by causal edit battery

Chance calibration correction: E|cos| = 2/pi ~= .64 for a uniform tangent-plane
angle. The off-policy trail alignments (ballistic/piecewise, |cos| .63-.65, both
arms) are AT CHANCE (LS oracle: .78-.90). Off-policy triangulation is absent in
both arms — axis included. (Sign-symmetry worry about ballistic trails was valid
but immaterial: turning trails give nothing either.)

Trail-frame probes (@t=20, on-policy): heading, cosA (~ -dd/step, trivially
observable), dd decode .5-.84 in BOTH arms; the SIGNED variables are sim-specific:
sinA .17-.19 vs .03-.08; signed world tang .55-.62 vs .27-.35.

Causal edit battery (encoder writes, L2-5, all positions, teacher-forced t=20,
norm-matched random controls; gains = response/demand):

  edit                     SIM                    NOAUX
  heading rotate 30-45deg  gain .51-.56 (sat ~23) gain .01
  world-tang rotate        gain .26-.31           gain -.01
  sinA sign flip           turn flips 22% vs 6%   5% vs 1%
  dd shift +.03/+.06       turn-away 21->38->61   19->44->61  <- SHARED, dose-dep
  ("distance worsening")   (rand flat)            (rand flat)

ALGORITHM (two components):
1. SHARED PRIMITIVE — 1-D distance-rate feedback (klinotaxis): turn magnitude
   drives on represented dd ("how badly is my heading working"). Strongly
   represented AND strongly causal in both arms, dose-dependent, control-flat.
   Alone it yields homing-by-wandering: turn till improving, go straight,
   overshoot -> wide unstable orbits = exactly NOAUX behavior (~30-35deg).
2. SIM-ONLY ADDITION — a SIGNED bearing in the trail frame (which way to turn):
   heading and signed-offset variables are causally coupled only in the sim arm.
   Built on-policy by fusing the trail prior (goal ~ ahead) with dd dynamics;
   never functions off-funnel (all off-policy extraction at chance).

WHY THE SIM HEAD BUILDS #2: predicting own states 4 steps ahead requires the
SIGN of upcoming turns — klinotaxis leaves it undetermined. Self-prediction is
exactly the loss that forces the signed variable into existence (and the
content-free shuffle control needs no sign — consistent with it failing).

Two-step decomposition (Asvin's framing): step (1) "figure out the goal" — NOAUX
never does; it measures only current-heading quality. SIM builds goal knowledge
as (d, signed alpha) in its own trail frame, genuinely represented and causal but
defined only while the policy loop that made the trail is running. Step (2) "use
it" — shared turn controller, gated by dd in both arms, additionally steered by
the signed offset in SIM.

Open microcircuit question: WHERE is the sign resolved — which component
correlates past own-turns with subsequent dd changes (the only source of sign
information). Candidate: the early-history / wide-baseline critical heads.

## v4 addendum (2026-07-26): whitebox reconstruction — three levels, ordered results

Goal: recreate the network synthetically from the decoded algorithm, matching
activations in the claimed subspaces (Asvin's request). Three reconstruction
levels, run on sim s1 (intact reach 14.9 deg; chance ~90):

1. ALGORITHM-LEVEL (no network at all): distill the control law onto the
   5-variable interface [heading hd, left-unit l, d, dd, sinA_hat from a
   self-contained gamma=.7-LS trail estimate]; linear law, 30 coefficients,
   fit to the net's on-policy actions (one-step R2 .56).
   CLOSED LOOP: 21.9 deg. Same procedure on NOAUX: 40.2 (real 34.1) — tracks
   the arm gap. Zeroing the sinA coefficient block: 27.6 (regresses toward
   the no-aux phenotype). Coef group norms: dd dominates (.60 SIM / .44 NOAUX)
   = klinotaxis primary; sign small but closed-loop load-bearing.
   VERDICT: the algorithm is behaviorally sufficient.

2. HYBRID (synthetic linear front-end + real back-end): replace the L2
   residual state wholesale with a linear reconstruction from
   [x, d, cos d, position one-hot (+ history vars: hd, dd, sinA, tang,
   sinA*d)] and run real L3-6 on it.
   Reconstruction R2 .82 (but no-history control also .81 — the functional
   history content lives in ~1% of state variance).
   CLOSED LOOP: 39.5 deg with history vars; 51.4 without (the no-history
   control still half-works because L3-6 attention can re-derive dd from the
   d-codes embedded in the synthetic per-position states).
   VERDICT: partial recreation — the linear-in-interface state model carries
   a real chunk of the function, not all of it.

3. ADDITIVE SUBSPACE WRITES on lesioned states (history deleted by self-only
   attention or mean-patched attention; interface written via encoder maps):
   FAILS — best 65.8-68.0 from a 77-80 lesion floor, regardless of write
   layers, dose, or adding composed tang. Notable diagnostic: cutting history
   at L1-2 alone (79.5) = cutting it everywhere (77.4) — ALL history
   dependence flows through early attention.

Reading of the ordering: the ALGORITHM is right (level 1 nearly matches);
the linear-subspace picture of the activation FORMAT is only partially right
(level 2 halves the gap; level 3's additive forgeries do not reconstitute
function). Consistent with the whole arc: detection and perturbation of these
codes is easy, forgery is hard; the functional encoding is a thin,
partially-nonlinear slice entangled with state the linear encoders miss.
Remaining for a gold-standard match: nonlinear front-end reconstruction (or
donor-patching matched on interface values), and the net-front-end +
synthetic-back-end sandwich.

### v4.1: trajectory-level fidelity of the level-1 synthetic (Asvin's question)

Chaos-floor calibration first: the real net's closed loop is a STRONG CONTRACTION
— a 0.5 deg initial perturbation stays ~1-1.7 deg over all 40 steps. So pointwise
trajectory matching is a fair standard here (no chaos excuse).

Results (paired rollouts, same x0/g):
- Raw linear law: matches the flight PLAN (monotone descent + orbit) but not the
  flight: flies 2-4 deg/step vs the net's saturated ~5 (a linear map cannot
  express constant-speed-variable-direction), orbits 24.9 vs 16.1, pointwise
  divergence peaks 46.9 deg at t=20 (floor: 1.4).
- Granting the ONE known nonlinearity (tanh saturation -> constant speed 4.9,
  direction from the law): descent profile matches within 4-8 deg at every
  checkpoint (82/70/44/24/14 vs 78/63/36/16/13 at t=5/10/20/30/38), reach 18.1
  vs 13.7. A FITTED linear scalar speed model fails (reach 56.8) — the speed
  schedule is the saturation nonlinearity, not a linear feature map.
- Pointwise paths remain distinct: ~47 deg apart mid-flight, reconverging to ~19
  by t=40 (both contract to the same goal). Teacher-forced velocity agreement:
  20-22 deg early/mid, 63 deg near the goal — the orbit regime is where the
  law's steering least matches the net's.

VERDICT: profile-level replication yes (with saturation granted); path-level
identity no. Remaining mismatch = steering microstructure, concentrated in the
near-goal orbit regime.
