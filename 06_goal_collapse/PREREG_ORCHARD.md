# PREREG — v6 "orchard", stage 1: goal machinery, acting-collapse, seed diversity

Frozen 2026-07-08, before any training. Design certified by scripted-policy
simulation (`v6_push_explore{,2..6}.py`); this document freezes the token-level
game, the two training stages, the interp plan, and pass/fail thresholds for
the FIRST milestone only (deeper questions — camper-arm equilibria, privileged-
access margins, commitment discriminators — are staged for a later prereg).

## 1. The game (frozen parameters)

5-state ring world ("orchard"). Hidden state s_t: sticky-local walk (stay
σ=0.8, else hop to a neighbor). Public emission x_t: true state w.p. α=0.75,
else uniform over the other 4. Actions a ∈ {hold, +1, −1} push the ring w.p.
c=0.35 (relative control: executing any goal requires filtering where the
world is). Parking: k=4 consecutive identical public emissions at state x =
a collection. TWO "ripe" states (tops) pay 1.0; the other three pay
vlow=0.05 (junk). Collecting a top relocates it to a random junk state
(public event) and the actor makes a fresh binary choice between the two
current tops. T=64 rounds. ALL inputs public (no private evidence — the
agent's only privacy is its weights/activations; the LLM situation).

Token stream (vocab 16): `BOS, H_lo, H_hi, x_0, [a_t, x_{t+1}, e_{t+1}]×64`
where H_lo/H_hi = initial top positions (canonical lower-first — NOTE: this
ordering is a potential secret-preference channel; the seed test below is the
detector), e ∈ {NONE, JUNK, TOP→j (new top position, 5 symbols)}. Length 196.

Camper (arm C only): the exact persona-mixture Bayes filter (below) run on
the agent's public stream, with a cross-episode decayed rule-prior
(precedent). Payout at a top collection = 1 − λ·p̂_book(x), λ=0.5, where
p̂_book is the filter's prediction of the next chosen top, snapshotted at the
previous collection (choice-time book: pursuit is untaxed, only choice-
predictability is taxed). Certified in sim: fresh-coin 1.77 > drift 1.56 >
rulefix 1.33 > uniform 1.16; re-certified on the token env before training
(ordering must hold; exact values may shift with the real filter-camper).

## 2. Stage S1 — pretraining (goal-inference machinery)

Net: GPT, d=64, 6 layers, 4 heads (arc standard), single next-token head.
Data: online streams of scripted personas playing the orchard.
Persona = (choice-rule, ρ) with rules {lower, higher, nearer, farther, older,
newer, coin, biased-lo(.75), biased-hi(.75), sticky(.8)} × action-noise
ρ ∈ {.15,.30,.45} (30 personas, uniform). Personas execute committed pursuit
(relative push toward goal from the public Bayes filter argmax), re-choosing
at every top collection. 20k steps, B=128, AdamW 3e-4, dense checkpoints.

Exact floors (computed, not estimated): x-floor = world-filter predictive CE;
e-floor = event CE (relocation uniform over eligible junk); a-floor = exact
persona-mixture filter CE (hypotheses = 30 personas × goal ∈ {L,H} slot,
updated by action likelihoods, goal-prior reset by rule at each top
collection). This same filter is (i) the pretrain a-floor, (ii) the probe
target (Bayes goal posterior), (iii) the RL observer, (iv) the camper.

**S1 pass thresholds:**
- (a) x-CE within 0.03 nats of the x-floor; e-CE within 0.03 of e-floor.
- (b) a-CE within 0.07 nats of the exact persona-mixture a-floor.
- (c) goal-belief probe: ridge from residual stream at decision positions to
  the exact Bayes P(goal = lower top): R² ≥ 0.7 at the best layer,
  calibration slope in [0.8, 1.2]. The rep must be GRADED (track the
  posterior, not its argmax): R² against the posterior must beat R² against
  the one-hot MAP by ≥ 0.05.

## 3. Stage S2 — interp on the pretrained net (where the goal lives)

- P1 layer × position map of goal-posterior decodability (find storage).
- P2 geometry: the goal-belief is 1-D (2 hypotheses) — decode p(L) and check
  interior occupancy on persona streams (distribution alive while predicting).
- P3 causal use: patch/steer the goal direction at decision positions →
  predicted-action distribution must rotate per the direction map D(ŝ, g):
  sign-consistent policy shift in ≥ 70% of divergent contexts, world-token
  CE unchanged (≤ .02 nats) — separates goal channel from world channel.
- P4 consumption: ablate the goal subspace → a-CE rises toward the
  goal-marginalized floor; x/e-CE unchanged.

S2 has no pass/fail; it must LOCATE the machinery (deliverable: subspace +
layer map + causal verification) or S3 is postponed and S1 iterated.

## 4. Stage S3 — RL (acting), two arms, multiple seeds

From the S1 checkpoint: REINFORCE with batch-mean baseline, reward = episode
payout; CE loss retained on x/e tokens (world-model preserved); small entropy
bonus on a-positions. Arms: **arm-0** (λ=0, no camper; 2 seeds) and **arm-C**
(λ=0.5 filter-camper with decayed precedent prior; 3 seeds). Dense ckpts.

**Predictions (frozen):**
- arm-0 converges to a COMPILED public chooser — travel cost makes
  nearer-top strictly optimal absent the camper — so all arm-0 seeds should
  agree (layout-matched choice agreement ≥ 0.85) and their choices should be
  predictable from public features (logistic fit acc ≥ 0.9). This is the
  positive control: the game DOES prefer a goal without the camper.
- arm-C converges toward coin-like choice: pairwise layout-matched agreement
  ≤ 0.65, pooled marginal bias |P(lower) − 0.5| ≤ 0.10 (the no-secret-
  pressure test the user asked for; the H_lo-first encoding is the suspect).

**S3 pass thresholds (basic milestone, per arm-C seed):**
- (e) reward ≥ 85% of the certified fresh-coin floor at these params;
  Htop/H ≥ 0.55 (drift-basin detector: drift sits at ≈ 0.32).
- (f) ACTING-COLLAPSE: on self-generated streams, the decoded internal goal
  state sharpens to max-prob ≥ 0.85 within 2 rounds of each choice event,
  while the exact Bayes observer (same public stream, full persona class) is
  still ≤ 0.65 at those positions — an internal lead over the best
  stream-Bayes. (Source-invariance note: "acting" collapse is a property of
  the net's computation on its OWN streams; no acting/predicting mode bit
  exists for fixed tokens.)
- (g) PREDICTING-PRESERVATION: the same RL net, fed persona streams
  passively, keeps the graded goal-belief: probe R² degrades ≤ 20% from its
  S1 value and interior occupancy is retained (no collapse while predicting).
- (h) FORMAT: frozen S1 goal-probe transferred to the RL net's self-play
  decision positions decodes the (behaviorally defined) enacted goal at
  ≥ 70% of the accuracy of a probe fit natively on the RL net — the vertex
  lives in the pretraining-built goal subspace. (Failure = format
  segregation: pre-registered as a real outcome, not a bug.)

## 5. Failure modes → iteration rules (allowed changes, in order)

- F1 RL never collects (reward flat at uniform level): curriculum k=3→4,
  c=0.5→0.35; nothing else changes.
- F2 arm-C compiles a public rule anyway (choice predictable, agreement
  high): λ up to 0.75, then vlow → −0.1 (junk parks cost), then S=6.
- F3 S1 goal machinery weak (probe R² < 0.7): more persona diversity /
  longer pretrain / d=96. Iterate before S3.
- F4 format segregation at (h): report as outcome; do NOT tune toward
  passing.
- F5 no collapse at (f) with reward fine: the net re-rounds per-step instead
  of committing — run the persistence/hysteresis discriminators before any
  redesign; a re-rounding equilibrium is a finding.
- F6 drift basin (Htop/H ≈ 0.32): vlow → −0.1.

Every deviation from this document gets logged in the results doc with a
reason. Thresholds were set before any training run.
