# AUXSWEEP — self-referential control auxiliaries + prediction-horizon sweep

Design doc, written before launch (2026-07-27). Tests the mechanism claim from
the circuit arc: **the sim loss accelerates goalnav training specifically by
manufacturing the TURN-SIGN / signed-bearing variable, whose policy-gradient is
symmetry-blocked; self-prediction of the FUTURE breaks the left/right symmetry
because the realized future says which way the trajectory bends.**

## Generative process / training (unchanged from goalnav.py)
S² navigation, obs=[x_t, d_t, dvalid], x_{t+1}=exp(0.06·tanh a)·x_t, L=40,
6L d96 4-head, batch 96, lr 3e-4, AdamW, 8000 steps, ploss = mean(1−⟨x,g*⟩)
by BPTT through the dynamics, plus λ=1 × auxiliary on a read-only linear head
(targets detached, trains the shared trunk). Seed 1 for all new runs.
Reference points (existing, seed 1): aux=self r=4 → reach 14.9; no-aux → 34.1;
shuffle (content-free) → 36.7.

## New runs (5)
| run | aux target | r | out |
|---|---|---|---|
| past   | own PAST states x_{t-1..t-4} (input-copyable; self-referential, NO future/turn-sign content) | 4 | goalnav_runs/gn_aux_past_s1 |
| vel    | current arriving velocity dir x_t−x_{t-1} (kinematic self-target, input-derivable) | 1 | goalnav_runs/gn_aux_vel_s1 |
| r=1    | own future x_{t+1} | 1 | goalnav_runs/gn_r1_s1 |
| r=2    | own future x_{t+1..t+2} | 2 | goalnav_runs/gn_r2_s1 |
| r=8    | own future x_{t+1..t+8} | 8 | goalnav_runs/gn_r8_s1 |

## Preregistered predictions & falsifiers
P1. **past and vel give little/no acceleration**: final reach ≈ no-aux band
    (≳25°), NOT the sim band (~13-15°). Falsifier: past or vel reaching ≤18°
    would break the "counterfactual future content is the active ingredient"
    claim — mere self-referential kinematic supervision would suffice.
P2. **r=1 recovers most of the effect** (turn-sign content is fully present in
    x_{t+1}) — reach well below no-aux, plausibly 15-20°.
P3. **Graded horizon effect via SNR**: the sign signal in the target scales
    with horizon (1-step bend deflects x_{t+1} by ~δ·sinα vs 4-step bend ~4x);
    so expect monotone improvement r=1→r=4 with saturation; r=8 ≈ r=4 (or
    slightly worse: fewer supervised positions, longer-horizon targets harder).
    Falsifier for the SNR story: r=1 ≈ r=4 exactly (sign content binary, SNR
    irrelevant) — acceptable, sharpens the mechanism; r=1 ≈ no-aux would
    falsify P2 and demand revision (e.g., multi-step rollout content needed).
P4. **Probe signature**: signed variables (sinA decode, estimate-bearing at
    t=8) strong in r≥1 future runs, weak (≈ no-aux levels: sinA ≲.08,
    estBrg ≲.6) in past/vel runs. This is the mechanism-level check, not just
    behavior.

## Analysis plan
Reach curves from json logs; final-ckpt probe battery (sinA/estBrg/tang
decodes, per goalnav_circuit.py conventions); if P1 holds, no further controls;
if violated, rerun with second seed before concluding.

## Files
Runs in tmux session `goalnav_aux` (GPU 6: past→vel→r8; GPU 7: r1→r2),
logs in logs/aux_*.log. Code: goalnav.py --aux {self,shuffle,past,vel}.

## RESULTS (2026-07-27, same day; seeds 2-3 of past-aux launched as confirmation)

| run | reach@8000 | sinA@t20 | estBrg@t8 | tangW@t20 | brg t30 L1->L6 |
|---|---|---|---|---|---|
| self r4 (orig) | 14.9 | .19 | .83 | .62 | .28 -> .51 |
| **PAST r4**    | **11.1** | .19 | .82 | .62 | .18 -> .50 |
| **VEL**        | **12.4** | .24 | .87 | .63 | .33 -> .54 |
| self r1        | 11.5 | .22 | .91 | .60 | .26 -> .41 |
| self r2        | 13.2 | .25 | .83 | .62 | .34 -> .50 |
| self r8        | 14.0 | .25 | .86 | .62 | .23 -> .51 |
| no-aux         | 32.6 | .08 | .59 | .35 | .28 -> .24 |

**P1 REFUTED** (counterexample: gn_aux_past_s1 reach 11.1 << falsifier 18; vel
12.4). Past-states prediction — input-copyable, zero future content — accelerates
AS MUCH AS future prediction (and converges fastest: 17.9 by step 4000).
**P2 confirmed** (r=1: 11.5). **P3 REFUTED** (horizon flat within seed noise).
**P4 -> scenario (a)**: past/vel nets develop the FULL signed-bearing system;
the signed variables come with ANY learnable own-trajectory auxiliary, not with
future-ness.

**REVISED MECHANISM**: the auxiliary's job is TRAIL LEGIBILITY. Any dense,
learnable target anchored to the net's own trajectory forces the trunk to
maintain a linearly-readable kinematic record (the past head's 7.5deg error
shows even copying is non-trivial — the record has real capacity cost). On that
substrate, the policy's own weak, symmetry-blocked gradient suffices to
assemble the signed bearing as a short-path READOUT — it could never build the
substrate itself (no-aux), but it can select from one. The shuffle control
slots in: another episode's future is UNLEARNABLE -> gradient noise -> worse
than nothing. Active ingredient = learnable own-trajectory content; NOT
prediction-of-the-future, NOT generic density.

SCOPE NOTE: in this game the observations ARE the own trajectory, so
self-specific vs any-learnable-record cannot be dissociated here. Designed
next control: append an exogenous observable stream (e.g., a random-walk
token) and an auxiliary predicting IT — if that also accelerates, the
mechanism is generic record-building; if not, self-anchored content matters.

## ROUND 2 (2026-07-27): greedy credit + exogenous-stream controls (prereg before launch)

Confirmation seeds first: past-aux s2 11.1, s3 9.9 — the P1 refutation is
3-seed solid (11.1/11.1/9.9 vs no-aux band 21.5-42.2).

New runs (seed 1, 8000 steps):
| run | change | tests |
|---|---|---|
| gn_greedy_s1 | HORIZON-1 credit (state+obs detached between steps; per-step gradient = pure regression onto signed oracle x cross g), NO aux | was DILUTION the blocker? |
| gn_exo_noaux_s1 | +exo channel y (drift .075 + noise .03 per step), no aux | re-anchor no-aux with distractor |
| gn_exo_self_s1 | +exo channel, aux = own future r4 | re-anchor sim with distractor |
| gn_exo_future_s1 | +exo channel, aux = y future r4 (learnable: must infer episode drift — estimation task matched in flavor to triangulation, ZERO consequence) | self-legibility vs generic-supervision |
| gn_exo_past_s1 | +exo channel, aux = y past r4 (record-keeping matched to 'past', wrong stream) | same, record flavor |

Preregistered predictions:
P5 (greedy): converges FAST (sim-band or better, possibly earliest), and builds
   the signed system WITHOUT any auxiliary (sinA >= .15, estBrg >= .8) — the
   undiluted oracle-signed gradient supervises the sign directly.
   Falsifier: greedy ~ no-aux => dilution was not the blocker; feature-growth
   difficulty dominates and the horizon-1 signal is insufficient too.
P6 (exo, self-legibility hypothesis): exo_future and exo_past land in the
   (exo-)no-aux band with weak signed probes; exo_self ~ sim band.
   Falsifier: exo arms accelerating like self-aux => the mechanism is generic
   learnable-record supervision, NOT self-anchored; the goalnav result then
   leaves the self-models story. Middle outcome (partial acceleration, no
   signed system) = split into generic + self-specific parts; probes arbitrate.

## ROUND 2 RESULTS (2026-07-27)

| run | reach@8000 | sinA | estBrg | tangW | brg t30 L1->L6 |
|---|---|---|---|---|---|
| greedy (horizon-1, no aux) | 49.1 (still descending) | .07 | .64 | .34 | .24 -> .23 flat |
| exo_noaux | 36.9 | .16 | .33 | .38 | .19 -> .19 flat |
| exo_self  | 11.9 | .37 | .85 | .65 | .28 -> .53 RISE |
| exo_future | 21.0 | .10 | .61 | .38 | .19 -> .13 DECAY |
| exo_past   | 27.5 | .14 | .66 | .31 | .15 -> .12 DECAY |

**P5 (greedy) unresolved at matched budget — magnitude confound**: horizon-1
gives each action 1 loss term instead of ~40, so aggregate gradient is ~40x
smaller; greedy@8000 ~ full-BPTT@~400 (52.6), curve still descending, probes
early-training-like. Scale-corrected rerun launched (--pscale 40,
gn_greedy40_s1).

**P6: strong self-legibility form REFUTED; the middle outcome, with a clean
mechanistic split.** Predicting the zero-consequence drifting star DOES help
behavior (exo_future 21.0, exo_past 27.5 vs exo_noaux 36.9) — but builds NO
signed system (sinA .10-.14, tangW .31-.38, depth-DECAY, all ~no-aux levels).
Only the self-anchored aux builds the signed bearing (exo_self .37/.85/.65,
depth-rise). And exo_FUTURE > exo_past: the generic transferable part looks
like latent-ESTIMATION infrastructure (inferring the drift axis ~ analog of
triangulation), not record-keeping per se.

REVISED MECHANISM (v3, two components):
1. GENERIC learnable-supervision benefit (~9-16 deg of reach), available even
   from a zero-consequence stream, larger when the aux task exercises
   latent-inference machinery; does NOT build the signed system. (The
   optimization-shaping intuition lives here — though instantaneous gradient
   cosine does not carry it: all aux gradients incl. shuffle are ~orthogonal
   to the policy gradient; see gradient-geometry table in session notes.)
2. SELF-ANCHORED record benefit (the remaining ~9 deg + stability): only
   own-trajectory targets build the SIGNED bearing system — this part is
   genuinely self-specific and stays in the self-models story.
Gradient-geometry side-note (Matt Smith thread): policy-gradient batch SNR is
catastrophic early (cross-batch cosine -0.54 at ckpt 1000, low-rank
sign-flipping noise), greedy component = coherent core of early full gradient
(cos 0.81); ALL aux gradients ~orthogonal to policy gradient (|cos|<=.08,
shuffle indistinguishable) => Du-et-al cosine gating would fail here; the
aux effect is second-order (trajectory-level), not first-order alignment.

## ROUND 2b (2026-07-27): greedy resolved, exo destabilized by seeds

- gn_greedy40_s1 (pscale=40): reach 45.0 ~ unscaled greedy 49.1. The magnitude
  confound is retracted (Adam is ~scale-invariant anyway): HORIZON-1 CREDIT
  GENUINELY FAILS, n=2 (45-49), and is WORSE than full-BPTT no-aux (34) =>
  dilution was NOT the blocker; the long-horizon + observation-route credit is
  load-bearing (consistent w/ cos(greedy, full)=.81 early but .21 late). The
  undiluted signed-oracle regression alone cannot build the estimator.
- exo second seeds: exo_future_s2 51.6 (!), exo_past_s2 42.1 (!) vs s1
  21.0/27.5. Exo-arm spread (21-52) exceeds the no-aux band (21.5-42.2);
  with n=2 the exo mean ~ no-aux mean. The round-2 'generic ~9-16deg benefit'
  claim is SUSPENDED pending n=3 (exo_future_s3, exo_past_s3, exo_noaux_s2/s3
  launched). What holds regardless: self-anchored aux is reliable and tight
  (7.3-15.1 across 9 runs: self r1/r2/r4/r8, past x3 seeds, vel; alwayson);
  exo aux NEVER builds the signed system (probes, s1); no-aux never does.

## FINAL VERDICT (2026-07-27, n=3 exo seeds)

exo_noaux: 36.9/29.2/38.9 (35.0 +- 4.2) | exo_past: 27.5/42.1/38.3 (36.0 +- 6.2)
exo_future: 21.0/51.6/19.1 (30.6 +- 14.9, bimodal) | exo_self 11.9;
self-anchored family (no exo channel): 7.3-15.1 across 9 runs.

- exo_past == baseline exactly: predicting the star's past buys NOTHING.
- exo_future: no reliable benefit (two seeds ~19-21, one 51.6 — helps-or-hurts).
- Probes on ALL exo runs incl. the good seed (19.1): NO depth-rising signed
  system (sinA .10-.24, depth decay) — even behaviorally-good exo seeds
  navigate without the self-aux signature.
- P6 self-legibility: VINDICATED at the reliability level. The round-2
  'generic benefit' was seed luck.

THE AUXSWEEP BOTTOM LINE (three rounds, 19 training runs):
1. Prediction DIRECTION irrelevant (past = future = velocity; horizon flat).
2. Target REFERENT decisive: only the net's OWN trajectory — the self-caused,
   maximal-consequence stream — reliably accelerates (7-15 vs 29-52) and it
   ALWAYS builds the depth-rising signed-bearing system; zero-consequence
   targets never build it and don't reliably help.
3. Credit structure: full BPTT required; horizon-1 fails even undiluted
   (greedy 45-49 < no-aux 34; Adam scale-invariance retires the magnitude excuse).
4. Mechanism: self-anchored dense supervision -> legible self-record in trunk
   -> policy assembles signed bearing as readout -> policy gradients become
   coherent (batch-SNR -0.54 -> +0.57) -> fast stable convergence.
CONSEQUENCE is the operative property of the target stream. 'Predict the
world' helps reliably only when the world-part you predict is yourself.

## MATT-HYPOTHESIS DEEP TEST (2026-07-28; gradient stats + twophase.py)

1. FIRST-ORDER FORM (Du et al. cosine alignment) REJECTED WITH STATISTICS:
   cos of MEAN gradients (12+12 disjoint batches, 8 ckpts, both trajectories):
   all aux in [-.09, +.08], sign-inconsistent; SHUFFLE has the most positive
   early cosines (+.04..+.06) -> cosine gating would select the harmful aux;
   on the past-aux's own trajectory past~policy is mildly NEGATIVE; per-block:
   no hidden alignment, early OPPOSITION on in_proj (cos -.30 @ckpt1000).
2. VARIANCE PREMISE CONFIRMED 30-90x: policy-gradient SNR (||mean||^2/var)
   collapses to .04-.06 mid-training (at ckpt 4000 two independent 12-batch
   means are ORTHOGONAL - no stable descent direction); past-aux SNR 1.3-3.8.
   (SNR estimator has +1/n bias and is heavy-tail unstable at small n - the
   twophase nb=8 readings are inflated; comparisons within-script only.)
3. SEQUENTIAL SECOND-ORDER FORM FAILS (twophase.py, from noaux ckpt1000):
   1000 aux-only steps solve the aux task (sloss .0005) while DESTROYING the
   policy (reach 35.7 -> 89.5 = chance); subsequent 1000 policy-only steps end
   at 24.5 vs 26.9 control vs 27.6 shuffle-pretreated - no meaningful
   acceleration, no SNR gain at handoff. DESIGN CAVEAT that is itself the
   finding: aux-only training lets the policy collapse, so the record gets
   built on degenerate trajectories and the action readout de-calibrates -
   a sequential substrate-then-readout decomposition is NOT WELL-POSED here.
VERDICT: the auxiliary benefit is a property of the COUPLED dynamics
(interleaved co-training: policy loss continuously re-calibrates readouts on
a trunk kept legible by the aux, data stays on-policy) - it decomposes neither
into first-order alignment (Matt) nor into substrate-then-readout (our v2).
Proposed quantifier: lambda-alternation experiment (alternate aux-only/policy-
only every k steps, k in {1,10,100,500}) to measure the ratchet timescale.
