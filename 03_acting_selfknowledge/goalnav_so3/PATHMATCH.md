# Path-matching the goalnav policy with a synthetic algorithm — round log (2026-07-26)

Goal (Asvin): iterate the synthetic law until it recovers the NET'S PATH pointwise
(standard: paired closed-loop divergence vs the ~1.4deg chaos floor — the net is a
strong contraction, so path identity is achievable in principle), then probe the
winning algorithm's variables inside the net. Net: gn_rep_lam1.0_s1 (sim arm,
reach 13.7deg on the paired eval; chance 90).

## Round table (paired closed loop, same x0/goal; div = angle(real, synth) at t)

| law | teacher-forced angle t2-8/8-20/20-30/30-40 | div t10/t20/t30 | reach |
|---|---|---|---|
| v0 interface law + const speed (v4.1) | 20/22/35/63 | 37/47/31 | 18.1 |
| r1 + RBF(d) gains, s(d) lookup, fitted warmup | (36 pooled) | 37/53/32 | 20.0 |
| r2 + turn hysteresis + dual-gamma | 21/23/34/64 | - | - |
| r3 LMS-filter + pursuit, end-to-end SGD | 19/23/41/95 | 31/67/79 | 83 (diverged) |
| r5 lag-5 local law (MLP, 22f) | 18/24/36/44 | 16/30/30 | 22.9 |
| r6 lag-12 local law (MLP, 43f) | 18/22/32/31 | 16/31/31 | 27.4 |
| r7 DAgger on lag-12 (3 iters) | - | worse each iter | 34 -> 61 |

## Discoveries that reshape the algorithm story

1. **The net orbits WITHOUT an accurate bearing.** Best targeted linear probe for
   the unit bearing at orbit positions (t26-38, d<20): 51.8deg (L6), vs 4.5-9deg
   achievable by uniform-LS on the same data. The near-goal bearing simply is not
   (linearly) present. Pursuit-of-estimate is the WRONG model of the orbit.
2. **Orbit steering is fully state-determined and finite-window.** From the full
   L5/L6 residual an MLP predicts orbit velocity at 9.4/7.1deg (held out) — the
   information exists per position. Compact summaries plateau: bearing+kinematics
   40deg; lag-window scalars improve monotonically with window size (order 2: 63,
   order 5: 44, order 12 ~ orbit period: 31deg). Supports: the orbit is the LIMIT
   CYCLE OF THE KLINOTAXIS LOOP (hunting cycle on the recent dd sequence), whose
   phase lives in ~an orbit-period of lagged kinematics.
3. **gamma=0.7 recency was the synthetic's own orbit bug** (43-57deg bearing error
   near goal vs 4.5-9 for uniform memory) — but fixing it did NOT fix the orbit
   prediction (MLP with uniform-LS bearing still ~62deg): consistent with (1).
4. **BC wall + DAgger failure.** Teacher-forced gains stopped transferring to
   closed loop (compounding drift); naive DAgger degraded monotonically
   (off-manifold recovery interference). The imitation-learning fix, not more
   features, is the current bottleneck.

## State / artifacts
- /tmp/laglaw.pt (lag-5), /tmp/laglaw12.pt (lag-12), /tmp/laglaw_dagger.pt,
  /tmp/lms_pursuit.pt, /tmp/mlp13.pt. Feature builders inline in session
  2026-07-26 (lag_features: hd, l, d, dd lags 0..NL-1, relative-heading lags
  1..NL-1 as local cos/sin scalars, slow sinA/cosA from gamma-.7 LS).

## Next moves (concrete)
1. **DART instead of DAgger**: roll the NET with small action noise (tube around
   the expert manifold), label with the net's clean actions, train the law there.
   Avoids learner-drift interference; standard fix for exactly this failure.
2. Sequence-model law (tiny GRU/TCN over the scalar streams) with noise-injected
   training; distill to interpretable after it path-matches.
3. Then the probe-back program (variables are already named): decode lagged dd
   and relative-heading scalars from the residual by position (does the net cache
   a dd-buffer? which heads transport lag-k signals — the recency heads L1h2 etc.);
   causal lag-specific edits; and the orbit-cycle phase as a decodable variable.

## Probe-back round + Codex fresh-eyes (2026-07-27)

PROBE-BACK RESULTS (sim net unless noted):
- **dd LAG BUFFER IS REAL AND CACHED**: resid decodes dd_(t-k) for k=0..11 at
  R2 .66-.85 (t=20) / .35-.83 (t=32), BOTH arms; excess over a
  current-obs reconstruction baseline +.33-.72 at every lag => genuinely
  fetched history, not autocorrelation.
- **Transport is redundant**: no single head ablation drops the buffer decode
  by more than 0.06 (the max is L2h0 — the behaviorally critical head).
- **Buffer structure ~ literal multi-slot**: SVD of the 12 lag-decoder
  directions: participation ratio 9.1, 7/12 dims for 90% energy — near-
  orthogonal slots, not a 3-4-dim compressed filter bank (Codex #15 test).
- **Event clock exists** (Codex #7): steps-since-closest-approach decodes at
  .46-.63 from resid L6 vs .18-.29 from the dd-lag summary — an internal
  phase-like variable beyond raw lags.
- **Future-dd content**: resid predicts dd_(t+1) at .49 vs .26 from the full
  12-lag summary => internal cycle-phase information beyond the buffer.
- **Sim head is NOT the step-level phase readout**: its 7-8deg position
  accuracy exceeds the 5deg step size, so its implied next-step velocity is
  ~45deg off realized at ALL phases. It proves coarse trajectory knowledge only.

CODEX DIVERGENCE DECOMPOSITION (#37, applied to the lag-5 law):
pointwise 29.6deg = CURVE error 16.1deg + PHASE SLIP ~7 steps (most of an
orbit period). Half clock problem, half steering problem.

CODEX IDEAS BANK (raw: /tmp/codex_ideas.txt): highest-value untried:
#26 phase reset at periapsis (direct fix for the slip half); #22 fit the
phase-response curve, not next actions; #2 extremum-seeking test (turn ~
sum of deltaU_(t-k)*dd_(t-k) — the dither-correlate mechanism for sign);
#21 frequency-response sysID (delay line vs integrator vs oscillator have
distinct signatures); #23 multiple-shooting fit; #29 frozen-base residual
boosting instead of DAgger; #17/#19 value-only + key-only interventions to
separate lag SELECTION from lag CONTENT; #31/#32 CAVEAT: our dd-edits create
physically impossible histories — purity check pending.

NEXT LAW (round 8 plan): lag-law + explicit phase oscillator with periapsis
reset (event clock is now a MEASURED internal variable), fitted via
multiple-shooting; then value-only/key-only interventions to pin which
components write the clock.
