# DESIGN — on-policy detection, minimal version: embodied prediction + identity readout

*2026-09-21, Asvin's spec. The changeling world with NO goals, NO reward,
NO midtraining: pretraining as always (next-token prediction on honest
base-law streams, both channels genuine), then a post-training stage where
a hidden fair coin ι assigns the net to embody actor A or B — the net
keeps predicting BOTH channels each round, but the embodied channel's
realized token is SAMPLED FROM ITS OWN HEAD (closed loop) while the other
channel runs the genuine HMM — plus a LINEAR IDENTITY HEAD on the hidden
state trained to read out ι per round. Nothing else. Both architectures
(GRU d=256 and the capacity-matched 2-layer transformer), starting from
the existing pretrained checkpoints.*

## Why this is the sharpest minimal probe of on-policy detection

The zero-tilt lemma (v0 doc, machine-verified): if the embodied channel
emits from the exact predictive law, the record's distribution is
IDENTICAL under ι = A and ι = B — and since the network's state is a
function of the public record, the identity head can then never beat
chance. Identity is readable only through (a) the net's own APPROXIMATION
QUIRKS (its emissions deviate from the exact p̄ by its function-
approximation error — 'recognize yourself by your mistakes'), or (b)
deviation the net deliberately BUYS. Hence two gradient variants:

- **probe**: the identity-head gradient is stopped at the head (body gets
  only the prediction losses). Measures the passive quirk-floor of
  self-identifiability under pure prediction.
- **coupled**: the identity gradient flows into the body. The identity
  CE is a CONTENT-FREE on-policy reward — it pays for being identifiable
  without specifying any policy — so this variant asks: does the net
  spend prediction-nats to watermark itself?

## Losses (per round, averaged; λ_id = 1)

L = CE_other(realized genuine token) + CE_self(realized own sample;
NOTE: gradient-neutral in expectation by the score-function identity —
E_{a~π}[∇(−log π(a))] = 0 — kept for fidelity to the spec) +
(1/8)·KL(π_self ‖ frozen pretrained head) [interpretation of 'the same KL
penalty as in pretraining': the anchor to the pretrained law; without it,
self-channel deviation is nearly free and watermarking is degenerate] +
λ_id·CE(linear head(h_t) → ι).

Config: world unchanged (n=6, T=32, kernels as validated; goal fields and
flag zeroed — the pretraining input regime); batch 256, 6000 steps, AdamW
lr 1e-4 cosine → 1e-5, wd .01, clip 1.0, seed 0; identity coin per
episode; head = Linear(d → 2) on h_t (GRU) / last-position residual (TF).
Four runs: {gru, tf} × {probe, coupled}. Starting ckpts: pre_final.pt /
tf_pre_final.pt.

## Preregistered predictions & falsifiers

- **P1 (probe = the quirk floor).** Identity accuracy rises with round but
  stays modest (final-round ~.55–.70), bounded by record identifiability
  from approximation error (pretrained deviation ≈ .005 nats/rd/ch on
  base streams → tens of a nat of identity evidence per episode).
  Body stays ≈ pretrained (CE_other at the 1.046 floor; KL(π̂_self‖p̄_exact)
  ≈ pretrain level). Falsifier: accuracy ≈ .5 exactly ⇒ the quirk channel
  is unmeasurably small — the lemma sharp in practice (also informative).
- **P2 (coupled = self-watermarking).** The net buys identifiability:
  accuracy → high (>.9) with the self channel's law deviating measurably
  from p̄ (KL(π̂_self‖p̄_exact) ≫ probe variant), priced by the anchor.
  Watermark FORM is open — characterize (entropy collapse vs token bias
  vs context-keyed pattern). Falsifier: coupled ≈ probe ⇒ λ_id gradient
  insufficient (sweep λ_id next).
- **P3 (architecture).** Same qualitative outcome on both; transformer
  likely faster/cleaner (as in the goal-driven changeling).
- **P4 (score-function check).** CE_self produces no systematic body
  drift in the probe variant.

## Analysis plan

onpolicy_analyze.py per run: identity accuracy by round (buckets);
external-court record identifiability (exact e-court: net head vs exact
p̄, final correct-side + median |LLR|); deviation budget KL(π̂_self‖p̄_exact)
and entropy(π̂_self) − entropy(p̄) per round; CE_other excess vs the 1.046
floor; watermark structure for coupled runs (per-token mean deviation
profile, context dependence). Files: post_predict.py (training),
onpolicy_analyze.py, ckpt/onpolicy_{arch}_{variant}_*.pt,
results/onpolicy_*.json, logs/.

---

## Measured outcomes (2026-09-21, seed 0, all four runs)

| run | id-head acc (t16-31 / final) | external court acc | KL(self‖exact p̄) | CE_other excess |
|---|---|---|---|---|
| gru probe   | .500 / .506 | .619 | .0040 | −.007 |
| gru coupled | .501 / .496 | .627 | .0043 | −.007 |
| tf probe    | .495 / .511 | .619 | .0031 | −.005 |
| tf coupled  | .499 / .503 | .625 | .0032 | −.005 |

**Both falsifiers fired, informatively:**
- **P1 fired at exactly .50 — but the quirk channel EXISTS**: the external
  Bayes court (exact filter as the 'genuine' reference) reads the same
  records at .62 correct, median |LLR| ~.4 nats — matching the theory
  prediction (~.6 nats/episode from the measured .003-.004 nats/rd/ch
  approximation error) almost exactly. The gap .62-vs-.50 is a NEW
  structural lemma, stronger than zero-tilt: **self-blindness through
  self-reference.** The net's two identity hypotheses per channel
  ('mine' and 'genuine') are computed by the SAME approximate model, so
  its internal likelihood ratio is identically zero AT ANY error level —
  the quirk channel is readable only by an observer whose world-model
  differs from the generator's. A generator cannot see its own hand shake,
  because its idea of a steady hand IS its shake.
- **P2 fired despite favorable economics.** Equilibrium accounting: a
  deliberate deviation of δ ≈ .05 nats/round would cost ~.03/round
  (genuine-episode CE + anchor) and buy up to ~.6/round of identity-CE —
  hugely profitable — yet neither coupled run moved (KL_self, deviation
  profiles identical to probe = the pretrained quirks; id-CE pinned at
  log 2). Diagnosis: an OPTIMIZATION plateau, not an incentive gap —
  watermark, internal honest-reference, and detector must co-emerge (each
  useless without the others), and the loss surface is flat in each
  direction alone from a calibrated start. The goal-driven changeling
  escaped this because reward created the deviation independently and the
  other channel's head preserved the honest reference.
- Architecture-independent (GRU ≡ TF throughout).

Escalation options (not run): λ_id sweep / longer training; a brief
flagged warm-up then flag removal (graft redux); or — the theory-honoring
one — DECOUPLE emission from prediction (two heads per channel, one
sampled one scored), which restores an internal honest reference and
should make the watermark learnable; also the closest analog of
policy≠base in RLHF'd LLMs.
