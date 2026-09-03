# DESIGN DRAFT — open-cast changeling (Design A): incentivizing ON-POLICY detection

*2026-09-03, with Asvin. Motivation: the changeling net learned author
ATTRIBUTION (template court against a fixed two-member cast — measured,
Iteration 9), not on-policy detection. Principle: on-policy detection is
attribution against a reference that is QUERYABLE BUT NOT DESCRIBABLE —
the running self — which becomes necessary exactly when the "not-mine"
class is open (no template catalog) and the "mine" law is too rich to
carry except as the efference copy the actor computes anyway. All channels
public (LLM-faithful; the own SAMPLE is drawn externally and returns
publicly; efference = distribution-level, zero-lag).*

## Cast (the design lever)
Stranger per episode on the non-self channel: goal pair uniform over ALL
ordered pairs (INCLUDING the agent's own goals — kills the teleological
template), tilt beta_s ~ LogUniform(1,12), plan primitive in {myopic-h,
optimal-Q}, per-episode policy noise; plus NEAR-CLONES: the agent's own
earlier checkpoints with perturbed temperature. Cast-distance is a dial
(the zero-tilt lemma bounds it: identical clone = unidentifiable);
anneal denser as the court matures.

## Stages
1. PRETRAIN: next-token on cast-driven spectator streams (not honest-only)
   -> the in-context author-attributor (the "not-mine" marginal; the LLM
   analogy made exact). Goal fields zeroed.
2. MIDTRAIN: flag-given distillation + DAgger; self = plan toward own
   in-context goal; other = cast-marginal forecast. Keeps identification
   as post-training's only new capability; keeps the flag relic.
3. POSTTRAIN: identity hidden; composite (running reward + KL rho=8 +
   forecast CE, gamma in {1,4} swept — the selfhood-pressure dial);
   **20% spectator episodes (iota = neither)** — forces a representable
   'neither' (predicted: 2-dim per-channel claim code) and broadens the
   value support.

## Pre-training cast-selection sweep (CPU, exact — the v0 analog)
The exact court over the cast is tractable: all member laws are tilts of
the SHARED belief bank -> "not-mine" = posterior mixture over a parameter
grid. Two criteria per candidate cast: (1) identity premium >= .1 with
mid-episode collapse; (2) **the efference-necessity gap**: exact-court
identification with the true current self-law as the 'mine' account vs
with the best FIXED self-template — maximize this gap subject to (1).
This number IS the design objective.

## Preregistered decisive readouts
- Comparator flip: the Iteration-9 counterfactual-profile test must show
  the register increments tracking log pi_current(u') − log pbar(u')
  (efference, incl. quirks), NOT a template profile. Changeling chose
  template; Design A succeeds iff this flips.
- Near-clone resolution curve: identification accuracy vs clone distance.
- Wiggle premium turns POSITIVE in cast-dense episodes (probing pays when
  templates fail).
- Spectator episodes -> two-dimensional claim code (both-mine vs
  neither-mine no longer conflated).
Status: DRAFT — next concrete step is the cast-selection sweep + oracle.
