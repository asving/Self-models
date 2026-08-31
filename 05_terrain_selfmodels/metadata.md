# 05_terrain_selfmodels — the 2026-07-04..07 arc (one shared codebase)
Terrain world (ambush.py is the base module: World, Net, filters, tokens —
everything else imports it; cluster must stay flat). Question: when a net is
predicted/mirrored/copied, what self-knowledge does it build, and is any of
it privileged?
- ambush*: checkpoint mindreader (opponent = frozen reader of the agent);
  dodge discovery ~step 700; ambush_runs/{A,B,A_lag*}.
- mirror*: precedent mirror (opponent = agent's own in-episode per-key action
  histogram). Three-channel dissociation (record / match-detector / dodge);
  consult-then-compile; equilibrium = type-conditional NOISE (entropy 1.01
  vs 0.18 unwatched; identification ramp rounds 0-9) — see memory
  precedent-mirror-design incl. the 2026-07-15 corrections (the logged
  ent=0.03 was full-vocab mass-leak artifact; 91% mass on a junk token).
  mirror_circuit* = whitebox harness + verified-circuit figure (figs/).
- mirror2*: corrupted efference record (rho<1): premium certified 0.67 nats;
  camps-aware SMC referee.
- doppel*, doppel2*: zero-shot self-play transfer; twin-sync artifact; joint-
  token fix; kind-not-self verdict (sibling TV .068). doppel2_hop = the
  generalization-hop analysis.
- intent_probe.py: prospective intention pre-update probe (cond>marg>null).
Runs: {ambush,mirror,mirror2,doppel,doppel2}_runs/. Logs and figs local.
Memories: ambush-game-v1-results, precedent-mirror-design,
mirror2-corrupted-record, doppel-transfer-design, prospective-intent-probe.
