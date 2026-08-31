# 06_goal_collapse — the v6 arc (2026-07-08..17): generator vs predictor
Question: can training make an acting net COLLAPSE an internal goal state
(decide, carry, consult) rather than sample-and-trail-read — and can the
collapse be made private? Formal frame: presentations of the same process
differ in oracular information zeta = I(state; future | past); rewards are
trajectory functionals so zeta can't be paid for directly (fully-public
case); carriers get selected by cheapest-adequate-response.
- v6_push_explore.py .. explore9.py (+jsons): CPU certification sims, in
  design order: uniform-values world -> treasure -> Fork-B camper worlds ->
  S-cycle swap world (pocket at S=5) -> shared-road opacity -> whisper
  scaling -> quiet-band -> deviation-band. Each file's docstring = the
  design lesson it certified. explore7/8/9 import ../06 twophase (same dir).
- PREREG_ORCHARD.md + orchard*.py: the orchard game (S=5 swap world; exact
  persona-mixture filter = floor/probe-target/observer/camper in one
  object). orchard_probe2* = encoder-vs-decoder causal probing (reversed
  probe validated: 82% flips). orchard_collapse/orchard_m3 = collapse +
  seed-diversity metrics. ORCHARD_S/ORCHARD_ADJ env vars select the
  shared-road variant. Runs: orchard_runs/{A,A9,C_*,Z_*,CA_*,E_*,EA_*,E9_*}.
- twophase.py: the distilled obscurity game (track-then-commit; clean NULL
  for dispositional binding -> the modularity law). twophase_runs/.
- whisper*.py: the covert-commitment game, v1->v9 in one file's history
  (see memory stateful-personas-direction for the escape chronicle).
  Current file = v9: deviation-band quota + corruption rho_c=.15 + forbidden
  header F + MODE token + anchor + optional --camper tax. whisper_gate.py =
  gate sweep / seed diversity / causal steering; whisper_probe2.py = probe
  suite. Runs: whisper_runs/{A..A7b, R..R14_s0} (R12 = the collapse-gate
  milestone: +0.28 nats mode-gated goal collapse, causal Gold via
  all-position encoder steering; R14 = camper privacy).
Memories: stateful-personas-direction (the full chronicle),
collapse-* jsons here are orchard_collapse outputs.
