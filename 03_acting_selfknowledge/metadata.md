# 03_acting_selfknowledge — self-knowledge in acting nets (Fellows-era toys)
Each subfolder is a self-contained cluster (own runs/, logs/, figs/, core
symlinks). Common thread: a net that ACTS, probed for whether/where it knows
its own policy/state — precursors of the 05/06 arcs.
- ttt/: color-blind tic-tac-toe; stored-vs-re-derived self variable
  (storedself_check = the kept-variable causal test).
- rps/: RPS vs adaptive opponents; action-entropy dual role (reward-now vs
  info-later); rps_route = recompute-vs-message-pass routing result;
  wb_* = whitebox decodes; make_figs.py builds the entropy figure set
  (reads ../dp/dp_results.json).
- selfcancel/: forecaster whose prediction edits its own target.
- goalnav_so3/: continuous-control self-models (SO3 integrator, goal nav).
- agent_cont/: continuous agent, rubber-hand binding probes (cont_*, d2_*),
  HMM-hard variants; sweep scripts; aggregate_report -> figs/figures.html.
- dp/: Bellman/DP optimal baselines for rps entropy analysis.
- selfsim/: self-simulation probe.
