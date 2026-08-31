# self-models — project scaffold

Research project on **self-models in neural networks**, operationalized via **consequence**.
Deep context lives in auto-memory (`self-models-project`, `simplex-*`) and in `scoping/design_{A,B,C,D}_*.md`. Read those first.

## One-paragraph frame
The "self" a predictor learns = the part of its world-model with privileged access to its own future. **Consequence** of a circuit C = how much the *actual realized* next token (not the prediction) depends on C's output, through the model's action and the environment. Pure predictor of an exogenous process ⇒ consequence 0 for all circuits even at zero loss. Claim: high-consequence circuits get *modeled* by the rest of the net (efference copy / self-factor), learnable by ordinary next-token SGD on-policy without closing the gradient loop. Signature: the world-state update conditions on the internally-read action (`η' = g(η, x_{t+1}, a_t)`).

## Codebase
Reuse `~/comp_icl` (its `.venv`, py3.11): `generator.py` (Mess3 ops + `CompositionMixture` exact-Bayes belief oracle; `eps=0` = independent factors), `model.py` (custom GPT, `return_hidden`, hooks; no TransformerLens), `train.py` (online loop + oracle floors), `probe.py` (`ridge_fit`, `subspace`/`overlap`, causal steering). Run with `~/comp_icl/.venv/bin/python`.

## Designs (see scoping/)
- **A** action head: add a consequential action that feeds the env combiner f. Cleanest first test.
- **B** self-resampling (run *free*): read a factor's belief, sample a hidden state, reset that HMM, gen ~k tokens. Prediction: that factor's belief geometry *collapses* in entropy. Don't impose argmax — let collapse emerge.
- **C** alternating action/observation transducer (true agency; capstone).
- **D** multi-persona (self vs other; defer full version).

## Box etiquette (shared 8×H100, no scheduler)
`nvidia-smi` before any GPU job; only use a GPU at ~0 MiB & 0% util; claim via `CUDA_VISIBLE_DEVICES`; single GPU default; never kill others' procs; long jobs in tmux + `tee` logs. Belief-geometry analysis is CPU-only.

## Layout (reorganized 2026-07-17 — see METADATA.md for the full scheme)
- `core/` shared library (model/factors/whitebox/probes; clusters symlink it).
- `01_substrate/` (scoping + hmm_select + archive), `02_consequence_mess3/`,
  `03_acting_selfknowledge/{ttt,rps,selfcancel,goalnav_so3,agent_cont,dp,selfsim}`,
  `04_dinner/`, `05_terrain_selfmodels/` (ambush/mirror/mirror2/doppel arc),
  `06_goal_collapse/` (orchard/twophase/whisper arc), `notes/`, `papers/`.
- Run scripts with cwd = their cluster folder; BASE constants resolve to the
  folder; each folder has metadata.md.

## Reporting note: results reports must be self-contained per the "Reports & summaries" section of ~/.claude/CLAUDE.md (re-ground the game + define every metric inline; assume no short-term memory, deep long-term memory).
