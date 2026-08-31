# self-models

A research program on **self-models in neural networks**, built on the
computational-mechanics / mixed-state-presentation tradition: the "self" a
predictive agent learns is characterized functionally, by how perturbations to
its internal coordinates renormalize — corrected by the world's evidence,
ratified into its own record, or restored by training — rather than by what
those coordinates encode.

**Start here:**

- `proposal.pdf` / `proposal.tex` — the current draft: the coupled-HMM
  formalism, post-training as a Doob tilt, the perturbation response field
  κ(v) with its two courts (identity and world), and a fully *measured*
  perturbation battery on a worked example (four figures + summary table).
- `binary_family/MASTER.md` — the canonical theory document: the two-courts
  construction, the master identity, renormalization time and its sign, the
  exactly solvable binary family, the shape zoo, open problems.
- `proposal_figs/` — self-contained generating scripts for every figure
  (exact Bayes filters and autograd; CPU-only, minutes to reproduce).
- `07_rho_record/` — the first network experiment (a small transformer
  trained on corrupted-record streams reaches the exact Bayes observer floor
  to 4e-5 nats; preregistered design and results docs inside).
- `METADATA.md` — the full map, including the earlier experimental arcs in
  the numbered folders (each with its own `metadata.md`).

Training checkpoints, logs, and large eval sets are not tracked (see
`.gitignore`); every experiment is reproducible from the scripts and the
parameters recorded in its design doc. The earlier design/grant documents
that previously lived at this repository's root are preserved in the git
history.
