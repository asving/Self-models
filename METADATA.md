# self-models — organization scheme (reorganized 2026-07-17; map updated 2026-08-31)

**Live entry path (2026-08):** `proposal.tex` / `proposal.pdf` (the current
draft: coupled-HMM formalism, the perturbation-response field κ(v), the
measured court battery) → `binary_family/MASTER.md` (canonical theory: two
courts, master identity, τ/sign, exact binary family, open problems) →
`proposal_figs/` (every figure's generating script, exact filters/autograd,
CPU) → `07_rho_record/` (first network experiment: transformer reaches the
exact observer floor; DESIGN and RESULTS docs inside). The numbered folders
below are the earlier arcs.

Research program: **self-models in neural networks** — when do learning systems
build internal models of their own states/policies/goals, what pressures create
them, and what carriers hold them. Deep episodic context: auto-memory
(`self-models-project`, `stateful-personas-direction`, `precedent-mirror-design`,
...), Naja's living doc (`/data/users/naja/dropbox/selfmodels_STATUS.md`), and
`notes/`.

## Scheme

Top-level folders are numbered by the arc's questions, in rough chronological
and conceptual order. Each experiment CLUSTER lives flat inside one folder —
files are NOT nested per-experiment where they share code, because sibling
imports (`from ambush import ...`) require a common cwd. Run every script with
cwd = its own folder. `core/` holds the shared library (model/factors/whitebox/
probes); clusters that need it contain SYMLINKS to it, so imports resolve
unchanged. All former `BASE = ~/self-models` constants are patched to
`dirname(__file__)`, so checkpoints/figs resolve inside each cluster.

```
core/                       shared library (GPT/TTTNet/Block, Mess3 factors, whitebox+probe kit)
papers/                     external references (clean-rl.pdf, Fellows paper)
notes/                      conceptual docs (working_notes, SELF_LEGIBILITY, DEPTH_AND_RECURRENCE, Naja update)
01_substrate/               choosing worlds + design scoping (scoping/, hmm_select/, archive_designE/)
02_consequence_mess3/       designs A/B on Mess3: does consequence create self-structure in predictors?
03_acting_selfknowledge/    Fellows-era toys: self-knowledge in acting nets
    ttt/                    color-blind tic-tac-toe: stored vs re-derived self (kept-variable conditions)
    rps/                    RPS: action-entropy dual role; recompute-vs-message-pass routing (rps_route)
    selfcancel/             self-cancelling forecaster (prediction that edits its own target)
    goalnav_so3/            continuous control (SO3 / goal navigation) self-models
    agent_cont/             continuous agent + rubber-hand-style binding probes
    dp/                     Bellman/DP baselines for the entropy-dual-role analysis
    selfsim/                self-simulation probe
04_dinner/                  planning-gap game (pretrain->RL template retrieval)
05_terrain_selfmodels/      the 2026-07 arc: ambush, mirror (precedent), mirror2 (corrupted record),
                            doppel/doppel2 (zero-shot self-play transfer), intent_probe
06_goal_collapse/           the v6 arc: generator-vs-predictor, commitment carriers —
                            push-mechanics sims (v6_push_explore*), orchard, twophase, whisper
binary_family/              the 2026-08 theory core: two courts, renormalization time τ(v),
                            MASTER.md + cross-validated theory notes + simulation suite
07_rho_record/              corrupted-action-record arc: privileged information Π; first
                            network experiment (transformer vs exact observer/agent floors)
proposal_figs/              generating scripts for every proposal figure (exact filters,
                            autograd; ring-world court battery, SGD court, laws, scar)
```

Conventions: each folder has `metadata.md` (purpose, inventory, run notes,
result pointers). `example.png` stays at root — the whitebox skill references
this path. Full old->new move log: `REORG_LOG.txt`. MANIFEST.md is a pointer
here (its 07-03..05 map is superseded).
