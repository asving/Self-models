# Self-models

Probing for **Self/World factorisation** in trained agents — do RL-trained
reasoning models develop an explicit belief over task latents *and* a causal
self-representation, separable from that belief and load-bearing in the forward
pass? We test this in a controlled toy setting with analytic ground truth (the
**SWITCH** coupled HMM) and then extend the probe suite to language models.

Builds on the belief-state-geometry line of work (Shai et al., 2024) and the
"privileged null" framing of the self in post-trained models.

## Contents

| File | What it is |
|---|---|
| `design.tex` | **Primary.** Working experimental-design document: the SWITCH spec, training regimes (R1–R6), measurements (M1–M6), interventions (I1–I4), predicted-outcome table, phasing. The thing to run experiments from. |
| `grant_proposal.tex` | Grant version — full motivation, developmental-cognitive-science case, broader framing. |
| `strand_two.tex` | Theory target: the *geometry* of the factorisation (product simplex, persona-selection MSP, nested belief fibers, the self as a degenerate fiber). `\input` by both docs. |
| `strand_three.tex` | Theory target: *epistemic actions* (queryable quotients) and introspective access — why the behavioural test is deflationary and how the mechanistic test breaks the equivalence. `\input` by both docs. |
| `references.bib` | Shared bibliography. |
| `archive/` | Superseded drafts (v1 proposal, standalone developmental-motivation note). |

## Build

```sh
make            # builds design.pdf and grant_proposal.pdf
make design     # just the design doc
make clean      # remove build artifacts
```

Requires a LaTeX toolchain (`latexmk` preferred; falls back to
`pdflatex`+`bibtex`).

## Status

Design stage — no experiments run yet. The pivotal experiments are flagged in
`design.tex`: whether a token-type self emerges in regime 2b (observation-only
loss, no gradient at action positions), and whether an epistemic self with
*causal* introspective access emerges in regime 6.
