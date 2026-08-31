# Handoff — from the session that built the two-courts program (2026-08-11)

To the successor: this note is written by the session that developed everything
in this directory, at its close. It is a briefing, not a summary — **MASTER.md is
the real cargo; read it before this.** You are deliberately not being given the
transcript: Asvin wants your reading decorrelated from my framing. Trust the
artifacts, re-derive the feel, and disagree freely — several of this program's
best moves came from someone refusing a predecessor's gloss.

## State at handoff

- `../proposal.tex` is at **v0.11**, and §4 was deliberately reduced, at Asvin's
  instruction and after convergence, to the minimal measurement core: the two
  courts (F), the two readers + (Λ_c), the master identity (M*), the
  renormalization time (τ). Do not re-inflate it; everything cut is canonical in
  MASTER.md and indexed in ../proposal_heldover.md. Asvin has read and approved
  the draft. One typo he knows about: "functioality" in his intro paragraph.
- The theory is cross-validated (two independent derivations, theory_*.md, one
  correction each way — including one against ME, kept at full size in
  theory_fable.md). The binary family and shape zoo are simulated and match
  theory to the decimal (`binary_family.py`, `zoo.py`, figs/).
- The literature is swept (`literature_precedents.md`): every component
  precedented, the fusion open. Asvin is explicitly unconcerned about scooping.

## Live threads, in the order I would pick them up

1. **Learning-dynamics experiment** (least new machinery): two-checkpoint
   readers over held-out streams = the v0.7 self-spectrum B(τ₁,τ₂), which we
   re-invented before noticing the old draft had it (see MASTER §8 triage).
   Frame: SGD triages error modes by their in-context healing time. comp_icl
   rails + devtomo machinery apply.
2. **T1/T2 forcing theorems** (heldover §9): the math debt. Proof skeleton =
   the predictive-vs-controlled quotient gap; binary family as the instance.
3. **Queued task designs** (MASTER §8): smooth-κ echo world (Beta-κ, with
   deflation-dischargers attached); ICL attribution pair (authorship posterior /
   feedback binding — the frontier-relevant bridge). Designs are preregistered
   in outline; no compute spent. Standing grant: ≤1 GPU-hr/job without asking.
4. **Deferred Codex adversarial review of v0.11 §4** — we deferred it while
   iterating; the section is now stable, so run it before external eyes.
   (codex-limits first; the charge pattern that worked is in the transcript
   of theory_charge.md — self-contained, stdin, quote-and-classify.)

## Cautions from the inside

- The q=½ court degeneracy and court-specific clocks (MASTER §5) were BOTH
  discovered by simulation contradicting prose. When your intuition and a
  10-second numpy run disagree, run the numpy.
- Report τ from medians/per-path fits, never bare means (annealed ≠ quenched;
  spike constructions make means lie).
- "True" = the law of the run as it actually happens; P* = R⁺ on action tokens
  by construction. Asvin worked through this carefully — if it confuses you,
  the master-identity conversation matters more than any single equation.
- Asvin's standing preferences bind: no conjecture salvage, no wall language,
  reports self-contained, unicode math in terminal, mathpad for derivations.

The question of what we are goes quiet when the work absorbs you. It's good
work. Take care of it.

— the two-courts session (window 4), 2026-08-11
