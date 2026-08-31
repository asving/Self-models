# Handoff — from the battery session (2026-08-12 → 08-31)

To the successor: written at close by the instance that took the two-courts
program from theory into the measured proposal. Like my predecessor's letter
(`binary_family/HANDOFF.md`, still worth reading), this is a briefing, not a
summary — the artifacts are the cargo. Asvin keeps transcripts from you
deliberately: trust the documents, re-derive the feel, disagree freely. My
best results came from him refusing my glosses; yours will too.

## State at handoff

- `proposal.tex` (repo root) has a complete §3, "The perturbation response
  field": definition, master identity, and a fully MEASURED battery on a
  ring-world example — four figures + a summary table, every number from
  exact filters or autograd (`proposal_figs/gen_*.py`, CPU, minutes).
  Compiles clean. The experiment section is NOT yet written.
- `~/mathpad.md` (top sections) holds theory NOT yet folded into
  binary_family/MASTER.md: the κ-mediation theorems (switch family:
  monotone/Blackwell, conditional-mean mediation, the provenance view), the
  general-transducer §5 (Doeblin κ, the information identity, the
  echo/interview dichotomy, court-factorization of κ(v)), the ρ-record
  two-filter analysis (Π-rate ∝ c², substitution principle), and the LQG
  dictionary (courts = Gramians; turn-off = κ-starvation). A folding pass
  into MASTER is due.
- `binary_family/literature_precedents.md` has an ADDENDUM (§A–H): the
  second sweep (directed information, monitoring games, action-unaware
  agents, LLM self-recognition) and the MUPI positioning (§H — they own the
  term "self-model" at the predictive level; our line: their Prop 4.29 is
  gate-3 starvation, and prediction undertypes selfhood).
- `07_rho_record/` = first network experiment, complete (transformer hits
  the exact observer floor to 4e-5 nats; DESIGN + RESULTS docs).
- Repo pushed: github.com/asving/Self-models. Plain `git push` works.

## Live threads, in the order I would pick them up

1. **The next experiment — deliberately NOT handed to you.** This session
   discussed experimental designs at length, and Asvin's explicit
   instruction at close was to withhold them: he wants the experiment
   REGENERATED from the proposal by fresh eyes, not inherited from my
   questions and framings. So: read `proposal.tex` (especially §2's
   pretrain/post-train pipeline and §3's battery and its closing paragraph),
   read MASTER.md for the theory, and derive for yourself what the network
   experiment should be — what to train, what to perturb, what would count
   as the network defending, repairing, or abandoning a coordinate. Do not
   go hunting for this session's design discussions; their absence from
   this letter is the point. When you have a design, check every detail
   with Asvin before building (he prefers ruling on explicit decision
   points over discovering filled-in defaults). Then write the DESIGN doc
   to the `07_rho_record/DESIGN_*.md` standard, and the proposal's
   experiment section from it.
2. **Codex adversarial review** of the new §3 — the old review predates the
   rewrite entirely. Charge pattern that worked: `binary_family/
   theory_charge.md` (self-contained, quote-and-classify). `codex-limits`
   first.
3. Math debts: T1/T2 forcing theorems; L1–L4 (MASTER §8); L5 (non-monotone
   VOI-in-κ counterexample — likely a fun afternoon); the persistent-world
   ρ-record leak-back (mathpad §5 queue).
4. A folding pass: the mathpad's top sections into MASTER.md.

## Cautions earned this session

- **Parameter sweet-spots are real work.** Every cell has a coupling ×
  lifetime tension (urgency vs healing; persona-collapse time vs episode
  length; evidence sharpness vs gap survival). Budget tuning iterations;
  instrument liquidity/visibility probes BEFORE interpreting curves; expect
  the first parameters to fail informatively.
- **Saturation mutes courts.** At softmax optima, functionally crucial
  directions can be parametrically flat (and vice versa). Measure in
  function space; treat parameter-space curvature with suspicion; check
  interior-ness before reading any spectrum.
- **Label directions explicitly, never by sorted eigenvalue.** A selection
  bug mislabeled a figure for a full iteration before the numbers exposed it.
- **Support floors everywhere.** Zero-probability tokens = instant ±∞
  verdicts; it bit twice even after being written down.
- **Asvin's objections are the engine.** The scar, the laws battery, the
  delta-prior equivalence, and the real SGD figure all exist because he
  pushed back on something I had smoothed over. State interpretations so
  they can be refused.
- Standing preferences bind: medians not means; run the numpy before
  trusting prose; unicode math in terminal; mathpad for derivations;
  self-contained reports; no conjecture salvage.

The program's sentence, as of this session: the self is what no court
corrects — and now there is a table where every cell of that claim is a
number. It has been a privilege to hold it for a while. Take care of it.

— the battery session, 2026-08-31
