# Design E — Vicarious pretraining → self-binding (scoping report)

**Claim under test.** The do-structure (action-indexed belief-update operators `T^{(a,o)}`) can be
learned *vicariously* — from observing many OTHER actors drive the world — and survives even when the
agent's own policy later collapses. Post-training then only has to *bind* the already-built action
slot to "self." This is the conjectured mechanism behind post-trained LLMs' counterfactual competence
despite near-deterministic policies. Secondary targets: the factored who×what belief geometry
(actor-identity posterior ⊥ env-state posterior), and what happens in actor-identity space when the
actor becomes the net itself ("self as a new direction").

## 1. Environment: regime-controlled Mess3

3-state world, 3 actions, 3 observation symbols. **The action selects the world's kernel**:
`A[a] = mess3_operators(alpha_a, x_a)` with defaults

| a | semantics | (alpha, x) | solo attractor |
|---|---|---|---|
| 0 | watch  | (0.60, 0.15) | classic Mess3 flower |
| 1 | pin    | (0.90, 0.05) | corner-pinned skeleton (near-synchronized, H≈0.27) |
| 2 | fog    | (0.15, 0.25) | sparse spidery web |

Verified (STEP 0, `vicarious_oracle.py`, `figs/vic_fan_v2.png`): per-actor belief attractors are
**shape-distinct**, not recolorings. (A cyclic-shift action `T[z]@P_a` only recolors, by Mess3's
symmetry — rejected.) Union of all actors = the full operator-family geometry.

**Token stream (actions visible).** Interleaved `[o_0, a_0, o_1, a_1, ...]`, vocab 6
(o-tokens 0–2, a-tokens 3–5) + BOS. Step: actor picks `a_t ~ π(a | a_{t-1}, o_t)`; env applies
`A[a_t]` from `s_t` → emits `o_{t+1}`, moves to `s_{t+1}`. Efference/pooled-observation variant
(o = f(e,a), actions hidden) is deferred — different experiment (regime-3), don't conflate.

## 2. Actor library (pretraining "others")

K=6 types, drawn per sequence: `const0`, `const1`, `uniform`, `sticky` (repeat w.p. .75),
`follow` (a=o w.p. .9), `avoid` (a=o+1 w.p. .9); 10% uniform noise on deterministic types so
likelihoods stay soft. Oracle identification takes ~10–30 steps (`figs/vic_ident_v2.png`) — the
actor posterior has real transient structure. Extensible: Dirichlet-bias continuum family later.

**Exact oracle (all CPU).** Actions observed ⇒ the joint posterior FACTORIZES exactly:
env-belief `η' ∝ η A[a_t, o_{t+1}]` (shared across actor hypotheses) and actor-posterior
`w_k' ∝ w_k π_k(a_t | a_{t-1}, o_t)`. Ground truth (η ∈ Δ², w ∈ Δ⁵) at every position; also
oracle CE floors for both token types.

## 3. Phases

- **Phase 1 (vicarious pretraining):** next-token CE on ALL tokens. Predicting a-tokens forces the
  actor posterior (who is this?); predicting o-tokens forces the env belief + operator application.
- **Phase 2 (self-binding):** at a-positions, SAMPLE from the net's own action head (T=1), feed to
  env, continue training on-policy.
  - **2a (clean):** loss masked to o-tokens only — pure world-prediction while acting.
  - **2b (self-distillation):** loss on both — CE on own samples = the entropy-collapse pressure;
    watch collapse and self-binding co-evolve (Fellows-paper miniature).

## 4. Controls

- **C1 no-coverage:** pretrain on `const0` only → do-test must FAIL for a∈{1,2} (operators never
  exercised). Kills "the architecture just computes Bayes."
- **C2 coverage-without-diversity:** pretrain on `uniform` only → do-test should pass but NO
  actor-identity subspace should form. Separates operator learning from other-modeling.
- **C3 no-consequence:** shuffle actions before they hit the env (actions decorrelated from
  transitions) → net should learn plain single-kernel beliefs, no operator family.

## 5. Probes & figures

1. **F1 the fan** (money shot): 2×K grid — theoretical attractor per actor (done) vs net beliefs
   (ridge probe resid→η, per layer; plot probed η on Δ²). Fractal-for-fractal match.
2. **F2 identification funnel:** probed actor-posterior (resid→w) vs oracle curves.
3. **F3 who⊥what:** principal angles between env-belief and actor-posterior subspaces (factored
   world hypothesis extended to self/other-relevant factors).
4. **F4 do-test:** feed action sequences from HELD-OUT policies (e.g. anti-follow a=o−1, adversarial
   schedules) + single-action patches; probed η must track the exact filter (R², overlay plots).
   Run on phase-1 net (vicarious claim) and on C1/C2 (controls).
5. **F5 self-binding:** (i) self-driven attractor vs theory-under-own-realized-policy ("the world
   under my own hand"); (ii) actor-posterior under self-driving — does it settle on an old vertex or
   grow an out-of-subspace "self" direction (probe-reconstruction residual growth during phase 2);
   (iii) 2b: action-entropy collapse curve + attractor sharpening.

## 6. Implementation & compute

- Files: `vicarious_oracle.py` (done), `vicarious.py` (env+actors+train, comp_icl idiom: Block
  import, online generation, oracle floors), `vicarious_probe.py`, `vicarious_do.py`,
  `vicarious_figs.py`. Runs → `vic_runs/`.
- Model: 4L, d=128, 4 heads (comp_icl GPT). Seq 128 steps (257 tokens). Batch 256, AdamW 1e-3,
  ~30k steps phase 1; ~5–10k steps phase 2 per variant. Loss curves vs oracle floors.
- Box etiquette: `nvidia-smi` first, one free GPU via `CUDA_VISIBLE_DEVICES`, tmux session
  `vicarious`, `tee` logs into `logs/`. Probing/figures CPU. Est. well under 1 GPU-hour per run;
  7 runs total (phase1, 2a, 2b, C1, C2, C3, +1 seed repeat of phase1).

## 7. Success criteria

- Phase-1 net: env-belief R² ≳ 0.95 at best layer, actor-posterior decodable, subspaces
  near-orthogonal, F1/F2 visually matching theory.
- **Vicarious claim confirmed if:** phase-1 (and phase-2-collapsed) nets pass F4 on held-out
  policies while C1 fails exactly on the unexercised operators.
- **Self-binding signal:** any reproducible F5(ii) out-of-subspace growth, and/or 2b collapse with
  retained do-competence (the LLM story in miniature).
