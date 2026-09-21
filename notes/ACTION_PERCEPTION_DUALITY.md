# Action / perception as an arrow-of-time duality — first pass (2026-09-21)

Prompted by Asvin's intuition: perceiving = past channel → current internal state;
acting = current internal state → future channel; time-reversal dual. Second thought:
"perhaps the dual of acting is learning." This note records the formal homes for
the intuition, the correction I'd make, and what it suggests we measure. Codex
adversarial literature check logged in `notes/codex/duality_litcheck_*.log`.

## 1. The duality is already inside proposal.tex

- (M)  η_{t+1} ∝ η_t T^{(x)}  — the Bayes filter, a **measure** pushed forward from the past.
- (π_g) h_t(μ) = E_{π₀}[e^{βR(μ_f)} | μ_t = μ] — a **function** pulled backward from the future
  (backward-harmonic; Doob h-transform); π_g(a|μ) ∝ π₀(a|μ) E[h(μ') | μ, a].
- The pairing ⟨η T, h⟩ = ⟨η, T h⟩ is the L²-adjointness of the forward (Fokker–Planck /
  Kolmogorov-forward) operator on beliefs and the backward (Kolmogorov-backward) operator
  on values. The action logit is the log of this pairing. **This is the physics
  forward/backward arrow-of-time duality, and it pairs perception with action-selection.**
- Kalman 1960: LQE ↔ LQR under (A,B,C,t) ↔ (Aᵀ,Cᵀ,Bᵀ,−t); observability ↔ controllability.
  Todorov 2008 (CDC) "General duality between optimal control and estimation" extends it
  to KL/linearly-solvable control — exactly our tilt. Control-as-inference (Attias 2003;
  Toussaint–Storkey 2006; Kappen–Gómez–Opper 2012; Levine 2018): α-messages = filter,
  β-messages = value, reward = likelihood of a fictitious future "optimality" observation.

## 2. "The dual of acting is learning" — exact content

(†)  learn:  P(θ | x_{≤t}) ∝ P(θ) · exp(Σ_{s≤t} log p(x_s | θ))      tilt of a prior law by the REALIZED past
     act:    π_g(a | μ)   ∝ π₀(a | μ) · exp(log E[h(μ') | μ, a])      tilt of a prior law by an IMPOSED future

Same functional form (Gibbs tilt of a prior law by a log-likelihood; Korbak et al. 2022
already cited for the RL side). The asymmetry is not in the form but in the evidence:
the past has a record; the future's "observation" (success) is supplied by the agent.
**A goal is a self-supplied observation of the future.** Structural consequence for the
battery: the world court adjudicates realized tokens only, so nothing in the stream can
refute a fictitious observation ⇒ goal-register perturbations are ratified (Fig. 1
right) and only the reward court speaks. The reward column of Table 1 is the *dual
court* — the future-facing half of the same measurement that κ makes on the past.

## 3. Correction: strict time reversal pairs by timescale; the intuition detects the write/read split

2×2, {fast, slow} × {past, future}:
- fast, past: perception (Bayes filter η)          — fast, future: action selection (h-tilt at η)
- slow, past: learning (SGD/pretraining)            — slow, future: policy formation (RL post-training tilt; niche construction; performativity)

Strict time reversal is the column swap; acting↔learning is the anti-diagonal. What the
intuition is detecting is a different split: acting and learning are the two
**irreversible writes** (into the world's state n' = E(n,a), and into the weights θ),
while unifilar Bayes filtering is a lossless read. Both writes leave scars (proposal:
SGD heals function not parameters; emitted tokens cannot be un-emitted).

Perspectival form: in the symmetric two-transducer loop of §2, *my action is the
environment's perception* (the environment's state is its memory of my actions).
Consequence — the founding quantity — is the world's learning rate about the agent.
Bipartite stochastic thermodynamics has this exactly: d/dt I(X;Y) = İ^X + İ^Y with İ^Y
the sensor's "learning rate" (Barato–Hartich–Seifert 2014; Hartich–Barato–Seifert 2014;
Horowitz–Esposito PRX 2014), İ^X ≤ 0 for a non-acting sensor (the world forgets), gaining
a positive feedback term when the world's kernel depends on the agent. Second-law bounds
come from the time-reversal fluctuation theorem. (Sagawa–Ueda 2010/2012: the feedback
efficacy γ is computed on the time-reversed protocol — a reversed-protocol *quantity*,
NOT a measurement↔feedback duality theorem; Codex flagged my first phrasing as
overclaiming.) Ito–Sagawa 2013 generalize to arbitrary causal networks.

## 4. Where time reversal of the PROCESS is not control

Reversing the actual joint process gives the retrodictive (Bayes-inverse) kernel
K^R_t(z|z') = ρ_t(z) K_t(z'|z) / ρ_{t+1}(z') — Crutchfield–Ellison–Mahoney 2009 "Time's
barbed arrow", reverse ε-machine, causal irreversibility — i.e. retrodiction under the
existing policy and occupancy, not the controller. Control is the backward operator
applied to a reward, not to a distribution. The duality is of computations, never of
the world. The self/other swap is exact only in §2's symmetric loop and breaks once
reward enters.

**Where the inexactness bites in proposal.tex (side finding from the Codex pass, not
previously recorded in the repo).** (π_g) as written — h_t(μ) = E_{π₀}[e^{βR(μ_f)} | μ]
with π_g(a|μ) ∝ π₀(a|μ) E[h(μ')|μ,a] — is the *trajectory-level* tilt: it tilts the law
of x-tokens as well as a-tokens, so it is the "optimistic"/risk-seeking control-as-
inference variant (Levine 2018 §4; Todorov's linearly-solvable case). The KL-regularized
RL optimum with the environment kernel held fixed is the constrained soft-Bellman form
   V_t(μ) = (1/β) log Σ_a π₀(a|μ) exp(β E_x[V_{t+1}(μ')]),   π*(a|μ) ∝ π₀(a|μ) exp(β E_x[V_{t+1}(μ')]),
i.e. expectation over the environment step *inside* the exponent, not outside. The two
coincide when the belief step μ→μ' is deterministic given a, or as β→0; in the ring world
(slip 0.25, noisy readings) they differ. Duality-theme reading: learning conditions on
the whole record, acting may tilt only its own court — the a/x split appears inside the
duality itself. Whether real RL-trained nets sit nearer π* or π_g is an empirical question
for the network stage; the exact-filter battery is internally consistent either way.

Additional exact nonlinear dualities Codex surfaced: Fleming–Mitter 1982 (log transform
of the pathwise filtering PDE gives an HJB — nonlinear estimation ↔ auxiliary control);
Kim–Mehta 2024 arXiv:2208.06587 (minimum-variance filtering = BSDE control problem, an
exact nonlinear Kalman duality). Ortega–Braun 2010 arXiv:0911.5104 (Bayesian control rule:
interventions vs observations in adaptive control) — relevant to the interventional stance.

## 5. Other literature homes (one line each)

- Friston FEP: perception/action minimize one free energy from opposite sides ("which
  side yields"); learning = same, slow; niche construction (Constant et al. 2018) = slow action.
- Tishby–Polani 2011 "Information theory of decisions and actions": information gained
  vs control information injected, both with Bellman recursions.
- Empowerment (a→x capacity) vs predictive information; Abel et al. 2025 plasticity as
  mirror of empowerment (already in literature_precedents.md §A).
- Conant–Ashby 1970 good regulator theorem; Francis–Wonham internal model principle.
- Pontryagin costate (∂V/∂s, backward) vs backprop adjoint (∂L/∂θ, backward): acting
  and learning are the same adjoint computation differing in the variable adjointed.
- Categorical cybernetics (Capucci–Gavranović–Hedges–Rischel 2021; Smithe 2020 Bayesian
  lenses): backprop, open-game utility, Bayesian inversion, dynamic programming all
  backward passes of an optic.
- Schrödinger bridge (Léonard 2014): path law = prior × φ̂₀(x₀) × φ_T(x_T); prior η₀ and
  goal are the two self-supplied boundary potentials. (Pretty picture; not a claim.)

## 6. What it suggests we measure

1. **Desirability geometry, the dual of belief-state geometry.** Pure predictors need
   only α (Shai et al. simplex). Post-trained nets must carry (an approximation of) h —
   a point in the dual cone (positive functions on states mod scale). Exact h_t is
   computable in the ring world (proposal_figs machinery). Probe the post-trained
   network for h_t alongside η_t; predict a linearly decodable h-coordinate that is
   absent before post-training, with action logits linear in log⟨η, h⟩.
2. **The dual battery.** For each direction v: κ(v) (past courts: corrected/ratified/
   ignored) and C(v) := response of h (or of the realized-future KL) under do(v) — the
   original *consequence*. Self := ratified ∧ consequential. Plot κ against C.
