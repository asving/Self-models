# Literature precedents for the τ/sign two-courts program

*Web literature sweep, 2026-08-11. Companion to `MASTER.md`. Every citation below was
checked against live search results; the few not individually confirmed are marked
**[standard, not re-verified]** or **[partially verified]**. Nothing here is from memory alone.*

---

## 0. The construction, decomposed into components

For cross-referencing, the program's construction (MASTER.md §1) is split into seven
labeled components. An agent in a closed loop carries internal coordinates; to type a
coordinate direction v as *self* (commitment/identity) or *world* (belief), we:

- **C1 — two-reader LR monitoring.** Run two frozen copies of the agent's own model
  (R⁺ perturbed at t₀, R⁻ not) forward on the SAME realized stream, and accumulate the
  log-likelihood ratio of their per-token predictions.
- **C2 — court split.** Split Λ by token type: action tokens (identity court Λᵃ — did
  the perturbation become the agent's state?) vs observation tokens (world court Λˣ —
  did the world object to its content?).
- **C3 — efference tag.** The agent's private record of what it injected, needed to
  write the took-hypothesis and to exclude self-authored acts from evidence
  (court-specific clocks: forced acts are the injector's).
- **C4 — do vs injection.** Enact the perturbation internally (*do* on the coordinate)
  vs through the action channel only (inject its action-shadow); the a-court sign
  difference isolates penetration.
- **C5 — τ as the boundary quantity.** The renormalization/healing time of the
  disagreement is the *defining* quantity of the self/world boundary.
- **C6 — self = never-renormalizing directions.** Vertex/point-mass (mutually
  singular) initializations that no amount of shared stream reconciles.
- **C7 — target = internal coordinates of neural networks** (activation directions,
  measured by patch/steer interventions).

Headline verdict (details in §8): **every component individually has a literature,
several of them deep and old; the specific fusion C2+C5+C7 — per-channel court split,
renormalization-time-as-self-boundary, applied to NN internals — appears nowhere found.**

---

## 1. Control theory / fault detection & isolation (FDI)

This is the field that owns C1 in its engineering form: monitor a running system by
comparing model predictions against a realized stream.

### 1.1 Sequential change detection
- **Wald, A. (1945), "Sequential Tests of Statistical Hypotheses," *Annals of
  Mathematical Statistics* 16(2):117–186.** The SPRT: accumulate the two-hypothesis
  log-likelihood ratio until a threshold. **Overlap:** the identity court run between
  two roster policies IS an SPRT; the program's (τ*) = depth-in-nats / per-step-KL is
  Wald's expected-sample-size law, with the same overshoot corrections. **Missing:**
  open-loop data, no channels, no agent, hypotheses are exogenous rather than
  "my own state, perturbed vs not."
- **Page, E. S. (1954), "Continuous Inspection Schemes," *Biometrika* 41:100–115** and
  **Shiryaev, A. N. (1963), "On Optimum Methods in Quickest Detection Problems,"
  *Theory of Probability and Its Applications* 8(1):22–46.** CUSUM and Bayesian
  quickest detection: detect *when* a stream's law changed. **Overlap:** the
  took/not question at t₀ is a change-point question; the plateau-vs-drift dichotomy
  echoes their detectability theory. **Missing:** they detect *persistent* changes
  against a single null model; a perturbation that heals (renormalizes) is exactly the
  event their machinery is designed to ignore, whereas for us τ of the healing is the
  measurement.
- **Basseville, M. & Nikiforov, I. V. (1993), *Detection of Abrupt Changes: Theory and
  Application*, Prentice-Hall.** The canonical synthesis of the above for dynamical
  systems (residuals, GLR, CUSUM). Useful as the single reference for the whole toolbox.

### 1.2 Two-model monitoring on a shared realized stream
- **Willsky, A. S. & Jones, H. L. (1976), "A generalized likelihood ratio approach to
  the detection and estimation of jumps in linear systems," *IEEE Trans. Automatic
  Control* 21:108–112.** Closest FDI object to the reader pair: hypothesis "state
  jumped by ν at θ" vs "no jump," scored by likelihood ratio on the same realized
  innovations stream from a single Kalman filter. **Overlap:** perturbation-at-t₀
  hypothesis pairs, LR on shared data, matched filters. **Missing:** the plant has no
  identity coordinate — all tokens are observations, so there is nothing to split; the
  jump is assumed persistent (no healing time); the output is detect/estimate, not a
  self/world typing.
- **Magill, D. T. (1965), "Optimal adaptive estimation of sampled stochastic
  processes," *IEEE Trans. Automatic Control* 10(4):434–439** and **Blom, H. A. P. &
  Bar-Shalom, Y. (1988), "The interacting multiple model algorithm for systems with
  Markovian switching coefficients," *IEEE Trans. Automatic Control* 33(8):780–783.**
  Multiple-model adaptive estimation: a bank of frozen models, each with its own
  filter, weighted by accumulated predictive likelihood on the shared stream; IMM lets
  the true model switch. **Overlap:** the roster {π_θ} with posterior over Θ is
  literally an MMAE bank; the observer machine's joint posterior μ ∈ Δ(S×Θ) is
  IMM-shaped. **Missing:** models differ in *dynamics*, not in initialization of a
  shared law; no channel split; convergence of the model posterior is a nuisance to
  speed up, never read as a boundary.
- Residual generation / observer-based FDI (Beard–Jones fault detection filters and
  descendants; surveyed in Basseville–Nikiforov): an observer = a reader; the residual
  = its prediction error. Same overlaps and gaps as above. **[standard, not re-verified]**

### 1.3 Active probing
- **Feldbaum, A. A. (1960–61), "Dual control theory I–IV," *Automation and Remote
  Control* 21–22.** Control has dual roles: caution (exploit) and probing (inject
  signals to identify the system). **Overlap:** the *injection* protocol is a probing
  action in Feldbaum's exact sense; the program's "level-up" question (when does a
  trained agent internally represent the value of probing?) is dual control turned
  inward. **Missing:** the identification target is the plant's parameters, never the
  probe-author's own coordinates; no court split.
- **Campbell, S. L. & Nikoukhah, R. (2004), *Auxiliary Signal Design for Failure
  Detection*, Princeton University Press** (with Nikoukhah's 1998 *Automatica* work on
  guaranteed active failure detection). Design the smallest input guaranteeing that two
  candidate models' output sets separate — active model discrimination. **Overlap:**
  choosing the perturbation/injection to maximize court discrimination is exactly their
  problem; a useful import if we ever optimize probe design. **Missing:** offline
  input design for plants; no self-reference.

### 1.4 Closed-loop identification — the leak, already classical
- **Gustavsson, I., Ljung, L. & Söderström, T. (1977), "Identification of processes in
  closed loop — identifiability and accuracy aspects," *Automatica* 13(1):59–75.**
  Identifying a plant while a feedback controller is running is a known hard problem:
  the loop correlates input with noise and can make plant and controller parameters
  jointly unidentifiable; external excitation or controller switching restores
  identifiability. **Overlap — a real anticipation:** this is the program's *leak*
  (consequence c_w writes identity into the world-state; the world court hears identity
  cases at rate ∝ c_w²(β₂−β₁)²) stated as an identifiability obstruction, 50 years ago;
  their cure (external excitation) is our injection. **Missing:** they resolve the
  confound to estimate the plant; they never turn the confound itself into the object
  of study (factored regime ⇔ c_w = 0), and there is no per-channel evidence split —
  identifiability is analyzed jointly, not read as two courts with signs.

---

## 2. Statistics / probability / econ

This field owns the mathematics of C1, C5 (as merging rates), and C6.

- **Kakutani, S. (1948), "On equivalence of infinite product measures," *Annals of
  Mathematics* 49:214–224.** Product measures are equivalent or mutually singular —
  no middle ground. **Overlap:** C6's mathematical skeleton: "self = mutually singular
  initializations" is the singular branch; the program's open lemma L2 (adapted
  dichotomy via predictable Hellinger processes) is the semimartingale generalization,
  whose machinery exists in **Jacod, J. & Shiryaev, A. N., *Limit Theorems for
  Stochastic Processes*, Springer (1987/2003), Ch. IV** (Hellinger processes;
  predictable criteria for absolute continuity/singularity). **Missing:** pure measure
  theory; no agent, no channels, no interpretation of the singular branch as identity.
- **Blackwell, D. & Dubins, L. (1962), "Merging of Opinions with Increasing
  Information," *Annals of Mathematical Statistics* 33(3):882–886.** If P ≪ Q, their
  predictive conditionals merge uniformly. **Overlap:** the world court's healing is
  merging; never-renormalizing directions are exactly failures of the absolute-continuity
  premise. **Missing:** qualitative (no rate, no τ), single stream type, no
  interventions — merging is fate, not a measurement protocol.
- **Ville's inequality / test martingales → Ramdas, A., Grünwald, P., Vovk, V. &
  Shafer, G. (2023), "Game-Theoretic Statistics and Safe Anytime-Valid Inference,"
  *Statistical Science* 38(4):576–601.** (Ville, J. (1939), *Étude critique de la
  notion de collectif*, Gauthier-Villars **[standard, not re-verified]**.)
  **Overlap:** the court-correct barrier theorem — the false account's exponential is a
  nonnegative martingale with e^(−b) wrong-way tails — is their e-process machinery
  verbatim; anytime-validity is why the courts can be read at any stopping time.
  **Missing:** used for hypothesis testing; nobody reads a *pair* of e-processes split
  by channel as a boundary classifier.
- **Dawid, A. P. (1984), "Present Position and Potential Developments: … The
  Prequential Approach," *J. Royal Statistical Society A* 147(2):278–290.** Score
  models solely by their sequential forecasts of the realized stream. **Overlap:** the
  methodological charter for judging R⁺/R⁻ prequentially. **Missing:** everything else.
- **Perdomo, J., Zrnic, T., Mendler-Dünner, C. & Hardt, M. (2020), "Performative
  Prediction," *ICML*, PMLR 119:7599–7609.** Predictions that alter the distribution
  they predict; equilibrium notion "performative stability" — calibrated against the
  outcomes manifesting from acting on the prediction. **Overlap:** *ratification* (own
  actions as evidence pushing interior identity to a vertex; landing law shifted by
  exactly δ) is a stochastic-process version of performative stability; the
  self-fulfilling character of identity coordinates is their fixed-point phenomenon.
  **Missing:** one-shot risk minimization framing; no internal coordinates, no courts,
  no time constant.

---

## 3. Neuroscience / psychology

This field owns C3 and the *semantics* of C2 (which channel's mismatch means what),
with ~75 years of priority, but almost none of the formal machinery.

- **von Holst, E. & Mittelstaedt, H. (1950), "Das Reafferenzprinzip,"
  *Naturwissenschaften* 37:464–476** and **Sperry, R. W. (1950), "Neural basis of the
  spontaneous optokinetic response produced by visual inversion," *J. Comparative and
  Physiological Psychology* 43:482–489.** Efference copy / corollary discharge:
  a private copy of the motor command lets the system cancel self-generated
  (reafferent) input and treat only exafferent input as world-evidence. **Overlap:**
  C3 exactly — the efference tag is an efference copy, and "exclude self-authored acts
  from evidence" is the reafference principle. **Missing:** subtractive cancellation of
  a *known* self-signal, not hypothesis-writing for a likelihood court; no dual
  readers; no τ.
- **Bell, C. C. (1981), "An efference copy which is modified by reafferent input,"
  *Science* 214.** Mormyrid electric fish: anti-Hebbian plasticity builds a "negative
  image" of the predicted consequence of the fish's own discharge; after a perturbation
  the negative image *re-adapts with a measurable time course*. **Overlap:** the
  closest biological system with an actual measured healing time of a self-prediction
  after perturbation. **Missing:** the time constant is synaptic adaptation, not an
  evidence-accumulation τ; no two-model comparison.
- **Wolpert, D. M., Ghahramani, Z. & Jordan, M. I. (1995), "An Internal Model for
  Sensorimotor Integration," *Science* 269:1880–1882**; **Blakemore, S.-J., Wolpert,
  D. & Frith, C. (1998), "Central cancellation of self-produced tickle sensation,"
  *Nature Neuroscience* 1:635–640**; Frith's comparator account of delusions of
  control (Frith 1992, *The Cognitive Neuropsychology of Schizophrenia*, Erlbaum
  **[standard, not re-verified]**). Forward model predicts sensory consequences;
  match ⇒ self-attribution + attenuation; comparator failure ⇒ passivity experiences
  ("my actions are not mine"). **Overlap:** the comparator is a one-reader,
  subtractive version of the world court; delusions of control are the phenomenology
  of a miscalibrated identity court — the program's (S) sign table is a formalization
  of what the comparator literature gestures at. **Missing:** no accumulation, no
  explicit alternative model (a comparator has a prediction and a residual, not two
  rival laws), binary attribution instead of the (τ, sign, sign) triple.
- **Georgieff, N. & Jeannerod, M. (1998), "Beyond consciousness of external reality:
  a 'who' system for consciousness of action and self-consciousness," *Consciousness
  and Cognition* 7(3):465–477.** Names the dedicated attribution system; shared motor
  representations make agency attribution a genuine inference problem. **Overlap:**
  the identity court is a formal "who" system. **Missing:** no mechanism.
- **Synofzik, M., Vosgerau, G. & Newen, A. (2008), "Beyond the comparator model: a
  multifactorial two-step account of agency," *Consciousness and Cognition*
  17:219–239** and **Legaspi, R. & Toyoizumi, T. (2019), "A Bayesian psychophysics
  model of sense of agency," *Nature Communications* 10.** The modern replacements:
  agency = precision-weighted cue combination / Bayesian causal inference over "I
  caused it." **Overlap:** the most formal neuro models of self/other attribution;
  Bayesian, graded. **Missing:** trial-level cue fusion, not stream-level accumulated
  LR; no channel split; no healing time; no interventional contrast.
- **Watson, J. S. (1972), "Smiling, cooing, and 'the game'," *Merrill-Palmer
  Quarterly* 18:323–339**; **Bahrick, L. & Watson, J. S. (1985), "Detection of
  intermodal proprioceptive–visual contingency as a potential basis of self-perception
  in infancy," *Developmental Psychology* 21 [partially verified — title and year
  confirmed, venue standard]**; **Gergely, G. & Watson, J. S. (1996), "The social
  biofeedback theory of parental affect-mirroring," *Int. J. Psycho-Analysis* 77.**
  Infants detect response–stimulus contingency (kick → mobile moves), prefer and
  exploit it; perfect contingency reads as self, and around 3 months an innate
  contingency-detection module's target is *re-set* from perfect toward high-but-
  imperfect contingency (toward social others). **Overlap:** contingency magnitude is
  the consequence parameter c_w; self-detection by acting and measuring response
  statistics is the injection protocol run by infants; the 3-month re-set is a
  developmental re-tuning of the self/world boundary. Rovee-Collier's mobile paradigm
  is the standard experimental vehicle. **Missing:** contingency indices, not
  likelihood-ratio courts; no τ; no notion that the boundary is *defined* by healing.
- **Botvinick, M. & Cohen, J. (1998), "Rubber hands 'feel' touch that eyes see,"
  *Nature* 391:756** and **Wegner, D. M. & Wheatley, T. (1999), "Apparent mental
  causation," *American Psychologist* 54:480–492.** Body boundary and authorship are
  inferred, plastic, and foolable by correlation (priority/consistency/exclusivity).
  **Overlap:** evidence that biological self/world typing is a statistical inference
  with characteristic failure modes — the program's "false took" and "wrong-way
  plateau" regimes have behavioral analogues. **Missing:** no formal machinery.
- **Held, R. & Hein, A. (1963), "Movement-produced stimulation in the development of
  visually guided behavior," *J. Comparative and Physiological Psychology* 56:872–876.**
  Kitten carousel: active and passive kitten receive the SAME visual stream; only the
  active one (whose stream is tagged by its own motor commands) develops normal
  visually guided behavior. **Overlap:** the cleanest classical do-vs-injection (C4)
  precedent — identical realized stream, the only difference being whether the acts
  were self-authored; the outcome shows the efference tag is load-bearing.
  **Missing:** developmental outcome measure, not an online statistic.

---

## 4. Developmental / robotic self-modeling

The field that has actually *built* C1+C3 devices for self/other classification.

- **Gold, K. & Scassellati, B. (2009), "Using probabilistic reasoning over time to
  self-recognize," *Robotics and Autonomous Systems* 57(5):384–392** (earlier
  "A Bayesian Robot That Distinguishes 'Self' from 'Other'," CogSci 2007). A humanoid
  holds three generative models per visual object — "self" (motion follows my motor
  commands), "animate other," "inanimate" — and accumulates each model's likelihood of
  the observed motion stream, given its own timestamped motor record, to classify
  mirror image vs experimenter. **Overlap — the closest single ancestor found:**
  multiple frozen models scored by accumulated likelihood on a shared realized stream,
  with an efference record deciding what counts as evidence for "self." C1+C3 in
  running hardware. **Missing:** classifies *external percepts* (pixels), not internal
  coordinates; evidence is a single (observation) channel — there is no action court
  because the robot's own actions are input to the hypotheses, not scored tokens; no
  perturbation, no τ, no dichotomy structure; winner-take-all posterior, not (τ, sign,
  sign).
- **Stoytchev, A. (2011), "Self-detection in robots: a method based on detecting
  temporal contingencies," *Robotica* 29:1–21.** Learns its "perfect contingency"
  delay distribution between motor command and visual movement, then types visual
  features as self/other by probabilistic timing match. **Overlap:** Watson's
  contingency criterion made algorithmic; efference timestamps as tags. **Missing:**
  same as Gold–Scassellati; timing statistics rather than model-pair LR.
- **Bongard, J., Zykov, V. & Lipson, H. (2006), "Resilient Machines Through Continuous
  Self-Modeling," *Science* 314:1118–1121.** Robot actively probes (chooses maximally
  informative actions) to infer its own morphology, detects damage as model
  divergence, re-plans. **Overlap:** active self-identification (Feldbaum probing
  aimed at the self-model); damage detection = perturbation detection on own body
  coordinates. **Missing:** model-space search, not likelihood courts; the self/world
  boundary is assumed (the body is given as the modeling target), not discovered.
- **Lanillos, P. et al. (2020), "Robot self/other distinction: active inference meets
  neural networks learning in a mirror," *ECAI 2020* (IOS Press).** Self-recognition
  via accumulated prediction-error/free-energy evidence that observed motion matches
  self-generated predictions; works in mirrors, robot–robot, robot–human. **Overlap:**
  prediction-error accumulation as self-evidence; NN-learned forward models.
  **Missing:** single-model prediction error (no rival reader), external percepts,
  no τ/sign structure.

---

## 5. Theoretical biology / active inference / individuality

The field that owns the *question* (where is the organism/world boundary, statistically?)
— and answers it observationally rather than interventionally.

- **Friston, K. (2013), "Life as we know it," *J. Royal Society Interface*
  10(86):20130475** (and Kirchhoff et al. 2018, "The Markov blankets of life," same
  journal). Self/world boundary = Markov blanket: a conditional-independence structure
  separating internal from external states. **Overlap:** the same target concept —
  a *statistical* self/world boundary for a coupled system. **Missing:** the blanket is
  a static factorization property of a stationary density; no perturbations, no
  streams, no time constant, no channels. It types *states* by graph position, not
  *directions* by dynamical fate.
- **Bruineberg, J., Dołęga, K., Dewhurst, J. & Baltieri, M. (2022), "The Emperor's New
  Markov Blankets," *Behavioral and Brain Sciences* 45.** The critique: Pearl-blanket
  instrumentalism vs Friston-blanket realism; blanket placement underdetermines agent
  boundaries. **Relevance:** the two-courts construction is a direct constructive
  answer to this underdetermination — replace the observational independence criterion
  with an interventional, per-channel, time-resolved one. Worth citing as the gap
  statement.
- **Krakauer, D., Bertschinger, N., Olbrich, E., Flack, J. C. & Ay, N. (2020), "The
  information theory of individuality," *Theory in Biosciences* 139:209–223.**
  Individuality = degree to which a subsystem propagates information from its own past
  into its own future (vs borrowing it from the environment); graded, with organismal /
  colonial / driven forms. **Overlap:** self/world by information flow through time;
  graded typing of coordinates. **Missing:** mutual-information decompositions on the
  *observational* process; no interventions, no likelihood courts, no healing time.
- **Maturana, H. & Varela, F. (1980), *Autopoiesis and Cognition: The Realization of
  the Living*, D. Reidel** and **Di Paolo, E. (2005), "Autopoiesis, adaptivity,
  teleology, agency," *Phenomenology and the Cognitive Sciences* 4 [partially verified
  — concept confirmed via secondary sources]**. Identity = a precarious, operationally
  closed process network that actively maintains itself; Di Paolo's viability set =
  the perturbations an identity can absorb without disintegration. **Overlap:**
  qualitative ancestor of C5/C6 — identity defined by response to perturbation, and
  "self = what does not renormalize back to the environment's statistics" is an
  evidential formalization of precarious closure. **Missing:** no probability at all.
- **Ashby, W. R. (1952), *Design for a Brain*, Chapman & Hall.** Essential variables
  and ultrastability: a two-timescale architecture where perturbations that ordinary
  regulation cannot heal trigger parameter re-organization. **Overlap:** the
  perturb-and-watch-the-return methodology; the distinction between disturbances the
  system absorbs (finite τ) and those that force identity change (never heals —
  step-change of parameters). **Missing:** dynamical, not evidential.

---

## 6. Physics / filtering mathematics — the two strongest formal precedents

### 6.1 Damage spreading: the reader pair, in statistical mechanics, since 1969
- **Kauffman, S. (1969), "Metabolic stability and epigenesis in randomly constructed
  genetic nets," *J. Theoretical Biology* 22 [partially verified — origin attribution
  confirmed via secondary sources]**; **Derrida, B. & Weisbuch, G. (1987)** on damage
  spreading in Ising/Kauffman systems **[venue not individually verified; the
  1987 attribution is confirmed in the damage-spreading literature]**; modern review
  material confirms the protocol. The protocol: make two replicas of a system,
  evolve both under the SAME thermal noise / update sequence, flip one spin in one
  replica at t₀, and watch whether the Hamming distance ("damage") heals to zero or
  spreads — the healing/spreading transition defines dynamical phases not visible in
  equilibrium quantities. **Overlap — structurally the reader protocol:** two copies
  of one law, shared realization, point perturbation, fate-of-the-difference as the
  measured object, and a *phase distinction* drawn from whether perturbations heal.
  This is C1's skeleton + C5 as an order parameter, two decades before filter
  stability. **Missing:** distance is Hamming, not log-likelihood (no evidential
  reading, no signs); no channels, no agent, no self-interpretation; the perturbed
  copy is the *system*, not a *model of* the system.
- **Gorin, T., Prosen, T., Seligman, T. H. & Žnidarič, M. (2006), "Dynamics of
  Loschmidt echoes and fidelity decay," *Physics Reports* 435:33–156.** Quantum/classical
  fidelity: overlap decay between two evolutions differing by a small perturbation;
  a taxonomy of decay regimes (analogous in spirit to the shape zoo). **Overlap:**
  perturbation-sensitivity with a regime classification. **Missing:** perturbed
  *dynamics* rather than perturbed *initialization on shared data*; no inference.

### 6.2 Filter stability: the world court's healing theorem already exists
- **Ocone, D. & Pardoux, E. (1996), "Asymptotic stability of the optimal filter with
  respect to its initial condition," *SIAM J. Control and Optimization* 34:226–243**
  and **van Handel, R. (2009), "Observability and nonlinear filtering," *Probability
  Theory and Related Fields* [volume/pages not individually verified; paper confirmed]**
  (also Chigansky & Liptser on exponential stability **[standard, not re-verified]**).
  Two copies of the SAME optimal filter, initialized differently, run on the SAME
  observation stream: under conditions they merge, and the literature's two known
  sufficient mechanisms are (i) **ergodicity of the signal** — the chain forgets its
  past, so the wrong prior stops mattering even with *uninformative observations* —
  and (ii) **observability/informative observations** — the data overwhelm the prior.
  **Overlap — the deepest formal precedent for the η side:** this is exactly the world
  court's healing, including the program's exact binary-family discovery that healing
  has two compounding routes (forgetting vs evidence) with τ set by the faster route
  and "healing with zero evidence" at q = ½: that is precisely mechanism (i) vs (ii).
  The open lemma L1 (uniform filter stability for the *controlled* closed-loop chain,
  where the policy can refuse to mix actions) sits exactly at this literature's known
  frontier — filter stability under degenerate/controlled observation processes.
  **Missing:** filter stability proves *whether/how fast* merging happens; it never
  splits evidence by channel, never attaches a sign (whose account was true), never
  reads *failure* of stability as identity, and has no efference/agency structure.

---

## 7. Machine learning / RL / interpretability

Owns C7, plus verbatim fragments of the courts.

- **Precup, D., Sutton, R. S. & Singh, S. (2000), "Eligibility Traces for Off-Policy
  Policy Evaluation," *ICML*, pp. 759–766.** Off-policy importance weights:
  Π_t π'(a_t|h_t)/π(a_t|h_t) over the realized trajectory — the action-court
  likelihood ratio, token for token. **Overlap:** the a-court statistic exists,
  ubiquitously, as an estimator correction. **Missing:** it is a *reweighting device*
  whose variance is a nuisance; nobody reads its drift/plateau/sign as a verdict about
  identity, and there is no observation court (the environment law is shared by
  construction).
- **Ramachandran, D. & Amir, E. (2007), "Bayesian Inverse Reinforcement Learning,"
  *IJCAI*, pp. 2586–2591.** Posterior over reward/policy parameters from action
  likelihoods given states. **Overlap:** the identity court as inference — action
  tokens carry likelihoods only in Θ. **Missing:** open-loop expert data; no world
  court, no perturbation, no τ.
- **Kenton, Z., Kumar, R., Farquhar, S., Richens, J., MacDermott, M. & Everitt, T.
  (2023), "Discovering Agents," *Artificial Intelligence* 322:103963.** First causal
  definition of agents: systems whose policy *would adapt* under intervention on how
  their actions influence the world; discovery algorithm over mechanised causal
  graphs. **Overlap:** interventional (not observational) drawing of the
  agent/non-agent boundary; philosophically the nearest ML relative of "self is what
  the do-operation reveals." **Missing:** graph-level and mechanism-level; no streams,
  no likelihood courts, no τ, no internal coordinates.
- **Demski, A. & Garrabrant, S. (2019), "Embedded Agency," arXiv:1902.09469.**
  Problem statement: agents embedded in their environment have no given self/world
  boundary. **Overlap:** the program is an answer to one of its named sub-problems.
- **Eberhardt, F., Glymour, C. & Scheines, R. (2005, UAI), "On the number of
  experiments sufficient…"** and **Tong, S. & Koller, D. (2001, IJCAI), "Active
  learning for structure in Bayesian networks."** Interventions as the discriminator
  between causal hypotheses; experiment selection. **Overlap:** the do-vs-observe
  logic underlying C4. **Missing:** structure learning over external variables.
- **Interpretability of NN internals (C7):**
  - **McGrath, T., Rahtz, M., Kramár, J., Mikulik, V. & Legg, S. (2023), "The Hydra
    Effect: Emergent Self-repair in Language Model Computations," arXiv:2307.15771.**
    Ablate an attention layer; downstream layers compensate. **Overlap:** *healing of
    internal perturbations* observed in transformers (within a forward pass across
    layers — depth-τ rather than time-τ). **Missing:** no formal measurement of a
    healing time, no likelihood courts, no self/world reading.
  - **Lindsey, J. (2025), "Emergent Introspective Awareness in Large Language Models,"
    Transformer Circuits (transformer-circuits.pub/2025/introspection), Anthropic.**
    Concept *injection* (activation steering = a do on internal coordinates) with the
    model asked whether it notices; ~20% detection at 0% false positives for
    Opus 4/4.1. Follow-ups confirmed and extended: "Latent Introspection: Models Can
    Detect Prior Concept Injections" (arXiv:2602.20031), "Mechanisms of Introspective
    Awareness" (Macar, Yang, Wang et al., arXiv:2603.21396), "Steering Awareness:
    Models Can Be Trained to Detect Activation Steering" (arXiv:2511.21399).
    **Overlap:** perturbations of internal coordinates with detection-of-authorship as
    the question — the do side of C4 on C7's substrate, and the self-report analogue
    of the identity court. **Missing:** verdict by *self-report*, not by a frozen
    reader pair's likelihood ratio; no observation court; no τ; no boundary criterion.
  - **Binder, F. J. et al. (2024), "Looking Inward: Language Models Can Learn About
    Themselves by Introspection," arXiv:2410.13787.** Self-prediction advantage over an
    equally-informed other-model — a privileged-access instrument (cf. the doppel
    design). **Panickssery, A., Bowman, S. R. & Feng, S. (2024), "LLM Evaluators
    Recognize and Favor Their Own Generations," *NeurIPS 2024*.** Self-recognition of
    own outputs. **Overlap:** self/other distinction measured in LLMs. **Missing:**
    behavioral, not coordinate-level; no courts.
  - **Chen, R., Arditi, A., Sleight, H., Evans, O. & Lindsey, J. (2025), "Persona
    Vectors: Monitoring and Controlling Character Traits in Language Models,"
    arXiv:2507.21509.** Identity-like coordinates (traits) as activation directions,
    monitored and steered. **Overlap:** establishes that z-type (identity) coordinates
    exist as directions in the residual stream — the natural first targets for the
    measurement protocol. **Missing:** no principled criterion for *why* these are
    identity rather than belief; that is exactly what (τ, sign Λᵃ, sign Λˣ) supplies.
- **O'Regan, J. K. & Noë, A. (2001), "A sensorimotor account of vision and visual
  consciousness," *Behavioral and Brain Sciences* 24:939–973.** Perception constituted
  by mastery of action-contingent input laws. **Overlap:** the general claim that
  what-you-are is written in action-conditioned statistics of the loop. **Missing:**
  no formalism relevant here.

---

## 8. Verdict

### (i) Which components are known, and where

| component | status | where |
|---|---|---|
| C1 two-reader LR on shared stream | **fully known**, deep | Wald 1945; Willsky–Jones 1976; Magill 1965/IMM 1988; Dawid 1984; Blackwell–Dubins 1962; e-processes (Ramdas et al. 2023); damage spreading (shared-noise replicas) |
| C2 court split by token type | **fragments only** | a-court alone: off-policy importance weights (Precup et al. 2000), Bayesian IRL (2007); x-court alone: filtering/FDI; the a/x confound named: closed-loop identifiability (Gustavsson et al. 1977). **No one runs both as a paired diagnostic and reads the sign pair.** |
| C3 efference tag | **known, old** | von Holst–Mittelstaedt 1950; Sperry 1950; Bell 1981; implemented: Gold–Scassellati 2009, Stoytchev 2011 |
| C4 do vs injection contrast | **known in analogue form** | Held–Hein 1963 (active/passive, same stream); passive-movement controls throughout the comparator literature; in NNs, steering (do) and prompting (injection) both exist but are never paired to isolate penetration |
| C5 τ as boundary-defining quantity | **exists as a quantity, never as a definition** | filter stability rates (Ocone–Pardoux 1996; van Handel 2009) — including the exact two-routes structure (forgetting vs evidence); damage-spreading transition; Loschmidt echo regimes; qualitatively: autopoiesis/viability (Di Paolo), ultrastability (Ashby) |
| C6 self = never-renormalizing / mutually singular | **mathematics known, interpretation new** | Kakutani 1948; Blackwell–Dubins absolute-continuity premise; Jacod–Shiryaev Hellinger machinery (for L2). No literature calls the singular branch "identity." |
| C7 internal NN coordinates as target | **substrate known** | activation patching/steering; persona vectors (Chen et al. 2025); concept injection + introspection (Lindsey 2025); Hydra effect healing (McGrath et al. 2023) |

### (ii) Does the specific fusion appear anywhere?

**No.** Searches targeted at every pairwise and triple combination (per-channel LR
split; healing-time-as-boundary; NN internals) found no instance of:

- a **per-channel court split** of accumulated model-pair evidence used as a
  *classifier of the perturbed coordinate* (action court = did it become me; world
  court = did the world object) — the split exists only as a confound to be engineered
  away (closed-loop ID) or as a single-channel estimator (importance sampling);
- **renormalization time as the *definition* of the self/world boundary** — τ exists
  as a stability rate (filtering) and an order parameter (damage spreading), but always
  as a property to be established or a phase to be mapped, never as the typing
  criterion for an agent's own coordinates;
- **any of this applied to internal coordinates of neural networks** — the nearest
  neighbors (concept injection + introspection; Hydra effect) use self-report or raw
  activation distance, not frozen-reader likelihood courts, and none extracts (τ,
  sign Λᵃ, sign Λˣ).

Nearest single works, ranked: **(1) Gold & Scassellati 2009** (accumulated
multi-model likelihood + efference record → self/other, in a real robot; missing
courts, τ, internality); **(2) the damage-spreading protocol** (two replicas, shared
noise, point perturbation, heal-or-spread as phase diagnostic; missing evidence
reading and agency); **(3) filter stability theory** (two copies of the agent's own
filter on the shared stream with merging rates — the world court's healing theorem,
including the forgetting-vs-evidence dichotomy; missing the identity side entirely).
The comparator-model tradition supplies the *semantics* the formal precedents lack,
and none of the three families cites the other two — the fusion is genuinely open.

### (iii) Load-bearing citations (all verified in search results unless marked)

1. **Wald, A. (1945).** "Sequential Tests of Statistical Hypotheses." *Ann. Math.
   Statist.* 16(2):117–186. — (τ*) is Wald's sample-size law.
2. **Kakutani, S. (1948).** "On equivalence of infinite product measures." *Ann.
   Math.* 49:214–224. — the dichotomy behind "self = mutually singular."
3. **Blackwell, D. & Dubins, L. (1962).** "Merging of Opinions with Increasing
   Information." *Ann. Math. Statist.* 33(3):882–886. — healing = merging; self =
   its failure.
4. **von Holst, E. & Mittelstaedt, H. (1950).** "Das Reafferenzprinzip."
   *Naturwissenschaften* 37:464–476. — the efference tag.
5. **Willsky, A. S. & Jones, H. L. (1976).** "A generalized likelihood ratio approach
   to the detection and estimation of jumps in linear systems." *IEEE Trans. Autom.
   Control* 21:108–112. — perturbation-hypothesis LR monitoring on a shared stream.
6. **Gustavsson, I., Ljung, L. & Söderström, T. (1977).** "Identification of processes
   in closed loop." *Automatica* 13(1):59–75. — the leak as classical identifiability
   failure; injection as its classical cure.
7. **Ocone, D. & Pardoux, E. (1996).** "Asymptotic stability of the optimal filter
   with respect to its initial condition." *SIAM J. Control Optim.* 34:226–243 — with
   **van Handel, R. (2009),** "Observability and nonlinear filtering," *Probab. Theory
   Relat. Fields*. — the world court's healing, two routes and all.
8. **Gold, K. & Scassellati, B. (2009).** "Using probabilistic reasoning over time to
   self-recognize." *Robotics and Autonomous Systems* 57(5):384–392. — closest
   end-to-end ancestor.
9. **Friston, K. (2013).** "Life as we know it." *J. R. Soc. Interface* 10(86):20130475
   — with **Bruineberg, J. et al. (2022),** "The Emperor's New Markov Blankets," *Behav.
   Brain Sci.* — the boundary question the courts answer interventionally.
10. **Ramdas, A., Grünwald, P., Vovk, V. & Shafer, G. (2023).** "Game-Theoretic
    Statistics and Safe Anytime-Valid Inference." *Statist. Sci.* 38(4):576–601. —
    the anytime-valid machinery of the court-correct barriers.
11. **Lindsey, J. (2025).** "Emergent Introspective Awareness in Large Language
    Models." Transformer Circuits, Anthropic. — the do-side of C4 on C7's substrate;
    the self-report baseline the courts replace.

Not individually re-verified by search (standard references used for context only):
Ville 1939; Frith 1992; Chigansky–Liptser; Beard–Jones FDI filters; exact venues of
Kauffman 1969 / Derrida–Weisbuch 1987 / Di Paolo 2005 (existence and attribution
confirmed via secondary sources in the search results).

---

## 9. Imports worth stealing (beyond citation)

- **Filter stability's two mechanisms** (signal ergodicity vs observability) formalize
  the binary family's "two routes compound; τ set by the faster, sign earned only by
  evidence" — cite it as the general statement behind τ_η, and use its *controlled-
  process* frontier to state L1 precisely.
- **Auxiliary-signal design** (Campbell–Nikoukhah) is the ready-made theory for
  optimizing the injection when court discrimination is expensive (the costly-probes
  environmental dial).
- **Damage spreading** supplies vocabulary for the network experiment: the τ/no-τ
  distinction as a *dynamical phase* of the trained system, invisible to equilibrium
  (behavioral) statistics — and a warning that τ can depend on the update rule
  (channel), which is the sign-flip figure's F(0) > 0 > F(q) in older clothes.
- **Overshoot corrections** from sequential analysis (already noted in MASTER §5.6)
  and **e-process confidence sequences** for reporting τ with anytime-valid bands.
- **Bruineberg et al.'s critique** is the cleanest published statement of the gap the
  program fills; positioning against it will do more work than positioning against
  the comparator literature.

---

# ADDENDUM — second sweep, 2026-08-13 (angles the first sweep's vocabulary couldn't reach)

*Trigger: the first sweep missed Abel et al.'s plasticity/empowerment paper. This
pass searched with vocabulary acquired since: monitoring games, directed
information, inverse dynamics, record corruption, action-unawareness. New
antecedents below, with differentiation verdicts. Collision status re-checked.*

## A. The missed dual, and its neighborhood

- **Abel, Bowling, Barreto, Dabney, et al., "Plasticity as the Mirror of
  Empowerment," arXiv:2505.10361 (2025).** Generalized directed information;
  empowerment (a→x capacity) and plasticity (x→a capacity) as mirror duals;
  tension theorem 𝔈+𝔓 ≤ interface budget. Same two channels our courts keep
  ledgers on, at capacity level: variational/ex-ante, no interventions, no
  internal state, no clock, no sign. Cite beside klyubin2005/salge2014.
- **Csaky, "Prediction and Empowerment: A Theory of Agency through Bridge
  Interfaces," arXiv:2605.06346 (2026).** Separates prediction/compression/
  empowerment; names "overwrite control" (making the future action-determined)
  vs task-relevant controllability — the formal name for our control-shortcut
  (clamping the echo world instead of learning it). Adjacent program, moving.
- **Csaky, "Artificial Agency Program," arXiv:2602.24100 (2026).** Curiosity/
  compression/empowerment agenda with budgeted observation-action-deliberation.
  Neighborhood-watch item, not an overlap.
- **Jaques et al., "Social Influence as Intrinsic Motivation," ICML 2019
  (arXiv:1810.08647).** Rewarding counterfactual influence of own actions on
  OTHER agents (≈ MI(action; others' behavior)). Consequence-as-reward for the
  other-directed field; our κ is the self-directed, represented version.
- **Jagadeesan, Hardt, Mendler-Dünner, "Performative Power," arXiv:2203.17232.**
  Measuring a platform's ability to steer participants = κ as a population-level
  estimand, from observational data. The performative line's quantitative arm
  (perdomo2020 already cited in proposal).

## B. Directed information — the annealed courts

- **Massey (1990); Massey & Massey (2005) conservation law.** MI between the
  halves of a dialogue = sum of the two one-way directed flows. The undirected/
  directed split behind the courts.
- **Permuter, Kim, Weissman, "Interpretations of Directed Information...,"
  IEEE-IT 57(6):3248–3259 (2011; arXiv:0912.4872).** DI = value of causal side
  information in horse-race gambling (Kelly increment) AND best error exponent
  for testing "does Y causally influence X." The expectation-level version of
  our two-reader court ledger (grant/deny a reader the source stream); our
  construction is its path-level, intervention-split, court-split refinement.
  Load-bearing citation for §4's eventual related-work.

## C. Games where the record is the battlefield

- **Aumann & Maschler (with Stearns), *Repeated Games with Incomplete
  Information* (1966–68/1995).** The informed player's actions leak type; optimal
  play = deliberate partial revelation (cav u). The identity court + exposure
  management, as game theory, sixty years early. Their uninformed player IS a
  Bayesian identity court.
- **Imperfect public monitoring (Green-Porter, APS; folk theorems).** Public
  noisy signal of actions; players privately know their OWN actions — the econ
  literature *assumes* variant-B efference natively. Our ρ is their monitoring
  precision, ours dialed on the self-channel.
- **Kandori & Obara, "Less is More: an observability paradox," J. Game Theory
  34:475–493 (2006).** Equilibrium payoffs can EXPAND as the public signal gets
  LESS sensitive to actions — the value of record corruption, in equilibrium
  terms. Game-theoretic cousin of "privacy has value under adversarial
  pressure" (our v3 design premise).

## D. Probing to identify — κ's evidence channel, formal layer

- **Chernoff, "Sequential design of experiments," Ann. Math. Stat. 30:755–770
  (1959).** Active hypothesis testing: choose experiments (actions) to maximize
  discrimination — VOI-driven probing with optimal error exponents.
- **Naghshvar & Javidi, "Active sequential hypothesis testing," Ann. Stat.
  41(6):2703–2738 (2013); Nitinawarat-Atia-Veeravalli, IEEE-TAC 58(10) (2013)
  controlled sensing.** Modern form. (Dual control/Feldbaum already in §1.3;
  these add the error-exponent theory our Wald/(τ*) analysis matches.)

## E. Action-unawareness and reconstructing the actor

- **Torresan, Suzuki, Kanai, Baltieri, "Active inference for action-unaware
  agents," arXiv:2508.12027 (Aug 2025). CLOSEST SINGLE ANTECEDENT to the ρ-record
  A/B axis.** Agents WITHOUT efference copy must infer their own past actions
  from observations; compared against action-aware agents on navigation.
  Differentiation: no record-fidelity dial (all-or-nothing, our ρ interpolates
  and yields the (1−2r)² law), no privileged-information accounting (Π-rate,
  floors), no trained-network optimality comparison, no representation claims —
  behavioral active-inference framing. Must-cite; likely friendly reviewers.
- **Baker et al., VPT (2022) inverse dynamics; Schmidt & Jiang, "Learning to
  Act without Actions" / LAPO, ICLR 2024; Bruce et al., Genie (2024) latent
  actions.** The observer's reconstruction problem (recover actions from the
  public record) as an ML industry: IDMs are F1-machinery at scale; latent-action
  models learn action variables with no action labels at all.
- **POMDP actuation-noise channel (commanded vs executed action; situation-
  calculus noisy effectors, Bacchus-Halpern-Levesque lineage).** A THIRD
  corruption locus we hadn't separated: noise between agent and world (agent
  knows the command, world hears noise) vs our ρ (world hears truth, record
  lies). Worth one design cell eventually; "Intention Inference Under Execution
  Noise," arXiv:2608.02440 (2026) is fresh nearby work (aleatoric/epistemic
  splitting of the identity court's docket).

## F. The self in the record — LLM empirics (additions to §7)

- **Panickssery, Bowman, Feng, "LLM Evaluators Recognize and Favor Their Own
  Generations," NeurIPS 2024 (arXiv:2404.13076).** Self-recognition capability
  linearly predicts self-preference — authorship posterior exists and has
  consumers. | **"LLM Self-Recognition: Steering and Retrieving Activation
  Signatures," arXiv:2606.06315 (2026)** — a steerable self-authorship direction
  in activations (our "authorship posterior as internal variable," found in the
  wild). | **arXiv:2410.02064** (Llama3 self-recognition inspect/control).
  | **"Emergent Introspective Awareness in LLMs," arXiv:2601.01828 (2026).**
- **Filler-token computation / encoded reasoning:** Lanham et al. / Roger &
  Greenblatt, "Preventing Language Models from Hiding Their Reasoning,"
  arXiv:2310.18512; Pfau, Merrill, Bowman, "Let's Think Dot by Dot" (2024)
  [ID from memory — verify]. Empirical support for §3's vacuous-token
  hiding-place remark (computation stored on semantically empty tokens).
- **Scherr et al., "Self-Supervised Learning Through Efference Copies,"
  arXiv:2210.09224.** Efference as an ML training signal.

## G. Frames and siblings previously uncited

- **De se / self-locating belief** (Perry 1979, Lewis 1979; SEP "Self-Locating
  Beliefs"): the philosophy of indexical content — what κ's self-typing is a
  mechanization of.
- **Shumailov et al., model collapse (2023) + self-consuming loops (Alemohammad
  et al.)**: MASTER §8 treats these as siblings of the entropy-collapse
  mechanism; they were absent from this file — now on record.
- **Eysenbach & Salakhutdinov, "Robust Predictable Control," arXiv:2109.03214.**
  Control-as-compression: agents steer toward compressible states — the
  prediction-loss shortcut, as a feature.

## Verdict update (supersedes §8(ii) at the margin)

The fusion — interventional, path-level, court-split, clocked-and-signed
measurement of self-directions, with a record-fidelity dial and exact
agent/observer floors, asked of trained networks as a representation question —
remains unclaimed. Nearest neighbors after this pass, in order: Torresan et al.
2025 (the A/B axis, behavioral, no accounting); Abel et al. 2025 (the two
channels, capacity-level); Permuter-Kim-Weissman 2011 (the annealed ledger);
Kandori-Obara 2006 (the value of record noise, equilibrium-level). Simplex
remains passive-prediction-only (progress report checked, July 2025). The
active theory-of-agency cluster (Csaky; DeepMind agency program) is the one to
watch — they have the dials but not the instrument.

## H. The embedded self-prediction frame (added 2026-08-15; MUST-POSITION)

- **Meulemans, Nasser, ..., Hutter, Sacramento, Agüera y Arcas, Richards,
  "Embedded Universal Predictive Intelligence," arXiv:2511.22226 (202pp, pure
  theory; Google Paradigms of Intelligence).** Flagged by window 0; overview at
  `~/mind-capture/mupi-2511.22226-overview.md`. Embedded agents hold Bayesian
  mixtures over universes CONTAINING THEMSELVES; **they name ρ(a_t|æ_<t) — the
  action-marginal of the agent's own mixture — "the self-model"** (their §3,
  Remark 3.3: the embedded best response ignores it; it converges to the actual
  policy on-distribution; §3.4 agents use it). Free will = posterior uncertainty
  over one's own policy (§6.5) — our Π, read from inside. Own actions as
  evidence about similar agents (Occam coupling) = the identity court on a
  population prior. All AIXI-lineage: uncomputable, asymptotic, no experiments,
  no measurement theory.
- **Term collision, not a program collision.** Their self-model is the
  *predictive/evidential* object — in our vocabulary the identity-court
  forecaster (the observer machine's action-predictive, run reflexively; our
  proposal §2 already constructs it). Our founding argument (§4 opening) is that
  this object undertypes selfhood: on the trajectory distribution a modeled
  clock and a commitment are identical; self-modelhood needs interventional
  typing (τ, signs), consequence (κ), privilege accounting (Π, record fidelity),
  and the representation question. The proposal must now cite them and draw
  this line explicitly — they own the term at the predictive level.
- **Prop 4.29 (theirs) = κ-starvation as equilibrium (ours).** Any deterministic
  policy profile is a subjective embedded equilibrium under dogmatic beliefs:
  off-path beliefs are never tested by the very policy they justify. This is
  the equilibrium-theoretic form of gate-3 starvation — helplessness/absorbing
  miscalibration (κ̂ dogma never probed) — and it certifies from inside their
  own framework that predictive self-consistency admits pathological selves the
  courts would expose. Their open problem (exploration escaping dogmatic
  equilibria "without unsafe randomness") is exactly where our machinery plugs
  in: the three gates say when scheduled DETERMINISTIC variation suffices
  (low-dimensional shared consequence parameters, e.g. κ — metronome probes),
  and the derandomized-explorer analysis says evidence needs off-habit, not
  random.
