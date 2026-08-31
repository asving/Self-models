A. ADVERSARIAL

1. The “Recovery of the decomposition” claim is false under its stated assumptions. Let the environment contain a static bit \(W\), observed through i.i.d. flips with probability \(q<1/2\); its Bayes filter is informative and synchronizing. Independently define
   \[
   z_{t+1}=a_t,\qquad P(a_t=z_t\mid z_t)=0.9.
   \]
   Thus \(z_t\) is reconstructed entirely from the realized stream, satisfying (b). After a forced window, the first free action may distinguish the “own” and “external” readings, but both readings then consume that same post-window action and acquire the identical state \(z_{t+1}=a_t\). Consequently \(\Lambda_s\) contains at most one informative increment and converges to a finite value, not \(+\infty\).

2. The claim also fails for \(\eta\)-directions because no observability condition connects \(\eta\) to the policy. In the same informative, synchronizing environment, let \(\pi(a\mid\eta,z)\) be independent of \(\eta\). Every \(\eta\)-direction then has the same “action-shadow,” the injection does nothing, and \(D_s\equiv0\). The classification calls it settable rather than world-typed. A weaker counterexample uses a threshold policy and a perturbation that does not cross a threshold.

3. There is a direct tension between “healed by expression” and \(\Lambda_s\to+\infty\). If expression kills the discrepancy at the first post-window act, as (H) explicitly says, the own and external predictive laws should thereafter coincide. Only finitely many nonzero log-likelihood increments remain, so \(\Lambda_s\) normally plateaus. Divergence requires persistent positive information per step, an extra assumption almost opposite to one-step healing.

4. Assumption (b), that \(g\) “consumes” realized acts, does not imply adoption. A valid update can ignore its action argument, retain a private component, average actions slowly, store their parity, or treat them as observations rather than commands. Even “\(z\) is reconstructed from the stream” does not imply that one act overwrites \(z\), much less that all \(z\)-directions heal after one expression.

5. The recovery construction is circular about the decomposition it claims to recover. An “\(\eta\)-direction” or “\(z\)-direction” can be injected only after coordinates corresponding to \(\eta\) and \(z\) have already been identified in \(S\)'s model. The premise even says \((\eta',z')\) track the true \((\eta,z)\). Testing already separated coordinate classes does not recover the separation “with no access to referents.”

6. The classification is not invariant under reparameterization. Replace model coordinates by
   \[
   u=\eta+z,\qquad w=\eta-z.
   \]
   Generic \(u\)- or \(w\)-directions perturb both mechanisms and can heal partly through evidence and partly through expression. Nothing in (H) identifies a unique factorization, subspace, or nonlinear foliation corresponding to the original \((\eta,z)\).

7. The Screening-off lemma establishes, at most, an equality under a strong modular causal model: the environment’s transition law must depend on the agent solely through emitted actions, and intervening on \(\xi\) must not alter any shared noise, side channel, timing, or physical variable. Those assumptions are embedded in \(\Phi\), not established for the stated “pair of transducers.”

8. The right-hand side of the Screening-off equality is effectively a full policy intervention extending over all future times. That is much stronger than “purely” playing a finite sequence of actions. It requires sampling from the counterfactual policy adaptively after every future observation. The finite injection defined immediately afterward does not realize the lemma’s infinite internal counterfactual.

9. The lemma does not show that \(S\) can perform the intervention. \(S\) is initially said only to “contribute” to actions and to have the channel “partly” available, but the injection assumes it can replace the complete action law. For example, if the emitted action is the XOR of \(S\)'s proposal and a private controller bit, \(S\) contributes causally but cannot implement an arbitrary counterfactual action policy.

10. “Every internal counterfactual is realizable” confuses matching the environment’s marginal response with realizing the agent counterfactual. Action replacement can reproduce the distribution of \(x_{\ge t}\), but generally not the counterfactual internal trajectory, correlations between internal and external variables, or downstream quantities depending directly on the internals.

11. The assertion that an agent cannot “plant evidence” on the \(x\)-channel is too strong. In a controlled observation model, actions can determine sensor position, measurement resolution, queried variable, or even the observation itself. For \(x_t=a_t\), the agent literally determines the next observation. Calling this merely choosing “which world is being asked” does not support the later mathematical distinction between content and index.

12. The statement that an action’s likelihood ratio between environmental hypotheses is \(1\) requires actions to be conditionally independent of the environment given public history. That fails whenever the agent has private observations or hidden state correlated with the environment—precisely among the hiding places admitted in Section 3. Example: \(W\in\{0,1\}\), the agent privately observes \(W\), and emits \(a=W\); then the action has an infinite likelihood ratio between the two hypotheses.

13. Adding \(v\) to \((\eta',z')\) may be meaningless. A generic vector takes \(\eta'+v\) outside the probability simplex, and \(Z\) was introduced as a quotient rather than a vector space. No tangent space, chart, projection, perturbation magnitude, or admissible-direction condition is supplied.

14. The injection applies the same \(v\) to every rolled-forward state. That is not generally the action sequence generated by a one-time intervention \(do(\xi_{t_0}=\xi_{t_0}+v)\), because nonlinear dynamics transport the perturbation. Thus the injection does not in general simulate the counterfactual invoked by Screening-off.

15. \(P^{-}\) in (D) is underdefined in a way that changes the answer. If the frozen model is nevertheless filtered normally on the entire injected history and uses the same stream-reconstructed \(g\), then an exact model reconstructs the same post-injection state as the agent and \(D_s=0\). If forced actions are withheld from \(g\), \(P^{-}\) is actually the later-defined external reading. The text needs one of these incompatible meanings.

16. “Frozen” is internally ambiguous. The model is said not to update, yet it is “rolled forward” and conditioned on everything that happened. Freezing parameters, freezing the latent state, freezing the posterior, and withholding selected state updates produce different reference laws and different scores.

17. \(D_s\) is only a behavioral pseudometric on internal state. Any two states with the same current policy distribution have zero divergence even if they have different future dynamics. Thus \(D\equiv0\) need not mean “another perfectly typical life”; it can mean a policy-null direction, a delayed effect, a symmetry, or an unsuccessful injection.

18. Current-action KL can miss delayed deviations. For example, let \(z\) be a commitment scheduled to affect the action only ten steps later. Then \(D_s=0\) for the first nine steps despite a causally potent self-state difference. Classification based on its profile needs a horizon-aware predictive divergence, not just the next-action law.

19. The Self-ratification lemma is false for ordinary online refitting. Let the true agent emit i.i.d. \(\mathrm{Bernoulli}(0.9)\) actions and let the reference refit a Bernoulli parameter after each observation using the empirical frequency. It has “absorbed what the agent just did,” but
   \[
   D_{\mathrm{KL}}(\mathrm{Bern}(0.9)\Vert\mathrm{Bern}(\hat p_t))
   \]
   is generally nonzero at every finite time. Refitting at behavior’s rate does not imply exact interpolation of the agent’s conditional law.

20. An even faster learner need not make \(D\) vanish: setting its next prediction to the last observed action yields a point mass and can make the next KL infinite. The lemma holds only if “re-fit” is defined to force the reference prediction to equal the true current policy, in which case it is tautological rather than a timescale result.

21. The claim \(\Lambda_s\equiv0\) under rapid refitting is separately unsupported. The own and external models can be refitted at every step while retaining different latent routing rules for injected acts. Equal update rates do not make their predictive distributions equal.

22. \(\Lambda_s\) is a cumulative log-likelihood ratio, not “log-odds,” unless hypothesis priors are specified. Its sign on a realized path need not identify adoption or repair; even when the own model is true, individual increments can be negative.

23. Divergence of \(\Lambda_s\) requires conditions such as persistent separation of the two predictive laws, positive asymptotic KL rate, integrability, and an ergodic or martingale law. None follows from “self-typed.” The two models may merge, alternate, become observationally equivalent, or differ only finitely often.

24. Likewise, \(\Lambda_s\to-\infty\) does not follow from a private commitment. A private commitment can affect only one action, be probabilistically weak, be revealed and then forgotten, or make own and external predictions mutually equivalent. Repair can therefore produce a finite log Bayes factor.

25. The world-typed plateau claim is not implied by filter synchronization. The own and external readings differ in their \(g\)-updates, not just their filters. Even after their \(\eta\) components synchronize, their \(z\) components can preserve an injected act forever, so their action laws and \(\Lambda\) need not merge.

26. Filter synchronization itself normally compares filters exposed to the same action-observation sequence under suitable absolute-continuity and mixing conditions. An injection changes the action policy and potentially the controlled hidden-state trajectory. Merely citing an informative channel does not provide the required uniform controlled-filter stability.

27. “Healed by evidence” is not necessarily monotone in a scalar noise parameter \(q\). Active sensing, state-dependent observability, policy thresholds, and controlled transitions can make additional observation noise change actions and hence improve or worsen synchronization non-monotonically.

28. The rows of (H) are neither exhaustive nor disjoint. \(D_s\) may oscillate, remain at a positive constant, decay partially, recur, grow and later decay, or contain simultaneous evidence- and expression-dependent components. “Growing” can overlap with transient lock-in, while \(D\equiv0\) is already a limiting form of immediate healing.

29. A growing divergence does not establish “rewriting the law that scores it.” The law is explicitly frozen, so it is not being rewritten. Growth can instead arise from unstable state dynamics, accumulating environmental effects, chaotic amplification, or support mismatch.

30. KL support failures can dominate the proposed phenomenology. If the frozen reference assigns zero probability to an action the agent can take, \(D_s\) or a \(\Lambda\) increment is infinite immediately. Such infinity says nothing by itself about world/self typing, adoption, or timescale.

31. The asymptotic statements do not specify whether they hold almost surely, in probability, or in expectation. \(D_s\) and \(\Lambda_s\) are random variables along the injected trajectory; pathwise divergence can fail even when expected log-likelihood drift is positive.

32. The theorem-like classification has no specified perturbation size. Large \(\eta\)-injections can cross basins, alter the environment irreversibly, or destroy absolute continuity; small \(z\)-injections may be policy-invisible. Type can therefore depend on amplitude, not just direction.

33. The “steering is not setting” remark does not resolve controlled-environment counterexamples. If a perturbed belief causes actions that permanently change the hidden world, the posterior may converge to the perturbed belief because the world was changed to match it. From the proposed scores alone, there may be no identifiable distinction between this route and commitment adoption.

34. Calling \(\kappa\) an “ordinary, evidence-driven belief” is unjustified without a hypothesis space, prior, likelihood model, or update rule. Since its supposed evidence \(\Lambda\) is itself computed from two model-dependent counterfactual routings, \(\kappa\) is not yet a defined Bayesian quantity.

35. The final feedback loop is potentially self-confirming rather than calibrated: \(\kappa\) selects which directions are injected, and only injected directions generate evidence for \(\kappa\). Without forced exploration or identifiability assumptions, many incorrect fields are absorbing—not only a uniformly low “helpless” field.

B. FRESH-READER

1. The central mathematical objects \(P^{-}\), \(P^{\mathrm{own}}\), and \(P^{\mathrm{ext}}\) are not formally defined as kernels or state-update procedures. A fresh reader cannot reproduce either score from the given definitions.

2. “Frozen model” is ambiguous: it could mean fixed parameters, fixed posterior, fixed hidden state, no online training, or an unperturbed counterfactual state. The text simultaneously freezes it and rolls it forward on new evidence.

3. The claim to recover the decomposition is hard to parse because injections are already described as being along known \(\eta\)- and \(z\)-directions. It is unclear what is unknown to \(S\), what coordinate system it possesses, and what output counts as successful recovery.

4. A “direction” \(v\) is undefined. The space containing it, its scale, admissibility, relationship to the simplex constraint, and meaning for a quotient-valued \(z\) are all unstated.

5. The map between \(S\)'s variables \((\eta',z')\) and the actual variables \((\eta,z)\) is unspecified. “Tracking” could mean equality, decoding, approximate prediction, or an unidentified change of coordinates.

6. The Screening-off equation does not define \(\xi'_\tau\) precisely. It is unclear whether it is the agent’s actual state, a counterfactual state, or \(S\)'s modeled state, and which random variables are shared between the two sides.

7. The notation \(do(\xi_t=\xi')\) presupposes a structural causal model and intervention semantics that have not been given. The earlier cycle map is a stochastic update scheme, not by itself a complete SCM.

8. “Every internal counterfactual is realizable purely on the action channel” overstates what the displayed equality visibly establishes. The equality concerns only the marginal law of future observations, not internal variables or the complete loop trajectory.

9. It is unclear how much control \(S\) has. The section alternates among “contributes to,” “partly its to write,” “the surface \(S\) owns,” and complete replacement of the action distribution.

10. The injection is defined using \(\pi(\cdot\mid(\eta'_\tau,z'_\tau)+v)\), but no procedure explains how \(S\) samples, forces, patches, or combines those actions with the rest of \(A\)'s controller.

11. “Action-shadow” appears later in the neural-network subsection without definition. Presumably it means the action distribution induced by a latent perturbation, but that relationship is never formalized.

12. The history notation is inconsistent or at least unexplained: \(h_t\), \(h_{\le t}\), \(h'_{t_0+s}\), and \(h'_\tau\) are used without saying exactly whether the current action or observation is included. This matters for every conditional probability in (D) and \(\Lambda\).

13. The timing in (D) is unclear. At \(t_0+s\), is \(\pi(\cdot\mid\xi_{t_0+s})\) predicting \(a_{t_0+s}\) before it occurs, while the conditioning history excludes that action? The notation does not settle this.

14. “The agent as its state has it” and “the agent as its own record implies” are intuitive glosses, but no mathematical distinction between those two states has been defined.

15. The phrase “without the perturbation” conflicts with conditioning \(P^{-}\) on the perturbed realized history \(h'\). A reader needs to know exactly which causal updates are retained and which are suppressed.

16. The summation in \(\Lambda_s\) needs a stated domain such as \(s\ge k\). For \(s<k\), its bounds are reversed or the sum must be declared empty.

17. \(\Lambda_s\) is called “authorship log-odds,” but no authorship hypotheses or prior odds are introduced. The displayed quantity is only a likelihood-ratio sum.

18. Zero-probability actions are not addressed. Both KL and the logarithmic ratio may be infinite or undefined, especially for deterministic policies, which the proposal repeatedly discusses.

19. The probability mode of all claimed limits is missing. “Drifts without bound,” “plateaus,” and “\(\to\pm\infty\)” could refer to sample paths, expectation, median behavior, or an asymptotic rate.

20. The four labels in (H)—“settable,” “world-typed,” “self-typed,” and “lock-in”—lack quantitative definitions. No threshold, time horizon, derivative, decay law, or statistical decision rule is provided.

21. “Healed by evidence” says the rate “scales with” informativeness and “vanishes with it,” but neither rate nor informativeness is mathematically defined. The binary-flip example does not generalize this notion.

22. “Healed by expression” is defined as “killed at the first post-window act,” but “killed” could mean \(D=0\) exactly, falls below a threshold, or becomes asymptotically negligible.

23. The phrase “whatever the channel” is ambiguous: action channel, evidence channel, observation-noise parameter, or the environment transducer as a whole.

24. “The two readings merge as the filter re-synchronizes” skips a necessary step. The readings also maintain commitment states through different \(g\)-updates; filter synchronization alone does not visibly merge their complete predictive laws.

25. The cited synchronization condition is not stated in usable form. A fresh reader needs to know which filters, driven by which common inputs, synchronize under what mixing or observability assumptions.

26. “Informative channel” is treated as equivalent to \(q\) bounded away from \(1/2\), but this only makes sense for an unstated binary symmetric observation model. The general transducer has no \(q\).

27. The relationship between intrinsic perturbations in the passive intuition and action injections in the formal experiment is not established. They act on different variables and need not generate equivalent state trajectories.

28. The statement that \(g\) treats action as “content” while the filter treats it “only as an index” is metaphorical. Both are mathematical arguments to update functions, and no invariant criterion distinguishes content from index.

29. “Adoptable” and “steerable” are introduced as if formally established, but neither term is defined and neither conclusion follows for arbitrary \(g\) and \(T^{(a,x)}\).

30. The section does not explain whether the true \(\xi_{t_0+s}\) in (D) is observable to the experimenter after \(S\) has overridden actions, or how it is recovered when the agent has hidden randomness.

31. “Computable by \(S\) from the stream alone” is misleading without listing what \(S\) knows. Computing the two reference laws appears also to require the frozen model, its state, \(g\), the filter kernels, and knowledge of which acts were injected.

32. “Reference lag” is invoked in the definition of \(\kappa\) but never parameterized. There is no update schedule, lag length, separation-of-timescales condition, or dependence of the field on that choice.

33. The codomain of the field \(v\mapsto\mathrm{type}(v)\) is only an informal four-way taxonomy. It is unclear how an “estimate” of a categorical field is represented, calibrated, or updated from \(\Lambda\).

34. The claim that \(\Lambda\) is “exactly” \(\kappa\)'s evidence is unexplained because \(\Lambda\) compares authorship readings, whereas (H) is principally defined using \(D\) and an evidence-noise sweep.

35. “Spend the field” and “\(\kappa\) gates commitment” have no corresponding variables or equations. No action-selection rule involving \(\kappa\) appears in the formal model.

36. “Acting threshold” is introduced only in the absorbing-state discussion. Its value, role, and relationship to injection frequency are not defined.

37. The comparison with \(\Pi\) presupposes that naturally occurring perturbations arrive at a rate tied to posterior identification of agent identity. No link between perturbation frequency and \(\Pi_t\) was established in Section 3.

38. “Curable only by the holder” is a substantive causal claim unsupported by the model; external interventions could alter priors, thresholds, observations, or actions.

39. The neural-network implementation does not say how a latent “candidate direction” is converted into a token-level action-shadow. This conversion is the key experimental bridge and may require precisely the decoding the paragraph says is unnecessary.

40. “Patched emissions” and “prefill edits” are treated as interchangeable injections, although one changes an emitted action and the other may modify an observed context token never produced by the agent.

41. The actor/reader slip test does not formally define the two “seats,” how otherwise identical contexts encode ownership, or what counts as “repair,” “re-decide,” and “Bayes-follow.”

42. The assertion that a base model should adopt “essentially any prefill as self” does not follow transparently from teacher forcing. Training without authorship labels does not imply invariance to role tokens, discourse structure, quoted text, or other contextual evidence of who generated a token.

43. Calling RLHF’s KL-to-base penalty “exactly” (D) is unclear and likely mismatched: standard RLHF penalties compare the trained/current policy with a base/reference policy on sampled contexts, whereas (D) compares the true agent’s current state-conditioned policy with a specially frozen, counterfactually routed model after an injection.
tokens used
18,312
A. ADVERSARIAL

1. The “Recovery of the decomposition” claim is false under its stated assumptions. Let the environment contain a static bit \(W\), observed through i.i.d. flips with probability \(q<1/2\); its Bayes filter is informative and synchronizing. Independently define
   \[
   z_{t+1}=a_t,\qquad P(a_t=z_t\mid z_t)=0.9.
   \]
   Thus \(z_t\) is reconstructed entirely from the realized stream, satisfying (b). After a forced window, the first free action may distinguish the “own” and “external” readings, but both readings then consume that same post-window action and acquire the identical state \(z_{t+1}=a_t\). Consequently \(\Lambda_s\) contains at most one informative increment and converges to a finite value, not \(+\infty\).

2. The claim also fails for \(\eta\)-directions because no observability condition connects \(\eta\) to the policy. In the same informative, synchronizing environment, let \(\pi(a\mid\eta,z)\) be independent of \(\eta\). Every \(\eta\)-direction then has the same “action-shadow,” the injection does nothing, and \(D_s\equiv0\). The classification calls it settable rather than world-typed. A weaker counterexample uses a threshold policy and a perturbation that does not cross a threshold.

3. There is a direct tension between “healed by expression” and \(\Lambda_s\to+\infty\). If expression kills the discrepancy at the first post-window act, as (H) explicitly says, the own and external predictive laws should thereafter coincide. Only finitely many nonzero log-likelihood increments remain, so \(\Lambda_s\) normally plateaus. Divergence requires persistent positive information per step, an extra assumption almost opposite to one-step healing.

4. Assumption (b), that \(g\) “consumes” realized acts, does not imply adoption. A valid update can ignore its action argument, retain a private component, average actions slowly, store their parity, or treat them as observations rather than commands. Even “\(z\) is reconstructed from the stream” does not imply that one act overwrites \(z\), much less that all \(z\)-directions heal after one expression.

5. The recovery construction is circular about the decomposition it claims to recover. An “\(\eta\)-direction” or “\(z\)-direction” can be injected only after coordinates corresponding to \(\eta\) and \(z\) have already been identified in \(S\)'s model. The premise even says \((\eta',z')\) track the true \((\eta,z)\). Testing already separated coordinate classes does not recover the separation “with no access to referents.”

6. The classification is not invariant under reparameterization. Replace model coordinates by
   \[
   u=\eta+z,\qquad w=\eta-z.
   \]
   Generic \(u\)- or \(w\)-directions perturb both mechanisms and can heal partly through evidence and partly through expression. Nothing in (H) identifies a unique factorization, subspace, or nonlinear foliation corresponding to the original \((\eta,z)\).

7. The Screening-off lemma establishes, at most, an equality under a strong modular causal model: the environment’s transition law must depend on the agent solely through emitted actions, and intervening on \(\xi\) must not alter any shared noise, side channel, timing, or physical variable. Those assumptions are embedded in \(\Phi\), not established for the stated “pair of transducers.”

8. The right-hand side of the Screening-off equality is effectively a full policy intervention extending over all future times. That is much stronger than “purely” playing a finite sequence of actions. It requires sampling from the counterfactual policy adaptively after every future observation. The finite injection defined immediately afterward does not realize the lemma’s infinite internal counterfactual.

9. The lemma does not show that \(S\) can perform the intervention. \(S\) is initially said only to “contribute” to actions and to have the channel “partly” available, but the injection assumes it can replace the complete action law. For example, if the emitted action is the XOR of \(S\)'s proposal and a private controller bit, \(S\) contributes causally but cannot implement an arbitrary counterfactual action policy.

10. “Every internal counterfactual is realizable” confuses matching the environment’s marginal response with realizing the agent counterfactual. Action replacement can reproduce the distribution of \(x_{\ge t}\), but generally not the counterfactual internal trajectory, correlations between internal and external variables, or downstream quantities depending directly on the internals.

11. The assertion that an agent cannot “plant evidence” on the \(x\)-channel is too strong. In a controlled observation model, actions can determine sensor position, measurement resolution, queried variable, or even the observation itself. For \(x_t=a_t\), the agent literally determines the next observation. Calling this merely choosing “which world is being asked” does not support the later mathematical distinction between content and index.

12. The statement that an action’s likelihood ratio between environmental hypotheses is \(1\) requires actions to be conditionally independent of the environment given public history. That fails whenever the agent has private observations or hidden state correlated with the environment—precisely among the hiding places admitted in Section 3. Example: \(W\in\{0,1\}\), the agent privately observes \(W\), and emits \(a=W\); then the action has an infinite likelihood ratio between the two hypotheses.

13. Adding \(v\) to \((\eta',z')\) may be meaningless. A generic vector takes \(\eta'+v\) outside the probability simplex, and \(Z\) was introduced as a quotient rather than a vector space. No tangent space, chart, projection, perturbation magnitude, or admissible-direction condition is supplied.

14. The injection applies the same \(v\) to every rolled-forward state. That is not generally the action sequence generated by a one-time intervention \(do(\xi_{t_0}=\xi_{t_0}+v)\), because nonlinear dynamics transport the perturbation. Thus the injection does not in general simulate the counterfactual invoked by Screening-off.

15. \(P^{-}\) in (D) is underdefined in a way that changes the answer. If the frozen model is nevertheless filtered normally on the entire injected history and uses the same stream-reconstructed \(g\), then an exact model reconstructs the same post-injection state as the agent and \(D_s=0\). If forced actions are withheld from \(g\), \(P^{-}\) is actually the later-defined external reading. The text needs one of these incompatible meanings.

16. “Frozen” is internally ambiguous. The model is said not to update, yet it is “rolled forward” and conditioned on everything that happened. Freezing parameters, freezing the latent state, freezing the posterior, and withholding selected state updates produce different reference laws and different scores.

17. \(D_s\) is only a behavioral pseudometric on internal state. Any two states with the same current policy distribution have zero divergence even if they have different future dynamics. Thus \(D\equiv0\) need not mean “another perfectly typical life”; it can mean a policy-null direction, a delayed effect, a symmetry, or an unsuccessful injection.

18. Current-action KL can miss delayed deviations. For example, let \(z\) be a commitment scheduled to affect the action only ten steps later. Then \(D_s=0\) for the first nine steps despite a causally potent self-state difference. Classification based on its profile needs a horizon-aware predictive divergence, not just the next-action law.

19. The Self-ratification lemma is false for ordinary online refitting. Let the true agent emit i.i.d. \(\mathrm{Bernoulli}(0.9)\) actions and let the reference refit a Bernoulli parameter after each observation using the empirical frequency. It has “absorbed what the agent just did,” but
   \[
   D_{\mathrm{KL}}(\mathrm{Bern}(0.9)\Vert\mathrm{Bern}(\hat p_t))
   \]
   is generally nonzero at every finite time. Refitting at behavior’s rate does not imply exact interpolation of the agent’s conditional law.

20. An even faster learner need not make \(D\) vanish: setting its next prediction to the last observed action yields a point mass and can make the next KL infinite. The lemma holds only if “re-fit” is defined to force the reference prediction to equal the true current policy, in which case it is tautological rather than a timescale result.

21. The claim \(\Lambda_s\equiv0\) under rapid refitting is separately unsupported. The own and external models can be refitted at every step while retaining different latent routing rules for injected acts. Equal update rates do not make their predictive distributions equal.

22. \(\Lambda_s\) is a cumulative log-likelihood ratio, not “log-odds,” unless hypothesis priors are specified. Its sign on a realized path need not identify adoption or repair; even when the own model is true, individual increments can be negative.

23. Divergence of \(\Lambda_s\) requires conditions such as persistent separation of the two predictive laws, positive asymptotic KL rate, integrability, and an ergodic or martingale law. None follows from “self-typed.” The two models may merge, alternate, become observationally equivalent, or differ only finitely often.

24. Likewise, \(\Lambda_s\to-\infty\) does not follow from a private commitment. A private commitment can affect only one action, be probabilistically weak, be revealed and then forgotten, or make own and external predictions mutually equivalent. Repair can therefore produce a finite log Bayes factor.

25. The world-typed plateau claim is not implied by filter synchronization. The own and external readings differ in their \(g\)-updates, not just their filters. Even after their \(\eta\) components synchronize, their \(z\) components can preserve an injected act forever, so their action laws and \(\Lambda\) need not merge.

26. Filter synchronization itself normally compares filters exposed to the same action-observation sequence under suitable absolute-continuity and mixing conditions. An injection changes the action policy and potentially the controlled hidden-state trajectory. Merely citing an informative channel does not provide the required uniform controlled-filter stability.

27. “Healed by evidence” is not necessarily monotone in a scalar noise parameter \(q\). Active sensing, state-dependent observability, policy thresholds, and controlled transitions can make additional observation noise change actions and hence improve or worsen synchronization non-monotonically.

28. The rows of (H) are neither exhaustive nor disjoint. \(D_s\) may oscillate, remain at a positive constant, decay partially, recur, grow and later decay, or contain simultaneous evidence- and expression-dependent components. “Growing” can overlap with transient lock-in, while \(D\equiv0\) is already a limiting form of immediate healing.

29. A growing divergence does not establish “rewriting the law that scores it.” The law is explicitly frozen, so it is not being rewritten. Growth can instead arise from unstable state dynamics, accumulating environmental effects, chaotic amplification, or support mismatch.

30. KL support failures can dominate the proposed phenomenology. If the frozen reference assigns zero probability to an action the agent can take, \(D_s\) or a \(\Lambda\) increment is infinite immediately. Such infinity says nothing by itself about world/self typing, adoption, or timescale.

31. The asymptotic statements do not specify whether they hold almost surely, in probability, or in expectation. \(D_s\) and \(\Lambda_s\) are random variables along the injected trajectory; pathwise divergence can fail even when expected log-likelihood drift is positive.

32. The theorem-like classification has no specified perturbation size. Large \(\eta\)-injections can cross basins, alter the environment irreversibly, or destroy absolute continuity; small \(z\)-injections may be policy-invisible. Type can therefore depend on amplitude, not just direction.

33. The “steering is not setting” remark does not resolve controlled-environment counterexamples. If a perturbed belief causes actions that permanently change the hidden world, the posterior may converge to the perturbed belief because the world was changed to match it. From the proposed scores alone, there may be no identifiable distinction between this route and commitment adoption.

34. Calling \(\kappa\) an “ordinary, evidence-driven belief” is unjustified without a hypothesis space, prior, likelihood model, or update rule. Since its supposed evidence \(\Lambda\) is itself computed from two model-dependent counterfactual routings, \(\kappa\) is not yet a defined Bayesian quantity.

35. The final feedback loop is potentially self-confirming rather than calibrated: \(\kappa\) selects which directions are injected, and only injected directions generate evidence for \(\kappa\). Without forced exploration or identifiability assumptions, many incorrect fields are absorbing—not only a uniformly low “helpless” field.

B. FRESH-READER

1. The central mathematical objects \(P^{-}\), \(P^{\mathrm{own}}\), and \(P^{\mathrm{ext}}\) are not formally defined as kernels or state-update procedures. A fresh reader cannot reproduce either score from the given definitions.

2. “Frozen model” is ambiguous: it could mean fixed parameters, fixed posterior, fixed hidden state, no online training, or an unperturbed counterfactual state. The text simultaneously freezes it and rolls it forward on new evidence.

3. The claim to recover the decomposition is hard to parse because injections are already described as being along known \(\eta\)- and \(z\)-directions. It is unclear what is unknown to \(S\), what coordinate system it possesses, and what output counts as successful recovery.

4. A “direction” \(v\) is undefined. The space containing it, its scale, admissibility, relationship to the simplex constraint, and meaning for a quotient-valued \(z\) are all unstated.

5. The map between \(S\)'s variables \((\eta',z')\) and the actual variables \((\eta,z)\) is unspecified. “Tracking” could mean equality, decoding, approximate prediction, or an unidentified change of coordinates.

6. The Screening-off equation does not define \(\xi'_\tau\) precisely. It is unclear whether it is the agent’s actual state, a counterfactual state, or \(S\)'s modeled state, and which random variables are shared between the two sides.

7. The notation \(do(\xi_t=\xi')\) presupposes a structural causal model and intervention semantics that have not been given. The earlier cycle map is a stochastic update scheme, not by itself a complete SCM.

8. “Every internal counterfactual is realizable purely on the action channel” overstates what the displayed equality visibly establishes. The equality concerns only the marginal law of future observations, not internal variables or the complete loop trajectory.

9. It is unclear how much control \(S\) has. The section alternates among “contributes to,” “partly its to write,” “the surface \(S\) owns,” and complete replacement of the action distribution.

10. The injection is defined using \(\pi(\cdot\mid(\eta'_\tau,z'_\tau)+v)\), but no procedure explains how \(S\) samples, forces, patches, or combines those actions with the rest of \(A\)'s controller.

11. “Action-shadow” appears later in the neural-network subsection without definition. Presumably it means the action distribution induced by a latent perturbation, but that relationship is never formalized.

12. The history notation is inconsistent or at least unexplained: \(h_t\), \(h_{\le t}\), \(h'_{t_0+s}\), and \(h'_\tau\) are used without saying exactly whether the current action or observation is included. This matters for every conditional probability in (D) and \(\Lambda\).

13. The timing in (D) is unclear. At \(t_0+s\), is \(\pi(\cdot\mid\xi_{t_0+s})\) predicting \(a_{t_0+s}\) before it occurs, while the conditioning history excludes that action? The notation does not settle this.

14. “The agent as its state has it” and “the agent as its own record implies” are intuitive glosses, but no mathematical distinction between those two states has been defined.

15. The phrase “without the perturbation” conflicts with conditioning \(P^{-}\) on the perturbed realized history \(h'\). A reader needs to know exactly which causal updates are retained and which are suppressed.

16. The summation in \(\Lambda_s\) needs a stated domain such as \(s\ge k\). For \(s<k\), its bounds are reversed or the sum must be declared empty.

17. \(\Lambda_s\) is called “authorship log-odds,” but no authorship hypotheses or prior odds are introduced. The displayed quantity is only a likelihood-ratio sum.

18. Zero-probability actions are not addressed. Both KL and the logarithmic ratio may be infinite or undefined, especially for deterministic policies, which the proposal repeatedly discusses.

19. The probability mode of all claimed limits is missing. “Drifts without bound,” “plateaus,” and “\(\to\pm\infty\)” could refer to sample paths, expectation, median behavior, or an asymptotic rate.

20. The four labels in (H)—“settable,” “world-typed,” “self-typed,” and “lock-in”—lack quantitative definitions. No threshold, time horizon, derivative, decay law, or statistical decision rule is provided.

21. “Healed by evidence” says the rate “scales with” informativeness and “vanishes with it,” but neither rate nor informativeness is mathematically defined. The binary-flip example does not generalize this notion.

22. “Healed by expression” is defined as “killed at the first post-window act,” but “killed” could mean \(D=0\) exactly, falls below a threshold, or becomes asymptotically negligible.

23. The phrase “whatever the channel” is ambiguous: action channel, evidence channel, observation-noise parameter, or the environment transducer as a whole.

24. “The two readings merge as the filter re-synchronizes” skips a necessary step. The readings also maintain commitment states through different \(g\)-updates; filter synchronization alone does not visibly merge their complete predictive laws.

25. The cited synchronization condition is not stated in usable form. A fresh reader needs to know which filters, driven by which common inputs, synchronize under what mixing or observability assumptions.

26. “Informative channel” is treated as equivalent to \(q\) bounded away from \(1/2\), but this only makes sense for an unstated binary symmetric observation model. The general transducer has no \(q\).

27. The relationship between intrinsic perturbations in the passive intuition and action injections in the formal experiment is not established. They act on different variables and need not generate equivalent state trajectories.

28. The statement that \(g\) treats action as “content” while the filter treats it “only as an index” is metaphorical. Both are mathematical arguments to update functions, and no invariant criterion distinguishes content from index.

29. “Adoptable” and “steerable” are introduced as if formally established, but neither term is defined and neither conclusion follows for arbitrary \(g\) and \(T^{(a,x)}\).

30. The section does not explain whether the true \(\xi_{t_0+s}\) in (D) is observable to the experimenter after \(S\) has overridden actions, or how it is recovered when the agent has hidden randomness.

31. “Computable by \(S\) from the stream alone” is misleading without listing what \(S\) knows. Computing the two reference laws appears also to require the frozen model, its state, \(g\), the filter kernels, and knowledge of which acts were injected.

32. “Reference lag” is invoked in the definition of \(\kappa\) but never parameterized. There is no update schedule, lag length, separation-of-timescales condition, or dependence of the field on that choice.

33. The codomain of the field \(v\mapsto\mathrm{type}(v)\) is only an informal four-way taxonomy. It is unclear how an “estimate” of a categorical field is represented, calibrated, or updated from \(\Lambda\).

34. The claim that \(\Lambda\) is “exactly” \(\kappa\)'s evidence is unexplained because \(\Lambda\) compares authorship readings, whereas (H) is principally defined using \(D\) and an evidence-noise sweep.

35. “Spend the field” and “\(\kappa\) gates commitment” have no corresponding variables or equations. No action-selection rule involving \(\kappa\) appears in the formal model.

36. “Acting threshold” is introduced only in the absorbing-state discussion. Its value, role, and relationship to injection frequency are not defined.

37. The comparison with \(\Pi\) presupposes that naturally occurring perturbations arrive at a rate tied to posterior identification of agent identity. No link between perturbation frequency and \(\Pi_t\) was established in Section 3.

38. “Curable only by the holder” is a substantive causal claim unsupported by the model; external interventions could alter priors, thresholds, observations, or actions.

39. The neural-network implementation does not say how a latent “candidate direction” is converted into a token-level action-shadow. This conversion is the key experimental bridge and may require precisely the decoding the paragraph says is unnecessary.

40. “Patched emissions” and “prefill edits” are treated as interchangeable injections, although one changes an emitted action and the other may modify an observed context token never produced by the agent.

41. The actor/reader slip test does not formally define the two “seats,” how otherwise identical contexts encode ownership, or what counts as “repair,” “re-decide,” and “Bayes-follow.”

42. The assertion that a base model should adopt “essentially any prefill as self” does not follow transparently from teacher forcing. Training without authorship labels does not imply invariance to role tokens, discourse structure, quoted text, or other contextual evidence of who generated a token.

43. Calling RLHF’s KL-to-base penalty “exactly” (D) is unclear and likely mismatched: standard RLHF penalties compare the trained/current policy with a base/reference policy on sampled contexts, whereas (D) compares the true agent’s current state-conditioned policy with a specially frozen, counterfactually routed model after an injection.
