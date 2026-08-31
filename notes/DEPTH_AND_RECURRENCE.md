# Depth as the unrolling of the belief↔action recurrence — when it is (and isn't) needed

*Project: self-models. This nails down a point we kept circling: where transformer DEPTH is the
resource for the action↔belief computation, and why our perception/filtering tasks never showed it.*

## The computation we're asking for
The optimal agent is a **time-recurrence**:
```
a_t   = f(b_t)                       # action = readout of the current belief
b_{t+1} = G(b_t, a_t, o_{t+1})       # belief update; the action enters (efference copy: decode o−a)
```
The action and belief are mutually dependent **through time** (staggered): `a_t` reads `b_t`; `b_{t+1}`
uses `a_t`. An **RNN** computes this by *looping* — one application of `G` per step, unbounded, shared
weights.

## The feedforward problem (the core point)
A **transformer is feedforward**: fixed number of layers, no runtime loop. It *cannot* iterate a
recurrence at runtime. To compute a T-step recurrence it must **unroll it across DEPTH**, and the
geometry of the unroll is forced by two facts:
1. information flows **forward in position** (causal attention) and **upward in layer**;
2. `a_t = f(b_t)` is a readout of the *full* belief, so it is finalized at some **depth** at position `t`.

Therefore `b_{t+1}` (which needs `a_t`) can only incorporate it at a **deeper** layer, and `a_{t+1}`
deeper still. The "correctly-computed" frontier marches **diagonally through the (position × layer)
grid** — roughly **one belief↔action update per ~c layers**, with attention chaining the time steps.

> **Depth = how many steps of the action↔belief recurrence the net can correctly chain.**
> This is a Picard / fixed-point-style iteration performed *through layers*, and it is the
> feedforward architecture standing in for an RNN's temporal loop. (≈ each layer advances the
> mutual action↔belief solution one iterate.)

## Why our tasks never made depth bite — the contractive shortcut
There is an escape from the diagonal march: compute `b_t` at **every position in parallel**, directly
from that position's own history, all at shallow layers. This is available **iff the recurrence is
contractive — iff the filter FORGETS.**

**IMPORTANT caveat (the belief is NOT a function of the raw observed tokens).** Because
`o_{t+1}=e_{t+1}+a_t`, the observation is the emission *plus the net's own action*; the belief requires
subtracting `a_t=f(b_t)`, which is **not** in the token stream. So the efference copy is genuinely
*necessary* (raw observations are insufficient — this is the self-model content, and it's why the
faithful rubber-hand works). The precise statement is therefore: `b_t` is a shallow function of recent
**(observations + the net's own recent actions)**, where the actions are themselves **shallowly
re-derived** from recent observations. It stays shallow for two contingent reasons:
- the filter is **contractive** — errors in mis-re-derived old actions *decay* (each observation
  re-anchors the belief), so only the *recent* actions matter, not the full history;
- within that short window, re-deriving the recent action is itself a smooth, low-depth computation.

So: **"necessary but shallow."** The action-correction is load-bearing but, for a contractive filter,
computed in parallel from a short window — not unrolled through depth. (Dissections confirm: the action
is re-derived *from the observation window*, causally real but depth-free.)

The action-corruption is also the **seed of non-contractivity**: a wrong re-derived `a_{t-1}` gives a
wrong `e_t→b_t→a_t`, an error that propagates forward. If observations keep re-anchoring (our tasks),
it *decays* (contractive → shallow). If observations are **weak/uninformative** so the agent must
**dead-reckon from its own actions**, the error *accumulates*, the belief depends on the whole action
history, and (with nonlinear, e.g. heading-dependent, integration) it becomes genuinely **non-contractive
→ deep**. That is a third depth route (besides planning and within-step fixed points), and it is
maximally "self": *path integration of one's own actions*.

**Precise principle:** `depth_needed = depth-complexity of (observations → belief/action)`.
Contractive/forgetting ⇒ low ⇒ **depth-flat**. The diagonal march only has to be paid when **no
shallow windowed shortcut exists.**

### Empirical record (all confirm the shortcut, all depth-flat)
- linear-Gaussian (posterior-mean readout): 1L ≈ 8L ≈ Kalman floor.
- nonlinear observation `h(s)=s²` (bimodal posterior): depth-flat, 1L (even d=8, ~1K params) at floor.
- aliased HMM (long-memory, ambiguous emissions): depth-flat.
- **nonlinear readout** `a≈E[tanh(s)]` (the "smooth-but-nonlinear" cell): **also depth-flat**
  (excess +0.003→+0.006, 1L≈8L). The nonlinear readout did NOT make the recurrence non-contractive.
- Dissections (`d2`, `cont_c8`): belief R² ~flat across layers; the previous action is **re-derived
  from a recent observation window**, not carried through a deep recurrence; no `t→t-1` routing edge;
  depth only polishes the readout. ⇒ the net takes the **shortcut**, not the march.

## Where the Picard-through-layers is UNAVOIDABLE (depth genuinely needed)
Kill the shortcut: make the recurrence **non-contractive** (it does not forget; the unroll can't be
collapsed to a shallow window). Then the transformer **must** perform the diagonal march, depth =
chain length, and the staircase should be directly observable in activations (belief/action R²
*climbing* with depth, a real `t→t-1` routing edge, and depth discrimination). These are
**deliberative**, not perceptual:
- **Forward self-simulation / planning**: emit the action you'd take K steps ahead — roll your own
  policy forward in one pass. Each step needs the previous, nothing forgets ⇒ genuinely depth-K.
  (This is also the *reflexive* self-model: model your own future policy.)
- **Within-step fixed points**: e.g. coupled corruption `o_t = h(s_t, a_t)` with `a_t=f(b_t)`, so
  `b_t` and `a_t` are co-determined and must be iterated to convergence *within a timestep*.

## The reconciliation (so we stop relitigating it)
- The **Picard-through-layers** account is the *correct* model of how a feedforward transformer
  realizes a recurrence; depth is its unroll budget. (This is the user's framing, and it's right.)
- "**depth-flat / attention parallelizes it**" is what happens **only when a contractive shortcut
  exists** — i.e. for forgetting filters, where `obs→belief` is low-depth.
- The entire back-and-forth reduces to one question: **does the recurrence forget?**
  - Perception / Bayesian filtering: **yes** → shallow windowed shortcut → depth-flat.
  - Deliberation / planning / fixed-points: **no** → must unroll through depth → depth discriminates.

**Consequence for the project:** stop trying to extract depth from perception (efference-copy
filtering is contractive, provably shallow — established across 4 task families). The next experiment
that *should* light up depth is **forward self-simulation** (and/or the coupled-corruption within-step
fixed point), where the action↔belief recurrence cannot be shortcut and the unroll-through-layers is
forced.
