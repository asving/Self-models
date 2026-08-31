# When being random makes you blind

*A small study of a tension between two pressures on a learning agent: the pressure to be
unpredictable, and the pressure to stay legible to yourself. Self-contained; everything you need to
rebuild the experiments is here.*

---

## 1. The puzzle

We had been chasing "self-models" in neural networks — the part of a network that has to keep track of
its *own* behavior in order to function. Over and over we ran into the same wall: whenever the best
thing to do is to be **random**, a network will happily become random, and then there is nothing
self-referential left to study. A coin doesn't need a self-model.

But we kept feeling there *should* be a counter-pressure. Here is the intuition, stated plainly:

> If your own actions affect what you see next, then to make sense of what you see you have to know
> what you did. And if you chose your action by flipping a coin and then forgot the result, you've
> thrown away exactly the information you need to interpret the world.

So there ought to be a pull *toward* being predictable — not to please anyone else, but so that **you
can understand your own observations**. We wanted the cleanest possible setting where this pull fights the
pull toward randomness, with a knob to slide between them, and with everything computable so we could
check the answer.

A few terms we'll use, defined once:

- **Action distribution / policy** `p`: at each moment the agent doesn't pick a fixed move, it picks a
  *probability distribution* over moves and then samples one. `p = (p₀, p₁, p₂)` for three moves.
- **Entropy** `H(p) = −Σᵢ pᵢ log pᵢ`: a number measuring how spread-out a distribution is. `H = 0`
  means "always the same move" (deterministic); `H = log 3 ≈ 1.10` means "perfectly uniform, maximally
  random" (for three moves). Entropy is our one-number summary of "how random is the agent."
- **Imperfect monitoring**: the agent does *not* get to see the opponent's move; it sees only a
  *pooled* signal that mixes its own move with the opponent's.
- **Efference copy**: the (biological) idea that to interpret a sensation caused partly by your own
  action, the brain keeps a copy of the motor command and subtracts its effect. Same idea here.

---

## 2. A clean toy: blindfolded rock–paper–scissors

Take ordinary rock–paper–scissors with moves labelled `0,1,2` on a cycle (each move beats the one
below it, mod 3). Two changes turn it into our laboratory:

**(a) The agent only sees the *outcome*, not the opponent's move.** Each round the agent plays `a` and
the opponent plays `b`, and the only thing reported back is

```
        o = (a − b) mod 3        ∈ {0 = tie, 1 = win, 2 = loss}.
```

This is the efference structure in its barest form: the observation `o` *pools* the agent's move and
the opponent's move. To recover what the opponent actually did, the agent must undo its own
contribution: `b = (a − o) mod 3`. **That subtraction needs `a` — the agent's own move.**

**(b) The agent never gets told its own realized move.** It outputs a policy `p`, samples a move
`a ~ p`, and the move is gone. So the agent only *knows* its move to the extent its policy was sharp.
If `p` was deterministic, it knows `a` exactly. If `p` was uniform, the realized `a` is a coin it
didn't keep — and then `o = (a − b)` tells it nothing about `b`, because it can't subtract a number it
doesn't know.

This last point is the whole game in one line:

> **The information the agent's observations carry about the opponent is gated by the agent's own
> entropy.** Sharp policy → you know your move → you can read the opponent. Uniform policy → you've
> blinded yourself.

### Why there is a genuine tension

Why would the agent ever *want* to be random? Because of the opponent. We give the opponent two
ingredients, mixed by a knob `β ∈ [0,1]`:

- a **hidden bias** `q` — a fixed, lopsided distribution over moves (drawn each game from a
  `Dirichlet(0.5)` distribution, which produces a different, usually-lopsided `q` every game). A biased
  opponent is *exploitable*: if you can figure out `q`, you just play the move that beats its favorite.
- a **best-responder** `BR(p)` — an opponent that can *simulate the agent's policy* `p` and plays the
  move most likely to beat it. (Concretely `BR(b) ∝ exp(γ · p[(b−1) mod 3])`, sharpness `γ = 6`. A
  best-responder *punishes predictability*: the sharper you are, the harder it beats you.)

So the agent is squeezed:

- To **exploit the bias** it must identify `q`, which (under imperfect monitoring) means decoding the
  opponent's moves, which means being **sharp** (low entropy).
- To **survive the best-responder** it must be **unpredictable** (high entropy).

And here's the trap door: under imperfect monitoring, *the only way to gather the information that
tells you whether the opponent is exploitable is to lower your entropy* — which is exactly what the
best-responder is waiting for. You cannot safely investigate.

`β` slides between the two worlds: `β = 0` is a pure exploitable bias (be sharp, cash in); `β = 1` is a
pure best-responder (be uniform, hide).

### The machinery (enough to rebuild it)

- The agent is a **small transformer** (2 layers, width 64, 4 heads) reading the sequence of past
  outcomes `o₁, o₂, …` and emitting, each round, action logits (→ policy `p`) and a value estimate.
  It does **not** receive its own past actions as input — consistent with "it must re-derive or
  remember them." Horizon `T = 40` rounds per game.
- It is trained by **REINFORCE**: a policy-gradient method that nudges up the probability of actions
  that were followed by high reward and down those followed by low reward, using the value estimate as
  a baseline. Reward each round is the game payoff (`+1` win, `0` tie, `−1` loss). **Crucially we add
  no entropy bonus** — many RL recipes bolt on a "stay random" term; we deliberately don't, so that
  whatever entropy emerges is the agent's *own* equilibrium choice, not something we paid for.
- Code: `rps_im.py` (the game + training). Batch 256, 4000 steps, lr 1e-3. Everything — the agent's
  realized moves, the opponent's moves, its hidden bias — is generated by us, so the "ground truth" is
  always available to check against.

---

## 3. What happened

### Result 1 — a sharp phase transition in self-chosen randomness

Train a separate agent for each `β` and read off its entropy. The agent finds the game-theoretic
optimum, and it switches regimes abruptly:

| `β` | 0.0 | 0.2 | 0.26 | 0.30 | 0.34 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|
| entropy | 0.07 | 0.15 | 0.31 | 0.47 | **1.10** | 1.10 | 1.10 |
| payoff/round | +0.44 | +0.17 | +0.10 | +0.02 | 0.00 | 0.00 | 0.00 |

Below `β* ≈ 0.32` the agent collapses toward a deterministic policy and cashes in the bias (positive
payoff). Above it, the agent sits at *exactly* `log 3` — perfectly uniform — and ties. There is a
narrow mixed band around `0.26–0.30` and then it saturates.

A side note worth keeping: at high `β` the agent does **not** suffer the usual "policy collapse" we'd
seen elsewhere (RL agents going deterministic even when they shouldn't). Here it stays *exactly*
uniform — because here determinism is genuinely *punished*. Collapse, it turns out, only happens when
nobody punishes predictability.

### Result 2 — the self-legibility trap

The per-`β` blend above mixes bias and best-responder *every turn*, which muddies the opponent. Cleaner:
each *game*, commit the opponent to one **pure** type — a fixed bias with probability `1−β`, a pure
best-responder with probability `β` — and let the agent infer which it's facing as it plays. Now we can
read the agent's entropy *separately* on the two kinds of game (we know which is which). The optimal
agent should exploit the bias games and go uniform on the best-responder games, banking the average.

It doesn't.

| `β` (best-responder fraction) | entropy on bias-games | payoff | achievable |
|---|---|---|---|
| 0.2 | **0.07** (exploits) | +0.33 | +0.35 |
| 0.5 | **1.10** (uniform!) | +0.005 | +0.22 |
| 0.8 | 1.10 | +0.004 | +0.09 |

At `β = 0.2` it does the right thing. But by `β = 0.5` it has **given up on exploiting altogether** —
it plays uniform even on the games where the opponent is a sitting-duck fixed bias, leaving roughly
`+0.22` per round on the table. Why? Because to *discover* that a given opponent is exploitable it
would have to lower its entropy and watch — and lowering entropy is dangerous when half the opponents
punish it for that. So when best-responders are common, the safe move is to stay uniform and *blind*,
and never find the easy money. **The self-blinding isn't merely a decoding penalty; it's an
exploration trap** — you can't gather the evidence that would justify acting, without taking the very
risk the evidence is supposed to rule out.

That's a striking story. But "the agent plays it safe under imperfect monitoring" has a boring
explanation too — *maybe imperfect monitoring just makes the opponent harder to estimate, and a
cautious agent plays safe, with nothing special about "self-legibility" at all.* The rest of the study
is the detective work of killing that and every other boring reading.

---

## 4. Ruling out the alternatives

We claimed something specific: the difficulty is not generic, it is that **the estimation noise is
coupled to the agent's own entropy**. Four experiments pin this down.

### (i) Full observation rescues it — so the barrier is the monitoring

Give the agent the opponent's move `b` directly (perfect monitoring), changing nothing else. Now it can
identify the opponent *while still playing uniform* — no entropy cost to looking. Result: it exploits
at **every** `β`.

| `β` | bias-game entropy | payoff (imperfect → full-obs) |
|---|---|---|
| 0.5 | 0.33 | **+0.005 → +0.235** |
| 0.8 | 0.36 | +0.004 → +0.092 |

The `+0.005 → +0.235` jump at `β = 0.5` *is* the cost of the trap. So the barrier is the imperfect
monitoring, not the payoffs or generic caution.

### (ii) Independent noise *doesn't* rescue it — so it's the *coupling*, not the difficulty

Full observation is "too easy" — maybe any easier estimation would help. So we made estimation hard in
a way that is **not coupled to the agent's entropy**: the agent sees the opponent's move `b` directly,
but corrupted independently — with probability `ρ` it sees `b`, otherwise a random symbol. Now the
agent can still gather information *while uniform* (the corruption doesn't care what the agent does),
just noisily. At matched difficulty, prediction: if our story is right, this escapes the trap (like
full-obs); if it's just "estimation is hard," it traps.

It escapes — every run:

| condition (`β = 0.5`) | bias-game entropy | payoff |
|---|---|---|
| coupled (our game) | 1.10 | +0.005 (trapped) |
| independent noise `ρ = 0.3` | 0.82 | +0.06 |
| independent noise `ρ = 0.5` | 0.51 | +0.13 |
| independent noise `ρ = 0.7` | 0.41 | +0.17 |

And monotone in `ρ`: more information → sharper exploitation → more payoff. So at *matched* estimation
difficulty, the only thing that decides trap-versus-escape is whether the observation noise is **coupled
to the agent's own entropy**. The boring "harder estimation" reading is dead.

### (iii) Forcing sharpness *causes* information — the mechanism, directly

The two above are behavioral. Here is the mechanism caught in the act. Take an agent already good at
exploiting a bias, put it against a fixed bias, and **force** its action entropy to a chosen level `s`
(`s = 1`: force the sharp move; `s = 0`: force uniform), then measure how much it has learned about the
opponent — both behaviorally (can its final policy beat the true bias?) and representationally (can we
linearly decode the true bias from its internal activations? — reported as `R²`, where 1.0 is perfect
and 0 is chance).

| forced sharpness `s` | exploitation quality | bias decodable from activations (`R²`) |
|---|---|---|
| 0.0 (uniform) | −0.01 (chance) | −0.10 (nothing there) |
| 0.5 | +0.36 | 0.50 |
| 1.0 (sharp) | +0.52 | 0.85 |

Forcing the actions sharper *causes* the opponent's identity to appear in the agent's head and *causes*
it to play well. Under forced-uniform actions there is literally **no opponent information in the
network** — not because the network is bad, but because uniform actions make the observations carry
none. Sharpening *is* the act of perception. (Code: `rps_probe.py`.)

### (iv) The cleanest cut — separating "look" from "exploit"

There's one subtle confound left. In rock–paper–scissors the *best response to a known bias is a single
pure move* — so a sharp policy could be sharp simply because *exploiting* is sharp, not because
*looking* wants it. The two motives happen to want the same thing (low entropy), so we can't tell them
apart.

So we built a game where **the optimal exploitation is itself spread-out (high entropy)** — and then any
sharpening *below* that level cannot be exploitation; it can only be looking.

**The forecasting game.** Same blindfolded structure, but the agent is no longer scored on winning —
it's scored on *predicting* the opponent. Each round it reports a distribution `p` (and, as before,
samples a move `a ~ p` that pools into the observation `o = (a − b) mod 3`). Its reward is the
**proper log-score** `log p(b)` — high when it put high probability on the move the opponent actually
made. A basic fact about log-scoring: the best report is the truth, `p = q` (the opponent's real
distribution), so

> the optimal forecast is `q` itself, whose entropy `H(q)` is generally **high** (the opponent is
> genuinely mixed).

Now the tension is naked. To forecast well you want `p = q` (spread out, entropy `H(q)`). But to *learn*
`q` you must decode the opponent, which needs a **sharp** `p` — a deliberately *wrong* forecast. So:

> **Any time the agent's entropy drops *below* `H(q)`, that is not forecasting and it is not
> exploitation — there is no payoff reason to be sharper than the truth. It can only be the agent
> paying score to gather information.** That downward dip is a clean fingerprint of looking.

We know `q` each game (we drew it), so we know `H(q)`, and we watch the within-game entropy. Prediction:
under imperfect monitoring the entropy should **dip below `H(q)`** early (over-sharpen to identify the
opponent) and then **relax up to `H(q)`** (settle into the honest forecast); under full observation
there's no reason to dip, so entropy should approach `H(q)` *from above*. (Code: `rps_forecast.py`,
with `α` the `Dirichlet` parameter setting how mixed `q` is — small `α` = peaky/easy to identify, large
`α` = uniform/hard.)

That is exactly what we see. With an easy-to-identify opponent (`α = 0.5`, so `H(q) = 0.66`):

| round → | 1 | 2 | 3 | 5 | 9 | 15 | settles |
|---|---|---|---|---|---|---|---|
| **imperfect** entropy | **0.38** | 0.59 | 0.73 | 0.80 | 0.84 | 0.88 | ~0.87 |
| full-obs entropy | 1.09 | 0.97 | 0.87 | 0.80 | 0.75 | 0.75 | ~0.74 |
| `H(q)` (optimal forecast) | 0.66 | — | — | — | — | — | 0.66 |

The imperfect-monitoring agent **starts far below `H(q)`** (0.38 vs 0.66) — it deliberately over-sharpens
to make its first observations legible and pin down `q` — then **relaxes upward** once it has. The
full-observation agent, needing no such trick, only ever approaches `H(q)` **from above**. Independent
noise behaves like full-obs (descends from above, no dip). Since the dip lives *below the exploitation
optimum*, it has no exploitation reading. This is the cleanest single picture of "the agent sharpens to
see."

One more thing this game taught us: when the opponent is *hard* to identify (`α = 1, 2`; `H(q)` larger),
the imperfect-monitoring agent doesn't dip — it **never learns at all**, sitting at uniform forever
(its score stuck at the value of guessing uniformly). The first step of learning — sharpen to decode —
is an immediate score loss the optimizer won't pay, so it never gets started. The trap, in its purest
form, doesn't just cap performance; it can block learning entirely. It is rescuable by a curriculum
(let the agent first learn on an easy, peaky opponent where sharp-and-honest coincide, then move it to
the hard one), which tells us the trap is as much about the *optimization landscape* as about the
equilibrium.

---

## 5. What it means

Put the four cuts together — full-obs rescues, independent noise (matched difficulty) does **not** trap,
forcing sharpness *causes* information, and the forecast entropy dips *below* the honest optimum — and
only one reading survives:

> **When your observations are corrupted by your own actions and you don't keep a copy of what you did,
> being deterministic is an act of perception.** Sharpening your policy is how you make your own
> contribution to the world knowable, and that is the only way to read the world through the
> efference. There is a real, measurable pull toward predictability that has nothing to do with the
> task reward and everything to do with *seeing* — and it stands in genuine tension with any pressure
> (here, an adversary) that wants you unpredictable.

And the tension has teeth. Because the only way to learn whether it's *safe* to be legible is to *be*
legible, the agent faces an exploration trap: it can leave easy reward uncollected (Result 2), and in
the starkest case fail to learn at all (the forecasting game with a hard opponent) — purely from the
cost of having to expose itself in order to look. Removing the coupling (full or independent
observation) or scaffolding past the first step (a curriculum) lifts the trap, which is itself the
proof that the coupling was the cause.

This is the cleanest handle we found on a "self-model" pressure: not a stored picture of the self, but
a *functional* demand — **know your own action, or go blind** — that shapes how random an agent is
willing to be.

---

## 6. How to reproduce

All in `~/self-models/`, run with the project venv (`~/comp_icl/.venv/bin/python`). Each run is a small
transformer, REINFORCE, no entropy bonus, `T = 40`, batch 256, ~4–5k steps; minutes on one GPU. Outputs
land in `rps_runs/` as `.json` logs (+ periodic checkpoints).

- **`rps_im.py`** — the blindfolded RPS game and trainer. Key flags:
  `--beta` (opponent mix / best-responder fraction); `--per_traj` (commit opponent to a pure type per
  game, vs. the per-turn blend); `--full_obs` (observe the opponent's move directly); `--noisy_obs ρ`
  (observe it w.p. `ρ` else random — the *independent-noise* control); `--init ckpt.pt` (warm-start, for
  the curriculum). It logs action entropy (overall, and split by opponent type under `--per_traj`) and
  payoff.
  - Phase transition: sweep `--beta`.
  - Trap: `--per_traj`, sweep `--beta`, read entropy on bias- vs best-responder-games.
  - Full-obs / independent-noise controls: add `--full_obs` or `--noisy_obs`.
  - Curriculum: train `--per_traj --beta 0` (all bias) → reuse via `--init`, then `--beta 0.5/0.8`.
- **`rps_probe.py`** — the clamp-entropy experiment: load a bias-exploiting net, force action sharpness
  `s`, measure exploitation quality and the linear decodability (`R²`) of the true bias from activations.
- **`rps_forecast.py`** — the forecasting game (proper log-score reward). Flags `--alpha` (how mixed the
  opponent `q` is), `--full_obs`, `--noisy_obs`. Logs entropy per round and `H(q)`; the quantity to watch
  is `entropy(round) − H(q)` going **negative** (the epistemic dip) under imperfect monitoring and not
  under full-obs.

Everything is small and fully synthetic, so the ground truth (realized moves, opponent moves, hidden
bias, `H(q)`) is always on hand to check the network against — which is what let us turn an intuition
into four independent, quantitative cuts.
