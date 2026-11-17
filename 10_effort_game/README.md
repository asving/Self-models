# The effort game

Scratch experiment begun by the self-models review session (2026-11-13), which left no notes and no
record of its run arguments. The setup and commands below were reconstructed on 2026-11-17 by
a later session from the code, the logs and a replication (see "Provenance").
Copied into `Self-models/10_effort_game/` on 2026-11-17; logs are under `results/` as .txt, weights and rollouts are not committed. Run scripts from this folder with the group-belief venv python (torch, numpy, `mps`).

## Question

Can a network trained only on the tokens it sees learn which part of the world responds to it,
and use that to act? Does it then steer by an internal belief about "which factor is me"?

## World (`world.py`)

- Three independent Mess3 processes ("factors"), x = 0.15, a = 0.6, each emitting one token in
  {0, 1, 2} per step. Episodes are T = 32 steps long.
- In each episode one factor k is chosen uniformly at random and given gain c_k = C0 = 0.8. The
  other two have gain 0. The agent is never told k.
- Each step the agent outputs, for each factor f, a token distribution p_f and an effort split
  w over the three factors (summing to 1). With probability c_f · w_f, factor f's next token is drawn
  from p_f, and its hidden state moves as if it had emitted that token: s' ~ T[x][s,·] / P(x|s).
  Otherwise the factor takes an ordinary Mess3 step.
- The reward per step is the number of factors showing token 0, so it lies between 0 and 3.
- `Oracle` is the exact Bayes filter for "which factor is active", given the agent's own (w, p).

## Network and training

- `net.py`: a 4-layer transformer (d = 128, 4 heads). Its input is the three tokens of the
  previous step. It has two heads: token logits (3×3), which serve both as the prediction and as the
  push distribution p, and effort logits (3). The effort head starts at zero, so w starts uniform.
  The agent's past efforts are not fed back. They are a deterministic function of what it has seen.
- `pretrain.py` → `pre.pt`: next-token prediction on gain-free episodes (3000 steps). It reaches the
  exact-filter floor.
- `post.py STEPS LAM TAG ENT` → `post_TAG.{pt,json}`: on-policy post-training.
  - The loss is −(advantage · log-likelihood of the realised tokens) + LAM · prediction CE − ENT · entropy(w).
  - The advantage is the discounted return (γ = 0.9) minus the batch mean.
  - The log-likelihood is the exact law of the realised token, log[(1 − c w) P(x|s) + c w p(x)],
    which is differentiable in both w and p.
  - Batch 256, AdamW at 3e-4. An evaluation every 250 steps uses 2000 episodes against the oracle.
- `analyze.py TAG` → a behavioural and probe report on 4000 episodes (log in `analyze_TAG.log`).

## Reference policies (`python world.py` → `baselines.log`; exact oracle, push p = (.98,.01,.01))

| C0 = 0.8, T = 32 | reward/step | median steps to 0.9 belief |
|---|---|---|
| agnostic (w uniform) | 1.188 | 26 |
| soft (w = posterior) | 1.359 | 16 |
| greedy (all effort on the posterior's best bet) | 1.472 | 6 |
| informed (knows k) | 1.530 | 4 |

The untouched rate is 1.00 (each factor shows 0 a third of the time).

## Runs

| tag | command | notes |
|---|---|---|
| a | `post.py 4000 1.0 a` (ENT = 0) | complete (2026-11-13) |
| b | `post.py 4000 1.0 b 0.05` | killed at step 250 on 2026-11-13; rerun in full 2026-11-17 (21 min on `mps`) |
| bsearch_* | `find_b.sh` | 250-step runs used to identify run b's arguments |

Originals from 2026-11-13 are in `orig/`.

## Provenance

- **Run a.** The command is inferred, not recorded. `arepro` (`post.py 250 1.0 arepro`) reproduces
  run a's step-250 evaluation to 3 decimals (reward 1.1291 vs 1.1285, wmax 0.9952 vs 0.9951). MPS is
  not bit-deterministic.
- **Run b.** ENT was found by matching run b's step-250 evaluation (`find_b.log`). ENT = 0.05 gives
  reward 1.0308 vs 1.0307, wmax 0.4024 vs 0.4023, and corr 0.479 vs 0.479. Neighbouring values
  (0.03, 0.07, 0.1) are clearly off. LAM = 1.0 is assumed, not searched.
- `post.py` was modified at 14:43, after run a finished (14:42). The replication above shows the edit
  did not change behaviour at ENT = 0, but the diff is unknown.

## Results (as of 2026-11-17)

The numbers below come from the step-4000 evaluation in `post_TAG.json` and from `analyze.py`
(4000 fresh episodes, logs in `analyze_TAG.log`). For comparison, the reference policies score
1.19 (even split), 1.36 (effort follows the posterior), 1.47 (greedy) and 1.53 (informed).

| | run a (ENT = 0) | run b (ENT = 0.05) |
|---|---|---|
| reward/step (late, t ≥ 16) | 1.136 (1.144) | 1.392 (1.449) |
| effort on the true factor at t = 1, 4, 8, 16, 31 | .33 .33 .33 .33 .33 | .48 .67 .76 .86 .93 |
| oracle belief in the true factor at the same steps | .37 .44 .49 .55 .58 | .39 .56 .70 .85 .94 |
| corr(effort, oracle belief), t ≥ 4 | −0.02 | 0.90 |
| switches of the top-effort factor per episode | 0 | 3.05 |
| head P(token 0), true factor / other factors, t ≥ 16 | .51 / .36 | .88 / .35 |
| linear read-out of the oracle belief from the final layer, held-out R² | 0.995 | 0.948 |

**Run a collapses.** Every step of every episode, it puts all its effort on factor z. It never
switches, and it scores below the even split (1.14 vs 1.19). Yet its final layer encodes the exact
belief about which factor is active (R² 0.995), and its token head pushes harder on the true factor
(.51 vs .36). The belief is present but has no effect on the effort. The effort head saturated early,
at step 250 wmax was already .995, and no gradient brought it back.

**Run b acts on its belief.**
- **It is a greedy, Bayes-tracking policy.** It starts every episode on factor z and keeps putting
  nearly all its effort on one factor (wmax ≈ .99). The factor it picks is the oracle's top factor
  about as often as the oracle itself would be right (.93 vs .96 at t = 31).
- **Its effort tracks the oracle belief.** Mean effort on a factor rises with belief: .01, .09, .38,
  .82, .93 and 1.00 across the belief bins 0–.1, .1–.3, .3–.5, .5–.7, .7–.9 and .9–1.
- **Behaviourally, it is win-stay / lose-shift.** After the pushed factor shows 0, it switches with
  probability .003. After it doesn't, it switches with probability .55 in steps 0–7 and .25 in steps 8–30.
  The falling switch rate is consistent with accumulated evidence, but this test alone does not
  separate "a belief that integrates history" from "a shift rule whose rate decays with t".
- **It falls short of the exact greedy policy (1.39 vs 1.47).** Part of the gap is the push quality:
  the network's P(token 0) on the true factor is .88, while the baseline pushes at .98.

**The entropy bonus is the knob that decides the outcome.** ENT = 0 collapses. ENT = 0.01, 0.03
and 0.05 learn identity tracking. ENT = 0.07 and 0.1 had not left the even split by step 250
(`find_b.log`), but were not run longer. LAM = 1 throughout; other values are untested.

Not done:
- seeds (each run is a single seed);
- whether run b uses its belief causally (steering the read-out direction);
- the reason the first-step choice is always z;
- longer ENT = 0.01 and 0.03 runs.
