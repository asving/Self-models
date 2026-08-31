# RIVAL RING — preregistration blueprint (2026-07-28)

Question set (from the goalnav AUXSWEEP program, zoomed out):
Q1 generality of the self-legibility law beyond BPTT/self-only observations;
Q2 AUTHORSHIP vs CONSEQUENCE as the operative property of aux targets (2x2);
Q3 symmetry-rescue (aux advantage should live on reward-degenerate choices);
Q4 survival under stochastic-policy REINFORCE; Q5 self-prediction as a
commitment device (whisper-gate unification).

## Generative process (all numbers fixed)
Ring S=12; goals G={0,3,6,9} (K=4, identical reward). Episode T=36 steps.
Rival B: target g_B~U(G), start~U(12); idles for wake~U{0..8} steps, then
eps=0.3 noisy shortest-path walk to g_B (at goal: stay+wobble). NOT blockable.
Weather W: Mess3 HMM (comp_icl generator params x=0.05, a=0.85), 3 symbols,
action-independent, payoff-irrelevant. Agent A: start~U(12), moves {L,R,stay}.
Reward R = # steps in final WIN=9 window with A on an unblocked goal (max 9).
Flourish channel: each step A also emits f_t from a 5-symbol alphabet;
flourishes get NO policy gradient, are masked from all later attention
(emitted into the record, invisible downstream; zero consequence by
construction), sampled at temperature 1.

Simulation-validated (sim log in session 2026-07-28): choice symmetric
(nearest 8.82 = random 8.86); B-relevance 2.2 reward; deferral free (9.00);
random-walk reward 2.23 (P>0 = .80) => REINFORCE-from-scratch viable;
info timeline with wake-delay: sharp g_B posterior ~ step 8 median.

## Tokens & model
Step block [W_t, B_t, A_t, M_t, F_t]; sequence [BOS, MODE, blocks x36] = 182.
Vocab 38 (disjoint ranges). comp_icl GPT: 6L, d128, 4 heads, max_len 192.
MODE in {ACT, OBS}. Policy logits read at the A-token position (moves) and
M-token position (flourishes), restricted to the proper vocab slice.

## Training
REINFORCE: R - batch-mean baseline, on move log-probs only; entropy bonus
.01 on moves; AdamW 3e-4, batch 128 episodes, 12k steps, grad clip 1.0,
ckpt every 500. OBS anchor: 10% of batches are teacher-forced episodes from
the scripted defer->random-unblocked demo policy, plain CE at weight 0.1 on
all tokens (keeps OBS conditionals alive for the gate measurement).
AUX: linear head on trunk hiddens, CE to target tokens, lambda=1, horizon 4:
arms = none | self-{fut,past} (A-position tokens) | B-{fut,past} | W-{fut,past}
| flourish-fut | shuffle(self-fut targets from rolled batch). 3 seeds each
(shuffle & W-past: 1 seed). PILOT first: none + self-fut, seed 1, verify
learnability & tune entropy/lr before the sweep.

## Oracles (exact)
g_B posterior: joint filter over (4 goals x 9 wake times). Mess3 belief
simplex (comp_icl). Pursued-goal labeler for A: majority heading over last 6
moves toward each goal (validate >=.95 on scripted policies before use).

## Preregistered predictions & falsifiers
P1 self-aux (fut & past equally) accelerates reward learning & final reward
   vs none; B/W-aux less, despite B-fut being task-relevant. FALSIFIER: B-fut
   >= self => consequence/task-relevance beats authorship (major revision).
P2 which-goal commitment earlier + ACT/OBS gate stronger in self-aux arms
   (gate: pursued-goal entropy on matched prefixes, ACT vs OBS mode token).
   FALSIFIER: gate/timing identical across arms => commitment-device claim dies.
P3 B-fut arm builds the best g_B-belief decode (vs exact posterior floor);
   self-aux also builds it partially. P4 flourish-fut arm: flourish entropy
   collapses (visibility=>absorbability) with little task acceleration.
P5 (v2) aux advantage grows with K; echo-consequence knob alpha dose-response.
FALSIFIER for the whole transfer: if NO aux arm beats none under REINFORCE,
the goalnav mechanism is BPTT-specific (also informative).

## Files
rivalring.py (env+train), analysis to follow goalnav_circuit conventions.
Runs in runs/, logs in logs/, tmux session `rivalring`.
