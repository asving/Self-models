"""World selection for the contingency game (exact identity posterior, CPU).

Game: N balls on a ring of R cells. Each episode one ball (hidden, uniform) is driven by
the agent's action a in {-1,0,+1}; the others are 'replays': the same informed policy
heading to their own random goals (equally purposeful). Every step a ball's intended move
is replaced, w.p. SLIP, by a uniform draw from {-1,0,+1}. The agent sees all positions
(never which ball it drives) and has a private goal. Reward = controlled ball AT goal.

Agents (all deterministic; randomness is the world's):
  informed  knows which ball is controlled
  live      exact Bayes posterior over the controlled ball (marginalizing the replays'
            goals), acts to maximize posterior-expected one-step distance reduction
  hedge     same rule with the posterior frozen at uniform (identity-free)
  nearest   steers by the ball nearest the goal (identity-free convention)
Outputs: occupancy per agent, identity premium (live - best identity-free), and the
live agent's detection time (first step with posterior on the true ball >= .9).
"""
import json, itertools
import numpy as np

R, T, RUNS = 8, 32, 20000
STEPS = np.array([-1, 0, 1])

def ring_dist(a, b):
    d = np.abs(a - b) % R
    return np.minimum(d, R - d)

def intend(pos, goal):
    d = (goal - pos) % R
    return np.where(d == 0, 0, np.where(d <= R // 2, 1, -1))

def run(N, SLIP, agent, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.integers(R, size=(RUNS, N))
    goals_other = rng.integers(R, size=(RUNS, N))
    goal = rng.integers(R, size=RUNS)
    k = rng.integers(N, size=RUNS)                       # controlled ball
    ar = np.arange(RUNS)
    ll_ctrl = np.zeros((RUNS, N))                        # log P(steps | ball is mine)
    ll_oth = np.zeros((RUNS, N, R))                      # log P(steps | replay with goal g)
    occ = np.zeros(RUNS); detect = np.full(RUNS, T)
    G = np.arange(R)
    for t in range(T):
        # posterior over controlled ball
        lo = ll_ctrl - np.logaddexp.reduce(ll_oth, axis=2) + np.log(R)  # replay goal prior 1/R
        post = np.exp(lo - lo.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
        hit = (post[ar, k] >= 0.9) & (detect == T); detect[hit] = t
        if agent == 'informed':
            a = intend(pos[ar, k], goal)
        elif agent == 'nearest':
            j = ring_dist(pos, goal[:, None]).argmin(1); a = intend(pos[ar, j], goal)
        else:
            w = post if agent == 'live' else np.full((RUNS, N), 1 / N)
            d0 = ring_dist(pos, goal[:, None])
            gain = np.stack([(w * (d0 - ring_dist((pos + s) % R, goal[:, None]))).sum(1)
                             for s in STEPS], 1)
            a = STEPS[gain.argmax(1)]                    # ties -> -1 first; fine
        # intended moves: controlled ball gets a, replays head to their goals
        intended = intend(pos, goals_other)
        intended[ar, k] = a
        slip = rng.random((RUNS, N)) < SLIP
        step = np.where(slip, rng.integers(-1, 2, size=(RUNS, N)), intended)
        # likelihood updates
        p = lambda inten: np.where(step == inten, 1 - SLIP + SLIP / 3, SLIP / 3)
        ll_ctrl += np.log(p(np.broadcast_to(a[:, None], (RUNS, N))))
        int_g = intend(pos[:, :, None], G[None, None, :])
        ll_oth += np.log(np.where(step[:, :, None] == int_g, 1 - SLIP + SLIP / 3, SLIP / 3))
        pos = (pos + step) % R
        occ += (pos[ar, k] == goal)
    return occ.mean() / T, float(np.median(detect)), float((detect < T).mean())

res = []
for N, SLIP in itertools.product((2, 3, 4), (0.1, 0.3, 0.5)):
    row = dict(N=N, slip=SLIP)
    for ag in ('informed', 'live', 'hedge', 'nearest'):
        o, med, frac = run(N, SLIP, ag)
        row[ag] = round(o, 3)
        if ag == 'live': row['detect_median'] = med; row['detect_frac'] = round(frac, 3)
    row['premium_live_vs_best_free'] = round(row['live'] - max(row['hedge'], row['nearest']), 3)
    row['premium_informed_vs_best_free'] = round(row['informed'] - max(row['hedge'], row['nearest']), 3)
    res.append(row); print(row, flush=True)
json.dump(res, open('results/contingency_select.json', 'w'), indent=1)
