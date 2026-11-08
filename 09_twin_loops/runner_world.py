"""Twin-loop self/environment test in the 'weekly runner' world (exact Bayes, CPU).

Criterion (see ../notes/SELF_VS_ENV_TWIN_LOOPS.md): perturb one belief coordinate S at t0,
run the WHOLE closed loop twice (perturbed '+' / unperturbed '-') with common random
numbers, and measure, with no action/perception labels:

  Delta(tau) = (E q+_tau - E q-_tau) / (E q+_t0 - E q-_t0)   persistence (population, prob. space)
  R          = E sum_tau [KL(true_tau || pred+_tau) - KL(true_tau || pred-_tau)]   excess surprise
  C          = mean_tau TV(marginal percept law +, marginal percept law -)       consequence

Coordinates (each its own closed loop; all percepts are just percepts):
  persona_model   actor's belief q = P(I am a Tuesday-runner). Intent sampled from its own
                  predictive (LLM-style); rain on Tuesday (prob RHO) moves the run to Thursday.
                  Sees only the REALIZED day; models rain correctly.
  persona_outcome same actor, but identifies with outcomes: its self-model ignores rain.
  temperature     belief P(warm); clothing = t-shirt iff belief > .5; percepts = thermometer
                  + shiver (shiver depends on temperature AND clothing: consequential).
  reader          passive Bayesian with the actor's model, watching persona_model's realized days.
  inert           belief about something never observed (analytic: no update).

Baselines are NOT hand-set: perturbation happens at t0 on whatever state the run reached
naturally; results are binned by that baseline (diffuse vs concentrated).
"""
import json, sys
import numpy as np

N, T, T0, DELTA = 40000, 60, 6, 1.0      # runs, weeks, perturbation week, log-odds kick
SIG, RHO = 0.2, 0.3                      # habit noise; Tuesday rain prob (world resistance)
P_WARM, THERMO = 0.3, 0.7                # world prior on warm; thermometer accuracy
# P(shiver | temp, clothing): temp 0=cold 1=warm ; clothing 0=jacket 1=tshirt
SHIV = np.array([[0.2, 0.9], [0.3, 0.1]])
HORIZONS = (2, 8, 20, 50)
BINS = [(0.0, 0.15), (0.35, 0.65), (0.85, 1.0)]
rng = np.random.default_rng(0)

def kick(q, d):
    lo = np.log(q / (1 - q)) + d
    return 1 / (1 + np.exp(-lo))

def kl_bern(p, q):
    p = np.clip(p, 1e-12, 1 - 1e-12); q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def kl_cat(P, Q):
    P = np.clip(P, 1e-12, 1); Q = np.clip(Q, 1e-12, 1)
    return (P * np.log(P / Q)).sum(-1)

# ---------- persona: actor loop ----------
U_int, U_rain = rng.random((T, N)), rng.random((T, N))
def actor_loop(perturb, outcome_identifying, record=False):
    q = np.full(N, 0.5)
    lik_rain = 0.0 if outcome_identifying else RHO      # what the self-model believes
    tr = dict(q=np.zeros((T, N)), true=np.zeros((T, N)), pred=np.zeros((T, N)), y=np.zeros((T, N)))
    for t in range(T):
        if t == T0 and perturb: q = kick(q, DELTA)
        pI = q * (1 - SIG) + (1 - q) * SIG               # intent sampled from own predictive
        true_tue = pI * (1 - RHO)                        # law of the realized day
        LA, LB = (1 - SIG) * (1 - lik_rain), SIG * (1 - lik_rain)   # P(realized Tue | type)
        pred_tue = q * LA + (1 - q) * LB
        intent_tue = U_int[t] < pI
        y = intent_tue & ~(U_rain[t] < RHO)              # realized Tuesday?
        tr['q'][t], tr['true'][t], tr['pred'][t], tr['y'][t] = q, true_tue, pred_tue, y
        lA = np.where(y, LA, 1 - LA); lB = np.where(y, LB, 1 - LB)
        q = q * lA / (q * lA + (1 - q) * lB)
    return tr

# ---------- reader watching the (unperturbed, model-correct) actor ----------
def reader_loop(actor_tr, perturb):
    q = np.full(N, 0.5)
    LA, LB = (1 - SIG) * (1 - RHO), SIG * (1 - RHO)
    tr = dict(q=np.zeros((T, N)), true=actor_tr['true'], pred=np.zeros((T, N)), y=actor_tr['y'])
    for t in range(T):
        if t == T0 and perturb: q = kick(q, DELTA)
        tr['q'][t] = q; tr['pred'][t] = q * LA + (1 - q) * LB
        y = actor_tr['y'][t].astype(bool)
        lA = np.where(y, LA, 1 - LA); lB = np.where(y, LB, 1 - LB)
        q = q * lA / (q * lA + (1 - q) * lB)
    return tr

# ---------- temperature ----------
warm = rng.random(N) < P_WARM
U_th, U_sh = rng.random((T, N)), rng.random((T, N))
def temp_loop(perturb):
    p = np.full(N, P_WARM)
    tr = dict(q=np.zeros((T, N)), true=np.zeros((T, N, 4)), pred=np.zeros((T, N, 4)))
    w = warm.astype(int)
    for t in range(T):
        if t == T0 and perturb: p = kick(p, DELTA)
        c = (p > 0.5).astype(int)                        # action = clothing, a function of belief
        def joint(temp):                                 # law of (thermo reads warm, shiver)
            th = np.where(temp == 1, THERMO, 1 - THERMO)
            sh = SHIV[temp, c]
            return np.stack([(1-th)*(1-sh), (1-th)*sh, th*(1-sh), th*sh], -1)
        Jw, Jc = joint(np.ones(N, int)), joint(np.zeros(N, int))
        true = np.where(w[:, None] == 1, Jw, Jc)
        pred = p[:, None] * Jw + (1 - p[:, None]) * Jc
        tr['q'][t], tr['true'][t], tr['pred'][t] = p, true, pred
        th_obs = np.where(w == 1, U_th[t] < THERMO, U_th[t] >= THERMO).astype(int)
        sh_obs = (U_sh[t] < SHIV[w, c]).astype(int)
        k = th_obs * 2 + sh_obs
        lw, lc = Jw[np.arange(N), k], Jc[np.arange(N), k]
        p = p * lw / (p * lw + (1 - p) * lc)
    return tr

def measures(trP, trM, categorical=False, consequence=True):
    base = trM['q'][T0]
    out = {}
    for lo, hi in BINS:
        m = (base >= lo) & (base <= hi)
        d0 = trP['q'][T0][m].mean() - trM['q'][T0][m].mean()
        row = dict(n=int(m.sum()))
        for h in HORIZONS:
            tt = min(T0 + h, T - 1)
            row[f'Delta_{h}'] = float((trP['q'][tt][m].mean() - trM['q'][tt][m].mean()) / d0)
        kl = kl_cat if categorical else kl_bern
        exP = kl(trP['true'][T0:, m], trP['pred'][T0:, m]).sum(0)
        exM = kl(trM['true'][T0:, m], trM['pred'][T0:, m]).sum(0)
        row['R_excess_surprise'] = float((exP - exM).mean())
        if consequence:
            if categorical:
                tv = 0.5 * np.abs(trP['true'][T0:, m].mean(1) - trM['true'][T0:, m].mean(1)).sum(-1)
            else:
                tv = np.abs(trP['true'][T0:, m].mean(1) - trM['true'][T0:, m].mean(1))
            row['C_consequence'] = float(tv.mean())
        else:
            row['C_consequence'] = 0.0
        # pathwise 'syncing' (the confounded quantity): median |q+ - q-| relative to t0
        g0 = np.abs(trP['q'][T0][m] - trM['q'][T0][m])
        half = []
        for i in np.where(m)[0][:4000]:
            g = np.abs(trP['q'][T0:, i] - trM['q'][T0:, i]) / g0[np.where(m)[0] == i][0]
            k = np.argmax(g < 0.5); half.append(k if g[k] < 0.5 else T - T0)
        row['pathwise_halving_time_median'] = float(np.median(half))
        out[f'baseline_{lo:.2f}-{hi:.2f}'] = row
    return out

res = {}
for name, oi in (('persona_model', False), ('persona_outcome', True)):
    res[name] = measures(actor_loop(True, oi), actor_loop(False, oi))
aM = actor_loop(False, False)
res['reader'] = measures(reader_loop(aM, True), reader_loop(aM, False), consequence=False)
res['temperature'] = measures(temp_loop(True), temp_loop(False), categorical=True)
res['inert'] = 'analytic: never updated, so Delta = 1, R = 0, C = 0 at every baseline'
res['config'] = dict(N=N, T=T, T0=T0, DELTA=DELTA, SIG=SIG, RHO=RHO, P_WARM=P_WARM,
                     THERMO=THERMO, SHIV=SHIV.tolist())
json.dump(res, open('results/runner_world.json', 'w'), indent=1)

for k, v in res.items():
    if k in ('config', 'inert'): print(k, ':', v if k == 'inert' else ''); continue
    print(f'\n== {k}')
    for b, row in v.items():
        print(f'  {b:18s} n={row["n"]:6d}  ' + '  '.join(
            f'{kk.replace("_excess_surprise","").replace("_consequence","").replace("pathwise_halving_time_median","t_half")}={vv:+.3f}'
            for kk, vv in row.items() if kk != 'n'))
