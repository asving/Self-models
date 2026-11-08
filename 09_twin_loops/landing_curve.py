"""Landing curve: E[final q] vs installed q, for both persona self-models (runner world)."""
import json, numpy as np
SIG, RHO, N, T = 0.2, 0.3, 20000, 300
rng = np.random.default_rng(1)
grid = np.linspace(0.02, 0.98, 25)
out = {}
for name, lik_rain in (('models_rain', RHO), ('outcome_identifying', 0.0)):
    LA, LB = (1-SIG)*(1-lik_rain), SIG*(1-lik_rain)
    ends = []
    for q0 in grid:
        q = np.full(N, q0)
        for t in range(T):
            pI = q*(1-SIG) + (1-q)*SIG
            y = (rng.random(N) < pI) & ~(rng.random(N) < RHO)
            lA = np.where(y, LA, 1-LA); lB = np.where(y, LB, 1-LB)
            q = q*lA/(q*lA + (1-q)*lB)
        ends.append([float(q.mean()), float((q > .5).mean())])
    out[name] = dict(installed=grid.tolist(), mean_final=[e[0] for e in ends], frac_tue=[e[1] for e in ends])
json.dump(out, open('results/landing_curve.json', 'w'), indent=1)
for name, d in out.items():
    print(name)
    for q0, m in zip(d['installed'][::3], d['mean_final'][::3]): print(f'  installed {q0:.2f} -> final {m:.3f}')
