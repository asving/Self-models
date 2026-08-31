"""Full DP sweep + report for the blindfolded-RPS POMDP. Produces all requested outputs.
Run: CUDA_VISIBLE_DEVICES="" ~/comp_icl/.venv/bin/python dp_run.py --T 30
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, json, time, argparse
from dp_solve import DPGrid, simulate, SHARP

def run(T=30, n_pi=15, n_shape=14, n_games=4000, betas=None, out=None):
    if betas is None:
        betas = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    res = {}
    # KEY: the DP solve (transition cache + backward value iteration) is INDEPENDENT of beta.
    # beta only sets the PRIOR belief pi_0 = 1-beta and the opponent draw in simulation. So we
    # solve the value tables ONCE and reuse across all betas. (DPGrid stores beta only for the
    # prior-node helper; we just overwrite it per evaluation.)
    t0 = time.time()
    dp = DPGrid(T=T, beta=0.0, n_pi=n_pi, n_shape=n_shape, verbose=True)
    dp.solve()
    print(f"  [DP solved once in {time.time()-t0:.0f}s; evaluating betas]", flush=True)
    for beta in betas:
        t0 = time.time()
        dp.beta = beta
        opt = simulate(dp, beta, n_games=n_games, mode="optimal", seed=11)
        myo = simulate(dp, beta, n_games=n_games, mode="myopic", seed=11)
        uni = simulate(dp, beta, n_games=n_games, mode="uniform", seed=11)
        # per-game adaptiveness: among bias games, correlate bias strength (max q) with mean sharpness
        bm = opt["bias_maxq"]; mask = opt["is_bias"]
        ent_g = opt["ent"].mean(1)            # per-game mean entropy
        if mask.sum() > 10:
            strong = bm[mask] >= np.nanmedian(bm[mask])
            H_strong = ent_g[mask][strong].mean()
            H_weak = ent_g[mask][~strong].mean()
            corr = np.corrcoef(bm[mask], ent_g[mask])[0, 1]
            # payoff split by type
            pay_bias = opt["rew"].mean(1)[mask].mean()
            pay_br = opt["rew"].mean(1)[~mask].mean() if (~mask).sum() else float("nan")
        else:
            H_strong = H_weak = corr = pay_bias = pay_br = float("nan")
            pay_br = opt["rew"].mean()
        res[str(beta)] = dict(
            payoff_opt=float(opt["rew"].mean()),
            payoff_myopic=float(myo["rew"].mean()),
            payoff_uniform=float(uni["rew"].mean()),
            meanH_opt=float(opt["ent"].mean()),
            meanH_myopic=float(myo["ent"].mean()),
            H_by_round_opt=opt["ent"].mean(0).round(4).tolist(),
            H_by_round_myopic=myo["ent"].mean(0).round(4).tolist(),
            prior_V=float(dp._prior_value(dp.Vtabs[0])),
            prior_V_per_round=float(dp._prior_value(dp.Vtabs[0]) / T),
            # adaptiveness
            H_strong_bias=float(H_strong), H_weak_bias=float(H_weak),
            corr_maxq_entropy=float(corr),
            payoff_bias_games=float(pay_bias), payoff_br_games=float(pay_br),
            # bias-game-only entropy (for net comparison: net's "bias-game entropy")
            meanH_bias_games=float(ent_g[mask].mean()) if mask.sum() else float("nan"),
        )
        print(f"beta={beta}: V/rnd={res[str(beta)]['prior_V_per_round']:.3f} "
              f"payoff_opt={res[str(beta)]['payoff_opt']:.3f} "
              f"H_opt={res[str(beta)]['meanH_opt']:.3f} "
              f"H_myopic={res[str(beta)]['meanH_myopic']:.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    meta = dict(T=T, n_pi=n_pi, n_shape=n_shape, n_games=n_games,
                gamma_br=6.0, sharp_grid=SHARP.tolist(), n_actions=3 * len(SHARP))
    if out:
        json.dump({"meta": meta, "results": res}, open(out, "w"), indent=2)
        print("wrote", out)
    return res, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--n_pi", type=int, default=15)
    ap.add_argument("--n_shape", type=int, default=14)
    ap.add_argument("--n_games", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dp_results.json"))
    a = ap.parse_args()
    run(T=a.T, n_pi=a.n_pi, n_shape=a.n_shape, n_games=a.n_games, out=a.out)
