"""Collect depth-sweep results for the HARD self-model variants and print depth->excess-MSE curves.
Reads runs/hard_sq_L*.json (cont nonlinear-obs square) and runs/hmm_L*.json (hard aliased HMM).
Excess = final action_MSE − floor. Uses the mean of the last 3 evals to reduce eval noise."""
import json, os, glob, sys
import numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))

def load(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(BASE, pattern))):
        d = json.load(open(f))
        L = d["args"]["n_layer"]; floor = d["floor"]
        log = d["log"]
        last = np.mean([e["action_mse"] for e in log[-3:]]) if len(log) >= 3 else log[-1]["action_mse"]
        rows.append((L, last, floor, last - floor, f))
    rows.sort()
    return rows

def show(name, pattern):
    rows = load(pattern)
    if not rows:
        print(f"\n[{name}] no runs found for {pattern}"); return
    floor = rows[0][2]
    print(f"\n=== {name}  (floor={floor:.4f}) ===")
    print(f"{'L':>3} {'act_MSE':>9} {'excess':>9}  {'excess/floor':>12}")
    for L, mse, fl, exc, _ in rows:
        print(f"{L:>3} {mse:>9.4f} {exc:>+9.4f}  {exc/fl:>11.1%}")
    # monotonicity check
    excs = [r[3] for r in rows]
    mono = all(excs[i] >= excs[i+1] - 1e-3 for i in range(len(excs)-1))
    print(f"shallowest L={rows[0][0]} excess={rows[0][3]:+.4f} | deepest L={rows[-1][0]} excess={rows[-1][3]:+.4f} "
          f"| drop={rows[0][3]-rows[-1][3]:+.4f} | monotone(non-increasing)={mono}")

if __name__ == "__main__":
    show("CONT nonlinear-obs square G=1.0, d_model=128", "runs/hard_sq_L*.json")
    show("CONT nonlinear-obs square G=1.0, d_model=8 (capacity-starved)", "runs/sqd8_L*.json")
    show("HARD aliased HMM N=6 n_alias=2 stay=0.9 se=0.3, d_model=128", "runs/hmm_L*.json")
