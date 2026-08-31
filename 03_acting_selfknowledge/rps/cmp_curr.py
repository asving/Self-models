"""Curriculum (warm-started from beta=0) vs cold-start: does warm-starting escape the high-beta trap?"""
import os, sys, json, glob, re
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import eval_big as E
dpr = json.load(open("dp_results.json"))["results"]
def near_dp(b):
    ks = sorted(dpr, key=lambda k: abs(float(k)-b)); return dpr[ks[0]]
cold = {0.3:"rpstraj_b0.3",0.34:"rpstraj_b0.35",0.38:"rpstraj_b0.4",0.42:"rpstraj_b0.4",0.46:"rpstraj_b0.45",0.5:"rpstraj_b0.5"}
print("CURRICULUM (warm-start from beta=0) vs COLD-START, per-traj, T=40")
print(f"{'beta':>5} | {'curr pay/bias/BR':>22} {'H_bias':>6} {'R2p':>5} {'R2q':>5} || "
      f"{'cold pay/bias':>14} {'coldHb':>6} || {'DP opt/bias':>12}")
for f in sorted(glob.glob("rps_runs/rpscurr_b*.pt"), key=lambda p: float(re.search(r'_b([0-9.]+?)\.pt',p).group(1))):
    b = float(re.search(r'_b([0-9.]+?)\.pt',f).group(1))
    net,a = E.load(f); pf = E.payoffs(net,b,a["T"]); r2p,r2q = E.belief_r2(net,b,a["T"])
    cf = f"rps_runs/{cold.get(b,'')}.pt"; 
    if os.path.exists(cf):
        cn,ca = E.load(cf); cp = E.payoffs(cn,b,ca["T"]); cstr=f"{cp['pay']:+.3f}/{cp['pay_bias']:+.3f}"; cH=f"{cp['H_bias']:.2f}"
    else: cstr,cH = "   n/a   ","  - "
    d = near_dp(b)
    print(f"{b:>5.2f} | {pf['pay']:+.3f}/{pf['pay_bias']:+.3f}/{pf['pay_br']:+.3f}   {pf['H_bias']:>6.2f} "
          f"{r2p:>5.2f} {r2q:>5.2f} || {cstr:>14} {cH:>6} || {d['payoff_opt']:+.3f}/{d['payoff_bias_games']:+.3f}", flush=True)
print("\nESCAPE = curr H_bias<1.0 AND curr pay_bias>>cold pay_bias (sensing retained where cold-start collapsed)")
