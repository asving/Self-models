import os, sys, json, glob, re
os.environ["CUDA_VISIBLE_DEVICES"]=""            # force CPU (GPUs are another user's)
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/comp_icl")); torch.set_num_threads(16)
import eval_big as E   # DEV resolves to cpu since CUDA hidden
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
print("device", E.DEV)
R=json.load(open("dp_results.json"))["results"]; dp={float(k):R[k] for k in R}
res={}
for b in [0.1,0.2,0.3,0.4,0.5]:
    f=f"rps_runs/rpsanneal_b{b}.pt"
    if not os.path.exists(f): continue
    net,a=E.load(f); pf=E.payoffs(net,b,a["T"],B=4000)
    res[b]=dict(pay=float(pf["pay"]),bias=float(pf["pay_bias"]),br=float(pf["pay_br"]),H=float(pf["H"]))
    print(f"anneal b={b}: pay {pf['pay']:+.3f} bias {pf['pay_bias']:+.3f} BR {pf['pay_br']:+.3f} H {pf['H']:.2f}",flush=True)
json.dump(res,open("figs/anneal_eval.json","w"),indent=2)
ab=sorted(res)
cold=json.load(open("figs/eval_big.json")); cb=sorted(float(k) for k in cold)
db=sorted(b for b in dp if b<=0.55)
fig,ax=plt.subplots(figsize=(7,4.6)); ax.axhline(0,color="#ccc",lw=0.8)
ax.plot(db,[dp[b]["payoff_opt"] for b in db],"--",color="#d62728",lw=2,label="Bellman optimal (DP)")
ax.plot(cb,[cold[f"{b:g}" if f"{b:g}" in cold else str(b)]["pay"] for b in cb],"-o",color="#7f7f7f",lw=1.8,ms=7,label="cold-start 6L net")
ax.plot(ab,[res[b]["pay"] for b in ab],"-o",color="#1f77b4",lw=2.4,ms=8,label="graded β-anneal (overall)")
ax.plot(ab,[res[b]["bias"] for b in ab],"-^",color="#2ca02c",lw=1.6,ms=6,label="anneal, bias-games")
ax.plot(ab,[res[b]["br"] for b in ab],"-v",color="#ff7f0e",lw=1.6,ms=6,label="anneal, BR-games")
ax.set_xlabel(r"$\beta$ = P(opponent is best-responder)"); ax.set_ylabel("reward / round")
ax.set_title("Graded β-anneal on the 6L RPS net: reward vs β",fontsize=12.5)
ax.legend(fontsize=9); ax.grid(alpha=0.25)
for s in ["top","right"]: ax.spines[s].set_visible(False)
fig.savefig("figs/anneal_reward_vs_beta.png",dpi=130,bbox_inches="tight"); print("wrote figs/anneal_reward_vs_beta.png")
