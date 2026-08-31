import os, re, glob, json
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# anneal payoff/round per beta (mean of last 3 evals for stability), from training logs
anneal = {}
for b in [0.1,0.2,0.3,0.4,0.5]:
    fs = sorted(glob.glob(f"logs/rpsanneal_b{b}.log"))
    if not fs: continue
    ps=[float(m) for m in re.findall(r"payoff/round\s+([+-][\d.]+)", open(fs[-1]).read())]
    if ps: anneal[b]=float(np.mean(ps[-3:]))
ab=sorted(anneal)

# cold-start big net (per-traj, from figs/eval_big.json) overall payoff
cold={}
if os.path.exists("figs/eval_big.json"):
    eb=json.load(open("figs/eval_big.json"))
    for k,v in eb.items(): cold[float(k)]=v["pay"]
cb=sorted(cold)

# DP optimal
R=json.load(open("dp_results.json"))["results"]
dp={float(k):R[k]["payoff_opt"] for k in R}
db=sorted(b for b in dp if b<=0.55)

fig,ax=plt.subplots(figsize=(7,4.6))
ax.axhline(0,color="#ccc",lw=0.8)
ax.plot([b for b in db], [dp[b] for b in db], "--", color="#d62728", lw=2, label="Bellman optimal (DP)")
ax.plot(cb,[cold[b] for b in cb],"-o",color="#7f7f7f",lw=1.8,ms=7,label="cold-start 6L net (from scratch)")
ax.plot(ab,[anneal[b] for b in ab],"-o",color="#1f77b4",lw=2.4,ms=8,label="graded β-anneal (warm-start chain)")
ax.set_xlabel(r"$\beta$ = P(opponent is best-responder)"); ax.set_ylabel("reward / round")
ax.set_title("Graded β-anneal escapes the high-β trap",fontsize=12.5)
ax.legend(fontsize=10); ax.grid(alpha=0.25)
for s in ["top","right"]: ax.spines[s].set_visible(False)
fig.savefig("figs/anneal_reward_vs_beta.png",dpi=130,bbox_inches="tight")
print("anneal:",{b:round(anneal[b],3) for b in ab})
print("cold  :",{b:round(cold[b],3) for b in cb})
print("dp    :",{b:round(dp[b],3) for b in db})
print("wrote figs/anneal_reward_vs_beta.png")
