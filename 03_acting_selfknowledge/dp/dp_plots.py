"""Render DP results to base64 PNGs for the artifact. Reads dp_results.json."""
import os, json, io, base64
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(BASE, "dp_results.json")))
R = d["results"]; meta = d["meta"]; T = meta["T"]
betas = sorted([float(b) for b in R.keys()])

def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig); return base64.b64encode(buf.getvalue()).decode()

imgs = {}

# 1) entropy vs round, optimal vs myopic, a few betas
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
show = [0.0, 0.2, 0.5, 1.0]
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(show)))
for c, b in zip(cmap, show):
    k = str(b)
    if k not in R: continue
    ax[0].plot(range(T), R[k]["H_by_round_opt"], color=c, label=f"β={b}")
    ax[1].plot(range(T), R[k]["H_by_round_myopic"], color=c, label=f"β={b}")
for a, t in zip(ax, ["OPTIMAL: action entropy vs round", "MYOPIC: action entropy vs round"]):
    a.axhline(np.log(3), ls=":", c="gray", lw=1, label="uniform (ln3)")
    a.set_xlabel("round t"); a.set_ylabel("action entropy (nats)"); a.set_title(t)
    a.set_ylim(0, 1.2); a.legend(fontsize=8)
imgs["ent_round"] = b64(fig)

# 2) mean entropy vs beta (opt vs myopic) + payoff vs beta
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
He = [R[str(b)]["meanH_opt"] for b in betas]
Hm = [R[str(b)]["meanH_myopic"] for b in betas]
ax[0].plot(betas, He, "o-", label="optimal")
ax[0].plot(betas, Hm, "s--", color="firebrick", label="myopic")
ax[0].axhline(np.log(3), ls=":", c="gray", label="uniform")
ax[0].set_xlabel("β (P best-responder)"); ax[0].set_ylabel("mean action entropy")
ax[0].set_title("Mean entropy vs β"); ax[0].legend(fontsize=8)
po = [R[str(b)]["payoff_opt"] for b in betas]
pm = [R[str(b)]["payoff_myopic"] for b in betas]
pu = [R[str(b)]["payoff_uniform"] for b in betas]
ax[1].plot(betas, po, "o-", label="optimal")
ax[1].plot(betas, pm, "s--", color="firebrick", label="myopic")
ax[1].plot(betas, pu, "^:", color="gray", label="uniform")
ax[1].axhline(0, c="k", lw=0.6)
ax[1].set_xlabel("β"); ax[1].set_ylabel("payoff / round"); ax[1].set_title("Payoff vs β")
ax[1].legend(fontsize=8)
imgs["vs_beta"] = b64(fig)

print(json.dumps(imgs))
json.dump(imgs, open(os.path.join(BASE, "dp_imgs.json"), "w"))
print("ok", file=__import__("sys").stderr)
