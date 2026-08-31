"""Self-cancellation figures:
 (1) over-confidence ratio vs training step (parsed from logs) -> shows the two-phase descent FROM ABOVE,
     with the closed-form prediction as a dashed target per g.
 (2) sigma_net vs sigma_honest vs step for one g -> the calibration-then-legibility two-timescale story.
 (3) a single rollout: true random walk e_t with the net's prediction mu_t +/- sigma_t band (and the
     honest Kalman band) over time, plus the (e_t, mu_t) tracking trajectory."""
import os, sys, glob, re
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
GS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
def pred_ratio(g):
    return np.sqrt(((g**2-1)+np.sqrt(1+6*g**2+g**4))/(4*g**2)) if g > 0 else 1.0
NET="#1f77b4"; HON="#888"; OPT="#d62728"
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":130})

def parse_log(g):
    fs = sorted(glob.glob(f"logs/sc_g{g}_*.log"))
    if not fs: return None
    steps, ratio, sn, sh = [], [], [], []
    for ln in open(fs[-1]):
        m = re.search(r"step\s+(\d+).*net_sigma_ss\s+([\d.]+)\s+honest_ss\s+([\d.]+)\s*\|\s*ratio\s+([\d.]+)", ln)
        if m:
            steps.append(int(m.group(1))); sn.append(float(m.group(2)))
            sh.append(float(m.group(3))); ratio.append(float(m.group(4)))
    return dict(step=np.array(steps), ratio=np.array(ratio), sn=np.array(sn), sh=np.array(sh))

# ---- Fig 1: ratio vs step, all g, with predicted targets ----
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(GS)))
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for g, c in zip(GS, cmap):
    d = parse_log(g)
    if d is None or len(d["step"]) == 0: continue
    ax.plot(d["step"], d["ratio"], "-o", color=c, ms=3, lw=1.6, label=f"g={g}")
    ax.axhline(pred_ratio(g), color=c, ls=":", lw=1, alpha=0.7)
ax.axhline(1/np.sqrt(2), color="k", ls="--", lw=1, alpha=0.5)
ax.text(ax.get_xlim()[1]*0.82, 1/np.sqrt(2)+0.005, r"$1/\sqrt{2}$ asymptote", fontsize=8.5)
ax.axhline(1.0, color="k", lw=0.6, alpha=0.4)
ax.set_xlabel("training step"); ax.set_ylabel(r"over-confidence ratio  $\sigma_{net}/\sigma_{honest}$")
ax.set_title("Descent from above: caution first, then over-confidence\n(dotted = closed-form optimum per g)", fontsize=11.5)
ax.legend(fontsize=8.5, ncol=2, loc="upper right")
fig.savefig("figs/sc_ratio_vs_step.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: sigma_net vs sigma_honest vs step (two-timescale), pick g=2.0 ----
d = parse_log(2.0)
if d and len(d["step"]):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(d["step"], d["sh"], "-s", color=HON, ms=3, lw=1.8, label=r"honest $\sigma$ (Kalman)")
    ax.plot(d["step"], d["sn"], "-o", color=NET, ms=3, lw=1.8, label=r"net $\sigma$ (reported)")
    ax.fill_between(d["step"], d["sn"], d["sh"], where=d["sn"]<d["sh"], color=OPT, alpha=0.15)
    ax.set_xlabel("training step"); ax.set_ylabel(r"steady-state $\sigma$  (g=2.0)")
    ax.set_title("Net $\\sigma$ falls toward honest, then dips below it (red = over-confidence)", fontsize=11)
    ax.legend(fontsize=9.5)
    fig.savefig("figs/sc_sigma_vs_step.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: a single rollout trajectory (needs a trained net) ----
import torch, torch.nn.functional as F
from selfcancel import RWForecastNet
def rollout_capture(path, g, sigma_eta=0.5, T=80, B=512, seed=3):
    ck = torch.load(path, map_location="cpu"); a = ck["args"]
    net = RWForecastNet(a["d_model"], a["n_layer"], a["n_head"], max(a["T"], T));
    # T may exceed trained T -> pos embedding size; reload only if fits
    net = RWForecastNet(a["d_model"], a["n_layer"], a["n_head"], a["T"]); net.load_state_dict(ck["state"]); net.eval()
    T = a["T"]; torch.manual_seed(seed)
    e = torch.randn(B); obs = torch.zeros(B, 0)
    es, mus, sigs, hon = [], [], [], []
    P = torch.ones(B)
    with torch.no_grad():
        for t in range(T):
            mu, ls = net(obs); sig = torch.exp(ls.clamp(-7, 3))
            pred_var = P + sigma_eta**2; hon.append(pred_var.sqrt())
            e = e + torch.randn(B)*sigma_eta
            eps = torch.randn(B); aa = mu + sig*eps; x = e + g*aa
            R = (g*sig)**2 + 1e-8; K = pred_var/(pred_var+R); P = (1-K)*pred_var
            es.append(e.clone()); mus.append(mu); sigs.append(sig)
            obs = torch.cat([obs, x[:, None]], 1)
    st = lambda L: torch.stack(L, 1).numpy()
    return st(es), st(mus), st(sigs), st(hon), a

gtraj = 2.0; pt = f"rps_runs/sc_g{gtraj}.pt"
if os.path.exists(pt):
    es, mus, sigs, hon, a = rollout_capture(pt, gtraj)
    i = 0; T = es.shape[1]; tt = np.arange(T)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    # time series
    ax1.plot(tt, es[i], "-", color="k", lw=2, label="true latent $e_t$")
    ax1.plot(tt, mus[i], "-", color=NET, lw=1.8, label=r"net prediction $\mu_t$")
    ax1.fill_between(tt, mus[i]-sigs[i], mus[i]+sigs[i], color=NET, alpha=0.20, label=r"net $\pm\sigma_t$")
    ax1.fill_between(tt, mus[i]-hon[i], mus[i]+hon[i], color=HON, alpha=0.0, edgecolor=HON, ls="--", lw=1.0, label=r"honest $\pm\sigma$")
    ax1.set_xlabel("time step $t$"); ax1.set_ylabel("latent value")
    ax1.set_title(f"Tracking a random walk (g={gtraj}): prediction + uncertainty band", fontsize=11)
    ax1.legend(fontsize=8.5, loc="best")
    # (e, mu) trajectory colored by time
    sc = ax2.scatter(es[i], mus[i], c=tt, cmap="viridis", s=22, zorder=3)
    lo = min(es[i].min(), mus[i].min()); hi = max(es[i].max(), mus[i].max())
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label=r"$\mu=e$ (perfect)")
    ax2.plot(es[i], mus[i], "-", color="#ccc", lw=0.8, zorder=2)
    ax2.set_xlabel("true latent $e_t$"); ax2.set_ylabel(r"net prediction $\mu_t$")
    ax2.set_title("Prediction vs truth (color = time)", fontsize=11); ax2.legend(fontsize=9)
    fig.colorbar(sc, ax=ax2, label="t")
    fig.savefig("figs/sc_trajectory.png", bbox_inches="tight"); plt.close(fig)
    print("wrote figs/sc_trajectory.png")
else:
    print(f"(trajectory skipped: {pt} not saved yet)")
print("wrote figs/sc_ratio_vs_step.png", "+ sc_sigma_vs_step.png" if d else "")
