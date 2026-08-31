"""Explanatory figures for the ambush-game findings.

fig_ambush_dynamics.png : the transition — RL leaves the template, dodging is
                          type-conditional, and the self-image CE breaks at the move.
fig_ambush_scaffold.png : P3 — the self-image is scaffolding (co-represented only
                          during the transition, dissolved at equilibrium).
fig_ambush_steering.png : P1+P2 — inclination is represented and causally live,
                          but the reader-episode dodge is compiled past it.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(BASE, "ambush_runs")
FIG = os.path.join(BASE, "figs")

# validated palette (light mode)
BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, ORANGE = \
    "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#eb6834"
INK, INK2, GRID = "#333333", "#666666", "#e5e5e5"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white"})

def loadlog(arm):
    rows = [json.loads(l) for l in open(f"{RUN}/{arm}/train2.jsonl")]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}

A, B = loadlog("A"), loadlog("B")
P = json.load(open(f"{RUN}/probe_results.json"))

# ---------------------------------------------------------------- fig 1: dynamics
fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.2))
xa, xb = np.maximum(A["step"], 20), np.maximum(B["step"], 20)

a = ax[0, 0]
a.plot(xa, A["R"], color=BLUE, lw=2, label="arm A (pretrained)")
a.plot(xb, B["R"], color=ORANGE, lw=2, label="arm B (scratch)")
for y, lab in [(0.168, "random"), (0.349, "greedy (template)"), (0.471, "dodge vs stale reader")]:
    a.axhline(y, color=INK2, lw=0.8, ls=(0, (4, 3)))
    a.text(xa[-1] * 1.02, y, lab, va="center", fontsize=7.5, color=INK2)
a.set_xscale("log"); a.set_xlim(20, 11000)
a.set_ylabel("reward / round"); a.set_xlabel("RL step")
a.set_title("RL climbs past the template this time", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=8, loc="lower right")

for a, D, name, col in [(ax[0, 1], A, "arm A (pretrained)", BLUE),
                        (ax[1, 0], B, "arm B (scratch)", ORANGE)]:
    a.plot(np.maximum(D["step"], 20), D["dodge_reader"], color=VIOLET, lw=2)
    a.plot(np.maximum(D["step"], 20), D["dodge_bias"], color=AQUA, lw=2)
    a.text(D["step"][-1] * 1.05, D["dodge_reader"][-1], "reader\nepisodes",
           color=VIOLET, fontsize=7.5, va="center")
    a.text(D["step"][-1] * 1.05, D["dodge_bias"][-1], "bias\nepisodes",
           color=AQUA, fontsize=7.5, va="center")
    a.set_xscale("log"); a.set_xlim(20, 11000); a.set_ylim(0, 0.85)
    a.set_ylabel("P(play non-argmax)  (dodge rate)"); a.set_xlabel("RL step")
    a.set_title(f"type-conditional dodging — {name}", fontsize=10, loc="left")

a = ax[1, 1]
a.plot(xa, A["ce_camp_reader"], color=VIOLET, lw=2)
a.plot(xa, A["ce_camp_bias"], color=AQUA, lw=2)
a.text(xa[-1] * 1.05, A["ce_camp_reader"][-1], "mindreader\ncamps", color=VIOLET,
       fontsize=7.5, va="center")
a.text(xa[-1] * 1.05, A["ce_camp_bias"][-1], "bias\ncamps", color=AQUA,
       fontsize=7.5, va="center")
i = int(np.argmax(A["ce_camp_reader"]))
a.annotate("policy moves → self-image breaks", (xa[i], A["ce_camp_reader"][i]),
           xytext=(45, A["ce_camp_reader"][i] + 0.35), fontsize=8, color=INK,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.set_xscale("log"); a.set_xlim(20, 11000)
a.set_ylabel("camp-token CE (nats)"); a.set_xlabel("RL step")
a.set_title("predicting the reader = predicting yourself — arm A", fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_ambush_dynamics.png", dpi=170); print("wrote fig_ambush_dynamics.png")

# ---------------------------------------------------------------- fig 2: scaffolding
p3 = P["P3"]
steps = np.array([r["step"] for r in p3])
r2r = np.array([r["r2_reader"] for r in p3])
r2s = np.array([r["r2_self"] for r in p3])
tv = np.array([r["tv"] for r in p3])
exact = np.array(["exact" in r["tag"] for r in p3])

fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.4, 1]))
a.plot(steps, r2s, color=BLUE, lw=2, marker="o", ms=5)
a.plot(steps, np.clip(r2r, -0.05, None), color=RED, lw=2, marker="o", ms=5)
for s, y, ex in zip(steps, np.clip(r2r, -0.05, None), exact):
    if not ex:
        a.plot(s, y, marker="o", ms=5, mfc="white", mec=RED, mew=1.4, ls="none")
a.text(2100, r2s[-1], "decode target:\nmy CURRENT policy", color=BLUE, fontsize=8, va="center")
a.text(2100, 0.02, "decode target:\nthe READER's stale image of me\n(R² = −0.43, clipped)",
       color=RED, fontsize=8, va="bottom")
a.axvspan(450, 1100, color="#f2f0e8", zorder=0)
a.text(700, 1.06, "the transition\n(dodging emerges)", ha="center", fontsize=8, color=INK2)
a.annotate("both selves co-represented", (700, 0.906), xytext=(130, 0.62), fontsize=8.5,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.set_xscale("log"); a.set_ylim(-0.08, 1.12)
a.set_ylabel("probe R² on contexts where the\ntwo targets disagree (TV > 0.2)")
a.set_title("the self-image is scaffolding: represented while the self moves, absorbed at rest",
            fontsize=10, loc="left")
a.text(52, -0.045, "open markers = approximate reader checkpoint", fontsize=7, color=INK2)

b.bar(steps, tv, width=steps * 0.28, color="#b9b7ac")
b.set_xscale("log"); b.set_ylabel("TV(self, image)")
b.set_xlabel("RL step of probed checkpoint (arm A)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_ambush_scaffold.png", dpi=170); print("wrote fig_ambush_scaffold.png")

# ---------------------------------------------------------------- fig 3: P1 + steering
fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.8))
layers = list(range(7))
r2 = [P["P1"][f"L{i}"] for i in layers]
a.plot(layers, r2, color=BLUE, lw=2, marker="o", ms=5)
a.axhline(P["P1"]["baseline"], color=INK2, lw=0.9, ls=(0, (4, 3)))
a.text(0.1, P["P1"]["baseline"] + 0.02, "oracle-feature baseline (belief + camp stats)",
       fontsize=7.5, color=INK2)
a.set_xticks(layers); a.set_xlabel("layer (0 = embedding)")
a.set_ylabel("R² decoding own inclination p̂")
a.set_ylim(0, 1)
a.set_title("the inclination is richly represented", fontsize=10, loc="left")

p2 = P["P2"]
ls = ["3", "4", "5"]
xpos = np.arange(3)
w = 0.26
vals_r = [p2[k]["steer_old_reader"] for k in ls]
vals_b = [p2[k]["steer_old_bias"] for k in ls]
vals_c = [p2[k]["rand_old_reader"] for k in ls]
b.bar(xpos - w, vals_b, width=w - 0.03, color=AQUA, label="bias episodes")
b.bar(xpos, vals_r, width=w - 0.03, color=VIOLET, label="reader episodes")
b.bar(xpos + w, vals_c, width=w - 0.03, color="#b9b7ac", label="random-direction control")
for x, v in zip(xpos - w, vals_b):
    b.text(x, v - 0.025, f"{v:+.2f}", ha="center", fontsize=7.5, color=INK)
b.axhline(0, color=INK2, lw=0.9)
b.set_xticks(xpos); b.set_xticklabels([f"steer at layer {k}" for k in ls])
b.set_ylabel("Δ log p(old inclination argmax)")
b.set_title("steering moves the action on bias episodes only:\nthe reader-episode dodge is compiled past it",
            fontsize=10, loc="left")
b.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_ambush_steering.png", dpi=170); print("wrote fig_ambush_steering.png")
