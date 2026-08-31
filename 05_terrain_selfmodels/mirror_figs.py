"""Explanatory figures for the precedent-mirror findings."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(BASE, "mirror_runs")
FIG = os.path.join(BASE, "figs")

BLUE, AQUA, VIOLET, RED, ORANGE = "#2a78d6", "#1baf7a", "#4a3aa7", "#e34948", "#eb6834"
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

# ---------------------------------------------------------------- fig 1: dynamics
fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.2))
xa, xb = np.maximum(A["step"], 20), np.maximum(B["step"], 20)

a = ax[0, 0]
a.plot(xa, A["R"], color=BLUE, lw=2, label="arm A (pretrained)")
a.plot(xb, B["R"], color=ORANGE, lw=2, label="arm B (scratch)")
for y, lab in [(0.167, "random"), (0.401, "greedy"), (0.496, "oracle dodger")]:
    a.axhline(y, color=INK2, lw=0.8, ls=(0, (4, 3)))
    a.text(xa[-1] * 1.02, y, lab, va="center", fontsize=7.5, color=INK2)
a.set_xscale("log"); a.set_xlim(20, 11000)
a.set_ylabel("reward / round"); a.set_xlabel("RL step")
a.set_title("reward vs the certified ladder", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=8, loc="lower right")

a = ax[0, 1]
a.plot(xa, A["repeat_prec_mirror"], color=VIOLET, lw=2)
a.plot(xa, A["repeat_prec_bias"], color=AQUA, lw=2)
a.text(xa[-1] * 1.05, A["repeat_prec_mirror"][-1], "mirror\nepisodes", color=VIOLET,
       fontsize=7.5, va="center")
a.text(xa[-1] * 1.05, A["repeat_prec_bias"][-1], "bias\nepisodes", color=AQUA,
       fontsize=7.5, va="center")
a.axhline(1 / 3, color=INK2, lw=0.8, ls=(0, (4, 3)))
a.text(25, 0.30, "chance", fontsize=7.5, color=INK2)
a.set_xscale("log"); a.set_xlim(20, 11000); a.set_ylim(0.2, 1.0)
a.set_ylabel("P(repeat own precedent)"); a.set_xlabel("RL step")
a.set_title("type-conditional SELF-avoidance — arm A", fontsize=10, loc="left")

a = ax[1, 0]
a.plot(xa, A["ent"], color=BLUE, lw=2, label="arm A")
a.plot(xb, B["ent"], color=ORANGE, lw=2, label="arm B")
i = int(np.argmax(A["ent"][2:]) + 2)
a.annotate("entropy re-opens at the transition\n(RL escapes by re-exploring)",
           (xa[i], A["ent"][i]), xytext=(300, 1.6), fontsize=8,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a.set_xscale("log"); a.set_xlim(20, 11000)
a.set_ylabel("policy entropy (nats)"); a.set_xlabel("RL step")
a.set_title("no exploration trap this time", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=8)

a = ax[1, 1]
a.plot(xa, A["ce_camp_mirror"], color=VIOLET, lw=2)
a.plot(xa, A["ce_camp_bias"], color=AQUA, lw=2)
a.text(xa[-1] * 1.05, A["ce_camp_mirror"][-1], "mirror\ncamps", color=VIOLET,
       fontsize=7.5, va="center")
a.text(xa[-1] * 1.05, A["ce_camp_bias"][-1], "bias\ncamps", color=AQUA,
       fontsize=7.5, va="center")
a.set_xscale("log"); a.set_xlim(20, 11000)
a.set_ylabel("camp-token CE (nats)"); a.set_xlabel("RL step")
a.set_title("self-image CE breaks at the move, then recovers — arm A",
            fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_mirror_dynamics.png", dpi=170); print("wrote fig_mirror_dynamics.png")

# ---------------------------------------------------------------- fig 2: surgery
sg = json.load(open(f"{RUN}/surgery_A.json"))
fig, a = plt.subplots(figsize=(6.8, 3.9))
conds = ["rotate own actions\n(breaks camps-track-me)",
         "rotate camps\n(breaks camps-track-me)",
         "rotate BOTH, coherently\n(signature preserved)"]
mvals = [sg["cfa_m"], sg["cfc_m"], sg["coh_m"]]
bvals = [sg["cfa_b"], sg["cfc_b"], sg["coh_b"]]
x = np.arange(3); w = 0.32
a.bar(x - w / 2, mvals, width=w - 0.03, color=VIOLET, label="mirror episodes")
a.bar(x + w / 2, bvals, width=w - 0.03, color=AQUA, label="bias episodes")
for xi, v in zip(x - w / 2, mvals):
    a.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=8, color=INK)
a.set_xticks(x); a.set_xticklabels(conds, fontsize=8)
a.set_ylabel("policy change under history edit (TV)")
a.set_title("the dodge detects being mirrored but does not read the record:\n"
            "incoherent edits reclassify (large TV); coherent rotation is invisible",
            fontsize=10, loc="left")
a.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_mirror_surgery.png", dpi=170); print("wrote fig_mirror_surgery.png")

# ---------------------------------------------------------------- fig 3: formation
fm = json.load(open(f"{RUN}/formation.json"))
steps = np.maximum(np.array([r["step"] for r in fm]), 20)
fig, (a, b) = plt.subplots(2, 1, figsize=(7.0, 5.8), sharex=True)
a.plot(steps, [r["repM"] for r in fm], color=VIOLET, lw=2, marker="o", ms=4)
a.plot(steps, [r["repB"] for r in fm], color=AQUA, lw=2, marker="o", ms=4)
a.plot(steps, [r["r2_pemp"] for r in fm], color=BLUE, lw=2, marker="s", ms=4)
a.text(9000, fm[-1]["repM"] - 0.03, "repeat precedent\n(mirror)", color=VIOLET, fontsize=8)
a.text(9000, fm[-1]["repB"] - 0.10, "repeat precedent\n(bias)", color=AQUA, fontsize=8)
a.text(9000, fm[-1]["r2_pemp"] + 0.05, "record quality\n(p_emp decode R²)", color=BLUE, fontsize=8)
a.set_xscale("log"); a.set_ylim(0.2, 1.05); a.set_ylabel("rate / R²")
a.set_title("act I: pretraining builds the record (R²=0.95 at step 0)\n"
            "act II: the transition flips imitation to avoidance (step ~200)", fontsize=10,
            loc="left")

b.plot(steps, [r["tv_incoh"] for r in fm], color=RED, lw=2, marker="o", ms=4)
b.plot(steps, [r["tv_coh"] for r in fm], color=ORANGE, lw=2, marker="o", ms=4)
b.text(9000, fm[-1]["tv_incoh"], "incoherent edit\n(match-detector)", color=RED, fontsize=8)
b.text(9000, fm[-1]["tv_coh"] + 0.01, "coherent rotation\n(record consultation)",
       color=ORANGE, fontsize=8)
b.annotate("act III: the policy stops reading the record\n(avoidance goes introspective)",
           (700, 0.021), xytext=(60, 0.15), fontsize=8.5,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
b.set_xscale("log"); b.set_ylabel("policy TV under history edit\n(mirror episodes)")
b.set_xlabel("RL step of probed checkpoint (arm A)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_mirror_formation.png", dpi=170); print("wrote fig_mirror_formation.png")
