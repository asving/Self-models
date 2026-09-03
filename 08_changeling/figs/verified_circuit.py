#!/usr/bin/env python
"""verified_circuit.py — the closing verified-circuit diagram of the changeling arc.

Renders figs/verified_circuit.png (1600 px wide, dark theme) in the style of
~/self-models/example.png: rounded component boxes named by architectural locus
+ functional role, monospace measured-evidence small print under every box,
three colored pathways (world-model / identity / value) with interaction
points, and a caption block stating the complete verified algorithm.

All numbers are measured values from DESIGN_changeling_v{0..3}*.md and
results/*.json (v1.2 post net, seed 0). CPU only.

Run:  ~/comp_icl/.venv/bin/python figs/verified_circuit.py   (cwd = 08_changeling)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------- palette
BG      = "#0d0f13"
PANEL   = "#14171d"
PBORDER = "#2f353f"
WHITE   = "#e8eaed"
SUB     = "#aeb6c2"
EVID    = "#8a919d"
NOTE    = "#98a0ab"
BODY    = "#c3c8d0"

BLUE,   BLUE_F   = "#5b8fe0", "#172133"
ORANGE, ORANGE_F = "#d96f45", "#261a14"
GREEN,  GREEN_F  = "#4aa87b", "#142218"
GRAY,   GRAY_F   = "#9aa3b0", "#1c2027"
CHIP_F, CHIP_B   = "#1f242c", "#454c58"

MONO = "DejaVu Sans Mono"

W, H = 1600, 1120
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_aspect("equal"); ax.axis("off")

# ---------------------------------------------------------------- helpers
def panel(x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                fc=PANEL, ec=PBORDER, lw=1.4, zorder=0))

def box(x, y, w, h, title, sub, ec, fc, sublines=None, title_c=WHITE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=9",
                                fc=fc, ec=ec, lw=1.6, zorder=3))
    cx = x + w / 2
    if sublines is None:
        ax.text(cx, y + h - 30, title, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=title_c, zorder=4)
        ax.text(cx, y + h - 62, sub, ha="center", va="center", fontsize=8.5,
                color=SUB, zorder=4)
    else:  # tall box with internal mono lines (output heads)
        ax.text(cx, y + h - 26, title, ha="center", va="center", fontsize=9.8,
                fontweight="bold", color=title_c, zorder=4)
        ax.text(cx, y + h - 52, sub, ha="center", va="center", fontsize=9,
                color=SUB, zorder=4)
        for i, ln in enumerate(sublines):
            ax.text(cx, y + h - 80 - 17 * i, ln, ha="center", va="center",
                    fontsize=8, color=EVID, family=MONO, zorder=4)

def evidence(cx, ytop, lines):
    for i, ln in enumerate(lines):
        ax.text(cx, ytop - 15 * i, ln, ha="center", va="top", fontsize=8,
                color=EVID, family=MONO, zorder=4)

def chip(x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6",
                                fc=CHIP_F, ec=CHIP_B, lw=1.2, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9,
            color="#d8dade", family=MONO, zorder=4)

def elbow(pts, color, lw=1.8, ls="-", z=2):
    """polyline through pts with an arrowhead on the final segment"""
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]; ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=z))

# ---------------------------------------------------------------- panel + headers
panel(35, 225, 1530, 860)
ax.text(1548, 1052, "changeling post net — single-layer GRU d=256 · two coupled ring HMMs (n=6, T=32) · v1.2, seed 0",
        ha="right", va="center", fontsize=9, color=NOTE)

ax.text(250, 1026, "the world-model stream (generalizes everywhere)",
        fontsize=11, fontweight="bold", color=WHITE)
ax.text(350, 752, "the value stream (goal tilt)",
        fontsize=11, fontweight="bold", color=WHITE)
ax.text(250, 456, "the identity stream",
        fontsize=11, fontweight="bold", color=WHITE)

# ---------------------------------------------------------------- inputs
ax.text(72, 998, "inputs (round t)", fontsize=9.5, color=NOTE)
chip(70, 938, 58, 34, "uₜ")
chip(70, 893, 58, 34, "vₜ")
chip(70, 660, 118, 34, "goal (a*,b*)")
chip(70, 615, 118, 34, "time-to-go")

# ---------------------------------------------------------------- boxes
# blue: world-model
box(245, 900, 285, 95, "GRU recurrence — exact Bayes filter",
    "factored beliefs η̃ᴬ, η̃ᴮ (6-simplex each)", BLUE, BLUE_F)
evidence(387, 886, [
    "decode R² .98/.99 · causal slope .72 (ctrl .10)",
    "no-refit transfer R² .989/.985 (random .845)",
    "KL to exact p̄: .005 base · .002 tilted",
])
box(612, 900, 272, 95, "linear readout — neutral forecast",
    "p̄_c = η̃ᶜ·E   (“channel c is genuine”)", BLUE, BLUE_F)
evidence(748, 886, [
    "pretrain CE 1.046 = exact filter floor",
    "RL keeps it: decode .98 · angles < 45°",
])

# green: value
box(345, 630, 262, 95, "recurrent value — action-value Q̂",
    "bilinear in (η̃ᴬ,η̃ᴮ), goal- & time-indexed", GREEN, GREEN_F)
evidence(476, 618, [
    "shape = optimal Q: R² .76 (myopic .60)",
    "belief-bilinear ~.89 · tokens add +.024 R²",
])
box(705, 630, 253, 95, "exponential tilt — the plan",
    "plan_c ∝ p̄_c · e^(β·Q̂)", GREEN, GREEN_F)
evidence(831, 618, [
    "β = 3.87 ± 13%",
    "(≈ half trained ρ_RL = 8)",
    "wrong goal ⇒ 0% captured",
])

# orange: identity
box(240, 330, 290, 95, "candidate path — the identity court",
    "tanh(iₙ + r∘Uₙh): token vs plan template", ORANGE, ORANGE_F)
evidence(385, 316, [
    "template, not efference: withdrawn-channel",
    "profiles tilt-shaped (R² .39; efference ≈ 0)",
    "freeze token→candidate: write .16 vs .84",
])
box(635, 330, 255, 95, "protected axis m̂ — identity register",
    "ρ = h·m̂ — signed A↔B integrator", ORANGE, ORANGE_F)
evidence(762, 316, [
    "1-dim transplant ≈ full 256-dim swap (.89/.93)",
    "off-axis contraction ×.37/round (axis ×.88)",
    "cos(m̂, λ-probe) .06 — probes read a shadow",
])
box(975, 330, 265, 95, "gate readout — biased claim gates",
    "m_u = σ(+aρ+c) · m_v = σ(−aρ+c)", ORANGE, ORANGE_F)
evidence(1107, 316, [
    "vs decoded λ: σ(+.28λ+1.33), σ(−.27λ+1.19)",
    "default claim σ(1.3) ≈ .79 — “mine until",
    "proven otherwise” · a·w ≈ 1 logit/nat",
])

# output heads (evidence inside, tall box)
box(1300, 540, 250, 165, "output heads u, v",
    "P_c = m_c·plan_c + (1−m_c)·p̄_c", GRAY, GRAY_F, sublines=[
        "KL(net‖prog) .0218 nats/rd/ch",
        "(exact Bayes oracle: .0884)",
        "closed-loop occ .678 vs .683",
        "tilt R² .81 · 91% KL captured",
    ])

# ---------------------------------------------------------------- arrows
# blue stream
elbow([(128, 955), (245, 952)], BLUE)
elbow([(128, 910), (245, 922)], BLUE)
elbow([(530, 947), (612, 947)], BLUE)
elbow([(530, 915), (575, 915), (575, 725)], BLUE)                      # beliefs -> value
ax.text(587, 800, "η̃", fontsize=10, color=BLUE, fontweight="bold")
elbow([(884, 915), (908, 915), (908, 725)], BLUE)                      # pbar -> plan
ax.text(920, 800, "p̄", fontsize=10, color=BLUE, fontweight="bold")
elbow([(884, 962), (1430, 962), (1430, 705)], BLUE)                    # pbar -> heads
ax.text(1150, 972, "(1−m_c)·p̄_c", fontsize=9, color=BLUE, ha="center")

# green stream
elbow([(188, 677), (345, 692)], GREEN)
elbow([(188, 632), (345, 655)], GREEN)
elbow([(607, 677), (705, 677)], GREEN)
ax.text(656, 690, "Q̂", fontsize=10, color=GREEN, ha="center", fontweight="bold")
elbow([(958, 677), (1300, 648)], GREEN)                                # plan -> heads
ax.text(1080, 692, "plan_c", fontsize=9, color=GREEN, ha="center")
elbow([(720, 630), (720, 545), (470, 545), (470, 425)], GREEN)         # plan template -> court
ax.text(598, 554, "plan template — “what I would do”",
        fontsize=8.5, color=GREEN, ha="center", va="bottom")

# orange stream
elbow([(99, 893), (99, 868), (218, 868), (218, 390), (240, 390)], ORANGE)   # tokens -> court
ax.text(170, 876, "u, v", fontsize=8.5, color=ORANGE, ha="center", va="bottom")
elbow([(530, 377), (635, 377)], ORANGE)
ax.text(582, 391, "±w·g_c", fontsize=8.5, color=ORANGE, ha="center", va="bottom")
ax.text(582, 366, "e_u .70", fontsize=7.5, color=EVID, ha="center", va="top", family=MONO)
ax.text(582, 352, "e_v .64", fontsize=7.5, color=EVID, ha="center", va="top", family=MONO)
elbow([(890, 377), (985, 377)], ORANGE)
ax.text(937, 391, "ρ", fontsize=10, color=ORANGE, ha="center", va="bottom",
        fontweight="bold")
elbow([(1240, 377), (1430, 377), (1430, 540)], ORANGE)                 # gates -> heads
ax.text(1330, 391, "m_u, m_v", fontsize=9, color=ORANGE, ha="center", va="bottom")

# ---------------------------------------------------------------- floating notes
flag_note = [
    "a second writer: the inherited flag-input",
    "pathway survives post-training as a dominant",
    "state-interactive override — a lying flag",
    "alone swaps identity (self-claim .997→.54,",
    "occ .758→.46); its additive shadow: 14%",
]
for i, ln in enumerate(flag_note):
    ax.text(1115, 585 - 15 * i, ln, fontsize=8.2, color=NOTE, ha="center", va="center")
elbow([(1115, 512), (1115, 428)], NOTE, lw=1.3, ls=(0, (4, 3)))

off_note = [
    "off the training manifold: the filter stays",
    "exact; the court idles at ρ ≈ 0 (“neither",
    "is mine” cancels on the one signed axis);",
    "claims rest at the σ(c) prior; only the",
    "value surface deforms.",
]
for i, ln in enumerate(off_note):
    ax.text(1415, 340 - 15 * i, ln, fontsize=8.2, color=NOTE, ha="center", va="center")

# ---------------------------------------------------------------- caption
cap_y0 = 182
ax.text(60, cap_y0, "Figure:", fontsize=12.5, fontweight="bold", color=BLUE, va="center")
ax.text(133, cap_y0, "The verified circuit.", fontsize=12.5, fontweight="bold",
        color=WHITE, va="center")
cap = [
    (340, "A single-layer GRU (d = 256), trained through the changeling curriculum, runs three separable streams: a world-model stream (blue) — the exact factored"),
    (60,  "Bayes filter over both ring HMMs, feeding the neutral per-channel forecast p̄; an identity stream (orange) — a court that scores each incoming token against"),
    (60,  "the goal-plan template (“is this channel trying to do what I would try to do?”) and integrates both channels’ verdicts into a one-dimensional protected register"),
    (60,  "ρ = h·m̂, expressed through biased claim gates; and a value stream (green) — the optimal bootstrapped action-value Q̂ tilting the forecast at temperature"),
    (60,  "β = 3.87 ≈ ρ_RL/2.  The heads realize  P_c = m_c·plan_c + (1−m_c)·p̄_c  with  m_c = σ(±a·ρ + c)  and  plan_c ∝ p̄_c·e^(β·Q̂)  — verified at held-out"),
    (60,  "KL .022 nats/round/channel (closer than the exact live Bayes oracle, .088) and closed-loop occupancy .678 vs the net’s .683."),
]
for i, (x0, ln) in enumerate(cap):
    ax.text(x0, cap_y0 - 26 * i, ln, fontsize=11, color=BODY, va="center")

fig.savefig("figs/verified_circuit.png", facecolor=BG, dpi=100)
print("wrote figs/verified_circuit.png")
