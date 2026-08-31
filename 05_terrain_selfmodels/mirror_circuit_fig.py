"""Final verified-circuit figure for the precedent-mirror net (example.png style)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, PANEL, BOX, EDGE = "#141413", "#1c1c1b", "#242422", "#3a3a38"
TXT, SUB, MONO = "#f5f4ef", "#c3c2b7", "#8f8e85"
BLUE, AQUA, VIOLET, RED, ORANGE = "#3987e5", "#199e70", "#9085e9", "#e66767", "#d95926"

fig = plt.figure(figsize=(12.4, 8.3), facecolor=BG)
ax = fig.add_axes([0.02, 0.13, 0.96, 0.84]); ax.set_facecolor(PANEL)
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.5, 0.5), 99, 61, boxstyle="round,pad=0.4",
                            fc=PANEL, ec=EDGE, lw=1))

def box(x, y, w, h, title, body, color, fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                fc=BOX, ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 1.7, title, ha="center", color=TXT, fontsize=fs,
            fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 1.3, body, ha="center", va="center", color=SUB,
            fontsize=fs - 1.3)

def note(x, y, s, color=MONO, fs=6.9, ha="center"):
    ax.text(x, y, s, ha=ha, va="top", color=color, fontsize=fs, family="monospace")

def arrow(x0, y0, x1, y1, color, ls="-", lw=1.6):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=2, shrinkB=2))

# ---- input stream (left)
ax.text(4.5, 57.5, "the episode stream", color=SUB, fontsize=8)
for i, (tok, c) in enumerate([("x", MONO), ("a", BLUE), ("c", VIOLET)] * 2):
    y = 52 - i * 4.4
    ax.add_patch(FancyBboxPatch((3, y), 3.4, 3.1, boxstyle="round,pad=0.25",
                                fc=BOX, ec=c, lw=1.1))
    ax.text(4.7, y + 1.5, tok, ha="center", va="center", color=TXT, fontsize=9)

# ---- pathway titles
ax.text(12, 58.8, "the record stream (kept for prediction, not for action)",
        color=BLUE, fontsize=9.5, fontweight="bold")
ax.text(12, 20.5, "the terrain→inclination→dodge pathway (the policy)",
        color=ORANGE, fontsize=9.5, fontweight="bold")

# ---- record stream (top, blue)
box(11, 48, 17, 8, "L1h0 + L1h1 attention", "same-key OWN-ACTION\ngatherers (redundant pair)", BLUE)
note(19.5, 46.9, "attn mass on own a-tokens,\nkey-matched: .36 / .39")
box(32, 48, 15, 8, "the record  p̂_emp", "per-key empirical\nself-distribution", BLUE)
note(39.5, 46.9, "decode R² = .88 (peak L3)\nsteer it: policy Δ ≈ 0 (null)")
box(51, 48, 15, 8, "L3h0 attention", "same-key CAMP memory\n(reputation stream)", VIOLET)
note(58.5, 46.9, "attn mass on camps .42\nablate: ΔCE_mirror +1.29,\npolicy TV .02  — prediction-only")
box(70, 48, 16, 8, "MLP4 · hub", "mirror-camp predictive\nsoftmax(γ·p̂) + gate input", BLUE)
note(78, 46.9, "ablate: ΔCE_mirror +47.7\nvs bias +1.5 — mirror-specific")
arrow(28.3, 52, 31.7, 52, BLUE); arrow(47.3, 52, 50.7, 52, VIOLET)
arrow(66.3, 52, 69.7, 52, BLUE)
arrow(86.3, 52, 92.3, 42.5, BLUE)
box(89, 37, 9.5, 6, "camp\nlogits", "CE .77", BLUE, fs=8)

# ---- match-detector (middle, red)
box(33, 30, 21, 8.5, "L2h3 · the match-detector",
    "“are the camps tracking ME?”\nreads same-key actions + camps", RED)
note(43.5, 28.8, "attn: own-a .35, camps .15 | ablate: type-conditionality .60→.19,"
     "\nTV_mirror .41 vs TV_bias .06 | incoherent-edit TV .33, coherent .009")
arrow(19.5, 47.4, 36, 38.6, RED, ls=(0, (4, 3)))
arrow(58.5, 47.4, 50, 38.6, RED, ls=(0, (4, 3)))
ax.text(66, 33.5, "compares the two streams:\nkeys on the correlation,\nnot the content", color=SUB,
        fontsize=7.3, ha="left")

# ---- policy pathway (bottom, orange)
box(11, 9.5, 16, 8, "belief filter η", "distributed in MLPs 2/5\n(parallel re-derivation)", ORANGE)
note(19, 8.4, "terrain .59; ablate mlp5/2:\nΔterrain −.13 / −.09")
box(32, 9.5, 15, 8, "inclination", "“what I’d play here”\n(the introspected variable)", ORANGE)
note(39.5, 8.4, "= live decision pipeline,\nnot a record lookup")
box(52, 9.5, 20, 8, "the dodge (gated)", "mirror: act AWAY from own\ninclination · bias: play it", ORANGE)
note(62, 8.4, "coherent history rotation: TV .009 (record never consulted)\n"
     "out-evades the bookkeeping oracle: intercepted .36 vs .54")
box(78, 9.5, 9.5, 8, "action\nlogits", "H ≈ .03", ORANGE, fs=8)
arrow(27.3, 13.5, 31.7, 13.5, ORANGE); arrow(47.3, 13.5, 51.7, 13.5, ORANGE)
arrow(72.3, 13.5, 77.7, 13.5, ORANGE)
arrow(43.5, 29.4, 58, 18.2, RED, lw=2.2)
ax.text(54.5, 24.5, "the gate", color=RED, fontsize=8.2, fontweight="bold")

# ---- formation strip
note(50, 6.0, "formation (same components probed at RL steps 0 / 200 / 8000):\n"
     "gate ablation Δtype-cond:  −.005 → −.27 → −.60  (born at the transition)      "
     "record ablation:  ΔCE .07 → .14 → 5.3, policy TV → .03  (consolidates; decouples from action)",
     color=MONO, fs=6.8)

fig.text(0.035, 0.085, "The verified circuit.", color=BLUE, fontsize=10.5, fontweight="bold")
fig.text(0.035, 0.012,
         "A record stream (top) aggregates own past actions by key-matched attention (L1h0/h1), holds the empirical self-distribution (R² .88), and joins the\n"
         "camp-memory head (L3h0) in MLP4 to predict the mirror — ablating any of it moves camp prediction, not behavior. The match-detector (L2h3) compares\n"
         "the two streams and gates the policy: with the gate on, the dodge plays away from the live inclination — never consulting the record it maintains\n"
         "(coherent-rotation TV .009). Recompute harness matches model.forward to ~1e-3 (TF32; measured effects ≥100× larger); single-component ablations;\n"
         "roles assigned on the final net and tracked back through checkpoints.",
         color=SUB, fontsize=8.3, va="bottom")
fig.savefig("/data/users/asvin/self-models/figs/fig_mirror_circuit.png", dpi=160,
            facecolor=BG)
print("wrote fig_mirror_circuit.png")
