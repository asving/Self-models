"""Build the 'dual role of action entropy' figure set: learned net vs optimal (Bellman DP) vs
myopic, plus the mechanism (clamp probe) and the conceptual reward-now/info-later decomposition.
Renders matplotlib PNGs, base64-embeds them into figs/figures.html (for the Artifact tool)."""
import base64, glob, io, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
LOG3 = float(np.log(3))
NET = "#1f77b4"; OPT = "#d62728"; MYO = "#7f7f7f"; INFO = "#2ca02c"; ACC = "#9467bd"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})


def png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


# ----- data: net (per-turn mix) sweep -----
net = {}
for f in glob.glob("rps_runs/rps_b*.json") + glob.glob("rps_runs/rpsfine_b*.json"):
    d = json.load(open(f)); a = d["args"]
    if not a.get("per_traj", False):
        net[round(a["beta"], 3)] = (d["log"][-1]["entropy"], d["log"][-1]["payoff"])
nb = sorted(net); nH = [net[b][0] for b in nb]; nP = [net[b][1] for b in nb]

# ----- data: Bellman DP -----
R = json.load(open("../dp/dp_results.json"))["results"]
db = sorted(float(k) for k in R)
dH = [R[f"{b:g}" if f"{b:g}" in R else str(b)][0] if False else R[[k for k in R if float(k) == b][0]]["meanH_opt"] for b in db]
def get(b, key): return R[[k for k in R if float(k) == b][0]][key]
dH = [get(b, "meanH_opt") for b in db]
dP_opt = [get(b, "payoff_opt") for b in db]
dP_myo = [get(b, "payoff_myopic") for b in db]

# ----- data: coarse sense-then-decide heuristic optimum (rps_adaptive PART A) -----
heur = {}
if os.path.exists("figs/adaptive.out"):
    for ln in open("figs/adaptive.out"):
        m = re.match(r"\s*([01]\.\d+)\s*\|\s*m=.*\|\s*[+-][\d.]+\s*\|\s*([\d.]+)", ln)
        if m: heur[float(m.group(1))] = float(m.group(2))
hb = sorted(heur); hH = [heur[b] for b in hb]

# ----- data: clamp probe (decode R^2 + exploit quality vs forced entropy) -----
probe = []  # (forced_action_entropy, quality, R2)
if os.path.exists("figs/probe.out"):
    for ln in open("figs/probe.out"):
        m = re.match(r"\s*s=([\d.]+)\s*\|.*?\|\s*([-+][\d.]+)\s*\|\s*([-+]?[\d.]+)", ln)
        if m:
            s, qual, r2 = float(m.group(1)), float(m.group(2)), float(m.group(3))
            # forced action: argmax w.p. s else uniform -> p_max=(2s+1)/3, p_off=(1-s)/3
            pm = (2 * s + 1) / 3; po = (1 - s) / 3
            Hf = -(pm * np.log(pm + 1e-12) + 2 * po * np.log(po + 1e-12))
            probe.append((Hf, qual, r2))
probe.sort()

figs = {}

# ================= Panel A: entropy vs beta (net vs optimal vs myopic) =================
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.axhline(LOG3, color=MYO, ls="--", lw=2, label="reward-now-only (myopic) = uniform")
ax.plot(db, dH, "-s", color=OPT, lw=2, ms=6, label="optimal (Bellman DP)")
if hb: ax.plot(hb, hH, ":^", color="#ff7f0e", lw=1.8, ms=6, label="optimal (coarse sense-then-act model)")
ax.plot(nb, nH, "-o", color=NET, lw=2, ms=6, label="learned net")
ax.set_xlabel(r"$\beta$  =  P(opponent is a best-responder)")
ax.set_ylabel("mean action entropy (nats)")
ax.set_ylim(-0.05, LOG3 + 0.08); ax.set_xlim(-0.02, 1.02)
ax.annotate("info-pull wins:\nsharpen to decode\nthe opponent", xy=(0.05, 0.12), xytext=(0.12, 0.55),
            fontsize=10, color=NET, ha="left",
            arrowprops=dict(arrowstyle="->", color=NET, lw=1.2))
ax.annotate("reward-now wins:\nstay uniform to dodge\nthe best-responder", xy=(0.8, LOG3), xytext=(0.45, 0.62),
            fontsize=10, color="#444", ha="left",
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
ax.legend(fontsize=9.5, loc="center right")
ax.set_title("Optimal action entropy rises with the cost of being legible", fontsize=12.5)
figs["A"] = png(fig)

# ================= Panel B: payoff vs beta (value of information) =================
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.fill_between(db, dP_myo, dP_opt, color=OPT, alpha=0.12)
ax.plot(db, dP_opt, "-s", color=OPT, lw=2, ms=6, label="optimal (senses, then exploits)")
ax.plot(db, dP_myo, "-d", color=MYO, lw=2, ms=6, label="myopic (never senses) ≈ uniform")
ax.set_xlabel(r"$\beta$  =  P(opponent is a best-responder)")
ax.set_ylabel("reward / round")
ax.set_xlim(-0.02, 0.74)
ax.annotate("value of information\n(reward the myopic\nleaves on the table)", xy=(0.18, 0.13),
            xytext=(0.28, 0.24), fontsize=10, color=OPT,
            arrowprops=dict(arrowstyle="->", color=OPT, lw=1.2))
ax.legend(fontsize=10, loc="upper right")
ax.set_title("The myopic that ignores information collects nothing", fontsize=12.5)
figs["B"] = png(fig)

# ================= Panel C: within-game trajectory (sense-then-exploit) =================
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for b, c, lab in [(0.3, "#ff7f0e", r"optimal, $\beta=0.3$"), (0.5, OPT, r"optimal, $\beta=0.5$")]:
    tr = get(b, "H_by_round_opt"); ax.plot(range(1, len(tr) + 1), tr, "-o", color=c, ms=3.5, lw=1.8, label=lab)
myo = get(0.5, "H_by_round_myopic")
ax.plot(range(1, len(myo) + 1), myo, "--", color=MYO, lw=2, label="myopic (flat = uniform)")
ax.axhline(LOG3, color=MYO, ls=":", lw=1, alpha=0.6)
ax.set_xlabel("round within game"); ax.set_ylabel("action entropy (nats)")
ax.set_ylim(-0.05, LOG3 + 0.08)
ax.annotate("sense:\nplay sharp to read\nthe opponent (pay reward now)", xy=(1.2, 0.14), xytext=(3.2, 0.30),
            fontsize=10, color="#444", arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
ax.annotate("exploit / hedge:\ncash in later", xy=(14, 0.92), xytext=(9, 0.55),
            fontsize=10, color="#444", arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
ax.legend(fontsize=10, loc="lower right")
ax.set_title("Now vs later, within one game: sense early, exploit late", fontsize=12.5)
figs["C"] = png(fig)

# ================= Panel D: mechanism -- legibility buys information =================
if probe:
    e = [p[0] for p in probe]; q = [p[1] for p in probe]; r2 = [p[2] for p in probe]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(e, r2, "-o", color=ACC, lw=2, ms=7, label="opponent decode $R^2$ (from residual)")
    ax.plot(e, q, "-s", color=INFO, lw=2, ms=7, label="exploitation quality")
    ax.set_xlabel("agent's own action entropy (forced)")
    ax.set_ylabel("information recovered about opponent")
    ax.invert_xaxis()  # sharp (low entropy) on the right -> reads "sharpen => gain info"
    ax.annotate("uniform action:\nself-illegible,\nzero information", xy=(LOG3, r2[-1] if e[-1] > e[0] else r2[0]),
                xytext=(0.75, 0.15), fontsize=10, color="#444",
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title("Why entropy carries information: legibility = deconvolvability", fontsize=12.5)
    figs["D"] = png(fig)

# ================= Panel E: the decomposition (reward-now vs info-later) =================
# fixed mid-game belief, beta=0.5. symmetric sharp policy: prob s on counter, (1-s)/2 each off.
# u = (3s-1)/2 in [0,1] is the channel amplitude. reward-now linear in u; info-value ~ u^2 (|P1|^2 law).
s = np.linspace(1/3, 1.0, 200)
u = (3 * s - 1) / 2
H = -(s * np.log(s) + (1 - s) * np.log((1 - s) / 2 + 1e-12))
delta, gBR, beta_e, kappa = 0.5, 1.2, 0.5, 0.55
reward_now = ((1 - beta_e) * delta - beta_e * gBR) * u      # <0 slope at high beta: wants uniform
info_value = kappa * u**2                                    # wants sharp (more deconvolvable)
total = reward_now + info_value
i_tot = int(np.argmax(total)); i_now = int(np.argmax(reward_now))
fig, ax = plt.subplots(figsize=(6.8, 4.6))
ax.plot(H, reward_now, color=MYO, lw=2.2, label="reward now (vs best-responder)")
ax.plot(H, info_value, color=INFO, lw=2.2, label="value of information (for later)")
ax.plot(H, total, color=OPT, lw=2.8, label="total value")
ax.axvline(H[i_tot], color=OPT, ls="--", lw=1.4)
ax.axvline(H[i_now], color=MYO, ls=":", lw=1.4)
ax.scatter([H[i_tot]], [total[i_tot]], color=OPT, zorder=5, s=45)
ax.set_xlabel("action entropy (nats)  —  sharp ←         → uniform")
ax.set_ylabel("value (arb. units)")
ax.annotate("reward-now optimum:\nuniform (don't get countered)", xy=(H[i_now], reward_now[i_now]),
            xytext=(0.55, reward_now.min() * 0.8), fontsize=9.5, color="#444",
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.1))
ax.annotate("optimum:\nsacrifice some reward now\nto buy information", xy=(H[i_tot], total[i_tot]),
            xytext=(0.05, total.max() * 0.7), fontsize=9.5, color=OPT,
            arrowprops=dict(arrowstyle="->", color=OPT, lw=1.1))
ax.legend(fontsize=10, loc="lower center")
ax.set_title("The dual role, decomposed (illustrative, $\\beta=0.5$)", fontsize=12.5)
figs["E"] = png(fig)

# ================= Panel F: belief-state decodability (from belief_probe.py) =================
if os.path.exists("figs/belief_probe.png"):
    figs["F"] = base64.b64encode(open("figs/belief_probe.png", "rb").read()).decode()

# ================= Panel G: scaling depth -- the trap moves but doesn't vanish =================
if os.path.exists("figs/eval_big.json"):
    eb = json.load(open("figs/eval_big.json"))
    gb = sorted(float(k) for k in eb); get_b = lambda b, k: eb[[kk for kk in eb if float(kk) == b][0]][k]
    big_bias = [get_b(b, "pay_bias") for b in gb]
    big_r2p = [get_b(b, "r2p") for b in gb]; big_r2q = [get_b(b, "r2q") for b in gb]
    big_Hb = [get_b(b, "H_bias") for b in gb]
    old_bias = {0.2: 0.435, 0.3: 0.266, 0.4: 0.001, 0.5: -0.000}   # 2L net, per-traj T=40 (measured)
    dp_bias = {0.2: 0.329, 0.3: 0.249, 0.4: 0.315, 0.5: 0.231}     # DP optimal, bias-games
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    # left: bias-game payoff vs beta
    ob = sorted(old_bias); ax1.plot(ob, [old_bias[b] for b in ob], "-o", color=MYO, lw=2, ms=7, label="2L/d64 (old)")
    ax1.plot(gb, big_bias, "-o", color=NET, lw=2.4, ms=8, label="6L/d256 (deeper)")
    dpb = sorted(dp_bias); ax1.plot(dpb, [dp_bias[b] for b in dpb], "--", color=OPT, lw=2, label="Bellman optimal")
    for b, h, y in zip(gb, big_Hb, big_bias):       # mark trapped (uniform) big-net points
        if h > 0.9: ax1.annotate("trapped", xy=(b, y), xytext=(b, y + 0.06), fontsize=8,
                                 color="#b00", ha="center")
    ax1.axhline(0, color="#ccc", lw=0.8)
    ax1.set_xlabel(r"$\beta$"); ax1.set_ylabel("reward / round on exploitable (bias) games")
    ax1.legend(fontsize=9.5, loc="upper right"); ax1.set_title("Depth pushes the sensing regime higher", fontsize=12)
    # right: representation tracks escape
    ax2.plot(gb, big_r2p, "-o", color=OPT, lw=2, ms=7, label="belief decode $R^2$(p)")
    ax2.plot(gb, big_r2q, "-s", color=INFO, lw=2, ms=7, label="belief decode $R^2(\\hat q)$")
    ax2b = ax2.twinx(); ax2b.plot(gb, big_Hb, ":^", color=MYO, lw=2, ms=7, label="H(bias-games)")
    ax2b.set_ylabel("action entropy on bias games", color=MYO); ax2b.set_ylim(-0.05, LOG3 + 0.05)
    ax2b.axhline(LOG3, color=MYO, ls=":", lw=0.8, alpha=0.5)
    ax2.set_xlabel(r"$\beta$"); ax2.set_ylabel("belief decodability $R^2$"); ax2.set_ylim(-0.05, 1.02)
    ax2.legend(fontsize=9, loc="center left"); ax2.set_title("Representation collapses with behavior", fontsize=12)
    for a in (ax1, ax2): [a.spines[s].set_visible(False) for s in ("top", "right")]
    ax2b.spines["top"].set_visible(False)
    figs["G"] = png(fig)

# ================= Panels H/I/J: the continuous self-cancellation task =================
for key, fn in [("H", "figs/sc_ratio_vs_step.png"), ("I", "figs/sc_sigma_vs_step.png"),
                ("J", "figs/sc_trajectory.png"), ("K", "figs/sc_theory_curve.png")]:
    if os.path.exists(fn):
        figs[key] = base64.b64encode(open(fn, "rb").read()).decode()

# ================= assemble HTML =================
order = ["E", "A", "F", "G", "D", "C", "B", "H", "K", "I", "J"]
titles = {
    "E": "1. The thesis in one picture: action entropy trades reward-now against information-for-later",
    "A": "2. The learned net plays this balance — not the myopic one",
    "F": "2b. The net literally carries the optimal belief state — but only while it is sensing",
    "G": "2c. Scaling the net deeper (6L/d256): the trap moves to higher β but doesn't vanish",
    "H": "6. A cleaner task (continuous, single-output): the same self-legibility effect, trainable",
    "K": "7. The effect matches a parameter-free closed-form optimum",
    "I": "8. Why it converges from above: caution first, then over-confidence",
    "J": "9. What the net actually does: tracking a random walk with an over-tight uncertainty band",
    "D": "3. The mechanism: low entropy makes your own action legible, which is what carries opponent-information",
    "C": "4. The same trade-off across time within a single game",
    "B": "5. Why it matters: the value of the information the myopic forgoes",
}
caps = {
    "E": "For a fixed mid-game belief, sharpening your action (left) raises how much the pooled outcome "
         "tells you about the opponent (green, the future payoff of that information) but exposes you to the "
         "best-responder now (grey slopes down). Their sum (red) peaks at an <b>interior</b> entropy: the "
         "agent deliberately gives up some immediate reward to stay legible enough to learn. Illustrative "
         "curves from the analytic model (reward-now linear in the action's Fourier amplitude, "
         "information ∝ |P̂₁|²).",
    "A": "Mean action entropy vs. how often the opponent best-responds (β). A purely myopic / "
         "certainty-equivalent agent can't break the symmetric tie, so it sits at uniform everywhere "
         "(dashed). The optimal policy and the trained net instead <b>sharpen</b> at low β to decode an "
         "exploitable opponent, and relax toward uniform as the best-responder — which punishes "
         "legibility — comes to dominate. The DP's coarse action grid inflates its entropy somewhat; the "
         "net and the coarse sense-then-act model sit lower, but all three share the rising shape.",
    "F": "Ridge-probe of the residual stream for the exact Bayes belief the optimal (Bellman) agent "
         "maintains: <b>p</b> = P(opponent is a best-responder), <b>q̂</b> = posterior-mean bias estimate, "
         "<b>κ</b> = legibility-weighted concentration. At low β the net is in its sense-then-exploit "
         "regime and the type posterior p and bias estimate q̂ are <b>linearly decodable</b> (R² ≈ 0.5–0.8) "
         "— it is running the optimal inference. At β ≥ 0.4 the net falls into the exploration trap "
         "(plays uniform, never senses) and p, q̂ drop to <b>zero decodability</b>: the belief "
         "representation disappears exactly when the behavior gives up. (κ tracks the round index and is "
         "near-constant once actions go uniform, so treat it as a weak/positional control, not a win.)",
    "G": "A deeper/wider net (6 layers, d=256, vs the 2-layer/d64 used everywhere else) trained on the "
         "same per-trajectory task. <b>Left:</b> reward on the exploitable (bias) games. The old 2L net "
         "collapses to zero by β=0.4; the deeper net keeps exploiting up to β=0.34 and again at 0.42 — and "
         "at low β actually <i>exceeds</i> the (grid-restricted, lower-bound) Bellman optimum by playing a "
         "continuous, sharper exploit. But it still <b>traps at β=0.38 and 0.50</b>, and the escape is "
         "<b>non-monotonic in β</b> — the signature of a stochastic optimization barrier (you tunnel through "
         "or you don't), not a capacity ceiling. <b>Right:</b> the same coupling as panel 2b, now for the "
         "deeper net — belief decodability R²(p), R²(q̂) drop to zero at exactly the β where the action "
         "entropy on bias-games snaps back to uniform (dotted). Representation, sensing, and reward are one "
         "switch. (Single seed per β; the 0.38-vs-0.42 reversal is almost certainly seed noise — a "
         "multi-seed sweep would measure the escape probability directly.)",
    "H": "The RPS trap is a categorical/adversarial accident; this task isolates the same idea cleanly. A "
         "hidden latent does a random walk; the net emits ONE Gaussian N(μ,σ) that is BOTH its forecast of "
         "the next latent (proper log-score) AND its action, which pools into the next observation "
         "(x = e + g·a). It can subtract its own mean but not its own sample noise, so reporting a smaller σ "
         "buys a cleaner future read of the latent — at the cost of an honestly-calibrated forecast now. "
         "Every run starts <b>over-dispersed</b> (ratio > 1: untrained, can't track yet, and the score "
         "punishes under-confidence catastrophically vs over-confidence mildly), then descends and settles "
         "<b>below</b> the honest std — self-legibility-induced over-confidence — ordered by the pooling "
         "gain g toward the 1/√2 floor. Fully differentiable (reparameterized), no REINFORCE, no plateau.",
    "K": "The measured steady-state over-confidence ratio σ_net/σ_honest vs. the closed-form optimum "
         "[(g²−1)+√(1+6g²+g⁴)]/(4g²) (derived from the Kalman steady state; the drift magnitude cancels, so "
         "the ratio depends only on g). The trained nets land on the curve with no fitting, from 1 (g→0, no "
         "coupling) down toward the 1/√2 ≈ 0.707 asymptote.",
    "I": "Steady-state σ over training (g=2). The net starts above the honest (Kalman) std, crashes, and "
         "overshoots straight through it into a persistent over-confident regime (red gap). The descent is "
         "always from above — the proper score makes under-confidence catastrophic and over-confidence "
         "cheap, so SGD stays on the cautious side and only tightens as the estimate earns it.",
    "J": "One episode (g=2): the true latent random walk (black), the net's prediction μ (blue) tracking it "
         "with a filtering lag, and the net's ±σ band — which sits <b>inside</b> an honest filter's band "
         "(it reports tighter than its true predictive uncertainty). Right: the prediction-vs-truth phase "
         "portrait, hugging the μ=e diagonal and drifting with the walk over time.",
    "D": "Causal clamp probe on a net trained to decode+exploit a fixed bias: we <b>force</b> its action "
         "entropy and read how much it recovers about the opponent. Driving the action toward uniform "
         "(right) drops both the linear decode-R² of the true bias from the residual and the net's "
         "exploitation quality to zero — a uniform action is self-illegible, so its own sampling noise "
         "drowns the opponent's signal in the pooled outcome. This is the green curve of panel 1, measured.",
    "C": "Optimal action entropy round-by-round within one game. The agent plays a <b>sharp, low-entropy</b> "
         "move in the first round(s) — paying the best-responder cost to make that round's outcome "
         "legible — then relaxes upward once the opponent's type is resolved. The myopic agent (dashed) "
         "stays uniform throughout. 'Reward now vs. information for later', laid out in time.",
    "B": "Reward per round in the Bellman model. The myopic agent never senses, so it never finds the "
         "exploitable opponents and earns ≈ 0. The shaded gap is the <b>value of information</b> — "
         "exactly the future reward that justifies spending entropy on legibility in the panels above.",
}
html = ['<title>Action entropy: reward now vs. information for later</title>',
        '<style>body{max-width:860px;margin:0 auto;padding:28px 20px;font-family:-apple-system,'
        'Segoe UI,Roboto,sans-serif;line-height:1.5;color:#1a1a1a}h1{font-size:1.5rem}'
        'h2{font-size:1.1rem;margin-top:2.2rem;color:#111}p.cap{color:#444;font-size:0.95rem}'
        'img{width:100%;max-width:680px;display:block;margin:0.6rem auto;border:1px solid #eee;border-radius:6px}'
        '.lead{font-size:1.05rem;color:#333;border-left:3px solid #d62728;padding-left:14px}</style>',
        '<h1>The dual role of action entropy</h1>',
        '<p class="lead">An agent’s action distribution does two jobs at once: its <b>spread now</b> sets '
        'how much reward it collects this step, and its <b>sharpness</b> sets how much its own action can be '
        'subtracted back out of the next observation — i.e. how much it learns for later. Blindfolded '
        'rock–paper–scissors under imperfect monitoring isolates the tension. Below: the learned net '
        'against the optimal (Bellman) and myopic policies.</p>']
for k in order:
    if k in figs:
        if k == "H":
            html.append('<hr style="margin:2.5rem 0 0;border:none;border-top:2px solid #d62728">')
            html.append('<p class="lead">A second, cleaner instantiation — continuous, single-output, and '
                        'trainable without the optimization trap that plagued RPS. Same thesis: an agent '
                        'lowers its own action entropy below the honestly-calibrated level to keep itself '
                        'legible to its future self.</p>')
        html.append(f'<h2>{titles[k]}</h2>')
        html.append(f'<img src="data:image/png;base64,{figs[k]}" alt="{k}">')
        html.append(f'<p class="cap">{caps[k]}</p>')
open("figs/figures.html", "w").write("\n".join(html))
for k in figs:                                   # standalone PNGs (VS Code renders these natively)
    open(f"figs/panel_{k}.png", "wb").write(base64.b64decode(figs[k]))
print("panels:", [k for k in order if k in figs], "| net betas:", nb, "| probe rows:", len(probe), "| heur betas:", hb)
print("wrote figs/figures.html")
