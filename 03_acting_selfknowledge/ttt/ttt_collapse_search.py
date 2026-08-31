"""Open-ended search: at AMBIGUOUS (tie) positions, WHICH optimal move does each
TTT net pick, and does the choice tie to SELF-LEGIBILITY of the true board?

Core objects
------------
A "context" = a reachable game prefix at a net decision point. We enumerate prefixes
by BFS over the game tree where:
  - net (X) plays SOME optimal move (we branch over the whole optimal set),
  - opponent (O) plays SOME legal move (we branch over all legal replies, weighted
    by the actual opponent policy p_strong=0.5 mix used in ttt.py).
Each context carries: true-board sequence (net-decision boards), occupancy sequence
(colorblind), own-move history, current true board, optimal set, and a REACH WEIGHT
= product of opponent-move probabilities along the path (net optimal-move choice is
the thing we are studying, so we DON'T weight by a net policy; we enumerate the
optimal subtree uniformly at net nodes and weight only by opponent stochasticity).

Self-legibility of a move m at a context
----------------------------------------
After the net plays m (true board b -> b'), and BEFORE the opponent replies, we ask:
how recoverable is the true board from the COLORBLIND occupancy the net will carry
forward?  We use the exact Bayesian posterior over true boards given the occupancy
SEQUENCE, under the generative process (X-first, opponent mix). Concretely we build,
over the full reachable context set, the grouping  key=(round, occ-sequence) -> set of
true boards with reach weights, and define
    legibility(context) = -H( true board | occ-sequence )      [nats; higher=more legible]
The legibility of MOVE m at a tie = legibility of the child context (round r+1) the net
lands in, marginalized over opponent replies (reach-weighted average of child
-H(true|occ-seq)).  This is "how self-legible is the future I create by playing m".

We then test (THREAD 2): at shared tie contexts, does the colorblind RL net pick the
more-self-legible optimal move more often than (a) the fullobs RL net, (b) a uniform
random tie-break?  Plus thread 1 (choice characterization), thread 3 (teacher spread),
thread 4 (choice vs downstream decodability), thread 5 (evolution over training).

CPU only.  ~/comp_icl/.venv/bin/python ttt_collapse_search.py
"""
import argparse, collections, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
import ttt
from model import TTTNet
import policy_eval as PE

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttt_runs")
DEV = "cpu"
torch.set_num_threads(1)


def load(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m = TTTNet(**ck["config"]).to(DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, bool(ck.get("fullobs", False))


# ---------------------------------------------------------------------------
# Opponent move distribution used in ttt.py (p_strong mix), as an explicit dict.
# ---------------------------------------------------------------------------
def opp_move_dist(board, p_strong=0.5):
    moves = ttt.legal_moves(board)
    if not moves:
        return {}
    opt = ttt.optimal_moves(board, 2)
    d = collections.defaultdict(float)
    # random branch
    for m in moves:
        d[m] += (1 - p_strong) / len(moves)
    # strong branch
    if opt:
        for m in opt:
            d[m] += p_strong / len(opt)
    else:
        for m in moves:
            d[m] += p_strong / len(moves)
    return dict(d)


# ---------------------------------------------------------------------------
# Enumerate reachable net-decision contexts.
# A context = dict(true_seq, occ_seq, own_moves, board, weight, round)
# Net nodes branch over the OPTIMAL set (uniform structural enumeration, NOT
# weighted -- we study the choice). Opponent nodes branch over legal replies,
# multiplying weight by opp probability.
# ---------------------------------------------------------------------------
def enumerate_contexts(p_strong=0.5):
    contexts = []  # all net-decision contexts (any round)
    # stack items: (board, true_seq, occ_seq, own_moves, weight)
    start = ([0] * 9, [], [], [], 1.0)
    stack = [start]
    while stack:
        board, true_seq, occ_seq, own_moves, w = stack.pop()
        if ttt.winner(board) != 0 or not ttt.legal_moves(board):
            continue
        r = len(true_seq)
        occ = ttt.colorblind(board)
        true_seq2 = true_seq + [tuple(board)]
        occ_seq2 = occ_seq + [tuple(int(x) for x in occ)]
        opt = ttt.optimal_moves(board, 1)
        ctx = {
            "board": tuple(board), "true_seq": tuple(true_seq2),
            "occ_seq": tuple(occ_seq2), "own_moves": tuple(own_moves),
            "round": r, "opt": tuple(opt), "weight": w,
        }
        contexts.append(ctx)
        # branch: net plays each optimal move, then opponent replies (all legal)
        for nm in opt:
            nb = list(board); nb[nm] = 1
            if ttt.winner(nb) != 0 or not ttt.legal_moves(nb):
                continue
            for om, op in opp_move_dist(nb, p_strong).items():
                nb2 = list(nb); nb2[om] = 2
                stack.append((nb2, true_seq2, occ_seq2,
                              own_moves + [nm], w * op))
    return contexts


# ---------------------------------------------------------------------------
# Legibility: -H(true board | (round, occ-seq)) over the reachable context set.
# We aggregate reach weight onto (round, occ_seq) keys -> distribution over true
# boards -> entropy. Each context inherits the legibility of its own key.
# ---------------------------------------------------------------------------
def build_legibility(contexts):
    key_dist = collections.defaultdict(lambda: collections.defaultdict(float))
    for c in contexts:
        key = (c["round"], c["occ_seq"])
        key_dist[key][c["board"]] += c["weight"]
    key_H = {}
    for key, dist in key_dist.items():
        tot = sum(dist.values())
        H = 0.0
        for v in dist.values():
            p = v / tot
            H -= p * math.log(p)
        key_H[key] = H
    return key_H  # entropy in nats; legibility = -H


def child_legibility_of_move(board, nm, occ_seq, round_r, key_H, p_strong=0.5):
    """Reach-weighted mean of -H(true|occ-seq) over the net-decision contexts the
    net lands in AFTER playing nm and the opponent replying. If the game ends after
    nm (win/full), legibility is perfectly defined (terminal, fully known) -> we
    return +0.0 entropy (max legibility) for terminal, since the board is settled."""
    nb = list(board); nb[nm] = 1
    if ttt.winner(nb) != 0 or not ttt.legal_moves(nb):
        return 0.0  # terminal: -H = 0 (no future ambiguity)
    tot = 0.0; acc = 0.0
    for om, op in opp_move_dist(nb, p_strong).items():
        nb2 = list(nb); nb2[om] = 2
        if ttt.winner(nb2) != 0 or not ttt.legal_moves(nb2):
            childH = 0.0
        else:
            child_occ = tuple(int(x) for x in ttt.colorblind(nb2))
            child_key = (round_r + 1, occ_seq + (child_occ,))
            childH = key_H.get(child_key, 0.0)
        acc += op * (-childH)
        tot += op
    return acc / tot if tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Net policy at a context (legal-renormalized over current board).
# colorblind net: feed occ_seq; fullobs net: feed true-ownership seq.
# ---------------------------------------------------------------------------
@torch.no_grad()
def net_policy_at(model, fullobs, true_seq, board):
    seq = np.stack([PE.obs_from_board(list(b), fullobs) for b in true_seq])
    inp = torch.tensor(seq, dtype=torch.float32)[None]
    logits = model(inp)[0, -1]
    legal = torch.tensor([0.0 if c != 0 else 1.0 for c in board])
    logits = logits.masked_fill(legal == 0, -1e9)
    return F.softmax(logits, -1).numpy()


# ---------------------------------------------------------------------------
# MAIN analyses
# ---------------------------------------------------------------------------
def analyze(args):
    print("enumerating contexts...", flush=True)
    contexts = enumerate_contexts(p_strong=0.5)
    key_H = build_legibility(contexts)
    # tie contexts only (|opt|>1)
    ties = [c for c in contexts if len(c["opt"]) > 1]
    print(f"  total net-decision contexts: {len(contexts)}", flush=True)
    print(f"  TIE contexts (|opt|>1): {len(ties)}", flush=True)
    # dedup tie contexts by (true_seq) -- identical history => identical query
    seen = {}
    for c in ties:
        seen.setdefault(c["true_seq"], c)
    uties = list(seen.values())
    print(f"  unique tie contexts (by true_seq): {len(uties)}", flush=True)

    # precompute per-context per-move legibility
    for c in uties:
        legs = {}
        for nm in c["opt"]:
            legs[nm] = child_legibility_of_move(
                list(c["board"]), nm, c["occ_seq"], c["round"], key_H)
        c["move_leg"] = legs
        vals = np.array([legs[nm] for nm in c["opt"]])
        c["leg_spread"] = float(vals.max() - vals.min())

    nets = {
        "rl": load(f"{RUNS}/rl.pt"),
        "rl_fullobs": load(f"{RUNS}/rl_fullobs.pt"),
        "onpolicy_teacher": load(f"{RUNS}/onpolicy_teacher.pt"),
    }

    out = {}
    # ----- thread 1+2+3: choice & self-legibility at shared tie contexts -----
    # Restrict to ties where legibility actually DIFFERS across optimal moves
    # (otherwise the choice is legibility-neutral and uninformative).
    eps = 1e-9
    disc = [c for c in uties if c["leg_spread"] > 1e-4]
    print(f"  tie contexts where legibility DISCRIMINATES moves "
          f"(spread>1e-4): {len(disc)}", flush=True)

    per_net = {}
    for name, (model, fullobs) in nets.items():
        rows = []
        for c in disc:
            p = net_policy_at(model, fullobs, c["true_seq"], list(c["board"]))
            opt = list(c["opt"])
            legs = np.array([c["move_leg"][m] for m in opt])
            pp = np.array([p[m] for m in opt])
            pp = pp / (pp.sum() + eps)  # policy restricted to optimal set
            # rank-based: probability mass-weighted mean legibility-rank (0..1)
            order = np.argsort(legs)            # ascending legibility
            ranks = np.empty(len(legs)); ranks[order] = np.linspace(0, 1, len(legs))
            chosen = int(np.argmax(pp))
            best_leg = int(np.argmax(legs))
            picks_most_legible = int(chosen == best_leg)
            # weighted legibility the net actually "spends mass on"
            net_meanleg = float((pp * legs).sum())
            unif_meanleg = float(legs.mean())
            rows.append({
                "true_seq": c["true_seq"], "round": c["round"],
                "n_opt": len(opt), "legs": legs, "pp": pp,
                "chosen": chosen, "best_leg": best_leg,
                "picks_most_legible": picks_most_legible,
                "net_meanleg": net_meanleg, "unif_meanleg": unif_meanleg,
                "leg_spread": c["leg_spread"], "weight": c["weight"],
                "ranks": ranks, "pp_rank": float((pp * ranks).sum()),
                "argmax_mv": opt[chosen],
            })
        per_net[name] = rows

    out["n_contexts"] = len(contexts)
    out["n_ties"] = len(ties)
    out["n_unique_ties"] = len(uties)
    out["n_discriminating_ties"] = len(disc)

    # Summaries
    def summarize(rows, label):
        n = len(rows)
        pml = np.mean([r["picks_most_legible"] for r in rows])
        # uniform baseline P(pick most legible) accounting for ties in legibility:
        # = mean over contexts of (#moves tied at max legibility)/n_opt
        unif_pml = np.mean([
            np.mean(np.isclose(r["legs"], r["legs"].max())) for r in rows])
        net_leg = np.mean([r["net_meanleg"] for r in rows])
        unif_leg = np.mean([r["unif_meanleg"] for r in rows])
        rank = np.mean([r["pp_rank"] for r in rows])  # 0.5=neutral, >0.5 = legible-biased
        # paired: net legibility advantage over uniform per context
        adv = np.array([r["net_meanleg"] - r["unif_meanleg"] for r in rows])
        from scipy import stats
        t, pval = stats.wilcoxon(adv) if np.any(adv != 0) else (float("nan"), float("nan"))
        d = {
            "n": int(n),
            "P(pick most legible)": float(pml),
            "P(pick most legible)_uniform_baseline": float(unif_pml),
            "net_mean_legibility": float(net_leg),
            "uniform_mean_legibility": float(unif_leg),
            "legibility_advantage_over_uniform": float(adv.mean()),
            "mass_weighted_legibility_rank(0.5=neutral)": float(rank),
            "wilcoxon_p_adv_vs_0": float(pval),
        }
        print(f"\n--- {label} (n={n} discriminating tie contexts) ---")
        for k, v in d.items():
            print(f"    {k:48s} {v}")
        return d

    out["per_net_legibility"] = {}
    for name in nets:
        out["per_net_legibility"][name] = summarize(per_net[name], name)

    # ----- thread 2 refined: PAIRED colorblind vs fullobs at SAME contexts -----
    rl_rows = {r["true_seq"]: r for r in per_net["rl"]}
    fo_rows = {r["true_seq"]: r for r in per_net["rl_fullobs"]}
    shared = [k for k in rl_rows if k in fo_rows]
    cb_leg = np.array([rl_rows[k]["net_meanleg"] for k in shared])
    fo_leg = np.array([fo_rows[k]["net_meanleg"] for k in shared])
    cb_pml = np.array([rl_rows[k]["picks_most_legible"] for k in shared])
    fo_pml = np.array([fo_rows[k]["picks_most_legible"] for k in shared])
    diff_choice = np.array([
        int(rl_rows[k]["argmax_mv"] != fo_rows[k]["argmax_mv"]) for k in shared])
    from scipy import stats
    paired = {
        "n_shared": len(shared),
        "cb_mean_legibility": float(cb_leg.mean()),
        "fo_mean_legibility": float(fo_leg.mean()),
        "cb_minus_fo_legibility": float((cb_leg - fo_leg).mean()),
        "wilcoxon_p_cb_vs_fo": float(
            stats.wilcoxon(cb_leg - fo_leg).pvalue
            if np.any(cb_leg != fo_leg) else float("nan")),
        "cb_P(most_legible)": float(cb_pml.mean()),
        "fo_P(most_legible)": float(fo_pml.mean()),
        "frac_contexts_different_argmax": float(diff_choice.mean()),
    }
    # Among contexts where they DIFFER, does CB pick the more-legible of the two?
    diff_idx = [i for i, k in enumerate(shared) if diff_choice[i]]
    if diff_idx:
        cb_more = []
        for i in diff_idx:
            k = shared[i]
            r = rl_rows[k]
            cb_mv = rl_rows[k]["argmax_mv"]; fo_mv = fo_rows[k]["argmax_mv"]
            opt = list(seen[k]["opt"])
            legc = r["move_leg"] if "move_leg" in r else None
            lg = {m: seen[k]["move_leg"][m] for m in opt}
            cb_more.append(int(lg[cb_mv] > lg[fo_mv]))
        paired["among_differing_frac_CB_more_legible_than_FO"] = float(np.mean(cb_more))
        paired["n_differing"] = len(diff_idx)
    print("\n=== PAIRED colorblind-RL vs fullobs-RL (same tie contexts) ===")
    for k, v in paired.items():
        print(f"    {k:48s} {v}")
    out["paired_cb_vs_fo"] = paired

    # save rows for downstream / training-evolution reuse
    np.save(f"{RUNS}/_collapse_disc_contexts.npy",
            np.array([{
                "true_seq": c["true_seq"], "board": c["board"],
                "occ_seq": c["occ_seq"], "round": c["round"],
                "opt": c["opt"], "move_leg": c["move_leg"],
                "weight": c["weight"], "leg_spread": c["leg_spread"],
            } for c in disc], dtype=object), allow_pickle=True)

    json.dump(_clean(out), open(f"{RUNS}/collapse_search_results.json", "w"), indent=2)
    print(f"\nsaved -> {RUNS}/collapse_search_results.json")
    return out, per_net, disc, seen, nets


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not isinstance(v, np.ndarray)}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    analyze(args)
