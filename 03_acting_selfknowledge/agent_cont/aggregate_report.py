"""Aggregate the four-way policy-entropy comparison + self-legibility gap + play
strength, and print trajectory tables. Reads the saved summary JSONs / checkpoints.
"""
import json, os
import numpy as np
import torch

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttt_runs")

BASELINE = {  # open-loop mid_4x96 measured via policy_eval (matches prior nets)
    "opening_entropy": 2.197, "opening_maxprob": 0.111,
    "tie_entropy": 1.630, "tie_maxprob": 0.277, "tie_acc": 0.955,
    "move_acc": 0.908,
}


def load_ck(tag):
    p = f"{RUNS}/{tag}.pt"
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def main():
    rows = [("open_loop (mid_4x96)", BASELINE, None, None)]
    for tag in ["onpolicy_teacher", "rl", "rl_fullobs"]:
        ck = load_ck(tag)
        if ck is None:
            print(f"[{tag}] not finished yet"); continue
        rows.append((tag, ck["collapse"], ck["play_random"], ck["play_optimal"]))

    print("\n================ FOUR-WAY POLICY-ENTROPY COMPARISON (FINAL) ================")
    hdr = f"{'run':24s} {'open_H':>7s} {'open_maxp':>9s} {'tie_H':>6s} {'tie_maxp':>8s} {'tie_acc':>7s} {'move_acc':>8s}"
    print(hdr); print("-" * len(hdr))
    for name, cm, _, _ in rows:
        print(f"{name:24s} {cm['opening_entropy']:7.3f} {cm['opening_maxprob']:9.3f} "
              f"{cm['tie_entropy']:6.3f} {cm.get('tie_maxprob',float('nan')):8.3f} "
              f"{cm.get('tie_acc',float('nan')):7.3f} {cm['move_acc']:8.3f}")

    # self-legibility gap = (fullobs entropy) - (colorblind entropy); positive => CB collapses more
    cb = load_ck("rl"); fo = load_ck("rl_fullobs")
    if cb and fo:
        print("\n================ SELF-LEGIBILITY GAP (RL color-blind vs fully-observed) ================")
        for key, lbl in [("opening_entropy", "opening entropy"),
                          ("tie_entropy", "tie entropy"),
                          ("opening_maxprob", "opening max-prob"),
                          ("tie_maxprob", "tie max-prob")]:
            c = cb["collapse"][key]; f = fo["collapse"][key]
            print(f"  {lbl:18s}: colorblind={c:.3f}  fullobs={f:.3f}  "
                  f"gap(fo-cb)={f-c:+.3f}")
        print("  (entropy gap>0 / maxprob gap<0 => color-blind collapses MORE => self-legibility effect)")

    print("\n================ PLAY STRENGTH (FINAL) ================")
    for name, cm, pr, po in rows:
        if pr is None: continue
        print(f"  {name:24s} vs RANDOM  W/D/L = {pr['win']:.3f}/{pr['draw']:.3f}/{pr['loss']:.3f}")
        print(f"  {'':24s} vs OPTIMAL W/D/L = {po['win']:.3f}/{po['draw']:.3f}/{po['loss']:.3f}")

    # trajectories
    for tag in ["onpolicy_teacher", "rl", "rl_fullobs"]:
        ck = load_ck(tag)
        if ck is None: continue
        print(f"\n---- TRAJECTORY [{tag}] ----")
        print(f"{'it':>5s} {'open_H':>7s} {'open_mp':>7s} {'tie_H':>6s} {'move_acc':>8s} {'win_rand':>8s} {'draw_opt':>8s}")
        for h in ck["history"]:
            pr = h.get("vs_random", {}); po = h.get("vs_optimal", {})
            print(f"{h['it']:5d} {h['opening_entropy']:7.3f} {h['opening_maxprob']:7.3f} "
                  f"{h['tie_entropy']:6.3f} {h['move_acc']:8.3f} "
                  f"{pr.get('win',float('nan')):8.3f} {po.get('draw',float('nan')):8.3f}")


if __name__ == "__main__":
    main()
