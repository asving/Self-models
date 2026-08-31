"""Train one TTTNet config; eval move-accuracy + play strength; save artifacts.

Usage:
  python train.py --tag wide_shallow --n_layer 2 --d_model 256 --gpu 0
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

import ttt
from data import build_dataset
from model import TTTNet, n_params

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttt_runs")


def masked_ce(logits, target, legal, valid):
    # logits,target,legal: (B,L,9); valid: (B,L)
    logits = logits.masked_fill(legal == 0, -1e9)
    logp = F.log_softmax(logits, -1)
    loss_per = -(target * logp).sum(-1)  # (B,L)
    v = valid.reshape(-1)
    return (loss_per.reshape(-1) * v).sum() / v.sum()


@torch.no_grad()
def move_accuracy(model, d, device):
    """Frac of net decisions where argmax-legal move is in the optimal set."""
    occ = d["occ"].to(device); tgt = d["target"].to(device)
    legal = d["legal"].to(device); valid = d["valid"].to(device)
    logits = model(occ).masked_fill(legal == 0, -1e9)  # (B,L,9)
    pred = logits.argmax(-1)  # (B,L)
    B, L = pred.shape
    chosen_mass = torch.gather(tgt, -1, pred.unsqueeze(-1)).squeeze(-1)  # (B,L)
    correct = (chosen_mass > 0).float()
    v = valid
    overall = (correct * v).sum() / v.sum()
    # per-round (round index r)
    per_round = []
    for r in range(L):
        m = v[:, r]
        if m.sum() > 0:
            per_round.append(((correct[:, r] * m).sum() / m.sum()).item())
        else:
            per_round.append(float("nan"))
    return overall.item(), per_round


@torch.no_grad()
def net_play(model, device, opponent="random", n_games=2000, seed=123):
    """Net plays full games online: at each of its turns it argmaxes the legal
    move from the occupancy-sequence-so-far.  Opponent random or optimal.
    Returns (win, draw, loss) fractions for the net (X)."""
    rng = np.random.default_rng(seed)
    res = {"win": 0, "draw": 0, "loss": 0}
    for _ in range(n_games):
        board = [0] * 9
        occ_seq = []
        while True:
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            occ = ttt.colorblind(board)
            occ_seq.append(occ)
            inp = torch.tensor(np.stack(occ_seq), dtype=torch.float32,
                               device=device)[None]  # (1,L,9)
            logits = model(inp)[0, -1]  # (9,)
            legal_mask = torch.tensor(
                [0.0 if c != 0 else 1.0 for c in board], device=device)
            logits = logits.masked_fill(legal_mask == 0, -1e9)
            mv = int(logits.argmax().item())
            board[mv] = 1
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            if opponent == "random":
                omv = int(rng.choice(ttt.legal_moves(board)))
            else:  # optimal opponent (picks uniformly among its optimal moves)
                opt = ttt.optimal_moves(board, 2)
                omv = int(rng.choice(opt)) if opt else int(rng.choice(ttt.legal_moves(board)))
            board[omv] = 2
        w = ttt.winner(board)
        if w == 1:
            res["win"] += 1
        elif w == 2:
            res["loss"] += 1
        else:
            res["draw"] += 1
    n = sum(res.values())
    return {k: v / n for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_layer", type=int, required=True)
    ap.add_argument("--d_model", type=int, required=True)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(RUNS, exist_ok=True)

    # ensure n_head divides d_model
    nh = args.n_head
    while args.d_model % nh != 0:
        nh -= 1

    print(f"[{args.tag}] building data...", flush=True)
    d_train = build_dataset(args.n_train, seed=1, p_strong=0.5)
    d_test = build_dataset(8000, seed=999, p_strong=0.5)

    model = TTTNet(d_model=args.d_model, n_layer=args.n_layer, n_head=nh,
                   max_len=6).to(device)
    npar = n_params(model)
    print(f"[{args.tag}] n_layer={args.n_layer} d_model={args.d_model} "
          f"n_head={nh} params={npar}", flush=True)

    occ = d_train["occ"].to(device); tgt = d_train["target"].to(device)
    legal = d_train["legal"].to(device); valid = d_train["valid"].to(device)
    N = occ.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    model.train()
    for s in range(args.steps):
        bi = torch.randint(0, N, (args.bs,), device=device)
        logits = model(occ[bi])
        loss = masked_ce(logits, tgt[bi], legal[bi], valid[bi])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 1000 == 0 or s == args.steps - 1:
            model.eval()
            acc, _ = move_accuracy(model, d_test, device)
            print(f"[{args.tag}] step {s:5d} loss {loss.item():.4f} "
                  f"test-acc {acc:.4f}", flush=True)
            model.train()

    model.eval()
    acc, per_round = move_accuracy(model, d_test, device)
    play_rand = net_play(model, device, "random", n_games=2000, seed=11)
    play_opt = net_play(model, device, "optimal", n_games=2000, seed=22)
    print(f"[{args.tag}] FINAL move-acc={acc:.4f} per_round={[round(x,3) for x in per_round]}")
    print(f"[{args.tag}] vs RANDOM  {play_rand}")
    print(f"[{args.tag}] vs OPTIMAL {play_opt}")

    ckpt = {
        "state_dict": model.state_dict(),
        "config": model.cfg,
        "tag": args.tag,
        "n_params": npar,
        "move_acc": acc,
        "per_round_acc": per_round,
        "play_random": play_rand,
        "play_optimal": play_opt,
    }
    path = os.path.join(RUNS, f"{args.tag}.pt")
    torch.save(ckpt, path)
    print(f"[{args.tag}] saved -> {path}")

    # append summary row
    summ = {
        "tag": args.tag, "n_layer": args.n_layer, "d_model": args.d_model,
        "n_head": nh, "params": npar, "move_acc": acc,
        "per_round_acc": per_round,
        "vs_random": play_rand, "vs_optimal": play_opt,
    }
    with open(os.path.join(RUNS, f"{args.tag}_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)


if __name__ == "__main__":
    main()
