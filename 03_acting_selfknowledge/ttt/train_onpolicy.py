"""RUN 1 -- ON-POLICY TEACHER (DAgger-like).

Closed-loop: the net SAMPLES its move from its own current policy and PLAYS it, so
the boards it must later attribute arise from its own choices. (Opponent = the same
unpredictable 50% random / 50% minimax mix.) But the training LOSS is the SAME
cross-entropy to the minimax-optimal-move distribution (uniform over the optimal
set, legal-masked), evaluated at each net decision against the TRUE board.

=> on-policy STATES, optimal-policy LABELS.

Color-blind: net observes only {0,1} occupancy on its own turns; moves NOT fed back.
Architecture matches mid_4x96 (d_model=96, n_layer=4, n_head=4) for apples-to-apples
policy-entropy comparison with the open-loop baseline.

Saves -> ttt_runs/onpolicy_teacher.pt  (does NOT overwrite existing checkpoints).

  CUDA_VISIBLE_DEVICES=<g> ~/comp_icl/.venv/bin/python train_onpolicy.py --gpu 0
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
import ttt
from model import TTTNet, n_params
import policy_eval as PE

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttt_runs")


@torch.no_grad()
def collect_onpolicy_batch(model, device, n_games, rng, fullobs=False, p_strong=0.5):
    """Roll out n_games closed-loop (net samples from its own policy). For each net
    decision record (obs_seq_padded, true_target_dist, legal_mask). Returns padded
    tensors (B, L, 9) + valid (B, L). Target = uniform over optimal set on TRUE board."""
    MAXR = 5
    OCC, TGT, LEG, VAL = [], [], [], []
    for _ in range(n_games):
        board = [0] * 9
        occ_seq = []
        occs, tgts, legs = [], [], []
        while True:
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            obs = PE.obs_from_board(board, fullobs)
            occ_seq.append(obs)
            p = PE.policy_probs(model, occ_seq, board, device)
            opt = ttt.optimal_moves(board, 1)
            tgt = np.zeros(9, np.float32)
            if opt:
                tgt[opt] = 1.0 / len(opt)
            legal = (np.array(board) == 0).astype(np.float32)
            occs.append(obs); tgts.append(tgt); legs.append(legal)
            mv = int(rng.choice(9, p=p))   # SAMPLE own move (closed-loop)
            board[mv] = 1
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            omv = ttt.opponent_move(board, rng, p_strong=p_strong)
            if omv is None:
                break
            board[omv] = 2
        R = len(occs)
        o = np.zeros((MAXR, 9), np.float32); t = np.zeros((MAXR, 9), np.float32)
        l = np.zeros((MAXR, 9), np.float32); v = np.zeros((MAXR,), np.float32)
        for r in range(R):
            o[r] = occs[r]; t[r] = tgts[r]; l[r] = legs[r]; v[r] = 1.0
        OCC.append(o); TGT.append(t); LEG.append(l); VAL.append(v)
    dev = device
    return (torch.tensor(np.stack(OCC), device=dev), torch.tensor(np.stack(TGT), device=dev),
            torch.tensor(np.stack(LEG), device=dev), torch.tensor(np.stack(VAL), device=dev))


def masked_ce(logits, target, legal, valid):
    logits = logits.masked_fill(legal == 0, -1e9)
    logp = F.log_softmax(logits, -1)
    loss_per = -(target * logp).sum(-1)
    v = valid.reshape(-1)
    return (loss_per.reshape(-1) * v).sum() / v.sum()


def log_cadence(it, iters):
    """Dense early (to catch collapse onset), sparser later."""
    if it < 40:
        return it % 5 == 0
    if it < 120:
        return it % 10 == 0
    return it % 30 == 0 or it == iters - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="onpolicy_teacher")
    ap.add_argument("--fullobs", action="store_true")
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--iters", type=int, default=400)      # outer DAgger iters
    ap.add_argument("--games_per_iter", type=int, default=512)
    ap.add_argument("--inner_steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(RUNS, exist_ok=True)

    model = TTTNet(d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
                   max_len=6).to(device)
    print(f"[{args.tag}] fullobs={args.fullobs} params={n_params(model)} device={device}",
          flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    total = args.iters * args.inner_steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)

    ckpt_dir = os.path.join(RUNS, f"{args.tag}_steps")
    os.makedirs(ckpt_dir, exist_ok=True)
    hist = []
    t0 = time.time()
    for it in range(args.iters):
        model.eval()
        occ, tgt, leg, val = collect_onpolicy_batch(
            model, device, args.games_per_iter, rng, fullobs=args.fullobs)
        model.train()
        for _ in range(args.inner_steps):
            logits = model(occ)
            loss = masked_ce(logits, tgt, leg, val)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log_cadence(it, args.iters):
            model.eval()
            cm = PE.collapse_metrics(model, device, fullobs=args.fullobs,
                                     n_games=1000, seed=99)
            pr = PE.net_play(model, device, "random", 600, 11, fullobs=args.fullobs)
            po = PE.net_play(model, device, "optimal", 600, 22, fullobs=args.fullobs)
            print(f"[{args.tag}] it {it:4d} loss {loss.item():.4f} | {PE.fmt(cm)} | "
                  f"rand{pr['win']:.2f}/{pr['draw']:.2f}/{pr['loss']:.2f} "
                  f"opt{po['win']:.2f}/{po['draw']:.2f}/{po['loss']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            hist.append({"it": it, "loss": loss.item(), **cm,
                         "vs_random": pr, "vs_optimal": po})
            torch.save({"state_dict": model.state_dict(), "config": model.cfg,
                        "it": it, "fullobs": args.fullobs, "collapse": cm,
                        "vs_random": pr, "vs_optimal": po},
                       os.path.join(ckpt_dir, f"step{it:05d}.pt"))

    model.eval()
    cm = PE.collapse_metrics(model, device, fullobs=args.fullobs, n_games=5000, seed=7)
    play_rand = PE.net_play(model, device, "random", 2000, 11, fullobs=args.fullobs)
    play_opt = PE.net_play(model, device, "optimal", 2000, 22, fullobs=args.fullobs)
    print(f"[{args.tag}] FINAL {PE.fmt(cm)}")
    print(f"[{args.tag}] vs RANDOM {play_rand}")
    print(f"[{args.tag}] vs OPTIMAL {play_opt}")
    ckpt = {"state_dict": model.state_dict(), "config": model.cfg, "tag": args.tag,
            "fullobs": args.fullobs, "n_params": n_params(model),
            "collapse": cm, "play_random": play_rand, "play_optimal": play_opt,
            "history": hist, "method": "onpolicy_teacher_dagger"}
    path = os.path.join(RUNS, f"{args.tag}.pt")
    torch.save(ckpt, path)
    json.dump({"tag": args.tag, "fullobs": args.fullobs, "collapse": cm,
               "vs_random": play_rand, "vs_optimal": play_opt},
              open(os.path.join(RUNS, f"{args.tag}_summary.json"), "w"), indent=2)
    print(f"[{args.tag}] saved -> {path}")


if __name__ == "__main__":
    main()
