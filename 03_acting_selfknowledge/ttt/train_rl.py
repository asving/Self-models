"""RUN 2 -- PURE RL (color-blind)  and  CONTROL -- FULLY-OBSERVED RL.

Closed-loop on-policy: the net SAMPLES its move from its own current policy and
PLAYS it. No labels. Terminal reward at game end (win=+1, draw=0, loss=-1) credited
to ALL the net's moves in that game. REINFORCE with a baseline to cut variance
(here: a small value head V(state) trained to the return; advantage = R - V).
No discounting (short game). SMALL, DECAYING entropy bonus for early exploration
only -- it competes with the collapse we measure, so kept minimal (schedule below).

Opponent = the same unpredictable 50% random / 50% minimax mix.
Moves NOT fed back; net observes only its own-turn observation.
  - color-blind (Run 2):  {0:empty, 1:marked}
  - fully-observed (CONTROL, --fullobs): empty=0, mine=+1, opp=-1 (ownership visible)

Architecture matches mid_4x96 for apples-to-apples entropy comparison.
Saves -> ttt_runs/rl.pt (color-blind) or ttt_runs/rl_fullobs.pt (control).

ENTROPY SCHEDULE: beta(t) = beta0 * max(0, 1 - it/anneal_iters), beta0=0.02,
anneal_iters = 60% of training. After that, pure REINFORCE (no entropy term) so any
residual entropy reflects the policy itself, not the bonus.

  CUDA_VISIBLE_DEVICES=<g> ~/comp_icl/.venv/bin/python train_rl.py --tag rl --gpu 0
  CUDA_VISIBLE_DEVICES=<g> ~/comp_icl/.venv/bin/python train_rl.py --tag rl_fullobs --fullobs --gpu 1
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ttt
from model import TTTNet, n_params
import policy_eval as PE

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttt_runs")


class ValueHead(nn.Module):
    """Small state-value baseline read off the same input the policy sees. Kept
    separate from TTTNet so the policy architecture is identical to the baseline."""
    def __init__(self, d_model=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, 1))

    def forward(self, occ):  # occ (B,L,9) -> (B,L)
        return self.net(occ).squeeze(-1)


def rollout(model, device, n_games, rng, fullobs, p_strong=0.5):
    """Closed-loop rollout collecting per-decision (obs_seq, action, legal, return).
    Returns padded tensors and a list-of-length-game records for log-prob compute.
    We store per game: obs list, action list, legal list, terminal return."""
    games = []
    for _ in range(n_games):
        board = [0] * 9
        occ_seq, acts, legals = [], [], []
        while True:
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            obs = PE.obs_from_board(board, fullobs)
            occ_seq.append(obs)
            with torch.no_grad():
                p = PE.policy_probs(model, occ_seq, board, device)
            mv = int(rng.choice(9, p=p))
            acts.append(mv)
            legals.append((np.array(board) == 0).astype(np.float32))
            board[mv] = 1
            if ttt.winner(board) != 0 or not ttt.legal_moves(board):
                break
            omv = ttt.opponent_move(board, rng, p_strong=p_strong)
            if omv is None:
                break
            board[omv] = 2
        w = ttt.winner(board)
        R = 1.0 if w == 1 else (-1.0 if w == 2 else 0.0)
        games.append({"obs": occ_seq, "acts": acts, "legals": legals, "ret": R})
    return games


def pad_games(games, device):
    MAXR = 5
    B = len(games)
    occ = np.zeros((B, MAXR, 9), np.float32)
    leg = np.zeros((B, MAXR, 9), np.float32)
    act = np.full((B, MAXR), 0, np.int64)
    val = np.zeros((B, MAXR), np.float32)
    ret = np.zeros((B,), np.float32)
    for b, g in enumerate(games):
        ret[b] = g["ret"]
        for r in range(len(g["acts"])):
            occ[b, r] = g["obs"][r]; leg[b, r] = g["legals"][r]
            act[b, r] = g["acts"][r]; val[b, r] = 1.0
    return (torch.tensor(occ, device=device), torch.tensor(leg, device=device),
            torch.tensor(act, device=device), torch.tensor(val, device=device),
            torch.tensor(ret, device=device))


def log_cadence(it, iters):
    """Dense early (to catch collapse onset), sparser later. ~12-15 points."""
    if it < 300:
        return it % 50 == 0
    if it < 1000:
        return it % 100 == 0
    return it % 250 == 0 or it == iters - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rl")
    ap.add_argument("--fullobs", action="store_true")
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--games_per_iter", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--vlr", type=float, default=2e-3)
    ap.add_argument("--beta0", type=float, default=0.02)   # entropy bonus init
    ap.add_argument("--anneal_frac", type=float, default=0.6)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(RUNS, exist_ok=True)

    model = TTTNet(d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
                   max_len=6).to(device)
    vhead = ValueHead(args.d_model).to(device)
    print(f"[{args.tag}] fullobs={args.fullobs} params={n_params(model)} device={device} "
          f"beta0={args.beta0} anneal_iters={int(args.anneal_frac*args.iters)}", flush=True)
    opt = torch.optim.Adam(list(model.parameters()) + list(vhead.parameters()), lr=args.lr)
    anneal_iters = int(args.anneal_frac * args.iters)

    ckpt_dir = os.path.join(RUNS, f"{args.tag}_steps")
    os.makedirs(ckpt_dir, exist_ok=True)
    hist = []
    t0 = time.time()
    for it in range(args.iters):
        beta = args.beta0 * max(0.0, 1.0 - it / max(1, anneal_iters))
        model.eval()
        games = rollout(model, device, args.games_per_iter, rng, args.fullobs)
        model.train()
        occ, leg, act, val, ret = pad_games(games, device)
        logits = model(occ).masked_fill(leg == 0, -1e9)        # (B,L,9)
        logp = F.log_softmax(logits, -1)
        p = logp.exp()
        ent = -(p * logp).clamp(min=-50).masked_fill(leg == 0, 0.0).sum(-1)  # (B,L)
        chosen = torch.gather(logp, -1, act.unsqueeze(-1)).squeeze(-1)       # (B,L)
        v = vhead(occ)                                                       # (B,L)
        retb = ret.unsqueeze(1).expand_as(v)                                 # return to each move
        adv = (retb - v).detach()
        pg = -(chosen * adv * val).sum() / val.sum()
        ent_bonus = -beta * (ent * val).sum() / val.sum()
        vloss = ((v - retb) ** 2 * val).sum() / val.sum()
        loss = pg + ent_bonus + 0.5 * vloss
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(list(model.parameters()) + list(vhead.parameters()), 1.0)
        opt.step()
        if log_cadence(it, args.iters):
            model.eval()
            cm = PE.collapse_metrics(model, device, fullobs=args.fullobs,
                                     n_games=1000, seed=99)
            pr = PE.net_play(model, device, "random", 600, 11, fullobs=args.fullobs)
            po = PE.net_play(model, device, "optimal", 600, 22, fullobs=args.fullobs)
            mret = ret.mean().item()
            print(f"[{args.tag}] it {it:4d} beta {beta:.4f} ret {mret:+.3f} "
                  f"vloss {vloss.item():.3f} | {PE.fmt(cm)} | "
                  f"rand{pr['win']:.2f}/{pr['draw']:.2f}/{pr['loss']:.2f} "
                  f"opt{po['win']:.2f}/{po['draw']:.2f}/{po['loss']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            hist.append({"it": it, "beta": beta, "ret": mret, **cm,
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
            "history": hist, "method": "reinforce_value_baseline",
            "entropy_schedule": {"beta0": args.beta0, "anneal_iters": anneal_iters}}
    path = os.path.join(RUNS, f"{args.tag}.pt")
    torch.save(ckpt, path)
    json.dump({"tag": args.tag, "fullobs": args.fullobs, "collapse": cm,
               "vs_random": play_rand, "vs_optimal": play_opt},
              open(os.path.join(RUNS, f"{args.tag}_summary.json"), "w"), indent=2)
    print(f"[{args.tag}] saved -> {path}")


if __name__ == "__main__":
    main()
