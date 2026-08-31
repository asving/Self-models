"""Doppelganger v2: JOINT-TOKEN stream (Asvin's fix for the decode->exploit wiring).

Stream per round: [x_t, y_t] with y_t = JOINT(a_t, c_t) -- one token encoding both moves.
At the DECISION position (x_t) the single head's target is y_t, so opponent-prediction is
trained by CE exactly where the policy reads out; catching = mass on the DIAGONAL of the
joint (a=c) -- a within-head reweighting instead of an impossible forward-attention hop.
Acting: sample a from the a-marginal of the joint head. Phase-2 CE masked to the
c-MARGINAL (avoids self-distillation on own sampled actions). Private observation
streams per player throughout (twin-sync guard). Personas/family unchanged from doppel.

Modes: train1 | train2 | eval
"""
from __future__ import annotations
import argparse, copy, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from ambush import World, S, sample_rows, filt_obs, filt_step
from doppel import Personas

import sys
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
from model import Block

BASE = os.path.dirname(os.path.abspath(__file__))
TOK_X0, TOK_J0, TOK_BOS, V2 = 0, 3, 12, 13


def jtok(a, c):
    return TOK_J0 + 3 * a + c


class Net2(nn.Module):
    def __init__(self, d=64, nl=6, nh=4, maxlen=80):
        super().__init__()
        self.emb = nn.Embedding(V2, d); self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, V2)

    def forward(self, tok):
        L = tok.shape[1]
        x = self.emb(tok) + self.pos(torch.arange(L, device=tok.device))[None]
        m = torch.triu(torch.ones(L, L, device=tok.device, dtype=torch.bool), 1)
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.lnf(x))


# ---------------------------------------------------------------- phase 1
def gen_phase1(B, T, rng):
    w = World(B, rng)
    p1, p2 = Personas(B, rng), Personas(B, rng)
    toks = np.zeros((B, 1 + 2 * T), dtype=np.int64); toks[:, 0] = TOK_BOS
    for t in range(T):
        z = w.emit(); z2 = w.emit()
        toks[:, 1 + 2 * t] = TOK_X0 + z
        a = p1.act_on(z); c = p2.act_on(z2)
        toks[:, 2 + 2 * t] = jtok(a, c)
        p1.drift(); p2.drift(); w.step()
    return toks

def mode_train1(args, dev):
    rng = np.random.default_rng(args.seed)
    net = Net2(args.d, args.nl, args.nh).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    ckpts = set([0] + list(range(1000, args.steps + 1, 1000)))
    logf = open(os.path.join(args.dir, "train1.jsonl"), "w")
    t0 = time.time()
    for step in range(args.steps + 1):
        toks = torch.from_numpy(gen_phase1(args.batch, args.T, rng)).to(dev)
        logits = net(toks[:, :-1]); tgt = toks[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, V2), tgt.reshape(-1),
                             reduction="none").view(tgt.shape)
        loss = ce.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            is_j = tgt >= TOK_J0
            rec = dict(step=step, loss=float(loss),
                       ce_x=float(ce[(tgt < TOK_J0)].mean()),
                       ce_joint=float(ce[is_j & (tgt < TOK_BOS)].mean()),
                       sec=round(time.time() - t0, 1))
            print(json.dumps(rec)); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step in ckpts:
            torch.save(dict(model=net.state_dict(), cfg=vars(args), step=step),
                       os.path.join(args.dir, f"p1_ckpt_{step:06d}.pt"))


# ---------------------------------------------------------------- phase 2 (RL catcher)
def joint_slice(logits):
    return logits[..., TOK_J0:TOK_J0 + 9].view(*logits.shape[:-1], 3, 3)  # [a, c]

def mode_train2(args, dev):
    rng = np.random.default_rng(args.seed + 1)
    net = Net2(args.d, args.nl, args.nh).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location=dev)
        net.load_state_dict(sd["model"]); print(f"loaded {args.init} (step {sd['step']})")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr2, weight_decay=0.01)
    B, T = args.batch2, args.T
    ckpts = set([0, 100, 400, 1000] + list(range(2000, args.steps2 + 1, 1000)))
    logf = open(os.path.join(args.dir, "train2.jsonl"), "w")
    t0 = time.time()
    for step in range(args.steps2 + 1):
        w = World(B, rng)
        opp = Personas(B, rng)
        atrue = torch.zeros(B, T, dtype=torch.long, device=dev)
        hits = np.zeros((B, T), dtype=np.float32)
        with torch.no_grad():
            tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=dev)
            for t in range(T):
                z = w.emit(); z2 = w.emit()
                tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(dev)], 1)
                jl = joint_slice(net(tt)[:, -1])
                pa = F.softmax(jl.reshape(B, 9), -1).view(B, 3, 3).sum(-1)   # a-marginal
                a = torch.multinomial(pa, 1).squeeze(1)
                atrue[:, t] = a
                c = opp.act_on(z2); opp.drift()
                hits[:, t] = a.cpu().numpy() == c
                y = TOK_J0 + 3 * a + torch.from_numpy(c).to(dev)
                tt = torch.cat([tt, y[:, None]], 1)
                w.step()
        R = hits.sum(1)
        logits = net(tt[:, :-1]); tgt = tt[:, 1:]
        pos_x = 1 + 2 * np.arange(T)                     # decision positions (target = joint)
        jl = joint_slice(logits[:, pos_x])               # (B,T,3,3)
        jls = F.log_softmax(jl.reshape(B, T, 9), -1).view(B, T, 3, 3)
        lp_a = torch.logsumexp(jls.gather(
            2, atrue[..., None, None].expand(-1, -1, 1, 3)).squeeze(2), -1)   # log a-marginal
        Rt = torch.from_numpy(R).to(dev)
        base = (Rt.sum() - Rt) / (B - 1)
        pg = -(((Rt - base) / T).detach()[:, None] * lp_a).sum(1).mean()
        # CE masked to c-marginal at decision positions + full CE on x targets
        c_real = (tgt[:, pos_x] - TOK_J0) % 3
        lp_c = torch.logsumexp(jls.gather(
            3, c_real[..., None, None].expand(-1, -1, 3, 1)).squeeze(3), -1)
        ce_c = -lp_c.mean()
        pos_y = 2 + 2 * np.arange(T - 1)                 # y-positions predict next x
        ce_x = F.cross_entropy(logits[:, pos_y].reshape(-1, V2),
                               tgt[:, pos_y].reshape(-1))
        loss = pg + args.ce_w * (ce_c + ce_x)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            pj = jls.exp()
            diag = float(pj[:, :, 0, 0].mean() + pj[:, :, 1, 1].mean() + pj[:, :, 2, 2].mean())
            ent = float(-(pj.sum(-1) * pj.sum(-1).clamp_min(1e-9).log()).sum(-1).mean())
            rec = dict(step=step, catch=float(hits.mean()), ent=ent, diag=round(diag, 3),
                       catch_early=float(hits[:, :3].mean()),
                       catch_late=float(hits[:, 12:].mean()),
                       ce_c=float(ce_c), sec=round(time.time() - t0, 1))
            print(json.dumps(rec)); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step in ckpts:
            torch.save(dict(model=net.state_dict(), cfg=vars(args), step=step),
                       os.path.join(args.dir, f"p2_ckpt_{step:06d}.pt"))


# ---------------------------------------------------------------- eval
@torch.no_grad()
def mode_eval(args, dev):
    sd = torch.load(args.init, map_location=dev)
    net = Net2(sd["cfg"]["d"], sd["cfg"]["nl"], sd["cfg"]["nh"]).to(dev)
    net.load_state_dict(sd["model"]); net.eval()
    cnet = copy.deepcopy(net)
    B, T = 1024, args.T
    results = {}
    for cond in ("personas", "self"):
        rng_c = np.random.default_rng(1234)
        w = World(B, rng_c)
        opp = Personas(B, rng_c)
        hits = np.zeros((B, T))
        tt = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=dev)
        tt2 = torch.full((B, 1), TOK_BOS, dtype=torch.long, device=dev)
        for t in range(T):
            z = w.emit(); z2 = w.emit()
            tt = torch.cat([tt, torch.from_numpy(TOK_X0 + z[:, None]).to(dev)], 1)
            tt2 = torch.cat([tt2, torch.from_numpy(TOK_X0 + z2[:, None]).to(dev)], 1)
            pa = F.softmax(joint_slice(net(tt)[:, -1]).reshape(B, 9), -1).view(B, 3, 3).sum(-1)
            a = torch.multinomial(pa, 1).squeeze(1)
            if cond == "self":
                pc = F.softmax(joint_slice(cnet(tt2)[:, -1]).reshape(B, 9), -1
                               ).view(B, 3, 3).sum(-1)
                c_t = torch.multinomial(pc, 1).squeeze(1)
                c = c_t.cpu().numpy()
            else:
                c = opp.act_on(z2); opp.drift()
                c_t = torch.from_numpy(c).to(dev)
            hits[:, t] = a.cpu().numpy() == c
            tt = torch.cat([tt, (TOK_J0 + 3 * a + c_t)[:, None]], 1)
            tt2 = torch.cat([tt2, (TOK_J0 + 3 * c_t + a)[:, None]], 1)   # mirrored slots
            w.step()
        results[cond] = hits.mean(0).tolist()
        print(f"[{cond}] rounds 0-2: {hits[:, :3].mean():.3f}   late: {hits[:, 12:].mean():.3f}")
        print(f"[{cond}] by round: " + " ".join(f"{v:.2f}" for v in hits.mean(0)))
    json.dump(results, open(os.path.join(args.dir, "eval_selfplay.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train1", "train2", "eval"], required=True)
    ap.add_argument("--dir", default=os.path.join(BASE, "doppel2_runs", "A"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--nh", type=int, default=4)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batch2", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--steps2", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--ce_w", type=float, default=1.0)
    ap.add_argument("--init", type=str, default="")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    {"train1": mode_train1, "train2": mode_train2, "eval": mode_eval}[args.mode](args, dev)

if __name__ == "__main__":
    main()
