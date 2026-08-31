"""Evaluate the deeper/bigger per-traj nets (rpsbig_b*): payoff vs the Bellman optimum (split by
opponent type) and linear decodability of the belief state (p, q_hat). GPU-accelerated."""
import os, sys, json, glob, re
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/comp_icl")); torch.set_num_threads(8)
from rps_im import RPSNet, rollout, GAMMA_BR
from belief_probe import belief_traj, ridge_r2
DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0)
dpr = json.load(open("dp_results.json"))["results"]


def load(path):
    ck = torch.load(path, map_location=DEV); a = ck["args"]
    net = RPSNet(a["d_model"], a["n_layer"], a["n_head"], a["T"]).to(DEV); net.load_state_dict(ck["state"]); net.eval()
    return net, a


@torch.no_grad()
def payoffs(net, beta, T, B=6000):
    is_br = torch.rand(B, device=DEV) < beta
    g = rng.gamma(0.5, 1.0, size=(B, 3)); bias = torch.tensor(g / g.sum(1, keepdims=True), dtype=torch.float32, device=DEV)
    _, _, pay, ent = rollout(net, B, T, DEV, beta, bias, per_traj=True, is_br=is_br)
    pg = pay.mean(1).cpu().numpy(); eg = ent.mean(1).cpu().numpy(); m = (~is_br).cpu().numpy()
    return dict(pay=pg.mean(), pay_bias=pg[m].mean(), pay_br=pg[~m].mean(),
                H=eg.mean(), H_bias=eg[m].mean(), H_br=eg[~m].mean())


@torch.no_grad()
def capture(net, beta, T, B=2500):
    """closed-loop rollout capturing post-lnf residual + (a,o,p) per step; GPU."""
    is_br = torch.rand(B, device=DEV) < beta
    g = rng.gamma(0.5, 1.0, size=(B, 3)); bias = torch.tensor(g / g.sum(1, keepdims=True), dtype=torch.float32, device=DEV)
    seq = torch.full((B, 1), 3, dtype=torch.long, device=DEV)
    Hs, A, O, P = [], [], [], []
    for t in range(T):
        L = seq.shape[1]
        x = net.emb(seq) + net.pos(torch.arange(L, device=DEV))[None]
        mask = torch.triu(torch.ones(L, L, device=DEV, dtype=torch.bool), 1)
        for blk in net.blocks: x = blk(x, mask)
        xf = net.lnf(x); Hs.append(xf[:, -1].cpu().numpy())
        logits = net.act_head(xf[:, -1]); p = torch.softmax(logits, -1); P.append(p.cpu().numpy())
        a = torch.multinomial(p, 1).squeeze(1)
        winprob = p[:, [2, 0, 1]]; br = torch.softmax(GAMMA_BR * winprob, -1)
        q = torch.where(is_br[:, None], br, bias); b = torch.multinomial(q, 1).squeeze(1)
        o = (a - b) % 3; A.append(a.cpu().numpy()); O.append(o.cpu().numpy())
        seq = torch.cat([seq, o[:, None]], 1)
    st = lambda L: np.stack(L, 1)
    return dict(H=[st(Hs)], a=st(A), o=st(O), p=st(P), is_br=is_br.cpu().numpy(), bias=bias.cpu().numpy())


def belief_r2(net, beta, T):
    roll = capture(net, beta, T)
    pt, qh, kp = belief_traj(roll, beta)
    flat = lambda z: z.reshape(-1, *z.shape[2:]) if z.ndim > 2 else z.reshape(-1)
    N = pt.size; perm = rng.permutation(N)
    X = flat(roll["H"][-1])[perm]
    return float(np.mean(ridge_r2(X, flat(pt)[perm, None]))), float(np.mean(ridge_r2(X, flat(qh)[perm])))


def beta_of(path): return float(re.search(r"_b([\d.]+)\.pt$", path).group(1))


if __name__ == "__main__":
    print(f"device={DEV}")
    files = sorted(glob.glob("rps_runs/rpsbig_b*.pt"), key=beta_of)
    print(f"{'beta':>5} | {'arch':>10} | {'pay':>7} {'bias':>7} {'BR':>7} | {'H_bias':>6} | "
          f"{'R2(p)':>6} {'R2(q)':>6} || {'DPopt':>6} {'DPbias':>7} | {'old_pay':>7} {'old_bias':>8}")
    rows = {}
    for f in files:
        net, a = load(f); b = beta_of(f); T = a["T"]
        pf = payoffs(net, b, T); r2p, r2q = belief_r2(net, b, T)
        d = dpr.get(str(b), dpr.get(f"{b:g}", {}))
        dpo = d.get("payoff_opt", float("nan")); dpb = d.get("payoff_bias_games", float("nan"))
        of = f"rps_runs/rpstraj_b{b:g}.pt"
        if os.path.exists(of):
            onet, oa = load(of); opf = payoffs(onet, b, oa["T"]); old_p, old_b = opf["pay"], opf["pay_bias"]
        else:
            old_p = old_b = float("nan")
        rows[b] = dict(pay=float(pf["pay"]), pay_bias=float(pf["pay_bias"]), pay_br=float(pf["pay_br"]),
                       H_bias=float(pf["H_bias"]), r2p=r2p, r2q=r2q, dp_opt=dpo, dp_bias=dpb,
                       old_pay=float(old_p), old_bias=float(old_b), arch=f"{a['n_layer']}L/d{a['d_model']}")
        r = rows[b]
        print(f"{b:>5.2f} | {r['arch']:>10} | {r['pay']:+.3f} {r['pay_bias']:+.3f} {r['pay_br']:+.3f} | "
              f"{r['H_bias']:>6.2f} | {r2p:>6.2f} {r2q:>6.2f} || {dpo:+.3f} {dpb:+.3f} | "
              f"{old_p:+.3f} {old_b:+.3f}", flush=True)
    json.dump(rows, open("figs/eval_big.json", "w"), indent=2)
    print("\nKEY: does the bigger/deeper net escape the trap (pay_bias -> DP_bias, H_bias < 1.1, R2 > 0) "
          "at beta where the old 2L net collapsed (>=0.4)?")
