"""White-box the FULL-OBSERVATION forecaster (rps_runs/fc_fullobs_a0.5.pt) -- the CONTRAST CONTROL
to the imperfect-monitoring forecaster.

Setup (read rps_forecast.py + rps_im.py):
  - Forecasting game, n=3. Opponent draws q ~ Dirichlet(alpha=0.5) per game, fixed for the game.
    Each round opponent plays b_t ~ q.
  - FULL OBSERVATION: the net's input token each round IS b_t directly (start token = 3).
    The net outputs p_t (3 logits via act_head). Proper-score reward log p_t(b_t); optimum p_t = q.
  - Because it SEES b_t, the optimal/only thing it needs is a running empirical frequency of the
    observed b's (a Dirichlet-posterior count tracker). NO self-action, NO mod-3 decode, NO
    self-legibility.

This script delivers:
  1. VARIABLES: decode the running q-estimate from the residual stream; R^2 vs true q and vs the
     running empirical frequency. Show it tracks the empirical count, not self-anything.
  2. CIRCUIT: an explicit synthetic program -- a smoothed running frequency (Dirichlet-posterior with
     pseudocount kappa, optional recency/EMA) -- fit to reproduce p_t from the b-sequence. Verify
     KL(net||synthetic) on held-out games + entropy-vs-round trajectory (descends to H(q) from ABOVE,
     no dip).
  3. CONTRAST: confirm there is NO self-policy variable and NO mod-3 decode step in this net.

CPU only. Does not touch any .pt for writing.
"""
import os, sys
import numpy as np
import torch, torch.nn.functional as F

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
torch.set_num_threads(4)
from rps_im import RPSNet  # noqa: E402

DEV = "cpu"
N = 3
NAME = "fc_fullobs_a0.5"
ALPHA = 0.5


def load(name):
    ck = torch.load(os.path.expanduser(f"~/self-models/rps_runs/{name}.pt"), map_location=DEV)
    a = ck["args"]
    net = RPSNet(a["d_model"], a["n_layer"], a["n_head"], a["T"])
    net.load_state_dict(ck["state"]); net.eval()
    return net, a


def gen_games(B, T, rng):
    """Draw q ~ Dir(alpha), then b_t ~ q. Returns (seq (B,T+1) with start token N, b (B,T), q (B,N))."""
    g = rng.gamma(ALPHA, 1.0, size=(B, N))
    q = g / g.sum(1, keepdims=True)
    # sample b_t ~ q independently each round
    b = np.array([rng.choice(N, size=T, p=q[i]) for i in range(B)])
    seq = np.concatenate([np.full((B, 1), N, dtype=np.int64), b.astype(np.int64)], axis=1)
    return (torch.tensor(seq), torch.tensor(b), torch.tensor(q, dtype=torch.float32))


@torch.no_grad()
def net_run(net, seq):
    """Return per-position policy p (B,L,3) and lnf residual (B,L,d). Position t (0-indexed) sees
    tokens[0..t]; its policy is the forecast p_{t+1} for the NEXT round's b."""
    L = seq.shape[1]
    x = net.emb(seq) + net.pos(torch.arange(L))[None]
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    for blk in net.blocks:
        x = blk(x, mask)
    resid = net.lnf(x)
    logits = net.act_head(resid)
    p = F.softmax(logits, -1)
    return p, resid


def kl(p, qd, eps=1e-9):
    return (p * ((p + eps).log() - (qd + eps).log())).sum(-1)


def main():
    net, a = load(NAME)
    T = a["T"]
    print(f"=== White-box: {NAME} ({a['n_layer']}L d{a['d_model']} {a['n_head']}h T={T}) full_obs={a.get('full_obs')} ===")
    rng = np.random.default_rng(0)

    # ---- big eval batch -------------------------------------------------
    B = 4000
    seq, b, q = gen_games(B, T, rng)            # seq (B,T+1), b (B,T), q (B,N)
    p_net, resid = net_run(net, seq)            # p_net (B,T+1,3), resid (B,T+1,d)

    # Position-t forecast (t=0..T-1) is BEFORE seeing round t's b? No: position t sees seq[:t+1].
    # seq[0]=start, seq[1]=b_0, ... seq[t]=b_{t-1}. So position t (t>=1) has observed b_0..b_{t-1}.
    # Its forecast targets the NEXT opponent move. We align: forecast at position t -> uses counts of
    # b_0..b_{t-1} (t observations). We'll index rounds by k = number of b's observed = position index.
    # Position 0 (start only): 0 observations. Position k (1<=k<=T): observed b_0..b_{k-1}.

    # ---- running empirical counts --------------------------------------
    # counts[:,k,:] = histogram of b_0..b_{k-1} over the 3 symbols (k observations). k=0..T.
    onehot = F.one_hot(b, N).float()                          # (B,T,3)
    csum = onehot.cumsum(1)                                    # (B,T,3): csum[:,k-1]=counts of first k
    counts = torch.cat([torch.zeros(B, 1, N), csum], 1)        # (B,T+1,3): counts[:,k]=first k obs
    k_obs = counts.sum(-1, keepdim=True)                       # (B,T+1,1)

    # =====================================================================
    # 1) VARIABLE DECODE: running q-estimate in the residual stream
    # =====================================================================
    # We decode q-estimate from resid via ridge regression onto the EMPIRICAL FREQUENCY (uniform when 0
    # obs). Use positions k>=1 (at least 1 obs) pooled across batch and rounds. Fit on half, test on half.
    def emp_freq(c, k):  # smoothed by Laplace +1 just for a stable target; report both raw & smoothed
        return (c + 1e-9) / (k + 1e-9 + (k < 0.5).float())  # k=0 -> uniform-ish; handled below

    freq = torch.where(k_obs > 0, counts / k_obs.clamp(min=1), torch.full_like(counts, 1.0 / N))  # (B,T+1,3)
    Hq = -(q * (q + 1e-9).log()).sum(-1)                      # (B,)

    # flatten positions k=1..T (skip start; resid there has no info)
    R = resid[:, 1:, :].reshape(-1, resid.shape[-1]).numpy()  # (B*T, d)
    Y_q = q[:, None, :].expand(-1, T, -1).reshape(-1, N).numpy()       # true q (constant within game)
    Y_freq = freq[:, 1:, :].reshape(-1, N).numpy()                    # running empirical frequency
    P_net_flat = p_net[:, 1:, :].reshape(-1, N).numpy()

    def ridge_decode(R, Y, lam=10.0):
        ntr = R.shape[0] // 2
        Rtr, Rte = R[:ntr], R[ntr:]
        Ytr, Yte = Y[:ntr], Y[ntr:]
        d = R.shape[1]
        W = np.linalg.solve(Rtr.T @ Rtr + lam * np.eye(d), Rtr.T @ Ytr)
        pred = Rte @ W
        ss_res = ((pred - Yte) ** 2).sum(0)
        ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0)
        r2 = 1 - ss_res.sum() / ss_tot.sum()
        return r2, W

    r2_q, _ = ridge_decode(R, Y_q)
    r2_freq, _ = ridge_decode(R, Y_freq)
    print("\n[1] VARIABLE DECODE (ridge from lnf residual, held-out half)")
    print(f"    R^2 decode of TRUE q           : {r2_q:.3f}")
    print(f"    R^2 decode of RUNNING FREQUENCY : {r2_freq:.3f}   (the running empirical count)")

    # Decode q-estimate per round-bucket to show it sharpens with #obs (R^2 vs true q rises with k)
    print("    R^2(decode true q) by #observations k (does the running estimate sharpen?):")
    for kk in [1, 2, 3, 5, 10, 20, 39]:
        if kk > T - 1: continue
        Rk = resid[:, kk, :].numpy(); Yk = q.numpy()
        r2k, _ = ridge_decode(Rk, Yk)
        print(f"        k={kk:3d}: R^2(true q)={r2k:.3f}")

    # =====================================================================
    # 2) CIRCUIT: synthetic Dirichlet-posterior / smoothed-frequency tracker
    # =====================================================================
    # Candidate family: p_synth(k) = (counts_k + kappa) / (k + 3*kappa), optionally with EMA recency.
    # We FIT kappa (pseudocount) and an optional EMA decay gamma to minimize mean KL(net||synth) on a
    # TRAIN split of games, then report KL on a HELD-OUT split. Also try a logit-temperature.
    Btr = B // 2

    def synth_dirichlet(counts, k_obs, kappa):
        return (counts + kappa) / (k_obs + N * kappa)

    def synth_ema(b_seq, gamma, kappa, T):
        # exponential-recency weighted counts: w_k = sum_{j<k} gamma^{k-1-j} onehot(b_j)
        Bn = b_seq.shape[0]
        oh = F.one_hot(b_seq, N).float()  # (B,T,3)
        ema = torch.zeros(Bn, T + 1, N)
        wsum = torch.zeros(Bn, T + 1, 1)
        cur = torch.zeros(Bn, N); cw = torch.zeros(Bn, 1)
        for k in range(T):
            cur = gamma * cur + oh[:, k]
            cw = gamma * cw + 1.0
            ema[:, k + 1] = cur
            wsum[:, k + 1] = cw
        return (ema + kappa) / (wsum + N * kappa)

    def mean_kl_dir(kappa, idx):
        synth = synth_dirichlet(counts[idx], k_obs[idx], kappa)[:, 1:]
        return kl(p_net[idx][:, 1:], synth).mean().item()

    # grid-search kappa (plain Dirichlet count tracker)
    kappas = np.linspace(0.05, 3.0, 60)
    tr_idx = torch.arange(Btr); te_idx = torch.arange(Btr, B)
    kl_tr = [mean_kl_dir(k, tr_idx) for k in kappas]
    kappa_best = float(kappas[int(np.argmin(kl_tr))])
    kl_te_dir = mean_kl_dir(kappa_best, te_idx)

    # EMA variant: search gamma x kappa on train
    best = (1e9, None, None)
    for gamma in [1.0, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8]:
        for kp in np.linspace(0.05, 2.0, 25):
            synth = synth_ema(b[tr_idx], gamma, kp, T)[:, 1:]
            v = kl(p_net[tr_idx][:, 1:], synth).mean().item()
            if v < best[0]:
                best = (v, gamma, kp)
    _, gamma_best, kappa_ema = best
    synth_te = synth_ema(b[te_idx], gamma_best, kappa_ema, T)[:, 1:]
    kl_te_ema = kl(p_net[te_idx][:, 1:], synth_te).mean().item()

    # baselines for KL scale: net vs true q, and net vs uniform
    kl_net_trueq = kl(p_net[te_idx][:, 1:], q[te_idx][:, None, :].expand(-1, T, -1)).mean().item()
    unif = torch.full((len(te_idx), T, N), 1.0 / N)
    kl_net_unif = kl(p_net[te_idx][:, 1:], unif).mean().item()

    print("\n[2] SYNTHETIC CIRCUIT (smoothed running frequency = Dirichlet-posterior count tracker)")
    print(f"    Plain count tracker p=(counts+kappa)/(k+3kappa): kappa*={kappa_best:.3f}")
    print(f"        held-out mean KL(net||synth) = {kl_te_dir:.4f} nats")
    print(f"    EMA-recency variant: gamma*={gamma_best:.3f} kappa*={kappa_ema:.3f}")
    print(f"        held-out mean KL(net||synth) = {kl_te_ema:.4f} nats")
    print(f"    Reference scales: KL(net||true q)={kl_net_trueq:.4f}  KL(net||uniform)={kl_net_unif:.4f}")

    # KL by round for the best synthetic
    use_gamma, use_kappa = (gamma_best, kappa_ema) if kl_te_ema < kl_te_dir else (1.0, kappa_best)
    synth_full = synth_ema(b, use_gamma, use_kappa, T)  # (B,T+1,3)
    klr = kl(p_net[:, 1:], synth_full[:, 1:]).mean(0)    # (T,)
    print(f"    chosen circuit: gamma={use_gamma:.3f} kappa={use_kappa:.3f}; KL by round (k=1,2,3,5,10,20,39):")
    for kk in [1, 2, 3, 5, 10, 20, 39]:
        if kk - 1 < klr.shape[0]:
            print(f"        round k={kk:3d}: KL={klr[kk-1].item():.4f}")

    # =====================================================================
    #   entropy-vs-round trajectory: descends to H(q) from ABOVE, NO dip
    # =====================================================================
    ent_net = -(p_net * (p_net + 1e-9).log()).sum(-1)       # (B,T+1)
    ent_synth = -(synth_full * (synth_full + 1e-9).log()).sum(-1)
    Hq_mean = Hq.mean().item()
    print("\n[2b] ENTROPY TRAJECTORY (should approach H(q) from ABOVE, no dip below H(q))")
    print(f"     H(q) mean = {Hq_mean:.3f}   (uniform = {np.log(3):.3f})")
    print("     round k :  net-entropy  synth-entropy   (net - H(q))")
    for kk in [1, 2, 3, 5, 9, 15, 25, 39, 40]:
        if kk <= T:
            en = ent_net[:, kk].mean().item(); es = ent_synth[:, kk].mean().item()
            print(f"        k={kk:3d} :   {en:.3f}        {es:.3f}        {en - Hq_mean:+.3f}")
    # the IN-GAME forecasts are positions k=1..T-1 (position k forecasts round k+1, which exists for
    # k<=T-1). Position T=40 forecasts a 41st round never trained -> edge artifact; report both.
    dip_ingame = (ent_net[:, 1:T].mean(0) - Hq_mean).min().item()
    dip_all = (ent_net[:, 1:].mean(0) - Hq_mean).min().item()
    print(f"     min(net-entropy - H(q)) over IN-GAME rounds k=1..{T-1} = {dip_ingame:+.3f}   (>=0 => NO dip)")
    print(f"     min including last position k={T} (untrained edge, forecasts round {T+1}) = {dip_all:+.3f}")

    # =====================================================================
    # 3) CONTRAST: confirm NO self-policy & NO mod-3 decode
    # =====================================================================
    print("\n[3] CONTRAST -- what is ABSENT vs the imperfect-monitoring forecaster")
    # (a) Permutation test: the net's forecast depends ONLY on the multiset of observed b's (counts),
    #     not on order / not on any self-action. If true, shuffling the order of observed b's within a
    #     game leaves the FINAL forecast (nearly) unchanged -- a pure frequency tracker is order-invariant
    #     up to recency. We test the FINAL-position forecast under random permutations of the b-sequence.
    permB = 1500
    seqp, bp, qp = gen_games(permB, T, np.random.default_rng(123))
    p_orig, _ = net_run(net, seqp)
    p_orig_last = p_orig[:, -1]                              # forecast after all T obs
    # permute order of b within each game
    klperm = []
    for _ in range(5):
        perm = torch.argsort(torch.rand(permB, T), dim=1)
        bperm = torch.gather(bp, 1, perm)
        seqperm = torch.cat([torch.full((permB, 1), N, dtype=torch.long), bperm], 1)
        pp, _ = net_run(net, seqperm)
        klperm.append(kl(p_orig_last, pp[:, -1]).mean().item())
    print(f"    (a) order-permutation of observed b's: KL(final forecast orig||permuted) = "
          f"{np.mean(klperm):.4f} nats  (small => forecast is a fn of COUNTS, not order/self-action)")

    # (b) mod-3 decode probe: in the imperfect net, b must be recovered as (a-o)%3 from a SELF action.
    #     Here input IS b, so there should be NO advantage to a 'decode' variable. We confirm the net's
    #     forecast is explained by counts of the INPUT TOKENS directly (already shown by KL above), and
    #     that there is no self-action channel: the model has no action input and we never feed a_t.
    #     Quantify: residual decodes the running-frequency of the INPUT b's at high R^2 (shown in [1]).
    #     Contrast statement printed below.
    print("    (b) No mod-3 decode: input token == b_t, so b is observed, not recovered from (a-o)%3.")
    print("        The forecast is a direct count of input tokens (R^2(freq)={:.3f}, KL={:.4f}).".format(
        r2_freq, min(kl_te_dir, kl_te_ema)))
    print("    (c) No self-policy variable: net has NO action input; forecast is order-/self-invariant.")

    # Save a compact synthetic-circuit spec to a numpy file for reuse (NOT a .pt)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wb_fc_fullobs_circuit.npz")
    np.savez(out, kappa_dir=kappa_best, kl_dir=kl_te_dir, gamma_ema=gamma_best, kappa_ema=kappa_ema,
             kl_ema=kl_te_ema, r2_q=r2_q, r2_freq=r2_freq, Hq=Hq_mean, dip_all=dip_all,
             chosen_gamma=use_gamma, chosen_kappa=use_kappa, dip_ingame=dip_ingame)
    print(f"\nsaved circuit spec -> {out}")


if __name__ == "__main__":
    main()
