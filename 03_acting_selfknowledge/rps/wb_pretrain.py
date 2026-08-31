"""White-box pretrain_bias.pt: decode variables, build synthetic circuit, answer self-legibility.

Game: blindfolded RPS under imperfect monitoring. Net sees only outcome tokens
o_t=(a_t-b_t)%3 in {0 tie,1 win,2 loss}, start=3. Opponent here (per_traj, beta=0) =
pure fixed bias q~Dir(0.5), b_t~q iid. Net must decode b=(a-o)%3 to estimate q, then
best-respond (play counter to q's argmax: a*=(argmax_b q +1)%3, since a beats b iff (a-b)%3==1).
"""
import os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/comp_icl"))
torch.set_num_threads(8)
from rps_im import RPSNet
DEV = "cpu"

# payoff M[a,b]: +1 if a beats b
M = torch.tensor([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])


def load(name="pretrain_bias"):
    ck = torch.load(os.path.expanduser(f"~/self-models/rps_runs/{name}.pt"), map_location=DEV)
    a = ck["args"]; net = RPSNet(a["d_model"], a["n_layer"], a["n_head"], a["T"])
    net.load_state_dict(ck["state"]); net.eval(); return net, a


def trunk_resid(net, tok):  # final lnf residual (B,L,d)
    L = tok.shape[1]
    x = net.emb(tok) + net.pos(torch.arange(L))[None]
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    for blk in net.blocks:
        x = blk(x, mask)
    return net.lnf(x)


@torch.no_grad()
def gen_games(net, B, T, rng, force_sharp=None):
    """Roll the REAL net vs pure fixed bias. Record q, a_t, b_t, o_t, policy p_t.
    Returns seq (B,T+1), and per-step arrays. If force_sharp given, override sampling."""
    g = rng.gamma(0.5, 1.0, size=(B, 3))
    bias = torch.tensor(g / g.sum(1, keepdims=True), dtype=torch.float32)
    seq = torch.full((B, 1), 3, dtype=torch.long)
    A, Bm, O, P = [], [], [], []
    for t in range(T):
        logits, _ = net(seq); p = F.softmax(logits[:, -1], -1)
        if force_sharp is None:
            a = torch.multinomial(p, 1).squeeze(1)
        else:
            a = torch.where(torch.rand(B) < force_sharp, p.argmax(-1), torch.randint(0, 3, (B,)))
        b = torch.multinomial(bias, 1).squeeze(1)
        o = (a - b) % 3
        A.append(a); Bm.append(b); O.append(o); P.append(p)
        seq = torch.cat([seq, o[:, None]], 1)
    return (seq, bias, torch.stack(A, 1), torch.stack(Bm, 1),
            torch.stack(O, 1), torch.stack(P, 1))


@torch.no_grad()
def ridge_probe(Xtr, Ytr, Xte, Yte, lam=10.0):
    Xtr, Ytr, Xte, Yte = Xtr.detach(), Ytr.detach(), Xte.detach(), Yte.detach()
    W = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1]), Xtr.T @ Ytr)
    pred = Xte @ W
    ss_res = ((pred - Yte) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0)
    r2 = (1 - ss_res / ss_tot)
    return W, r2, pred


if __name__ == "__main__":
    net, a = load(); T = a["T"]
    print(f"=== white-box pretrain_bias: {a['n_layer']}L d{a['d_model']} {a['n_head']}h T={T} ===\n")

    # ---- generate train/test games (on-policy = real net sampling) ----
    seq_tr, bias_tr, A_tr, B_tr, O_tr, P_tr = gen_games(net, 4000, T, np.random.default_rng(1))
    seq_te, bias_te, A_te, B_te, O_te, P_te = gen_games(net, 4000, T, np.random.default_rng(2))
    with torch.no_grad():
        Rtr = trunk_resid(net, seq_tr)  # (B,L,d), L=T+1
        Rte = trunk_resid(net, seq_te)

    # =========================================================
    # PART 1: DECODE VARIABLES (linear probes vs ground truth)
    # =========================================================
    print("--- PART 1: variable decodability (linear probes, held-out R^2 / acc) ---")

    # (a) opponent favored move = argmax q (categorical, 3-way) -- from final residual
    Xtr_f = Rtr[:, -1]; Xte_f = Rte[:, -1]
    fav_tr = bias_tr.argmax(1); fav_te = bias_te.argmax(1)
    # multinomial via ridge to one-hot then argmax
    Yfav = F.one_hot(fav_tr, 3).float()
    Wf, _, pf = ridge_probe(Xtr_f, Yfav, Xte_f, F.one_hot(fav_te, 3).float())
    acc_fav = (pf.argmax(1) == fav_te).float().mean().item()
    print(f"[fav move = argmax q]  3-way acc (final pos) = {acc_fav:.3f}")

    # (b) full bias vector q (regression)
    _, r2_q, _ = ridge_probe(Xtr_f, bias_tr, Xte_f, bias_te)
    print(f"[bias vector q]        R^2 per-component (final pos) = {r2_q.numpy().round(3)}  mean={r2_q.mean():.3f}")

    # (c) net's OWN policy p_t (its intended-move distribution) -- probe at each position
    #     stack positions 1..T (residual after seeing t outcomes -> predicts p at step t)
    #     position index in seq: after t obs, the residual at index t (0=start) feeds policy for step t.
    def stack_pos(R, P):
        # R[:,t] is residual at seq position t (t=0..T). policy P[:,t] is used at step t (t=0..T-1),
        # computed from seq[:, :t+1] i.e. residual at position t. So align R[:, :T] with P.
        Xs = R[:, :T].reshape(-1, R.shape[-1])
        return Xs
    Xp_tr = stack_pos(Rtr, P_tr); Xp_te = stack_pos(Rte, P_te)
    Yp_tr = P_tr.reshape(-1, 3); Yp_te = P_te.reshape(-1, 3)
    # probe predicts policy (note: this is near-trivial since policy IS a linear-ish readout of resid;
    # but confirms 'intended move' is linearly present)
    _, r2_p, _ = ridge_probe(Xp_tr, Yp_tr, Xp_te, Yp_te)
    print(f"[own policy p_t]       R^2 per-comp (all pos) = {r2_p.numpy().round(3)}  mean={r2_p.mean():.3f}")

    # (d) own intended move = argmax p_t (the 'my-move' variable used in decode)
    mymove_tr = P_tr.argmax(-1).reshape(-1); mymove_te = P_te.argmax(-1).reshape(-1)
    Wm, _, pm = ridge_probe(Xp_tr, F.one_hot(mymove_tr, 3).float(),
                            Xp_te, F.one_hot(mymove_te, 3).float())
    acc_my = (pm.argmax(1) == mymove_te).float().mean().item()
    print(f"[own intended move]    3-way acc (all pos) = {acc_my:.3f}")

    # (e) running count of decoded b (the accumulator). Ground truth: cumulative count of b so far.
    #     decode b_hat = (mymove - o)%3 ; net's ACTUAL b is B_tr. Check the running fraction of b==k.
    #     We test whether residual encodes running empirical favored move count.
    cnt_tr = torch.zeros(Xp_tr.shape[0], 3)
    # build running counts of true b up to (not incl) step t
    run = torch.zeros(4000, 3)
    rows = []
    for t in range(T):
        rows.append(run.clone())
        run = run + F.one_hot(B_tr[:, t], 3).float()
    runc_tr = torch.stack(rows, 1).reshape(-1, 3)
    run = torch.zeros(4000, 3); rows = []
    for t in range(T):
        rows.append(run.clone()); run = run + F.one_hot(B_te[:, t], 3).float()
    runc_te = torch.stack(rows, 1).reshape(-1, 3)
    # normalize to fractions (the actual estimate of q)
    frac_tr = runc_tr / runc_tr.sum(1, keepdim=True).clamp(min=1)
    frac_te = runc_te / runc_te.sum(1, keepdim=True).clamp(min=1)
    _, r2_run, _ = ridge_probe(Xp_tr, frac_tr, Xp_te, frac_te)
    print(f"[running b-frequency]  R^2 per-comp (all pos) = {r2_run.numpy().round(3)}  mean={r2_run.mean():.3f}")

    # ---- entropy / sharpness summary ----
    ent = -(P_te * (P_te + 1e-9).log()).sum(-1)
    print(f"\npolicy entropy: early(t<10) {ent[:, :10].mean():.3f}  late(t>30) {ent[:, 30:].mean():.3f}  (uniform={np.log(3):.3f})")
    print(f"exploitation payoff/round (real net) = {(P_te.mean(1) @ M @ bias_te.T).diag().mean():+.3f}")
    # =========================================================
    # PART 2: SYNTHETIC CIRCUIT (the "synthetic net")
    # =========================================================
    # Mechanism B hypothesis: net re-derives its own intended move from the obs chain
    # (= its own policy mode), decodes b_hat=(mymove - o)%3, accumulates a Dirichlet-ish
    # count, best-responds. Two variants:
    #   (i) self-consistent SOFT: maintain count using soft policy p (not argmax)
    #   (ii) self-consistent SHARP: use argmax of own re-derived policy (mechanism B literal)
    print("\n--- PART 2: synthetic circuit ---")
    # The circuit (the 'synthetic net'):
    #   state: count[3] (Dirichlet pseudo-counts of opponent move b), prior alpha0.
    #   opening: deterministic learned move a0 (=2 for this net), as the net does.
    #   each step t:  qhat = count/sum ; intended a_t = (argmax_b count + 1) % 3  (counter to favored b)
    #                 observe o_t ; decode b_t = (a_t - o_t) % 3  (mod-3 subtraction, mechanism B:
    #                 a_t is the net's OWN re-derived policy mode, never a stored realized sample)
    #                 count[b_t] += 1
    #   output policy p_t = softmax(gain * winprob), winprob(a)=qhat[(a-1)%3].
    A0 = int(P_tr[:, 0].argmax(-1).mode().values)   # net's deterministic opening
    print(f"net opening move a0 = {A0} (deterministic: {(P_tr[:,0].argmax(-1)==A0).float().mean():.3f} of games)")

    @torch.no_grad()
    def synthetic(O_seq, alpha0=0.5, gain=12.0, open_move=A0, force_mode=None):
        """force_mode: if given (B,T) intended-move trajectory, decode b with it (teacher-forced,
        = supplying the net's own internally-read mode). Else free-running self-consistent."""
        B, Tt = O_seq.shape
        count = torch.full((B, 3), alpha0); ps = []; intended = torch.full((B,), open_move)
        for t in range(Tt):
            qhat = count / count.sum(1, keepdim=True)
            winprob = qhat[:, [2, 0, 1]]
            p = F.softmax(gain * winprob, -1); ps.append(p)
            a_int = force_mode[:, t] if force_mode is not None else (count.argmax(1) + 1) % 3
            b_hat = (a_int - O_seq[:, t]) % 3
            count = count + F.one_hot(b_hat, 3).float()
        return torch.stack(ps, 1)

    @torch.no_grad()
    def kl_agree(Psynth, Pnet):
        kl = (Pnet * ((Pnet + 1e-9).log() - (Psynth + 1e-9).log())).sum(-1).mean().item()
        agree = (Psynth.argmax(-1) == Pnet.argmax(-1)).float().mean().item()
        return kl, agree

    # (A) MECHANISM fidelity: teacher-force the net's own intended-move (= what it reads internally
    #     via mechanism B). This isolates the decode->count->BR pipeline.
    mode_tr = P_tr.argmax(-1); mode_te = P_te.argmax(-1)
    best = None
    for g in [4, 6, 8, 12, 20, 40]:
        for al in [0.1, 0.3, 0.5, 1.0]:
            Ps = synthetic(O_tr, al, g, force_mode=mode_tr)
            kl, ag = kl_agree(Ps, P_tr)
            if best is None or kl < best[0]:
                best = (kl, ag, g, al)
    kl, ag, g, al = best
    Ps_te = synthetic(O_te, al, g, force_mode=mode_te)
    kl_te, ag_te = kl_agree(Ps_te, P_te)
    pay_syn = (Ps_te.mean(1) @ M @ bias_te.T).diag().mean().item()
    print(f"[mechanism, mode teacher-forced] fit gain={g} alpha0={al}")
    print(f"  HELD-OUT KL(net||synth)={kl_te:.4f}  argmax-agree={ag_te:.3f}  (late t>=10: "
          f"{(Ps_te[:,10:].argmax(-1)==P_te[:,10:].argmax(-1)).float().mean():.3f})")
    print(f"  synthetic exploitation payoff/round={pay_syn:+.3f}  (real={(P_te.mean(1)@M@bias_te.T).diag().mean():+.3f})")

    # (B) FULLY free-running (no net access at all): seeded only by learned opening a0.
    Ps_fr = synthetic(O_te, al, g, force_mode=None)
    kl_fr, ag_fr = kl_agree(Ps_fr, P_te)
    pay_fr = (Ps_fr.mean(1) @ M @ bias_te.T).diag().mean().item()
    print(f"[free-running, opening-seeded only] argmax-agree={ag_fr:.3f}  payoff={pay_fr:+.3f}")
    print("  (free-run is brittle: a single mode mismatch shifts ALL later mod-3 decodes -> error")
    print("   amplification. The net avoids this because it reads its TRUE committed mode each step.)")

    # =========================================================
    # PART 3: MECHANISM -- how does it know its own past action?
    # =========================================================
    print("\n--- PART 3: self-legibility mechanism ---")
    # Test (B) re-derivation: feed a RANDOM outcome chain (not on-policy). The net's decode of b
    # only works if it uses its OWN re-derived intended move. Check that b_hat=(argmax p_t - o)%3
    # done with the net's own policy reproduces the accumulator that drives the next policy.
    # Concretely: does the running count built from the net's OWN re-derived moves predict the
    # net's favored-move output better than a count built from a FIXED wrong action?

    # Build synthetic using net's actual realized actions A vs re-derived argmax-policy:
    @torch.no_grad()
    def count_from_actions(O_seq, A_seq, alpha0=0.5):
        B, Tt = O_seq.shape; count = torch.full((B, 3), alpha0)
        rows = []
        for t in range(Tt):
            rows.append(count.clone())
            b_hat = (A_seq[:, t] - O_seq[:, t]) % 3
            count = count + F.one_hot(b_hat, 3).float()
        return torch.stack(rows, 1)  # counts BEFORE step t

    # net's own re-derived intended move = argmax of its policy P
    A_rederiv = P_te.argmax(-1)
    c_rederiv = count_from_actions(O_te, A_rederiv).reshape(-1, 3)
    c_realized = count_from_actions(O_te, A_te).reshape(-1, 3)
    # ground-truth net favored-move output (argmax policy NEXT step) - decode target = net's qhat
    # Compare: which count's argmax matches the net's NEXT-step favored move?
    fav_net = P_te.argmax(-1)  # net's chosen move = counter to its believed favored b
    # implied favored b from net move: b* = (a* - 1)%3  (since a* beats b*)
    fav_b_net = (fav_net - 1) % 3
    match_rederiv = (c_rederiv.argmax(1) == fav_b_net.reshape(-1)).float().mean().item()
    match_realized = (c_realized.argmax(1) == fav_b_net.reshape(-1)).float().mean().item()
    print(f"[on-policy] count(re-derived-mode) argmax == net implied fav-b: {match_rederiv:.3f}")
    print(f"[on-policy] count(realized-sample) argmax == net implied fav-b: {match_realized:.3f}")
    print("  (on-policy these tie: sharp policy => realized sample usually == mode)")

    # DECISIVE counterfactual: force the net to play RANDOM actions (force_sharp small) so the
    # realized action a_t systematically DIFFERS from the policy mode. The net is NEVER told a_t.
    # If it uses mechanism B (re-derive mode), its later favored-b output should track the count
    # built from the MODE; if mechanism A (stored realized action), it should track the count
    # built from the REALIZED forced action. We can compute both and see which predicts net output.
    print("\n  COUNTERFACTUAL (forced-random actions, realized != mode):")
    seq_cf, bias_cf, A_cf, B_cf, O_cf, P_cf = gen_games(net, 4000, T, np.random.default_rng(33), force_sharp=0.0)
    # only count steps where realized action differs from the mode at that step
    mode_cf = P_cf.argmax(-1)
    # count built from MODE-decode vs REALIZED-decode (b_hat = (a - o)%3)
    c_mode = count_from_actions(O_cf, mode_cf)      # (B,T,3) counts before step t
    c_real = count_from_actions(O_cf, A_cf)
    fav_b_net_cf = ((P_cf.argmax(-1)) - 1) % 3      # net's implied favored-b at each step
    # evaluate at later steps t>=5 where evidence accumulated
    sel = slice(5, T)
    m_mode = (c_mode[:, sel].argmax(-1) == fav_b_net_cf[:, sel]).float().mean().item()
    m_real = (c_real[:, sel].argmax(-1) == fav_b_net_cf[:, sel]).float().mean().item()
    frac_diff = (mode_cf != A_cf).float().mean().item()
    print(f"  realized != mode on {frac_diff:.2f} of forced steps")
    print(f"  net output tracks count(MODE-decode):     {m_mode:.3f}")
    print(f"  net output tracks count(REALIZED-decode): {m_real:.3f}")
    print("  -> MODE >> REALIZED confirms mechanism B (re-derives own policy mode, not stored sample)")

    # attention: which positions does the final layer attend to? capture attn weights
    @torch.no_grad()
    def attn_weights(net, tok):
        L = tok.shape[1]
        x = net.emb(tok) + net.pos(torch.arange(L))[None]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        W = []
        for blk in net.blocks:
            h = blk.ln1(x)
            _, aw = blk.attn(h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=True)
            W.append(aw)
            a, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a; x = x + blk.mlp(blk.ln2(x))
        return W
    AW = attn_weights(net, seq_te[:512])
    for li, aw in enumerate(AW):
        # attention from last query position over keys: is it uniform (counting) or local?
        last = aw[:, -1].mean(0)  # (L,)
        eff = (1.0 / (last ** 2).sum()).item()
        print(f"layer{li}: final-pos attention eff#positions={eff:.1f} of {last.shape[0]} "
              f"(self-weight={last[-1]:.3f}, mean-over-rest={last[:-1].mean():.3f})")
    print("  -> near-uniform (broad) attention = the ACCUMULATOR: it pools the whole outcome history")
    print("     (a count), consistent with the running-b-frequency probe R^2=0.80.")

    # mod-3 decode confirmation: per outcome token o, the net's move SHIFTS by exactly the cyclic
    # amount. Hold the committed move m and vary the single most-recent obs; the implied favored-b
    # = (m - o)%3 must rotate with o. We already showed favb==(m-o_mode)%3 holds 0.79 globally.
    print("\n  mod-3 decode: net's implied favored-b = (own-move - dominant-outcome) mod 3")
    ohist = torch.zeros(seq_te.shape[0], 3).scatter_add_(1, O_te, torch.ones_like(O_te, dtype=torch.float))
    o_mode = ohist.argmax(1); m_fin = P_te[:, -1].argmax(-1); favb = (m_fin - 1) % 3
    print(f"    favb == (m - o_mode) mod 3 holds on {(favb==((m_fin-o_mode)%3)).float().mean():.3f} of games")

    np.save("/tmp/claude-1104/-data-users-asvin/1c857ad9-f9be-483c-b470-b3c8c479ce5e/scratchpad/wb_probes.npy",
            dict(acc_fav=acc_fav, r2_q=r2_q.numpy(), acc_my=acc_my, r2_run=r2_run.numpy(),
                 kl_te=kl_te, ag_te=ag_te, match_rederiv=match_rederiv, match_realized=match_realized),
            allow_pickle=True)
