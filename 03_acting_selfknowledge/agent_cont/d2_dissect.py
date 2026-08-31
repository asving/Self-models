"""Dissect & reduce the d2 closed-loop agent (2x128, emit=0.9, stay=0.6).

Three tasks (run as: python d2_dissect.py {belief,rubber,reduce} ):
  belief  : confirm L1 residual carries the action-conditioned oracle filter (R2);
            net action ~= argmax(oracle belief).
  rubber  : causal efference-copy rubber-hand. Find the direction encoding the net's
            own previous committed action a_{t-1}; patch it to a counterfactual a';
            propagate; test if belief/state/next-obs mis-decode toward decode-with-a'.
  reduce  : pure-numpy simulator of d2's algorithm (action-conditioned filter driven by
            net's own action dist + MAP/softmax read-off); fit to the net.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/comp_icl"))
import probes
import agent as A

BASE = os.path.dirname(os.path.abspath(__file__))
DEV = "cpu"          # everything here is tiny
rng = np.random.default_rng(0)
Q = 3


def load(tag="d2"):
    ck = torch.load(BASE + f"/runs/agent_{tag}.pt", map_location="cpu")
    a = ck["args"]
    A.set_env(a.get("emit", 0.6), a.get("stay", 0.6))            # CRITICAL: this net's env
    net = A.Agent(a["d_model"], a["n_layer"], a["n_head"], a["L"]).to(DEV)
    net.load_state_dict(ck["state"]); net.eval()
    return net, a


def gen(net, a, B=800):
    T = torch.tensor(A.T0, dtype=torch.float32); E = torch.tensor(A.EM, dtype=torch.float32)
    pi = torch.tensor(A.PI, dtype=torch.float32)
    det, ol = a.get("det_action", False), a.get("open_loop", False)
    with torch.no_grad():
        obs, states = A.rollout(net, B, a["L"], DEV, T, E, pi, det, ol)
        al, ola = net(obs); p = F.softmax(al, -1)
        _, hs = net.backbone(obs, return_hidden=True)
        H = [net.backbone.lnf(h).numpy() for h in hs]
    return obs.numpy(), states.numpy(), p.numpy(), al.numpy(), ola.numpy(), H, ol


def probe_acc(X, lab, nc, tr=0.7):
    n = len(X); c = int(n * tr); idx = rng.permutation(n)
    W, b, _ = probes.ridge_fit(X[idx[:c]], np.eye(nc)[lab[idx[:c]]])
    return ((X[idx[c:]] @ W + b).argmax(1) == lab[idx[c:]]).mean()


# ===================================================================== #
def task_belief(tag="d2"):
    net, a = load(tag); d = a["d_model"]
    obs, st, p, al, ola, H, ol = gen(net, a)
    bel = np.stack([A.oracle_filter(obs[i], p[i], ol) for i in range(len(obs))])
    L = a["L"]; tail = slice(L // 2, None)
    net_a = p.argmax(-1)
    print(f"=== TASK 1: belief = action-conditioned filter?  agent_{tag} ({a['n_layer']}x{d}, "
          f"emit={a['emit']} stay={a['stay']}) ===")
    for i, h in enumerate(H):
        r2 = probes.ridge_fit(h.reshape(-1, d), bel.reshape(-1, Q))[2]
        sacc = probe_acc(h.reshape(-1, d), st.reshape(-1), Q)
        bmap = probe_acc(h.reshape(-1, d), bel.argmax(-1).reshape(-1), Q)
        print(f"  L{i} resid | oracle-belief R2={r2:.3f} | state-acc={sacc:.3f} | belief-MAP-acc={bmap:.3f}")
    # net action vs oracle belief MAP
    agree = (net_a[:, tail] == bel.argmax(-1)[:, tail]).mean()
    net_acc = (net_a[:, tail] == st[:, tail]).mean()
    orc_acc = (bel.argmax(-1)[:, tail] == st[:, tail]).mean()
    print(f"  net-action == oracle-belief-MAP : {agree:.3f}")
    print(f"  net act_acc={net_acc:.3f}  oracle MAP acc={orc_acc:.3f}  gap={orc_acc-net_acc:+.3f}")


# ===================================================================== #
# Rubber-hand: patch the internal record of a_{t-1} on the backbone residual.
# Hook target: output of block 0 (resid after L0), which feeds block 1 where the
# action subtraction / belief update happens. We find the a_{t-1} readout direction
# there and replace the a_{t-1} component with a counterfactual one-hot.

def fit_action_readout(H_layer, prev_a):
    """Ridge: resid (at positions 1..) -> one-hot(a_{t-1}). Returns W(d,3), b(3,)."""
    d = H_layer.shape[-1]
    X = H_layer[:, 1:, :].reshape(-1, d)          # position t carries a_{t-1}
    y = prev_a.reshape(-1)
    W, b, r2 = probes.ridge_fit(X, np.eye(Q)[y])
    return W, b, r2


def task_rubber(tag="d2"):
    net, a = load(tag); d = a["d_model"]; L = a["L"]; ol = a.get("open_loop", False)
    obs, st, p, al, ola, H, _ = gen(net, a, B=800)
    net_a = p.argmax(-1)
    prev_a = net_a[:, :-1]                          # a_{t-1} committed (decisive: env samples, but argmax is the trace)

    # --- where is a_{t-1} read? probe both layers' resid for the *committed* prev action ---
    print(f"=== TASK 2: causal efference-copy rubber-hand  agent_{tag} ===")
    for i, h in enumerate(H):
        Wp, bp, r2 = fit_action_readout(h, prev_a)
        acc = ((h[:, 1:, :].reshape(-1, d) @ Wp + bp).argmax(1) == prev_a.reshape(-1)).mean()
        print(f"  prev-action a_(t-1) readout @ L{i} resid: R2={r2:.3f} acc={acc:.3f}")

    # We patch on resid AFTER block 0 (input to block 1). Fit readout there.
    LP = 0                                          # patch layer = output of block LP
    Wp, bp, _ = fit_action_readout(H[LP], prev_a)
    # min-norm direction set: pseudo-inverse of W maps a desired logit-shift to a resid shift.
    # Simpler & cleaner: replace the readout-subspace component. Define the 3 class directions
    # as rows mapped via the readout's right-inverse so that adding delta moves the decoded
    # one-hot from a to a'.  We use the least-squares "decoder-aligned encoder": Wp (d,3).
    # To move decoded logits by +e_{a'} - e_{a}, add resid shift = Wp @ pinv? Use encoder = Wp (d,3)
    # columns; min-norm resid achieving logit change dl is Wp @ (Wp^T Wp)^-1 dl.
    G = np.linalg.inv(Wp.T @ Wp)                     # (3,3)
    def encode_shift(a_from, a_to):                  # min-norm resid shift moving readout a->a'
        dl = (np.eye(Q)[a_to] - np.eye(Q)[a_from])   # (...,3) target logit change
        return dl @ G @ Wp.T                         # (...,d)

    # --- counterfactual env: ground-truth "decode with a'" hypothesis ---
    # The belief update for current state uses EM[k,(o_t - a_{t-1})%Q]. Under patch a_{t-1}->a',
    # the net should behave as if it had taken a' last step. The cleanest oracle prediction:
    # rerun oracle_filter but with the *patched* action distribution at step t-1.

    # We'll patch a single chosen step t* per trajectory and read the effect at t*.
    # Build a torch forward hook on block LP that adds encode_shift at the chosen positions.
    obs_t = torch.tensor(obs)
    shift_buf = {"delta": None}                      # (B,L,d) additive shift, set per run
    blk = net.backbone.blocks[LP]
    def hook(module, inp, out):
        if shift_buf["delta"] is not None:
            return out + torch.tensor(shift_buf["delta"], dtype=out.dtype)
        return out
    handle = blk.register_forward_hook(hook)

    def run_patched(delta):
        shift_buf["delta"] = delta
        with torch.no_grad():
            al2, ola2 = net(obs_t)
        shift_buf["delta"] = None
        return F.softmax(al2, -1).numpy(), al2.numpy(), ola2.numpy()

    # ---------- SANITY: does the patch actually change the *decoded* prev-action? ----------
    # Apply at every position, a_{t-1} -> (a_{t-1}+1)%3, then re-probe prev-action at L1.
    a_from = prev_a                                  # (B,L-1)
    a_to = (prev_a + 1) % Q
    delta = np.zeros((len(obs), L, d), dtype=np.float32)
    delta[:, 1:, :] = encode_shift(a_from, a_to)     # patch positions 1..L-1 (which carry a_{t-1})
    p2, al2, ola2 = run_patched(delta)
    # re-probe a_{t-1} at L1 on patched run (need patched hidden) — recompute hidden under hook
    shift_buf["delta"] = delta
    with torch.no_grad():
        _, hs2 = net.backbone(obs_t, return_hidden=True)
        H2 = [net.backbone.lnf(h).numpy() for h in hs2]
    shift_buf["delta"] = None
    Wp1, bp1, _ = fit_action_readout(H[1], prev_a)   # readout fitted on clean L1
    dec_clean = (H[1][:, 1:, :].reshape(-1, d) @ Wp1 + bp1).argmax(1)
    dec_patch = (H2[1][:, 1:, :].reshape(-1, d) @ Wp1 + bp1).argmax(1)
    print(f"\n  [SANITY] L1 decoded prev-action: clean matches true a_(t-1)={ (dec_clean==prev_a.reshape(-1)).mean():.3f}; "
          f"under patch a->a+1, decoded == a+1 (target)={ (dec_patch==a_to.reshape(-1)).mean():.3f}, "
          f"== a (orig)={ (dec_patch==prev_a.reshape(-1)).mean():.3f}")

    # ---------- THE RUBBER-HAND TEST ----------
    # For each counterfactual shift k in {1,2}: patch a_{t-1} -> (a_{t-1}+k). Then compare the net's
    # downstream belief/state/next-obs to two ground-truth hypotheses computed by the oracle filter:
    #   H_real : oracle filter using the net's TRUE committed actions (no patch)
    #   H_fake : oracle filter where, at each step, the action that DECODES o is shifted by +k
    #            i.e. the net believes it took a'=a+k. Concretely we feed the oracle a one-hot at a+k.
    # Prediction of "rubber hand": patched net belief moves toward H_fake (mis-decodes o as o-a').
    #
    # Operationalize via the NEXT-OBS prediction head and via the action (=belief MAP):
    #  - state/belief: argmax of net action under patch should match H_fake's MAP more than H_real's.
    #  - next-obs: o_{t+1}=(e_{t+1}+a_t). The net's obs head predicts the *token*. Under the belief
    #    shift the predicted obs distribution should move.

    # Build H_real and H_fake belief (full trajectory patch, all positions) ---------------
    def oracle_with_actions(obs_i, p_i):
        return A.oracle_filter(obs_i, p_i, ol)
    # net action dist is p (B,L,3). For H_fake we roll the action one-hots by +k in obs-decode sense.
    print("\n  shift k | net-action-MAP agrees with: REAL-decode | FAKE-decode(a'=a+k)  (tail, patched run)")
    L2 = slice(L // 2, None)
    for k in (1, 2):
        a_to_k = (prev_a + k) % Q
        delta = np.zeros((len(obs), L, d), dtype=np.float32)
        delta[:, 1:, :] = encode_shift(prev_a, a_to_k)
        p_k, al_k, ola_k = run_patched(delta)
        netmap_k = p_k.argmax(-1)                    # net's belief-MAP (=action) under patch

        # Ground-truth hypotheses for the CURRENT-STATE belief MAP under each interpretation.
        # The state the net should infer at step t depends on which action it thinks it took at t-1.
        # REAL: net took a_{t-1}; o_t reveals e_t=o_t-a_{t-1}; state ~ e_t (emit=0.9 dominant).
        # FAKE: net thinks it took a'=a_{t-1}+k; reads e'_t=o_t-a'=e_t-k; AND the closed-loop
        #       transition shift differs. Cleanest signature: the inferred emission/state shifts by -k.
        # Compute both via the oracle filter run with the corresponding action one-hots.
        bel_real = np.stack([oracle_with_actions(obs[i], np.eye(Q)[prev_a_full(net_a)[i]])
                             for i in range(len(obs))])
        bel_fake = np.stack([oracle_with_actions(obs[i], np.eye(Q)[(prev_a_full(net_a)[i] + k) % Q])
                             for i in range(len(obs))])
        agr_real = (netmap_k[:, L2] == bel_real.argmax(-1)[:, L2]).mean()
        agr_fake = (netmap_k[:, L2] == bel_fake.argmax(-1)[:, L2]).mean()
        print(f"     k={k}   |        {agr_real:.3f}        |        {agr_fake:.3f}")

    # ---------- quantitative next-obs / belief RMSE toward each hypothesis ----------
    # Use the obs-prediction head. Patch a_{t-1}->a+k at ALL positions, look at predicted next-obs
    # distribution vs what real/fake decode imply. Simpler clean metric: how far the patched net's
    # action distribution (its belief proxy) is from clean, in the direction predicted by -k emission shift.
    handle.remove()


def prev_a_full(net_a):
    """Action one-hot index aligned per position: a_{t-1} at position t. position 0 -> a_{-1}=0."""
    pa = np.zeros_like(net_a)
    pa[:, 1:] = net_a[:, :-1]
    return pa


# ===================================================================== #
def task_reduce(tag="d2"):
    """Pure-numpy simulator of d2: action-conditioned belief filter driven by net's own action
    dist, with a temperature/sharpening read-off. Fit beta to net action dist; report KL & match."""
    net, a = load(tag); L = a["L"]; ol = a.get("open_loop", False)
    obs, st, p, al, ola, H, _ = gen(net, a, B=1000)
    # The reduced program: belief b_t, action p_t = softmax(beta * log b_t) (sharpening of belief),
    # closed loop. We feed the SAME observation stream the net saw (teacher-forced) and drive the
    # filter with the program's OWN action dist. Compare program action vs net action.
    T0, EM, PI = A.T0, A.EM, A.PI

    def simulate(beta, drive="net"):
        """Belief filter on the net's realized obs. The belief UPDATE must condition on the action
        distribution that actually generated the obs (the net's p) for consistency; `drive` selects
        whether the filter dynamics use the net's p ('net') or the program's own sharpened read-off
        ('self', a fully-standalone simulator). The read-off action is always softmax(beta*log b)."""
        B = len(obs)
        bel = np.tile(PI, (B, 1))
        progp = np.zeros((B, L, Q)); progmap = np.zeros((B, L), int); bels = np.zeros((B, L, Q))
        for t in range(L):
            bels[:, t] = bel
            logit = beta * np.log(bel + 1e-9)
            pt = np.exp(logit - logit.max(1, keepdims=True)); pt /= pt.sum(1, keepdims=True)
            progp[:, t] = pt; progmap[:, t] = pt.argmax(1)
            if t == L - 1: break
            adist = p[:, t] if drive == "net" else pt    # action dist driving the obs-decode
            o_next = obs[:, t + 1]
            Tb = bel @ T0                              # (B,3) predictive
            bn = np.zeros((B, Q))
            for k in range(Q):
                for av in range(Q):
                    trans = Tb[:, k] if ol else Tb[:, (k - av) % Q]
                    bn[:, k] += adist[:, av] * trans * EM[k, (o_next - av) % Q]
            bn /= bn.sum(1, keepdims=True); bel = bn
        return progp, progmap, bels

    tail = slice(L // 2, None)
    net_map = p.argmax(-1)
    print(f"=== TASK 3: reduced numpy program  agent_{tag} ===")
    net_acc = (net_map[:, tail] == st[:, tail]).mean()
    bel = np.stack([A.oracle_filter(obs[i], p[i], ol) for i in range(len(obs))])
    orc_acc = (bel.argmax(-1)[:, tail] == st[:, tail]).mean()
    print(f"  [ref] net track_acc={net_acc:.3f}  oracle(MAP) track_acc={orc_acc:.3f}\n")
    for drive in ("net", "self"):
        print(f"  --- filter dynamics driven by {'NET action dist (replay)' if drive=='net' else 'PROGRAM own read-off (standalone)'} ---")
        best = None
        for beta in [1, 2, 4, 8, 16, 1e9]:
            progp, progmap, bels = simulate(beta, drive)
            match = (progmap[:, tail] == net_map[:, tail]).mean()
            eps = 1e-9
            kl = (p[:, tail] * (np.log(p[:, tail] + eps) - np.log(progp[:, tail] + eps))).sum(-1).mean()
            tr = (progmap[:, tail] == st[:, tail]).mean()
            print(f"    beta={beta:>6}: prog-vs-net match={match:.3f}  KL(net||prog)={kl:.3f}  prog track_acc={tr:.3f}")
            if best is None or match > best[1]:
                best = (beta, match, tr)
        print(f"    best beta={best[0]}: match={best[1]:.3f} track={best[2]:.3f}")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "belief"
    tag = sys.argv[2] if len(sys.argv) > 2 else "d2"
    {"belief": task_belief, "rubber": task_rubber, "reduce": task_reduce}[task](tag)
